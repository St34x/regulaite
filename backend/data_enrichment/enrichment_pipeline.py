# plugins/regul_aite/backend/data_enrichment/enrichment_pipeline.py
import logging
import os
from typing import Dict, List, Any, Optional, Set, Tuple
import json
from neo4j import GraphDatabase

from .entity_extractor import EntityExtractor
from .concept_extractor import ConceptExtractor
from .regulatory_analyzer import RegulatoryAnalyzer
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class EnrichmentPipeline:
    """
    Enrichment pipeline for documents. Combines entity extraction, concept extraction,
    and regulatory analysis into a single pipeline.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        spacy_model: str = "en_core_web_sm",
        regulatory_domain: bool = True,
        multilingual: bool = True,
        max_models: int = 3
    ):
        """
        Initialize the enrichment pipeline.

        Args:
            neo4j_uri: URI for Neo4j database
            neo4j_user: Username for Neo4j
            neo4j_password: Password for Neo4j
            spacy_model: Name of spaCy model to use
            regulatory_domain: Whether to include regulatory domain knowledge
            multilingual: Whether to enable multilingual support
            max_models: Maximum number of language models to keep in memory
        """
        # Neo4j connection settings
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

        # Initialize Neo4j connection
        self.driver = None
        self._connect_to_neo4j()

        # Save multilingual setting
        self.multilingual = multilingual

        # Initialize language detector
        self.language_detector = LanguageDetector(fallback_language='en')

        # Initialize enrichment components
        self.entity_extractor = EntityExtractor(
            spacy_model=spacy_model,
            multilingual=multilingual,
            max_models=max_models
        )

        self.concept_extractor = ConceptExtractor(
            spacy_model=spacy_model,
            regulatory_domain=regulatory_domain,
            multilingual=multilingual,
            max_models=max_models
        )

        self.regulatory_analyzer = RegulatoryAnalyzer(multilingual=multilingual)

        logger.info(f"Enrichment pipeline initialized with multilingual support: {multilingual}")

    def _connect_to_neo4j(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=3600
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected to Neo4j' AS message")
                for record in result:
                    logger.info(record["message"])
            logger.info(f"Enrichment pipeline connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def enrich_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enrich text with extracted entities, concepts, and regulatory analysis.

        Args:
            text: Text to enrich
            context: Optional context information

        Returns:
            Dictionary of enrichment results
        """
        # Initialize context if not provided
        if context is None:
            context = {}

        # Detect language if multilingual support is enabled
        language_info = None
        lang_code = None

        if self.multilingual:
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']
            logger.info(f"Detected language: {language_info['language_name']} ({lang_code})")

            # Add language info to context
            context['language'] = {
                'code': lang_code,
                'name': language_info['language_name'],
                'confidence': language_info['confidence'],
                'is_supported': language_info['is_supported'],
                'model': language_info['model_name']
            }

        # Extract entities with language awareness
        logger.info("Extracting entities from text")
        entity_dict = self.entity_extractor.extract_entities(text, lang_code)

        # Extract concepts with language awareness
        logger.info("Extracting concepts from text")
        concepts = self.concept_extractor.extract_concepts(text, lang_code)

        # Perform regulatory analysis
        logger.info("Performing regulatory analysis")
        regulatory_analysis = self.regulatory_analyzer.analyze_text(text, lang_code)

        # Build the enrichment result
        result = {
            "entities": entity_dict,
            "concepts": concepts,
            "regulatory_analysis": regulatory_analysis,
            "context": context
        }

        # Count items
        entity_count = sum(len(entities) for entities in entity_dict.values())
        concept_count = len(concepts)
        requirement_count = len(regulatory_analysis.get("requirements", []))
        has_regulatory_content = regulatory_analysis.get("summary", {}).get("has_regulatory_content", False)

        # Add counts to the result
        result["stats"] = {
            "entity_count": entity_count,
            "concept_count": concept_count,
            "requirement_count": requirement_count,
            "has_regulatory_content": has_regulatory_content
        }

        return result

    def enrich_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Enrich a document stored in Neo4j.

        Args:
            doc_id: Document ID to enrich

        Returns:
            Dictionary with enrichment results
        """
        logger.info(f"Enriching document: {doc_id}")
        try:
            # Get all chunks for the document
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                    RETURN c.chunk_id AS chunk_id, c.text AS text, c.section AS section
                    ORDER BY c.index
                    """,
                    doc_id=doc_id
                )

                chunks = [{"chunk_id": record["chunk_id"], "text": record["text"], "section": record["section"]}
                          for record in result]

            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return {"status": "error", "message": "No chunks found for document"}

            # Get document language
            lang_code = None
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.language AS language
                    """,
                    doc_id=doc_id
                )
                record = result.single()
                if record and record["language"]:
                    lang_code = record["language"]

            # Combine chunks into a single text
            all_text = "\n\n".join([chunk["text"] for chunk in chunks])

            # Enrich the combined text
            enrichment_result = self.enrich_text(all_text, {"doc_id": doc_id})

            # Store the enrichment results back to Neo4j
            entity_count = enrichment_result["stats"]["entity_count"]
            concept_count = enrichment_result["stats"]["concept_count"]
            requirement_count = enrichment_result["stats"]["requirement_count"]
            has_regulatory_content = enrichment_result["stats"]["has_regulatory_content"]

            # Update document with statistics
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    SET d.entity_count = $entity_count,
                        d.concept_count = $concept_count,
                        d.requirement_count = $requirement_count,
                        d.has_regulatory_content = $has_regulatory_content,
                        d.enriched = true,
                        d.enriched_at = datetime()
                    """,
                    doc_id=doc_id,
                    entity_count=entity_count,
                    concept_count=concept_count,
                    requirement_count=requirement_count,
                    has_regulatory_content=has_regulatory_content
                )

            # Create and link entities
            self._store_entities(doc_id, enrichment_result["entities"])

            # Create and link concepts
            self._store_concepts(doc_id, enrichment_result["concepts"])

            # Create and link regulatory items
            self._store_regulatory_items(doc_id, enrichment_result["regulatory_analysis"])

            logger.info(f"Document {doc_id} enriched with {entity_count} entities, {concept_count} concepts, and {requirement_count} requirements")
            return {
                "status": "success",
                "doc_id": doc_id,
                "entities": entity_count,
                "concepts": concept_count,
                "requirements": requirement_count,
                "has_regulatory_content": has_regulatory_content
            }

        except Exception as e:
            logger.error(f"Error enriching document {doc_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _store_entities(self, doc_id: str, entities: Dict[str, List[Dict[str, Any]]]):
        """Store entities in Neo4j and link them to the document"""
        if not entities:
            return

        try:
            # Get document name for metadata
            with self.driver.session() as session:
                doc_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.name as doc_name
                    """,
                    doc_id=doc_id
                )
                record = doc_result.single()
                doc_name = record["doc_name"] if record else "Unknown Document"

                # Check if Entity label exists in the database
                labels_result = session.run(
                    """
                    CALL db.labels() YIELD label
                    RETURN collect(label) as labels
                    """
                )
                node_labels = labels_result.single()["labels"]

                # Create Entity label if it doesn't exist
                if "Entity" not in node_labels:
                    session.run(
                        """
                        CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
                        CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name);
                        """
                    )
                    logger.info("Created Entity label with constraints and indices")

            with self.driver.session() as session:
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        # Create a consistent ID based on entity text and type
                        entity_text = entity["text"]
                        normalized_text = entity.get("normalized", entity["text"].lower())

                        try:
                            # Use OPTIONAL MATCH for safer querying of properties that might not exist
                            check_result = session.run(
                                """
                                MATCH (e:Entity {name: $name, type: $type})
                                RETURN CASE WHEN e.doc_names IS NOT NULL THEN e.doc_names ELSE [] END as doc_names
                                """,
                                name=entity_text,
                                type=entity_type
                            )

                            existing_record = check_result.single()
                            if existing_record:
                                # Entity exists, let's update doc_names
                                doc_names = existing_record["doc_names"] or []
                                if isinstance(doc_names, list) and doc_name not in doc_names:
                                    doc_names.append(doc_name)
                                else:
                                    doc_names = [doc_name]

                                # Use MERGE to handle uniqueness constraints
                                session.run(
                                    """
                                    MERGE (e:Entity {name: $name, type: $type})
                                    ON CREATE SET e.id = $entity_id,
                                                e.normalized = $normalized,
                                                e.created_at = datetime(),
                                                e.doc_names = $doc_names
                                    ON MATCH SET e.updated_at = datetime(),
                                                e.frequency = COALESCE(e.frequency, 0) + 1,
                                                e.doc_names = $doc_names
                                    """,
                                    entity_id=f"entity_{hash((entity_text, entity_type))}",
                                    name=entity_text,
                                    type=entity_type,
                                    normalized=normalized_text,
                                    doc_names=doc_names
                                )
                            else:
                                # New entity, set doc_names to [doc_name]
                                session.run(
                                    """
                                    MERGE (e:Entity {name: $name, type: $type})
                                    ON CREATE SET e.id = $entity_id,
                                                e.normalized = $normalized,
                                                e.created_at = datetime(),
                                                e.doc_names = $doc_names
                                    ON MATCH SET e.updated_at = datetime(),
                                                e.frequency = COALESCE(e.frequency, 0) + 1,
                                                e.doc_names = $doc_names
                                    """,
                                    entity_id=f"entity_{hash((entity_text, entity_type))}",
                                    name=entity_text,
                                    type=entity_type,
                                    normalized=normalized_text,
                                    doc_names=[doc_name]
                                )

                            # Link entity to document (create relationship only if it doesn't exist)
                            session.run(
                                """
                                MATCH (d:Document {doc_id: $doc_id})
                                MATCH (e:Entity {name: $name, type: $type})
                                MERGE (d)-[:HAS_ENTITY]->(e)
                                """,
                                doc_id=doc_id,
                                name=entity_text,
                                type=entity_type
                            )
                        except Exception as e:
                            logger.error(f"Error storing entity '{entity_text}' (type: {entity_type}): {str(e)}")
                            # Continue with the next entity
                            continue
        except Exception as e:
            logger.error(f"Error in _store_entities for document {doc_id}: {str(e)}")
            # We still want to proceed with other enrichment steps, so we just log the error

    def _store_concepts(self, doc_id: str, concepts: List[Dict[str, Any]]):
        """Store concepts in Neo4j and link them to the document"""
        if not concepts:
            return

        try:
            # Get document name for metadata
            with self.driver.session() as session:
                doc_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.name as doc_name
                    """,
                    doc_id=doc_id
                )
                record = doc_result.single()
                doc_name = record["doc_name"] if record else "Unknown Document"

                # Check if Concept label exists in the database
                labels_result = session.run(
                    """
                    CALL db.labels() YIELD label
                    RETURN collect(label) as labels
                    """
                )
                node_labels = labels_result.single()["labels"]

                # Create Concept label if it doesn't exist
                if "Concept" not in node_labels:
                    session.run(
                        """
                        CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE
                        """
                    )
                    logger.info("Created Concept label with constraint")

            with self.driver.session() as session:
                for concept in concepts:
                    concept_text = concept["text"]
                    lemma = concept.get("lemma", concept_text.lower())
                    concept_type = concept.get("type", "general")

                    try:
                        # Use OPTIONAL MATCH for safer querying of properties that might not exist
                        check_result = session.run(
                            """
                            MATCH (c:Concept {name: $name})
                            RETURN CASE WHEN c.doc_names IS NOT NULL THEN c.doc_names ELSE [] END as doc_names
                            """,
                            name=concept_text
                        )

                        existing_record = check_result.single()
                        if existing_record:
                            # Concept exists, let's update doc_names
                            doc_names = existing_record["doc_names"] or []
                            if isinstance(doc_names, list) and doc_name not in doc_names:
                                doc_names.append(doc_name)
                            else:
                                doc_names = [doc_name]

                            # Use MERGE to handle uniqueness constraints
                            session.run(
                                """
                                MERGE (c:Concept {name: $name})
                                ON CREATE SET c.id = $concept_id,
                                            c.type = $type,
                                            c.lemma = $lemma,
                                            c.created_at = datetime(),
                                            c.doc_names = $doc_names
                                ON MATCH SET c.updated_at = datetime(),
                                            c.frequency = COALESCE(c.frequency, 0) + 1,
                                            c.doc_names = $doc_names
                                """,
                                concept_id=f"concept_{hash(concept_text.lower())}",
                                name=concept_text,
                                type=concept_type,
                                lemma=lemma,
                                doc_names=doc_names
                            )
                        else:
                            # New concept, set doc_names to [doc_name]
                            session.run(
                                """
                                MERGE (c:Concept {name: $name})
                                ON CREATE SET c.id = $concept_id,
                                            c.type = $type,
                                            c.lemma = $lemma,
                                            c.created_at = datetime(),
                                            c.doc_names = $doc_names
                                ON MATCH SET c.updated_at = datetime(),
                                            c.frequency = COALESCE(c.frequency, 0) + 1,
                                            c.doc_names = $doc_names
                                """,
                                concept_id=f"concept_{hash(concept_text.lower())}",
                                name=concept_text,
                                type=concept_type,
                                lemma=lemma,
                                doc_names=[doc_name]
                            )

                        # Link concept to document (create relationship only if it doesn't exist)
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            MATCH (c:Concept {name: $name})
                            MERGE (d)-[:HAS_CONCEPT]->(c)
                            """,
                            doc_id=doc_id,
                            name=concept_text
                        )
                    except Exception as e:
                        logger.error(f"Error storing concept '{concept_text}': {str(e)}")
                        # Continue with the next concept
                        continue
        except Exception as e:
            logger.error(f"Error in _store_concepts for document {doc_id}: {str(e)}")

    def _store_regulatory_items(self, doc_id: str, regulatory: Dict[str, Any]):
        """Store regulatory items in Neo4j and link them to the document"""
        if not regulatory:
            return

        try:
            # Get document name for metadata
            with self.driver.session() as session:
                doc_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.name as doc_name
                    """,
                    doc_id=doc_id
                )
                record = doc_result.single()
                doc_name = record["doc_name"] if record else "Unknown Document"

                # First check which node labels exist in the database to avoid warnings
                labels_result = session.run(
                    """
                    CALL db.labels() YIELD label
                    RETURN collect(label) as labels
                    """
                )
                node_labels = labels_result.single()["labels"]
                logger.info(f"Available node labels: {node_labels}")

            with self.driver.session() as session:
                # Store regulatory summary
                summary = regulatory.get("summary", {})
                if summary:
                    try:
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            SET d.regulatory_score = $score,
                                d.legislation_count = $legislation,
                                d.deadline_count = $deadlines
                            """,
                            doc_id=doc_id,
                            score=summary.get("regulatory_score", 0),
                            legislation=summary.get("legislation_count", 0),
                            deadlines=summary.get("deadline_count", 0)
                        )
                    except Exception as e:
                        logger.error(f"Error storing regulatory summary: {str(e)}")

                # Store legislation references
                for legislation in regulatory.get("legislation_references", []):
                    try:
                        reference = legislation.get("reference", "")
                        leg_type = legislation.get("type", "Unknown")
                        leg_text = legislation.get("text", "")

                        # Create Legislation label if it doesn't exist
                        if "Legislation" not in node_labels:
                            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Legislation) REQUIRE l.reference IS UNIQUE")
                            node_labels.append("Legislation")
                            logger.info("Created Legislation label and constraint")

                        # Check if legislation already exists and has doc_names
                        check_result = session.run(
                            """
                            MATCH (l:Legislation {reference: $reference, type: $type})
                            RETURN CASE WHEN l.doc_names IS NOT NULL THEN l.doc_names ELSE [] END as doc_names
                            """,
                            reference=reference,
                            type=leg_type
                        )

                        existing_record = check_result.single()
                        if existing_record:
                            # Legislation exists, let's update doc_names
                            doc_names = existing_record["doc_names"] or []
                            if isinstance(doc_names, list) and doc_name not in doc_names:
                                doc_names.append(doc_name)
                            else:
                                doc_names = [doc_name]

                            # Use MERGE for legislation references
                            session.run(
                                """
                                MERGE (l:Legislation {reference: $reference, type: $type})
                                ON CREATE SET l.id = $leg_id,
                                            l.text = $text,
                                            l.created_at = datetime(),
                                            l.doc_names = $doc_names
                                ON MATCH SET l.updated_at = datetime(),
                                            l.frequency = COALESCE(l.frequency, 0) + 1,
                                            l.doc_names = $doc_names
                                """,
                                leg_id=f"leg_{hash((reference, leg_type))}",
                                reference=reference,
                                type=leg_type,
                                text=leg_text,
                                doc_names=doc_names
                            )
                        else:
                            # New legislation, set doc_names to [doc_name]
                            session.run(
                                """
                                MERGE (l:Legislation {reference: $reference, type: $type})
                                ON CREATE SET l.id = $leg_id,
                                            l.text = $text,
                                            l.created_at = datetime(),
                                            l.doc_names = $doc_names
                                ON MATCH SET l.updated_at = datetime(),
                                            l.frequency = COALESCE(l.frequency, 0) + 1,
                                            l.doc_names = $doc_names
                                """,
                                leg_id=f"leg_{hash((reference, leg_type))}",
                                reference=reference,
                                type=leg_type,
                                text=leg_text,
                                doc_names=[doc_name]
                            )

                        # Link legislation to document
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            MATCH (l:Legislation {reference: $reference, type: $type})
                            MERGE (d)-[:REFERENCES_LEGISLATION]->(l)
                            """,
                            doc_id=doc_id,
                            reference=reference,
                            type=leg_type
                        )
                    except Exception as e:
                        logger.error(f"Error storing legislation reference '{legislation.get('reference', 'Unknown')}': {str(e)}")
                        continue

                # Store requirements - only if Requirement label exists or create it
                if regulatory.get("requirements", []):
                    # Create Requirement label if it doesn't exist
                    if "Requirement" not in node_labels:
                        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE")
                        node_labels.append("Requirement")
                        logger.info("Created Requirement label and constraint")

                    for req in regulatory.get("requirements", []):
                        try:
                            req_text = req.get("text", "")
                            requirement = req.get("requirement", "")

                            # Check if requirement already exists and has doc_names
                            check_result = session.run(
                                """
                                MATCH (r:Requirement {text: $text})
                                RETURN CASE WHEN r.doc_names IS NOT NULL THEN r.doc_names ELSE [] END as doc_names
                                """,
                                text=req_text
                            )

                            existing_record = check_result.single()
                            if existing_record:
                                # Requirement exists, let's update doc_names
                                doc_names = existing_record["doc_names"] or []
                                if isinstance(doc_names, list) and doc_name not in doc_names:
                                    doc_names.append(doc_name)
                                else:
                                    doc_names = [doc_name]

                                # Use MERGE for requirements
                                session.run(
                                    """
                                    MERGE (r:Requirement {text: $text})
                                    ON CREATE SET r.id = $req_id,
                                                r.requirement = $requirement,
                                                r.created_at = datetime(),
                                                r.doc_names = $doc_names
                                    ON MATCH SET r.updated_at = datetime(),
                                                r.frequency = COALESCE(r.frequency, 0) + 1,
                                                r.doc_names = $doc_names
                                    """,
                                    req_id=f"req_{hash(req_text)}",
                                    text=req_text,
                                    requirement=requirement,
                                    doc_names=doc_names
                                )
                            else:
                                # New requirement, set doc_names to [doc_name]
                                session.run(
                                    """
                                    MERGE (r:Requirement {text: $text})
                                    ON CREATE SET r.id = $req_id,
                                                r.requirement = $requirement,
                                                r.created_at = datetime(),
                                                r.doc_names = $doc_names
                                    ON MATCH SET r.updated_at = datetime(),
                                                r.frequency = COALESCE(r.frequency, 0) + 1,
                                                r.doc_names = $doc_names
                                    """,
                                    req_id=f"req_{hash(req_text)}",
                                    text=req_text,
                                    requirement=requirement,
                                    doc_names=[doc_name]
                                )

                            # Link requirement to document
                            session.run(
                                """
                                MATCH (d:Document {doc_id: $doc_id})
                                MATCH (r:Requirement {text: $text})
                                MERGE (d)-[:HAS_REQUIREMENT]->(r)
                                """,
                                doc_id=doc_id,
                                text=req_text
                            )
                        except Exception as e:
                            logger.error(f"Error storing requirement: {str(e)}")
                            continue

                # Store deadlines if present - only if Deadline label exists or create it
                if regulatory.get("deadlines", []):
                    # Create Deadline label if it doesn't exist
                    if "Deadline" not in node_labels:
                        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (dl:Deadline) REQUIRE dl.id IS UNIQUE")
                        node_labels.append("Deadline")
                        logger.info("Created Deadline label and constraint")

                    for deadline in regulatory.get("deadlines", []):
                        try:
                            deadline_text = deadline.get("text", "")
                            deadline_date = deadline.get("date", "")
                            deadline_type = deadline.get("type", "unknown")

                            # Check if deadline already exists and has doc_names
                            check_result = session.run(
                                """
                                MATCH (dl:Deadline {text: $text})
                                RETURN CASE WHEN dl.doc_names IS NOT NULL THEN dl.doc_names ELSE [] END as doc_names
                                """,
                                text=deadline_text
                            )

                            existing_record = check_result.single()
                            if existing_record:
                                # Deadline exists, let's update doc_names
                                doc_names = existing_record["doc_names"] or []
                                if isinstance(doc_names, list) and doc_name not in doc_names:
                                    doc_names.append(doc_name)
                                else:
                                    doc_names = [doc_name]

                                # Use MERGE for deadlines
                                session.run(
                                    """
                                    MERGE (dl:Deadline {text: $text})
                                    ON CREATE SET dl.id = $deadline_id,
                                                dl.date = $date,
                                                dl.type = $type,
                                                dl.created_at = datetime(),
                                                dl.doc_names = $doc_names
                                    ON MATCH SET dl.updated_at = datetime(),
                                                dl.frequency = COALESCE(dl.frequency, 0) + 1,
                                                dl.doc_names = $doc_names
                                    """,
                                    deadline_id=f"deadline_{hash(deadline_text)}",
                                    text=deadline_text,
                                    date=deadline_date,
                                    type=deadline_type,
                                    doc_names=doc_names
                                )
                            else:
                                # New deadline, set doc_names to [doc_name]
                                session.run(
                                    """
                                    MERGE (dl:Deadline {text: $text})
                                    ON CREATE SET dl.id = $deadline_id,
                                                dl.date = $date,
                                                dl.type = $type,
                                                dl.created_at = datetime(),
                                                dl.doc_names = $doc_names
                                    ON MATCH SET dl.updated_at = datetime(),
                                                dl.frequency = COALESCE(dl.frequency, 0) + 1,
                                                dl.doc_names = $doc_names
                                    """,
                                    deadline_id=f"deadline_{hash(deadline_text)}",
                                    text=deadline_text,
                                    date=deadline_date,
                                    type=deadline_type,
                                    doc_names=[doc_name]
                                )

                            # Link deadline to document
                            session.run(
                                """
                                MATCH (d:Document {doc_id: $doc_id})
                                MATCH (dl:Deadline {text: $text})
                                MERGE (d)-[:HAS_DEADLINE]->(dl)
                                """,
                                doc_id=doc_id,
                                text=deadline_text
                            )
                        except Exception as e:
                            logger.error(f"Error storing deadline: {str(e)}")
                            continue

        except Exception as e:
            logger.error(f"Error in _store_regulatory_items for document {doc_id}: {str(e)}")
