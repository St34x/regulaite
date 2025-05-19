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
        
        doc_id_for_log = context.get("doc_id", "unknown_doc")

        # Detect language if multilingual support is enabled
        language_info = None
        lang_code = None

        if self.multilingual:
            # Log a snippet of the text being language-detected by EnrichmentPipeline
            text_snippet_for_log = text[:min(200, len(text))].replace("\n", " ")
            logger.info(f"EnrichmentPipeline: Detecting language for doc_id: {doc_id_for_log} from text snippet: '{text_snippet_for_log}...'")
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']
            logger.info(f"EnrichmentPipeline: Detected language for doc_id: {doc_id_for_log} is '{lang_code}' (Name: {language_info.get('language_name')}, Confidence: {language_info.get('confidence')})")

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
        """Store entities in Neo4j and link them to the document using batch operations"""
        if not entities:
            return

        try:
            doc_name = "Unknown Document"
            with self.driver.session() as session:
                # Get document name for metadata
                doc_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.name as doc_name
                    """,
                    doc_id=doc_id
                )
                record = doc_result.single()
                if record and record["doc_name"]:
                    doc_name = record["doc_name"]

                # Ensure Entity label, constraints, and indices exist
                # This is done once before batching
                labels_result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                node_labels = labels_result.single()["labels"]
                if "Entity" not in node_labels:
                    session.run(
                        """
                        CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
                        CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name);
                        CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type);
                        """
                    )
                    logger.info("Created Entity label with constraints and indices")
            
            entities_batch = []
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_text = entity["text"]
                    normalized_text = entity.get("normalized", entity["text"].lower())
                    # Create a consistent ID based on entity text and type
                    entity_id = f"entity_{hash((entity_text, entity_type))}"
                    entities_batch.append({
                        "entity_id": entity_id,
                        "name": entity_text,
                        "type": entity_type,
                        "normalized": normalized_text,
                        "doc_id": doc_id,
                        "doc_name": doc_name 
                    })

            if not entities_batch:
                return

            with self.driver.session() as session:
                session.run(
                    """
                    UNWIND $entities_batch AS props
                    
                    MERGE (e:Entity {name: props.name, type: props.type})
                    ON CREATE SET 
                        e.id = props.entity_id,
                        e.normalized = props.normalized,
                        e.created_at = datetime(),
                        e.doc_names = [props.doc_name],
                        e.frequency = 1
                    ON MATCH SET 
                        e.updated_at = datetime(),
                        e.frequency = COALESCE(e.frequency, 0) + 1,
                        // Add doc_name to list if not already present, handling null doc_names
                        e.doc_names = CASE
                                        WHEN props.doc_name IN COALESCE(e.doc_names, []) THEN e.doc_names
                                        ELSE COALESCE(e.doc_names, []) + [props.doc_name]
                                      END
                    
                    WITH e, props.doc_id AS doc_id_val
                    MATCH (d:Document {doc_id: doc_id_val})
                    MERGE (d)-[r:HAS_ENTITY]->(e)
                    ON CREATE SET r.created_at = datetime()
                    ON MATCH SET r.updated_at = datetime()
                    """,
                    entities_batch=entities_batch
                )
            logger.info(f"Stored/updated {len(entities_batch)} entities for document {doc_id} using batch operation.")

        except Exception as e:
            logger.error(f"Error in _store_entities for document {doc_id}: {str(e)}")
            # We still want to proceed with other enrichment steps, so we just log the error

    def _store_concepts(self, doc_id: str, concepts: List[Dict[str, Any]]):
        """Store concepts in Neo4j and link them to the document using batch operations"""
        if not concepts:
            return

        try:
            doc_name = "Unknown Document"
            with self.driver.session() as session:
                # Get document name for metadata
                doc_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d.name as doc_name
                    """,
                    doc_id=doc_id
                )
                record = doc_result.single()
                if record and record["doc_name"]:
                    doc_name = record["doc_name"]
                
                # Ensure Concept label and constraint exist
                labels_result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                node_labels = labels_result.single()["labels"]
                if "Concept" not in node_labels:
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE") 
                    session.run("CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.name)")
                    logger.info("Created Concept label with constraint and index")

            concepts_batch = []
            for concept in concepts:
                concept_text = concept["text"]
                lemma = concept.get("lemma", concept_text.lower())
                concept_type = concept.get("type", "general")
                concept_id = f"concept_{hash(concept_text.lower())}" 
                concepts_batch.append({
                    "concept_id": concept_id,
                    "name": concept_text,
                    "type": concept_type,
                    "lemma": lemma,
                    "doc_id": doc_id,
                    "doc_name": doc_name
                })
            
            if not concepts_batch:
                return

            with self.driver.session() as session:
                session.run(
                    """
                    UNWIND $concepts_batch AS props
                    
                    MERGE (c:Concept {name: props.name}) 
                    ON CREATE SET 
                        c.id = props.concept_id,
                        c.type = props.type,
                        c.lemma = props.lemma,
                        c.created_at = datetime(),
                        c.doc_names = [props.doc_name],
                        c.frequency = 1
                    ON MATCH SET 
                        c.updated_at = datetime(),
                        c.frequency = COALESCE(c.frequency, 0) + 1,
                        c.doc_names = CASE
                                        WHEN props.doc_name IN COALESCE(c.doc_names, []) THEN c.doc_names
                                        ELSE COALESCE(c.doc_names, []) + [props.doc_name]
                                      END
                        // c.type = props.type, // Uncomment if these should be updated on match
                        // c.lemma = props.lemma
                    
                    WITH c, props.doc_id AS doc_id_val
                    MATCH (d:Document {doc_id: doc_id_val})
                    MERGE (d)-[r:HAS_CONCEPT]->(c)
                    ON CREATE SET r.created_at = datetime()
                    ON MATCH SET r.updated_at = datetime()
                    """,
                    concepts_batch=concepts_batch
                )
            logger.info(f"Stored/updated {len(concepts_batch)} concepts for document {doc_id} using batch operation.")

        except Exception as e:
            logger.error(f"Error in _store_concepts for document {doc_id}: {str(e)}")

    def _store_regulatory_items(self, doc_id: str, regulatory: Dict[str, Any]):
        """Store regulatory items in Neo4j and link them to the document, using batch operations for list items."""
        if not regulatory:
            return

        try:
            doc_name = "Unknown Document"
            # Store regulatory summary (this is a single update on the document node)
            summary = regulatory.get("summary", {})
            if summary:
                 with self.driver.session() as session: 
                    session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        SET d.regulatory_score = $score,
                            d.legislation_count = $legislation_count,
                            d.deadline_count = $deadlines
                        """,
                        doc_id=doc_id,
                        score=summary.get("regulatory_score", 0),
                        legislation_count=summary.get("legislation_count", 0),
                        deadlines=summary.get("deadline_count", 0)
                    )
                    doc_result = session.run("MATCH (d:Document {doc_id: $doc_id}) RETURN d.name as doc_name", doc_id=doc_id)
                    record = doc_result.single()
                    if record and record["doc_name"]:
                        doc_name = record["doc_name"]

            all_node_labels = []
            with self.driver.session() as session:
                labels_result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                record = labels_result.single()
                if record and record["labels"]:
                    all_node_labels = record["labels"]

            legislation_list = regulatory.get("legislation_references", [])
            if legislation_list:
                with self.driver.session() as session: 
                    if "Legislation" not in all_node_labels:
                        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Legislation) REQUIRE l.reference IS UNIQUE")
                        session.run("CREATE INDEX IF NOT EXISTS FOR (l:Legislation) ON (l.reference)")
                        session.run("CREATE INDEX IF NOT EXISTS FOR (l:Legislation) ON (l.type)") 
                        logger.info("Created Legislation label with constraint and index(es)")
                        all_node_labels.append("Legislation") 

                legislation_batch = []
                for leg_item in legislation_list:
                    reference = leg_item.get("reference", "")
                    leg_type = leg_item.get("type", "Unknown")
                    leg_text = leg_item.get("text", "")
                    leg_id = f"leg_{hash((reference, leg_type))}" 
                    if not reference: 
                        logger.warning(f"Skipping legislation with empty reference for doc {doc_id}")
                        continue
                    legislation_batch.append({
                        "leg_id": leg_id,
                        "reference": reference,
                        "type": leg_type,
                        "text": leg_text,
                        "doc_id": doc_id,
                        "doc_name": doc_name
                    })
                
                if legislation_batch:
                    with self.driver.session() as session:
                        session.run(
                            """
                            UNWIND $legislation_batch AS props
                            
                            MERGE (l:Legislation {reference: props.reference, type: props.type}) 
                            ON CREATE SET 
                                l.id = props.leg_id,
                                l.text = props.text,
                                l.created_at = datetime(),
                                l.doc_names = [props.doc_name],
                                l.frequency = 1
                            ON MATCH SET 
                                l.updated_at = datetime(),
                                l.frequency = COALESCE(l.frequency, 0) + 1,
                                l.doc_names = CASE
                                                WHEN props.doc_name IN COALESCE(l.doc_names, []) THEN l.doc_names
                                                ELSE COALESCE(l.doc_names, []) + [props.doc_name]
                                              END
                                // l.text = props.text // uncomment if text should be updated on match
                            
                            WITH l, props.doc_id AS doc_id_val
                            MATCH (d:Document {doc_id: doc_id_val})
                            MERGE (d)-[r:REFERENCES_LEGISLATION]->(l)
                            ON CREATE SET r.created_at = datetime()
                            ON MATCH SET r.updated_at = datetime()
                            """,
                            legislation_batch=legislation_batch
                        )
                        logger.info(f"Stored/updated {len(legislation_batch)} legislation items for document {doc_id} using batch.")

            requirements_list = regulatory.get("requirements", [])
            if requirements_list:
                with self.driver.session() as session: 
                    if "Requirement" not in all_node_labels:
                        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.text IS UNIQUE")
                        session.run("CREATE INDEX IF NOT EXISTS FOR (r:Requirement) ON (r.text)") 
                        logger.info("Created Requirement label with constraint and index")
                        all_node_labels.append("Requirement") 

                requirements_batch = []
                for req_item in requirements_list:
                    req_text = req_item.get("text", "") 
                    requirement_detail = req_item.get("requirement", "") 
                    req_id = f"req_{hash(req_text)}"
                    if not req_text: 
                        logger.warning(f"Skipping requirement with empty text for doc {doc_id}")
                        continue
                    requirements_batch.append({
                        "req_id": req_id,
                        "text": req_text,
                        "requirement_detail": requirement_detail,
                        "doc_id": doc_id,
                        "doc_name": doc_name
                    })

                if requirements_batch:
                    with self.driver.session() as session:
                        session.run(
                            """
                            UNWIND $requirements_batch AS props
                            
                            MERGE (r:Requirement {text: props.text}) 
                            ON CREATE SET 
                                r.id = props.req_id, 
                                r.requirement = props.requirement_detail,
                                r.created_at = datetime(),
                                r.doc_names = [props.doc_name],
                                r.frequency = 1
                            ON MATCH SET 
                                r.updated_at = datetime(),
                                r.frequency = COALESCE(r.frequency, 0) + 1,
                                r.doc_names = CASE
                                                WHEN props.doc_name IN COALESCE(r.doc_names, []) THEN r.doc_names
                                                ELSE COALESCE(r.doc_names, []) + [props.doc_name]
                                              END
                                // r.requirement = props.requirement_detail // uncomment if detail should be updated
                            
                            WITH r, props.doc_id AS doc_id_val
                            MATCH (d:Document {doc_id: doc_id_val})
                            MERGE (d)-[rel:HAS_REQUIREMENT]->(r) 
                            ON CREATE SET rel.created_at = datetime()
                            ON MATCH SET rel.updated_at = datetime()
                            """,
                            requirements_batch=requirements_batch
                        )
                        logger.info(f"Stored/updated {len(requirements_batch)} requirement items for document {doc_id} using batch.")
        
        except Exception as e:
            logger.error(f"Error in _store_regulatory_items for document {doc_id}: {str(e)}")
