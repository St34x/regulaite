import logging
import os # For environment variables for credentials
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple, Union, cast
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, Record, Result
from neo4j.graph import Node as Neo4jNode
from pydantic import BaseModel
import datetime
import json

from ..graph_components.nodes import Node, ConceptNode, DocumentNode, QueryNode, ResponseNode
from ..graph_components.edges import Edge, RelationshipType

logger = logging.getLogger(__name__)

# Configuration - Load from environment variables or a config file
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

TNode = TypeVar('TNode', bound=Node)
TEdge = TypeVar('TEdge', bound=Edge)

class GraphInterface:
    """Interface for interacting with the Neo4j graph database."""

    _driver: Optional[AsyncDriver] = None

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        logger.info(f"GraphInterface initialized for URI: {self.uri}")

    def _sanitize_properties(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize properties to ensure they're Neo4j compatible.
        Neo4j only accepts primitive types (string, int, float, bool) and arrays of primitive types.
        This method converts complex objects to string representations.
        
        Args:
            props: Dictionary of properties to sanitize
            
        Returns:
            Dictionary with sanitized properties
        """
        sanitized = {}
        for key, value in props.items():
            # Skip None values
            if value is None:
                continue
                
            # Handle primitive types
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            # Handle lists/arrays
            elif isinstance(value, list):
                # Check if all elements are primitive
                all_primitive = all(isinstance(item, (str, int, float, bool)) for item in value)
                if all_primitive:
                    sanitized[key] = value
                else:
                    # Convert non-primitive lists to string representation
                    sanitized[key] = str(value)
            # Handle dictionaries and other complex objects
            else:
                # Convert to string representation
                sanitized[key] = str(value)
                
        return sanitized

    async def connect(self):
        """Establishes the connection to the Neo4j database."""
        if not self._driver or self._driver.closed(): # Check if driver is closed
            try:
                self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
                await self._driver.verify_connectivity()
                logger.info(f"Successfully connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self._driver = None
                raise

    async def close(self):
        """Closes the connection to the Neo4j database."""
        if self._driver and not self._driver.closed():
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    async def _ensure_connection(self):
        """Ensure we have a working Neo4j connection."""
        if not self._driver:
            await self.connect()
        
        # Check for connection validity in a better way
        try:
            # Simple test query to verify connection
            await self._driver.execute_query("RETURN 1 as n")
            return True
        except Exception as e:
            logger.warning(f"Neo4j connection test failed: {e}. Reconnecting...")
            await self.connect()
            return True

    async def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Record]:
        await self._ensure_connection()
        async with self._driver.session() as session: # type: AsyncSession
            try:
                result: Result = await session.run(query, parameters)
                records = [record async for record in result]
                return records
            except Exception as e:
                logger.error(f"Error executing Cypher query \"{query[:100]}...\": {e}", exc_info=True)
                raise

    def _props_to_cypher_string(self, props: Dict[str, Any], param_name: str = "props_to_set") -> str:
        """Converts a properties dictionary to a Cypher SET string parts, avoiding setting id or label."""
        set_clauses = []
        for key, value in props.items():
            if key not in ['id', 'label', 'created_at', 'updated_at']: # Avoid overwriting immutable or auto fields
                set_clauses.append(f"n.{key} = ${param_name}.{key}")
        return ", ".join(set_clauses)

    def _serialize_node_props(self, node: BaseModel) -> Dict[str, Any]:
        """
        Converts a node into a dictionary suitable for Neo4j.
        Ensures all values are properly serialized.
        """
        # Convert the node to a dict
        node_dict = node.dict()
        
        # Handle special properties
        for key, value in node_dict.items():
            # Convert datetime objects to ISO format strings
            if isinstance(value, datetime.datetime):
                node_dict[key] = value.isoformat()
            # Convert dictionary attributes to JSON strings to avoid Neo4j type issues
            elif isinstance(value, dict):
                node_dict[key] = json.dumps(value)
            # Handle lists with potential dict items
            elif isinstance(value, list):
                # Check if list contains dictionaries that need serialization
                if any(isinstance(item, dict) for item in value):
                    node_dict[key] = json.dumps(value)
        
        # Labels should match the class name
        node_dict["labels"] = [node.__class__.__name__]
        
        return node_dict

    def _deserialize_neo4j_node(self, neo4j_node: Dict[str, Any], target_class: Type[TNode]) -> TNode:
        """
        Converts a Neo4j node (as dictionary) to a Pydantic model instance.
        Handles proper deserialization of specialized types.
        """
        props = dict(neo4j_node)
        
        # Process properties before creating the model
        for key, value in list(props.items()):
            # Handle JSON strings stored as attributes or metadata
            if key in ["attributes", "metadata"] and isinstance(value, str):
                try:
                    props[key] = json.loads(value)
                except json.JSONDecodeError:
                    # If it's not valid JSON, keep as string
                    pass
            # Handle datetime strings
            elif key in ["created_at", "updated_at"] and isinstance(value, str):
                try:
                    props[key] = datetime.datetime.fromisoformat(value)
                except ValueError:
                    # If it's not a valid datetime string, keep as string
                    pass
        
        # Remove Neo4j specific fields
        if "labels" in props:
            del props["labels"]
        
        # Instantiate the target class
        return target_class(**props)

    async def add_node(self, node_data: TNode) -> TNode:
        """
        Add a node to the graph database.
        
        Args:
            node_data: A BaseNode instance (DocumentNode, QueryNode, etc.)
            
        Returns:
            Updated node with database ID
        """
        # Prepare node properties
        node_props = self._serialize_node_props(node_data)
        
        # Extract the label from class name
        label = node_data.__class__.__name__
        
        # Check if this is an update (node has an ID)
        if hasattr(node_data, "id") and node_data.id:
            # Update existing node
            query = f"""
            MATCH (n:{label} {{id: $id}})
            SET n += $props
            RETURN n
            """
            results = await self._execute_query(query, {
                "id": node_data.id,
                "props": {k: v for k, v in node_props.items() if k != "id"}
            })
        else:
            # Create new node
            query = f"""
            CREATE (n:{label} $props)
            RETURN n
            """
            results = await self._execute_query(query, {"props": node_props})
        
        if not results:
            raise ValueError(f"Failed to add/update node: {node_data}")
        
        # Return the updated node with database-assigned properties
        db_props = results[0].get("n", {})
        if isinstance(db_props, Neo4jNode):
            db_props = dict(db_props)
        
        # Deserialize the node
        return self._deserialize_neo4j_node(db_props, type(node_data))

    async def update_node_properties(self, node_id: str, properties_to_update: Dict[str, Any], node_label: Optional[str]=None) -> bool:
        if not properties_to_update:
            logger.warning(f"No properties provided to update for node ID: {node_id}")
            return False
        
        # Sanitize properties
        sanitized_props = self._sanitize_properties(properties_to_update)
        
        set_clauses = []
        for key in sanitized_props.keys():
            # Prevent trying to update 'id' or 'label' via this method
            if key in ['id', 'label', 'created_at']:
                logger.warning(f"Skipping disallowed property update for '{key}' on node {node_id}")
                continue
            set_clauses.append(f"n.{key} = $props.{key}")
        
        if not set_clauses:
            logger.info(f"No valid properties to update for node {node_id} after filtering.")
            return False

        match_clause = f"MATCH (n {{id: $node_id}})" if not node_label else f"MATCH (n:{node_label} {{id: $node_id}})"
        cypher = (
            f"{match_clause} "
            f"SET {', '.join(set_clauses)}, n.updated_at = datetime() "
            f"RETURN n"
        )
        params = {"node_id": node_id, "props": sanitized_props}
        records = await self._execute_query(cypher, params)
        if records and records[0]["n"]:
            logger.info(f"Updated properties for node ID: {node_id}")
            return True
        logger.warning(f"Failed to update properties for node ID: {node_id} (node not found or no properties changed).")
        return False

    async def add_edge(self, edge_data: TEdge) -> TEdge:
        """Adds an edge between two nodes."""
        source_id = edge_data.source_node_id
        target_id = edge_data.target_node_id
        relationship_type = edge_data.type
        
        # Prepare edge properties
        edge_props = edge_data.dict(exclude={"source_node_id", "target_node_id", "type"})
        
        # Handle special properties
        for key, value in edge_props.items():
            if isinstance(value, datetime.datetime):
                edge_props[key] = value.isoformat()
            elif isinstance(value, dict):
                edge_props[key] = json.dumps(value)
        
        cypher = """
        MATCH (source) WHERE source.id = $source_id
        MATCH (target) WHERE target.id = $target_id
        CREATE (source)-[r:$type $props]->(target)
        RETURN r
        """
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "type": relationship_type,
            "props": edge_props
        }
        
        try:
            results = await self._execute_query(cypher, params)
            if results:
                logger.info(f"Added edge: {source_id} -[{relationship_type}]-> {target_id}")
                return edge_data
            else:
                logger.error(f"Failed to add edge: {source_id} -[{relationship_type}]-> {target_id}")
                raise ValueError(f"Failed to add edge")
        except Exception as e:
            logger.error(f"Error adding edge: {e}")
            raise

    async def get_node_by_id(self, node_id: str, node_type: Type[TNode]) -> Optional[TNode]:
        try:
            node_label = node_type().label
        except AttributeError:
            logger.error(f"Cannot determine label for node_type {node_type.__name__}. It must have a 'label' attribute.")
            return None
        except Exception as e:
            logger.error(f"Error instantiating {node_type.__name__} to get label: {e}")
            return None

        cypher = f"MATCH (n:{node_label} {{id: $node_id}}) RETURN n"
        records = await self._execute_query(cypher, {"node_id": node_id})
        if records and records[0]["n"]:
            node_props = dict(records[0]["n"])
            for key, value in node_props.items():
                if hasattr(value, 'to_native'):
                    node_props[key] = value.to_native()
            return node_type(**node_props)
        return None

    async def update_document_retrieval_stats(self, document_id: str, increment_count: int = 1) -> bool:
        cypher = (
            f"MATCH (d:Document {{id: $document_id}}) "
            f"SET d.retrieval_count = coalesce(d.retrieval_count, 0) + $increment, d.last_retrieved_at = datetime() "
            f"RETURN d.retrieval_count"
        )
        records = await self._execute_query(cypher, {"document_id": document_id, "increment": increment_count})
        if records:
            logger.info(f"Updated retrieval stats for Document ID: {document_id}. New count: {records[0][0]}")
            return True
        logger.warning(f"Failed to update retrieval stats for Document ID: {document_id} (not found or no change).")
        return False

    async def update_edge_properties(self, edge_type: str, source_node_id: str, target_node_id: str, properties_update: Dict[str, Any]) -> bool:
        """
        Updates the properties of an edge identified by type, source, and target nodes.
        
        Args:
            edge_type: The type of the edge (e.g., "RETRIEVED_FOR")
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            properties_update: Dictionary of property names and values to update
            
        Returns:
            True if the edge was found and updated, False otherwise
        """
        if not properties_update:
            logger.warning(f"No properties provided to update for edge from {source_node_id} to {target_node_id}")
            return False
        
        try:
            # Convert string edge_type to enum if needed
            if isinstance(edge_type, str):
                try:
                    rel_type = RelationshipType[edge_type]
                    edge_type = rel_type.value
                except (KeyError, AttributeError):
                    # If not found in enum, use the string directly
                    pass
                    
            # Build SET clauses for properties
            set_clauses = []
            for key in properties_update.keys():
                if key not in ['id', 'created_at']:
                    set_clauses.append(f"r.{key} = $props.{key}")
            
            if not set_clauses:
                logger.info(f"No valid properties to update for edge from {source_node_id} to {target_node_id}")
                return False
                
            # Execute the query
            cypher = (
                f"MATCH (a {{id: $source_id}})-[r:{edge_type}]->(b {{id: $target_id}}) "
                f"SET {', '.join(set_clauses)}, r.updated_at = datetime() "
                f"RETURN r"
            )
            params = {"source_id": source_node_id, "target_id": target_node_id, "props": properties_update}
            records = await self._execute_query(cypher, params)
            
            if records and records[0]["r"]:
                logger.info(f"Updated properties for edge from {source_node_id} to {target_node_id} of type {edge_type}")
                return True
                
            logger.warning(f"Edge from {source_node_id} to {target_node_id} of type {edge_type} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error updating edge properties: {e}", exc_info=True)
            return False

    async def update_edge_properties_by_nodes_and_type(
        self, source_id: str, target_id: str, rel_type: RelationshipType,
        properties_to_update: Dict[str, Any]
    ) -> bool:
        if not properties_to_update:
            return False
        set_clauses = [f"r.{key} = $props.{key}" for key in properties_to_update.keys()]
        cypher = (
            f"MATCH (a {{id: $source_id}})-[r:{rel_type.value}]->(b {{id: $target_id}}) "
            f"SET {', '.join(set_clauses)}, r.updated_at = datetime() "
            f"RETURN r"
        )
        params = {"source_id": source_id, "target_id": target_id, "props": properties_to_update}
        records = await self._execute_query(cypher, params)
        return bool(records)
    
    async def adjust_edge_weight(
        self, source_id: str, target_id: str, rel_type: RelationshipType,
        adjustment: float, weight_property: str = "weight", 
        min_weight: Optional[float] = None, max_weight: Optional[float] = None
    ) -> bool:
        cypher = (
            f"MATCH (a {{id: $source_id}})-[r:{rel_type.value}]->(b {{id: $target_id}}) "
            f"SET r.{weight_property} = coalesce(r.{weight_property}, 0) + $adjustment, r.updated_at = datetime() "
            # Optional clamping
            + (f", r.{weight_property} = CASE WHEN r.{weight_property} < $min_w THEN $min_w ELSE r.{weight_property} END " if min_weight is not None else "")
            + (f", r.{weight_property} = CASE WHEN r.{weight_property} > $max_w THEN $max_w ELSE r.{weight_property} END " if max_weight is not None else "")
            + f"RETURN r.{weight_property}"
        )
        params = {"source_id": source_id, "target_id": target_id, "adjustment": adjustment}
        if min_weight is not None: params["min_w"] = min_weight
        if max_weight is not None: params["max_w"] = max_weight
        
        records = await self._execute_query(cypher, params)
        if records:
            logger.info(f"Adjusted weight for {rel_type.value} edge between {source_id} and {target_id} to {records[0][0]}.")
            return True
        return False

    async def get_query_for_response(self, response_id: str) -> Optional[QueryNode]:
        cypher = (
            f"MATCH (q:Query)-[:{RelationshipType.LED_TO.value}]->(r:Response {{id: $response_id}}) "
            f"RETURN q"
        )
        records = await self._execute_query(cypher, {"response_id": response_id})
        if records and records[0]["q"]:
            props = dict(records[0]["q"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            return QueryNode(**props)
        return None

    async def get_concepts_by_terms(self, terms: List[str], limit_per_term: int = 3) -> List[ConceptNode]:
        """Finds ConceptNodes where the name matches any of the provided terms.
        Note: This is a simple implementation. For better performance with many terms,
        a single Cypher query using IN or UNWIND would be more efficient.
        """
        await self._ensure_connection()
        all_found_concepts: Dict[str, ConceptNode] = {} # Use dict to avoid duplicates by ID

        for term in terms:
            # This uses an existing method that likely queries for exact matches on a 'name' property.
            # If ConceptNode has a different primary text property, adjust find_concepts_by_name_exact or this logic.
            try:
                # Assuming find_concepts_by_name_exact handles its own session/query execution
                concepts = await self.find_concepts_by_name_exact(concept_name=term, limit=limit_per_term)
                for concept in concepts:
                    if concept.id not in all_found_concepts:
                        all_found_concepts[concept.id] = concept
            except Exception as e:
                logger.error(f"Error fetching concepts for term '{term}': {e}", exc_info=True)
                # Continue with other terms
        
        logger.info(f"Found {len(all_found_concepts)} unique concepts for terms: {terms}")
        return list(all_found_concepts.values())

    async def check_relationship_exists(self, source_id: str, target_id: str, rel_type: RelationshipType) -> bool:
        cypher = (
            f"OPTIONAL MATCH (a {{id: $source_id}})-[:{rel_type.value}]->(b {{id: $target_id}}) "
            f"RETURN b IS NOT NULL AS link_exists"
        )
        records = await self._execute_query(cypher, {"source_id": source_id, "target_id": target_id})
        return records[0]["link_exists"] if records else False

    async def find_concepts_by_name_exact(self, concept_name: str, limit: int = 5) -> List[ConceptNode]:
        cypher = "MATCH (c:Concept {name: $name}) RETURN c LIMIT $limit"
        records = await self._execute_query(cypher, {"name": concept_name, "limit": limit})
        concepts = []
        for record in records:
            props = dict(record["c"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            concepts.append(ConceptNode(**props))
        return concepts

    async def get_nodes_linked_from(self, source_node_id: str, relationship_type: RelationshipType, target_node_type: Type[TNode], limit: int = 10) -> List[TNode]:
        try: target_label = target_node_type().label
        except: return []
        cypher = (
            f"MATCH (a {{id: $source_id}})-[:{relationship_type.value}]->(b:{target_label}) "
            f"RETURN b LIMIT $limit"
        )
        records = await self._execute_query(cypher, {"source_id": source_node_id, "limit": limit})
        nodes = []
        for record in records:
            props = dict(record["b"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            nodes.append(target_node_type(**props))
        return nodes

    async def get_nodes_linked_to(self, target_node_id: str, relationship_type: RelationshipType, source_node_type: Type[TNode], limit: int = 10) -> List[TNode]:
        try: source_label = source_node_type().label
        except: return []
        cypher = (
            f"MATCH (a:{source_label})-[:{relationship_type.value}]->(b {{id: $target_node_id}}) "
            f"RETURN a LIMIT $limit"
        )
        records = await self._execute_query(cypher, {"target_node_id": target_node_id, "limit": limit})
        nodes = []
        for record in records:
            props = dict(record["a"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            nodes.append(source_node_type(**props))
        return nodes

    # Methods from previous TODO list needing specific implementation based on use case:
    # - get_docs_from_successful_similar_queries -> Complex, involves similarity + path traversals
    # - find_successful_reformulations -> Needs definition of "successful" and query similarity logic

    # Additional methods to complete the implementation

    async def find_concepts_by_name_fuzzy(self, query_term: str, min_similarity: float = 0.7, limit: int = 5) -> List[ConceptNode]:
        """
        Finds concepts with names similar to the query term using fuzzy matching.
        
        Args:
            query_term: The search term to match concept names against
            min_similarity: Minimum similarity score (0.0-1.0) to include a result
            limit: Maximum number of concepts to return
            
        Returns:
            List of ConceptNode objects with similar names
        """
        # Use Cypher's apoc.text.fuzzyMatch if available
        try:
            # First check if APOC is available
            check_query = "CALL dbms.procedures() YIELD name WHERE name CONTAINS 'apoc.text.fuzzyMatch' RETURN count(*) > 0 AS has_fuzzy"
            records = await self._execute_query(check_query)
            has_fuzzy = records[0]["has_fuzzy"] if records else False
            
            if has_fuzzy:
                # Use APOC's fuzzy matching function
                cypher = """
                MATCH (c:Concept)
                WITH c, apoc.text.fuzzyMatch(c.name, $query_term) AS score
                WHERE score >= $min_similarity
                RETURN c, score
                ORDER BY score DESC
                LIMIT $limit
                """
                params = {"query_term": query_term, "min_similarity": min_similarity, "limit": limit}
            else:
                # Fallback using basic substring matching with CONTAINS
                logger.warning("APOC fuzzy matching not available, falling back to CONTAINS")
                cypher = """
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($query_part)
                OR toLower($query_part) CONTAINS toLower(c.name)
                RETURN c, 0.9 AS score
                LIMIT $limit
                """
                params = {"query_part": query_term, "limit": limit}
                
            records = await self._execute_query(cypher, params)
            concepts = []
            for record in records:
                props = dict(record["c"])
                for key, value in props.items():
                    if hasattr(value, 'to_native'): props[key] = value.to_native()
                # Add the similarity score to the attributes
                if "attributes" not in props:
                    props["attributes"] = {}
                props["attributes"]["match_score"] = record["score"]
                concepts.append(ConceptNode(**props))
            return concepts
        except Exception as e:
            logger.error(f"Error in fuzzy concept search: {e}", exc_info=True)
            return []

    async def find_concepts_by_terms_fuzzy(self, terms: List[str], min_similarity: float = 0.6, limit_per_term: int = 3) -> List[ConceptNode]:
        """
        Find concepts matching multiple terms with fuzzy matching.
        
        Args:
            terms: List of search terms
            min_similarity: Minimum similarity score for matching
            limit_per_term: Maximum number of concepts to return per term
            
        Returns:
            List of matched ConceptNode objects, deduplicated
        """
        await self._ensure_connection()
        all_found_concepts: Dict[str, ConceptNode] = {}  # Use dict to avoid duplicates by ID
        
        for term in terms:
            try:
                # Try fuzzy matching first
                concepts = await self.find_concepts_by_name_fuzzy(
                    query_term=term, 
                    min_similarity=min_similarity,
                    limit=limit_per_term
                )
                
                # If fuzzy search returns nothing, try exact matching
                if not concepts:
                    concepts = await self.find_concepts_by_name_exact(term, limit=limit_per_term)
                
                # Add unique concepts to our result
                for concept in concepts:
                    if concept.id not in all_found_concepts:
                        all_found_concepts[concept.id] = concept
            except Exception as e:
                logger.error(f"Error finding concepts for term '{term}': {e}", exc_info=True)
                # Continue with other terms
        
        logger.info(f"Found {len(all_found_concepts)} unique concepts for terms: {terms}")
        return list(all_found_concepts.values())

    async def get_documents_containing_concept(self, concept_id: str, limit: int = 10) -> List[DocumentNode]:
        """
        Retrieves documents that contain references to a given concept.
        
        Args:
            concept_id: ID of the concept to find in documents
            limit: Maximum number of documents to return
            
        Returns:
            List of DocumentNode objects that contain the concept
        """
        cypher = """
        MATCH (c:Concept {id: $concept_id})<-[:CONTAINS]-(d:Document)
        RETURN d
        LIMIT $limit
        """
        records = await self._execute_query(cypher, {"concept_id": concept_id, "limit": limit})
        
        docs = []
        for record in records:
            props = dict(record["d"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            docs.append(DocumentNode(**props))
        
        return docs

    async def find_successful_reformulations(self, original_query_text: str, success_threshold: float = 0.8, limit: int = 5) -> List[dict]:
        """
        Finds examples of successful query reformulations similar to the given original query.
        
        Args:
            original_query_text: Text of the original query to find similar reformulations for
            success_threshold: Minimum success score to consider a reformulation successful
            limit: Maximum number of successful patterns to return
            
        Returns:
            List of dictionaries with original_query, reformulated_query, and success_score
        """
        # This query finds query nodes that are similar to our input,
        # then looks for reformulations of those queries that led to successful responses
        cypher = """
        // Find similar original queries
        MATCH (q1:Query)
        WHERE q1.original_user_input CONTAINS $query_substring OR $query_substring CONTAINS q1.original_user_input
        
        // Find their reformulations
        MATCH (q1)-[:REFORMULATED_AS]->(q2:Query)
        
        // Find successful responses generated from those reformulations
        MATCH (q2)-[led:LED_TO]->(r:Response)
        WHERE led.success_metric >= $threshold
        
        RETURN q1.original_user_input AS original_query, 
               q2.reformulated_query_text AS reformulated_query,
               led.success_metric AS success_score
        ORDER BY success_score DESC
        LIMIT $limit
        """
        
        # Use some substring of the original query to find similar queries
        query_substring = ' '.join(original_query_text.split()[:5])  # First 5 words as a simple approach
        
        params = {
            "query_substring": query_substring,
            "threshold": success_threshold,
            "limit": limit
        }
        
        try:
            records = await self._execute_query(cypher, params)
            results = []
            for record in records:
                results.append({
                    "original_query": record["original_query"],
                    "reformulated_query": record["reformulated_query"],
                    "success_score": record["success_score"]
                })
            return results
        except Exception as e:
            logger.error(f"Error finding successful reformulations: {e}", exc_info=True)
            return []
            
    async def get_docs_from_successful_similar_queries(self, query_text: str, success_threshold: float = 0.7, limit: int = 5) -> List[Tuple[DocumentNode, float]]:
        """
        Finds documents that were successfully used to answer similar queries.
        
        Args:
            query_text: Text of the query to find similar answers for
            success_threshold: Minimum success score to consider a response successful
            limit: Maximum number of documents to return
            
        Returns:
            List of tuples of (DocumentNode, relevance_score)
        """
        # This complex query finds:
        # 1. Queries similar to our input
        # 2. Successful responses to those queries
        # 3. Documents that contributed to those successful responses
        cypher = """
        // Find similar queries
        MATCH (q:Query)
        WHERE q.original_user_input CONTAINS $query_substring OR $query_substring CONTAINS q.original_user_input
        
        // Find successful responses to those queries
        MATCH (q)-[led:LED_TO]->(r:Response)
        WHERE led.success_metric >= $threshold
        
        // Find documents that contributed to those responses
        MATCH (r)-[gen:GENERATED_FROM]->(d:Document)
        
        // Return documents with a combined relevance score
        RETURN d, 
               gen.contribution_score * led.success_metric AS relevance_score
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        
        # Use a substring of the query text for similarity matching
        query_substring = ' '.join(query_text.split()[:5])  # First 5 words
        
        params = {
            "query_substring": query_substring,
            "threshold": success_threshold,
            "limit": limit
        }
        
        try:
            records = await self._execute_query(cypher, params)
            results = []
            for record in records:
                doc_props = dict(record["d"])
                for key, value in doc_props.items():
                    if hasattr(value, 'to_native'): doc_props[key] = value.to_native()
                doc_node = DocumentNode(**doc_props)
                relevance = record["relevance_score"]
                results.append((doc_node, relevance))
            return results
        except Exception as e:
            logger.error(f"Error finding documents from similar queries: {e}", exc_info=True)
            return []

    async def get_parent_documents(self, doc_id: str, limit: int = 5) -> List[DocumentNode]:
        """
        Retrieves parent documents for a given document.
        Parent documents are those connected by a PARENT_OF relationship.
        
        Args:
            doc_id: ID of the document to find parents for
            limit: Maximum number of parents to return
            
        Returns:
            List of parent DocumentNode objects
        """
        cypher = """
        MATCH (parent:Document)-[:PARENT_OF]->(child:Document {id: $doc_id})
        RETURN parent
        LIMIT $limit
        """
        records = await self._execute_query(cypher, {"doc_id": doc_id, "limit": limit})
        
        parents = []
        for record in records:
            props = dict(record["parent"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            parents.append(DocumentNode(**props))
        
        return parents
        
    async def get_child_documents(self, doc_id: str, limit: int = 5) -> List[DocumentNode]:
        """
        Retrieves child documents for a given document.
        Child documents are those that the parent connects to via PARENT_OF.
        
        Args:
            doc_id: ID of the document to find children for
            limit: Maximum number of children to return
            
        Returns:
            List of child DocumentNode objects
        """
        cypher = """
        MATCH (parent:Document {id: $doc_id})-[:PARENT_OF]->(child:Document)
        RETURN child
        LIMIT $limit
        """
        records = await self._execute_query(cypher, {"doc_id": doc_id, "limit": limit})
        
        children = []
        for record in records:
            props = dict(record["child"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            children.append(DocumentNode(**props))
        
        return children
    
    async def get_documents_by_concept(self, concept_id: str, limit: int = 10) -> List[DocumentNode]:
        """
        Retrieves documents related to a specific concept.
        This is an alias for get_documents_containing_concept for backward compatibility.
        
        Args:
            concept_id: ID of the concept
            limit: Maximum number of documents to return
            
        Returns:
            List of DocumentNode objects related to the concept
        """
        return await self.get_documents_containing_concept(concept_id, limit)
    
    async def get_documents_by_source(self, source: str, limit: int = 10) -> List[DocumentNode]:
        """
        Retrieves documents from a specific source.
        
        Args:
            source: Source identifier (e.g., website, database, publication)
            limit: Maximum number of documents to return
            
        Returns:
            List of DocumentNode objects from the specified source
        """
        cypher = """
        MATCH (d:Document)
        WHERE d.source = $source OR d.source_name = $source
        RETURN d
        LIMIT $limit
        """
        records = await self._execute_query(cypher, {"source": source, "limit": limit})
        
        docs = []
        for record in records:
            props = dict(record["d"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            docs.append(DocumentNode(**props))
        
        return docs
    
    async def get_documents_by_author(self, author: str, limit: int = 10) -> List[DocumentNode]:
        """
        Retrieves documents by a specific author.
        
        Args:
            author: Author name or identifier
            limit: Maximum number of documents to return
            
        Returns:
            List of DocumentNode objects by the specified author
        """
        cypher = """
        MATCH (d:Document)
        WHERE d.author = $author OR $author IN d.authors
        RETURN d
        LIMIT $limit
        """
        records = await self._execute_query(cypher, {"author": author, "limit": limit})
        
        docs = []
        for record in records:
            props = dict(record["d"])
            for key, value in props.items():
                if hasattr(value, 'to_native'): props[key] = value.to_native()
            docs.append(DocumentNode(**props))
        
        return docs

# Example usage (for testing or integration):
async def main():
    """Test function demonstrating basic GraphInterface usage with regulatory concepts"""
    # Replace with your actual credentials (or use environment variables)
    graph_db = GraphInterface(
        uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    await graph_db.connect()

    try:
        print("Connected to Neo4j database.")
        
        # 1. Create regulatory concept nodes
        gdpr = ConceptNode(
            name="GDPR",
            definition="General Data Protection Regulation - EU regulation on data protection and privacy",
            aliases=["General Data Protection Regulation"],
            domain="regulatory_compliance",
            popularity_score=0.9
        )
        
        data_privacy = ConceptNode(
            name="Data Privacy",
            definition="The proper handling of sensitive personal information",
            aliases=["Privacy", "Information Privacy"],
            domain="regulatory_compliance",
            popularity_score=0.8
        )
        
        # 2. Add nodes to the graph
        gdpr_node = await graph_db.add_node(gdpr)
        print(f"Added concept: {gdpr_node.name} (ID: {gdpr_node.id})")
        
        privacy_node = await graph_db.add_node(data_privacy)
        print(f"Added concept: {privacy_node.name} (ID: {privacy_node.id})")
        
        # 3. Connect concepts with relationships
        from ..graph_components.edges import RelatedToEdge, ContainsEdge
        
        # GDPR contains Data Privacy provisions
        contains_edge = ContainsEdge(
            source_node_id=gdpr_node.id,
            target_node_id=privacy_node.id,
            properties={"strength": 0.9, "description": "GDPR has significant data privacy provisions"}
        )
        await graph_db.add_edge(contains_edge)
        print(f"Added relationship: {gdpr_node.name} CONTAINS {privacy_node.name}")
        
        # 4. Retrieve concepts by name
        found_concepts = await graph_db.find_concepts_by_name_exact("GDPR")
        if found_concepts:
            print(f"Retrieved {len(found_concepts)} concept(s) by exact name")
            print(f"First match: {found_concepts[0].name} - {found_concepts[0].definition[:50]}...")
        
        # 5. Find concepts by fuzzy matching
        fuzzy_concepts = await graph_db.find_concepts_by_name_fuzzy("privacy", min_similarity=0.6)
        if fuzzy_concepts:
            print(f"Retrieved {len(fuzzy_concepts)} concept(s) by fuzzy matching")
            for concept in fuzzy_concepts:
                print(f"- {concept.name} (ID: {concept.id})")
        
        # 6. Update node properties
        await graph_db.update_node_properties(
            gdpr_node.id, 
            {"popularity_score": 0.95, "attributes": {"year_introduced": 2018}}
        )
        print(f"Updated properties for {gdpr_node.name}")
        
        # 7. Retrieve updated node
        updated_gdpr = await graph_db.get_node_by_id(gdpr_node.id, ConceptNode)
        if updated_gdpr:
            print(f"Retrieved updated node: {updated_gdpr.name}")
            print(f"New popularity score: {updated_gdpr.popularity_score}")
            print(f"Attributes: {updated_gdpr.attributes}")
            
    except Exception as e:
        import traceback
        print(f"Error in GraphInterface test: {e}")
        traceback.print_exc()
        
    finally:
        await graph_db.close()
        print("Closed Neo4j connection.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 