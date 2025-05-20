# backend/llamaIndex_rag/neo4j_vector_store.py
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

import uuid
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class Neo4jVectorStore(VectorStore):
    """Neo4j Vector Store implementation using Neo4j's vector search capabilities."""

    stores_text: bool = True
    flat_metadata: bool = True

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        node_label: str = "TextChunk",
        text_property: str = "text",
        embedding_property: str = "embedding",
        metadata_property: str = "metadata",
        default_vector_dim: int = 384,
        similarity_function: str = "cosine",
        index_name: str = "text_chunk_embedding",
        collection_name: Optional[str] = None,
        language: Optional[str] = "fr"
    ) -> None:
        """Initialize Neo4j Vector Store.

        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            node_label: Label of the node to store vectors
            text_property: Property name for storing text
            embedding_property: Property name for storing embeddings
            metadata_property: Property name for storing metadata
            default_vector_dim: Default vector dimension
            similarity_function: Similarity function for vector search
            index_name: Name of the vector index
            collection_name: Optional collection name for language-specific handling
            language: Optional language code for language-specific handling
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info(f"Connected to Neo4j at {uri}")
            
            # Test connection
            with self.driver.session(database=database) as session:
                result = session.run("RETURN 'Connected to Neo4j' AS message")
                for record in result:
                    logger.info(record["message"])
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

        self.database = database
        self.node_label = node_label
        self.text_property = text_property
        self.embedding_property = embedding_property
        self.metadata_property = metadata_property
        self.default_vector_dim = default_vector_dim
        self.similarity_function = similarity_function
        self.index_name = index_name
        self.collection_name = collection_name
        self.language = language
        
        # Initialize vector index
        self._ensure_vector_index()
        
    def _ensure_vector_index(self) -> None:
        """Create vector index if it doesn't exist."""
        try:
            with self.driver.session(database=self.database) as session:
                # Check Neo4j version
                result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
                for record in result:
                    if record["name"] == "Neo4j Kernel":
                        neo4j_version = record["versions"][0]
                        logger.info(f"Neo4j version: {neo4j_version}")
                        break
                
                # Create vector index
                create_index_query = f"""
                CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS 
                FOR (n:{self.node_label}) 
                ON (n.{self.embedding_property})
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {self.default_vector_dim},
                    `vector.similarity_function`: '{self.similarity_function}'
                  }}
                }}
                """
                session.run(create_index_query)
                logger.info(f"Created/ensured vector index {self.index_name} on {self.node_label}.{self.embedding_property}")
                
                # Create fulltext index for hybrid search
                fulltext_index_name = f"{self.node_label.lower()}_text"
                create_fulltext_query = f"""
                CREATE FULLTEXT INDEX {fulltext_index_name} IF NOT EXISTS 
                FOR (n:{self.node_label}) 
                ON EACH [n.{self.text_property}]
                """
                session.run(create_fulltext_query)
                logger.info(f"Created/ensured fulltext index {fulltext_index_name} on {self.node_label}.{self.text_property}")
                
        except Exception as e:
            logger.error(f"Error ensuring vector index: {str(e)}")
            raise

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of nodes to add
            add_kwargs: Additional arguments

        Returns:
            List of node IDs
        """
        ids = []
        
        try:
            with self.driver.session(database=self.database) as session:
                for node in nodes:
                    # Generate ID if not present
                    if node.id_ is None or not node.id_:
                        node_id = str(uuid.uuid4())
                    else:
                        node_id = node.id_
                    
                    # Get text
                    text = node.get_content(metadata_mode=MetadataMode.NONE)
                    
                    # Get embedding
                    embedding = node.get_embedding()
                    if embedding is None:
                        logger.warning(f"Node {node_id} has no embedding. Skipping.")
                        continue
                    
                    # Get metadata
                    metadata = node_to_metadata_dict(
                        node, remove_text=True, flat_metadata=self.flat_metadata
                    )
                    
                    # Add language if available
                    if self.language:
                        metadata["language"] = self.language
                    
                    # Add collection info if available
                    if self.collection_name:
                        metadata["collection"] = self.collection_name
                    
                    # Extract key metadata fields to store as direct properties
                    chunk_id = metadata.get("chunk_id", node_id)
                    doc_id = metadata.get("doc_id", "")
                    doc_name = metadata.get("doc_name", "")
                    language = metadata.get("language", self.language or "")
                    section = metadata.get("section", "")
                    page_num = metadata.get("page_num", 0)
                    order_index = metadata.get("order_index", 0)
                    category = metadata.get("category", "")
                    author = metadata.get("author", "")
                    file_type = metadata.get("file_type", "")
                    
                    # Create node in Neo4j
                    create_query = f"""
                    CREATE (n:{self.node_label} {{
                        id: $id,
                        chunk_id: $chunk_id,
                        doc_id: $doc_id,
                        doc_name: $doc_name,
                        {self.text_property}: $text,
                        {self.embedding_property}: $embedding,
                        language: $language,
                        section: $section,
                        page_num: $page_num,
                        order_index: $order_index,
                        category: $category,
                        author: $author, 
                        file_type: $file_type
                    }})
                    RETURN n.id as id
                    """
                    
                    result = session.run(
                        create_query,
                        id=node_id,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        text=text,
                        embedding=embedding,
                        language=language,
                        section=section,
                        page_num=page_num,
                        order_index=order_index,
                        category=category,
                        author=author,
                        file_type=file_type
                    )
                    
                    for record in result:
                        ids.append(record["id"])
                        
                    logger.info(f"Added node {node_id} to Neo4j vector store")
                
        except Exception as e:
            logger.error(f"Error adding nodes to Neo4j vector store: {str(e)}")
            raise
            
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from the vector store.

        Args:
            ref_doc_id: The document ID to delete
            delete_kwargs: Additional arguments
        """
        try:
            with self.driver.session(database=self.database) as session:
                # Check if we need to delete by document reference or by node ID
                if delete_kwargs.get("delete_from_metadata", False):
                    # Delete by document reference in metadata
                    delete_query = f"""
                    MATCH (n:{self.node_label})
                    WHERE n.doc_id = $ref_doc_id
                    DETACH DELETE n
                    """
                else:
                    # Delete by node ID
                    delete_query = f"""
                    MATCH (n:{self.node_label} {{id: $ref_doc_id}})
                    DETACH DELETE n
                    """
                
                session.run(delete_query, ref_doc_id=ref_doc_id)
                logger.info(f"Deleted nodes with reference to document {ref_doc_id} from Neo4j vector store")
                
        except Exception as e:
            logger.error(f"Error deleting nodes from Neo4j vector store: {str(e)}")
            raise

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        Args:
            query: Vector store query
            kwargs: Additional arguments

        Returns:
            Vector store query result
        """
        query_embedding = query.query_embedding
        if query_embedding is None:
            raise ValueError("Query embedding is required for vector search")
            
        similarity_top_k = query.similarity_top_k
        
        try:
            with self.driver.session(database=self.database) as session:
                # Build filter conditions if provided
                filter_conditions = []
                filter_params = {}
                
                if query.filters:
                    for key, value in query.filters.items():
                        # Handle direct property filters
                        filter_conditions.append(f"n.{key} = ${key}")
                        filter_params[key] = value
                
                # Add language filter if available
                if self.language:
                    filter_conditions.append(f"n.language = $language")
                    filter_params["language"] = self.language
                    
                # Add collection filter if available
                if self.collection_name:
                    filter_conditions.append(f"n.collection = $collection")
                    filter_params["collection"] = self.collection_name
                
                where_clause = ""
                if filter_conditions:
                    where_clause = "WHERE " + " AND ".join(filter_conditions)
                
                # Perform vector search
                vector_query = f"""
                MATCH (n:{self.node_label})
                {where_clause}
                WITH n, vector.similarity.{self.similarity_function}(n.{self.embedding_property}, $embedding) AS score
                ORDER BY score DESC
                LIMIT $top_k
                RETURN 
                    n.id as id, 
                    n.{self.text_property} as text, 
                    n.chunk_id as chunk_id,
                    n.doc_id as doc_id,
                    n.doc_name as doc_name,
                    n.language as language,
                    n.section as section,
                    n.page_num as page_num,
                    n.order_index as order_index,
                    n.category as category,
                    n.author as author,
                    n.file_type as file_type,
                    score
                """
                
                query_params = {
                    "embedding": query_embedding,
                    "top_k": similarity_top_k,
                    **filter_params
                }
                
                result = session.run(vector_query, **query_params)
                
                nodes = []
                similarities = []
                ids = []
                
                for record in result:
                    node_id = record["id"]
                    node_text = record["text"]
                    node_score = record["score"]
                    
                    # Build metadata from individual properties
                    node_metadata = {
                        "chunk_id": record["chunk_id"],
                        "doc_id": record["doc_id"],
                        "doc_name": record["doc_name"],
                        "language": record["language"],
                        "section": record["section"],
                        "page_num": record["page_num"],
                        "order_index": record["order_index"],
                        "category": record["category"],
                        "author": record["author"],
                        "file_type": record["file_type"]
                    }
                    
                    # Create TextNode
                    node = TextNode(
                        text=node_text,
                        id_=node_id,
                        metadata=node_metadata,
                    )
                    
                    nodes.append(node)
                    similarities.append(node_score)
                    ids.append(node_id)
                
                logger.info(f"Retrieved {len(nodes)} nodes from Neo4j vector store")
                
                return VectorStoreQueryResult(
                    nodes=nodes,
                    similarities=similarities,
                    ids=ids,
                )
                
        except Exception as e:
            logger.error(f"Error querying Neo4j vector store: {str(e)}")
            raise

    def update(
        self,
        nodes: List[BaseNode],
        **update_kwargs: Any,
    ) -> List[str]:
        """Update nodes in the vector store.

        Args:
            nodes: List of nodes to update
            update_kwargs: Additional arguments

        Returns:
            List of updated node IDs
        """
        ids = []
        
        try:
            with self.driver.session(database=self.database) as session:
                for node in nodes:
                    node_id = node.id_
                    if not node_id:
                        logger.warning("Cannot update node without ID. Skipping.")
                        continue
                    
                    # Get text
                    text = node.get_content(metadata_mode=MetadataMode.NONE)
                    
                    # Get embedding
                    embedding = node.get_embedding()
                    if embedding is None:
                        logger.warning(f"Node {node_id} has no embedding. Skipping.")
                        continue
                    
                    # Get metadata
                    metadata = node_to_metadata_dict(
                        node, remove_text=True, flat_metadata=self.flat_metadata
                    )
                    
                    # Add language if available
                    if self.language:
                        metadata["language"] = self.language
                    
                    # Add collection info if available
                    if self.collection_name:
                        metadata["collection"] = self.collection_name
                    
                    # Extract key metadata fields
                    chunk_id = metadata.get("chunk_id", node_id)
                    doc_id = metadata.get("doc_id", "")
                    doc_name = metadata.get("doc_name", "")
                    language = metadata.get("language", self.language or "")
                    section = metadata.get("section", "")
                    page_num = metadata.get("page_num", 0)
                    order_index = metadata.get("order_index", 0)
                    category = metadata.get("category", "")
                    author = metadata.get("author", "")
                    file_type = metadata.get("file_type", "")
                        
                    # Update node in Neo4j
                    update_query = f"""
                    MERGE (n:{self.node_label} {{id: $id}})
                    SET n.{self.text_property} = $text,
                        n.{self.embedding_property} = $embedding,
                        n.chunk_id = $chunk_id,
                        n.doc_id = $doc_id,
                        n.doc_name = $doc_name,
                        n.language = $language,
                        n.section = $section,
                        n.page_num = $page_num,
                        n.order_index = $order_index,
                        n.category = $category,
                        n.author = $author,
                        n.file_type = $file_type
                    RETURN n.id as id
                    """
                    
                    result = session.run(
                        update_query,
                        id=node_id,
                        text=text,
                        embedding=embedding,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        language=language,
                        section=section,
                        page_num=page_num,
                        order_index=order_index,
                        category=category,
                        author=author,
                        file_type=file_type
                    )
                    
                    for record in result:
                        ids.append(record["id"])
                        
                    logger.info(f"Updated node {node_id} in Neo4j vector store")
                
        except Exception as e:
            logger.error(f"Error updating nodes in Neo4j vector store: {str(e)}")
            raise
            
        return ids
        
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection") 