import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.schema import Document, NodeWithScore, TextNode
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings_fastembed import FastEmbedEmbedding
from llama_index.llms_openai import OpenAI

from .neo4j_vector_store import Neo4jVectorStore

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG System implementation using Neo4j for vector storage."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: Optional[str] = None,
        neo4j_database: str = "neo4j",
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        embed_dimensionality: int = 384,
        qdrant_url: Optional[str] = None,  # Not used but kept for compatibility
    ):
        """Initialize RAG System.

        Args:
            neo4j_uri: Neo4j URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key (optional)
            neo4j_database: Neo4j database name
            embed_model_name: Name of the embedding model to use
            embed_dimensionality: Dimensionality of the embeddings
            qdrant_url: Legacy parameter, not used
        """
        logger.info("Initializing RAG System")
        
        try:
            # Initialize the embedding model
            self.embed_model = self._initialize_embedding_model(embed_model_name)
            
            # Initialize Neo4j vector store
            self.vector_store = Neo4jVectorStore(
                uri=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password,
                database=neo4j_database,
                node_label="TextChunk",
                text_property="text",
                embedding_property="embedding",
                metadata_property="metadata",
                default_vector_dim=embed_dimensionality,
                similarity_function="cosine"
            )
            
            # Initialize OpenAI if API key is provided
            if openai_api_key:
                self.llm = OpenAI(api_key=openai_api_key, temperature=0.1)
            else:
                self.llm = None
                logger.warning("No OpenAI API key provided. LLM functionality will be limited.")
            
            logger.info("Neo4j vector store set up successfully")
            logger.info("RAG system initialized with Neo4j vector store")
        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {str(e)}")
            raise

    def _initialize_embedding_model(self, model_name: str) -> BaseEmbedding:
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            
        Returns:
            An initialized embedding model
        """
        try:
            # Try FastEmbed (faster and more efficient)
            try:
                logger.info(f"Initializing FastEmbed with model {model_name}")
                return FastEmbedEmbedding(model_name=model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize FastEmbed: {str(e)}")
                # Fall back to a simpler approach
                from llama_index.core.embeddings import resolve_embed_model
                logger.info(f"Falling back to resolve_embed_model for {model_name}")
                return resolve_embed_model(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def index_documents(self, documents: List[Document]) -> List[str]:
        """Index documents in the vector store.
        
        Args:
            documents: List of documents to index
            
        Returns:
            List of document IDs
        """
        try:
            # Convert documents to nodes with embeddings
            nodes = []
            
            for doc in documents:
                # Create text nodes from document
                text_chunks = self._split_document(doc)
                
                # Generate embeddings for each chunk
                for chunk in text_chunks:
                    if not chunk.embedding:
                        embedding = self.embed_model.get_text_embedding(
                            chunk.get_content()
                        )
                        chunk.embedding = embedding
                    nodes.append(chunk)
            
            # Add nodes to vector store
            doc_ids = self.vector_store.add(nodes)
            logger.info(f"Indexed {len(doc_ids)} document chunks")
            
            return doc_ids
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def _split_document(self, document: Document) -> List[TextNode]:
        """Split document into chunks for indexing.
        
        Args:
            document: Document to split
            
        Returns:
            List of text nodes
        """
        # Simple implementation - in a real system, use a proper text splitter
        # This is a placeholder that just creates a single node per document
        node = TextNode(
            text=document.get_content(),
            metadata=document.metadata
        )
        return [node]

    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of nodes with relevance scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Prepare query bundle
            query_bundle = QueryBundle(
                query_str=query,
                embedding=query_embedding
            )
            
            # Query vector store
            results = self.vector_store.query(
                query_bundle,
                similarity_top_k=top_k
            )
            
            logger.info(f"Retrieved {len(results.nodes)} relevant chunks for query: {query}")
            
            return results.nodes
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    def delete_documents(self, doc_id: str) -> None:
        """Delete documents from the vector store.
        
        Args:
            doc_id: Document ID to delete
        """
        try:
            self.vector_store.delete(doc_id)
            logger.info(f"Deleted document with ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

    def close(self) -> None:
        """Close connections to resources."""
        try:
            self.vector_store.close()
            logger.info("RAG system resources released")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}") 