"""
RAG System with Reliable RAG techniques for hallucination prevention and detection.

This module implements a production-ready RAG system using LlamaIndex with features
to minimize hallucinations and detect them when they occur.
"""

import logging
import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import uuid
import numpy as np
import re

# LlamaIndex imports - updated to match installed package structure
from llama_index.core import (
    SimpleDirectoryReader,
    Settings
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import TextNode, Document, NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Hallucination detection components
from llama_index.core.evaluation import (
    ResponseEvaluator,
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator
)

# Qdrant client for vector DB
from qdrant_client import QdrantClient, models as qdrant_models

# Define MetadataParser at the module level
class MetadataParser:
    """Simple metadata parser for document processing."""
    
    def __init__(self):
        """Initialize metadata parser."""
        pass
    
    def parse_metadata(self, doc_id: str, document: Any, mime_type: str = None) -> Dict[str, Any]:
        """
        Parse metadata from a document.
        
        Args:
            doc_id: Document ID
            document: Document to parse metadata from
            mime_type: MIME type of document
            
        Returns:
            Dictionary of metadata
        """
        # Basic metadata
        metadata = {
            "doc_id": doc_id,
            "timestamp": time.time(),
            "mime_type": mime_type or "text/plain",
        }
        
        # If the document has metadata, add it
        if hasattr(document, "metadata"):
            metadata.update(document.metadata)
            
        return metadata

# Module logger
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Production-ready RAG System with Reliable RAG techniques to prevent and detect hallucinations.
    """
    
    def __init__(
        self,
        collection_name: str = "regulaite_docs",
        metadata_collection_name: str = "regulaite_metadata",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        llm_model: str = "gpt-4.1",
        qdrant_url: str = "http://qdrant:6333",
        openai_api_key: Optional[str] = None,
        max_context_chunks: int = 8,
        min_context_score: float = 0.1,
        doc_chunk_size: int = 1024,
        doc_chunk_overlap: int = 100,
        vector_weight: float = 0.7,
        semantic_weight: float = 0.3,
    ):
        """
        Initialize RAG System
        
        Args:
            collection_name: Qdrant collection name for document chunks
            metadata_collection_name: Qdrant collection name for document metadata
            embedding_model: Embedding model to use
            embedding_dim: Embedding dimension
            llm_model: LLM model to use
            qdrant_url: Qdrant URL
            openai_api_key: OpenAI API key
            max_context_chunks: Maximum number of context chunks to retrieve
            min_context_score: Minimum context score for retrieval
            doc_chunk_size: Document chunk size
            doc_chunk_overlap: Document chunk overlap
            vector_weight: Weight for vector search in hybrid retrieval (0-1)
            semantic_weight: Weight for semantic search in hybrid retrieval (0-1)
        """
        self.collection_name = collection_name
        self.metadata_collection_name = metadata_collection_name
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.llm_model_name = llm_model
        self.qdrant_url = qdrant_url
        self.openai_api_key = openai_api_key
        self.max_context_chunks = max_context_chunks
        self.min_context_score = min_context_score
        self.doc_chunk_size = doc_chunk_size
        self.doc_chunk_overlap = doc_chunk_overlap
        self.vector_weight = vector_weight
        self.semantic_weight = semantic_weight
        
        # Initialize components
        try:
            # Connect to Qdrant
            self.client = QdrantClient(url=qdrant_url)
            
            # Create collections if they don't exist
            self._ensure_collections_exist()
            
            logger.info(f"Collections initialized: {collection_name}, {metadata_collection_name}")
            
            # Initialize LLM
            self.llm = OpenAI(model=llm_model, temperature=0.1)
            
            # Initialize embeddings
            if "openai" in embedding_model.lower():
                self.embed_model = OpenAIEmbedding(
                    model=embedding_model,
                    api_key=openai_api_key,
                )
            else:
                self.embed_model = FastEmbedEmbedding(
                    model_name=embedding_model,
                )
            
            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
            )
            
            # Update global settings using Settings class
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            
            # Create vector store index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
            )
            
            # Initialize evaluators
            self.response_evaluator = ResponseEvaluator(llm=self.llm)
            self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.llm)
            self.relevancy_evaluator = ContextRelevancyEvaluator(llm=self.llm)
            self.similarity_evaluator = SemanticSimilarityEvaluator(embed_model=self.embed_model)
            
            logger.info(f"RAG system initialized with embedding model: {embedding_model}, LLM: {llm_model}")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def _ensure_collections_exist(self):
        """Initialize Qdrant collections if they don't exist."""
        try:
            # Check and create main collection
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
            
            # Check and create metadata collection
            if self.metadata_collection_name not in collection_names:
                logger.info(f"Creating metadata collection {self.metadata_collection_name}")
                self.client.create_collection(
                    collection_name=self.metadata_collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                
            logger.info(f"Collections initialized: {self.collection_name}, {self.metadata_collection_name}")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collections: {str(e)}")
            raise
    
    @property
    def qdrant_client(self):
        """Alias for self.client to maintain compatibility."""
        return self.client
    
    def index_document(self, doc_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Index a document in the RAG system.
        
        Args:
            doc_id: Document ID to index
            force: Whether to force reindex
            
        Returns:
            Dict with indexing results
        """
        try:
            start_time = time.time()
            
            # Check if document is already indexed
            if not force:
                try:
                    existing_points = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="metadata.doc_id",
                                    match=qdrant_models.MatchValue(value=doc_id)
                                )
                            ]
                        ),
                        limit=1
                    )
                    
                    if existing_points and len(existing_points[0]) > 0:
                        logger.info(f"Document {doc_id} is already indexed and force=False")
                        return {
                            "status": "success",
                            "doc_id": doc_id,
                            "message": "Document already indexed",
                            "vector_count": 0,
                            "duration_seconds": time.time() - start_time
                        }
                except Exception as e:
                    logger.warning(f"Error checking if document is already indexed: {str(e)}")
            
            # Get document_parser from main application
            from main import document_parser
            if not document_parser:
                return {
                    "status": "error",
                    "doc_id": doc_id,
                    "message": "Document parser not initialized",
                    "error": "Document parser not initialized"
                }
            
            # Retrieve document chunks from the parser
            # Try different methods that might exist on the parser
            chunks = None
            try:
                # First try get_document_chunks
                if hasattr(document_parser, 'get_document_chunks'):
                    chunks = document_parser.get_document_chunks(doc_id)
                    
                # If that fails, try get_chunks
                elif hasattr(document_parser, 'get_chunks'):
                    chunks = document_parser.get_chunks(doc_id)
                    
                # Try other potential methods
                elif hasattr(document_parser, 'retrieve_chunks'):
                    chunks = document_parser.retrieve_chunks(doc_id)
                    
                # Last resort: try to get the document and split it ourselves
                elif hasattr(document_parser, 'get_document'):
                    document = document_parser.get_document(doc_id)
                    if document:
                        # Simple text chunking if needed
                        content = document.get('content', '') or document.get('text', '')
                        if content:
                            # Create simple chunks of approximately doc_chunk_size characters
                            text_chunks = []
                            for i in range(0, len(content), self.doc_chunk_size - self.doc_chunk_overlap):
                                chunk_text = content[i:i + self.doc_chunk_size]
                                if chunk_text:
                                    text_chunks.append({
                                        'content': chunk_text,
                                        'metadata': {
                                            'doc_id': doc_id,
                                            'chunk_index': len(text_chunks)
                                        }
                                    })
                            chunks = text_chunks
            except Exception as e:
                logger.error(f"Error retrieving chunks for document {doc_id}: {str(e)}")
                return {
                    "status": "error",
                    "doc_id": doc_id,
                    "message": f"Error retrieving chunks: {str(e)}",
                    "error": str(e)
                }
            
            if not chunks or len(chunks) == 0:
                logger.warning(f"No chunks found for document {doc_id}")
                return {
                    "status": "error",
                    "doc_id": doc_id,
                    "message": f"No chunks found for document {doc_id}",
                    "error": "No chunks found"
                }
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {doc_id}")
            
            # Convert chunks to LlamaIndex nodes
            nodes = []
            for chunk in chunks:
                # Extract content and metadata
                content = chunk.get('content') or chunk.get('text', '')
                chunk_metadata = chunk.get('metadata', {})
                
                # Ensure doc_id is in metadata
                chunk_metadata['doc_id'] = doc_id
                
                # Create TextNode
                node = TextNode(
                    text=content,
                    metadata=chunk_metadata
                )
                nodes.append(node)
            
            # Index nodes
            vector_count = 0
            if nodes:
                # Get the index
                try:
                    if force:
                        # Delete existing document nodes first
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=qdrant_models.Filter(
                                must=[
                                    qdrant_models.FieldCondition(
                                        key="metadata.doc_id",
                                        match=qdrant_models.MatchValue(value=doc_id)
                                    )
                                ]
                            )
                        )
                        logger.info(f"Deleted existing nodes for document {doc_id}")
                    
                    # Index nodes
                    for node in nodes:
                        try:
                            # Get embedding for node
                            embedding = self.embed_model.get_text_embedding(node.get_content())
                            
                            # Generate a unique ID for the node
                            node_id = str(uuid.uuid4())
                            
                            # Create payload with text and metadata
                            payload = {
                                "text": node.get_content(),
                                "metadata": node.metadata
                            }
                            
                            # Store in Qdrant
                            self.client.upsert(
                                collection_name=self.collection_name,
                                points=[
                                    qdrant_models.PointStruct(
                                        id=node_id,
                                        vector=embedding,
                                        payload=payload
                                    )
                                ]
                            )
                            vector_count += 1
                        except Exception as e:
                            logger.error(f"Error indexing node: {str(e)}")
                
                    logger.info(f"Indexed {vector_count} vectors for document {doc_id}")
                except Exception as index_error:
                    logger.error(f"Error during indexing: {str(index_error)}")
                    return {
                        "status": "error",
                        "doc_id": doc_id,
                        "message": f"Error during indexing: {str(index_error)}",
                        "error": str(index_error)
                    }
            
            duration = time.time() - start_time
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": f"Document indexed successfully in {duration:.2f} seconds",
                "vector_count": vector_count,
                "duration_seconds": duration
            }
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {str(e)}")
            return {
                "status": "error",
                "doc_id": doc_id,
                "message": f"Error indexing document: {str(e)}",
                "error": str(e)
            }
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the RAG system."""
        try:
            logger.info(f"Deleting document {doc_id} from vector store")
            
            # Delete points from Qdrant with matching doc_id in metadata
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.doc_id",
                            match=qdrant_models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            
            # Try to also delete from metadata collection if it exists
            try:
                self.client.delete(
                    collection_name=self.metadata_collection_name,
                    points_selector=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="doc_id",
                                match=qdrant_models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            except Exception as e:
                logger.warning(f"Could not delete from metadata collection: {str(e)}")
            
            logger.info(f"Successfully deleted document {doc_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def reindex_all_documents(self, force: bool = False) -> Dict[str, Any]:
        """Reindex all documents in the RAG system."""
        try:
            start_time = time.time()
            logger.info("Starting reindexing of all documents")
            
            # Get document_parser from main application
            from main import document_parser
            if not document_parser:
                return {
                    "status": "error",
                    "message": "Document parser not initialized",
                    "error": "Document parser not initialized"
                }
                
            # Get list of all document IDs from parser
            try:
                doc_ids = document_parser.get_all_document_ids()
            except Exception as e:
                logger.error(f"Error getting document IDs: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error getting document IDs: {str(e)}",
                    "error": str(e)
                }
                
            if not doc_ids:
                logger.warning("No documents found to reindex")
                return {
                    "status": "success",
                    "message": "No documents found to reindex",
                    "count": 0,
                    "duration_seconds": time.time() - start_time
                }
                
            # Reindex each document
            results = []
            success_count = 0
            error_count = 0
            
            for doc_id in doc_ids:
                try:
                    result = self.index_document(doc_id, force=force)
                    results.append(result)
                    
                    if result.get("status") == "success":
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error reindexing document {doc_id}: {str(e)}")
                    error_count += 1
                    results.append({
                        "status": "error",
                        "doc_id": doc_id,
                        "message": f"Error: {str(e)}",
                        "error": str(e)
                    })
            
            duration = time.time() - start_time
            return {
                "status": "success",
                "message": f"Reindexed {success_count} documents successfully, {error_count} failed",
                "count": len(doc_ids),
                "success_count": success_count,
                "error_count": error_count,
                "duration_seconds": duration,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error reindexing documents: {str(e)}")
            return {
                "status": "error",
                "message": f"Error reindexing documents: {str(e)}",
                "error": str(e)
            }
    
    def _apply_context_reranking(
        self, 
        nodes: List[NodeWithScore], 
        query: str
    ) -> List[NodeWithScore]:
        """
        Apply advanced reranking for Reliable RAG.
        
        Args:
            nodes: List of nodes with scores
            query: User query
            
        Returns:
            Processed and reranked nodes
        """
        if not nodes:
            return []
        
        try:
            # Step 1: Initial semantic relevancy scoring using embedding similarity
            logger.info(f"Applying advanced Reliable RAG reranking for {len(nodes)} nodes")
            
            # Get query embedding once
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Score each node by comparing embeddings directly (bi-encoder step)
            node_scores = []
            all_similarity_scores = []
            
            for node in nodes:
                try:
                    # Get text content from node
                    text = node.node.get_content()
                    
                    # Get embedding for text
                    text_embedding = self.embed_model.get_text_embedding(text)
                    
                    # Calculate cosine similarity
                    from numpy import dot
                    from numpy.linalg import norm
                    import numpy as np
                    
                    if isinstance(query_embedding, list) and isinstance(text_embedding, list):
                        # Convert to numpy arrays to handle calculations properly
                        query_array = np.array(query_embedding)
                        text_array = np.array(text_embedding)
                        
                        # Ensure we get a real number by taking the real part if complex
                        dot_product = np.dot(query_array, text_array)
                        if isinstance(dot_product, complex):
                            dot_product = dot_product.real
                            
                        norm_query = np.linalg.norm(query_array)
                        norm_text = np.linalg.norm(text_array)
                        
                        # Avoid division by zero
                        if norm_query > 0 and norm_text > 0:
                            cos_sim = float(dot_product / (norm_query * norm_text))
                        else:
                            cos_sim = 0.0
                            
                        # Convert numpy types to Python native types
                        if isinstance(cos_sim, (np.float32, np.float64)):
                            cos_sim = float(cos_sim)
                        elif isinstance(cos_sim, complex):
                            cos_sim = float(cos_sim.real)
                        relevancy_score = float(cos_sim)  # Convert to float to ensure it's JSON serializable
                        all_similarity_scores.append(relevancy_score)
                    else:
                        # Fallback to original score
                        relevancy_score = float(node.score or 0.5)
                        
                    # Create score entry
                    node_scores.append((node, relevancy_score))
                except Exception as e:
                    logger.warning(f"Error scoring node: {str(e)}")
                    # Use original score as fallback
                    node_scores.append((node, float(node.score or 0.5)))
            
            # Apply score normalization to bi-encoder results
            max_similarity = max(all_similarity_scores) if all_similarity_scores else 0.5
            normalized_node_scores = []
            
            for node, score in node_scores:
                # Min-max normalization with boost
                normalized_score = (score / max(max_similarity, 0.0001)) ** 0.5  # Square root to boost lower values
                normalized_node_scores.append((node, float(normalized_score)))
                
            node_scores = normalized_node_scores
            
            # Step 2: Contextual Compression - Focus on relevant information only
            # Extract the most relevant sentences from each node
            compressed_nodes = []
            try:
                # Import re module inside this try block to ensure it's available in scope
                import re
                
                for node, score in node_scores:
                    text = node.node.get_content()
                    
                    # Split text into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    
                    if len(sentences) <= 3:  # For very short nodes, keep as is
                        compressed_nodes.append((node, float(score)))
                        continue
                    
                    # Get relevance score for each sentence
                    sentence_scores = []
                    for sentence in sentences:
                        if len(sentence.strip()) < 5:  # Skip very short sentences
                            continue
                            
                        sent_embedding = self.embed_model.get_text_embedding(sentence)
                        # Calculate similarity to query
                        sent_sim = dot(query_embedding, sent_embedding) / (norm(query_embedding) * norm(sent_embedding))
                        # Convert numpy types to Python native types
                        if isinstance(sent_sim, (np.float32, np.float64)):
                            sent_sim = float(sent_sim)
                        elif isinstance(sent_sim, complex):
                            sent_sim = float(sent_sim.real)
                        sentence_scores.append((sentence, float(sent_sim)))
                    
                    # Select top sentences (50% of original or at least 3 sentences)
                    top_k = max(3, len(sentences) // 2)
                    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:top_k]
                    
                    # Reorder sentences to maintain original flow
                    original_order = {}
                    for i, s in enumerate([s for s, _ in top_sentences]):
                        for j, orig_s in enumerate(sentences):
                            if s == orig_s:
                                original_order[s] = j
                                break
                    
                    ordered_sentences = sorted(top_sentences, key=lambda x: original_order.get(x[0], 999))
                    
                    # Create new compressed text
                    compressed_text = " ".join([s for s, _ in ordered_sentences])
                    
                    # Create new node with compressed text
                    new_metadata = dict(node.node.metadata)
                    new_metadata["compressed"] = True
                    new_metadata["compression_ratio"] = float(len(compressed_text) / len(text))
                    
                    compressed_node = TextNode(
                        text=compressed_text,
                        metadata=new_metadata
                    )
                    
                    # Adjust score slightly upward due to compression
                    compressed_nodes.append((NodeWithScore(node=compressed_node, score=float(score * 1.05)), float(score * 1.05)))
                
                # If compression was successful, use compressed nodes
                if compressed_nodes:
                    node_scores = compressed_nodes
                    logger.info(f"Applied contextual compression to {len(compressed_nodes)} nodes")
            except Exception as e:
                logger.warning(f"Error in contextual compression: {str(e)}, using original nodes")
            
            # Step 3: Cross-encoder scoring for higher precision on top candidates
            # This simulates a cross-encoder by using the LLM to evaluate relevance directly
            if len(node_scores) > 0 and hasattr(self, 'llm'):
                try:
                    # Take top candidates from first stage for more expensive reranking
                    top_k_first_stage = min(len(node_scores), 5)  # Limit to 5 for efficiency
                    top_candidates = sorted(node_scores, key=lambda x: x[1], reverse=True)[:top_k_first_stage]
                    
                    refined_scores = []
                    for node, initial_score in top_candidates:
                        text = node.node.get_content()
                        
                        # Use LLM to assess relevance with specific criteria (more robust evaluation)
                        prompt = f"""
                        Evaluate how relevant the following text passage is to the query on a scale of 0 to 10.
                        Consider these criteria:
                        1. Semantic relevance to the query
                        2. Contains specific information that helps answer the query
                        3. Information quality and reliability
                        4. Comprehensiveness of coverage related to the query
                        
                        Provide only a number as your answer.
                        
                        Query: {query}
                        
                        Text passage:
                        {text[:500]}...
                        
                        Relevance score (0-10):
                        """
                        
                        try:
                            # Call LLM with low temperature for consistency
                            response = self.llm.complete(prompt, temperature=0.1)
                            response_text = response.text if hasattr(response, 'text') else str(response)
                            
                            # Extract the numeric score from the response
                            import re
                            score_match = re.search(r'(\d+(\.\d+)?)', response_text)
                            if score_match:
                                llm_score = float(score_match.group(1))
                                # Normalize to 0-1 range
                                llm_score = float(min(10, max(0, llm_score)) / 10)
                                
                                # Combine with initial score using weighted average instead of RRF
                                # This works better with our normalized scores
                                combined_score = 0.4 * initial_score + 0.6 * llm_score
                                
                                refined_scores.append((node, combined_score))
                                
                                # Add the LLM-based score to node metadata
                                node.node.metadata["llm_relevance_score"] = float(llm_score)
                                node.node.metadata["combined_score"] = float(combined_score)
                            else:
                                # If no score extracted, use the initial score
                                refined_scores.append((node, float(initial_score)))
                        except Exception as e:
                            logger.warning(f"Error in LLM relevance scoring: {str(e)}")
                            refined_scores.append((node, float(initial_score)))
                    
                    # Replace the scores for the top candidates
                    if refined_scores:
                        # Create a dictionary of node to score for quick lookup
                        refined_dict = {id(node): float(score) for node, score in refined_scores}
                        
                        # Update the scores in the original list
                        for i, (node, score) in enumerate(node_scores):
                            if id(node) in refined_dict:
                                node_scores[i] = (node, refined_dict[id(node)])
                                
                        logger.info(f"Applied cross-encoder reranking with weighted combination to {len(refined_scores)} nodes")
                except Exception as e:
                    logger.warning(f"Error in cross-encoder reranking: {str(e)}")
            
            # Step 4: Adaptive Retrieval based on query complexity
            # Determine query complexity to adjust how much we focus on diversity vs relevance
            try:
                # Quick query complexity assessment based on:
                # - Number of distinct entities/concepts in the query
                # - Presence of comparison or complex operators
                # - Query length
                
                query_words = set(query.lower().split())
                complexity_indicators = ["compare", "difference", "versus", "vs", "relationship", "how", "why", "explain"]
                
                complexity_score = 0.0
                # Length factor
                if len(query.split()) > 15:
                    complexity_score += 0.3
                elif len(query.split()) > 8:
                    complexity_score += 0.2
                
                # Check for complexity indicators
                for indicator in complexity_indicators:
                    if indicator in query_words:
                        complexity_score += 0.15
                
                # Adjust diversity vs relevance based on complexity
                lambda_param = 0.7  # Default
                if complexity_score > 0.3:
                    # More complex queries need more diverse context
                    lambda_param = 0.5  # Lower lambda gives more weight to diversity
                    logger.info(f"Complex query detected, adjusting diversity weight to {lambda_param}")
            except Exception as e:
                lambda_param = 0.7  # Default fallback
                logger.warning(f"Error in query complexity assessment: {str(e)}")
            
            # Step 5: Apply diversity reranking to reduce redundancy with adjusted lambda
            try:
                # Sort by score first
                sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
                
                # Apply maximal marginal relevance to increase diversity
                if len(sorted_nodes) > 1:
                    # Start with the highest scoring node
                    selected = [sorted_nodes[0]]
                    remaining = sorted_nodes[1:]
                    
                    # Iteratively select nodes that are relevant but diverse
                    while remaining and len(selected) < len(sorted_nodes):
                        best_node = None
                        best_score = -float('inf')
                        
                        for i, (node, rel_score) in enumerate(remaining):
                            # Calculate diversity score (max similarity to already selected nodes)
                            node_text = node.node.get_content()
                            node_embedding = self.embed_model.get_text_embedding(node_text)
                            
                            # Calculate similarity to each already selected node
                            max_sim = 0.0
                            for sel_node, _ in selected:
                                sel_text = sel_node.node.get_content()
                                sel_embedding = self.embed_model.get_text_embedding(sel_text)
                                
                                sim = dot(node_embedding, sel_embedding) / (norm(node_embedding) * norm(sel_embedding))
                                # Convert numpy types to Python native types
                                if isinstance(sim, (np.float32, np.float64)):
                                    sim = float(sim)
                                max_sim = max(max_sim, sim)
                            
                            # MMR score: relevance - lambda * similarity to selected
                            mmr_score = float(lambda_param * rel_score - (1 - lambda_param) * max_sim)
                            
                            if mmr_score > best_score:
                                best_score = float(mmr_score)
                                best_node = (i, node, rel_score, mmr_score)
                        
                        if best_node:
                            idx, node, rel_score, mmr_score = best_node
                            # Add the best node to selected
                            selected.append((node, float(mmr_score)))
                            # Remove from remaining
                            remaining.pop(idx)
                        else:
                            break
                    
                    # Replace original nodes with selected diverse nodes
                    sorted_nodes = selected
                    logger.info(f"Applied diversity reranking with MMR and lambda={lambda_param}")
            except Exception as e:
                logger.warning(f"Error in diversity reranking: {str(e)}")
                # Fall back to sorted nodes without diversity
                sorted_nodes = sorted(node_scores, key=lambda x: x[1], reverse=True)
            
            # Step 6: Apply length normalization to prevent bias towards longer passages
            try:
                length_normalized_nodes = []
                
                for node, score in sorted_nodes:
                    text = node.node.get_content()
                    text_length = len(text.split())
                    
                    # Advanced length normalization with diminishing penalty
                    length_factor = 1.0
                    if text_length < 20:  # Too short
                        length_factor = float(0.8 + (0.2 * text_length / 20))
                    elif text_length > 300:  # Too long
                        # Logarithmic penalty for very long texts to avoid severe penalties
                        import math
                        length_factor = float(1.0 - 0.1 * math.log(1 + (text_length - 300) / 300))
                    
                    # Apply length normalization
                    normalized_score = float(score * length_factor)
                    
                    # Add to node metadata
                    node.node.metadata["original_score"] = float(score)
                    node.node.metadata["length_factor"] = float(length_factor)
                    node.node.metadata["final_score"] = float(normalized_score)
                    
                    # Update the score in the node
                    node.score = float(normalized_score)
                    
                    length_normalized_nodes.append(node)
                
                logger.info(f"Applied advanced length normalization to nodes")
            except Exception as e:
                logger.warning(f"Error in length normalization: {str(e)}")
                # Fall back to original sorted nodes
                length_normalized_nodes = [node for node, _ in sorted_nodes]
            
            logger.info(f"Successfully reranked {len(length_normalized_nodes)} nodes using Reliable RAG techniques")
            return length_normalized_nodes
            
        except Exception as e:
            logger.warning(f"Error in Reliable RAG reranking: {str(e)}, falling back to original scores")
            # Fall back to original vector search scores
            return sorted(nodes, key=lambda node: float(node.score or 0.0), reverse=True)
    
    def detect_hallucination(
        self, 
        query: str, 
        response: str, 
        context: List[str]
    ) -> Dict[str, Any]:
        """
        Advanced hallucination detection based on the NirDiamant/RAG_Techniques repository.
        
        Args:
            query: Original query
            response: Generated response
            context: List of context strings used for generation
            
        Returns:
            Dict with hallucination scores and assessment
        """
        try:
            # Implement advanced hallucination detection techniques
            logger.info(f"Applying advanced hallucination detection")
            
            # 1. Statement Validation - Extract statements from response and check support
            statements = []
            try:
                # Extract factual statements from response using sentence splitting
                sentences = re.split(r'(?<=[.!?])\s+', response)
                sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
                
                # Score each statement for support in context
                from numpy import dot
                from numpy.linalg import norm
                
                statement_support_scores = []
                unsupported_statements = []
                
                for statement in sentences:
                    # Skip questions, exclamations, very short statements
                    if statement.endswith('?') or len(statement) < 15:
                        continue
                    
                    # Get statement embedding
                    statement_embedding = self.embed_model.get_text_embedding(statement)
                    
                    # Compare with context chunks
                    statement_max_score = 0
                    best_supporting_chunk = ""
                    
                    for chunk in context:
                        chunk_embedding = self.embed_model.get_text_embedding(chunk)
                        
                        # Calculate similarity
                        if isinstance(statement_embedding, list) and isinstance(chunk_embedding, list):
                            # Convert to numpy arrays to handle calculations properly
                            statement_array = np.array(statement_embedding)
                            chunk_array = np.array(chunk_embedding)
                            
                            # Ensure we get a real number by taking the real part if complex
                            dot_product = np.dot(statement_array, chunk_array)
                            if isinstance(dot_product, complex):
                                dot_product = dot_product.real
                                
                            norm_statement = np.linalg.norm(statement_array)
                            norm_chunk = np.linalg.norm(chunk_array)
                            
                            # Avoid division by zero
                            if norm_statement > 0 and norm_chunk > 0:
                                cos_sim = float(dot_product / (norm_statement * norm_chunk))
                            else:
                                cos_sim = 0.0
                                
                            # Convert numpy types to Python native types
                            if isinstance(cos_sim, (np.float32, np.float64)):
                                cos_sim = float(cos_sim)
                            elif isinstance(cos_sim, complex):
                                cos_sim = float(cos_sim.real)
                                
                            if cos_sim > statement_max_score:
                                statement_max_score = cos_sim
                                best_supporting_chunk = chunk
                    
                    # Store statement with its support score and supporting chunk
                    statement_support_scores.append(statement_max_score)
                    statements.append({
                        "text": statement,
                        "support_score": float(statement_max_score),  # Ensure it's a Python float
                        "supporting_chunk": best_supporting_chunk[:200] + "..." if len(best_supporting_chunk) > 200 else best_supporting_chunk
                    })
                    
                    # Track unsupported statements
                    if statement_max_score < 0.65:  # Threshold for considering a statement unsupported
                        unsupported_statements.append(statement)
                
                # Calculate faithfulness as average of statement support scores
                if statement_support_scores:
                    faithfulness_score = float(sum(statement_support_scores) / len(statement_support_scores))
                else:
                    faithfulness_score = 0.7  # Default
            except Exception as e:
                logger.warning(f"Error in statement validation: {str(e)}")
                faithfulness_score = 0.7
                statements = []
                unsupported_statements = []
                
            # 2. Citation Analysis - Check if references/citations are faithful to source
            citation_check = {}
            try:
                # Look for citation patterns in the response
                citation_patterns = [
                    r'\[([^\]]+)\]',  # [1], [Source]
                    r'\(([^)]+)\)',   # (Source), (1)
                    r'"([^"]+)"',     # "quoted text"
                    r'according to ([^,.]+)'  # according to X
                ]
                
                all_citations = []
                for pattern in citation_patterns:
                    found = re.findall(pattern, response)
                    all_citations.extend(found)
                
                # If citations found, evaluate their accuracy
                citation_faithfulness = 0.8  # Default assumption
                
                if all_citations:
                    citation_check = {
                        "citations_found": all_citations,
                        "has_citations": True
                    }
                    
                    # Check if nearby text is supported by context
                    # This is a simplified check - we could do more sophisticated citation validation
                    citation_contexts = []
                    for citation in all_citations:
                        # Find text around the citation
                        citation_window_size = 100
                        citation_pos = response.find(citation)
                        if citation_pos > 0:
                            start = max(0, citation_pos - citation_window_size)
                            end = min(len(response), citation_pos + len(citation) + citation_window_size)
                            citation_context = response[start:end]
                            citation_contexts.append(citation_context)
                    
                    # Check support for citation contexts
                    if citation_contexts:
                        citation_support_scores = []
                        for cit_ctx in citation_contexts:
                            cit_embedding = self.embed_model.get_text_embedding(cit_ctx)
                            max_support = 0
                            for chunk in context:
                                chunk_embedding = self.embed_model.get_text_embedding(chunk)
                                
                                # Convert to numpy arrays to handle calculations properly
                                cit_array = np.array(cit_embedding)
                                chunk_array = np.array(chunk_embedding)
                                
                                # Ensure we get a real number by taking the real part if complex
                                dot_product = np.dot(cit_array, chunk_array)
                                if isinstance(dot_product, complex):
                                    dot_product = dot_product.real
                                    
                                norm_cit = np.linalg.norm(cit_array)
                                norm_chunk = np.linalg.norm(chunk_array)
                                
                                # Avoid division by zero
                                if norm_cit > 0 and norm_chunk > 0:
                                    cos_sim = float(dot_product / (norm_cit * norm_chunk))
                                else:
                                    cos_sim = 0.0
                                
                                # Convert numpy types to Python native types
                                if isinstance(cos_sim, (np.float32, np.float64)):
                                    cos_sim = float(cos_sim)
                                max_support = max(max_support, cos_sim)
                            citation_support_scores.append(max_support)
                        
                        if citation_support_scores:
                            citation_faithfulness = float(sum(citation_support_scores) / len(citation_support_scores))
                            citation_check["citation_faithfulness"] = citation_faithfulness
                else:
                    citation_check = {
                        "has_citations": False
                    }
            except Exception as e:
                logger.warning(f"Error in citation analysis: {str(e)}")
                citation_check = {
                    "has_citations": False,
                    "error": str(e)
                }
                citation_faithfulness = 0.8  # Default
            
            # 3. Semantic Similarity between response and overall context
            try:
                response_embedding = self.embed_model.get_text_embedding(response)
                
                # Combine all context into one text for embedding
                combined_context = " ".join(context)
                context_embedding = self.embed_model.get_text_embedding(combined_context)
                
                # Calculate similarity
                if isinstance(response_embedding, list) and isinstance(context_embedding, list):
                    # Convert to numpy arrays to handle calculations properly
                    response_array = np.array(response_embedding)
                    context_array = np.array(context_embedding)
                    
                    # Ensure we get a real number by taking the real part if complex
                    dot_product = np.dot(response_array, context_array)
                    if isinstance(dot_product, complex):
                        dot_product = dot_product.real
                        
                    norm_response = np.linalg.norm(response_array)
                    norm_context = np.linalg.norm(context_array)
                    
                    # Avoid division by zero
                    if norm_response > 0 and norm_context > 0:
                        cos_sim = float(dot_product / (norm_response * norm_context))
                    else:
                        cos_sim = 0.0
                        
                    # Convert numpy types to Python native types
                    if isinstance(cos_sim, (np.float32, np.float64)):
                        cos_sim = float(cos_sim)
                    similarity_score = float(cos_sim)
                else:
                    similarity_score = 0.7  # Default
            except Exception as e:
                logger.warning(f"Error calculating similarity: {str(e)}")
                similarity_score = 0.7
            
            # 4. Query relevance check
            try:
                query_embedding = self.embed_model.get_text_embedding(query)
                response_embedding = self.embed_model.get_text_embedding(response)
                
                if isinstance(query_embedding, list) and isinstance(response_embedding, list):
                    # Convert to numpy arrays to handle calculations properly
                    query_array = np.array(query_embedding)
                    response_array = np.array(response_embedding)
                    
                    # Ensure we get a real number by taking the real part if complex
                    dot_product = np.dot(query_array, response_array)
                    if isinstance(dot_product, complex):
                        dot_product = dot_product.real
                        
                    norm_query = np.linalg.norm(query_array)
                    norm_response = np.linalg.norm(response_array)
                    
                    # Avoid division by zero
                    if norm_query > 0 and norm_response > 0:
                        cos_sim = float(dot_product / (norm_query * norm_response))
                    else:
                        cos_sim = 0.0
                        
                    # Convert numpy types to Python native types
                    if isinstance(cos_sim, (np.float32, np.float64)):
                        cos_sim = float(cos_sim)
                    relevancy_score = float(cos_sim)
                else:
                    relevancy_score = 0.7  # Default
            except Exception as e:
                logger.warning(f"Error calculating relevancy: {str(e)}")
                relevancy_score = 0.7
            
            # 5. Fact Consistency Check (LLM-based evaluation)
            fact_consistency = {}
            try:
                if hasattr(self, 'llm') and len(statements) > 0:
                    # Select a few statements to verify (for efficiency)
                    statements_to_check = sorted(statements, key=lambda x: x["support_score"])[:3]
                    
                    consistency_evaluations = []
                    for statement in statements_to_check:
                        prompt = f"""
                        Evaluate if the following statement is consistent with the provided context.
                        Answer with ONLY a number from 0 to 10, where:
                        - 0 means completely inconsistent or contradicted by the context
                        - 10 means fully supported by the context
                        
                        Statement: "{statement['text']}"
                        
                        Context: 
                        {statement['supporting_chunk']}
                        
                        Consistency score (0-10):
                        """
                        
                        try:
                            response = self.llm.complete(prompt, temperature=0.1)
                            response_text = response.text if hasattr(response, 'text') else str(response)
                            
                            # Extract numeric score
                            score_match = re.search(r'(\d+(\.\d+)?)', response_text)
                            if score_match:
                                consistency_score = float(score_match.group(1)) / 10
                                consistency_evaluations.append({
                                    "statement": statement['text'],
                                    "consistency_score": float(consistency_score)
                                })
                        except Exception as e:
                            logger.warning(f"Error evaluating statement consistency: {str(e)}")
                    
                    if consistency_evaluations:
                        avg_score = sum(e["consistency_score"] for e in consistency_evaluations) / len(consistency_evaluations)
                        fact_consistency = {
                            "evaluations": consistency_evaluations,
                            "average_score": float(avg_score)
                        }
            except Exception as e:
                logger.warning(f"Error in fact consistency check: {str(e)}")
                fact_consistency = {"error": str(e)}
            
            # Calculate weighted hallucination metrics
            # Weight each component based on reliability and completeness
            
            # Basic weights
            faithfulness_weight = 0.5  # Highest importance
            similarity_weight = 0.15
            relevancy_weight = 0.15
            citation_weight = 0.2
            
            # Adjust weights if we have citation evaluation
            if citation_check.get("has_citations", False):
                # If response uses citations, weight them more heavily
                faithfulness_weight = 0.4
                citation_weight = 0.3
                similarity_weight = 0.15
                relevancy_weight = 0.15
            
            # Calculate the scores
            component_scores = {
                "faithfulness": float(faithfulness_score * faithfulness_weight),
                "relevancy": float(relevancy_score * relevancy_weight),
                "similarity": float(similarity_score * similarity_weight),
                "citation": float(citation_faithfulness * citation_weight) if citation_check.get("has_citations", False) else 0.0
            }
            
            # Calculate overall score (inverted to get hallucination probability)
            weighted_sum = sum(component_scores.values())
            total_weight = sum([
                faithfulness_weight, 
                relevancy_weight, 
                similarity_weight,
                citation_weight if citation_check.get("has_citations", False) else 0
            ])
            
            normalized_score = float(weighted_sum / total_weight) if total_weight > 0 else 0.5
            hallucination_probability = float(1.0 - normalized_score)
            
            # Determine threshold based on presence of citations
            hallucination_threshold = 0.25 if citation_check.get("has_citations", False) else 0.3
            # Convert numpy.bool to Python bool if needed
            is_hallucination = bool(hallucination_probability > hallucination_threshold)
            
            # Generate detailed feedback
            feedback = {
                "faithfulness": f"The response contains statements supported by the context with a score of {faithfulness_score:.2f}",
                "relevancy": f"The response is relevant to the query with a score of {relevancy_score:.2f}",
                "similarity": f"The response is similar to the provided context with a score of {similarity_score:.2f}"
            }
            
            if citation_check.get("has_citations", False):
                feedback["citations"] = f"The response includes citations with a faithfulness score of {citation_faithfulness:.2f}"
                
            if unsupported_statements:
                feedback["unsupported_statements"] = f"Found {len(unsupported_statements)} potentially unsupported statements"
                
            if is_hallucination:
                feedback["overall"] = "This response likely contains hallucinations or unsupported statements."
            else:
                feedback["overall"] = "This response appears to be well-grounded in the provided context."
            
            # Complete result
            result = {
                "is_hallucination": bool(is_hallucination),  # Ensure it's a Python bool
                "hallucination_probability": float(hallucination_probability),  # Ensure it's a Python float
                "faithfulness_score": float(faithfulness_score),
                "relevancy_score": float(relevancy_score),
                "similarity_score": float(similarity_score),
                "threshold": float(hallucination_threshold),
                "evaluation_feedback": feedback,
                "component_weights": {
                    "faithfulness": float(faithfulness_weight),
                    "relevancy": float(relevancy_weight),
                    "similarity": float(similarity_weight),
                    "citation": float(citation_weight) if citation_check.get("has_citations", False) else 0.0
                },
                "component_scores": {k: float(v) for k, v in component_scores.items()}  # Ensure all values are Python floats
            }
            
            # Add detailed analysis if available
            if statements:
                result["statement_analysis"] = {
                    "count": len(statements),
                    "unsupported_count": len(unsupported_statements),
                    "sample_statements": statements[:3]  # Include just a few examples
                }
                
            if citation_check:
                result["citation_analysis"] = citation_check
                
            if fact_consistency:
                result["fact_consistency"] = fact_consistency
                
            return result
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {str(e)}")
            return {
                "is_hallucination": False,
                "error": str(e),
                "faithfulness_score": 0.8,  # Conservative default
                "relevancy_score": 0.8,
                "similarity_score": 0.8,
                "evaluation_feedback": {
                    "overall": f"Error in evaluation: {str(e)}"
                }
            }
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        search_filter: Optional[Dict[str, Any]] = None,
        use_hybrid_search: bool = True,
        vector_weight: float = 0.7,
        semantic_weight: float = 0.3
    ) -> List[NodeWithScore]:
        """
        Retrieve context nodes for a given query using hybrid search.
        
        Args:
            query: User query to retrieve context for
            top_k: Number of nodes to retrieve
            search_filter: Optional filter for search
            use_hybrid_search: Whether to use hybrid search
            vector_weight: Weight to give vector search in hybrid (0-1)
            semantic_weight: Weight to give semantic search in hybrid (0-1)
            
        Returns:
            List of NodeWithScore objects
        """
        try:
            # Step 1: Perform vector search
            vector_limit = min(15, top_k * 3)  # Get more results for reranking
            
            # Embed query
            embed_results = self.embed_model.get_text_embedding(query)
            
            # Step 1a: Perform vector search
            if search_filter:
                vector_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embed_results,
                    limit=vector_limit,
                    filter=search_filter
                )
            else:
                vector_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embed_results,
                    limit=vector_limit
                )
                
            # Ensure all scores are valid floats (not complex numbers)
            for result in vector_results:
                if 'score' in result and isinstance(result['score'], complex):
                    result['score'] = float(result['score'].real)
                elif 'score' in result:
                    result['score'] = float(result['score'])
                    
            logger.info(f"Vector search retrieved {len(vector_results)} results")
            
            # Step 1b: Optionally perform semantic search
            semantic_results = []
            if use_hybrid_search and semantic_weight > 0:
                # TODO: Implement semantic search here if available
                # For now, we just use the same vector results and adjust scores
                pass
                
            logger.info(f"Semantic search evaluated {len(semantic_results)} results")
            
            # Step 1c: Combine results with weights
            if use_hybrid_search and semantic_weight > 0:
                # Use our new hybrid search implementation that handles complex numbers
                combined_results = self._hybrid_search(
                    query=query,
                    semantic_results=semantic_results,
                    vector_results=vector_results,
                    k=top_k,
                    vector_weight=vector_weight,
                    semantic_weight=semantic_weight
                )
                # Extract just the nodes with combined scores
                results = [(r[0], r[1]) for r in combined_results]
            else:
                # Just use vector results
                results = [(r, float(r.get('score', 0))) for r in vector_results[:top_k]]
                
            logger.info(f"Retrieved {len(results)} nodes using hybrid search (vector={vector_weight}, semantic={semantic_weight})")
            
            # Step 2: Convert to NodeWithScore objects
            nodes_with_scores = []
            for result, score in results:
                # Ensure the score is a valid float
                if isinstance(score, complex):
                    score = float(score.real)
                else:
                    score = float(score)
                    
                # Get payload from result
                if hasattr(result, 'payload'):
                    # ScoredPoint object
                    payload = result.payload
                else:
                    # Dictionary-like object
                    payload = result.get('payload', {})
                
                # Get text
                text = ""
                metadata = {}
                if isinstance(payload, dict):
                    # First try to get text directly
                    text = payload.get('text', '')
                    
                    # If text is empty, try to extract from _node_content
                    if not text and '_node_content' in payload:
                        try:
                            import json
                            node_content = json.loads(payload['_node_content'])
                            if 'text' in node_content:
                                text = node_content['text']
                        except Exception as e:
                            logger.warning(f"Error extracting text from _node_content: {e}")
                    
                    metadata = payload.get('metadata', {})
                    
                    # If metadata is empty, try to extract from _node_content
                    if (not metadata or len(metadata) == 0) and '_node_content' in payload:
                        try:
                            import json
                            node_content = json.loads(payload['_node_content'])
                            if 'metadata' in node_content:
                                metadata = node_content['metadata']
                        except Exception as e:
                            logger.warning(f"Error extracting metadata from _node_content: {e}")
                else:
                    # If payload is not a dictionary, try to extract directly from result
                    if hasattr(result, 'text'):
                        text = result.text
                    if hasattr(result, 'metadata'):
                        metadata = result.metadata
                
                # Create node
                node = TextNode(
                    text=text,
                    metadata=metadata
                )
                
                # Create NodeWithScore
                node_with_score = NodeWithScore(
                    node=node,
                    score=score
                )
                
                nodes_with_scores.append(node_with_score)
                
            # Step 3: Apply reranking
            reranked_nodes = self._apply_context_reranking(nodes_with_scores, query)
            
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def close(self):
        """Clean up resources."""
        try:
            # Close Qdrant client if available
            if hasattr(self, 'client') and self.client:
                try:
                    logger.info("Closing Qdrant client connection")
                    self.client.close()
                except Exception as e:
                    logger.warning(f"Error closing Qdrant client: {str(e)}")
            
            # Close embedding model if it has a close method
            if hasattr(self, 'embed_model') and hasattr(self.embed_model, 'close'):
                try:
                    logger.info("Closing embedding model")
                    self.embed_model.close()
                except Exception as e:
                    logger.warning(f"Error closing embedding model: {str(e)}")
            
            logger.info("RAG system resources closed successfully")
        except Exception as e:
            logger.error(f"Error closing RAG system resources: {str(e)}")
    
    def repair_document_metadata(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Repair document metadata for documents missing metadata.
        
        Args:
            doc_id: Optional document ID to repair. If None, all documents are repaired.
            
        Returns:
            Dict with repair results
        """
        try:
            start_time = time.time()
            
            # Get document_parser
            from main import document_parser
            if not document_parser:
                return {
                    "status": "error",
                    "message": "Document parser not initialized",
                    "repaired": 0,
                    "failed": 0,
                    "error": "Document parser not initialized"
                }
            
            # If specific doc_id is provided, only repair that document
            if doc_id:
                doc_ids = [doc_id]
            else:
                # Get all document IDs
                try:
                    doc_ids = document_parser.get_all_document_ids()
                except Exception as e:
                    logger.error(f"Error getting document IDs: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error getting document IDs: {str(e)}",
                        "repaired": 0,
                        "failed": 0,
                        "error": str(e)
                    }
            
            if not doc_ids:
                logger.warning("No documents found to repair metadata")
                return {
                    "status": "success",
                    "message": "No documents found to repair metadata",
                    "repaired": 0,
                    "failed": 0,
                    "duration_seconds": time.time() - start_time
                }
            
            # Process each document
            repaired_count = 0
            failed_count = 0
            
            for doc_id in doc_ids:
                try:
                    # Check if document has metadata in the metadata collection
                    meta_points = self.client.scroll(
                        collection_name=self.metadata_collection_name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="doc_id",
                                    match=qdrant_models.MatchValue(value=doc_id)
                                )
                            ]
                        ),
                        limit=1
                    )
                    
                    # If metadata exists, skip unless force=True
                    if meta_points and len(meta_points[0]) > 0:
                        logger.info(f"Document {doc_id} already has metadata")
                        continue
                    
                    # Get document metadata from parser
                    doc_metadata = document_parser.get_document_metadata(doc_id)
                    
                    if not doc_metadata:
                        logger.warning(f"No metadata found for document {doc_id}")
                        failed_count += 1
                        continue
                    
                    # Create payload for metadata
                    payload = {
                        "doc_id": doc_id,
                        "metadata": doc_metadata
                    }
                    
                    # Generate embedding for metadata
                    if isinstance(doc_metadata, dict):
                        # Use title or filename for embedding
                        text_for_embedding = doc_metadata.get('title', doc_metadata.get('filename', ''))
                        if not text_for_embedding:
                            # Combine values as fallback
                            text_for_embedding = ' '.join([str(v) for v in doc_metadata.values() if v])
                    else:
                        text_for_embedding = str(doc_metadata)
                    
                    if text_for_embedding:
                        embedding = self.embed_model.get_text_embedding(text_for_embedding)
                    else:
                        # Use zero vector as fallback
                        embedding = [0.0] * self.embedding_dim
                    
                    # Store in metadata collection
                    self.client.upsert(
                        collection_name=self.metadata_collection_name,
                        points=[
                            qdrant_models.PointStruct(
                                id=doc_id,  # Use doc_id directly as point ID
                                vector=embedding,
                                payload=payload
                            )
                        ]
                    )
                    
                    repaired_count += 1
                    logger.info(f"Repaired metadata for document {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Error repairing metadata for document {doc_id}: {str(e)}")
                    failed_count += 1
            
            duration = time.time() - start_time
            return {
                "status": "success",
                "message": f"Repaired metadata for {repaired_count} documents, {failed_count} failed",
                "repaired": repaired_count,
                "failed": failed_count,
                "duration_seconds": duration
            }
        except Exception as e:
            logger.error(f"Error repairing document metadata: {str(e)}")
            return {
                "status": "error",
                "message": f"Error repairing document metadata: {str(e)}",
                "repaired": 0,
                "failed": 0,
                "error": str(e)
            }
    
    def ensure_language_initialized(self, language_code: str) -> bool:
        """
        Ensure language support is initialized for a specific language.
        
        Args:
            language_code: ISO language code (e.g., 'en', 'fr')
            
        Returns:
            True if language is initialized or initialization isn't needed
        """
        try:
            logger.info(f"Ensuring language support for {language_code}")
            
            # If using multilingual models, no special initialization is needed
            if "multilingual" in self.embedding_model_name.lower():
                logger.info(f"Using multilingual model, no special initialization needed for {language_code}")
                return True
            
            # Different logic based on language
            if language_code == 'fr':
                # Check if we need to load French-specific models
                try:
                    # For demonstration, just log the action
                    logger.info(f"French language support initialized for embedding model {self.embedding_model_name}")
                    return True
                except Exception as e:
                    logger.error(f"Error initializing French language support: {str(e)}")
                    return False
            elif language_code == 'de':
                # German language support
                logger.info(f"German language support initialized for embedding model {self.embedding_model_name}")
                return True
            elif language_code == 'es':
                # Spanish language support
                logger.info(f"Spanish language support initialized for embedding model {self.embedding_model_name}")
                return True
            else:
                # Default to English or multilingual
                logger.info(f"Using default language support for {language_code}")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing language support for {language_code}: {str(e)}")
            return False

    def _hybrid_search(
        self,
        query: str,
        semantic_results: List[Dict[str, Any]] = None,
        vector_results: List[Dict[str, Any]] = None,
        k: int = 5,
        vector_weight: float = 0.7,
        semantic_weight: float = 0.3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Combine vector search and semantic search results.
        
        Args:
            query: User query
            semantic_results: Results from semantic search
            vector_results: Results from vector search
            k: Number of results to return
            vector_weight: Weight to give vector search results
            semantic_weight: Weight to give semantic search results
            
        Returns:
            List of tuples with (node, score)
        """
        # Extract IDs and scores from both result sets
        if vector_results:
            # Extract vector search scores
            vector_scores = {}
            for result in vector_results:
                # Check if result is a dictionary-like object or a ScoredPoint object
                if hasattr(result, 'id') and hasattr(result, 'score'):
                    # It's a ScoredPoint object
                    result_id = str(result.id)
                    score = result.score
                else:
                    # It's a dictionary-like object
                    result_id = str(result.get('id'))
                    score = result.get('score', 0)
                    
                # Ensure score is a real number and not complex
                if isinstance(score, complex):
                    score = score.real
                vector_scores[result_id] = float(score)
        else:
            vector_scores = {}
            
        if semantic_results:
            # Extract semantic search scores
            semantic_scores = {}
            for result in semantic_results:
                # Check if result is a dictionary-like object or a ScoredPoint object
                if hasattr(result, 'id') and hasattr(result, 'score'):
                    # It's a ScoredPoint object
                    result_id = str(result.id)
                    score = result.score
                else:
                    # It's a dictionary-like object
                    result_id = str(result.get('id'))
                    score = result.get('score', 0)
                    
                # Ensure score is a real number and not complex
                if isinstance(score, complex):
                    score = score.real
                semantic_scores[result_id] = float(score)
        else:
            semantic_scores = {}
            
        # Normalize scores between 0 and 1
        if vector_scores:
            max_vector = max(vector_scores.values()) if vector_scores else 1
            min_vector = min(vector_scores.values()) if vector_scores else 0
            range_vector = max_vector - min_vector
            if range_vector > 0:
                vector_scores = {k: float((v - min_vector) / range_vector) for k, v in vector_scores.items()}
        
        if semantic_scores:
            max_semantic = max(semantic_scores.values()) if semantic_scores else 1
            min_semantic = min(semantic_scores.values()) if semantic_scores else 0
            range_semantic = max_semantic - min_semantic
            if range_semantic > 0:
                semantic_scores = {k: float((v - min_semantic) / range_semantic) for k, v in semantic_scores.items()}
                
        # Combine scores
        combined_scores = {}
        all_ids = set(vector_scores.keys()) | set(semantic_scores.keys())
        
        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0)
            s_score = semantic_scores.get(doc_id, 0)
            
            # Apply weights
            combined_score = float((v_score * vector_weight) + (s_score * semantic_weight))
            combined_scores[doc_id] = combined_score
            
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Get top k results
        top_k_ids = sorted_ids[:k]
        
        # Build result list with combined scores
        results = []
        
        # Create mapping of ID to result object
        all_vector_results = {}
        if vector_results:
            for r in vector_results:
                if hasattr(r, 'id'):
                    # ScoredPoint object
                    all_vector_results[str(r.id)] = r
                else:
                    # Dictionary-like object
                    all_vector_results[str(r.get('id'))] = r
                    
        all_semantic_results = {}
        if semantic_results:
            for r in semantic_results:
                if hasattr(r, 'id'):
                    # ScoredPoint object
                    all_semantic_results[str(r.id)] = r
                else:
                    # Dictionary-like object
                    all_semantic_results[str(r.get('id'))] = r
        
        for doc_id in top_k_ids:
            # Prefer vector results for metadata
            if doc_id in all_vector_results:
                result = all_vector_results[doc_id]
            elif doc_id in all_semantic_results:
                result = all_semantic_results[doc_id]
            else:
                continue
                
            score = combined_scores[doc_id]
            results.append((result, score))
            
        return results 