import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .graph_interface import GraphInterface
from ..graph_components.nodes import ConceptNode, DocumentNode, QueryNode

logger = logging.getLogger(__name__)

class EmbeddingConfig(BaseModel):
    """Configuration for the embedding integration layer."""
    default_embedding_dim: int = Field(default=1536, description="Default dimension for embeddings")
    index_refresh_interval: int = Field(default=3600, description="Interval in seconds for refreshing the vector index")
    hybrid_search_weight: float = Field(default=0.7, description="Weight for vector similarity in hybrid search (0-1)")
    max_concept_distance: float = Field(default=0.75, description="Maximum cosine distance for concepts to be considered related")
    store_embeddings_in_graph: bool = Field(default=True, description="Whether to store embeddings in the graph nodes")
    use_cached_embeddings: bool = Field(default=True, description="Whether to use cached embeddings when available")

class SearchResult(BaseModel):
    """Structure for search results from the embedding integration layer."""
    node_id: str
    node_type: str
    similarity_score: float
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingIntegrationLayer:
    """
    Bridges between vector embeddings and graph data.
    
    This layer provides functionality for:
    1. Embedding generation and management for graph nodes
    2. Vector similarity search across node types
    3. Hybrid search combining vector similarity with graph properties
    4. Embedding-based concept mapping
    5. Dynamic weighting based on contextual relevance
    """
    
    def __init__(
        self, 
        embedding_service: Any,
        graph_interface: GraphInterface,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the embedding integration layer.
        
        Args:
            embedding_service: The service that generates embeddings (implementation-specific)
            graph_interface: Interface to the knowledge graph
            config: Configuration options for the embedding layer
        """
        self.embedding_service = embedding_service
        self.graph = graph_interface
        self.config = config or EmbeddingConfig()
        self._vector_cache = {}  # Temp memory cache for frequently used vectors
        logger.info("EmbeddingIntegrationLayer initialized")
    
    async def get_embedding_for_text(self, text: str, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Get embedding vector for a text string.
        
        Args:
            text: The text to embed
            cache_key: Optional key to cache the embedding result
            
        Returns:
            Numpy array of embedding vector
        """
        # Check cache first if a cache key is provided
        if cache_key and cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
            
        # Generate embedding using the provided service
        try:
            embedding = await self.embedding_service.embed_text(text)
            
            # Cache if requested
            if cache_key:
                self._vector_cache[cache_key] = embedding
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}", exc_info=True)
            # Return zeros array as fallback
            return np.zeros(self.config.default_embedding_dim)
    
    async def update_node_embedding(self, node_id: str, node_type: str, text_for_embedding: str) -> bool:
        """
        Update the embedding for a specific node in the graph.
        
        Args:
            node_id: ID of the node to update
            node_type: Type of the node (e.g., 'Document', 'Concept')
            text_for_embedding: Text content to generate embedding from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate the embedding
            embedding = await self.get_embedding_for_text(text_for_embedding)
            
            # Convert to list for storage
            embedding_list = embedding.tolist()
            
            # Update the node properties in the graph
            properties_to_update = {
                "embedding": embedding_list,
                "embedding_updated_at": datetime.utcnow().isoformat()
            }
            
            success = await self.graph.update_node_properties(
                node_id=node_id,
                properties_to_update=properties_to_update,
                node_label=node_type
            )
            
            return success
        except Exception as e:
            logger.error(f"Error updating embedding for node {node_id}: {e}", exc_info=True)
            return False
    
    async def embed_document(self, document: DocumentNode) -> DocumentNode:
        """
        Generate and store embeddings for a document node.
        
        Args:
            document: The DocumentNode to embed
            
        Returns:
            Updated DocumentNode with embedding
        """
        # Check if we need to generate a new embedding
        if (hasattr(document, "embedding") and document.embedding is not None and 
            self.config.use_cached_embeddings):
            logger.info(f"Using existing embedding for document {document.id}")
            return document
            
        # Determine text to embed - this depends on your document structure
        text_to_embed = document.content
        if hasattr(document, "title") and document.title:
            text_to_embed = f"{document.title}\n\n{text_to_embed}"
            
        # Generate embedding
        embedding = await self.get_embedding_for_text(text_to_embed)
        
        # Update document with embedding
        if self.config.store_embeddings_in_graph:
            await self.update_node_embedding(
                node_id=document.id,
                node_type="Document",
                text_for_embedding=text_to_embed
            )
        
        # Update the local object
        document.embedding = embedding.tolist()
        return document
    
    async def embed_concept(self, concept: ConceptNode) -> ConceptNode:
        """
        Generate and store embeddings for a concept node.
        
        Args:
            concept: The ConceptNode to embed
            
        Returns:
            Updated ConceptNode with embedding
        """
        # Check if we can use existing embedding
        if (hasattr(concept, "embedding") and concept.embedding is not None and
            self.config.use_cached_embeddings):
            logger.info(f"Using existing embedding for concept {concept.id}")
            return concept
            
        # Determine text to embed
        text_to_embed = concept.name
        if hasattr(concept, "definition") and concept.definition:
            text_to_embed = f"{concept.name}: {concept.definition}"
            
        # Generate embedding
        embedding = await self.get_embedding_for_text(text_to_embed)
        
        # Update concept with embedding
        if self.config.store_embeddings_in_graph:
            await self.update_node_embedding(
                node_id=concept.id,
                node_type="Concept",
                text_for_embedding=text_to_embed
            )
        
        # Update the local object
        concept.embedding = embedding.tolist()
        return concept
    
    async def hybrid_search(
        self,
        query_text: str,
        node_type: str = "Document",
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: Optional[float] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Perform a hybrid search that combines vector similarity with graph relationship strength.
        
        Args:
            query_text: The query text to search for
            node_type: Type of nodes to search (Document, Concept, etc.)
            filters: Optional filters to apply to search results
            vector_weight: Weight for vector similarity vs. graph relationships (0-1)
                          If None, uses config.hybrid_search_weight
            limit: Maximum number of results to return
            
        Returns:
            List of search results with combined scores
        """
        # Use configured weight if not specified
        if vector_weight is None:
            vector_weight = self.config.hybrid_search_weight
        
        # Ensure vector_weight is in valid range
        vector_weight = max(0.0, min(1.0, vector_weight))
        graph_weight = 1.0 - vector_weight
        
        try:
            # Step 1: Get query embedding
            query_embedding = await self.get_embedding_for_text(query_text)
            
            # Step 2: Extract concepts from query to enhance graph search
            from ..processing_nodes.input_processor import extract_key_concepts
            query_concepts = await extract_key_concepts(query_text, self.graph)
            
            # Step 3: Find documents through vector search
            vector_results = []
            if vector_weight > 0:
                if hasattr(self.embedding_service, 'search'):
                    # Use the embedding service's native search if available
                    vector_results = await self.embedding_service.search(
                        query_text=query_text,
                        collection_name=node_type,
                        filters=filters,
                        limit=limit*2  # Get more results for hybrid scoring
                    )
                else:
                    # Fallback: Get all nodes and compute similarity ourselves
                    # This would be inefficient in a real system but works for demonstration
                    logger.warning("Using inefficient vector search fallback")
                    # Get nodes from graph database (implementation would depend on your system)
                    all_nodes = await self._get_all_nodes_of_type(node_type, limit=100)  # Get a reasonable number
                    for node in all_nodes:
                        if not hasattr(node, 'embedding') or not node.embedding:
                            continue
                        node_embedding = np.array(node.embedding)
                        similarity = self.cosine_similarity(query_embedding, node_embedding)
                        vector_results.append({
                            'node_id': node.id,
                            'similarity': similarity,
                            'node': node
                        })
                    # Sort by similarity
                    vector_results.sort(key=lambda x: x['similarity'], reverse=True)
                    vector_results = vector_results[:limit*2]
            
            # Step 4: Find documents through graph relationships
            graph_results = []
            if graph_weight > 0 and query_concepts:
                # Find documents related to extracted concepts
                for concept in query_concepts:
                    concept_docs = await self.graph.get_documents_by_concept(
                        concept_id=concept.id,
                        limit=limit
                    )
                    for doc in concept_docs:
                        # Get relationship strength from graph
                        rel_exists = await self.graph.check_relationship_exists(
                            concept.id, doc.id, "CONTAINS"
                        )
                        # Add to graph results
                        if rel_exists:
                            # You would ideally get an actual weight/strength value here
                            relationship_strength = 0.8  # Default strength if not available
                            graph_results.append({
                                'node_id': doc.id,
                                'similarity': relationship_strength,
                                'node': doc
                            })
            
            # Step 5: Combine results with weighted scoring
            combined_results = {}
            
            # Process vector results
            for result in vector_results:
                node_id = result['node_id']
                if node_id not in combined_results:
                    combined_results[node_id] = {
                        'node_id': node_id,
                        'node': result.get('node'),
                        'vector_score': result['similarity'],
                        'graph_score': 0.0
                    }
                else:
                    combined_results[node_id]['vector_score'] = result['similarity']
            
            # Process graph results
            for result in graph_results:
                node_id = result['node_id']
                if node_id not in combined_results:
                    combined_results[node_id] = {
                        'node_id': node_id,
                        'node': result.get('node'),
                        'vector_score': 0.0,
                        'graph_score': result['similarity']
                    }
                else:
                    combined_results[node_id]['graph_score'] = max(
                        combined_results[node_id]['graph_score'],
                        result['similarity']
                    )
            
            # Calculate combined scores
            final_results = []
            for node_id, data in combined_results.items():
                combined_score = (
                    vector_weight * data['vector_score'] + 
                    graph_weight * data['graph_score']
                )
                
                node = data['node']
                if not node and self.graph:
                    # Retrieve node if we only have the ID
                    if node_type == "Document":
                        node = await self.graph.get_node_by_id(node_id, DocumentNode)
                    elif node_type == "Concept":
                        node = await self.graph.get_node_by_id(node_id, ConceptNode)
                
                # Create search result
                if node:
                    text_content = getattr(node, 'content', 
                                  getattr(node, 'definition', 
                                  getattr(node, 'name', '')))
                    
                    metadata = getattr(node, 'metadata', {})
                    if hasattr(node, 'title') and node.title:
                        metadata['title'] = node.title
                    if hasattr(node, 'source') and node.source:
                        metadata['source'] = node.source
                    
                    result = SearchResult(
                        node_id=node_id,
                        node_type=node_type,
                        similarity_score=combined_score,
                        text_content=text_content,
                        metadata=metadata
                    )
                    final_results.append(result)
            
            # Sort results by combined score
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top results
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}", exc_info=True)
            return []

    async def _get_all_nodes_of_type(self, node_type: str, limit: int = 100) -> List[Any]:
        """Helper method to get nodes of a specific type."""
        # This is a placeholder - implementation would depend on your system
        # In a real system, you would have a more efficient way to get nodes
        nodes = []
        try:
            # Use the appropriate method based on node type
            if node_type == "Document":
                # Implement document retrieval logic
                pass
            elif node_type == "Concept":
                # Implement concept retrieval logic
                pass
        except Exception as e:
            logger.error(f"Error getting nodes of type {node_type}: {e}")
        return nodes
    
    async def find_related_concepts(
        self,
        text_or_concept: Union[str, ConceptNode],
        max_distance: Optional[float] = None,
        limit: int = 5
    ) -> List[Tuple[ConceptNode, float]]:
        """
        Find concepts related to given text or concept.
        
        Args:
            text_or_concept: Text string or ConceptNode to find related concepts for
            max_distance: Maximum semantic distance (1-similarity) to include
            limit: Maximum number of related concepts to return
            
        Returns:
            List of (ConceptNode, similarity_score) tuples
        """
        try:
            # Get the embedding for the source
            if isinstance(text_or_concept, str):
                # This is text, generate embedding
                source_embedding = await self.get_embedding_for_text(text_or_concept)
            else:
                # This is a concept, get or generate its embedding
                concept = await self.embed_concept(text_or_concept)
                source_embedding = np.array(concept.embedding)
            
            # Set max distance
            max_dist = max_distance if max_distance is not None else self.config.max_concept_distance
            
            # Placeholder for real implementation
            # In a real system, you would:
            # 1. Search your vector index for similar concept embeddings
            # 2. Filter by distance threshold
            # 3. Retrieve the actual concept nodes
            
            logger.warning("Find related concepts is currently a placeholder")
            
            # Return empty list for now
            return []
            
        except Exception as e:
            logger.error(f"Error finding related concepts: {e}", exc_info=True)
            return []
    
    async def attach_embeddings_to_search_results(
        self,
        search_results: List[Dict[str, Any]],
        text_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Generate and attach embeddings to search results.
        
        Args:
            search_results: List of search result dictionaries
            text_key: Key in each result dictionary that contains the text to embed
            
        Returns:
            Search results with embeddings added
        """
        enhanced_results = []
        
        for result in search_results:
            if text_key in result and result[text_key]:
                # Generate embedding
                embedding = await self.get_embedding_for_text(result[text_key])
                # Add to result
                result["embedding"] = embedding.tolist()
            enhanced_results.append(result)
            
        return enhanced_results
    
    async def embed_query(self, query: Union[str, QueryNode]) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text or QueryNode to embed
            
        Returns:
            Embedding vector
        """
        if isinstance(query, str):
            # This is just text
            return await self.get_embedding_for_text(query)
        else:
            # This is a QueryNode
            query_text = query.query_text
            if hasattr(query, "original_user_input") and query.original_user_input:
                # Use both reformulated query and original input for better semantic representation
                query_text = f"{query.query_text}\n{query.original_user_input}"
            return await self.get_embedding_for_text(query_text, cache_key=f"query_{query.id}")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def clear_cache(self) -> None:
        """Clear the vector cache."""
        self._vector_cache = {}
        logger.info("Vector cache cleared")
        
    async def update_all_document_embeddings(self, batch_size: int = 10) -> Tuple[int, int]:
        """
        Update embeddings for all documents in the graph.
        
        Args:
            batch_size: Number of documents to process in each batch
            
        Returns:
            Tuple of (success_count, failed_count)
        """
        # This is a placeholder implementation
        # In a real system, you would:
        # 1. Query all documents from the graph
        # 2. Process them in batches
        # 3. Update embeddings for each document
        
        logger.info("Document embedding update is a placeholder - no documents processed")
        return (0, 0) 