import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from pydantic import BaseModel
from ..graph_components.nodes import DocumentNode, ConceptNode
from .graph_interface import GraphInterface

# Try to import the RAG system - this path might need to be adjusted
try:
    from llamaIndex_rag.rag import RAGSystem
    RAG_SYSTEM_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import RAGSystem. Vector search functionality will be limited.")
    RAG_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)

class HybridSearchResult(BaseModel):
    document: DocumentNode
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    combined_score: float
    source_details: Dict[str, Any] = {} # e.g., {"vector_match": True, "graph_path": "..."}

class EmbeddingIntegrationLayer:
    """Bridges vector embeddings with graph structures for enhanced RAG."""

    def __init__(self,
                 graph_interface: GraphInterface,
                 vector_db_client: Optional[Any] = None, # e.g., RAGSystem or a specific vector store client
                 config: Optional[Dict[str, Any]] = None):
        self.graph_interface = graph_interface
        self.vector_db_client = vector_db_client
        self.config = config if config else {}
        
        # Set default configurations
        self.default_vector_weight = self.config.get("default_vector_weight", 0.6)
        self.default_graph_weight = self.config.get("default_graph_weight", 0.4)
        self.min_vector_threshold = self.config.get("min_vector_threshold", 0.2)
        self.min_graph_threshold = self.config.get("min_graph_threshold", 0.2)
        self.use_concept_expansion = self.config.get("use_concept_expansion", True)
        self.concept_expansion_limit = self.config.get("concept_expansion_limit", 3)
        self.recency_boost_factor = self.config.get("recency_boost_factor", 1.05)
        self.dual_source_boost = self.config.get("dual_source_boost", 1.1)
        
        # Performance monitoring
        self.perf_metrics = {
            "total_searches": 0,
            "avg_search_time": 0,
            "cache_hits": 0
        }
        
        # Simple cache for frequent queries (optional)
        self.result_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_seconds", 300)  # 5 minutes default
        self.max_cache_size = self.config.get("max_cache_size", 100)
        
        logger.info(f"EmbeddingIntegrationLayer initialized with vector client: {vector_db_client is not None}, concept expansion: {self.use_concept_expansion}")

    async def hybrid_search(
        self, 
        query_text: str, 
        query_embedding: Optional[List[float]] = None, # Optional if can be generated
        top_k_vector: int = 5,
        top_k_graph: int = 5,
        top_k_final: int = 5,
        # Concepts relevant to the query, potentially from an earlier step
        related_concepts: Optional[List[ConceptNode]] = None, 
        # Weighting for combining scores, e.g. {"vector": 0.6, "graph": 0.4}
        score_weights: Optional[Dict[str, float]] = None,
        # Additional options
        use_cache: bool = True,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[HybridSearchResult]:
        """
        Performs a hybrid search combining vector similarity and graph-based relevance.

        Args:
            query_text: The search query.
            query_embedding: Pre-computed embedding for the query.
            top_k_vector: Number of results from vector search.
            top_k_graph: Number of results from graph search.
            top_k_final: Final number of results after combination.
            related_concepts: Concepts to leverage for graph search.
            score_weights: Weights for combining vector and graph scores.
            use_cache: Whether to use and update the result cache.
            filter_criteria: Additional filters for search (e.g., by document type).

        Returns:
            A list of HybridSearchResult objects.
        """
        start_time = time.time()
        self.perf_metrics["total_searches"] += 1
        
        logger.info(f"Performing hybrid search for query: {query_text[:50]}...")
        if score_weights is None:
            score_weights = {"vector": self.default_vector_weight, "graph": self.default_graph_weight}

        # Check cache if enabled
        cache_key = None
        if use_cache:
            cache_key = f"{query_text}|v:{top_k_vector}|g:{top_k_graph}|f:{top_k_final}|c:{len(related_concepts) if related_concepts else 0}"
            if filter_criteria:
                cache_key += f"|f:{hash(frozenset(filter_criteria.items()))}"
            
            if cache_key in self.result_cache:
                cache_entry = self.result_cache[cache_key]
                # Check if cache entry is still valid
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    self.perf_metrics["cache_hits"] += 1
                    logger.info(f"Cache hit for query: {query_text[:30]}...")
                    return cache_entry["results"]
                else:
                    # Remove expired entry
                    del self.result_cache[cache_key]

        results_map: Dict[str, HybridSearchResult] = {}

        # 1. Vector Search (if vector_db_client is available)
        if self.vector_db_client and RAG_SYSTEM_AVAILABLE:
            try:
                # Create search parameters
                search_params = {}
                if filter_criteria:
                    search_params["filter"] = filter_criteria
                
                # Assuming vector_db_client has a search method that returns (DocumentNode, score) tuples
                vector_results = await self.vector_db_client.search(
                    query=query_text, 
                    top_k=top_k_vector, 
                    query_embedding=query_embedding,
                    **search_params
                )
                
                for doc_node, v_score in vector_results:
                    # Only include vector results above a threshold
                    if v_score >= self.min_vector_threshold:
                        # Apply recency boost if enabled
                        final_score = v_score
                        recency_info = {}
                        
                        if self.config.get("use_recency_boost", True) and hasattr(doc_node, "last_updated"):
                            # Apply a small boost to newer documents (if last_updated is available)
                            try:
                                if doc_node.last_updated:
                                    # Simple recency boost: multiply by a factor based on recency
                                    # This will vary by implementation but here's a basic approach
                                    now = datetime.now()
                                    if isinstance(doc_node.last_updated, str):
                                        # Convert string to datetime if needed
                                        try:
                                            last_updated = datetime.fromisoformat(doc_node.last_updated)
                                        except ValueError:
                                            # Skip boost if date parsing fails
                                            last_updated = None
                                    else:
                                        last_updated = doc_node.last_updated
                                    
                                    if last_updated:
                                        # Calculate days since last update
                                        days_old = (now - last_updated).days
                                        # Boost newer documents (with a cap)
                                        recency_boost = max(1.0, self.recency_boost_factor - (days_old * 0.01))
                                        final_score = v_score * recency_boost
                                        recency_info = {
                                            "days_old": days_old,
                                            "recency_boost": recency_boost
                                        }
                            except Exception as e:
                                logger.warning(f"Error applying recency boost: {e}")
                        
                        results_map[doc_node.id] = HybridSearchResult(
                            document=doc_node, 
                            vector_score=v_score, 
                            combined_score=final_score * score_weights.get("vector", self.default_vector_weight),
                            source_details={
                                "vector_match": True, 
                                "initial_vector_score": v_score,
                                "recency_info": recency_info
                            }
                        )
                
                logger.info(f"Vector search returned {len(vector_results)} results, {len(results_map)} above threshold")
            except Exception as e:
                logger.error(f"Error during vector search: {e}", exc_info=True)
                # Continue with graph search even if vector search fails
        else:
            logger.warning("Vector DB client not available for hybrid search. Continuing with graph search only.")

        # 2. Graph-enhanced Search
        if self.graph_interface and related_concepts:
            try:
                graph_doc_info: Dict[str, Dict[str, Any]] = {}  # Track documents retrieved by graph, with metadata
                
                # First approach: Get documents directly linked to concepts
                for concept in related_concepts:
                    try:
                        # Find documents that relate to this concept
                        # First try specific method if available (possibly better optimized)
                        try:
                            # Check if the specific method exists
                            linked_docs = []
                            if hasattr(self.graph_interface, "get_documents_containing_concept"):
                                linked_docs = await self.graph_interface.get_documents_containing_concept(
                                    concept_id=concept.id, 
                                    limit=top_k_graph
                                )
                            else:
                                # Use generic method that should always exist
                                linked_docs = await self.graph_interface.get_nodes_linked_from(
                                    source_node_id=concept.id,
                                    relationship_type="CONTAINED_IN",  # This needs to match your schema
                                    target_node_type=DocumentNode,
                                    limit=top_k_graph
                                )
                            
                            # Process the documents found through graph relationships
                            for doc_node in linked_docs:
                                if doc_node.id not in graph_doc_info:
                                    # Initial score for document related to concept
                                    graph_doc_info[doc_node.id] = {
                                        "document": doc_node,
                                        "score": 0.7,  # Base score for direct concept relationship
                                        "concepts": [concept.name],
                                        "paths": [f"direct:{concept.name}"]
                                    }
                                else:
                                    # Boost score if found through multiple concepts
                                    current_info = graph_doc_info[doc_node.id]
                                    current_info["score"] = min(current_info["score"] + 0.1, 1.0)
                                    current_info["concepts"].append(concept.name)
                                    current_info["paths"].append(f"direct:{concept.name}")
                        
                        except Exception as e:
                            logger.error(f"Error retrieving documents for concept {concept.name}: {e}", exc_info=True)
                            # Continue with other concepts
                    
                    except Exception as e:
                        logger.error(f"Error processing concept {concept.name}: {e}", exc_info=True)
                        # Continue with other concepts
                
                # Second approach: Find related concepts to our initial concepts and get their documents
                if self.use_concept_expansion:
                    try:
                        for concept in related_concepts:
                            # Get related concepts (say, 2nd degree connections)
                            related_concept_nodes = await self.graph_interface.get_nodes_linked_from(
                                source_node_id=concept.id,
                                relationship_type="RELATED_TO",  # This needs to match your schema
                                target_node_type=ConceptNode,
                                limit=self.concept_expansion_limit  # Don't go too broad
                            )
                            
                            # For each related concept, get its documents
                            for related_concept in related_concept_nodes:
                                try:
                                    related_docs = await self.graph_interface.get_nodes_linked_from(
                                        source_node_id=related_concept.id,
                                        relationship_type="CONTAINED_IN",  # This needs to match your schema
                                        target_node_type=DocumentNode,
                                        limit=top_k_graph // 2  # Get fewer docs per related concept
                                    )
                                    
                                    # Process with lower base score since these are 2nd degree connections
                                    for doc_node in related_docs:
                                        if doc_node.id not in graph_doc_info:
                                            graph_doc_info[doc_node.id] = {
                                                "document": doc_node,
                                                "score": 0.5,  # Lower score for indirect relationship
                                                "concepts": [f"{concept.name}->{related_concept.name}"],
                                                "paths": [f"indirect:{concept.name}->{related_concept.name}"]
                                            }
                                        else:
                                            # Small boost for additional path
                                            current_info = graph_doc_info[doc_node.id]
                                            current_info["score"] = min(current_info["score"] + 0.05, 1.0)
                                            current_info["concepts"].append(f"{concept.name}->{related_concept.name}")
                                            current_info["paths"].append(f"indirect:{concept.name}->{related_concept.name}")
                                
                                except Exception as e:
                                    logger.error(f"Error retrieving documents for related concept {related_concept.name}: {e}", exc_info=True)
                                    # Continue with other related concepts
                    
                    except Exception as e:
                        logger.error(f"Error in concept expansion: {e}", exc_info=True)
                        # Continue with main algorithm
                
                # Apply filtering if specified
                if filter_criteria and hasattr(doc_node, "attributes"):
                    filtered_graph_doc_info = {}
                    for doc_id, info in graph_doc_info.items():
                        doc_node = info["document"]
                        # Basic attribute filtering (this would need to be customized based on document structure)
                        matches_filter = True
                        for filter_key, filter_value in filter_criteria.items():
                            if filter_key in doc_node.attributes:
                                if doc_node.attributes[filter_key] != filter_value:
                                    matches_filter = False
                                    break
                            else:
                                # If attribute doesn't exist, consider it non-matching
                                matches_filter = False
                                break
                        
                        if matches_filter:
                            filtered_graph_doc_info[doc_id] = info
                    
                    graph_doc_info = filtered_graph_doc_info
                
                # Add graph-based documents to results
                for doc_id, info in graph_doc_info.items():
                    doc_node = info["document"]
                    g_score = info["score"]
                    
                    # Only add results above threshold
                    if g_score >= self.min_graph_threshold:
                        if doc_id in results_map:
                            # Document already in results from vector search
                            result = results_map[doc_id]
                            result.graph_score = g_score
                            result.source_details.update({
                                "graph_match": True,
                                "graph_concepts": info["concepts"],
                                "graph_paths": info["paths"]
                            })
                            # Recalculate combined score with both vector and graph components
                            if result.vector_score is not None:
                                result.combined_score = (
                                    result.vector_score * score_weights.get("vector", self.default_vector_weight) +
                                    g_score * score_weights.get("graph", self.default_graph_weight)
                                )
                            else:
                                result.combined_score = g_score * score_weights.get("graph", 1.0)
                        else:
                            # New document from graph search
                            results_map[doc_id] = HybridSearchResult(
                                document=doc_node,
                                graph_score=g_score,
                                combined_score=g_score * score_weights.get("graph", 1.0),
                                source_details={
                                    "graph_match": True,
                                    "graph_concepts": info["concepts"],
                                    "graph_paths": info["paths"]
                                }
                            )
                
                logger.info(f"Graph search found {len(graph_doc_info)} documents")
            
            except Exception as e:
                logger.error(f"Error during graph-enhanced search: {e}", exc_info=True)
                # Continue with what we have
        else:
            if not related_concepts:
                logger.warning("No related concepts provided for graph search")
            if not self.graph_interface:
                logger.warning("Graph interface not available for hybrid search")

        # 3. Combine, filter and rank final results
        final_results = []
        for doc_id, result in results_map.items():
            # Ensure we have a valid document
            if result.document:
                # Apply any final modifications to score if needed
                # For example, boost documents that have both vector and graph scores
                if result.vector_score is not None and result.graph_score is not None:
                    # Optional boost for docs found through both methods
                    result.combined_score *= self.dual_source_boost
                    result.source_details["dual_source_boost"] = self.dual_source_boost
                
                final_results.append(result)
        
        # Sort by combined score and take top_k_final
        sorted_results = sorted(final_results, key=lambda x: x.combined_score, reverse=True)
        top_results = sorted_results[:top_k_final]
        
        # Update performance metrics
        search_time = time.time() - start_time
        self.perf_metrics["avg_search_time"] = (
            (self.perf_metrics["avg_search_time"] * (self.perf_metrics["total_searches"] - 1) + search_time) / 
            self.perf_metrics["total_searches"]
        )
        
        # Update cache if enabled
        if use_cache and cache_key:
            # Limit cache size by removing oldest entries if needed
            if len(self.result_cache) >= self.max_cache_size:
                oldest_key = min(self.result_cache.items(), key=lambda x: x[1]["timestamp"])[0]
                del self.result_cache[oldest_key]
            
            self.result_cache[cache_key] = {
                "results": top_results,
                "timestamp": time.time()
            }
        
        logger.info(f"Hybrid search generated {len(final_results)} candidates, returning top {len(top_results)}")
        return top_results

    async def update_embeddings_from_graph_learning(self, concept_id: str, updated_related_concepts: List[ConceptNode]):
        """
        Updates embeddings based on graph learning (e.g., new strong relationships).
        
        This implements a simplified version where we might update document metadata or
        concept relationships in the vector store based on graph learning.
        """
        if not self.vector_db_client:
            logger.warning(f"Cannot update embeddings: vector DB client not available")
            return
        
        try:
            # 1. Get the concept that's being updated
            concept = await self.graph_interface.get_node_by_id(concept_id, ConceptNode)
            if not concept:
                logger.error(f"Cannot update embeddings: concept {concept_id} not found")
                return
            
            # 2. Get documents containing this concept
            docs_with_concept = await self.graph_interface.get_nodes_linked_from(
                source_node_id=concept_id,
                relationship_type="CONTAINED_IN",  # Adjust to match your schema
                target_node_type=DocumentNode
            )
            
            # 3. Update document metadata in vector store (if supported)
            # This is highly dependent on your vector DB implementation
            if hasattr(self.vector_db_client, "update_document_metadata"):
                for doc in docs_with_concept:
                    # Create updated metadata with concept relationships
                    related_concept_names = [rc.name for rc in updated_related_concepts]
                    
                    # Get timestamp for update
                    current_time = datetime.now().isoformat()
                    
                    updated_metadata = {
                        "related_concepts": related_concept_names,
                        "last_updated": current_time,
                        "concept_relationships_updated": True
                    }
                    
                    # Update the document's metadata in the vector store
                    await self.vector_db_client.update_document_metadata(
                        doc_id=doc.id,
                        metadata=updated_metadata
                    )
                    
                logger.info(f"Updated vector DB metadata for {len(docs_with_concept)} documents related to concept '{concept.name}'")
            else:
                logger.warning("Vector DB client doesn't support updating document metadata")
            
            # 4. Optionally, update concept representations if your system supports it
            if hasattr(self.vector_db_client, "update_concept_relationships"):
                relationship_data = []
                for related_concept in updated_related_concepts:
                    relationship_data.append({
                        "concept_id": related_concept.id,
                        "concept_name": related_concept.name,
                        "relationship_strength": 0.8  # Could be computed elsewhere
                    })
                
                await self.vector_db_client.update_concept_relationships(
                    concept_id=concept_id,
                    concept_name=concept.name,
                    related_concepts=relationship_data
                )
                logger.info(f"Updated concept relationships for '{concept.name}' with {len(relationship_data)} related concepts")
            
        except Exception as e:
            logger.error(f"Error updating embeddings from graph learning: {e}", exc_info=True)

    async def manage_embedding_refresh(self, document_node: DocumentNode, event_type: str = "update"):
        """
        Manages refreshing embeddings when documents change (CRUD operations).
        
        Args:
            document_node: The document that has been created, updated, or deleted.
            event_type: "create", "update", or "delete".
        """
        if not self.vector_db_client:
            logger.warning(f"Cannot manage embedding refresh: vector DB client not available")
            return
        
        try:
            if event_type in ["create", "update"]:
                # Check if vector_db_client has an appropriate method
                if hasattr(self.vector_db_client, "index_document"):
                    # Extract concepts and metadata for better indexing
                    document_concepts = []
                    
                    # Try to get linked concepts if possible
                    if self.graph_interface:
                        try:
                            concepts = await self.graph_interface.get_nodes_linked_to(
                                target_node_id=document_node.id,
                                relationship_type="CONTAINED_IN",  # Assuming this relationship direction
                                source_node_type=ConceptNode
                            )
                            document_concepts = [c.name for c in concepts]
                        except Exception as e:
                            logger.warning(f"Error retrieving concepts for document {document_node.id}: {e}")
                    
                    # Enrich document with concepts if found
                    if document_concepts and hasattr(document_node, "attributes"):
                        # Update document attributes with concepts for better indexing
                        document_node.attributes["related_concepts"] = document_concepts
                        document_node.attributes["last_indexed"] = datetime.now().isoformat()
                    
                    # Index or re-index the document
                    await self.vector_db_client.index_document(document_node)
                    logger.info(f"Successfully {event_type}d document {document_node.id} in vector store")
                    
                    # Clear cache entries that might include this document
                    self._clear_related_cache_entries(document_node.id)
                    
                else:
                    logger.warning(f"Vector DB client doesn't support indexing documents")
            
            elif event_type == "delete":
                # Check if vector_db_client has an appropriate method
                if hasattr(self.vector_db_client, "delete_document"):
                    # Delete the document from the vector store
                    await self.vector_db_client.delete_document(document_node.id)
                    logger.info(f"Successfully deleted document {document_node.id} from vector store")
                    
                    # Clear any cache entries that might include this document
                    self._clear_related_cache_entries(document_node.id)
                    
                else:
                    logger.warning(f"Vector DB client doesn't support deleting documents")
            
            else:
                logger.warning(f"Unknown event type: {event_type}")
        
        except Exception as e:
            logger.error(f"Error managing embedding refresh for document {document_node.id} ({event_type}): {e}", exc_info=True)

    def _clear_related_cache_entries(self, document_id: str):
        """Helper method to clear cache entries that might contain a specific document."""
        # This is a simple implementation - just clear the entire cache
        # A more sophisticated approach would track document IDs in results
        self.result_cache.clear()
        logger.debug(f"Cleared result cache after document {document_id} was modified")

    async def align_vector_similarity_with_graph(self, concept1: ConceptNode, concept2: ConceptNode, graph_relationship_strength: float):
        """
        Attempts to align vector similarity with graph relationship strength.
        
        This could involve adjusting parameters in a hybrid search, or influencing
        embedding models based on graph relationships.
        """
        if not self.vector_db_client:
            logger.warning(f"Cannot align vector similarity: vector DB client not available")
            return
        
        # Store this relationship in our configuration to influence hybrid search
        relationship_key = f"{concept1.name}_{concept2.name}"
        
        # Update score weights in our config
        if "concept_relationship_weights" not in self.config:
            self.config["concept_relationship_weights"] = {}
        
        self.config["concept_relationship_weights"][relationship_key] = graph_relationship_strength
        
        logger.info(f"Updated relationship weight between '{concept1.name}' and '{concept2.name}' to {graph_relationship_strength}")
        
        # If the vector DB supports explicit concept relationship updating
        if hasattr(self.vector_db_client, "update_concept_pair_relationship"):
            try:
                await self.vector_db_client.update_concept_pair_relationship(
                    concept1_id=concept1.id,
                    concept2_id=concept2.id,
                    relationship_strength=graph_relationship_strength
                )
                logger.info(f"Updated concept relationship in vector DB: {concept1.name} - {concept2.name}")
            except Exception as e:
                logger.error(f"Error updating concept relationship in vector DB: {e}")
        
        # Clear result cache after updating relationships
        self.result_cache.clear() 