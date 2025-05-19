# plugins/regul_aite/backend/llamaIndex_rag/rag.py
import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
import uuid
import openai

# Updated imports for the current LlamaIndex structure
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.core.retrievers import BaseRetriever

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from neo4j import GraphDatabase
import time
import numpy as np
import Stemmer

from data_enrichment.language_detector import LanguageDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try importing BM25Retriever from different locations
try:
    from llama_index_retrievers_bm25 import BM25Retriever
except ImportError:
  try:
      from llama_index_retrievers_bm25.retrievers import BM25Retriever
  except ImportError:
      try:
          from llama_index.retrievers.bm25 import BM25Retriever
      except ImportError:
          logger.error("Failed to import BM25Retriever, hybrid search will be disabled")
          BM25Retriever = None

# Define BaseRetrieverOutput
class BaseRetrieverOutput:
    def __init__(self, nodes):
        self.nodes = nodes

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines vector search and BM25 keyword search.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BM25Retriever,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 5
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_retriever: Vector-based retriever
            keyword_retriever: Keyword-based retriever
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            top_k: Number of results to retrieve
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k

        # Normalize weights if they don't sum to 1
        total_weight = vector_weight + keyword_weight
        if abs(total_weight - 1.0) > 1e-9:
            self.vector_weight = vector_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight

        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> BaseRetrieverOutput:
        """
        Retrieve nodes using hybrid search.

        Args:
            query_bundle: Query bundle

        Returns:
            BaseRetrieverOutput containing retrieved nodes
        """
        # Get vector search results
        vector_results = self.vector_retriever.retrieve(query_bundle)

        # Get keyword search results
        keyword_results = self.keyword_retriever.retrieve(query_bundle)

        # Combine results
        node_dict = {}

        # Add vector results with vector_weight
        for node in vector_results:
            node_id = node.node.node_id
            node_dict[node_id] = {
                "node": node.node,
                "score": node.score * self.vector_weight,
                "source": "vector"
            }

        # Add or update with keyword results
        for node in keyword_results:
            node_id = node.node.node_id
            if node_id in node_dict:
                # If already in results, add the weighted keyword score
                node_dict[node_id]["score"] += node.score * self.keyword_weight
                node_dict[node_id]["source"] = "both"
            else:
                # If not in results, add with the keyword score
                node_dict[node_id] = {
                    "node": node.node,
                    "score": node.score * self.keyword_weight,
                    "source": "keyword"
                }

        # Sort by score and take top_k
        sorted_nodes = sorted(
            node_dict.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:self.top_k]

        # Convert to NodeWithScore format
        from llama_index.core.schema import NodeWithScore
        results = [
            NodeWithScore(
                node=item["node"],
                score=item["score"]
            ) for item in sorted_nodes
        ]

        return BaseRetrieverOutput(nodes=results)

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Override the base retrieve method to ensure we always return a List[NodeWithScore].
        
        Args:
            query_bundle: Query bundle
            
        Returns:
            List of NodeWithScore objects
        """
        output = self._retrieve(query_bundle)
        if hasattr(output, 'nodes'):
            return output.nodes
        elif isinstance(output, list):
            return output
        else:
            logger.error(f"Unexpected output type from _retrieve: {type(output)}")
            return []

class HierarchicalRetriever(BaseRetriever):
    """
    Retriever that considers document hierarchy for better contextual retrieval.
    Enhances retrieval by considering the document's hierarchical structure and section context.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        neo4j_driver,
        top_k: int = 5,
        parent_boost: float = 0.2,
        sibling_boost: float = 0.1,
        context_window: int = 2
    ):
        """
        Initialize the hierarchical retriever.

        Args:
            base_retriever: The underlying retriever to use for initial retrieval
            neo4j_driver: Neo4j driver for retrieving document structure
            top_k: Number of results to retrieve
            parent_boost: Score boost for parent sections
            sibling_boost: Score boost for sibling sections
            context_window: Number of siblings to consider before and after a retrieved chunk
        """
        self.base_retriever = base_retriever
        self.neo4j_driver = neo4j_driver
        self.top_k = top_k
        self.parent_boost = parent_boost
        self.sibling_boost = sibling_boost
        self.context_window = context_window
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> BaseRetrieverOutput:
        """
        Retrieve nodes using hierarchical context-aware retrieval.

        Args:
            query_bundle: Query bundle

        Returns:
            BaseRetrieverOutput containing retrieved nodes
        """
        # Get base results - CALL _retrieve DIRECTLY to avoid problematic base implementation
        try:
            base_result = self.base_retriever._retrieve(query_bundle)
            
            # Handle different return types
            base_nodes_with_scores = []
            if hasattr(base_result, 'nodes'):
                base_nodes_with_scores = base_result.nodes
            elif isinstance(base_result, list):
                base_nodes_with_scores = base_result
            else:
                logger.error(f"Unexpected base retriever output type: {type(base_result)}")
                return BaseRetrieverOutput(nodes=[])
        except Exception as e:
            logger.error(f"Error calling base retriever: {str(e)}")
            return BaseRetrieverOutput(nodes=[])

        # Check if we have Neo4j connectivity before trying hierarchical retrieval
        if not self.neo4j_driver:
            logger.warning("No Neo4j driver available, skipping hierarchical retrieval")
            return BaseRetrieverOutput(nodes=base_nodes_with_scores)

        # Extract node IDs from base results
        initial_nodes = {}
        for node in base_nodes_with_scores:
            node_id = node.node.node_id
            initial_nodes[node_id] = {
                "node": node.node,
                "score": node.score,
                "source": "direct"
            }

        # If no results or too few, just return what we have
        if len(initial_nodes) < 2:
            return BaseRetrieverOutput(nodes=base_nodes_with_scores)

        # Get hierarchical context from Neo4j
        augmented_nodes = self._augment_with_hierarchical_context(initial_nodes)

        # Sort by score and take top_k
        sorted_nodes = sorted(
            augmented_nodes.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:self.top_k]

        # Convert to NodeWithScore format
        from llama_index.core.schema import NodeWithScore
        results = [
            NodeWithScore(
                node=item["node"],
                score=item["score"]
            ) for item in sorted_nodes
        ]

        return BaseRetrieverOutput(nodes=results)

    def _augment_with_hierarchical_context(self, initial_nodes: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Augment the initial nodes with hierarchical context from Neo4j.

        Args:
            initial_nodes: Dictionary of initial nodes with node_id as key

        Returns:
            Dictionary of augmented nodes with additional context
        """
        try:
            # Get the IDs of initial nodes
            node_ids = list(initial_nodes.keys())

            # Find parent, sibling, and child nodes in Neo4j
            with self.neo4j_driver.session() as session:
                # Query for hierarchical relationships
                # MODIFIED QUERY:
                # - Uses c.order_index
                # - Identifies parent Section node via Document and c.section property
                # - Identifies sibling Chunks via shared Document and section property
                result = session.run("""
                    MATCH (c:Chunk) WHERE c.id IN $node_ids
                    
                    // Find the Document this chunk belongs to
                    OPTIONAL MATCH (doc:Document)-[:CONTAINS]->(c)

                    // Directly find sections/chunks based on section name without requiring HAS_SECTION relationship
                    WITH c, doc
                    
                    // Get parent section info directly from chunk properties
                    WITH c, doc,
                         CASE WHEN c.section IS NOT NULL 
                              THEN {id: c.section, name: c.section, type: 'parent_section'} 
                              ELSE NULL 
                         END as parent_section_info
                    
                    // Find sibling Chunks: other chunks in the same document and same section name
                    WITH c, parent_section_info
                    OPTIONAL MATCH (sibling_c_node:Chunk {doc_id: c.doc_id, section: c.section})
                    WHERE sibling_c_node.id <> c.id

                    // Sequence information using order_index
                    WITH c, parent_section_info, sibling_c_node,
                         CASE
                           WHEN sibling_c_node IS NOT NULL AND c.order_index IS NOT NULL AND sibling_c_node.order_index IS NOT NULL
                           THEN abs(sibling_c_node.order_index - c.order_index)
                           ELSE null
                         END as distance
                    WHERE distance IS NULL OR distance <= $context_window
                    
                    // Group by chunk ID to fix the aggregation issue
                    WITH c.id as chunk_id, 
                         CASE WHEN parent_section_info IS NOT NULL 
                              THEN collect(DISTINCT parent_section_info) 
                              ELSE [] 
                         END as parent_section_nodes,
                         collect(DISTINCT {id: sibling_c_node.chunk_id, type: 'sibling_chunk', distance: distance}) as sibling_chunk_nodes
                    
                    RETURN chunk_id, 
                           parent_section_nodes,
                           sibling_chunk_nodes
                """, node_ids=node_ids, context_window=self.context_window)

                # Process results and add to initial nodes
                augmented_nodes = initial_nodes.copy()

                # Track additional nodes to fetch
                additional_node_ids = set()

                # Store raw query results for second pass to avoid re-iteration or complex data structures
                processed_records = list(result)

                # First pass: identify additional nodes to retrieve
                for record in processed_records:
                    # chunk_id = record["chunk_id"] # Chunk itself is already in initial_nodes
                    parent_section_nodes = record["parent_section_nodes"]
                    sibling_chunk_nodes = record["sibling_chunk_nodes"]

                    # Add parent section IDs
                    for parent_sec_node in parent_section_nodes:
                        if parent_sec_node["id"] and parent_sec_node["id"] not in augmented_nodes:
                            additional_node_ids.add(parent_sec_node["id"])

                    # Add sibling chunk IDs
                    for sibling_chunk_node in sibling_chunk_nodes:
                        if sibling_chunk_node["id"] and sibling_chunk_node["id"] not in augmented_nodes:
                            additional_node_ids.add(sibling_chunk_node["id"])

                # If we have additional nodes to fetch, get them from Neo4j
                if additional_node_ids:
                    # Convert to list for Neo4j query
                    add_ids = list(additional_node_ids)

                    # Fetch the additional nodes (could be Section nodes or Chunk nodes)
                    # Ensure 'text' and 'metadata' are handled, using defaults if not present (e.g. for Section nodes if they don't have 'text')
                    add_result = session.run("""
                        MATCH (n) WHERE n.id IN $node_ids OR n.chunk_id IN $node_ids OR n.section_id IN $node_ids
                        RETURN n.id as id, 
                               COALESCE(n.text, n.name, n.content, 'No text content') as text, 
                               // Attempt to get metadata, if not, check for properties typical of Section or Chunk
                               CASE 
                                 WHEN n.metadata IS NOT NULL THEN n.metadata
                                 ELSE properties(n) // Fallback to all properties as metadata
                               END as metadata,
                               labels(n) as labels 
                    """, node_ids=add_ids)

                    # Create document nodes for additional content
                    for record in add_result:
                        node_id = record["id"]
                        text = record["text"]
                        metadata = record["metadata"] if isinstance(record["metadata"], dict) else {}
                        
                        # Add label info to metadata for clarity
                        if record["labels"]:
                            metadata["node_type"] = record["labels"][0] # Assuming primary label

                        # Create a TextNode (generic enough for Chunks or Sections if they have text)
                        from llama_index.core.schema import TextNode
                        doc_node = TextNode(
                            text=text,
                            id_=node_id,
                            metadata=metadata
                        )

                        # Add to augmented nodes with zero score initially
                        augmented_nodes[node_id] = {
                            "node": doc_node,
                            "score": 0,
                            "source": "hierarchical"
                        }

                # Second pass: apply score adjustments
                for record in processed_records:
                    chunk_id = record["chunk_id"]
                    if chunk_id not in augmented_nodes:
                        continue

                    base_score = augmented_nodes[chunk_id]["score"]
                    parent_section_nodes = record["parent_section_nodes"]
                    sibling_chunk_nodes = record["sibling_chunk_nodes"]

                    # Boost parent section nodes
                    for parent_sec_node in parent_section_nodes:
                        parent_id = parent_sec_node["id"] # This is section_id
                        if parent_id and parent_id in augmented_nodes:
                            # Set parent score to a fraction of this node's score
                            augmented_nodes[parent_id]["score"] = max(
                                augmented_nodes[parent_id]["score"],
                                base_score * self.parent_boost
                            )

                    # Boost sibling chunk nodes based on distance
                    for sibling_chunk_node in sibling_chunk_nodes:
                        sibling_id = sibling_chunk_node["id"] # This is chunk_id
                        if sibling_id and sibling_id in augmented_nodes:
                            distance = sibling_chunk_node["distance"]
                            if distance is None: # Should not happen with current query if sibling_chunk_node exists
                                distance = self.context_window + 1 
                            # Apply distance-based decay
                            distance_factor = 1 - (distance / (self.context_window + 1))
                            sibling_boost_score = self.sibling_boost * distance_factor

                            # Set sibling score with boost
                            augmented_nodes[sibling_id]["score"] = max(
                                augmented_nodes[sibling_id]["score"],
                                base_score * sibling_boost_score
                            )
                return augmented_nodes
        except Exception as e:
            logger.error(f"Error in hierarchical context retrieval: {str(e)}")
            return initial_nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Override the base retrieve method to ensure we always return a List[NodeWithScore].
        
        Args:
            query_bundle: Query bundle
            
        Returns:
            List of NodeWithScore objects
        """
        output = self._retrieve(query_bundle)
        if hasattr(output, 'nodes'):
            return output.nodes
        elif isinstance(output, list):
            return output
        else:
            logger.error(f"Unexpected output type from _retrieve: {type(output)}")
            return []

class RAGSystem:
    """
    Multilingual Retrieval-Augmented Generation (RAG) system using LlamaIndex.
    Supports multiple languages with specialized embedding models for each.
    Uses lazy-loading of models to optimize memory usage.
    """

    DEFAULT_EMBED_DIM = 384
    DEFAULT_MODEL_KEY = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Updated to a good multilingual model

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        qdrant_url: str = None,
        qdrant_collection_prefix: str = "regulaite_docs",
        openai_api_key: str = None,
        default_lang: str = "fr",  # Updated default language
        chunk_size: int = 1000,
        preload_languages: List[str] = ["fr"],  # Updated to preload French by default
        hybrid_search: bool = True,
        vector_weight: float = 0.5,  # Adjusted to give less weight to vector search
        keyword_weight: float = 0.5,  # Increased keyword weight for better term matching
        hierarchical_retrieval: bool = True,
        parent_boost: float = 0.2,
        sibling_boost: float = 0.1,
        use_reranker: bool = True,
        reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # Updated multilingual reranker
        reranker_top_n: int = 10,  # Increased to get more candidates for reranking
        reranker_threshold: float = -5.0,  # New parameter to filter out low-scoring results
        query_expansion: bool = True  # New parameter to enable query expansion
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_driver = None
        self._connect_to_neo4j()

        self.qdrant_url = qdrant_url
        self.qdrant_collection_prefix = qdrant_collection_prefix
        self.qdrant_client = None
        if qdrant_url:
            try:
                self.qdrant_client = QdrantClient(url=qdrant_url)
                logger.info(f"Connected to Qdrant at {qdrant_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant at {qdrant_url}: {e}")
                self.qdrant_client = None # Ensure it's None if connection failed
        else:
            logger.warning("Qdrant URL not provided, vector store operations will be limited.")

        self.openai_api_key = openai_api_key
        self.default_lang = default_lang
        self.chunk_size = chunk_size
        self.language_settings = {}
        self.language_detector = LanguageDetector()
        
        self.hybrid_search_enabled = hybrid_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.hierarchical_retrieval_enabled = hierarchical_retrieval
        self.parent_boost = parent_boost
        self.sibling_boost = sibling_boost

        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model_name
        self.reranker_top_n = reranker_top_n
        self.reranker_threshold = reranker_threshold  # New parameter
        self.query_expansion = query_expansion  # New parameter
        self.reranker = None
        
        # Try initializing reranker immediately
        if self.use_reranker:
            try:
                self.reranker = SentenceTransformerRerank(
                    model_name=self.reranker_model_name,
                    top_n=self.reranker_top_n
                )
                logger.info(f"Initialized reranker with model {self.reranker_model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {str(e)}")

        if preload_languages:
            for lang_code in preload_languages:
                self._initialize_language(lang_code)
        self._initialize_language(self.default_lang)
        logger.info(f"RAG System initialized with default language: {self.default_lang}")

    def _connect_to_neo4j(self):
        """Establish connection to Neo4j with retry logic"""
        max_retries = 3
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                logger.info(f"Attempting to connect to Neo4j at {self.neo4j_uri} (attempt {retry_count + 1}/{max_retries})")

                self.neo4j_driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                    max_connection_lifetime=3600
                )

                # Test connection
                with self.neo4j_driver.session() as session:
                    result = session.run("RETURN 'Connected to Neo4j' AS message")
                    for record in result:
                        logger.info(record["message"])

                logger.info(f"Multilingual RAG system connected to Neo4j at {self.neo4j_uri}")
                return  # Successfully connected

            except Exception as e:
                last_error = e
                retry_count += 1

                # Log the specific error
                if "AuthenticationRateLimit" in str(e):
                    logger.error("Neo4j authentication rate limit reached. Waiting before retry...")
                    # Wait longer when hitting rate limit
                    time.sleep(30)  # 30 second delay
                else:
                    logger.error(f"Failed to connect to Neo4j (attempt {retry_count}/{max_retries}): {str(e)}")
                    time.sleep(5)  # 5 second delay for other errors

        # If we get here, all retries failed
        logger.error(f"Failed to connect to Neo4j after {max_retries} attempts. Last error: {str(last_error)}")
        raise last_error

    def _get_vector_dim_for_model(self, model_name: str) -> int:
        return self.DEFAULT_EMBED_DIM

    def _initialize_language(self, lang_code: str) -> bool:
        if lang_code in self.language_settings:
            return True

        try:
            logger.info(f"Initializing resources for language: {lang_code}")

            embed_model = FastEmbedEmbedding(model_name=self.DEFAULT_MODEL_KEY)
            logger.info(f"Successfully initialized FastEmbedEmbedding model {self.DEFAULT_MODEL_KEY} for language: {lang_code}")
            
            vector_dim = self._get_vector_dim_for_model(self.DEFAULT_MODEL_KEY)
            logger.info(f"Using vector dimension {vector_dim} for FastEmbed")
            
            collection_name = f"{self.qdrant_collection_prefix}_{lang_code}"
            
            if not self.qdrant_client:
                logger.error("Qdrant client not available. Cannot initialize language.")
                return False

            try:
                collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
                logger.info(f"Collection {collection_name} exists.")
                
                existing_vector_config = collection_info.config.params.vectors
                current_vector_name = "text-dense" # The name we intend to use

                if isinstance(existing_vector_config, dict): # Named vectors
                    if current_vector_name not in existing_vector_config or \
                       existing_vector_config[current_vector_name].size != vector_dim:
                        logger.warning(f"Collection {collection_name} has incompatible vector config. Recreating.")
                        self.qdrant_client.delete_collection(collection_name=collection_name)
                        raise ValueError("Recreating collection") # Force recreation
                elif hasattr(existing_vector_config, 'size'): # Single unnamed vector
                    if existing_vector_config.size != vector_dim:
                        logger.warning(f"Collection {collection_name} has incompatible vector dimension. Recreating.")
                        self.qdrant_client.delete_collection(collection_name=collection_name)
                        raise ValueError("Recreating collection") # Force recreation
                else: # Unknown config, recreate
                    logger.warning(f"Unknown vector config for {collection_name}. Recreating.")
                    self.qdrant_client.delete_collection(collection_name=collection_name)
                    raise ValueError("Recreating collection")


            except Exception as e: # Catches Qdrant client errors (e.g. collection not found) or our ValueError
                if "not found" in str(e).lower() or "recreating collection" in str(e).lower() :
                    logger.info(f"Creating new collection {collection_name} or recreating due to: {str(e)}")
                    self.qdrant_client.recreate_collection( # Use recreate_collection for simplicity
                        collection_name=collection_name,
                        vectors_config={
                            "text-dense": rest.VectorParams(size=vector_dim, distance=rest.Distance.COSINE)
                        }
                    )
                    logger.info(f"Collection {collection_name} created/recreated with vector 'text-dense' dim {vector_dim}")
                else:
                    logger.error(f"Error checking/creating Qdrant collection {collection_name}: {e}")
                    return False
            
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                vector_name="text-dense" 
            )
            logger.info(f"Initialized QdrantVectorStore for {collection_name} with vector_name='text-dense'")

            # Configure global LlamaIndex settings for this operation context if needed
            # Settings.embed_model = embed_model # This might be better done locally if possible

            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model # Pass embed_model for from_vector_store
            )
            logger.info(f"Initialized VectorStoreIndex for {collection_name}")

            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=10, 
                embed_model=embed_model, 
                vector_store_query_mode=VectorStoreQueryMode.DEFAULT
            )
            logger.info(f"Base VectorIndexRetriever initialized for {lang_code} with mode DEFAULT")

            # Store the main components
            self.language_settings[lang_code] = {
                "embed_model_container": {"model": embed_model}, # Wrapped the embed_model
                "vector_store": vector_store,
                "index": index,
                "vector_retriever": vector_retriever, # This is the base dense retriever
                "retriever": vector_retriever # Start with this, potentially wrap it later
            }

            # BM25 Retriever Setup (if enabled globally)
            if self.hybrid_search_enabled and BM25Retriever is not None:
                try:
                    # Fetch all nodes from Qdrant for BM25. This can be memory intensive.
                    # Consider a more sophisticated way if collections are very large.
                    logger.info(f"Attempting to initialize BM25Retriever for {lang_code}. Fetching nodes...")
                    all_nodes_for_bm25 = []
                    # Scroll through all points in the Qdrant collection
                    offset = None
                    while True:
                        points_page, next_offset = self.qdrant_client.scroll(
                            collection_name=collection_name,
                            limit=250, # Adjust batch size as needed
                            offset=offset,
                            with_payload=True,
                            with_vectors=False # No need for vectors for BM25
                        )
                        for point in points_page:
                            node_text = point.payload.get("text", "") if point.payload else ""
                            node_metadata = point.payload if point.payload else {}
                            # Ensure 'chunk_id' and 'doc_id' are in metadata for BM25 nodes if possible
                            if 'chunk_id' not in node_metadata: node_metadata['chunk_id'] = point.id
                            if 'doc_id' not in node_metadata and 'doc_id' in point.payload:
                                node_metadata['doc_id'] = point.payload['doc_id']

                            all_nodes_for_bm25.append(TextNode(text=node_text, id_=point.id, metadata=node_metadata))
                        
                        if next_offset is None:
                            break
                        offset = next_offset
                    
                    logger.info(f"Fetched {len(all_nodes_for_bm25)} nodes for BM25 for {lang_code}.")
                    if all_nodes_for_bm25:
                        # Determine language for BM25 retriever
                        # Use lang_code, which is the language of the documents being indexed/retrieved for
                        bm25_language = lang_code
                        logger.info(f"Using language '{bm25_language}' for BM25Retriever.")

                        bm25_retriever = BM25Retriever.from_defaults(
                            nodes=all_nodes_for_bm25,
                            similarity_top_k=10,
                            language=bm25_language, # Added language
                            stemmer=Stemmer.Stemmer(bm25_language) if bm25_language else None # Changed from Stemmer(bm25_language)
                        )
                        self.language_settings[lang_code]["bm25_retriever"] = bm25_retriever
                        logger.info(f"BM25Retriever initialized for {lang_code} with language '{bm25_language}'.")
                        
                        # If BM25 is setup, the main retriever becomes HybridRetriever
                        self.language_settings[lang_code]["retriever"] = HybridRetriever(
                            vector_retriever=vector_retriever,
                            keyword_retriever=bm25_retriever,
                            vector_weight=self.vector_weight,
                            keyword_weight=self.keyword_weight,
                            top_k=10 
                        )
                        logger.info(f"HybridRetriever (Vector+BM25) set as main retriever for {lang_code}.")
                    else:
                        logger.warning(f"No nodes found to initialize BM25Retriever for {lang_code}. Hybrid search (BM25 part) will be disabled for this language.")
                        self.language_settings[lang_code]["bm25_retriever"] = None

                except Exception as e:
                    logger.error(f"Failed to initialize BM25Retriever for {lang_code}: {e}", exc_info=True)
                    self.language_settings[lang_code]["bm25_retriever"] = None
            
            # Hierarchical Retriever Setup (if enabled globally and Neo4j is available)
            # This wraps the current main retriever (which could be VectorIndexRetriever or HybridRetriever)
            if self.hierarchical_retrieval_enabled and self.neo4j_driver:
                current_main_retriever = self.language_settings[lang_code]["retriever"]
                self.language_settings[lang_code]["retriever"] = HierarchicalRetriever(
                    base_retriever=current_main_retriever,
                    neo4j_driver=self.neo4j_driver,
                    top_k=10, 
                    parent_boost=self.parent_boost,
                    sibling_boost=self.sibling_boost
                )
                logger.info(f"HierarchicalRetriever wrapped main retriever for {lang_code}.")
            
            logger.info(f"Language settings for {lang_code} fully populated. Embedding model container is: {self.language_settings[lang_code].get('embed_model_container')}")
            return True

        except Exception as e:
            logger.error(f"Fatal error initializing language {lang_code}: {e}", exc_info=True)
            return False

    def _expand_query(self, query: str, lang_code: str = "fr") -> str:
        """
        Expand query with synonyms or related terms to improve retrieval.
        
        Args:
            query: Original query string
            lang_code: Language code for query expansion
            
        Returns:
            Expanded query string
        """
        # Simple query expansion logic - can be enhanced with more sophisticated methods
        if not self.query_expansion:
            return query
            
        try:
            # For French queries about risk levels
            if lang_code == "fr" and any(term in query.lower() for term in ["risque", "critique", "niveau", "score", "note"]):
                expanded_terms = []
                
                # Add specific risk-related terms
                if "critique" in query.lower():
                    expanded_terms.extend(["critique", "sévère", "grave", "majeur", "important"])
                
                if "risque" in query.lower():
                    expanded_terms.extend(["danger", "menace", "vulnérabilité"])
                    
                if any(level in query.lower() for level in ["niveau", "score", "note"]):
                    expanded_terms.extend(["classification", "évaluation", "catégorie"])
                
                # Create expanded query with original plus new terms
                if expanded_terms:
                    expanded_query = f"{query} {' '.join(expanded_terms)}"
                    logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
                    return expanded_query
            
            return query
        except Exception as e:
            logger.error(f"Error in query expansion: {str(e)}")
            return query  # Fall back to original query

    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """
        Check if a document's metadata matches filter criteria.
        
        Args:
            metadata: Document metadata
            filter_criteria: Filter criteria dict
            
        Returns:
            Whether the document matches the filter
        """
        if not filter_criteria:
            return True
            
        for field, value in filter_criteria.items():
            # Handle special filter operations
            if field == "any_of":
                # any_of: At least one of the nested conditions must match
                if not isinstance(value, list) or not value:
                    continue
                    
                any_matched = False
                for sub_filter in value:
                    if self._matches_filter(metadata, sub_filter):
                        any_matched = True
                        break
                        
                if not any_matched:
                    return False
                    
            elif field == "all_of":
                # all_of: All nested conditions must match
                if not isinstance(value, list) or not value:
                    continue
                    
                for sub_filter in value:
                    if not self._matches_filter(metadata, sub_filter):
                        return False
                
            elif field == "not":
                # not: Inverts the matching of nested conditions
                if not isinstance(value, dict) or not value:
                    continue
                    
                if self._matches_filter(metadata, value):
                    return False
                    
            # Handle field-specific filtering
            elif field in metadata:
                meta_value = metadata[field]
                
                # Handle different filter value types
                if isinstance(value, dict):
                    # Complex comparison operators (eq, gt, lt, etc.)
                    if not self._apply_comparison_filter(meta_value, value):
                        return False
                elif isinstance(value, list):
                    # List of possible values (IN operator)
                    if meta_value not in value:
                        return False
                else:
                    # Direct equality comparison
                    if meta_value != value:
                        return False
            else:
                # Field not in metadata
                return False
                
        return True

    def _get_context_based_filter(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Generate query-specific filter criteria based on the query content.
        
        Args:
            query: The query string
            
        Returns:
            Filter criteria dictionary or None
        """
        lower_query = query.lower()
        
        # For risk-related questions, prioritize security documentation
        if any(term in lower_query for term in ["risque", "critique", "danger", "menace", "vulnerability", "sécurité"]):
            return {
                "any_of": [
                    {"category": "security"},
                    {"category": "risk"},
                    {"category": "compliance"},
                    {"doc_name": "PSSI"} # Prioritize security policy documents
                ]
            }
            
        # For compliance questions
        if any(term in lower_query for term in ["conforme", "conformité", "rgpd", "gdpr", "compliance"]):
            return {
                "any_of": [
                    {"category": "compliance"},
                    {"category": "legal"},
                    {"category": "regulatory"}
                ]
            }
            
        # No specific filter for other query types
        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cross_lingual: bool = True, # If true, attempts to retrieve from default_lang if query lang is not initialized
        use_hybrid: Optional[bool] = None, # Overrides system hybrid setting
        filter_criteria: Optional[Dict[str, Any]] = None,
        use_neo4j: bool = True,  # Parameter to control Neo4j usage for hierarchical parts
        use_hierarchical: Optional[bool] = None, # Overrides system hierarchical setting
        use_query_expansion: Optional[bool] = None, # Overrides system query expansion setting
        auto_filter: bool = True  # Whether to apply automatic context-based filtering
    ) -> List[Dict[str, Any]]:
        """
        Retrieves documents based on a query, with multilingual support.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to retrieve
            use_cross_lingual: Whether to attempt retrieval from default language if query language not initialized
            use_hybrid: Whether to use hybrid search (overrides default setting)
            filter_criteria: Dictionary of metadata filters to apply
            use_neo4j: Whether to utilize Neo4j for hierarchical retrieval
            use_hierarchical: Whether to use hierarchical retrieval (overrides default setting)
            use_query_expansion: Whether to use query expansion (overrides default setting)
            auto_filter: Whether to apply automatic context-based filtering
            
        Returns:
            List of dictionaries containing retrieved information
        """
        if not query or not query.strip():
            logger.warning("Empty query received, cannot perform retrieval")
            return []
            
        query_lang = self.detect_language(query)
        logger.info(f"Detected query language: {query_lang}")
        
        # Apply context-based filtering if enabled
        context_filter = None
        if auto_filter and not filter_criteria:
            context_filter = self._get_context_based_filter(query)
            if context_filter:
                logger.info(f"Applied context-based filter: {context_filter}")
                filter_criteria = context_filter

        # Determine which language to use for retrieval
        target_lang = self.ensure_language_initialized(query_lang)
        if not target_lang:
            # If primary language not available, try default language for cross-lingual search
            if use_cross_lingual and query_lang != self.default_lang:
                logger.info(f"Query language {query_lang} not available, attempting cross-lingual search with {self.default_lang}")
                target_lang = self.ensure_language_initialized(self.default_lang)
                if not target_lang:
                    logger.error(f"Failed to initialize default language {self.default_lang}")
                    return []
            else:
                logger.error(f"Language {query_lang} not available and cross-lingual search not enabled")
                return []
                
        logger.info(f"Using language {target_lang} for retrieval")
        
        # Apply query expansion if enabled
        should_expand = self.query_expansion if use_query_expansion is None else use_query_expansion
        expanded_query = self._expand_query(query, target_lang) if should_expand else query
        
        # Get the retriever for this language
        lang_settings = self.language_settings.get(target_lang)
        if not lang_settings:
            logger.error(f"Language {target_lang} settings not found")
            return []
            
        # Create query bundle with expanded query
        query_bundle = QueryBundle(query_str=expanded_query)
        
        # Get the appropriate retriever based on settings and overrides
        retriever = self._get_retriever_for_query(
            lang_settings=lang_settings,
            use_hybrid=use_hybrid,
            use_neo4j=use_neo4j,
            use_hierarchical=use_hierarchical,
        )
        
        if not retriever:
            logger.error("No suitable retriever found")
            return []
            
        # Retrieve nodes
        try:
            logger.info(f"Retrieving with {type(retriever).__name__}")
            output = retriever._retrieve(query_bundle)
            
            # Handle different return types from the retriever
            retrieved_nodes = []
            if hasattr(output, 'nodes'):
                # It's a BaseRetrieverOutput object
                retrieved_nodes = output.nodes
            elif isinstance(output, list):
                # It's a list of NodeWithScore
                if not output:
                    logger.info("Retriever returned empty list")
                    return []
                if all(isinstance(n, NodeWithScore) for n in output):
                    retrieved_nodes = output
                else:
                    logger.error(f"Retriever returned list with non-NodeWithScore elements")
                    return []
            else:
                logger.error(f"Retriever returned unexpected type: {type(output)}")
                return []
                
            logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
                
            # Apply reranker if enabled
            if self.reranker and self.use_reranker and retrieved_nodes:
                logger.info(f"Applying reranking with {self.reranker_model_name}")
                try:
                    # Get more candidates for reranking if possible
                    reranked_nodes = self.reranker.postprocess_nodes(
                        retrieved_nodes,
                        query_bundle=QueryBundle(query_str=query)  # Use original query for reranking
                    )
                    logger.info(f"Reranked nodes: {len(reranked_nodes)}")
                    retrieved_nodes = reranked_nodes
                    
                    # Apply threshold to filter out low-scoring results
                    if self.reranker_threshold is not None:
                        original_count = len(retrieved_nodes)
                        retrieved_nodes = [n for n in retrieved_nodes if n.score >= self.reranker_threshold]
                        logger.info(f"Applied score threshold {self.reranker_threshold}: {original_count} -> {len(retrieved_nodes)} nodes")
                except Exception as e:
                    logger.error(f"Error during reranking: {str(e)}", exc_info=True)
            
            # Process results into final format
            final_results = []
            for node_with_score in retrieved_nodes:
                node = node_with_score.node
                score = node_with_score.score
                
                # Create result dictionary with node data
                result = {
                    "text": node.get_content(),
                    "metadata": node.metadata or {},
                    "score": score,
                    "doc_id": node.metadata.get("doc_id", node.node_id),
                    "chunk_id": node.node_id
                }
                
                # Add reranker information if used
                if self.reranker and self.use_reranker:
                    result["metadata"]["reranked_score"] = score
                    result["metadata"]["reranker_model"] = self.reranker_model_name
                
                # Apply filter criteria if provided
                if filter_criteria:
                    if not self._matches_filter(result["metadata"], filter_criteria):
                        continue
                
                final_results.append(result)
            
            # Limit to top_k results
            final_results = final_results[:top_k]
            logger.info(f"Returning {len(final_results)} final results")
            
            return final_results
                
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
            return []

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text.

        Args:
            text: Text to analyze

        Returns:
            Language code
        """
        lang_info = self.language_detector.detect_language(text)
        lang_code = lang_info["language_code"]

        # If language not supported, use multilingual or default
        if lang_code not in self.language_settings:
            if "multilingual" in self.language_settings:
                return "multilingual"
            return self.default_lang

        return lang_code

    def ensure_language_initialized(self, lang_code: str) -> str:
        """
        Ensure a language is initialized, or fall back to an alternative.

        Args:
            lang_code: Desired language code

        Returns:
            The initialized language code (might be different if fallback was used)
        """
        # Try to initialize the requested language
        if lang_code not in self.language_settings:
            success = self._initialize_language(lang_code)
            if success:
                return lang_code

        # If we get here, either initialization failed or was not needed
        # Check if the language is already initialized
        if lang_code in self.language_settings:
            return lang_code

        # Try multilingual fallback
        if "multilingual" in self.language_settings:
            if "multilingual" not in self.language_settings:
                success = self._initialize_language("multilingual")
                if success:
                    return "multilingual"
            elif "multilingual" in self.language_settings:
                return "multilingual"

        # Try default language as a final fallback
        if self.default_lang not in self.language_settings:
            success = self._initialize_language(self.default_lang)
            if success:
                return self.default_lang
        elif self.default_lang in self.language_settings:
            return self.default_lang

        # If we get here, nothing worked
        logger.error(f"Failed to initialize any language model (requested: {lang_code})")
        raise ValueError(f"Failed to initialize any language model (requested: {lang_code})")

    def _apply_comparison_filter(self, value: Any, filter_dict: Dict[str, Any]) -> bool:
        """
        Apply a comparison filter based on the operator specified in the filter_dict.

        Args:
            value: The value to compare
            filter_dict: Dictionary containing the filter criteria

        Returns:
            True if the value passes the filter, False otherwise
        """
        operator = filter_dict['operator']
        comparison_value = filter_dict['value']

        if operator == '>':
            return value > comparison_value
        elif operator == '<':
            return value < comparison_value
        elif operator == '>=':
            return value >= comparison_value
        elif operator == '<=':
            return value <= comparison_value
        elif operator == '==':
            return value == comparison_value
        elif operator == '!=':
            return value != comparison_value
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Multilingual RAG system Neo4j connection closed")

    def _initialize_fallback_placeholder(self):
        """
        Initialize a fallback placeholder language resource when all other
        initialization attempts fail. This allows the system to continue
        operating with empty retrieval results rather than crashing.
        """
        logger.warning("Initializing fallback placeholder for language resources")
        
        # Create empty placeholder resources
        class EmptyRetriever(BaseRetriever):
            def _retrieve(self, query_bundle):
                logger.warning(f"Using empty retriever for query: {query_bundle.query_str}")
                return BaseRetrieverOutput(nodes=[])
        
        # Store fallback resources
        self.language_settings["fallback"] = {
            "embedding_model": None,
            "vector_retriever": EmptyRetriever(),
        }
        
        logger.info("Fallback placeholder initialized - system will return empty results")

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from all vector stores.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            logger.info(f"Deleting document {doc_id} from vector stores")
            
            # Check all language collections
            deleted = False
            deletion_errors = []
            
            # Get list of all collections
            try:
                collections_info = self.qdrant_client.get_collections()
                collections = collections_info.collections
                collection_names = [c.name for c in collections if c.name.startswith(self.qdrant_collection_prefix)]
            except Exception as e:
                logger.error(f"Error getting collections from Qdrant: {str(e)}")
                # Try to reconnect to Qdrant
                try:
                    logger.info("Attempting to reconnect to Qdrant...")
                    self.qdrant_client = QdrantClient(url=self.qdrant_url)
                    collections_info = self.qdrant_client.get_collections()
                    collections = collections_info.collections
                    collection_names = [c.name for c in collections if c.name.startswith(self.qdrant_collection_prefix)]
                except Exception as reconnect_error:
                    logger.error(f"Failed to reconnect to Qdrant: {str(reconnect_error)}")
                    return False
            
            if not collection_names:
                logger.warning(f"No collections found with prefix {self.qdrant_collection_prefix}")
                # Not really an error - document might not be in any collection yet
                return True
                
            logger.info(f"Found collections: {collection_names}")
            
            # Delete from each collection
            for collection_name in collection_names:
                # Use retry logic for each collection
                max_retries = 3
                retry_delay = 1  # seconds
                
                for retry in range(max_retries):
                    try:
                        # Check if the collection has points with payload fields
                        try:
                            # Get collection info to check if it has points and the payload schema
                            collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
                            if collection_info.points_count == 0:
                                logger.info(f"Collection {collection_name} is empty, skipping")
                                break  # Skip to next collection
                        except Exception as e:
                            logger.warning(f"Failed to get collection info for {collection_name}: {str(e)}")
                            # Continue anyway - we'll try to find points

                        # Determine correct payload field for filtering
                        # This is crucial - sometimes it might be stored in metadata.doc_id instead of doc_id
                        possible_payload_fields = ["doc_id", "metadata.doc_id", "document_id", "document"]
                        payload_field = "doc_id"  # Default field
                        
                        # First, check if we can find any points with any of the potential doc_id fields
                        found_points = False
                        
                        for field in possible_payload_fields:
                            try:
                                # Try the current field
                                logger.info(f"Checking if collection {collection_name} has points with {field} = {doc_id}")
                                
                                # Create the appropriate filter based on field structure
                                if "." in field:
                                    # Nested field like metadata.doc_id
                                    parent, child = field.split(".", 1)
                                    filter_condition = rest.Filter(
                                        must=[
                                            rest.FieldCondition(
                                                key=parent,
                                                match=rest.MatchAny(
                                                    any=[{child: doc_id}]
                                                )
                                            )
                                        ]
                                    )
                                else:
                                    # Top-level field
                                    filter_condition = rest.Filter(
                                        must=[
                                            rest.FieldCondition(
                                                key=field,
                                                match=rest.MatchValue(value=doc_id)
                                            )
                                        ]
                                    )
                                
                                # Try to find points with this field
                                search_result = self.qdrant_client.scroll(
                                    collection_name=collection_name,
                                    scroll_filter=filter_condition,
                                    limit=1  # Just need to know if any exist
                                )
                                
                                points, _ = search_result
                                
                                if points:
                                    logger.info(f"Found points with {field} = {doc_id}")
                                    payload_field = field
                                    found_points = True
                                    break
                            except Exception as field_error:
                                logger.warning(f"Error checking field {field}: {str(field_error)}")
                        
                        if not found_points:
                            logger.info(f"No points found for document {doc_id} in collection {collection_name}")
                            break  # No need for further retries
                        
                        # Now get all points to delete using the correct field
                        logger.info(f"Using payload field '{payload_field}' for deletion filter in {collection_name}")
                        
                        # Create the appropriate filter based on field structure
                        if "." in payload_field:
                            # Nested field like metadata.doc_id
                            parent, child = payload_field.split(".", 1)
                            filter_condition = rest.Filter(
                                must=[
                                    rest.FieldCondition(
                                        key=parent,
                                        match=rest.MatchAny(
                                            any=[{child: doc_id}]
                                        )
                                    )
                                ]
                            )
                        else:
                            # Top-level field
                            filter_condition = rest.Filter(
                                must=[
                                    rest.FieldCondition(
                                        key=payload_field,
                                        match=rest.MatchValue(value=doc_id)
                                    )
                                ]
                            )
                        
                        search_result = self.qdrant_client.scroll(
                            collection_name=collection_name,
                            scroll_filter=filter_condition,
                            limit=100  # Get up to 100 matching points
                        )
                        
                        points, next_page_offset = search_result
                        
                        if not points:
                            logger.info(f"No points found for document {doc_id} in collection {collection_name}")
                            break  # No need for further retries
                        
                        logger.info(f"Found {len(points)} points to delete in collection {collection_name}")
                        
                        # We have two options: 
                        # 1. Delete by IDs if we have a manageable number of points
                        # 2. Delete by filter for larger sets
                        
                        if len(points) <= 100:
                            # Extract point IDs and delete them directly
                            point_ids = [point.id for point in points]
                            
                            # Delete by IDs
                            self.qdrant_client.delete(
                                collection_name=collection_name,
                                points_selector=rest.PointIdsList(
                                    points=point_ids
                                )
                            )
                            logger.info(f"Deleted {len(point_ids)} points by ID from collection {collection_name}")
                        else:
                            # For larger sets, use filter-based deletion
                            # Use the same filter that we used to find the points
                            self.qdrant_client.delete(
                                collection_name=collection_name,
                                points_selector=rest.FilterSelector(
                                    filter=filter_condition
                                )
                            )
                            logger.info(f"Deleted points by filter from collection {collection_name}")
                        
                        deleted = True
                        break  # Success, no need for further retries
                        
                    except Exception as e:
                        error_msg = f"Error deleting from collection {collection_name} (attempt {retry+1}/{max_retries}): {str(e)}"
                        logger.error(error_msg)
                        
                        if retry < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            deletion_errors.append(error_msg)
            
            if deletion_errors and not deleted:
                logger.error(f"All deletion attempts failed: {deletion_errors}")
                return False
                
            # If we reach here, at least one collection was processed successfully or had no points to delete
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document from vector stores: {str(e)}")
            # Check for specific error types and provide more detailed information
            if "connection" in str(e).lower():
                logger.error("Connection error occurred. Qdrant service may be unavailable.")
            elif "timeout" in str(e).lower():
                logger.error("Timeout error occurred. The operation took too long to complete.")
            return False

    def index_document(self, doc_id: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index a document in the vector store.
        
        Args:
            doc_id: Document ID to index
            force_reindex: Whether to force reindexing if already indexed
            
        Returns:
            Dictionary with indexing results
        """
        try:
            logger.info(f"Indexing document {doc_id} in vector stores")
            
            # 1. Get document chunks from Neo4j
            if not self.neo4j_driver:
                self._connect_to_neo4j()
                
            if not self.neo4j_driver:
                logger.error("Failed to connect to Neo4j, cannot index document")
                return {
                    "doc_id": doc_id, "vector_count": 0, "status": "error",
                    "message": f"Failed to connect to Neo4j, cannot index document"
                }
                
            with self.neo4j_driver.session() as session:
                # Check if document exists
                doc_check = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id}) 
                    RETURN d.name as name, 
                           COALESCE(d.title, d.name) as title, 
                           COALESCE(d.language, 'en') as language,
                           COALESCE(d.category, '') as category, 
                           COALESCE(d.author, '') as author, 
                           COALESCE(d.file_type, '') as file_type,
                           COALESCE(d.is_indexed, false) as is_indexed
                    """,
                    doc_id=doc_id
                )
                
                doc_record = doc_check.single()
                if not doc_record:
                    logger.error(f"Document {doc_id} not found in Neo4j")
                    return {
                        "doc_id": doc_id, "vector_count": 0, "status": "error",
                        "message": f"Document {doc_id} not found in Neo4j"
                    }
                
                # Check if document is already indexed and we're not forcing reindex
                if not force_reindex and doc_record.get("is_indexed") == True:
                    logger.info(f"Document {doc_id} is already indexed. Use force_reindex=True to reindex.")
                    return {
                        "doc_id": doc_id,
                        "vector_count": 0, # Or fetch existing count if important
                        "message": "Document already indexed",
                        "status": "skipped"
                    }
                    
                # Handle potentially missing properties with safe gets
                doc_name = doc_record.get("title") or doc_record.get("name", f"Document-{doc_id}")
                # doc_language = doc_record.get("language") or self.default_lang # Original line

                # --- MODIFICATION FOR FRENCH-ONLY ---
                detected_doc_language = doc_record.get("language")
                if detected_doc_language != 'fr':
                    logger.warning(f"Document {doc_id} (title: {doc_name}) detected by parser as '{detected_doc_language}'. Forcing to 'fr' for indexing as per French-only requirement.")
                    doc_language = 'fr'
                else:
                    doc_language = 'fr' # It's already French
                # --- END MODIFICATION ---

                doc_category = doc_record.get("category", "")
                doc_author = doc_record.get("author", "")
                doc_file_type = doc_record.get("file_type", "")
                
                # Get all chunks for the document
                chunk_query = """
                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                RETURN c.chunk_id as chunk_id, c.text as text, 
                       COALESCE(c.section, 'Default') as section, 
                       COALESCE(c.page_num, 0) as page_num, 
                       COALESCE(c.order_index, 0) as order_index
                ORDER BY COALESCE(c.order_index, 0)
                """
                
                chunks_result = session.run(chunk_query, doc_id=doc_id)
                chunks = list(chunks_result)
                
                if not chunks:
                    logger.warning(f"Document {doc_id} has no chunks to index")
                    return {
                        "doc_id": doc_id,
                        "vector_count": 0,
                        "message": "Document has no chunks to index",
                        "status": "skipped"
                    }
                    
                # Ensure the language is initialized
                actual_lang = self.ensure_language_initialized(doc_language)
                
                # --- ADDED SAFETY CHECK FOR FRENCH --- 
                if actual_lang != 'fr':
                    logger.error(f"CRITICAL: actual_lang in index_document is '{actual_lang}' instead of 'fr' for doc {doc_id} (title: {doc_name}). This indicates a problem in ensure_language_initialized or default_lang setup.")
                    return {
                        "doc_id": doc_id, "vector_count": 0, "status": "error",
                        "message": f"Language initialization for indexing yielded '{actual_lang}' instead of 'fr'."
                    }
                # --- END SAFETY CHECK --- 

                # Get the correct resources for this language
                if actual_lang not in self.language_settings: # This check might be redundant if ensure_language_initialized is robust
                    logger.error(f"Language {actual_lang} not available in language_settings for indexing doc {doc_id}")
                    return {
                        "doc_id": doc_id, "vector_count": 0, "status": "error",
                        "message": f"Language {actual_lang} not available in language_settings for indexing"
                    }
                    
                resources = self.language_settings[actual_lang]
                logger.info(f"Resources content for {actual_lang} in index_document (doc_id: {doc_id}): {str(resources)[:500]}...")
                
                embed_model = None # Initialize embed_model to None
                embed_model_container = resources.get("embed_model_container")
                logger.info(f"Retrieved embed_model_container for {actual_lang} (doc_id: {doc_id}): {embed_model_container}")

                if embed_model_container and isinstance(embed_model_container, dict):
                    embed_model = embed_model_container.get("model")
                    logger.info(f"Retrieved embed_model from container for {actual_lang} (doc_id: {doc_id}): {embed_model}")
                else:
                    logger.warning(f"embed_model_container was missing or not a dict for {actual_lang} (doc_id: {doc_id}).")

                if not embed_model:
                    logger.error(f"No embedding model retrieved for language {actual_lang} (doc_id: {doc_id}) [Container Check]")
                    return {
                        "doc_id": doc_id, "vector_count": 0, "status": "error",
                        "message": f"No embedding model retrieved for language {actual_lang} (doc_id: {doc_id}) [Container Check]"
                    }
                
                # 2. Convert chunks to nodes and embed them
                indexed_chunks = 0
                for chunk in chunks:
                    try:
                        chunk_id = chunk["chunk_id"]
                        text = chunk["text"] # Get the text for payload and embedding
                        
                        # If chunk_id is None or empty, generate a new one
                        if not chunk_id:
                            chunk_id = f"{doc_id}_chunk_{chunk['order_index']}"
                            logger.warning(f"Generated chunk_id for missing ID: {chunk_id}")
                        
                        # Generate a UUID for Qdrant point ID
                        point_id = str(uuid.uuid4())
                        
                        # Create metadata
                        metadata = {
                            "doc_id": doc_id,
                            "doc_name": doc_name,
                            "chunk_id": chunk_id,
                            "language": actual_lang,
                            "section": chunk["section"] or "Default",
                            "page_num": chunk["page_num"] or 0,
                            "order_index": chunk["order_index"] or 0,
                            "category": doc_category or "",
                            "author": doc_author or "",
                            "file_type": doc_file_type or "",
                            "text": text  # Store the text content in the payload for BM25
                        }
                        
                        # Create vector
                        if not text or len(text.strip()) == 0:
                            logger.warning(f"Skipping empty chunk: {chunk_id}")
                            continue
                            
                        # Get embedding from the model
                        try:
                            embedding = embed_model.get_text_embedding(text)
                            logger.info(f"Generated embedding of size {len(embedding)} for chunk {chunk_id}")
                        except Exception as embed_error:
                            logger.error(f"Error generating embedding for chunk {chunk_id}: {str(embed_error)}")
                            continue
                        
                        # Add to Qdrant
                        collection_name = f"{self.qdrant_collection_prefix}_{actual_lang}"
                        vector_name_to_use = "text-dense" # Standardize to use 'text-dense'
                        
                        # Check if collection exists, create if it doesn't
                        try:
                            collection_info = self.qdrant_client.get_collection(collection_name)
                            logger.info(f"Collection {collection_name} exists with {collection_info.points_count} points")
                            
                            # Verify existing collection uses the correct vector name, otherwise log a warning
                            if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                                if not (hasattr(collection_info.config.params, 'vectors') and \
                                        vector_name_to_use in collection_info.config.params.vectors):
                                    logger.warning(
                                        f"Collection {collection_name} exists but may not be configured for vector name '{vector_name_to_use}'. Expected config: {collection_info.config.params.vectors}"
                                    )
                            
                        except Exception as collection_error:
                            # Simplified error handling: if any error (likely collection not found), try to create.
                            logger.info(f"Attempting to create collection {collection_name} as it might not exist or other error occurred: {str(collection_error)}")
                            vector_size = len(embedding)
                            
                            try:
                                self.qdrant_client.create_collection(
                                    collection_name=collection_name,
                                    vectors_config={
                                        vector_name_to_use: rest.VectorParams(
                                            size=vector_size,
                                            distance=rest.Distance.COSINE
                                        )
                                    }
                                )
                                logger.info(f"Created new collection {collection_name} with vector name '{vector_name_to_use}' and size {vector_size}")
                            except Exception as create_exc:
                                logger.error(f"Failed to create collection {collection_name}: {create_exc}")
                                continue # Skip this chunk if collection creation fails
                        
                        # Upsert the point using UUID as the point_id
                        logger.info(f"Upserting point {point_id} for chunk {chunk_id} in collection {collection_name} with vector name {vector_name_to_use}")
                        
                        try:
                            # Convert embedding to list if it's a numpy array
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                                
                            # Prepare vector with the standardized name
                            vector_dict = {vector_name_to_use: embedding}
                            
                            result = self.qdrant_client.upsert(
                                collection_name=collection_name,
                                points=[
                                    rest.PointStruct(
                                        id=point_id,
                                        vector=vector_dict,  # Use named vector format with appropriate name
                                        payload=metadata
                                    )
                                ]
                            )
                            logger.info(f"Successfully upserted point in {collection_name}: {result}")
                            indexed_chunks += 1
                        except Exception as upsert_error:
                            logger.error(f"Error upserting point to Qdrant: {str(upsert_error)}")
                            # Continue with other chunks
                    except Exception as chunk_error:
                        logger.error(f"Error indexing chunk {chunk.get('chunk_id')}: {str(chunk_error)}")
                
                if indexed_chunks > 0:
                    logger.info(f"Successfully indexed {indexed_chunks} chunks from document {doc_id}")
                    
                    # After successful indexing, update the document's is_indexed status
                    try:
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            SET d.is_indexed = true,
                                d.indexed_at = datetime(),
                                d.language = COALESCE(d.language, $language)
                            """,
                            doc_id=doc_id,
                            language=actual_lang
                        )
                        logger.info(f"Updated document {doc_id} as indexed in Neo4j")
                    except Exception as update_error:
                        logger.error(f"Failed to update document indexing status: {str(update_error)}")
                    
                    return {
                        "doc_id": doc_id,
                        "vector_count": indexed_chunks,
                        "language": actual_lang,
                        "status": "success",
                        "message": f"Successfully indexed {indexed_chunks} chunks"
                    }
                else:
                    logger.error(f"Failed to index any chunks for document {doc_id}")
                    return {
                        "doc_id": doc_id,
                        "vector_count": 0,
                        "status": "error",
                        "message": "Failed to index any chunks"
                    }
                
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {str(e)}")
            return {
                "doc_id": doc_id,
                "vector_count": 0,
                "status": "error",
                "message": f"Error: {str(e)}"
            }

    def get_initialized_languages(self) -> List[str]:
        """
        Get a list of currently initialized languages in the system
        
        Returns:
            List of language codes that have been initialized
        """
        return list(self.language_settings.keys())

    def _get_retriever_for_query(
        self,
        lang_settings: Dict[str, Any],
        use_hybrid: Optional[bool] = None,
        use_neo4j: bool = True,
        use_hierarchical: Optional[bool] = None
    ) -> Optional[BaseRetriever]:
        """
        Constructs and returns the appropriate retriever based on settings.
        
        Args:
            lang_settings: Language-specific settings including base retrievers
            use_hybrid: Whether to use hybrid search (overrides system setting)
            use_neo4j: Whether to use Neo4j for hierarchical retrieval
            use_hierarchical: Whether to use hierarchical retrieval (overrides system setting)
            
        Returns:
            Configured retriever or None if no suitable retriever could be constructed
        """
        # Start with the base vector retriever
        vector_retriever = lang_settings.get("vector_retriever")
        if not vector_retriever:
            logger.error("Vector retriever not available in language settings")
            return None
            
        current_retriever = vector_retriever
        
        # Determine if hybrid search should be used
        final_use_hybrid = self.hybrid_search_enabled if use_hybrid is None else use_hybrid
        if final_use_hybrid and BM25Retriever is not None:
            keyword_retriever = lang_settings.get("bm25_retriever")
            if keyword_retriever:
                logger.info("Using hybrid retriever (vector + keyword)")
                current_retriever = HybridRetriever(
                    vector_retriever=vector_retriever,
                    keyword_retriever=keyword_retriever,
                    vector_weight=self.vector_weight,
                    keyword_weight=self.keyword_weight
                )
            else:
                logger.warning("BM25 retriever not available, falling back to vector-only search")
                
        # Determine if hierarchical retrieval should be used
        final_use_hierarchical = self.hierarchical_retrieval_enabled if use_hierarchical is None else use_hierarchical
        if final_use_hierarchical and use_neo4j and self.neo4j_driver:
            logger.info("Using hierarchical retriever")
            current_retriever = HierarchicalRetriever(
                base_retriever=current_retriever,
                neo4j_driver=self.neo4j_driver,
                parent_boost=self.parent_boost,
                sibling_boost=self.sibling_boost
            )
            
        return current_retriever
    