# plugins/regul_aite/backend/llamaIndex_rag/rag.py
import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict
import uuid

# Updated imports for the current LlamaIndex structure
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.core.retrievers import BaseRetriever

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from neo4j import GraphDatabase
import time
import numpy as np

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

        # Ensure weights sum to 1
        total_weight = vector_weight + keyword_weight
        if total_weight != 1.0:
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
        # Get base results
        base_results = self.base_retriever.retrieve(query_bundle)

        # Check if we have Neo4j connectivity before trying hierarchical retrieval
        if not self.neo4j_driver:
            logger.warning("No Neo4j driver available, skipping hierarchical retrieval")
            return base_results

        # Extract node IDs from base results
        initial_nodes = {}
        for node in base_results:
            node_id = node.node.node_id
            initial_nodes[node_id] = {
                "node": node.node,
                "score": node.score,
                "source": "direct"
            }

        # If no results or too few, just return what we have
        if len(initial_nodes) < 2:
            return base_results

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
                result = session.run("""
                    MATCH (c:Chunk) WHERE c.id IN $node_ids
                    // Find parents
                    OPTIONAL MATCH (c)-[:PART_OF]->(parent:Section)
                    // Find siblings (chunks in the same section)
                    OPTIONAL MATCH (parent)<-[:PART_OF]-(sibling:Chunk)
                    WHERE sibling.id <> c.id
                    // Find sequence information
                    WITH c, parent, sibling,
                         CASE WHEN sibling IS NOT NULL THEN abs(sibling.sequence_num - c.sequence_num) ELSE null END as distance
                    WHERE distance IS NULL OR distance <= $context_window
                    RETURN c.id as chunk_id,
                           collect(DISTINCT {id: parent.id, type: 'parent'}) as parents,
                           collect(DISTINCT {id: sibling.id, type: 'sibling', distance: distance}) as siblings
                """, node_ids=node_ids, context_window=self.context_window)

                # Process results and add to initial nodes
                augmented_nodes = initial_nodes.copy()

                # Track additional nodes to fetch
                additional_node_ids = set()

                # First pass: identify additional nodes to retrieve
                for record in result:
                    chunk_id = record["chunk_id"]
                    parents = record["parents"]
                    siblings = record["siblings"]

                    # Add parent IDs
                    for parent in parents:
                        if parent["id"] and parent["id"] not in augmented_nodes:
                            additional_node_ids.add(parent["id"])

                    # Add sibling IDs
                    for sibling in siblings:
                        if sibling["id"] and sibling["id"] not in augmented_nodes:
                            additional_node_ids.add(sibling["id"])

                # If we have additional nodes to fetch, get them from Neo4j
                if additional_node_ids:
                    # Convert to list for Neo4j query
                    add_ids = list(additional_node_ids)

                    # Fetch the additional nodes
                    add_result = session.run("""
                        MATCH (n) WHERE n.id IN $node_ids
                        RETURN n.id as id, n.text as text, n.metadata as metadata
                    """, node_ids=add_ids)

                    # Create document nodes for additional content
                    for record in add_result:
                        node_id = record["id"]
                        text = record["text"]
                        metadata = record["metadata"] if record["metadata"] else {}

                        # Create a Document node
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
                for record in result:
                    chunk_id = record["chunk_id"]
                    if chunk_id not in augmented_nodes:
                        continue

                    base_score = augmented_nodes[chunk_id]["score"]
                    parents = record["parents"]
                    siblings = record["siblings"]

                    # Boost parent nodes
                    for parent in parents:
                        parent_id = parent["id"]
                        if parent_id and parent_id in augmented_nodes:
                            # Set parent score to a fraction of this node's score
                            augmented_nodes[parent_id]["score"] = max(
                                augmented_nodes[parent_id]["score"],
                                base_score * self.parent_boost
                            )

                    # Boost sibling nodes based on distance
                    for sibling in siblings:
                        sibling_id = sibling["id"]
                        if sibling_id and sibling_id in augmented_nodes:
                            distance = sibling["distance"]
                            # Apply distance-based decay
                            distance_factor = 1 - (distance / (self.context_window + 1))
                            sibling_boost = self.sibling_boost * distance_factor

                            # Set sibling score with boost
                            augmented_nodes[sibling_id]["score"] = max(
                                augmented_nodes[sibling_id]["score"],
                                base_score * sibling_boost
                            )

                return augmented_nodes

        except Exception as e:
            logger.error(f"Error in hierarchical context retrieval: {str(e)}")
            return initial_nodes

class RAGSystem:
    """
    Multilingual Retrieval-Augmented Generation (RAG) system using LlamaIndex.
    Supports multiple languages with specialized embedding models for each.
    Uses lazy-loading of models to optimize memory usage.
    """

    # Default embedding dimensions for FastEmbed
    DEFAULT_EMBED_DIM = 384  # Update to match the actual dimension of FastEmbed's default model
    
    # Keeping this for backward compatibility but we'll use FastEmbed for all languages
    LANGUAGE_EMBEDDING_MODELS = {
        'en': "default",
        'de': "default",
        'es': "default", 
        'fr': "default",
        'it': "default",
        'nl': "default",
        'pt': "default",
        'multi': "default"
    }

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        qdrant_url: str = None,
        qdrant_collection_prefix: str = "regulaite_docs",
        openai_api_key: str = None,
        default_lang: str = "en",
        chunk_size: int = 1000,
        preload_languages: List[str] = None,
        hybrid_search: bool = True,  # New parameter for hybrid search
        vector_weight: float = 0.7,  # New parameter for vector search weight
        keyword_weight: float = 0.3,  # New parameter for keyword search weight
        hierarchical_retrieval: bool = True,  # New parameter for hierarchical retrieval
        parent_boost: float = 0.2,   # New parameter for parent boost
        sibling_boost: float = 0.1,  # New parameter for sibling boost
    ):
        """
        Initialize the multilingual RAG system with lazy-loading of models.

        Args:
            neo4j_uri: URI for the Neo4j database
            neo4j_user: Username for Neo4j
            neo4j_password: Password for Neo4j
            qdrant_url: URL for the Qdrant server (defaults to env var)
            qdrant_collection_prefix: Prefix for language-specific collections
            openai_api_key: OpenAI API key for LLM
            default_lang: Default language code
            chunk_size: Size of chunks for text splitting
            preload_languages: List of language codes to preload
            hybrid_search: Whether to use hybrid search (vector + keyword)
            vector_weight: Weight for vector search in hybrid retrieval (0-1)
            keyword_weight: Weight for keyword search in hybrid retrieval (0-1)
            hierarchical_retrieval: Whether to use hierarchical retrieval
            parent_boost: Score boost for parent sections in hierarchical retrieval
            sibling_boost: Score boost for sibling sections in hierarchical retrieval
        """
        # Store initialization parameters
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Get Qdrant URL from environment if not provided
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")

        # Collections
        self.qdrant_collection_prefix = qdrant_collection_prefix

        # OpenAI key (if used)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

        # Default language
        self.default_lang = default_lang

        # Chunking parameters
        self.chunk_size = chunk_size

        # Retrieval options
        self.hybrid_search = hybrid_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.hierarchical_retrieval = hierarchical_retrieval
        self.parent_boost = parent_boost
        self.sibling_boost = sibling_boost

        # Language detector
        self.language_detector = LanguageDetector()

        # Initialize Neo4j connection
        self.neo4j_driver = None
        self._connect_to_neo4j()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=self.qdrant_url)

        # Dictionary to store language-specific resources
        self.language_resources = {}

        # Preload specified languages
        if preload_languages:
            for lang in preload_languages:
                self._initialize_language(lang)

        # Always initialize default language
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
        """
        Get the vector dimension for a specific model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Vector dimension for the model
        """
        # For FastEmbed models, use the default dimension
        return self.DEFAULT_EMBED_DIM

    def _initialize_language(self, lang_code: str) -> bool:
        """
        Initialize resources for a specific language (lazy-loading).

        Args:
            lang_code: Language code to initialize

        Returns:
            True if initialization was successful, False otherwise
        """
        # Skip if already initialized
        if lang_code in self.language_resources:
            return True

        try:
            logger.info(f"Initializing resources for language: {lang_code}")

            # Create embedding model - use FastEmbed with consistent model for all languages
            try:
                # Use BAAI/bge-small-en-v1.5 which has 384 dimensions
                embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
                logger.info(f"Successfully initialized FastEmbedEmbedding with model BAAI/bge-small-en-v1.5 for language: {lang_code}")
            except Exception as emb_err:
                logger.error(f"Error creating FastEmbedEmbedding: {str(emb_err)}")
                return False
            
            # Get model-specific vector dimensions
            vector_dim = self._get_vector_dim_for_model("default")
            logger.info(f"Using vector dimension {vector_dim} for FastEmbed")
            
            # Check if Qdrant collection exists
            collection_name = f"{self.qdrant_collection_prefix}_{lang_code}"
            try:
                collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
                logger.info(f"Collection {collection_name} exists: {collection_info}")
                
                # Check if existing collection has the correct vector dimension
                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                    existing_dim = None
                    if hasattr(collection_info.config.params, 'vectors'):
                        if 'embed' in collection_info.config.params.vectors:
                            existing_dim = collection_info.config.params.vectors['embed'].size
                        elif 'default' in collection_info.config.params.vectors:
                            existing_dim = collection_info.config.params.vectors['default'].size
                    
                    if existing_dim is not None and existing_dim != vector_dim:
                        logger.warning(f"Collection {collection_name} has dimension {existing_dim}, but we need {vector_dim}")
                        logger.warning(f"Recreating collection {collection_name} with correct dimension")
                        
                        # Get existing points if available
                        try:
                            existing_points = []
                            scroll_result = self.qdrant_client.scroll(
                                collection_name=collection_name,
                                limit=100
                            )
                            points, next_page_offset = scroll_result
                            existing_points.extend(points)
                            
                            # If there are points in the collection, save metadata for later re-indexing
                            if existing_points:
                                logger.warning(f"Found {len(existing_points)} points in collection. You'll need to re-index these documents.")
                                doc_ids = set()
                                for point in existing_points:
                                    if hasattr(point, 'payload') and point.payload:
                                        if 'doc_id' in point.payload:
                                            doc_ids.add(point.payload['doc_id'])
                                
                                logger.warning(f"Documents to re-index: {doc_ids}")
                        except Exception as e:
                            logger.error(f"Error retrieving existing points: {str(e)}")
                        
                        # Delete the collection and recreate it
                        self.qdrant_client.delete_collection(collection_name=collection_name)
                        logger.info(f"Deleted collection {collection_name}")
                        
                        # Create a new collection with correct dimensions
                        self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config={
                                "embed": {
                                    "size": vector_dim,
                                    "distance": "Cosine"
                                }
                            }
                        )
                        logger.info(f"Recreated collection {collection_name} with dimension {vector_dim}")
            except Exception as e:
                # Collection doesn't exist, create it
                logger.info(f"Creating new collection {collection_name}: {str(e)}")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "embed": {
                            "size": vector_dim,
                            "distance": "Cosine"
                        }
                    }
                )
            
            # Initialize vector store
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
                vector_name="embed"
            )
            
            # Create vector index
            Settings.embed_model = embed_model
            vector_index = VectorStoreIndex.from_vector_store(vector_store)
            
            # Create a proper retriever from the index
            vector_retriever = vector_index.as_retriever(similarity_top_k=5)
            
            # Store resources
            self.language_resources[lang_code] = {
                "embedding_model": embed_model,
                "vector_retriever": vector_retriever
            }
            
            # Add BM25 retriever if available
            if BM25Retriever is not None:
                try:
                    # Create a new VectorStoreIndex with the right vector name for BM25
                    # to ensure it uses the correct docstore
                    bm25_vector_store = QdrantVectorStore(
                        client=self.qdrant_client,
                        collection_name=collection_name,
                        vector_name="embed"
                    )
                    bm25_index = VectorStoreIndex.from_vector_store(bm25_vector_store)
                    
                    # Create BM25 retriever from the docstore
                    self.language_resources[lang_code]["bm25_retriever"] = BM25Retriever.from_defaults(
                        docstore=bm25_index.docstore,
                        similarity_top_k=5
                    )
                except Exception as bm25_error:
                    logger.warning(f"Failed to initialize BM25Retriever: {str(bm25_error)}")
                    # Continue without BM25 retriever
            
            # Mark as initialized
            return True

        except Exception as e:
            logger.error(f"Error initializing resources for language {lang_code}: {str(e)}")
            if "max() arg is an empty sequence" in str(e):
                logger.warning(f"This may indicate that the Qdrant collection {self.qdrant_collection_prefix}_{lang_code} is empty or not properly configured")
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cross_lingual: bool = True,
        use_hybrid: Optional[bool] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        use_neo4j: bool = True,  # New parameter to control Neo4j usage
        use_hierarchical: Optional[bool] = None  # New parameter to control hierarchical retrieval
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            use_cross_lingual: Whether to use cross-lingual retrieval if language detection is different
            use_hybrid: Whether to use hybrid retrieval (overrides instance setting)
            filter_criteria: Optional criteria to filter results (metadata filtering)
            use_neo4j: Whether to use Neo4j for additional context retrieval
            use_hierarchical: Whether to use hierarchical retrieval (overrides instance setting)

        Returns:
            List of retrieved documents
        """
        if not query.strip():
            logger.warning("Empty query received")
            return []

        # Detect query language
        query_lang = self.detect_language(query)
        logger.info(f"Query language detected: {query_lang}")

        # Handle fallback case if no languages were initialized properly
        if not self.language_resources and "fallback" not in self.language_resources:
            logger.warning("No language models initialized, initializing fallback placeholder")
            self._initialize_fallback_placeholder()
            return []
            
        # If only fallback is available, use it
        if list(self.language_resources.keys()) == ["fallback"]:
            logger.warning("Only fallback placeholder available, returning empty results")
            return []

        # Determine retrieval languages
        retrieval_languages = [query_lang]
        if use_cross_lingual and query_lang != self.default_lang:
            retrieval_languages.append(self.default_lang)
            logger.info(f"Using cross-lingual retrieval with languages: {retrieval_languages}")

        # Determine whether to use hybrid search
        if use_hybrid is None:
            use_hybrid = self.hybrid_search

        # Determine whether to use hierarchical retrieval
        if use_hierarchical is None:
            use_hierarchical = self.hierarchical_retrieval

        # Ensure all languages are initialized
        for lang in retrieval_languages:
            actual_lang = self.ensure_language_initialized(lang)
            # Replace with actual language if it was a fallback
            if actual_lang != lang:
                index = retrieval_languages.index(lang)
                retrieval_languages[index] = actual_lang

        # Get results from each language
        all_results = []
        for lang in retrieval_languages:
            logger.info(f"Retrieving for language: {lang}")

            # Get the right index and search resources
            if lang not in self.language_resources:
                logger.warning(f"Language {lang} not initialized, skipping")
                continue

            resources = self.language_resources[lang]

            # Select appropriate retriever based on settings
            retriever = None

            # Build base retriever (vector or hybrid)
            if use_hybrid and BM25Retriever is not None and resources.get("bm25_retriever"):
                vector_retriever = resources["vector_retriever"]
                keyword_retriever = resources["bm25_retriever"]

                try:
                    # Create hybrid retriever
                    base_retriever = HybridRetriever(
                        vector_retriever=vector_retriever,
                        keyword_retriever=keyword_retriever,
                        vector_weight=self.vector_weight,
                        keyword_weight=self.keyword_weight,
                        top_k=top_k
                    )
                    logger.info(f"Using hybrid retriever for {lang}")
                except Exception as hybrid_error:
                    logger.warning(f"Failed to create hybrid retriever: {str(hybrid_error)}, falling back to vector search")
                    base_retriever = resources["vector_retriever"]
            else:
                # Use just vector retrieval
                base_retriever = resources["vector_retriever"]
                if use_hybrid:
                    logger.warning("Hybrid search requested but BM25Retriever not available, falling back to vector search")

            # Apply hierarchical retrieval if enabled
            if use_hierarchical and self.neo4j_driver and use_neo4j:
                retriever = HierarchicalRetriever(
                    base_retriever=base_retriever,
                    neo4j_driver=self.neo4j_driver,
                    top_k=top_k,
                    parent_boost=self.parent_boost,
                    sibling_boost=self.sibling_boost
                )
                logger.info(f"Using hierarchical retriever for {lang}")
            else:
                retriever = base_retriever

            # Build query bundle
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=query)

            # Apply metadata filtering if specified
            if filter_criteria and isinstance(filter_criteria, dict):
                # Create a metadata filter function
                def metadata_filter_fn(node):
                    # Only apply filter if node has metadata
                    if not hasattr(node, 'metadata') or not node.metadata:
                        return False

                    # Check each filter criterion
                    for key, value in filter_criteria.items():
                        if key not in node.metadata:
                            return False

                        # Handle different types of values
                        if isinstance(value, list):
                            # List type - check if any value matches
                            if node.metadata[key] not in value:
                                return False
                        elif isinstance(value, dict) and 'operator' in value:
                            # Dict with operator for numeric comparisons
                            if not self._apply_comparison_filter(node.metadata[key], value):
                                return False
                        else:
                            # Direct equality comparison
                            if node.metadata[key] != value:
                                return False

                    # If passed all checks, include this node
                    return True

                # Apply the filter function to the retriever
                retriever.filter_fn = metadata_filter_fn
                logger.info(f"Applied metadata filters: {filter_criteria}")

            # Execute retrieval
            try:
                # Check if collection is empty before attempting retrieval
                try:
                    # Get collection info to check if it has points
                    collection_info = self.qdrant_client.get_collection(
                        collection_name=f"{self.qdrant_collection_prefix}_{lang}"
                    )
                    point_count = collection_info.points_count
                    
                    if point_count == 0:
                        logger.warning(f"Collection {self.qdrant_collection_prefix}_{lang} is empty, skipping retrieval")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to check collection status: {str(e)}")
                
                # Attempt retrieval if we get here
                retrieval_results = retriever.retrieve(query_bundle)

                # Convert to standard format
                for result in retrieval_results.nodes:
                    node = result.node

                    # Extract text and metadata
                    result_item = {
                        "text": node.text,
                        "score": result.score,
                        "id": node.node_id,
                        "metadata": node.metadata or {},
                        "language": lang
                    }

                    all_results.append(result_item)

            except Exception as e:
                logger.error(f"Error during retrieval for language {lang}: {str(e)}")

        return all_results

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
        if lang_code not in self.LANGUAGE_EMBEDDING_MODELS:
            if "multi" in self.LANGUAGE_EMBEDDING_MODELS:
                return "multi"
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
        if lang_code not in self.language_resources:
            success = self._initialize_language(lang_code)
            if success:
                return lang_code

        # If we get here, either initialization failed or was not needed
        # Check if the language is already initialized
        if lang_code in self.language_resources:
            return lang_code

        # Try multilingual fallback
        if "multi" in self.LANGUAGE_EMBEDDING_MODELS:
            if "multi" not in self.language_resources:
                success = self._initialize_language("multi")
                if success:
                    return "multi"
            elif "multi" in self.language_resources:
                return "multi"

        # Try default language as a final fallback
        if self.default_lang not in self.language_resources:
            success = self._initialize_language(self.default_lang)
            if success:
                return self.default_lang
        elif self.default_lang in self.language_resources:
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
        self.language_resources["fallback"] = {
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
                return False
                
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
                    return False
                
                # Check if document is already indexed and we're not forcing reindex
                if not force_reindex and doc_record.get("is_indexed") == True:
                    logger.info(f"Document {doc_id} is already indexed. Use force_reindex=True to reindex.")
                    return {
                        "doc_id": doc_id,
                        "vector_count": 0,
                        "message": "Document already indexed"
                    }
                    
                # Handle potentially missing properties with safe gets
                doc_name = doc_record.get("title") or doc_record.get("name", f"Document-{doc_id}")
                doc_language = doc_record.get("language") or self.default_lang
                doc_category = doc_record.get("category", "")
                doc_author = doc_record.get("author", "")
                doc_file_type = doc_record.get("file_type", "")
                
                # Get all chunks for the document
                chunk_query = """
                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                RETURN c.chunk_id as chunk_id, c.text as text, 
                       COALESCE(c.section, 'Default') as section, 
                       COALESCE(c.page_num, 0) as page_num, 
                       COALESCE(c.order_index, c.index, 0) as order_index
                ORDER BY COALESCE(c.order_index, c.index, 0)
                """
                
                chunks_result = session.run(chunk_query, doc_id=doc_id)
                chunks = list(chunks_result)
                
                if not chunks:
                    logger.warning(f"Document {doc_id} has no chunks to index")
                    return {
                        "doc_id": doc_id,
                        "vector_count": 0,
                        "message": "Document has no chunks to index"
                    }
                    
                # Ensure the language is initialized
                actual_lang = self.ensure_language_initialized(doc_language)
                
                # Get the correct resources for this language
                if actual_lang not in self.language_resources:
                    logger.error(f"Language {actual_lang} not available for indexing")
                    return False
                    
                resources = self.language_resources[actual_lang]
                embed_model = resources.get("embedding_model")
                if not embed_model:
                    logger.error(f"No embedding model for language {actual_lang}")
                    return False
                
                # 2. Convert chunks to nodes and embed them
                indexed_chunks = 0
                for chunk in chunks:
                    try:
                        chunk_id = chunk["chunk_id"]
                        
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
                            "file_type": doc_file_type or ""
                        }
                        
                        # Create vector
                        text = chunk["text"]
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
                        
                        # Check if collection exists, create if it doesn't
                        try:
                            collection_info = self.qdrant_client.get_collection(collection_name)
                            logger.info(f"Collection {collection_name} exists with {collection_info.points_count} points")
                            
                            # Check vector name configuration to determine proper vector name
                            vector_name = "default"
                            if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                                if hasattr(collection_info.config.params, 'vectors') and 'embed' in collection_info.config.params.vectors:
                                    vector_name = "embed"
                                    logger.info(f"Using 'embed' as vector name for existing collection {collection_name}")
                            
                        except Exception as collection_error:
                            logger.info(f"Creating new collection {collection_name}: {str(collection_error)}")
                            vector_size = len(embedding)
                            vector_name = "default"  # Use default for new collections
                            
                            self.qdrant_client.create_collection(
                                collection_name=collection_name,
                                vectors_config=rest.VectorParams(
                                    size=vector_size,
                                    distance=rest.Distance.COSINE
                                ),
                                # Define the vector name explicitly as the default
                                vectors={
                                    vector_name: rest.VectorParams(
                                        size=vector_size,
                                        distance=rest.Distance.COSINE
                                    )
                                }
                            )
                            logger.info(f"Created new collection {collection_name} with vector size {vector_size}")
                        
                        # Upsert the point using UUID as the point_id
                        logger.info(f"Upserting point {point_id} for chunk {chunk_id} in collection {collection_name} with vector name {vector_name}")
                        
                        try:
                            # Convert embedding to list if it's a numpy array
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                                
                            # Prepare vector with the appropriate name
                            vector_dict = {vector_name: embedding}
                            
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
        return list(self.language_resources.keys())
    