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

    # Mapping of language codes to embedding models
    LANGUAGE_EMBEDDING_MODELS = {
        'en': "BAAI/bge-small-en-v1.5",       # English
        'de': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # German
        'es': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Spanish
        'fr': "sentence-transformers/distiluse-base-multilingual-cased-v2",   # French - Updated
        'it': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Italian
        'nl': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Dutch
        'pt': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Portuguese
        'multi': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Multilingual fallback
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

            # Get embedding model name for this language
            if lang_code not in self.LANGUAGE_EMBEDDING_MODELS:
                logger.warning(f"No embedding model defined for language: {lang_code}")
                return False

            model_name = self.LANGUAGE_EMBEDDING_MODELS[lang_code]

            # Create embedding model
            embed_model = FastEmbedEmbedding(model_name=model_name)
            
            # Get model-specific vector dimensions
            vector_dim = self._get_vector_dim_for_model(model_name)
            logger.info(f"Using vector dimension {vector_dim} for model {model_name}")
            
            # Check if Qdrant collection exists
            collection_name = f"{self.qdrant_collection_prefix}_{lang_code}"
            try:
                collection_exists = self.qdrant_client.get_collection(collection_name=collection_name)
                logger.info(f"Collection {collection_name} exists: {collection_exists}")
            except Exception as e:
                # Collection doesn't exist, create it
                logger.info(f"Creating new collection {collection_name}: {str(e)}")
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "embed": {
                            "size": vector_dim,  # Use model-specific dimension
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

    def _get_vector_dim_for_model(self, model_name: str) -> int:
        """
        Get the correct vector dimension for a specific model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Vector dimension size
        """
        # Known vector dimensions for specific models
        dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
            "sentence-transformers/distiluse-base-multilingual-cased-v2": 512
        }
        
        # Return known dimension or default to 384
        return dimensions.get(model_name, 384)

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

        # Last resort: default language
        if self.default_lang not in self.language_resources:
            success = self._initialize_language(self.default_lang)
            if not success:
                logger.error(f"Failed to initialize any language models")
                # Rather than raising an exception, we'll initialize a fallback placeholder
                # This allows the system to continue operating with no retrieval
                self._initialize_fallback_placeholder()
                return "fallback"

        return self.default_lang

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
