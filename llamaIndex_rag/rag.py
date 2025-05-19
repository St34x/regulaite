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
# This seems to be a custom class definition.
# class BaseRetrieverOutput:
#     def __init__(self, nodes: List[NodeWithScore]):
#         self.nodes = nodes

class HybridRetriever(BaseRetriever):
    """Hybrid retriever that combines vector search and BM25 keyword search."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        keyword_retriever: BM25Retriever, 
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 5
    ):
        """Initialize the hybrid retriever.

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

        total_weight = vector_weight + keyword_weight
        if abs(total_weight - 1.0) > 1e-9:
            self.vector_weight = vector_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight

        super().__init__()
        
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using hybrid search.
        
        This method overrides the base retrieve to ensure we always return a list of NodeWithScore objects
        rather than a BaseRetrieverOutput object.
        
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
            logger.error(f"Unexpected result type from _retrieve: {type(output)}")
            return []

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

# End of content to restore before HierarchicalRetriever

class HierarchicalRetriever(BaseRetriever):
    """Retriever that considers document hierarchy for better contextual retrieval.
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
        """Initialize the hierarchical retriever.

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
        
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using hierarchical context-aware retrieval.
        
        This method overrides the base retrieve to ensure we always return a list of NodeWithScore objects
        rather than a BaseRetrieverOutput object.
        
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
            logger.error(f"Unexpected result type from _retrieve: {type(output)}")
            return []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using hierarchical context-aware retrieval.

        Args:
            query_bundle: Query bundle

        Returns:
            List of NodeWithScore objects
        """
        # Get base results - call _retrieve directly to avoid problematic base implementation
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
                return []
        except Exception as e:
            logger.error(f"Error calling base retriever: {str(e)}")
            return []

        if not self.neo4j_driver:
            logger.warning("No Neo4j driver available, skipping hierarchical retrieval, returning base results.")
            return base_nodes_with_scores

        initial_nodes_data = {}
        for node_with_score in base_nodes_with_scores:
            node = node_with_score.node
            node_id = node.node_id
            initial_nodes_data[node_id] = {
                "node": node,
                "score": node_with_score.score,
                "source": "direct"
            }

        # If too few initial nodes to warrant hierarchical augmentation, return them directly.
        # Consider making this threshold configurable.
        if len(initial_nodes_data) < 1: # If 0 or 1 node, no hierarchy to explore based on pairs
            logger.info(f"Too few initial nodes ({len(initial_nodes_data)}) for hierarchical augmentation, returning base results.")
            return base_nodes_with_scores

        augmented_nodes_data = self._augment_with_hierarchical_context(initial_nodes_data)

        sorted_nodes_data = sorted(
            augmented_nodes_data.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:self.top_k]

        final_results = [
            NodeWithScore(
                node=item["node"],
                score=item["score"]
            ) for item in sorted_nodes_data
        ]
        return final_results

    def _augment_with_hierarchical_context(self, initial_nodes_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Augment the initial nodes with hierarchical context from Neo4j.

        Args:
            initial_nodes_data: Dictionary of initial nodes data with node_id as key

        Returns:
            Dictionary of augmented nodes data with additional context
        """
        try:
            node_ids = list(initial_nodes_data.keys())
            with self.neo4j_driver.session() as session:
                # Note: Escaped triple quotes for the Cypher query string literal
                result = session.run('''
                    MATCH (c:Chunk) WHERE c.id IN $node_ids
                    OPTIONAL MATCH (c)-[:PART_OF]->(parent:Section)
                    OPTIONAL MATCH (parent)<-[:PART_OF]-(sibling:Chunk)
                    WHERE sibling.id <> c.id
                    WITH c, parent, sibling,
                         CASE WHEN sibling IS NOT NULL THEN abs(sibling.sequence_num - c.sequence_num) ELSE null END as distance
                    WHERE distance IS NULL OR distance <= $context_window
                    RETURN c.id as chunk_id,
                           collect(DISTINCT {id: parent.id, type: \'parent\'}) as parents,
                           collect(DISTINCT {id: sibling.id, type: \'sibling\', distance: distance}) as siblings
                ''', node_ids=node_ids, context_window=self.context_window)

                augmented_nodes_data = initial_nodes_data.copy()
                additional_node_ids_to_fetch = set()

                for record in result:
                    chunk_id = record["chunk_id"]
                    parents = record["parents"]
                    siblings = record["siblings"]
                    for parent_data in parents:
                        if parent_data["id"] and parent_data["id"] not in augmented_nodes_data:
                            additional_node_ids_to_fetch.add(parent_data["id"])
                    for sibling_data in siblings:
                        if sibling_data["id"] and sibling_data["id"] not in augmented_nodes_data:
                            additional_node_ids_to_fetch.add(sibling_data["id"])

                if additional_node_ids_to_fetch:
                    add_ids_list = list(additional_node_ids_to_fetch)
                    add_result = session.run('''
                        MATCH (n) WHERE n.id IN $node_ids
                        RETURN n.id as id, n.text as text, n.metadata as metadata
                    ''', node_ids=add_ids_list)

                    for record in add_result:
                        node_id = record["id"]
                        text = record["text"]
                        metadata = record["metadata"] if record["metadata"] else {}
                        doc_node = TextNode(
                            text=text,
                            id_=node_id,
                            metadata=metadata
                        )
                        augmented_nodes_data[node_id] = {
                            "node": doc_node,
                            "score": 0,
                            "source": "hierarchical"
                        }

                for record in result:
                    chunk_id = record["chunk_id"]
                    if chunk_id not in augmented_nodes_data:
                        continue
                    base_score = augmented_nodes_data[chunk_id]["score"]
                    parents = record["parents"]
                    siblings = record["siblings"]

                    for parent_data in parents:
                        parent_id = parent_data["id"]
                        if parent_id and parent_id in augmented_nodes_data:
                            augmented_nodes_data[parent_id]["score"] = max(
                                augmented_nodes_data[parent_id]["score"],
                                base_score * self.parent_boost
                            )
                    for sibling_data in siblings:
                        sibling_id = sibling_data["id"]
                        if sibling_id and sibling_id in augmented_nodes_data:
                            distance = sibling_data["distance"]
                            if distance is not None:
                                distance_factor = 1 - (distance / (self.context_window + 1))
                                sibling_boost_factor = self.sibling_boost * distance_factor
                                augmented_nodes_data[sibling_id]["score"] = max(
                                    augmented_nodes_data[sibling_id]["score"],
                                    base_score * sibling_boost_factor
                                )
                return augmented_nodes_data
        except Exception as e:
            logger.error(f"Error in hierarchical context retrieval: {str(e)}", exc_info=True)
            return initial_nodes_data

# ... existing code ... 

class RAGSystem:
    """Multilingual Retrieval Augmented Generation (RAG) system.
    Supports multiple languages, Qdrant for vector storage, Neo4j for graph-based
    hierarchical retrieval, and hybrid search (vector + BM25).
    """
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        qdrant_url: str = None,
        qdrant_collection_prefix: str = "regulaite_docs",
        openai_api_key: str = None, # Retained for potential direct OpenAI calls if ever needed
        default_lang: str = "fr",
        chunk_size: int = 1000,
        preload_languages: List[str] = ["fr"],
        hybrid_search: bool = True,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        hierarchical_retrieval: bool = True,
        parent_boost: float = 0.2,
        sibling_boost: float = 0.1,
        use_reranker: bool = True,
        reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        reranker_top_n: int = 5
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_driver = None
        try:
            self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.neo4j_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}. Hierarchical retrieval will be disabled.")
            self.neo4j_driver = None

        self.qdrant_collection_prefix = qdrant_collection_prefix
        self.qdrant_client = None
        if qdrant_url:
            try:
                self.qdrant_client = QdrantClient(url=qdrant_url)
                logger.info(f"Connected to Qdrant at {qdrant_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant at {qdrant_url}: {e}")
                self.qdrant_client = None
        else:
            logger.warning("Qdrant URL not provided, vector store operations will be limited.")
        
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
            logger.info("OpenAI API key configured.")

        self.default_lang = default_lang
        self.chunk_size = chunk_size
        self.language_settings: Dict[str, Dict[str, Any]] = {}
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
        self.reranker = None
        if self.use_reranker:
            try:
                self.reranker = SentenceTransformerRerank(
                    model=self.reranker_model_name,
                    top_n=self.reranker_top_n
                )
                logger.info(f"Initialized SentenceTransformerRerank with model: {self.reranker_model_name}, top_n: {self.reranker_top_n}")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformerRerank: {e}. Reranking will be disabled.")
                self.reranker = None
                self.use_reranker = False

        if preload_languages:
            for lang_code in preload_languages:
                self._initialize_language(lang_code)
        if self.default_lang not in self.language_settings:
            self._initialize_language(self.default_lang)

    def _get_qdrant_collection_name(self, lang_code: str) -> str:
        return f"{self.qdrant_collection_prefix}_{lang_code}"

    def _initialize_language(self, lang_code: str) -> bool:
        if lang_code in self.language_settings and self.language_settings[lang_code].get("index") is not None:
            logger.info(f"Language {lang_code} already initialized.")
            return True
        
        logger.info(f"Initializing language: {lang_code}...")
        if not self.qdrant_client:
            logger.error(f"Qdrant client not available. Cannot initialize language {lang_code}.")
            return False

        try:
            embed_model_name = "BAAI/bge-base-en"
            embed_model = FastEmbedEmbedding(model_name=embed_model_name)
            logger.info(f"FastEmbedEmbedding model '{embed_model_name}' loaded for {lang_code}.")

            Settings.embed_model = embed_model

            collection_name = self._get_qdrant_collection_name(lang_code)
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
            )
            logger.info(f"QdrantVectorStore initialized for collection: {collection_name}")

            try:
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model,
                    storage_context=storage_context
                )
                logger.info(f"Successfully loaded index from Qdrant collection '{collection_name}'.")
            except Exception as e:
                logger.info(f"Could not load index from Qdrant for '{collection_name}' (may be normal if first time): {e}. Will proceed as if new.")
                index = VectorStoreIndex([], embed_model=embed_model, storage_context=StorageContext.from_defaults(vector_store=vector_store))
                logger.info(f"Created new/empty VectorStoreIndex for '{collection_name}'.")

            vector_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=10,
                vector_store_query_mode=VectorStoreQueryMode.DEFAULT, 
            )
            logger.info(f"Base VectorIndexRetriever initialized for {lang_code} with mode DEFAULT")

            self.language_settings[lang_code] = {
                "embed_model_container": {"model": embed_model}, 
                "vector_store": vector_store,
                "index": index,
                "vector_retriever": vector_retriever,
                "retriever": vector_retriever 
            }

            if self.hybrid_search_enabled and BM25Retriever is not None:
                try:
                    logger.info(f"Attempting to initialize BM25Retriever for {lang_code}. Fetching nodes...")
                    all_nodes_for_bm25 = []
                    offset = None
                    try:
                        self.qdrant_client.get_collection(collection_name=collection_name)
                    except Exception as e:
                        logger.warning(f"Qdrant collection '{collection_name}' does not exist or error checking: {e}. Skipping BM25 for now.")
                        raise
                        
                    while True:
                        points_page, next_offset = self.qdrant_client.scroll(
                            collection_name=collection_name,
                            limit=250, 
                            offset=offset,
                            with_payload=True,
                            with_vectors=False 
                        )
                        for point in points_page:
                            node_text = point.payload.get("text", "") if point.payload else ""
                            node_metadata = point.payload if point.payload else {}
                            if 'chunk_id' not in node_metadata: node_metadata['chunk_id'] = point.id
                            if 'doc_id' not in node_metadata and point.payload and 'doc_id' in point.payload:
                                node_metadata['doc_id'] = point.payload['doc_id']
                            all_nodes_for_bm25.append(TextNode(text=node_text, id_=point.id, metadata=node_metadata))
                        if next_offset is None: break
                        offset = next_offset
                    
                    logger.info(f"Fetched {len(all_nodes_for_bm25)} nodes for BM25 for {lang_code}.")
                    if all_nodes_for_bm25:
                        bm25_language = lang_code
                        logger.info(f"Using language '{bm25_language}' for BM25Retriever.")
                        current_stemmer = Stemmer.Stemmer(bm25_language) if bm25_language else None
                        bm25_retriever = BM25Retriever.from_defaults(
                            nodes=all_nodes_for_bm25,
                            similarity_top_k=10,
                            language=bm25_language,
                            stemmer=current_stemmer
                        )
                        self.language_settings[lang_code]["bm25_retriever"] = bm25_retriever
                        logger.info(f"BM25Retriever initialized for {lang_code} with language '{bm25_language}'.")
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
            
            logger.info(f"Language settings for {lang_code} fully populated. Main retriever is {type(self.language_settings[lang_code]['retriever']).__name__}")
            return True

        except Exception as e:
            logger.error(f"Fatal error initializing language {lang_code}: {e}", exc_info=True)
            if lang_code in self.language_settings:
                del self.language_settings[lang_code]
            return False

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cross_lingual: bool = True,
        use_hybrid: Optional[bool] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        use_neo4j: bool = True, 
        use_hierarchical: Optional[bool] = None 
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks based on the query."""
        if not query.strip():
            logger.warning("Empty query received")
            return []

        query_lang_detected = self.language_detector.detect_language(query)["language_code"]
        logger.info(f"Detected query language: {query_lang_detected}")

        target_lang_code = self.ensure_language_initialized(query_lang_detected)
        if not target_lang_code:
            if use_cross_lingual and query_lang_detected != self.default_lang:
                logger.warning(f"Query language '{query_lang_detected}' not initialized. Attempting retrieval in default language '{self.default_lang}'")
                target_lang_code = self.ensure_language_initialized(self.default_lang)
                if not target_lang_code:
                    logger.error(f"Default language '{self.default_lang}' also not initialized. Cannot retrieve.")
                    return []
            else:
                logger.error(f"Query language '{query_lang_detected}' not initialized and cross-lingual search disabled or not applicable. Cannot retrieve.")
                return []
        
        lang_settings = self.language_settings.get(target_lang_code)
        if not lang_settings or not lang_settings.get("retriever"): 
            logger.error(f"Retriever for language '{target_lang_code}' is not available.")
            return []

        final_retriever = lang_settings["retriever"]
        
        query_bundle = QueryBundle(query_str=query)
        retrieved_nodes_with_scores = []

        try:
            output = final_retriever._retrieve(query_bundle)
            # Handle different return types from the retriever
            if hasattr(output, 'nodes'):
                # It's a BaseRetrieverOutput object
                retrieved_nodes_with_scores = output.nodes
            elif isinstance(output, list):
                # It's a list of NodeWithScore
                if not output:
                    logger.info("Retriever returned empty list.")
                    return []
                if all(isinstance(n, NodeWithScore) for n in output):
                    retrieved_nodes_with_scores = output
                else:
                    logger.error(f"Retriever {type(final_retriever).__name__}._retrieve() returned list with non-NodeWithScore elements.")
                    return []
            else:
                logger.error(f"Retriever {type(final_retriever).__name__}._retrieve() returned unsupported type: {type(output)}")
                return []
            
        except Exception as e:
            logger.error(f"Error during retrieval with {type(final_retriever).__name__}: {e}", exc_info=True)
            return []

        logger.info(f"Retrieved {len(retrieved_nodes_with_scores)} nodes initially with {type(final_retriever).__name__}.")

        if self.reranker and self.use_reranker and retrieved_nodes_with_scores:
            logger.info(f"Applying reranking. Initial nodes for reranking: {len(retrieved_nodes_with_scores)}, reranker top_n: {self.reranker_top_n}")
            try:
                reranked_nodes_with_scores = self.reranker.postprocess_nodes(
                    retrieved_nodes_with_scores,
                    query_bundle=query_bundle
                )
                logger.info(f"Nodes after reranking: {len(reranked_nodes_with_scores)}")
                retrieved_nodes_with_scores = reranked_nodes_with_scores 
            except Exception as e:
                logger.error(f"Error during reranking: {e}", exc_info=True)

        final_results_list = []
        if retrieved_nodes_with_scores:
            for res_node in retrieved_nodes_with_scores:
                node_obj = res_node.node
                score = res_node.score
                node_data = {
                    "text": node_obj.get_content(),
                    "metadata": node_obj.metadata or {},
                    "score": score,
                    "doc_id": node_obj.metadata.get("doc_id", node_obj.node_id),
                    "chunk_id": node_obj.node_id,
                }
                if self.reranker and self.use_reranker:
                    node_data["metadata"]["reranked_score"] = score
                    node_data["metadata"]["reranker_model"] = self.reranker_model_name
                
                if filter_criteria:
                    if not self._matches_filter(node_data["metadata"], filter_criteria):
                        continue

                final_results_list.append(node_data)
        
        final_output = final_results_list[:top_k]

        logger.info(f"Returning {len(final_output)} nodes after all processing for query: '{query[:50]}...'")
        return final_output

    def _matches_filter(self, node_metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Checks if a node's metadata matches the given filter criteria."""
        if not node_metadata:
            return False

        for key, condition in filter_criteria.items():
            node_value = node_metadata.get(key)
            if node_value is None:
                return False

            if isinstance(condition, dict) and "operator" in condition and "value" in condition:
                op = condition["operator"].lower()
                val = condition["value"]
                if op == "in":
                    if not isinstance(val, list):
                        logger.warning(f"IN operator expects a list value for key '{key}', got {type(val)}")
                        return False
                    if node_value not in val:
                        return False
                elif op == "nin":
                    if not isinstance(val, list):
                        logger.warning(f"NIN operator expects a list value for key '{key}', got {type(val)}")
                        return False
                    if node_value in val:
                        return False
                elif op == "==" or op == "eq":
                    if node_value != val:
                        return False
                elif op == "!=" or op == "ne":
                    if node_value == val:
                        return False
                elif op == ">" or op == "gt":
                    if not (isinstance(node_value, (int, float)) and isinstance(val, (int, float)) and node_value > val):
                        return False
                elif op == ">=" or op == "gte":
                    if not (isinstance(node_value, (int, float)) and isinstance(val, (int, float)) and node_value >= val):
                        return False
                elif op == "<" or op == "lt":
                    if not (isinstance(node_value, (int, float)) and isinstance(val, (int, float)) and node_value < val):
                        return False
                elif op == "<=" or op == "lte":
                    if not (isinstance(node_value, (int, float)) and isinstance(val, (int, float)) and node_value <= val):
                        return False
                else:
                    logger.warning(f"Unsupported filter operator '{op}' for key '{key}'")
                    return False
            else:
                if node_value != condition:
                    return False
        return True

    def detect_language(self, text: str) -> str:
        """Detect the language of a text."""
        lang_info = self.language_detector.detect_language(text)
        return lang_info.get("language_code", self.default_lang)

    def ensure_language_initialized(self, lang_code: str) -> Optional[str]:
        """Ensures a language is initialized. Returns the language code that is usable, or None."""
        if lang_code in self.language_settings:
            return lang_code
        if self._initialize_language(lang_code):
            return lang_code
        logger.warning(f"Failed to initialize requested language '{lang_code}'. Check logs for details.")
        return None

    def index_documents(
        self,
        documents: List[Union[Document, Dict[str, Any]]],
        doc_id: str,
        lang_code: Optional[str] = None,
        batch_size: int = 100
    ) -> None:
        """Indexes a list of LlamaIndex Documents or dictionaries into the appropriate Qdrant collection."""
        target_lang_code = lang_code if lang_code else self.default_lang
        
        initialized_lang = self.ensure_language_initialized(target_lang_code)
        if not initialized_lang:
            logger.error(f"Cannot index documents: Language '{target_lang_code}' could not be initialized.")
            return
        target_lang_code = initialized_lang

        lang_specific_settings = self.language_settings[target_lang_code]
        index = lang_specific_settings["index"]
        embed_model = lang_specific_settings["embed_model_container"]["model"]
        
        original_settings_embed_model = Settings.embed_model
        Settings.embed_model = embed_model
        
        llama_documents = []
        for doc_data in documents:
            if isinstance(doc_data, Document):
                if 'doc_id' not in doc_data.metadata:
                    doc_data.metadata['doc_id'] = doc_id
                llama_documents.append(doc_data)
            elif isinstance(doc_data, dict):
                text_content = doc_data.get("text", "")
                metadata = doc_data.get("metadata", {})
                if 'doc_id' not in metadata:
                    metadata['doc_id'] = doc_id
                llama_documents.append(Document(text=text_content, metadata=metadata))
            else:
                logger.warning(f"Skipping unsupported document type: {type(doc_data)}")
        
        if not llama_documents:
            logger.info("No valid documents to index.")
            Settings.embed_model = original_settings_embed_model
            return

        node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_size // 10)
        nodes = node_parser.get_nodes_from_documents(llama_documents, show_progress=True)
        
        for node in nodes:
            if "chunk_id" not in node.metadata:
                node.metadata["chunk_id"] = str(uuid.uuid4())
            if "doc_id" not in node.metadata:
                node.metadata["doc_id"] = doc_id
            if "language" not in node.metadata:
                node.metadata["language"] = target_lang_code

        logger.info(f"Inserting {len(nodes)} nodes into index for language '{target_lang_code}' and doc_id '{doc_id}'.")
        index.insert_nodes(nodes, show_progress=True)
        logger.info(f"Successfully indexed {len(nodes)} nodes for doc_id '{doc_id}' in language '{target_lang_code}'.")

        logger.info(f"Re-initializing language '{target_lang_code}' to update retrievers after indexing.")
        self._initialize_language(target_lang_code)
        
        Settings.embed_model = original_settings_embed_model

    def delete_document(self, doc_id: str, lang_code: Optional[str] = None) -> bool:
        """Deletes all chunks/nodes associated with a document_id from all relevant Qdrant collections."""
        if not self.qdrant_client:
            logger.error("Qdrant client not available. Cannot delete document.")
            return False

        target_languages = [lang_code] if lang_code else list(self.language_settings.keys())
        if not target_languages and self.default_lang:
             target_languages = [self.default_lang]
        elif not target_languages and not self.default_lang:
            logger.error("No languages specified or initialized, and no default language. Cannot determine collection to delete from.")
            return False

        all_deleted_successfully = True
        possible_payload_fields_for_doc_id = ["doc_id", "metadata.doc_id"]

        for lc in target_languages:
            collection_name = self._get_qdrant_collection_name(lc)
            logger.info(f"Attempting to delete document {doc_id} from Qdrant collection: {collection_name}")
            
            deleted_in_collection = False
            for payload_field in possible_payload_fields_for_doc_id:
                try:
                    try:
                        self.qdrant_client.get_collection(collection_name=collection_name)
                    except Exception:
                        logger.warning(f"Collection {collection_name} does not exist. Skipping deletion for this language.")
                        deleted_in_collection = True
                        break 

                    points_to_delete_ids = []
                    offset = None
                    while True:
                        filter_condition = None
                        if "." in payload_field:
                            parent, child = payload_field.split(".",1)
                            filter_condition = rest.Filter(must=[
                                rest.FieldCondition(key=parent, match=rest.MatchAny(any=[{child: doc_id}]))
                            ])
                        else:
                            filter_condition = rest.Filter(must=[
                                rest.FieldCondition(key=payload_field, match=rest.MatchValue(value=doc_id))
                            ])

                        search_result, next_page_offset = self.qdrant_client.scroll(
                            collection_name=collection_name,
                            scroll_filter=filter_condition,
                            limit=250, 
                            offset=offset,
                            with_payload=False,
                            with_vectors=False
                        )
                        if not search_result:
                            break
                        points_to_delete_ids.extend([p.id for p in search_result])
                        if next_page_offset is None:
                            break
                        offset = next_page_offset

                    if points_to_delete_ids:
                        logger.info(f"Found {len(points_to_delete_ids)} points to delete for doc_id '{doc_id}' (field: {payload_field}) in {collection_name}.")
                        self.qdrant_client.delete_points(collection_name=collection_name, points_selector=points_to_delete_ids)
                        logger.info(f"Deletion request sent for {len(points_to_delete_ids)} points from {collection_name}.")
                        deleted_in_collection = True 
                        break
                    else:
                        logger.info(f"No points found for doc_id '{doc_id}' using field '{payload_field}' in {collection_name}.")

                except Exception as e:
                    logger.error(f"Error deleting points for doc_id '{doc_id}' from {collection_name} using field '{payload_field}': {e}", exc_info=True)
                    all_deleted_successfully = False
                    break
            
            if not deleted_in_collection:
                logger.warning(f"Document {doc_id} not found or not deleted in collection {collection_name} using any checked fields.")

        if all_deleted_successfully:
            logger.info(f"Document {doc_id} and its associated chunks successfully processed for deletion across specified languages.")
        else:
            logger.warning(f"Deletion process for document {doc_id} completed with some issues. Check logs.")
        
        target_langs_for_reinit = [lang_code] if lang_code else list(self.language_settings.keys())
        for lc_reinit in target_langs_for_reinit:
            if lc_reinit in self.language_settings:
                logger.info(f"Re-initializing language '{lc_reinit}' to update retrievers after deletion.")
                self._initialize_language(lc_reinit)

        return all_deleted_successfully

    def close(self):
        """Close connections and clean up resources."""
        if self.neo4j_driver:
            try:
                self.neo4j_driver.close()
                logger.info("Neo4j connection closed.")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")
        if self.qdrant_client:
            try:
                logger.info("Qdrant client operations finished. (No explicit close() usually needed for REST client)")
            except Exception as e:
                logger.error(f"Error with Qdrant client during close: {e}")
        logger.info("RAGSystem shutdown complete.")

# Example Usage (Illustrative - not for direct execution without setup)
if __name__ == '__main__':
    logger.info("Illustrative RAGSystem Usage Example")

    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    rag_system = RAGSystem(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        qdrant_url=QDRANT_URL,
        openai_api_key=OPENAI_API_KEY,
        default_lang="en",
        preload_languages=["en", "fr"],
        hybrid_search=True,
        hierarchical_retrieval=True,
        use_reranker=True
    )

    logger.info("Illustrative RAGSystem Usage Example")

    # Configuration (replace with your actual credentials and URLs)
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

    # Initialize RAG System
    rag_system = RAGSystem(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        qdrant_url=QDRANT_URL,
        openai_api_key=OPENAI_API_KEY,
        default_lang="en",
        preload_languages=["en", "fr"],
        hybrid_search=True,
        hierarchical_retrieval=True,
        use_reranker=True
    )

    # Example: Indexing some documents (assuming Document is LlamaIndex Document)
    # sample_doc_id = "doc_example_123"
    # sample_documents_en = [
    #     Document(text="This is a sample document about apples in English.", metadata={"source": "manual", "doc_id": sample_doc_id, "language": "en"}),
    #     Document(text="Apples are healthy and delicious.", metadata={"source": "manual", "doc_id": sample_doc_id, "language": "en"})
    # ]
    # sample_documents_fr = [
    #     Document(text="Ceci est un document exemple sur les pommes en français.", metadata={"source": "manual", "doc_id": sample_doc_id, "language": "fr"}),
    #     Document(text="Les pommes sont saines et délicieuses.", metadata={"source": "manual", "doc_id": sample_doc_id, "language": "fr"})
    # ]
    # rag_system.index_documents(sample_documents_en, doc_id=sample_doc_id, lang_code="en")
    # rag_system.index_documents(sample_documents_fr, doc_id=sample_doc_id, lang_code="fr")

    # Example: Retrieving documents
    # query_en = "tell me about apples"
    # results_en = rag_system.retrieve(query_en, top_k=3)
    # logger.info(f"Results for English query '{query_en}':")
    # for res in results_en:
    #     logger.info(f"  Score: {res['score']:.4f}, Chunk ID: {res['chunk_id']}, Text: {res['text'][:100]}...")
    #     logger.info(f"  Metadata: {res['metadata']}")

    # query_fr = "parle-moi des pommes"
    # results_fr = rag_system.retrieve(query_fr, top_k=3, use_cross_lingual=False)
    # logger.info(f"Results for French query '{query_fr}':")
    # for res in results_fr:
    #     logger.info(f"  Score: {res['score']:.4f}, Chunk ID: {res['chunk_id']}, Text: {res['text'][:100]}...")
    #     logger.info(f"  Metadata: {res['metadata']}")

    # Example: Deleting a document
    # if rag_system.delete_document(sample_doc_id):
    #     logger.info(f"Successfully deleted document {sample_doc_id}")
    # else:
    #     logger.warning(f"Could not fully delete document {sample_doc_id}")

    # Close connections
    rag_system.close() 