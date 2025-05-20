import logging
from typing import Any, Dict, List, Optional, Tuple
import time

from pydantic import BaseModel

from .base_node import BaseProcessingNode
from ..graph_components.nodes import QueryNode, DocumentNode, ConceptNode
from ..graph_components.edges import RetrievedForEdge, RelatedToEdge
from ..integration_components.graph_interface import GraphInterface
from ..integration_components.embedding_integrator import EmbeddingIntegrationLayer, HybridSearchResult

try:
    from llamaIndex_rag.rag import RAGSystem
    RAG_SYSTEM_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import RAGSystem. Vector search functionality will be limited.")
    RAG_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGRetrieverInput(BaseModel):
    current_query_node: QueryNode
    # Concepts relevant to the current query, identified earlier or by this node
    relevant_concepts: List[ConceptNode] = [] 
    # Parameters for retrieval, e.g., top_k
    retrieval_params: Dict[str, Any] = {"top_k": 5}

class RetrievedDocument(BaseModel):
    document_node: DocumentNode
    relevance_score: float
    retrieval_source: str # e.g., "vector_search", "graph_walk", "linked_concept"

class RAGRetrieverOutput(BaseModel):
    retrieved_documents: List[RetrievedDocument]
    # Any new statistics or insights gathered during retrieval
    retrieval_metadata: Dict[str, Any] = {}

# Alias to match what workflow_engine.py expects
RAGRetrievalOutput = RAGRetrieverOutput

class RAGRetrievalNode(BaseProcessingNode):
    """Retrieves relevant information from knowledge sources using RAG and graph."""

    def __init__(self, node_config: Optional[Dict[str, Any]] = None, existing_rag_system: Optional[Any] = None):
        super().__init__(node_config)
        self.rag_system = existing_rag_system
        # Configure default parameters
        config = node_config or {}
        self.top_k_vector = config.get("top_k_vector", 5)
        self.top_k_graph = config.get("top_k_graph", 5)
        self.top_k_final = config.get("top_k_final", 5)
        # Score weights for hybrid search
        self.score_weights = config.get("score_weights", {"vector": 0.6, "graph": 0.4})
        # Minimum scores for relevance filter
        self.min_relevance_score = config.get("min_relevance_score", 0.2)
        # Whether to use recent document boost
        self.use_recency_boost = config.get("use_recency_boost", True)
        # Whether to enable document diversity
        self.enable_diversity = config.get("enable_diversity", True)
        self.default_limit = config.get("default_limit", 10)
        self.context_expansion_enabled = config.get("context_expansion_enabled", True)

    async def execute(self, input_data, context: Dict[str, Any]) -> RAGRetrievalOutput:
        """
        Retrieves relevant documents based on the query node.
        
        Args:
            input_data: Contains the query node and search parameters.
            context: Workflow context with services.

        Returns:
            A RAGRetrievalOutput object with retrieved documents.
        """
        try:
            # Extract query from input_data
            if isinstance(input_data, dict):
                query_node = input_data.get("query_node")
                search_parameters = input_data.get("search_parameters", {})
            else:
                query_node = input_data.query_node
                search_parameters = input_data.search_parameters if hasattr(input_data, "search_parameters") else {}
            
            if not query_node:
                logger.error(f"[{self.get_name()}] No query node provided in input data")
                return RAGRetrievalOutput(retrieved_documents=[])
            
            # Get the query text from the node (prefer reformulated if available)
            if hasattr(query_node, "reformulated_query_text") and query_node.reformulated_query_text:
                query_text = query_node.reformulated_query_text
            elif hasattr(query_node, "query_text") and query_node.query_text:
                query_text = query_node.query_text
            else:
                query_text = str(query_node)  # Fallback
            
            logger.info(f"[{self.get_name()}] Starting retrieval for query: {query_text[:100]}...")
            
            # Get embedding service
            embedding_service = context.get("embedding_service")
            if not embedding_service:
                logger.error(f"[{self.get_name()}] No embedding service provided in context")
                return RAGRetrievalOutput(retrieved_documents=[])
            
            # Prepare search parameters with defaults
            limit = search_parameters.get("limit", self.default_limit)
            min_score = search_parameters.get("min_score", self.min_relevance_score)
            filter_criteria = search_parameters.get("filter_criteria", {})
            
            # Apply any query-based filters based on detected entities
            augmented_filters = await self._augment_filters_from_query(query_text, filter_criteria, context)

            retrieval_start = time.time()
            
            # Get document nodes from embedding service
            try:
                retrieved_docs = await embedding_service.query_documents(
                    query_text=query_text,
                    limit=limit,
                    min_score=min_score,
                    filters=augmented_filters
                )
                
                logger.info(f"[{self.get_name()}] Retrieved {len(retrieved_docs)} documents")
                
                # Extract text from documents if needed
                # Make sure we extract key segments or highlights when available
                for doc in retrieved_docs:
                    # If the document has highlight or key segments fields, ensure they're preserved
                    if hasattr(doc, 'highlights') and doc.highlights:
                        logger.info(f"[{self.get_name()}] Document {doc.id} has highlights")
                    elif hasattr(doc, 'key_segments') and doc.key_segments:
                        logger.info(f"[{self.get_name()}] Document {doc.id} has key segments")
                    # If no highlights, try to extract some based on query terms
                    else:
                        self._add_query_based_highlights(doc, query_text)
                
                # Optionally expand documents with relevant context if graph interface is available
                graph_interface = context.get("graph_interface")
                if graph_interface and self.context_expansion_enabled:
                    enhanced_docs = await self._expand_with_related_documents(
                        retrieved_docs, 
                        query_node, 
                        graph_interface
                    )
                    # Combine and deduplicate
                    all_docs = retrieved_docs + enhanced_docs
                    # Remove duplicates based on ID
                    seen_ids = set()
                    unique_docs = []
                    for doc in all_docs:
                        if doc.id not in seen_ids:
                            seen_ids.add(doc.id)
                            unique_docs.append(doc)
                    
                    retrieved_docs = unique_docs[:limit]  # Respect the original limit
                
                retrieval_time = time.time() - retrieval_start
                
                # Return the structured output
                return RAGRetrievalOutput(
                    retrieved_documents=retrieved_docs,
                    retrieval_metadata={
                        "retrieval_time": retrieval_time,
                        "num_documents": len(retrieved_docs),
                        "query_used": query_text,
                        "filters_applied": augmented_filters
                    }
                )
                
            except Exception as e:
                logger.error(f"[{self.get_name()}] Error during embedding search: {e}", exc_info=True)
                return RAGRetrievalOutput(
                    retrieved_documents=[],
                    retrieval_metadata={"error": str(e)}
                )
                
        except Exception as e:
            logger.error(f"[{self.get_name()}] Error in execution: {e}", exc_info=True)
            return RAGRetrievalOutput(retrieved_documents=[])

    def _add_query_based_highlights(self, doc, query_text):
        """
        Adds query-based highlights to a document by finding sentences containing query terms.
        
        Args:
            doc: The document to enhance with highlights
            query_text: The query text to find in the document
        """
        # Set up content field
        content_field = "text_content" if hasattr(doc, "text_content") else "content"
        doc_content = getattr(doc, content_field, "")
        
        if not doc_content:
            return
            
        # Extract key terms from query (filter out common words)
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "of", "is", "are"}
        query_terms = [term.lower() for term in query_text.split() if term.lower() not in common_words and len(term) > 3]
        
        # Find sentences containing query terms
        import re
        sentences = re.split(r'(?<=[.!?])\s+', doc_content)
        highlights = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains any query term
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                highlights.append(sentence)
                
            # Limit to 5 highlights
            if len(highlights) >= 5:
                break
                
        # If we found highlights, add them to the document
        if highlights:
            # Add as a new attribute
            setattr(doc, "highlights", highlights)
            # Also set a highlights_text field for easy access
            setattr(doc, "highlights_text", "\n".join(highlights))
            logger.info(f"[{self.get_name()}] Added {len(highlights)} query-based highlights to document {doc.id}")
        
    async def _extract_query_info_with_llm(self, query_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses LLM to extract structured information from the query text.
        
        Args:
            query_text: The user query text
            context: Workflow context with services
            
        Returns:
            Dictionary with extracted information including entities, intent, filters, etc.
        """
        # Try different LLM services that might be in the context
        llm_service = context.get("llm_client") or context.get("llm_service")
        
        if not llm_service:
            logger.warning(f"[{self.get_name()}] No LLM service found in context, skipping LLM extraction")
            return {}
            
        try:
            # Define the extraction prompt
            extraction_prompt = f"""
            Analyze the following user query and extract key information in JSON format.
            
            Query: "{query_text}"
            
            Extract the following information:
            - main_topic: The primary subject of the query
            - entities: All named entities (people, organizations, products, etc.)
            - time_references: Any dates, time periods, or temporal references
            - document_types: Types of documents being requested (reports, policies, emails, etc.)
            - filters: Additional filtering criteria (categories, departments, etc.)
            - sorting_preferences: How results should be sorted (recency, relevance, etc.)
            - specific_fields: Fields the user wants to see in the results
            
            Format your response as a valid JSON object with these fields. Include only fields that are actually present in the query.
            Your response should be valid JSON only, with no additional explanatory text.
            """
            
            # Call LLM to extract info - handle different types of LLM clients
            response = None
            
            # Try the standard generate_text method first
            if hasattr(llm_service, 'generate_text'):
                logger.info(f"[{self.get_name()}] Using generate_text method for query extraction")
                response = await llm_service.generate_text(
                    prompt=extraction_prompt,
                    max_tokens=1024,
                    temperature=0.1
                )
            # If not available, try OpenAI-style interfaces
            elif hasattr(llm_service, 'chat') and hasattr(llm_service.chat, 'completions') and hasattr(llm_service.chat.completions, 'create'):
                # Modern OpenAI client
                logger.info(f"[{self.get_name()}] Using OpenAI chat completions for query extraction")
                try:
                    # Check if the create method is coroutine function (async)
                    import inspect
                    create_method = llm_service.chat.completions.create
                    if inspect.iscoroutinefunction(create_method):
                        # Use await for async function
                        completion = await create_method(
                            model="gpt-4",  # Use an appropriate model
                            messages=[
                                {"role": "system", "content": "You extract structured information from queries in JSON format."},
                                {"role": "user", "content": extraction_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1024
                        )
                    else:
                        # Call synchronously
                        completion = create_method(
                            model="gpt-4",  # Use an appropriate model
                            messages=[
                                {"role": "system", "content": "You extract structured information from queries in JSON format."},
                                {"role": "user", "content": extraction_prompt}
                            ],
                            temperature=0.1,
                            max_tokens=1024
                        )
                    
                    # Handle different response formats
                    if hasattr(completion, 'choices') and hasattr(completion.choices[0], 'message'):
                        response = completion.choices[0].message.content
                    elif isinstance(completion, dict) and 'choices' in completion:
                        response = completion['choices'][0]['message']['content']
                    else:
                        logger.error(f"[{self.get_name()}] Unknown response format from OpenAI")
                        response = ""
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error using OpenAI client: {e}", exc_info=True)
                    response = ""
            # Legacy OpenAI Completion API
            elif hasattr(llm_service, 'Completion') and hasattr(llm_service.Completion, 'create'):
                logger.info(f"[{self.get_name()}] Using legacy OpenAI Completion API for query extraction")
                try:
                    # Check if the create method is coroutine function (async)
                    import inspect
                    create_method = llm_service.Completion.create
                    if inspect.iscoroutinefunction(create_method):
                        # Use await for async function
                        completion = await create_method(
                            engine="text-davinci-003",  # Or appropriate model
                            prompt=extraction_prompt,
                            temperature=0.1,
                            max_tokens=1024
                        )
                    else:
                        # Call synchronously
                        completion = create_method(
                            engine="text-davinci-003",  # Or appropriate model
                            prompt=extraction_prompt,
                            temperature=0.1,
                            max_tokens=1024
                        )
                    
                    # Handle different response formats
                    if hasattr(completion, 'choices') and hasattr(completion.choices[0], 'text'):
                        response = completion.choices[0].text
                    elif isinstance(completion, dict) and 'choices' in completion:
                        response = completion['choices'][0]['text']
                    else:
                        logger.error(f"[{self.get_name()}] Unknown response format from OpenAI Completion")
                        response = ""
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error using OpenAI Completion client: {e}", exc_info=True)
                    response = ""
                
            if not response:
                logger.error(f"[{self.get_name()}] No compatible method found to call LLM service")
                return {}
            
            # Parse the JSON response
            import json
            try:
                # Extract JSON structure from potential text wrapper
                json_str = response
                # If the response contains markdown code blocks, extract just the JSON part
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                
                # Clean up potential line prefixes (like "> " in markdown quotes)
                clean_lines = []
                for line in json_str.split("\n"):
                    clean_lines.append(line.lstrip("> "))
                json_str = "\n".join(clean_lines)
                
                # Try to find JSON boundaries if the response contains text around the JSON
                if not json_str.strip().startswith("{"):
                    # Try to find the JSON part
                    start_idx = json_str.find("{")
                    end_idx = json_str.rfind("}")
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = json_str[start_idx:end_idx+1]
                    else:
                        raise ValueError("No JSON object found in response")
                
                extracted_info = json.loads(json_str)
                logger.info(f"[{self.get_name()}] Successfully extracted structured info from query using LLM")
                return extracted_info
            except json.JSONDecodeError as e:
                logger.error(f"[{self.get_name()}] Failed to parse LLM response as JSON: {e}")
                logger.debug(f"[{self.get_name()}] Raw LLM response: {response}")
                return {}
                
        except Exception as e:
            logger.error(f"[{self.get_name()}] Error extracting info with LLM: {e}", exc_info=True)
            return {}

    async def _augment_filters_from_query(self, query_text, filter_criteria, context: Dict[str, Any] = None):
        """
        Analyzes query text to extract entities and create additional filter criteria.
        Uses both rule-based methods and LLM extraction when available.
        
        Args:
            query_text: The user query text
            filter_criteria: Existing filter criteria
            context: Workflow context with services
            
        Returns:
            Enhanced filter criteria with detected entities
        """
        # Start with any existing filter criteria
        augmented_filters = filter_criteria.copy() if filter_criteria else {}
        
        # First try LLM-based extraction if context is provided
        if context:
            llm_extracted_info = await self._extract_query_info_with_llm(query_text, context)
            
            if llm_extracted_info:
                # Process time references
                if 'time_references' in llm_extracted_info and llm_extracted_info['time_references']:
                    time_refs = llm_extracted_info['time_references']
                    if isinstance(time_refs, list) and time_refs:
                        augmented_filters['date_range'] = {'contains': time_refs[0]}
                    elif isinstance(time_refs, str):
                        augmented_filters['date_range'] = {'contains': time_refs}
                    elif isinstance(time_refs, dict):
                        # Assume the LLM returned a structured date range with start/end
                        augmented_filters['date_range'] = time_refs
                
                # Process document types
                if 'document_types' in llm_extracted_info and llm_extracted_info['document_types']:
                    doc_types = llm_extracted_info['document_types']
                    if isinstance(doc_types, list) and doc_types:
                        augmented_filters['document_type'] = doc_types[0]
                    elif isinstance(doc_types, str):
                        augmented_filters['document_type'] = doc_types
                
                # Process entities for organization/author filters
                if 'entities' in llm_extracted_info and llm_extracted_info['entities']:
                    entities = llm_extracted_info['entities']
                    if isinstance(entities, dict):
                        # Structured entities by type
                        if 'organizations' in entities and entities['organizations']:
                            orgs = entities['organizations']
                            if isinstance(orgs, list) and orgs:
                                augmented_filters['organization'] = orgs[0]
                            elif isinstance(orgs, str):
                                augmented_filters['organization'] = orgs
                                
                        if 'people' in entities and entities['people']:
                            people = entities['people']
                            if isinstance(people, list) and people:
                                augmented_filters['author'] = people[0]
                            elif isinstance(people, str):
                                augmented_filters['author'] = people
                    elif isinstance(entities, list) and entities:
                        # Just a list of entities, use the first one as a general entity filter
                        augmented_filters['entity'] = entities[0]
                
                # Process sorting preferences
                if 'sorting_preferences' in llm_extracted_info and llm_extracted_info['sorting_preferences']:
                    sorting = llm_extracted_info['sorting_preferences']
                    if isinstance(sorting, str) and 'recent' in sorting.lower():
                        augmented_filters['recency_boost'] = True
                    elif isinstance(sorting, list) and any('recent' in s.lower() for s in sorting if isinstance(s, str)):
                        augmented_filters['recency_boost'] = True
                        
                # Process any additional filters the LLM extracted
                if 'filters' in llm_extracted_info and llm_extracted_info['filters']:
                    filters = llm_extracted_info['filters']
                    if isinstance(filters, dict):
                        # Merge any additional filters directly
                        for key, value in filters.items():
                            if key not in augmented_filters:
                                augmented_filters[key] = value
                    elif isinstance(filters, list) and filters:
                        # For list of filters, use as tags
                        augmented_filters['tags'] = filters
                
                logger.info(f"[{self.get_name()}] Added filters from LLM extraction: {augmented_filters}")
        
        # Fall back to rule-based extraction or supplement LLM extraction with rules
        try:
            # Extract dates if not already captured by LLM
            if 'date_range' not in augmented_filters:
                import re
                # Match dates in various formats (e.g., 2023-01-01, Jan 2023, January 2023)
                date_patterns = [
                    r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # ISO format: 2023-01-01
                    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # US/EU format: 01/01/2023 or 01/01/23
                    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month name: January 1, 2023
                    r'\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # EU format: 1 January 2023
                    r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b',  # Month and year: January 2023
                    r'\b\d{4}\b'  # Just year: 2023
                ]
                
                found_dates = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, query_text, re.IGNORECASE)
                    if matches:
                        found_dates.extend(matches)
                
                if found_dates:
                    logger.info(f"[{self.get_name()}] Found date references in query: {found_dates}")
                    augmented_filters['date_range'] = {'contains': found_dates[0]}
            
            # Document type detection if not already captured by LLM
            if 'document_type' not in augmented_filters:
                doc_type_keywords = {
                    'report': ['report', 'analysis', 'overview', 'study'],
                    'policy': ['policy', 'guideline', 'procedure', 'protocol'],
                    'legal': ['law', 'regulation', 'legal', 'compliance', 'statute'],
                    'correspondence': ['letter', 'email', 'message', 'communication'],
                    'financial': ['financial', 'budget', 'expense', 'cost', 'revenue']
                }
                
                # Check if query contains document type keywords
                query_lower = query_text.lower()
                for doc_type, keywords in doc_type_keywords.items():
                    if any(keyword in query_lower for keyword in keywords):
                        logger.info(f"[{self.get_name()}] Detected document type: {doc_type}")
                        augmented_filters['document_type'] = doc_type
                        break
            
            # Detect if the query is asking for recent documents if not already captured
            if 'recency_boost' not in augmented_filters:
                recency_keywords = ['recent', 'latest', 'newest', 'current', 'last']
                query_lower = query_text.lower()
                if any(keyword in query_lower for keyword in recency_keywords):
                    logger.info(f"[{self.get_name()}] Query indicates preference for recent documents")
                    augmented_filters['recency_boost'] = True
            
        except Exception as e:
            logger.error(f"[{self.get_name()}] Error in rule-based filter augmentation: {e}", exc_info=True)
        
        return augmented_filters

    async def _expand_with_related_documents(self, retrieved_docs, query_node, graph_interface):
        """
        Expands the set of retrieved documents with related documents from the graph.
        
        This enriches the search results by:
        1. Finding parent/child relationships of already retrieved documents
        2. Finding semantically related documents through concept connections
        3. Including documents written by same authors or from same sources
        
        Args:
            retrieved_docs: List of initially retrieved document nodes
            query_node: The query node for context 
            graph_interface: Interface to the knowledge graph
            
        Returns:
            Additional document nodes to consider for retrieval
        """
        if not retrieved_docs or not graph_interface:
            return []
            
        additional_docs = []
        seen_doc_ids = {doc.id for doc in retrieved_docs}
        
        try:
            # 1. Find parent/child documents for existing results
            for doc in retrieved_docs[:5]:  # Limit to top 5 to avoid too many graph queries
                try:
                    # Get parent documents
                    parent_docs = await graph_interface.get_parent_documents(doc.id)
                    for parent_doc in parent_docs:
                        if parent_doc.id not in seen_doc_ids:
                            seen_doc_ids.add(parent_doc.id)
                            additional_docs.append(parent_doc)
                            
                    # Get child documents
                    child_docs = await graph_interface.get_child_documents(doc.id)
                    for child_doc in child_docs:
                        if child_doc.id not in seen_doc_ids:
                            seen_doc_ids.add(child_doc.id)
                            additional_docs.append(child_doc)
                            
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error getting related docs for {doc.id}: {e}", exc_info=True)
                    continue
            
            # 2. Find documents related through shared concepts
            # Extract concepts from the query node if available
            relevant_concept_ids = []
            if hasattr(query_node, 'related_concept_ids') and query_node.related_concept_ids:
                relevant_concept_ids = query_node.related_concept_ids
                
            if relevant_concept_ids:
                for concept_id in relevant_concept_ids[:3]:  # Limit to top 3 concepts
                    try:
                        # Get documents related to this concept
                        concept_docs = await graph_interface.get_documents_by_concept(concept_id)
                        for doc in concept_docs:
                            if doc.id not in seen_doc_ids:
                                seen_doc_ids.add(doc.id)
                                additional_docs.append(doc)
                    except Exception as e:
                        logger.error(f"[{self.get_name()}] Error getting docs for concept {concept_id}: {e}", exc_info=True)
                        continue
            
            # 3. Find documents from the same sources/authors
            sources = set()
            authors = set()
            
            # Extract sources and authors from retrieved docs
            for doc in retrieved_docs:
                if hasattr(doc, 'source') and doc.source:
                    sources.add(doc.source)
                if hasattr(doc, 'author') and doc.author:
                    authors.add(doc.author)
            
            # Get additional documents by source
            if sources:
                for source in list(sources)[:2]:  # Limit to 2 sources
                    try:
                        source_docs = await graph_interface.get_documents_by_source(source)
                        for doc in source_docs:
                            if doc.id not in seen_doc_ids:
                                seen_doc_ids.add(doc.id)
                                additional_docs.append(doc)
                    except Exception as e:
                        logger.error(f"[{self.get_name()}] Error getting docs for source {source}: {e}", exc_info=True)
                        continue
            
            # Get additional documents by author
            if authors:
                for author in list(authors)[:2]:  # Limit to 2 authors
                    try:
                        author_docs = await graph_interface.get_documents_by_author(author)
                        for doc in author_docs:
                            if doc.id not in seen_doc_ids:
                                seen_doc_ids.add(doc.id)
                                additional_docs.append(doc)
                    except Exception as e:
                        logger.error(f"[{self.get_name()}] Error getting docs for author {author}: {e}", exc_info=True)
                        continue
            
            logger.info(f"[{self.get_name()}] Found {len(additional_docs)} additional related documents")
            return additional_docs
            
        except Exception as e:
            logger.error(f"[{self.get_name()}] Error expanding related documents: {e}", exc_info=True)
            return []
