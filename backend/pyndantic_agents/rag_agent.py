"""
RAG Agent for understanding user requests and retrieving relevant context.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from openai import AsyncOpenAI
import os
import uuid
import time

from .base_agent import BaseAgent, AgentInput, AgentOutput
from llamaIndex_rag.rag import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryUnderstandingOutput(AgentOutput):
    """Enhanced output for query understanding"""
    query_type: str = "general"
    extracted_entities: Optional[List[Dict[str, Any]]] = None
    reformulated_query: Optional[str] = None

class RAGAgent(BaseAgent):
    """
    Agent that understands user requests and searches for relevant context.
    Leverages LlamaIndex RAG system to retrieve information.
    """

    def __init__(
        self,
        rag_system: RAGSystem,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        use_hyde: bool = True,
        hyde_top_k: int = 3,
        **kwargs
    ):
        """
        Initialize the RAG agent.

        Args:
            rag_system: Instance of the RAGSystem
            openai_api_key: OpenAI API key for LLM calls
            model: LLM model to use for understanding and HyDE
            use_hyde: Whether to use Hypothetical Document Embeddings
            hyde_top_k: Number of documents to retrieve in the HyDE pass
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.rag_system = rag_system
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.use_hyde = use_hyde
        self.hyde_top_k = hyde_top_k
        logger.info(f"Initialized RAG agent with model {self.model}, HyDE enabled: {self.use_hyde}")

    async def understand_query(self, query: str) -> Dict[str, Any]:
        """
        Use the LLM to understand the query, extract entities, and determine intent.

        Args:
            query: The user query

        Returns:
            Dictionary with query understanding data
        """
        system_prompt = """You are an AI assistant that helps understand user queries about governance, risk, and compliance (GRC).
Your task is to:
1. Identify the type of query (e.g., regulatory question, risk assessment, compliance check, general information)
2. Extract key entities mentioned (regulations, companies, risks, processes, etc.)
3. Reformulate the query to optimize for information retrieval if needed
4. Determine which specific domain this falls under (regulatory compliance, risk management, governance, cybersecurity, etc.)

Respond ONLY with a JSON object with the following structure:
{
  "query_type": "string",
  "domain": "string",
  "entities": [{"name": "string", "type": "string"}],
  "reformulated_query": "string",
  "reasoning": "string"
}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            understanding = json.loads(content)
            logger.info(f"Query understanding: {understanding}")
            return understanding

        except Exception as e:
            logger.error(f"Error in query understanding: {str(e)}")
            # Return a basic fallback structure
            return {
                "query_type": "general",
                "domain": "unknown",
                "entities": [],
                "reformulated_query": query,
                "reasoning": "Error in query understanding"
            }

    async def _generate_hypothetical_document(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical document for a given query using an LLM.
        """
        system_prompt = "You are an AI assistant. Based on the following user query, generate a concise, factual document that you believe would be a perfect answer to the query. Focus on providing key information and entities relevant to the query. Do not include any conversational fluff or introductory/concluding remarks. Just the document text."
        
        try:
            logger.info(f"[RAGAgent_HyDE] Generating hypothetical document for query: '{query[:100]}...'")
            response = await self.client.chat.completions.create(
                model=self.model, # Use the agent's configured model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.2, # Lower temperature for more factual generation
                max_tokens=500 # Limit the length of the hypothetical document
            )
            hypothetical_doc = response.choices[0].message.content
            if hypothetical_doc:
                logger.info(f"[RAGAgent_HyDE] Generated hypothetical document (first 100 chars): '{hypothetical_doc[:100]}...'")
                return hypothetical_doc.strip()
            else:
                logger.warning("[RAGAgent_HyDE] Hypothetical document generation returned empty content.")
                return None
        except Exception as e:
            logger.error(f"[RAGAgent_HyDE] Error generating hypothetical document: {str(e)}")
            return None

    async def retrieve_context(self, query: str, understanding: Dict[str, Any], top_k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None, use_neo4j: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on the query and understanding, potentially using HyDE.
        Loops with varying strategies until context is found or a maximum attempt limit is reached.

        Args:
            query: Original user query
            understanding: Query understanding from understand_query
            top_k: Number of results to retrieve
            filter_criteria: Optional criteria to filter results (metadata filtering for RAGSystem)
            use_neo4j: Whether to use Neo4j for additional context retrieval by RAGSystem (for main passes)

        Returns:
            List of context items
        """
        try:
            retrieval_id_base = str(uuid.uuid4())[:8]
            max_attempts = 5  # Max retry attempts
            final_results: List[Dict[str, Any]] = [] # Store results from a successful attempt

            for attempt in range(1, max_attempts + 1):
                current_retrieval_id = f"{retrieval_id_base}_attempt_{attempt}"
                logger.info(f"[RAGAgent:{current_retrieval_id}] Starting context retrieval for query: '{query}' (Attempt {attempt}/{max_attempts})")

                if attempt > 1:
                    await asyncio.sleep(1 * attempt) # Exponential backoff for sleep, 1s, 2s, 3s...
                    logger.info(f"[RAGAgent:{current_retrieval_id}] Retrying context retrieval. Previous attempt yielded no results.")

                # Determine search query for this attempt
                current_search_query = query # Default to original query
                if attempt == 1:
                    reformulated_q = understanding.get("reformulated_query")
                    if reformulated_q and reformulated_q != query:
                        current_search_query = reformulated_q
                        logger.info(f"[RAGAgent:{current_retrieval_id}] Using reformulated query for attempt 1: '{current_search_query}'")
                    else:
                        logger.info(f"[RAGAgent:{current_retrieval_id}] Using original query for attempt 1: '{current_search_query}'")
                else: # For attempts > 1, always use original query or try other variations if implemented
                    logger.info(f"[RAGAgent:{current_retrieval_id}] Using original query for attempt {attempt}: '{current_search_query}'")
                
                entities = understanding.get("entities", [])
                attempt_results: List[Dict[str, Any]] = []
                processed_chunk_ids_for_attempt: set[str] = set()

                # 1. HyDE Pass (if enabled) for current_search_query
                if self.use_hyde:
                    hypothetical_document = await self._generate_hypothetical_document(current_search_query)
                    if hypothetical_document:
                        logger.info(f"[RAGAgent:{current_retrieval_id}_HyDE] Retrieving context with HyDE query (hypothetical doc), filter: {filter_criteria}, use_neo4j: {use_neo4j}, top_k: {self.hyde_top_k}")
                        hyde_start_time = time.time()
                        hyde_nodes = self.rag_system.retrieve(
                            query=hypothetical_document,
                            top_k=self.hyde_top_k,
                            filter_criteria=filter_criteria,
                            use_neo4j=use_neo4j # Use the passed-in use_neo4j for main passes
                        )
                        hyde_retrieval_time = time.time() - hyde_start_time
                        logger.info(f"[RAGAgent:{current_retrieval_id}_HyDE] Retrieved {len(hyde_nodes)} context items using HyDE in {hyde_retrieval_time:.2f}s")
                        for node in hyde_nodes:
                            chunk_id = node.get("metadata", {}).get("chunk_id")
                            if chunk_id and chunk_id not in processed_chunk_ids_for_attempt:
                                node["retrieval_method"] = "hyde"
                                attempt_results.append(node)
                                processed_chunk_ids_for_attempt.add(chunk_id)
                    else:
                        logger.warning(f"[RAGAgent:{current_retrieval_id}_HyDE] Skipping HyDE pass as hypothetical document generation failed.")

                # 2. Standard Retrieval Pass for current_search_query
                logger.info(f"[RAGAgent:{current_retrieval_id}] Retrieving context with query: '{current_search_query}', filter: {filter_criteria}, use_neo4j: {use_neo4j}, top_k: {top_k}")
                standard_start_time = time.time()
                standard_nodes = self.rag_system.retrieve(
                    query=current_search_query,
                    top_k=top_k,
                    filter_criteria=filter_criteria,
                    use_neo4j=use_neo4j # Use the passed-in use_neo4j for main passes
                )
                standard_retrieval_time = time.time() - standard_start_time
                logger.info(f"[RAGAgent:{current_retrieval_id}] Retrieved {len(standard_nodes)} context items using standard query in {standard_retrieval_time:.2f}s")
                for node in standard_nodes:
                    chunk_id = node.get("metadata", {}).get("chunk_id")
                    if chunk_id and chunk_id not in processed_chunk_ids_for_attempt:
                        node["retrieval_method"] = "standard"
                        attempt_results.append(node)
                        processed_chunk_ids_for_attempt.add(chunk_id)
                
                logger.info(f"[RAGAgent:{current_retrieval_id}] After HyDE and Standard passes, found {len(attempt_results)} unique items for this attempt.")

                # 3. Enhanced Entity-Focused Search / Fallback if results are poor or entities missed
                missing_entities_names: set[str] = set()
                if entities:
                    entity_names_lower = {e.get("name", "").lower() for e in entities if e.get("name")}
                    found_entities_in_attempt: set[str] = set()
                    for item in attempt_results:
                        text_lower = item.get("text", "").lower()
                        for entity_name_l in entity_names_lower:
                            if entity_name_l in text_lower:
                                found_entities_in_attempt.add(entity_name_l)
                    missing_entities_names = entity_names_lower - found_entities_in_attempt
                    if missing_entities_names:
                        logger.warning(f"[RAGAgent:{current_retrieval_id}] Missing key entities in attempt's results: {missing_entities_names}")
                
                needs_fallback_search = (not attempt_results) or missing_entities_names

                if needs_fallback_search:
                    logger.info(f"[RAGAgent:{current_retrieval_id}] Attempting enhanced fallback search (empty results or missing entities).")
                    
                    fallback_query = " ".join(missing_entities_names) if missing_entities_names else current_search_query
                    
                    if fallback_query: # Ensure there's something to query
                        logger.info(f"[RAGAgent:{current_retrieval_id}] Fallback RAG call with query: '{fallback_query}' and forcing Neo4j, top_k={top_k}.")
                        try:
                            fallback_nodes = self.rag_system.retrieve(
                                query=fallback_query,
                                top_k=top_k, # Use main top_k for fallback, or a specific one like min(3 * len(missing_entities_names), top_k)
                                filter_criteria=filter_criteria,
                                use_neo4j=True  # Force Neo4j usage for this fallback call
                            )
                            if fallback_nodes:
                                new_items_from_fallback_count = 0
                                for node in fallback_nodes:
                                    chunk_id = node.get("metadata", {}).get("chunk_id")
                                    if chunk_id and chunk_id not in processed_chunk_ids_for_attempt:
                                        node["retrieval_method"] = "fallback_neo4j"
                                        attempt_results.append(node)
                                        processed_chunk_ids_for_attempt.add(chunk_id)
                                        new_items_from_fallback_count +=1
                                if new_items_from_fallback_count > 0:
                                    logger.info(f"[RAGAgent:{current_retrieval_id}] Added {new_items_from_fallback_count} unique items from fallback search.")
                                else:
                                    logger.info(f"[RAGAgent:{current_retrieval_id}] Fallback search found items, but all were duplicates of existing attempt results.")
                            else:
                                logger.info(f"[RAGAgent:{current_retrieval_id}] Fallback RAG call found no items.")
                        except Exception as e_fallback:
                            logger.error(f"[RAGAgent:{current_retrieval_id}] Error in fallback search: {str(e_fallback)}")
                    else:
                         logger.warning(f"[RAGAgent:{current_retrieval_id}] Fallback search skipped as fallback_query was empty (e.g. no missing entities and current_search_query was empty).")
                
                if attempt_results:
                    final_results = attempt_results
                    logger.info(f"[RAGAgent:{current_retrieval_id}] Attempt {attempt} successful. Found {len(final_results)} context items. Proceeding with these results.")
                    break # Exit the retry loop as results have been found

                logger.warning(f"[RAGAgent:{current_retrieval_id}] Attempt {attempt} yielded no results even after fallback. Retrying if attempts remain.")
            
            # After the loop (either break due to results, or max_attempts reached)
            if not final_results:
                logger.error(f"[RAGAgent:{retrieval_id_base}] All {max_attempts} attempts failed. No context found for query: '{query}'. Returning empty list.")
                return []

            # Log document sources and completion message
            doc_sources = {}
            for res_item in final_results:
                doc_id = res_item.get("metadata", {}).get("doc_id", "unknown")
                doc_sources[doc_id] = doc_sources.get(doc_id, 0) + 1
            
            final_retrieval_id_log = f"{retrieval_id_base}_attempt_{attempt}" if final_results else retrieval_id_base
            logger.info(f"[RAGAgent:{final_retrieval_id_log}] Context sources (doc_id: count): {doc_sources}")
            logger.info(f"[RAGAgent:{final_retrieval_id_log}] Context retrieval completed with {len(final_results)} items after {attempt} attempt(s).")

            return final_results

        except Exception as e:
            log_id_on_exception = retrieval_id_base if 'retrieval_id_base' in locals() else "UNKNOWN_ID"
            logger.error(f"[RAGAgent:{log_id_on_exception}] Critical error during context retrieval process: {str(e)}")
            return []

    async def process(self, input_data: AgentInput) -> QueryUnderstandingOutput:
        """
        Process the input query, understand it, and retrieve relevant context.

        Args:
            input_data: Agent input data

        Returns:
            QueryUnderstandingOutput with the response and retrieved context
        """
        self._log_processing(input_data)
        query = input_data.query

        try:
            # Step 1: Understand the query
            understanding = await self.understand_query(query)

            # Step 2: Retrieve relevant context
            # Extract top_k, filter_criteria, and use_neo4j from input_data.parameters if available, otherwise use defaults
            agent_params = input_data.parameters if input_data.parameters else {}
            top_k_param = agent_params.get("top_k", 5)
            # Assuming filter_criteria and use_neo4j might also be in parameters,
            # otherwise using defaults set in retrieve_context method signature.
            filter_criteria_param = agent_params.get("filter_criteria") # Will be None if not present
            use_neo4j_param = agent_params.get("use_neo4j") # Will be None if not present

            context_results = await self.retrieve_context(
                query=query,
                understanding=understanding,
                top_k=top_k_param,
                filter_criteria=filter_criteria_param, # Pass along, will default to None in method if still None
                use_neo4j=use_neo4j_param if use_neo4j_param is not None else True # Default to True if not specified
            )

            # Step 3: Prepare response
            return QueryUnderstandingOutput(
                response=f"I understand you're asking about {understanding.get('domain', 'this topic')}. I found {len(context_results)} relevant sources.",
                context_used=context_results,
                confidence=0.8 if context_results else 0.4,
                reasoning=understanding.get("reasoning", ""),
                query_type=understanding.get("query_type", "general"),
                extracted_entities=understanding.get("entities", []),
                reformulated_query=understanding.get("reformulated_query", None),
                additional_data={"understanding": understanding}
            )

        except Exception as e:
            logger.error(f"Error processing in RAG agent: {str(e)}")
            return QueryUnderstandingOutput(
                response="I encountered an error while processing your request.",
                context_used=[],
                confidence=0.1,
                reasoning="Error during processing",
                query_type="error"
            )
