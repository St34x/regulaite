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
        **kwargs
    ):
        """
        Initialize the RAG agent.

        Args:
            rag_system: Instance of the RAGSystem
            openai_api_key: OpenAI API key for LLM calls
            model: LLM model to use
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.rag_system = rag_system
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        logger.info(f"Initialized RAG agent with model {self.model}")

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

    async def retrieve_context(self, query: str, understanding: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on the query and understanding.

        Args:
            query: Original user query
            understanding: Query understanding from understand_query
            top_k: Number of results to retrieve

        Returns:
            List of context items
        """
        try:
            # Generate a unique ID for this retrieval operation
            retrieval_id = str(uuid.uuid4())[:8]
            logger.info(f"[RAGAgent:{retrieval_id}] Starting context retrieval for query: '{query}'")
            logger.info(f"[RAGAgent:{retrieval_id}] Query understanding: {understanding}")

            # Use the reformulated query if available
            search_query = understanding.get("reformulated_query", query)
            if search_query != query:
                logger.info(f"[RAGAgent:{retrieval_id}] Using reformulated query: '{search_query}' (original: '{query}')")

            # Build filter criteria based on extracted entities
            filter_criteria = None
            entities = understanding.get("entities", [])

            if entities:
                filter_criteria = {}
                logger.info(f"[RAGAgent:{retrieval_id}] Extracted entities: {entities}")

                for entity in entities:
                    entity_type = entity.get("type", "").lower()
                    entity_name = entity.get("name", "")

                    # Map entity types to metadata fields
                    if entity_type in ["regulation", "standard", "law", "directive"]:
                        if "doc_name" not in filter_criteria:
                            filter_criteria["doc_name"] = []
                        filter_criteria["doc_name"].append(entity_name)
                        logger.info(f"[RAGAgent:{retrieval_id}] Added filter for {entity_type}: {entity_name}")

                # Log the constructed filter criteria
                if filter_criteria:
                    logger.info(f"[RAGAgent:{retrieval_id}] Using filter criteria: {filter_criteria}")
                else:
                    logger.info(f"[RAGAgent:{retrieval_id}] No applicable filter criteria created from entities")

            # Determine if we need to use Neo4j based on query complexity
            use_neo4j = True
            query_complexity = understanding.get("query_type", "").lower()

            # For simple factual queries, vector search might be sufficient
            if query_complexity in ["simple", "factual"]:
                use_neo4j = False
                logger.info(f"[RAGAgent:{retrieval_id}] Simple/factual query detected, using only vector search")
            else:
                logger.info(f"[RAGAgent:{retrieval_id}] Complex query detected ({query_complexity}), will use Neo4j if needed")

            # Retrieve context
            logger.info(f"[RAGAgent:{retrieval_id}] Retrieving context with query: '{search_query}', filter: {filter_criteria}, use_neo4j: {use_neo4j}")

            start_time = time.time()
            results = self.rag_system.retrieve(
                query=search_query,
                top_k=top_k,
                filter_criteria=filter_criteria,
                use_neo4j=use_neo4j
            )
            retrieval_time = time.time() - start_time

            logger.info(f"[RAGAgent:{retrieval_id}] Retrieved {len(results)} context items in {retrieval_time:.2f}s")

            # Evaluate context quality
            if len(results) == 0:
                logger.warning(f"[RAGAgent:{retrieval_id}] No context found! This may result in hallucinations.")
            elif len(results) < top_k * 0.5:
                logger.warning(f"[RAGAgent:{retrieval_id}] Limited context ({len(results)} < {top_k/2}) found. May need to expand search.")

            # Check if results cover all key entities
            if entities:
                entity_names = [e.get("name", "").lower() for e in entities]
                found_entities = set()

                for result in results:
                    text = result.get("text", "").lower()
                    for entity in entity_names:
                        if entity and entity in text:
                            found_entities.add(entity)

                missing_entities = set(entity_names) - found_entities
                if missing_entities:
                    logger.warning(f"[RAGAgent:{retrieval_id}] Missing key entities in results: {missing_entities}")

                    # Try to find more context specifically for these entities
                    if use_neo4j == False:
                        logger.info(f"[RAGAgent:{retrieval_id}] Expanding search to include Neo4j to find missing entities")
                        try:
                            # Second retrieval attempt with Neo4j enabled
                            additional_results = self.rag_system.retrieve(
                                query=" ".join(missing_entities),  # Use missing entities as query
                                top_k=min(3, len(missing_entities)),
                                filter_criteria=filter_criteria,
                                use_neo4j=True  # Force Neo4j usage
                            )

                            if additional_results:
                                logger.info(f"[RAGAgent:{retrieval_id}] Found {len(additional_results)} additional context items for missing entities")

                                # Add non-duplicate results
                                existing_ids = set(r.get("metadata", {}).get("chunk_id", "") for r in results)
                                new_results = [r for r in additional_results if r.get("metadata", {}).get("chunk_id", "") not in existing_ids]

                                if new_results:
                                    logger.info(f"[RAGAgent:{retrieval_id}] Adding {len(new_results)} unique additional context items")
                                    results.extend(new_results)
                        except Exception as e:
                            logger.error(f"[RAGAgent:{retrieval_id}] Error in additional entity search: {str(e)}")

            # Log document sources in results
            doc_sources = {}
            for result in results:
                doc_id = result.get("metadata", {}).get("doc_id", "unknown")
                doc_sources[doc_id] = doc_sources.get(doc_id, 0) + 1

            logger.info(f"[RAGAgent:{retrieval_id}] Context sources (doc_id: count): {doc_sources}")
            logger.info(f"[RAGAgent:{retrieval_id}] Context retrieval completed")

            return results

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
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
            context_results = await self.retrieve_context(
                query=query,
                understanding=understanding,
                top_k=input_data.parameters.get("top_k", 5) if input_data.parameters else 5
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
