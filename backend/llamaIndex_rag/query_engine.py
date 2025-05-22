"""
RAG Query Engine implementation with Reliable RAG techniques to minimize hallucinations.

This module implements a production-ready RAG query engine using LlamaIndex
with features to generate reliable, accurate responses and detect hallucinations.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import re
import numpy as np

# LlamaIndex imports - updated to match installed package structure
from llama_index.core import (
    Settings,
    get_response_synthesizer,
    PromptHelper,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

# Hallucination prevention
from llama_index.core.evaluation import (
    ResponseEvaluator,
    FaithfulnessEvaluator,
)

# Local imports
from llamaIndex_rag.rag import RAGSystem

logger = logging.getLogger(__name__)

# Enhanced templates for reliable RAG
RELIABLE_RAG_TEMPLATE = """You are an AI assistant designed to provide accurate, truthful responses based on the provided context.

CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer ONLY based on the provided context. Do not use prior knowledge.
2. If the context doesn't contain enough information, acknowledge limitations by stating "Based on the provided information, I cannot fully answer this question" and suggest what might help.
3. Be specific and precise. Avoid broad claims unless explicitly supported by context.
4. For factual statements, cite the relevant part of the context (e.g., "According to the context...").
5. If different parts of the context present conflicting information, acknowledge the conflict.
6. Maintain the same level of technical detail as present in the context.
7. Present statistics and numerical data exactly as they appear in the context.
8. Avoid speculation beyond what's in the context, even if it seems reasonable.

ANSWER:"""

# Template with self-critique and revision strategy
SELF_CRITIQUE_TEMPLATE = """You are an AI assistant designed to provide accurate, truthful responses based on the provided context.

CONTEXT:
{context}

QUERY: {query}

PROCESS:
First, draft an answer based solely on the provided context.
Then, critically evaluate your draft answer by checking:
1. Is every claim directly supported by specific content in the context?
2. Have I included information not present in the context?
3. Have I maintained appropriate uncertainty when the context is ambiguous?
4. Have I accurately represented numerical data from the context?

After your evaluation, revise your answer to fix any issues identified.

FINAL ANSWER (make sure this only contains factual information from the context):"""

# Template for handling uncertainty when context is insufficient
UNCERTAINTY_TEMPLATE = """You are an AI assistant designed to handle uncertainty appropriately.

CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Determine if the context contains sufficient information to answer the query.
2. If the context is sufficient, provide a detailed answer based solely on the context.
3. If the context is partially sufficient, clearly indicate which parts of the query you can address and acknowledge the limitations for other parts.
4. If the context is insufficient, state that you don't have enough information to provide a reliable answer.
5. Do not speculate beyond what's in the context, even if the speculation seems reasonable.
6. Maintain appropriate levels of certainty/uncertainty in your language based on the strength of evidence in the context.

ANSWER:"""

# Template for query reformulation
QUERY_REFORMULATION_TEMPLATE = """You are an AI assistant that helps to reformulate queries to improve retrieval results.

ORIGINAL QUERY: {query}

TASK:
Please reformulate the original query to make it more effective for information retrieval. Generate 2-3 alternative versions that:
1. Clarify any ambiguities in the original query
2. Include relevant keywords that might help with retrieval
3. Break complex queries into simpler components
4. Express the information need more explicitly

FORMAT YOUR RESPONSE AS A JSON ARRAY OF STRINGS, CONTAINING THE REFORMULATED QUERIES ONLY:
"""

# Template for source attribution and verification
SOURCE_ATTRIBUTION_TEMPLATE = """You are an AI assistant that provides trustworthy answers with proper source attribution.

CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
1. Answer the query based solely on the provided context
2. For each main point in your answer, cite the specific part of the context that supports it
3. Use the format [Context X] to cite sources, where X is the context number
4. If different sources provide conflicting information, acknowledge this and explain the differences
5. Maintain the level of detail and technical language present in the context
6. Structure your answer in a clear, logical manner

ANSWER:"""

class RAGQueryEngine:
    """
    Production-ready RAG Query Engine with reliability techniques to minimize hallucinations.
    """
    
    def __init__(
        self,
        rag_system: RAGSystem,
        model_name: str = "gpt-4.1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        streaming: bool = False,
        default_prompt: Optional[str] = None,
        use_self_critique: bool = True
    ):
        """
        Initialize RAG query engine.
        
        Args:
            rag_system: RAG system to use for retrieving context
            model_name: Name of the LLM model to use for generation
            temperature: Temperature for generation (lower = more deterministic)
            max_tokens: Maximum tokens in generated response
            streaming: Whether to stream responses by default
            default_prompt: Default prompt template to use for responses
            use_self_critique: Whether to use self-critique for hallucination reduction
        """
        self.rag_system = rag_system
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming = streaming
        
        # Initialize LLM
        self.llm = OpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # No need to create settings explicitly, use global settings with our local llm
        Settings.llm = self.llm
        Settings.embed_model = self.rag_system.embed_model
        
        # Set default prompt - now using the enhanced reliable RAG template
        self.default_prompt = default_prompt or PromptTemplate(RELIABLE_RAG_TEMPLATE)
        
        # Initialize specialized prompts for different scenarios
        self.self_critique_prompt = PromptTemplate(SELF_CRITIQUE_TEMPLATE)
        self.uncertainty_prompt = PromptTemplate(UNCERTAINTY_TEMPLATE)
        self.query_reformulation_prompt = PromptTemplate(QUERY_REFORMULATION_TEMPLATE)
        self.source_attribution_prompt = PromptTemplate(SOURCE_ATTRIBUTION_TEMPLATE)
        
        # Initialize self critique
        self.use_self_critique = use_self_critique
        
        # Enhanced retrieval settings
        self.use_query_reformulation = True
        self.use_multi_retrieval = True
        self.use_source_attribution = True
        
        logger.info(f"Enhanced RAG query engine initialized with model: {model_name}, temperature: {temperature}")
    
    def update_model(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        default_prompt: Optional[str] = None
    ):
        """Update model parameters."""
        if model_name:
            self.model_name = model_name
        
        if temperature is not None:
            self.temperature = temperature
        
        if max_tokens:
            self.max_tokens = max_tokens
            
        if default_prompt:
            self.default_prompt = default_prompt
        
        # Reinitialize LLM with new parameters
        self.llm = OpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        logger.info(f"RAG query engine updated with model: {self.model_name}, temperature: {self.temperature}")
    
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        search_filter: Optional[Dict[str, Any]] = None,
        custom_prompt: Optional[str] = None,
        streaming: Optional[bool] = True,
        return_contexts: bool = True,
        use_self_critique: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with reliable RAG techniques.
        
        Args:
            query_text: Query to answer
            top_k: Number of context chunks to retrieve
            search_filter: Metadata filters for retrieval
            custom_prompt: Custom prompt template for response
            streaming: Whether to stream the response
            return_contexts: Whether to return context in response
            use_self_critique: Whether to use self-critique for hallucination reduction
            
        Returns:
            Dict with query results, including answer and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Step 1: Query Reformulation (if enabled)
            reformulated_queries = []
            if self.use_query_reformulation:
                reformulated_queries = await self._reformulate_query(query_text)
                logger.info(f"Reformulated query into {len(reformulated_queries)} variations")
            
            # Always include the original query
            all_queries = [query_text] + reformulated_queries
            
            # Step 2: Query Complexity Assessment
            query_complexity = await self._assess_query_complexity(query_text)
            logger.info(f"Query complexity assessed as: {query_complexity}")
            
            # Adjust retrieval parameters based on complexity
            adjusted_top_k = top_k
            if query_complexity == "high":
                # For complex queries, retrieve more context
                adjusted_top_k = min(top_k + 3, 12)  # Get more context but with a reasonable limit
                logger.info(f"Increased context retrieval to {adjusted_top_k} chunks due to high complexity")
            
            # Step 3: Multi-strategy Retrieval
            retrieved_nodes = []
            if self.use_multi_retrieval and reformulated_queries:
                # Use multiple queries for retrieval
                all_nodes = []
                
                # First retrieve with original query
                original_nodes = self.rag_system.retrieve_context(
                    query=query_text,
                    top_k=adjusted_top_k,
                    search_filter=search_filter
                )
                
                # Add retrieval method metadata
                for node in original_nodes:
                    if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                        node.node.metadata["retrieval_method"] = "original_query"
                
                all_nodes.extend(original_nodes)
                
                # Then retrieve with reformulated queries, but with fewer results each
                reformulation_top_k = max(2, adjusted_top_k // 2)
                for i, ref_query in enumerate(reformulated_queries[:2]):  # Limit to top 2 reformulations
                    ref_nodes = self.rag_system.retrieve_context(
                        query=ref_query,
                        top_k=reformulation_top_k,
                        search_filter=search_filter
                    )
                    
                    # Add retrieval method metadata
                    for node in ref_nodes:
                        if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                            node.node.metadata["retrieval_method"] = f"reformulation_{i+1}"
                    
                    all_nodes.extend(ref_nodes)
                
                # Deduplicate nodes (in case the same document was retrieved by multiple queries)
                seen_texts = set()
                unique_nodes = []
                
                for node in all_nodes:
                    if hasattr(node, 'node'):
                        node_text = node.node.get_content()
                        # Use a simple hash of the text to identify duplicates
                        text_hash = hash(node_text[:100])  # Use first 100 chars for hashing
                        
                        if text_hash not in seen_texts:
                            seen_texts.add(text_hash)
                            unique_nodes.append(node)
                
                # Keep only the top nodes by score
                retrieved_nodes = sorted(unique_nodes, key=lambda n: n.score or 0, reverse=True)[:adjusted_top_k]
                logger.info(f"Retrieved {len(retrieved_nodes)} unique nodes using multi-query retrieval")
            else:
                # Use standard retrieval with original query
                retrieved_nodes = self.rag_system.retrieve_context(
                    query=query_text,
                    top_k=adjusted_top_k,
                    search_filter=search_filter
                )
            
            # Step 4: Custom Semantic Reranking
            if hasattr(self.rag_system, '_apply_context_reranking'):
                retrieved_nodes = self.rag_system._apply_context_reranking(retrieved_nodes, query_text)
                
            # Convert nodes to text for context
            context_texts = [node.node.get_content() for node in retrieved_nodes]
            
            # Step 5: Context Assessment and Quality Check
            context_quality = await self._assess_context_quality(query_text, context_texts)
            logger.info(f"Context quality assessed as: {context_quality}")
            
            # Step 6: Select appropriate prompt template based on context quality and source attribution preference
            use_source_attribution = self.use_source_attribution and context_quality != "insufficient"
            if use_source_attribution:
                prompt_template = self.source_attribution_prompt
                logger.info("Using source attribution prompt template")
            else:
                prompt_template = self._select_prompt_template(context_quality, custom_prompt)
                logger.info(f"Selected prompt template type: {prompt_template.__class__.__name__}")
            
            # Format context with enhanced context formatting
            enhanced_context = await self._enhance_context_formatting(query_text, context_texts)
            combined_context = enhanced_context if enhanced_context else "\n\n".join(context_texts)
            
            # Step 7: Generate response with the selected strategy
            if use_self_critique and self.use_self_critique and context_quality != "insufficient":
                # First generate an initial answer
                formatted_initial_prompt = self.default_prompt.format(
                    context=combined_context,
                    query=query_text
                )
                
                # Generate initial response
                initial_response = self.llm.complete(formatted_initial_prompt)
                initial_response_text = initial_response.text if hasattr(initial_response, 'text') else str(initial_response)
                
                # Then use self-critique to improve it
                formatted_prompt = self.self_critique_prompt.format(
                    context=combined_context,
                    query=query_text,
                    initial_answer=initial_response_text
                )
                
                # Generate final response with self-critique
                response = self.llm.complete(formatted_prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
            else:
                # Regular prompt formatting based on selected template
                formatted_prompt = prompt_template.format(
                    context=combined_context,
                    query=query_text
                )
                
                # Generate response
                response = self.llm.complete(formatted_prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Step 8: Detect and address hallucinations
            hallucination_result = self.rag_system.detect_hallucination(
                query=query_text,
                response=response_text,
                context=context_texts
            )
            
            # If high hallucination probability is detected, try to correct the response
            if hallucination_result.get("is_hallucination", False) and not use_self_critique:
                logger.warning(f"Detected potential hallucination, regenerating with self-critique prompt")
                
                # Use self-critique prompt for regeneration
                formatted_prompt = self.self_critique_prompt.format(
                    context=combined_context,
                    query=query_text,
                    initial_answer=response_text  # Use the initial response as input to self-critique
                )
                
                # Regenerate response
                response = self.llm.complete(formatted_prompt)
                response_text = response.text if hasattr(response, 'text') else str(response)
                
                # Re-evaluate
                hallucination_result = self.rag_system.detect_hallucination(
                    query=query_text,
                    response=response_text,
                    context=context_texts
                )
                
                # If still hallucinating, add a disclaimer
                if hallucination_result.get("is_hallucination", False):
                    response_text = f"[Note: This response may contain some uncertainty due to limited context.]\n\n{response_text}"
            
            # Calculate end time
            end_time = asyncio.get_event_loop().time()
            duration_seconds = end_time - start_time
            
            # Prepare result
            result = {
                "query": query_text,
                "answer": response_text,
                "duration_seconds": duration_seconds,
                "model": self.model_name,
                "hallucination_metrics": hallucination_result,
                "context_quality": context_quality,
                "query_complexity": query_complexity,
                "reformulated_queries": reformulated_queries if self.use_query_reformulation else []
            }
            
            # Include contexts if requested
            if return_contexts:
                result["contexts"] = []
                for i, node in enumerate(retrieved_nodes):
                    reliability_score = node.node.metadata.get("reliability_score", node.score)
                    retrieval_method = node.node.metadata.get("retrieval_method", "default")
                    result["contexts"].append({
                        "text": node.node.get_content(),
                        "score": node.score if hasattr(node, 'score') else None,
                        "metadata": node.node.metadata if hasattr(node.node, 'metadata') else {},
                        "document_id": node.node.metadata.get("document_id") if hasattr(node.node, 'metadata') else None,
                        "reliability_score": reliability_score,
                        "retrieval_method": retrieval_method
                    })
            
            return result
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return {
                "query": query_text,
                "answer": f"Error processing query: {str(e)}",
                "error": str(e)
            }
    
    async def _reformulate_query(self, query: str) -> List[str]:
        """
        Reformulate the query to increase retrieval effectiveness.
        
        Args:
            query: The original query text
            
        Returns:
            List of reformulated queries
        """
        try:
            logger.info(f"Reformulating query: {query}")
            
            # Format the prompt for query reformulation
            formatted_prompt = self.query_reformulation_prompt.format(query=query)
            
            # Generate reformulations
            response = self.llm.complete(formatted_prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Try to parse the JSON response
            try:
                # Extract JSON array from the response if needed
                json_match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
                    
                reformulated_queries = json.loads(json_text)
                
                # Ensure it's a list of strings
                if isinstance(reformulated_queries, list):
                    reformulated_queries = [str(q) for q in reformulated_queries if q]
                    logger.info(f"Generated {len(reformulated_queries)} query reformulations")
                    return reformulated_queries[:3]  # Limit to 3 reformulations
                else:
                    logger.warning("Query reformulation didn't return a list, using original query")
                    return []
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract queries using regex
                logger.warning("JSON parsing failed for query reformulation, trying regex extraction")
                # Look for quoted strings that might be reformulated queries
                matches = re.findall(r'"([^"]+)"', response_text)
                if matches:
                    return matches[:3]  # Limit to 3 reformulations
                
                # Otherwise try to split by newlines or numbers
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                # Remove any numbering at the start of lines
                clean_lines = [re.sub(r'^[\d\.\)\-\s]+', '', line).strip() for line in lines]
                # Remove any quotes
                clean_lines = [line.strip('"\'') for line in clean_lines if len(line) > 10]
                
                if clean_lines:
                    return clean_lines[:3]  # Limit to 3 reformulations
                
                return []
        except Exception as e:
            logger.warning(f"Error in query reformulation: {str(e)}")
            return []  # Fall back to empty list
    
    async def _assess_query_complexity(self, query: str) -> str:
        """
        Assess the complexity of a query to determine appropriate retrieval and answering strategies.
        
        Args:
            query: The query text
            
        Returns:
            Complexity level: "low", "medium", or "high"
        """
        try:
            # Simple heuristics for query complexity assessment
            words = query.split()
            
            # Check if query has multiple components/questions
            question_marks = query.count('?')
            sentence_count = max(1, len([s for s in query.split('.') if len(s.strip()) > 0]))
            
            # Check for complexity indicators
            complex_indicators = [
                "compare", "contrast", "analyze", "explain", "why", "how",
                "relationship", "difference", "similarities", "pros and cons",
                "advantages", "disadvantages", "detailed"
            ]
            
            # Count indicators
            indicator_count = sum(1 for word in words if any(ind in word.lower() for ind in complex_indicators))
            
            # Determine complexity
            if len(words) > 25 or question_marks > 1 or indicator_count >= 2:
                return "high"
            elif len(words) > 12 or sentence_count > 1 or indicator_count >= 1:
                return "medium"
            else:
                return "low"
        except Exception as e:
            logger.warning(f"Error assessing query complexity: {str(e)}")
            return "medium"  # Default to medium if assessment fails
    
    async def _assess_context_quality(self, query: str, contexts: List[str]) -> str:
        """
        Assess the quality and relevance of retrieved context relative to the query.
        
        Args:
            query: The query text
            contexts: List of context passages
            
        Returns:
            Quality assessment: "high", "medium", "low", or "insufficient"
        """
        try:
            if not contexts:
                return "insufficient"
            
            # Get query embedding
            query_embedding = self.rag_system.embed_model.get_text_embedding(query)
            
            # Calculate relevance scores for each context
            relevance_scores = []
            for context in contexts:
                context_embedding = self.rag_system.embed_model.get_text_embedding(context)
                
                # Calculate cosine similarity
                if isinstance(query_embedding, list) and isinstance(context_embedding, list):
                    # Convert to numpy arrays to handle calculations properly
                    query_array = np.array(query_embedding)
                    context_array = np.array(context_embedding)
                    
                    # Ensure we get a real number by taking the real part if complex
                    dot_product = np.dot(query_array, context_array)
                    if isinstance(dot_product, complex):
                        dot_product = dot_product.real
                        
                    norm_query = np.linalg.norm(query_array)
                    norm_context = np.linalg.norm(context_array)
                    
                    # Avoid division by zero
                    if norm_query > 0 and norm_context > 0:
                        cos_sim = float(dot_product / (norm_query * norm_context))
                    else:
                        cos_sim = 0.0
                        
                    relevance_scores.append(float(cos_sim))
                else:
                    relevance_scores.append(0.5)  # Default mid-range score
            
            # Calculate average and max relevance
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            max_relevance = max(relevance_scores) if relevance_scores else 0
            
            # Assess context quality based on relevance metrics
            if max_relevance > 0.85 and avg_relevance > 0.7:
                return "high"
            elif max_relevance > 0.75 and avg_relevance > 0.6:
                return "medium"
            elif max_relevance > 0.65:
                return "low"
            else:
                return "insufficient"
                
        except Exception as e:
            logger.warning(f"Error assessing context quality: {str(e)}")
            if contexts:
                return "medium"  # Default to medium if assessment fails but contexts exist
            else:
                return "insufficient"
    
    def _select_prompt_template(self, context_quality: str, custom_prompt: Optional[str] = None) -> PromptTemplate:
        """
        Select the appropriate prompt template based on context quality.
        
        Args:
            context_quality: Quality assessment of context
            custom_prompt: Optional custom prompt to use
            
        Returns:
            Selected prompt template
        """
        # If custom prompt is provided, use it
        if custom_prompt:
            return PromptTemplate(custom_prompt)
            
        # Select based on context quality
        if context_quality == "insufficient":
            return self.uncertainty_prompt
        elif context_quality == "low":
            return self.self_critique_prompt  # Use self-critique for low quality context
        else:
            return self.default_prompt
    
    async def _enhance_context_formatting(self, query: str, contexts: List[str]) -> Optional[str]:
        """
        Enhance context formatting to highlight relevant sections and improve readability.
        
        Args:
            query: The query text
            contexts: List of context passages
            
        Returns:
            Formatted context string or None if enhancement fails
        """
        try:
            if not contexts:
                return None
                
            # For now, a simple enhancement with numbering and source attribution
            formatted_contexts = []
            for i, context in enumerate(contexts):
                # Add section number and formatting
                formatted_context = f"[CONTEXT {i+1}]\n{context.strip()}\n"
                formatted_contexts.append(formatted_context)
                
            # Combine contexts with clear separation
            return "\n".join(formatted_contexts)
        except Exception as e:
            logger.warning(f"Error enhancing context formatting: {str(e)}")
            return None  # Return None to fall back to standard formatting
    
    def generate_custom_response(
        self,
        query_text: str,
        context: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate a custom response with provided context and query.
        
        Args:
            query_text: Query to answer
            context: Context to use for generation
            prompt_template: Custom prompt template
            
        Returns:
            Generated response
        """
        try:
            # Use provided prompt or default
            template = PromptTemplate(prompt_template) if prompt_template else self.default_prompt
            
            # Format prompt
            formatted_prompt = template.format(
                context=context,
                query=query_text
            )
            
            # Generate response
            response = self.llm.complete(formatted_prompt)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Error generating custom response: {str(e)}")
            return f"Error generating response: {str(e)}" 