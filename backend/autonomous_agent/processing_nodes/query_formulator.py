import logging
from typing import Any, Dict, List, Optional
import json
import re
from collections import Counter

from pydantic import BaseModel

from .base_node import BaseProcessingNode
from ..graph_components.nodes import QueryNode, ConceptNode
from ..graph_components.edges import ReformulatedAsEdge, MentionsEdge
from ..integration_components.graph_interface import GraphInterface # To interact with graph
from llm_services.llm_client import LLMClient # For LLM-based reformulation

logger = logging.getLogger(__name__)

class QueryFormulatorInput(BaseModel):
    original_query_node: QueryNode
    key_terms: List[str]
    # Potentially, context from input processing like meta-instructions
    meta_instructions: Dict[str, Any]
    # Access to concept nodes (e.g., retrieved based on key_terms)
    related_concepts: List[ConceptNode] = [] 

class QueryFormulatorOutput(BaseModel):
    reformulated_query_node: QueryNode
    # True if a new query was generated, False if original is deemed sufficient
    was_reformulated: bool 

# Export as an alias to match what workflow_engine.py is expecting
QueryFormulationOutput = QueryFormulatorOutput

class QueryFormulationNode(BaseProcessingNode):
    """Transforms user questions into optimized search queries."""

    def __init__(self, node_config: Optional[Dict[str, Any]] = None):
        super().__init__(node_config)
        # Configure default parameters
        config = node_config or {}
        self.max_concepts_to_use = config.get("max_concepts_to_use", 5)
        self.min_concept_relevance = config.get("min_concept_relevance", 0.3)
        self.llm_temperature = config.get("llm_temperature", 0.3)
        self.llm_max_tokens = config.get("llm_max_tokens", 200)
        self.custom_prompt_suffix = config.get("custom_prompt_suffix")

    async def execute(self, input_data: Any, context: Dict[str, Any]) -> QueryFormulatorOutput:
        """
        Formulates or reformulates a query for optimal retrieval.

        Args:
            input_data: Contains the original query node, key terms, and related concepts.
            context: Workflow context, including graph interface, LLM clients.

        Returns:
            A QueryFormulatorOutput object with the (potentially) reformulated QueryNode.
        """
        # Convert input dictionary to QueryFormulatorInput if needed
        if isinstance(input_data, dict):
            input_data = QueryFormulatorInput(**input_data)
        
        logger.info(f"[{self.get_name()}] Starting query formulation for: {input_data.original_query_node.reformulated_query_text[:100]}...")

        graph: Optional[GraphInterface] = context.get("graph_interface")
        llm_client: Optional[LLMClient] = context.get("llm_client")

        reformulated_text = input_data.original_query_node.reformulated_query_text
        reformulation_strategy = "none"
        was_reformulated = False

        # 1. Filter concepts by relevance if we have too many
        filtered_concepts = input_data.related_concepts
        if len(filtered_concepts) > self.max_concepts_to_use:
            # Prioritize more relevant concepts if we have relevance scores
            # Otherwise, just take the first N
            filtered_concepts = filtered_concepts[:self.max_concepts_to_use]
            logger.info(f"[{self.get_name()}] Limited concepts to top {self.max_concepts_to_use} out of {len(input_data.related_concepts)}")
        
        expanded_terms = []
        for concept in filtered_concepts:
            expanded_terms.append(concept.name)
        
        # 2. Use concept nodes to expand the original query
        if expanded_terms:
            # Use LLM for more natural and effective query reformulation
            if llm_client:
                # Prepare a structured prompt for the LLM
                domain_context = ""
                if input_data.meta_instructions.get("domain"):
                    domain_context = f"\nThe question is in the domain of: {input_data.meta_instructions.get('domain')}\n"
                
                prompt = f"""
I need to reformulate a search query to make it more effective for information retrieval.

Original query: "{reformulated_text}"{domain_context}

Related concepts I should incorporate naturally into the query: {', '.join(expanded_terms)}

Meta-instructions for reformulation:
- Make the query more specific and targeted
- Maintain the original intent and requirements
- Add relevant domain terminology from the concepts
- Make it a single cohesive query, not multiple questions
- Keep the query concise (generally under 100 words)

Reformulated query:
"""
                # Append custom suffix if provided
                if self.custom_prompt_suffix:
                    prompt += f"\n\nAdditional instructions: {self.custom_prompt_suffix}\n"
                    logger.info(f"[{self.get_name()}] Appended custom prompt suffix: {self.custom_prompt_suffix}")

                try:
                    # Use the LLM client to generate the reformulated query
                    reformulated_text = await llm_client.generate_text(
                        prompt=prompt,
                        max_tokens=self.llm_max_tokens,
                        temperature=self.llm_temperature
                    )
                    # Clean up the reformulated text
                    reformulated_text = reformulated_text.strip().strip('"')
                    reformulation_strategy = "concept_expansion_llm"
                    was_reformulated = True
                    logger.info(f"[{self.get_name()}] LLM-based query reformulation successful")
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error during LLM reformulation: {e}", exc_info=True)
                    # Fall back to simple concatenation
                    potential_reformulation = reformulated_text + " including concepts like " + ", ".join(list(set(expanded_terms)))
                    reformulated_text = potential_reformulation
                    reformulation_strategy = "concept_concatenation_fallback"
                    was_reformulated = True
                    logger.warning(f"[{self.get_name()}] Fell back to simple concept concatenation due to LLM error")
            else:
                # Fall back to simple concatenation if no LLM client
                potential_reformulation = reformulated_text + " including concepts like " + ", ".join(list(set(expanded_terms)))
                reformulated_text = potential_reformulation
                reformulation_strategy = "concept_concatenation"
                was_reformulated = True
                logger.info(f"[{self.get_name()}] Expanded query with concepts: {expanded_terms}")

        # 3. Incorporate domain knowledge if available
        if input_data.meta_instructions.get("domain_specific_formatting") and not reformulation_strategy.startswith("concept_expansion_llm"):
            # Apply domain-specific formatting if LLM wasn't used (which already does this)
            domain_format = input_data.meta_instructions.get("domain_specific_formatting")
            if domain_format == "scientific":
                reformulated_text = f"scientific research on {reformulated_text}"
                reformulation_strategy = f"{reformulation_strategy}_scientific"
                was_reformulated = True
            elif domain_format == "legal":
                reformulated_text = f"legal definition and implications of {reformulated_text}"
                reformulation_strategy = f"{reformulation_strategy}_legal"
                was_reformulated = True
            # Add other domain formats as needed

        # If no specific reformulation happened, use the original query
        if not was_reformulated:
            logger.info(f"[{self.get_name()}] No reformulation applied. Using original query text.")
            return QueryFormulatorOutput(
                reformulated_query_node=input_data.original_query_node, # Return original if no change
                was_reformulated=False
            )

        # 4. Create reformulated QueryNode in the graph
        reformulated_query_node = QueryNode(
            original_user_input=input_data.original_query_node.original_user_input, # Keep track of the very first input
            reformulated_query_text=reformulated_text,
            is_reformulated=True,
            contextual_info=input_data.original_query_node.contextual_info, # Carry over context
            attributes={
                "source": "query_formulation_node",
                "reformulation_strategy": reformulation_strategy,
                "original_query_id": input_data.original_query_node.id,
                "used_concepts": [concept.name for concept in filtered_concepts]
            }
        )

        if graph:
            try:
                # Persist the reformulated query node to the graph
                updated_node = await graph.add_node(reformulated_query_node)
                reformulated_query_node = updated_node  # Use the updated node with DB-assigned fields
                logger.info(f"[{self.get_name()}] Created reformulated QueryNode ID: {reformulated_query_node.id}")

                # 5. Connect new query to original user query (and the immediate parent query)
                reform_edge = ReformulatedAsEdge(
                    source_node_id=input_data.original_query_node.id,
                    target_node_id=reformulated_query_node.id,
                    properties={
                        "strategy": reformulation_strategy,
                        "confidence": 0.8 if "llm" in reformulation_strategy else 0.6
                    }
                )
                await graph.add_edge(reform_edge)
                logger.info(f"[{self.get_name()}] Linked original query {input_data.original_query_node.id} to reformulated {reformulated_query_node.id}")
                
                # Create MENTIONS edges for concepts that were used in reformulation
                if expanded_terms and filtered_concepts:
                    for concept in filtered_concepts:
                        mentions_edge = MentionsEdge(
                            source_node_id=reformulated_query_node.id, 
                            target_node_id=concept.id,
                            properties={"weight": 1.0}  # Could be weighted by relevance if available
                        )
                        await graph.add_edge(mentions_edge)
                    logger.info(f"[{self.get_name()}] Added MENTIONS edges for {len(filtered_concepts)} concepts")
            except Exception as e:
                logger.error(f"[{self.get_name()}] Error persisting query and relationships to graph: {e}", exc_info=True)
                # Continue with the query even if graph persistence failed

        return QueryFormulatorOutput(
            reformulated_query_node=reformulated_query_node,
            was_reformulated=True
        )
    
    async def _extract_key_terms(self, text: str, context: Dict[str, Any]) -> List[str]:
        """
        Extract key terms from text using LLM or fallback to basic extraction.
        
        Args:
            text: Input text to extract key terms from
            context: Processing context that contains the LLM client
            
        Returns:
            List of extracted key terms
        """
        # Get LLM client from context
        llm_client = context.get("llm_client")
        
        # Try sophisticated extraction with LLM
        if llm_client:
            try:
                # Craft a prompt for key term extraction
                prompt = (
                    f"Extract the 5-7 most important key terms or concepts from this text. "
                    f"Focus on domain-specific terms, entities, and concepts that are central to the meaning. "
                    f"Return ONLY a JSON array of terms, nothing else.\n\n"
                    f"Text: {text}\n\n"
                    f"Key terms:"
                )
                
                # Call LLM with JSON format response
                response = await llm_client.generate_text(prompt, response_format="json_array")
                
                # Parse the response
                try:
                    terms = json.loads(response)
                    if isinstance(terms, list) and terms:
                        # Filter out any empty or very short terms
                        valid_terms = [term for term in terms if isinstance(term, str) and len(term.strip()) > 2]
                        if valid_terms:
                            logger.info(f"Extracted {len(valid_terms)} key terms using LLM")
                            return valid_terms[:10]  # Limit to 10 terms maximum
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response as JSON: {e}. Response: {response[:100]}...")
                    # Fall through to basic extraction
            except Exception as e:
                logger.warning(f"LLM key term extraction failed: {e}. Using basic extraction as fallback.")
                # Fall through to basic extraction
        
        # Basic extraction as fallback
        logger.info("Using basic key term extraction as fallback")
        # More comprehensive stopwords list
        stopwords = set([
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "and", "or", "but", "if", "then", "else", "when", "where", "how",
            "what", "why", "who", "which", "this", "that", "these", "those",
            "to", "of", "in", "for", "with", "on", "at", "by", "from", "up",
            "about", "into", "over", "after", "before", "between", "under",
            "above", "through", "during", "since", "without", "within",
            "can", "could", "may", "might", "will", "would", "should", "shall",
            "must", "have", "has", "had", "do", "does", "did", "not", "my",
            "your", "our", "their", "his", "her", "its", "I", "you", "we", "they",
            "he", "she", "it", "me", "him", "us", "them"
        ])
        
        # Extract potential terms
        # Clean the text
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize and filter
        terms = []
        for term in cleaned_text.split():
            term = term.strip()
            if term and term.lower() not in stopwords and len(term) > 2:
                terms.append(term)
        
        # Get the most frequent terms, excluding stopwords
        term_counter = Counter(terms)
        most_common_terms = [term for term, count in term_counter.most_common(10)]
        
        # Alternatively, just take the first N terms after filtering
        if not most_common_terms:
            most_common_terms = terms[:10]
            
        return most_common_terms 