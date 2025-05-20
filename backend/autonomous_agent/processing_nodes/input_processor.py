import logging
from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel, Field

from .base_node import BaseProcessingNode
from ..graph_components.nodes import QueryNode, ConceptNode
from ..integration_components.graph_interface import GraphInterface
from llm_services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Add a module-level function for extracting key concepts
async def extract_key_concepts(query_text: str, graph_interface) -> List[ConceptNode]:
    """
    Extract key concepts from a query text to enhance graph-based retrieval.
    
    Args:
        query_text: The query text to extract concepts from
        graph_interface: Interface to the knowledge graph
        
    Returns:
        List of identified concept nodes from the graph
    """
    try:
        # 1. Perform basic text preprocessing
        clean_text = query_text.lower()
        
        # 2. Extract potential concept terms
        # For complex extraction, you might want to use NLP libraries like spaCy
        # This is a simplified approach that extracts:
        # - Noun phrases (simplified)
        # - Key domain terms
        
        # Split into sentences and words
        sentences = clean_text.split('.')
        words = clean_text.split()
        
        # Look for phrases of 1-3 words that might be concepts
        potential_concepts = set()
        
        # Add single words (excluding stopwords)
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'to', 'from', 
                    'in', 'out', 'on', 'off', 'over', 'under', 'for', 'of', 'by', 'with',
                    'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom'}
        
        for word in words:
            # Strip punctuation
            word = word.strip('.,;:!?()"\'')
            if word and word not in stopwords and len(word) > 1:
                potential_concepts.add(word)
        
        # Add bigrams and trigrams
        word_seq = [w.strip('.,;:!?()"\'') for w in words if w.strip('.,;:!?()"\'')]
        for i in range(len(word_seq) - 1):
            bigram = f"{word_seq[i]} {word_seq[i+1]}"
            if all(w not in stopwords for w in bigram.split()):
                potential_concepts.add(bigram)
            
            if i < len(word_seq) - 2:
                trigram = f"{word_seq[i]} {word_seq[i+1]} {word_seq[i+2]}"
                if all(w not in stopwords for w in trigram.split()):
                    potential_concepts.add(trigram)
        
        # 3. Look up these potential concepts in the graph
        concepts = []
        for term in potential_concepts:
            # First try exact match
            exact_concepts = await graph_interface.find_concepts_by_name_exact(term)
            if exact_concepts:
                concepts.extend(exact_concepts)
                continue
            
            # Then try fuzzy match for those not found exactly
            fuzzy_concepts = await graph_interface.find_concepts_by_name_fuzzy(term)
            concepts.extend(fuzzy_concepts)
        
        # 4. Remove duplicates by ID
        unique_concepts = {}
        for concept in concepts:
            if concept.id not in unique_concepts:
                unique_concepts[concept.id] = concept
        
        return list(unique_concepts.values())
        
    except Exception as e:
        logger.error(f"Error extracting key concepts: {e}", exc_info=True)
        return []

class UserInput(BaseModel):
    raw_prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_context: Dict[str, Any] = Field(default_factory=dict)

class LLMParsedInput(BaseModel):
    core_question: str = Field(description="The essential question or task identified in the user's prompt.")
    key_terms: List[str] = Field(default_factory=list, description="Significant keywords or phrases for search and concept matching.")
    meta_instructions: Dict[str, Any] = Field(default_factory=dict, description="Explicit or implicit instructions about response format, style, language, etc.")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Named entities or specific contextual details mentioned.")

class InputProcessorOutput(BaseModel):
    initial_query_node: QueryNode
    # These fields are derived from LLMParsedInput or directly from UserInput
    core_question: str 
    key_terms: List[str]
    meta_instructions: Dict[str, Any]
    contextual_info_extracted: Dict[str, Any] # Merges LLM extracted entities with user-provided additional_context

class InputProcessingNode(BaseProcessingNode):
    """Parses the user's input using an LLM to extract structured information and creates an initial QueryNode."""

    def __init__(self, node_config: Optional[Dict[str, Any]] = None):
        super().__init__(node_config)
        # Configure default parameters
        config = node_config or {}
        self.llm_temperature = config.get("llm_temperature", 0.2)
        self.llm_max_tokens = config.get("llm_max_tokens", 500)
        self.use_mock_responses = config.get("use_mock_responses", False)

    async def _parse_prompt_with_llm(self, llm_client: Any, raw_prompt: str) -> Optional[LLMParsedInput]:
        """Uses LLM to parse the raw prompt into structured data."""
        # This prompt engineering is crucial and would need refinement.
        # It asks the LLM to return a JSON object matching LLMParsedInput structure.
        structured_prompt = f'''
Analyze the following user prompt and extract the specified information.
Return the output as a JSON object with the following keys:
- "core_question": The essential question or task identified in the user's prompt.
- "key_terms": A list of significant keywords or phrases (max 7-10) for search and concept matching.
- "meta_instructions": A dictionary of explicit or implicit instructions (e.g., {{ "language": "fr", "format": "bullet_points" }}).
- "extracted_entities": A dictionary of important named entities or specific contextual details mentioned by the user (e.g., {{ "product_name": "XYZ Widget", "date_range": "next 5 days" }}).

User Prompt:
---
{raw_prompt}
---

JSON Output:
'''
        try:
            # If configured to use mock responses for testing/development
            if self.use_mock_responses:
                logger.warning(f"[{self.get_name()}] Using mock responses for LLM parsing. NOT FOR PRODUCTION USE.")
                # Mocked response for development - REMOVE FOR PRODUCTION
                if "tell me about dogs" in raw_prompt.lower():
                    parsed_output = LLMParsedInput(
                        core_question="Information about dogs",
                        key_terms=["dogs", "canine information"],
                        meta_instructions={},
                        extracted_entities={}
                    )
                elif "what is the capital of france in spanish" in raw_prompt.lower():
                    parsed_output = LLMParsedInput(
                        core_question="Capital of France",
                        key_terms=["France", "capital", "Paris"],
                        meta_instructions={"language": "es"},
                        extracted_entities={"country": "France"}
                    )
                else:
                    # Fallback for mock
                    parsed_output = LLMParsedInput(
                        core_question=raw_prompt,
                        key_terms=[term for term in raw_prompt.split()[:5]], # simple split for mock
                        meta_instructions={},
                        extracted_entities={}
                    )
                return parsed_output

            # Check if client is OpenAI client or has standard methods
            is_openai_client = hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions")
            
            # Attempt to use structured output generation if available (non-OpenAI client)
            if not is_openai_client and hasattr(llm_client, "generate_structured_output"):
                try:
                    # Use direct structured output generation with Pydantic model
                    parsed_output = await llm_client.generate_structured_output(
                        prompt=structured_prompt, 
                        output_model=LLMParsedInput,
                        temperature=self.llm_temperature,
                        max_tokens=self.llm_max_tokens
                    )
                    if parsed_output:
                        logger.info(f"[{self.get_name()}] Successfully parsed input using structured LLM output")
                        return parsed_output
                    else:
                        raise ValueError("LLM returned None for structured output")
                except Exception as e:
                    logger.warning(f"[{self.get_name()}] Structured output generation failed: {e}. Falling back to text generation.")
                    # Continue to fallback method
            
            # Check the type of LLM client and adapt accordingly
            response_json_str = ""
            
            # For OpenAI client
            if is_openai_client:
                logger.info(f"[{self.get_name()}] Using OpenAI client for LLM")
                try:
                    response = llm_client.chat.completions.create(
                        model="gpt-4", # Use a model that can structure JSON output
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that responds with properly formatted JSON."},
                            {"role": "user", "content": structured_prompt}
                        ],
                        temperature=self.llm_temperature,
                        max_tokens=self.llm_max_tokens
                    )
                    response_json_str = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error calling OpenAI API: {e}")
                    raise e
            # For other clients with generate_text
            elif hasattr(llm_client, "generate_text"):
                logger.info(f"[{self.get_name()}] Using generate_text method")
                try:
                    response_json_str = await llm_client.generate_text(
                        prompt=structured_prompt, 
                        temperature=self.llm_temperature,
                        max_tokens=self.llm_max_tokens
                    )
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error with text generation: {e}")
                    raise e
            # Last resort: try a generic method call
            else:
                logger.warning(f"[{self.get_name()}] No standard method found on LLM client, using generic call")
                # Generic call that should work with most LLM clients
                try:
                    if hasattr(llm_client, "complete"):
                        response_data = await llm_client.complete(
                            prompt=structured_prompt,
                            temperature=self.llm_temperature,
                            max_tokens=self.llm_max_tokens
                        )
                        if isinstance(response_data, str):
                            response_json_str = response_data
                        elif hasattr(response_data, "text"):
                            response_json_str = response_data.text
                        elif isinstance(response_data, dict) and "text" in response_data:
                            response_json_str = response_data["text"]
                        else:
                            response_json_str = str(response_data)
                    else:
                        raise ValueError(f"LLM client has no usable method for text generation. Client type: {type(llm_client)}")
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error with generic LLM call: {e}")
                    raise e
            
            # Extract JSON if it's embedded in other text
            json_start = response_json_str.find('{')
            json_end = response_json_str.rfind('}')
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_json_str[json_start:json_end+1]
                try:
                    # Parse the JSON and validate against our model
                    data = json.loads(json_text)
                    parsed_output = LLMParsedInput(**data)
                    logger.info(f"[{self.get_name()}] Successfully parsed input using JSON extraction from text")
                    return parsed_output
                except (json.JSONDecodeError, ValueError) as json_err:
                    logger.error(f"[{self.get_name()}] Error parsing JSON from LLM response: {json_err}")
                    raise ValueError(f"Invalid JSON structure in LLM response: {json_err}")
            else:
                logger.error(f"[{self.get_name()}] No JSON object found in LLM response: {response_json_str[:100]}...")
                raise ValueError("No JSON object found in LLM response")

        except Exception as e:
            logger.error(f"[{self.get_name()}] Error parsing prompt with LLM: {e}", exc_info=True)
            # Fallback: use raw prompt as core question, basic keyword extraction
            return LLMParsedInput(
                core_question=raw_prompt.strip(),
                key_terms=[term.lower().strip(".,!?") for term in raw_prompt.split() if len(term.strip(".,!?")) > 2][:5],
                meta_instructions={},
                extracted_entities={}
            )

    async def execute(self, input_data: Any, context: Dict[str, Any]) -> InputProcessorOutput:
        # Handle both string and UserInput object
        if isinstance(input_data, str):
            raw_prompt = input_data
            user_id = context.get("user_id")
            session_id = context.get("session_id")
            additional_context = context.get("additional_context", {})
            # Convert string input to UserInput object
            input_obj = UserInput(
                raw_prompt=raw_prompt,
                user_id=user_id,
                session_id=session_id,
                additional_context=additional_context
            )
            logger.info(f"[{self.get_name()}] Processing user input (from string): {raw_prompt[:100]}...")
        elif isinstance(input_data, UserInput):
            input_obj = input_data
            logger.info(f"[{self.get_name()}] Processing user input: {input_obj.raw_prompt[:100]}...")
        else:
            # If input is neither string nor UserInput, try to convert it
            try:
                if hasattr(input_data, "raw_prompt"):
                    raw_prompt = input_data.raw_prompt
                    user_id = getattr(input_data, "user_id", context.get("user_id"))
                    session_id = getattr(input_data, "session_id", context.get("session_id"))
                    additional_context = getattr(input_data, "additional_context", {})
                    input_obj = UserInput(
                        raw_prompt=raw_prompt,
                        user_id=user_id,
                        session_id=session_id,
                        additional_context=additional_context
                    )
                    logger.info(f"[{self.get_name()}] Processing user input (from compatible object): {raw_prompt[:100]}...")
                else:
                    # Use string representation as fallback
                    raw_prompt = str(input_data)
                    user_id = context.get("user_id")
                    session_id = context.get("session_id")
                    additional_context = context.get("additional_context", {})
                    input_obj = UserInput(
                        raw_prompt=raw_prompt,
                        user_id=user_id,
                        session_id=session_id,
                        additional_context=additional_context
                    )
                    logger.warning(f"[{self.get_name()}] Converting unknown input type to string: {raw_prompt[:100]}...")
            except Exception as e:
                logger.error(f"[{self.get_name()}] Error processing input: {e}", exc_info=True)
                # Create a minimal valid input as last resort
                raw_prompt = str(input_data) if input_data else "Empty query"
                input_obj = UserInput(
                    raw_prompt=raw_prompt,
                    user_id=context.get("user_id"),
                    session_id=context.get("session_id")
                )
                logger.error(f"[{self.get_name()}] Using fallback input: {raw_prompt[:100]}")
        
        graph_interface: Optional[GraphInterface] = context.get("graph_interface")
        llm_client: Optional[LLMClient] = context.get("llm_client")

        if not llm_client:
            logger.error(f"[{self.get_name()}] LLMClient not found in context. Cannot parse prompt effectively.")
            # Fallback to very basic parsing if no LLM
            core_q = input_obj.raw_prompt.strip()
            k_terms = [term.lower().strip(".,!?") for term in core_q.split() if len(term.strip(".,!?")) > 2][:5]
            m_instr: Dict[str, Any] = {}
            e_entities = {}
        else:
            parsed_llm_input = await self._parse_prompt_with_llm(llm_client, input_obj.raw_prompt)
            if parsed_llm_input:
                core_q = parsed_llm_input.core_question
                k_terms = parsed_llm_input.key_terms
                m_instr = parsed_llm_input.meta_instructions
                e_entities = parsed_llm_input.extracted_entities
            else: # Should not happen if _parse_prompt_with_llm always returns a fallback
                core_q = input_obj.raw_prompt.strip()
                k_terms = [term.lower().strip(".,!?") for term in core_q.split() if len(term.strip(".,!?")) > 2][:5]
                m_instr = {}
                e_entities = {}
        
        # Combine LLM-extracted entities with any user-provided additional context
        combined_contextual_info = {}
        
        # Ensure all values are primitive types for Neo4j compatibility
        if e_entities:
            for key, value in e_entities.items():
                if isinstance(value, (str, int, float, bool)):
                    combined_contextual_info[key] = value
                else:
                    # Convert complex objects to string representation
                    combined_contextual_info[key] = str(value)
        
        if input_obj.additional_context:
            for key, value in input_obj.additional_context.items():
                if isinstance(value, (str, int, float, bool)):
                    combined_contextual_info[key] = value
                else:
                    # Convert complex objects to string representation
                    combined_contextual_info[key] = str(value)
                    
        # Add user_id and session_id as strings
        if input_obj.user_id:
            combined_contextual_info["user_id_str"] = str(input_obj.user_id)
        if input_obj.session_id:
            combined_contextual_info["session_id_str"] = str(input_obj.session_id)

        # Create the initial QueryNode with the core question as both query_text and reformulated_query_text
        initial_query_node = QueryNode(
            original_user_input=input_obj.raw_prompt,
            reformulated_query_text=core_q,  # Initially, the core question is the reformulated query
            query_text=core_q,  # Add the required query_text field
            is_reformulated=False,
            contextual_info=combined_contextual_info,
            attributes={
                "source": "user_direct_input",
                "key_terms": k_terms,  # Store key terms in attributes for future reference
                "query_analysis_confidence": 0.8 if llm_client else 0.4  # Higher confidence with LLM
            }
        )

        if graph_interface:
            try:
                updated_node = await graph_interface.add_node(initial_query_node)
                initial_query_node = updated_node # Use the node returned from DB (e.g., with created_at)
                logger.info(f"[{self.get_name()}] Created and persisted initial QueryNode with ID: {initial_query_node.id}")
            except Exception as e:
                logger.error(f"[{self.get_name()}] Error persisting QueryNode: {e}", exc_info=True)
                # Decide: proceed with non-persisted node or raise error? For now, proceed.
        else:
            logger.warning(f"[{self.get_name()}] Graph interface not found in context. QueryNode not persisted.")

        logger.info(f"[{self.get_name()}] Core question: {core_q}")
        logger.info(f"[{self.get_name()}] Key terms: {k_terms}")
        if m_instr:
            logger.info(f"[{self.get_name()}] Meta instructions: {m_instr}")

        return InputProcessorOutput(
            initial_query_node=initial_query_node,
            core_question=core_q,
            key_terms=k_terms,
            meta_instructions=m_instr,
            contextual_info_extracted=combined_contextual_info # Pass the merged context
        ) 

    async def extract_key_concepts(self, query_text: str, graph_interface) -> List[ConceptNode]:
        """
        Extract key concepts from a query text to enhance graph-based retrieval.
        
        Args:
            query_text: The query text to extract concepts from
            graph_interface: Interface to the knowledge graph
            
        Returns:
            List of identified concept nodes from the graph
        """
        # This method delegates to the module-level function with the same name
        return await extract_key_concepts(query_text, graph_interface) 