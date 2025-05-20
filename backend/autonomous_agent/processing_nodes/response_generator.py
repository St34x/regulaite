import logging
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .base_node import BaseProcessingNode
from ..processing_nodes.response_planner import ResponsePlan, PlannedResponseChunk
from ..graph_components.nodes import ResponseNode, QueryNode, ConceptNode, DocumentNode
from ..graph_components.edges import LedToEdge, GeneratedFromEdge, ContainsEdge # (Response)-[CONTAINS]->Concept
from ..integration_components.graph_interface import GraphInterface
from llm_services.llm_client import LLMClient

logger = logging.getLogger(__name__)

class ResponseGeneratorInput(BaseModel):
    response_plan: ResponsePlan
    original_query_node: QueryNode # To link the response to
    # The documents that contributed to the plan, for creating GENERATED_FROM edges
    contributing_documents: List[DocumentNode] 
    # Meta-instructions for LLM (already in plan, but might be useful separately)
    meta_instructions: Dict[str, Any] 

class GeneratedResponse(BaseModel):
    response_node: ResponseNode
    # Concepts extracted from the generated response text itself
    newly_extracted_concepts: List[ConceptNode] = [] 

class ResponseGenerationNode(BaseProcessingNode):
    """Creates the final response and updates the knowledge graph."""
    
    def __init__(self, node_config: Optional[Dict[str, Any]] = None):
        super().__init__(node_config)
        # Configure default parameters
        config = node_config or {}
        self.generation_temperature = config.get("generation_temperature", 0.7)
        self.max_generation_tokens = config.get("max_generation_tokens", 1000)
        self.concept_extraction_enabled = config.get("concept_extraction_enabled", True)
        self.concept_extraction_temperature = config.get("concept_extraction_temperature", 0.3)

    async def execute(self, input_data, context):
        try:
            # Extract required fields from input_data safely
            if isinstance(input_data, dict):
                plan = input_data.get('response_plan')
                original_query_node = input_data.get('original_query_node')
                contributing_documents = input_data.get('contributing_documents', [])
            else:
                plan = input_data.response_plan
                original_query_node = input_data.original_query_node
                contributing_documents = input_data.contributing_documents

            if not plan:
                logger.error(f"[ResponseGenerationNode] No response plan found in the input data.")
                raise ValueError("Missing response plan")

            # Determine query id and original user input safely
            if original_query_node is None:
                logger.warning(f"[ResponseGenerationNode] original_query_node is None, using default values")
                query_id = "unknown"
                original_user_input = "Unknown query"
            elif isinstance(original_query_node, dict):
                query_id = original_query_node.get('id', 'unknown')
                original_user_input = original_query_node.get('original_user_input', '')
            else:
                query_id = original_query_node.id
                original_user_input = original_query_node.original_user_input

            logger.info(f"[{self.get_name()}] Generating response based on plan for query ID: {query_id}")

            graph: Optional[GraphInterface] = context.get("graph_interface")
            llm_client: Optional[LLMClient] = context.get("llm_client")

            # 1. Generate natural language response following the response plan
            final_response_text = ""
            
            # Construct a detailed prompt for the LLM
            if llm_client:
                generation_prompt = f"You are an AI assistant. Generate a response based on the following plan and information.\n\n"
                generation_prompt += f"Original User Query: {original_user_input}\n\n"
                
                # Add formatting instructions if available
                format_instructions = []
                if plan.style:
                    format_instructions.append(f"Style: {plan.style}")
                if format_instructions:
                    generation_prompt += f"Format Instructions: {', '.join(format_instructions)}\n\n"
                
                # Add structural guidance
                if plan.structure_template:
                    generation_prompt += f"Follow this structure: {plan.structure_template}\n\n"
                
                # Add introduction if available
                if plan.introduction:
                    generation_prompt += f"Introduction: {plan.introduction}\n\n"
                
                # Add main content points
                generation_prompt += "Main Points to cover (synthesize these into a coherent response):\n"
                for i, chunk in enumerate(plan.main_points):
                    source_info = f"from {chunk.source_id}" if getattr(chunk, 'source_id', None) else "from analysis"
                    generation_prompt += f"- Point {i+1} ({source_info}): {chunk.chunk_text}\n"
                generation_prompt += "\n"
                
                # Add conclusion and caveats
                if plan.conclusion:
                    generation_prompt += f"Conclusion: {plan.conclusion}\n\n"
                if plan.warnings_or_caveats:
                    generation_prompt += f"Important Caveats: {'; '.join(plan.warnings_or_caveats)}\n\n"
                
                # Final instructions
                generation_prompt += """
Instructions for Response Generation:
1. Synthesize the information into a cohesive, natural-sounding response
2. Ensure the response is well-organized and follows the plan
+3. IMPORTANT: Incorporate the specific content from the provided document chunks - this is critical
+4. Use direct information from the retrieved documents to support your answer
+5. Do not use phrases like "According to the information provided" - instead, integrate the content naturally
+6. Your response MUST be based on the retrieved information, not general knowledge
"""

                try:
                    # Generate the response using the LLM
                    final_response_text = await llm_client.generate_text(
                        prompt=generation_prompt,
                        temperature=self.generation_temperature,
                        max_tokens=self.max_generation_tokens
                    )
                    logger.info(f"[{self.get_name()}] Successfully generated response using LLM")
                except Exception as e:
                    logger.error(f"[{self.get_name()}] LLM response generation failed: {e}. Using fallback response.")
                    final_response_text = self._generate_fallback_response(plan, original_user_input)
            else:
                # Fallback if no LLM: Concatenate planned parts (basic template)
                final_response_text = self._generate_fallback_response(plan, original_user_input)
            
            logger.info(f"[{self.get_name()}] Generated response text (first 100 chars): {final_response_text[:100]}...")

            # 2. Create ResponseNode in the graph database
            # Get the query_id safely (required field for ResponseNode)
            if original_query_node is None:
                query_id = "unknown_query_id"
            else:
                query_id = original_query_node.get('id') if isinstance(original_query_node, dict) else original_query_node.id

            response_node = ResponseNode(
                response_text=final_response_text,
                query_id=query_id,
                attributes={
                    "language": plan.language,
                    "style": plan.style,
                    "generation_model": getattr(llm_client, "default_model_name", "gpt-3.5-turbo") if llm_client else "template",
                    "source_document_count": len(contributing_documents),
                    "source_document_ids": [doc.id for doc in contributing_documents if hasattr(doc, 'id')]
                },
                metadata={}  # Explicitly set as empty dict instead of letting it be serialized as string
            )
            
            newly_extracted_concepts: List[ConceptNode] = []
            
            # 3. Update the graph if available
            if graph:
                try:
                    # Add the response node to the graph
                    updated_response_node = await graph.add_node(response_node)
                    response_node = updated_response_node  # Use the updated node with DB-assigned fields
                    logger.info(f"[{self.get_name()}] Created ResponseNode ID: {response_node.id}")

                    # Connect response to query (LED_TO)
                    if original_query_node is not None:
                        source_id = original_query_node.get('id') if isinstance(original_query_node, dict) else original_query_node.id
                        led_to_edge = LedToEdge(
                            source_node_id=source_id,
                            target_node_id=response_node.id,
                            properties={
                                "confidence": plan.confidence if hasattr(plan, "confidence") else 0.7,
                                "document_count": len(contributing_documents)
                            } 
                        )
                        await graph.add_edge(led_to_edge)
                        logger.info(f"[{self.get_name()}] Linked response to query {source_id}")
                    else:
                        logger.warning(f"[{self.get_name()}] Cannot link response to query - original_query_node is None")

                    # Connect response to contributing documents (GENERATED_FROM)
                    for doc_node in contributing_documents:
                        contribution_score = 0.5  # Default
                        relevant_chunks = [c for c in plan.main_points if hasattr(c, "document_id") and c.document_id == (doc_node.get('id') if isinstance(doc_node, dict) else doc_node.id)]
                        if relevant_chunks:
                            contribution_score = max(chunk.relevance_score for chunk in relevant_chunks if hasattr(chunk, "relevance_score"))
                        gen_from_edge = GeneratedFromEdge(
                            source_node_id=response_node.id,
                            target_node_id=(doc_node.get('id') if isinstance(doc_node, dict) else doc_node.id),
                            properties={"contribution_score": contribution_score}
                        )
                        await graph.add_edge(gen_from_edge)
                    
                    logger.info(f"[{self.get_name()}] Linked ResponseNode to QueryNode and {len(contributing_documents)} DocumentNodes.")
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Error updating graph: {e}", exc_info=True)

            # 4. Extract new concepts from the generated response if enabled
            if llm_client and graph and self.concept_extraction_enabled and final_response_text:
                try:
                    # Prepare the concept extraction prompt
                    concept_extraction_prompt = f"""
Extract key concepts (entities, topics, ideas) from the following text. 
For each concept, provide a brief definition.

Text:
{final_response_text[:1500]}  # Limit length to avoid token overflow

Format your response as a JSON array of objects with "name" and "definition" fields:
[
  {{"name": "Concept1", "definition": "Brief definition of Concept1"}},
  {{"name": "Concept2", "definition": "Brief definition of Concept2"}},
  ...
]

Include only substantive concepts (no general words).
Limit to at most 5 important concepts.
JSON response:
"""

                    # Get concepts from LLM
                    try:
                        # Try different API methods that might be available
                        concepts_json_str = None
                        concepts_data = []
                        
                        # First try generate_text if it exists
                        if hasattr(llm_client, 'generate_text'):
                            concepts_json_str = await llm_client.generate_text(
                                prompt=concept_extraction_prompt,
                                temperature=self.concept_extraction_temperature,
                                max_tokens=500
                            )
                        # If not, try with different OpenAI APIs that might be available
                        elif hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions') and hasattr(llm_client.chat.completions, 'create'):
                            # Modern OpenAI client
                            try:
                                # Check if the create method is coroutine function (async)
                                import inspect
                                create_method = llm_client.chat.completions.create
                                if inspect.iscoroutinefunction(create_method):
                                    # Use await for async function
                                    response = await create_method(
                                        model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": concept_extraction_prompt}],
                                        temperature=self.concept_extraction_temperature,
                                        max_tokens=500
                                    )
                                else:
                                    # Call synchronously
                                    response = create_method(
                                        model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": concept_extraction_prompt}],
                                        temperature=self.concept_extraction_temperature,
                                        max_tokens=500
                                    )
                                # Handle different response formats
                                if hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
                                    concepts_json_str = response.choices[0].message.content
                                elif isinstance(response, dict) and 'choices' in response:
                                    concepts_json_str = response['choices'][0]['message']['content']
                            except Exception as e:
                                logger.error(f"[{self.get_name()}] Error using OpenAI client: {e}", exc_info=True)
                        elif hasattr(llm_client, 'Completion') and hasattr(llm_client.Completion, 'create'):
                            # Legacy OpenAI client
                            try:
                                # Check if the create method is coroutine function (async)
                                import inspect
                                create_method = llm_client.Completion.create
                                if inspect.iscoroutinefunction(create_method):
                                    # Use await for async function
                                    response = await create_method(
                                        engine="text-davinci-003",
                                        prompt=concept_extraction_prompt,
                                        temperature=self.concept_extraction_temperature,
                                        max_tokens=500
                                    )
                                else:
                                    # Call synchronously
                                    response = create_method(
                                        engine="text-davinci-003",
                                        prompt=concept_extraction_prompt,
                                        temperature=self.concept_extraction_temperature,
                                        max_tokens=500
                                    )
                                
                                # Handle different response formats
                                if hasattr(response, 'choices') and hasattr(response.choices[0], 'text'):
                                    concepts_json_str = response.choices[0].text
                                elif isinstance(response, dict) and 'choices' in response:
                                    concepts_json_str = response['choices'][0]['text']
                            except Exception as e:
                                logger.error(f"[{self.get_name()}] Error using OpenAI Completion client: {e}", exc_info=True)
                        
                        # Extract and parse JSON if we got a response
                        if concepts_json_str:
                            json_start = concepts_json_str.find('[')
                            json_end = concepts_json_str.rfind(']')
                            
                            if json_start >= 0 and json_end > json_start:
                                concepts_json = concepts_json_str[json_start:json_end+1]
                                try:
                                    concepts_data = json.loads(concepts_json)
                                except json.JSONDecodeError:
                                    logger.warning(f"[{self.get_name()}] Failed to parse concepts JSON")
                                    concepts_data = []
                        else:
                            logger.warning(f"[{self.get_name()}] No API method found to extract concepts - skipping concept extraction")
                    
                    except (json.JSONDecodeError, Exception) as e:
                        logger.error(f"[{self.get_name()}] Error extracting concepts: {e}", exc_info=True)
                    
                    # Process each concept
                    for concept_data in concepts_data:
                        concept_name = concept_data.get("name", "").strip()
                        concept_definition = concept_data.get("definition", "").strip()
                        
                        if concept_name:
                            # Create or retrieve the concept node
                            # First check if it already exists
                            existing_concepts = await graph.find_concepts_by_name_exact(concept_name)
                            
                            if existing_concepts:
                                concept_node = existing_concepts[0]
                                # If it exists but has no definition, update it
                                if concept_definition and not concept_node.attributes.get("definition"):
                                    await graph.update_node_properties(
                                        concept_node.id,
                                        {"attributes.definition": concept_definition}
                                    )
                            else:
                                # Create new concept node
                                concept_node = ConceptNode(
                                    name=concept_name,
                                    attributes={
                                        "definition": concept_definition,
                                        "source": "response_extraction",
                                        "extraction_confidence": 0.7
                                    }
                                )
                                concept_node = await graph.add_node(concept_node)
                                newly_extracted_concepts.append(concept_node)
                            
                            # Link the response to the concept
                            contains_edge = ContainsEdge(
                                source_node_id=response_node.id,
                                target_node_id=concept_node.id,
                                properties={"extraction_method": "llm"}
                            )
                            await graph.add_edge(contains_edge)
                    
                    logger.info(f"[{self.get_name()}] Extracted {len(newly_extracted_concepts)} new concepts from response")
                except Exception as e:
                    logger.error(f"[{self.get_name()}] Concept extraction failed: {e}", exc_info=True)

            return GeneratedResponse(
                response_node=response_node,
                newly_extracted_concepts=newly_extracted_concepts
            )
        except Exception as e:
            logger.error(f"[{self.get_name()}] Execution failed: {e}", exc_info=True)
            raise
    
    def _generate_fallback_response(self, plan: ResponsePlan, original_query: str) -> str:
        """Creates a basic templated response when LLM generation fails."""
        final_response_text = f"Here's what I found about: {original_query}\n\n"
        
        if plan.introduction:
            final_response_text += plan.introduction + "\n\n"
            
        # Always include retrieved content to ensure RAG works even in fallback mode
        if plan.main_points:
            final_response_text += "Key information from retrieved sources:\n\n"
            for i, chunk in enumerate(plan.main_points):
                # Only include the source ID if it exists
                source_info = f" (Source: {chunk.source_id})" if hasattr(chunk, "source_id") and chunk.source_id else ""
                # Ensure we include the full chunk text
                chunk_text = chunk.chunk_text.strip()
                final_response_text += f"• {chunk_text}{source_info}\n\n"
        else:
            final_response_text += "I couldn't find specific information to answer your query.\n\n"
            
        if plan.conclusion:
            final_response_text += plan.conclusion + "\n\n"
            
        if plan.warnings_or_caveats:
            final_response_text += "Note: " + "; ".join(plan.warnings_or_caveats) + "\n"
            
        if not plan.main_points:
            final_response_text += "I couldn't find specific information to answer your query. Please try rephrasing or providing more details."
            
        return final_response_text 