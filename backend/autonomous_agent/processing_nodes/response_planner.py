import logging
from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel, Field

from .base_node import BaseProcessingNode
from ..processing_nodes.result_evaluator import EvaluatedDocument
from ..graph_components.nodes import QueryNode #, ResponseNode (for historical patterns)
from ..integration_components.graph_interface import GraphInterface
# from llm_services.llm_client import LLMClient

logger = logging.getLogger(__name__)

class PlannedResponseChunk(BaseModel):
    """A chunk of information to be included in the response."""
    document_id: str
    source_id: str
    chunk_text: str
    relevance_score: float
    original_query_aspect_addressed: Optional[str] = None

class ResponsePlan(BaseModel):
    """The plan for generating a response, including structure and content elements."""
    introduction: str
    main_points: List[PlannedResponseChunk]
    conclusion: str
    warnings_or_caveats: List[str] = Field(default_factory=list)
    structure_template: str = "standard_intro_points_conclusion"
    language: Optional[str] = "en"
    style: Optional[str] = "neutral"
    tone: Optional[str] = "informative"
    detail_level: Optional[str] = "balanced"

class ResponsePlannerInput(BaseModel):
    """Input for the ResponsePlanningNode."""
    original_query_node: QueryNode
    evaluated_documents: List[EvaluatedDocument]
    meta_instructions: Dict[str, Any] = Field(default_factory=dict)

class ResponsePlannerOutput(BaseModel):
    """Output from the ResponsePlanningNode."""
    response_plan: ResponsePlan
    metrics: Dict[str, Any] = Field(default_factory=dict)

class ResponsePlanningNode(BaseProcessingNode):
    """Organizes retrieved information into a coherent structure for response generation."""

    async def execute(self, input_data: ResponsePlannerInput, context: Dict[str, Any]) -> ResponsePlannerOutput:
        """
        Creates a plan for generating the final response.

        Args:
            input_data: Contains the query, evaluated documents, and meta-instructions.
            context: Workflow context.

        Returns:
            A ResponsePlan object.
        """
        logger.info(f"[{self.get_name()}] Planning response for query: {input_data.original_query_node.original_user_input[:100]}...")

        graph: GraphInterface = context.get("graph_interface")
        llm_client = context.get("llm_client")

        main_points: List[PlannedResponseChunk] = []
        
        # 1. Identify most relevant information chunks (use relevant_snippets from EvaluatedDocument)
        for eval_doc in sorted(input_data.evaluated_documents, key=lambda d: d.final_relevance_score or 0, reverse=True):
            if eval_doc.relevant_snippets:
                for snippet in eval_doc.relevant_snippets:
                    main_points.append(PlannedResponseChunk(
                        document_id=eval_doc.document_node.id,
                        source_id=eval_doc.document_node.source_id if hasattr(eval_doc.document_node, "source_id") else eval_doc.document_node.source,
                        chunk_text=snippet,
                        relevance_score=eval_doc.final_relevance_score or eval_doc.relevance_score
                    ))
            # else, if no snippets but doc is highly relevant, maybe take a summary (placeholder)
            elif (eval_doc.final_relevance_score or eval_doc.relevance_score) > 0.7:
                content_field = "text_content" if hasattr(eval_doc.document_node, "text_content") else "content"
                doc_content = getattr(eval_doc.document_node, content_field, "")
                if doc_content:
                    # Use LLM to generate a better summary if available
                    if llm_client:
                        try:
                            summary_prompt = f"Summarize the following content in relation to this query: '{input_data.original_query_node.original_user_input}'\n\nContent: {doc_content[:1500]}..."
                            pseudo_snippet = await llm_client.generate_text(summary_prompt, max_tokens=300)
                        except Exception as e:
                            logger.warning(f"LLM summarization failed: {e}. Using text prefix as fallback.")
                            pseudo_snippet = doc_content[:300] + "..." # First 300 chars as fallback
                    else:
                        pseudo_snippet = doc_content[:300] + "..." # First 300 chars
                    
                    main_points.append(PlannedResponseChunk(
                        document_id=eval_doc.document_node.id,
                        source_id=eval_doc.document_node.source_id if hasattr(eval_doc.document_node, "source_id") else eval_doc.document_node.source,
                        chunk_text=pseudo_snippet,
                        relevance_score=eval_doc.final_relevance_score or eval_doc.relevance_score,
                        original_query_aspect_addressed="general_relevance_fallback_snippet"
                    ))

        # Truncate main points if too many, or implement more sophisticated selection/synthesis
        if len(main_points) > 10 and llm_client:
            # If we have an LLM client, try to merge similar points
            merged_points = await self._merge_similar_points(main_points, llm_client)
            main_points = merged_points[:10]  # Still limit to 10 after merging
        else:
            main_points = main_points[:10]  # Limit to top 10 chunks

        # 2. Structure information according to user's meta-instructions
        language = input_data.meta_instructions.get("language", "en")
        style = input_data.meta_instructions.get("style", "neutral")
        custom_format = input_data.meta_instructions.get("format")
        tone = input_data.meta_instructions.get("tone", "informative")
        detail_level = input_data.meta_instructions.get("detail_level", "balanced")

        # 3. Incorporate successful response patterns from similar queries
        historical_structure = None
        if graph and llm_client:
            try:
                # Find historical responses to similar queries
                similar_responses = await graph.find_successful_reformulations(
                    original_query_text=input_data.original_query_node.original_user_input,
                    success_threshold=0.8,
                    limit=3
                )
                
                if similar_responses:
                    # Use historical successful responses as reference
                    historical_structure = await self._extract_response_template(
                        similar_responses, 
                        input_data.original_query_node.original_user_input, 
                        llm_client
                    )
                    logger.info(f"Found {len(similar_responses)} similar successful responses to use as reference")
            except Exception as e:
                logger.warning(f"Error retrieving historical responses: {e}")

        # 4. Plan how to address all aspects of the original question using LLM
        introduction = f"Based on the information retrieved for your query: '{input_data.original_query_node.original_user_input}'."
        conclusion = "Please let me know if you need further clarification."
        structure_template = "standard_intro_points_conclusion"
        warnings_or_caveats = []
        
        # Apply basic format templates first
        if custom_format == "list" and main_points:
            introduction = f"Here is a list of points regarding your query: '{input_data.original_query_node.original_user_input}':"
            structure_template = "list_format"
        elif custom_format == "faq" and main_points:
            introduction = f"Here are frequently asked questions about '{input_data.original_query_node.original_user_input}':"
            structure_template = "faq_format"
        elif custom_format == "step_by_step" and main_points:
            introduction = f"Here's a step-by-step guide regarding '{input_data.original_query_node.original_user_input}':"
            structure_template = "step_by_step_format"

        # Use LLM to generate a more sophisticated plan/outline
        llm_plan = None
        if llm_client:
            try:
                plan_prompt = f"Original Query: {input_data.original_query_node.original_user_input}\n"
                plan_prompt += f"Meta-instructions: Language: {language}, Style: {style}, Format: {custom_format or 'standard'}, Tone: {tone}, Detail Level: {detail_level}\n"
                plan_prompt += f"Key information chunks found:\n"
                for i, chunk in enumerate(main_points[:5]): # Top 5 chunks for brevity
                    plan_prompt += f"{i+1}. From {chunk.source_id} (Score: {chunk.relevance_score:.2f}): {chunk.chunk_text[:150]}...\n"
                
                if historical_structure:
                    plan_prompt += f"\nSuccessful response template from similar queries: {historical_structure}\n"
                    
                plan_prompt += "\nCreate a detailed response plan. This includes: 1. A brief introduction. 2. An ordered list of main points/sections, indicating which chunks support which point. 3. A brief conclusion. 4. Any necessary caveats. Respond ONLY in JSON format: { \"introduction\": \"...\", \"main_points_outline\": [ {\"point_summary\":\"...\", \"supporting_chunk_indices\":[0,1]} ], \"conclusion\": \"...\", \"warnings_or_caveats\": [] }"
                
                # Use generate_text method which is available in the LLM client
                raw_plan = await llm_client.generate_text(plan_prompt, max_tokens=1000)
                
                # Parse and validate the LLM response
                # First check if we got JSON and try to parse it
                if raw_plan.strip().startswith('{') and raw_plan.strip().endswith('}'):
                    try:
                        plan_data = json.loads(raw_plan)
                        # Only update fields specified in the plan_data
                        introduction = plan_data.get("introduction", introduction)
                        conclusion = plan_data.get("conclusion", conclusion)
                        if "main_points_outline" in plan_data:
                            structured_main_points = []
                            for point_outline in plan_data["main_points_outline"]:
                                if "supporting_chunk_indices" in point_outline and "point_summary" in point_outline:
                                    for idx in point_outline["supporting_chunk_indices"]:
                                        if 0 <= idx < len(main_points):
                                            chunk = main_points[idx]
                                            # Create a new chunk with the LLM's point summary as the aspect addressed
                                            structured_main_points.append(PlannedResponseChunk(
                                                document_id=chunk.document_id,
                                                source_id=chunk.source_id,
                                                chunk_text=chunk.chunk_text,
                                                relevance_score=chunk.relevance_score,
                                                original_query_aspect_addressed=point_outline["point_summary"]
                                            ))
                        
                        # If we successfully created structured points, use them
                        if structured_main_points:
                            main_points = structured_main_points
                            
                            # Additional properties if provided
                            structure_template = plan_data.get("structure", structure_template)
                            if "warnings_or_caveats" in plan_data:
                                warnings_or_caveats = plan_data["warnings_or_caveats"]
                            if "style" in plan_data:
                                style = plan_data["style"]
                            
                            logger.info(f"[{self.get_name()}] Successfully parsed JSON response plan from LLM")
                            self._enhance_plan_with_source_docs(
                                ResponsePlan(
                                    introduction=introduction,
                                    main_points=main_points,
                                    conclusion=conclusion,
                                    warnings_or_caveats=warnings_or_caveats,
                                    structure_template=structure_template,
                                    language=language,
                                    style=style,
                                    tone=tone,
                                    detail_level=detail_level
                                ),
                                input_data.evaluated_documents
                            )
                        else:
                            # Fall back to basic structure without using custom parser
                            logger.info(f"[{self.get_name()}] Using default structure after JSON parse failure")
                    except json.JSONDecodeError:
                        logger.warning(f"[{self.get_name()}] Failed to parse LLM response as JSON: {raw_plan[:100]}...")
                        # Fall back to basic structure without using custom parser
                        logger.info(f"[{self.get_name()}] Using default structure after JSON parse failure")
                else:
                    # If not JSON, just extract any useful text
                    logger.warning(f"[{self.get_name()}] Response was not in JSON format")
                    
                    # Look for introduction section
                    intro_match = raw_plan.lower().find("introduction")
                    if intro_match > -1:
                        next_section = raw_plan.find("\n\n", intro_match)
                        if next_section > -1:
                            introduction = raw_plan[intro_match + 12:next_section].strip()
                    
                    # Look for conclusion section
                    conclusion_match = raw_plan.lower().find("conclusion")
                    if conclusion_match > -1:
                        next_section = raw_plan.find("\n\n", conclusion_match)
                        if next_section > -1:
                            conclusion = raw_plan[conclusion_match + 10:next_section].strip()
                        else:
                            conclusion = raw_plan[conclusion_match + 10:].strip()
                
                # Enhance the plan with insights from source docs
                self._enhance_plan_with_source_docs(
                    ResponsePlan(
                        introduction=introduction,
                        main_points=main_points,
                        conclusion=conclusion,
                        warnings_or_caveats=warnings_or_caveats,
                        structure_template=structure_template,
                        language=language,
                        style=style,
                        tone=tone,
                        detail_level=detail_level
                    ),
                    input_data.evaluated_documents
                )
                
                # If specified, adjust tone/style according to user preferences
                if style_preference and llm_client:
                    try:
                        style_prompt = f"""
Adjust the style of my response plan to be {style_preference}.
Original plan details:
- Introduction: {introduction}
- Main points: {len(main_points)} points
- Conclusion: {conclusion}

Provide a short style guide for how to transform this content to the desired style.
"""
                        style_guide = await llm_client.generate_text(style_prompt, max_tokens=300)
                        if style_guide:
                            style = f"{style_preference}: {style_guide[:100]}..."
                    except Exception as e:
                        logger.warning(f"Style adjustment failed: {e}")
                        style = style_preference  # Just use the preference without a guide

            except Exception as e:
                logger.error(f"[{self.get_name()}] LLM response planning failed: {e}. Using basic plan.")
                # Fall back to basic planning method
                response_plan = ResponsePlan(
                    introduction=introduction,
                    main_points=main_points,
                    conclusion=conclusion,
                    warnings_or_caveats=warnings_or_caveats,
                    structure_template=structure_template,
                    language=language,
                    style=style,
                    tone=tone,
                    detail_level=detail_level
                )
                self._create_basic_plan(input_data.evaluated_documents, response_plan)
                return ResponsePlannerOutput(response_plan=response_plan)

        response_plan = ResponsePlan(
            introduction=introduction,
            main_points=main_points,
            conclusion=conclusion,
            warnings_or_caveats=warnings_or_caveats,
            structure_template=structure_template,
            language=language,
            style=style,
            tone=tone,
            detail_level=detail_level
        )

        logger.info(f"[{self.get_name()}] Response plan created. Introduction: {response_plan.introduction[:50]}... {len(response_plan.main_points)} main points.")

        return ResponsePlannerOutput(response_plan=response_plan)
        
    async def _extract_response_template(
        self, 
        similar_responses: List[Dict[str, Any]], 
        current_query: str,
        llm_client
    ) -> Optional[str]:
        """Extract a response template from similar historical responses."""
        if not similar_responses or not llm_client:
            return None
            
        try:
            # Format the similar responses for the LLM
            similar_responses_text = "\n".join([
                f"Original Query: {resp['original_query']}\n"
                f"Reformulated Query: {resp['reformulated_query']}\n"
                f"Success Score: {resp['success_score']}\n"
                f"---"
                for resp in similar_responses
            ])
            
            template_prompt = (
                f"Based on these successful query-response pairs:\n\n{similar_responses_text}\n\n"
                f"Extract a high-level response template or pattern that could be applied to this new query: "
                f"'{current_query}'\n\nResponse template (include placeholders for specific information):"
            )
            
            template = await llm_client.generate_text(template_prompt, max_tokens=300)
            return template
        except Exception as e:
            logger.warning(f"Error extracting response template: {e}")
            return None
            
    async def _merge_similar_points(
        self, 
        points: List[PlannedResponseChunk], 
        llm_client
    ) -> List[PlannedResponseChunk]:
        """Merge similar points to reduce redundancy using LLM."""
        if len(points) <= 10 or not llm_client:
            return points
            
        try:
            # Prepare chunks for the LLM
            chunks_text = "\n".join([
                f"{i}. Source: {chunk.source_id}, Score: {chunk.relevance_score:.2f}\n"
                f"Text: {chunk.chunk_text[:200]}...\n"
                for i, chunk in enumerate(points)
            ])
            
            merge_prompt = (
                f"Analyze these information chunks and identify groups of related/similar chunks "
                f"that could be merged:\n\n{chunks_text}\n\n"
                f"Return a JSON array of groups, where each group is an array of chunk indices "
                f"that should be merged. Include only indices that should be merged, not standalone chunks. "
                f"Example response format: [[0, 3, 5], [2, 7]]"
            )
            
            merge_response = await llm_client.generate_text(merge_prompt, response_format="json_array")
            merge_groups = json.loads(merge_response)
            
            # Create merged chunks and keep standalone chunks
            processed_indices = set()
            result = []
            
            # Process merge groups
            for group in merge_groups:
                if not group or not isinstance(group, list):
                    continue
                    
                valid_indices = [i for i in group if 0 <= i < len(points)]
                if not valid_indices:
                    continue
                    
                chunks_to_merge = [points[i] for i in valid_indices]
                processed_indices.update(valid_indices)
                
                # Merge the chunks
                merged_text = "\n".join([c.chunk_text for c in chunks_to_merge])
                
                # Use LLM to synthesize the merged chunks
                synthesis_prompt = f"Synthesize this related information into a coherent paragraph, removing redundancies:\n\n{merged_text}"
                synthesized_text = await llm_client.generate_text(synthesis_prompt, max_tokens=400)
                
                # Create merged chunk
                result.append(PlannedResponseChunk(
                    document_id="|".join([c.document_id for c in chunks_to_merge]),
                    source_id="|".join(set([c.source_id for c in chunks_to_merge])),
                    chunk_text=synthesized_text,
                    relevance_score=max([c.relevance_score for c in chunks_to_merge]),
                    original_query_aspect_addressed="merged_content"
                ))
            
            # Add standalone chunks
            for i, chunk in enumerate(points):
                if i not in processed_indices:
                    result.append(chunk)
                    
            # Sort by relevance
            result.sort(key=lambda x: x.relevance_score, reverse=True)
            return result
        except Exception as e:
            logger.warning(f"Error merging similar points: {e}")
            return points 

    def _create_basic_plan(self, evaluated_documents, response_plan):
        """
        Create a basic response plan when LLM-based planning fails.
        
        Args:
            evaluated_documents: List of evaluated documents
            response_plan: Existing response plan to modify
        """
        logger.info(f"[{self.get_name()}] Creating basic response plan")
        
        # Add a generic introduction if not already set
        if not response_plan.introduction:
            response_plan.introduction = "Here's what I found based on your query. The following information comes directly from the documents I retrieved:"
        
        # Make sure we have main points by using document content directly
        if not response_plan.main_points:
            for doc in sorted(evaluated_documents, key=lambda d: d.final_relevance_score or 0, reverse=True):
                # Try to get relevant_snippets first since they're more targeted
                if hasattr(doc, 'relevant_snippets') and doc.relevant_snippets:
                    # Add each relevant snippet as a separate point for better structure
                    for snippet in doc.relevant_snippets:
                        if snippet and len(snippet.strip()) > 10:  # Ensure it's not just whitespace
                            response_plan.main_points.append(PlannedResponseChunk(
                                document_id=doc.document_node.id,
                                source_id=getattr(doc.document_node, "source_id", "") or getattr(doc.document_node, "source", ""),
                                chunk_text=snippet,
                                relevance_score=doc.final_relevance_score or doc.relevance_score or 0.7,
                                original_query_aspect_addressed="relevant_snippet"
                            ))
                else:
                    # Fall back to full document content if no snippets
                    content_field = "text_content" if hasattr(doc.document_node, "text_content") else "content"
                    doc_content = getattr(doc.document_node, content_field, "")
                    
                    if doc_content:
                        # Extract more meaningful chunks by splitting on paragraphs
                        paragraphs = [p for p in doc_content.split('\n\n') if p.strip()]
                        if paragraphs:
                            # Use paragraphs if available (more meaningful than arbitrary character limits)
                            for i, para in enumerate(paragraphs[:3]):  # Limit to first 3 paragraphs
                                if len(para.strip()) > 10:  # Ensure it's not just whitespace
                                    response_plan.main_points.append(PlannedResponseChunk(
                                        document_id=doc.document_node.id,
                                        source_id=getattr(doc.document_node, "source_id", "") or getattr(doc.document_node, "source", ""),
                                        chunk_text=para[:500] + ("..." if len(para) > 500 else ""),
                                        relevance_score=doc.final_relevance_score or doc.relevance_score or 0.5 - (i * 0.1),  # Decrease score for later paragraphs
                                        original_query_aspect_addressed=f"paragraph_{i+1}"
                                    ))
                        else:
                            # Fall back to character-based excerpt if no paragraphs
                            excerpt = doc_content[:500] + "..." if len(doc_content) > 500 else doc_content
                            response_plan.main_points.append(PlannedResponseChunk(
                                document_id=doc.document_node.id,
                                source_id=getattr(doc.document_node, "source_id", "") or getattr(doc.document_node, "source", ""),
                                chunk_text=excerpt,
                                relevance_score=doc.final_relevance_score or doc.relevance_score or 0.5,
                                original_query_aspect_addressed="document_excerpt"
                            ))
        
        # Add a generic conclusion if not already set
        if not response_plan.conclusion:
            response_plan.conclusion = "The information above comes directly from the documents I retrieved. Let me know if you need any clarification or have follow-up questions."
        
        # Add a caveat about basic planning if none exist
        if not response_plan.warnings_or_caveats:
            response_plan.warnings_or_caveats = ["This response is based directly on the content of the documents I retrieved, presented in order of relevance."]
            
        # Limit to a reasonable number of points
        response_plan.main_points = response_plan.main_points[:10]  # Allow more points to ensure comprehensive context
        
        return response_plan

    def _enhance_plan_with_source_docs(self, response_plan, evaluated_documents):
        """Enhance the plan with information from source documents."""
        # This should analyze the source documents to find additional context or evidence
        # For now, implementation is simple - just verify/update the source IDs and document IDs
        for point in response_plan.main_points:
            if not point.source_id:
                for doc in evaluated_documents:
                    if doc.document_node.id == point.document_id:
                        point.source_id = doc.document_node.source_id if hasattr(doc.document_node, "source_id") else doc.document_node.source
        
        return response_plan 