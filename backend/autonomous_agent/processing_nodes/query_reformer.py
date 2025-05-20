import logging
from typing import Any, Dict, List, Optional
import json
from datetime import datetime

from pydantic import BaseModel, Field

from .base_node import BaseProcessingNode, ProcessingStepResult
from ..graph_components.nodes import QueryNode, ConceptNode
from ..graph_components.edges import ReformulatedAsEdge
from ..integration_components.graph_interface import GraphInterface
from llm_services.llm_client import LLMClient

logger = logging.getLogger(__name__)

class QueryReformerInput(BaseModel):
    """Input for the QueryReformulationNode."""
    # The query that led to insufficient results
    previous_query_node: QueryNode 
    # Gaps identified by the ResultEvaluationNode
    information_gaps: List[str] 
    # Concepts related to the original query or identified as potentially relevant
    related_concepts: List[ConceptNode] = [] 
    # Meta-instructions from original input, if they should guide reformulation
    meta_instructions: Dict[str, Any] = Field(default_factory=dict)
    # History of reformulations in this cycle to avoid loops or apply different strategies
    reformulation_history: List[Dict[str, Any]] = Field(default_factory=list)

class QueryReformerOutput(ProcessingStepResult):
    """Output from query reformulation."""
    step_name: str = "query_reformulation"
    status: str
    reformulated_query: str
    reformulation_strategy: str
    rationale: str
    used_concepts: Optional[List[Dict[str, Any]]] = None

class QueryReformulationNode(BaseProcessingNode):
    """Generates better queries when initial results are insufficient, focusing on filling gaps."""

    async def execute(self, input_data: Any, context: Dict[str, Any]) -> QueryReformerOutput:
        """
        Execute query reformulation.
        
        Args:
            input_data: Input data containing original query, retrieval results, and evaluation
            context: Workflow context
            
        Returns:
            Reformulated query and metadata
        """
        # Extract necessary components from input
        original_query, retrieval_result, evaluation_result = self._extract_from_input(input_data)
        
        # Get query text
        query_text = self._get_query_text(original_query)
        
        # Get LLM client from context
        llm_client = context.get("llm_client")
        if not llm_client:
            raise ValueError("LLM client is required for query reformulation")
            
        # Get graph interface from context
        graph_interface = context.get("graph_interface")
        
        # Check if reformulation is needed
        needs_reformulation = True  # Default to True to ensure we continue with reformulation
        
        # First check for needs_reformulation flag
        if hasattr(evaluation_result, 'needs_reformulation'):
            needs_reformulation = evaluation_result.needs_reformulation
        # Fallback to requires_reformulation for compatibility
        elif hasattr(evaluation_result, 'requires_reformulation'):
            needs_reformulation = evaluation_result.requires_reformulation
            
        if not needs_reformulation:
            logger.info("Query reformulation not needed according to evaluation")
            return QueryReformerOutput(
                status="skipped",
                reformulated_query=query_text,
                reformulation_strategy="none",
                rationale="Evaluation indicated reformulation was not needed"
            )
        
        # Identify information gaps
        information_gaps = []
        if hasattr(evaluation_result, 'information_gaps'):
            information_gaps = evaluation_result.information_gaps
        
        # Get related concepts if available
        related_concepts = []
        if hasattr(evaluation_result, 'related_concepts'):
            related_concepts = evaluation_result.related_concepts
            
        # Track the reformulation iteration in context
        reformulation_count = context.get('reformulation_count', 0)
        
        # Choose reformulation strategy based on available data
        if information_gaps and llm_client:
            logger.info(f"Using gap-focused reformulation with {len(information_gaps)} identified gaps")
            reformulated_query = await self._generate_gap_focused_query(query_text, information_gaps, llm_client)
            strategy = "gap_focused"
            rationale = f"Reformulated to address information gaps: {', '.join(information_gaps[:2])}"
        elif related_concepts and llm_client:
            logger.info(f"Using concept expansion with {len(related_concepts)} related concepts")
            # Use concept objects or just the names based on what's available
            concept_objects = []
            for concept in related_concepts:
                if isinstance(concept, dict) and "name" in concept:
                    # Create minimal concept objects from dictionaries
                    from ..graph_components.nodes import ConceptNode
                    concept_obj = ConceptNode(
                        id=concept.get("id", f"temp-{len(concept_objects)}"),
                        name=concept["name"],
                        definition=concept.get("definition", "")
                    )
                    concept_objects.append(concept_obj)
                else:
                    concept_objects.append(concept)
                    
            reformulated_query = await self._apply_concept_expansion(query_text, concept_objects, llm_client)
            strategy = "concept_expansion"
            rationale = f"Expanded with related concepts: {', '.join([c.name if hasattr(c, 'name') else str(c) for c in concept_objects[:2]])}"
        else:
            # Fallback - ensure the query is different enough from the original
            logger.info(f"Using fallback reformulation strategy (iteration {reformulation_count})")
            reformulated_query = self._ensure_query_difference(query_text, information_gaps, reformulation_count)
            strategy = "fallback"
            rationale = f"Applied general reformulation techniques (iteration {reformulation_count})"
            
        # Create a new QueryNode for the reformulated query
        if graph_interface and hasattr(original_query, "id"):
            try:
                from ..graph_components.nodes import QueryNode
                from ..graph_components.edges import ReformulatedAsEdge
                
                # Create the new query node
                new_query = QueryNode(
                    original_user_input=getattr(original_query, "original_user_input", query_text),
                    query_text=reformulated_query,
                    is_reformulation=True,
                    reformulated_from_id=original_query.id,
                    reformulated_query_text=reformulated_query,
                    attributes={
                        "reformulation_strategy": strategy,
                        "reformulation_iteration": reformulation_count,
                        "source": "autonomous_agent"
                    }
                )
                
                # Persist to graph
                persisted_query = await graph_interface.add_node(new_query)
                
                # Create edge between original and reformulated query
                edge = ReformulatedAsEdge(
                    source_node_id=original_query.id,
                    target_node_id=persisted_query.id,
                    properties={
                        "strategy": strategy,
                        "rationale": rationale,
                        "iteration": reformulation_count
                    }
                )
                await graph_interface.add_edge(edge)
                
                logger.info(f"Created and persisted reformulated query with ID: {persisted_query.id}")
                
                # Return the reformulated query node itself
                return persisted_query
                
            except Exception as e:
                logger.error(f"Error persisting reformulated query: {e}", exc_info=True)
                # Continue with returning the standard output
                
        # Return standard output if we didn't return the persisted query node
        return QueryReformerOutput(
            status="completed",
            reformulated_query=reformulated_query,
            reformulation_strategy=strategy,
            rationale=rationale,
            used_concepts=[c.name if hasattr(c, "name") else str(c) for c in related_concepts[:3]] if related_concepts else None
        )
    
    def _extract_from_input(self, input_data):
        """Extract components from input data."""
        if isinstance(input_data, dict):
            # Try to get the components from the dictionary
            original_query = input_data.get('original_query', input_data.get('query', None))
            retrieval_result = input_data.get('retrieval_result', None)
            evaluation_result = input_data.get('evaluation_result', None)
            return original_query, retrieval_result, evaluation_result
        
        # If not a dictionary, try to access attributes
        original_query = getattr(input_data, 'original_query', 
                        getattr(input_data, 'query', None))
        retrieval_result = getattr(input_data, 'retrieval_result', None)
        evaluation_result = getattr(input_data, 'evaluation_result', None)
        
        return original_query, retrieval_result, evaluation_result
    
    def _get_query_text(self, query_node):
        """Extract the query text from the query node."""
        if isinstance(query_node, str):
            return query_node
            
        # Try to get the query text from the node
        query_text = getattr(query_node, 'query_text', 
                    getattr(query_node, 'reformulated_query_text', 
                    getattr(query_node, 'original_user_input', str(query_node))))
        
        return query_text
    
    async def _apply_basic_gap_strategy(self, original_query: str, gaps: List[str]) -> str:
        """Apply a basic gap-filling strategy without LLM."""
        if not gaps:
            return f"{original_query} (refined)"
            
        # Use the first gap to augment the query
        first_gap = gaps[0]
        # Extract key terms from the gap (simple approach: first few words)
        gap_terms = first_gap.split(' ')[:5]  # First 5 words
        
        return f"{original_query} focusing on {' '.join(gap_terms)}"
    
    def _ensure_query_difference(self, original_query: str, gaps: List[str], iteration: int) -> str:
        """Ensure the reformulated query is different from the original."""
        suffixes = [
            " (with more specific details)",
            " (with broader context)",
            " (focusing on technical aspects)",
            " (considering alternatives)",
            " (clarified)",
        ]
        
        # Use modulo to cycle through suffixes if we have many iterations
        suffix_idx = (iteration - 1) % len(suffixes)
        
        # Apply a suffix to make it different
        modified_query = original_query + suffixes[suffix_idx]
        
        # If we have gaps, also append some gap information
        if gaps and iteration > 1:  # For second+ iterations, be more aggressive
            gap_idx = (iteration - 1) % len(gaps)
            gap_terms = gaps[gap_idx].split(' ')[:3]  # First 3 words of the selected gap
            modified_query += f" including {' '.join(gap_terms)}"
            
        return modified_query
    
    async def _generate_gap_focused_query(self, original_query: str, gaps: List[str], llm_client) -> str:
        """Generate a gap-focused query using LLM with specialized prompt."""
        try:
            # Create a more directed prompt focusing specifically on gaps
            gap_prompt = (
                f"I need to reformulate this search query: \"{original_query}\"\n\n"
                f"The search returned insufficient information specifically about:\n"
            )
            
            # Add each gap as a bullet point
            for gap in gaps:
                gap_prompt += f"- {gap}\n"
                
            gap_prompt += (
                f"\nPlease rewrite the query to specifically target these missing information gaps. "
                f"The query should be direct and focused on retrieving the missing information. "
                f"Do not make the query overly complex or verbose."
            )
            
            reformulated = await llm_client.generate_text(gap_prompt, max_tokens=200)
            if reformulated and reformulated.strip() and reformulated.strip().lower() != original_query.lower():
                return reformulated.strip()
                
            # If we got back something unusable, fall back to basic strategy
            return self._apply_basic_gap_strategy(original_query, gaps)
            
        except Exception as e:
            logger.error(f"Gap-focused query generation failed: {e}")
            return self._apply_basic_gap_strategy(original_query, gaps)
    
    async def _apply_concept_expansion(self, original_query: str, concepts: List[ConceptNode], llm_client) -> str:
        """Expand the query using related concepts."""
        if not concepts or not llm_client:
            return original_query
            
        try:
            # Extract concept names and definitions
            concept_info = []
            for concept in concepts:
                info = f"{concept.name}"
                if hasattr(concept, "definition") and concept.definition:
                    info += f" - {concept.definition[:100]}"  # Limit definition length
                concept_info.append(info)
                
            concept_text = "\n".join([f"- {c}" for c in concept_info[:3]])  # Limit to top 3
            
            expansion_prompt = (
                f"Original query: \"{original_query}\"\n\n"
                f"The following concepts are relevant to this query:\n{concept_text}\n\n"
                f"Please reformulate the query to better incorporate these concepts and retrieve more relevant information. "
                f"The query should remain focused and clear."
            )
            
            expanded_query = await llm_client.generate_text(expansion_prompt, max_tokens=200)
            if expanded_query and expanded_query.strip() and expanded_query.strip().lower() != original_query.lower():
                return expanded_query.strip()
            
            # If expansion didn't work, just append the concept names
            concept_names = [c.name for c in concepts[:2]]  # Top 2 concepts
            return f"{original_query} related to {', '.join(concept_names)}"
            
        except Exception as e:
            logger.error(f"Concept expansion failed: {e}")
            # Simple fallback - just append the first concept name
            if concepts:
                return f"{original_query} related to {concepts[0].name}"
            return original_query 