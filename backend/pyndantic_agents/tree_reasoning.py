"""
Tree-based reasoning system for agents.
Implements a decision tree structure that guides agent reasoning.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import asyncio
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os

# Added import for NodeWithScore
from llama_index.core.schema import NodeWithScore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DecisionNode(BaseModel):
    """
    A node in a decision tree representing a decision point or action.
    """
    id: str = Field(..., description="Unique identifier for the node")
    description: str = Field(..., description="Description of what this node does")
    prompt: str = Field(..., description="Prompt to send to the LLM for this decision")

    # Possible next nodes
    children: Dict[str, str] = Field(default_factory=dict, description="Mapping of decision values to child node IDs")

    # For leaf nodes (actions)
    is_leaf: bool = Field(False, description="Whether this is a leaf node (action node)")
    action: Optional[str] = Field(None, description="Action to take if this is a leaf node")
    response_template: Optional[str] = Field(None, description="Template for generating the response")

    # Enhanced features
    is_probabilistic: bool = Field(False, description="Whether this node uses probabilistic decisions rather than deterministic")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold for a decision to be considered valid")
    explore_multiple_paths: bool = Field(False, description="Whether to explore multiple high-confidence paths")
    domain: Optional[str] = Field(None, description="Domain/category this decision node belongs to (e.g., 'regulatory', 'security')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this node")
    # Runtime context for the node, including text and sources
    runtime_context: Dict[str, Any] = Field(default_factory=dict, description="Runtime context for the node, including formatted text and source nodes")

    # For advanced decision nodes
    fallback_node_id: Optional[str] = Field(None, description="Node to fall back to if no decision meets confidence threshold")
    requires_context: bool = Field(False, description="Whether this node requires specific context to make a decision")
    context_requirements: List[str] = Field(default_factory=list, description="List of context requirements (e.g., required entity types)")

class DecisionTree(BaseModel):
    """
    A decision tree that guides the agent's reasoning process.
    """
    id: str = Field(..., description="Unique identifier for the tree")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Description of what this tree does")
    root_node_id: str = Field(..., description="ID of the root node")
    nodes: Dict[str, DecisionNode] = Field(..., description="Mapping of node IDs to nodes")
    version: str = Field("1.0", description="Version of the tree")

    # Enhanced features
    domain: Optional[str] = Field(None, description="Domain this tree specializes in (e.g., 'compliance', 'risk')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for this tree")
    required_context_types: List[str] = Field(default_factory=list, description="Types of context this tree requires")
    fallback_strategy: str = Field("default", description="Strategy to use when decisions fail ('default', 'backtrack', 'skip')")
    max_paths_to_explore: int = Field(1, description="Maximum number of paths to explore for probabilistic nodes")

class TreeReasoningAgent:
    """
    Agent that uses tree-based reasoning to make decisions.
    """

    def __init__(
        self,
        tree: DecisionTree,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        use_probabilistic: bool = True,
        max_exploration_paths: Optional[int] = None,
    ):
        """
        Initialize the tree reasoning agent.

        Args:
            tree: Decision tree to use
            openai_api_key: OpenAI API key for LLM calls
            model: LLM model to use
            use_probabilistic: Whether to use probabilistic decision making
            max_exploration_paths: Maximum number of paths to explore for probabilistic nodes
        """
        self.tree = tree
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.use_probabilistic = use_probabilistic
        self.max_exploration_paths = max_exploration_paths or tree.max_paths_to_explore

        # Decision path tracking
        self.decision_path = []

        # For tracking multiple paths in probabilistic mode
        self.path_results = []

        logger.info(f"Initialized TreeReasoningAgent with tree {tree.id} ({tree.name})")

    async def process(
        self,
        query: str,
        initial_retrieved_nodes: Optional[List[NodeWithScore]] = None,
        max_depth: int = 10,
        agent_settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process the input through the decision tree.

        Args:
            query: User query
            initial_retrieved_nodes: Optional list of NodeWithScore from RAG for initial context
            max_depth: Maximum depth to traverse to prevent infinite loops
            agent_settings: Optional agent settings from the request

        Returns:
            Dictionary with the response and reasoning path
        """
        # Reset decision path
        self.decision_path = []
        self.path_results = []

        # Format the initial context with sources
        current_formatted_context_str = self._format_context_with_sources(
            base_text="",
            retrieved_nodes=initial_retrieved_nodes
        )
        
        # Check context sufficiency for tree
        # if self.tree.required_context_types and initial_retrieved_nodes: # This was the line causing potential issues
            # missing_context = self._check_missing_context(initial_retrieved_nodes, self.tree.required_context_types) # This would likely error
            # if missing_context:
            #     logger.warning(f"Missing required context types: {missing_context}")

        # Start from the root node
        if self.use_probabilistic and self.max_exploration_paths > 1:
            # Process with multiple path exploration
            await self._process_multiple_paths(query, current_formatted_context_str, initial_retrieved_nodes, max_depth)

            # Merge and rank results
            if self.path_results:
                best_result = max(self.path_results, key=lambda x: x.get("confidence", 0))
                best_result["alternative_paths"] = [
                    {
                        "path": r["decision_path"],
                        "confidence": r["confidence"],
                        "response": r["response"]
                    }
                    for r in self.path_results if r != best_result
                ]
                return best_result

            # Fallback if no paths were successful
            return {
                "response": "I couldn't determine a confident response based on my decision process.",
                "decision_path": [],
                "confidence": 0.0,
                "error": "No valid decision paths found",
                "sources": []  # Add empty sources list for consistency
            }
        else:
            # Process with single path
            return await self._process_single_path(query, current_formatted_context_str, initial_retrieved_nodes, max_depth)

    async def _process_single_path(
        self,
        query: str,
        initial_formatted_context_str: str,
        initial_retrieved_nodes: Optional[List[NodeWithScore]],
        max_depth: int
    ) -> Dict[str, Any]:
        """
        Process a single decision path through the tree.

        Args:
            query: User query
            initial_formatted_context_str: Formatted context string from initial RAG retrieval
            initial_retrieved_nodes: The initial list of NodeWithScore objects from RAG
            max_depth: Maximum depth to traverse

        Returns:
            Result dictionary
        """
        current_node_id = self.tree.root_node_id
        depth = 0
        path_confidence = 1.0  # Start with full confidence
        
        # This will hold the NodeWithScore objects relevant at each step. Starts with initial.
        current_step_nodes = initial_retrieved_nodes if initial_retrieved_nodes else []
        # The formatted text to be used in prompts at this step
        current_formatted_prompt_context = initial_formatted_context_str

        while depth < max_depth:
            # Get current node
            if current_node_id not in self.tree.nodes:
                logger.error(f"Node {current_node_id} not found in tree")
                break

            current_node = self.tree.nodes[current_node_id]
            # Store current sources in the node for potential later use or logging
            current_node.runtime_context = {"sources": current_step_nodes, "formatted_text": current_formatted_prompt_context}

            # If leaf node, return action
            if current_node.is_leaf:
                # Leaf node uses the context (text and sources) accumulated up to this point
                response, confidence, sources = await self._generate_leaf_response(current_node, query, current_formatted_prompt_context, current_step_nodes)
                self.decision_path.append({
                    "node_id": current_node_id,
                    "decision": "leaf",
                    "confidence": confidence
                })

                # Combine path confidence with leaf confidence
                final_confidence = path_confidence * confidence

                return {
                    "response": response,
                    "decision_path": self.decision_path,
                    "final_node_id": current_node_id,
                    "confidence": final_confidence,
                    "sources": sources
                }

            # Make decision at this node
            if self.use_probabilistic and current_node.is_probabilistic:
                decision, confidence = await self._make_probabilistic_decision(current_node, query, current_formatted_prompt_context)
            else:
                decision = await self._make_decision(current_node, query, current_formatted_prompt_context)
                confidence = 1.0  # Default confidence for non-probabilistic

            # Update path confidence
            path_confidence *= confidence

            self.decision_path.append({
                "node_id": current_node_id,
                "decision": decision,
                "confidence": confidence
            })

            # Check confidence threshold
            if confidence < current_node.confidence_threshold:
                logger.warning(f"Decision confidence {confidence} below threshold {current_node.confidence_threshold}")

                # Use fallback if available
                if current_node.fallback_node_id:
                    logger.info(f"Using fallback node {current_node.fallback_node_id}")
                    current_node_id = current_node.fallback_node_id
                    continue

            # Move to next node
            if decision in current_node.children:
                current_node_id = current_node.children[decision]
            else:
                logger.warning(f"Decision '{decision}' not found in children of node {current_node_id}")
                break

            depth += 1

        # If we get here, we hit max depth or an error occurred
        logger.warning(f"Decision tree traversal stopped at depth {depth}, node {current_node_id}")

        return {
            "response": "I'm not sure how to respond to that based on my decision process.",
            "decision_path": self.decision_path,
            "final_node_id": current_node_id,
            "confidence": path_confidence,
            "error": "Max depth reached or invalid decision"
        }

    async def _process_multiple_paths(
        self,
        query: str,
        initial_formatted_context_str: str,
        initial_retrieved_nodes: Optional[List[NodeWithScore]],
        max_depth: int
    ) -> None:
        """
        Process multiple decision paths through the tree (for probabilistic exploration).

        Args:
            query: User query
            initial_formatted_context_str: Formatted context string from initial RAG retrieval
            initial_retrieved_nodes: The initial list of NodeWithScore objects from RAG
            max_depth: Maximum depth to traverse
        """
        # This is a simplified version for multiple paths.
        # A full implementation would involve more complex state management
        # and potentially exploring paths in parallel or with beam search.

        # Get the root node
        root_node_id = self.tree.root_node_id
        if root_node_id not in self.tree.nodes:
            logger.error(f"Root node {root_node_id} not found in tree")
            return

        root_node = self.tree.nodes[root_node_id]

        # If root is leaf
        if root_node.is_leaf:
            response, confidence = await self._generate_leaf_response(root_node, query, initial_formatted_context_str, initial_retrieved_nodes)
            self.path_results.append({
                "response": response,
                "decision_path": [{"node_id": root_node_id, "decision": "leaf", "confidence": confidence}],
                "confidence": confidence
            })
            return

        # Make initial decisions from the root
        if root_node.is_probabilistic and self.use_probabilistic:
            decisions_with_confidence = await self._make_multiple_decisions(
                root_node, query, initial_formatted_context_str, top_k=self.max_exploration_paths
            )
        else:
            # For non-probabilistic or single path, make one decision
            decision = await self._make_decision(root_node, query, initial_formatted_context_str)
            decisions_with_confidence = [(decision, 1.0)] # Assume full confidence

        # Explore paths for each decision
        for decision, confidence in decisions_with_confidence:
            if confidence < root_node.confidence_threshold:
                logger.info(f"Skipping path for decision '{decision}' due to low confidence {confidence}")
                continue

            if decision in root_node.children:
                current_path = [{"node_id": root_node_id, "decision": decision, "confidence": confidence}]
                await self._explore_path(
                    root_node.children[decision],
                    query,
                    initial_formatted_context_str,
                    initial_retrieved_nodes,
                    max_depth -1, # Decrement depth
                    current_path,
                    confidence # Initial path confidence
                )
            else:
                logger.warning(f"Decision '{decision}' has no child from root node {root_node_id}")
        
        if not self.path_results:
            logger.warning("No successful paths found during multiple path exploration.")

    async def _explore_path(
        self,
        current_node_id: str,
        query: str,
        current_formatted_context_str: str,
        current_retrieved_nodes: Optional[List[NodeWithScore]],
        remaining_depth: int,
        current_path: List[Dict[str, Any]],
        path_confidence: float
    ):
        """
        Recursively explore a path in the decision tree.

        Args:
            current_node_id: ID of the current node
            query: User query
            current_formatted_context_str: Formatted context string for the current state of the path
            current_retrieved_nodes: List of NodeWithScore for the current state of the path
            remaining_depth: Depth remaining for traversal
            current_path: Current decision path taken
            path_confidence: Accumulated confidence for the current path
        """
        if remaining_depth <= 0:
            logger.info(f"Max depth reached for path: {current_path}")
            # Potentially generate a response based on current path if needed
            return

        if current_node_id not in self.tree.nodes:
            logger.error(f"Node {current_node_id} not found in tree during exploration")
            return

        node = self.tree.nodes[current_node_id]
        node.runtime_context = {"sources": current_retrieved_nodes, "formatted_text": current_formatted_context_str}

        # If leaf node, record result
        if node.is_leaf:
            response, confidence = await self._generate_leaf_response(node, query, current_formatted_context_str, current_retrieved_nodes)
            final_confidence = path_confidence * confidence
            self.path_results.append({
                "response": response,
                "decision_path": current_path,
                "confidence": final_confidence
            })
            return

        # Make decision(s) at this node
        if self.use_probabilistic and node.is_probabilistic and node.explore_multiple_paths:
            # Explore multiple decisions if configured
            decisions_with_confidence = await self._make_multiple_decisions(
                node,
                query,
                current_formatted_context_str,
                top_k=self.max_exploration_paths
            )
            for decision, confidence in decisions_with_confidence:
                if confidence < node.confidence_threshold:
                    logger.info(f"Skipping decision '{decision}' due to low confidence {confidence}")
                    continue

                if decision in node.children:
                    new_path = current_path + [{
                        "node_id": current_node_id,
                        "decision": decision,
                        "confidence": confidence
                    }]
                    await self._explore_path(
                        node.children[decision],
                        query,
                        current_formatted_context_str,
                        current_retrieved_nodes,
                        remaining_depth - 1,
                        new_path,
                        path_confidence * confidence
                    )
                else:
                    logger.warning(f"Decision '{decision}' not found in children of node {current_node_id} during exploration")
        else: # Single decision path (either deterministic or probabilistic choosing one)
            if self.use_probabilistic and node.is_probabilistic:
                decision, confidence = await self._make_probabilistic_decision(node, query, current_formatted_context_str)
            else:
                decision = await self._make_decision(node, query, current_formatted_context_str)
                confidence = 1.0

            new_confidence = path_confidence * confidence
            new_path = current_path + [{
                "node_id": current_node_id,
                "decision": decision,
                "confidence": confidence
            }]

            if confidence < node.confidence_threshold:
                logger.warning(f"Decision confidence {confidence} below threshold {node.confidence_threshold} for node {current_node_id}")
                # Fallback logic if any (currently not fully exploring fallback in multi-path here)
                # For simplicity, this branch stops if below threshold in multi-explore.
                # A more robust version might use fallback_node_id.
                return # Stop this path if confidence is too low

            if decision in node.children:
                await self._explore_path(
                    node.children[decision],
                    query,
                    current_formatted_context_str,
                    current_retrieved_nodes,
                    remaining_depth - 1,
                    new_path,
                    new_confidence
                )
            else:
                logger.warning(f"Decision '{decision}' not found in children of node {current_node_id} during exploration")

    def _format_context_with_sources(self, base_text: str, retrieved_nodes: Optional[List[NodeWithScore]]) -> str:
        """
        Formats the retrieved LlamaIndex nodes into a string with source citations
        and a list of sources at the end.

        Args:
            base_text: Any base text to include before the context.
            retrieved_nodes: List of NodeWithScore objects from RAG.

        Returns:
            A string containing the formatted context with source citations.
        """
        if not retrieved_nodes:
            return f"{base_text}\n\nNo context documents were retrieved." if base_text else "No context documents were retrieved."

        context_parts = []
        source_details_list = []

        for i, node in enumerate(retrieved_nodes):
            source_id = i + 1
            # Ensure metadata exists and get doc_name, otherwise provide a default
            if hasattr(node, 'metadata'):
                metadata = node.metadata
            elif isinstance(node, dict) and 'metadata' in node:
                metadata = node['metadata']
            else:
                metadata = None
                
            doc_name = metadata.get('doc_name', f'Document {source_id}') if metadata else f'Document {source_id}'
            
            # Attempt to get file_path if doc_name is generic, or prefer file_path
            file_path = metadata.get('file_path', doc_name) if metadata else doc_name

            # Prefer a more specific name if available, e.g., from file_path
            display_source_name = file_path

            # Fix: Get content from node correctly depending on its type
            if hasattr(node, 'get_content'):
                content = node.get_content().strip()
            elif isinstance(node, dict) and 'content' in node:
                content = node['content'].strip()
            else:
                content = str(node).strip()

            # Fix: Get score from node correctly depending on its type
            if hasattr(node, 'get_score'):
                score = node.get_score() or 'N/A'
            elif isinstance(node, dict) and 'score' in node:
                score = node['score'] or 'N/A'
            else:
                score = 'N/A'

            context_parts.append(f"[Source {source_id}]\n{content}")
            source_details_list.append(f"{source_id}. {display_source_name} (Score: {score :.2f})")
        
        formatted_sources_references = "\n\nSources:\n" + "\n".join(source_details_list)
        
        full_context_str = "\n\n".join(context_parts)
        
        final_output = ""
        if base_text:
            final_output += base_text + "\n\n"
        
        final_output += "Relevant Context from Documents:\n" + full_context_str + formatted_sources_references
        
        return final_output

    def _check_missing_context(self, context: List[Dict[str, Any]], required_types: List[str]) -> List[str]:
        """Check for missing required context types"""
        available_types = set()
        for ctx in context:
            metadata = ctx.get("metadata", {})
            ctx_type = metadata.get("type", "")
            if ctx_type:
                available_types.add(ctx_type)

        missing = [req for req in required_types if req not in available_types]
        return missing

    async def _make_decision(self, node: DecisionNode, query: str, context_text: str) -> str:
        """
        Make a decision at a node using LLM.
        Assumes node.prompt is a template with {query} and {context}.
        The context_text provided here should already be formatted with sources.
        """
        # Prompt for decision making
        prompt_template = node.prompt
        # Ensure children are presented as options
        options_str = ", ".join(node.children.keys())
        
        # Enhanced prompt for sourcing
        filled_prompt = prompt_template.format(
            query=query,
            context=context_text, # This context_text includes [Source X] markers
            options=options_str
        )
        
        final_prompt = (
            f"{filled_prompt}\n\n"
            f"The user's query is: {query}\n"
            f"Based on the query, the provided context (please cite sources as [Source X] if used), and the current decision point described as '{node.description}', "
            f"choose one of the following actions or next steps: {options_str}.\n"
            f"Your response should be ONLY the chosen action/step name from the list."
            f"If you use information from the context, ensure your reasoning (even if not explicitly stated in the final choice) considers it. Your direct answer must be one of: {options_str}."
            f"Always respond in the same language as the user's query."
        )

        logger.info(f"Making decision for node {node.id} with prompt:\n{final_prompt}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful assistant that provides detailed and sourced answers."},
                          {"role": "user", "content": final_prompt}],
                temperature=0.7, # Allow some creativity for well-formed responses
            )
            decision = response.choices[0].message.content.strip()

            # Validate and clean decision
            valid_decisions = list(node.children.keys())

            # Check if decision is valid
            if decision in valid_decisions:
                return decision

            # Try matching with case insensitivity
            for valid in valid_decisions:
                if valid.lower() == decision.lower():
                    return valid

            # If nothing matched, use first option as fallback
            logger.warning(f"Invalid decision '{decision}', using first option")
            return valid_decisions[0] if valid_decisions else "unknown"

        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            valid_decisions = list(node.children.keys())
            return valid_decisions[0] if valid_decisions else "unknown"

    async def _make_probabilistic_decision(
        self,
        node: DecisionNode,
        query: str,
        context_text: str
    ) -> Tuple[str, float]:
        """
        Make a probabilistic decision at a node using LLM.
        The context_text provided here should already be formatted with sources.
        """
        # Prompt for probabilistic decision making
        prompt_template = node.prompt  # Assuming this prompt asks for probabilities or a ranked list
        options_str = ", ".join(node.children.keys())

        # Enhanced prompt for sourcing
        filled_prompt = prompt_template.format(
            query=query,
            context=context_text, # This context_text includes [Source X] markers
            options=options_str
        )
        
        final_prompt = (
            f"{filled_prompt}\n\n"
            f"The user's query is: {query}\n"
            f"Context (cite sources as [Source X] if used):\n{context_text}\n\n"
            f"Considering the query, context, and the current decision point '{node.description}', "
            f"evaluate the following options: {options_str}.\n"
            f"Respond with a JSON object where keys are option names and values are confidence scores (0.0 to 1.0). Example: {{'option_a': 0.8, 'option_b': 0.2}}. "
            f"Ensure your reasoning for the scores considers the provided context and cite sources if applicable in your internal thought process."
            f"Always respond in the same language as the user's query."
        )

        logger.info(f"Making probabilistic decision for node {node.id} with prompt:\n{final_prompt}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful assistant that provides detailed and sourced answers."},
                          {"role": "user", "content": final_prompt}],
                temperature=0.7, # Allow some creativity for well-formed responses
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            decision = result.get("decision", "").strip()
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            # Log the reasoning
            logger.info(f"Decision reasoning: {reasoning}")

            # Validate and clean decision
            valid_decisions = list(node.children.keys())

            # Check if decision is valid
            if decision in valid_decisions:
                return decision, confidence

            # Try matching with case insensitivity
            for valid in valid_decisions:
                if valid.lower() == decision.lower():
                    return valid, confidence

            # If nothing matched, use first option as fallback with reduced confidence
            logger.warning(f"Invalid decision '{decision}', using first option with reduced confidence")
            return valid_decisions[0] if valid_decisions else "unknown", confidence * 0.5

        except Exception as e:
            logger.error(f"Error making probabilistic decision: {str(e)}")
            valid_decisions = list(node.children.keys())
            return valid_decisions[0] if valid_decisions else "unknown", 0.3

    async def _make_multiple_decisions(
        self,
        node: DecisionNode,
        query: str,
        context_text: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Generate multiple decisions with confidence scores using LLM.
        The context_text provided here should already be formatted with sources.
        """
        # Prompt for multiple decision generation
        prompt_template = node.prompt  # Expects a prompt that can lead to multiple choices
        options_str = ", ".join(node.children.keys())

        # Enhanced prompt for sourcing
        filled_prompt = prompt_template.format(
            query=query,
            context=context_text, # This context_text includes [Source X] markers
            options=options_str
        )

        final_prompt = (
            f"{filled_prompt}\n\n"
            f"The user's query is: {query}\n"
            f"Context (cite sources as [Source X] if used):\n{context_text}\n\n"
            f"Considering the query, context, and the current decision point '{node.description}', "
            f"identify the top {top_k} most relevant next steps or decisions from the options: {options_str}.\n"
            f"Respond with a JSON array of objects, each object having 'decision' (the option name) and 'confidence' (0.0 to 1.0). Example: [{{'decision': 'option_a', 'confidence': 0.9}}, {{'decision': 'option_b', 'confidence': 0.7}}]. "
            f"Ensure your reasoning for the scores considers the provided context and cite sources if applicable in your internal thought process."
            f"Always respond in the same language as the user's query."
        )

        logger.info(f"Making multiple decisions for node {node.id} with prompt:\n{final_prompt}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful assistant that provides detailed and sourced answers."},
                          {"role": "user", "content": final_prompt}],
                temperature=0.7, # Allow some creativity for well-formed responses
            )
            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            rankings = result.get("rankings", [])
            valid_decisions = list(node.children.keys())

            # Validate and clean decisions
            validated_rankings = []
            for rank in rankings:
                decision = rank.get("decision", "").strip()
                confidence = float(rank.get("confidence", 0.5))

                # Check if decision is valid
                if decision in valid_decisions:
                    validated_rankings.append((decision, confidence))
                    continue

                # Try matching with case insensitivity
                for valid in valid_decisions:
                    if valid.lower() == decision.lower():
                        validated_rankings.append((valid, confidence))
                        break

            # If no valid decisions, use first option as fallback
            if not validated_rankings and valid_decisions:
                validated_rankings.append((valid_decisions[0], 0.3))

            return validated_rankings

        except Exception as e:
            logger.error(f"Error making multiple decisions: {str(e)}")
            valid_decisions = list(node.children.keys())
            if valid_decisions:
                return [(valid_decisions[0], 0.3)]
            return [("unknown", 0.1)]

    async def _generate_leaf_response(
        self,
        node: DecisionNode,
        query: str,
        context_text: str,
        retrieved_nodes: Optional[List[NodeWithScore]]
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Generate a response for a leaf node using LLM.
        The context_text provided here should already be formatted with sources.
        """
        if node.action and node.response_template:
            # Template-based response
            try:
                # The context_text already contains source information.
                # The template should be designed to incorporate this naturally.
                # For example, the template might say: "Based on the information: {context}, the answer is..."
                # And the LLM, when filling it, should use the [Source X] citations from the context_text.
                
                prompt_for_leaf = (
                    f"User Query: {query}\n\n"
                    f"Context from retrieved documents (sources are cited as [Source X]):\n{context_text}\n\n"
                    f"Task: You are at a final step in a decision process. The decision taken is '{node.description}'.\n"
                    f"Action to perform: {node.action}.\n"
                    f"Response template: {node.response_template}\n\n"
                    f"Instructions: Generate a final response for the user by filling in the response template. "
                    f"Use the information from the provided context. If you use specific information from the context, "
                    f"ensure the corresponding [Source X] citations are included in your final answer. "
                    f"If the context is not relevant, synthesize an answer based on the query and action. Be comprehensive. "
                    f"Always respond in the same language as the user's query."
                )

                logger.info(f"Generating leaf response for node {node.id} with prompt:\n{prompt_for_leaf}")

                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "You are a helpful assistant that provides detailed and sourced answers."},
                              {"role": "user", "content": prompt_for_leaf}],
                    temperature=0.7, # Allow some creativity for well-formed responses
                )
                response_content = completion.choices[0].message.content.strip()
                
                # Basic confidence - can be improved with LLM eval or specific metrics
                confidence = 0.85  # Default confidence for a generated leaf response
                
                # Check if sources were cited (simple check)
                if "[Source" in response_content and retrieved_nodes: # Checks if citation markers are present
                    logger.info(f"Leaf response for node {node.id} appears to cite sources.")
                elif retrieved_nodes:
                    logger.warning(f"Leaf response for node {node.id} was generated with context, but might be missing [Source X] citations.")

                # Prepare source information from retrieved nodes
                sources = []
                if retrieved_nodes:
                    for i, node in enumerate(retrieved_nodes):
                        # Ensure metadata exists
                        if hasattr(node, 'metadata'):
                            metadata = node.metadata
                        elif isinstance(node, dict) and 'metadata' in node:
                            metadata = node['metadata']
                        else:
                            metadata = {}
                            
                        # Get document name
                        doc_name = metadata.get('doc_name', f'Document {i+1}') if metadata else f'Document {i+1}'
                        
                        # Get file path if available
                        file_path = metadata.get('file_path', doc_name) if metadata else doc_name
                        
                        # Get score
                        if hasattr(node, 'get_score'):
                            score = node.get_score() or 0
                        elif isinstance(node, dict) and 'score' in node:
                            score = node['score'] or 0
                        else:
                            score = 0
                            
                        # Create source info
                        source_info = {
                            "id": i + 1,
                            "title": doc_name,
                            "file_path": file_path,
                            "score": score,
                            "chunk_id": metadata.get("chunk_id", "")
                        }
                        sources.append(source_info)

                return response_content, confidence, sources

            except Exception as e:
                logger.error(f"Error generating templated response for leaf node {node.id}: {str(e)}")
                return f"Error generating response: {str(e)}", 0.3, []
        else:
            # Fallback if no template or action
            logger.warning(f"Leaf node {node.id} has no action or response template.")
            return f"Reached end of decision path at node {node.id} ({node.description}). No specific action defined.", 0.5, []

def create_default_decision_tree() -> DecisionTree:
    """
    Create a default decision tree for general query understanding.

    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="default_understanding",
        name="Default Query Understanding",
        description="A general-purpose decision tree for understanding and routing user queries",
        root_node_id="root",
        nodes={
            "root": DecisionNode(
                id="root",
                description="Determine the high-level query type",
                prompt="Analyze the following query and determine what type of request it is:\n\n{query}\n\n{context}",
                children={
                    "regulatory": "regulatory_query",
                    "risk": "risk_query",
                    "compliance": "compliance_query",
                    "general": "general_query"
                }
            ),
            "regulatory_query": DecisionNode(
                id="regulatory_query",
                description="Handle regulatory queries",
                prompt="This is a regulatory query. Determine which specific regulation it relates to:\n\n{query}\n\n{context}",
                children={
                    "specific": "known_regulation",
                    "general": "general_regulation"
                }
            ),
            "risk_query": DecisionNode(
                id="risk_query",
                description="Handle risk-related queries",
                prompt="This is a risk-related query. Determine whether it's about risk assessment, mitigation, or management:\n\n{query}\n\n{context}",
                children={
                    "assessment": "risk_assessment",
                    "mitigation": "risk_mitigation",
                    "management": "risk_management"
                }
            ),
            "compliance_query": DecisionNode(
                id="compliance_query",
                description="Handle compliance queries",
                prompt="This is a compliance query. Determine if it's about checking compliance or implementing compliance measures:\n\n{query}\n\n{context}",
                children={
                    "check": "compliance_check",
                    "implement": "compliance_implement"
                }
            ),
            "general_query": DecisionNode(
                id="general_query",
                description="Handle general information queries",
                prompt="This is a general information query. Determine if it's asking for definitions, examples, or processes:\n\n{query}\n\n{context}",
                children={
                    "definition": "provide_definition",
                    "example": "provide_example",
                    "process": "explain_process",
                    "other": "general_information"
                }
            ),
            "known_regulation": DecisionNode(
                id="known_regulation",
                description="Provide information about a specific regulation",
                prompt="Determine which regulation is being asked about:\n\n{query}\n\n{context}",
                is_leaf=True,
                action="provide_specific_regulation_information",
                response_template="Regarding your query about specific regulations ({query}), and based on the available documents, here is the information:"
            ),
            "general_regulation": DecisionNode(
                id="general_regulation",
                description="Provide general regulatory information",
                is_leaf=True,
                prompt="",
                action="provide_general_regulatory_information",
                response_template="Regarding your question about regulations ({query}), the following information, drawn from the provided context, should help you understand the regulatory landscape:"
            ),
            "risk_assessment": DecisionNode(
                id="risk_assessment",
                description="Provide risk assessment information",
                is_leaf=True,
                prompt="",
                action="provide_risk_assessment_information",
                response_template="For your risk assessment query ({query}), the following information from the relevant documents should assist your process:"
            ),
            "risk_mitigation": DecisionNode(
                id="risk_mitigation",
                description="Provide risk mitigation strategies",
                is_leaf=True,
                prompt="",
                action="provide_risk_mitigation_strategies",
                response_template="Based on your query ({query}), here are risk mitigation strategies derived from the provided context:"
            ),
            "risk_management": DecisionNode(
                id="risk_management",
                description="Provide risk management information",
                is_leaf=True,
                prompt="",
                action="provide_risk_management_information",
                response_template="Regarding your question about risk management ({query}), information from the provided documents suggests the following approach:"
            ),
            "compliance_check": DecisionNode(
                id="compliance_check",
                description="Provide compliance checking information",
                is_leaf=True,
                prompt="",
                action="provide_compliance_checking_information",
                response_template="To help you check compliance against your requirements ({query}), based on the provided context:"
            ),
            "compliance_implement": DecisionNode(
                id="compliance_implement",
                description="Provide compliance implementation guidance",
                is_leaf=True,
                prompt="",
                action="provide_compliance_implementation_guidance",
                response_template="For implementing compliance measures related to your query ({query}), please consider the following guidance from the available documents:"
            ),
            "provide_definition": DecisionNode(
                id="provide_definition",
                description="Provide definition of a term",
                is_leaf=True,
                prompt="",
                action="provide_definition",
                response_template="Regarding the definition for your query ({query}), the provided documents indicate the following:"
            ),
            "provide_example": DecisionNode(
                id="provide_example",
                description="Provide examples",
                is_leaf=True,
                prompt="",
                action="provide_examples",
                response_template="Here are examples based on your query ({query}), drawn from the provided context:"
            ),
            "explain_process": DecisionNode(
                id="explain_process",
                description="Explain a process",
                is_leaf=True,
                prompt="",
                action="explain_process",
                response_template="Regarding the process you asked about ({query}), here's an explanation based on the available information:"
            ),
            "general_information": DecisionNode(
                id="general_information",
                description="Provide general information",
                is_leaf=True,
                prompt="",
                action="provide_general_information",
                response_template="Based on your query ({query}) and the information in the provided documents:"
            )
        }
    )

    return tree
