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
        context: Optional[List[Dict[str, Any]]] = None,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Process the input through the decision tree.

        Args:
            query: User query
            context: Optional context from RAG
            max_depth: Maximum depth to traverse to prevent infinite loops

        Returns:
            Dictionary with the response and reasoning path
        """
        # Reset decision path
        self.decision_path = []
        self.path_results = []

        # Format the context for insertion in prompts
        context_text = self._format_context(context)

        # Check context sufficiency for tree
        if self.tree.required_context_types and context:
            missing_context = self._check_missing_context(context, self.tree.required_context_types)
            if missing_context:
                logger.warning(f"Missing required context types: {missing_context}")

        # Start from the root node
        if self.use_probabilistic and self.max_exploration_paths > 1:
            # Process with multiple path exploration
            await self._process_multiple_paths(query, context_text, max_depth)

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
                "error": "No valid decision paths found"
            }
        else:
            # Process with single path
            return await self._process_single_path(query, context_text, max_depth)

    async def _process_single_path(
        self,
        query: str,
        context_text: str,
        max_depth: int
    ) -> Dict[str, Any]:
        """
        Process a single decision path through the tree.

        Args:
            query: User query
            context_text: Formatted context text
            max_depth: Maximum depth to traverse

        Returns:
            Result dictionary
        """
        current_node_id = self.tree.root_node_id
        depth = 0
        path_confidence = 1.0  # Start with full confidence

        while depth < max_depth:
            # Get current node
            if current_node_id not in self.tree.nodes:
                logger.error(f"Node {current_node_id} not found in tree")
                break

            current_node = self.tree.nodes[current_node_id]

            # If leaf node, return action
            if current_node.is_leaf:
                response, confidence = await self._generate_leaf_response(current_node, query, context_text)
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
                    "confidence": final_confidence
                }

            # Make decision at this node
            if self.use_probabilistic and current_node.is_probabilistic:
                decision, confidence = await self._make_probabilistic_decision(current_node, query, context_text)
            else:
                decision = await self._make_decision(current_node, query, context_text)
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
        context_text: str,
        max_depth: int
    ) -> None:
        """
        Process multiple decision paths through the tree in parallel.

        Args:
            query: User query
            context_text: Formatted context text
            max_depth: Maximum depth to traverse
        """
        # Start with the root node
        root_node_id = self.tree.root_node_id

        # Queue for BFS-like exploration
        path_queue = [{"node_id": root_node_id, "path": [], "confidence": 1.0}]
        completed_paths = []

        while path_queue and len(completed_paths) < self.max_exploration_paths:
            current_path = path_queue.pop(0)
            current_node_id = current_path["node_id"]
            current_path_nodes = current_path["path"]
            current_confidence = current_path["confidence"]

            if current_node_id not in self.tree.nodes:
                continue

            current_node = self.tree.nodes[current_node_id]

            # If leaf node, complete this path
            if current_node.is_leaf:
                response, confidence = await self._generate_leaf_response(current_node, query, context_text)
                final_path = current_path_nodes + [{
                    "node_id": current_node_id,
                    "decision": "leaf",
                    "confidence": confidence
                }]

                final_confidence = current_confidence * confidence

                completed_paths.append({
                    "response": response,
                    "decision_path": final_path,
                    "final_node_id": current_node_id,
                    "confidence": final_confidence
                })
                continue

            # For probabilistic nodes, consider multiple paths
            if self.use_probabilistic and current_node.is_probabilistic:
                decisions = await self._make_multiple_decisions(
                    current_node,
                    query,
                    context_text,
                    top_k=min(len(current_node.children), 3)  # Consider top 3 decisions at most
                )

                # Add each decision path to the queue
                for decision, confidence in decisions:
                    if decision in current_node.children:
                        new_path = current_path_nodes + [{
                            "node_id": current_node_id,
                            "decision": decision,
                            "confidence": confidence
                        }]

                        path_queue.append({
                            "node_id": current_node.children[decision],
                            "path": new_path,
                            "confidence": current_confidence * confidence
                        })
            else:
                # For non-probabilistic nodes, just take the single decision
                decision = await self._make_decision(current_node, query, context_text)
                confidence = 1.0

                if decision in current_node.children:
                    new_path = current_path_nodes + [{
                        "node_id": current_node_id,
                        "decision": decision,
                        "confidence": confidence
                    }]

                    path_queue.append({
                        "node_id": current_node.children[decision],
                        "path": new_path,
                        "confidence": current_confidence
                    })

            # Sort the queue by confidence score
            path_queue.sort(key=lambda x: x["confidence"], reverse=True)

            # Limit queue size
            path_queue = path_queue[:self.max_exploration_paths * 2]

        # Save the completed paths
        self.path_results = completed_paths

    def _format_context(self, context: Optional[List[Dict[str, Any]]]) -> str:
        """Format context for prompts"""
        if not context:
            return ""

        context_text = "\n\nRelevant Context:\n"
        for i, ctx in enumerate(context):
            context_text += f"\n--- Source {i+1} ---\n{ctx.get('text', '')}\n"
        return context_text

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
        Make a decision at a decision node.

        Args:
            node: Current decision node
            query: User query
            context_text: Formatted context text

        Returns:
            Decision string
        """
        # Existing implementation for deterministic decisions
        # Prepare prompt with context and possible decisions
        full_prompt = node.prompt.replace("{QUERY}", query).replace("{CONTEXT}", context_text)

        # Add options for structured output
        options_text = "\n\nPossible decisions:\n"
        for option in node.children.keys():
            options_text += f"- {option}\n"

        full_prompt += options_text
        full_prompt += "\n\nRespond with ONLY the decision value from the list above, no other text."

        # Make API call
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a decision-making assistant. Follow the instructions exactly."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0,
                max_tokens=50
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
        Make a probabilistic decision with confidence score.

        Args:
            node: Current decision node
            query: User query
            context_text: Formatted context text

        Returns:
            Tuple of (decision, confidence)
        """
        # Prepare prompt with context and possible decisions
        full_prompt = node.prompt.replace("{QUERY}", query).replace("{CONTEXT}", context_text)

        # Add options for structured output
        options_text = "\n\nPossible decisions:\n"
        for option in node.children.keys():
            options_text += f"- {option}\n"

        full_prompt += options_text
        full_prompt += "\n\nRespond with a JSON object containing your decision and confidence level (0-1):\n"
        full_prompt += """{
  "decision": "chosen_option",
  "confidence": 0.8,
  "reasoning": "Brief explanation of your choice"
}"""

        # Make API call
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a decision-making assistant. Follow the instructions exactly."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=200
            )

            content = response.choices[0].message.content
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
        Make multiple probabilistic decisions with confidence scores.

        Args:
            node: Current decision node
            query: User query
            context_text: Formatted context text
            top_k: Number of top decisions to return

        Returns:
            List of (decision, confidence) tuples
        """
        # Prepare prompt with context and possible decisions
        full_prompt = node.prompt.replace("{QUERY}", query).replace("{CONTEXT}", context_text)

        # Add options for structured output
        options_text = "\n\nPossible decisions:\n"
        for option in node.children.keys():
            options_text += f"- {option}\n"

        full_prompt += options_text
        full_prompt += f"\n\nRespond with a JSON object ranking your top {top_k} decisions with confidence levels (0-1):\n"
        full_prompt += """{
  "rankings": [
    {"decision": "option1", "confidence": 0.8, "reasoning": "Brief reasoning"},
    {"decision": "option2", "confidence": 0.6, "reasoning": "Brief reasoning"},
    {"decision": "option3", "confidence": 0.4, "reasoning": "Brief reasoning"}
  ]
}"""

        # Make API call
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a decision-making assistant. Follow the instructions exactly."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=400
            )

            content = response.choices[0].message.content
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
        context_text: str
    ) -> Tuple[str, float]:
        """
        Generate a response at a leaf node.

        Args:
            node: Current leaf node
            query: User query
            context_text: Formatted context text

        Returns:
            Tuple of (response, confidence)
        """
        if node.response_template:
            # Use the template
            response_template = node.response_template.replace("{QUERY}", query).replace("{CONTEXT}", context_text)

            # Add instruction for confidence score
            template_with_confidence = response_template + "\n\n---\nAfter generating your response above, provide a confidence score from 0 to 1 about how well you were able to answer based on the available context. Format: 'CONFIDENCE: 0.X'"

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear, accurate responses based on the available context."},
                        {"role": "user", "content": template_with_confidence}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                content = response.choices[0].message.content

                # Extract confidence if provided
                confidence = 0.8  # Default confidence
                if "CONFIDENCE:" in content:
                    parts = content.split("CONFIDENCE:")
                    response_text = parts[0].strip()
                    try:
                        conf_value = float(parts[1].strip())
                        if 0 <= conf_value <= 1:
                            confidence = conf_value
                    except:
                        pass
                else:
                    response_text = content

                return response_text, confidence

            except Exception as e:
                logger.error(f"Error generating leaf response: {str(e)}")
                return f"I'm having trouble generating a response for your query.", 0.3
        else:
            return f"I've reached a conclusion, but I don't have a response template for node {node.id}.", 0.5

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
                prompt="Analyze the following query and determine what type of request it is:\n\n{QUERY}\n\n{CONTEXT}",
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
                prompt="This is a regulatory query. Determine which specific regulation it relates to:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "specific": "known_regulation",
                    "general": "general_regulation"
                }
            ),
            "risk_query": DecisionNode(
                id="risk_query",
                description="Handle risk-related queries",
                prompt="This is a risk-related query. Determine whether it's about risk assessment, mitigation, or management:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "assessment": "risk_assessment",
                    "mitigation": "risk_mitigation",
                    "management": "risk_management"
                }
            ),
            "compliance_query": DecisionNode(
                id="compliance_query",
                description="Handle compliance queries",
                prompt="This is a compliance query. Determine if it's about checking compliance or implementing compliance measures:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "check": "compliance_check",
                    "implement": "compliance_implement"
                }
            ),
            "general_query": DecisionNode(
                id="general_query",
                description="Handle general information queries",
                prompt="This is a general information query. Determine if it's asking for definitions, examples, or processes:\n\n{QUERY}\n\n{CONTEXT}",
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
                prompt="Determine which regulation is being asked about:\n\n{QUERY}\n\n{CONTEXT}",
                is_leaf=True,
                action="provide_specific_regulation_information",
                response_template="Based on your query about specific regulations and the context I have, here's what you need to know:\n\n{CONTEXT}\n\nTo answer your question: {QUERY}"
            ),
            "general_regulation": DecisionNode(
                id="general_regulation",
                description="Provide general regulatory information",
                is_leaf=True,
                prompt="",
                action="provide_general_regulatory_information",
                response_template="Regarding your question about regulations:\n\n{CONTEXT}\n\nThis should help you understand the regulatory landscape related to your query."
            ),
            "risk_assessment": DecisionNode(
                id="risk_assessment",
                description="Provide risk assessment information",
                is_leaf=True,
                prompt="",
                action="provide_risk_assessment_information",
                response_template="For your risk assessment query:\n\n{CONTEXT}\n\nThis information should help you with your risk assessment process."
            ),
            "risk_mitigation": DecisionNode(
                id="risk_mitigation",
                description="Provide risk mitigation strategies",
                is_leaf=True,
                prompt="",
                action="provide_risk_mitigation_strategies",
                response_template="Here are risk mitigation strategies based on your query:\n\n{CONTEXT}\n\nThese approaches should help address the risks you're concerned about."
            ),
            "risk_management": DecisionNode(
                id="risk_management",
                description="Provide risk management information",
                is_leaf=True,
                prompt="",
                action="provide_risk_management_information",
                response_template="Regarding your question about risk management:\n\n{CONTEXT}\n\nThis information should help with your risk management approach."
            ),
            "compliance_check": DecisionNode(
                id="compliance_check",
                description="Provide compliance checking information",
                is_leaf=True,
                prompt="",
                action="provide_compliance_checking_information",
                response_template="For checking compliance against your requirements:\n\n{CONTEXT}\n\nThis should help you evaluate your compliance status."
            ),
            "compliance_implement": DecisionNode(
                id="compliance_implement",
                description="Provide compliance implementation guidance",
                is_leaf=True,
                prompt="",
                action="provide_compliance_implementation_guidance",
                response_template="To implement compliance measures based on your query:\n\n{CONTEXT}\n\nFollow these guidelines to ensure proper compliance implementation."
            ),
            "provide_definition": DecisionNode(
                id="provide_definition",
                description="Provide definition of a term",
                is_leaf=True,
                prompt="",
                action="provide_definition",
                response_template="Here's the definition you're looking for:\n\n{CONTEXT}\n\nI hope this clarifies the term for you."
            ),
            "provide_example": DecisionNode(
                id="provide_example",
                description="Provide examples",
                is_leaf=True,
                prompt="",
                action="provide_examples",
                response_template="Here are examples based on your query:\n\n{CONTEXT}\n\nThese examples should illustrate the concept you're asking about."
            ),
            "explain_process": DecisionNode(
                id="explain_process",
                description="Explain a process",
                is_leaf=True,
                prompt="",
                action="explain_process",
                response_template="Here's an explanation of the process:\n\n{CONTEXT}\n\nThis should help you understand how the process works."
            ),
            "general_information": DecisionNode(
                id="general_information",
                description="Provide general information",
                is_leaf=True,
                prompt="",
                action="provide_general_information",
                response_template="Based on your query:\n\n{CONTEXT}\n\nI hope this information is helpful."
            )
        }
    )

    return tree
