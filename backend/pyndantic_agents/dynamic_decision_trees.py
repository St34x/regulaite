"""
Dynamic decision tree generator for adaptive reasoning based on query content.
Builds decision trees on-the-fly based on query analysis.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import os
import uuid
from openai import AsyncOpenAI

from .tree_reasoning import DecisionNode, DecisionTree, TreeReasoningAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicTreeGenerator:
    """
    Generates decision trees dynamically based on query analysis.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        template_trees: Optional[Dict[str, DecisionTree]] = None
    ):
        """
        Initialize the dynamic tree generator.

        Args:
            openai_api_key: OpenAI API key
            model: LLM model to use
            template_trees: Optional pre-defined template trees to use as starting points
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.template_trees = template_trees or {}

        logger.info("Initialized DynamicTreeGenerator")

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine its domain, complexity, and structure.

        Args:
            query: User query to analyze

        Returns:
            Analysis dictionary
        """
        system_prompt = """You are an AI specialized in analyzing queries to determine their domain, complexity, and structure.
You need to categorize the query and identify what kind of decision structure would best answer it.

Output a JSON object with the following structure:
{
  "domain": "The primary domain the query belongs to",
  "subdomains": ["List of relevant subdomains"],
  "query_type": "question_answering|problem_solving|decision_making|information_retrieval",
  "complexity": "simple|moderate|complex",
  "key_concepts": ["List of key concepts or entities in the query"],
  "required_context_types": ["Types of information needed to answer"],
  "decision_structure": "linear|branching|hierarchical",
  "estimated_steps": 3,
  "recommended_template": "Name of template if applicable"
}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this query: {query}"}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=500
            )

            content = response.choices[0].message.content
            analysis = json.loads(content)

            logger.info(f"Query analysis: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                "domain": "general",
                "subdomains": [],
                "query_type": "question_answering",
                "complexity": "moderate",
                "key_concepts": [],
                "required_context_types": [],
                "decision_structure": "linear",
                "estimated_steps": 3,
                "recommended_template": None
            }

    async def generate_tree(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> DecisionTree:
        """
        Generate a decision tree based on query analysis.

        Args:
            query: User query
            context: Optional context information

        Returns:
            Generated DecisionTree
        """
        # Analyze the query
        analysis = await self.analyze_query(query)

        # Select base template if available
        template_id = analysis.get("recommended_template")
        base_tree = None

        if template_id and template_id in self.template_trees:
            logger.info(f"Using template tree: {template_id}")
            base_tree = self.template_trees[template_id]
            return await self._adapt_template_tree(base_tree, analysis, query, context)
        else:
            # Generate a new tree from scratch
            logger.info(f"Generating new tree for domain: {analysis.get('domain')}")
            return await self._generate_new_tree(analysis, query, context)

    async def _adapt_template_tree(
        self,
        base_tree: DecisionTree,
        analysis: Dict[str, Any],
        query: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> DecisionTree:
        """
        Adapt a template tree based on query analysis.

        Args:
            base_tree: Base template tree
            analysis: Query analysis
            query: Original query
            context: Optional context

        Returns:
            Adapted DecisionTree
        """
        # Create a new tree based on the template
        tree_id = f"{base_tree.id}_{str(uuid.uuid4())[:8]}"
        tree_name = f"Dynamic {base_tree.name} for: {query[:50]}..."

        # Copy nodes from base tree
        nodes = {node_id: node.copy(deep=True) for node_id, node in base_tree.nodes.items()}

        # Update tree metadata
        tree = DecisionTree(
            id=tree_id,
            name=tree_name,
            description=f"Dynamically adapted from {base_tree.name} template for query: {query}",
            root_node_id=base_tree.root_node_id,
            nodes=nodes,
            version=f"{base_tree.version}-dynamic",
            domain=analysis.get("domain"),
            metadata={
                "source_query": query,
                "analysis": analysis,
                "base_template": base_tree.id
            },
            required_context_types=analysis.get("required_context_types", []),
            max_paths_to_explore=2 if analysis.get("complexity") == "complex" else 1
        )

        # Customize node prompts for this specific query
        await self._customize_node_prompts(tree, query, analysis)

        return tree

    async def _generate_new_tree(
        self,
        analysis: Dict[str, Any],
        query: str,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> DecisionTree:
        """
        Generate a completely new decision tree.

        Args:
            analysis: Query analysis
            query: Original query
            context: Optional context

        Returns:
            Generated DecisionTree
        """
        system_prompt = """You are an AI specialized in creating decision trees for reasoning about complex queries.
You need to design a decision tree with appropriate nodes for answering a specific query.

A decision tree has the following structure:
1. A root node that determines the main category or approach
2. Branch nodes that make specific decisions about how to proceed
3. Leaf nodes that provide specific responses

Your task is to create a JSON schema for a decision tree with the following structure:
{
  "root_node_id": "root",
  "nodes": {
    "root": {
      "id": "root",
      "description": "Description of the root node",
      "prompt": "Prompt to determine the branch",
      "is_leaf": false,
      "is_probabilistic": true,
      "children": {
        "option1": "node_id_for_option1",
        "option2": "node_id_for_option2"
      }
    },
    "node_id_for_option1": {
      "id": "node_id_for_option1",
      "description": "Description",
      "prompt": "Prompt for this decision",
      "is_leaf": false,
      "children": {
        "suboption1": "leaf_node_id1",
        "suboption2": "leaf_node_id2"
      }
    },
    "leaf_node_id1": {
      "id": "leaf_node_id1",
      "description": "Description of leaf node",
      "prompt": "",
      "is_leaf": true,
      "response_template": "Template for response with {QUERY} and {CONTEXT} placeholders"
    }
  }
}

Create a decision tree specifically optimized for answering this query:"""

        try:
            # Extract features from context if available
            context_summary = ""
            if context:
                context_texts = [item.get("text", "")[:200] for item in context[:3]]
                context_summary = "Some relevant context snippets:\n" + "\n".join(context_texts)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nAnalysis: {json.dumps(analysis)}\n\n{context_summary}"}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Extract JSON from the response
            tree_def = None
            try:
                # Find JSON content
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                    tree_def = json.loads(json_content)
                else:
                    # Try to find JSON block
                    import re
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        tree_def = json.loads(json_match.group(1))
            except Exception as e:
                logger.error(f"Error parsing JSON from response: {str(e)}")
                logger.debug(f"Response content: {content}")
                return self._create_fallback_tree(query, analysis)

            if not tree_def or "nodes" not in tree_def:
                return self._create_fallback_tree(query, analysis)

            # Convert the tree definition to a DecisionTree
            nodes = {}
            for node_id, node_def in tree_def.get("nodes", {}).items():
                # Create DecisionNode objects
                nodes[node_id] = DecisionNode(
                    id=node_def.get("id", node_id),
                    description=node_def.get("description", ""),
                    prompt=node_def.get("prompt", ""),
                    children=node_def.get("children", {}),
                    is_leaf=node_def.get("is_leaf", False),
                    is_probabilistic=node_def.get("is_probabilistic", node_id == "root"),  # Root is probabilistic by default
                    action=node_def.get("action"),
                    response_template=node_def.get("response_template"),
                    confidence_threshold=node_def.get("confidence_threshold", 0.7),
                    explore_multiple_paths=node_def.get("explore_multiple_paths", False),
                    domain=analysis.get("domain"),
                    fallback_node_id=node_def.get("fallback_node_id")
                )

            # Create the tree
            tree_id = f"dynamic_{str(uuid.uuid4())[:8]}"
            tree = DecisionTree(
                id=tree_id,
                name=f"Dynamic tree for: {query[:50]}...",
                description=f"Dynamically generated for query: {query}",
                root_node_id=tree_def.get("root_node_id", "root"),
                nodes=nodes,
                version="1.0-dynamic",
                domain=analysis.get("domain"),
                metadata={
                    "source_query": query,
                    "analysis": analysis
                },
                required_context_types=analysis.get("required_context_types", []),
                max_paths_to_explore=2 if analysis.get("complexity") == "complex" else 1
            )

            logger.info(f"Generated new decision tree with {len(nodes)} nodes for query: {query}")
            return tree

        except Exception as e:
            logger.error(f"Error generating decision tree: {str(e)}")
            return self._create_fallback_tree(query, analysis)

    def _create_fallback_tree(self, query: str, analysis: Dict[str, Any]) -> DecisionTree:
        """Create a simple fallback tree when generation fails"""
        tree_id = f"fallback_{str(uuid.uuid4())[:8]}"

        # Create a simple linear tree
        nodes = {
            "root": DecisionNode(
                id="root",
                description="Determine approach for answering the query",
                prompt="Determine the best approach for answering this query:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "direct_answer": "direct_answer",
                    "need_more_info": "need_more_info",
                    "complex_analysis": "complex_analysis"
                },
                is_probabilistic=True
            ),
            "direct_answer": DecisionNode(
                id="direct_answer",
                description="Provide a direct answer",
                prompt="",
                is_leaf=True,
                response_template="Here's a direct answer to your question:\n\n{CONTEXT}\n\nBased on this information, I can tell you that: [provide concise answer based on the context]"
            ),
            "need_more_info": DecisionNode(
                id="need_more_info",
                description="Request more information",
                prompt="",
                is_leaf=True,
                response_template="I need some additional information to properly answer your question about {QUERY}. Based on the context I have:\n\n{CONTEXT}\n\nCould you provide more details about [specific aspects needed]?"
            ),
            "complex_analysis": DecisionNode(
                id="complex_analysis",
                description="Provide detailed analysis",
                prompt="",
                is_leaf=True,
                response_template="Your question requires a detailed analysis. Based on the available information:\n\n{CONTEXT}\n\nHere's my analysis: [comprehensive explanation addressing the query]"
            )
        }

        return DecisionTree(
            id=tree_id,
            name=f"Fallback tree for: {query[:50]}...",
            description=f"Simple fallback tree for query: {query}",
            root_node_id="root",
            nodes=nodes,
            version="1.0-fallback",
            domain=analysis.get("domain", "general"),
            metadata={
                "source_query": query,
                "is_fallback": True
            }
        )

    async def _customize_node_prompts(
        self,
        tree: DecisionTree,
        query: str,
        analysis: Dict[str, Any]
    ) -> None:
        """
        Customize node prompts for a specific query.

        Args:
            tree: Tree to customize
            query: Original query
            analysis: Query analysis
        """
        system_prompt = """You are an AI specialized in optimizing decision tree prompts for specific queries.
Your task is to enhance the effectiveness of decision prompts by making them specifically relevant to the query at hand.

For each decision node prompt, you should:
1. Make it specific to the query domain
2. Include relevant terminology from the query
3. Focus on discriminating between the available options
4. Keep the core decision logic intact

You will be given:
1. The original query
2. The decision node's current prompt
3. The available options (decisions)
4. Analysis of the query

Respond with ONLY the updated prompt text.
"""

        # Select important nodes to customize (root and first level)
        key_nodes = [tree.root_node_id]
        root_node = tree.nodes.get(tree.root_node_id)
        if root_node:
            key_nodes.extend(root_node.children.values())

        # Deduplicate
        key_nodes = list(set(key_nodes))

        # Customize each key node
        for node_id in key_nodes:
            node = tree.nodes.get(node_id)
            if not node or node.is_leaf:
                continue

            try:
                options_text = "\n".join([f"- {option}" for option in node.children.keys()])

                prompt_request = f"""Original query: {query}

Node description: {node.description}

Current prompt:
{node.prompt}

Available options:
{options_text}

Query analysis:
{json.dumps(analysis, indent=2)}

Provide an updated prompt that is specifically tailored to this query while maintaining the same decision logic:"""

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_request}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                updated_prompt = response.choices[0].message.content.strip()

                # Update the node prompt
                tree.nodes[node_id].prompt = updated_prompt
                logger.info(f"Customized prompt for node {node_id}")

            except Exception as e:
                logger.error(f"Error customizing prompt for node {node_id}: {str(e)}")


class DynamicTreeAgent:
    """
    Agent that dynamically generates and uses decision trees based on query analysis.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        template_trees: Optional[Dict[str, DecisionTree]] = None,
        cache_trees: bool = True
    ):
        """
        Initialize the dynamic tree agent.

        Args:
            openai_api_key: OpenAI API key
            model: LLM model to use
            template_trees: Optional template trees to use
            cache_trees: Whether to cache generated trees
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.template_trees = template_trees or {}
        self.cache_trees = cache_trees

        # Tree generator
        self.tree_generator = DynamicTreeGenerator(
            openai_api_key=self.openai_api_key,
            model=self.model,
            template_trees=self.template_trees
        )

        # Cache for generated trees
        self.tree_cache = {}

        logger.info("Initialized DynamicTreeAgent")

    async def process(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        reuse_tree: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query by dynamically generating and using a decision tree.

        Args:
            query: User query
            context: Optional context information
            reuse_tree: Whether to reuse cached trees for similar queries

        Returns:
            Processing result
        """
        start_time = __import__('time').time()

        # Check cache for similar queries
        cache_key = None
        if self.cache_trees and reuse_tree:
            cache_key = self._generate_cache_key(query)
            if cache_key in self.tree_cache:
                logger.info(f"Using cached tree for similar query: {cache_key}")
                tree = self.tree_cache[cache_key]["tree"]
            else:
                # Generate new tree
                tree = await self.tree_generator.generate_tree(query, context)

                # Cache the tree
                if self.cache_trees:
                    self.tree_cache[cache_key] = {
                        "tree": tree,
                        "query": query,
                        "time": start_time
                    }
        else:
            # Generate new tree
            tree = await self.tree_generator.generate_tree(query, context)

        # Create tree reasoning agent
        tree_agent = TreeReasoningAgent(
            tree=tree,
            openai_api_key=self.openai_api_key,
            model=self.model,
            use_probabilistic=True
        )

        # Process with the tree
        result = await tree_agent.process(query, context)

        # Add tree information to result
        result["tree_id"] = tree.id
        result["tree_name"] = tree.name
        result["tree_generated"] = True
        result["tree_generation_time"] = __import__('time').time() - start_time

        return result

    def _generate_cache_key(self, query: str) -> str:
        """Generate a cache key for a query"""
        # Simple approach: take first 5 words
        words = query.lower().split()[:5]
        return "_".join(words)

    def clear_cache(self, max_age: Optional[float] = None):
        """
        Clear the tree cache.

        Args:
            max_age: Optional maximum age in seconds. If provided, only clears trees older than this.
        """
        if max_age is None:
            self.tree_cache.clear()
            logger.info("Cleared entire tree cache")
        else:
            current_time = __import__('time').time()
            keys_to_remove = []
            for key, cache_item in self.tree_cache.items():
                if current_time - cache_item["time"] > max_age:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.tree_cache[key]

            logger.info(f"Cleared {len(keys_to_remove)} trees from cache (older than {max_age}s)")
