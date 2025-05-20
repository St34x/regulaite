import logging
from typing import Dict, Any, List, Optional, Union
import asyncio
import uuid
from datetime import datetime
import json
import openai
from openai import OpenAI

from .base_agent import BaseAgent
from .agent_models import AgentConfig
from autonomous_agent.integration_components.tree_reasoning_adapter import TreeReasoningAdapter, DecisionTree, TreeNode

logger = logging.getLogger(__name__)

class TreeReasoningAgent(BaseAgent):
    """
    TreeReasoningAgent that uses tree-based reasoning for query processing.
    This implementation directly uses the OpenAI client for simplicity and reliability.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
        decision_tree: Optional[Union[Dict[str, Any], DecisionTree]] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        rag_system: Any = None
    ):
        """
        Initialize the tree reasoning agent.
        
        Args:
            agent_id: Unique ID for the agent
            config: Configuration for the agent
            decision_tree: Decision tree to use for reasoning
            neo4j_uri: URI for Neo4j connection
            neo4j_user: Username for Neo4j connection
            neo4j_password: Password for Neo4j connection
            openai_api_key: API key for OpenAI
            rag_system: RAG system for retrieving context
        """
        super().__init__(agent_id or f"tree_agent_{uuid.uuid4()}")
        
        # Convert dict config to AgentConfig if needed
        if isinstance(config, dict):
            self.config = AgentConfig(**config)
        else:
            self.config = config or AgentConfig(
                name="Tree Reasoning Agent",
                description="Agent for tree-based reasoning",
                model="gpt-4",
                temperature=0.7,
                max_tokens=2048,
                include_context=True
            )
            
        # Initialize decision tree
        if isinstance(decision_tree, dict):
            self.decision_tree = DecisionTree.from_dict(decision_tree)
        else:
            self.decision_tree = decision_tree
            
        # Set up Neo4j connection
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Set up OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
            
        # Set up RAG system
        self.rag_system = rag_system
        
        logger.info(f"TreeReasoningAgent initialized with ID: {self.agent_id}")
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with the given query.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Result of agent execution
        """
        try:
            logger.info(f"TreeReasoningAgent executing with query: {query}")
            
            if not self.openai_client:
                return {"error": "OpenAI client not initialized", "analysis": "Unable to process query without OpenAI client"}
            
            if not self.decision_tree:
                logger.warning("No decision tree provided, generating a generic response")
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that specializes in regulatory information, especially related to healthcare data."},
                        {"role": "user", "content": query}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                response_text = response.choices[0].message.content
                
                return {
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": query,
                    "analysis": response_text,
                    "summary": response_text
                }
            
            # Begin tree traversal with the root node
            current_node_id = self.decision_tree.root_node_id
            traversal_history = []
            context = {"query": query}
            response_text = "No response generated"
            
            while current_node_id:
                current_node = self.decision_tree.nodes.get(current_node_id)
                if not current_node:
                    logger.error(f"Node {current_node_id} not found in decision tree")
                    break
                
                traversal_history.append({
                    "node_id": current_node_id,
                    "node_type": current_node.type,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                if current_node.type == "decision":
                    next_node_id = self._process_decision(current_node, query, context)
                elif current_node.type == "action":
                    result = self._process_action(current_node, query, context)
                    context["last_action_result"] = result
                    next_node_id = current_node.next_node_id
                elif current_node.type == "response":
                    response_text = self._process_response(current_node, query, context)
                    break
                else:
                    logger.error(f"Unknown node type: {current_node.type}")
                    break
                
                current_node_id = next_node_id
            
            # Format the result
            return {
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "analysis": response_text,
                "summary": response_text,
                "visited_nodes": [node["node_id"] for node in traversal_history],
                "final_result": {
                    "response": response_text
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing tree reasoning agent: {e}", exc_info=True)
            error_message = f"I encountered an error while processing your request: {str(e)}"
            return {
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "error": str(e),
                "analysis": error_message,
                "summary": error_message
            }
    
    def _process_decision(self, node: TreeNode, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Process a decision node to determine the next node."""
        try:
            decision_query = node.content.get("query", "")
            options = node.content.get("options", [])
            
            if not options:
                logger.error(f"Decision node {node.id} has no options")
                return None
                
            prompt = f"""
            Based on the user query: "{query}"
            
            I need to choose the most appropriate option for the following decision:
            {decision_query}
            
            Available options:
            {json.dumps(options, indent=2)}
            
            Choose exactly ONE option from the list above.
            Respond ONLY with the selected option value (e.g., "option1").
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a decision-making assistant. Respond only with the option value."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            decision = response.choices[0].message.content.strip()
            logger.info(f"Decision for node {node.id}: {decision}")
            
            # Find the matching option
            for option in options:
                if option["value"] in decision:
                    logger.info(f"Selected option: {option['value']} -> {option['next']}")
                    return option["next"]
            
            # No match found, use the first option as fallback
            logger.warning(f"No matching option found in response: {decision}. Using default.")
            return options[0]["next"]
            
        except Exception as e:
            logger.error(f"Error in decision node processing: {e}", exc_info=True)
            if options:
                return options[0]["next"]
            return None
    
    def _process_action(self, node: TreeNode, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action node to perform an action."""
        try:
            action = node.content.get("action", "unknown")
            logger.info(f"Processing action: {action}")
            
            if action == "retrieve_context" and self.rag_system:
                # Retrieve context from RAG system
                try:
                    context_results = self.rag_system.retrieve(query, top_k=5)
                    context_text = []
                    sources = []
                    
                    for result in context_results:
                        if hasattr(result, "text"):
                            context_text.append(result.text)
                        elif hasattr(result, "content"):
                            context_text.append(result.content)
                        elif isinstance(result, dict) and "text" in result:
                            context_text.append(result["text"])
                        
                        if hasattr(result, "metadata"):
                            sources.append(result.metadata)
                        elif isinstance(result, dict) and "metadata" in result:
                            sources.append(result["metadata"])
                    
                    return {
                        "action": action,
                        "context": context_text,
                        "sources": sources
                    }
                except Exception as e:
                    logger.error(f"Error retrieving context: {e}", exc_info=True)
                    return {"action": action, "error": str(e)}
            
            elif action == "retrieve_facts" and self.rag_system:
                # Similar to retrieve_context but can be customized for facts
                try:
                    context_results = self.rag_system.retrieve(query, top_k=5)
                    facts = []
                    sources = []
                    
                    for result in context_results:
                        if hasattr(result, "text"):
                            facts.append(result.text)
                        elif hasattr(result, "content"):
                            facts.append(result.content)
                        elif isinstance(result, dict) and "text" in result:
                            facts.append(result["text"])
                        
                        if hasattr(result, "metadata"):
                            sources.append(result.metadata)
                        elif isinstance(result, dict) and "metadata" in result:
                            sources.append(result["metadata"])
                    
                    return {
                        "action": action,
                        "facts": facts,
                        "sources": sources
                    }
                except Exception as e:
                    logger.error(f"Error retrieving facts: {e}", exc_info=True)
                    return {"action": action, "error": str(e)}
            
            elif action == "perform_analysis":
                # Get previous context if available
                previous_context = []
                if "last_action_result" in context:
                    if "context" in context["last_action_result"]:
                        previous_context = context["last_action_result"]["context"]
                    elif "facts" in context["last_action_result"]:
                        previous_context = context["last_action_result"]["facts"]
                
                # Prepare prompt with context
                context_text = ""
                if previous_context:
                    context_text = "Based on this information:\n\n"
                    for i, ctx in enumerate(previous_context):
                        context_text += f"{i+1}. {ctx}\n\n"
                
                prompt = f"{context_text}Analyze the following query: {query}\n\nProvide a detailed analysis."
                
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are an analytical assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                analysis = response.choices[0].message.content
                
                return {
                    "action": action,
                    "analysis": analysis
                }
            
            else:
                logger.warning(f"Unknown action: {action}")
                return {"action": action, "error": f"Action {action} not implemented"}
                
        except Exception as e:
            logger.error(f"Error in action node processing: {e}", exc_info=True)
            return {"action": "error", "error": str(e)}
    
    def _process_response(self, node: TreeNode, query: str, context: Dict[str, Any]) -> str:
        """Process a response node to generate the final response."""
        try:
            response_template = node.content.get("response_template", "{result}")
            
            # Get previous result if available
            result_content = ""
            if "last_action_result" in context:
                if "analysis" in context["last_action_result"]:
                    result_content = context["last_action_result"]["analysis"]
                elif "facts" in context["last_action_result"]:
                    facts = context["last_action_result"]["facts"]
                    result_content = "Based on the information I've found:\n\n"
                    for i, fact in enumerate(facts):
                        result_content += f"{i+1}. {fact}\n\n"
                elif "context" in context["last_action_result"]:
                    context_items = context["last_action_result"]["context"]
                    result_content = "Based on the information I've found:\n\n"
                    for i, item in enumerate(context_items):
                        result_content += f"{i+1}. {item}\n\n"
            
            # If no previous result, generate a response using the LLM
            if not result_content:
                prompt = f"Please provide a comprehensive response to this query about healthcare regulations: {query}"
                
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in healthcare regulations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                result_content = response.choices[0].message.content
            
            # Format the response
            try:
                response_text = response_template.format(result=result_content)
            except Exception as template_error:
                logger.error(f"Error formatting response template: {template_error}")
                response_text = result_content
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in response node processing: {e}", exc_info=True)
            return f"I encountered an error while generating a response: {str(e)}"
    
    def close(self):
        """Clean up resources."""
        logger.info(f"TreeReasoningAgent {self.agent_id} closed")


def get_default_tree() -> Dict[str, Any]:
    """Get a default decision tree for use in testing or as a fallback."""
    return {
        "id": f"default_tree_{uuid.uuid4()}",
        "name": "Default Decision Tree",
        "description": "Default tree for regulatory information processing",
        "root_node": "start",
        "nodes": {
            "start": {
                "id": "start",
                "type": "decision",
                "query": "What type of information is the user seeking?",
                "options": [
                    {"value": "factual", "label": "Factual Information", "next": "retrieve_facts"},
                    {"value": "analytical", "label": "Analysis", "next": "analyze"},
                    {"value": "procedural", "label": "Process Information", "next": "retrieve_procedures"}
                ]
            },
            "retrieve_facts": {
                "id": "retrieve_facts",
                "type": "action",
                "action": "retrieve_facts",
                "next": "generate_response"
            },
            "analyze": {
                "id": "analyze",
                "type": "action",
                "action": "perform_analysis",
                "next": "generate_response"
            },
            "retrieve_procedures": {
                "id": "retrieve_procedures",
                "type": "action",
                "action": "retrieve_facts",
                "next": "generate_response"
            },
            "generate_response": {
                "id": "generate_response",
                "type": "response",
                "response_template": "{result}"
            }
        }
    }

def create_default_decision_tree() -> DecisionTree:
    """Create a default decision tree."""
    tree_dict = get_default_tree()
    return DecisionTree.from_dict(tree_dict) 