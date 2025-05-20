"""
Factory class for creating autonomous agents through the adapter pattern.
This replaces the classic agent factory in the backend.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI

from .agent_adapter import AutonomousAgentAdapter
from .graph_interface import GraphInterface
from llamaIndex_rag.rag import RAGSystem
from .tree_reasoning_adapter import TreeReasoningAdapter, DecisionTree

logger = logging.getLogger(__name__)

class AutonomousAgentFactory:
    """
    Factory class for creating autonomous agents.
    This provides a drop-in replacement for the classic agent factory.
    """
    
    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        openai_client: Optional[OpenAI] = None,
        **kwargs
    ):
        """
        Initialize the agent factory with required dependencies.
        
        Args:
            rag_system: RAG system for retrieving context
            openai_client: OpenAI client for language model interactions
            **kwargs: Additional keyword arguments for configuration
        """
        self.rag_system = rag_system
        self.openai_client = openai_client
        self.config = kwargs
        
        # Create graph interface from RAG system if provided
        self.graph_interface = None
        if rag_system:
            # Get the embedding dimension from RAGSystem's default value
            embedding_dimension = getattr(rag_system, 'DEFAULT_EMBED_DIM', 384)
            self.graph_interface = GraphInterface(
                uri=rag_system.neo4j_uri,
                user=rag_system.neo4j_user,
                password=rag_system.neo4j_password
            )
        
        logger.info("AutonomousAgentFactory initialized")
    
    def create_agent(self, agent_type: str, **kwargs) -> Union[AutonomousAgentAdapter, TreeReasoningAdapter]:
        """
        Create an autonomous agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            **kwargs: Additional configuration for the agent
            
        Returns:
            An instantiated autonomous agent adapter or tree reasoning adapter.
        """
        logger.info(f"Creating autonomous agent of type: {agent_type}")
        
        # Merge passed configuration with default configuration
        agent_config = {**self.config, **kwargs}

        if agent_type == "tree_reasoning_agent":
            # Get the decision tree for this agent type
            # For now, using get_default_tree. This could be made more sophisticated.
            tree_structure = self.get_default_tree(agent_type)
            decision_tree = DecisionTree.from_dict(tree_structure)
            
            return TreeReasoningAdapter(
                tree=decision_tree,
                graph_interface=self.graph_interface,
                embedding_service=self.rag_system, # RAG system can act as embedding service
                llm_client=self.openai_client,
                config=agent_config
            )
        else:
            # Create and return the standard agent adapter
            return AutonomousAgentAdapter(
                agent_type=agent_type,
                graph_interface=self.graph_interface,
                embedding_service=self.rag_system,  # The RAG system also provides embedding services
                llm_client=self.openai_client,
                config=agent_config
            )
    
    def get_agent_types(self) -> Dict[str, str]:
        """
        Get a dictionary of available agent types.
        This mimics the functionality of the classic agent factory.
        
        Returns:
            Dictionary mapping agent type IDs to descriptions
        """
        # Define the supported agent types
        # This should match the agent types that were available in the classic implementation
        return {
            "rag": "RAG Agent for general context-aware responses",
            "regulatory": "Regulatory compliance analysis agent",
            "policy": "Policy analysis and comparison agent",
            "cybersecurity": "Cybersecurity assessment and guidance agent",
            "risk": "Risk identification and mitigation agent",
            "legal": "Legal document analysis agent",
            "research": "Research assistant for regulatory topics",
            "grc": "Governance, Risk, and Compliance agent",
            "tree_reasoning_agent": "Agent that uses a decision tree for reasoning"
        }
    
    def get_default_tree(self, agent_type: str) -> Dict[str, Any]:
        """
        Get the default decision tree for an agent type.
        This provides compatibility with the tree reasoning functionality.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Default decision tree structure as a dictionary
        """
        # Return a simple default tree structure based on agent type
        # In a full implementation, these would be more sophisticated and agent-specific
        return {
            "id": f"{agent_type}_default_tree",
            "name": f"Default {agent_type.capitalize()} Tree",
            "description": f"Default decision tree for {agent_type} agent",
            "root_node": "start",
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "decision",
                    "query": "What is the nature of the user's query?",
                    "options": [
                        {"value": "factual", "label": "Factual Query", "next": "factual_node"},
                        {"value": "procedural", "label": "Procedural Query", "next": "procedural_node"},
                        {"value": "analytical", "label": "Analytical Query", "next": "analytical_node"}
                    ]
                },
                "factual_node": {
                    "id": "factual_node",
                    "type": "action",
                    "action": "retrieve_facts",
                    "next": "response_node"
                },
                "procedural_node": {
                    "id": "procedural_node",
                    "type": "action",
                    "action": "retrieve_procedures",
                    "next": "response_node"
                },
                "analytical_node": {
                    "id": "analytical_node",
                    "type": "action",
                    "action": "perform_analysis",
                    "next": "response_node"
                },
                "response_node": {
                    "id": "response_node",
                    "type": "response",
                    "response_template": "Based on my analysis, {result}."
                }
            }
        }


# Singleton instance of the factory
_factory_instance = None

def get_agent_factory(
    rag_system: Optional[RAGSystem] = None,
    openai_client: Optional[OpenAI] = None,
    **kwargs
) -> AutonomousAgentFactory:
    """
    Get or create a singleton instance of the agent factory.
    
    Args:
        rag_system: RAG system for retrieving context
        openai_client: OpenAI client for language model interaction
        **kwargs: Additional configuration for the factory
        
    Returns:
        Singleton instance of the agent factory
    """
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = AutonomousAgentFactory(
            rag_system=rag_system,
            openai_client=openai_client,
            **kwargs
        )
    # Ensure that if RAG system or OpenAI client is passed again, 
    # and the instance exists, we update them if they were None previously.
    # This is important if the factory is initialized early without all dependencies.
    elif rag_system and not _factory_instance.rag_system:
        _factory_instance.rag_system = rag_system
        # Potentially re-initialize graph_interface if rag_system was None before
        if rag_system.neo4j_uri and rag_system.neo4j_user and rag_system.neo4j_password:
             embedding_dimension = getattr(rag_system, 'DEFAULT_EMBED_DIM', 384)
             _factory_instance.graph_interface = GraphInterface(
                uri=rag_system.neo4j_uri,
                user=rag_system.neo4j_user,
                password=rag_system.neo4j_password
            )

    elif openai_client and not _factory_instance.openai_client:
        _factory_instance.openai_client = openai_client
            
    return _factory_instance

def create_agent(agent_type: str, **kwargs) -> Union[AutonomousAgentAdapter, TreeReasoningAdapter]:
    """
    Create an agent of the specified type using the factory.
    This is a convenience function for compatibility with the classic implementation.
    
    Args:
        agent_type: Type of agent to create
        **kwargs: Additional configuration for the agent
        
    Returns:
        An instantiated agent
    """
    factory = get_agent_factory()
    return factory.create_agent(agent_type, **kwargs)

def get_agent_types() -> Dict[str, str]:
    """
    Get a dictionary of available agent types.
    This is a convenience function for compatibility with the classic implementation.
    
    Returns:
        Dictionary mapping agent type IDs to descriptions
    """
    factory = get_agent_factory()
    return factory.get_agent_types()

def get_default_tree(agent_type: str) -> Dict[str, Any]:
    """
    Get the default decision tree for an agent type.
    This is a convenience function for compatibility with the classic implementation.
    
    Args:
        agent_type: Type of agent
        
    Returns:
        Default decision tree structure as a dictionary
    """
    factory = get_agent_factory()
    return factory.get_default_tree(agent_type) 