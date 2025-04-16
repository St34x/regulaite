"""
Agent factory for instantiating different types of agents.
"""
import logging
from typing import Dict, Any, Optional, Type, List
import uuid
import os

from .base_agent import BaseAgent
from .rag_agent import RAGAgent
from .cybersecurity_agents import (
    VulnerabilityAssessmentAgent,
    ComplianceMappingAgent,
    ThreatModelingAgent
)
from llamaIndex_rag.rag import RAGSystem
from .agents import ResearchAgent, RegulatoryAgent
from .tree_reasoning import TreeReasoningAgent, create_default_decision_tree
from .dynamic_decision_trees import DynamicTreeAgent
from config.llm_config import LLMConfig, get_provider_specific_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Registry of available agent types
_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "rag": RAGAgent,
    "vulnerability_assessment": VulnerabilityAssessmentAgent,
    "compliance_mapping": ComplianceMappingAgent,
    "threat_modeling": ThreatModelingAgent,
}

def register_agent_type(agent_type: str, agent_class: Type[BaseAgent]) -> None:
    """
    Register a new agent type.

    Args:
        agent_type: The identifier for the agent type
        agent_class: The agent class to register
    """
    _AGENT_REGISTRY[agent_type] = agent_class
    logger.info(f"Registered agent type: {agent_type}")

def get_agent_types() -> Dict[str, str]:
    """
    Get available agent types.

    Returns:
        Dictionary of agent types to descriptions
    """
    return {
        "rag": "Retrieval-Augmented Generation agent for general queries",
        "regulatory": "Agent specialized in regulatory compliance questions",
        "research": "Agent specialized in research and analysis questions",
        "tree_reasoning": "Agent that uses decision trees for structured reasoning",
        "dynamic_tree": "Agent that dynamically generates decision trees based on query content",
        "vulnerability_assessment": "Agent specialized in vulnerability assessment",
        "compliance_mapping": "Agent specialized in mapping between compliance frameworks",
        "threat_modeling": "Agent specialized in threat modeling"
    }

def create_agent(
    agent_type: str,
    agent_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    rag_system: Optional[RAGSystem] = None,
    llm_config: Optional[LLMConfig] = None,
    **kwargs
) -> BaseAgent:
    """
    Create an agent of the specified type.

    Args:
        agent_type: Type of agent to create
        agent_id: Optional ID for the agent
        config: Optional agent configuration
        rag_system: Optional RAG system instance
        llm_config: Optional LLM configuration
        **kwargs: Additional arguments to pass to the agent constructor

    Returns:
        Agent instance
    """
    # Generate random ID if not provided
    if not agent_id:
        agent_id = str(uuid.uuid4())

    # Use provided LLM config or create default one
    if not llm_config:
        llm_config = LLMConfig(
            provider=kwargs.get("provider", "openai"),
            model=kwargs.get("model", "gpt-4"),
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY", "")),
            api_url=kwargs.get("api_url", None),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
            top_p=kwargs.get("top_p", None),
            frequency_penalty=kwargs.get("frequency_penalty", None),
            presence_penalty=kwargs.get("presence_penalty", None),
            stop_sequences=kwargs.get("stop_sequences", None),
        )

    # Remove LLM config parameters from kwargs to avoid duplication
    for param in ["provider", "model", "api_key", "api_url", "temperature",
                 "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
                 "stop_sequences"]:
        kwargs.pop(param, None)

    # Set a default config if not provided
    if not config:
        config = {
            "name": f"{agent_type.capitalize()} Agent",
            "description": f"Agent specialized in {agent_type}",
            # Add LLM config to agent config
            "llm": get_provider_specific_config(llm_config)
        }
    else:
        # Ensure the LLM config is in the agent config
        config["llm"] = get_provider_specific_config(llm_config)

    # Create agent based on type
    agent_classes = {
        "rag": RAGAgent,
        "regulatory": RegulatoryAgent,
        "research": ResearchAgent,
        "tree_reasoning": TreeReasoningAgent,
        "dynamic_tree": DynamicTreeAgent,
        "vulnerability_assessment": VulnerabilityAssessmentAgent,
        "compliance_mapping": ComplianceMappingAgent,
        "threat_modeling": ThreatModelingAgent,
        # Add more agent types here
    }

    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        logger.error(f"Unknown agent type: {agent_type}")
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Special handling for tree reasoning agent
    if agent_type.lower() == "tree_reasoning" and "tree" not in kwargs:
        # Create a default decision tree if none provided
        kwargs["tree"] = create_default_decision_tree()

    # Special handling for dynamic tree agent
    if agent_type.lower() == "dynamic_tree":
        # Dynamic tree agent doesn't follow the same constructor pattern
        return agent_class(
            openai_api_key=llm_config.api_key or os.getenv("OPENAI_API_KEY", ""),
            model=llm_config.model,
            api_url=llm_config.api_url,
            temperature=llm_config.temperature,
            cache_trees=kwargs.get("cache_trees", True)
        )

    return agent_class(agent_id=agent_id, config=config, rag_system=rag_system, **kwargs)
