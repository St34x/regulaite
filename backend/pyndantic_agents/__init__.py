# pyndantic_agents package
"""
Intelligent agent framework for the RegulAite platform.
Uses a combination of LLMs, RAG, and Pydantic for validation.
"""

# Import all the key modules so they're available from the package
from .base_agent import BaseAgent, AgentInput, AgentOutput
from .rag_agent import RAGAgent, QueryUnderstandingOutput
from .tree_reasoning import TreeReasoningAgent, DecisionNode, DecisionTree
from .decision_trees import get_tree, get_available_trees
from .agent_factory import create_agent, get_agent_types
from .cybersecurity_agents import (
    VulnerabilityAssessmentAgent, VulnerabilityAssessmentOutput,
    ComplianceMappingAgent, ComplianceMappingOutput,
    ThreatModelingAgent, ThreatModelOutput
)

# Import cybersecurity trees (this will also register them)
import pyndantic_agents.cybersecurity_trees

# Version
__version__ = "0.1.0" 