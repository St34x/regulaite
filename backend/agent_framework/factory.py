"""
Factory for creating and initializing agents in the RegulAIte Agent Framework.

This module provides factory functions for creating different types of agents
with all necessary components.
"""
from typing import Dict, List, Optional, Any, Union
import logging

from .agent import Agent
from .rag_agent import RAGAgent
from .orchestrator import OrchestratorAgent
from .tool_registry import ToolRegistry
from .query_parser import QueryParser
from .integrations.rag_integration import get_rag_integration
from .integrations.llm_integration import get_llm_integration, LLMIntegration, get_llm_client

# Set up logging
logger = logging.getLogger(__name__)

# Backward compatibility alias
LLMClient = LLMIntegration

async def create_rag_agent(agent_id: str = "rag_agent",
                     name: str = "RAG Agent",
                     tool_registry: Optional[ToolRegistry] = None,
                     query_parser: Optional[QueryParser] = None,
                     model: str = "gpt-4",
                     max_sources: int = 5,
                     **kwargs) -> RAGAgent:
    """
    Create and initialize a RAG agent with all necessary components.
    
    Args:
        agent_id: Unique identifier for the agent
        name: Human-readable name for the agent
        tool_registry: Registry of tools (if None, a new one will be created)
        query_parser: Query parser (if None, a new one will be created)
        model: LLM model to use
        max_sources: Maximum number of sources to retrieve
        **kwargs: Additional arguments for the agent
        
    Returns:
        An initialized RAG agent
    """
    logger.info(f"Creating RAG agent: {agent_id}")
    
    # Create components if not provided
    if tool_registry is None:
        tool_registry = ToolRegistry()
        
    # Get integrations first
    rag_integration = get_rag_integration()
    llm_integration = get_llm_integration(model=model)
        
    if query_parser is None:
        query_parser = QueryParser(llm_client=llm_integration)
        
    # Create and initialize the agent
    agent = RAGAgent(
        agent_id=agent_id,
        name=name,
        tool_registry=tool_registry,
        query_parser=query_parser,
        retrieval_system=rag_integration,
        llm_client=llm_integration,
        max_sources=max_sources
    )
    
    # Discover and register tools
    try:
        # First try the correct package path for tools
        tool_ids = tool_registry.discover_tools("agent_framework.tools")
        logger.info(f"Registered {len(tool_ids)} tools: {', '.join(tool_ids)}")
        
        # If no tools were discovered, try to import and register them directly
        if len(tool_ids) == 0:
            logger.warning("No tools discovered via package discovery, trying direct import")
            try:
                from agent_framework.tools.search_tools import (
                    query_reformulation, 
                    filter_search, 
                    extract_search_entities
                )
                
                # Register tools directly
                tool_registry.register(query_reformulation)
                tool_registry.register(filter_search)
                tool_registry.register(extract_search_entities)
                
                logger.info("Successfully registered tools via direct import: query_reformulation, filter_search, extract_search_entities")
            except ImportError as e:
                logger.error(f"Could not import tools directly: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error discovering tools: {str(e)}")
    
    return agent

async def create_orchestrator_agent(
    agent_id: str = "orchestrator",
    name: str = "Orchestrateur Principal GRC",
    llm_client: Optional[LLMClient] = None,
    **kwargs
) -> OrchestratorAgent:
    """
    Crée et initialise l'agent orchestrateur principal.
    
    Args:
        agent_id: Identifiant unique de l'agent
        name: Nom humain de l'agent
        llm_client: Client LLM (si None, utilise le client par défaut)
        **kwargs: Arguments supplémentaires
        
    Returns:
        Agent orchestrateur initialisé
    """
    logger.info(f"Création de l'agent orchestrateur: {agent_id}")
    
    # Créer le client LLM si non fourni
    if llm_client is None:
        llm_client = get_llm_client()
    
    # Créer l'agent orchestrateur
    orchestrator = OrchestratorAgent(llm_client=llm_client)
    
    return orchestrator

async def create_specialized_agents(
    orchestrator: OrchestratorAgent,
    rag_system = None,
    **kwargs
) -> Dict[str, Agent]:
    """
    Crée et enregistre les agents spécialisés dans l'orchestrateur.
    
    Args:
        orchestrator: Agent orchestrateur principal
        rag_system: Système RAG pour les agents
        **kwargs: Arguments supplémentaires
        
    Returns:
        Dictionnaire des agents spécialisés créés
    """
    logger.info("Création des agents spécialisés")
    
    specialized_agents = {}
    
    # Pour l'instant, créer des agents RAG spécialisés avec des paramètres différents
    # TODO: Implémenter les vrais agents spécialisés
    
    # Agent d'évaluation des risques
    risk_agent = await create_rag_agent(
        agent_id="risk_assessment",
        name="Agent d'Évaluation des Risques",
        **kwargs
    )
    specialized_agents["risk_assessment"] = risk_agent
    orchestrator.register_agent("risk_assessment", risk_agent)
    
    # Agent d'analyse de conformité 
    compliance_agent = await create_rag_agent(
        agent_id="compliance_analysis",
        name="Agent d'Analyse de Conformité",
        **kwargs
    )
    specialized_agents["compliance_analysis"] = compliance_agent
    orchestrator.register_agent("compliance_analysis", compliance_agent)
    
    # Agent d'analyse de gouvernance
    governance_agent = await create_rag_agent(
        agent_id="governance_analysis",
        name="Agent d'Analyse de Gouvernance",
        **kwargs
    )
    specialized_agents["governance_analysis"] = governance_agent
    orchestrator.register_agent("governance_analysis", governance_agent)
    
    logger.info(f"Agents spécialisés créés: {list(specialized_agents.keys())}")
    
    return specialized_agents

async def get_agent(agent_type: str, **kwargs) -> Agent:
    """
    Get an agent of the specified type.
    
    Args:
        agent_type: Type of agent to create
        **kwargs: Additional arguments for the agent
        
    Returns:
        An initialized agent
    """
    if agent_type == "rag":
        return await create_rag_agent(**kwargs)
    elif agent_type == "orchestrator":
        return await create_orchestrator_agent(**kwargs)
    else:
        logger.error(f"Unsupported agent type: {agent_type}")
        raise ValueError(f"Unsupported agent type: {agent_type}")

# Agent instances cache
_agent_instances = {}

async def get_agent_instance(agent_type: str, agent_id: Optional[str] = None, **kwargs) -> Agent:
    """
    Get a cached agent instance, creating it if it doesn't exist.
    
    Args:
        agent_type: Type of agent to get
        agent_id: Unique identifier for the agent (if None, a default ID will be used)
        **kwargs: Additional arguments for creating the agent
        
    Returns:
        An agent instance
    """
    global _agent_instances
    
    # Generate a default agent ID if not provided
    if agent_id is None:
        agent_id = f"{agent_type}_default"
        
    # Create a cache key
    cache_key = f"{agent_type}_{agent_id}"
    
    # Return cached instance if available
    if cache_key in _agent_instances:
        return _agent_instances[cache_key]
        
    # Create a new instance
    agent = await get_agent(agent_type, agent_id=agent_id, **kwargs)
    
    # Cache the instance
    _agent_instances[cache_key] = agent
    
    return agent 

async def initialize_complete_agent_system(rag_system=None, **kwargs) -> OrchestratorAgent:
    """
    Initialise le système complet d'agents avec orchestrateur et agents spécialisés.
    
    Args:
        rag_system: Système RAG à utiliser
        **kwargs: Arguments supplémentaires
        
    Returns:
        Agent orchestrateur avec tous les agents spécialisés enregistrés
    """
    logger.info("Initialisation du système complet d'agents")
    
    # Créer l'orchestrateur
    orchestrator = await create_orchestrator_agent(**kwargs)
    
    # Créer et enregistrer les agents spécialisés
    specialized_agents = await create_specialized_agents(
        orchestrator=orchestrator,
        rag_system=rag_system,
        **kwargs
    )
    
    logger.info(f"Système d'agents initialisé avec {len(specialized_agents)} agents spécialisés")
    
    return orchestrator 