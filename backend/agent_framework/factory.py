"""
Factory for creating and initializing agents in the RegulAIte Agent Framework.

This module provides factory functions for creating different types of agents
with all necessary components.
"""
from typing import Dict, List, Optional, Any, Union
import logging

from .agent import Agent
from .rag_agent import RAGAgent
from .tool_registry import ToolRegistry
from .query_parser import QueryParser
from .integrations.rag_integration import get_rag_integration
from .integrations.llm_integration import get_llm_integration

# Set up logging
logger = logging.getLogger(__name__)

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