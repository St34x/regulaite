"""
Chat Integration for the RegulAIte Agent Framework.

This module provides integration between the chat router and the agent framework.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import sys
from pathlib import Path
import json
import time

# Set up logging
logger = logging.getLogger(__name__)

# Remove import at the module level to fix circular imports
# from ..factory import get_agent_instance
from ..agent import Query, AgentResponse, QueryContext
from ..response_generator import ResponseGenerator, ResponseFormat

class ChatIntegration:
    """
    Integration between the chat router and the agent framework.
    
    This class provides methods for processing chat requests using the agent
    framework.
    """
    
    def __init__(self):
        """Initialize the chat integration."""
        self.response_generator = ResponseGenerator()
        
    async def process_chat_request(self, 
                                  request_data: Dict[str, Any], 
                                  use_agent: bool = True) -> Dict[str, Any]:
        """
        Process a chat request using the agent framework autonomously.
        
        Args:
            request_data: The chat request data
            use_agent: Whether to use an agent (defaults to True for autonomous operation)
            
        Returns:
            The response from the agent
        """
        # Import get_agent_instance lazily to avoid circular imports
        from ..factory import get_agent_instance
        
        # Extract information from the request
        messages = request_data.get("messages", [])
        model = request_data.get("model", "gpt-4.1")
        session_id = request_data.get("session_id")
        include_context = request_data.get("include_context", True)
        context_query = request_data.get("context_query")
        response_format = request_data.get("response_format", "text")
        
        # Get the last user message
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
                
        if not user_message:
            return {
                "message": "No user message found in the request",
                "error": True,
                "model": model,
                "agent_used": False,
                "context_used": False,
                "session_id": session_id,
                "timestamp": time.time()
            }
            
        try:
            # Record start time for performance measurement
            start_time = time.time()
            
            # Autonomously determine the best agent type - for now we use RAG as it's most general
            # In the future, this could analyze the query to determine the best agent type
            agent_type = "rag"
            
            # Get the agent instance without requiring manual parameters
            agent = await get_agent_instance(
                agent_type=agent_type,
                model=model
            )
            
            # Create query context
            query_context = QueryContext(
                session_id=session_id,
                metadata={"previous_messages": messages[:-1] if len(messages) > 1 else []}
            )
            
            # Create the query
            query = Query(
                query_text=user_message,
                context=query_context
            )
            
            # Process the query
            agent_response = await agent.process_query(query)
            
            # Ensure we have a valid response
            if not agent_response or not agent_response.content:
                return {
                    "message": "I apologize, but I wasn't able to generate a response to your query. Please try rephrasing your question.",
                    "error": False,
                    "model": model,
                    "agent_type": agent_type,
                    "agent_used": True,
                    "context_used": False,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
            
            # Format the response
            formatted_response = await self.response_generator.generate(
                response=agent_response,
                query=query,
                format=response_format
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create the response object
            response = {
                "message": formatted_response.content,
                "model": model,
                "agent_type": agent_type,
                "agent_used": True,
                "tree_reasoning_used": False,
                "context_used": agent_response.context_used,
                "session_id": session_id,
                "timestamp": formatted_response.timestamp,
                "execution_time": execution_time,
                "sources": agent_response.metadata.get("sources"),
                "tools_used": agent_response.tools_used,
                "metadata": agent_response.metadata
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing chat request with agent: {str(e)}")
            return {
                "message": f"I encountered an error while processing your request: {str(e)}. Please try again with a different approach.",
                "error": True,
                "model": model,
                "agent_used": True,
                "context_used": False,
                "session_id": session_id,
                "timestamp": time.time()
            }
    
    async def get_agent_for_chat(self, model: str) -> Dict[str, Any]:
        """
        Get information about the autonomous agent for the chat UI.
        
        Args:
            model: Model to use
            
        Returns:
            Information about the agent
        """
        # Import get_agent_instance lazily to avoid circular imports
        from ..factory import get_agent_instance
        
        try:
            # Autonomously determine the best agent type
            agent_type = "rag"
            
            # Get the agent instance
            agent = await get_agent_instance(
                agent_type=agent_type,
                model=model
            )
            
            # Get information about the agent
            return {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "type": agent_type,
                "model": model,
                "available": True
            }
            
        except Exception as e:
            logger.error(f"Error getting agent information: {str(e)}")
            return {
                "agent_id": "unknown",
                "name": "Autonomous Agent",
                "type": "autonomous",
                "model": model,
                "available": False,
                "error": str(e)
            }

# Singleton instance
_chat_integration = None

def get_chat_integration():
    """
    Get the chat integration instance.
    
    Returns:
        The chat integration instance
    """
    global _chat_integration
    
    if _chat_integration is None:
        _chat_integration = ChatIntegration()
        
    return _chat_integration 