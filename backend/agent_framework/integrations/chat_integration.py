"""
Chat Integration for the RegulAIte Agent Framework.

This module provides integration between the chat router and the agent framework.
"""
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import sys
from pathlib import Path
import json
import time
import asyncio

# Set up logging
logger = logging.getLogger(__name__)

# Remove import at the module level to fix circular imports
# from ..factory import get_agent_instance
from ..agent import Query, AgentResponse, QueryContext, Agent
from ..response_generator import ResponseGenerator, ResponseFormat
from ..rag_agent import RAGAgent

class ChatIntegration:
    """
    Integration layer between the chat system and agent framework.
    
    This class handles the communication between the chat API and
    the various agents, managing request routing and response formatting.
    """
    
    def __init__(self):
        """Initialize the chat integration."""
        self.response_generator = ResponseGenerator()
        self.logger = logging.getLogger("chat_integration")
        
        # Available agent types
        self.agent_types = {
            "rag": RAGAgent,
            "general": Agent
        }
        
        # Step streaming callback
        self.step_callback = None
        
    def set_step_callback(self, callback: Callable):
        """
        Set a callback function to receive agent processing steps.
        
        Args:
            callback: Function to call with step updates
        """
        self.step_callback = callback
        
    async def _emit_agent_step(self, step_data: Dict[str, Any]):
        """
        Emit an agent processing step.
        
        Args:
            step_data: Step information to emit
        """
        if self.step_callback:
            try:
                if asyncio.iscoroutinefunction(self.step_callback):
                    await self.step_callback(step_data)
                else:
                    self.step_callback(step_data)
            except Exception as e:
                self.logger.error(f"Error in step callback: {e}")
        
    async def process_chat_request(
        self, 
        request_data: Dict[str, Any], 
        use_agent: bool = True,  # Always True - agents are always enabled
        agent_type: str = "rag"
    ) -> Dict[str, Any]:
        """
        Process a chat request using the agent framework.
        
        Args:
            request_data: The chat request data
            use_agent: Whether to use agent processing (always True)
            agent_type: Type of agent to use
            
        Returns:
            Dict containing the response and metadata
        """
        # Force agent usage - agents are always enabled
        use_agent = True
        
        try:
            # Extract request information
            messages = request_data.get("messages", [])
            model = request_data.get("model", "gpt-4")
            session_id = request_data.get("session_id")
            include_context = request_data.get("include_context", True)
            context_query = request_data.get("context_query")
            response_format = request_data.get("response_format", "markdown")
            
            # Get the user message (last message should be from user)
            user_message = ""
            if messages:
                user_message = messages[-1].get("content", "")
            
            if not user_message:
                return {
                    "message": "No user message found in the request.",
                    "error": True,
                    "model": model,
                    "agent_type": agent_type,
                    "agent_used": False,
                    "context_used": False,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
            
            # Record start time for performance measurement
            start_time = time.time()
            
            # Emit initial step
            await self._emit_agent_step({
                "step": "initialization",
                "message": "Initializing agent processing...",
                "details": f"Using {agent_type} agent with {model} model",
                "progress": 5,
                "timestamp": time.time()
            })
            
            # Autonomously determine the best agent type - for now we use RAG as it's most general
            # In the future, this could analyze the query to determine the best agent type
            agent_type = "rag"
            
            # Get the agent instance without requiring manual parameters
            agent = await get_agent_instance(
                agent_type=agent_type,
                model=model
            )
            
            # Set up step callback for the agent
            execution_id = f"exec_{int(time.time())}_{session_id}"
            agent.set_step_callback(self._emit_agent_step, execution_id)
            
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
            
            # Emit agent start step
            await self._emit_agent_step({
                "step": "agent_start",
                "message": f"Starting {agent_type} agent processing...",
                "details": f"Query: {user_message[:100]}{'...' if len(user_message) > 100 else ''}",
                "progress": 15,
                "timestamp": time.time(),
                "execution_id": execution_id
            })
            
            # Process the query
            agent_response = await agent.process_query(query)
            
            # Ensure we have a valid response
            if not agent_response or not agent_response.content:
                await self._emit_agent_step({
                    "step": "agent_error",
                    "message": "Agent failed to generate a response",
                    "details": "No content returned from agent",
                    "timestamp": time.time(),
                    "execution_id": execution_id
                })
                
                return {
                    "message": "I apologize, but I wasn't able to generate a response to your query. Please try rephrasing your question.",
                    "error": False,
                    "model": model,
                    "agent_type": agent_type,
                    "agent_used": True,  # Always True since agents are always enabled
                    "context_used": False,
                    "session_id": session_id,
                    "timestamp": time.time()
                }
            
            # Format the response
            await self._emit_agent_step({
                "step": "response_formatting",
                "message": "Formatting response...",
                "details": f"Response length: {len(agent_response.content)} characters",
                "progress": 95,
                "timestamp": time.time(),
                "execution_id": execution_id
            })
            
            formatted_response = await self.response_generator.generate(
                response=agent_response,
                query=query,
                format=response_format
            )
            
            # Extract the content string from the FormattedResponse object
            if hasattr(formatted_response, 'content'):
                response_content = formatted_response.content
            elif isinstance(formatted_response, str):
                response_content = formatted_response
            else:
                # Fallback: convert to string
                response_content = str(formatted_response)
                
            # Ensure we have a valid string
            if not isinstance(response_content, str):
                response_content = str(response_content)
                
            # Ensure the response is not empty
            if not response_content.strip():
                response_content = "I apologize, but I wasn't able to generate a meaningful response to your query. Please try rephrasing your question."
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Final completion step
            await self._emit_agent_step({
                "step": "completion",
                "message": "Agent processing completed successfully!",
                "details": f"Total execution time: {execution_time:.2f}s",
                "progress": 100,
                "timestamp": time.time(),
                "execution_id": execution_id
            })
            
            # Return the formatted response with metadata
            return {
                "message": response_content,  # Use the extracted content string
                "error": False,
                "model": model,
                "agent_type": agent_type,
                "agent_used": True,  # Always True since agents are always enabled
                "context_used": agent_response.context_used,
                "tools_used": agent_response.tools_used,
                "sources": getattr(agent_response, 'sources', []),
                "metadata": getattr(agent_response, 'metadata', {}),
                "session_id": session_id,
                "execution_time": execution_time,
                "execution_id": execution_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in agent processing: {str(e)}")
            
            # Emit error step
            await self._emit_agent_step({
                "step": "error",
                "message": f"Agent processing failed: {str(e)}",
                "details": str(e),
                "timestamp": time.time()
            })
            
            return {
                "message": f"I encountered an error while processing your request: {str(e)}",
                "error": True,
                "model": model,
                "agent_type": agent_type,
                "agent_used": True,  # Always True since agents are always enabled
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

# Global instance
_chat_integration = None

def get_chat_integration() -> ChatIntegration:
    """Get the global chat integration instance."""
    global _chat_integration
    if _chat_integration is None:
        _chat_integration = ChatIntegration()
    return _chat_integration

async def get_agent_instance(agent_type: str = "rag", model: str = "gpt-4") -> Agent:
    """
    Get an agent instance of the specified type.
    
    Args:
        agent_type: Type of agent to create
        model: Model to use with the agent
        
    Returns:
        Agent instance
    """
    # Import the factory function to properly create agents with all integrations
    from ..factory import get_agent_instance as factory_get_agent_instance
    
    # Use the factory function which properly initializes all integrations
    return await factory_get_agent_instance(
        agent_type=agent_type,
        model=model
    ) 