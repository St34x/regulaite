"""
FastAPI router for chat endpoints, chat history management and agent integration.
"""
import logging
import json
import uuid
import time
from typing import List, Dict, Any, Optional, Literal, Union
from fastapi import APIRouter, Depends, HTTPException, Body, Request, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import os
from fastapi.responses import StreamingResponse
import mariadb
from datetime import timedelta
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# Models for API
class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    stream: bool = Field(False, description="Whether to stream the response")
    model: str = Field("gpt-4", description="Model to use for generation")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens in response")
    include_context: bool = Field(True, description="Whether to include RAG context")
    context_query: Optional[str] = Field(None, description="Query to use for retrieving context")
    retrieval_type: Optional[str] = Field("auto", description="Type of retrieval to use: 'hybrid', 'vector', or 'auto' (default)")
    use_agent: bool = Field(False, description="Whether to use an agent for processing")
    agent_type: Optional[str] = Field(None, description="Type of agent to use if use_agent is True")
    agent_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the agent")
    use_tree_reasoning: bool = Field(False, description="Whether to use tree-based reasoning")
    tree_template: Optional[str] = Field(None, description="ID of the decision tree template to use")
    custom_tree: Optional[Dict[str, Any]] = Field(None, description="Custom decision tree for reasoning")
    session_id: Optional[str] = Field(None, description="Session ID for chat history")


class ChatResponse(BaseModel):
    """Response for a chat completion."""
    message: str = Field(..., description="Assistant response message")
    model: str = Field(..., description="Model used for generation")
    agent_type: Optional[str] = Field(None, description="Type of agent used (if any)")
    agent_used: bool = Field(False, description="Whether an agent was used")
    tree_reasoning_used: bool = Field(False, description="Whether tree reasoning was used")
    context_used: bool = Field(False, description="Whether context was used")
    session_id: str = Field(..., description="Session ID for chat history")
    timestamp: str = Field(..., description="Timestamp of the response")
    execution_id: Optional[str] = Field(None, description="ID of the execution for tracking progress")


class ChatHistoryEntry(BaseModel):
    """Entry in chat history."""
    message_text: str = Field(..., description="Message text")
    message_role: str = Field(..., description="Message role (user or assistant)")
    timestamp: str = Field(..., description="Timestamp of the message")


class ChatHistoryResponse(BaseModel):
    """Response for chat history."""
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatHistoryEntry] = Field(..., description="Chat messages")
    count: int = Field(..., description="Number of messages")


class ChatSessionsResponse(BaseModel):
    """Response for listing chat sessions."""
    sessions: List[Dict[str, Any]] = Field(..., description="List of chat sessions")
    count: int = Field(..., description="Number of sessions")


class AgentProgressResponse(BaseModel):
    """Response for agent execution progress."""
    execution_id: str = Field(..., description="ID of the execution")
    agent_id: str = Field(..., description="ID of the agent")
    progress_percent: float = Field(..., description="Percentage of completion (0-100)")
    status: str = Field(..., description="Status of the execution (running, completed, failed)")
    status_message: Optional[str] = Field(None, description="Status message or description")
    timestamp: str = Field(..., description="Timestamp of the progress update")


# Dependency to get the database connection
async def get_db_connection():
    """Get the MariaDB database connection from main application."""
    from main import get_mariadb_connection
    return get_mariadb_connection()


# Dependency to get the RAG system
async def get_rag_system():
    """Get the RAG system from main application."""
    from main import rag_system
    return rag_system


# Utility function to track agent execution
async def track_agent_execution(
    agent_id: str,
    session_id: str,
    task: str,
    model: str,
    start_time: float,
    tokens: Optional[Dict[str, int]] = None,
    error: bool = False,
    error_message: str = None
) -> int:
    """
    Track agent execution in the database.
    
    Args:
        agent_id: ID of the agent
        session_id: ID of the chat session
        task: The task or query for the agent
        model: Model used for the agent
        start_time: Start time of execution (as returned by time.time())
        tokens: Optional dictionary with token counts
        error: Whether an error occurred
        error_message: Error message if an error occurred
        
    Returns:
        ID of the execution record
    """
    try:
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Get token counts if available
        prompt_tokens = tokens.get("prompt_tokens", 0) if tokens else 0
        completion_tokens = tokens.get("completion_tokens", 0) if tokens else 0
        total_tokens = tokens.get("total_tokens", 0) if tokens else 0
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor()
        
        # Insert execution record
        cursor.execute(
            """
            INSERT INTO agent_executions (
                agent_id, session_id, task, model,
                response_time_ms, token_count, prompt_token_count,
                completion_token_count, error, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id, session_id, task, model,
                response_time_ms, total_tokens, prompt_tokens,
                completion_tokens, error, error_message
            )
        )
        
        conn.commit()
        execution_id = cursor.lastrowid
        
        # Initialize progress at 100% if already completed
        if not error:
            cursor.execute(
                """
                INSERT INTO agent_progress (
                    execution_id, progress_percent, status, status_message
                ) VALUES (?, ?, ?, ?)
                """,
                (execution_id, 100.0, "completed", "Task completed successfully")
            )
        else:
            cursor.execute(
                """
                INSERT INTO agent_progress (
                    execution_id, progress_percent, status, status_message
                ) VALUES (?, ?, ?, ?)
                """,
                (execution_id, 0.0, "failed", error_message or "Task failed")
            )
            
        conn.commit()
        conn.close()
        
        return execution_id
    except Exception as e:
        logger.error(f"Error tracking agent execution: {str(e)}")
        return None


# Background task for updating analytics
async def update_agent_analytics(agent_id: str, execution_id: int, rating: Optional[int] = None):
    """
    Update agent analytics based on execution data.
    
    Args:
        agent_id: ID of the agent
        execution_id: ID of the execution
        rating: Optional rating from feedback
    """
    try:
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get execution data
        cursor.execute(
            """
            SELECT 
                response_time_ms, error,
                DATE(timestamp) as execution_date,
                user_id
            FROM agent_executions ae
            JOIN chat_history ch ON ae.session_id = ch.session_id
            WHERE ae.id = ?
            LIMIT 1
            """,
            (execution_id,)
        )
        
        execution = cursor.fetchone()
        if not execution:
            conn.close()
            return
            
        # Get or create analytics record for the day
        cursor.execute(
            """
            SELECT * FROM agent_analytics
            WHERE agent_id = ? AND day = ?
            """,
            (agent_id, execution["execution_date"])
        )
        
        analytics = cursor.fetchone()
        
        if analytics:
            # Update existing record
            cursor.execute(
                """
                UPDATE agent_analytics SET
                    execution_count = execution_count + 1,
                    avg_response_time_ms = ((avg_response_time_ms * execution_count) + ?) / (execution_count + 1),
                    error_rate = ((error_rate * execution_count) + ?) / (execution_count + 1)
                WHERE id = ?
                """,
                (
                    execution["response_time_ms"] or 0,
                    1 if execution["error"] else 0,
                    analytics["id"]
                )
            )
        else:
            # Create new record
            cursor.execute(
                """
                INSERT INTO agent_analytics (
                    agent_id, day, execution_count, avg_response_time_ms, error_rate
                ) VALUES (?, ?, 1, ?, ?)
                """,
                (
                    agent_id,
                    execution["execution_date"],
                    execution["response_time_ms"] or 0,
                    1 if execution["error"] else 0
                )
            )
            
        # Update rating if provided
        if rating is not None and analytics:
            cursor.execute(
                """
                UPDATE agent_analytics SET
                    avg_rating = ((avg_rating * IFNULL(rating_count, 0)) + ?) / (IFNULL(rating_count, 0) + 1),
                    rating_count = IFNULL(rating_count, 0) + 1
                WHERE id = ?
                """,
                (rating, analytics["id"])
            )
            
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error updating agent analytics: {str(e)}")


# Helper function to extract user ID from request
async def extract_user_id_from_request(req: Request, provided_user_id: Optional[str] = None):
    """Extract user ID from request headers or provided user_id parameter.
    
    Args:
        req: FastAPI Request object
        provided_user_id: User ID directly provided to the endpoint
        
    Returns:
        User ID if found, None otherwise
    """
    user_id = provided_user_id
    
    # If no user_id is provided, try to get it from the request header
    if not user_id and req:
        user_id = req.headers.get("X-User-ID")
        
        # If still no user_id, try to extract from Authorization header
        if not user_id and "Authorization" in req.headers:
            auth_header = req.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    token = auth_header.replace("Bearer ", "")
                    # Import here to avoid circular imports
                    import jwt
                    from routers.auth_router import SECRET_KEY, ALGORITHM
                    
                    # Decode token and extract user ID
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    user_id = payload.get("sub") or payload.get("user_id")
                except Exception as e:
                    logger.error(f"Error extracting user ID from token: {str(e)}")
    
    return user_id


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request, background_tasks: BackgroundTasks):
    """Chat with the RAG-enhanced LLM or AI agent."""
    try:
        execution_id = None
        execution_start_time = time.time()

        # Get messages from request
        messages = request.messages

        # Get the last user message
        user_message = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found in the chat history"
            )
            
        # Handle short follow-up messages (like "20", "yes", etc.) by adding context
        if len(user_message.strip()) <= 20 and len(messages) >= 2:
            # Find the previous user and assistant messages
            previous_messages = list(messages)
            previous_messages.reverse()
            
            # Skip the current user message
            previous_messages = previous_messages[1:]
            
            # Look for the previous user message and assistant response
            prev_user_message = next((m.content for m in previous_messages if m.role == "user"), None)
            prev_assistant_message = next((m.content for m in previous_messages if m.role == "assistant"), None)
            
            # If we found both, add a system message with context
            if prev_user_message and prev_assistant_message:
                context_message = ChatMessage(
                    role="system",
                    content=f"The user previously asked: \"{prev_user_message}\". You responded with: \"{prev_assistant_message}\". The user's follow-up message is: \"{user_message}\". Remember to maintain context from the previous exchange, especially if the follow-up message is short or ambiguous."
                )
                messages.insert(0, context_message)

        # Initialize database connection for chat history
        try:
            conn = await get_db_connection()
            cursor = conn.cursor()
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error connecting to database: {str(e)}"
            )

        # Generate a session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        user_id = await extract_user_id_from_request(req)
        
        # Authentication is always required for chat functionality
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for chat functionality"
            )

        # Store user message in chat history
        try:
            cursor.execute(
                """
                INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, session_id, user_message, "user")
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing user message in chat history: {str(e)}")
            # Continue anyway, as this is not critical

        # Get the RAG system
        rag_system = await get_rag_system()
        
        # Track token usage
        token_usage = {}

        # If agent-based processing is requested
        if request.use_agent and request.agent_type:
            from pyndantic_agents.agents import create_agent, AgentConfig
            from pyndantic_agents.tree_reasoning import TreeReasoningAgent, DecisionTree
            from pyndantic_agents.decision_trees import get_available_trees, create_default_decision_tree

            # Create agent execution tracking in progress state
            if not request.stream:
                # For non-streaming, we'll create a background task for progress updates
                cursor.execute(
                    """
                    INSERT INTO agent_executions (
                        agent_id, session_id, task, model, error
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (request.agent_type, session_id, user_message, request.model, False)
                )
                conn.commit()
                execution_id = cursor.lastrowid
                
                # Initialize progress
                cursor.execute(
                    """
                    INSERT INTO agent_progress (
                        execution_id, progress_percent, status, status_message
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (execution_id, 0.0, "running", "Task started")
                )
                conn.commit()

            # Handle tree-based reasoning if requested
            if request.use_tree_reasoning:
                # Create agent configuration
                agent_config = AgentConfig(
                    name="Tree Reasoning Agent",
                    description="Agent for tree-based reasoning",
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    include_context=request.include_context,
                    context_query=request.context_query,
                    max_context_results=5
                )

                # Determine which decision tree to use
                decision_tree = None

                # Use custom tree if provided
                if request.custom_tree:
                    try:
                        decision_tree = DecisionTree(**request.custom_tree)
                        logger.info(f"Using custom decision tree: {decision_tree.name}")
                    except Exception as e:
                        logger.error(f"Error parsing custom decision tree: {str(e)}")
                        # Fall through to other options if parsing fails

                # Use template tree if specified (and custom tree wasn't used)
                if decision_tree is None and request.tree_template:
                    available_trees = get_available_trees()
                    if request.tree_template == "default":
                        decision_tree = create_default_decision_tree()
                        logger.info(f"Using default decision tree template")
                    elif request.tree_template in available_trees:
                        decision_tree = available_trees[request.tree_template]
                        logger.info(f"Using template tree: {request.tree_template} - {decision_tree.name}")
                    else:
                        logger.warning(f"Template '{request.tree_template}' not found, using default tree")
                        decision_tree = create_default_decision_tree()

                # Use default decision tree if no other option was successful
                if decision_tree is None:
                    decision_tree = create_default_decision_tree()
                    logger.info(f"Using default decision tree (none specified or failed to parse)")

                # Create tree reasoning agent with Neo4j credentials from env vars or parent application
                from main import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY

                # Update progress if available
                if execution_id:
                    cursor.execute(
                        """
                        UPDATE agent_progress SET
                            progress_percent = ?, status_message = ?
                        WHERE execution_id = ?
                        """,
                        (10.0, "Initializing tree reasoning agent", execution_id)
                    )
                    conn.commit()

                agent = TreeReasoningAgent(
                    agent_id=f"tree_agent_{uuid.uuid4()}",
                    config=agent_config,
                    decision_tree=decision_tree,
                    neo4j_uri=NEO4J_URI,
                    neo4j_user=NEO4J_USER,
                    neo4j_password=NEO4J_PASSWORD,
                    openai_api_key=OPENAI_API_KEY,
                    rag_system=rag_system
                )

                try:
                    # Update progress if available
                    if execution_id:
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                progress_percent = ?, status_message = ?
                            WHERE execution_id = ?
                            """,
                            (25.0, "Executing tree reasoning", execution_id)
                        )
                        conn.commit()
                    
                    # Execute the task
                    result = agent.execute(user_message)
                    
                    # Get token usage if available
                    if hasattr(agent, 'usage') and agent.usage:
                        token_usage = agent.usage
                        
                    # Update progress if available
                    if execution_id:
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                progress_percent = ?, status_message = ?
                            WHERE execution_id = ?
                            """,
                            (75.0, "Formatting response", execution_id)
                        )
                        conn.commit()

                    # Clean up
                    agent.close()

                    # Format response from tree reasoning
                    if "final_result" in result and "response" in result["final_result"]:
                        assistant_message = result["final_result"]["response"]
                    elif "final_result" in result and isinstance(result["final_result"], dict) and "completion" in result["final_result"]:
                        assistant_message = result["final_result"]["completion"]
                    elif "final_result" in result:
                        assistant_message = str(result["final_result"])
                    else:
                        assistant_message = f"Tree-based reasoning completed. Visited {len(result.get('visited_nodes', []))} decision nodes."

                    # Include reasoning explanation if available
                    if "reasoning_explanation" in result:
                        # Only include the explanation if specifically requested or for debugging
                        if request.agent_params and request.agent_params.get("show_reasoning", False):
                            assistant_message += "\n\n" + result["reasoning_explanation"]
                            
                    # Update execution record with successful completion
                    if execution_id:
                        # Track execution in background task
                        background_tasks.add_task(
                            update_agent_analytics,
                            "tree_reasoning",
                            execution_id
                        )
                        
                        # Update progress to 100%
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                progress_percent = 100.0, status = 'completed', status_message = ?
                            WHERE execution_id = ?
                            """,
                            ("Tree reasoning completed successfully", execution_id)
                        )
                        conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error in tree reasoning agent: {str(e)}")
                    
                    # Update execution record with error
                    if execution_id:
                        cursor.execute(
                            """
                            UPDATE agent_executions SET
                                error = 1, error_message = ?
                            WHERE id = ?
                            """,
                            (str(e), execution_id)
                        )
                        
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                status = 'failed', status_message = ?, progress_percent = 0
                            WHERE execution_id = ?
                            """,
                            (f"Error: {str(e)}", execution_id)
                        )
                        conn.commit()
                        
                    # Set a fallback error message
                    assistant_message = f"I encountered an error while processing your request with tree reasoning: {str(e)}. Please try again or use a different approach."

                # Handle streaming for tree-based reasoning responses if requested
                if request.stream:
                    async def generate():
                        # Start streaming response
                        yield "data: {\"event\": \"start\", \"session_id\": \"" + session_id + "\"}\n\n"
                        
                        # For tree-based reasoning responses, we deliver the entire result at once
                        yield f"data: {json.dumps({'content': assistant_message, 'event': 'chunk'})}\n\n"
                        
                        # End streaming response
                        yield f"data: {json.dumps({'event': 'end'})}\n\n"
                        
                        # Store the assistant response in chat history
                        try:
                            cursor.execute(
                                """
                                INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                                VALUES (?, ?, ?, ?)
                                """,
                                (user_id, session_id, assistant_message, "assistant")
                            )
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Error storing assistant response in chat history: {str(e)}")
                        
                        # Close database connection
                        conn.close()
                    
                    return StreamingResponse(generate(), media_type="text/event-stream")

            else:
                # Standard agent-based processing
                # Create an agent configuration
                agent_config = AgentConfig(
                    name=f"{request.agent_type.capitalize()} Agent",
                    description=f"Agent for handling {request.agent_type} tasks",
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    include_context=request.include_context,
                    context_query=request.context_query,
                    max_context_results=5
                )

                # Get Neo4j and OpenAI credentials from env vars or parent application
                from main import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY

                # Update progress if available
                if execution_id:
                    cursor.execute(
                        """
                        UPDATE agent_progress SET
                            progress_percent = ?, status_message = ?
                        WHERE execution_id = ?
                        """,
                        (20.0, f"Initializing {request.agent_type} agent", execution_id)
                    )
                    conn.commit()

                try:
                    # Create the agent
                    agent = create_agent(
                        agent_type=request.agent_type,
                        config=agent_config,
                        neo4j_uri=NEO4J_URI,
                        neo4j_user=NEO4J_USER,
                        neo4j_password=NEO4J_PASSWORD,
                        openai_api_key=OPENAI_API_KEY,
                        rag_system=rag_system
                    )
                    
                    # Update progress if available
                    if execution_id:
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                progress_percent = ?, status_message = ?
                            WHERE execution_id = ?
                            """,
                            (40.0, f"Executing {request.agent_type} agent", execution_id)
                        )
                        conn.commit()

                    # Execute the task using the agent
                    result = agent.execute(user_message)
                    
                    # Get token usage if available
                    if hasattr(agent, 'usage') and agent.usage:
                        token_usage = agent.usage
                        
                    # Update progress if available
                    if execution_id:
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                progress_percent = ?, status_message = ?
                            WHERE execution_id = ?
                            """,
                            (80.0, "Processing agent result", execution_id)
                        )
                        conn.commit()

                    # Clean up the agent
                    agent.close()

                    # Format agent response
                    if "analysis" in result:
                        assistant_message = result["analysis"]
                    elif "summary" in result:
                        assistant_message = result["summary"]
                    else:
                        assistant_message = str(result)
                    
                    # Update execution record with successful completion
                    if execution_id:
                        # Track execution in background task
                        background_tasks.add_task(
                            update_agent_analytics,
                            request.agent_type,
                            execution_id
                        )
                        
                        # Update progress to 100%
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                progress_percent = 100.0, status = 'completed', status_message = ?
                            WHERE execution_id = ?
                            """,
                            (f"{request.agent_type.capitalize()} agent completed successfully", execution_id)
                        )
                        conn.commit()
                        
                except Exception as e:
                    logger.error(f"Error in {request.agent_type} agent: {str(e)}")
                    
                    # Update execution record with error
                    if execution_id:
                        cursor.execute(
                            """
                            UPDATE agent_executions SET
                                error = 1, error_message = ?
                            WHERE id = ?
                            """,
                            (str(e), execution_id)
                        )
                        
                        cursor.execute(
                            """
                            UPDATE agent_progress SET
                                status = 'failed', status_message = ?, progress_percent = 0
                            WHERE execution_id = ?
                            """,
                            (f"Error: {str(e)}", execution_id)
                        )
                        conn.commit()
                        
                    # Set a fallback error message
                    assistant_message = f"I encountered an error while processing your request with the {request.agent_type} agent: {str(e)}. Please try again or use a different approach."

                # Handle streaming for agent responses if requested
                if request.stream:
                    async def generate():
                        # Start streaming response
                        yield "data: {\"event\": \"start\", \"session_id\": \"" + session_id + "\"}\n\n"
                        
                        # For agent responses, we deliver the entire result at once
                        # since agents typically don't provide incremental results
                        yield f"data: {json.dumps({'content': assistant_message, 'event': 'chunk'})}\n\n"
                        
                        # End streaming response
                        yield f"data: {json.dumps({'event': 'end'})}\n\n"
                        
                        # Store the assistant response in chat history
                        try:
                            cursor.execute(
                                """
                                INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                                VALUES (?, ?, ?, ?)
                                """,
                                (user_id, session_id, assistant_message, "assistant")
                            )
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Error storing assistant response in chat history: {str(e)}")
                        
                        # Close database connection
                        conn.close()
                    
                    return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            # Standard RAG-based processing
            # Import OpenAI client from main application
            from main import OpenAI, OPENAI_API_KEY

            # Create OpenAI client
            client = OpenAI(api_key=OPENAI_API_KEY)

            # Determine if we should use context
            if request.include_context:
                # Use context_query if provided, otherwise use the user message
                query = request.context_query or user_message

                # Retrieve context using RAG system
                try:
                    # Convert retrieval_type to use_hybrid parameter
                    use_hybrid = None  # Default (auto)
                    if request.retrieval_type == "hybrid":
                        use_hybrid = True
                    elif request.retrieval_type == "vector":
                        use_hybrid = False
                        
                    retrieved_nodes = rag_system.retrieve(
                        query, 
                        top_k=5, 
                        use_hybrid=use_hybrid
                    )
                except Exception as e:
                    logger.error(f"Error retrieving context from RAG system: {str(e)}")
                    # Continue without context if retrieval fails
                    retrieved_nodes = []

                # Format context
                if retrieved_nodes:
                    context_parts = []
                    for node in retrieved_nodes:
                        source = f"{node['metadata'].get('doc_name', 'Unknown document')}"
                        if 'section' in node['metadata'] and node['metadata']['section'] != 'Unknown':
                            source += f", Section: {node['metadata']['section']}"

                        context_parts.append(f"Content: {node['text']}\nSource: {source}\n")

                    context_text = "\n".join(context_parts)

                    # Add context to messages
                    system_message = next((m for m in messages if m.role == "system"), None)
                    if system_message:
                        system_message.content += f"\n\nContext for answering this question:\n{context_text}"
                    else:
                        messages.insert(0, ChatMessage(
                            role="system",
                            content=f"You are an AI assistant with access to the following context information:\n\n{context_text}"
                        ))
                else:
                    # Add a default system message if no context was retrieved
                    if not any(m.role == "system" for m in messages):
                        messages.insert(0, ChatMessage(
                            role="system",
                            content="You are an AI assistant that helps users with questions. If you don't know the answer, just say so."
                        ))

            # Add special handling for counting or numbering tasks
            # Get the last two messages if they exist
            if len(messages) >= 3:
                # Check if we're in a counting or numbering scenario
                user_messages = [m for m in messages if m.role == "user"]
                counting_pattern = re.compile(r'^count\s+(to|until|from)\s+\d+(\s+to\s+\d+)?$', re.IGNORECASE)
                
                # Check if previous message was about counting
                if len(user_messages) >= 2:
                    previous_user_msg = user_messages[-2].content.lower()
                    current_user_msg = user_messages[-1].content.lower()
                    
                    # Check if we have a counting request followed by a number
                    if counting_pattern.match(previous_user_msg) and re.match(r'^\d+$', current_user_msg):
                        # Add specific system instruction for sequential counting
                        messages.insert(0, ChatMessage(
                            role="system",
                            content="IMPORTANT: The user is engaging in a counting exercise. If they provide just a number like '20', you should interpret this as a request to count to that number. For example, if they say '20', respond by counting from 1 to 20."
                        ))
                    
                    # Check for simple number sequence
                    if re.match(r'^\d+$', previous_user_msg) and re.match(r'^\d+$', current_user_msg):
                        try:
                            prev_num = int(previous_user_msg)
                            curr_num = int(current_user_msg)
                            # If the second number could be a continuation
                            if curr_num > prev_num and curr_num <= prev_num + 100:
                                messages.insert(0, ChatMessage(
                                    role="system",
                                    content=f"IMPORTANT: The user seems to be continuing a number sequence. The previous number was {prev_num}, and now they've provided {curr_num}. If they're counting, continue the sequence from {curr_num} by providing the next numbers in order."
                                ))
                        except ValueError:
                            pass  # Not numeric values

            # Convert messages for OpenAI API
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Get chat completion
            if request.stream:
                # Handle streaming response
                async def generate():
                    # Start streaming response
                    yield "data: {\"event\": \"start\", \"session_id\": \"" + session_id + "\"}\n\n"
                    
                    full_response = ""
                    
                    # Create streaming completion
                    stream = client.chat.completions.create(
                        model=request.model,
                        messages=openai_messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=True
                    )
                    
                    # Stream chunks to the client
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield f"data: {json.dumps({'content': content, 'event': 'chunk'})}\n\n"
                    
                    # End streaming response
                    yield f"data: {json.dumps({'event': 'end'})}\n\n"
                    
                    # Store the full assistant response in chat history
                    try:
                        cursor.execute(
                            """
                            INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                            VALUES (?, ?, ?, ?)
                            """,
                            (user_id, session_id, full_response, "assistant")
                        )
                        conn.commit()
                    except Exception as e:
                        logger.error(f"Error storing assistant response in chat history: {str(e)}")
                    
                    # Close database connection
                    conn.close()
                
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                response = client.chat.completions.create(
                    model=request.model,
                    messages=openai_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                # Extract assistant message
                assistant_message = response.choices[0].message.content
                
                # Get token usage
                if hasattr(response, 'usage'):
                    token_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

        # Store assistant response in chat history (only for non-streaming responses)
        # For streaming responses, the chat history is stored in the generate functions
        if not request.stream:
            try:
                cursor.execute(
                    """
                    INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, session_id, assistant_message, "assistant")
                )
                conn.commit()
            except Exception as e:
                logger.error(f"Error storing assistant response in chat history: {str(e)}")
                # Continue anyway, as this is not critical

            # Close database connection
            conn.close()
            
            # For agent-based processing, also track execution in background if not already tracked
            if request.use_agent and request.agent_type and not execution_id:
                # Track execution after the fact
                background_tasks.add_task(
                    track_agent_execution,
                    agent_id=request.agent_type if not request.use_tree_reasoning else "tree_reasoning",
                    session_id=session_id,
                    task=user_message,
                    model=request.model,
                    start_time=execution_start_time,
                    tokens=token_usage
                )

            return ChatResponse(
                message=assistant_message,
                model=request.model,
                agent_type=request.agent_type if request.use_agent else None,
                agent_used=request.use_agent,
                tree_reasoning_used=request.use_tree_reasoning if request.use_agent else False,
                context_used=request.include_context,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                execution_id=str(execution_id) if execution_id else None
            )
        # For streaming responses, we've already returned a StreamingResponse

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in chat endpoint: {str(e)}"
        )


async def process_history_entries(entries):
    """Process chat history entries to ensure proper format.
    
    Args:
        entries: List of chat history entries from database
        
    Returns:
        Processed list of entries with correct data types
    """
    processed_entries = []
    for entry in entries:
        # Convert datetime timestamp to ISO format string
        if isinstance(entry["timestamp"], datetime):
            entry["timestamp"] = entry["timestamp"].isoformat()
        processed_entries.append(entry)
    return processed_entries


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    skip_old: bool = False,
    max_age_days: Optional[int] = None,
    req: Request = None
):
    """Get chat history for a session.
    
    Args:
        session_id: ID of the chat session
        limit: Maximum number of messages to return (default: 50)
        skip_old: If true, only return messages within max_age_days
        max_age_days: Maximum age of messages to retrieve
    """
    try:
        # Authentication might be required for accessing chat history
        user_id = await extract_user_id_from_request(req)
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Verify the session belongs to the user if user_id is provided
        if user_id and os.getenv("REQUIRE_AUTH", "false").lower() == "true":
            cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ? AND user_id = ?
                LIMIT 1
                """,
                (session_id, user_id)
            )
            
            result = cursor.fetchone()
            if result and result["count"] == 0:
                conn.close()
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to access this chat session"
                )

        # Base query
        query = """
            SELECT message_text, message_role, timestamp
            FROM chat_history
            WHERE session_id = ?
        """
        params = [session_id]

        # Add user check if user_id is available
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        # Add age filter if requested
        if skip_old or max_age_days:
            if max_age_days:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)"
                params.append(max_age_days)
            elif skip_old:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)"

        # Add ordering and limit
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        history = cursor.fetchall()
        conn.close()

        # Process the timestamps to ensure they are strings
        processed_history = await process_history_entries(history)

        return ChatHistoryResponse(
            session_id=session_id,
            messages=processed_history,
            count=len(processed_history)
        )
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )


@router.get("/sessions", response_model=ChatSessionsResponse)
async def get_chat_sessions(
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    req: Request = None
):
    """Get chat sessions for the current user.
    
    Args:
        user_id: Optional user ID to filter sessions
        limit: Maximum number of sessions to return
        offset: Offset for pagination
    """
    try:
        # Authentication is required for viewing sessions
        user_id = await extract_user_id_from_request(req, user_id)
        
        if not user_id:
            raise HTTPException(
                status_code=401, 
                detail="Authentication required to view chat sessions"
            )

        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if the chat_sessions table exists
        try:
            cursor.execute("SHOW TABLES LIKE 'chat_sessions'")
            table_exists = cursor.fetchone() is not None
        except:
            table_exists = False
            
        if table_exists:
            # Query the dedicated sessions table if it exists
            query = """
                SELECT 
                    session_id, 
                    user_id,
                    title,
                    created_at,
                    last_message_time,
                    preview,
                    message_count
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY last_message_time DESC
                LIMIT ? OFFSET ?
            """
            params = [user_id, limit, offset]
            
            cursor.execute(query, params)
            sessions = cursor.fetchall()
        else:
            # Fallback to aggregating from chat_history (legacy approach)
            query = """
                SELECT session_id, user_id,
                       MAX(timestamp) as last_message_time,
                       COUNT(*) as message_count
                FROM chat_history
                WHERE user_id = ?
                GROUP BY session_id, user_id
                ORDER BY last_message_time DESC
                LIMIT ? OFFSET ?
            """
            params = [user_id, limit, offset]

            cursor.execute(query, params)
            sessions = cursor.fetchall()

            # For each session, fetch the first message to use as a title
            for session in sessions:
                # Convert datetime to string
                if isinstance(session["last_message_time"], datetime):
                    session["last_message_time"] = session["last_message_time"].isoformat()
                    
                title_query = """
                    SELECT message_text
                    FROM chat_history
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT 1
                """
                cursor.execute(title_query, [session["session_id"]])
                first_message = cursor.fetchone()

                if first_message:
                    # Truncate long messages and use as session title
                    title = first_message["message_text"]
                    if len(title) > 100:
                        title = title[:97] + "..."
                    session["title"] = title
                else:
                    session["title"] = "Untitled Session"

        # Convert all datetime objects to ISO format strings
        for session in sessions:
            for key, value in session.items():
                if isinstance(value, datetime):
                    session[key] = value.isoformat()

        conn.close()

        return ChatSessionsResponse(
            sessions=sessions,
            count=len(sessions)
        )
    except Exception as e:
        logger.error(f"Error retrieving chat sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat sessions: {str(e)}"
        )


@router.delete("/history/{session_id}")
async def delete_chat_history(
    session_id: str, 
    before_days: Optional[int] = None,
    req: Request = None
):
    """Delete chat history for a specific session.
    
    Args:
        session_id: ID of the chat session to delete
        before_days: Optional number of days to keep (delete everything older)
    """
    try:
        # Authentication might be required for deleting chat history
        user_id = await extract_user_id_from_request(req)
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor()

        # If auth is required or user_id is available, verify ownership
        if user_id:
            # First check if the session belongs to the user or if it exists at all
            verify_cursor = conn.cursor(dictionary=True)
            verify_cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ? AND user_id = ?
                LIMIT 1
                """,
                (session_id, user_id)
            )
            
            result = verify_cursor.fetchone()
            
            # Also check if the session exists at all (for any user)
            verify_cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ?
                LIMIT 1
                """,
                (session_id,)
            )
            
            session_exists = verify_cursor.fetchone()
            verify_cursor.close()
            
            # If the session exists but doesn't belong to the user, deny access
            if session_exists and session_exists["count"] > 0 and result and result["count"] == 0:
                conn.close()
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to delete this chat session"
                )
            
            # If session doesn't exist for anyone, it might be a new session without messages yet
            # We'll consider this a successful deletion (since there's nothing to delete)
            if session_exists and session_exists["count"] == 0:
                conn.close()
                return {
                    "session_id": session_id,
                    "deleted_count": 0,
                    "message": "Session had no messages to delete"
                }

            # Base delete query with user check
            query = "DELETE FROM chat_history WHERE session_id = ? AND user_id = ?"
            params = [session_id, user_id]
        else:
            # Base delete query without user check
            query = "DELETE FROM chat_history WHERE session_id = ?"
            params = [session_id]

        # Add age filter if requested
        if before_days is not None:
            query += " AND timestamp < DATE_SUB(NOW(), INTERVAL ? DAY)"
            params.append(before_days)

        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} messages from chat history"
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions with proper status codes
        raise
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat history: {str(e)}"
        )


@router.delete("/history")
async def delete_all_chat_history(
    before_days: Optional[int] = None, 
    user_id: Optional[str] = None,
    req: Request = None
):
    """Delete all chat history for the current user.
    
    Args:
        before_days: Optional number of days to keep (delete everything older)
        user_id: Optional user ID to filter sessions
    """
    try:
        # Authentication is required for deleting chat history
        user_id = await extract_user_id_from_request(req, user_id)
        
        if not user_id:
            raise HTTPException(
                status_code=401, 
                detail="Authentication required to delete chat history"
            )
            
        conn = await get_db_connection()
        cursor = conn.cursor()

        # Base delete query
        query = "DELETE FROM chat_history WHERE 1=1"
        params = []

        # Add filters if requested
        if before_days is not None:
            query += " AND timestamp < DATE_SUB(NOW(), INTERVAL ? DAY)"
            params.append(before_days)

        # Only allow users to delete their own messages
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} messages from chat history"
        }
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat history: {str(e)}"
        )


@router.get("/messages", response_model=ChatHistoryResponse)
async def get_chat_messages(
    session_id: str,
    limit: int = 50,
    skip_old: bool = False,
    max_age_days: Optional[int] = None,
    req: Request = None
):
    """Get messages for a chat session (alias for get_chat_history).
    
    Args:
        session_id: ID of the chat session
        limit: Maximum number of messages to return (default: 50)
        skip_old: If true, only return messages within max_age_days 
        max_age_days: Maximum age of messages to retrieve
    """
    try:
        # Authentication might be required for accessing chat history
        user_id = await extract_user_id_from_request(req)
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Base query
        query = """
            SELECT message_text, message_role, timestamp
            FROM chat_history
            WHERE session_id = ?
        """
        params = [session_id]

        # Add user check if user_id is available
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        # Add age filter if requested
        if skip_old or max_age_days:
            if max_age_days:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)"
                params.append(max_age_days)
            elif skip_old:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)"

        # Add ordering and limit
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        history = cursor.fetchall()
        conn.close()

        # Process the timestamps to ensure they are strings
        processed_history = await process_history_entries(history)

        return ChatHistoryResponse(
            session_id=session_id,
            messages=processed_history,
            count=len(processed_history)
        )
    except Exception as e:
        logger.error(f"Error retrieving chat messages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat messages: {str(e)}"
        )


@router.post("/sessions", response_model=dict)
async def create_chat_session(
    user_id: Optional[str] = None,
    req: Request = None
):
    """Create a new chat session.
    
    Args:
        user_id: The user ID to associate with the session (optional)
    """
    try:
        # Authentication is always required for creating new chat sessions
        user_id = await extract_user_id_from_request(req, user_id)
        
        # If no user_id is available, return error
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to create chat sessions"
            )
        
        # Generate a unique session ID
        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Insert session into the chat_sessions table if it exists
        conn = await get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Check if the chat_sessions table exists
            cursor.execute("SHOW TABLES LIKE 'chat_sessions'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Insert into the dedicated sessions table
                cursor.execute(
                    """
                    INSERT INTO chat_sessions 
                    (session_id, user_id, title, created_at, last_message_time)
                    VALUES (?, ?, ?, NOW(), NOW())
                    """,
                    (session_id, user_id, "New Conversation")
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to insert into chat_sessions table: {str(e)}")
        finally:
            conn.close()
        
        # Return the session ID
        return {
            "session_id": session_id,
            "user_id": user_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating chat session: {str(e)}"
        )


@router.get("/sessions/{session_id}/messages", response_model=ChatHistoryResponse)
async def get_session_messages(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    skip_old: bool = False,
    max_age_days: Optional[int] = None,
    req: Request = None
):
    """Get messages for a specific chat session.
    
    Args:
        session_id: ID of the chat session
        limit: Maximum number of messages to return (default: 50)
        offset: Offset for pagination
        skip_old: If true, only return messages within max_age_days
        max_age_days: Maximum age of messages to retrieve
    """
    try:
        # Authentication might be required for accessing chat history
        user_id = await extract_user_id_from_request(req)
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Verify the session belongs to the user if user_id is provided
        if user_id and os.getenv("REQUIRE_AUTH", "false").lower() == "true":
            cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ? AND user_id = ?
                LIMIT 1
                """,
                (session_id, user_id)
            )
            
            result = cursor.fetchone()
            
            # Also check if the session exists at all (for any user)
            cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ?
                LIMIT 1
                """,
                (session_id,)
            )
            
            session_exists = cursor.fetchone()
            
            # If session doesn't exist at all
            if not session_exists or session_exists["count"] == 0:
                conn.close()
                raise HTTPException(
                    status_code=404,
                    detail="Chat session not found. It may have been deleted."
                )
            
            # If the session exists but doesn't belong to the user, deny access
            if result and result["count"] == 0:
                conn.close()
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to access this chat session"
                )

        # Base query
        query = """
            SELECT message_text, message_role, timestamp
            FROM chat_history
            WHERE session_id = ?
        """
        params = [session_id]

        # Add user check if user_id is available
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        # Add age filter if requested
        if skip_old or max_age_days:
            if max_age_days:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)"
                params.append(max_age_days)
            elif skip_old:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)"

        # Add ordering and limit
        query += " ORDER BY timestamp ASC"
        
        # Add pagination if offset is provided
        if offset > 0:
            query += " LIMIT ?, ?"
            params.extend([offset, limit])
        else:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        history = cursor.fetchall()
        conn.close()

        # If no history found, check if session actually exists
        if not history:
            # For an existing session with no messages, return empty list
            return ChatHistoryResponse(
                session_id=session_id,
                messages=[],
                count=0
            )

        # Process the timestamps to ensure they are strings
        processed_history = await process_history_entries(history)

        return ChatHistoryResponse(
            session_id=session_id,
            messages=processed_history,
            count=len(processed_history)
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving session messages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving session messages: {str(e)}"
        )


@router.get("/progress/{execution_id}", response_model=AgentProgressResponse)
async def get_agent_progress(execution_id: str):
    """Get progress information for an agent execution."""
    try:
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Query execution info
        cursor.execute(
            """
            SELECT 
                ae.agent_id,
                ap.progress_percent,
                ap.status,
                ap.status_message,
                ap.timestamp
            FROM agent_progress ap
            JOIN agent_executions ae ON ap.execution_id = ae.id
            WHERE ap.execution_id = ?
            ORDER BY ap.timestamp DESC
            LIMIT 1
            """,
            (execution_id,)
        )
        
        progress = cursor.fetchone()
        
        if not progress:
            raise HTTPException(
                status_code=404,
                detail=f"Progress information not found for execution: {execution_id}"
            )
            
        conn.close()
        
        return AgentProgressResponse(
            execution_id=execution_id,
            agent_id=progress["agent_id"],
            progress_percent=progress["progress_percent"],
            status=progress["status"],
            status_message=progress["status_message"],
            timestamp=progress["timestamp"].isoformat() if isinstance(progress["timestamp"], datetime) else progress["timestamp"]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent progress: {str(e)}"
        )


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_chat_session(
    session_id: str,
    before_days: Optional[int] = None,
    req: Request = None
):
    """Delete a specific chat session.
    
    Args:
        session_id: ID of the chat session to delete
        before_days: Optional number of days to keep (delete everything older)
    """
    try:
        # Authentication might be required for deleting chat history
        user_id = await extract_user_id_from_request(req)
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor()

        # If auth is required or user_id is available, verify ownership
        if user_id:
            # First check if the session belongs to the user or if it exists at all
            verify_cursor = conn.cursor(dictionary=True)
            verify_cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ? AND user_id = ?
                LIMIT 1
                """,
                (session_id, user_id)
            )
            
            result = verify_cursor.fetchone()
            
            # Also check if the session exists at all (for any user)
            verify_cursor.execute(
                """
                SELECT COUNT(*) as count 
                FROM chat_history 
                WHERE session_id = ?
                LIMIT 1
                """,
                (session_id,)
            )
            
            session_exists = verify_cursor.fetchone()
            verify_cursor.close()
            
            # If the session exists but doesn't belong to the user, deny access
            if session_exists and session_exists["count"] > 0 and result and result["count"] == 0:
                conn.close()
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to delete this chat session"
                )
            
            # If session doesn't exist for anyone, it might be a new session without messages yet
            # We'll consider this a successful deletion (since there's nothing to delete)
            if session_exists and session_exists["count"] == 0:
                conn.close()
                return {
                    "session_id": session_id,
                    "deleted_count": 0,
                    "message": "Session had no messages to delete"
                }

            # Base delete query with user check
            query = "DELETE FROM chat_history WHERE session_id = ? AND user_id = ?"
            params = [session_id, user_id]
        else:
            # Base delete query without user check
            query = "DELETE FROM chat_history WHERE session_id = ?"
            params = [session_id]

        # Add age filter if requested
        if before_days is not None:
            query += " AND timestamp < DATE_SUB(NOW(), INTERVAL ? DAY)"
            params.append(before_days)

        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} messages from chat history"
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions with proper status codes
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat session: {str(e)}"
        )
