"""
FastAPI router for chat endpoints, chat history management and agent integration.
"""
import logging
import json
import uuid
import time
import asyncio
from typing import List, Dict, Any, Optional, Literal, Union
from fastapi import APIRouter, Depends, HTTPException, Body, Request, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import os
from fastapi.responses import StreamingResponse
import mariadb
from datetime import timedelta
import re
from openai import OpenAI, AsyncOpenAI

# Import autonomous agent components instead of pyndantic agents
from autonomous_agent.integration_components.agent_factory import create_agent, get_agent_types
from autonomous_agent.integration_components.agent_adapter import AutonomousAgentAdapter
from autonomous_agent.integration_components.tree_reasoning_adapter import TreeReasoningAdapter, DecisionTree
from llamaIndex_rag.rag import RAGSystem, NodeWithScore

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
    stream: bool = Field(True, description="Whether to stream the response")
    model: str = Field("gpt-4", description="Model to use for generation")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens in response")
    include_context: bool = Field(True, description="Whether to include RAG context")
    context_query: Optional[str] = Field(None, description="Query to use for retrieving context")
    retrieval_type: Optional[str] = Field("auto", description="Type of retrieval to use: 'hybrid', 'vector', or 'auto' (default)")
    use_agent: bool = Field(True, description="Whether to use an agent for processing")
    agent_type: Optional[str] = Field("rag", description="Type of agent to use if use_agent is True")
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
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used for generating the response")


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
        # Instead of returning None, return a default execution ID for errors
        # This avoids the FastAPI response validation error
        return -1  # Use -1 to indicate an error occurred, but still return an int


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
    """Chat endpoint with RAG integration."""
    execution_id = None
    execution_start = time.time()
    
    try:
        # Extract user ID
        user_id = await extract_user_id_from_request(req, None)
        
        # Create a session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Prepare response
        response = ChatResponse(
            message="",
            model=request.model,
            agent_type=request.agent_type if request.use_agent else None,
            agent_used=request.use_agent,
            tree_reasoning_used=request.use_tree_reasoning,
            context_used=request.include_context,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            sources=[]
        )
        
        # Save user messages to history
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if user_messages:
            background_tasks.add_task(
                save_chat_message, 
                session_id, 
                user_id, 
                [user_messages[-1]]  # Just save the last user message
            )
            
        # Determine query text
        query_text = request.context_query
        if not query_text and user_messages:
            query_text = user_messages[-1].content
        
        # Use improved RAG with context handling for direct questions
        if (not request.use_agent and request.include_context and query_text and 
            not request.use_tree_reasoning):
            # Use the new query_with_context method for better RAG integration
            try:
                # Get RAG system
                rag_system = await get_rag_system()
                
                # Call the improved query_with_context method
                rag_result = rag_system.query_with_context(
                    query=query_text,
                    top_k=5,  # Retrieve 5 documents for context
                    use_hybrid=True  # Use hybrid search for better retrieval
                )
                
                # Extract response and sources
                response_text = rag_result.get("response", "I couldn't find relevant information to answer your query.")
                sources = rag_result.get("sources", [])
                
                # For streaming responses
                if request.stream:
                    async def generate():
                        # Yield headers
                        yield f"data: {json.dumps({'content': '', 'sources': [], 'done': False})}\n\n"
                        
                        # Simulate streaming by sending response in chunks
                        chunk_size = 10  # Characters per chunk
                        for i in range(0, len(response_text), chunk_size):
                            chunk = response_text[i:i+chunk_size]
                            yield f"data: {json.dumps({'content': chunk, 'sources': [], 'done': False})}\n\n"
                            await asyncio.sleep(0.03)  # Simulate typing speed
                            
                        # Send final message with sources
                        formatted_sources = []
                        for src in sources:
                            formatted_source = {
                                "text": src.get("text", ""),
                                "document_id": src.get("document_id", ""),
                                "document_name": src.get("document_name", "Unknown"),
                                "score": src.get("score", 0.0),
                                "section": src.get("section", "")
                            }
                            formatted_sources.append(formatted_source)
                            
                        yield f"data: {json.dumps({'content': '', 'sources': formatted_sources, 'done': True})}\n\n"
                    
                    # Save the assistant response to history
                    background_tasks.add_task(
                        save_chat_message,
                        session_id,
                        user_id,
                        [ChatMessage(role="assistant", content=response_text)]
                    )
                    
                    # Track the use of RAG system
                    execution_id = await track_agent_execution(
                        agent_id="improved_rag",
                        session_id=session_id,
                        task=query_text,
                        model="gpt-4-turbo",  # Used in query_with_context
                        start_time=execution_start
                    )
                    
                    # Make sure execution_id is a string
                    execution_id = str(execution_id) if execution_id else None
                    
                    # Return streaming response
                    return StreamingResponse(
                        generate(),
                        media_type="text/event-stream"
                    )
                else:
                    # Non-streaming response with the improved RAG context
                    # Track the use of RAG system
                    execution_id = await track_agent_execution(
                        agent_id="improved_rag",
                        session_id=session_id,
                        task=query_text,
                        model="gpt-4-turbo",  # Used in query_with_context
                        start_time=execution_start
                    )
                    
                    # Make sure execution_id is a string
                    execution_id = str(execution_id) if execution_id else None
                    
                    # Save the assistant response to history
                    background_tasks.add_task(
                        save_chat_message,
                        session_id,
                        user_id,
                        [ChatMessage(role="assistant", content=response_text)]
                    )
                    
                    # Format the sources
                    formatted_sources = []
                    for src in sources:
                        formatted_source = {
                            "text": src.get("text", ""),
                            "document_id": src.get("document_id", ""),
                            "document_name": src.get("document_name", "Unknown"),
                            "score": src.get("score", 0.0),
                            "section": src.get("section", "")
                        }
                        formatted_sources.append(formatted_source)
                    
                    # Return the response
                    return ChatResponse(
                        message=response_text,
                        model="gpt-4-turbo",
                        agent_type=None,
                        agent_used=False,
                        tree_reasoning_used=False,
                        context_used=True,
                        session_id=session_id,
                        timestamp=datetime.now().isoformat(),
                        execution_id=execution_id,
                        sources=formatted_sources
                    )
                
            except Exception as e:
                logger.error(f"Error using improved RAG: {str(e)}", exc_info=True)
                # Fall back to standard behavior on error
        else:
            # Handle tree reasoning if requested
            if request.use_tree_reasoning:
                try:
                    # Get RAG system and OpenAI client from main application
                    from main import rag_system, openai_client
                    
                    # Create the tree reasoning agent
                    tree_data = None
                    
                    if request.tree_template:
                        # Get predefined tree template
                        from autonomous_agent.integration_components.agent_factory import get_agent_factory
                        factory = get_agent_factory()
                        
                        # Get the default tree for a specific agent type
                        agent_type = request.agent_type or "rag"  # Default to RAG agent if none specified
                        tree_data = factory.get_default_tree(agent_type)
                        
                    elif request.custom_tree:
                        # Use custom tree provided in the request
                        tree_data = request.custom_tree
                    else:
                        # No tree specified, use a default general-purpose tree
                        from autonomous_agent.integration_components.agent_factory import get_agent_factory
                        factory = get_agent_factory()
                        tree_data = factory.get_default_tree("rag")  # Use RAG agent's default tree
                    
                    # Create tree reasoning adapter
                    tree_agent = TreeReasoningAdapter(
                        tree=tree_data,
                        graph_interface=None,  # Will be set up by the adapter
                        embedding_service=rag_system,
                        llm_client=openai_client,
                        config={
                            "max_reformulation_attempts": 1,
                            "timeout_seconds": 60
                        }
                    )
                    
                    # Track the start time for execution timing
                    start_time = time.time()
                    execution_id = str(uuid.uuid4())
                    
                    # Process with tree reasoning agent (non-streaming only for now)
                    result = await tree_agent.process(
                        user_input=query_text,
                        session_id=session_id,
                        user_id=user_id,
                        model=request.model,
                        include_context=request.include_context,
                        context_query=request.context_query
                    )
                    
                    # Track execution in background
                    execution_time = time.time() - start_time
                    background_tasks.add_task(
                        track_agent_execution,
                        agent_id="tree_reasoning",
                        session_id=session_id,
                        task=query_text,
                        model=request.model,
                        start_time=start_time,
                        tokens={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        error=False
                    )
                    
                    # Format the response
                    response_text = result.get("response", "")
                    sources = result.get("source_documents", [])
                    
                    # Save the assistant response to history
                    background_tasks.add_task(
                        save_chat_message,
                        session_id,
                        user_id,
                        [ChatMessage(role="assistant", content=response_text)]
                    )
                    
                    # Return the response
                    response.message = response_text
                    response.sources = sources
                    return response
                    
                except Exception as e:
                    logger.error(f"Error processing with tree reasoning: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing with tree reasoning: {str(e)}"
                    )
            else:
                # Standard chat processing (no agent or tree reasoning)
                try:
                    # Get OpenAI client and RAG system from main application
                    from main import openai_client, rag_system
                    
                    # Track the start time for execution timing
                    start_time = time.time()
                    execution_id = str(uuid.uuid4())
                    
                    # Process streaming or non-streaming
                    if request.stream:
                        async def generate():
                            try:
                                # Prepare message history
                                message_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
                                
                                # Add context from RAG if requested
                                context = ""
                                sources = []
                                context_used = False
                                
                                if request.include_context and rag_system:
                                    context_query = request.context_query or query_text
                                    retrieval_type = request.retrieval_type or "auto"
                                    
                                    # Get context from RAG system
                                    if retrieval_type == "auto":
                                        context_results = rag_system.retrieve(
                                            query=context_query, 
                                            top_k=3
                                        )
                                    elif retrieval_type == "hybrid":
                                        context_results = rag_system.retrieve(
                                            query=context_query,
                                            use_hybrid=True,
                                            top_k=3
                                        )
                                    else:  # vector
                                        context_results = rag_system.retrieve(
                                            query=context_query,
                                            use_hybrid=False,
                                            top_k=3
                                        )

                                    if context_results:
                                        context_used = True
                                        context = "Here's some relevant context that might help answering the query:\n\n"
                                        
                                        for i, result in enumerate(context_results):
                                            if "text" in result:
                                                context += f"[{i+1}] {result['text']}\n\n"
                                            
                                            # Add to sources if metadata is available
                                            if "metadata" in result and result["metadata"]:
                                                source_data = result["metadata"].copy()
                                                # Ensure "text" is not duplicated in sources
                                                if "text" in source_data:
                                                    del source_data["text"]
                                                
                                                # Add snippet to source
                                                source_data["snippet"] = result.get("text", "")[:200] + "..."
                                                sources.append(source_data)
                                        
                                        # Add context as a system message
                                        message_history.insert(0, {
                                            "role": "system", 
                                            "content": context
                                        })
                                
                                # Create streaming completion
                                stream = openai_client.chat.completions.create(
                                    model=request.model,
                                    messages=message_history,
                                    temperature=request.temperature,
                                    max_tokens=request.max_tokens,
                                    stream=True
                                )
                                
                                # Stream the response
                                response_text = ""
                                for chunk in stream:
                                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                        content = chunk.choices[0].delta.content
                                        response_text += content
                                        
                                        # Format the chunk for SSE
                                        response = {
                                            "text": content,
                                            "done": False,
                                            "session_id": session_id,
                                            "execution_id": execution_id
                                        }
                                        
                                        yield f"data: {json.dumps(response)}\n\n"
                                        
                                        # Add a small delay to simulate natural typing
                                        await asyncio.sleep(0.01)
                                
                                # Final response with sources
                                final_response = {
                                    "text": "",
                                    "done": True,
                                    "session_id": session_id,
                                    "execution_id": execution_id
                                }
                                
                                if sources:
                                    final_response["sources"] = sources
                                    
                                yield f"data: {json.dumps(final_response)}\n\n"
                                
                                # Track usage
                                background_tasks.add_task(
                                    save_chat_message,
                                    session_id=session_id,
                                    user_id=user_id,
                                    messages=[ChatMessage(role="assistant", content=response_text)]
                                )
                                
                            except Exception as e:
                                logger.error(f"Error in standard chat streaming: {str(e)}", exc_info=True)
                                error_response = {
                                    "text": f"Error: {str(e)}",
                                    "done": True,
                                    "error": True,
                                    "session_id": session_id,
                                    "execution_id": execution_id
                                }
                                yield f"data: {json.dumps(error_response)}\n\n"
                    
                    # Return streaming response
                    return StreamingResponse(
                        generate(),
                        media_type="text/event-stream"
                    )
                except Exception as e:
                    logger.error(f"Error in standard chat processing: {str(e)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing chat: {str(e)}"
                    )
        return response
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

async def save_chat_message(session_id: str, user_id: Optional[str], messages: List[ChatMessage]):
    """Save chat messages to the database."""
    try:
        conn = await get_db_connection()
        cursor = conn.cursor()
        
        # Use a default user_id if none is provided
        if user_id is None:
            user_id = f"guest_{str(uuid.uuid4())[:8]}"
            logger.warning(f"No user_id provided for chat message in session {session_id}, using generated ID: {user_id}")
        
        for msg in messages:
            cursor.execute(
                """
                INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, session_id, msg.content, msg.role)
            )
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving chat message: {str(e)}")


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
        user_id: Optional user ID to filter sessions (primarily for admin use, normally inferred from token)
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
