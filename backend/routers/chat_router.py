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
import asyncio

# Import OpenAI clients for both sync and async operations
from openai import OpenAI, AsyncOpenAI

# Import agent framework integrations
from agent_framework.integrations.chat_integration import get_chat_integration

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
    use_agent: bool = Field(True, description="Whether to use an agent for processing")
    use_tree_reasoning: bool = Field(False, description="Whether to use tree-based reasoning")
    tree_template: Optional[str] = Field(None, description="ID of the decision tree template to use")
    custom_tree: Optional[Dict[str, Any]] = Field(None, description="Custom decision tree for reasoning")
    session_id: Optional[str] = Field(None, description="Session ID for chat history")


class SourceInfo(BaseModel):
    """Information about a source used in RAG retrieval."""
    doc_id: Optional[str] = Field(None, description="Document ID")
    page_number: Optional[int] = Field(1, description="Page number in document")
    score: Optional[float] = Field(None, description="Relevance score")
    retrieval_method: Optional[str] = Field("HyPE", description="Method used for retrieval")
    title: Optional[str] = Field(None, description="Document title if available")
    content: Optional[str] = Field(None, description="Actual text content from the document chunk")


class ChatResponse(BaseModel):
    """Response for a chat completion."""
    message: str = Field(..., description="Assistant response message")
    model: str = Field(..., description="Model used for generation")
    agent_used: bool = Field(False, description="Whether an agent was used")
    tree_reasoning_used: bool = Field(False, description="Whether tree reasoning was used")
    context_used: bool = Field(False, description="Whether context was used")
    session_id: str = Field(..., description="Session ID for chat history")
    timestamp: str = Field(..., description="Timestamp of the response")
    execution_id: Optional[str] = Field(None, description="ID of the execution for tracking progress")
    sources: Optional[List[SourceInfo]] = Field(None, description="Sources used in the response")
    context_quality: Optional[str] = Field(None, description="Quality assessment of the context")
    hallucination_risk: Optional[float] = Field(None, description="Risk of hallucination in the response")
    internal_thoughts: Optional[str] = Field(None, description="Internal thoughts and reasoning process")
    tools_used: Optional[List[str]] = Field(None, description="List of tools used by the agent")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata from agent processing")


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


# Dependency to get the query engine
async def get_rag_query_engine():
    """Get the RAG query engine from main application."""
    from main import rag_query_engine
    return rag_query_engine


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
    """
    Process a chat request and generate a response.
    
    Optionally include context from the RAG system and/or use an agent for processing.
    """
    # Check if OpenAI API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured"
        )
    
    # Extract user ID from request if available
    user_id = await extract_user_id_from_request(req)
    
    # Generate session ID if not provided
    session_id = request.session_id or f"session_{uuid.uuid4()}"
    
    # Get last user message for context retrieval if needed
    last_user_message = next(
        (msg.content for msg in reversed(request.messages) if msg.role == "user"),
        None
    )
    
    # Context retrieval
    context = None
    context_used = False
    context_result = None
    
    if request.include_context and last_user_message:
        # Use context_query if provided, otherwise use last user message
        query = request.context_query or last_user_message
        
        try:
            # Use LlamaIndex RAG system for context retrieval
            rag_query_engine = await get_rag_query_engine()
            
            # Get context for the query
            context_result = await retrieve_context_for_query(
                query=query,
                top_k=5,
                search_filter=None,
                rag_query_engine=rag_query_engine
            )
            
            if context_result.get("status") == "success" and context_result.get("context"):
                # Extract context and sources
                context_texts = context_result.get("context", [])
                sources = context_result.get("sources", [])
                
                # Format context for prompt with source attributions
                if context_texts:
                    formatted_contexts = []
                    for i, text in enumerate(context_texts):
                        source_info = ""
                        if i < len(sources):
                            source = sources[i]
                            doc_id = source.get("doc_id", "unknown")
                            page = source.get("page_number")
                            page_info = f", page {page}" if page else ""
                            title = source.get("title", "")
                            title_info = f" - {title}" if title else ""
                            source_info = f" [Source {i+1}: Document {doc_id}{page_info}{title_info}]"
                        
                        formatted_contexts.append(f"Context {i+1}:{source_info}\n{text}")
                    
                    # Add a system instruction to use source attributions
                    source_instruction = """
You will be provided with context information from various sources. When answering:
1. Include source attributions like [Source 1], [Source 2], etc. when referencing specific information
2. Address all aspects of the query using the given context
3. If the context is insufficient to fully answer the query, acknowledge the limitations
4. Prioritize information from sources with higher relevance (they are provided in order of relevance)
"""
                    
                    # Add the system instruction at the beginning of messages
                    messages_with_context = list(request.messages)  # Create a copy of the messages
                    
                    if messages_with_context and messages_with_context[0].role == "system":
                        # Append to existing system message
                        messages_with_context[0].content = source_instruction + "\n\n" + messages_with_context[0].content
                    else:
                        # Add a new system message
                        messages_with_context.insert(0, ChatMessage(role="system", content=source_instruction))
                    
                    context = "\n\n".join(formatted_contexts)
                    context_used = True
                    
                    # Add system message with the formatted context
                    context_message = ChatMessage(
                        role="system",
                        content=f"Please use the following context information to answer the user's question:\n\n{context}"
                    )
                    messages_with_context.insert(1 if messages_with_context[0].role == "system" else 0, context_message)
                    
                    # Use the updated messages with context
                    request.messages = messages_with_context
                    
                    logger.info(f"Retrieved {len(context_texts)} context chunks for query: {query}")
                else:
                    logger.info(f"No context found for query: {query}")
            else:
                logger.warning(f"Context retrieval failed: {context_result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            
    # Rest of the chat function continues as before
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
        execution_id = None
        context_used = False
        sources = None
        internal_thoughts = None
        assistant_message = ""  # Initialize to ensure it always has a value

        # If agent-based processing is requested
        if request.use_agent and not request.stream:
            # Agent processing only works with non-streaming requests for now
            try:
                # Use the chat integration for agent processing
                chat_integration = get_chat_integration()
                
                # Prepare request data for the chat integration
                request_data = {
                    "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
                    "model": request.model,
                    "session_id": session_id,
                    "include_context": request.include_context,
                    "context_query": request.context_query,
                    "response_format": "text"
                }
                
                # Process with the agent framework
                agent_response = await chat_integration.process_chat_request(
                    request_data=request_data,
                    use_agent=True
                )
                
                if agent_response.get("error"):
                    raise Exception(agent_response.get("message", "Agent processing failed"))
                
                assistant_message = agent_response.get("message", "")
                context_used = agent_response.get("context_used", False)
                sources = agent_response.get("sources")
                
                # Extract agent metadata
                agent_tools_used = agent_response.get("tools_used")
                agent_metadata = agent_response.get("metadata", {})
                
                # If we have sources from the agent, format them for the response
                if sources:
                    # Create a compatible context_result structure for the return statement
                    context_result = {
                        "sources": sources,
                        "context_quality": "agent_processed",
                        "hallucination_risk": None
                    }
                
                # Create agent execution tracking
                cursor.execute(
                    """
                    INSERT INTO agent_executions (
                        agent_id, session_id, task, model, error
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (request.model, session_id, user_message, request.model, False)
                )
                conn.commit()
                execution_id = cursor.lastrowid
                
                # Initialize and complete progress
                cursor.execute(
                    """
                    INSERT INTO agent_progress (
                        execution_id, progress_percent, status, status_message
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (execution_id, 100.0, "completed", "Agent processing completed")
                )
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error in agent processing: {str(e)}")
                
                # Update execution record with error if exists
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
                    
                # Set a fallback error message and fall through to standard RAG processing
                logger.info("Falling back to standard RAG processing due to agent error")
                assistant_message = ""  # Reset to trigger standard processing
                
        if not assistant_message:
            # Standard RAG-based processing (used when no agent or agent failed or streaming enabled)
            # Import OpenAI API key from main application
            from main import OPENAI_API_KEY

            # Create OpenAI client - use AsyncOpenAI for streaming to prevent blocking
            if request.stream:
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            else:
                client = OpenAI(api_key=OPENAI_API_KEY)

            # Convert messages for OpenAI API
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Detect language from user message and create appropriate system prompt
            detected_language = detect_language(user_message)
            logger.info(f"Detected language: {detected_language} for message: {user_message[:50]}...")
            
            # Add system message with instruction to include internal thoughts
            openai_messages.insert(0, {
                "role": "system", 
                "content": get_system_prompt_for_language(detected_language, context_used)
            })

            # Get chat completion
            if request.stream:
                # Handle streaming response - IMPROVED VERSION BASED ON BEST PRACTICES
                async def generate():
                    try:
                        # Start streaming response with request tracking
                        request_start = time.time()
                        yield json.dumps({
                            "type": "start",
                            "timestamp": datetime.now().isoformat(),
                            "request_id": f"stream_{int(request_start)}_{uuid.uuid4().hex[:8]}"
                        }) + "\n"
                        
                        # Simplified processing steps
                        processing_steps = [
                            "Analyzing your query...",
                            "Searching knowledge base...", 
                            "Retrieving relevant context...",
                            "Generating response..."
                        ]
                        
                        # Send processing updates quickly
                        for i, step_message in enumerate(processing_steps):
                            yield json.dumps({
                                "type": "processing",
                                "state": step_message,
                                "step": f"step_{i+1}",
                                "step_number": i + 1,
                                "total_steps": len(processing_steps),
                                "timestamp": datetime.now().isoformat()
                            }) + "\n"
                            
                            # Only add minimal delay for context retrieval step
                            if i == 2:  # Context retrieval step
                                await asyncio.sleep(0.1)
                        
                        # Context insights if available
                        if context_used and context_result:
                            source_count = len(context_result.get("sources", []))
                            yield json.dumps({
                                "type": "processing",
                                "state": f"Found {source_count} relevant sources",
                                "step": "context_ready",
                                "context_metadata": {
                                    "source_count": source_count,
                                    "context_quality": context_result.get("context_quality")
                                },
                                "timestamp": datetime.now().isoformat()
                            }) + "\n"
                        
                        # IMPROVED STREAMING WITH REAL-TIME DEDUPLICATION
                        try:
                            logger.info("Starting improved OpenAI streaming with deduplication")
                            
                            # Create the stream
                            stream = await client.chat.completions.create(
                                model=request.model,
                                messages=openai_messages,
                                temperature=request.temperature,
                                max_tokens=request.max_tokens,
                                stream=True
                            )
                            
                            # Advanced token collection with real-time deduplication
                            collected_tokens = []
                            full_response = ""  # Track full accumulated response
                            internal_thought_content = []
                            in_internal_thoughts = False
                            last_sent_content = ""  # Track what we've already sent to prevent duplication
                            
                            # Process stream chunks with improved logic
                            async for chunk in stream:
                                if (hasattr(chunk, 'choices') and 
                                    len(chunk.choices) > 0 and 
                                    hasattr(chunk.choices[0], 'delta') and 
                                    hasattr(chunk.choices[0].delta, 'content') and 
                                    chunk.choices[0].delta.content is not None):
                                    
                                    content = chunk.choices[0].delta.content
                                    
                                    # Skip empty content
                                    if not content:
                                        continue
                                    
                                    # Accumulate all content for full tracking
                                    full_response += content
                                    collected_tokens.append(content)
                                    
                                    # Real-time internal thoughts detection
                                    if "<internal_thoughts>" in content:
                                        in_internal_thoughts = True
                                        # Split content at the tag
                                        parts = content.split("<internal_thoughts>", 1)
                                        if parts[0]:
                                            # Send the part before the tag
                                            new_content = parts[0]
                                            # Check for duplication against last sent content
                                            if not last_sent_content or not new_content.startswith(last_sent_content[-min(len(last_sent_content), 50):]):
                                                yield json.dumps({
                                                    "type": "token",
                                                    "content": new_content
                                                }) + "\n"
                                                last_sent_content += new_content
                                        
                                        # Start collecting internal thoughts
                                        if len(parts) > 1:
                                            internal_thought_content.append(parts[1])
                                        continue
                                    
                                    elif "</internal_thoughts>" in content and in_internal_thoughts:
                                        # End of internal thoughts
                                        parts = content.split("</internal_thoughts>", 1)
                                        if parts[0]:
                                            internal_thought_content.append(parts[0])
                                        
                                        # Send internal thoughts as processing update
                                        if internal_thought_content:
                                            thoughts_text = "".join(internal_thought_content)
                                            yield json.dumps({
                                                "type": "processing",
                                                "state": "Processing internal reasoning",
                                                "step": "reasoning",
                                                "internal_thoughts": thoughts_text,
                                                "timestamp": datetime.now().isoformat()
                                            }) + "\n"
                                        
                                        in_internal_thoughts = False
                                        
                                        # Send content after the closing tag
                                        if len(parts) > 1 and parts[1]:
                                            new_content = parts[1]
                                            # Check for duplication
                                            if not last_sent_content or not new_content.startswith(last_sent_content[-min(len(last_sent_content), 50):]):
                                                yield json.dumps({
                                                    "type": "token",
                                                    "content": new_content
                                                }) + "\n"
                                                last_sent_content += new_content
                                        continue
                                    
                                    elif in_internal_thoughts:
                                        # Accumulate internal thoughts content
                                        internal_thought_content.append(content)
                                        continue
                                    
                                    else:
                                        # Normal content - apply real-time deduplication
                                        # Check if this content would create a duplication
                                        if last_sent_content:
                                            # Look for duplications where the new content starts with the end of the last sent content
                                            overlap_check_length = min(len(last_sent_content), 100)
                                            recent_content = last_sent_content[-overlap_check_length:] if overlap_check_length > 0 else ""
                                            
                                            # If the new content is a repetition of recent content, skip it
                                            if recent_content and content in recent_content:
                                                logger.debug(f"Skipping duplicate content: '{content}'")
                                                continue
                                                
                                            # Check for partial overlaps at word boundaries
                                            words_recent = recent_content.split()
                                            words_new = content.split()
                                            
                                            # If new content starts with the same words as recent content ends, it might be a duplication
                                            if (len(words_recent) > 0 and len(words_new) > 0 and 
                                                len(words_recent) >= 2 and len(words_new) >= 2):
                                                
                                                # Check if last 2-3 words of recent content match start of new content
                                                for check_len in [3, 2]:
                                                    if (len(words_recent) >= check_len and len(words_new) >= check_len and
                                                        words_recent[-check_len:] == words_new[:check_len]):
                                                        logger.debug(f"Skipping overlapping duplicate: '{content}'")
                                                        continue
                                        
                                        # Content passes deduplication checks - send it
                                        yield json.dumps({
                                            "type": "token",
                                            "content": content
                                        }) + "\n"
                                        last_sent_content += content
                            
                            logger.info(f"OpenAI streaming completed successfully")
                            
                        except Exception as e:
                            logger.error(f"Error in OpenAI streaming: {str(e)}")
                            yield json.dumps({
                                "type": "error",
                                "message": f"Error during response generation: {str(e)}",
                                "error_code": "GENERATION_ERROR"
                            }) + "\n"
                            return
                        
                        # Process the final accumulated response
                        final_response = full_response
                        final_internal_thoughts = None
                        
                        # Extract internal thoughts from full response if any were missed
                        if internal_thought_content:
                            final_internal_thoughts = "".join(internal_thought_content)
                        else:
                            # Fallback: extract from full response
                            internal_thoughts_match = re.search(r'<internal_thoughts>(.*?)</internal_thoughts>', final_response, re.DOTALL)
                            if internal_thoughts_match:
                                final_internal_thoughts = internal_thoughts_match.group(1).strip()
                        
                        # Clean the final response (remove internal thoughts tags)
                        cleaned_response = re.sub(r'<internal_thoughts>.*?</internal_thoughts>', '', final_response, flags=re.DOTALL).strip()
                        cleaned_response = re.sub(r'</?internal[^>]*thoughts[^>]*>', '', cleaned_response).strip()
                        
                        # Advanced deduplication on the final response
                        # This handles any remaining issues that real-time deduplication missed
                        words = cleaned_response.split()
                        deduplicated_words = []
                        i = 0
                        
                        while i < len(words):
                            current_word = words[i]
                            
                            # Look ahead for potential duplications
                            skip_count = 0
                            
                            # Check for immediate word repetition
                            if i + 1 < len(words) and words[i] == words[i + 1]:
                                skip_count = 1
                            
                            # Check for phrase repetitions (2-3 words)
                            elif i + 3 < len(words):
                                # Check 2-word phrase repetition
                                if (words[i:i+2] == words[i+2:i+4]):
                                    skip_count = 2
                                # Check 3-word phrase repetition
                                elif i + 5 < len(words) and words[i:i+3] == words[i+3:i+6]:
                                    skip_count = 3
                            
                            deduplicated_words.append(current_word)
                            i += 1 + skip_count
                        
                        cleaned_response = ' '.join(deduplicated_words).strip()
                        
                        # Additional cleanup for punctuation duplications
                        cleaned_response = re.sub(r'([.!?])\s*\1+', r'\1', cleaned_response)  # Remove repeated punctuation
                        cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()  # Clean excessive whitespace
                        
                        # Store the response in chat history
                        try:
                            cursor.execute(
                                """
                                INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                                VALUES (?, ?, ?, ?)
                                """,
                                (user_id, session_id, cleaned_response, "assistant")
                            )
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Error storing assistant message in chat history: {str(e)}")
                        
                        # Extract sources from context result if available
                        sources = []
                        if context_result and context_result.get("status") == "success":
                            sources = context_result.get("sources", [])
                        
                        # Send completion event
                        yield json.dumps({
                            "type": "end",
                            "message": cleaned_response,
                            "model": request.model,
                            "context_used": context_used,
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                            "sources": sources,
                            "context_quality": context_result.get("context_quality") if context_result else None,
                            "hallucination_risk": context_result.get("hallucination_risk") if context_result else None,
                            "internal_thoughts": final_internal_thoughts
                        }) + "\n"
                        
                        logger.info("Streaming response completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Error in streaming generator: {str(e)}")
                        yield json.dumps({
                            "type": "error",
                            "message": f"Generator error: {str(e)}"
                        }) + "\n"

                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # Non-streaming response
                response = client.chat.completions.create(
                    model=request.model,
                    messages=openai_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                # Extract assistant message
                assistant_message = response.choices[0].message.content
                
                # Extract internal thoughts if present
                internal_thoughts = None
                internal_thoughts_match = re.search(r'<internal_thoughts>(.*?)</internal_thoughts>', assistant_message, re.DOTALL)
                if internal_thoughts_match:
                    internal_thoughts = internal_thoughts_match.group(1).strip()
                    # Remove internal thoughts from the message
                    assistant_message = re.sub(r'<internal_thoughts>.*?</internal_thoughts>', '', assistant_message, flags=re.DOTALL).strip()
                    # Log successful extraction
                    logger.info(f"Successfully extracted internal thoughts: {internal_thoughts[:50]}...")
                else:
                    logger.info("No internal thoughts found in response")
                
                # Get token usage
                if hasattr(response, 'usage'):
                    token_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

        # Store assistant response in chat history (only for non-streaming responses)
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
            
            # Ensure assistant_message is not empty
            if not assistant_message or assistant_message.strip() == "":
                assistant_message = "I apologize, but I wasn't able to generate a response to your query. Please try rephrasing your question or try again."
            
            # Return final chat response
            return ChatResponse(
                message=assistant_message,
                model=request.model,
                agent_used=request.use_agent,
                tree_reasoning_used=request.use_tree_reasoning,
                context_used=context_used,
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                execution_id=str(execution_id) if execution_id else None,
                sources=context_result.get("sources") if context_result else None,
                context_quality=context_result.get("context_quality") if context_result else None,
                hallucination_risk=context_result.get("hallucination_risk") if context_result else None,
                internal_thoughts=internal_thoughts,
                tools_used=agent_tools_used if 'agent_tools_used' in locals() else None,
                metadata=agent_metadata if 'agent_metadata' in locals() else None
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
        
        # Log the deletion request with detailed information
        logger.info(f"Received request to delete chat history. session_id={session_id}, user_id={user_id}, before_days={before_days}")
        
        # Get database connection
        conn = await get_db_connection()
        cursor = conn.cursor()
        logger.info(f"Database connection established for history deletion. session_id={session_id}")

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
            logger.info(f"Ownership verification: found {result['count']} messages for user. session_id={session_id}, user_id={user_id}")
            
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
            logger.info(f"Session verification: found {session_exists['count']} total messages. session_id={session_id}")
            
            # If the session exists but doesn't belong to the user, deny access
            if session_exists and session_exists["count"] > 0 and result and result["count"] == 0:
                logger.warning(f"Unauthorized attempt to delete chat history. session_id={session_id}, requesting_user={user_id}")
                conn.close()
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to delete this chat session"
                )
            
            # If session doesn't exist for anyone, it might be a new session without messages yet
            # We'll consider this a successful deletion (since there's nothing to delete)
            if session_exists and session_exists["count"] == 0:
                logger.info(f"No messages found for session. session_id={session_id}, user_id={user_id}")
                conn.close()
                return {
                    "session_id": session_id,
                    "deleted_count": 0,
                    "message": "Session had no messages to delete"
                }

            logger.info(f"Authorized chat history deletion. session_id={session_id}, user_id={user_id}")
            
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
            logger.info(f"Deleting messages older than {before_days} days for session_id={session_id}")

        logger.info(f"Executing history deletion query. session_id={session_id}")
        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        logger.info(f"Deleted {deleted_count} messages from chat history. session_id={session_id}")
        
        conn.commit()
        logger.info(f"Database transaction committed. session_id={session_id}")
        
        conn.close()
        logger.info(f"Database connection closed after history deletion. session_id={session_id}")

        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} messages from chat history"
        }
    except HTTPException as e:
        # Re-raise HTTP exceptions with proper status codes
        logger.warning(f"HTTP exception while deleting chat history: {str(e)}, session_id={session_id}")
        raise
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}, session_id={session_id}, user_id={user_id if 'user_id' in locals() else None}")
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
    """Delete a chat session and its messages."""
    # Extract user ID from request
    user_id = await extract_user_id_from_request(req)
    
    # Log the deletion request with more details
    logger.info(f"Received request to delete chat session. session_id={session_id}, user_id={user_id}, before_days={before_days}")
    
    # Initialize database connection
    try:
        conn = await get_db_connection()
        cursor = conn.cursor()
        logger.info(f"Database connection established for session deletion. session_id={session_id}")
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to database: {str(e)}"
        )
        
    # Check if session exists
    try:
        cursor.execute(
            """
            SELECT session_id, user_id FROM chat_sessions 
            WHERE session_id = ?
            """,
            (session_id,)
        )
        session = cursor.fetchone()
        
        # If session not found in chat_sessions, check chat_history as fallback
        if not session:
            logger.info(f"Session not found in chat_sessions, checking chat_history. session_id={session_id}")
            cursor.execute(
                """
                SELECT session_id, user_id FROM chat_history 
                WHERE session_id = ?
                GROUP BY session_id, user_id
                """,
                (session_id,)
            )
            session = cursor.fetchone()
            
        if not session:
            logger.warning(f"Session not found for deletion. session_id={session_id}, user_id={user_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Session with ID {session_id} not found"
            )
            
        # Verify ownership
        if session[1] != user_id:
            logger.warning(f"Unauthorized attempt to delete session. session_id={session_id}, requesting_user={user_id}, owner_user={session[1]}")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to delete this session"
            )
            
        logger.info(f"Authorized session deletion. session_id={session_id}, user_id={user_id}")
            
        # Construct date filter if before_days is provided
        date_filter = ""
        date_filter_params = ()
        if before_days is not None:
            # Use MariaDB date syntax instead of SQLite
            date_filter = "AND timestamp < DATE_SUB(NOW(), INTERVAL ? DAY)"
            date_filter_params = (before_days,)
            logger.info(f"Deleting messages older than {before_days} days for session_id={session_id}")
            
        # Delete messages
        logger.info(f"Executing message deletion for session_id={session_id}")
        cursor.execute(
            f"""
            DELETE FROM chat_history 
            WHERE session_id = ? {date_filter}
            """,
            (session_id,) + date_filter_params
        )
        messages_deleted = cursor.rowcount
        logger.info(f"Deleted {messages_deleted} messages from session_id={session_id}")
        
        # If deleting all messages (no date filter or all messages match filter)
        if before_days is None or messages_deleted > 0:
            # Check if any messages remain
            cursor.execute(
                """
                SELECT COUNT(*) FROM chat_history 
                WHERE session_id = ?
                """,
                (session_id,)
            )
            remaining_messages = cursor.fetchone()[0]
            logger.info(f"Remaining messages after deletion: {remaining_messages}, session_id={session_id}")
            
            # If no messages remain, delete the session record
            if remaining_messages == 0:
                logger.info(f"No messages remain, deleting session record. session_id={session_id}")
                try:
                    cursor.execute(
                        """
                        DELETE FROM chat_sessions 
                        WHERE session_id = ?
                        """,
                        (session_id,)
                    )
                    logger.info(f"Session record deleted successfully. session_id={session_id}")
                except Exception as e:
                    # Log but continue if deleting from chat_sessions fails
                    logger.warning(f"Could not delete from chat_sessions: {str(e)}, session_id={session_id}")
                
        conn.commit()
        logger.info(f"Chat session deletion completed successfully. session_id={session_id}, messages_deleted={messages_deleted}")
        
        return {
            "status": "success",
            "message": f"Chat session {session_id} deleted successfully",
            "messages_deleted": messages_deleted
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}, session_id={session_id}, user_id={user_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat session: {str(e)}"
        )
    finally:
        # Close database connection
        if conn:
            conn.close()
            logger.info(f"Database connection closed after session deletion. session_id={session_id}")


@router.post("/rag")
async def chat_with_rag(
    payload: Dict[str, Any] = Body(...),
    req: Request = None,
    background_tasks: BackgroundTasks = None
):
    """
    Process a chat request with RAG (Retrieval-Augmented Generation) enabled by default.
    
    This endpoint is a streamlined version of the regular chat endpoint that always uses RAG.
    Supports both streaming and non-streaming responses based on the request.
    """
    start_time = time.time()
    
    # Check if we have valid messages in the payload
    if not payload.get("messages"):
        raise HTTPException(
            status_code=400,
            detail="Messages array is required"
        )
    
    # Log the incoming messages for debugging
    logger.info(f"Received {len(payload.get('messages', []))} messages in /chat/rag endpoint")
    for i, msg in enumerate(payload.get("messages", [])):
        logger.info(f"Message {i}: role={msg.get('role')}, content_length={len(msg.get('content', ''))}")
    
    # Convert the dict-based messages to ChatMessage objects
    try:
        messages = [
            ChatMessage(
                role=msg.get("role"),
                content=msg.get("content")
            ) for msg in payload.get("messages")
        ]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid message format: {str(e)}"
        )
    
    # Create a ChatRequest object with RAG enabled and streaming by default
    request = ChatRequest(
        messages=messages,
        stream=payload.get("stream", True),  # Enable streaming by default
        model=payload.get("model", "gpt-4"),
        temperature=payload.get("temperature", 0.7),
        max_tokens=payload.get("max_tokens", 2048),
        include_context=True,  # Always include context for RAG
        context_query=payload.get("context_query"),
        retrieval_type=payload.get("retrieval_type", "auto"),
        use_agent=payload.get("use_agent", False),  # Disable agent by default to prevent hanging
        use_tree_reasoning=payload.get("use_tree_reasoning", False),
        tree_template=payload.get("tree_template"),
        custom_tree=payload.get("custom_tree"),
        session_id=payload.get("session_id")
    )
    
    # Enhanced request tracking
    request_id = f"rag_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Processing RAG request {request_id} - Agent enabled: {request.use_agent}, Streaming: {request.stream}")
    
    # TEMPORARY TESTING WORKAROUND:
    # Create a modified request object with the auth check bypassed
    if req is None or not await extract_user_id_from_request(req):
        # Create a modified Request object with a test user ID for testing
        from fastapi import Request
        from starlette.datastructures import Headers
        
        # Create headers with a test user ID
        test_user_id = "test_user_123"
        headers = {"X-User-ID": test_user_id}
        
        # Create a Request object with these headers
        # We need to preserve the original request if it exists
        if req is not None:
            # Copy the original request but add our test user ID
            # This is a simplified approach - in a real scenario, you'd properly
            # duplicate all needed properties
            mock_req = Request(scope=req.scope)
            mock_req._headers = Headers({**dict(req.headers), **headers})
            req = mock_req
        else:
            # Create a minimal request object with headers
            from starlette.datastructures import Headers, MutableHeaders
            from starlette.types import Scope
            
            scope = {
                "type": "http", 
                "headers": [(k.encode("latin1"), v.encode("latin1")) for k, v in headers.items()]
            }
            req = Request(scope=scope)
    
    # Add timeout handling for the entire request
    try:
        # Set up request timeout (5 minutes max)
        timeout_task = asyncio.create_task(asyncio.sleep(300))  # 5 minutes
        chat_task = asyncio.create_task(chat(request, req, background_tasks))
        
        # Wait for either the chat to complete or timeout
        done, pending = await asyncio.wait(
            [chat_task, timeout_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Check if we got a result or timed out
        if chat_task in done:
            result = await chat_task
            elapsed_time = time.time() - start_time
            logger.info(f"RAG request {request_id} completed successfully in {elapsed_time:.2f}s")
            return result
        else:
            # Request timed out
            elapsed_time = time.time() - start_time
            logger.error(f"RAG request {request_id} timed out after {elapsed_time:.2f}s")
            
            if request.stream:
                # Return a streaming error response
                async def timeout_generator():
                    yield json.dumps({
                        "type": "error",
                        "message": "Request timed out. The query is taking too long to process. Please try a simpler question or try again later.",
                        "error_code": "TIMEOUT",
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"
                
                return StreamingResponse(timeout_generator(), media_type="text/event-stream")
            else:
                raise HTTPException(
                    status_code=504,
                    detail="Request timed out. The query is taking too long to process. Please try a simpler question or try again later."
                )
                
    except asyncio.CancelledError:
        elapsed_time = time.time() - start_time
        logger.info(f"RAG request {request_id} was cancelled after {elapsed_time:.2f}s")
        raise HTTPException(
            status_code=499,
            detail="Request was cancelled"
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"RAG request {request_id} failed after {elapsed_time:.2f}s: {str(e)}")
        
        if request.stream:
            # Return a streaming error response
            async def error_generator():
                yield json.dumps({
                    "type": "error",
                    "message": f"An error occurred: {str(e)}",
                    "error_code": "PROCESSING_ERROR",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
            
            return StreamingResponse(error_generator(), media_type="text/event-stream")
        else:
            raise e


async def retrieve_context_for_query(
    query: str,
    top_k: int = 5,
    search_filter: Optional[Dict[str, Any]] = None,
    rag_query_engine = None
) -> Dict[str, Any]:
    """
    Retrieve context for a query using the RAG system.
    
    Args:
        query: The query to retrieve context for
        top_k: Maximum number of results to retrieve
        search_filter: Optional metadata filters
        rag_query_engine: Optional pre-fetched RAG query engine
        
    Returns:
        Dict with retrieved context and status
    """
    logger.info(f"Retrieving context for query: {query}")
    
    if not rag_query_engine:
        # Get the query engine from the main app
        from main import rag_query_engine
    
    if not rag_query_engine:
        logger.warning("RAG query engine not available")
        return {
            "status": "error",
            "message": "RAG system not available",
            "context": [],
            "sources": []
        }
    
    try:
        # Retrieve context without generating a response
        result = await rag_query_engine.query(
            query_text=query,
            top_k=top_k,
            search_filter=search_filter,
            streaming=False,
            custom_prompt=None
        )
        
        # Check if we have valid context in the response
        if result and "contexts" in result:
            # Extract contexts and prepare response
            contexts = result.get("contexts", [])
            context_texts = [ctx.get("text", "") for ctx in contexts]
            sources = []
            
            # Extract source information with metadata directly from Qdrant
            for ctx in contexts:
                metadata = ctx.get("metadata", {})
                doc_id = metadata.get("doc_id", ctx.get("document_id", "unknown"))
                
                # Basic source info
                source = {
                    "doc_id": doc_id,
                    "page_number": metadata.get("page_num", metadata.get("page_number", ctx.get("page_num", 1))),
                    "score": float(ctx.get("score", 0)),
                    "retrieval_method": "HyPE",
                    "title": metadata.get("title") or metadata.get("filename", ""),
                    "content": ctx.get("text", "")  # Include the actual content
                }
                
                # Include enhanced details from Reliable RAG if available
                for key in ["original_score", "length_factor", "final_score", "llm_relevance_score", "rrf_score", "compressed"]:
                    if key in metadata:
                        source[key] = metadata[key]
                
                sources.append(source)
            
            # Extract hallucination risk from result if available
            hallucination_risk = None
            if "hallucination_metrics" in result:
                hallucination_metrics = result.get("hallucination_metrics", {})
                hallucination_risk = hallucination_metrics.get("hallucination_probability")
                
            return {
                "status": "success",
                "context": context_texts,
                "sources": sources,
                "context_quality": result.get("context_quality", "medium"),
                "query_complexity": result.get("query_complexity", "medium"),
                "hallucination_risk": hallucination_risk
            }
        elif "answer" in result:
            # If there's an answer but no contexts, return the answer as context
            # Check for hallucination risk
            hallucination_risk = None
            if "hallucination_metrics" in result:
                hallucination_metrics = result.get("hallucination_metrics", {})
                hallucination_risk = hallucination_metrics.get("hallucination_probability")
                
            return {
                "status": "success",
                "context": ["No specific context found, but generated answer: " + result.get("answer", "")],
                "sources": [],
                "message": "Generated answer without specific context",
                "hallucination_risk": hallucination_risk
            }
        else:
            logger.warning(f"No context found in RAG response")
            return {
                "status": "success",
                "message": "No context found for query",
                "context": [],
                "sources": []
            }
    except Exception as e:
        logger.warning(f"Context retrieval failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving context: {str(e)}",
            "context": [],
            "sources": []
        }

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns language code (en, fr, es, etc.)
    """
    # Simple language detection based on common words and patterns
    text_lower = text.lower()
    
    # French indicators
    french_indicators = [
        'le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'et ', 'est ', 'un ', 'une ',
        'dans ', 'pour ', 'avec ', 'sur ', 'par ', 'ce ', 'qui ', 'que ', 'comment ',
        'où ', 'quand ', 'pourquoi ', 'qu\'', 'c\'', 'd\'', 'l\'', 'n\'', 'tion ',
        'ment ', 'ées ', 'ent ', 'sont ', 'ont', 'était', 'avait', 'sera', 'sécurité',
        'réseau', 'conformité', 'réglementation', 'politique', 'gestion', 'contrôle'
    ]
    
    # Spanish indicators
    spanish_indicators = [
        'el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'y ', 'es ', 'un ', 'una ',
        'en ', 'con ', 'por ', 'para ', 'que ', 'como ', 'donde ', 'cuando ',
        'por qué ', 'cómo ', 'ción ', 'mente ', 'ado ', 'ida ', 'son ', 'han',
        'seguridad', 'red', 'cumplimiento'
    ]
    
    # English is default, but check for specific patterns
    english_indicators = [
        'the ', 'and ', 'is ', 'are ', 'was ', 'were ', 'a ', 'an ', 'in ', 'on ',
        'at ', 'by ', 'for ', 'with ', 'to ', 'of ', 'that ', 'this ', 'what ',
        'how ', 'when ', 'where ', 'why ', 'tion ', 'ment ', 'ing ', 'ed ',
        'security', 'network', 'compliance', 'regulation', 'policy', 'management'
    ]
    
    # Count indicators for each language
    french_score = sum(1 for indicator in french_indicators if indicator in text_lower)
    spanish_score = sum(1 for indicator in spanish_indicators if indicator in text_lower)
    english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
    
    # Determine language based on highest score
    if french_score > english_score and french_score > spanish_score:
        return 'fr'
    elif spanish_score > english_score and spanish_score > french_score:
        return 'es'
    else:
        return 'en'  # Default to English

def get_system_prompt_for_language(language: str, context_available: bool = False) -> str:
    """
    Get system prompt in the specified language.
    """
    prompts = {
        'fr': {
            'base': """Vous êtes un assistant spécialisé dans l'analyse réglementaire et la conformité.

Lorsque vous répondez, incluez vos réflexions internes et votre processus de raisonnement dans des balises <internal_thoughts>. 
Ces réflexions doivent expliquer votre approche pour répondre à la question et toute information clé.
Par exemple :

<internal_thoughts>
D'abord, je dois comprendre quelles réglementations s'appliquent à ce scénario.
Je devrais vérifier s'il y a des cadres de conformité spécifiques mentionnés.
Les considérations clés semblent être X, Y, et Z.
</internal_thoughts>

Ensuite, fournissez votre réponse réelle à l'utilisateur sans ces balises.
IMPORTANT : Répondez TOUJOURS en français, même si des informations en anglais sont fournies dans le contexte.""",
            'with_context': """Vous êtes un assistant spécialisé dans l'analyse réglementaire et la conformité.

Vous avez accès à des documents de contexte pertinents pour aider à répondre aux questions de l'utilisateur. Utilisez ces informations pour fournir des réponses précises et détaillées.

Lorsque vous répondez, incluez vos réflexions internes et votre processus de raisonnement dans des balises <internal_thoughts>. 
Ces réflexions doivent expliquer votre approche pour répondre à la question et toute information clé.
Par exemple :

<internal_thoughts>
D'abord, je dois analyser le contexte fourni pour trouver des informations pertinentes.
Je vois des informations sur la sécurité réseau dans le contexte.
Les points clés à aborder sont X, Y, et Z basés sur la documentation.
</internal_thoughts>

Ensuite, fournissez votre réponse réelle à l'utilisateur sans ces balises.
IMPORTANT : Répondez TOUJOURS en français, même si des informations en anglais sont fournies dans le contexte."""
        },
        'es': {
            'base': """Eres un asistente especializado en análisis regulatorio y cumplimiento.

Al responder, incluye tus pensamientos internos y proceso de razonamiento envueltos en etiquetas <internal_thoughts>. 
Estos pensamientos deben explicar tu enfoque para responder la pregunta y cualquier información clave.
Por ejemplo:

<internal_thoughts>
Primero, necesito entender qué regulaciones se aplican a este escenario.
Debería verificar si hay marcos de cumplimiento específicos mencionados.
Las consideraciones clave parecen ser X, Y, y Z.
</internal_thoughts>

Luego proporciona tu respuesta real al usuario sin estas etiquetas.
IMPORTANTE: Responde SIEMPRE en español, incluso si se proporciona información en inglés en el contexto.""",
            'with_context': """Eres un asistente especializado en análisis regulatorio y cumplimiento.

Tienes acceso a documentos de contexto relevantes para ayudar a responder las preguntas del usuario. Usa esta información para proporcionar respuestas precisas y detalladas.

Al responder, incluye tus pensamientos internos y proceso de razonamiento envueltos en etiquetas <internal_thoughts>. 
Estos pensamientos deben explicar tu enfoque para responder la pregunta y cualquier información clave.
Por ejemplo:

<internal_thoughts>
Primero, necesito analizar el contexto proporcionado para encontrar información relevante.
Veo información sobre seguridad de red en el contexto.
Los puntos clave a abordar son X, Y, y Z basados en la documentación.
</internal_thoughts>

Luego proporciona tu respuesta real al usuario sin estas etiquetas.
IMPORTANTE: Responde SIEMPRE en español, incluso si se proporciona información en inglés en el contexto."""
        },
        'en': {
            'base': """You are a helpful assistant specializing in regulatory analysis and compliance.

When responding, include your internal thoughts and reasoning process wrapped in <internal_thoughts> tags. 
These thoughts should explain your approach to answering the question and any key insights.
For example:

<internal_thoughts>
First, I need to understand what regulations apply to this scenario.
I should check if there are any specific compliance frameworks mentioned.
The key considerations seem to be X, Y, and Z.
</internal_thoughts>

Then provide your actual response to the user without these tags.
IMPORTANT: Always respond in English, even if information in other languages is provided in the context.""",
            'with_context': """You are a helpful assistant specializing in regulatory analysis and compliance.

You have access to relevant context documents to help answer the user's questions. Use this information to provide accurate and detailed responses.

When responding, include your internal thoughts and reasoning process wrapped in <internal_thoughts> tags. 
These thoughts should explain your approach to answering the question and any key insights.
For example:

<internal_thoughts>
First, I need to analyze the provided context to find relevant information.
I can see information about network security in the context.
The key points to address are X, Y, and Z based on the documentation.
</internal_thoughts>

Then provide your actual response to the user without these tags.
IMPORTANT: Always respond in English, even if information in other languages is provided in the context."""
        }
    }
    
    language_prompts = prompts.get(language, prompts['en'])  # Default to English
    return language_prompts['with_context'] if context_available else language_prompts['base']
