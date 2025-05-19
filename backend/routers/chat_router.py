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
from openai import OpenAI, AsyncOpenAI

from pyndantic_agents.agent_factory import create_agent
from pyndantic_agents.base_agent import BaseAgent, AgentInput
from pyndantic_agents.rag_agent import RAGAgent
from llamaIndex_rag.rag import RAGSystem, NodeWithScore
from pyndantic_agents.tree_reasoning import TreeReasoningAgent

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
    """Handle chat requests, optionally using RAG and agents."""
    session_id = request.session_id or str(uuid.uuid4())
    user_id = await extract_user_id_from_request(req, provided_user_id=req.headers.get("X-User-ID")) # Example user ID extraction
    start_time = time.time()
    execution_id = str(uuid.uuid4()) # Unique ID for this chat interaction or agent execution

    # Ensure OPENAI_API_KEY is available if any LLM call is to be made
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set. LLM calls will fail.")
        # Potentially raise HTTPException here if critical for all paths

    # Initialize RAG system 
    rag_system_instance = await get_rag_system() # RAGSystem instance
    
    response_message = ""
    context_used_flag = False
    agent_used_flag = False
    tree_reasoning_used_flag = False
    final_agent_type = request.agent_type
    agent: Optional[BaseAgent] = None # Define agent variable
    sources = None # Initialize sources variable

    try:
        if request.use_agent and request.agent_type:
            agent_used_flag = True
            logger.info(f"Using agent: {request.agent_type} for session {session_id}")
            
            agent_query = request.messages[-1].content if request.messages else ""
            if not agent_query:
                raise HTTPException(status_code=400, detail="Cannot use agent with empty message list.")

            agent_settings = request.agent_params or {}
            # Ensure necessary LLM parameters are in agent_settings for create_agent
            agent_settings.setdefault('model', request.model)
            agent_settings.setdefault('temperature', request.temperature)
            agent_settings.setdefault('max_tokens', request.max_tokens)
            # openai_api_key is passed directly to create_agent or handled by its llm_config
            
            # Parameters for create_agent
            create_agent_params = {
                "agent_type": request.agent_type,
                "rag_system": rag_system_instance,
                "api_key": openai_api_key, # create_agent handles this for its LLMConfig
                # Pass through existing agent_settings which might have model, temp, etc.
                # create_agent will prioritize its specific llm_config args then kwargs
                **agent_settings 
            }

            if request.agent_type == "TreeReasoningAgent" or request.use_tree_reasoning:
                tree_reasoning_used_flag = True
                final_agent_type = "TreeReasoningAgent" # This is for the response model, can remain CamelCase
                logger.info(f"Initiating TreeReasoningAgent with settings: {agent_settings}")

                # Get RAG context for the TreeReasoningAgent
                initial_retrieval_top_k = agent_settings.get("initial_retrieval_top_k", 5)
                retrieved_nodes = None
                if initial_retrieval_top_k > 0 and rag_system_instance:
                    logger.info(f"[TreeReasoningAgent] Retrieving top {initial_retrieval_top_k} documents for query: {agent_query[:100]}...")
                    retrieved_nodes = rag_system_instance.retrieve( 
                        agent_query, 
                        top_k=initial_retrieval_top_k,
                        use_query_expansion=True,
                        auto_filter=True,
                        use_hierarchical=True
                    )
                    context_used_flag = True if retrieved_nodes else False
                    logger.info(f"[TreeReasoningAgent] Retrieved {len(retrieved_nodes) if retrieved_nodes else 0} nodes.")
                    
                    # Extract sources information
                    if retrieved_nodes:
                        sources = []
                        for i, node in enumerate(retrieved_nodes):
                            source_info = {
                                "id": i + 1,
                                "title": node.get("metadata", {}).get("doc_name", f"Document {i+1}"),
                                "file_path": node.get("metadata", {}).get("file_path", ""),
                                "score": node.get("score", 0),
                                "chunk_id": node.get("chunk_id", ""),
                                "text_preview": node.get("text", "") # Include full text without truncation
                            }
                            sources.append(source_info)

                # Update create_agent_params for TreeReasoningAgent specific needs
                create_agent_params["agent_type"] = "tree_reasoning" # Changed to lowercase to match factory key
                tree_id = request.tree_template or agent_settings.get("tree_template", "default_tree")
                create_agent_params['tree_id'] = tree_id
                if request.custom_tree:
                    create_agent_params['custom_tree_config'] = request.custom_tree

                try:
                    agent = create_agent(**create_agent_params)
                    if not isinstance(agent, TreeReasoningAgent):
                        logger.error(f"Created agent is not a TreeReasoningAgent instance as expected.")
                        raise HTTPException(status_code=500, detail="Agent creation failed for TreeReasoningAgent.")
                except Exception as e:
                    logger.error(f"Error creating TreeReasoningAgent: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Could not create TreeReasoningAgent: {str(e)}")

                # Ensure agent is a TreeReasoningAgent for type safety before calling process
                if isinstance(agent, TreeReasoningAgent):
                    agent_result_dict = await agent.process(
                        query=agent_query, 
                        initial_retrieved_nodes=retrieved_nodes, 
                        max_depth=agent_settings.get("max_depth", 10),
                        agent_settings=agent_settings 
                    )
                    response_message = agent_result_dict.get("response", "Tree agent did not return a standard response.")
                    # Check if agent result includes source information
                    if "sources" in agent_result_dict:
                        sources = agent_result_dict["sources"]
                else: # Should have been caught by the isinstance check above
                    raise HTTPException(status_code=500, detail="Failed to correctly initialize TreeReasoningAgent.")


            elif request.agent_type == "RAGAgent": 
                final_agent_type = "RAGAgent"
                logger.info(f"Initiating RAGAgent with settings: {agent_settings}")
                create_agent_params["agent_type"] = "RAGAgent" # Explicitly set
                
                try:
                    agent = create_agent(**create_agent_params)
                    if not isinstance(agent, RAGAgent):
                        logger.error(f"Created agent is not a RAGAgent instance as expected.")
                        raise HTTPException(status_code=500, detail="Agent creation failed for RAGAgent.")
                except Exception as e:
                    logger.error(f"Error creating RAGAgent: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Could not create RAGAgent: {str(e)}")

                rag_agent_process_params = {
                    "filter_criteria": agent_settings.get("filter_criteria"),
                    "use_neo4j": agent_settings.get("use_neo4j", False), # RAGAgent might have its own defaults
                    "top_k": agent_settings.get("top_k", rag_system_instance.DEFAULT_TOP_K if rag_system_instance else 5),
                    "retrieval_id": execution_id 
                }
                logger.info(f"Calling RAGAgent.process with params: {rag_agent_process_params}")
                # Ensure agent is a RAGAgent for type safety
                if isinstance(agent, RAGAgent):
                    agent_response_obj = await agent.process(query=agent_query, **rag_agent_process_params)
                    response_message = agent_response_obj.response
                    context_used_flag = bool(agent_response_obj.context_used)
                    if agent_response_obj.context_used:
                        sources = []
                        for i, node_data in enumerate(agent_response_obj.context_used):
                            # node_data is a dict like {"text": ..., "metadata": ..., "score": ..., "doc_id": ..., "chunk_id": ...}
                            metadata = node_data.get("metadata", {})
                            source_info = {
                                "id": metadata.get("chunk_id", f"rag_source_{i+1}"), # Use chunk_id or a generated ID
                                "title": metadata.get("doc_name", metadata.get("file_name", f"Source {i+1}")),
                                "file_path": metadata.get("file_path", metadata.get("file_name", "")), # Prefer file_path
                                "score": node_data.get("score"),
                                "text_preview": node_data.get("text", ""), # Include full text without truncation
                                "chunk_id": node_data.get("chunk_id", "")
                            }
                            sources.append(source_info)
                else: # Should have been caught by the isinstance check above
                     raise HTTPException(status_code=500, detail="Failed to correctly initialize RAGAgent.")


            else: # Generic agent
                logger.info(f"Attempting to use generic agent type: {request.agent_type}")
                create_agent_params["agent_type"] = request.agent_type # Explicitly set
                try:
                    agent = create_agent(**create_agent_params)
                    if not isinstance(agent, BaseAgent):
                        logger.error(f"Created agent is not a BaseAgent instance as expected.")
                        raise HTTPException(status_code=500, detail=f"Agent creation failed for {request.agent_type}.")

                    if hasattr(agent, "process") and callable(agent.process):
                        # Prepare AgentInput for BaseAgent derived agents
                        current_process_kwargs = {**agent_settings}
                        # Remove params consumed by create_agent or not directly part of AgentInput.parameters
                        for k_to_pop in ['model', 'temperature', 'max_tokens', 'api_key', 'rag_system', 'tree_id', 'custom_tree_config']:
                            current_process_kwargs.pop(k_to_pop, None)
                        
                        # Construct AgentInput
                        # query is mandatory for AgentInput
                        # context can be None
                        # parameters will be the remaining agent_settings/process_kwargs
                        agent_input_obj = AgentInput(query=agent_query, parameters=current_process_kwargs)
                        
                        # Call agent.process with the AgentInput object
                        agent_result = await agent.process(agent_input_obj)
                        
                        # AgentOutput (and its subclasses like VulnerabilityAssessmentOutput) is a Pydantic model
                        # It should have a 'response' field. We can access it directly or via .dict() / .model_dump()
                        if hasattr(agent_result, 'response') and isinstance(agent_result.response, str):
                            response_message = agent_result.response
                        elif isinstance(agent_result, str): # Fallback if process directly returns a string
                            response_message = agent_result
                        elif isinstance(agent_result, dict) and "response" in agent_result: # Fallback for dict
                            response_message = agent_result["response"]
                        else:
                            # If agent_result is an AgentOutput model, convert to dict to log
                            log_result = agent_result
                            if hasattr(agent_result, 'model_dump'):
                                log_result = agent_result.model_dump()
                            elif hasattr(agent_result, 'dict'):
                                log_result = agent_result.dict()
                            response_message = "Agent processed the request but returned an unknown or non-standard format."
                            logger.warning(f"Agent {request.agent_type} returned: {log_result}. Expected AgentOutput with a 'response' string or direct string.")
                    else:
                        response_message = f"Agent type {request.agent_type} does not have a callable 'process' method."
                        logger.error(response_message)
                        raise HTTPException(status_code=501, detail=response_message)
                        
                except Exception as e:
                    logger.error(f"Error processing with agent {request.agent_type}: {str(e)}")
                    # Check if it's a ValueError from create_agent for an unknown agent type
                    if isinstance(e, ValueError) and "Unknown agent type" in str(e):
                        raise HTTPException(status_code=400, detail=str(e))
                    response_message = f"Error with agent {request.agent_type}: {str(e)}"
                    # raise HTTPException(status_code=500, detail=response_message) # Avoid double raising if already an HTTPException

        else: # Direct LLM call
            logger.info(f"Direct LLM call for session {session_id}. Model: {request.model}")
            llm_query = request.messages[-1].content if request.messages else ""
            if not llm_query:
                raise HTTPException(status_code=400, detail="Cannot process empty message list.")

            context_str = ""
            retrieved_docs_for_direct_llm = None
            if request.include_context and rag_system_instance:
                logger.info(f"Retrieving context for direct LLM call. Query: {llm_query[:100]}...")
                # Use context_query if provided, else use the main llm_query
                query_for_rag = request.context_query or llm_query
                retrieved_docs_for_direct_llm = rag_system_instance.retrieve(
                    query_for_rag,
                    top_k=request.agent_params.get("top_k", 5) if request.agent_params else 5,
                    use_query_expansion=True,
                    auto_filter=True,
                    use_hierarchical=True
                )
                if retrieved_docs_for_direct_llm:
                    context_used_flag = True
                    # Format for direct LLM - simple concatenation for now
                    # This formatting does not use [Source X] by default for direct LLM calls yet.
                    # Could be enhanced to use _format_context_with_sources from an agent or utility.
                    context_items_text = []
                    sources = []
                    
                    for i, node in enumerate(retrieved_docs_for_direct_llm):
                        source_id = i + 1
                        
                        # Extract metadata for source info
                        metadata = node.get("metadata", {}) if isinstance(node, dict) else (node.metadata if hasattr(node, "metadata") else {})
                        doc_name = metadata.get("doc_name", f"Document {source_id}")
                        file_path = metadata.get("file_path", doc_name)
                        
                        # Get content from node based on its type
                        if isinstance(node, dict):
                            content = node.get("text", "")
                        elif hasattr(node, "get_content"):
                            content = node.get_content()
                        else:
                            content = str(node)
                            
                        # Add formatted content with source marker
                        context_items_text.append(f"[Source {source_id}]\n{content}")
                        
                        # Create source info for response
                        source_info = {
                            "id": source_id,
                            "title": doc_name,
                            "file_path": file_path,
                            "score": node.get("score", 0) if isinstance(node, dict) else (node.score if hasattr(node, "score") else 0),
                            "chunk_id": metadata.get("chunk_id", ""),
                            "text_preview": content # Include full text without truncation
                        }
                        sources.append(source_info)
                    
                    context_str = "\n\n".join(context_items_text)
                    logger.info(f"Retrieved {len(retrieved_docs_for_direct_llm)} context items for direct LLM.")
                else:
                    logger.info("No context items retrieved for direct LLM.")
            
            # Construct messages for OpenAI API
            messages_for_api = []
            if context_str:
                messages_for_api.append({"role": "system", "content": f"Use the following context to answer the user\'s query. If the context is not relevant, answer based on your general knowledge. Always respond in the same language as the user's query.\n\nContext:\n{context_str}"})    
            for msg in request.messages:
                messages_for_api.append({"role": msg.role, "content": msg.content})

            if request.stream:
                logger.info("Streaming response for direct LLM call.")
                # Placeholder for actual streaming logic
                async def generate():
                    # Start streaming response
                    try:
                        client = AsyncOpenAI(api_key=openai_api_key)
                        stream = await client.chat.completions.create(
                            model=request.model,
                            messages=messages_for_api,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            stream=True,
                        )
                        accumulated_response = ""
                        
                        # Send initial event
                        yield f"data: {json.dumps({'event': 'start', 'session_id': session_id})}\n\n"
                        
                        async for chunk in stream:
                            content = chunk.choices[0].delta.content or ""
                            accumulated_response += content
                            
                            # Send chunk in a format the frontend expects
                            yield f"data: {json.dumps({'event': 'chunk', 'content': content})}\n\n"
                        
                        # Send final event with metadata
                        metadata = {
                            'event': 'end', 
                            'session_id': session_id,
                            'model': request.model,
                            'context_used': context_used_flag,
                            'sources': sources
                        }
                        yield f"data: {json.dumps(metadata)}\n\n"
                        
                        # Send the complete message for reference
                        complete_data = {
                            'type': 'metadata',
                            'data': {
                                'message': accumulated_response,
                                'model': request.model,
                                'session_id': session_id,
                                'context_used': context_used_flag,
                                'sources': sources
                            }
                        }
                        yield f"data: {json.dumps(complete_data)}\n\n"
                        
                        # After stream, save to history
                        # Using a background task to avoid delaying the stream closing
                        history_entry = ChatMessage(role="assistant", content=accumulated_response)
                        background_tasks.add_task(save_chat_message, session_id, user_id, request.messages + [history_entry])
                        # Track execution after response is fully sent
                        background_tasks.add_task(track_agent_execution, 
                            agent_id="direct_llm", session_id=session_id, task=llm_query, model=request.model, 
                            start_time=start_time, error=False # TODO: Add token tracking if possible from stream
                        )

                    except Exception as e:
                        logger.error(f"Error during streaming: {str(e)}")
                        # Send an error message through the stream if possible
                        error_payload = {
                            'error': True, 
                            'message': f"Error during streaming: {str(e)}",
                            'is_last': True
                        }
                        yield f"data: {json.dumps(error_payload)}\n\n"
                return StreamingResponse(generate(), media_type="text/event-stream")
            else: # Non-streaming direct LLM call
                logger.info("Non-streaming direct LLM call.")
                client = AsyncOpenAI(api_key=openai_api_key)
                completion = await client.chat.completions.create(
                    model=request.model,
                    messages=messages_for_api,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                response_message = completion.choices[0].message.content.strip()
                # TODO: Add token tracking from completion.usage
                # track_agent_execution(...) with token counts

        # Save message to history (if not streamed, as streaming handles its own saving)
        if not (request.use_agent == False and request.stream):
            current_messages = list(request.messages) # Make a mutable copy
            current_messages.append(ChatMessage(role="assistant", content=response_message))
            background_tasks.add_task(save_chat_message, session_id, user_id, current_messages)
        
        # Track execution (if not streamed, as streaming handles its own tracking)
        # For agent calls, or non-streamed direct LLM.
        if not (request.use_agent == False and request.stream):
            agent_id_for_tracking = final_agent_type if agent_used_flag else "direct_llm_nonstream"
            query_for_tracking = request.messages[-1].content if request.messages else ""
            background_tasks.add_task(track_agent_execution, 
                agent_id=agent_id_for_tracking, session_id=session_id, task=query_for_tracking, 
                model=request.model, start_time=start_time, error=False
            )

        return ChatResponse(
            message=response_message,
            model=request.model,
            agent_type=final_agent_type if agent_used_flag else None,
            agent_used=agent_used_flag,
            tree_reasoning_used=tree_reasoning_used_flag,
            context_used=context_used_flag,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            execution_id=execution_id, # Pass back execution_id
            sources=sources  # Include sources in the response
        )
    
    except HTTPException as http_exc:
        logger.error(f"HTTPException in chat endpoint: {http_exc.detail}", exc_info=True)
        # Track execution with error
        query_for_tracking = request.messages[-1].content if request.messages and len(request.messages) > 0 else ""
        agent_id_for_tracking = final_agent_type if agent_used_flag else "direct_llm_error"
        background_tasks.add_task(track_agent_execution, 
            agent_id=agent_id_for_tracking, session_id=session_id, task=query_for_tracking, model=request.model, 
            start_time=start_time, error=True, error_message=http_exc.detail
        )
        raise http_exc # Re-raise the exception

    except Exception as e:
        logger.error(f"General error in chat endpoint: {str(e)}", exc_info=True)
        # Track execution with error
        query_for_tracking = request.messages[-1].content if request.messages and len(request.messages) > 0 else ""
        agent_id_for_tracking = final_agent_type if agent_used_flag else "direct_llm_error"
        background_tasks.add_task(track_agent_execution, 
            agent_id=agent_id_for_tracking, session_id=session_id, task=query_for_tracking, model=request.model, 
            start_time=start_time, error=True, error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

async def save_chat_message(session_id: str, user_id: Optional[str], messages: List[ChatMessage]):
    """Save chat messages to the database."""
    try:
        conn = await get_db_connection()
        cursor = conn.cursor()
        
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
