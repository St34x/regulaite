"""
FastAPI router for agent metadata, capabilities and documentation.
"""
import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
import os
import sys
from pathlib import Path

# Add the backend directory to the path so we can import the agent framework
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Import the agent framework
from agent_framework.factory import get_agent_instance
from agent_framework.tool_registry import ToolRegistry, ToolMetadata
from agent_framework.agent import Agent, Query, AgentResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/agents",
    tags=["agents"],
    responses={404: {"description": "Not found"}},
)

# Models for API
class AgentCapability(BaseModel):
    """Capability of an agent."""
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of the capability")
    requires_context: bool = Field(False, description="Whether the capability requires context")
    examples: List[str] = Field(default_factory=list, description="Example queries or prompts")


class AgentParameter(BaseModel):
    """Parameter for agent configuration."""
    name: str = Field(..., description="Name of the parameter")
    description: str = Field(..., description="Description of the parameter")
    type: str = Field(..., description="Type of the parameter (string, number, boolean, etc.)")
    required: bool = Field(False, description="Whether the parameter is required")
    default: Optional[Any] = Field(None, description="Default value of the parameter")
    options: Optional[List[Any]] = Field(None, description="Available options for the parameter")


class AgentMetadata(BaseModel):
    """Metadata about an agent."""
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Display name for the agent")
    description: str = Field(..., description="Description of the agent")
    version: str = Field(..., description="Version of the agent")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Capabilities of the agent")
    parameters: List[AgentParameter] = Field(default_factory=list, description="Parameters for agent configuration")
    model_requirements: Dict[str, Any] = Field(default_factory=dict, description="Model requirements for the agent")
    context_usage: str = Field("optional", description="How the agent uses context (required, optional, none)")
    author: Optional[str] = Field(None, description="Author of the agent")
    documentation_url: Optional[str] = Field(None, description="URL to documentation")
    icon: Optional[str] = Field(None, description="Icon for the agent")
    tags: List[str] = Field(default_factory=list, description="Tags for the agent")


class AgentUsageExample(BaseModel):
    """Example of agent usage."""
    query: str = Field(..., description="Example query")
    description: str = Field(..., description="Description of the example")
    result_summary: Optional[str] = Field(None, description="Summary of expected result")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the example")


class AgentDocumentation(BaseModel):
    """Documentation for an agent."""
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Display name for the agent")
    description: str = Field(..., description="Description of the agent")
    long_description: str = Field(..., description="Long-form description of the agent")
    usage_examples: List[AgentUsageExample] = Field(default_factory=list, description="Examples of agent usage")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the agent")
    best_practices: List[str] = Field(default_factory=list, description="Best practices for using the agent")
    parameter_details: Optional[Dict[str, str]] = Field(None, description="Detailed descriptions of parameters")
    faq: Optional[List[Dict[str, str]]] = Field(None, description="Frequently asked questions")
    related_agents: Optional[List[str]] = Field(None, description="Related agents")


class AgentDecisionTree(BaseModel):
    """Decision tree for an agent."""
    id: str = Field(..., description="Unique identifier for the tree")
    name: str = Field(..., description="Name of the tree")
    description: str = Field(..., description="Description of the tree")
    nodes: Dict[str, Any] = Field(..., description="Nodes of the tree")
    agent_id: str = Field(..., description="ID of the agent this tree belongs to")
    version: str = Field(..., description="Version of the tree")
    is_default: bool = Field(False, description="Whether this is the default tree for the agent")


class AgentFeedback(BaseModel):
    """User feedback for an agent response."""
    agent_id: str = Field(..., description="ID of the agent")
    session_id: str = Field(..., description="ID of the chat session")
    message_id: Optional[str] = Field(None, description="ID of the specific message")
    rating: int = Field(..., description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")
    timestamp: Optional[str] = Field(None, description="Timestamp of the feedback")
    context_used: Optional[bool] = Field(None, description="Whether context was used in the response")
    model: Optional[str] = Field(None, description="Model used for the response")


class AgentHealthCheck(BaseModel):
    """Health check information for an agent."""
    agent_id: str = Field(..., description="ID of the agent")
    status: str = Field(..., description="Current status (online, offline, degraded)")
    version: str = Field(..., description="Version of the agent")
    last_execution: Optional[str] = Field(None, description="Timestamp of last execution")
    error_count: Optional[int] = Field(None, description="Number of errors in recent executions")
    avg_response_time: Optional[float] = Field(None, description="Average response time in milliseconds")


class AgentRequest(BaseModel):
    """Request for executing an agent."""
    agent_type: str = Field(..., description="Type of agent to use")
    query: str = Field(..., description="Query to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the agent")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    include_context: bool = Field(True, description="Whether to include context in the response")
    model: Optional[str] = Field("gpt-4", description="Model to use for generation")


class AgentResponse(BaseModel):
    """Response from executing an agent."""
    agent_id: str = Field(..., description="ID of the agent")
    query: str = Field(..., description="Query executed")
    response: str = Field(..., description="Response from the agent")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used in the response")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in the response")
    context_used: bool = Field(False, description="Whether context was used in the response")
    execution_time: float = Field(..., description="Time taken to execute the query in seconds")
    model: Optional[str] = Field(None, description="Model used for generation")

# Singleton tool registry for the router
_tool_registry = ToolRegistry()

# Dependency to get agent types from the agent framework
async def get_agent_types():
    """Get available agent types from the agent framework."""
    return {
        "rag": "Retrieval-augmented generation agent",
        "qa": "Question answering agent",
        "summarization": "Document summarization agent"
    }


@router.get("/types", response_model=Dict[str, str])
async def list_agent_types():
    """List all available agent types."""
    agent_types = await get_agent_types()
    return agent_types


@router.get("/metadata", response_model=List[AgentMetadata])
async def get_agents_metadata():
    """Get metadata for all available agents."""
    try:
        # Get agent types first
        agent_types = await get_agent_types()

        # Create metadata for each agent type
        metadata_list = []
        for agent_id, description in agent_types.items():
            # Define capabilities based on agent type
            capabilities = []
            if agent_id == "rag":
                capabilities = [
                    AgentCapability(
                        name="Context-enhanced responses", 
                        description="Use relevant documents to enhance responses",
                        requires_context=True,
                        examples=[
                            "What regulations apply to GDPR data transfers?",
                            "Explain the requirements for financial compliance under MiFID II"
                        ]
                    ),
                    AgentCapability(
                        name="Document-based Q&A",
                        description="Answer questions based on document content",
                        requires_context=True,
                        examples=[
                            "What does Article 83 of the GDPR say about penalties?",
                            "Find information about PSD2 strong customer authentication"
                        ]
                    )
                ]
            elif agent_id == "qa":
                capabilities = [
                    AgentCapability(
                        name="Direct questioning",
                        description="Answer questions directly using the model's knowledge",
                        requires_context=False,
                        examples=[
                            "What is the purpose of a compliance program?",
                            "Explain the difference between regulations and directives"
                        ]
                    )
                ]
            elif agent_id == "summarization":
                capabilities = [
                    AgentCapability(
                        name="Document summarization",
                        description="Summarize document content",
                        requires_context=True,
                        examples=[
                            "Summarize this document about Basel III",
                            "Give me a summary of the MIFID II regulation"
                        ]
                    )
                ]

            # Define parameters based on agent type
            parameters = [
                AgentParameter(
                    name="model",
                    description="Language model to use",
                    type="string",
                    required=False,
                    default="gpt-4",
                    options=["gpt-4", "gpt-3.5-turbo"]
                ),
                AgentParameter(
                    name="max_sources",
                    description="Maximum number of sources to use",
                    type="integer",
                    required=False,
                    default=5,
                    options=[1, 3, 5, 10]
                )
            ]

            # Create the metadata
            metadata = AgentMetadata(
                id=agent_id,
                name=f"{agent_id.capitalize()} Agent",
                description=description,
                version="1.0.0",
                capabilities=capabilities,
                parameters=parameters,
                model_requirements={"recommended": "gpt-4"},
                context_usage="optional" if agent_id in ["rag", "summarization"] else "none",
                author="RegulAIte",
                tags=[agent_id]
            )

            metadata_list.append(metadata)

        return metadata_list

    except Exception as e:
        logger.error(f"Error retrieving agent metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent metadata: {str(e)}"
        )


@router.get("/metadata/{agent_id}", response_model=AgentMetadata)
async def get_agent_metadata(agent_id: str):
    """Get metadata for a specific agent."""
    try:
        # Get all agent metadata
        metadata_list = await get_agents_metadata()

        # Find the requested agent
        for metadata in metadata_list:
            if metadata.id == agent_id:
                return metadata

        # If we get here, the agent was not found
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found: {agent_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent metadata: {str(e)}"
        )


@router.get("/documentation/{agent_id}", response_model=AgentDocumentation)
async def get_agent_documentation(agent_id: str):
    """Get documentation for a specific agent."""
    try:
        # Get the agent metadata first
        metadata = await get_agent_metadata(agent_id)

        # Create example usage examples based on agent capabilities
        usage_examples = []
        for capability in metadata.capabilities:
            for example in capability.examples:
                usage_examples.append(AgentUsageExample(
                    query=example,
                    description=f"Example of {capability.name}",
                    result_summary=f"The agent will use its {capability.name} capability to respond"
                ))

        # Define limitations based on agent type
        limitations = []
        if agent_id == "rag":
            limitations = [
                "Responses are limited to the information available in the knowledge base",
                "May not have the latest information if the knowledge base is not up-to-date",
                "Complex reasoning across multiple documents may be limited"
            ]
        elif agent_id == "qa":
            limitations = [
                "Responses are limited to the model's pre-trained knowledge",
                "May not have domain-specific expertise without context",
                "Cannot access real-time information"
            ]
        elif agent_id == "summarization":
            limitations = [
                "Summary quality depends on the clarity and structure of the source document",
                "May miss nuanced details in very technical documents",
                "Length of summary is limited by token constraints"
            ]

        # Define best practices based on agent type
        best_practices = [
            "Provide clear, specific queries",
            "Include key terms and concepts in your query",
            "Use follow-up questions to refine responses"
        ]

        # Create the documentation
        documentation = AgentDocumentation(
            id=metadata.id,
            name=metadata.name,
            description=metadata.description,
            long_description=f"The {metadata.name} provides advanced capabilities for {metadata.description.lower()}. "
                            f"It leverages a powerful language model combined with a knowledge retrieval system to "
                            f"deliver accurate, contextually-relevant responses to your queries.",
            usage_examples=usage_examples,
            limitations=limitations,
            best_practices=best_practices,
            parameter_details={
                "model": "The language model used for generating responses. Higher-tier models like GPT-4 provide better reasoning but may be slower.",
                "max_sources": "Maximum number of sources to retrieve from the knowledge base. Higher values provide more context but may introduce noise."
            },
            faq=[
                {"question": "How accurate are the responses?", "answer": "The agent strives for high accuracy, but responses should be verified when used for critical applications."},
                {"question": "Can it access the internet?", "answer": "No, the agent can only access information from its knowledge base and training data."},
                {"question": "How is context determined?", "answer": "Context is retrieved based on semantic relevance to your query using vector similarity search."}
            ]
        )

        return documentation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent documentation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent documentation: {str(e)}"
        )


@router.get("/capabilities", response_model=List[AgentCapability])
async def get_all_capabilities():
    """Get all available agent capabilities."""
    try:
        # Get all agent metadata
        metadata_list = await get_agents_metadata()

        # Collect all capabilities
        all_capabilities = []
        for metadata in metadata_list:
            all_capabilities.extend(metadata.capabilities)

        # Remove duplicates based on name
        unique_capabilities = []
        capability_names = set()
        for capability in all_capabilities:
            if capability.name not in capability_names:
                unique_capabilities.append(capability)
                capability_names.add(capability.name)

        return unique_capabilities

    except Exception as e:
        logger.error(f"Error retrieving agent capabilities: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent capabilities: {str(e)}"
        )


@router.get("/tags", response_model=List[str])
async def get_all_tags():
    """Get all available agent tags."""
    try:
        # Get all agent metadata
        metadata_list = await get_agents_metadata()

        # Collect all tags
        all_tags = []
        for metadata in metadata_list:
            all_tags.extend(metadata.tags)

        # Remove duplicates
        unique_tags = list(set(all_tags))

        return unique_tags

    except Exception as e:
        logger.error(f"Error retrieving agent tags: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent tags: {str(e)}"
        )


@router.post("/feedback", response_model=Dict[str, Any])
async def submit_agent_feedback(feedback: AgentFeedback):
    """Submit feedback for an agent response."""
    try:
        # Validate the agent ID
        try:
            await get_agent_metadata(feedback.agent_id)
        except HTTPException:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {feedback.agent_id}"
            )

        # In a real implementation, this would store the feedback in a database
        # For now, just log it
        logger.info(f"Received feedback for agent {feedback.agent_id}: {feedback.rating}/5")
        if feedback.feedback_text:
            logger.info(f"Feedback text: {feedback.feedback_text}")

        return {
            "success": True,
            "message": "Feedback received",
            "feedback_id": f"feedback_{feedback.agent_id}_{feedback.session_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )


@router.get("/health", response_model=List[AgentHealthCheck])
async def get_agents_health():
    """Get health information for all agents."""
    try:
        # Get agent types
        agent_types = await get_agent_types()

        # Create health checks for each agent type
        health_checks = []
        for agent_id in agent_types:
            # In a real implementation, this would check the actual health
            # of the agent. For now, just report all as online.
            health_check = AgentHealthCheck(
                agent_id=agent_id,
                status="online",
                version="1.0.0",
                error_count=0,
                avg_response_time=1.5  # seconds
            )
            health_checks.append(health_check)

        return health_checks

    except Exception as e:
        logger.error(f"Error retrieving agents health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agents health: {str(e)}"
        )


@router.get("/health/{agent_id}", response_model=AgentHealthCheck)
async def get_agent_health(agent_id: str):
    """Get health information for a specific agent."""
    try:
        # Get all health checks
        health_checks = await get_agents_health()

        # Find the requested agent
        for health_check in health_checks:
            if health_check.agent_id == agent_id:
                return health_check

        # If we get here, the agent was not found
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found: {agent_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent health: {str(e)}"
        )


@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute an agent with the given query."""
    try:
        import time
        start_time = time.time()

        # Get the agent instance
        agent = await get_agent_instance(
            agent_type=request.agent_type,
            model=request.model or "gpt-4"
        )

        # Create the query
        query = Query(
            query_text=request.query,
            parameters=request.parameters or {}
        )

        # Add session_id to context if provided
        if request.session_id:
            query.context.session_id = request.session_id

        # Execute the agent
        agent_response = await agent.process_query(query)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create the response
        response = AgentResponse(
            agent_id=agent.agent_id,
            query=request.query,
            response=agent_response.content,
            sources=agent_response.metadata.get("sources"),
            tools_used=agent_response.tools_used,
            context_used=agent_response.context_used,
            execution_time=execution_time,
            model=request.model
        )

        return response

    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent: {str(e)}"
        )
