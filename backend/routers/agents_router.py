"""
FastAPI router for agent metadata, capabilities and documentation.
"""
import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
import os

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


# Dependency to get agent types from the main app
async def get_agent_types():
    """Get agent types from Pyndantic agents."""
    # Since pyndantic_agents is removed, return default agent types
    return {
        "rag": "Retrieval-augmented generation agent",
        "qa": "Question answering agent",
        "summarization": "Document summarization agent"
    }


# Dependency to get decision trees from the main app
async def get_decision_trees():
    """Get available decision trees."""
    # Since pyndantic_agents is removed, return an empty dict
    return {}


@router.get("/types", response_model=Dict[str, str])
async def list_agent_types():
    """List all available agent types."""
    agent_types = await get_agent_types()
    return agent_types


@router.get("/trees", response_model=Dict[str, Dict[str, Any]])
async def list_decision_trees():
    """List all available decision trees."""
    trees = await get_decision_trees()
    return trees


@router.get("/metadata", response_model=List[AgentMetadata])
async def get_agents_metadata():
    """Get metadata for all available agents."""
    try:
        # Get agent types first
        agent_types = await get_agent_types()

        # Since pyndantic_agents is removed, provide default metadata
        metadata_dict = {
            agent_id: {
                "id": agent_id,
                "name": f"{agent_id.capitalize()} Agent",
                "description": description,
                "version": "1.0.0",
                "capabilities": [],
                "parameters": [],
                "model_requirements": {"recommended": "gpt-4"},
                "context_usage": "optional",
                "author": "RegulAIte",
                "documentation_url": None,
                "icon": None,
                "tags": [agent_id]
            }
            for agent_id, description in agent_types.items()
        }

        # Return metadata as list
        return [AgentMetadata(**metadata) for metadata in metadata_dict.values()]

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
        # Re-raise HTTP exceptions
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
        # Since pyndantic_agents is removed, create minimal documentation on the fly
        try:
            metadata = await get_agent_metadata(agent_id)

            # Create minimal documentation
            return AgentDocumentation(
                id=agent_id,
                name=metadata.name,
                description=metadata.description,
                long_description=f"Detailed documentation for {metadata.name} is not available yet.",
                usage_examples=[
                    AgentUsageExample(
                        query=f"Help me with {agent_id} analysis",
                        description=f"Basic {agent_id} analysis request"
                    )
                ],
                limitations=["Documentation is under development."],
                best_practices=["Refer to the general agent documentation for best practices."]
            )
        except HTTPException:
            # If agent metadata is not found, raise 404
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {agent_id}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent documentation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent documentation: {str(e)}"
        )


@router.get("/trees/{tree_id}", response_model=AgentDecisionTree)
async def get_decision_tree(tree_id: str):
    """Get a specific decision tree."""
    try:
        # Get all trees
        trees = await get_decision_trees()

        # Check if the requested tree exists
        if tree_id not in trees:
            raise HTTPException(
                status_code=404,
                detail=f"Decision tree not found: {tree_id}"
            )

        # Get the tree
        tree_data = trees[tree_id]

        # Extract tree metadata and structure
        tree = AgentDecisionTree(
            id=tree_id,
            name=tree_data.get("name", f"Tree {tree_id}"),
            description=tree_data.get("description", "No description available"),
            nodes=tree_data.get("nodes", {}),
            agent_id=tree_data.get("agent_id", "unknown"),
            version=tree_data.get("version", "1.0.0"),
            is_default=tree_data.get("is_default", False)
        )

        return tree

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving decision tree: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving decision tree: {str(e)}"
        )


@router.get("/visualize-tree/{tree_id}")
async def visualize_decision_tree(tree_id: str, format: str = "svg"):
    """Visualize a decision tree."""
    # Since pyndantic_agents is removed, this feature is not available
    raise HTTPException(
        status_code=501,
        detail="Tree visualization is not available after removing pyndantic_agents module"
    )


@router.get("/capabilities", response_model=List[AgentCapability])
async def get_all_capabilities():
    """Get all agent capabilities."""
    try:
        # Load unique capabilities across all agents
        agent_metadata = await get_agents_metadata()

        # Collect all capabilities
        capability_dict = {}
        for agent in agent_metadata:
            for capability in agent.capabilities:
                if capability.name not in capability_dict:
                    capability_dict[capability.name] = capability

        # Return as list
        return list(capability_dict.values())

    except Exception as e:
        logger.error(f"Error retrieving agent capabilities: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent capabilities: {str(e)}"
        )


@router.get("/tags", response_model=List[str])
async def get_all_tags():
    """Get all agent tags."""
    try:
        # Load unique tags across all agents
        agent_metadata = await get_agents_metadata()

        # Collect all tags
        tags = set()
        for agent in agent_metadata:
            for tag in agent.tags:
                tags.add(tag)

        # Return as sorted list
        return sorted(list(tags))

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
        # Set timestamp if not provided
        if not feedback.timestamp:
            from datetime import datetime
            feedback.timestamp = datetime.now().isoformat()
            
        # Get database connection
        from main import get_mariadb_connection
        conn = get_mariadb_connection()
        cursor = conn.cursor()
        
        # Insert feedback into database
        cursor.execute(
            """
            INSERT INTO agent_feedback (
                agent_id, session_id, message_id, rating, 
                feedback_text, timestamp, context_used, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback.agent_id, feedback.session_id, 
                feedback.message_id or "", feedback.rating,
                feedback.feedback_text or "", feedback.timestamp,
                feedback.context_used, feedback.model or ""
            )
        )
        
        conn.commit()
        
        # Get the inserted id
        feedback_id = cursor.lastrowid
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id
        }
        
    except Exception as e:
        logger.error(f"Error submitting agent feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting agent feedback: {str(e)}"
        )


@router.get("/health", response_model=List[AgentHealthCheck])
async def get_agents_health():
    """Get health information for all agents."""
    try:
        # Get all agent types
        agent_types = await get_agent_types()
        
        # Get database connection
        from main import get_mariadb_connection
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get health status for all agents from the database
        health_checks = []
        
        for agent_id in agent_types.keys():
            try:
                # Query metrics from database if available
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as execution_count,
                        MAX(timestamp) as last_execution,
                        AVG(response_time_ms) as avg_response_time,
                        SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
                    FROM agent_executions
                    WHERE agent_id = ? AND timestamp > DATE_SUB(NOW(), INTERVAL 24 HOUR)
                    """,
                    (agent_id,)
                )
                
                result = cursor.fetchone()
                
                # Determine status based on metrics
                status = "online"
                if result and result["error_count"] > 0:
                    if result["error_count"] / max(1, result["execution_count"]) > 0.5:
                        status = "degraded"
                
                # Create health check
                health_checks.append(
                    AgentHealthCheck(
                        agent_id=agent_id,
                        status=status,
                        version="1.0.0", # This should ideally come from agent metadata
                        last_execution=result["last_execution"] if result and result["last_execution"] else None,
                        error_count=result["error_count"] if result else 0,
                        avg_response_time=result["avg_response_time"] if result and result["avg_response_time"] else None
                    )
                )
            except Exception as agent_e:
                logger.warning(f"Error getting health for agent {agent_id}: {str(agent_e)}")
                # Still include the agent in results, but mark as unknown status
                health_checks.append(
                    AgentHealthCheck(
                        agent_id=agent_id,
                        status="unknown",
                        version="1.0.0"
                    )
                )
        
        return health_checks
        
    except Exception as e:
        logger.error(f"Error retrieving agent health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent health: {str(e)}"
        )


@router.get("/health/{agent_id}", response_model=AgentHealthCheck)
async def get_agent_health(agent_id: str):
    """Get health information for a specific agent."""
    try:
        # Check if agent exists
        agent_types = await get_agent_types()
        if agent_id not in agent_types:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {agent_id}"
            )
        
        # Get database connection
        from main import get_mariadb_connection
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Query metrics from database if available
        cursor.execute(
            """
            SELECT 
                COUNT(*) as execution_count,
                MAX(timestamp) as last_execution,
                AVG(response_time_ms) as avg_response_time,
                SUM(CASE WHEN error = 1 THEN 1 ELSE 0 END) as error_count
            FROM agent_executions
            WHERE agent_id = ? AND timestamp > DATE_SUB(NOW(), INTERVAL 24 HOUR)
            """,
            (agent_id,)
        )
        
        result = cursor.fetchone()
        
        # Determine status based on metrics
        status = "online"
        if result and result["error_count"] > 0:
            if result["error_count"] / max(1, result["execution_count"]) > 0.5:
                status = "degraded"
        
        # Create health check
        return AgentHealthCheck(
            agent_id=agent_id,
            status=status,
            version="1.0.0", # This should ideally come from agent metadata
            last_execution=result["last_execution"] if result and result["last_execution"] else None,
            error_count=result["error_count"] if result else 0,
            avg_response_time=result["avg_response_time"] if result and result["avg_response_time"] else None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving agent health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving agent health: {str(e)}"
        )
