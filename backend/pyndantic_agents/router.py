"""
FastAPI router for agent endpoints.
"""
import logging
import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body, Response
from pydantic import BaseModel, Field
import os

from .base_agent import AgentInput, AgentOutput
from .rag_agent import RAGAgent, QueryUnderstandingOutput
from .tree_reasoning import TreeReasoningAgent
from .decision_trees import get_tree, get_available_trees
from .cybersecurity_agents import (
    VulnerabilityAssessmentAgent, VulnerabilityAssessmentOutput,
    ComplianceMappingAgent, ComplianceMappingOutput,
    ThreatModelingAgent, ThreatModelOutput
)
from .agent_factory import create_agent, get_agent_types
from .dynamic_decision_trees import DynamicTreeAgent, DynamicTreeGenerator
from .tree_visualizer import generate_tree_visualization

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
class AgentRequest(BaseModel):
    """Request to an agent"""
    query: str = Field(..., description="User query or request")
    agent_type: str = Field(..., description="Type of agent to use")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")
    tree_template_id: Optional[str] = Field(None, description="ID of decision tree to use")
    use_tree_reasoning: bool = Field(False, description="Whether to use tree-based reasoning")
    use_dynamic_tree: bool = Field(False, description="Whether to dynamically generate a decision tree")
    cache_dynamic_tree: bool = Field(True, description="Whether to cache dynamically generated trees")

class AgentResponse(BaseModel):
    """Response from an agent"""
    response: str = Field(..., description="Agent response")
    context_used: Optional[List[Dict[str, Any]]] = Field(None, description="Context used in the response")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning process")
    agent_type: str = Field(..., description="Type of agent used")
    tree_id: Optional[str] = Field(None, description="ID of decision tree used")
    decision_path: Optional[List[Dict[str, Any]]] = Field(None, description="Decision path if tree reasoning was used")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional data returned by the agent")

class VulnerabilityRequest(BaseModel):
    """Request for vulnerability assessment"""
    query: str = Field(..., description="Vulnerability assessment request")
    systems: Optional[List[str]] = Field(None, description="Systems to assess")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class ComplianceMappingRequest(BaseModel):
    """Request for compliance mapping"""
    query: str = Field(..., description="Compliance mapping request")
    source_framework: Optional[str] = Field(None, description="Source compliance framework")
    target_framework: Optional[str] = Field(None, description="Target compliance framework")
    specific_controls: Optional[List[str]] = Field(None, description="Specific controls to map")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class ThreatModelRequest(BaseModel):
    """Request for threat modeling"""
    query: str = Field(..., description="Threat modeling request")
    system_name: Optional[str] = Field(None, description="Name of the system to model")
    system_components: Optional[List[str]] = Field(None, description="Components of the system")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

# Dependency to get the RAG system instance
async def get_rag_system():
    """Get the RAG system instance from the application state"""
    from main import rag_system
    return rag_system

# Helper function to convert agent output to standard response
def convert_to_response(output: AgentOutput, agent_type: str, tree_id: Optional[str] = None, decision_path: Optional[List[Dict[str, Any]]] = None) -> AgentResponse:
    """Convert agent output to standard API response"""
    return AgentResponse(
        response=output.response,
        context_used=output.context_used,
        confidence=output.confidence,
        reasoning=output.reasoning,
        agent_type=agent_type,
        tree_id=tree_id,
        decision_path=decision_path,
        additional_data=output.additional_data
    )

@router.get("/types")
async def list_agent_types():
    """
    List all available agent types.

    Returns:
        Dictionary of agent type identifiers to descriptions
    """
    return get_agent_types()

@router.get("/trees")
async def list_decision_trees():
    """
    List all available decision trees.

    Returns:
        Dictionary of decision tree metadata
    """
    return get_available_trees()

@router.post("/process", response_model=AgentResponse)
async def process_with_agent(
    request: AgentRequest,
    rag_system = Depends(get_rag_system)
):
    """
    Process a request with an agent.

    Args:
        request: Agent request

    Returns:
        Agent response
    """
    try:
        logger.info(f"Processing request with agent type: {request.agent_type}")

        # Create the appropriate agent using the factory
        try:
            agent = create_agent(
                agent_type=request.agent_type,
                rag_system=rag_system,
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                model=request.parameters.get("model", "gpt-4") if request.parameters else "gpt-4"
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Create input data
        input_data = AgentInput(
            query=request.query,
            parameters=request.parameters
        )

        # Process with agent
        if request.use_dynamic_tree:
            # Use dynamic tree-based reasoning
            dynamic_agent = DynamicTreeAgent(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                model=request.parameters.get("model", "gpt-4") if request.parameters else "gpt-4",
                cache_trees=request.cache_dynamic_tree
            )

            # First get the context with our agent if not a RAG agent
            if request.agent_type.lower() != "rag":
                agent_output = await agent.process(input_data)
                context = agent_output.context_used
            else:
                # For RAG agent, we'll get context directly in the dynamic agent
                rag_agent = RAGAgent(
                    rag_system=rag_system,
                    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                    model=request.parameters.get("model", "gpt-4") if request.parameters else "gpt-4"
                )

                # Process to get context
                understanding = await rag_agent.understand_query(request.query)
                context = await rag_agent.retrieve_context(
                    query=request.query,
                    understanding=understanding,
                    top_k=request.parameters.get("top_k", 5) if request.parameters else 5
                )

            # Process with dynamic tree agent
            tree_result = await dynamic_agent.process(
                query=request.query,
                context=context
            )

            # Convert to standard response format
            return AgentResponse(
                response=tree_result["response"],
                context_used=context,
                confidence=tree_result.get("confidence", 0.7),
                reasoning="Dynamic decision tree reasoning",
                agent_type=f"{request.agent_type}_dynamic_tree",
                tree_id=tree_result.get("tree_id"),
                decision_path=tree_result.get("decision_path", []),
                additional_data={
                    "tree_name": tree_result.get("tree_name"),
                    "tree_generated": tree_result.get("tree_generated", True),
                    "tree_generation_time": tree_result.get("tree_generation_time")
                }
            )
        elif not request.use_tree_reasoning:
            # Use standard agent processing
            agent_output = await agent.process(input_data)

            # Convert to API response
            return convert_to_response(agent_output, request.agent_type)
        else:
            # Use tree-based reasoning
            if not request.tree_template_id:
                raise HTTPException(status_code=400, detail="Tree template ID is required for tree reasoning")

            # Get the decision tree
            tree = get_tree(request.tree_template_id)
            if not tree:
                raise HTTPException(status_code=404, detail=f"Decision tree template not found: {request.tree_template_id}")

            # First get the context with our agent
            agent_output = await agent.process(input_data)
            context = agent_output.context_used

            # Then use the tree reasoning agent
            tree_agent = TreeReasoningAgent(
                tree=tree,
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                model=request.parameters.get("model", "gpt-4") if request.parameters else "gpt-4"
            )

            tree_result = await tree_agent.process(
                query=request.query,
                context=context
            )

            # Combine the results
            return AgentResponse(
                response=tree_result["response"],
                context_used=context,
                confidence=agent_output.confidence,
                reasoning=agent_output.reasoning,
                agent_type=request.agent_type,
                tree_id=request.tree_template_id,
                decision_path=tree_result["decision_path"],
                additional_data={
                    "agent_output": agent_output.additional_data,
                    "tree_result": tree_result
                }
            )

    except Exception as e:
        logger.error(f"Error processing with agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/vulnerability-assessment", response_model=AgentResponse)
async def vulnerability_assessment(
    request: VulnerabilityRequest,
    rag_system = Depends(get_rag_system)
):
    """
    Perform a vulnerability assessment.

    Args:
        request: Vulnerability assessment request

    Returns:
        Vulnerability assessment results
    """
    try:
        # Create specialized vulnerability assessment agent
        agent = VulnerabilityAssessmentAgent(
            rag_system=rag_system,
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        # Prepare parameters with systems list if provided
        parameters = request.parameters or {}
        if request.systems:
            parameters["systems"] = request.systems

        # Process the request
        input_data = AgentInput(
            query=request.query,
            parameters=parameters
        )

        output = await agent.process(input_data)

        # Return standard format
        return convert_to_response(output, "vulnerability_assessment")

    except Exception as e:
        logger.error(f"Error in vulnerability assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing vulnerability assessment: {str(e)}")

@router.post("/compliance-mapping", response_model=AgentResponse)
async def compliance_mapping(
    request: ComplianceMappingRequest,
    rag_system = Depends(get_rag_system)
):
    """
    Perform compliance mapping between frameworks.

    Args:
        request: Compliance mapping request

    Returns:
        Compliance mapping results
    """
    try:
        # Create specialized compliance mapping agent
        agent = ComplianceMappingAgent(
            rag_system=rag_system,
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        # Prepare parameters
        parameters = request.parameters or {}
        if request.source_framework:
            parameters["source_framework"] = request.source_framework
        if request.target_framework:
            parameters["target_framework"] = request.target_framework
        if request.specific_controls:
            parameters["specific_controls"] = request.specific_controls

        # Process the request
        input_data = AgentInput(
            query=request.query,
            parameters=parameters
        )

        output = await agent.process(input_data)

        # Return standard format
        return convert_to_response(output, "compliance_mapping")

    except Exception as e:
        logger.error(f"Error in compliance mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing compliance mapping: {str(e)}")

@router.post("/threat-modeling", response_model=AgentResponse)
async def threat_modeling(
    request: ThreatModelRequest,
    rag_system = Depends(get_rag_system)
):
    """
    Create a threat model for a system.

    Args:
        request: Threat modeling request

    Returns:
        Threat model results
    """
    try:
        # Create specialized threat modeling agent
        agent = ThreatModelingAgent(
            rag_system=rag_system,
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        # Prepare parameters
        parameters = request.parameters or {}
        if request.system_name:
            parameters["system_name"] = request.system_name
        if request.system_components:
            parameters["system_components"] = request.system_components

        # Process the request
        input_data = AgentInput(
            query=request.query,
            parameters=parameters
        )

        output = await agent.process(input_data)

        # Return standard format
        return convert_to_response(output, "threat_modeling")

    except Exception as e:
        logger.error(f"Error in threat modeling: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing threat model: {str(e)}")

@router.post("/understand")
async def understand_query(
    query: str = Body(..., embed=True),
    rag_system = Depends(get_rag_system)
):
    """
    Understand a query and extract entities, intent, etc.

    Args:
        query: Query to understand

    Returns:
        Query understanding
    """
    try:
        agent = RAGAgent(
            rag_system=rag_system,
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        understanding = await agent.understand_query(query)
        return understanding

    except Exception as e:
        logger.error(f"Error understanding query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error understanding query: {str(e)}")

@router.post("/retrieve-context")
async def retrieve_context(
    query: str = Body(...),
    limit: int = Body(5),
    rag_system = Depends(get_rag_system)
):
    """
    Retrieve context for a query.

    Args:
        query: Query to retrieve context for
        limit: Maximum number of results

    Returns:
        List of context items
    """
    try:
        agent = RAGAgent(
            rag_system=rag_system,
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        # First understand the query
        understanding = await agent.understand_query(query)

        # Then retrieve context
        context = await agent.retrieve_context(
            query=query,
            understanding=understanding,
            top_k=limit
        )

        return {
            "context": context,
            "understanding": understanding
        }

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

@router.post("/generate-dynamic-tree")
async def generate_dynamic_tree(
    query: str = Body(...),
    context: Optional[List[Dict[str, Any]]] = Body(None),
    rag_system = Depends(get_rag_system)
):
    """
    Generate a dynamic decision tree based on a query.

    Args:
        query: Query to generate tree for
        context: Optional context to use for tree generation

    Returns:
        Generated tree structure
    """
    try:
        # If context not provided, retrieve it
        if not context:
            rag_agent = RAGAgent(
                rag_system=rag_system,
                openai_api_key=os.getenv("OPENAI_API_KEY", "")
            )

            # Process to get context
            understanding = await rag_agent.understand_query(query)
            context = await rag_agent.retrieve_context(
                query=query,
                understanding=understanding,
                top_k=5
            )

        # Create tree generator
        generator = DynamicTreeGenerator(
            openai_api_key=os.getenv("OPENAI_API_KEY", "")
        )

        # Generate tree
        tree = await generator.generate_tree(query, context)

        # Convert to dictionary for API response
        return {
            "tree_id": tree.id,
            "tree_name": tree.name,
            "root_node_id": tree.root_node_id,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "description": node.description,
                    "is_leaf": node.is_leaf,
                    "is_probabilistic": node.is_probabilistic,
                    "children": node.children
                }
                for node_id, node in tree.nodes.items()
            },
            "domain": tree.domain,
            "version": tree.version
        }

    except Exception as e:
        logger.error(f"Error generating dynamic tree: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating dynamic tree: {str(e)}")

@router.post("/visualize-tree")
async def visualize_tree(
    tree_id: str = Body(...),
    format: str = Body("svg")
):
    """
    Generate a visualization of a decision tree.

    Args:
        tree_id: ID of the tree to visualize
        format: Output format (svg, png, dot)

    Returns:
        Tree visualization
    """
    try:
        # Get the tree
        tree = get_tree(tree_id)
        if not tree:
            raise HTTPException(status_code=404, detail=f"Tree not found: {tree_id}")

        # Generate visualization
        graph_data = generate_tree_visualization(tree, output_format=format)

        if format == "svg":
            return Response(content=graph_data, media_type="image/svg+xml")
        elif format == "png":
            return Response(content=graph_data, media_type="image/png")
        else:
            return {"dot_data": graph_data}

    except Exception as e:
        logger.error(f"Error visualizing tree: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error visualizing tree: {str(e)}")
