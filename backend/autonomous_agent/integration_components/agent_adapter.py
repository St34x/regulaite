"""
Adapter class that allows the autonomous agent to be used as a replacement for classic agents.
This provides compatibility with existing API endpoints while leveraging the autonomous agent architecture.
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import time
import uuid

from .workflow_engine import WorkflowEngine, WorkflowConfig, WorkflowState
from .graph_interface import GraphInterface
from ..processing_nodes.input_processor import InputProcessingNode
from ..processing_nodes.query_formulator import QueryFormulationNode
from ..processing_nodes.rag_retriever import RAGRetrievalNode
from ..processing_nodes.result_evaluator import ResultEvaluationNode
from ..processing_nodes.query_reformer import QueryReformulationNode
from ..processing_nodes.response_planner import ResponsePlanningNode
from ..processing_nodes.response_generator import ResponseGenerationNode
from ..processing_nodes.feedback_collector import FeedbackCollectionNode

logger = logging.getLogger(__name__)

class AutonomousAgentAdapter:
    """
    Adapter class that provides compatibility with the existing agent interface
    while using the autonomous agent's workflow engine under the hood.
    """
    
    def __init__(
        self,
        agent_type: str,
        graph_interface: Optional[GraphInterface] = None,
        embedding_service: Any = None,
        llm_client: Any = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the autonomous agent adapter.
        
        Args:
            agent_type: Type of agent to emulate (used for compatibility)
            graph_interface: Interface to the knowledge graph
            embedding_service: Service for generating embeddings
            llm_client: Client for language model interactions
            config: Configuration parameters for the workflow engine
        """
        self.agent_type = agent_type
        self.graph_interface = graph_interface
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        
        # Initialize workflow engine with appropriate configuration
        # Node-specific configs can be passed via self.config.get("node_configs")
        # and then used in _setup_nodes.
        self.node_configs = config.pop("node_configs", {}) if config else {}

        workflow_config = WorkflowConfig(
            max_reformulation_attempts=config.get("max_reformulation_attempts", 2) if config else 2,
            return_intermediate_results=config.get("return_intermediate_results", False) if config else False,
            save_workflow_history=config.get("save_workflow_history", True) if config else True,
            timeout_seconds=config.get("timeout_seconds", 60) if config else 60,
            enable_async_processing=config.get("enable_async_processing", True) if config else True,
            node_configs=self.node_configs
        )
        
        self.workflow_engine = WorkflowEngine(config=workflow_config)
        
        # Set up the default nodes based on agent type
        self._setup_nodes(agent_type)
        
        logger.info(f"AutonomousAgentAdapter initialized for agent type: {agent_type}")
        
    def _setup_nodes(self, agent_type: str):
        """
        Set up the appropriate processing nodes based on agent type.
        This allows customization of the workflow based on the emulated agent type.
        
        Args:
            agent_type: Type of agent to emulate
        """
        # Common nodes for all agent types
        self.workflow_engine.register_node(
            WorkflowState.PROCESSING_INPUT,
            InputProcessingNode(**self.node_configs.get(WorkflowState.PROCESSING_INPUT, {}))
        )
        
        # Example of agent-specific node configuration
        qfn_config = self.node_configs.get(WorkflowState.FORMULATING_QUERY, {})
        if agent_type == "research":
            qfn_config["custom_prompt_suffix"] = "Focus on academic sources and comprehensive literature review for this research query."
        elif agent_type == "regulatory":
            qfn_config["custom_prompt_suffix"] = "Ensure the query targets specific regulatory articles and compliance requirements."
        
        self.workflow_engine.register_node(
            WorkflowState.FORMULATING_QUERY,
            QueryFormulationNode(**qfn_config)
        )
        
        self.workflow_engine.register_node(
            WorkflowState.RETRIEVING_INFORMATION,
            RAGRetrievalNode(**self.node_configs.get(WorkflowState.RETRIEVING_INFORMATION, {}))
        )
        
        self.workflow_engine.register_node(
            WorkflowState.EVALUATING_RESULTS,
            ResultEvaluationNode(**self.node_configs.get(WorkflowState.EVALUATING_RESULTS, {}))
        )
        
        self.workflow_engine.register_node(
            WorkflowState.REFORMULATING_QUERY,
            QueryReformulationNode(**self.node_configs.get(WorkflowState.REFORMULATING_QUERY, {}))
        )
        
        self.workflow_engine.register_node(
            WorkflowState.PLANNING_RESPONSE,
            ResponsePlanningNode(**self.node_configs.get(WorkflowState.PLANNING_RESPONSE, {}))
        )
        
        self.workflow_engine.register_node(
            WorkflowState.GENERATING_RESPONSE,
            ResponseGenerationNode(**self.node_configs.get(WorkflowState.GENERATING_RESPONSE, {}))
        )
        
        self.workflow_engine.register_node(
            WorkflowState.COLLECTING_FEEDBACK,
            FeedbackCollectionNode(**self.node_configs.get(WorkflowState.COLLECTING_FEEDBACK, {}))
        )
        
        # TODO: Agent type-specific node customization could be added here
        # For example, different types of agents might use different versions of certain nodes
        # or might have different configurations for the same node types
    
    async def process(self, user_input: str, session_id: str, user_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a user query using the autonomous agent workflow.
        This method follows the same interface as the classic agents for compatibility.
        
        Args:
            user_input: User query or input text
            session_id: Session ID for the conversation
            user_id: Optional user ID
            **kwargs: Additional parameters that may be passed to the agent
            
        Returns:
            Dict containing the agent's response and metadata
        """
        start_time = time.time()
        
        # Extract relevant parameters from kwargs
        model = kwargs.get("model", "gpt-4")
        include_context = kwargs.get("include_context", True)
        
        # Special handling for context if a specific query was provided
        context_query = kwargs.get("context_query", None)
        if context_query:
            # Use the provided context query instead of the user input for context retrieval
            logger.info(f"Using specified context query: {context_query}")
        
        # Prepare initial context for the workflow
        initial_context = {
            "model": model,
            "include_context": include_context,
            "context_query": context_query,
            "agent_type": self.agent_type,
            **kwargs  # Include any other parameters
        }
        
        try:
            # Execute the workflow
            workflow_result = await self.workflow_engine.execute_workflow(
                user_input=user_input,
                graph_interface=self.graph_interface,
                embedding_service=self.embedding_service,
                llm_client=self.llm_client,
                session_id=session_id,
                user_id=user_id,
                initial_context=initial_context
            )
            
            # Extract response content, handling both string and dict formats
            if isinstance(workflow_result.get("response"), str):
                response_content = workflow_result.get("response", "")
            else:
                response_content = workflow_result.get("response", {}).get("content", "")
            
            # Extract sources/citations if available
            sources = workflow_result.get("metadata", {}).get("sources", [])
            if not sources and isinstance(workflow_result.get("response"), dict):
                # Backward compatibility with old format
                sources = workflow_result.get("response", {}).get("sources", [])
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Format the response to match the expected format from classic agents
            formatted_response = {
                "response": response_content,
                "source_documents": sources,
                "execution_time": execution_time,
                "agent_type": self.agent_type,
                "model": model,
                "session_id": session_id,
                "workflow_id": workflow_result.get("workflow_id", str(uuid.uuid4())),
                "timestamp": datetime.utcnow().isoformat(),
                "context_used": include_context,
                # Additional metadata about the processing steps
                "processing_details": {
                    "reformulations": workflow_result.get("reformulation_count", 0),
                    "workflow_states": [step.get("state") for step in workflow_result.get("steps", [])]
                }
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing with autonomous agent: {str(e)}", exc_info=True)
            # Return error response in the same format expected from classic agents
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "source_documents": [],
                "execution_time": time.time() - start_time,
                "agent_type": self.agent_type,
                "model": model,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def process_with_streaming(self, user_input: str, session_id: str, user_id: Optional[str] = None, **kwargs):
        """
        Process a user query with streaming response.
        
        Args:
            user_input: User query or input text
            session_id: Session ID for the conversation
            user_id: Optional user ID
            **kwargs: Additional parameters
            
        Returns:
            An async generator that yields response chunks
        """
        # For now, process normally and then simulate streaming
        # In a real implementation, this would tap into the streaming capabilities of the workflow engine
        result = await self.process(user_input, session_id, user_id, **kwargs)
        
        # Simulate streaming by yielding the response in chunks
        response_text = result.get("response", "")
        chunk_size = 10  # Number of characters per chunk
        
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield {
                "text": chunk,
                "done": i + chunk_size >= len(response_text),
                "sources": result.get("source_documents", []) if i + chunk_size >= len(response_text) else None
            }
            await asyncio.sleep(0.05)  # Small delay to simulate streaming 