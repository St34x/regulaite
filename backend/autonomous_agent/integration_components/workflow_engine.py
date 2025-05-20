import logging
import asyncio
from typing import Dict, Any, List, Type, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import os
import uuid

from ..processing_nodes.base_node import BaseProcessingNode, ProcessingStepResult
from ..processing_nodes.input_processor import InputProcessingNode, InputProcessorOutput
from ..processing_nodes.query_formulator import QueryFormulationNode, QueryFormulationOutput, QueryFormulatorInput
from ..processing_nodes.rag_retriever import RAGRetrievalNode, RAGRetrievalOutput
from ..processing_nodes.result_evaluator import ResultEvaluationNode, ResultEvaluationOutput
from ..processing_nodes.query_reformer import QueryReformulationNode, QueryReformerOutput
from ..processing_nodes.response_planner import ResponsePlanningNode, ResponsePlannerOutput, ResponsePlannerInput
from ..processing_nodes.response_generator import ResponseGenerationNode, GeneratedResponse, ResponseGeneratorInput
from ..processing_nodes.feedback_collector import FeedbackCollectionNode, FeedbackCollectorOutput
from .graph_interface import GraphInterface

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    INITIALIZED = "initialized"
    PROCESSING_INPUT = "processing_input"
    FORMULATING_QUERY = "formulating_query"
    RETRIEVING_INFORMATION = "retrieving_information"
    EVALUATING_RESULTS = "evaluating_results"
    REFORMULATING_QUERY = "reformulating_query"
    PLANNING_RESPONSE = "planning_response"
    GENERATING_RESPONSE = "generating_response"
    COLLECTING_FEEDBACK = "collecting_feedback"
    COMPLETED = "completed"
    ERROR = "error"

class WorkflowStepResult(BaseModel):
    state: WorkflowState
    step_name: str
    output: Any
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metrics: Dict[str, Any] = Field(default_factory=dict)

class WorkflowHistory(BaseModel):
    workflow_id: str
    session_id: str
    user_id: Optional[str] = None
    steps: List[WorkflowStepResult] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    overall_metrics: Dict[str, Any] = Field(default_factory=dict)

class WorkflowConfig(BaseModel):
    """Configuration for the WorkflowEngine"""
    max_reformulation_attempts: int = Field(default=2, description="Maximum number of query reformulation attempts")
    return_intermediate_results: bool = Field(default=False, description="Whether to return intermediate results")
    save_workflow_history: bool = Field(default=True, description="Whether to save workflow history")
    timeout_seconds: Optional[int] = Field(default=60, description="Timeout for the entire workflow in seconds")
    enable_async_processing: bool = Field(default=True, description="Whether to enable async processing")
    node_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Configuration for individual nodes")

class WorkflowEngine:
    """
    Orchestrates the flow between different processing nodes in the autonomous agent workflow.
    This engine manages the execution sequence, handles state transitions, and maintains workflow context.
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.state = WorkflowState.INITIALIZED
        self.nodes: Dict[WorkflowState, BaseProcessingNode] = {}
        self.context: Dict[str, Any] = {}
        self.history: Optional[WorkflowHistory] = None
        self._setup_default_nodes()
        logger.info(f"WorkflowEngine initialized with config: {self.config}")

    def _setup_default_nodes(self):
        """Set up the default processing nodes for the workflow."""
        # This will be implemented to initialize each node
        # For now, we'll just log that this needs implementation
        logger.info("Default nodes setup method called - needs implementation with actual node classes")
        # The actual implementation will look like:
        # self.nodes[WorkflowState.PROCESSING_INPUT] = InputProcessingNode(...)
        # self.nodes[WorkflowState.FORMULATING_QUERY] = QueryFormulationNode(...)
        # etc.

    def register_node(self, state: WorkflowState, node: BaseProcessingNode):
        """Register a custom processing node for a specific workflow state."""
        self.nodes[state] = node
        logger.info(f"Registered custom node {node.__class__.__name__} for state {state.value}")
        return self

    def initialize_workflow(self, session_id: str, user_id: Optional[str] = None) -> str:
        """
        Initialize a new workflow with a unique ID and prepare the context.
        
        Args:
            session_id: ID of the user session
            user_id: Optional ID of the user
            
        Returns:
            The generated workflow ID
        """
        workflow_id = str(uuid.uuid4())
        self.history = WorkflowHistory(
            workflow_id=workflow_id,
            session_id=session_id,
            user_id=user_id
        )
        
        # Set up the base context that will be passed to all nodes
        self.context = {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "user_id": user_id,
            "start_time": datetime.utcnow().isoformat(),
            "reformulation_count": 0,
            # These would typically be injected by your application
            "graph_interface": None,  # Will be set when execute_workflow is called
            "embedding_service": None,  # Will be set when execute_workflow is called
            "llm_client": None,  # Will be set when execute_workflow is called
        }
        
        self.state = WorkflowState.INITIALIZED
        logger.info(f"Initialized workflow {workflow_id} for session {session_id}")
        return workflow_id

    async def _execute_processing_step(
        self, 
        state: WorkflowState, 
        input_data: Any, 
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single processing step of the workflow."""
        await self._transition_to(state)
        processing_node = self.nodes.get(state)
        
        if not processing_node:
            logger.error(f"[WorkflowEngine] Processing node not found for state: {state}")
            raise ValueError(f"Required processing node not found for state: {state}")
        
        result = await processing_node.execute(input_data, context)
        self._record_step(state, processing_node.get_name(), result)
        logger.info(f"[WorkflowEngine] {state} step completed")
        
        return result
        
    async def execute_workflow(
        self, 
        user_input: Any,
        graph_interface: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the workflow for a given query.
        
        This method supports two parameter formats for backward compatibility:
        1. The original format with graph_interface, embedding_service, etc.
        2. The new format with workflow_id and context
        
        Args:
            user_input: The user input to process
            graph_interface: Interface to the knowledge graph
            embedding_service: Service for generating embeddings
            llm_client: Client for language model interactions
            session_id: ID of the current session
            user_id: Optional ID of the user
            initial_context: Optional additional context to include
            workflow_id: ID of the workflow (new format)
            context: Context dictionary (new format)
            
        Returns:
            The workflow results including the final response
        """
        # Handle the new format if workflow_id is provided
        if workflow_id:
            if not hasattr(self, 'active_workflows'):
                self.active_workflows = {}
                
            if workflow_id not in self.active_workflows:
                logger.error(f"Workflow {workflow_id} not found")
                raise ValueError(f"Workflow {workflow_id} not found")
                
            self.context = context or {}
            self.context["workflow_id"] = workflow_id
        # Handle the original format
        else:
            # Initialize a new workflow
            workflow_id = self.initialize_workflow(session_id or str(uuid.uuid4()), user_id)
            
            # Update context with required services and initial context
            self.context.update({
                "graph_interface": graph_interface,
                "embedding_service": embedding_service,
                "llm_client": llm_client,
                "user_input": user_input,
                "workflow_id": workflow_id
            })
            
            # Add initial context if provided
            if initial_context:
                self.context.update(initial_context)
        
        # Ensure all required nodes exist
        self._validate_required_nodes()
        
        # Begin the workflow
        try:
            # 1. Input Processing
            logger.info(f"[WorkflowEngine] Starting input processing for workflow {workflow_id}")
            input_result = await self._execute_processing_step(
                WorkflowState.PROCESSING_INPUT,
                user_input,
                self.context
            )
            
            # 2. Query Formulation
            logger.info(f"[WorkflowEngine] Formulating query for workflow {workflow_id}")
            query_result = await self._execute_processing_step(
                WorkflowState.FORMULATING_QUERY,
                input_result,
                self.context
            )
            
            # 3. RAG Retrieval
            logger.info(f"[WorkflowEngine] Retrieving information for workflow {workflow_id}")
            retrieval_result = await self._execute_processing_step(
                WorkflowState.RETRIEVING_INFORMATION,
                query_result,
                self.context
            )
            
            # 4. Result Evaluation
            logger.info(f"[WorkflowEngine] Evaluating results for workflow {workflow_id}")
            evaluation_result = await self._execute_processing_step(
                WorkflowState.EVALUATING_RESULTS,
                retrieval_result,
                self.context
            )
            
            # 5. Query Reformulation (if needed)
            if hasattr(evaluation_result, 'needs_reformulation') and evaluation_result.needs_reformulation:
                logger.info(f"[WorkflowEngine] Reformulating query for workflow {workflow_id}")
                
                # Track reformulation attempts
                self.context['reformulation_count'] = self.context.get('reformulation_count', 0) + 1
                
                # Check if we've exceeded the maximum reformulation attempts
                if self.context['reformulation_count'] > self.config.max_reformulation_attempts:
                    logger.warning(f"[WorkflowEngine] Maximum reformulation attempts exceeded for workflow {workflow_id}")
                else:
                    # Execute query reformulation
                    reformulation_result = await self._execute_processing_step(
                        WorkflowState.REFORMULATING_QUERY,
                        {
                            'original_query': query_result,
                            'retrieval_result': retrieval_result,
                            'evaluation_result': evaluation_result
                        },
                        self.context
                    )
                    
                    # Retry retrieval with the reformulated query
                    logger.info(f"[WorkflowEngine] Retrying retrieval with reformulated query for workflow {workflow_id}")
                    retrieval_result = await self._execute_processing_step(
                        WorkflowState.RETRIEVING_INFORMATION,
                        reformulation_result,
                        self.context
                    )
                    
                    # Re-evaluate the results
                    logger.info(f"[WorkflowEngine] Re-evaluating results for workflow {workflow_id}")
                    evaluation_result = await self._execute_processing_step(
                        WorkflowState.EVALUATING_RESULTS,
                        retrieval_result,
                        self.context
                    )
            
            # 6. Response Planning
            logger.info(f"[WorkflowEngine] Planning response for workflow {workflow_id}")
            
            # Prepare input for response planning
            planning_input = {
                'query': query_result,
                'retrieval_result': retrieval_result,
                'evaluation_result': evaluation_result,
                # Include the original input processing result for context
                'input_result': input_result
            }
            
            response_plan = await self._execute_processing_step(
                WorkflowState.PLANNING_RESPONSE,
                planning_input,
                self.context
            )
            
            # 7. Response Generation
            logger.info(f"[WorkflowEngine] Generating response for workflow {workflow_id}")
            
            # Prepare input for response generation
            generation_input = {
                'response_plan': response_plan,
                'query': query_result,
                'retrieved_documents': retrieval_result,
                # Include original user input for context
                'user_input': user_input
            }
            
            response = await self._execute_processing_step(
                WorkflowState.GENERATING_RESPONSE,
                generation_input,
                self.context
            )
            
            # Update workflow state
            await self._transition_to(WorkflowState.COMPLETED)
            
            # Record completion metrics
            self._record_completion_metrics(response)
            
            # Return the final result
            return {
                'workflow_id': workflow_id,
                'status': 'completed',
                'response': response.content if hasattr(response, 'content') else response,
                'metadata': {
                    'query': query_result.query_text if hasattr(query_result, 'query_text') else str(query_result),
                    'reformulation_count': self.context.get('reformulation_count', 0),
                    'document_count': len(retrieval_result.documents) if hasattr(retrieval_result, 'documents') else 0,
                    'sources': response.sources if hasattr(response, 'sources') else [],
                    'execution_time': self._calculate_execution_time(),
                    'model': self.context.get('model', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"[WorkflowEngine] Error executing workflow: {e}", exc_info=True)
            await self._transition_to(WorkflowState.ERROR)
            
            # Return error response
            return {
                'workflow_id': workflow_id,
                'status': 'error',
                'error': str(e),
                'response': f"I'm sorry, but I encountered an error processing your request: {str(e)}",
                'metadata': {
                    'execution_time': self._calculate_execution_time(),
                    'state': self.state.value
                }
            }
    
    def _validate_required_nodes(self):
        """Validate that all required processing nodes exist."""
        required_states = [
            WorkflowState.PROCESSING_INPUT,
            WorkflowState.FORMULATING_QUERY,
            WorkflowState.RETRIEVING_INFORMATION,
            WorkflowState.EVALUATING_RESULTS,
            WorkflowState.PLANNING_RESPONSE,
            WorkflowState.GENERATING_RESPONSE
        ]
        
        missing_nodes = []
        for state in required_states:
            if state not in self.nodes:
                missing_nodes.append(state.value)
        
        if missing_nodes:
            logger.error(f"Missing required processing nodes: {', '.join(missing_nodes)}")
            raise ValueError(f"Missing required processing nodes: {', '.join(missing_nodes)}")
    
    def _record_completion_metrics(self, response):
        """Record metrics upon workflow completion."""
        if self.history:
            self.history.overall_metrics.update({
                'completion_time': datetime.utcnow().isoformat(),
                'total_steps': len(self.history.steps),
                'reformulation_count': self.context.get('reformulation_count', 0),
                'response_length': len(str(response)) if response else 0
            })
    
    def _calculate_execution_time(self):
        """Calculate the total execution time of the workflow."""
        if 'start_time' in self.context:
            start_time = self.context['start_time']
            if isinstance(start_time, str):
                try:
                    start_datetime = datetime.fromisoformat(start_time)
                    execution_time = (datetime.utcnow() - start_datetime).total_seconds()
                    return execution_time
                except (ValueError, TypeError):
                    pass
        return None

    async def process_feedback(
        self,
        feedback_data: Any,
        workflow_id: str,
        response_id: str
    ) -> Dict[str, Any]:
        """
        Process user feedback for a specific response.
        
        Args:
            feedback_data: The feedback provided by the user
            workflow_id: ID of the workflow that generated the response
            response_id: ID of the response
            
        Returns:
            Result of feedback processing
        """
        try:
            # Set the state to feedback collection
            self.state = WorkflowState.COLLECTING_FEEDBACK
            
            # Feedback processing logic will be implemented here
            logger.info(f"Feedback processing not yet implemented for response {response_id}")
            
            # Return a dummy result for now
            return {
                "status": "success",
                "feedback_processed": True,
                "message": "Feedback processing is under construction."
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "feedback_processed": False
            }

    def _record_step(self, state: WorkflowState, step_name: str, output: Any):
        """Record a workflow step in the history."""
        if not self.history:
            return
            
        # Extract metrics if available
        metrics = {}
        if hasattr(output, "metrics"):
            metrics = output.metrics
        elif isinstance(output, dict) and "metrics" in output:
            metrics = output["metrics"]
            
        # Create and add the step result
        step_result = WorkflowStepResult(
            state=state,
            step_name=step_name,
            output=output if not isinstance(output, BaseModel) else output.model_dump(),
            timestamp=datetime.utcnow().isoformat(),
            metrics=metrics
        )
        
        self.history.steps.append(step_result)
        self.history.updated_at = datetime.utcnow().isoformat()

    async def _transition_to(self, new_state: WorkflowState):
        """Handle state transition logic."""
        logger.info(f"Workflow {self.context.get('workflow_id', 'unknown')} transitioning from {self.state.value} to {new_state.value}")
        self.state = new_state
        return new_state

# Example of how the engine might be instantiated and used:
async def main_workflow_example():
    """Example usage of the WorkflowEngine with proper instantiation of required services"""
    try:
        # Initialize required services
        # 1. Create and connect to graph database
        graph_interface = GraphInterface(
            uri=os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        await graph_interface.connect()
        
        # 2. Initialize LLM client
        from llm_services.llm_client import LLMClient
        llm_client = LLMClient()  # Configure with appropriate parameters
        
        # 3. Initialize embedding service
        from .embedding_integration import EmbeddingIntegrationLayer
        embedding_service = EmbeddingIntegrationLayer()  # Initialize with appropriate params
        
        # 4. Set up the workflow engine with appropriate configuration
        workflow_config = WorkflowConfig(
            max_reformulation_attempts=2,
            return_intermediate_results=True,  # Useful for testing
            save_workflow_history=True
        )
        engine = WorkflowEngine(config=workflow_config)
        
        # 5. Set up the required processing nodes
        engine.register_node(WorkflowState.PROCESSING_INPUT, InputProcessingNode())
        engine.register_node(WorkflowState.FORMULATING_QUERY, QueryFormulationNode())
        engine.register_node(WorkflowState.RETRIEVING_INFORMATION, RAGRetrievalNode())
        engine.register_node(WorkflowState.EVALUATING_RESULTS, ResultEvaluationNode())
        engine.register_node(WorkflowState.REFORMULATING_QUERY, QueryReformulationNode())
        engine.register_node(WorkflowState.PLANNING_RESPONSE, ResponsePlanningNode())
        engine.register_node(WorkflowState.GENERATING_RESPONSE, ResponseGenerationNode())
        engine.register_node(WorkflowState.COLLECTING_FEEDBACK, FeedbackCollectionNode())
        
        # 6. Execute the workflow with a sample query
        user_input = "What are the key regulatory requirements for financial services in the EU?"
        session_id = "test_session_123"
        user_id = "test_user_456"
        
        result = await engine.execute_workflow(
            user_input=user_input,
            graph_interface=graph_interface,
            embedding_service=embedding_service,
            llm_client=llm_client,
            session_id=session_id,
            user_id=user_id,
            initial_context={"domain": "financial_regulations"}
        )
        
        # 7. Process the results
        if "error" in result:
            print(f"Workflow Error: {result['error']}")
        else:
            print("\n=== Workflow Result ===")
            print(f"Response: {result['response'][:500]}...")
            print(f"Query reformulations: {result['metadata']['reformulation_count']}")
            print(f"Sources used: {len(result['metadata']['sources'])}")
            
            # 8. Process feedback if desired
            if result.get("response_id"):
                feedback_data = {
                    "rating": 4,
                    "comment": "Good response but could include more detail on MiFID II.",
                    "is_helpful": True
                }
                
                feedback_result = await engine.process_feedback(
                    feedback_data=feedback_data,
                    workflow_id=result["workflow_id"],
                    response_id=result["response_id"]
                )
                print(f"Feedback processing status: {feedback_result['status']}")
    
    except Exception as e:
        import traceback
        print(f"Error running workflow example: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up connections
        await graph_interface.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_workflow_example()) 