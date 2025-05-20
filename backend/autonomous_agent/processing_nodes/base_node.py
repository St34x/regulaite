import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ProcessingStepResult(BaseModel):
    """Base class for all processing step results."""
    step_name: str
    status: str
    metrics: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True

class BaseProcessingNode(ABC):
    """
    Base class for all processing nodes in the autonomous agent workflow.
    
    Each concrete node implementation should:
    1. Inherit from this class
    2. Implement execute() method
    3. Optionally override other methods like validate_input()
    """
    
    def __init__(self, node_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processing node with optional configuration.
        
        Args:
            node_config: Dictionary of configuration parameters for this node
        """
        self.config = node_config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initialized {self.name} with config: {self.config}")
    
    def get_name(self) -> str:
        """Get the name of this processing node."""
        return self.name
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate that the input data is appropriate for this node.
        
        Args:
            input_data: The input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Base implementation just returns True
        # Subclasses should override with specific validation logic
        return True
    
    @abstractmethod
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """
        Execute the processing logic of this node.
        
        Args:
            input_data: The input data for this node
            context: The shared context for the workflow
            
        Returns:
            Output data from this node
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics from this node's execution.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        logger.info(f"Metrics for {self.name}: {metrics}")
        
    def handle_error(self, error: Exception, input_data: Any, context: Dict[str, Any]) -> Any:
        """
        Handle errors that occur during execution.
        
        Args:
            error: The exception that occurred
            input_data: The input data that was being processed
            context: The workflow context
            
        Returns:
            Fallback result or raises the exception
        """
        logger.error(f"Error in {self.name}: {str(error)}", exc_info=True)
        # Default behavior is to re-raise
        raise error 