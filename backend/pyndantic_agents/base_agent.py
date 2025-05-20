import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat()
        logger.info(f"Initialized agent with ID: {self.agent_id}")
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with the given query.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Result of agent execution
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def close(self):
        """Clean up resources."""
        logger.info(f"Closing agent with ID: {self.agent_id}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.close()
        except Exception as e:
            logger.error(f"Error closing agent {self.agent_id}: {e}")
    
    def __repr__(self):
        """String representation of the agent."""
        return f"{self.__class__.__name__}(agent_id={self.agent_id})" 