"""
Base Agent class for RegulAite.
"""
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    """Input data for agents"""
    query: str = Field(..., description="User query or request")
    context: Optional[List[Dict[str, Any]]] = Field(None, description="Context from RAG system")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")
    
class AgentOutput(BaseModel):
    """Output data from agents"""
    response: str = Field(..., description="Agent response")
    context_used: Optional[List[Dict[str, Any]]] = Field(None, description="Context used in the response")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning process")
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional data returned by the agent")

class BaseAgent(ABC):
    """
    Base agent class that defines the interface for all agents.
    """
    
    def __init__(self, agent_id: Optional[str] = None, **kwargs):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
            **kwargs: Additional parameters
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.parameters = kwargs
        self.context = []
        logger.info(f"Initialized agent {self.__class__.__name__} with ID {self.agent_id}")
    
    @abstractmethod
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process an input and generate a response.
        
        Args:
            input_data: Input data containing query and optional context
            
        Returns:
            AgentOutput with the response and metadata
        """
        pass
    
    def _log_processing(self, input_data: AgentInput) -> None:
        """Log the processing of input data"""
        logger.info(f"Agent {self.agent_id} processing query: {input_data.query[:50]}...")
    
    def _validate_output(self, output: AgentOutput) -> bool:
        """
        Validate the output before returning it.
        
        Args:
            output: The output to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        if not output.response:
            raise ValueError("Agent must provide a non-empty response")
        
        if output.confidence < 0 or output.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {output.confidence}")
        
        return True 

    def _get_context(self, query: str) -> str:
        """
        Get context from RAG system.
        
        Args:
            query: Query string for context retrieval
            
        Returns:
            Context string
        """
        if not self.rag_system:
            return ""
            
        # Use the provided query or the input query
        context_query = self.config.context_query or query
        
        # Convert retrieval_type to use_hybrid parameter
        use_hybrid = None  # Default (auto)
        if hasattr(self.config, 'retrieval_type'):
            if self.config.retrieval_type == "hybrid":
                use_hybrid = True
            elif self.config.retrieval_type == "vector":
                use_hybrid = False
        
        # Get context from RAG system
        try:
            nodes = self.rag_system.retrieve(
                context_query, 
                top_k=self.config.max_context_results,
                use_hybrid=use_hybrid
            )
            
            if not nodes:
                return ""
                
            # Format context
            context_parts = []
            for node in nodes:
                # Extract source information
                source = node["metadata"].get("doc_name", "Unknown document")
                if "section" in node["metadata"] and node["metadata"]["section"] != "Unknown":
                    source += f" - {node['metadata']['section']}"
                    
                # Format content
                context_parts.append(
                    f"Content: {node['text']}\n"
                    f"Source: {source}\n"
                )
                
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "" 