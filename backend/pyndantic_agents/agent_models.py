from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    model: str = Field("gpt-4", description="Model to use for generation")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens in response")
    include_context: bool = Field(True, description="Whether to include RAG context")
    context_query: Optional[str] = Field(None, description="Query to use for retrieving context")
    max_context_results: int = Field(5, description="Maximum number of context results to retrieve")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the agent") 