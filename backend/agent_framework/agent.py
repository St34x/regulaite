"""
Core Agent implementation for the RegulAIte Agent Framework.
"""
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field, model_validator
import logging
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    """Types of user query intents."""
    QUESTION = "question"
    COMMAND = "command"
    CLARIFICATION = "clarification"
    UNKNOWN = "unknown"

class QueryContext(BaseModel):
    """Context information about a query."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[str] = None
    previous_interactions: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Query(BaseModel):
    """Incoming user query with metadata."""
    query_text: str = Field(..., description="The raw query text from the user")
    intent: IntentType = Field(default=IntentType.UNKNOWN)
    context: QueryContext = Field(default_factory=QueryContext)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def classify_intent(self):
        """Automatically classify the intent if not provided."""
        if self.intent == IntentType.UNKNOWN:
            # Simple classification - can be enhanced later
            query_lower = self.query_text.lower()
            if query_lower.endswith('?'):
                self.intent = IntentType.QUESTION
            elif query_lower.startswith(('find', 'search', 'get', 'retrieve', 'show')):
                self.intent = IntentType.COMMAND
            elif any(x in query_lower for x in ["what do you mean", "can you explain", "please clarify"]):
                self.intent = IntentType.CLARIFICATION
        return self

class ToolResult(BaseModel):
    """Result from executing a tool."""
    tool_id: str
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResponse(BaseModel):
    """Response from the agent."""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    tools_used: List[str] = Field(default_factory=list)
    context_used: bool = False
    confidence: float = 1.0
    thinking: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class Agent:
    """
    Core Agent class for orchestrating query processing.
    
    This agent processes user queries, selects appropriate tools,
    and generates responses based on tool outputs.
    """
    
    def __init__(self, agent_id: str, name: str, tools: Optional[Dict[str, Callable]] = None):
        """
        Initialize the agent with its identity and tools.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            tools: Dictionary of tools available to this agent
        """
        self.agent_id = agent_id
        self.name = name
        self.tools = tools or {}
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
    async def process_query(self, query: Union[str, Query]) -> AgentResponse:
        """
        Process a user query and generate a response.
        
        Args:
            query: Either a raw query string or a Query object
            
        Returns:
            An AgentResponse object with the agent's response
        """
        # Convert string queries to Query objects
        if isinstance(query, str):
            query = Query(query_text=query)
            
        self.logger.info(f"Processing query: {query.query_text}")
        self.logger.info(f"Detected intent: {query.intent}")
        
        # This is where the core agent logic will go
        # 1. Select appropriate tools based on intent
        # 2. Execute tools in appropriate sequence
        # 3. Generate response based on tool outputs
        
        # Placeholder logic - will be replaced in subclasses
        response = AgentResponse(
            content=f"I received your query: {query.query_text}",
            tools_used=[],
            context_used=False
        )
        
        return response
    
    async def execute_tool(self, tool_id: str, **kwargs) -> ToolResult:
        """
        Execute a specific tool with the given parameters.
        
        Args:
            tool_id: ID of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            A ToolResult with the output of the tool execution
        """
        if tool_id not in self.tools:
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error=f"Tool not found: {tool_id}"
            )
            
        try:
            tool_func = self.tools[tool_id]
            result = await tool_func(**kwargs)
            return ToolResult(
                tool_id=tool_id,
                success=True,
                result=result
            )
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_id}: {str(e)}")
            return ToolResult(
                tool_id=tool_id,
                success=False,
                error=str(e)
            ) 