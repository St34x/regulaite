"""
Response Generator for the RegulAIte Agent Framework.

This module provides functionality for generating, formatting, and validating
responses from the agent system.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator
import logging
import json
from datetime import datetime

from .agent import AgentResponse, Query
from .query_parser import ParsedQuery, QueryCategory

# Set up logging
logger = logging.getLogger(__name__)

class SourceInfo(BaseModel):
    """Information about a source used in a response."""
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None
    source_type: str = "document"

class ResponseFormat(str):
    """Response format type."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"

class FormattedResponse(BaseModel):
    """A fully formatted response ready for delivery to the user."""
    response_id: str
    content: str
    format: str = ResponseFormat.TEXT
    sources: Optional[List[SourceInfo]] = None
    confidence: float = 1.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_content_format(self):
        """Validate that the content matches the specified format."""
        if self.format == ResponseFormat.JSON:
            try:
                # Check if content is valid JSON
                json.loads(self.content)
            except json.JSONDecodeError:
                # If not valid JSON, wrap in a JSON object
                self.content = json.dumps({"response": self.content})
                
        return self

class ResponseGenerator:
    """
    Generator for formatting and validating agent responses.
    """
    
    def __init__(self):
        """Initialize the response generator."""
        self.formatters = {
            ResponseFormat.TEXT: self._format_text,
            ResponseFormat.JSON: self._format_json,
            ResponseFormat.MARKDOWN: self._format_markdown,
            ResponseFormat.HTML: self._format_html
        }
        
    async def generate(self, 
                      response: AgentResponse, 
                      query: Union[Query, ParsedQuery],
                      format: str = ResponseFormat.TEXT) -> FormattedResponse:
        """
        Generate a formatted response.
        
        Args:
            response: The raw agent response
            query: The original query
            format: The desired output format
            
        Returns:
            A formatted response
        """
        # Get the appropriate formatter
        formatter = self.formatters.get(format, self._format_text)
        
        # Format the content
        formatted_content = formatter(response.content, query)
        
        # Create the formatted response
        formatted_response = FormattedResponse(
            response_id=response.response_id,
            content=formatted_content,
            format=format,
            confidence=response.confidence,
            metadata={
                "query": query.dict() if hasattr(query, "dict") else {"query_text": str(query)},
                "tools_used": response.tools_used,
                "context_used": response.context_used,
                **response.metadata
            }
        )
        
        # Add sources if available
        if "sources" in response.metadata:
            formatted_response.sources = [
                SourceInfo(**source) if isinstance(source, dict) else source
                for source in response.metadata["sources"]
            ]
            
        # Log the response generation
        logger.info(f"Generated {format} response for query: {query.query_text if hasattr(query, 'query_text') else query}")
        
        return formatted_response
        
    def _format_text(self, content: str, query: Union[Query, ParsedQuery]) -> str:
        """
        Format the response as plain text.
        
        Args:
            content: The response content
            query: The original query
            
        Returns:
            Formatted plain text response
        """
        # For text format, just return the content as is
        return content
        
    def _format_json(self, content: str, query: Union[Query, ParsedQuery]) -> str:
        """
        Format the response as JSON.
        
        Args:
            content: The response content
            query: The original query
            
        Returns:
            Formatted JSON response
        """
        # Create a response object
        response_obj = {
            "response": content,
            "query": query.query_text if hasattr(query, "query_text") else str(query)
        }
        
        # Add category if available
        if hasattr(query, "category"):
            response_obj["category"] = query.category
            
        # Add entities if available
        if hasattr(query, "entities") and query.entities:
            response_obj["entities"] = query.entities
            
        return json.dumps(response_obj)
        
    def _format_markdown(self, content: str, query: Union[Query, ParsedQuery]) -> str:
        """
        Format the response as Markdown.
        
        Args:
            content: The response content
            query: The original query
            
        Returns:
            Formatted Markdown response
        """
        # For now, just add some basic formatting
        formatted = f"## Response\n\n{content}\n\n"
        
        # Add sources section if available in query metadata
        if hasattr(query, "metadata") and "sources" in query.metadata:
            formatted += "\n\n## Sources\n\n"
            for i, source in enumerate(query.metadata["sources"]):
                title = source.get("title", f"Source {i+1}")
                url = source.get("url", "")
                
                if url:
                    formatted += f"- [{title}]({url})\n"
                else:
                    formatted += f"- {title}\n"
        
        return formatted
        
    def _format_html(self, content: str, query: Union[Query, ParsedQuery]) -> str:
        """
        Format the response as HTML.
        
        Args:
            content: The response content
            query: The original query
            
        Returns:
            Formatted HTML response
        """
        # Convert the content to HTML
        # For now, just wrap in basic HTML tags and escape special characters
        import html
        
        escaped_content = html.escape(content)
        formatted = f"<div class='response'><p>{escaped_content}</p></div>"
        
        # Add sources section if available in query metadata
        if hasattr(query, "metadata") and "sources" in query.metadata:
            formatted += "<div class='sources'><h3>Sources</h3><ul>"
            for i, source in enumerate(query.metadata["sources"]):
                title = html.escape(source.get("title", f"Source {i+1}"))
                url = source.get("url", "")
                
                if url:
                    formatted += f"<li><a href='{html.escape(url)}'>{title}</a></li>"
                else:
                    formatted += f"<li>{title}</li>"
            
            formatted += "</ul></div>"
            
        return formatted 