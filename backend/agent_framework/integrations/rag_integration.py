"""
RAG Integration for the RegulAIte Agent Framework.

This module provides integration with the existing RAG system.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import json
import sys
import os
from pathlib import Path

# Add the backend directory to the path so we can import from it
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Import the RAG system
try:
    from rag.query_engine import get_query_engine, QueryEngine
except ImportError:
    # If the RAG system can't be imported, provide a mock version
    QueryEngine = object
    def get_query_engine():
        return None

# Set up logging
logger = logging.getLogger(__name__)

class RAGIntegration:
    """
    Integration with the existing RAG system.
    
    This class provides a bridge between the Agent Framework and the
    existing RAG system.
    """
    
    def __init__(self, query_engine=None):
        """
        Initialize the RAG integration.
        
        Args:
            query_engine: An existing QueryEngine instance to use
        """
        self.query_engine = query_engine
        
        # Try to initialize the query engine if not provided
        if self.query_engine is None:
            try:
                # Try to get the query engine from the main application
                try:
                    from main import rag_query_engine
                    if rag_query_engine is not None:
                        self.query_engine = rag_query_engine
                        logger.info("Successfully initialized RAG query engine from main application")
                    else:
                        logger.warning("RAG query engine not available in main application")
                except ImportError:
                    logger.warning("Could not import RAG query engine from main application")
                
                # If still no query engine, try the generic get_query_engine function
                if self.query_engine is None:
                    self.query_engine = get_query_engine()
                    if self.query_engine:
                        logger.info("Successfully initialized RAG query engine from get_query_engine")
                    else:
                        logger.warning("get_query_engine returned None")
                        
            except Exception as e:
                logger.error(f"Failed to initialize RAG query engine: {str(e)}")
                self.query_engine = None
                
    async def retrieve(self, query: str, top_k: int = 5, search_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the RAG system.
        
        Args:
            query: The query to retrieve documents for
            top_k: Maximum number of documents to retrieve
            search_filter: Optional filter for the search
            
        Returns:
            Dictionary with retrieval results and sources
        """
        if self.query_engine is None:
            logger.error("Cannot retrieve documents: RAG query engine not initialized")
            return {"results": [], "sources": []}
            
        try:
            logger.info(f"Retrieving documents for query: {query}")
            
            # Call the query engine with the appropriate parameters
            # The interface might vary based on the actual implementation
            if hasattr(self.query_engine, 'retrieve'):
                # If the query engine has a retrieve method, use it
                retrieval_result = await self.query_engine.retrieve(
                    query, 
                    top_k=top_k, 
                    search_filter=search_filter
                )
                
                # Process the results into a standard format
                return self._process_retrieval_result(retrieval_result)
            elif hasattr(self.query_engine, 'query'):
                # If the query engine only has a query method, use it
                # and extract the context used for the response
                query_result = await self.query_engine.query(
                    query,
                    top_k=top_k,
                    search_filter=search_filter
                )
                
                # Process the query result to extract context
                return self._process_query_result(query_result)
            else:
                logger.error("RAG query engine does not have retrieve or query methods")
                return {"results": [], "sources": []}
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return {"results": [], "sources": []}
            
    async def query(self, query: str, **kwargs) -> str:
        """
        Query the RAG system for a response.
        
        Args:
            query: The query to process
            **kwargs: Additional parameters for the query
            
        Returns:
            The response from the RAG system
        """
        if self.query_engine is None:
            logger.error("Cannot query: RAG query engine not initialized")
            return "I'm sorry, but I cannot access the knowledge base at the moment."
            
        try:
            logger.info(f"Querying RAG system: {query}")
            
            # Call the query engine
            if hasattr(self.query_engine, 'query'):
                response = await self.query_engine.query(query, **kwargs)
                
                # If the response is a dictionary, extract the response text
                if isinstance(response, dict) and "response" in response:
                    return response["response"]
                elif isinstance(response, dict) and "answer" in response:
                    return response["answer"]
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            else:
                logger.error("RAG query engine does not have a query method")
                return "I'm sorry, but I cannot process your query at the moment."
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
            
    def _process_retrieval_result(self, result: Any) -> Dict[str, Any]:
        """
        Process the retrieval result into a standard format.
        
        Args:
            result: The retrieval result from the RAG system
            
        Returns:
            Dictionary with processed results and sources
        """
        # Handle different result formats
        if result is None:
            return {"results": [], "sources": []}
            
        if isinstance(result, dict):
            # If the result is already a dictionary, check for required keys
            results = result.get("results", [])
            sources = result.get("sources", [])
            
            # If sources is not in the result, try to extract from the results
            if not sources and results:
                sources = []
                for i, text in enumerate(results):
                    source = {"id": f"source_{i}", "title": f"Source {i+1}"}
                    
                    # Try to extract metadata if available
                    if isinstance(text, dict) and "metadata" in text:
                        source.update(text["metadata"])
                        
                    sources.append(source)
            
            return {
                "results": results,
                "sources": sources
            }
        elif isinstance(result, list):
            # If the result is a list, assume it's a list of documents
            results = result
            sources = []
            
            # Create sources based on the results
            for i, text in enumerate(results):
                source = {"id": f"source_{i}", "title": f"Source {i+1}"}
                
                # Try to extract metadata if available
                if isinstance(text, dict):
                    if "metadata" in text:
                        source.update(text["metadata"])
                    if "text" in text:
                        results[i] = text["text"]
                        
                sources.append(source)
            
            return {
                "results": results,
                "sources": sources
            }
        else:
            # If the result is some other type, convert to string
            return {
                "results": [str(result)],
                "sources": [{"id": "source_0", "title": "Source 1"}]
            }
            
    def _process_query_result(self, result: Any) -> Dict[str, Any]:
        """
        Process the query result to extract context.
        
        Args:
            result: The query result from the RAG system
            
        Returns:
            Dictionary with extracted context and sources
        """
        if result is None:
            return {"results": [], "sources": []}
            
        if isinstance(result, dict):
            # Try to extract context from the result
            context = result.get("context", [])
            if not context:
                # Check other possible keys for context
                context = result.get("contexts", [])
            if not context:
                context = result.get("documents", [])
            
            # Try to extract sources
            sources = result.get("sources", [])
            if not sources:
                # Check other possible keys for sources
                sources = result.get("citations", [])
                
            # If sources is still empty, create from context
            if not sources and context:
                sources = []
                for i, text in enumerate(context):
                    source = {"id": f"source_{i}", "title": f"Source {i+1}"}
                    
                    # Try to extract metadata if available
                    if isinstance(text, dict) and "metadata" in text:
                        source.update(text["metadata"])
                        
                    sources.append(source)
            
            return {
                "results": context,
                "sources": sources
            }
        else:
            # If the result is not a dictionary, we can't extract context
            return {"results": [], "sources": []}

# Singleton instance
_rag_integration = None

def get_rag_integration():
    """
    Get the RAG integration instance.
    
    Returns:
        The RAG integration instance
    """
    global _rag_integration
    
    if _rag_integration is None:
        _rag_integration = RAGIntegration()
        
    return _rag_integration 