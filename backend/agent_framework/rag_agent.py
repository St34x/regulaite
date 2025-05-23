"""
RAG Agent implementation for the RegulAIte Agent Framework.

This module provides a Retrieval-Augmented Generation (RAG) agent that combines
the agent foundation with retrieval capabilities.
"""
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import json
import asyncio

from .agent import Agent, AgentResponse, Query
from .query_parser import ParsedQuery, QueryParser
from .tool_registry import ToolRegistry

# Set up logging
logger = logging.getLogger(__name__)

class RAGAgent(Agent):
    """
    Retrieval-Augmented Generation agent that combines query processing
    with document retrieval capabilities.
    """
    
    def __init__(self, 
                agent_id: str = "rag_agent", 
                name: str = "RAG Agent",
                tool_registry: Optional[ToolRegistry] = None,
                query_parser: Optional[QueryParser] = None,
                retrieval_system=None,
                llm_client=None,
                max_sources: int = 5):
        """
        Initialize the RAG agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            tool_registry: Registry of tools available to this agent
            query_parser: Parser for processing queries
            retrieval_system: System for retrieving documents
            llm_client: Client for LLM-based generation
            max_sources: Maximum number of sources to retrieve
        """
        super().__init__(agent_id, name)
        
        self.tool_registry = tool_registry or ToolRegistry()
        self.query_parser = query_parser or QueryParser()
        self.retrieval_system = retrieval_system
        self.llm_client = llm_client
        self.max_sources = max_sources
        
        # Register standard RAG tools
        self._register_standard_tools()
        
    def _register_standard_tools(self):
        """Register standard tools for the RAG agent."""
        # This would register tools like document retrieval, query reformulation, etc.
        # For now, just a placeholder
        pass
        
    async def process_query(self, query: Union[str, Query, ParsedQuery]) -> AgentResponse:
        """
        Process a user query and generate a response with RAG capabilities.
        
        Args:
            query: The query to process
            
        Returns:
            An AgentResponse with the agent's response
        """
        # Parse the query if it's not already parsed
        if isinstance(query, str):
            query = Query(query_text=query)
        
        if not isinstance(query, ParsedQuery):
            parsed_query = await self.query_parser.parse(query)
        else:
            parsed_query = query
            
        self.logger.info(f"Processing query with RAG agent: {parsed_query.query_text}")
        self.logger.info(f"Query category: {getattr(parsed_query, 'category', 'unknown')}")
        
        # Initialize response objects
        context = []
        tool_results = []
        sources = []
        enhanced_query = parsed_query.query_text
        
        # Step 1: Select and execute relevant tools FIRST (before RAG)
        # Tools can help enhance the query, extract entities, or provide additional context
        tool_ids = await self.tool_registry.select_tools(parsed_query.query_text)
        self.logger.info(f"Selected tools: {tool_ids}")
        
        for tool_id in tool_ids:
            tool = self.tool_registry.get_tool(tool_id)
            if tool:
                try:
                    # Execute the tool with just the query parameter
                    # Most tools expect a 'query' parameter
                    self.logger.info(f"Executing tool: {tool_id}")
                    
                    # All tools in our system are async, so await them properly
                    result = await tool(query=parsed_query.query_text)
                    
                    # Store successful tool results (ensure JSON serializable)
                    serializable_result = result
                    try:
                        json.dumps(result)  # Test if result is JSON serializable
                    except (TypeError, ValueError):
                        # Convert non-serializable result to string
                        serializable_result = str(result)
                    
                    tool_results.append({
                        "tool_id": tool_id,
                        "result": serializable_result,
                        "success": True
                    })
                    
                    self.logger.info(f"Tool {tool_id} execution successful")
                    
                    # Use tool results to enhance the query for RAG
                    if tool_id == "query_reformulation" and isinstance(result, dict):
                        reformulations = result.get("reformulations", [])
                        if reformulations:
                            # Use the best reformulation for RAG
                            enhanced_query = reformulations[0]
                            self.logger.info(f"Enhanced query with reformulation: {enhanced_query}")
                    
                    elif tool_id == "extract_search_entities" and isinstance(result, dict):
                        # Use extracted entities to enhance the search
                        entities = result
                        entity_terms = []
                        for entity_type, entity_list in entities.items():
                            if entity_list and entity_type in ['keywords', 'regulations', 'organizations']:
                                entity_terms.extend(entity_list)
                        
                        if entity_terms:
                            # Add important entities to the query for better retrieval
                            enhanced_query = f"{parsed_query.query_text} {' '.join(entity_terms[:5])}"
                            self.logger.info(f"Enhanced query with entities: {enhanced_query}")
                            
                except Exception as e:
                    self.logger.error(f"Error executing tool {tool_id}: {str(e)}")
                    tool_results.append({
                        "tool_id": tool_id,
                        "result": None,
                        "success": False,
                        "error": str(e)
                    })
        
        # Step 2: Retrieve relevant context using enhanced query (after tools)
        if self.retrieval_system and getattr(parsed_query, 'category', None) != 'system':
            try:
                self.logger.info(f"Retrieving context from RAG system using enhanced query: {enhanced_query}")
                retrieval_result = await self.retrieval_system.retrieve(
                    enhanced_query,  # Use enhanced query instead of original
                    top_k=self.max_sources
                )
                
                if retrieval_result:
                    if isinstance(retrieval_result, dict) and "results" in retrieval_result:
                        # Handle the case where retrieval returns a dict with results
                        context = retrieval_result["results"]
                        if "sources" in retrieval_result:
                            sources = retrieval_result["sources"]
                    elif isinstance(retrieval_result, list):
                        # Handle the case where retrieval returns a list directly
                        context = retrieval_result
                        
                    self.logger.info(f"Retrieved {len(context)} context items")
            except Exception as e:
                self.logger.error(f"Error retrieving context: {str(e)}")
        
        # Step 3: Generate response using LLM with context and tool results
        response_content = ""
        if self.llm_client:
            try:
                # Detect language from the original query
                from .integrations.llm_integration import detect_language, get_language_instruction
                detected_language = detect_language(parsed_query.query_text)
                language_instruction = get_language_instruction(detected_language)
                
                self.logger.info(f"RAG Agent detected language: {detected_language} for original query: {parsed_query.query_text}")
                
                # Prepare context string
                context_str = ""
                if context:
                    context_str = "\n\n".join([
                        f"Context {i+1}:\n{ctx}" 
                        for i, ctx in enumerate(context)
                    ])
                
                # Prepare tool results string
                tools_str = ""
                successful_tools = [tr for tr in tool_results if tr["success"]]
                if successful_tools:
                    tools_str = "\n\n".join([
                        f"Tool {tr['tool_id']} result:\n{json.dumps(tr['result'])}"
                        for tr in successful_tools
                    ])
                
                # Prepare the prompt
                context_part = ""
                if context_str:
                    context_part = f"Context:\n{context_str}\n\n"
                
                tools_part = ""
                if tools_str:
                    tools_part = f"Tool Analysis Results:\n{tools_str}\n\n"
                
                prompt = f"""
                Original Query: {parsed_query.query_text}
                
                {tools_part}{context_part}Please provide a helpful response to the query based on the provided information.
                If the context doesn't contain relevant information, say so and provide a general response.
                If you're using information from the context, cite the relevant context numbers.
                Consider any tool analysis results when crafting your response.
                """
                
                # Call the LLM with explicit language instruction and disable auto-detection
                self.logger.info("Generating response with LLM")
                llm_response = await self.llm_client.generate(
                    prompt, 
                    system_message=language_instruction,
                    auto_language_detection=False  # Disable auto-detection since we're providing explicit instruction
                )
                
                if llm_response:
                    response_content = llm_response
                else:
                    response_content = "I wasn't able to generate a response. Please try again."
            except Exception as e:
                self.logger.error(f"Error generating response with LLM: {str(e)}")
                response_content = "I encountered an error while generating a response. Please try again."
        else:
            # Fallback if no LLM client is available
            if context:
                response_content = (
                    f"I found {len(context)} relevant documents for your query '{parsed_query.query_text}'. "
                    f"However, I cannot provide a detailed analysis as the language model is not available. "
                    f"Please check the system configuration."
                )
            else:
                response_content = (
                    f"I processed your query '{parsed_query.query_text}' but couldn't find relevant information "
                    f"and the language model is not available for generating a response. "
                    f"Please check the system configuration or try a different query."
                )
        
        # Create the final response
        response = AgentResponse(
            content=response_content,
            tools_used=[tr["tool_id"] for tr in tool_results if tr["success"]],
            context_used=len(context) > 0,
            metadata={
                "sources": sources,
                "context_count": len(context),
                "tool_count": len([tr for tr in tool_results if tr["success"]]),
                "enhanced_query": enhanced_query,
                "tool_results": tool_results
            }
        )
        
        return response 