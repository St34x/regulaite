"""
Adapter for tree reasoning functionality using the autonomous agent architecture.
This replaces the classic TreeReasoningAgent implementation.
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
from ..processing_nodes.base_node import ProcessingStepResult

logger = logging.getLogger(__name__)

class TreeNode:
    """
    Represents a node in the decision tree.
    This is a simpler version than in the classic implementation.
    """
    def __init__(
        self, 
        node_id: str, 
        node_type: str, 
        content: Dict[str, Any],
        next_node_id: Optional[str] = None
    ):
        self.id = node_id
        self.type = node_type
        self.content = content
        self.next_node_id = next_node_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "next_node_id": self.next_node_id
        }


class DecisionTree:
    """
    Represents a decision tree for reasoning.
    This is a simpler version than in the classic implementation.
    """
    def __init__(
        self, 
        tree_id: str, 
        name: str, 
        description: str,
        nodes: Dict[str, Dict[str, Any]],
        root_node_id: str
    ):
        self.id = tree_id
        self.name = name
        self.description = description
        
        # Convert dictionary nodes to TreeNode objects
        self.nodes = {}
        for node_id, node_data in nodes.items():
            node_type = node_data.get("type", "unknown")
            content = {k: v for k, v in node_data.items() if k not in ["id", "type", "next"]}
            next_node_id = node_data.get("next", None)
            self.nodes[node_id] = TreeNode(node_id, node_type, content, next_node_id)
        
        self.root_node_id = root_node_id
    
    @classmethod
    def from_dict(cls, tree_dict: Dict[str, Any]) -> 'DecisionTree':
        """Create a DecisionTree from a dictionary representation."""
        return cls(
            tree_id=tree_dict.get("id", str(uuid.uuid4())),
            name=tree_dict.get("name", "Unnamed Tree"),
            description=tree_dict.get("description", ""),
            nodes=tree_dict.get("nodes", {}),
            root_node_id=tree_dict.get("root_node", "start")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_node": self.root_node_id
        }


class TreeReasoningAdapter:
    """
    Adapter that implements tree reasoning using the autonomous agent architecture.
    This provides compatibility with the classic TreeReasoningAgent.
    """
    
    def __init__(
        self,
        tree: Union[Dict[str, Any], DecisionTree],
        graph_interface: Optional[GraphInterface] = None,
        embedding_service: Any = None,
        llm_client: Any = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the tree reasoning adapter.
        
        Args:
            tree: Decision tree as a dictionary or DecisionTree object
            graph_interface: Interface to the knowledge graph
            embedding_service: Service for generating embeddings
            llm_client: Client for language model interactions
            config: Configuration parameters
        """
        self.graph_interface = graph_interface
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        
        # Initialize the tree
        if isinstance(tree, dict):
            self.tree = DecisionTree.from_dict(tree)
        else:
            self.tree = tree
        
        # Initialize workflow engine for tree traversal
        workflow_config = WorkflowConfig(
            max_reformulation_attempts=config.get("max_reformulation_attempts", 1) if config else 1,
            return_intermediate_results=True,  # We need intermediate results for tree traversal
            save_workflow_history=True,  # Save history for debugging
            timeout_seconds=config.get("timeout_seconds", 60) if config else 60,
            enable_async_processing=True,
            node_configs=config.get("node_configs", {}) if config else {}
        )
        
        self.workflow_engine = WorkflowEngine(config=workflow_config)
        
        logger.info(f"TreeReasoningAdapter initialized with tree: {self.tree.id}")
    
    async def process(self, user_input: str, session_id: str, user_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a user query using tree-based reasoning.
        
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
        
        try:
            # Initialize the workflow for this request
            workflow_id = self.workflow_engine.initialize_workflow(session_id, user_id)
            
            # Start tree traversal from the root node
            current_node_id = self.tree.root_node_id
            traversal_history = []
            
            # Prepare the initial context for LLM
            path_context = {
                "user_query": user_input,
                "tree_id": self.tree.id,
                "tree_name": self.tree.name,
                "current_node": current_node_id,
                "traversal_history": traversal_history
            }
            
            # Prepare the full context for the workflow
            context = {
                "model": model,
                "include_context": include_context,
                "workflow_id": workflow_id,
                "session_id": session_id, 
                "user_id": user_id,
                "graph_interface": self.graph_interface,
                "embedding_service": self.embedding_service,
                "llm_client": self.llm_client,
                "path_context": path_context,
                **kwargs
            }
            
            # Begin traversal of the tree
            response = await self._traverse_tree(
                user_input=user_input,
                context=context,
                start_node_id=current_node_id
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Format the response to match the expected format from classic agents
            formatted_response = {
                "response": response.get("content", ""),
                "source_documents": response.get("sources", []),
                "execution_time": execution_time,
                "agent_type": "tree_reasoning",
                "model": model,
                "session_id": session_id,
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat(),
                "context_used": include_context,
                "tree_id": self.tree.id,
                "traversal_path": traversal_history,
                "tree_reasoning_used": True
            }
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in tree reasoning process: {str(e)}", exc_info=True)
            # Return error response
            return {
                "response": f"I encountered an error while processing your request with tree reasoning: {str(e)}",
                "source_documents": [],
                "execution_time": time.time() - start_time,
                "agent_type": "tree_reasoning",
                "model": model,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "tree_id": self.tree.id,
                "tree_reasoning_used": True
            }
    
    async def _traverse_tree(
        self,
        user_input: str,
        context: Dict[str, Any],
        start_node_id: str
    ) -> Dict[str, Any]:
        """
        Traverse the decision tree based on user input and context.
        
        Args:
            user_input: User query text
            context: Processing context
            start_node_id: ID of the node to start traversal from
            
        Returns:
            Final response from the tree traversal
        """
        current_node_id = start_node_id
        traversal_history = context.get("path_context", {}).get("traversal_history", [])
        
        # Record start node in traversal history
        traversal_history.append({
            "node_id": current_node_id,
            "node_type": self.tree.nodes[current_node_id].type,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update context with traversal history
        context["path_context"]["traversal_history"] = traversal_history
        
        # Maximum depth to prevent infinite loops
        max_depth = 20
        current_depth = 0
        
        response = {
            "content": "No response generated",
            "sources": []
        }
        
        # Traverse the tree until we reach a terminal node or max depth
        while current_node_id and current_depth < max_depth:
            current_depth += 1
            
            # Get the current node
            if current_node_id not in self.tree.nodes:
                logger.error(f"Node ID not found in tree: {current_node_id}")
                response["content"] = f"Error: Node {current_node_id} not found in decision tree."
                break
            
            current_node = self.tree.nodes[current_node_id]
            node_type = current_node.type
            
            logger.info(f"Traversing node: {current_node_id} of type {node_type}")
            
            # Update context with current node
            context["path_context"]["current_node"] = current_node_id
            
            # Process the node based on its type
            if node_type == "decision":
                # For decision nodes, call a processing function that determines the next node
                next_node_id = await self._process_decision_node(
                    node=current_node,
                    user_input=user_input,
                    context=context
                )
                
            elif node_type == "action":
                # For action nodes, perform the specified action and proceed to next node
                action_result = await self._process_action_node(
                    node=current_node,
                    user_input=user_input,
                    context=context
                )
                
                # Store the action result in context for use by future nodes
                context["last_action_result"] = action_result
                
                # Get the next node ID from the current node
                next_node_id = current_node.next_node_id
                
            elif node_type == "response":
                # For response nodes, generate the final response and terminate traversal
                response = await self._process_response_node(
                    node=current_node,
                    user_input=user_input,
                    context=context
                )
                
                # Response nodes are terminal, so set next_node_id to None
                next_node_id = None
                
            else:
                # Unknown node type, log error and terminate traversal
                logger.error(f"Unknown node type: {node_type}")
                response["content"] = f"Error: Unknown node type {node_type} in decision tree."
                break
            
            # Record the node traversal in history
            traversal_history.append({
                "node_id": current_node_id,
                "node_type": node_type,
                "timestamp": datetime.utcnow().isoformat(),
                "next_node": next_node_id
            })
            
            # Update the current node ID for the next iteration
            current_node_id = next_node_id
            
            # Update context with traversal history
            context["path_context"]["traversal_history"] = traversal_history
        
        # Log the final traversal path
        logger.info(f"Tree traversal complete. Path: {[step['node_id'] for step in traversal_history]}")
        
        return response
    
    async def _process_decision_node(
        self,
        node: TreeNode,
        user_input: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Process a decision node to determine the next node in the traversal.
        
        Args:
            node: Current tree node
            user_input: User query text
            context: Processing context
            
        Returns:
            ID of the next node to traverse to
        """
        try:
            # Get the decision query from the node
            query = node.content.get("query", "Make a decision based on the user input.")
            options = node.content.get("options", [])
            
            if not options:
                logger.error(f"Decision node {node.id} has no options")
                return None
            
            # Prepare prompt for LLM to make a decision
            prompt = f"""
            Based on the user query: "{user_input}"
            
            You need to choose the most appropriate option for the following decision:
            {query}
            
            Available options:
            {json.dumps(options, indent=2)}
            
            Choose the value of exactly ONE option from the list above.
            Respond ONLY with the selected option value (e.g., "option1").
            """
            
            # Call LLM to make the decision
            llm_client = context.get("llm_client")
            if not llm_client:
                logger.error("LLM client not available in context")
                return None
            
            # Get model from context or use default
            model = context.get("model", "gpt-4")
            
            # Make LLM call to get the decision
            try:
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a decision-making assistant. Respond only with the specific option value requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more deterministic decisions
                    max_tokens=50     # We only need a short response
                )
                
                # Extract the decision from the response
                decision_text = response.choices[0].message.content.strip()
                logger.info(f"LLM decision for node {node.id}: {decision_text}")
                
                # Find the matching option
                for option in options:
                    if option["value"] in decision_text:
                        logger.info(f"Selected option: {option['value']} -> {option['next']}")
                        return option["next"]
                
                # If no exact match, try a more flexible approach
                for option in options:
                    if option["value"].lower() in decision_text.lower():
                        logger.info(f"Fuzzy matched option: {option['value']} -> {option['next']}")
                        return option["next"]
                
                # If still no match, try to match based on label
                for option in options:
                    if "label" in option and option["label"].lower() in decision_text.lower():
                        logger.info(f"Label matched option: {option['value']} -> {option['next']}")
                        return option["next"]
                
                # No match found, use the first option as default
                logger.warning(f"No matching option found in response: {decision_text}. Using default.")
                return options[0].get("next", None)
                
            except Exception as e:
                logger.error(f"Error calling LLM for decision: {str(e)}")
                # Fall back to first option in case of error
                return options[0].get("next", None)
                
        except Exception as e:
            logger.error(f"Error processing decision node: {str(e)}")
            return None
    
    async def _process_action_node(
        self,
        node: TreeNode,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an action node to perform an action.
        
        Args:
            node: Current tree node
            user_input: User query text
            context: Processing context
            
        Returns:
            Result of the action
        """
        try:
            # Get the action from the node
            action = node.content.get("action", "unknown_action")
            
            logger.info(f"Processing action node {node.id} with action: {action}")
            
            # Handle different action types
            if action == "retrieve_context":
                # Retrieve context from RAG system
                if context.get("include_context", True) and context.get("embedding_service"):
                    embedding_service = context.get("embedding_service")
                    result = await self._retrieve_context(user_input, embedding_service)
                    return {
                        "action": action,
                        "success": True,
                        "context": result.get("context", []),
                        "sources": result.get("sources", [])
                    }
                else:
                    return {
                        "action": action,
                        "success": False,
                        "error": "Context retrieval not available or disabled"
                    }
                    
            elif action == "retrieve_facts":
                # Retrieve factual information (similar to context retrieval but more focused)
                embedding_service = context.get("embedding_service")
                result = await self._retrieve_context(
                    user_input, 
                    embedding_service, 
                    filter_criteria={"document_type": "factual"}
                )
                return {
                    "action": action,
                    "success": True,
                    "facts": result.get("context", []),
                    "sources": result.get("sources", [])
                }
                
            elif action == "retrieve_procedures":
                # Retrieve procedural information
                embedding_service = context.get("embedding_service")
                result = await self._retrieve_context(
                    user_input, 
                    embedding_service, 
                    filter_criteria={"document_type": "procedural"}
                )
                return {
                    "action": action,
                    "success": True,
                    "procedures": result.get("context", []),
                    "sources": result.get("sources", [])
                }
                
            elif action == "perform_analysis":
                # Perform analysis using LLM
                llm_client = context.get("llm_client")
                if not llm_client:
                    return {
                        "action": action,
                        "success": False,
                        "error": "LLM client not available"
                    }
                
                # Get any context retrieved in previous nodes
                previous_context = []
                if "last_action_result" in context and "context" in context["last_action_result"]:
                    previous_context = context["last_action_result"]["context"]
                
                # Prepare analysis prompt
                prompt = f"""
                Analyze the following query: "{user_input}"
                
                {f"Using this context: {json.dumps(previous_context, indent=2)}" if previous_context else ""}
                
                Provide a detailed analysis including:
                1. Key points and implications
                2. Relevant considerations
                3. Potential risks or challenges
                4. Recommended approach
                """
                
                # Call LLM for analysis
                model = context.get("model", "gpt-4")
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an analytical assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                analysis = response.choices[0].message.content
                
                return {
                    "action": action,
                    "success": True,
                    "analysis": analysis
                }
                
            else:
                # Unknown action
                logger.warning(f"Unknown action: {action}")
                return {
                    "action": action,
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Error processing action node: {str(e)}", exc_info=True)
            return {
                "action": "error",
                "success": False,
                "error": str(e)
            }
    
    async def _process_response_node(
        self,
        node: TreeNode,
        user_input: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a response node to generate the final response.
        
        Args:
            node: Current tree node
            user_input: User query text
            context: Processing context
            
        Returns:
            Final response
        """
        try:
            # Get the response template from the node
            response_template = node.content.get(
                "response_template", 
                "Based on my analysis, I can provide the following information: {result}"
            )
            
            # Get results from previous actions
            result_content = ""
            sources = []
            
            if "last_action_result" in context:
                action_result = context["last_action_result"]
                
                # Extract different possible result types
                if "analysis" in action_result:
                    result_content = action_result["analysis"]
                elif "facts" in action_result:
                    result_content = "Based on the facts gathered: " + json.dumps(action_result["facts"])
                elif "procedures" in action_result:
                    result_content = "The relevant procedures are: " + json.dumps(action_result["procedures"])
                elif "context" in action_result:
                    result_content = "Based on the retrieved information: " + json.dumps(action_result["context"])
                
                # Extract sources if available
                sources = action_result.get("sources", [])
            
            # If no result content is available, generate a response using LLM
            if not result_content:
                llm_client = context.get("llm_client")
                if llm_client:
                    # Prepare prompt for response generation
                    prompt = f"""
Generate a comprehensive response to the following query:
"{user_input}"

Your response should be helpful, accurate, and concise.
**Crucially, respond in the same language as the user's query shown above.**
"""
                    
                    # Get traversal history from context to provide more information to the LLM
                    traversal_history = context.get("path_context", {}).get("traversal_history", [])
                    if traversal_history:
                        history_summary = "Decision path: " + " -> ".join([step.get("node_id", "unknown") for step in traversal_history])
                        prompt += f"\n\nAdditional context:\n{history_summary}"
                    
                    # Call LLM to generate response
                    model = context.get("model", "gpt-4")
                    try:
                        response = llm_client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant specializing in regulatory information."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=context.get("max_tokens", 1000)
                        )
                        
                        result_content = response.choices[0].message.content
                        logger.info(f"Generated response with LLM for node {node.id}")
                    except Exception as e:
                        logger.error(f"Error generating response with LLM: {str(e)}")
                        result_content = f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
                else:
                    result_content = "No information is available to answer your query at this time."
            
            # Format the final response using the template
            try:
                response_content = response_template.format(result=result_content)
            except Exception as e:
                logger.error(f"Error formatting response with template: {str(e)}")
                response_content = result_content  # Fall back to raw result if template formatting fails
            
            return {
                "content": response_content,
                "sources": sources
            }
                
        except Exception as e:
            logger.error(f"Error processing response node: {str(e)}")
            return {
                "content": f"I encountered an error processing your request: {str(e)}",
                "sources": []
            }
    
    async def _retrieve_context(
        self, 
        query: str, 
        embedding_service: Any,
        filter_criteria: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve context from the embedding service.
        
        Args:
            query: Query text
            embedding_service: Service for retrieving context
            filter_criteria: Optional filter criteria for retrieval
            limit: Maximum number of results to retrieve
            
        Returns:
            Retrieved context and sources
        """
        try:
            if not embedding_service:
                logger.warning("No embedding service provided for context retrieval")
                return {"context": [], "sources": []}
            
            # Determine how to use the embedding service based on available methods
            results = None
            
            # Try to use the most common method patterns for embedding services
            try:
                logger.info(f"Attempting to retrieve context with query: {query}")
                
                # Check if it's a RAG system with retrieve method
                if hasattr(embedding_service, "retrieve"):
                    logger.info("Using retrieve method")
                    import inspect
                    if inspect.iscoroutinefunction(embedding_service.retrieve):
                        results = await embedding_service.retrieve(query=query, top_k=limit)
                    else:
                        results = embedding_service.retrieve(query=query, top_k=limit)
                
                # Check if it's a Qdrant client
                elif hasattr(embedding_service, "search") and hasattr(embedding_service, "get_collection"):
                    logger.info("Using Qdrant search method")
                    # This is likely a Qdrant client
                    collection_name = "documents"  # Default collection name
                    if hasattr(embedding_service, "collection_name"):
                        collection_name = embedding_service.collection_name
                    
                    search_results = embedding_service.search(
                        collection_name=collection_name,
                        query_vector=await self._get_embedding(query, embedding_service),
                        limit=limit
                    )
                    results = search_results
                
                # Check for semantic_search method
                elif hasattr(embedding_service, "semantic_search"):
                    logger.info("Using semantic_search method")
                    import inspect
                    if inspect.iscoroutinefunction(embedding_service.semantic_search):
                        results = await embedding_service.semantic_search(query=query, limit=limit)
                    else:
                        results = embedding_service.semantic_search(query=query, limit=limit)
                
                # Check for query method
                elif hasattr(embedding_service, "query"):
                    logger.info("Using query method")
                    import inspect
                    if inspect.iscoroutinefunction(embedding_service.query):
                        results = await embedding_service.query(text=query, top_k=limit)
                    else:
                        results = embedding_service.query(text=query, top_k=limit)
                
                # If we have a rag_system attribute, use that
                elif hasattr(embedding_service, "rag_system") and embedding_service.rag_system:
                    logger.info("Using rag_system attribute")
                    rag = embedding_service.rag_system
                    if hasattr(rag, "retrieve"):
                        if inspect.iscoroutinefunction(rag.retrieve):
                            results = await rag.retrieve(query=query, top_k=limit)
                        else:
                            results = rag.retrieve(query=query, top_k=limit)
                
                else:
                    logger.warning(f"Could not determine how to use embedding service of type {type(embedding_service)}")
                    # Try a generic call as last resort
                    if callable(embedding_service):
                        results = embedding_service(query=query, limit=limit)
            
            except Exception as e:
                logger.error(f"Error calling embedding service: {str(e)}", exc_info=True)
                return {"context": [], "sources": [], "error": str(e)}
            
            if results is None:
                logger.warning("No results returned from embedding service")
                return {"context": [], "sources": []}
            
            # Process the results based on their format
            context_items = []
            sources = []
            
            logger.info(f"Processing results of type: {type(results)}")
            
            # Handle different result formats
            if isinstance(results, list):
                for item in results:
                    # Extract text and metadata based on common patterns
                    text = None
                    metadata = {}
                    
                    if isinstance(item, dict):
                        # Common dictionary formats
                        if "text" in item:
                            text = item["text"]
                        elif "content" in item:
                            text = item["content"]
                        elif "passage" in item:
                            text = item["passage"]
                        elif "chunk" in item:
                            text = item["chunk"]
                        # Handle payload from vector DB
                        elif "payload" in item and isinstance(item["payload"], dict):
                            payload = item["payload"]
                            if "text" in payload:
                                text = payload["text"]
                            elif "content" in payload:
                                text = payload["content"]
                            # Also check for metadata in payload
                            if "metadata" in payload:
                                metadata = payload["metadata"]
                        
                        # Get metadata from various locations
                        if "metadata" in item:
                            metadata = item["metadata"]
                        elif "source" in item:
                            metadata = {"source": item["source"]}
                    
                    # Handle object-style results
                    elif hasattr(item, "text"):
                        text = item.text
                        if hasattr(item, "metadata"):
                            metadata = item.metadata
                    elif hasattr(item, "content"):
                        text = item.content
                        if hasattr(item, "metadata"):
                            metadata = item.metadata
                    
                    # Add to context if we found text
                    if text:
                        context_items.append(text)
                        if metadata:
                            sources.append(metadata)
            
            # Handle dictionary results
            elif isinstance(results, dict):
                # Handle common dictionary formats
                if "documents" in results and isinstance(results["documents"], list):
                    context_items = results["documents"]
                elif "texts" in results and isinstance(results["texts"], list):
                    context_items = results["texts"]
                elif "context" in results and isinstance(results["context"], list):
                    context_items = results["context"]
                elif "passages" in results and isinstance(results["passages"], list):
                    context_items = results["passages"]
                
                # Handle metadata/sources
                if "metadatas" in results and isinstance(results["metadatas"], list):
                    sources = results["metadatas"]
                elif "sources" in results and isinstance(results["sources"], list):
                    sources = results["sources"]
                elif "metadata" in results and isinstance(results["metadata"], list):
                    sources = results["metadata"]
            
            # Handle object-style results
            elif hasattr(results, "documents") and isinstance(results.documents, list):
                context_items = results.documents
                if hasattr(results, "metadatas") and isinstance(results.metadatas, list):
                    sources = results.metadatas
            elif hasattr(results, "texts") and isinstance(results.texts, list):
                context_items = results.texts
                if hasattr(results, "metadatas") and isinstance(results.metadatas, list):
                    sources = results.metadatas
            
            logger.info(f"Retrieved {len(context_items)} context items from embedding service")
            
            return {
                "context": context_items,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return {"context": [], "sources": [], "error": str(e)}
            
    async def _get_embedding(self, text: str, embedding_service: Any) -> List[float]:
        """Helper method to get embeddings from various embedding services"""
        try:
            # Try common embedding methods
            if hasattr(embedding_service, "get_embedding"):
                import inspect
                if inspect.iscoroutinefunction(embedding_service.get_embedding):
                    return await embedding_service.get_embedding(text)
                else:
                    return embedding_service.get_embedding(text)
            elif hasattr(embedding_service, "embed_query"):
                import inspect
                if inspect.iscoroutinefunction(embedding_service.embed_query):
                    return await embedding_service.embed_query(text)
                else:
                    return embedding_service.embed_query(text)
            elif hasattr(embedding_service, "embed"):
                import inspect
                if inspect.iscoroutinefunction(embedding_service.embed):
                    return await embedding_service.embed(text)
                else:
                    return embedding_service.embed(text)
            else:
                # If all else fails, try to find a model in the embedding service
                if hasattr(embedding_service, "embedding_model"):
                    model = embedding_service.embedding_model
                    if hasattr(model, "get_embedding"):
                        return model.get_embedding(text)
                    elif hasattr(model, "embed_query"):
                        return model.embed_query(text)
                    elif hasattr(model, "embed"):
                        return model.embed(text)
                
                logger.warning("Could not find embedding method")
                # Return a zero vector as fallback (not ideal but prevents errors)
                return [0.0] * 1536  # Standard OpenAI embedding size
                
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}", exc_info=True)
            # Return a zero vector as fallback
            return [0.0] * 1536  # Standard OpenAI embedding size 