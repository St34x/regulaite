# plugins/regul_aite/backend/pyndantic_agents/agents.py

import logging
import os
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
import openai
from datetime import datetime

# Import RAG system
from llamaIndex_rag.rag import RAGSystem

from .tree_reasoning import TreeReasoningAgent, create_default_decision_tree


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class AgentAction(BaseModel):
    """Action that an agent can take"""
    action_type: str = Field(..., description="Type of action to perform")
    action_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    completion_status: Optional[bool] = Field(None, description="Whether the action was completed successfully")

class AgentObservation(BaseModel):
    """Observation made by an agent while executing an action"""
    content: str = Field(..., description="Content of the observation")
    source: Optional[str] = Field(None, description="Source of the observation")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the observation was made")

class AgentThought(BaseModel):
    """Thought process of an agent"""
    content: str = Field(..., description="Content of the thought")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the thought was made")

class AgentState(BaseModel):
    """State of an agent during execution"""
    actions: List[AgentAction] = Field(default_factory=list, description="Actions taken by the agent")
    observations: List[AgentObservation] = Field(default_factory=list, description="Observations made by the agent")
    thoughts: List[AgentThought] = Field(default_factory=list, description="Thoughts of the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information for the agent")

class AgentConfig(BaseModel):
    """Configuration for an agent"""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    model: str = Field("gpt-4", description="LLM model to use")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens in response")
    include_context: bool = Field(True, description="Whether to include RAG context")
    tools: List[str] = Field(default_factory=list, description="Tools available to the agent")
    context_query: Optional[str] = Field(None, description="Query to use for initial RAG context")
    retrieval_type: Optional[str] = Field("auto", description="Type of retrieval to use: 'hybrid', 'vector', or 'auto' (default)")
    max_context_results: int = Field(5, description="Maximum number of context results to retrieve")

class BaseAgent:
    """Base class for all agents in the system"""

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        rag_system: Optional[RAGSystem] = None,
    ):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration for the agent
            neo4j_uri: URI for Neo4j database
            neo4j_user: Username for Neo4j
            neo4j_password: Password for Neo4j
            openai_api_key: OpenAI API key
            qdrant_url: URL for Qdrant vector store
            rag_system: Optional pre-initialized RAG system to use
        """
        self.agent_id = agent_id
        self.config = config
        self.state = AgentState()

        # Neo4j connection
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_driver = None

        # OpenAI API
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

        # Qdrant URL
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")

        # Connect to Neo4j
        self._connect_to_neo4j()

        # Initialize or use provided RAG system
        self.rag_system = rag_system
        if not self.rag_system and self.config.include_context:
            self._initialize_rag_system()

    def _initialize_rag_system(self):
        """Initialize the RAG system for context retrieval"""
        try:
            logger.info(f"Agent {self.agent_id} initializing RAG system")
            self.rag_system = RAGSystem(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                qdrant_url=self.qdrant_url,
                openai_api_key=self.openai_api_key
            )
            logger.info(f"Agent {self.agent_id} RAG system initialized")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to initialize RAG system: {str(e)}")
            self.rag_system = None

    def _connect_to_neo4j(self):
        """Establish connection to Neo4j"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=3600
            )

            # Test connection
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 'Connected to Neo4j' AS message")
                for record in result:
                    logger.info(f"Agent {self.agent_id}: {record['message']}")

            logger.info(f"Agent {self.agent_id} connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to connect to Neo4j: {str(e)}")
            raise

    def add_thought(self, content: str) -> AgentThought:
        """Add a thought to the agent's state"""
        thought = AgentThought(content=content)
        self.state.thoughts.append(thought)
        return thought

    def add_observation(self, content: str, source: Optional[str] = None) -> AgentObservation:
        """Add an observation to the agent's state"""
        observation = AgentObservation(content=content, source=source)
        self.state.observations.append(observation)
        return observation

    def add_action(self, action_type: str, action_params: Dict[str, Any] = None) -> AgentAction:
        """Add an action to the agent's state"""
        if action_params is None:
            action_params = {}
        action = AgentAction(action_type=action_type, action_params=action_params)
        self.state.actions.append(action)
        return action

    def query_knowledge_graph(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query the knowledge graph (Neo4j)"""
        if params is None:
            params = {}

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, params)
                records = [record.data() for record in result]
                return records
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to query knowledge graph: {str(e)}")
            self.add_observation(f"Failed to query knowledge graph: {str(e)}", "error")
            return []

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve context from RAG system.

        Args:
            query: The query to search for
            top_k: Maximum number of results to return (default: config value)

        Returns:
            List of context results
        """
        if not self.rag_system:
            logger.warning(f"Agent {self.agent_id} has no RAG system to retrieve context")
            self.add_observation("No RAG system available for context retrieval", "warning")
            return []

        if top_k is None:
            top_k = self.config.max_context_results

        try:
            logger.info(f"Agent {self.agent_id} retrieving context for query: {query}")
            results = self.rag_system.retrieve(query, top_k=top_k)

            if results:
                self.add_observation(f"Retrieved {len(results)} context items from knowledge base", "rag")
            else:
                self.add_observation("No relevant context found in knowledge base", "rag")

            return results
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to retrieve context: {str(e)}")
            self.add_observation(f"Failed to retrieve context: {str(e)}", "error")
            return []

    def format_context_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context into a string for prompt.

        Args:
            results: List of context results from RAG

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context available."

        context_parts = []
        for i, result in enumerate(results):
            source = f"{result['metadata'].get('doc_name', 'Unknown document')}"
            if 'section' in result['metadata'] and result['metadata']['section'] != 'Unknown':
                source += f" - {result['metadata']['section']}"

            context_parts.append(f"Context {i+1}: [Source: {source}]\n{result['text'].strip()}")

        return "\n\n".join(context_parts)

    def generate_completion(self, prompt: str, include_context: bool = None) -> str:
        """Generate completion using OpenAI API with optional context enhancement"""
        if not self.openai_api_key:
            return "OpenAI API key not provided. Cannot generate response."

        try:
            # Determine if we should include context
            use_context = include_context if include_context is not None else self.config.include_context

            # Get context if needed
            context_str = ""
            if use_context and self.rag_system:
                # Use context_query if specified, otherwise use the prompt
                query = self.config.context_query or prompt

                # Get context from RAG
                context_results = self.retrieve_context(query)

                # Format context
                if context_results:
                    context_str = self.format_context_for_prompt(context_results)
                    self.add_thought(f"Using context for completion: {len(context_results)} items")

            # Construct the full prompt with context if available
            full_prompt = prompt
            if context_str and use_context:
                full_prompt = f"""
Please use the following context information to inform your response:

{context_str}

Based on the above context, please respond to:

{prompt}
"""

            # Log the prompt building process
            if context_str and use_context:
                self.add_thought("Enhanced prompt with relevant context")
            else:
                self.add_thought("Using prompt without additional context")

            # Call OpenAI API
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to generate completion: {str(e)}")
            self.add_observation(f"Failed to generate completion: {str(e)}", "error")
            return f"Error generating completion: {str(e)}"

    def execute(self, task: str) -> Dict[str, Any]:
        """
        Execute a task using the agent.
        This method should be overridden by subclasses.

        Args:
            task: Task description

        Returns:
            Result of the task execution
        """
        raise NotImplementedError("Subclasses must implement the execute method")

    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info(f"Agent {self.agent_id} Neo4j connection closed")

        if self.rag_system:
            self.rag_system.close()
            logger.info(f"Agent {self.agent_id} RAG system closed")


class RegulatoryAgent(BaseAgent):
    """Agent specializing in regulatory analysis"""

    def extract_regulatory_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract regulatory entities from text"""
        # First check for context in the knowledge base
        if self.config.include_context and self.rag_system:
            context_results = self.retrieve_context(f"regulatory entities {text}")
            if context_results:
                self.add_thought("Using knowledge base context for regulatory entity extraction")

        prompt = f"""
        Extract regulatory entities from the following text.
        Entities include: regulations, laws, compliance requirements, deadlines, and relevant authorities.
        For each entity, provide the name, type, and description.

        Text: {text}
        """

        completion = self.generate_completion(prompt)
        self.add_thought(f"Extracted regulatory entities: {completion}")

        # This is a simplified approach - in a real system you'd parse the completion
        # to extract structured entities
        return [{"content": completion}]

    def analyze_compliance_status(self, document_id: str) -> Dict[str, Any]:
        """Analyze compliance status of a document"""
        self.add_thought(f"Analyzing compliance status for document {document_id}")

        # Get document content from Neo4j
        document_data = self.query_knowledge_graph(
            """
            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
            RETURN c.text as text
            ORDER BY c.index
            """,
            {"doc_id": document_id}
        )

        if not document_data:
            self.add_observation(f"No content found for document {document_id}", "error")
            return {"status": "error", "message": "Document not found"}

        # Combine document chunks
        document_text = "\n\n".join([record.get("text", "") for record in document_data])

        # Retrieve additional context from RAG using the document as query
        additional_context = ""
        if self.config.include_context and self.rag_system:
            # Extract key terms from document for better context retrieval
            key_terms_prompt = f"""
            Extract 3-5 key regulatory terms or concepts from this document excerpt for compliance analysis:

            {document_text[:2000]}  # Sample of document for key terms
            """

            key_terms = self.generate_completion(key_terms_prompt, include_context=False)
            self.add_thought(f"Extracted key terms for context search: {key_terms}")

            # Use key terms to search for relevant context
            rag_results = self.retrieve_context(f"compliance requirements {key_terms}")

            if rag_results:
                context_texts = []
                for result in rag_results:
                    source = f"{result['metadata'].get('doc_name', 'Unknown document')}"
                    if 'section' in result['metadata']:
                        source += f" - {result['metadata']['section']}"
                    context_texts.append(f"SOURCE: {source}\n{result['text']}")

                additional_context = "\n\n".join(context_texts)
                self.add_observation(f"Found {len(rag_results)} relevant compliance documents in knowledge base", "rag")

        # Generate compliance analysis
        prompt = f"""
        Analyze the following document for regulatory compliance.
        Identify:
        1. Key compliance requirements
        2. Compliance status (compliant, non-compliant, unclear)
        3. Key risks and gaps
        4. Recommended actions

        Document: {document_text[:8000]}  # Truncate to avoid token limits
        """

        # Add additional context if available
        if additional_context:
            prompt += f"""

            Additional relevant compliance information from knowledge base:
            {additional_context}
            """

        analysis = self.generate_completion(prompt, include_context=False)  # We already included context manually
        self.add_thought(f"Generated compliance analysis")
        self.add_observation(analysis, "compliance_analysis")

        return {
            "status": "success",
            "document_id": document_id,
            "analysis": analysis
        }

    def execute(self, task: str) -> Dict[str, Any]:
        """Execute a regulatory task"""
        self.add_thought(f"Received task: {task}")

        # Determine task type
        if "extract" in task.lower() and "entities" in task.lower():
            # Extract entities from text
            self.add_thought("Task identified as entity extraction")
            text = task.split("from:", 1)[1].strip() if "from:" in task else task
            entities = self.extract_regulatory_entities(text)
            return {"status": "success", "task_type": "entity_extraction", "entities": entities}

        elif "analyze" in task.lower() and "compliance" in task.lower():
            # Analyze compliance status
            self.add_thought("Task identified as compliance analysis")
            doc_id = None
            if "document:" in task.lower():
                doc_id = task.split("document:", 1)[1].strip()
            elif "doc_id:" in task.lower():
                doc_id = task.split("doc_id:", 1)[1].strip()

            if doc_id:
                return self.analyze_compliance_status(doc_id)
            else:
                return {"status": "error", "message": "No document ID provided for compliance analysis"}

        else:
            # Default to general regulatory analysis with RAG context
            self.add_thought("Task identified as general regulatory query")

            # This task type always benefits from context
            analysis = self.generate_completion(task)
            self.add_observation(analysis, "general_analysis")
            return {"status": "success", "task_type": "general_analysis", "analysis": analysis}


class ResearchAgent(BaseAgent):
    """Agent specializing in regulatory research"""

    def search_knowledge_base(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base for information related to the query"""
        # Use RAG system if available for better semantic search
        if self.rag_system:
            self.add_thought(f"Using RAG system to search for: '{query}'")
            search_results = self.retrieve_context(query, top_k=limit)

            if search_results:
                self.add_observation(f"Found {len(search_results)} relevant results for '{query}'", "rag")
                return search_results
            else:
                self.add_observation(f"No relevant results found for '{query}' using RAG", "rag")

        # Fallback to Neo4j text search if RAG fails or is unavailable
        self.add_thought("Performing Neo4j text search as fallback")
        related_chunks = self.query_knowledge_graph(
            """
            MATCH (c:Chunk)
            WHERE c.text CONTAINS $query
            RETURN c.chunk_id as id, c.text as text, c.section as section
            LIMIT $limit
            """,
            {"query": query, "limit": limit}
        )

        self.add_thought(f"Found {len(related_chunks)} chunks related to '{query}' using Neo4j")

        # Format the results to match RAG result structure
        formatted_results = []
        for chunk in related_chunks:
            formatted_results.append({
                "text": chunk.get("text", ""),
                "metadata": {
                    "chunk_id": chunk.get("id", ""),
                    "section": chunk.get("section", "Unknown"),
                    "doc_name": "Unknown"  # Neo4j query doesn't provide doc_name
                }
            })

        self.add_observation(f"Search results for '{query}': {len(formatted_results)} results", "neo4j")
        return formatted_results

    def summarize_research(self, chunks: List[Dict[str, Any]]) -> str:
        """Summarize research findings from document chunks"""
        if not chunks:
            return "No relevant information found."

        # Prepare text from chunks
        texts = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict) and "text" in chunk:
                # RAG system format
                source = ""
                if "metadata" in chunk:
                    source = f"{chunk['metadata'].get('doc_name', 'Unknown document')}"
                    if 'section' in chunk['metadata'] and chunk['metadata']['section'] != 'Unknown':
                        source += f" - {chunk['metadata']['section']}"
                texts.append(f"Chunk {i+1} (Source: {source}): {chunk['text']}")
            else:
                # Neo4j format (fallback)
                section = chunk.get('section', 'Unknown')
                texts.append(f"Chunk {i+1} (Section: {section}): {chunk.get('text', '')}")

        combined_text = "\n\n".join(texts)

        prompt = f"""
        Summarize the following information into a cohesive research summary.
        Focus on key regulatory requirements, insights, and implications.

        {combined_text[:8000]}  # Truncate to avoid token limits
        """

        # Generate the summary without additional context (we've already included the relevant chunks)
        summary = self.generate_completion(prompt, include_context=False)
        self.add_thought("Generated research summary")

        return summary

    def execute(self, task: str) -> Dict[str, Any]:
        """Execute a research task"""
        self.add_thought(f"Received task: {task}")

        # Extract the research query
        query = task
        limit = 10  # Default limit

        # Check if there's a specific limit requested
        if "limit:" in task.lower():
            parts = task.lower().split("limit:", 1)
            query = parts[0].strip()
            try:
                limit = int(parts[1].strip())
            except ValueError:
                pass

        # Analyze the query to identify key topics
        analysis_prompt = f"""
        Identify the main research topics and key terms in this query:

        {query}

        Return only 3-5 key terms or phrases that would be most useful for searching a regulatory knowledge base.
        """

        key_terms = self.generate_completion(analysis_prompt, include_context=False)
        self.add_thought(f"Identified key search terms: {key_terms}")

        # Use the key terms for a more focused search
        search_results = self.search_knowledge_base(f"{query} {key_terms}", limit)

        # If we got too few results, try with just the original query
        if len(search_results) < 3 and len(key_terms) > 0:
            self.add_thought("Initial search returned few results, trying with original query")
            additional_results = self.search_knowledge_base(query, limit - len(search_results))

            # Add any new results that aren't duplicates
            existing_ids = {result.get("metadata", {}).get("chunk_id", "") for result in search_results}
            for result in additional_results:
                if result.get("metadata", {}).get("chunk_id", "") not in existing_ids:
                    search_results.append(result)

            self.add_thought(f"Combined search found {len(search_results)} results")

        # Generate a summary of findings
        summary = self.summarize_research(search_results)
        self.add_observation(summary, "research_summary")

        # Format detailed results
        detailed_results = []
        for result in search_results:
            if "metadata" in result:
                # RAG format
                formatted_result = {
                    "text": result["text"],
                    "source": result["metadata"].get("doc_name", "Unknown document"),
                    "section": result["metadata"].get("section", "Unknown")
                }
            else:
                # Neo4j format (fallback)
                formatted_result = result

            detailed_results.append(formatted_result)

        return {
            "status": "success",
            "task_type": "research",
            "query": query,
            "result_count": len(search_results),
            "summary": summary,
            "detailed_results": detailed_results
        }


# Factory function to create agents
def create_agent(
    agent_type: str,
    agent_id: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    rag_system: Optional[RAGSystem] = None,
    **kwargs
) -> BaseAgent:
    """
    Create an agent of the specified type.

    Args:
        agent_type: Type of agent to create
        agent_id: ID for the agent (generated if not provided)
        config: Configuration for the agent
        rag_system: Optional pre-initialized RAG system to use
        **kwargs: Additional arguments to pass to the agent constructor

    Returns:
        An initialized agent
    """
    if agent_id is None:
        agent_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    if config is None:
        config = AgentConfig(
            name=f"{agent_type.capitalize()} Agent",
            description=f"A {agent_type} agent for regulatory AI"
        )

    agent_classes = {
        "regulatory": RegulatoryAgent,
        "research": ResearchAgent,
        "tree_reasoning": TreeReasoningAgent,
        # Add more agent types here
    }

    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        logger.error(f"Unknown agent type: {agent_type}")
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Special handling for tree reasoning agent
    if agent_type.lower() == "tree_reasoning" and "decision_tree" not in kwargs:
        # Create a default decision tree if none provided
        kwargs["decision_tree"] = create_default_decision_tree()

    return agent_class(agent_id=agent_id, config=config, rag_system=rag_system, **kwargs)
