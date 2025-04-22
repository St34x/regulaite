"""
Parsing Quality Evaluation
==========================

This script evaluates how the quality of document parsing influences 
the quality of responses from different agents in the RegulAIte system.

It compares responses from different agents when using documents parsed with:
- Doctly API
- LlamaParse API
- Unstructured API
- Unstructured local docker container

The evaluation runs a set of predefined test queries against each agent type
and collects metrics on response quality.
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI

# Add the parent directory to the path to import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components from the backend
from pyndantic_agents.base_agent import BaseAgent, AgentInput, AgentOutput
from pyndantic_agents.agent_factory import create_agent
from pyndantic_agents.rag_agent import RAGAgent
from pyndantic_agents.cybersecurity_agents import (
    VulnerabilityAssessmentAgent, 
    ComplianceMappingAgent,
    ThreatModelingAgent
)
from pyndantic_agents.tree_reasoning import TreeReasoningAgent
from pyndantic_agents.dynamic_decision_trees import DynamicTreeAgent
from llamaIndex_rag.rag import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parsing_quality_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Agent types to test
AGENT_TYPES = [
    "regulatory",
    "research",
    "tree_reasoning",
    "dynamic_tree",
    "vulnerability_assessment",
    "compliance_mapping",
    "threat_modeling"
]

# Parsing methods used
PARSING_METHODS = [
    "doctly",
    "llamaparse",
    "unstructured_api", 
    "unstructured_local"
]

# Test queries by category
TEST_QUERIES = {
    "regulatory": [
        "What are our key compliance obligations based on our internal policies?",
        "How does our data protection policy align with GDPR requirements?",
        "What controls should we implement to meet ISO 27001 requirements?",
        "Explain how our organization handles risk assessment according to internal policies",
        "What are the consequences of non-compliance with our security policies?"
    ],
    "risk_assessment": [
        "What vulnerabilities were identified in the EBIOS RM report?",
        "What are the highest risk threats in our environment according to our documentation?",
        "How do we address supply chain risks according to our policies?", 
        "What is our risk acceptance threshold according to documentation?",
        "What mitigation strategies are recommended for the top threats?"
    ],
    "threat_modeling": [
        "What are the key attack vectors identified in our documents?",
        "How do we prioritize security threats according to our methodology?",
        "What countermeasures are suggested for phishing attacks?",
        "Explain our defense-in-depth strategy based on documentation",
        "How do we assess emerging threats according to our framework?"
    ],
    "compliance_mapping": [
        "Map our internal security controls to NIST CSF framework",
        "How do our policies align with ISO 27001 controls?",
        "What gaps exist between our controls and regulatory requirements?",
        "Which compliance requirements are addressed by our data backup policy?",
        "Create a mapping between our access control policy and PCI DSS requirements"
    ]
}

class EvaluationMetrics:
    """Class to store evaluation metrics for a test run"""
    def __init__(self):
        self.results = []
        
    def add_result(self, 
                  agent_type: str,
                  parsing_method: str, 
                  query: str,
                  query_category: str,
                  response: str,
                  response_time: float,
                  evaluation_score: Optional[float] = None,
                  evaluation_feedback: Optional[str] = None,
                  tokens_used: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """Add a test result to the metrics collection"""
        result = {
            "agent_type": agent_type,
            "parsing_method": parsing_method,
            "query": query,
            "query_category": query_category,
            "response": response,
            "response_time": response_time,
            "evaluation_score": evaluation_score,
            "evaluation_feedback": evaluation_feedback,
            "tokens_used": tokens_used,
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            result.update(metadata)
            
        self.results.append(result)
        
    def save_to_json(self, filename: str):
        """Save results to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved evaluation results to {filename}")
        
    def save_to_csv(self, filename: str):
        """Save results to a CSV file"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        logger.info(f"Saved evaluation results to {filename}")
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the evaluation results"""
        df = pd.DataFrame(self.results)
        
        summary = {
            "total_tests": len(df),
            "avg_score_by_agent": df.groupby("agent_type")["evaluation_score"].mean().to_dict(),
            "avg_score_by_parsing": df.groupby("parsing_method")["evaluation_score"].mean().to_dict(),
            "avg_score_by_category": df.groupby("query_category")["evaluation_score"].mean().to_dict(),
            "avg_time_by_agent": df.groupby("agent_type")["response_time"].mean().to_dict(),
            "avg_time_by_parsing": df.groupby("parsing_method")["response_time"].mean().to_dict(),
            "best_agent_parsing_combo": df.groupby(["agent_type", "parsing_method"])["evaluation_score"].mean().idxmax(),
            "worst_agent_parsing_combo": df.groupby(["agent_type", "parsing_method"])["evaluation_score"].mean().idxmin(),
        }
        
        return summary

class EvaluationAgent:
    """Agent for evaluating the quality of responses from other agents"""
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 verbose: bool = False):
        """Initialize the evaluation agent"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        self.verbose = verbose
        logger.info(f"Initialized EvaluationAgent with model {self.model}")
        
    async def evaluate_response(self, 
                               query: str, 
                               response: str, 
                               agent_type: str,
                               category: str) -> Tuple[float, str]:
        """
        Evaluate the quality of a response using LLM.
        
        Args:
            query: The original query
            response: The agent's response
            agent_type: The type of agent that generated the response
            category: The category of the query
            
        Returns:
            Tuple of (score from 0-10, feedback as string)
        """
        system_prompt = """You are an expert evaluator for AI assistant responses in the Governance, Risk and Compliance (GRC) domain.
Your job is to rate responses from various AI agents on a scale of 0 to 10 based on:

1. Accuracy (0-3 points): Is the information factually correct?
2. Relevance (0-2 points): Does the response address the specific query?
3. Completeness (0-2 points): Does the response provide comprehensive information?
4. Clarity (0-1 point): Is the response well-structured and easy to understand?
5. Actionability (0-2 points): Does the response provide practical, actionable insights?

Provide a detailed explanation for your rating, mentioning strengths and weaknesses.
Respond ONLY with a JSON object with the following structure:
{
  "score": X.X,
  "feedback": "Detailed explanation of the rating",
  "breakdown": {
    "accuracy": X,
    "relevance": X,
    "completeness": X, 
    "clarity": X,
    "actionability": X
  }
}"""

        user_prompt = f"""I need you to evaluate a response from a {agent_type} agent to a query in the {category} category.

Query: {query}

Response: {response}

Please rate this response on our 0-10 scale with a detailed explanation."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            evaluation = json.loads(content)
            
            if self.verbose:
                logger.info(f"Evaluation for {agent_type} on {category} query:")
                logger.info(f"Score: {evaluation['score']}/10")
                logger.info(f"Feedback: {evaluation['feedback']}")
                logger.info(f"Breakdown: {evaluation['breakdown']}")
                
            return evaluation["score"], evaluation["feedback"]
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return 0.0, f"Evaluation failed: {str(e)}"

class TestRunner:
    """Class for running evaluation tests"""
    def __init__(self, 
                 rag_system: RAGSystem,
                 evaluator: EvaluationAgent,
                 metrics: EvaluationMetrics,
                 openai_api_key: Optional[str] = None,
                 agent_model: str = "gpt-4",
                 output_dir: str = "./evaluation_results",
                 verbose: bool = False):
        """Initialize the test runner"""
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.metrics = metrics
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.agent_model = agent_model
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Track currently running test info
        self.current_test = {
            "agent_type": None,
            "parsing_method": None,
            "query": None,
            "category": None
        }
        
        logger.info(f"Initialized TestRunner with model {agent_model}")
        
    async def run_tests(self, 
                       agent_types: Optional[List[str]] = None,
                       parsing_methods: Optional[List[str]] = None,
                       query_categories: Optional[List[str]] = None):
        """
        Run evaluation tests for specified agent types and parsing methods.
        
        Args:
            agent_types: List of agent types to test (defaults to all)
            parsing_methods: List of parsing methods to test (defaults to all)
            query_categories: List of query categories to test (defaults to all)
        """
        agent_types = agent_types or AGENT_TYPES
        parsing_methods = parsing_methods or PARSING_METHODS
        query_categories = query_categories or list(TEST_QUERIES.keys())
        
        # Run tests for all combinations
        for agent_type in agent_types:
            for parsing_method in parsing_methods:
                # Set the active parsing method in the RAG system
                # This would depend on how your system handles different parsing methods
                # Here assuming there's a configuration or metadata that controls this
                await self._set_active_parsing_method(parsing_method)
                
                for category in query_categories:
                    if category in TEST_QUERIES:
                        for query in TEST_QUERIES[category]:
                            self.current_test = {
                                "agent_type": agent_type,
                                "parsing_method": parsing_method,
                                "query": query,
                                "category": category
                            }
                            
                            logger.info(f"Testing {agent_type} agent with {parsing_method} parsing on {category} query")
                            
                            # Run the test
                            await self._run_single_test(agent_type, parsing_method, query, category)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics.save_to_json(f"{self.output_dir}/results_{timestamp}.json")
        self.metrics.save_to_csv(f"{self.output_dir}/results_{timestamp}.csv")
        
        # Generate and save summary
        summary = self.metrics.generate_summary()
        with open(f"{self.output_dir}/summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
    
    async def _set_active_parsing_method(self, parsing_method: str):
        """
        Set the active parsing method in the RAG system.
        This is a placeholder - implement according to your system.
        
        Args:
            parsing_method: The parsing method to activate
        """
        # This is a placeholder - in an actual implementation, you would
        # update the RAG system configuration or metadata to use the specified
        # parsing method for retrieving documents
        logger.info(f"Setting active parsing method to {parsing_method}")
        # Example: await self.rag_system.set_parsing_filter(parsing_method)
    
    async def _run_single_test(self, 
                              agent_type: str,
                              parsing_method: str,
                              query: str,
                              category: str):
        """
        Run a single test case.
        
        Args:
            agent_type: Type of agent to test
            parsing_method: Parsing method used for documents
            query: Query to test
            category: Category of the query
        """
        try:
            # Create agent
            agent = await self._create_agent(agent_type)
            
            # Track response time
            start_time = time.time()
            
            # Create agent input
            agent_input = AgentInput(
                query=query,
                # Any other parameters needed for your agents
            )
            
            # Process query with agent
            response = await agent.process(agent_input)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract response text
            response_text = response.response if hasattr(response, "response") else str(response)
            
            # Evaluate response
            score, feedback = await self.evaluator.evaluate_response(
                query, response_text, agent_type, category
            )
            
            # Extract token usage if available
            tokens_used = None
            if hasattr(response, "token_usage") and response.token_usage:
                tokens_used = response.token_usage.get("total_tokens")
            
            # Collect metadata if available
            metadata = {}
            if hasattr(response, "metadata"):
                metadata["agent_metadata"] = response.metadata
                
            # Add result to metrics
            self.metrics.add_result(
                agent_type=agent_type,
                parsing_method=parsing_method,
                query=query,
                query_category=category,
                response=response_text,
                response_time=response_time,
                evaluation_score=score,
                evaluation_feedback=feedback,
                tokens_used=tokens_used,
                metadata=metadata
            )
            
            logger.info(f"Test completed: {agent_type} - {parsing_method} - Score: {score}/10")
            
        except Exception as e:
            logger.error(f"Error running test: {str(e)}")
            # Record the error as a result
            self.metrics.add_result(
                agent_type=agent_type,
                parsing_method=parsing_method,
                query=query,
                query_category=category,
                response=f"ERROR: {str(e)}",
                response_time=0.0,
                evaluation_score=0.0,
                evaluation_feedback=f"Test failed with error: {str(e)}"
            )
    
    async def _create_agent(self, agent_type: str) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            
        Returns:
            Initialized agent
        """
        # Create agent based on type
        if agent_type == "regulatory":
            from pyndantic_agents.agents import RegulatoryAgent
            agent = RegulatoryAgent(
                agent_id=f"test_{agent_type}",
                config={
                    "name": "Regulatory Agent",
                    "description": "Agent specialized in regulatory compliance questions",
                    "model": self.agent_model,
                    "include_context": True,
                },
                rag_system=self.rag_system,
                openai_api_key=self.openai_api_key
            )
        elif agent_type == "research":
            from pyndantic_agents.agents import ResearchAgent
            agent = ResearchAgent(
                agent_id=f"test_{agent_type}",
                config={
                    "name": "Research Agent",
                    "description": "Agent specialized in research and analysis questions",
                    "model": self.agent_model,
                    "include_context": True,
                },
                rag_system=self.rag_system,
                openai_api_key=self.openai_api_key
            )
        elif agent_type == "tree_reasoning":
            from pyndantic_agents.decision_trees import get_default_tree
            agent = TreeReasoningAgent(
                tree=get_default_tree(),
                openai_api_key=self.openai_api_key,
                model=self.agent_model
            )
        elif agent_type == "dynamic_tree":
            agent = DynamicTreeAgent(
                openai_api_key=self.openai_api_key,
                model=self.agent_model,
                rag_system=self.rag_system
            )
        elif agent_type == "vulnerability_assessment":
            agent = VulnerabilityAssessmentAgent(
                rag_system=self.rag_system,
                openai_api_key=self.openai_api_key,
                model=self.agent_model
            )
        elif agent_type == "compliance_mapping":
            agent = ComplianceMappingAgent(
                rag_system=self.rag_system,
                openai_api_key=self.openai_api_key,
                model=self.agent_model
            )
        elif agent_type == "threat_modeling":
            agent = ThreatModelingAgent(
                rag_system=self.rag_system,
                openai_api_key=self.openai_api_key,
                model=self.agent_model
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent

async def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate parsing quality impact on agent responses")
    parser.add_argument("--agents", nargs="+", choices=AGENT_TYPES, 
                        help="Agent types to test (default: all)")
    parser.add_argument("--parsing-methods", nargs="+", choices=PARSING_METHODS,
                        help="Parsing methods to test (default: all)")
    parser.add_argument("--categories", nargs="+", choices=list(TEST_QUERIES.keys()),
                        help="Query categories to test (default: all)")
    parser.add_argument("--output-dir", default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--agent-model", default="gpt-4",
                        help="Model to use for agents")
    parser.add_argument("--evaluator-model", default="gpt-4",
                        help="Model to use for evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    # Get environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Initialize RAG system
    rag_system = RAGSystem(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        qdrant_url=qdrant_url,
        openai_api_key=openai_api_key,
        hybrid_search=True,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    # Initialize evaluator
    evaluator = EvaluationAgent(
        openai_api_key=openai_api_key,
        model=args.evaluator_model,
        verbose=args.verbose
    )
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    
    # Initialize and run tests
    runner = TestRunner(
        rag_system=rag_system,
        evaluator=evaluator,
        metrics=metrics,
        openai_api_key=openai_api_key,
        agent_model=args.agent_model,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Run tests
    summary = await runner.run_tests(
        agent_types=args.agents,
        parsing_methods=args.parsing_methods,
        query_categories=args.categories
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))
    
    # Generate visualizations if pandas and matplotlib are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create directory for visualizations
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load results into DataFrame
        df = pd.DataFrame(metrics.results)
        
        # Heatmap of agent performance by parsing method
        plt.figure(figsize=(12, 8))
        heatmap_data = df.pivot_table(
            values="evaluation_score", 
            index="agent_type", 
            columns="parsing_method", 
            aggfunc="mean"
        )
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Average Evaluation Score by Agent and Parsing Method")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "agent_parsing_heatmap.png"))
        
        # Boxplot of scores by parsing method
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="parsing_method", y="evaluation_score", data=df)
        plt.title("Distribution of Evaluation Scores by Parsing Method")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "parsing_score_boxplot.png"))
        
        # Response time by agent type
        plt.figure(figsize=(10, 6))
        sns.barplot(x="agent_type", y="response_time", data=df)
        plt.title("Average Response Time by Agent Type")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "agent_response_time.png"))
        
        print(f"Visualizations saved to {viz_dir}")
        
    except ImportError:
        print("Matplotlib and/or seaborn not available. Skipping visualizations.")

if __name__ == "__main__":
    asyncio.run(main()) 