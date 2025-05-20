import logging
import json
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

from pydantic import BaseModel, Field

from .base_node import BaseProcessingNode, ProcessingStepResult
from .rag_retriever import RetrievedDocument # Input from RAG node
from ..graph_components.nodes import QueryNode, DocumentNode
from ..integration_components.graph_interface import GraphInterface
from llm_services.llm_client import LLMClient # For LLM-based evaluation

logger = logging.getLogger(__name__)

class EvaluationCriteria(str, Enum):
    """Criteria used to evaluate retrieved documents."""
    RELEVANCE = "relevance"
    FACTUALITY = "factuality"
    RECENCY = "recency"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"

class DocumentEvaluation(BaseModel):
    document: RetrievedDocument
    scores: Dict[str, float] = Field(default_factory=dict)
    overall_score: float
    notes: Optional[str] = None

class EvaluatedDocument(RetrievedDocument):
    # Potentially add more evaluation-specific fields if needed
    final_relevance_score: Optional[float] = None # If different from initial retrieval score
    # Contains snippets from the document that are most relevant to the query
    relevant_snippets: List[str] = []

class ResultEvaluationOutput(ProcessingStepResult):
    """Output from the result evaluation step."""
    step_name: str = "result_evaluation"
    status: str
    is_sufficient: bool
    needs_reformulation: bool  # Explicit flag for query reformulation
    information_quality_score: float
    information_gaps: List[str]
    evaluated_documents: List[EvaluatedDocument]
    evaluation_rationale: str
    related_concepts: Optional[List[Dict[str, Any]]] = None

class ResultEvaluatorInput(BaseModel):
    current_query_node: QueryNode
    retrieved_documents: List[RetrievedDocument]
    # Threshold for determining sufficiency, could be configurable
    sufficiency_threshold: float = 0.7 

class ResultEvaluatorOutput(BaseModel):
    evaluated_documents: List[EvaluatedDocument]
    # Overall assessment of whether the retrieved info is sufficient
    is_sufficient: bool
    # If not sufficient, what information is missing or unclear
    information_gaps: List[str] = [] 
    # Decision on whether to proceed to response generation or reformulate query
    requires_reformulation: bool
    needs_reformulation: bool = Field(default=None, description="Alias for requires_reformulation for workflow compatibility")
    # Confidence in the current set of documents to answer the query
    overall_confidence: float
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure needs_reformulation is always set to match requires_reformulation
        if self.needs_reformulation is None:
            self.needs_reformulation = self.requires_reformulation

class ResultEvaluationNode(BaseProcessingNode):
    """Assess the quality and relevance of retrieved information."""
    
    def __init__(self, node_config: Optional[Dict[str, Any]] = None):
        super().__init__(node_config)
        # Configure default parameters
        config = node_config or {}
        self.use_llm_evaluation = config.get("use_llm_evaluation", True)
        self.min_documents_required = config.get("min_documents_required", 1)
        self.evaluation_temperature = config.get("evaluation_temperature", 0.2)
        self.max_evaluation_tokens = config.get("max_evaluation_tokens", 300)
        self.snippet_limit = config.get("snippet_limit", 3)

    async def execute(self, input_data: Any, context: Dict[str, Any]) -> ResultEvaluationOutput:
        """
        Execute the result evaluation step.
        
        Args:
            input_data: Either a RAGRetrievalOutput or dict with retrieved_documents and query_node
            context: Workflow context
            
        Returns:
            Evaluation of the retrieval results
        """
        self.llm_client = context.get("llm_client")
        if not self.llm_client:
            raise ValueError("LLM client is required for result evaluation")
            
        # Get the graph interface from context
        self.graph_interface = context.get("graph_interface")
        
        # Extract retrieved documents and query from input
        retrieval_result, query_node = self._extract_from_input(input_data)
        
        # Standardize documents for evaluation
        documents_to_evaluate = self._standardize_documents(retrieval_result)
        
        # Evaluate each document
        evaluated_docs = await self._evaluate_documents(documents_to_evaluate, query_node)
        
        # Analyze information gaps
        gaps_analysis = await self._analyze_information_gaps(evaluated_docs, query_node)
        
        # Determine if the information is sufficient
        information_quality_score = self._calculate_information_quality(evaluated_docs)
        is_sufficient = information_quality_score >= self.config.get("sufficient_quality_threshold", 0.7)
        
        # Determine if query reformulation is needed
        # We need reformulation if:
        # 1. Information is not sufficient (score below threshold)
        # 2. There are significant information gaps identified
        # 3. No relevant documents were found
        significant_gaps = len(gaps_analysis["information_gaps"]) > 0
        no_relevant_docs = all(doc.relevance_score < 0.5 for doc in evaluated_docs)
        needs_reformulation = not is_sufficient or significant_gaps or no_relevant_docs
        
        # Find related concepts that could help with query reformulation
        related_concepts = []
        if needs_reformulation and self.graph_interface:
            related_concepts = await self._get_related_concepts(query_node, evaluated_docs)
        
        return ResultEvaluationOutput(
            status="completed",
            is_sufficient=is_sufficient,
            needs_reformulation=needs_reformulation,
            information_quality_score=information_quality_score,
            information_gaps=gaps_analysis["information_gaps"],
            evaluated_documents=evaluated_docs,
            evaluation_rationale=gaps_analysis["evaluation_rationale"],
            related_concepts=related_concepts
        ) 

    def _extract_from_input(self, input_data):
        """Extract retrieval result and query node from the input data."""
        if hasattr(input_data, 'retrieved_documents') and hasattr(input_data, 'query_node'):
            # Direct input with the expected fields
            return input_data.retrieved_documents, input_data.query_node
        elif hasattr(input_data, 'documents') and hasattr(input_data, 'query'):
            # Input from RAGRetrievalOutput
            return input_data.documents, input_data.query
        elif isinstance(input_data, dict):
            # Dictionary input - look for the right keys
            retrieval_result = input_data.get('retrieved_documents', input_data.get('documents', []))
            query_node = input_data.get('query_node', input_data.get('query', None))
            return retrieval_result, query_node
        else:
            # Unsupported input format
            logger.error(f"Unsupported input format for result evaluator: {type(input_data)}")
            raise ValueError(f"Unsupported input format for result evaluator: {type(input_data)}")
    
    def _standardize_documents(self, documents):
        """Standardize documents to a consistent format for evaluation."""
        standardized_docs = []
        for doc in documents:
            # If it's already an EvaluatedDocument, use it as is
            if isinstance(doc, EvaluatedDocument):
                standardized_docs.append(doc)
                continue
                
            # If it's a dict or has appropriate attributes, convert to EvaluatedDocument
            doc_id = getattr(doc, 'id', getattr(doc, 'document_id', None))
            if not doc_id and isinstance(doc, dict):
                doc_id = doc.get('id', doc.get('document_id', None))
                
            content = getattr(doc, 'content', getattr(doc, 'text_content', None))
            if not content and isinstance(doc, dict):
                content = doc.get('content', doc.get('text_content', None))
                
            relevance = getattr(doc, 'relevance_score', 0.0)
            if isinstance(doc, dict):
                relevance = doc.get('relevance_score', relevance)
                
            metadata = getattr(doc, 'metadata', {})
            if isinstance(doc, dict) and 'metadata' in doc:
                metadata = doc['metadata']
                
            # Create a standardized document
            std_doc = EvaluatedDocument(
                document_id=doc_id,
                content=content,
                relevance_score=relevance,
                metadata=metadata,
                relevant_snippets=[]
            )
            standardized_docs.append(std_doc)
            
        return standardized_docs
    
    async def _evaluate_documents(self, documents, query_node):
        """Evaluate the relevance of documents to the query and extract relevant snippets."""
        if not documents:
            logger.warning("No documents to evaluate")
            return []
            
        if not self.llm_client:
            logger.warning("No LLM client available for document evaluation")
            return documents
            
        query_text = getattr(query_node, 'query_text', 
                     getattr(query_node, 'reformulated_query_text', 
                     getattr(query_node, 'original_user_input', str(query_node))))
            
        evaluated_docs = []
        for doc in documents:
            try:
                # Skip evaluation if no content
                if not doc.content:
                    doc.relevance_score = 0.0
                    doc.relevant_snippets = []
                    evaluated_docs.append(doc)
                    continue
                    
                # Limit content length for evaluation
                content_for_eval = doc.content[:5000] + ("..." if len(doc.content) > 5000 else "")
                
                # Create evaluation prompt
                eval_prompt = f"""
                Query: {query_text}
                
                Document Content:
                {content_for_eval}
                
                Task:
                1. Assess the relevance of this document to the query on a scale of 0.0 to 1.0
                2. Extract up to 3 most relevant snippets (sentences or short paragraphs)
                3. Explain briefly why this document is or isn't relevant
                
                Respond in JSON format with these fields:
                - relevance_score: float between 0 and 1
                - relevant_snippets: array of strings
                - reasoning: string explaining the relevance assessment
                """
                
                # Get LLM evaluation
                response = await self.llm_client.chat_completion([
                    {"role": "system", "content": "You are an expert document evaluator. Assess document relevance to queries accurately."},
                    {"role": "user", "content": eval_prompt}
                ])
                
                evaluation = {}
                try:
                    # Parse the response
                    response_text = response['choices'][0]['message']['content']
                    
                    # Extract JSON if enclosed in markdown or other text
                    import json
                    import re
                    
                    # Try to find JSON block
                    json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        # If no code block, try to find JSON object directly
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}')
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end+1]
                        else:
                            json_str = response_text
                    
                    evaluation = json.loads(json_str)
                except Exception as e:
                    logger.error(f"Failed to parse LLM evaluation response: {e}")
                    evaluation = {
                        "relevance_score": 0.0,
                        "relevant_snippets": [],
                        "reasoning": "Failed to parse evaluation"
                    }
                
                # Update document with evaluation results
                doc.relevance_score = evaluation.get('relevance_score', doc.relevance_score)
                doc.relevant_snippets = evaluation.get('relevant_snippets', [])
                doc.evaluation_notes = evaluation.get('reasoning', "")
                
                evaluated_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error evaluating document {doc.document_id}: {e}")
                # Add the document with original values
                evaluated_docs.append(doc)
        
        # Sort by relevance score
        evaluated_docs.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)
        return evaluated_docs
    
    async def _analyze_information_gaps(self, evaluated_docs, query_node):
        """Analyze information gaps in the retrieved documents."""
        if not self.llm_client:
            return {
                "information_gaps": ["Unable to analyze information gaps without LLM client"],
                "evaluation_rationale": "No LLM client available"
            }
            
        # Get query text
        query_text = getattr(query_node, 'query_text', 
                     getattr(query_node, 'reformulated_query_text', 
                     getattr(query_node, 'original_user_input', str(query_node))))
        
        # Prepare summaries of the retrieved documents
        doc_summaries = []
        for i, doc in enumerate(evaluated_docs[:5]):  # Top 5 docs for analysis
            snippets = "\n".join([f"- {s}" for s in doc.relevant_snippets[:2]])
            if not snippets:
                snippets = f"- {doc.content[:200]}..." if doc.content else "- No content available"
            
            doc_summaries.append(f"Document {i+1} (relevance: {doc.relevance_score:.2f}):\n{snippets}")
        
        summary_text = "\n\n".join(doc_summaries)
        
        # Prepare the gap analysis prompt
        gap_prompt = f"""
        Query: {query_text}
        
        Retrieved Information:
        {summary_text if summary_text else "No relevant documents found."}
        
        Based on the query and the retrieved information, please analyze:
        
        1. What specific information is missing to fully answer the query?
        2. Are there any important aspects of the query not covered by the retrieved documents?
        3. Is the retrieved information sufficient to provide a comprehensive answer?
        
        Provide your analysis in JSON format with these fields:
        - information_gaps: array of strings describing specific missing information
        - is_sufficient: boolean indicating whether the retrieved information is sufficient
        - evaluation_rationale: string explaining your overall assessment
        """
        
        try:
            # Get LLM analysis
            response = await self.llm_client.chat_completion([
                {"role": "system", "content": "You are an expert research analyst. Identify information gaps in research results."},
                {"role": "user", "content": gap_prompt}
            ])
            
            analysis = {}
            try:
                # Parse the response
                response_text = response['choices'][0]['message']['content']
                
                # Extract JSON
                import json
                import re
                
                # Try to find JSON block
                json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # If no code block, try to find JSON object directly
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end+1]
                    else:
                        json_str = response_text
                
                analysis = json.loads(json_str)
            except Exception as e:
                logger.error(f"Failed to parse LLM gap analysis response: {e}")
                # Create default analysis
                if not evaluated_docs:
                    analysis = {
                        "information_gaps": ["No documents retrieved for the query"],
                        "is_sufficient": False,
                        "evaluation_rationale": "No information available to evaluate"
                    }
                else:
                    analysis = {
                        "information_gaps": ["Unable to properly analyze information gaps"],
                        "is_sufficient": any(doc.relevance_score > 0.7 for doc in evaluated_docs),
                        "evaluation_rationale": "Analysis based on document relevance scores only"
                    }
            
            return {
                "information_gaps": analysis.get("information_gaps", []),
                "evaluation_rationale": analysis.get("evaluation_rationale", "")
            }
            
        except Exception as e:
            logger.error(f"Error analyzing information gaps: {e}")
            return {
                "information_gaps": ["Error analyzing information gaps"],
                "evaluation_rationale": f"Error during analysis: {str(e)}"
            }
    
    def _calculate_information_quality(self, evaluated_docs):
        """Calculate the overall information quality score based on document evaluations."""
        if not evaluated_docs:
            return 0.0
            
        # Calculate weighted average relevance
        total_relevance = sum(doc.relevance_score for doc in evaluated_docs)
        
        # Get number of documents with high relevance (>0.7)
        high_relevance_count = sum(1 for doc in evaluated_docs if doc.relevance_score > 0.7)
        
        # Get total document count
        doc_count = len(evaluated_docs)
        
        # Ideal minimum number of documents
        min_docs = self.config.get("min_documents", 3)
        
        # Calculate document coverage score (saturates at min_docs)
        coverage_score = min(1.0, doc_count / min_docs)
        
        # Calculate average relevance
        avg_relevance = total_relevance / doc_count if doc_count > 0 else 0.0
        
        # Calculate high quality score: are there enough high-quality documents?
        high_quality_score = min(1.0, high_relevance_count / 2)  # Saturates at 2 high-quality docs
        
        # Calculate combined score with weighting
        combined_score = (
            0.5 * avg_relevance +    # Weight for average relevance
            0.3 * high_quality_score +  # Weight for high-quality documents
            0.2 * coverage_score        # Weight for document coverage
        )
        
        return combined_score
    
    async def _get_related_concepts(self, query_node, evaluated_docs):
        """Get related concepts that could help with query reformulation."""
        if not self.graph_interface:
            return []
            
        try:
            # Get query text
            query_text = getattr(query_node, 'query_text', 
                        getattr(query_node, 'reformulated_query_text', 
                        getattr(query_node, 'original_user_input', str(query_node))))
            
            # Find concepts related to the query
            from ..processing_nodes.input_processor import extract_key_concepts
            query_concepts = await extract_key_concepts(query_text, self.graph_interface)
            
            related_concepts = []
            for concept in query_concepts:
                # Get related concepts from the graph
                related = await self.graph_interface.get_nodes_linked_from(
                    source_node_id=concept.id,
                    relationship_type="RELATED_TO",
                    target_node_type=type(concept),
                    limit=5
                )
                for rel_concept in related:
                    related_concepts.append({
                        "id": rel_concept.id,
                        "name": rel_concept.name,
                        "definition": getattr(rel_concept, "definition", ""),
                        "source_concept": concept.name
                    })
            
            # Remove duplicates by ID
            unique_concepts = {}
            for concept in related_concepts:
                if concept["id"] not in unique_concepts:
                    unique_concepts[concept["id"]] = concept
            
            return list(unique_concepts.values())
            
        except Exception as e:
            logger.error(f"Error getting related concepts: {e}")
            return [] 