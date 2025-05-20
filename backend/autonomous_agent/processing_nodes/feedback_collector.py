import logging
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from .base_node import BaseProcessingNode
from ..graph_components.nodes import ResponseNode, QueryNode, DocumentNode, ConceptNode
from ..graph_components.edges import LedToEdge, RetrievedForEdge, ContainsEdge, RelatedToEdge, FeedbackOnEdge
from ..integration_components.graph_interface import GraphInterface

logger = logging.getLogger(__name__)

class FeedbackType(str, Enum):
    EXPLICIT_RATING = "explicit_rating" # e.g., 1-5 stars
    THUMBS_UP_DOWN = "thumbs_up_down"
    USER_COMMENT = "user_comment"
    CORRECTION = "correction" # User provided a corrected answer
    IMPLICIT_ENGAGEMENT = "implicit_engagement" # e.g., clicked a link in response, spent time on page
    FOLLOW_UP_QUERY_RELEVANCE = "follow_up_query_relevance" # If a follow-up query indicates previous answer was good/bad

class FeedbackData(BaseModel):
    response_id: str # ID of the ResponseNode this feedback pertains to
    feedback_type: FeedbackType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rating: Optional[float] = Field(None, description="Numerical rating, e.g., 1-5")
    is_positive: Optional[bool] = Field(None, description="For thumbs up/down or binary sentiment")
    comment_text: Optional[str] = None
    corrected_text: Optional[str] = None
    # For implicit feedback, e.g. {"action": "clicked_source_link", "link_url": "..."}
    details: Dict[str, Any] = Field(default_factory=dict)

class FeedbackCollectorInput(BaseModel):
    feedback_data: FeedbackData
    # Optionally, the ResponseNode itself if readily available
    response_node: Optional[ResponseNode] = None
    # Associated QueryNode could also be useful for context
    query_node: Optional[QueryNode] = None 
    # Documents that contributed to the response
    contributing_documents: List[DocumentNode] = []
    # Concepts involved in query/response
    involved_concepts: List[ConceptNode] = []

class FeedbackCollectorOutput(BaseModel):
    status: str
    message: str
    updated_response_node: Optional[ResponseNode] = None
    # IDs of graph elements whose scores/weights might have been adjusted
    graph_elements_adjusted: List[str] = [] 

class FeedbackCollectionNode(BaseProcessingNode):
    """Processes user feedback to improve system performance and update graph elements."""
    
    def __init__(self, node_config: Optional[Dict[str, Any]] = None):
        super().__init__(node_config)
        # Configure default parameters
        config = node_config or {}
        self.min_adjustment = config.get("min_adjustment", -0.2)
        self.max_adjustment = config.get("max_adjustment", 0.2)
        self.default_edge_weight = config.get("default_edge_weight", 0.5)
        self.feedback_source_prefix = config.get("feedback_source_prefix", "feedback_source_")

    async def execute(self, input_data: FeedbackCollectorInput, context: Dict[str, Any]) -> FeedbackCollectorOutput:
        """
        Processes feedback, updates scores and relationships in the graph.

        Args:
            input_data: Feedback data along with related graph nodes.
            context: Workflow context, primarily for graph_interface.

        Returns:
            Status of feedback processing.
        """
        fb_data = input_data.feedback_data
        logger.info(f"[{self.get_name()}] Processing feedback type '{fb_data.feedback_type}' for response ID: {fb_data.response_id}")

        graph: GraphInterface = context.get("graph_interface")
        if not graph:
            logger.error(f"[{self.get_name()}] GraphInterface not found in context. Cannot process feedback.")
            return FeedbackCollectorOutput(status="error", message="GraphInterface not available.")

        response_node_id = fb_data.response_id
        updated_response_node = input_data.response_node
        graph_elements_adjusted = []

        # Fetch nodes if not provided
        if not updated_response_node:
            updated_response_node = await graph.get_node_by_id(response_node_id, ResponseNode)
            if not updated_response_node:
                logger.error(f"[{self.get_name()}] ResponseNode {response_node_id} not found.")
                return FeedbackCollectorOutput(status="error", message=f"ResponseNode {response_node_id} not found.")
        
        query_node = input_data.query_node
        if not query_node:
           # Try to find the query that LED_TO this response
           query_node = await graph.get_query_for_response(response_node_id)

        # 1. Update effectiveness scores on ResponseNodes
        current_effectiveness = getattr(updated_response_node, "effectiveness_score", None)
        new_effectiveness = current_effectiveness

        # Initialize feedback field if it doesn't exist
        if not hasattr(updated_response_node, "feedback") or updated_response_node.feedback is None:
            updated_response_node.feedback = {}

        if fb_data.feedback_type == FeedbackType.EXPLICIT_RATING and fb_data.rating is not None:
            # Assuming rating is 1-5, normalize to 0-1 for effectiveness
            new_effectiveness = (fb_data.rating - 1) / 4 
            updated_response_node.feedback["explicit_rating"] = fb_data.rating
        elif fb_data.feedback_type == FeedbackType.THUMBS_UP_DOWN and fb_data.is_positive is not None:
            new_effectiveness = 1.0 if fb_data.is_positive else 0.0
            updated_response_node.feedback["thumbs_up_down"] = "up" if fb_data.is_positive else "down"
        elif fb_data.feedback_type == FeedbackType.USER_COMMENT and fb_data.comment_text:
            updated_response_node.feedback["user_comment"] = fb_data.comment_text
            # A comment by itself doesn't change effectiveness unless we add sentiment analysis
            # This could be a good place to use LLM for sentiment analysis
        elif fb_data.feedback_type == FeedbackType.CORRECTION and fb_data.corrected_text:
            updated_response_node.feedback["user_correction"] = fb_data.corrected_text
            new_effectiveness = 0.1 # Low score for original if corrected
        elif fb_data.feedback_type == FeedbackType.IMPLICIT_ENGAGEMENT:
            updated_response_node.feedback["implicit_engagement"] = fb_data.details
            # For implicit engagement, adjust slightly based on action type
            if "action" in fb_data.details:
                action = fb_data.details.get("action", "")
                if action in ["clicked_source", "saved_response", "shared_response"]:
                    # These actions suggest the response was helpful
                    new_effectiveness = min(1.0, (current_effectiveness or 0.5) + 0.1)
                elif action in ["dismissed", "ignored"]:
                    # These actions suggest the response wasn't helpful
                    new_effectiveness = max(0.0, (current_effectiveness or 0.5) - 0.1)

        if new_effectiveness is not None:
            # Update the ResponseNode in the graph
            try:
                update_data = {
                    "effectiveness_score": new_effectiveness,
                    "feedback": updated_response_node.feedback,
                    "last_feedback_at": datetime.utcnow().isoformat()
                }
                await graph.update_node_properties(response_node_id, update_data)
                updated_response_node.effectiveness_score = new_effectiveness
                graph_elements_adjusted.append(response_node_id)
                logger.info(f"[{self.get_name()}] Updated effectiveness for ResponseNode {response_node_id} to {new_effectiveness}")
            except Exception as e:
                logger.error(f"[{self.get_name()}] Error updating ResponseNode properties: {e}")

        # 2. Adjust success rates of query-response relationships (LED_TO edge)
        if query_node and new_effectiveness is not None:
            try:
                await graph.update_edge_properties(
                    edge_type="LED_TO", 
                    source_node_id=query_node.id, 
                    target_node_id=response_node_id, 
                    properties_update={
                        "success_metric": new_effectiveness,
                        "last_feedback_type": fb_data.feedback_type.value,
                        "last_feedback_at": datetime.utcnow().isoformat()
                    }
                )
                graph_elements_adjusted.append(f"LED_TO_EDGE:{query_node.id}-{response_node_id}")
                logger.info(f"[{self.get_name()}] Updated LED_TO edge between query {query_node.id} and response {response_node_id}")
            except Exception as e:
                logger.error(f"[{self.get_name()}] Error updating LED_TO edge: {e}")

        # 3. Strengthen or weaken document-concept relationships (CONTAINS, RELATED_TO)
        # and document-query relationships (RETRIEVED_FOR)
        if new_effectiveness is not None:
            adjustment_factor = (new_effectiveness - 0.5) * 0.4  # Scale adjustment, e.g., -0.2 to +0.2
            adjustment_factor = max(self.min_adjustment, min(self.max_adjustment, adjustment_factor))
            
            # For each contributing document, adjust relevant edges
            for doc in input_data.contributing_documents:
                # Adjust RETRIEVED_FOR from this doc to the query
                if query_node:
                    try:
                        # Check if edge exists
                        edge_exists = await graph.check_relationship_exists(
                            source_id=doc.id, 
                            target_id=query_node.id, 
                            rel_type=RetrievedForEdge.get_relationship_type()
                        )
                        
                        if edge_exists:
                            # Adjust the weight
                            await graph.adjust_edge_weight(
                                source_id=doc.id, 
                                target_id=query_node.id, 
                                rel_type=RetrievedForEdge.get_relationship_type(),
                                adjustment=adjustment_factor,
                                weight_property="relevance_score",
                                min_weight=0.1,
                                max_weight=1.0
                            )
                            graph_elements_adjusted.append(f"RETRIEVED_FOR_EDGE:{doc.id}-{query_node.id}")
                            logger.info(f"[{self.get_name()}] Adjusted RETRIEVED_FOR edge from doc {doc.id} to query {query_node.id}")
                    except Exception as e:
                        logger.error(f"[{self.get_name()}] Error adjusting document-query relationship: {e}")
                
                # For concepts within this document that were relevant to the query
                for concept in input_data.involved_concepts:
                    try:
                        # Check if this concept is in this document
                        concept_in_doc = await graph.check_relationship_exists(
                            source_id=doc.id, 
                            target_id=concept.id, 
                            rel_type=ContainsEdge.get_relationship_type()
                        )
                        
                        if concept_in_doc:
                            # Adjust the CONTAINS edge weight
                            await graph.adjust_edge_weight(
                                source_id=doc.id, 
                                target_id=concept.id, 
                                rel_type=ContainsEdge.get_relationship_type(),
                                adjustment=adjustment_factor,
                                weight_property="weight",
                                min_weight=0.1,
                                max_weight=1.0
                            )
                            graph_elements_adjusted.append(f"CONTAINS_EDGE:{doc.id}-{concept.id}")
                            logger.info(f"[{self.get_name()}] Adjusted CONTAINS edge from doc {doc.id} to concept {concept.id}")
                    except Exception as e:
                        logger.error(f"[{self.get_name()}] Error adjusting document-concept relationship: {e}")

        # 4. Create FeedbackOnEdge (User -> Response or FeedbackSource -> Response)
        feedback_source_id = fb_data.user_id if fb_data.user_id else f"{self.feedback_source_prefix}{fb_data.session_id or 'anonymous'}"
        
        try:
            # Prepare edge properties, filtering out None values
            feedback_properties = {
                "feedback_type": fb_data.feedback_type.value,
                "timestamp": fb_data.timestamp.isoformat()
            }
            
            if fb_data.rating is not None:
                feedback_properties["rating"] = fb_data.rating
            if fb_data.is_positive is not None:
                feedback_properties["is_positive"] = fb_data.is_positive
            if fb_data.comment_text:
                feedback_properties["has_comment"] = True
                feedback_properties["comment_length"] = len(fb_data.comment_text)
            if fb_data.corrected_text:
                feedback_properties["has_correction"] = True
                feedback_properties["correction_length"] = len(fb_data.corrected_text)
            if fb_data.details:
                feedback_properties["details"] = fb_data.details
            
            # Create and add the feedback edge
            feedback_edge = FeedbackOnEdge(
                source_node_id=feedback_source_id,
                target_node_id=response_node_id,
                properties=feedback_properties
            )
            
            await graph.add_edge(feedback_edge)
            logger.info(f"[{self.get_name()}] Created FeedbackOnEdge from {feedback_source_id} to response {response_node_id}")
        except Exception as e:
            logger.error(f"[{self.get_name()}] Error creating FeedbackOnEdge: {e}")

        logger.info(f"[{self.get_name()}] Feedback processing finished. Adjustments: {len(graph_elements_adjusted)} elements.")

        return FeedbackCollectorOutput(
            status="success",
            message=f"Feedback of type '{fb_data.feedback_type}' processed for response {fb_data.response_id}.",
            updated_response_node=updated_response_node,
            graph_elements_adjusted=list(set(graph_elements_adjusted))
        ) 