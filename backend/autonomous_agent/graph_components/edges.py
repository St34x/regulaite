import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

class RelationshipType(str, Enum):
    """Types of relationships between nodes in the knowledge graph."""
    CONTAINS = "CONTAINS"  # Document contains Concept
    RELATED_TO = "RELATED_TO"  # Concept is related to Concept
    RETRIEVED_FOR = "RETRIEVED_FOR"  # Document was retrieved for a Query
    LED_TO = "LED_TO"  # Query led to a Response
    REFORMULATED_AS = "REFORMULATED_AS"  # Query was reformulated as another Query
    GENERATED_FROM = "GENERATED_FROM"  # Response was generated from a Document
    MENTIONS = "MENTIONS"  # Document mentions a Concept
    FEEDBACK_ON = "FEEDBACK_ON"  # User provided feedback on a Response
    CHUNK_OF = "CHUNK_OF"  # Chunk is part of a Document
    USED_IN = "USED_IN"  # Concept was used in a Query
    SUBMITTED_BY = "SUBMITTED_BY"  # Query was submitted by a User
    INTERACTED_WITH = "INTERACTED_WITH"  # User interacted with a Document/Response

class Edge(BaseModel):
    """Base class for all graph edges."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: RelationshipType
    source_node_id: str
    target_node_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_timestamps(self):
        """Ensure timestamps are set."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        return self
        
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        """Get the relationship type of this edge."""
        raise NotImplementedError("Subclasses must implement get_relationship_type")

class ContainsEdge(Edge):
    """Represents that a document contains a concept."""
    type: RelationshipType = RelationshipType.CONTAINS
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.CONTAINS

class RelatedToEdge(Edge):
    """Represents that a concept is related to another concept."""
    type: RelationshipType = RelationshipType.RELATED_TO
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.RELATED_TO

class RetrievedForEdge(Edge):
    """Represents that a document was retrieved for a query."""
    type: RelationshipType = RelationshipType.RETRIEVED_FOR
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.RETRIEVED_FOR

class LedToEdge(Edge):
    """Represents that a query led to a response."""
    type: RelationshipType = RelationshipType.LED_TO
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.LED_TO

class ReformulatedAsEdge(Edge):
    """Represents that a query was reformulated as another query."""
    type: RelationshipType = RelationshipType.REFORMULATED_AS
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.REFORMULATED_AS

class GeneratedFromEdge(Edge):
    """Represents that a response was generated from a document."""
    type: RelationshipType = RelationshipType.GENERATED_FROM
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.GENERATED_FROM

class MentionsEdge(Edge):
    """Represents that a document mentions a concept."""
    type: RelationshipType = RelationshipType.MENTIONS
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.MENTIONS

class FeedbackOnEdge(Edge):
    """Represents that a user provided feedback on a response."""
    type: RelationshipType = RelationshipType.FEEDBACK_ON
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.FEEDBACK_ON

class ChunkOfEdge(Edge):
    """Represents that a chunk is part of a document."""
    type: RelationshipType = RelationshipType.CHUNK_OF
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.CHUNK_OF

class UsedInEdge(Edge):
    """Represents that a concept was used in a query."""
    type: RelationshipType = RelationshipType.USED_IN
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.USED_IN

class SubmittedByEdge(Edge):
    """Represents that a query was submitted by a user."""
    type: RelationshipType = RelationshipType.SUBMITTED_BY
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.SUBMITTED_BY

class InteractedWithEdge(Edge):
    """Represents that a user interacted with a document or response."""
    type: RelationshipType = RelationshipType.INTERACTED_WITH
    
    @classmethod
    def get_relationship_type(cls) -> RelationshipType:
        return RelationshipType.INTERACTED_WITH 