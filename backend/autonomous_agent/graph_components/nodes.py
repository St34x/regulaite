import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

class Node(BaseModel):
    """Base class for all graph nodes."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_timestamps(self):
        """Ensure timestamps are set."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        return self

class ConceptNode(Node):
    """Represents a concept or entity in the knowledge graph."""
    label: str = "Concept"
    name: str
    definition: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    popularity_score: float = 0.0
    domain: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    related_concept_ids: List[str] = Field(default_factory=list)

class DocumentNode(Node):
    """Represents a document or content piece in the knowledge graph."""
    label: str = "Document"
    title: Optional[str] = None
    content: str
    source: Optional[str] = None
    url: Optional[str] = None
    document_type: str = "text"
    embedding: Optional[List[float]] = None
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_count: int = 0
    last_retrieved_at: Optional[str] = None
    sentiment_score: Optional[float] = None

class QueryNode(Node):
    """Represents a user query in the knowledge graph."""
    label: str = "Query"
    original_user_input: str
    query_text: str
    embedding: Optional[List[float]] = None
    reformulated_from_id: Optional[str] = None
    is_reformulation: bool = False
    reformulated_query_text: Optional[str] = None
    intent_classification: Optional[str] = None
    entities: List[Dict[str, str]] = Field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    search_config: Dict[str, Any] = Field(default_factory=dict)
    contextual_info: Dict[str, Any] = Field(default_factory=dict)

class ResponseNode(Node):
    """Represents a response generated for a query."""
    label: str = "Response"
    response_text: str
    query_id: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_document_ids: List[str] = Field(default_factory=list)
    effectiveness_score: Optional[float] = None
    feedback: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class FeedbackSourceNode(Node):
    """Represents a source of feedback (typically a user)."""
    label: str = "FeedbackSource"
    source_type: str = "user"  # Can be "user", "system", "expert", etc.
    source_id: Optional[str] = None
    reliability_score: float = 0.8
    feedback_count: int = 0
    
class UserNode(Node):
    """Represents a user in the system."""
    label: str = "User"
    username: Optional[str] = None
    email: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    expertise_areas: List[str] = Field(default_factory=list)
    trust_score: float = 0.5
    
class ChunkNode(Node):
    """Represents a chunk of a document for fine-grained retrieval."""
    label: str = "Chunk"
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict) 