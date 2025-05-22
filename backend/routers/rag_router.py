"""
Router for LlamaIndex RAG system operations.
"""
import logging
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import query engine
from llamaIndex_rag.query_engine import RAGQueryEngine
from llamaIndex_rag.rag import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}},
)

# Models for API
class RAGQuery(BaseModel):
    """Query for RAG system."""
    query: str = Field(..., description="Query to retrieve context for")
    top_k: int = Field(5, description="Number of results to return")
    search_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    synthesize: bool = Field(True, description="Whether to synthesize a response")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt for synthesis")
    streaming: Optional[bool] = Field(None, description="Whether to stream the response")
    show_hallucination_indicators: bool = Field(True, description="Whether to return hallucination indicators for UI")
    use_self_critique: bool = Field(True, description="Whether to use self-critique for hallucination reduction")


class RAGIndexRequest(BaseModel):
    """Request to index a document."""
    doc_id: str = Field(..., description="Document ID to index")
    force: bool = Field(False, description="Whether to force reindex")


class RAGBulkIndexRequest(BaseModel):
    """Request to bulk index documents."""
    doc_ids: List[str] = Field(..., description="Document IDs to index")
    force: bool = Field(False, description="Whether to force reindex")


class RAGConfig(BaseModel):
    """Configuration for RAG system."""
    collection_name: Optional[str] = Field(None, description="Qdrant collection name")
    metadata_collection_name: Optional[str] = Field(None, description="Qdrant metadata collection name")
    embedding_model: Optional[str] = Field(None, description="Model name for embeddings")
    embedding_dim: Optional[int] = Field(None, description="Embedding dimension")
    llm_model: Optional[str] = Field(None, description="LLM model name")
    temperature: Optional[float] = Field(None, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response")
    vector_weight: Optional[float] = Field(None, description="Weight for vector search in hybrid retrieval (0-1)")
    semantic_weight: Optional[float] = Field(None, description="Weight for semantic search in hybrid retrieval (0-1)")
    sentence_window_size: Optional[int] = Field(None, description="Number of sentences for context window")
    default_prompt: Optional[str] = Field(None, description="Default prompt template for answer synthesis")


class RepairMetadataRequest(BaseModel):
    """Request for repairing document metadata."""
    doc_id: Optional[str] = Field(None, description="Specific document ID to repair, if empty repairs all")
    force: bool = Field(False, description="Force repair even if metadata exists")


class RepairMetadataResponse(BaseModel):
    """Response for metadata repair operation."""
    status: str = Field(..., description="Operation status")
    repaired: int = Field(..., description="Number of documents repaired")
    failed: int = Field(..., description="Number of documents that failed to repair")
    message: str = Field(..., description="Result message")
    duration_seconds: Optional[float] = Field(None, description="Duration of repair operation in seconds")


# Dependency to get RAG system from main application
async def get_rag_system():
    """Get the RAG system from main application."""
    from main import rag_system
    return rag_system


# Dependency to get query engine
async def get_query_engine():
    """Get the RAG query engine."""
    from main import rag_query_engine
    return rag_query_engine


@router.post("/query", response_model=Dict[str, Any])
async def query_rag(
    request: RAGQuery,
    query_engine: RAGQueryEngine = Depends(get_query_engine)
):
    """Query the RAG system with context retrieval and optional response synthesis."""
    try:
        result = await query_engine.query(
            query_text=request.query,
            top_k=request.top_k,
            search_filter=request.search_filter,
            custom_prompt=request.custom_prompt,
            streaming=request.streaming,
            use_self_critique=request.use_self_critique
        )
        
        # If hallucination indicators are not requested, remove hallucination metrics from result
        if not request.show_hallucination_indicators and "hallucination_metrics" in result:
            del result["hallucination_metrics"]
            
        return result
    except Exception as e:
        logger.error(f"Error querying RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying RAG system: {str(e)}")


@router.post("/index", response_model=Dict[str, Any])
async def index_document(
    request: RAGIndexRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Index a document in the RAG system."""
    try:
        result = rag_system.index_document(request.doc_id)
        return result
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")


@router.post("/reindex-all", response_model=Dict[str, Any])
async def reindex_all_documents(
    force: bool = False,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Reindex all documents in the RAG system."""
    try:
        result = rag_system.reindex_all_documents(force=force)
        return result
    except Exception as e:
        logger.error(f"Error reindexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reindexing documents: {str(e)}")


@router.post("/bulk-index", response_model=Dict[str, Any])
async def bulk_index_documents(
    request: RAGBulkIndexRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Bulk index documents in the RAG system."""
    try:
        results = []
        for doc_id in request.doc_ids:
            result = rag_system.index_document(doc_id)
            results.append(result)
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = len(results) - success_count
        
        return {
            "status": "success",
            "message": f"Indexed {success_count} documents, {error_count} failed",
            "total": len(request.doc_ids),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error bulk indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error bulk indexing documents: {str(e)}")


@router.get("/config", response_model=RAGConfig)
async def get_rag_config(
    rag_system: RAGSystem = Depends(get_rag_system),
    query_engine: RAGQueryEngine = Depends(get_query_engine)
):
    """Get current RAG system configuration."""
    try:
        config = {
            "collection_name": rag_system.collection_name,
            "metadata_collection_name": rag_system.metadata_collection_name,
            "embedding_model": rag_system.embedding_model,
            "embedding_dim": rag_system.embedding_dim,
            "vector_weight": getattr(rag_system, "vector_weight", 0.7),
            "semantic_weight": getattr(rag_system, "semantic_weight", 0.3),
            "sentence_window_size": getattr(rag_system, "sentence_window_size", 3),
            "llm_model": query_engine.model_name,
            "temperature": query_engine.temperature,
            "max_tokens": query_engine.max_tokens,
            "default_prompt": getattr(query_engine, "default_prompt", None)
        }
        return config
    except Exception as e:
        logger.error(f"Error retrieving RAG config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving RAG config: {str(e)}")


@router.post("/config", response_model=RAGConfig)
async def update_rag_config(
    config: RAGConfig,
    rag_system: RAGSystem = Depends(get_rag_system),
    query_engine: RAGQueryEngine = Depends(get_query_engine)
):
    """Update RAG system configuration."""
    try:
        # Update query engine config
        if config.llm_model or config.temperature is not None or config.max_tokens is not None or config.default_prompt:
            query_engine.update_model(
                model_name=config.llm_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                default_prompt=config.default_prompt
            )
        
        # Update RAG system parameters that can be updated without reinitialization
        if hasattr(rag_system, "vector_weight") and config.vector_weight is not None:
            rag_system.vector_weight = config.vector_weight
            
        if hasattr(rag_system, "semantic_weight") and config.semantic_weight is not None:
            rag_system.semantic_weight = config.semantic_weight
        
        if hasattr(rag_system, "sentence_window_size") and config.sentence_window_size is not None:
            rag_system.sentence_window_size = config.sentence_window_size
  
        # Return current config
        updated_config = {
            "collection_name": rag_system.collection_name,
            "metadata_collection_name": rag_system.metadata_collection_name,
            "embedding_model": rag_system.embedding_model,
            "embedding_dim": rag_system.embedding_dim,
            "vector_weight": getattr(rag_system, "vector_weight", 0.7),
            "semantic_weight": getattr(rag_system, "semantic_weight", 0.3),
            "sentence_window_size": getattr(rag_system, "sentence_window_size", 3),
            "llm_model": query_engine.model_name,
            "temperature": query_engine.temperature,
            "max_tokens": query_engine.max_tokens,
            "default_prompt": getattr(query_engine, "default_prompt", None)
        }
        return updated_config
    except Exception as e:
        logger.error(f"Error updating RAG config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating RAG config: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Delete a document from the system."""
    try:
        success = rag_system.delete_document(doc_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Document {doc_id} deleted successfully",
                "doc_id": doc_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to delete document {doc_id} or document not found"
            )
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.get("/health")
async def rag_health_check(
    rag_system: RAGSystem = Depends(get_rag_system),
    query_engine: RAGQueryEngine = Depends(get_query_engine)
):
    """Check health status of RAG system components."""
    try:
        # Check Qdrant connection
        qdrant_ok = False
        try:
            collections = rag_system.qdrant_client.get_collections()
            qdrant_ok = True
        except Exception as e:
            logger.error(f"Qdrant connection error: {str(e)}")
        
        # Check LLM availability
        llm_ok = query_engine.llm is not None
        
        # Return health status
        return {
            "status": "healthy" if (qdrant_ok and llm_ok) else "unhealthy",
            "components": {
                "qdrant": "connected" if qdrant_ok else "disconnected",
                "llm": "available" if llm_ok else "unavailable",
                "embedding_model": rag_system.embedding_model
            }
        }
    except Exception as e:
        logger.error(f"Error checking RAG health: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/repair-metadata", response_model=RepairMetadataResponse)
async def repair_document_metadata(request: RepairMetadataRequest):
    """
    Repair document metadata for documents missing metadata.
    
    This will check for documents with chunks but missing metadata and create default metadata for them.
    Can be used to fix indexing issues for documents where metadata is missing.
    """
    try:
        # Get RAG system
        rag_system = await get_rag_system()
        if not rag_system:
            raise HTTPException(
                status_code=500,
                detail="RAG system not initialized"
            )
        
        # Repair metadata
        result = rag_system.repair_document_metadata(doc_id=request.doc_id)
        
        return {
            "status": result.get("status", "error"),
            "repaired": result.get("repaired", 0),
            "failed": result.get("failed", 0),
            "message": result.get("message", ""),
            "duration_seconds": result.get("duration_seconds")
        }
    
    except Exception as e:
        logger.error(f"Error repairing document metadata: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error repairing document metadata: {str(e)}"
        ) 