"""
FastAPI router for document management, upload and configuration.
"""
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
from datetime import datetime

# Neo4j imports
from neo4j.time import DateTime as Neo4jDateTime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)

# Helper function to serialize Neo4j DateTime objects
def neo4j_datetime_serializer(obj):
    """Custom serializer for Neo4j DateTime objects."""
    if isinstance(obj, Neo4jDateTime):
        # Convert Neo4j DateTime to Python datetime
        return datetime(
            obj.year, obj.month, obj.day,
            obj.hour, obj.minute, obj.second,
            obj.nanosecond // 1000000
        ).isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Models for API
class DocumentMetadata(BaseModel):
    """Additional metadata for document processing."""
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source")
    publish_date: Optional[str] = Field(None, description="Document publication date")
    category: Optional[str] = Field(None, description="Document category")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom metadata fields")


class DocumentProcessResponse(BaseModel):
    """Response for document processing."""
    doc_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks processed")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Processing message")


class DocumentIndexStatus(BaseModel):
    """Status of document indexing."""
    doc_id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Indexing status")
    vector_count: Optional[int] = Field(None, description="Number of vectors indexed")
    message: str = Field(..., description="Status message")


class DocumentConfigUpdate(BaseModel):
    """Update to document processing configuration."""
    chunk_size: Optional[int] = Field(None, description="Chunk size for document processing")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap for document processing")
    default_language: Optional[str] = Field(None, description="Default language for processing")
    auto_detect_language: Optional[bool] = Field(None, description="Whether to auto-detect language")
    use_nlp: Optional[bool] = Field(None, description="Whether to use NLP for entity extraction")
    use_enrichment: Optional[bool] = Field(None, description="Whether to use data enrichment")
    index_immediately: Optional[bool] = Field(None, description="Whether to index documents immediately after processing")
    embedding_model: Optional[str] = Field(None, description="Model to use for embeddings")
    default_embedding_dim: Optional[int] = Field(None, description="Dimension of default embeddings")
    max_file_size_mb: Optional[int] = Field(None, description="Maximum file size in MB")
    allowed_file_types: Optional[List[str]] = Field(None, description="List of allowed file extensions")


class DocumentConfig(DocumentConfigUpdate):
    """Document processing configuration."""
    pass


class DocumentSearchRequest(BaseModel):
    """Request for document search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Maximum number of results")
    offset: int = Field(0, description="Offset for pagination")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    hybrid_search: Optional[bool] = Field(True, description="Whether to use hybrid search")


class DocumentStatsResponse(BaseModel):
    """Response with document statistics."""
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    documents_by_type: Dict[str, int] = Field(..., description="Document count by type")
    documents_by_language: Dict[str, int] = Field(..., description="Document count by language")
    recent_uploads: List[Dict[str, Any]] = Field(..., description="Recent document uploads")
    total_storage_mb: float = Field(..., description="Total storage used in MB")


# Dependency to get Neo4j driver
async def get_neo4j_driver():
    """Get the Neo4j driver from main application."""
    from main import driver
    return driver


# Dependency to get document parser
async def get_document_parser():
    """Get the document parser from main application."""
    from main import document_parser
    return document_parser


# Dependency to get RAG system
async def get_rag_system():
    """Get the RAG system from main application."""
    from main import rag_system
    return rag_system


# Dependency to get task router for queue
async def get_task_router():
    """Get the task router from main application."""
    from routers.task_router import queue_document_processing
    return queue_document_processing


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    use_nlp: bool = Form(True),
    use_enrichment: bool = Form(True),
    detect_language: bool = Form(True),
    language: Optional[str] = Form(None),
    use_queue: bool = Form(False)
):
    """Process a document using Unstructured API and spaCy NLP, then store in Neo4j."""

    # Import and forward to existing implementation in main.py
    from main import process_document as main_process_document

    response = await main_process_document(
        file=file,
        doc_id=doc_id,
        metadata=metadata,
        use_nlp=use_nlp,
        use_enrichment=use_enrichment,
        detect_language=detect_language,
        language=language,
        use_queue=use_queue
    )

    return response


@router.get("", response_model=Dict[str, Any])
async def list_documents(
    limit: int = 10,
    offset: int = 0,
    sort_by: str = "created",
    sort_order: str = "desc",
    file_type: Optional[str] = None,
    category: Optional[str] = None,
    language: Optional[str] = None,
    tags: Optional[str] = None,
    search: Optional[str] = None
):
    """List all documents with optional filtering and sorting."""
    try:
        # Get neo4j driver
        driver = await get_neo4j_driver()

        with driver.session() as session:
            # Start building the query
            query_parts = ["MATCH (d:Document)"]
            params = {}

            # Add filters if provided
            filters = []

            if file_type:
                filters.append("d.file_type = $file_type")
                params["file_type"] = file_type

            if category:
                filters.append("d.category = $category")
                params["category"] = category

            if language:
                filters.append("d.language = $language")
                params["language"] = language

            if tags:
                tag_list = tags.split(",")
                filters.append("ANY(tag IN $tags WHERE tag IN d.tags)")
                params["tags"] = tag_list

            if search:
                filters.append("(d.title CONTAINS $search OR d.doc_id CONTAINS $search)")
                params["search"] = search

            # Combine filters if any
            if filters:
                query_parts.append("WHERE " + " AND ".join(filters))

            # Count total documents
            count_query = "\n".join(query_parts) + "\nRETURN COUNT(d) as total"
            count_result = session.run(count_query, params).single()
            total = count_result["total"] if count_result else 0

            # Add sorting and pagination
            query_parts.append(f"RETURN d ORDER BY d.{sort_by} {sort_order.upper()}")
            query_parts.append(f"SKIP {offset} LIMIT {limit}")

            # Execute the query
            query = "\n".join(query_parts)
            results = session.run(query, params)

            # Process results
            documents = []
            for record in results:
                doc = record["d"]

                # Convert Neo4j values to Python
                doc_dict = dict(doc.items())

                # Convert any Neo4j DateTime objects to ISO format strings
                for key, value in doc_dict.items():
                    if isinstance(value, Neo4jDateTime):
                        doc_dict[key] = neo4j_datetime_serializer(value)

                documents.append(doc_dict)

            return {"documents": documents, "total": total}

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@router.get("/{doc_id}", response_model=Dict[str, Any])
async def get_document(doc_id: str, include_chunks: bool = False, include_entities: bool = False):
    """Get document metadata and optionally chunks and entities."""
    try:
        # Get neo4j driver
        driver = await get_neo4j_driver()

        with driver.session() as session:
            # Get document metadata
            doc_result = session.run(
                "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
                doc_id=doc_id
            )

            doc_record = doc_result.single()
            if not doc_record:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {doc_id}"
                )

            document = dict(doc_record["d"])
            response = {"document": document}

            # Get document chunks if requested
            if include_chunks:
                chunks_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                    RETURN c
                    ORDER BY c.index
                    """,
                    doc_id=doc_id
                )

                chunks = [dict(record["c"]) for record in chunks_result]
                response["chunks"] = chunks
                response["chunk_count"] = len(chunks)

            # Get document entities if requested
            if include_entities:
                entities_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(:Chunk)-[:HAS_ENTITY]->(e:Entity)
                    RETURN DISTINCT e
                    ORDER BY e.name
                    """,
                    doc_id=doc_id
                )

                entities = [dict(record["e"]) for record in entities_result]
                response["entities"] = entities
                response["entity_count"] = len(entities)

            return response

    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document: {str(e)}"
        )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, delete_from_index: bool = True):
    """Delete document and its chunks from Neo4j and optionally from vector store."""
    try:
        # Get dependencies
        driver = await get_neo4j_driver()
        rag_system = await get_rag_system()

        # First delete from Neo4j
        with driver.session() as session:
            # Check if document exists
            doc_check = session.run(
                "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) as count",
                doc_id=doc_id
            )

            if doc_check.single()["count"] == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {doc_id}"
                )

            # Delete all relationships and nodes related to the document
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                OPTIONAL MATCH (c)-[r]->(n)
                DELETE r, c, d
                RETURN count(c) as chunks_deleted
                """,
                doc_id=doc_id
            )

            chunks_deleted = result.single()["chunks_deleted"]

        # Delete from vector index if requested
        index_message = ""
        if delete_from_index and chunks_deleted > 0:
            try:
                # Use RAG system to delete from index
                rag_system.delete_document(doc_id)
                index_message = ", and deleted from vector index"
            except Exception as e:
                logger.error(f"Error deleting document from vector index: {str(e)}")
                index_message = f", but failed to delete from vector index: {str(e)}"

        return {
            "doc_id": doc_id,
            "deleted": True,
            "chunks_deleted": chunks_deleted,
            "message": f"Document and {chunks_deleted} chunks deleted from database{index_message}"
        }

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/index/{doc_id}", response_model=DocumentIndexStatus)
async def index_document(doc_id: str, force_reindex: bool = False):
    """Index a document in the vector store."""
    try:
        # Get dependencies
        driver = await get_neo4j_driver()
        rag_system = await get_rag_system()

        # Check if document exists in Neo4j
        with driver.session() as session:
            doc_check = session.run(
                "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) as count",
                doc_id=doc_id
            )

            if doc_check.single()["count"] == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document not found: {doc_id}"
                )

        # Index the document
        result = rag_system.index_document(doc_id, force_reindex=force_reindex)

        return DocumentIndexStatus(
            doc_id=doc_id,
            status="success",
            vector_count=result.get("vector_count", 0) if isinstance(result, dict) else None,
            message=f"Document successfully indexed in vector store"
        )

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing document: {str(e)}"
        )


@router.post("/search", response_model=Dict[str, Any])
async def search_documents(request: DocumentSearchRequest):
    """Search for documents using semantic search."""
    try:
        # Get RAG system
        rag_system = await get_rag_system()

        # Prepare search parameters
        hybrid = request.hybrid_search if request.hybrid_search is not None else True

        # Use RAG system to search
        results = rag_system.retrieve(
            query=request.query,
            top_k=request.limit,
            filters=request.filters,
            hybrid_search=hybrid
        )

        # Process results
        processed_results = []
        seen_docs = set()

        for item in results:
            doc_id = item["metadata"].get("doc_id")

            # Create a result entry
            result_entry = {
                "doc_id": doc_id,
                "document_title": item["metadata"].get("doc_name", "Unknown document"),
                "text": item["text"],
                "section": item["metadata"].get("section", "Unknown"),
                "score": item.get("score", 0),
                "metadata": item["metadata"]
            }

            processed_results.append(result_entry)
            seen_docs.add(doc_id)

        return {
            "results": processed_results,
            "query": request.query,
            "count": len(processed_results),
            "unique_documents": len(seen_docs)
        }

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )


@router.get("/config", response_model=DocumentConfig)
async def get_document_config():
    """Get document processing configuration."""
    try:
        # Get document parser to access its configuration
        doc_parser = await get_document_parser()

        # Get configuration from database
        from main import get_mariadb_connection

        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT setting_key, setting_value
            FROM regulaite_settings
            WHERE setting_key LIKE 'doc_%'
            """
        )

        settings = {row['setting_key'][4:]: row['setting_value'] for row in cursor.fetchall()}
        conn.close()

        # Convert types as needed
        result = DocumentConfig(
            chunk_size=int(settings.get('chunk_size', 1000)),
            chunk_overlap=int(settings.get('chunk_overlap', 200)),
            default_language=settings.get('default_language', 'en'),
            auto_detect_language=settings.get('auto_detect_language', 'true').lower() == 'true',
            use_nlp=settings.get('use_nlp', 'true').lower() == 'true',
            use_enrichment=settings.get('use_enrichment', 'true').lower() == 'true',
            index_immediately=settings.get('index_immediately', 'true').lower() == 'true',
            embedding_model=settings.get('embedding_model', 'text-embedding-ada-002'),
            default_embedding_dim=int(settings.get('default_embedding_dim', 1536)),
            max_file_size_mb=int(settings.get('max_file_size_mb', 10)),
            allowed_file_types=settings.get('allowed_file_types', 'pdf,docx,txt,md,html').split(',')
        )

        return result

    except Exception as e:
        logger.error(f"Error retrieving document configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document configuration: {str(e)}"
        )


@router.post("/config", response_model=DocumentConfig)
async def update_document_config(config: DocumentConfigUpdate):
    """Update document processing configuration."""
    try:
        # Get document parser
        doc_parser = await get_document_parser()

        # Get database connection
        from main import get_mariadb_connection

        conn = get_mariadb_connection()
        cursor = conn.cursor()

        # Update each provided setting
        if config.dict(exclude_none=True):
            for key, value in config.dict(exclude_none=True).items():
                # Convert lists to comma-separated strings
                if isinstance(value, list):
                    value = ','.join(value)

                # Convert booleans to strings
                if isinstance(value, bool):
                    value = str(value).lower()

                # Update or insert the setting
                cursor.execute(
                    """
                    INSERT INTO regulaite_settings (setting_key, setting_value)
                    VALUES (?, ?)
                    ON DUPLICATE KEY UPDATE setting_value = ?
                    """,
                    (f"doc_{key}", str(value), str(value))
                )

        conn.commit()
        conn.close()

        # Return the updated configuration
        return await get_document_config()

    except Exception as e:
        logger.error(f"Error updating document configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating document configuration: {str(e)}"
        )


@router.get("/stats", response_model=DocumentStatsResponse)
async def get_document_stats():
    """Get statistics about documents in the system."""
    try:
        # Get neo4j driver
        driver = await get_neo4j_driver()

        with driver.session() as session:
            # Get total document count
            total_doc_result = session.run("MATCH (d:Document) RETURN count(d) as count")
            total_documents = total_doc_result.single()["count"]

            # Get total chunk count
            total_chunk_result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
            total_chunks = total_chunk_result.single()["count"]

            # Get documents by type
            type_result = session.run(
                """
                MATCH (d:Document)
                RETURN d.file_type as type, count(d) as count
                ORDER BY count DESC
                """
            )
            documents_by_type = {record["type"] or "unknown": record["count"] for record in type_result}

            # Get documents by language
            lang_result = session.run(
                """
                MATCH (d:Document)
                RETURN d.language as language, count(d) as count
                ORDER BY count DESC
                """
            )
            documents_by_language = {record["language"] or "unknown": record["count"] for record in lang_result}

            # Get recent uploads
            recent_result = session.run(
                """
                MATCH (d:Document)
                RETURN d.doc_id as doc_id, d.title as title, d.created as created,
                       d.file_type as file_type, d.language as language,
                       d.original_filename as filename, d.size as size
                ORDER BY d.created DESC
                LIMIT 5
                """
            )
            recent_uploads = [dict(record) for record in recent_result]

            # Calculate total storage
            total_storage = 0
            if total_documents > 0:
                storage_result = session.run("MATCH (d:Document) RETURN sum(d.size) as total_size")
                total_storage = storage_result.single()["total_size"] or 0

            # Convert to MB
            total_storage_mb = round(total_storage / (1024 * 1024), 2)

            return DocumentStatsResponse(
                total_documents=total_documents,
                total_chunks=total_chunks,
                documents_by_type=documents_by_type,
                documents_by_language=documents_by_language,
                recent_uploads=recent_uploads,
                total_storage_mb=total_storage_mb
            )

    except Exception as e:
        logger.error(f"Error retrieving document statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document statistics: {str(e)}"
        )
