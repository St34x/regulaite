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
from datetime import datetime as py_datetime
import asyncio

# Neo4j imports
from neo4j.time import DateTime as Neo4jDateTime

# Import parser types
from unstructured_parser.base_parser import ParserType

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
        return py_datetime(
            obj.year, obj.month, obj.day,
            obj.hour, obj.minute, obj.second,
            obj.nanosecond // 1000000
        ).isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Models for API
class DocumentMetadata(BaseModel):
    """Document metadata for API responses."""
    doc_id: str
    title: str
    name: str
    is_indexed: bool
    file_type: str = ""
    description: str = ""
    language: str = "en"
    size: int = 0
    page_count: int = 0
    chunk_count: int = 0
    created_at: Optional[str] = None
    tags: List[str] = []
    category: str = ""
    author: str = ""
    status: str = "active"


class DocumentDetail(DocumentMetadata):
    """Detailed document information."""
    indexed_at: Optional[str] = None
    # Additional fields can be added here


class DocumentUploadMetadata(BaseModel):
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
    use_queue: bool = Form(False),
    parser_type: str = Form(ParserType.UNSTRUCTURED.value),
    extract_images: bool = Form(False)
):
    """Process a document using the specified parser and store in Neo4j."""

    # Validate parser type
    try:
        if parser_type not in [pt.value for pt in ParserType]:
            logger.warning(f"Invalid parser type: {parser_type}, using default: {ParserType.UNSTRUCTURED.value}")
            parser_type = ParserType.UNSTRUCTURED.value
    except Exception as e:
        logger.warning(f"Error validating parser type: {str(e)}, using default")
        parser_type = ParserType.UNSTRUCTURED.value

    # Add parser type to metadata
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse metadata JSON: {metadata}")
    
    metadata_dict["parser_type"] = parser_type
    
    # Add parser settings to metadata
    if "parser_settings" not in metadata_dict:
        metadata_dict["parser_settings"] = {}
    
    # Set extract_images in parser settings
    metadata_dict["parser_settings"]["extract_images"] = extract_images

    # Convert back to JSON string
    metadata = json.dumps(metadata_dict)

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
        use_queue=use_queue,
        parser_type=parser_type
    )

    return response


@router.get("/", response_model=List[DocumentMetadata])
async def document_list(
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    filter_tags: Optional[List[str]] = Query(None),
    filter_status: Optional[List[str]] = Query(None),
    search_query: Optional[str] = None
):
    """
    Get a list of documents with optional filtering and search.
    """
    # Validate sort parameters
    valid_sort_fields = ["created_at", "title", "file_type", "size", "language"]
    valid_sort_orders = ["asc", "desc"]

    if sort_by not in valid_sort_fields:
        sort_by = "created_at"
    if sort_order not in valid_sort_orders:
        sort_order = "desc"

    # Neo4j connection (assuming get_neo4j_driver() is available and preferred)
    # from neo4j import GraphDatabase
    # from config.settings import get_settings
    # settings = get_settings()
    # neo4j_driver = GraphDatabase.driver(
    #     settings.NEO4J_URI,
    #     auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    # )
    # For consistency with other parts of the codebase, let's assume a get_neo4j_driver exists
    from main import get_neo4j_driver # Assuming this function exists and provides the driver
    neo4j_driver = await get_neo4j_driver()


    documents = []
    try:
        with neo4j_driver.session() as session:
            query_conditions = []
            params = {"skip": skip, "limit": limit}

            base_query = "MATCH (d:Document)"

            if search_query:
                # Basic search: case-insensitive on title and name
                # More advanced search might use full-text indexes if configured in Neo4j
                query_conditions.append("(toLower(d.title) CONTAINS toLower($search_query) OR toLower(d.name) CONTAINS toLower($search_query))")
                params["search_query"] = search_query
            
            if filter_tags:
                # Assuming tags is a list property on the node
                query_conditions.append("ANY(tag IN $filter_tags WHERE tag IN d.tags)")
                params["filter_tags"] = filter_tags

            if filter_status:
                query_conditions.append("d.status IN $filter_status")
                params["filter_status"] = filter_status
            
            where_clause = ""
            if query_conditions:
                where_clause = "WHERE " + " AND ".join(query_conditions)

            # Define the expression for creation date, using UTC for Neo4j's datetime()
            created_at_expression = "COALESCE(d.created, d.created_at, datetime({timezone: 'UTC'}))"
            
            # Determine the actual field/expression to sort by
            if sort_by == "created_at":
                order_by_target_expression = created_at_expression
            elif sort_by == "size":
                 order_by_target_expression = f"COALESCE(d.{sort_by}, 0)" # Ensure size has a default for sorting
            else:
                # For other fields like title, file_type, language, assume they exist or COALESCE in RETURN handles display
                # If sorting on these can also fail due to missing props, they'd need COALESCE too.
                order_by_target_expression = f"d.{sort_by}"

            # Construct the full query
            full_query = f"""
            {base_query}
            {where_clause}
            WITH d // Ensure 'd' is available for RETURN and ORDER BY
            RETURN d.doc_id as doc_id,
                   COALESCE(d.title, d.name, 'Untitled') as title,
                   COALESCE(d.is_indexed, false) as is_indexed,
                   COALESCE(d.name, '') as name,
                   COALESCE(d.file_type, '') as file_type,
                   COALESCE(d.description, '') as description,
                   COALESCE(d.language, 'en') as language,
                   COALESCE(d.size, 0) as size,
                   COALESCE(d.page_count, 0) as page_count,
                   COALESCE(d.chunk_count, 0) as chunk_count,
                   {created_at_expression} as created_at, // Use the robust expression for the returned field
                   COALESCE(d.tags, []) as tags,
                   COALESCE(d.category, 'Uncategorized') as category,
                   COALESCE(d.author, 'Unknown') as author,
                   COALESCE(d.status, 'active') as status
            ORDER BY {order_by_target_expression} {sort_order}
            SKIP $skip LIMIT $limit
            """
            
            logger.debug(f"Executing Cypher query for document list: {{full_query}} with params: {{params}}")
            result = session.run(full_query, **params)

            # Process the results
            for record_data in result:
                doc = dict(record_data)
                
                created_at_val = doc.get("created_at")
                if isinstance(created_at_val, Neo4jDateTime):
                    doc["created_at"] = created_at_val.to_native().isoformat()
                elif isinstance(created_at_val, py_datetime):
                    doc["created_at"] = created_at_val.isoformat()
                elif created_at_val is None: # Should not happen with COALESCE to datetime()
                    logger.warning(f"Document {{doc.get('doc_id')}} has null created_at even after COALESCE. Defaulting to now.")
                    doc["created_at"] = py_datetime.now(py_datetime.timezone.utc).isoformat()
                # If it's already a string and potentially problematic, it will pass through here.
                # The frontend's new Date() will be the final judge.

                # Ensure all fields for DocumentMetadata are present with defaults
                # The COALESCE in Cypher should handle most of this.
                metadata_doc = DocumentMetadata(
                    doc_id=doc.get("doc_id", str(uuid.uuid4())),
                    title=doc.get("title", "Untitled"),
                    name=doc.get("name", ""),
                    is_indexed=doc.get("is_indexed", False),
                    file_type=doc.get("file_type", ""),
                    description=doc.get("description", ""),
                    language=doc.get("language", "en"),
                    size=doc.get("size", 0),
                    page_count=doc.get("page_count", 0),
                    chunk_count=doc.get("chunk_count", 0),
                    created_at=doc.get("created_at"), # Already processed
                    tags=doc.get("tags", []),
                    category=doc.get("category", "Uncategorized"),
                    author=doc.get("author", "Unknown"),
                    status=doc.get("status", "active")
                )
                documents.append(metadata_doc)

    except Exception as e:
        logger.error(f"Error fetching document list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        neo4j_driver.close()

    return documents


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
                RETURN COALESCE(d.file_type, 'unknown') as type, count(d) as count
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
                RETURN d.doc_id as doc_id, d.title as title, d.created_at as created,
                       COALESCE(d.file_type, 'unknown') as file_type, d.language as language,
                       COALESCE(d.file_name, 'N/A') as filename, d.size as size
                ORDER BY d.created_at DESC
                LIMIT 5
                """
            )
            recent_uploads_raw = [dict(record) for record in recent_result]
            
            recent_uploads = []
            for upload in recent_uploads_raw:
                created_val = upload.get('created')
                if isinstance(created_val, Neo4jDateTime):
                    # Convert to python datetime, then to ISO string
                    native_dt = created_val.to_native()
                    upload['created'] = native_dt.isoformat()
                elif isinstance(created_val, py_datetime):
                    # If it's already a Python datetime, ensure it's ISO string
                    upload['created'] = created_val.isoformat()
                recent_uploads.append(upload)

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


@router.get("/parsers", response_model=Dict[str, List[Dict[str, str]]])
async def get_parser_types():
    """Get available parser types for document processing."""
    parser_types = []
    
    for parser in ParserType:
        parser_info = {
            "id": parser.value,
            "name": parser.name.replace("_", " ").title(),
            "description": get_parser_description(parser)
        }
        parser_types.append(parser_info)
    
    return {"parsers": parser_types}

def get_parser_description(parser_type: ParserType) -> str:
    """Get the description for a parser type."""
    descriptions = {
        ParserType.UNSTRUCTURED: "Default document parser that extracts text and structure from various document formats",
        ParserType.UNSTRUCTURED_CLOUD: "Cloud-based version of Unstructured with enhanced capabilities",
        ParserType.DOCTLY: "Specialized parser for legal and regulatory documents with enhanced metadata extraction",
        ParserType.LLAMAPARSE: "AI-powered document parser based on LlamaIndex with advanced semantic understanding"
    }
    return descriptions.get(parser_type, "Document parser")

@router.post("/reindex-all", response_model=Dict[str, Any])
async def reindex_all_documents(force: bool = False):
    """Reindex all documents in the vector store."""
    try:
        # Get dependencies
        driver = await get_neo4j_driver()
        rag_system = await get_rag_system()

        # Get all document IDs from Neo4j
        with driver.session() as session:
            docs_result = session.run(
                """
                MATCH (d:Document)
                RETURN d.doc_id as doc_id, d.title as title, d.is_indexed as is_indexed
                """
            )
            
            documents = []
            for record in docs_result:
                doc_id = record["doc_id"]
                title = record["title"]
                is_indexed = record["is_indexed"]
                
                # Skip already indexed documents unless force is True
                if not force and is_indexed:
                    continue
                    
                documents.append({
                    "doc_id": doc_id,
                    "title": title
                })

        # No documents to reindex
        if not documents:
            return {
                "status": "success",
                "message": "No documents need reindexing",
                "total": 0,
                "documents": []
            }
            
        # Reindex each document
        results = []
        for doc in documents:
            doc_id = doc["doc_id"]
            title = doc["title"]
            
            try:
                logger.info(f"Reindexing document: {doc_id} - {title}")
                result = rag_system.index_document(doc_id, force_reindex=True)
                
                # Add document info to results
                doc_result = {
                    "doc_id": doc_id,
                    "title": title,
                    "status": result.get("status", "unknown"),
                    "vector_count": result.get("vector_count", 0)
                }
                results.append(doc_result)
                
                # Don't overload the system
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error reindexing document {doc_id}: {str(e)}")
                results.append({
                    "doc_id": doc_id,
                    "title": title,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "status": "success",
            "message": f"Reindexed {len(results)} documents",
            "total": len(results),
            "documents": results
        }

    except Exception as e:
        logger.error(f"Error reindexing all documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reindexing all documents: {str(e)}"
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


@router.get("/{doc_id}", response_model=Dict[str, Any])
async def get_document(doc_id: str, include_chunks: bool = False, include_entities: bool = False):
    """Get document metadata and optionally chunks and entities."""
    try:
        # Get neo4j driver
        driver = await get_neo4j_driver()

        with driver.session() as session:
            # Get document metadata
            doc_result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d.doc_id as doc_id, 
                       COALESCE(d.title, d.name) as title, 
                       COALESCE(d.name, '') as name,
                       COALESCE(d.is_indexed, false) as is_indexed,
                       COALESCE(d.file_type, '') as file_type, 
                       COALESCE(d.description, '') as description,
                       COALESCE(d.language, 'en') as language,
                       COALESCE(d.size, 0) as size,
                       COALESCE(d.page_count, 0) as page_count,
                       COALESCE(d.chunk_count, 0) as chunk_count,
                       COALESCE(d.created, d.created_at, datetime()) as created_at,
                       COALESCE(d.indexed_at, null) as indexed_at,
                       COALESCE(d.tags, []) as tags,
                       COALESCE(d.category, '') as category,
                       COALESCE(d.author, '') as author,
                       COALESCE(d.status, 'active') as status
                """,
                doc_id=doc_id
            )
            
            record = doc_result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
                
            document_raw = dict(record)

            # Convert created_at
            created_at_val = document_raw.get("created_at")
            if isinstance(created_at_val, Neo4jDateTime):
                document_raw["created_at"] = created_at_val.to_native().isoformat()
            elif isinstance(created_at_val, py_datetime):
                document_raw["created_at"] = created_at_val.isoformat()

            # Convert indexed_at
            indexed_at_val = document_raw.get("indexed_at")
            if indexed_at_val is not None: # Can be None
                if isinstance(indexed_at_val, Neo4jDateTime):
                    document_raw["indexed_at"] = indexed_at_val.to_native().isoformat()
                elif isinstance(indexed_at_val, py_datetime):
                    document_raw["indexed_at"] = indexed_at_val.isoformat()
            
            document = {
                "doc_id": document_raw.get("doc_id"),
                "title": document_raw.get("title"),
                "name": document_raw.get("name"),
                "is_indexed": document_raw.get("is_indexed"),
                "file_type": document_raw.get("file_type"),
                "description": document_raw.get("description"),
                "language": document_raw.get("language"),
                "size": document_raw.get("size"),
                "page_count": document_raw.get("page_count"),
                "chunk_count": document_raw.get("chunk_count"),
                "created_at": document_raw.get("created_at"), # Already converted
                "indexed_at": document_raw.get("indexed_at"), # Already converted
                "tags": document_raw.get("tags"),
                "category": document_raw.get("category"),
                "author": document_raw.get("author"),
                "status": document_raw.get("status")
            }
            
            # Get chunk count
            chunk_result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                RETURN count(c) as chunk_count
                """,
                doc_id=doc_id
            )
            
            chunk_record = chunk_result.single()
            document["chunk_count"] = chunk_record["chunk_count"] if chunk_record else 0
            
            return document
            
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        neo4j_driver.close()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, delete_from_index: bool = True):
    """Delete document and its chunks from Neo4j and optionally from vector store."""
    try:
        # Get dependencies
        driver = await get_neo4j_driver()
        rag_system = await get_rag_system()

        # First try to delete from vector index if requested
        if delete_from_index:
            try:
                # Use RAG system to delete from index
                logger.info(f"Deleting document {doc_id} from vector index")
                rag_deleted = rag_system.delete_document(doc_id)
                index_message = ", and deleted from vector index"
                
                if not rag_deleted:
                    logger.warning(f"Document {doc_id} may not exist in vector index or encountered errors during deletion")
            except Exception as e:
                # Log error but continue - we still want to delete from Neo4j
                logger.error(f"Error deleting document from vector index: {str(e)}")
                index_message = f", but failed to delete from vector index: {str(e)}"
                
                # Don't raise exception here - continue with Neo4j deletion

        # Then delete from Neo4j using a transaction to ensure atomicity
        try:
            with driver.session() as session:
                # Check if document exists
                doc_check = session.run(
                    "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) as count",
                    doc_id=doc_id
                )

                doc_exists = doc_check.single()["count"] > 0
                
                if not doc_exists:
                    # Before raising a 404, check if there are orphaned chunks to clean up
                    chunk_check = session.run(
                        """
                        MATCH (c:Chunk {doc_id: $doc_id})
                        RETURN count(c) as chunk_count
                        """,
                        doc_id=doc_id
                    )
                    
                    chunk_count = chunk_check.single()["chunk_count"]
                    
                    if chunk_count > 0:
                        # We have orphaned chunks but no document - clean them up
                        logger.warning(f"Document {doc_id} not found, but found {chunk_count} orphaned chunks to clean up")
                        
                        # Use a transaction to delete orphaned chunks
                        tx = session.begin_transaction()
                        try:
                            # Delete relationships from chunks first
                            tx.run(
                                """
                                MATCH (c:Chunk {doc_id: $doc_id})
                                OPTIONAL MATCH (c)-[r]-()
                                DELETE r
                                """,
                                doc_id=doc_id
                            )
                            
                            # Then delete the chunks
                            tx.run(
                                """
                                MATCH (c:Chunk {doc_id: $doc_id})
                                DELETE c
                                """,
                                doc_id=doc_id
                            )
                            
                            tx.commit()
                            logger.info(f"Cleaned up {chunk_count} orphaned chunks for document {doc_id}")
                            
                            return {
                                "doc_id": doc_id,
                                "deleted": False,
                                "document_found": False,
                                "chunks_deleted": chunk_count,
                                "message": f"Document not found but {chunk_count} orphaned chunks were cleaned up{locals().get('index_message', '')}"
                            }
                        except Exception as tx_error:
                            tx.rollback()
                            logger.error(f"Error cleaning up orphaned chunks: {str(tx_error)}")
                            raise tx_error
                    else:
                        # No document and no chunks - nothing to do
                        raise HTTPException(
                            status_code=404,
                            detail=f"Document not found: {doc_id}"
                        )
                
                # First count the chunks that will be deleted
                count_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                    RETURN count(c) as chunk_count
                    """,
                    doc_id=doc_id
                )
                
                chunks_count = count_result.single()["chunk_count"]
                if chunks_count is None:
                    chunks_count = 0
                
                # Also look for chunks with the doc_id but missing the proper relationship
                missing_rel_result = session.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})
                    WHERE NOT EXISTS {
                        MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c)
                    }
                    RETURN count(c) as orphan_count
                    """,
                    doc_id=doc_id
                )
                
                orphan_count = missing_rel_result.single()["orphan_count"]
                if orphan_count > 0:
                    logger.warning(f"Found {orphan_count} chunks missing proper relationships to document {doc_id}")
                
                total_chunks = chunks_count + orphan_count
                logger.info(f"Found {total_chunks} total chunks to delete for document {doc_id}")
                
                # Use a transaction for the deletion to ensure atomicity
                tx = session.begin_transaction()
                try:
                    # First delete the relationships between document-connected chunks and other nodes
                    if chunks_count > 0:
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                            OPTIONAL MATCH (c)-[cr]-()
                            DELETE cr
                            """,
                            doc_id=doc_id
                        )
                        
                        # Then delete the document-connected chunk nodes themselves
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                            DETACH DELETE c
                            """,
                            doc_id=doc_id
                        )
                    
                    # Delete any orphaned chunks with the same doc_id but missing relationships
                    if orphan_count > 0:
                        tx.run(
                            """
                            MATCH (c:Chunk {doc_id: $doc_id})
                            WHERE NOT EXISTS {
                                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c)
                            }
                            OPTIONAL MATCH (c)-[r]-()
                            DELETE r
                            """,
                            doc_id=doc_id
                        )
                        
                        tx.run(
                            """
                            MATCH (c:Chunk {doc_id: $doc_id})
                            WHERE NOT EXISTS {
                                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c)
                            }
                            DELETE c
                            """,
                            doc_id=doc_id
                        )
                    
                    # Delete document relationships
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        OPTIONAL MATCH (d)-[r]-()
                        DELETE r
                        """,
                        doc_id=doc_id
                    )
                    
                    # Finally delete the document node
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        DELETE d
                        """,
                        doc_id=doc_id
                    )
                    
                    # Commit the transaction
                    tx.commit()
                    logger.info(f"Deleted document {doc_id} with {total_chunks} chunks from Neo4j")
                    chunks_deleted = total_chunks
                except Exception as tx_error:
                    # Rollback on error
                    tx.rollback()
                    logger.error(f"Transaction error while deleting document from Neo4j: {str(tx_error)}")
                    raise tx_error
                
                # Verify deletion was successful
                verify_result = session.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})
                    RETURN count(c) as remaining_chunks
                    """,
                    doc_id=doc_id
                )
                
                remaining_chunks = verify_result.single()["remaining_chunks"]
                if remaining_chunks > 0:
                    logger.warning(f"Found {remaining_chunks} chunks still remaining after deletion")
                    
                    # Try one more direct deletion of any remaining chunks
                    cleanup_tx = session.begin_transaction()
                    try:
                        cleanup_tx.run(
                            """
                            MATCH (c:Chunk {doc_id: $doc_id})
                            OPTIONAL MATCH (c)-[r]-()
                            DELETE r
                            """,
                            doc_id=doc_id
                        )
                        
                        cleanup_tx.run(
                            """
                            MATCH (c:Chunk {doc_id: $doc_id})
                            DELETE c
                            """,
                            doc_id=doc_id
                        )
                        
                        cleanup_tx.commit()
                        logger.info(f"Cleaned up {remaining_chunks} remaining chunks in verification step")
                        chunks_deleted += remaining_chunks
                    except Exception as cleanup_error:
                        cleanup_tx.rollback()
                        logger.error(f"Error in verification cleanup: {str(cleanup_error)}")
                
        except Exception as neo4j_error:
            logger.error(f"Error deleting document from Neo4j: {str(neo4j_error)}")
            # Check if another error handling has already set index_message
            if not locals().get('index_message'):
                index_message = ""
                
            # If we failed to delete from both vector store and Neo4j, that's a bigger problem
            if "index_message" in locals() and "failed to delete from vector index" in index_message:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete document from both Neo4j and vector index. Neo4j error: {str(neo4j_error)}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error deleting document from Neo4j: {str(neo4j_error)}"
                )

        # If we get here, at least Neo4j deletion was successful
        return {
            "doc_id": doc_id,
            "deleted": True,
            "chunks_deleted": chunks_deleted,
            "message": f"Document and {chunks_deleted} chunks deleted from database{locals().get('index_message', '')}"
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
        # First check if document exists
        neo4j_driver = await get_neo4j_driver()
        
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) as count",
                doc_id=doc_id
            )
            
            if result.single()["count"] == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {doc_id} not found"
                )
        
        # Get RAG system and index document
        rag_system = await get_rag_system()
        index_result = rag_system.index_document(doc_id, force_reindex=force_reindex)
        
        # Check the result format
        if isinstance(index_result, dict):
            # Check if it already contains a "status" key
            if "status" in index_result:
                status = index_result["status"]
                vector_count = index_result.get("vector_count", 0)
                message = index_result.get("message", "")
            # Check if it has positive vector count
            elif "vector_count" in index_result and index_result["vector_count"] > 0:
                status = "success"
                vector_count = index_result["vector_count"]
                message = index_result.get("message", "Document indexed successfully")
            # Check if it's already indexed
            elif index_result.get("message") == "Document already indexed":
                status = "success"
                vector_count = 0
                message = "Document already indexed"
            # Otherwise consider it failed
            else:
                status = "error"
                vector_count = 0
                message = index_result.get("message", "Failed to index document")
        elif index_result:
            # Boolean True result
            status = "success"
            vector_count = None
            message = "Document successfully indexed"
        else:
            # Failed indexing
            status = "error"
            vector_count = 0
            message = "Failed to index document"
        
        return {
            "doc_id": doc_id,
            "status": status,
            "vector_count": vector_count,
            "message": message
        }
    
    except HTTPException as e:
        raise e
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


@router.post("/maintenance/standardize-properties", response_model=Dict[str, Any])
async def standardize_document_properties():
    """
    Maintenance endpoint to standardize document properties.
    This ensures all documents have consistent property names.
    """
    try:
        # Get neo4j driver
        driver = await get_neo4j_driver()

        with driver.session() as session:
            # Standardize file_type property
            file_type_result = session.run(
                """
                MATCH (d:Document)
                WHERE d.filetype IS NOT NULL AND d.file_type IS NULL
                SET d.file_type = d.filetype
                RETURN count(d) as updated_count
                """
            )
            updated_file_type = file_type_result.single()["updated_count"]

            # Add other property standardizations here if needed

            return {
                "status": "success",
                "message": f"Standardized document properties successfully",
                "updated_file_type_count": updated_file_type
            }

    except Exception as e:
        logger.error(f"Error standardizing document properties: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error standardizing document properties: {str(e)}"
        )


@router.post("/maintenance/cleanup-orphaned-chunks", response_model=Dict[str, Any])
async def cleanup_orphaned_chunks():
    """Maintenance endpoint to clean up orphaned chunks that have no associated document."""
    try:
        # Get Neo4j driver
        driver = await get_neo4j_driver()
        
        with driver.session() as session:
            # First count all orphaned chunks
            count_result = session.run(
                """
                MATCH (c:Chunk)
                WHERE NOT EXISTS {
                    MATCH (d:Document {doc_id: c.doc_id})
                }
                RETURN count(c) as orphan_count
                """
            )
            
            orphan_count = count_result.single()["orphan_count"]
            
            if orphan_count == 0:
                return {
                    "status": "success",
                    "message": "No orphaned chunks found to clean up",
                    "chunks_deleted": 0
                }
            
            # Get the doc_ids of orphaned chunks for reporting
            doc_ids_result = session.run(
                """
                MATCH (c:Chunk)
                WHERE NOT EXISTS {
                    MATCH (d:Document {doc_id: c.doc_id})
                }
                RETURN c.doc_id as doc_id, count(c) as chunk_count
                """
            )
            
            orphaned_docs = [{"doc_id": record["doc_id"], "chunk_count": record["chunk_count"]} 
                            for record in doc_ids_result]
            
            # Start a transaction for deletion
            tx = session.begin_transaction()
            try:
                # Delete relationships from orphaned chunks first
                rel_result = tx.run(
                    """
                    MATCH (c:Chunk)
                    WHERE NOT EXISTS {
                        MATCH (d:Document {doc_id: c.doc_id})
                    }
                    OPTIONAL MATCH (c)-[r]-()
                    DELETE r
                    RETURN count(r) as rel_count
                    """
                )
                
                rel_count = rel_result.single()["rel_count"]
                
                # Now delete the orphaned chunks
                tx.run(
                    """
                    MATCH (c:Chunk)
                    WHERE NOT EXISTS {
                        MATCH (d:Document {doc_id: c.doc_id})
                    }
                    DELETE c
                    """
                )
                
                tx.commit()
                logger.info(f"Cleaned up {orphan_count} orphaned chunks with {rel_count} relationships")
                
                return {
                    "status": "success",
                    "message": f"Successfully cleaned up {orphan_count} orphaned chunks",
                    "chunks_deleted": orphan_count,
                    "relationships_deleted": rel_count,
                    "affected_documents": orphaned_docs
                }
            except Exception as tx_error:
                tx.rollback()
                logger.error(f"Error in cleanup transaction: {str(tx_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Transaction error during cleanup: {str(tx_error)}"
                )
    
    except Exception as e:
        logger.error(f"Error in orphaned chunks cleanup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up orphaned chunks: {str(e)}"
        )


@router.delete("/maintenance/force-delete-chunks/{doc_id}", response_model=Dict[str, Any])
async def force_delete_chunks(doc_id: str):
    """Force delete all chunks with a given doc_id, even if the document itself doesn't exist."""
    try:
        # Get Neo4j driver
        driver = await get_neo4j_driver()
        
        with driver.session() as session:
            # First count the chunks to delete
            count_result = session.run(
                """
                MATCH (c:Chunk {doc_id: $doc_id})
                RETURN count(c) as chunk_count
                """,
                doc_id=doc_id
            )
            
            chunk_count = count_result.single()["chunk_count"]
            
            if chunk_count == 0:
                return {
                    "status": "success",
                    "message": f"No chunks found for document ID {doc_id}",
                    "chunks_deleted": 0
                }
            
            # Start a transaction for deletion
            tx = session.begin_transaction()
            try:
                # Delete relationships from chunks first
                rel_result = tx.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})
                    OPTIONAL MATCH (c)-[r]-()
                    DELETE r
                    RETURN count(r) as rel_count
                    """,
                    doc_id=doc_id
                )
                
                rel_count = rel_result.single()["rel_count"]
                
                # Now delete the chunks
                tx.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id})
                    DELETE c
                    """,
                    doc_id=doc_id
                )
                
                tx.commit()
                logger.info(f"Force deleted {chunk_count} chunks for document {doc_id}")
                
                # Also check if document exists
                doc_check = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN count(d) as doc_count
                    """,
                    doc_id=doc_id
                )
                
                doc_exists = doc_check.single()["doc_count"] > 0
                
                # If document still exists and user wants to delete chunks, they probably
                # want to delete the document too
                if doc_exists:
                    doc_tx = session.begin_transaction()
                    try:
                        # Delete document relationships
                        doc_tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            OPTIONAL MATCH (d)-[r]-()
                            DELETE r
                            """,
                            doc_id=doc_id
                        )
                        
                        # Delete document node
                        doc_tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            DELETE d
                            """,
                            doc_id=doc_id
                        )
                        
                        doc_tx.commit()
                        logger.info(f"Also deleted document node for {doc_id}")
                        
                        return {
                            "status": "success",
                            "message": f"Successfully force deleted {chunk_count} chunks and document node for {doc_id}",
                            "chunks_deleted": chunk_count,
                            "relationships_deleted": rel_count,
                            "document_deleted": True
                        }
                    except Exception as doc_error:
                        doc_tx.rollback()
                        logger.error(f"Error deleting document node: {str(doc_error)}")
                        # Continue with just reporting chunk deletion
                
                return {
                    "status": "success",
                    "message": f"Successfully force deleted {chunk_count} chunks for document {doc_id}",
                    "chunks_deleted": chunk_count,
                    "relationships_deleted": rel_count,
                    "document_deleted": False
                }
                
            except Exception as tx_error:
                tx.rollback()
                logger.error(f"Error in force delete transaction: {str(tx_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Transaction error during force delete: {str(tx_error)}"
                )
    
    except Exception as e:
        logger.error(f"Error force deleting chunks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error force deleting chunks: {str(e)}"
        )
