# plugins/regul_aite/backend/main.py
from fastapi import FastAPI, HTTPException, Depends, Body, File, UploadFile, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import os
import time
import uuid
import json
import mariadb
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime, timedelta
import neo4j
from neo4j import GraphDatabase
from neo4j.time import DateTime as Neo4jDateTime
from openai import OpenAI
from unstructured_parser.base_parser import BaseParser, ParserType
import threading
import sys
import random
import string
import shutil
import tempfile
import asyncio
import requests
import uvicorn
import mysql.connector
from enum import Enum

# Custom JSON encoder for Neo4j DateTime objects
class Neo4jDateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Neo4jDateTime):
            # Convert Neo4j DateTime to Python datetime
            dt = datetime(
                obj.year, obj.month, obj.day,
                obj.hour, obj.minute, obj.second,
                obj.nanosecond // 1000000
            )
            return dt.isoformat()
        return super().default(obj)

# Import task router from routers package instead of directly from queuing_sys
from routers.task_router import task_router

# Import document parser
from unstructured_parser.document_parser import DocumentParser

# Import RAG system
from llamaIndex_rag.rag import RAGSystem

# Import Pyndantic Agents router
from pyndantic_agents.router import router as agents_router

from pyndantic_agents.tree_reasoning import TreeReasoningAgent, DecisionNode, DecisionTree, create_default_decision_tree
from pyndantic_agents.decision_trees import get_available_trees

# Import our new custom routers
from routers.chat_router import router as chat_router
from routers.document_router import router as document_router
from routers.config_router import router as config_router
from routers.agents_router import router as agents_metadata_router
from routers.welcome_router import router as welcome_router
from routers.auth_router import router as auth_router

# Import model preloading function
from llamaIndex_rag.preload_models import preload_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI instance
app = FastAPI(
    title="RegulAIte API",
    description="Backend API for RegulAIte application",
    version="1.0.0",
    # Add custom JSON encoder for Neo4j DateTime objects to ensure proper serialization
    json_encoders={Neo4jDateTime: lambda dt: datetime(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second,
        dt.nanosecond // 1000000
    ).isoformat()}
)

app.include_router(task_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(agents_router)
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(config_router)
app.include_router(agents_metadata_router)
app.include_router(welcome_router)
app.include_router(auth_router)

# Configuration from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
MARIADB_HOST = os.getenv("MARIADB_HOST", "mariadb")
MARIADB_DATABASE = os.getenv("MARIADB_DATABASE", "regulaite")
MARIADB_USER = os.getenv("MARIADB_USER", "regulaite_user")
MARIADB_PASSWORD = os.getenv("MARIADB_PASSWORD", "SecureP@ssw0rd!")

# Connect to Neo4j with retry logic
driver = None
max_retries = 3
retry_count = 0
last_error = None

while retry_count < max_retries and driver is None:
    try:
        logger.info(f"Attempting to connect to Neo4j at {NEO4J_URI} (attempt {retry_count + 1}/{max_retries})")

        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600
        )

        # Verify connectivity
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' as status")
            for record in result:
                logger.info(f"Neo4j connection: {record['status']}")

        logger.info(f"Connected to Neo4j at {NEO4J_URI}")

    except Exception as e:
        last_error = e
        retry_count += 1

        # Log the specific error
        if "AuthenticationRateLimit" in str(e):
            logger.error("Neo4j authentication rate limit reached. Waiting before retry...")
            # Wait longer when hitting rate limit
            time.sleep(30)  # 30 second delay
        else:
            logger.error(f"Failed to connect to Neo4j (attempt {retry_count}/{max_retries}): {str(e)}")
            time.sleep(5)  # 5 second delay for other errors

if driver is None:
    logger.error(f"Failed to connect to Neo4j after {max_retries} attempts. Last error: {str(last_error)}")
    # Continue without failing to allow for retries

# Initialize document parser with Neo4j credentials
document_parser = DocumentParser(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    use_enrichment=True
)

# Initialize RAG system with Qdrant
rag_system = RAGSystem(
    neo4j_uri=NEO4J_URI,
    neo4j_user=NEO4J_USER,
    neo4j_password=NEO4J_PASSWORD,
    qdrant_url=QDRANT_URL,
    openai_api_key=OPENAI_API_KEY,
    hybrid_search=True,     # Enable hybrid search
    vector_weight=0.7,      # Set vector search weight
    keyword_weight=0.3      # Set keyword search weight
)

# Model preloading thread
model_thread = None
model_loading_status = {
    "started": False, 
    "completed": False, 
    "results": None, 
    "start_time": None,
    "end_time": None,
    "languages": [],
    "progress": 0,
    "total_languages": 0,
    "current_language": None,
    "error": None,
    "language_status": {},  # Will track per-language loading status
    "language_errors": {}  # Will track per-language errors
}

@app.on_event("startup")
def startup_event():
    """
    Startup event handler to initialize application
    """
    logger.info("Starting RegulAIte API...")
    
    # Start model preloading in a separate thread
    # Default to multi-language model for better coverage
    start_model_preloading(['multi'])

def start_model_preloading(languages: List[str]):
    """
    Start preloading embedding models for the specified languages in a background thread
    
    Args:
        languages: List of language codes to preload models for
    """
    global model_thread, model_loading_status
    
    # If preloading is already in progress, don't start again
    if model_loading_status["started"] and not model_loading_status["completed"]:
        logger.info("Model preloading already in progress")
        return
    
    # If previous preloading failed with an error, clear the error state before starting again
    if model_loading_status.get("error"):
        logger.info("Clearing previous preloading error state before starting new preloading")
    
    if not languages:
        languages = ['en', 'it', 'de', 'multi']  # Default set of languages
    
    # Initialize/reset the model loading status
    model_loading_status = {
        "started": True,
        "completed": False,
        "languages": languages,
        "current_language": None,
        "progress": 0,
        "start_time": time.time(),
        "end_time": None,
        "results": None,
        "error": None,
        "language_status": {},  # Will track per-language loading status
        "language_errors": {},  # Will track per-language errors
        "total_languages": len(languages)
    }
    
    def preload_worker():
        try:
            logger.info(f"Starting model preloading for languages: {languages}")
            
            # Define the progress callback function
            def progress_callback(language, progress, completed, error):
                # Update the model loading status with current language and progress
                model_loading_status["current_language"] = language
                
                # Calculate overall progress based on individual language progress
                # Find the index of current language in languages list
                try:
                    lang_idx = languages.index(language)
                    # Calculate overall progress: 
                    # (completed languages * 100 + current language progress) / total languages
                    overall_progress = (lang_idx * 100 + progress) / len(languages)
                    model_loading_status["progress"] = round(overall_progress, 1)
                except ValueError:
                    # Language not found in list, use progress as is
                    model_loading_status["progress"] = progress
                
                # If there was an error, log it
                if error:
                    logger.error(f"Error loading model for language '{language}': {error}")
                    model_loading_status["language_errors"][language] = error
                
                # If a language completed loading, update its status
                if completed:
                    if not error:
                        logger.info(f"Completed loading model for language '{language}'")
                    model_loading_status["language_status"][language] = {
                        "completed": True,
                        "success": not bool(error),
                        "error": error
                    }
            
            # Call the preload_models function with the requested languages and progress callback
            results = preload_models(languages, progress_callback=progress_callback)
            
            # Calculate overall stats
            elapsed = time.time() - model_loading_status["start_time"]
            success_count = sum(1 for status in results.values() if status)
            
            logger.info(f"Model preloading completed in {elapsed:.2f} seconds. "
                       f"Success: {success_count}/{len(languages)} languages")
            
            # Update final status
            model_loading_status["completed"] = True
            model_loading_status["results"] = results
            model_loading_status["end_time"] = time.time()
            model_loading_status["progress"] = 100
            
        except Exception as e:
            error_msg = f"Critical error during model preloading: {str(e)}"
            logger.error(error_msg)
            
            # Update error status
            model_loading_status["completed"] = True
            model_loading_status["results"] = {"error": str(e)}
            model_loading_status["end_time"] = time.time()
            model_loading_status["error"] = error_msg
    
    # Create and start the thread with a meaningful name
    model_thread = threading.Thread(
        target=preload_worker, 
        daemon=True,
        name=f"ModelPreloader-{'-'.join(languages)}"
    )
    model_thread.start()
    logger.info(f"Model preloading thread started for languages: {', '.join(languages)}")

@app.get("/api/status")
def get_status():
    """
    Get API status including model preloading status
    """
    status_info = {
        "status": "running",
        "model_loading": {
            "started": model_loading_status["started"],
            "completed": model_loading_status["completed"],
            "languages": model_loading_status["languages"],
            "current_language": model_loading_status["current_language"],
            "progress": model_loading_status["progress"]
        }
    }
    
    # Include timing information if available
    if model_loading_status["start_time"]:
        status_info["model_loading"]["start_time"] = model_loading_status["start_time"]
        
        if model_loading_status["end_time"]:
            status_info["model_loading"]["end_time"] = model_loading_status["end_time"]
            status_info["model_loading"]["elapsed_seconds"] = round(
                model_loading_status["end_time"] - model_loading_status["start_time"], 2)
    
    # Include detailed language status if available
    if model_loading_status.get("language_status"):
        status_info["model_loading"]["language_status"] = model_loading_status["language_status"]
    
    # Include language-specific errors if any
    if model_loading_status.get("language_errors"):
        status_info["model_loading"]["language_errors"] = model_loading_status["language_errors"]
    
    # Include results if completed
    if model_loading_status["completed"] and model_loading_status["results"]:
        status_info["model_loading"]["results"] = model_loading_status["results"]
    
    # Include error information if any
    if model_loading_status.get("error"):
        status_info["model_loading"]["error"] = model_loading_status["error"]
    
    return status_info

@app.get("/")
def root():
    """
    Root endpoint
    """
    return {"message": "Welcome to RegulAIte API"}

# Models
class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    stream: bool = Field(False, description="Whether to stream the response")
    model: str = Field("gpt-4", description="Model to use for generation")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: int = Field(2048, description="Maximum tokens in response")
    include_context: bool = Field(True, description="Whether to include RAG context")
    context_query: Optional[str] = Field(None, description="Query to use for retrieving context")
    use_agent: bool = Field(False, description="Whether to use an agent for processing")
    agent_type: Optional[str] = Field(None, description="Type of agent to use if use_agent is True")
    agent_params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for the agent")
    use_tree_reasoning: bool = Field(False, description="Whether to use tree-based reasoning")
    tree_template: Optional[str] = Field(None, description="ID of the decision tree template to use")
    custom_tree: Optional[Dict[str, Any]] = Field(None, description="Custom decision tree for reasoning")


class SearchRequest(BaseModel):
    """Request for search."""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Maximum number of results")
    filter_criteria: Optional[Dict[str, Any]] = Field(None, description="Metadata filter criteria")


class SearchResult(BaseModel):
    """Result from a search query."""
    entity: Optional[str] = None
    related: Optional[str] = None
    relationship: Optional[str] = None
    document: Optional[str] = None
    section: Optional[str] = None
    relevance: Optional[float] = None


class SearchResponse(BaseModel):
    """Response for a search query."""
    results: List[SearchResult]
    query: str
    timestamp: str


class DocumentProcessResponse(BaseModel):
    """Response for document processing."""
    doc_id: str
    filename: str
    chunk_count: int
    status: str
    message: str


class ContextRequest(BaseModel):
    """Request for retrieving context."""
    query: str = Field(..., description="Query to use for retrieving context")
    limit: int = Field(5, description="Maximum number of results to return")
    agent_id: Optional[str] = Field(None, description="ID of the agent requesting context (for logging)")
    use_neo4j: bool = Field(True, description="Whether to include Neo4j graph-based context")


# Helper functions
def get_db():
    """Get Neo4j database connection."""
    if driver is None:
        raise HTTPException(status_code=500, detail="Database connection not available")
    return driver


def get_mariadb_connection():
    """Get a connection to MariaDB."""
    conn = None
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries and conn is None:
        try:
            logger.info(f"Attempting to connect to MariaDB at {MARIADB_HOST} (attempt {retry_count + 1}/{max_retries})")
            
            conn = mariadb.connect(
                host=MARIADB_HOST,
                user=MARIADB_USER,
                password=MARIADB_PASSWORD,
                database=MARIADB_DATABASE,
                port=3306,
                autocommit=False  # Disable autocommit to manage transactions
            )
            
            # Verify connectivity with a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            logger.info(f"Connected to MariaDB at {MARIADB_HOST}")
            
            # Initialize database tables if needed
            initialize_database(conn)
            
            return conn
            
        except mariadb.Error as e:
            last_error = e
            retry_count += 1
            logger.error(f"Failed to connect to MariaDB (attempt {retry_count}/{max_retries}): {str(e)}")
            time.sleep(5)  # 5 second delay between retries
    
    logger.error(f"Failed to connect to MariaDB after {max_retries} attempts. Last error: {str(last_error)}")
    raise HTTPException(
        status_code=500,
        detail="Database connection not available"
    )


def initialize_database(conn):
    """Initialize database tables if they don't exist."""
    try:
        cursor = conn.cursor()
        
        # Create regulaite_settings table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS regulaite_settings (
            setting_key VARCHAR(255) PRIMARY KEY,
            setting_value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            description TEXT
        ) ENGINE=InnoDB;
        """)
        
        # Insert default settings if they don't exist
        cursor.execute("""
        INSERT IGNORE INTO regulaite_settings (setting_key, setting_value, description) VALUES
        ('llm_model', 'gpt-4', 'Default LLM model'),
        ('llm_temperature', '0.7', 'Default temperature for LLM'),
        ('llm_max_tokens', '2048', 'Default max tokens for LLM'),
        ('llm_top_p', '1', 'Default top_p value for LLM'),
        ('enable_chat_history', 'true', 'Whether to save chat history');
        """)
        
        # Create users table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(36) PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(255) NOT NULL,
            company VARCHAR(255),
            username VARCHAR(255) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL,
            settings JSON
        ) ENGINE=InnoDB;
        """)
        
        # Create refresh tokens table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token_id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            refresh_token VARCHAR(255) NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """)
        
        # Create other necessary tables...
        
        conn.commit()
        cursor.close()
    except mariadb.Error as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()


@app.get("/")
def read_root():
    """Root endpoint."""
    logger.info("Root endpoint called")
    return {
        "status": "ok",
        "message": "RegulAite API is running",
        "version": "1.0.0",
        "neo4j_connected": driver is not None,
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint called")

    neo4j_status = "disconnected"
    if driver:
        try:
            with driver.session() as session:
                result = session.run("RETURN 1 as n")
                if result.single()["n"] == 1:
                    neo4j_status = "connected"
        except:
            pass

    qdrant_status = "disconnected"
    try:
        # Simple check if Qdrant client is initialized
        if rag_system.qdrant_client:
            collections = rag_system.qdrant_client.get_collections()
            if collections:
                qdrant_status = "connected"
    except:
        pass

    mariadb_status = "disconnected"
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        mariadb_status = "connected"
    except:
        pass

    return {
        "status": "healthy",
        "components": {
            "neo4j": neo4j_status,
            "qdrant": qdrant_status,
            "mariadb": mariadb_status,
            "api": "healthy"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/documents/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    use_nlp: bool = Form(True),
    use_enrichment: bool = Form(True),
    detect_language: bool = Form(True),
    language: Optional[str] = Form(None),
    use_queue: bool = Form(False),
    parser_type: str = Form(ParserType.UNSTRUCTURED.value)
):
    """Process a document and store it in the database."""
    try:
        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4()}"

        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata JSON: {metadata}")

        # Add file metadata
        doc_metadata["original_filename"] = file.filename
        doc_metadata["content_type"] = file.content_type
        doc_metadata["size"] = 0  # Will be updated with actual size
        doc_metadata["use_nlp"] = use_nlp
        doc_metadata["use_enrichment"] = use_enrichment
        doc_metadata["parser_type"] = parser_type  # Store the parser type used

        # Extract file extension and store file_type
        file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
        doc_metadata["file_type"] = file_ext  # Standardized to file_type

        # Add language info if provided
        if language:
            doc_metadata["language"] = language
            doc_metadata["language_detect"] = False
        else:
            doc_metadata["language_detect"] = detect_language

        # Read file content
        file_content = await file.read()
        doc_metadata["size"] = len(file_content)

        # Use queue for processing if requested
        if use_queue:
            # Queue the document processing task
            from routers.task_router import queue_document_processing
            
            # Create a new UploadFile with the read content reset
            file.file.seek(0)
            
            response = await queue_document_processing(
                file=file,
                doc_id=doc_id,
                metadata=json.dumps(doc_metadata),
                use_nlp=use_nlp,
                use_enrichment=use_enrichment,
                detect_language=detect_language,
                language=language,
                parser_type=parser_type
            )
            
            return {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunk_count": 0,
                "status": "queued",
                "message": f"Document queued for processing (Task ID: {response.task_id})"
            }

        # Process document - get the appropriate parser if not using the default
        parser = document_parser
        if parser_type != ParserType.UNSTRUCTURED.value:
            # Get parser using the factory method
            try:
                # Get parser using the factory method
                parser = BaseParser.get_parser(
                    parser_type=ParserType(parser_type),
                    neo4j_uri=os.getenv("NEO4J_URI"),
                    neo4j_user=os.getenv("NEO4J_USER"),
                    neo4j_password=os.getenv("NEO4J_PASSWORD")
                )
            except Exception as e:
                logger.error(f"Error creating parser of type {parser_type}: {str(e)}")
                # Fall back to default parser
                parser = document_parser
        
        # Check if extract_images is in parser_settings
        extract_images = False
        if "parser_settings" in doc_metadata and isinstance(doc_metadata["parser_settings"], dict):
            if "extract_images" in doc_metadata["parser_settings"]:
                extract_images = bool(doc_metadata["parser_settings"]["extract_images"])
                logger.info(f"Setting extract_images={extract_images} from parser settings")
                
                # If using the default parser, we need to modify its settings
                if parser == document_parser:
                    parser.extract_images = extract_images
        
        # Process the document with the selected parser
        try:
            result = parser.process_document(
                file_content=file_content,
                file_name=file.filename,
                doc_id=doc_id,
                doc_metadata=doc_metadata,
                enrich=use_enrichment,
                detect_language=detect_language
            )

            # Extract statistics from the result
            processed_doc_id = result["doc_id"]
            chunk_count = result.get("chunk_count", 0)
            section_count = result.get("section_count", 0)
            entity_count = result.get("entity_count", 0)
            concept_count = result.get("concept_count", 0)
            requirement_count = result.get("requirement_count", 0)
            has_regulatory_content = result.get("has_regulatory_content", False)
            detected_language = result.get("language")
            language_name = result.get("language_name")
            image_count = result.get("image_count", 0)

            # Create language info part of the message
            language_msg = ""
            if detected_language:
                language_msg = f" (Language: {language_name or detected_language})"
                
            # Add image extraction info to message
            image_msg = ""
            if extract_images and image_count > 0:
                image_msg = f" with {image_count} extracted images"

            # Index document in Qdrant through RAG system
            try:
                # Check if indexing is configured
                index_immediately = True  # Default to True
                try:
                    # Get setting from MariaDB
                    conn = get_mariadb_connection()
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute(
                        """
                        SELECT setting_value
                        FROM regulaite_settings
                        WHERE setting_key = 'doc_index_immediately'
                        """
                    )
                    result = cursor.fetchone()
                    if result:
                        index_immediately = result['setting_value'].lower() == 'true'
                    conn.close()
                except Exception as config_e:
                    logger.warning(f"Could not get index_immediately setting, using default: {str(config_e)}")
                
                if index_immediately:
                    logger.info(f"Indexing document {processed_doc_id} in Qdrant")
                    index_result = rag_system.index_document(processed_doc_id)
                    
                    if isinstance(index_result, dict):
                        if index_result.get("status") == "success":
                            vector_count = index_result.get("vector_count", 0)
                            logger.info(f"Document {processed_doc_id} indexed in Qdrant with {vector_count} vectors")
                        elif "vector_count" in index_result and index_result.get("vector_count", 0) > 0:
                            vector_count = index_result.get("vector_count", 0)
                            logger.info(f"Document {processed_doc_id} indexed in Qdrant with {vector_count} vectors")
                        elif index_result.get("message") == "Document already indexed":
                            logger.info(f"Document {processed_doc_id} was already indexed")
                        else:
                            logger.warning(f"Indexing completed but may have issues: {index_result}")
                    else:
                        logger.warning(f"Unexpected indexing result format: {type(index_result)} - {index_result}")
                else:
                    logger.info(f"Immediate indexing disabled, document {processed_doc_id} will not be indexed now")
            except Exception as e:
                logger.error(f"Error indexing document in Qdrant: {str(e)}", exc_info=True)
                # Continue without failing as the document is already processed in Neo4j

            # Make sure to close the custom parser if we created one
            if parser != document_parser:
                try:
                    parser.close()
                except:
                    pass

            return {
                "doc_id": processed_doc_id,
                "filename": file.filename,
                "chunk_count": chunk_count,
                "status": "success",
                "message": f"Document processed successfully with {parser_type} parser{language_msg} with {chunk_count} chunks, {section_count} sections, {entity_count} entities, and {concept_count} concepts{image_msg}" +
                          (f", including {requirement_count} regulatory requirements" if has_regulatory_content else "")
            }

        except Exception as e:
            # Make sure to close the custom parser if we created one
            if parser != document_parser:
                try:
                    parser.close()
                except:
                    pass
                    
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error handling document upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error handling document upload: {str(e)}"
        )


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document metadata and chunks."""
    try:
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

            # Get document chunks
            chunks_result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                RETURN c
                ORDER BY c.index
                """,
                doc_id=doc_id
            )

            chunks = [dict(record["c"]) for record in chunks_result]

            return {
                "document": document,
                "chunks": chunks,
                "chunk_count": len(chunks)
            }

    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document: {str(e)}"
        )


@app.get("/documents")
async def list_documents(limit: int = 10, offset: int = 0):
    """List all documents."""
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                RETURN d
                ORDER BY d.created_at DESC
                SKIP $offset
                LIMIT $limit
                """,
                offset=offset,
                limit=limit
            )

            documents = [dict(record["d"]) for record in result]

            # Get total count
            count_result = session.run("MATCH (d:Document) RETURN count(d) as count")
            total_count = count_result.single()["count"]

            return {
                "documents": documents,
                "total": total_count,
                "limit": limit,
                "offset": offset
            }

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for documents and entities related to a query."""
    try:
        # Extract filter criteria if present
        filter_criteria = request.filter_criteria if hasattr(request, "filter_criteria") else None

        # Use RAG system to perform search with hybrid retrieval
        results = rag_system.retrieve(
            request.query,
            top_k=request.limit,
            use_hybrid=True,
            filter_criteria=filter_criteria
        )

        # Format results
        search_results = []
        for result in results:
            search_result = SearchResult(
                document=result["metadata"].get("doc_name"),
                section=result["metadata"].get("section"),
                relevance=result.get("score", 0.0)
            )
            search_results.append(search_result)

        return SearchResponse(
            results=search_results,
            query=request.query,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )


@app.post("/context/retrieve")
async def retrieve_context(request: ContextRequest):
    """Retrieve context from RAG system for agents or UI."""
    try:
        # Log the request
        logger.info(f"Context retrieval request: {request.query[:100]}... (Agent: {request.agent_id or 'None'})")

        # Use RAG system to retrieve context with the new parameter
        results = rag_system.retrieve(
            request.query,
            top_k=request.limit,
            use_neo4j=request.use_neo4j
        )

        # Format results for return
        formatted_results = []
        for result in results:
            # Get source information
            if "source_type" in result and result["source_type"] == "graph_database":
                source = "Neo4j Graph"
                if "related_entities" in result["metadata"] and result["metadata"]["related_entities"]:
                    source += f" (Related entities: {', '.join(result['metadata']['related_entities'][:3])})"
            else:
                source = result["metadata"].get("doc_name", "Unknown document")
                if "section" in result["metadata"] and result["metadata"]["section"] != "Unknown":
                    source += f" - {result['metadata']['section']}"

            formatted_result = {
                "text": result["text"],
                "source": source,
                "score": result.get("score", 0.0),
                "metadata": result["metadata"],
                "source_type": result.get("source_type", "vector_database")
            }
            formatted_results.append(formatted_result)

        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}"
        )


@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    """Chat with the RAG-enhanced LLM or AI agent."""
    try:
        # Get messages from request
        messages = request.messages

        # Get the last user message
        user_message = next((m.content for m in reversed(messages) if m.role == "user"), None)

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message found in the chat history"
            )

        # If agent-based processing is requested
        if request.use_agent and request.agent_type:

            # ---- TREE REASONING SECTION START ----
            # Check if tree reasoning is requested
            if request.use_tree_reasoning:
                # Create agent configuration
                agent_config = AgentConfig(
                    name="Tree Reasoning Agent",
                    description="Agent for tree-based reasoning",
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    include_context=request.include_context,
                    context_query=request.context_query,
                    max_context_results=5
                )

                # Determine which decision tree to use
                decision_tree = None

                # Use custom tree if provided
                if request.custom_tree:
                    try:
                        decision_tree = DecisionTree(**request.custom_tree)
                        logger.info(f"Using custom decision tree: {decision_tree.name}")
                    except Exception as e:
                        logger.error(f"Error parsing custom decision tree: {str(e)}")
                        # Fall through to other options if parsing fails

                # Use template tree if specified (and custom tree wasn't used)
                if decision_tree is None and request.tree_template:
                    available_trees = get_available_trees()
                    if request.tree_template == "default":
                        decision_tree = create_default_decision_tree()
                        logger.info(f"Using default decision tree template")
                    elif request.tree_template in available_trees:
                        decision_tree = available_trees[request.tree_template]
                        logger.info(f"Using template tree: {request.tree_template} - {decision_tree.name}")
                    else:
                        logger.warning(f"Template '{request.tree_template}' not found, using default tree")
                        decision_tree = create_default_decision_tree()

                # Use default decision tree if no other option was successful
                if decision_tree is None:
                    decision_tree = create_default_decision_tree()
                    logger.info(f"Using default decision tree (none specified or failed to parse)")

                # Create tree reasoning agent
                agent = TreeReasoningAgent(
                    agent_id=f"tree_agent_{uuid.uuid4()}",
                    config=agent_config,
                    decision_tree=decision_tree,
                    neo4j_uri=NEO4J_URI,
                    neo4j_user=NEO4J_USER,
                    neo4j_password=NEO4J_PASSWORD,
                    openai_api_key=OPENAI_API_KEY,
                    rag_system=rag_system
                )

                # Execute the task
                result = agent.execute(user_message)

                # Clean up
                agent.close()

                # Format response from tree reasoning
                if "final_result" in result and "response" in result["final_result"]:
                    assistant_message = result["final_result"]["response"]
                elif "final_result" in result and isinstance(result["final_result"], dict) and "completion" in result["final_result"]:
                    assistant_message = result["final_result"]["completion"]
                elif "final_result" in result:
                    assistant_message = str(result["final_result"])
                else:
                    assistant_message = f"Tree-based reasoning completed. Visited {len(result.get('visited_nodes', []))} decision nodes."

                # Include reasoning explanation if available
                if "reasoning_explanation" in result:
                    # Only include the explanation if specifically requested or for debugging
                    if request.agent_params and request.agent_params.get("show_reasoning", False):
                        assistant_message += "\n\n" + result["reasoning_explanation"]
            # ---- TREE REASONING SECTION END ----

            else:
                # Original agent-based processing
                # Import agent-related components
                from pyndantic_agents.agents import create_agent, AgentConfig

                # Create an agent configuration
                agent_config = AgentConfig(
                    name=f"{request.agent_type.capitalize()} Agent",
                    description=f"Agent for handling {request.agent_type} tasks",
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    include_context=request.include_context,
                    context_query=request.context_query,
                    max_context_results=5
                )

                # Create the agent with the shared RAG system
                agent = create_agent(
                    agent_type=request.agent_type,
                    config=agent_config,
                    neo4j_uri=NEO4J_URI,
                    neo4j_user=NEO4J_USER,
                    neo4j_password=NEO4J_PASSWORD,
                    openai_api_key=OPENAI_API_KEY,
                    rag_system=rag_system  # Pass the pre-initialized RAG system
                )

                # Execute the task using the agent
                result = agent.execute(user_message)

                # Clean up the agent
                agent.close()

                # Format agent response
                if "analysis" in result:
                    assistant_message = result["analysis"]
                elif "summary" in result:
                    assistant_message = result["summary"]
                else:
                    assistant_message = str(result)

            # Store the conversation in MariaDB
            try:
                conn = get_mariadb_connection()
                cursor = conn.cursor()

                # Generate a session ID if not present
                session_id = req.headers.get("X-Session-ID", str(uuid.uuid4()))
                user_id = req.headers.get("X-User-ID")
                
                # If no user ID is provided and authentication is required, return error
                if not user_id and os.getenv("REQUIRE_AUTH", "false").lower() == "true":
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required"
                    )
                    
                # Use a generated user ID if none provided
                if not user_id:
                    user_id = f"guest_{str(uuid.uuid4())[:8]}"

                # Store user message
                cursor.execute(
                    """
                    INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, session_id, user_message, "user")
                )

                # Store assistant response
                cursor.execute(
                    """
                    INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, session_id, assistant_message, "assistant")
                )

                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error storing chat history: {str(e)}")
                # Continue anyway, as this is not critical

            return {
                "message": assistant_message,
                "model": request.model,
                "agent_type": request.agent_type,
                "agent_used": True,
                "tree_reasoning_used": request.use_tree_reasoning,
                "context_used": request.include_context,
                "timestamp": datetime.now().isoformat()
            }

        # Standard RAG-based processing
        else:
            # Determine if we should use context
            if request.include_context:
                # Use context_query if provided, otherwise use the user message
                query = request.context_query or user_message

                # Retrieve context using RAG system
                retrieved_nodes = rag_system.retrieve(query, top_k=5)

                # Format context
                if retrieved_nodes:
                    context_parts = []
                    for node in retrieved_nodes:
                        source = f"{node['metadata'].get('doc_name', 'Unknown document')}"
                        if 'section' in node['metadata'] and node['metadata']['section'] != 'Unknown':
                            source += f" - {node['metadata']['section']}"

                        context_parts.append(f"Source: {source}\nContent: {node['text']}")

                    context = "\n\n".join(context_parts)
                else:
                    context = "No relevant context found in the knowledge base."
            else:
                context = None

            # Generate response based on chat history and context
            if context:
                # Add system message with context
                system_message = f"""You are an AI assistant that helps with regulatory information.
    Answer based on the information in the provided context.
    If the answer cannot be found in the context, say so politely.

    Context information:
    {context}
    """
                # Prepend system message to the chat history
                messages = [ChatMessage(role="system", content=system_message)] + messages

            # Format messages for OpenAI
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]

            # Call OpenAI with streaming if requested
            if request.stream:
                # For streaming, return a StreamingResponse
                async def generate():
                    client = OpenAI(api_key=OPENAI_API_KEY)

                    completion = client.chat.completions.create(
                        model=request.model,
                        messages=openai_messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        stream=True
                    )

                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"

                    yield f"data: [DONE]\n\n"

                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                # For non-streaming, return a regular response
                client = OpenAI(api_key=OPENAI_API_KEY)

                completion = client.chat.completions.create(
                    model=request.model,
                    messages=openai_messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )

                # Extract the assistant's response
                assistant_message = completion.choices[0].message.content

                # Store the conversation in MariaDB
                try:
                    conn = get_mariadb_connection()
                    cursor = conn.cursor()

                    # Generate a session ID if not present
                    session_id = req.headers.get("X-Session-ID", str(uuid.uuid4()))
                    user_id = req.headers.get("X-User-ID")
                    
                    # If no user ID is provided and authentication is required, return error
                    if not user_id and os.getenv("REQUIRE_AUTH", "false").lower() == "true":
                        raise HTTPException(
                            status_code=401,
                            detail="Authentication required"
                        )
                        
                    # Use a generated user ID if none provided
                    if not user_id:
                        user_id = f"guest_{str(uuid.uuid4())[:8]}"

                    # Store user message
                    cursor.execute(
                        """
                        INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                        VALUES (?, ?, ?, ?)
                        """,
                        (user_id, session_id, user_message, "user")
                    )

                    # Store assistant response
                    cursor.execute(
                        """
                        INSERT INTO chat_history (user_id, session_id, message_text, message_role)
                        VALUES (?, ?, ?, ?)
                        """,
                        (user_id, session_id, assistant_message, "assistant")
                    )

                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Error storing chat history: {str(e)}")
                    # Continue anyway, as this is not critical

                return {
                    "message": assistant_message,
                    "model": request.model,
                    "agent_used": False,
                    "tree_reasoning_used": False,
                    "context_used": request.include_context,
                    "timestamp": datetime.now().isoformat()
                }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chat response: {str(e)}"
        )


@app.get("/chat/history")
async def get_chat_history(session_id: str, limit: int = 50, skip_old: bool = False, max_age_days: int = None):
    """Get chat history for a session.

    Args:
        session_id: The session ID to get chat history for
        limit: Maximum number of messages to return (default: 50)
        skip_old: Whether to skip old messages entirely
        max_age_days: Maximum age of messages in days to return
    """
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)

        # Base query
        query = """
            SELECT message_text, message_role, timestamp
            FROM chat_history
            WHERE session_id = ?
        """
        params = [session_id]

        # Add age filter if requested
        if skip_old or max_age_days:
            if max_age_days:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL ? DAY)"
                params.append(max_age_days)
            elif skip_old:
                query += " AND timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)"

        # Add ordering and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        history = cursor.fetchall()
        conn.close()

        return {
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chat history: {str(e)}"
        )


@app.delete("/chat/history/{session_id}")
async def delete_chat_history(session_id: str, before_days: int = None):
    """Delete chat history for a session.

    Args:
        session_id: The session ID to delete chat history for
        before_days: If provided, only delete messages older than this many days
    """
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor()

        # Base delete query
        query = "DELETE FROM chat_history WHERE session_id = ?"
        params = [session_id]

        # Add age filter if requested
        if before_days is not None:
            query += " AND timestamp < DATE_SUB(NOW(), INTERVAL ? DAY)"
            params.append(before_days)

        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "session_id": session_id,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} messages from chat history"
        }
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat history: {str(e)}"
        )


@app.delete("/chat/history")
async def delete_all_chat_history(before_days: int = None, user_id: str = None):
    """Delete all chat history or filter by user.

    Args:
        before_days: If provided, only delete messages older than this many days
        user_id: If provided, only delete messages for this user
    """
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor()

        # Base delete query
        query = "DELETE FROM chat_history WHERE 1=1"
        params = []

        # Add filters if requested
        if before_days is not None:
            query += " AND timestamp < DATE_SUB(NOW(), INTERVAL ? DAY)"
            params.append(before_days)

        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        cursor.execute(query, params)
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} messages from chat history"
        }
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat history: {str(e)}"
        )


@app.get("/settings")
async def get_settings():
    """Get global settings."""
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT setting_key, setting_value
            FROM regulaite_settings
            """
        )

        settings = {row['setting_key']: row['setting_value'] for row in cursor.fetchall()}
        conn.close()

        # Format LLM settings if they exist
        llm_settings = {}

        # Extract LLM settings with prefix llm_
        for key, value in list(settings.items()):
            if key.startswith('llm_'):
                # Remove the prefix for cleaner API response
                clean_key = key[4:]  # Remove 'llm_' prefix

                # Handle special types
                if clean_key in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
                    try:
                        llm_settings[clean_key] = float(value)
                    except ValueError:
                        llm_settings[clean_key] = value
                elif clean_key in ['max_tokens']:
                    try:
                        llm_settings[clean_key] = int(value)
                    except ValueError:
                        llm_settings[clean_key] = value
                else:
                    llm_settings[clean_key] = value

                # Remove these from the main settings dict to avoid duplication
                settings.pop(key)

        # Add the formatted LLM settings to the response if they exist
        if llm_settings:
            settings['llm_config'] = llm_settings

        return settings
    except Exception as e:
        logger.error(f"Error retrieving settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving settings: {str(e)}"
        )


@app.post("/settings")
async def update_settings(settings: Dict[str, Any]):
    """Update global settings."""
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor()

        # Handle LLM config if it exists in the payload
        if 'llm_config' in settings:
            llm_config = settings.pop('llm_config')

            # Flatten LLM config with llm_ prefix
            for key, value in llm_config.items():
                settings[f'llm_{key}'] = value

        for key, value in settings.items():
            cursor.execute(
                """
                INSERT INTO regulaite_settings (setting_key, setting_value)
                VALUES (?, ?)
                ON DUPLICATE KEY UPDATE setting_value = ?
                """,
                (key, str(value), str(value))
            )

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": f"Updated {len(settings)} settings"
        }
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating settings: {str(e)}"
        )


# Add a specific endpoint just for LLM configuration
@app.get("/settings/llm")
async def get_llm_settings():
    """Get LLM configuration settings."""
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT setting_key, setting_value
            FROM regulaite_settings
            WHERE setting_key LIKE 'llm_%'
            """
        )

        settings = {row['setting_key']: row['setting_value'] for row in cursor.fetchall()}
        conn.close()

        # Format settings (remove prefix and convert types)
        llm_config = {}
        for key, value in settings.items():
            clean_key = key[4:]  # Remove 'llm_' prefix

            # Handle special types
            if clean_key in ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty']:
                try:
                    llm_config[clean_key] = float(value)
                except ValueError:
                    llm_config[clean_key] = value
            elif clean_key in ['max_tokens']:
                try:
                    llm_config[clean_key] = int(value)
                except ValueError:
                    llm_config[clean_key] = value
            else:
                llm_config[clean_key] = value

        return llm_config
    except Exception as e:
        logger.error(f"Error retrieving LLM settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving LLM settings: {str(e)}"
        )


@app.post("/settings/llm")
async def update_llm_settings(llm_config: Dict[str, Any]):
    """Update LLM configuration settings."""
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor()

        # Add llm_ prefix to all keys
        settings = {f'llm_{key}': value for key, value in llm_config.items()}

        for key, value in settings.items():
            cursor.execute(
                """
                INSERT INTO regulaite_settings (setting_key, setting_value)
                VALUES (?, ?)
                ON DUPLICATE KEY UPDATE setting_value = ?
                """,
                (key, str(value), str(value))
            )

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": f"Updated {len(settings)} LLM settings"
        }
    except Exception as e:
        logger.error(f"Error updating LLM settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating LLM settings: {str(e)}"
        )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, background_tasks: BackgroundTasks = None):
    """Delete a document from the system."""
    try:
        logger.info(f"Processing request to delete document: {doc_id}")

        # Step 1: Delete from RAG system (Qdrant vectors)
        rag_deleted = rag_system.delete_document(doc_id)
        if not rag_deleted:
            logger.warning(f"Failed to delete document {doc_id} from RAG system or document not found in vector store")
            # Continue with Neo4j deletion even if Qdrant deletion failed

        # Step 2: Delete from Neo4j
        try:
            # Using purge_orphans=True to ensure all related nodes are properly cleaned up
            neo4j_deleted = document_parser.delete_document(doc_id, purge_orphans=True)
            if isinstance(neo4j_deleted, dict) and neo4j_deleted.get("status") == "error":
                logger.error(f"Error deleting document from Neo4j: {neo4j_deleted.get('message', 'Unknown error')}")
                # Only raise 404 if document was truly not found
                if neo4j_deleted.get("message") == "Document not found":
                    raise HTTPException(
                        status_code=404,
                        detail=f"Document not found in Neo4j: {doc_id}"
                    )
                # Otherwise attempt direct chunk cleanup by doc_id
                else:
                    logger.warning(f"Attempting direct cleanup of chunks for document {doc_id}")
                    try:
                        # Direct cleanup using Neo4j driver
                        with driver.session() as session:
                            # Start transaction for atomic deletion
                            tx = session.begin_transaction()
                            try:
                                # Get chunk count
                                chunk_count_result = tx.run(
                                    """
                                    MATCH (c:Chunk {doc_id: $doc_id})
                                    RETURN count(c) as chunk_count
                                    """,
                                    doc_id=doc_id
                                )
                                
                                chunk_count = chunk_count_result.single()["chunk_count"] if chunk_count_result.peek() else 0
                                
                                if chunk_count > 0:
                                    logger.info(f"Found {chunk_count} orphaned chunks to delete with direct cleanup")
                                    # Delete relationships from orphaned chunks first
                                    tx.run(
                                        """
                                        MATCH (c:Chunk {doc_id: $doc_id})
                                        OPTIONAL MATCH (c)-[r]-()
                                        DELETE r
                                        """,
                                        doc_id=doc_id
                                    )
                                    
                                    # Delete the orphaned chunks
                                    tx.run(
                                        """
                                        MATCH (c:Chunk {doc_id: $doc_id})
                                        DELETE c
                                        """,
                                        doc_id=doc_id
                                    )
                                
                                tx.commit()
                                logger.info(f"Direct cleanup successful, deleted {chunk_count} orphaned chunks")
                                
                                neo4j_deleted = {
                                    "document_deleted": False,  # Document wasn't found
                                    "chunks_deleted": chunk_count,
                                    "orphaned_chunks_deleted": chunk_count,
                                    "relationships_deleted": 0
                                }
                            except Exception as tx_error:
                                tx.rollback()
                                logger.error(f"Error in direct cleanup transaction: {str(tx_error)}")
                                raise tx_error
                    except Exception as cleanup_error:
                        logger.error(f"Direct cleanup failed: {str(cleanup_error)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to clean up orphaned chunks: {str(cleanup_error)}"
                        )
        except Exception as neo4j_error:
            logger.error(f"Error deleting document {doc_id} from Neo4j: {str(neo4j_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Document deleted from vector store but error occurred when deleting from Neo4j: {str(neo4j_error)}"
            )

        # Format deletion statistics for response if available
        deletion_stats = {}
        if isinstance(neo4j_deleted, dict):
            deletion_stats = {
                "document_deleted": neo4j_deleted.get("document_deleted", True),
                "chunks_deleted": neo4j_deleted.get("chunks_deleted", 0),
                "relationships_deleted": neo4j_deleted.get("relationships_deleted", 0),
                "orphaned_chunks_deleted": neo4j_deleted.get("orphaned_chunks_deleted", 0)
            }

        # Step 3: Run a periodic orphaned chunk cleanup (background task)
        try:
            if driver:
                background_tasks.add_task(clean_orphaned_chunks)
        except Exception as cleanup_error:
            logger.warning(f"Background cleanup task error: {str(cleanup_error)}")

        return {
            "status": "success",
            "message": f"Document {doc_id} deleted successfully",
            "doc_id": doc_id,
            "stats": deletion_stats
        }

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


async def clean_orphaned_chunks():
    """Background task to clean up any orphaned chunks in the database."""
    try:
        if not driver:
            logger.warning("Cannot clean orphaned chunks: Neo4j driver not initialized")
            return
            
        logger.info("Running background task to clean up orphaned chunks")
        with driver.session() as session:
            # First, check for orphaned chunks (chunks without document)
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
            
            if orphan_count > 0:
                logger.info(f"Found {orphan_count} orphaned chunks to clean up")
                
                tx = session.begin_transaction()
                try:
                    # Delete relationships first
                    tx.run(
                        """
                        MATCH (c:Chunk)
                        WHERE NOT EXISTS {
                            MATCH (d:Document {doc_id: c.doc_id})
                        }
                        OPTIONAL MATCH (c)-[r]-()
                        DELETE r
                        """
                    )
                    
                    # Then delete the chunks
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
                    logger.info(f"Successfully cleaned up {orphan_count} orphaned chunks")
                except Exception as tx_error:
                    tx.rollback()
                    logger.error(f"Error in orphaned chunks cleanup: {str(tx_error)}")
            else:
                logger.info("No orphaned chunks found to clean up")
    
    except Exception as e:
        logger.error(f"Error in background cleanup task: {str(e)}")


@app.get("/debug/qdrant/collections")
async def debug_qdrant_collections():
    """Debug endpoint to check Qdrant collections and document count"""
    try:
        # Get all collections
        collections_info = rag_system.qdrant_client.get_collections()
        collections = collections_info.collections

        result = {
            "collections": []
        }

        for collection in collections:
            collection_name = collection.name

            # Get collection info
            try:
                collection_info = rag_system.qdrant_client.get_collection(collection_name)
                count = rag_system.qdrant_client.count(collection_name)

                # Get a sample of points if there are any
                sample_points = []
                if count.count > 0:
                    sample = rag_system.qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=3  # Get at most 3 samples
                    )

                    for point in sample[0]:
                        if hasattr(point, 'payload') and point.payload:
                            payload_info = {
                                "id": point.id,
                                "metadata": point.payload.get("metadata", {})
                            }
                            sample_points.append(payload_info)

                collection_data = {
                    "name": collection_name,
                    "point_count": count.count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "sample_points": sample_points
                }

                result["collections"].append(collection_data)

            except Exception as e:
                result["collections"].append({
                    "name": collection_name,
                    "error": str(e)
                })

        # Also add information about initialized languages in RAG
        result["initialized_languages"] = rag_system.get_initialized_languages()

        return result

    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return {"error": str(e)}


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when shutting down."""
    logger.info("Shutting down API")

    if document_parser:
        try:
            document_parser.close()
            logger.info("Document parser connections closed")
        except:
            pass

    if rag_system:
        try:
            rag_system.close()
            logger.info("RAG system connections closed")
        except:
            pass

    if driver:
        try:
            driver.close()
            logger.info("Main Neo4j driver connection closed")
        except:
            pass


@app.get("/settings/user/{user_id}/parser")
async def get_user_parser_settings(user_id: str):
    """Get parser settings for a specific user."""
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)

        # First check if user exists
        cursor.execute(
            "SELECT user_id FROM users WHERE user_id = ?",
            (user_id,)
        )

        user = cursor.fetchone()
        if not user:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"User with ID {user_id} not found"
            )

        # Get user settings
        cursor.execute(
            "SELECT settings FROM users WHERE user_id = ?",
            (user_id,)
        )

        result = cursor.fetchone()
        conn.close()

        if result and result['settings']:
            try:
                settings = json.loads(result['settings'])
                if 'parser_settings' in settings:
                    return settings['parser_settings']
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in user settings for user {user_id}")

        # Return default settings if no user settings found
        return {
            "selected_parser": os.getenv("DEFAULT_PARSER_TYPE", "unstructured"),
            "parser_settings": {
                "extract_tables": True,
                "extract_metadata": True,
                "extract_images": False,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "chunking_strategy": "fixed"
            }
        }
    except Exception as e:
        logger.error(f"Error retrieving user parser settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving user parser settings: {str(e)}"
        )


@app.post("/settings/user/{user_id}/parser")
async def update_user_parser_settings(user_id: str, parser_settings: Dict[str, Any]):
    """Update parser settings for a specific user."""
    try:
        conn = get_mariadb_connection()
        cursor = conn.cursor(dictionary=True)

        # First check if user exists
        cursor.execute(
            "SELECT user_id, settings FROM users WHERE user_id = ?",
            (user_id,)
        )

        user = cursor.fetchone()
        if not user:
            conn.close()
            raise HTTPException(
                status_code=404,
                detail=f"User with ID {user_id} not found"
            )

        # Get existing settings or initialize empty dict
        user_settings = {}
        if user['settings']:
            try:
                user_settings = json.loads(user['settings'])
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in user settings for user {user_id}, initializing empty settings")

        # Update parser settings
        user_settings['parser_settings'] = parser_settings

        # Save updated settings
        cursor.execute(
            "UPDATE users SET settings = ? WHERE user_id = ?",
            (json.dumps(user_settings), user_id)
        )

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": "User parser settings updated"
        }
    except Exception as e:
        logger.error(f"Error updating user parser settings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating user parser settings: {str(e)}"
        )


@app.post("/maintenance/cleanup-orphaned-chunks")
async def cleanup_orphaned_chunks():
    """Manually clean up orphaned chunks in Neo4j."""
    try:
        if not driver:
            return {
                "status": "error", 
                "message": "Neo4j driver not initialized"
            }
            
        with driver.session() as session:
            # First, get a count of orphaned chunks
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
            
            # Start a transaction for the cleanup
            tx = session.begin_transaction()
            try:
                # Delete relationships first
                tx.run(
                    """
                    MATCH (c:Chunk)
                    WHERE NOT EXISTS {
                        MATCH (d:Document {doc_id: c.doc_id})
                    }
                    OPTIONAL MATCH (c)-[r]-()
                    DELETE r
                    """
                )
                
                # Then delete the chunks
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
                logger.info(f"Manual cleanup deleted {orphan_count} orphaned chunks")
                return {
                    "status": "success",
                    "message": f"Successfully cleaned up {orphan_count} orphaned chunks",
                    "chunks_deleted": orphan_count
                }
            except Exception as tx_error:
                tx.rollback()
                logger.error(f"Error in manual orphaned chunks cleanup: {str(tx_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Transaction error during cleanup: {str(tx_error)}"
                )
    
    except Exception as e:
        logger.error(f"Error in manual cleanup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning up orphaned chunks: {str(e)}"
        )
