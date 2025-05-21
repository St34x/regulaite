"""
FastAPI router for welcome page and dashboard data.
"""
import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
import os
from datetime import datetime, timedelta
from routers.auth_middleware import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/welcome",
    tags=["welcome"],
    responses={404: {"description": "Not found"}},
)

# Models for API
class WelcomeContent(BaseModel):
    """Content for the welcome page."""
    title: str = Field(..., description="Welcome title")
    subtitle: Optional[str] = Field(None, description="Welcome subtitle")
    intro_text: str = Field(..., description="Introduction text")
    features: List[Dict[str, str]] = Field(..., description="Features to highlight")
    getting_started: List[Dict[str, str]] = Field(..., description="Getting started steps")
    recent_updates: Optional[List[Dict[str, Any]]] = Field(None, description="Recent updates")
    cta_text: Optional[str] = Field(None, description="Call to action text")
    cta_link: Optional[str] = Field(None, description="Call to action link")


class DashboardStats(BaseModel):
    """Statistics for the dashboard."""
    document_count: int = Field(..., description="Total number of documents")
    agent_count: int = Field(..., description="Number of available agents")
    recent_document_count: int = Field(..., description="Number of recently added documents")
    recent_chat_count: int = Field(..., description="Number of recent chat sessions")
    storage_usage_mb: float = Field(..., description="Storage usage in MB")
    task_stats: Dict[str, int] = Field(..., description="Task statistics")


class DashboardActivity(BaseModel):
    """Recent activity for the dashboard."""
    recent_documents: List[Dict[str, Any]] = Field(..., description="Recently added documents")
    recent_chats: List[Dict[str, Any]] = Field(..., description="Recent chat sessions")
    recent_tasks: List[Dict[str, Any]] = Field(..., description="Recent tasks")


class DashboardResponse(BaseModel):
    """Dashboard data response."""
    stats: DashboardStats = Field(..., description="Dashboard statistics")
    activity: DashboardActivity = Field(..., description="Recent activity")
    system_status: Dict[str, Any] = Field(..., description="System status")


# Helper function to get database connection
async def get_db_connection():
    """Get MariaDB connection from main application."""
    from main import get_mariadb_connection
    return get_mariadb_connection()


# Helper function to get Neo4j driver
async def get_neo4j_driver():
    """Get the Neo4j driver from main application."""
    from main import driver
    return driver


@router.get("", response_model=WelcomeContent)
async def get_welcome_content():
    """Get welcome page content."""
    try:
        # Get welcome content from database
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT setting_key, setting_value
            FROM regulaite_settings
            WHERE setting_key LIKE 'welcome_%'
            """
        )

        settings = {row['setting_key'][8:]: row['setting_value'] for row in cursor.fetchall()}
        conn.close()

        # Check if we have welcome content stored
        if not settings or 'title' not in settings:
            # Provide default welcome content
            welcome_content = WelcomeContent(
                title="Welcome to RegulAite",
                subtitle="AI-powered Regulatory Compliance Solution",
                intro_text="RegulAite helps you navigate complex regulatory landscapes with AI-driven insights, document analysis, and intelligent agents.",
                features=[
                    {
                        "title": "AI Chat",
                        "description": "Chat with AI that understands your regulatory documents and compliance needs.",
                        "icon": "chat"
                    },
                    {
                        "title": "Document Analysis",
                        "description": "Extract knowledge and insights from your regulatory documents.",
                        "icon": "document"
                    },
                    {
                        "title": "Intelligent Agents",
                        "description": "Specialized AI agents for compliance mapping, gap analysis, and more.",
                        "icon": "agent"
                    },
                    {
                        "title": "Knowledge Graph",
                        "description": "Visualize relationships between regulatory concepts and requirements.",
                        "icon": "graph"
                    }
                ],
                getting_started=[
                    {
                        "step": "1",
                        "title": "Upload Documents",
                        "description": "Start by uploading your regulatory documents to build your knowledge base."
                    },
                    {
                        "step": "2",
                        "title": "Explore Insights",
                        "description": "Use AI chat to ask questions about your regulatory documents."
                    },
                    {
                        "step": "3",
                        "title": "Use Specialized Agents",
                        "description": "Leverage purpose-built agents for specific compliance tasks."
                    }
                ],
                cta_text="Get Started",
                cta_link="/documents"
            )

            # Return default content
            return welcome_content

        # Parse stored welcome content
        try:
            # Parse JSON fields
            features = json.loads(settings.get('features', '[]'))
            getting_started = json.loads(settings.get('getting_started', '[]'))
            recent_updates = json.loads(settings.get('recent_updates', '[]'))

            # Create welcome content from stored settings
            welcome_content = WelcomeContent(
                title=settings.get('title', 'Welcome to RegulAite'),
                subtitle=settings.get('subtitle'),
                intro_text=settings.get('intro_text', 'AI-powered regulatory compliance solution'),
                features=features,
                getting_started=getting_started,
                recent_updates=recent_updates,
                cta_text=settings.get('cta_text'),
                cta_link=settings.get('cta_link')
            )

            return welcome_content

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing welcome content: {str(e)}")
            # Fall back to default content
            welcome_content = WelcomeContent(
                title=settings.get('title', 'Welcome to RegulAite'),
                subtitle=settings.get('subtitle'),
                intro_text=settings.get('intro_text', 'AI-powered regulatory compliance solution'),
                features=[
                    {"title": "AI Chat", "description": "Chat with your documents", "icon": "chat"}
                ],
                getting_started=[
                    {"step": "1", "title": "Upload Documents", "description": "Start by uploading documents"}
                ]
            )

            return welcome_content

    except Exception as e:
        logger.error(f"Error retrieving welcome content: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving welcome content: {str(e)}"
        )


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data():
    """Get dashboard data and statistics."""
    try:
        # Get Neo4j driver for document stats
        neo4j_driver = await get_neo4j_driver()

        # Get MariaDB connection for chat and task stats
        db_conn = await get_db_connection()
        db_cursor = db_conn.cursor(dictionary=True)

        # Get document stats from Neo4j
        with neo4j_driver.session() as session:
            # Total document count
            doc_count_result = session.run("MATCH (d:Document) RETURN count(d) as count")
            document_count = doc_count_result.single()["count"]

            # Total storage usage
            storage_result = session.run("MATCH (d:Document) RETURN sum(d.size) as total_size")
            total_size = storage_result.single()["total_size"] or 0
            storage_usage_mb = round(total_size / (1024 * 1024), 2)

            # Recently added documents (last 7 days)
            recent_date = datetime.now() - timedelta(days=7)
            recent_docs_query = """
                MATCH (d:Document)
                WHERE d.created >= $recent_date
                RETURN count(d) as count
            """
            recent_docs_result = session.run(recent_docs_query, recent_date=recent_date.strftime("%Y-%m-%d"))
            recent_document_count = recent_docs_result.single()["count"]

            # Get recent document details
            recent_docs_details_query = """
                MATCH (d:Document)
                RETURN d.doc_id as doc_id, 
                       COALESCE(d.title, d.name, 'Untitled') as title,
                       COALESCE(d.original_filename, d.name, d.doc_id) as filename, 
                       COALESCE(d.created_at, d.created, datetime()) as created,
                       COALESCE(d.file_type, '') as file_type, 
                       COALESCE(d.size, 0) as size,
                       COALESCE(d.is_indexed, false) as is_indexed
                ORDER BY COALESCE(d.created_at, d.created, datetime()) DESC
                LIMIT 5
            """
            recent_docs_details = [dict(record) for record in session.run(recent_docs_details_query)]

        # Get chat stats from MariaDB
        # Recent chat sessions (last 7 days)
        recent_chats_query = """
            SELECT COUNT(DISTINCT session_id) as count
            FROM chat_history
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """
        db_cursor.execute(recent_chats_query)
        recent_chat_count = db_cursor.fetchone()["count"]

        # Get recent chat session details
        recent_chats_details_query = """
            SELECT session_id, user_id, MAX(timestamp) as last_message,
                   COUNT(*) as message_count,
                   (SELECT message_text FROM chat_history c2
                    WHERE c2.session_id = c1.session_id
                    ORDER BY timestamp ASC LIMIT 1) as first_message
            FROM chat_history c1
            GROUP BY session_id, user_id
            ORDER BY MAX(timestamp) DESC
            LIMIT 5
        """
        db_cursor.execute(recent_chats_details_query)
        recent_chats = db_cursor.fetchall()

        # Format chat sessions
        for chat in recent_chats:
            if chat["first_message"] and len(chat["first_message"]) > 100:
                chat["first_message"] = chat["first_message"][:97] + "..."

        # Get task stats from MariaDB
        task_stats_query = """
            SELECT status, COUNT(*) as count
            FROM tasks
            GROUP BY status
        """
        db_cursor.execute(task_stats_query)
        task_stats_rows = db_cursor.fetchall()

        # Format task stats
        task_stats = {row["status"]: row["count"] for row in task_stats_rows}

        # Get recent tasks
        recent_tasks_query = """
            SELECT task_id, task_type, status, created_at, completed_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT 5
        """
        db_cursor.execute(recent_tasks_query)
        recent_tasks = db_cursor.fetchall()

        # Get agent count
        try:
            from pyndantic_agents.agent_factory import get_agent_types
            agent_types = get_agent_types()
            agent_count = len(agent_types)
        except:
            logger.warning("Could not get agent count, using default")
            agent_count = 3  # Default if can't determine

        # Get system status
        system_status = {
            "status": "healthy",
            "components": {
                "database": "connected",
                "neo4j": "connected",
                "embedding_service": "operational",
                "task_queue": "operational"
            },
            "version": os.environ.get("APP_VERSION", "1.0.0"),
            "uptime": "unknown"  # Would need to track this separately
        }

        # Close database connection
        db_conn.close()

        # Construct response
        dashboard_stats = DashboardStats(
            document_count=document_count,
            agent_count=agent_count,
            recent_document_count=recent_document_count,
            recent_chat_count=recent_chat_count,
            storage_usage_mb=storage_usage_mb,
            task_stats=task_stats
        )

        dashboard_activity = DashboardActivity(
            recent_documents=recent_docs_details,
            recent_chats=recent_chats,
            recent_tasks=recent_tasks
        )

        return DashboardResponse(
            stats=dashboard_stats,
            activity=dashboard_activity,
            system_status=system_status
        )

    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dashboard data: {str(e)}"
        )


@router.get("/user-dashboard", response_model=DashboardResponse)
async def get_user_dashboard(current_user: dict = Depends(get_current_user)):
    """Get personalized dashboard data for authenticated user."""
    try:
        # Get Neo4j driver for document stats
        neo4j_driver = await get_neo4j_driver()
        
        # Get MariaDB connection for additional stats
        conn = await get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get document count - filter by user_id for personal documents
        user_id = current_user["user_id"]
        
        with neo4j_driver.session() as session:
            # Count documents specific to this user
            result = session.run(
                """
                MATCH (d:Document)
                WHERE COALESCE(d.owner_id, 'unknown') = $user_id
                RETURN count(d) as document_count
                """,
                user_id=user_id
            )
            user_document_count = result.single()["document_count"]
            
            # Count recently added documents by this user
            result = session.run(
                """
                MATCH (d:Document)
                WHERE COALESCE(d.owner_id, 'unknown') = $user_id 
                  AND COALESCE(d.created_at, d.created, datetime()) > datetime() - duration('P7D')
                RETURN count(d) as recent_document_count
                """,
                user_id=user_id
            )
            user_recent_document_count = result.single()["recent_document_count"]
            
            # Get storage used by this user's documents
            result = session.run(
                """
                MATCH (d:Document)
                WHERE COALESCE(d.owner_id, 'unknown') = $user_id
                RETURN sum(COALESCE(d.file_size, d.size, 0)) as total_size
                """,
                user_id=user_id
            )
            record = result.single()
            total_size = record["total_size"] if record["total_size"] is not None else 0
            storage_usage_mb = round(total_size / (1024 * 1024), 2)
        
        # Get recent chat sessions for this user
        cursor.execute(
            """
            SELECT COUNT(*) as chat_count
            FROM chat_sessions
            WHERE user_id = %s AND created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
            """,
            (user_id,)
        )
        recent_chat_count = cursor.fetchone()["chat_count"]
        
        # Get task statistics for this user
        cursor.execute(
            """
            SELECT status, COUNT(*) as count
            FROM tasks
            WHERE user_id = %s AND created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY status
            """,
            (user_id,)
        )
        task_stats = {row["status"]: row["count"] for row in cursor.fetchall()}
        
        # Get available agent count (same for all users)
        cursor.execute("SELECT COUNT(*) as count FROM agents WHERE is_active = 1")
        agent_count = cursor.fetchone()["count"]
        
        # Get recent documents
        cursor.execute(
            """
            SELECT d.document_id, d.filename, d.file_type, d.created_at
            FROM documents d
            WHERE d.user_id = %s
            ORDER BY d.created_at DESC
            LIMIT 5
            """,
            (user_id,)
        )
        recent_documents = cursor.fetchall()
        
        # Get recent chat sessions
        cursor.execute(
            """
            SELECT cs.session_id, cs.title, cs.created_at, COUNT(cm.message_id) as message_count
            FROM chat_sessions cs
            LEFT JOIN chat_messages cm ON cs.session_id = cm.session_id
            WHERE cs.user_id = %s
            GROUP BY cs.session_id
            ORDER BY cs.created_at DESC
            LIMIT 5
            """,
            (user_id,)
        )
        recent_chats = cursor.fetchall()
        
        # Get recent tasks
        cursor.execute(
            """
            SELECT task_id, task_type, status, created_at, completed_at
            FROM tasks
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (user_id,)
        )
        recent_tasks = cursor.fetchall()
        
        # Get system status
        cursor.execute(
            """
            SELECT component, status, last_check
            FROM system_status
            """
        )
        system_status = {row["component"]: {"status": row["status"], "last_check": row["last_check"]} 
                         for row in cursor.fetchall()}
        
        conn.close()
        
        # Create response
        response = DashboardResponse(
            stats=DashboardStats(
                document_count=user_document_count,
                agent_count=agent_count,
                recent_document_count=user_recent_document_count,
                recent_chat_count=recent_chat_count,
                storage_usage_mb=storage_usage_mb,
                task_stats=task_stats
            ),
            activity=DashboardActivity(
                recent_documents=recent_documents,
                recent_chats=recent_chats,
                recent_tasks=recent_tasks
            ),
            system_status=system_status
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving user dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving user dashboard data: {str(e)}"
        )
