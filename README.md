# RegulAite - Regulatory AI Assistant

RegulAite is a powerful AI-driven regulatory assistant built as a Kibana plugin. It enables users to process, analyze, and query regulatory documents using natural language, leveraging advanced RAG (Retrieval-Augmented Generation) and LLM technologies.

## Features

- **Document Management**: Upload, process, and manage regulatory documents
- **Smart Search & RAG**: Search document content using natural language queries
- **AI Chat Interface**: Interact with processed documents through a conversational interface
- **Agent System**: Specialized agents for regulatory analysis, vulnerability assessment, and compliance mapping
- **Task Queue Management**: Asynchronous processing of documents and tasks
- **User-specific Settings**: Personalized configuration stored per user
- **Multiple Document Parsers**: Support for different document parsing services

## Architecture

RegulAite uses a multi-component architecture:

- **Frontend**: React-based Kibana plugin UI
- **Backend**: FastAPI Python application
- **Databases**:
  - Neo4j (Graph database for knowledge representation)
  - Qdrant (Vector database for semantic search)
  - MariaDB (Relational database for metadata, settings, and chat history)
- **Document Processing**: Unstructured, Doctly, LlamaParse, and other document parsers
- **LLM Integration**: OpenAI, Anthropic, or other LLM providers

## Installation

RegulAite is installed as a Kibana plugin. Standard installation process:

1. Install the plugin in Kibana
2. Configure environment variables for database connections
3. Start Kibana with the plugin enabled

## Configuration

### Global Settings

Global settings are stored in the MariaDB database in the `regulaite_settings` table:

| Setting Key | Description | Default Value |
|-------------|-------------|---------------|
| `llm_model` | Default LLM model | gpt-4 |
| `llm_temperature` | Temperature for LLM generation | 0.7 |
| `llm_max_tokens` | Maximum tokens in LLM responses | 2048 |
| `llm_top_p` | Top-p sampling parameter | 1 |
| `enable_chat_history` | Whether to save chat history | true |

### LLM Provider Configuration

RegulAite supports multiple LLM providers:

- OpenAI
- Anthropic
- Azure OpenAI
- Local models

LLM configuration includes:
- Provider selection
- Model selection
- API key management
- Generation parameters (temperature, max tokens, etc.)

### Parser Configuration

Document parsers can be configured per user:

- **Unstructured (Local)**: Self-hosted Unstructured instance
- **Unstructured (Cloud)**: Unstructured.io cloud API
- **Doctly**: Doctly document parsing service
- **LlamaParse**: LlamaIndex document parsing service

Parser settings include:
- API URL
- API key
- Content extraction options (tables, metadata, images)
- Chunking settings (size, overlap, strategy)

## User Interface

### Settings Page

The settings page allows configuration of:

1. **General Settings**:
   - System name
   - System version
   - Organization name
   - Debug mode
   - Telemetry settings

2. **LLM Settings**:
   - Provider selection
   - Model selection
   - API keys
   - Generation parameters
   - Streaming options
   - RAG context usage

3. **Database Settings** (read-only):
   - Connection status for Neo4j, Qdrant, and MariaDB
   - Host information
   - Debug tools

4. **Document Parsers**:
   - Parser selection
   - Default parser setting
   - API configuration
   - Extraction options
   - Chunking settings

### Document Upload

The document upload interface supports:

- File selection and drag-and-drop
- Custom document ID
- Metadata in JSON format
- Parser selection
- Advanced parser settings
- Processing options (NLP, enrichment, language detection)
- Asynchronous processing (queue)

User settings for document processing are preserved between sessions.

## API Documentation

This section provides detailed information about all API endpoints, including call formats and response formats.

### System Management

#### `GET /`

Root endpoint to check if the API is running.

**Request:**
```
GET /api/regul_aite/
```

**Response:**
```json
{
  "status": "ok",
  "message": "RegulAite API is running",
  "version": "1.0.0",
  "neo4j_connected": true
}
```

#### `GET /health`

Health check endpoint for system components.

**Request:**
```
GET /api/regul_aite/health
```

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "neo4j": "connected",
    "qdrant": "connected",
    "mariadb": "connected",
    "api": "healthy"
  },
  "timestamp": "2023-04-11T12:34:56.789Z"
}
```

### Document Management

#### `POST /documents/process`
Process a document using the selected parser and store in Neo4j.

**Request:**
```
POST /api/regul_aite/documents/process
Content-Type: multipart/form-data

Form fields:
- file: [Binary file data]
- doc_id: "optional_custom_id" (optional)
- metadata: {"source": "regulatory authority", "jurisdiction": "EU"} (optional)
- use_nlp: true (optional, default: true)
- use_enrichment: true (optional, default: true)
- detect_language: true (optional, default: true)
- language: "en" (optional)
- parser_type: "unstructured" (optional)
- use_queue: false (optional, default: false)
```

**Response:**
```json
{
  "doc_id": "doc_1234abcd",
  "filename": "example.pdf",
  "chunk_count": 42,
  "status": "success",
  "message": "Document processed successfully (Language: English) with 42 chunks, 5 sections, 123 entities, and 37 concepts, including 8 regulatory requirements"
}
```

#### `GET /documents/{doc_id}`
Get document metadata and chunks.

**Request:**
```
GET /api/regul_aite/documents/doc_1234abcd
```

**Response:**
```json
{
  "document": {
    "doc_id": "doc_1234abcd",
    "name": "example.pdf",
    "created": "2023-04-11T12:34:56.789Z",
    "language": "en",
    "chunk_count": 42,
    "indexed": true
  },
  "chunks": [
    {
      "chunk_id": "doc_1234abcd_chunk_0",
      "text": "This is the content of the first chunk...",
      "index": 0,
      "section": "Introduction"
    }
  ],
  "chunk_count": 42
}
```

#### `GET /documents`
List all documents with pagination.

**Request:**
```
GET /api/regul_aite/documents?limit=10&offset=0
```

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "doc_1234abcd",
      "name": "example.pdf",
      "created": "2023-04-11T12:34:56.789Z",
      "language": "en",
      "chunk_count": 42
    }
  ],
  "total": 156,
  "limit": 10,
  "offset": 0
}
```

#### `DELETE /documents/{doc_id}`
Delete a document from the system.

**Request:**
```
DELETE /api/regul_aite/documents/doc_1234abcd
```

**Response:**
```json
{
  "status": "success",
  "message": "Document doc_1234abcd deleted successfully",
  "doc_id": "doc_1234abcd"
}
```

### Search & RAG

#### `POST /search`
Search for documents related to a query.

**Request:**
```
POST /api/regul_aite/search
Content-Type: application/json

{
  "query": "regulatory compliance requirements",
  "limit": 10,
  "filter_criteria": {
    "language": "en",
    "has_regulatory_content": true
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "document": "Compliance Handbook.pdf",
      "section": "Regulatory Requirements",
      "relevance": 0.92
    }
  ],
  "query": "regulatory compliance requirements",
  "timestamp": "2023-04-11T12:34:56.789Z"
}
```

#### `POST /context/retrieve`
Retrieve context from RAG system.

**Request:**
```
POST /api/regul_aite/context/retrieve
Content-Type: application/json

{
  "query": "regulatory compliance for banks",
  "limit": 5,
  "agent_id": "research_agent_123",
  "use_neo4j": true
}
```

**Response:**
```json
{
  "status": "success",
  "query": "regulatory compliance for banks",
  "results": [
    {
      "text": "Banks must comply with Basel III capital requirements...",
      "source": "Banking Regulations.pdf - Capital Requirements",
      "score": 0.95,
      "metadata": {
        "doc_id": "doc_5678efgh",
        "doc_name": "Banking Regulations.pdf",
        "section": "Capital Requirements",
        "index": 24,
        "chunk_id": "doc_5678efgh_chunk_24",
        "language": "en"
      },
      "source_type": "vector_database"
    }
  ],
  "count": 5,
  "timestamp": "2023-04-11T12:34:56.789Z"
}
```

### Chat & LLM Interaction

#### `POST /chat`
Chat with the RAG-enhanced LLM or AI agent.

**Request:**
```
POST /api/regul_aite/chat
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "content": "What are the main regulatory requirements for European banks?"
    }
  ],
  "stream": false,
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 2048,
  "include_context": true,
  "context_query": null,
  "use_agent": true,
  "agent_type": "regulatory",
  "agent_params": {}
}
```

**Response (Non-Streaming):**
```json
{
  "message": "European banks are primarily regulated under the Basel framework and EU banking regulations. The main requirements include...",
  "model": "gpt-4",
  "agent_type": "regulatory",
  "agent_used": true,
  "context_used": true,
  "timestamp": "2023-04-11T12:34:56.789Z"
}
```

**Response (Streaming):**
Server-Sent Events stream with each chunk having the format:
```
data: {"content": "European banks are"}

data: {"content": " primarily regulated under"}

...

data: [DONE]
```

#### `GET /chat/history`
Get chat history for a session.

**Request:**
```
GET /api/regul_aite/chat/history?session_id=abc123def456&limit=50
```

**Response:**
```json
{
  "session_id": "abc123def456",
  "messages": [
    {
      "message_text": "What are the main regulatory requirements for European banks?",
      "message_role": "user",
      "timestamp": "2023-04-11T12:30:00.000Z"
    },
    {
      "message_text": "European banks are primarily regulated under the Basel framework and EU banking regulations...",
      "message_role": "assistant",
      "timestamp": "2023-04-11T12:30:15.000Z"
    }
  ],
  "count": 2
}
```

#### `DELETE /chat/history`
Delete chat history for a session.

**Request:**
```
DELETE /api/regul_aite/chat/history?session_id=abc123def456
```

**Response:**
```json
{
  "status": "success",
  "message": "Chat history deleted for session abc123def456",
  "count": 2
}
```

### Settings Management

#### `GET /settings`
Get global settings.

**Request:**
```
GET /api/regul_aite/settings
```

**Response:**
```json
{
  "system_name": "RegulAite",
  "version": "1.0.0",
  "organization_name": "Example Corp",
  "debug_mode": false,
  "enable_telemetry": true,
  "llm_config": {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0
  }
}
```

#### `POST /settings`
Update global settings.

**Request:**
```
POST /api/regul_aite/settings
Content-Type: application/json

{
  "system_name": "RegulAite Custom",
  "version": "1.0.1",
  "organization_name": "Example Corp",
  "debug_mode": true,
  "enable_telemetry": false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Updated 5 settings"
}
```

#### `GET /settings/llm`
Get LLM configuration settings.

**Request:**
```
GET /api/regul_aite/settings/llm
```

**Response:**
```json
{
  "provider": "openai",
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stream": true,
  "use_rag_context": true
}
```

#### `POST /settings/llm`
Update LLM configuration.

**Request:**
```
POST /api/regul_aite/settings/llm
Content-Type: application/json

{
  "provider": "anthropic",
  "model": "claude-3-opus",
  "temperature": 0.5,
  "max_tokens": 4096,
  "top_p": 0.9,
  "stream": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Updated 6 LLM settings"
}
```

#### `GET /settings/user/{user_id}/parser`
Get parser settings for a specific user.

**Request:**
```
GET /api/regul_aite/settings/user/user123/parser
```

**Response:**
```json
{
  "selected_parser": "unstructured",
  "parser_settings": {
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": false,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "chunking_strategy": "fixed"
  }
}
```

#### `POST /settings/user/{user_id}/parser`
Update parser settings for a user.

**Request:**
```
POST /api/regul_aite/settings/user/user123/parser
Content-Type: application/json

{
  "selected_parser": "llamaparse",
  "parser_settings": {
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": true,
    "chunk_size": 2000,
    "chunk_overlap": 400,
    "chunking_strategy": "semantic"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "User parser settings updated"
}
```

### Agent System

#### `POST /agents/execute`
Execute a task using an agent.

**Request:**
```
POST /api/regul_aite/agents/execute
Content-Type: application/json

{
  "agent_type": "regulatory",
  "task": "Analyze Basel III capital requirements",
  "config": {
    "name": "Regulatory Analysis Agent",
    "description": "Agent for regulatory analysis",
    "include_context": true
  },
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 2048
  },
  "include_context": true,
  "context_query": "Basel III capital requirements"
}
```

**Response:**
```json
{
  "agent_id": "regulatory_20230411123456",
  "task": "Analyze Basel III capital requirements",
  "result": {
    "status": "success",
    "task_type": "regulatory_analysis",
    "analysis": "Basel III is a global regulatory framework that sets higher capital requirements for banks..."
  },
  "timestamp": "2023-04-11T12:34:56.789Z"
}
```

#### `GET /agents/types`
Get available agent types.

**Request:**
```
GET /api/regul_aite/agents/types
```

**Response:**
```json
[
  "regulatory",
  "research",
  "compliance",
  "vulnerability_assessment",
  "threat_modeling"
]
```

#### `GET /agents/{agent_id}/state`
Get the current state of an agent.

**Request:**
```
GET /api/regul_aite/agents/regulatory_20230411123456/state
```

**Response:**
```json
{
  "actions": [
    {
      "action_type": "analyze_document",
      "action_params": {
        "doc_id": "doc_5678efgh"
      },
      "completion_status": true
    }
  ],
  "observations": [
    {
      "content": "Document analyzed successfully",
      "source": "document_analyzer",
      "timestamp": "2023-04-11T12:34:50.789Z" 
    }
  ],
  "thoughts": [
    {
      "content": "I should analyze the capital requirements section first",
      "timestamp": "2023-04-11T12:34:45.789Z"
    }
  ],
  "context": {
    "doc_id": "doc_5678efgh"
  }
}
```

#### `POST /agents/{agent_id}/execute`
Execute a task using an existing agent.

**Request:**
```
POST /api/regul_aite/agents/regulatory_20230411123456/execute
Content-Type: application/json

{
  "task": "Compare Basel III and Basel IV requirements",
  "include_context": true,
  "context_query": "Basel III Basel IV comparison"
}
```

**Response:**
```json
{
  "agent_id": "regulatory_20230411123456",
  "task": "Compare Basel III and Basel IV requirements",
  "result": {
    "status": "success",
    "task_type": "regulatory_analysis",
    "analysis": "When comparing Basel III and Basel IV..."
  },
  "timestamp": "2023-04-11T12:40:56.789Z"
}
```

### Task Queue Management

#### `POST /tasks/documents/process`
Queue document for processing asynchronously.

**Request:**
```
POST /api/regul_aite/tasks/documents/process
Content-Type: multipart/form-data

Form fields:
- file: [Binary file data]
- doc_id: "optional_custom_id" (optional)
- metadata: {"source": "regulatory authority", "jurisdiction": "EU"} (optional)
- use_nlp: true (optional, default: true)
- use_enrichment: true (optional, default: true)
- detect_language: true (optional, default: true)
- language: "en" (optional)
- parser_type: "unstructured" (optional)
```

**Response:**
```json
{
  "task_id": "task_9876zyxw",
  "status": "pending",
  "message": "Document doc_1234abcd queued for processing",
  "created_at": "2023-04-11T12:34:56.789Z"
}
```

#### `POST /tasks/agents/execute`
Queue an agent task for execution.

**Request:**
```
POST /api/regul_aite/tasks/agents/execute
Content-Type: application/json

{
  "agent_type": "regulatory",
  "task": "Analyze Basel III capital requirements",
  "config": {
    "name": "Regulatory Analysis Agent",
    "include_context": true
  },
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.3
  },
  "include_context": true,
  "context_query": "Basel III capital requirements"
}
```

**Response:**
```json
{
  "task_id": "task_5432abcd",
  "status": "pending",
  "message": "Regulatory agent task queued for execution",
  "created_at": "2023-04-11T12:34:56.789Z"
}
```

#### `POST /tasks/documents/bulk-index`
Queue multiple documents for indexing.

**Request:**
```
POST /api/regul_aite/tasks/documents/bulk-index
Content-Type: application/json

{
  "doc_ids": ["doc_1234abcd", "doc_5678efgh"]
}
```

**Response:**
```json
{
  "task_id": "task_2468abcd",
  "status": "pending",
  "message": "Bulk indexing of 2 documents queued",
  "created_at": "2023-04-11T12:34:56.789Z"
}
```

#### `GET /tasks/status/{task_id}`
Get status of a queued task.

**Request:**
```
GET /api/regul_aite/tasks/status/task_9876zyxw
```

**Response (Pending):**
```json
{
  "task_id": "task_9876zyxw",
  "status": "pending",
  "message": "Task is pending execution"
}
```

**Response (Completed):**
```json
{
  "task_id": "task_9876zyxw",
  "status": "completed",
  "result": {
    "doc_id": "doc_1234abcd",
    "chunk_count": 42,
    "indexed": true
  },
  "completed_at": "2023-04-11T12:35:56.789Z"
}
```

#### `DELETE /tasks/cancel/{task_id}`
Cancel a queued task if possible.

**Request:**
```
DELETE /api/regul_aite/tasks/cancel/task_9876zyxw
```

**Response:**
```json
{
  "task_id": "task_9876zyxw",
  "status": "revoked",
  "message": "Task has been canceled"
}
```

#### `GET /tasks/active`
Get list of currently active tasks.

**Request:**
```
GET /api/regul_aite/tasks/active
```

**Response:**
```json
{
  "count": 2,
  "tasks": [
    {
      "task_id": "task_9876zyxw",
      "name": "process_document",
      "worker": "celery@worker1",
      "args": ["..."],
      "kwargs": {"doc_id": "doc_1234abcd"},
      "started_at": "2023-04-11T12:34:56.789Z"
    },
    {
      "task_id": "task_5432abcd",
      "name": "execute_agent_task",
      "worker": "celery@worker2",
      "args": ["..."],
      "kwargs": {"agent_type": "regulatory"},
      "started_at": "2023-04-11T12:35:00.000Z"
    }
  ]
}
```

#### `GET /tasks/documents/parser-settings`
Get parser configuration settings.

**Request:**
```
GET /api/regul_aite/tasks/documents/parser-settings
```

**Response:**
```json
{
  "unstructured_local": {
    "api_url": "http://unstructured:8000/general/v0/general",
    "api_key": "",
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": false,
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "unstructured_cloud": {
    "api_url": "https://api.unstructured.io/general/v0/general",
    "api_key": "***",
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": true,
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "doctly": {
    "api_url": "https://api.doctly.dev/v1/parse",
    "api_key": "***",
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": false,
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "llamaparse": {
    "api_url": "https://api.llamaindex.ai/v1/parsing",
    "api_key": "***",
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": false,
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "default_parser": "unstructured"
}
```

#### `POST /tasks/documents/parser-settings`
Update parser settings.

**Request:**
```
POST /api/regul_aite/tasks/documents/parser-settings
Content-Type: application/json

{
  "parser_id": "llamaparse",
  "settings": {
    "api_url": "https://api.llamaindex.ai/v1/parsing",
    "api_key": "new_api_key_here",
    "extract_tables": true,
    "extract_metadata": true,
    "extract_images": true,
    "chunk_size": 1500,
    "chunk_overlap": 300
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Settings for llamaparse parser updated",
  "parser_id": "llamaparse"
}
```

### Debug Endpoints

#### `GET /debug/qdrant/collections`
Debug endpoint to check Qdrant collections and document count.

**Request:**
```
GET /api/regul_aite/debug/qdrant/collections
```

**Response:**
```json
{
  "collections": [
    {
      "name": "regulaite_docs_en",
      "point_count": 1250,
      "vector_size": 384,
      "sample_points": [
        {
          "id": "doc_1234abcd_chunk_0",
          "metadata": {
            "doc_id": "doc_1234abcd",
            "doc_name": "example.pdf",
            "section": "Introduction"
          }
        }
      ]
    }
  ],
  "initialized_languages": ["en", "fr", "de"]
}
```

#### `POST /debug/index_document`
Debug endpoint to force reindex a document.

**Request:**
```
POST /api/regul_aite/debug/index_document
Content-Type: application/json

{
  "doc_id": "doc_1234abcd"
}
```

**Response:**
```json
{
  "doc_id": "doc_1234abcd",
  "indexed": true,
  "timestamp": "2023-04-11T12:34:56.789Z"
}
```

## Developer Information

### Folder Structure

```
plugins/regul_aite/
├── backend/
│   ├── config/
│   │   ├── mariadb/
│   │   │   ├── custom.cnf
│   │   │   └── init.sql
│   │   ├── __init__.py
│   │   └── llm_config.py
│   ├── routers/
│   │   └── task_router.py
│   └── main.py
├── public/
│   ├── components/
│   │   ├── document_upload.tsx
│   │   └── settings_page.tsx
│   └── services/
│       └── api_service.ts
└── README.md
```

### Database Schema

MariaDB tables:
- `regulaite_settings`: Global settings
- `chat_history`: User chat interactions
- `task_chat_messages`: Task-specific chat messages
- `tasks`: Task tracking information
- `users`: User information and settings

## Best Practices

1. **Document Processing**:
   - Use descriptive document IDs
   - Add comprehensive metadata
   - Consider using queue for large documents
   - Select the most appropriate parser for your document type

2. **LLM Configuration**:
   - Use lower temperatures (0.1-0.3) for factual responses
   - Use higher temperatures (0.7-0.9) for creative content
   - Adjust max tokens based on expected response length

3. **Security**:
   - Store API keys securely
   - Use distinct API keys for different environments
   - Review access logs periodically

## Troubleshooting

Common issues and solutions:

1. **Document Processing Failures**:
   - Check file format compatibility
   - Verify parser API is accessible
   - Check logs for detailed error messages

2. **LLM Connection Issues**:
   - Verify API key validity
   - Check network connectivity to provider
   - Confirm API URL is correct

3. **Performance Optimization**:
   - Adjust chunk size for better retrieval
   - Tune vector search parameters
   - Consider document preprocessing strategies

## Support and Resources

- Issue Reporting: [GitHub Issues](https://github.com/example/regulaite/issues)
- Documentation: [Full Documentation](https://docs.regulaite.com)
- Community: [Slack Channel](https://regulaite.slack.com)