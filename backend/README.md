# RegulAite Backend

RegulAite is an AI-powered regulatory compliance solution that helps organizations navigate complex regulatory landscapes, analyze documents, and ensure compliance.

## Features

### Chat Interface
- Chat with AI that understands your regulatory documents
- Save chat history for future reference
- Use specialized agents directly from chat
- Retrieve context from your documents for accurate answers

### Document Management
- Upload and process regulatory documents
- Configure document processing settings
- Search documents using hybrid semantic + keyword search
- View document statistics and insights

### Intelligent Agents
- RAG Agent - Answer questions using your document knowledge
- Compliance Mapping Agent - Map requirements between compliance frameworks
- Vulnerability Assessment Agent - Identify security risks and vulnerabilities
- Threat Modeling Agent - Create threat models for systems and processes
- Dynamic Tree Reasoning Agent - Use decision trees for complex reasoning tasks

### System Configuration
- Configure LLM settings (model, temperature, etc.)
- Configure RAG settings (search parameters, embedding model)
- Customize UI settings
- Set system-wide parameters

## Architecture

RegulAite uses a modular architecture:
- FastAPI for the backend API
- Neo4j for knowledge graph storage
- MariaDB for structured data
- Qdrant for vector search
- OpenAI for language models and embeddings

## API Routes

### Chat
- `POST /chat` - Chat with the AI
- `GET /chat/history` - Get chat history
- `GET /chat/sessions` - List chat sessions
- `DELETE /chat/history/{session_id}` - Delete chat history

### Documents
- `POST /documents/process` - Upload and process a document
- `GET /documents` - List all documents
- `GET /documents/{doc_id}` - Get document details
- `DELETE /documents/{doc_id}` - Delete a document
- `POST /documents/search` - Search documents
- `GET /documents/config` - Get document processing configuration

### Agents
- `GET /agents/types` - List available agent types
- `GET /agents/metadata` - Get agent metadata
- `GET /agents/documentation/{agent_id}` - Get detailed agent documentation
- `GET /agents/trees` - List available decision trees
- `GET /agents/trees/{tree_id}` - Get a specific decision tree
- `POST /agents/process` - Process a request with an agent

### Configuration
- `GET /config` - Get all configuration settings
- `POST /config/llm` - Update LLM configuration
- `POST /config/rag` - Update RAG configuration
- `POST /config/ui` - Update UI configuration
- `POST /config/system` - Update system configuration
- `GET /config/reset` - Reset configuration to defaults

### Welcome and Dashboard
- `GET /welcome` - Get welcome page content
- `GET /welcome/dashboard` - Get dashboard data and statistics

## Getting Started

1. Make sure environment variables are set in `.env` or environment:
   ```
   NEO4J_URI=bolt://neo4j:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   OPENAI_API_KEY=your-api-key
   QDRANT_URL=http://qdrant:6333
   MARIADB_HOST=mariadb
   MARIADB_DATABASE=regulaite
   MARIADB_USER=regulaite_user
   MARIADB_PASSWORD=SecureP@ssw0rd!
   ```

2. Run with uvicorn:
   ```
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Or use Docker:
   ```
   docker-compose up -d
   ```

## Documentation

API documentation is available at `/docs` when the server is running. 