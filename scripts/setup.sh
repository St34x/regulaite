#!/bin/bash
# Updated setup.sh for RegulAite Plugin with Docker image building
# This script configures the plugin and builds required Docker images

set -e  # Exit on error

# ANSI color codes for prettier output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PLUGIN_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
DOCKER_COMPOSE_FILE="$PLUGIN_DIR/docker-compose.yml"
ENV_FILE="$PLUGIN_DIR/.env"
BACKEND_IMAGE="ai_backend:latest"
FRONTEND_IMAGE="regulaite_frontend:latest"
FRONTEND_DIR="$PLUGIN_DIR/front-end"
BUILD_TARGET="development" # Default to development mode

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                        RegulAIte Plugin Setup                        ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
for cmd in docker grep sed; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed or not in PATH${NC}"
        exit 1
    fi
done

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies are installed and Docker is running${NC}"

# Step 2: Create a dedicated network (for containers to talk to each other)
NETWORK_NAME="regulaite_network"

# Create a new network if it doesn't exist
echo -e "${YELLOW}Setting up Docker network: ${NETWORK_NAME}${NC}"
if ! docker network ls | grep -q "$NETWORK_NAME"; then
    docker network create "$NETWORK_NAME"
    echo -e "${GREEN}✅ Network '$NETWORK_NAME' created${NC}"
else
    echo -e "${GREEN}✅ Network '$NETWORK_NAME' already exists${NC}"
fi

sleep 1  # Wait for network to be created

# Delete exitsting /data directory
# echo -e "${YELLOW}Cleaning up existing data directories...${NC}"
# sudo rm -fr "$PLUGIN_DIR/backend/data"
# echo -e "${GREEN}✅ Existing data directories removed${NC}"

# sleep 0.5  # Wait for directory to be removed

# echo -e "${YELLOW}Creating data directories...${NC}"
# mkdir -p "$PLUGIN_DIR/backend/data/neo4j/data"
# sleep 0.5  # Wait for directory to be created
# echo -e "${GREEN}✅ data/neo4j/data directory created${NC}"
# mkdir -p "$PLUGIN_DIR/backend/data/neo4j/logs"
# sleep 0.5  # Wait for directory to be created
# echo -e "${GREEN}✅ data/neo4j/logs directory created${NC}"
# mkdir -p "$PLUGIN_DIR/backend/data/neo4j/import"
# sleep 0.5  # Wait for directory to be created
# echo -e "${GREEN}✅ data/neo4j/import directory created${NC}"
# mkdir -p "$PLUGIN_DIR/backend/data/neo4j/plugins"
# sleep 0.5  # Wait for directory to be created
# echo -e "${GREEN}✅ data/neo4j/plugins directory created${NC}"
# mkdir -p "$PLUGIN_DIR/backend/data/regulaite-files"
# echo -e "${GREEN}✅ data/regulaite-files directory created${NC}"
# sleep 0.5  # Wait for directory to be created
# mkdir -p "$PLUGIN_DIR/backend/data/mariadb"
# echo -e "${GREEN}✅ $PLUGIN_DIR/backend/data/mariadb directory created${NC}"
# Create directory for Weaviate data
# mkdir -p "$PLUGIN_DIR/backend/data/qdrant"
# echo -e "${GREEN}✅ data/qdrant directory created${NC}"
# sleep 0.5  # Wait for directory to be created
# echo -e "${GREEN}✅ All directories created${NC}"


# Change data directory permissions (mysql user has UID 999 in the container)
# sudo chown -R 999:999 "$PLUGIN_DIR/backend/data/mariadb"

# Step 4: Update .env file - Using localhost configuration
echo -e "${YELLOW}Configuring environment variables...${NC}"

# Set Kibana host to localhost for simplicity
KIBANA_HOST="localhost"

# Step 4: Stop any running containers and start fresh
echo -e "${YELLOW}Stopping any running containers...${NC}"
docker-compose -f "$DOCKER_COMPOSE_FILE" down 2>/dev/null || true
echo -e "${GREEN}✅ Containers stopped${NC}"

# Create or update .env file
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Found existing .env file. Updating...${NC}"
else
    echo -e "${YELLOW}Creating new .env file...${NC}"
fi

# Prompt for OpenAI API key if not in .env
OPENAI_API_KEY=""
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    OPENAI_API_KEY="${OPENAI_API_KEY:-}"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -n "Enter your OpenAI API key (required for AI services): "
    read -r OPENAI_API_KEY

    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}No API key provided. Using placeholder - some features may not work correctly.${NC}"
        OPENAI_API_KEY="your-openai-api-key"
    fi
fi

# Prompt for MariaDB root password
MARIADB_ROOT_PASSWORD="${MARIADB_ROOT_PASSWORD:-}"
if [ -z "$MARIADB_ROOT_PASSWORD" ]; then
    echo -n "Enter a root password for MariaDB (leave blank for default 'StrongR00tP@ssW0rd!'): "
    read -r MARIADB_ROOT_PASSWORD

    if [ -z "$MARIADB_ROOT_PASSWORD" ]; then
        echo -e "${YELLOW}Using default root password for MariaDB${NC}"
        MARIADB_ROOT_PASSWORD="StrongR00tP@ssW0rd!"
    fi
fi

# Prompt for MariaDB user credentials
MARIADB_DATABASE="${MARIADB_DATABASE:-regulaite}"
MARIADB_USER="${MARIADB_USER:-regulaite_user}"
MARIADB_PASSWORD="${MARIADB_PASSWORD:-}"

# Prompt MariaDB user name if not in .env
if [ -z "$MARIADB_USER" ]; then
    echo -n "Enter a username for MariaDB (leave blank for default 'regulaite_user'): "
    read -r MARIADB_USER

    if [ -z "$MARIADB_USER" ]; then
        echo -e "${YELLOW}Using default username 'regulaite_user'${NC}"
        MARIADB_USER="regulaite_user"
    fi
fi

# Prompt for MariaDB password if not in .env
if [ -z "$MARIADB_PASSWORD" ]; then
    echo -n "Enter a password for MariaDB user '$MARIADB_USER' (leave blank for default 'SecureP@ssw0rd!'): "
    read -r MARIADB_PASSWORD

    if [ -z "$MARIADB_PASSWORD" ]; then
        echo -e "${YELLOW}Using default password for MariaDB user${NC}"
        MARIADB_PASSWORD="SecureP@ssw0rd!"
    fi
fi

# Prompt for MariaDB database name if not in .env
if [ -z "$MARIADB_DATABASE" ]; then
    echo -n "Enter a name for MariaDB database (leave blank for default 'regulaite'): "
    read -r MARIADB_DATABASE

    if [ -z "$MARIADB_DATABASE" ]; then
        echo -e "${YELLOW}Using default database name 'regulaite'${NC}"
        MARIADB_DATABASE="regulaite"
    fi
fi

# Prompt for Neo4j password if not in .env
NEO4J_PASSWORD=""
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"
fi

if [ -z "$NEO4J_PASSWORD" ]; then
    echo -n "Enter a password for Neo4j (required for database): "
    read -r NEO4J_PASSWORD

    if [ -z "$NEO4J_PASSWORD" ]; then
        echo -e "${RED}No password provided. Using default password 'password'${NC}"
        NEO4J_PASSWORD="password"
    fi
fi

# Prompt for Neo4j username if not in .env
NEO4J_USER="neo4j"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    NEO4J_USER="${NEO4J_USER:-}"
fi

if [ -z "$NEO4J_USER" ]; then
    echo -n "Enter a username for Neo4j (required for database, leave blank for default 'neo4j'): "
    read -r NEO4J_USER

    if [ -z "$NEO4J_USER" ]; then
        echo -e "${RED}No username provided. Using default username 'neo4j'${NC}"
        NEO4J_USER="neo4j"
    fi
fi

# Document Parser API Keys
echo -e "${YELLOW}Document Parser Configuration${NC}"

# Unstructured Cloud API Key
UNSTRUCTURED_CLOUD_API_KEY=""
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    UNSTRUCTURED_CLOUD_API_KEY="${UNSTRUCTURED_CLOUD_API_KEY:-}"
fi

echo -n "Enter Unstructured Cloud API Key (optional): "
read -r UNSTRUCTURED_CLOUD_API_KEY_INPUT
if [ -n "$UNSTRUCTURED_CLOUD_API_KEY_INPUT" ]; then
    UNSTRUCTURED_CLOUD_API_KEY="$UNSTRUCTURED_CLOUD_API_KEY_INPUT"
fi

# LlamaParse API Key
LLAMAPARSE_API_KEY=""
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    LLAMAPARSE_API_KEY="${LLAMAPARSE_API_KEY:-}"
fi

echo -n "Enter LlamaParse API Key (optional): "
read -r LLAMAPARSE_API_KEY_INPUT
if [ -n "$LLAMAPARSE_API_KEY_INPUT" ]; then
    LLAMAPARSE_API_KEY="$LLAMAPARSE_API_KEY_INPUT"
fi

# Doctly API Key
DOCTLY_API_KEY=""
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    DOCTLY_API_KEY="${DOCTLY_API_KEY:-}"
fi

echo -n "Enter Doctly API Key (optional): "
read -r DOCTLY_API_KEY_INPUT
if [ -n "$DOCTLY_API_KEY_INPUT" ]; then
    DOCTLY_API_KEY="$DOCTLY_API_KEY_INPUT"
fi

# Default Parser Selection
DEFAULT_PARSER_TYPE="unstructured"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    DEFAULT_PARSER_TYPE="${DEFAULT_PARSER_TYPE:-unstructured}"
fi

echo -e "${YELLOW}Select default document parser:${NC}"
echo "1) Unstructured (Local) - default"
echo "2) Unstructured Cloud"
echo "3) LlamaParse"
echo "4) Doctly"
read -rp "Enter your choice [1]: " parser_choice
parser_choice=${parser_choice:-1}

case $parser_choice in
    2)
        DEFAULT_PARSER_TYPE="unstructured_cloud"
        echo -e "${YELLOW}Setting default parser to: Unstructured Cloud${NC}"
        if [ -z "$UNSTRUCTURED_CLOUD_API_KEY" ]; then
            echo -e "${RED}Warning: No API key provided for Unstructured Cloud. This parser may not work.${NC}"
        fi
        ;;
    3)
        DEFAULT_PARSER_TYPE="llamaparse"
        echo -e "${YELLOW}Setting default parser to: LlamaParse${NC}"
        if [ -z "$LLAMAPARSE_API_KEY" ]; then
            echo -e "${RED}Warning: No API key provided for LlamaParse. This parser may not work.${NC}"
        fi
        ;;
    4)
        DEFAULT_PARSER_TYPE="doctly"
        echo -e "${YELLOW}Setting default parser to: Doctly${NC}"
        if [ -z "$DOCTLY_API_KEY" ]; then
            echo -e "${RED}Warning: No API key provided for Doctly. This parser may not work.${NC}"
        fi
        ;;
    *)
        echo -e "${YELLOW}Setting default parser to: Unstructured (Local)${NC}"
        ;;
esac

# Ask for environment type
echo -e "${YELLOW}Select environment type:${NC}"
echo "1) Development (runs as root, hot-reloading enabled)"
echo "2) Production (runs as non-root user, security hardened)"
read -rp "Enter your choice [1]: " env_choice
env_choice=${env_choice:-1}

if [ "$env_choice" = "2" ]; then
    BUILD_TARGET="production"
    echo -e "${YELLOW}Setting up PRODUCTION environment${NC}"
else
    echo -e "${YELLOW}Setting up DEVELOPMENT environment${NC}"
fi

# Create or update .env file with localhost configuration
cat > "$ENV_FILE" << EOL

#Build target
BUILD_TARGET=$BUILD_TARGET

# API Keys
OPENAI_API_KEY=$OPENAI_API_KEY

# Kibana Hostname
KIBANA_HOST=$KIBANA_HOST
KIBANA_PORT=5601

# MariaDB Configuration
MARIADB_ROOT_PASSWORD=$MARIADB_ROOT_PASSWORD
MARIADB_DATABASE=$MARIADB_DATABASE
MARIADB_USER=$MARIADB_USER
MARIADB_PASSWORD=$MARIADB_PASSWORD

# Qdrant Configuration
QDRANT_URL=http://qdrant:6333
QDRANT_GRPC_URL=http://qdrant:6334

# Neo4j Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_EXTERNAL_URI=bolt://neo4j:7687
NEO4J_USER=$NEO4J_USER
NEO4J_PASSWORD=$NEO4J_PASSWORD

# Unstructured Configuration
UNSTRUCTURED_API_URL=http://unstructured:8000/general/v0/general
UNSTRUCTURED_CLOUD_API_URL=https://api.unstructured.io/general/v0/general
UNSTRUCTURED_CLOUD_API_KEY=$UNSTRUCTURED_CLOUD_API_KEY
UNSTRUCTURED_EXTRA_ARGS=""

# LlamaParse Configuration
LLAMAPARSE_API_URL=https://api.llamaindex.ai/v1/parsing
LLAMAPARSE_API_KEY=$LLAMAPARSE_API_KEY

# Doctly Configuration
DOCTLY_API_URL=https://api.doctly.dev/v1/parse
DOCTLY_API_KEY=$DOCTLY_API_KEY

# Default Parser
DEFAULT_PARSER_TYPE=$DEFAULT_PARSER_TYPE

# Parser Settings
EXTRACT_TABLES=true
EXTRACT_METADATA=true
EXTRACT_IMAGES=false
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# AI_BACKEND Configuration
AI_BACKEND_API_URL=http://localhost:8090

# Processing Configuration
MAX_WORKERS=5

# Cache Directory
CACHE_DIR=/app/cache
EOL

echo -e "${GREEN}✅ Environment configuration completed${NC}"

# Step 6: Build the Pyndantic Docker image (Neo4j moved to external container only)
echo -e "${YELLOW}Building AI Backend Docker image...${NC}"

# Verfy if dockerfile exists for AI_BACKEND
AI_BACKEND_DIR="$PLUGIN_DIR/backend"
PY_DOCKERFILE="$AI_BACKEND_DIR/Dockerfile"

if [ -f "$PY_DOCKERFILE" ]; then
    echo -e "${YELLOW}✅ Found existing AI_BACKEND Dockerfile. ${NC}"
else
    echo -e "${RED}Dockerfile for AI_BACKEND doesn't exist!${NC}"
fi

# Step 6: Update docker-compose.yml to use our build target
echo -e "${YELLOW}Updating docker-compose.yml with build target...${NC}"
if grep -q "target:" "$DOCKER_COMPOSE_FILE"; then
    # Update existing target
    sed -i "s/target: .*/target: $BUILD_TARGET/" "$DOCKER_COMPOSE_FILE"
else
    # Add target if it doesn't exist
    sed -i "/context: \.\/backend/a \ \ \ \ \ \ target: $BUILD_TARGET" "$DOCKER_COMPOSE_FILE"
fi
echo -e "${GREEN}✅ Updated docker-compose.yml with build target: $BUILD_TARGET${NC}"


# Build the Docker image with specified target
echo -e "${YELLOW}Building Docker image: $BACKEND_IMAGE using target: $BUILD_TARGET${NC}"
if docker build --no-cache --target $BUILD_TARGET -t "$BACKEND_IMAGE" "$AI_BACKEND_DIR"; then
    echo -e "${GREEN}✅ Successfully built $BACKEND_IMAGE with target: $BUILD_TARGET${NC}"
else
    echo -e "${RED}Failed to build $BACKEND_IMAGE${NC}"
    echo -e "${YELLOW}Please check the Dockerfile and build log for errors.${NC}"
    exit 1
fi

# Build frontend image
echo -e "${YELLOW}Building frontend Docker image...${NC}"
if docker build --no-cache -t "$FRONTEND_IMAGE" "$FRONTEND_DIR"; then
    echo -e "${GREEN}✅ Successfully built $FRONTEND_IMAGE${NC}"
else
    echo -e "${RED}Failed to build $FRONTEND_IMAGE${NC}"
fi

# Start containers
echo -e "${YELLOW}Starting containers...${NC}"
docker-compose -f "$DOCKER_COMPOSE_FILE" up -d --remove-orphans

# Step 7: Wait for services to start
echo -e "${YELLOW}Waiting for services to initialize...${NC}"

# Wait for Neo4j to be ready
MAX_WAIT=60  # seconds
echo -e "${YELLOW}Waiting for Neo4j to start (this can take up to $MAX_WAIT seconds)...${NC}"
for i in $(seq 1 $MAX_WAIT); do
    if docker ps | grep -q "regulaite-neo4j" && docker logs regulaite-neo4j 2>&1 | grep -q "Remote interface available at"; then
        echo -e "${GREEN}✅ Neo4j is ready${NC}"
        break
    fi

    # Show progress
    if [ $((i % 5)) -eq 0 ]; then
        echo -ne "${YELLOW}Still waiting for Neo4j... ${i}s/$MAX_WAIT\r${NC}"
    fi

    # If we've reached max wait, show a message but continue
    if [ "$i" -eq $MAX_WAIT ]; then
        echo -e "${YELLOW}⚠️ Max wait time reached for Neo4j. Continuing anyway...${NC}"
    fi

    sleep 1
done

# Wait for MariaDB to be ready
echo -e "${YELLOW}Waiting for MariaDB to start...${NC}"
for i in $(seq 1 $MAX_WAIT); do
    if docker ps | grep -q "regulaite-mariadb" && docker exec regulaite-mariadb mariadb-admin ping -h localhost -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" --silent; then
        echo -e "${GREEN}✅ MariaDB is ready${NC}"
        break
    fi

    # Show progress
    if [ $((i % 5)) -eq 0 ]; then
        echo -ne "${YELLOW}Still waiting for MariaDB... ${i}s/$MAX_WAIT\r${NC}"
    fi

    # If we've reached max wait, show a message but continue
    if [ "$i" -eq $MAX_WAIT ]; then
        echo -e "${YELLOW}⚠️ Max wait time reached for MariaDB. Continuing anyway...${NC}"
    fi

    sleep 1
done

# Step 8: Initialize Neo4j Schema
# Show progress
for i in $(seq 1 $MAX_WAIT); do
    echo -ne "${YELLOW}Waiting for Neo4j to be fully operational... ${i}s\r${NC}"
    docker exec regulaite-neo4j bash -c "echo 'RETURN 1;' | cypher-shell -u \"$NEO4J_USER\" -p \"$NEO4J_PASSWORD\"" > /dev/null 2>&1 && break
    sleep 1
done

# Wait a bit longer to ensure Neo4j has properly initialized
echo -e "${YELLOW}Waiting for Neo4j to be fully operational...${NC}"

# Run the Cypher commands directly in the container
echo -e "${YELLOW}Running Neo4j schema initialization commands...${NC}"

docker exec regulaite-neo4j bash -c "
# Test authentication first
MAX_RETRIES=5
RETRY_COUNT=0
SUCCESS=false

while [ \$RETRY_COUNT -lt \$MAX_RETRIES ] && [ \$SUCCESS = false ]; do
  if echo 'RETURN 1;' | cypher-shell -u \"$NEO4J_USER\" -p \"$NEO4J_PASSWORD\" > /dev/null 2>&1; then
    SUCCESS=true
    echo 'Authentication successful, proceeding with schema initialization.'
  else
    RETRY_COUNT=\$((RETRY_COUNT+1))
    echo \"Authentication failed, attempt \$RETRY_COUNT of \$MAX_RETRIES. Waiting 20 seconds...\"
    sleep 20
  fi
done

if [ \$SUCCESS = false ]; then
  echo 'Failed to authenticate after multiple attempts. Aborting schema initialization.'
  exit 1
fi

# If authentication succeeded, run schema initialization
echo '
// Entity constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name);

// Concept constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;

// Document nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.content);
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.name);

// Section nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.title);

// File nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE;

// Create knowledge graph schema
MERGE (sc:SchemaInit {id: \"regulaite-schema\"})
SET sc.created = datetime(), sc.version = \"1.0\";
' | cypher-shell -u $NEO4J_USER -p \"$NEO4J_PASSWORD\"
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Neo4j schema initialized successfully${NC}"
else
    echo -e "${RED}Error initializing Neo4j schema${NC}"
    echo -e "${YELLOW}You may need to initialize the schema manually through the Neo4j Browser.${NC}"
fi

# Initialize MariaDB schema
echo -e "${YELLOW}Checking MariaDB schema status...${NC}"

# Check if the regulaite_settings table exists (used as a marker for schema initialization)
if ! docker exec regulaite-mariadb mariadb -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "SHOW TABLES LIKE 'regulaite_settings';" | grep -q "regulaite_settings"; then
    echo -e "${YELLOW}Schema not initialized. Applying database schema...${NC}"

    # First check if the init.sql script worked by itself (it should run on container start)
    sleep 3  # Give a moment for any auto-initialization to complete

    # Check again if tables exist after waiting
    if ! docker exec regulaite-mariadb mariadb -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "SHOW TABLES LIKE 'regulaite_settings';" | grep -q "regulaite_settings"; then
        echo -e "${YELLOW}Auto-initialization not detected. Running init.sql manually...${NC}"

        # Copy the init.sql file into the container
        docker cp "$PLUGIN_DIR/backend/config/mariadb/init.sql" regulaite-mariadb:/tmp/init.sql

        # Execute the SQL file using the root user instead of the application user
        if docker exec regulaite-mariadb sh -c "mariadb -u\"root\" -p\"$MARIADB_ROOT_PASSWORD\" \"$MARIADB_DATABASE\" < /tmp/init.sql"; then
            echo -e "${GREEN}✅ MariaDB schema initialized successfully from init.sql${NC}"
        else
            echo -e "${RED}Error initializing MariaDB schema from init.sql${NC}"

            # Fallback to minimal schema if the init.sql fails
            echo -e "${YELLOW}Attempting fallback schema initialization...${NC}"
            docker exec regulaite-mariadb mariadb -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "
                -- Table for global settings
                CREATE TABLE IF NOT EXISTS regulaite_settings (
                    setting_key VARCHAR(255) PRIMARY KEY,
                    setting_value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    description TEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

                -- Table for chat messages history
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    message_text TEXT NOT NULL,
                    message_role ENUM('user', 'assistant', 'system') NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    INDEX idx_user_id (user_id),
                    INDEX idx_session_id (session_id),
                    INDEX idx_timestamp (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

                -- Table for task chat messages
                CREATE TABLE IF NOT EXISTS task_chat_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    message_id VARCHAR(255) NOT NULL UNIQUE,
                    task_id VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    role ENUM('user', 'assistant', 'system') NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_message_id (message_id),
                    INDEX idx_task_id (task_id),
                    INDEX idx_timestamp (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

                -- Create task tracking table if needed
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id VARCHAR(255) PRIMARY KEY,
                    status ENUM('queued', 'processing', 'completed', 'failed', 'cancelled') NOT NULL,
                    task_type VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP NULL,
                    result JSON,
                    error TEXT,
                    message TEXT,
                    parameters JSON,
                    INDEX idx_status (status),
                    INDEX idx_task_type (task_type),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

                -- Create user table if needed for future authentication
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(255) PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    company VARCHAR(255),
                    username VARCHAR(255) UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL,
                    settings JSON,
                    INDEX idx_username (username),
                    INDEX idx_email (email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

                -- Insert default settings
                INSERT IGNORE INTO regulaite_settings (setting_key, setting_value, description) VALUES
                ('llm_model', 'gpt-4', 'Default LLM model'),
                ('llm_temperature', '0.7', 'Default temperature for LLM'),
                ('llm_max_tokens', '2048', 'Default max tokens for LLM'),
                ('llm_top_p', '1', 'Default top_p value for LLM'),
                ('enable_chat_history', 'true', 'Whether to save chat history');
            "

            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Fallback MariaDB schema initialized successfully${NC}"
            else
                echo -e "${RED}Critical error initializing MariaDB schema${NC}"
            fi
        fi
    else
        echo -e "${GREEN}✅ MariaDB schema already initialized via container startup${NC}"
    fi
else
    echo -e "${GREEN}✅ MariaDB schema already initialized${NC}"

    # Check for schema version and do migrations if needed
    echo -e "${YELLOW}Checking for schema updates...${NC}"

    # Check if users table has settings column - this is a critical column for the new user settings feature
    if ! docker exec regulaite-mariadb mariadb -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "SHOW COLUMNS FROM users LIKE 'settings';" | grep -q "settings"; then
        echo -e "${YELLOW}Migrating users table to add settings column...${NC}"
        docker exec regulaite-mariadb mariadb -u"$MARIADB_USER" -p"$MARIADB_PASSWORD" "$MARIADB_DATABASE" -e "
            ALTER TABLE users ADD COLUMN settings JSON;
        "
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Users table migration completed${NC}"
        else
            echo -e "${RED}Error migrating users table${NC}"
        fi
    fi
fi

# Final status and information
echo -e "${GREEN}===============================================${NC}"
echo -e "${BLUE}🎉 RegulAite setup completed!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo -e "${BLUE}Environment: ${BUILD_TARGET^^}${NC}"
echo ""
echo -e "${BLUE}All services should be accessible at:${NC}"
echo -e "  Neo4j Browser: http://localhost:7474${NC}"
echo -e "  Neo4j Bolt: bolt://localhost:7687${NC}"
echo -e "  MariaDB: localhost:3306${NC}"
echo -e "  AI_BACKEND API: http://localhost:8090${NC}"
echo -e "  Unstructured Healthcheck: http://localhost:9900/healthcheck${NC}"
echo -e "  Unstructured API: http://localhost:9900/general/v0/general${NC}"
echo -e "  Qdrant Healthcheck: http://localhost:6333/healthz${NC}"
echo -e "  Qdrant URL: http://localhost:6333/${NC}"
echo -e "  Qdrant gRPC URL: http://localhost:6334${NC}"
echo ""
echo -e "${YELLOW}Neo4j credentials:${NC}"
echo -e "  Username: $NEO4J_USER"
echo -e "  Password: $NEO4J_PASSWORD"
echo ""
echo -e "${YELLOW}MariaDB credentials:${NC}"
echo -e "  Database: $MARIADB_DATABASE"
echo -e "  Root Password: $MARIADB_ROOT_PASSWORD"
echo -e "  Username: $MARIADB_USER"
echo -e "  Password: $MARIADB_PASSWORD"
echo ""
echo -e "${YELLOW}If any services failed to start, check their logs:${NC}"
echo -e "  docker logs regulaite-neo4j"
echo -e "  docker logs regulaite-mariadb"
echo -e "  docker logs regulaite-ai-backend"
echo -e "  docker logs regulaite-qdrant"
echo -e "  docker logs regulaite-front-end"
echo -e "  docker logs regulaite-unstructured"
echo -e "  docker logs regulaite-redis"
echo -e "  docker logs regulaite-celery-worker"
echo -e "  docker logs regulaite-celery-flower"
echo ""
echo -e "${YELLOW}Navigate to your application at:${NC}"
echo -e "  http://localhost:3000${NC}"