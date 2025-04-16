-- Migration to add agent feedback and execution tracking
-- For MariaDB/MySQL

-- Table for storing feedback on agent responses
CREATE TABLE IF NOT EXISTS agent_feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    message_id VARCHAR(64) DEFAULT '',
    rating INT NOT NULL,
    feedback_text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    context_used BOOLEAN,
    model VARCHAR(64),
    
    INDEX (agent_id),
    INDEX (session_id)
);

-- Table for tracking agent executions
CREATE TABLE IF NOT EXISTS agent_executions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    task TEXT NOT NULL,
    model VARCHAR(64),
    response_time_ms INT,
    token_count INT,
    prompt_token_count INT,
    completion_token_count INT,
    error BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX (agent_id),
    INDEX (session_id),
    INDEX (timestamp)
);

-- Table for tracking agent execution progress (for long-running tasks)
CREATE TABLE IF NOT EXISTS agent_progress (
    id INT AUTO_INCREMENT PRIMARY KEY,
    execution_id INT NOT NULL,
    progress_percent FLOAT,
    status VARCHAR(32) NOT NULL,
    status_message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (execution_id) REFERENCES agent_executions(id) ON DELETE CASCADE,
    INDEX (execution_id)
);

-- Table for agent usage analytics
CREATE TABLE IF NOT EXISTS agent_analytics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_id VARCHAR(64) NOT NULL,
    day DATE NOT NULL,
    execution_count INT DEFAULT 0,
    avg_response_time_ms FLOAT,
    avg_rating FLOAT,
    error_rate FLOAT,
    unique_users INT DEFAULT 0,
    success_rate FLOAT,
    
    UNIQUE KEY (agent_id, day),
    INDEX (agent_id),
    INDEX (day)
); 