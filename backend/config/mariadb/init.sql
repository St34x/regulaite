-- Create or check regulaite database
CREATE DATABASE IF NOT EXISTS regulaite CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE regulaite;

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
) ENGINE=InnoDB;

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
) ENGINE=InnoDB;

-- Table for global settings
CREATE TABLE IF NOT EXISTS regulaite_settings (
    setting_key VARCHAR(255) PRIMARY KEY,
    setting_value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    description TEXT
) ENGINE=InnoDB;

-- Insert default settings if they don't exist
INSERT IGNORE INTO regulaite_settings (setting_key, setting_value, description) VALUES
('llm_model', 'gpt-4', 'Default LLM model'),
('llm_temperature', '0.7', 'Default temperature for LLM'),
('llm_max_tokens', '2048', 'Default max tokens for LLM'),
('llm_top_p', '1', 'Default top_p value for LLM'),
('enable_chat_history', 'true', 'Whether to save chat history');

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
) ENGINE=InnoDB;

-- Create user table if needed for future authentication
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    settings JSON,

    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB;

-- Grant privileges to regulaite_user
GRANT ALL PRIVILEGES ON regulaite.* TO 'regulaite_user'@'%';
FLUSH PRIVILEGES;
