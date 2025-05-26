-- Migration Script: Upgrade to Comprehensive RegulAite Schema
-- This script safely migrates existing RegulAite databases to the new comprehensive schema
-- Run this script if you have an existing RegulAite installation

-- Set session variables for safety
SET FOREIGN_KEY_CHECKS = 0;
SET AUTOCOMMIT = 0;
START TRANSACTION;

-- =============================================
-- BACKUP EXISTING DATA (Create backup tables)
-- =============================================

-- Backup existing tables before migration
CREATE TABLE IF NOT EXISTS backup_chat_history_pre_migration AS SELECT * FROM chat_history WHERE 1=0;
CREATE TABLE IF NOT EXISTS backup_users_pre_migration AS SELECT * FROM users WHERE 1=0;
CREATE TABLE IF NOT EXISTS backup_tasks_pre_migration AS SELECT * FROM tasks WHERE 1=0;

-- Insert existing data into backup tables
INSERT IGNORE INTO backup_chat_history_pre_migration SELECT * FROM chat_history;
INSERT IGNORE INTO backup_users_pre_migration SELECT * FROM users;
INSERT IGNORE INTO backup_tasks_pre_migration SELECT * FROM tasks;

-- =============================================
-- MODIFY EXISTING TABLES
-- =============================================

-- Update regulaite_settings table
ALTER TABLE regulaite_settings 
ADD COLUMN IF NOT EXISTS category VARCHAR(50) DEFAULT 'general',
ADD COLUMN IF NOT EXISTS is_system BOOLEAN DEFAULT FALSE,
ADD INDEX IF NOT EXISTS idx_category (category),
ADD INDEX IF NOT EXISTS idx_is_system (is_system);

-- Update users table
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS role ENUM('admin', 'analyst', 'auditor', 'viewer') DEFAULT 'analyst',
ADD COLUMN IF NOT EXISTS organization_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
ADD INDEX IF NOT EXISTS idx_organization_id (organization_id),
ADD INDEX IF NOT EXISTS idx_role (role),
ADD INDEX IF NOT EXISTS idx_is_active (is_active);

-- Update chat_sessions table
ALTER TABLE chat_sessions 
ADD COLUMN IF NOT EXISTS organization_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS context_type ENUM('general', 'analysis', 'document_review', 'compliance_check') DEFAULT 'general',
ADD COLUMN IF NOT EXISTS context_data JSON,
ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE,
ADD INDEX IF NOT EXISTS idx_organization_id (organization_id),
ADD INDEX IF NOT EXISTS idx_is_archived (is_archived);

-- Update chat_history table
ALTER TABLE chat_history 
ADD COLUMN IF NOT EXISTS organization_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS agent_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS context_used BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS tokens_used INT,
ADD COLUMN IF NOT EXISTS processing_time_ms INT,
ADD INDEX IF NOT EXISTS idx_organization_id (organization_id),
ADD INDEX IF NOT EXISTS idx_agent_id (agent_id);

-- Update tasks table
ALTER TABLE tasks 
ADD COLUMN IF NOT EXISTS organization_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS title VARCHAR(255),
ADD COLUMN IF NOT EXISTS description TEXT,
ADD COLUMN IF NOT EXISTS priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
ADD COLUMN IF NOT EXISTS progress_percent FLOAT DEFAULT 0,
ADD COLUMN IF NOT EXISTS started_at TIMESTAMP NULL,
ADD COLUMN IF NOT EXISTS estimated_duration INT,
ADD COLUMN IF NOT EXISTS actual_duration INT,
ADD COLUMN IF NOT EXISTS assigned_agent VARCHAR(64),
ADD INDEX IF NOT EXISTS idx_user_id (user_id),
ADD INDEX IF NOT EXISTS idx_organization_id (organization_id),
ADD INDEX IF NOT EXISTS idx_priority (priority),
ADD INDEX IF NOT EXISTS idx_assigned_agent (assigned_agent);

-- Modify tasks status enum to include new values
ALTER TABLE tasks MODIFY COLUMN status ENUM('queued', 'processing', 'completed', 'failed', 'cancelled', 'pending_review') NOT NULL;

-- Update agent_executions table if it exists
ALTER TABLE agent_executions 
ADD COLUMN IF NOT EXISTS execution_id VARCHAR(255) UNIQUE,
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS organization_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS input_data JSON,
ADD COLUMN IF NOT EXISTS output_data JSON,
ADD COLUMN IF NOT EXISTS cost_estimate DECIMAL(10,6),
ADD COLUMN IF NOT EXISTS context_used BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS context_sources JSON,
ADD INDEX IF NOT EXISTS idx_execution_id (execution_id),
ADD INDEX IF NOT EXISTS idx_user_id (user_id),
ADD INDEX IF NOT EXISTS idx_organization_id (organization_id),
ADD INDEX IF NOT EXISTS idx_error (error);

-- Update agent_feedback table if it exists
ALTER TABLE agent_feedback 
ADD COLUMN IF NOT EXISTS execution_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS feedback_type ENUM('accuracy', 'helpfulness', 'completeness', 'performance', 'other') DEFAULT 'other',
ADD INDEX IF NOT EXISTS idx_execution_id (execution_id),
ADD INDEX IF NOT EXISTS idx_user_id (user_id);

-- Modify rating constraint
ALTER TABLE agent_feedback ADD CONSTRAINT chk_rating CHECK (rating >= 1 AND rating <= 5);

-- Update agent_progress table if it exists
ALTER TABLE agent_progress 
ADD COLUMN IF NOT EXISTS step_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS total_steps INT,
ADD COLUMN IF NOT EXISTS current_step INT,
ADD INDEX IF NOT EXISTS idx_timestamp (timestamp);

-- Update agent_analytics table if it exists  
ALTER TABLE agent_analytics 
ADD COLUMN IF NOT EXISTS organization_id VARCHAR(50),
ADD COLUMN IF NOT EXISTS success_count INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS error_count INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_tokens INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_cost DECIMAL(10,6) DEFAULT 0,
DROP INDEX IF EXISTS agent_analytics_unique_key,
ADD UNIQUE KEY unique_agent_org_day (agent_id, organization_id, day),
ADD INDEX IF NOT EXISTS idx_organization_id (organization_id);

-- =============================================
-- CREATE NEW TABLES (IF NOT EXISTS)
-- =============================================

-- Organizations table
CREATE TABLE IF NOT EXISTS organizations (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    sector ENUM('banking', 'insurance', 'healthcare', 'energy', 'telecom', 'technology', 'public', 'general') DEFAULT 'general',
    size ENUM('startup', 'small', 'medium', 'large', 'enterprise') DEFAULT 'medium',
    employee_count INT DEFAULT 100,
    annual_revenue VARCHAR(50) DEFAULT '10M-100M',
    organization_type ENUM('startup', 'sme', 'large_corp', 'public_sector', 'healthcare', 'financial', 'technology', 'manufacturing', 'energy', 'telecom') DEFAULT 'technology',
    business_model ENUM('traditional', 'digital', 'hybrid', 'platform') DEFAULT 'traditional',
    digital_maturity ENUM('basic', 'intermediate', 'advanced', 'leading') DEFAULT 'intermediate',
    risk_appetite ENUM('conservative', 'moderate', 'aggressive') DEFAULT 'moderate',
    geographical_presence JSON,
    custom_settings JSON,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_sector (sector),
    INDEX idx_size (size),
    INDEX idx_organization_type (organization_type),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organization assets table
CREATE TABLE IF NOT EXISTS organization_assets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    asset_id VARCHAR(50) NOT NULL,
    asset_name VARCHAR(255) NOT NULL,
    asset_type ENUM('system', 'data', 'application', 'network', 'human', 'physical') NOT NULL,
    criticality ENUM('very_low', 'low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    description TEXT,
    owner VARCHAR(255),
    location VARCHAR(255),
    business_value ENUM('very_low', 'low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    regulatory_classification VARCHAR(100),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_org_asset (organization_id, asset_id),
    INDEX idx_organization_id (organization_id),
    INDEX idx_asset_type (asset_type),
    INDEX idx_criticality (criticality),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organization threat profiles table
CREATE TABLE IF NOT EXISTS organization_threat_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    threat_type VARCHAR(100) NOT NULL,
    likelihood ENUM('very_low', 'low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    sophistication ENUM('basic', 'intermediate', 'advanced', 'expert') DEFAULT 'intermediate',
    motivation ENUM('financial', 'espionage', 'activism', 'disruption') DEFAULT 'financial',
    resources ENUM('limited', 'moderate', 'substantial', 'extensive') DEFAULT 'moderate',
    persistence ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    INDEX idx_organization_id (organization_id),
    INDEX idx_threat_type (threat_type),
    INDEX idx_likelihood (likelihood),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organization regulatory environment table
CREATE TABLE IF NOT EXISTS organization_regulatory_env (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    frameworks JSON,
    regulatory_pressure ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    audit_frequency ENUM('quarterly', 'bi_annual', 'annual', 'ad_hoc') DEFAULT 'annual',
    penalties_exposure ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    external_oversight JSON,
    certification_requirements JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_org_regulatory (organization_id),
    INDEX idx_organization_id (organization_id),
    INDEX idx_regulatory_pressure (regulatory_pressure)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organization governance maturity table
CREATE TABLE IF NOT EXISTS organization_governance_maturity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    domain VARCHAR(50) NOT NULL,
    maturity_level ENUM('initial', 'developing', 'defined', 'managed', 'optimized') DEFAULT 'developing',
    assessment_date DATE,
    assessor VARCHAR(255),
    score INT,
    comments TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_org_domain (organization_id, domain),
    INDEX idx_organization_id (organization_id),
    INDEX idx_domain (domain),
    INDEX idx_maturity_level (maturity_level),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    organization_id VARCHAR(50),
    filename VARCHAR(500) NOT NULL,
    original_filename VARCHAR(500),
    file_path VARCHAR(1000),
    file_size BIGINT,
    file_hash VARCHAR(64),
    mime_type VARCHAR(100),
    document_type ENUM('policy', 'procedure', 'risk_assessment', 'audit_report', 'compliance_report', 'framework', 'evidence', 'other') DEFAULT 'other',
    classification ENUM('public', 'internal', 'confidential', 'restricted') DEFAULT 'internal',
    status ENUM('pending', 'processing', 'processed', 'failed', 'archived') DEFAULT 'pending',
    processing_status JSON,
    metadata JSON,
    uploaded_by VARCHAR(255),
    processed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL,
    FOREIGN KEY (uploaded_by) REFERENCES users(user_id) ON DELETE SET NULL,
    INDEX idx_organization_id (organization_id),
    INDEX idx_document_type (document_type),
    INDEX idx_status (status),
    INDEX idx_classification (classification),
    INDEX idx_uploaded_by (uploaded_by),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organization custom frameworks table
CREATE TABLE IF NOT EXISTS organization_custom_frameworks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    framework_name VARCHAR(100) NOT NULL,
    framework_version VARCHAR(20),
    framework_data JSON NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL,
    UNIQUE KEY unique_org_framework (organization_id, framework_name, framework_version),
    INDEX idx_organization_id (organization_id),
    INDEX idx_framework_name (framework_name),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    analysis_type ENUM('risk_assessment', 'compliance_analysis', 'governance_analysis', 'gap_analysis', 'vulnerability_assessment', 'audit_preparation') NOT NULL,
    analysis_id VARCHAR(100),
    title VARCHAR(255),
    description TEXT,
    result_data JSON NOT NULL,
    summary JSON,
    recommendations JSON,
    metadata JSON,
    status ENUM('pending', 'processing', 'completed', 'failed', 'archived') DEFAULT 'completed',
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    created_by VARCHAR(255),
    assigned_to VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    expires_at TIMESTAMP NULL,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (assigned_to) REFERENCES users(user_id) ON DELETE SET NULL,
    INDEX idx_organization_id (organization_id),
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_analysis_id (analysis_id),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organization assessment history table
CREATE TABLE IF NOT EXISTS organization_assessment_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    assessment_type VARCHAR(50) NOT NULL,
    assessment_date DATE NOT NULL,
    methodology VARCHAR(50),
    scope TEXT,
    assessor VARCHAR(255),
    score DECIMAL(5,2),
    summary TEXT,
    detailed_results JSON,
    recommendations JSON,
    status ENUM('draft', 'final', 'superseded') DEFAULT 'final',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (assessor) REFERENCES users(user_id) ON DELETE SET NULL,
    INDEX idx_organization_id (organization_id),
    INDEX idx_assessment_type (assessment_type),
    INDEX idx_assessment_date (assessment_date),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    agent_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type ENUM('universal_tool', 'risk_assessment', 'compliance_analysis', 'governance_analysis', 'gap_analysis', 'document_processor') NOT NULL,
    capabilities JSON,
    configuration JSON,
    model_config JSON,
    is_active BOOLEAN DEFAULT TRUE,
    version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_agent_type (agent_type),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System analytics table
CREATE TABLE IF NOT EXISTS system_analytics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6),
    metric_category ENUM('performance', 'usage', 'cost', 'quality', 'error') NOT NULL,
    organization_id VARCHAR(50),
    day DATE NOT NULL,
    metadata JSON,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_metric_org_day (metric_name, organization_id, day),
    INDEX idx_metric_name (metric_name),
    INDEX idx_metric_category (metric_category),
    INDEX idx_organization_id (organization_id),
    INDEX idx_day (day)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Audit trail table
CREATE TABLE IF NOT EXISTS audit_trail (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSON,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    INDEX idx_organization_id (organization_id),
    INDEX idx_user_id (user_id),
    INDEX idx_action (action),
    INDEX idx_resource_type (resource_type),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================
-- ADD NEW SETTINGS AND DEFAULT DATA
-- =============================================

-- Insert new settings
INSERT IGNORE INTO regulaite_settings (setting_key, setting_value, description, category, is_system) VALUES
('max_file_size_mb', '50', 'Maximum file upload size in MB', 'documents', FALSE),
('supported_file_types', '["pdf", "docx", "doc", "txt", "xlsx", "csv"]', 'Supported file types for upload', 'documents', FALSE),
('rag_chunk_size', '1000', 'Default chunk size for RAG processing', 'ai', TRUE),
('rag_chunk_overlap', '200', 'Default chunk overlap for RAG processing', 'ai', TRUE),
('default_analysis_retention_days', '365', 'Default retention period for analysis results', 'general', FALSE);

-- Update existing settings categories
UPDATE regulaite_settings SET category = 'ai', is_system = TRUE 
WHERE setting_key IN ('llm_model', 'llm_temperature', 'llm_max_tokens', 'llm_top_p');

-- Insert default organization if none exists
INSERT IGNORE INTO organizations (
    id, name, sector, size, organization_type, 
    business_model, digital_maturity, risk_appetite
) VALUES (
    'default_org', 
    'RegulAIte Demo Organization', 
    'technology', 
    'medium', 
    'technology',
    'digital', 
    'intermediate', 
    'moderate'
);

-- Insert default agents
INSERT IGNORE INTO agents (agent_id, name, description, agent_type, capabilities, configuration) VALUES
('universal_tools', 'Universal Tool Suite', 'Provides foundational GRC capabilities including document finding, entity extraction, and cross-referencing', 'universal_tool', 
 JSON_ARRAY('document_finder', 'entity_extractor', 'cross_reference', 'document_relationship_mapper', 'temporal_analyzer'),
 JSON_OBJECT('max_documents', 1000, 'entity_types', JSON_ARRAY('controls', 'risks', 'findings', 'requirements', 'assets'))),

('risk_assessment', 'Risk Assessment Module', 'Comprehensive risk analysis and assessment capabilities', 'risk_assessment',
 JSON_ARRAY('risk_identification', 'risk_analysis', 'risk_evaluation', 'risk_treatment', 'risk_monitoring'),
 JSON_OBJECT('risk_scales', JSON_OBJECT('likelihood', JSON_ARRAY('very_low', 'low', 'medium', 'high', 'very_high'), 'impact', JSON_ARRAY('very_low', 'low', 'medium', 'high', 'very_high')))),

('compliance_analysis', 'Compliance Analysis Module', 'Framework compliance analysis and gap identification', 'compliance_analysis',
 JSON_ARRAY('framework_mapping', 'gap_analysis', 'compliance_monitoring', 'control_assessment'),
 JSON_OBJECT('supported_frameworks', JSON_ARRAY('iso27001', 'iso27002', 'nist', 'sox', 'gdpr', 'hipaa'))),

('governance_analysis', 'Governance Analysis Module', 'Organizational governance maturity and effectiveness analysis', 'governance_analysis',
 JSON_ARRAY('maturity_assessment', 'governance_structure_analysis', 'policy_effectiveness', 'strategic_alignment'),
 JSON_OBJECT('maturity_model', 'cmmi', 'assessment_areas', JSON_ARRAY('strategic', 'risk', 'compliance', 'security'))),

('gap_analysis', 'Gap Analysis Module', 'Identifies gaps between current state and desired state', 'gap_analysis',
 JSON_ARRAY('current_state_assessment', 'target_state_definition', 'gap_identification', 'remediation_planning'),
 JSON_OBJECT('analysis_types', JSON_ARRAY('control_gaps', 'process_gaps', 'technology_gaps', 'skill_gaps'))),

('document_processor', 'Document Processing Module', 'Intelligent document processing and analysis', 'document_processor',
 JSON_ARRAY('document_parsing', 'content_extraction', 'classification', 'metadata_enrichment'),
 JSON_OBJECT('supported_formats', JSON_ARRAY('pdf', 'docx', 'doc', 'txt', 'xlsx'), 'extraction_methods', JSON_ARRAY('unstructured', 'llamaparse', 'doctly')));

-- =============================================
-- ADD FOREIGN KEY CONSTRAINTS SAFELY
-- =============================================

-- Add foreign key constraint for users table if organizations exist
SET @fk_exists = (SELECT COUNT(*) FROM information_schema.KEY_COLUMN_USAGE 
                  WHERE TABLE_SCHEMA = DATABASE() 
                  AND TABLE_NAME = 'users' 
                  AND CONSTRAINT_NAME = 'fk_users_organization');

SET @sql = IF(@fk_exists = 0, 
    'ALTER TABLE users ADD CONSTRAINT fk_users_organization FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL', 
    'SELECT "Foreign key already exists" as message');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Update existing users to default organization if they don't have one
UPDATE users SET organization_id = 'default_org' WHERE organization_id IS NULL;

-- =============================================
-- CREATE OR REPLACE VIEWS
-- =============================================

-- Drop existing views if they exist
DROP VIEW IF EXISTS organization_complete_profiles;
DROP VIEW IF EXISTS recent_analyses_by_org;
DROP VIEW IF EXISTS agent_performance_summary;

-- Create comprehensive organization profiles view
CREATE VIEW organization_complete_profiles AS
SELECT 
    o.*,
    ore.frameworks,
    ore.regulatory_pressure,
    ore.audit_frequency,
    COUNT(DISTINCT oa.id) as asset_count,
    COUNT(DISTINCT otp.id) as threat_count,
    AVG(CASE 
        WHEN ogm.maturity_level = 'initial' THEN 1
        WHEN ogm.maturity_level = 'developing' THEN 2
        WHEN ogm.maturity_level = 'defined' THEN 3
        WHEN ogm.maturity_level = 'managed' THEN 4
        WHEN ogm.maturity_level = 'optimized' THEN 5
        ELSE 2
    END) as avg_governance_maturity
FROM organizations o
LEFT JOIN organization_regulatory_env ore ON o.id = ore.organization_id
LEFT JOIN organization_assets oa ON o.id = oa.organization_id AND oa.active = 1
LEFT JOIN organization_threat_profiles otp ON o.id = otp.organization_id AND otp.active = 1
LEFT JOIN organization_governance_maturity ogm ON o.id = ogm.organization_id AND ogm.active = 1
WHERE o.active = 1
GROUP BY o.id, ore.frameworks, ore.regulatory_pressure, ore.audit_frequency;

-- Create recent analyses view
CREATE VIEW recent_analyses_by_org AS
SELECT 
    organization_id,
    analysis_type,
    COUNT(*) as analysis_count,
    MAX(created_at) as last_analysis_date,
    AVG(JSON_EXTRACT(metadata, '$.duration_seconds')) as avg_duration
FROM analysis_results 
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    AND status = 'completed'
GROUP BY organization_id, analysis_type;

-- Create agent performance summary view
CREATE VIEW agent_performance_summary AS
SELECT 
    a.agent_id,
    a.name,
    a.agent_type,
    COUNT(ae.id) as total_executions,
    COUNT(CASE WHEN ae.error = FALSE THEN 1 END) as successful_executions,
    AVG(ae.response_time_ms) as avg_response_time,
    AVG(af.rating) as avg_rating,
    SUM(ae.cost_estimate) as total_cost
FROM agents a
LEFT JOIN agent_executions ae ON a.agent_id = ae.agent_id
LEFT JOIN agent_feedback af ON ae.execution_id = af.execution_id
WHERE a.is_active = TRUE
    AND ae.timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY a.agent_id, a.name, a.agent_type;

-- =============================================
-- FINALIZE MIGRATION
-- =============================================

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- Commit the transaction
COMMIT;

-- Verify migration success
SELECT 
    'Migration completed successfully' as status,
    COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = DATABASE() 
    AND table_name IN ('organizations', 'agents', 'analysis_results', 'documents');

-- Show summary of new tables created
SELECT 
    table_name,
    table_rows,
    table_comment
FROM information_schema.tables 
WHERE table_schema = DATABASE() 
    AND table_name LIKE 'organization%'
ORDER BY table_name; 