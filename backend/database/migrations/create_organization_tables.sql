-- Migration: Create Organization Configuration Tables for RegulAIte
-- This adds organizational context and configuration management to RegulAIte

-- Table principale des organisations
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
);

-- Table des actifs organisationnels
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
);

-- Table des profils de menaces organisationnelles
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
);

-- Table de l'environnement réglementaire
CREATE TABLE IF NOT EXISTS organization_regulatory_env (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    frameworks JSON, -- Liste des frameworks applicables
    regulatory_pressure ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    audit_frequency ENUM('quarterly', 'bi_annual', 'annual', 'ad_hoc') DEFAULT 'annual',
    penalties_exposure ENUM('low', 'medium', 'high', 'very_high') DEFAULT 'medium',
    external_oversight JSON, -- Liste des autorités de supervision
    certification_requirements JSON, -- Liste des certifications requises
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_org_regulatory (organization_id),
    INDEX idx_organization_id (organization_id),
    INDEX idx_regulatory_pressure (regulatory_pressure)
);

-- Table de maturité de gouvernance
CREATE TABLE IF NOT EXISTS organization_governance_maturity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    domain VARCHAR(50) NOT NULL, -- strategic, risk, compliance, etc.
    maturity_level ENUM('initial', 'developing', 'defined', 'managed', 'optimized') DEFAULT 'developing',
    assessment_date DATE,
    assessor VARCHAR(255),
    score INT, -- Score numérique optionnel (0-100)
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
);

-- Table des résultats d'analyse
CREATE TABLE IF NOT EXISTS analysis_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    analysis_type ENUM('risk_assessment', 'compliance_analysis', 'governance_analysis', 'gap_analysis') NOT NULL,
    analysis_id VARCHAR(100), -- ID unique de l'analyse
    result_data JSON NOT NULL, -- Résultats complets de l'analyse
    metadata JSON, -- Métadonnées (versions, paramètres, etc.)
    status ENUM('pending', 'completed', 'failed', 'archived') DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NULL, -- Optionnel pour l'archivage automatique
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    INDEX idx_organization_id (organization_id),
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_analysis_id (analysis_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Table de configuration des frameworks personnalisés
CREATE TABLE IF NOT EXISTS organization_custom_frameworks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    framework_name VARCHAR(100) NOT NULL,
    framework_version VARCHAR(20),
    framework_data JSON NOT NULL, -- Structure complète du framework
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    UNIQUE KEY unique_org_framework (organization_id, framework_name, framework_version),
    INDEX idx_organization_id (organization_id),
    INDEX idx_framework_name (framework_name),
    INDEX idx_is_active (is_active)
);

-- Table d'historique des évaluations
CREATE TABLE IF NOT EXISTS organization_assessment_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organization_id VARCHAR(50) NOT NULL,
    assessment_type VARCHAR(50) NOT NULL,
    assessment_date DATE NOT NULL,
    methodology VARCHAR(50),
    scope TEXT,
    assessor VARCHAR(255),
    score DECIMAL(5,2), -- Score global (ex: 85.50)
    summary TEXT,
    detailed_results JSON,
    recommendations JSON,
    status ENUM('draft', 'final', 'superseded') DEFAULT 'final',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    INDEX idx_organization_id (organization_id),
    INDEX idx_assessment_type (assessment_type),
    INDEX idx_assessment_date (assessment_date),
    INDEX idx_status (status)
);

-- Insertion d'une organisation par défaut pour les tests
INSERT IGNORE INTO organizations (
    id, name, sector, size, organization_type, 
    business_model, digital_maturity, risk_appetite
) VALUES (
    'default_org', 
    'Organisation de test RegulAIte', 
    'technology', 
    'medium', 
    'technology',
    'digital', 
    'intermediate', 
    'moderate'
);

-- Insertion d'actifs par défaut
INSERT IGNORE INTO organization_assets (
    organization_id, asset_id, asset_name, asset_type, criticality, description
) VALUES 
    ('default_org', 'SYS-001', 'Systèmes d''information', 'system', 'high', 'Infrastructure IT principale'),
    ('default_org', 'DATA-001', 'Données sensibles', 'data', 'very_high', 'Données clients et données métier'),
    ('default_org', 'APP-001', 'Applications métier', 'application', 'high', 'Applications critiques business'),
    ('default_org', 'NET-001', 'Infrastructure réseau', 'network', 'medium', 'Réseau et télécommunications'),
    ('default_org', 'HR-001', 'Personnel', 'human', 'high', 'Ressources humaines et compétences');

-- Insertion de profils de menaces par défaut
INSERT IGNORE INTO organization_threat_profiles (
    organization_id, threat_type, likelihood, sophistication, motivation
) VALUES 
    ('default_org', 'Cybercriminels', 'medium', 'intermediate', 'financial'),
    ('default_org', 'Menaces internes', 'low', 'basic', 'financial'),
    ('default_org', 'Hacktivistes', 'low', 'intermediate', 'activism'),
    ('default_org', 'Erreurs humaines', 'high', 'basic', 'disruption');

-- Insertion d'un environnement réglementaire par défaut
INSERT IGNORE INTO organization_regulatory_env (
    organization_id, frameworks, regulatory_pressure, audit_frequency
) VALUES (
    'default_org', 
    JSON_ARRAY('iso27001', 'rgpd'), 
    'medium', 
    'annual'
);

-- Insertion de maturité de gouvernance par défaut
INSERT IGNORE INTO organization_governance_maturity (
    organization_id, domain, maturity_level, assessment_date
) VALUES 
    ('default_org', 'strategic', 'defined', CURDATE()),
    ('default_org', 'risk', 'developing', CURDATE()),
    ('default_org', 'compliance', 'defined', CURDATE()),
    ('default_org', 'security', 'developing', CURDATE());

-- Vues pour faciliter les requêtes

-- Vue complète des organisations avec leurs contextes
CREATE OR REPLACE VIEW organization_complete_profiles AS
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

-- Vue des analyses récentes par organisation
CREATE OR REPLACE VIEW recent_analyses_by_org AS
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