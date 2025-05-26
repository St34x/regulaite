"""
Agent Framework Modules - Modules spécialisés d'analyse GRC pour RegulAIte.
Avec intégration complète du contexte organisationnel et configuration adaptative.
"""

# Configuration organisationnelle
from .organization_config import (
    OrganizationConfigManager, OrganizationProfile, AssetTemplate, ThreatProfile,
    RegulatoryEnvironment, OrganizationType, RegulatorySector, ComplianceFramework,
    get_organization_config_manager
)

# Modules d'analyse spécialisés avec intégration organisationnelle
from .compliance_analysis_module import ComplianceAnalysisModule, get_compliance_analysis_module
from .governance_analysis_module import GovernanceAnalysisModule, get_governance_analysis_module
from .risk_assessment_module import RiskAssessmentModule, get_risk_assessment_module
from .gap_analysis_module import GapAnalysisModule, get_gap_analysis_module

# Framework Parser avec support multi-frameworks
from .framework_parser import FrameworkParser, get_framework_parser

__all__ = [
    # Configuration organisationnelle
    "OrganizationConfigManager",
    "OrganizationProfile", 
    "AssetTemplate",
    "ThreatProfile",
    "RegulatoryEnvironment",
    "OrganizationType",
    "RegulatorySector", 
    "ComplianceFramework",
    "get_organization_config_manager",
    
    # Modules d'analyse
    "ComplianceAnalysisModule",
    "GovernanceAnalysisModule", 
    "RiskAssessmentModule",
    "GapAnalysisModule",
    "FrameworkParser",
    
    # Factory functions
    "get_compliance_analysis_module",
    "get_governance_analysis_module",
    "get_risk_assessment_module", 
    "get_gap_analysis_module",
    "get_framework_parser"
] 