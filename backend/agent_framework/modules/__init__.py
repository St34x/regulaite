"""
Agent Framework Modules - Modules spécialisés pour l'analyse GRC.

Chaque module fournit une expertise spécialisée pour différents aspects
de l'analyse de gouvernance, risque et conformité.
"""

# Import the organization config module which is working
from .organization_config import (
    OrganizationConfigManager,
    OrganizationType,
    RegulatorySector,
    ComplianceFramework,
    OrganizationProfile,
    get_organization_config_manager
)

# Framework Parser avec support multi-frameworks (from tools directory)
from ..tools.framework_parser import FrameworkParser, get_framework_parser

# Comment out problematic modules for now - they have LLMIntegration vs LLMClient type issues
# from .compliance_analysis_module import ComplianceAnalysisModule, get_compliance_analysis_module
# from .governance_analysis_module import GovernanceAnalysisModule, get_governance_analysis_module
# from .risk_assessment_module import RiskAssessmentModule, get_risk_assessment_module
# from .gap_analysis_module import GapAnalysisModule, get_gap_analysis_module

__all__ = [
    # Organization Config (working)
    "OrganizationConfigManager",
    "OrganizationType", 
    "RegulatorySector",
    "ComplianceFramework",
    "OrganizationProfile",
    "get_organization_config_manager",
    
    # Framework Parser
    "FrameworkParser",
    "get_framework_parser",
    
    # Commented out until LLM type issues are resolved
    # "ComplianceAnalysisModule",
    # "get_compliance_analysis_module",
    # "GovernanceAnalysisModule", 
    # "get_governance_analysis_module",
    # "RiskAssessmentModule",
    # "get_risk_assessment_module",
    # "GapAnalysisModule",
    # "get_gap_analysis_module"
] 