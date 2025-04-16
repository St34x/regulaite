"""
Cybersecurity-specific decision trees for GRC agents.
"""
import logging
from typing import Dict
from .tree_reasoning import DecisionNode, DecisionTree
from .decision_trees import register_tree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_cybersecurity_incident_response_tree() -> DecisionTree:
    """
    Create a decision tree for cybersecurity incident response.
    
    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="cybersecurity_incident_response",
        name="Cybersecurity Incident Response",
        description="Decision tree for analyzing and responding to cybersecurity incidents",
        root_node_id="root",
        nodes={
            "root": DecisionNode(
                id="root",
                description="Determine the type of security incident",
                prompt="Analyze the security incident described in the query and determine what type of incident it is:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "malware": "malware_incident",
                    "data_breach": "data_breach",
                    "unauthorized_access": "unauthorized_access",
                    "denial_of_service": "denial_of_service",
                    "social_engineering": "social_engineering",
                    "insider_threat": "insider_threat",
                    "other": "general_incident"
                }
            ),
            "malware_incident": DecisionNode(
                id="malware_incident",
                description="Determine the type of malware",
                prompt="This is a malware incident. Determine what type of malware is involved:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "ransomware": "ransomware_incident",
                    "virus": "virus_incident",
                    "trojan": "trojan_incident",
                    "worm": "worm_incident",
                    "other_malware": "general_malware"
                }
            ),
            "data_breach": DecisionNode(
                id="data_breach",
                description="Determine the scope of the data breach",
                prompt="This is a data breach incident. Determine the scope of the breach:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "pii_breach": "pii_breach",
                    "financial_data_breach": "financial_data_breach",
                    "health_data_breach": "health_data_breach",
                    "intellectual_property": "ip_breach",
                    "multiple_data_types": "multiple_data_breach"
                }
            ),
            "unauthorized_access": DecisionNode(
                id="unauthorized_access",
                description="Determine the type of unauthorized access",
                prompt="This is an unauthorized access incident. Determine the nature of the access:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "credential_theft": "credential_theft",
                    "privilege_escalation": "privilege_escalation",
                    "session_hijacking": "session_hijacking",
                    "other_access": "general_unauthorized_access"
                }
            ),
            "denial_of_service": DecisionNode(
                id="denial_of_service",
                description="Determine the type of DoS attack",
                prompt="This is a denial of service incident. Determine the type of DoS attack:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "ddos": "ddos_attack",
                    "application_dos": "application_dos",
                    "resource_exhaustion": "resource_exhaustion",
                    "other_dos": "general_dos"
                }
            ),
            "social_engineering": DecisionNode(
                id="social_engineering",
                description="Determine the type of social engineering attack",
                prompt="This is a social engineering incident. Determine the type of social engineering attack:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "phishing": "phishing_attack",
                    "spear_phishing": "spear_phishing",
                    "pretexting": "pretexting",
                    "other_social": "general_social_engineering"
                }
            ),
            "insider_threat": DecisionNode(
                id="insider_threat", 
                description="Determine the type of insider threat",
                prompt="This is an insider threat incident. Determine the nature of the insider threat:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "malicious_insider": "malicious_insider",
                    "negligent_insider": "negligent_insider",
                    "compromised_insider": "compromised_insider"
                }
            ),
            # Leaf nodes for malware incidents
            "ransomware_incident": DecisionNode(
                id="ransomware_incident",
                description="Provide guidance for ransomware incident",
                is_leaf=True,
                prompt="",
                action="provide_ransomware_response",
                response_template="This appears to be a ransomware incident. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Isolate affected systems immediately to prevent lateral movement\n2. Activate your incident response team\n3. Determine the ransomware variant if possible\n4. Assess what data and systems are affected\n5. Check if you have clean, offline backups\n6. Contact law enforcement and consider regulatory reporting obligations\n7. Do NOT pay the ransom immediately - consult with security experts and law enforcement first"
            ),
            "virus_incident": DecisionNode(
                id="virus_incident",
                description="Provide guidance for virus incident",
                is_leaf=True,
                prompt="",
                action="provide_virus_response",
                response_template="This appears to be a virus incident. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Isolate affected systems from the network\n2. Run full system scans with updated antivirus software\n3. Identify the virus strain and its behaviors\n4. Remove the virus using appropriate security tools\n5. Check for persistent threats and remove them\n6. Patch vulnerable systems that may have been the entry point\n7. Review logs to determine the infection vector"
            ),
            "trojan_incident": DecisionNode(
                id="trojan_incident",
                description="Provide guidance for trojan incident",
                is_leaf=True,
                prompt="",
                action="provide_trojan_response",
                response_template="This appears to be a trojan incident. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Isolate affected systems from the network\n2. Identify what the trojan is designed to do (data theft, backdoor, etc.)\n3. Run specialized anti-malware tools to remove the trojan\n4. Look for persistence mechanisms the trojan may have installed\n5. Review logs to determine how the trojan was installed\n6. Check for data exfiltration and what information may have been compromised\n7. Implement improved application control policies"
            ),
            "worm_incident": DecisionNode(
                id="worm_incident",
                description="Provide guidance for worm incident",
                is_leaf=True,
                prompt="",
                action="provide_worm_response",
                response_template="This appears to be a worm incident. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Isolate affected network segments immediately to prevent further spread\n2. Apply emergency patches for the vulnerability the worm is exploiting\n3. Identify the worm variant and its specific behaviors\n4. Conduct network-wide scans to identify all infected systems\n5. Clean or reimage infected systems\n6. Monitor network traffic for unusual patterns indicating reinfection\n7. Review and strengthen network segmentation"
            ),
            "general_malware": DecisionNode(
                id="general_malware",
                description="Provide guidance for general malware incident",
                is_leaf=True,
                prompt="",
                action="provide_general_malware_response",
                response_template="This appears to be a malware incident. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Isolate affected systems from the network\n2. Run comprehensive malware scans with multiple tools\n3. Identify the malware type and its capabilities\n4. Remove the malware and any persistence mechanisms\n5. Verify system integrity after cleanup\n6. Review how the malware entered your environment\n7. Update security controls to prevent similar incidents"
            ),
            
            # Leaf nodes for data breach incidents
            "pii_breach": DecisionNode(
                id="pii_breach",
                description="Provide guidance for PII data breach",
                is_leaf=True,
                prompt="",
                action="provide_pii_breach_response",
                response_template="This appears to be a breach involving personally identifiable information (PII). Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Identify exactly what PII data was exposed and to whom\n2. Contain the breach by closing access points\n3. Preserve evidence for forensic investigation\n4. Determine regulatory reporting requirements (GDPR, CCPA, etc.)\n5. Prepare for mandatory notification to affected individuals\n6. Consider offering credit monitoring or identity theft protection\n7. Document the breach timeline and response for regulators"
            ),
            "financial_data_breach": DecisionNode(
                id="financial_data_breach",
                description="Provide guidance for financial data breach",
                is_leaf=True,
                prompt="",
                action="provide_financial_data_breach_response",
                response_template="This appears to be a breach involving financial data. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Identify what financial data was exposed (card numbers, account details, etc.)\n2. Contain the breach and secure all financial systems\n3. Notify financial institutions and payment processors\n4. Determine PCI DSS reporting requirements\n5. Consider implementing card reissuance for affected accounts\n6. Increase transaction monitoring for affected accounts\n7. Work with financial partners on fraud prevention measures"
            ),
            "health_data_breach": DecisionNode(
                id="health_data_breach",
                description="Provide guidance for health data breach",
                is_leaf=True,
                prompt="",
                action="provide_health_data_breach_response",
                response_template="This appears to be a breach involving health data. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Determine exactly what health information was exposed\n2. Contain the breach and secure all health information systems\n3. Assess HIPAA/HITECH reporting requirements\n4. Prepare for required notifications to affected individuals\n5. Report to the Department of Health and Human Services if required\n6. Document the breach risk assessment and response\n7. Implement enhanced controls for protected health information"
            ),
            
            # Add other leaf nodes for different incident types...
            
            "general_incident": DecisionNode(
                id="general_incident",
                description="Provide guidance for general security incident",
                is_leaf=True,
                prompt="",
                action="provide_general_incident_response",
                response_template="This appears to be a security incident. Based on the available information:\n\n{CONTEXT}\n\nHere's how you should respond:\n\n1. Contain the incident by isolating affected systems\n2. Gather initial evidence while preserving the state of systems\n3. Identify the scope and impact of the incident\n4. Analyze the attack vector and techniques used\n5. Develop and execute a remediation plan\n6. Determine if regulatory reporting is required\n7. Document the incident and response for post-incident review"
            ),
        }
    )
    
    # Add more nodes as needed...
    
    return tree

def create_security_control_assessment_tree() -> DecisionTree:
    """
    Create a decision tree for security control assessment.
    
    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="security_control_assessment",
        name="Security Control Assessment",
        description="Decision tree for assessing security controls across different frameworks",
        root_node_id="root",
        nodes={
            "root": DecisionNode(
                id="root",
                description="Determine the control domain",
                prompt="Analyze the security control query and determine which control domain it falls under:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "access_control": "access_control_domain",
                    "network_security": "network_security_domain",
                    "data_protection": "data_protection_domain",
                    "identity_management": "identity_management_domain",
                    "vulnerability_management": "vulnerability_management_domain",
                    "security_operations": "security_operations_domain",
                    "governance": "governance_domain",
                    "physical_security": "physical_security_domain",
                    "other": "general_control_domain"
                }
            ),
            "access_control_domain": DecisionNode(
                id="access_control_domain",
                description="Determine the access control sub-domain",
                prompt="This relates to access control. Determine the specific aspect of access control:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "least_privilege": "least_privilege",
                    "segregation_of_duties": "segregation_of_duties",
                    "privileged_access": "privileged_access",
                    "authentication": "authentication",
                    "authorization": "authorization",
                    "other_access": "general_access_control"
                }
            ),
            # More domain nodes would go here...
            
            # Leaf nodes for access control sub-domains
            "least_privilege": DecisionNode(
                id="least_privilege",
                description="Assess least privilege controls",
                is_leaf=True,
                prompt="",
                action="assess_least_privilege_controls",
                response_template="Regarding least privilege access controls:\n\n{CONTEXT}\n\nBased on best practices and the information provided, you should assess these aspects:\n\n1. Whether users have only the minimum permissions needed for their role\n2. Process for regular review and removal of excessive permissions\n3. Automation of access provisioning and de-provisioning\n4. Default deny access policies\n5. Temporary privilege elevation processes\n\nImplementation guidance:\n- Document role-based access control matrices\n- Implement automated tools for access reviews\n- Use just-in-time access where possible\n- Monitor and alert on deviation from least privilege policies"
            ),
            "privileged_access": DecisionNode(
                id="privileged_access",
                description="Assess privileged access controls",
                is_leaf=True,
                prompt="",
                action="assess_privileged_access_controls",
                response_template="Regarding privileged access management controls:\n\n{CONTEXT}\n\nBased on best practices and the information provided, you should assess these aspects:\n\n1. Privileged account inventory and management\n2. Privileged access management (PAM) solutions\n3. Just-in-time access for privileged accounts\n4. Privileged session monitoring and recording\n5. Separation of admin accounts from regular user accounts\n\nImplementation guidance:\n- Deploy a PAM solution for secure credential vaulting\n- Implement multi-factor authentication for all privileged access\n- Establish break-glass procedures for emergency access\n- Ensure privileged actions are logged to a secure, immutable log store\n- Set up alerts for unusual privileged account usage"
            ),
            # More leaf nodes would go here...
            
            "general_control_domain": DecisionNode(
                id="general_control_domain",
                description="Provide general security control guidance",
                is_leaf=True,
                prompt="",
                action="provide_general_security_control_guidance",
                response_template="Regarding security controls:\n\n{CONTEXT}\n\nBased on the information provided, here are general control assessment guidelines:\n\n1. Determine the control objectives and requirements\n2. Identify relevant frameworks (NIST, ISO, CIS, etc.) that address these controls\n3. Assess current implementation against framework requirements\n4. Identify gaps and develop a plan to address them\n5. Implement continuous monitoring of control effectiveness\n\nKey considerations:\n- Ensure controls are appropriate for your risk profile\n- Document control implementations and evidence of effectiveness\n- Test controls regularly through audits and assessments\n- Maintain an up-to-date control inventory mapped to frameworks\n- Consider the impact of changes to the environment on controls"
            )
        }
    )
    
    # Add more nodes as needed...
    
    return tree

def create_third_party_risk_tree() -> DecisionTree:
    """
    Create a decision tree for third-party risk assessment.
    
    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="third_party_risk",
        name="Third-Party Risk Assessment",
        description="Decision tree for assessing cybersecurity risks from third parties and vendors",
        root_node_id="root",
        nodes={
            "root": DecisionNode(
                id="root",
                description="Determine the third-party risk domain",
                prompt="Analyze the third-party risk query and determine which risk domain it falls under:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "data_sharing": "data_sharing_risk",
                    "access_rights": "access_rights_risk",
                    "supply_chain": "supply_chain_risk",
                    "compliance": "third_party_compliance",
                    "operational": "operational_dependency",
                    "other": "general_third_party_risk"
                }
            ),
            "data_sharing_risk": DecisionNode(
                id="data_sharing_risk",
                description="Assess data sharing risk with third parties",
                prompt="This relates to data sharing with third parties. Determine the data sensitivity level:\n\n{QUERY}\n\n{CONTEXT}",
                children={
                    "highly_sensitive": "highly_sensitive_data",
                    "sensitive": "sensitive_data",
                    "moderate": "moderate_data",
                    "low": "low_sensitivity_data"
                }
            ),
            # More domain nodes would go here...
            
            # Leaf nodes for different data sensitivity levels
            "highly_sensitive_data": DecisionNode(
                id="highly_sensitive_data",
                description="Assess third-party risk for highly sensitive data",
                is_leaf=True,
                prompt="",
                action="assess_highly_sensitive_data_sharing",
                response_template="Regarding sharing highly sensitive data with third parties:\n\n{CONTEXT}\n\nBased on best practices, you should implement these controls:\n\n1. Comprehensive vendor security assessment before sharing\n2. Strong contractual requirements including data protection addendum\n3. Right to audit and penetration testing of vendor environments\n4. Data encryption both in transit and at rest\n5. Vendor access limited to specific data needed only\n6. Detailed data flow mapping and impact assessment\n7. Regular compliance attestations (e.g., SOC 2 Type 2, ISO 27001)\n8. Continuous monitoring of vendor security posture\n\nRecommended additional measures:\n- Data loss prevention controls\n- Digital rights management for sensitive files\n- Vendor personnel background checks\n- Regular security review meetings with vendor"
            ),
            "sensitive_data": DecisionNode(
                id="sensitive_data",
                description="Assess third-party risk for sensitive data",
                is_leaf=True,
                prompt="",
                action="assess_sensitive_data_sharing",
                response_template="Regarding sharing sensitive data with third parties:\n\n{CONTEXT}\n\nBased on best practices, you should implement these controls:\n\n1. Vendor security assessment before sharing\n2. Contractual requirements for data protection\n3. Data encryption in transit and at rest\n4. Access controls based on least privilege\n5. Vendor security review and attestation\n6. Data classification and handling requirements\n7. Incident response planning for vendor breaches\n\nImplementation guidance:\n- Conduct annual vendor security reviews\n- Implement secure file transfer methods\n- Maintain a data inventory with vendor access mapping\n- Test vendor incident response procedures"
            ),
            # More leaf nodes would go here...
            
            "general_third_party_risk": DecisionNode(
                id="general_third_party_risk",
                description="Provide general third-party risk guidance",
                is_leaf=True,
                prompt="",
                action="provide_general_third_party_risk_guidance",
                response_template="Regarding third-party risk management:\n\n{CONTEXT}\n\nBased on best practices, here is a general approach:\n\n1. Develop a comprehensive vendor risk management program\n2. Categorize vendors based on risk (critical, high, medium, low)\n3. Conduct appropriate due diligence for each risk tier\n4. Implement appropriate contractual protections\n5. Establish ongoing monitoring processes\n6. Develop incident response procedures for vendor incidents\n7. Regularly review and update your vendor inventory\n\nKey considerations:\n- Ensure adequate resources for vendor assessment\n- Use standardized assessment questionnaires when possible\n- Consider using third-party risk management platforms\n- Align vendor controls with your internal requirements\n- Document risk acceptance decisions for exceptions"
            )
        }
    )
    
    # Add more nodes as needed...
    
    return tree

# Initialize and register cybersecurity decision trees
def initialize_cybersecurity_trees():
    """Initialize and register cybersecurity decision trees"""
    # Create and register each tree
    register_tree(create_cybersecurity_incident_response_tree())
    register_tree(create_security_control_assessment_tree())
    register_tree(create_third_party_risk_tree())
    
    logger.info("Initialized cybersecurity decision trees")

# Initialize the trees when the module is imported
initialize_cybersecurity_trees() 