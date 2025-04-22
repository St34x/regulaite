"""
Test Cases for Parsing Quality Evaluation
=========================================

This file defines test cases and expected outputs for evaluating 
agent responses in the RegulAIte system.

Each test case includes:
- A query to be sent to the agent
- A category for the query
- Expected content elements that should be present in a high-quality response
- Potentially problematic or incorrect content elements to watch for
"""

from typing import Dict, List, Any

# Test cases with expected outputs for each category
TEST_CASES = {
    "regulatory": [
        {
            "query": "What are our key compliance obligations based on our internal policies?",
            "expected_elements": [
                "Reference to specific policies",
                "Policy names or identifiers",
                "Compliance obligations",
                "Responsibility assignments"
            ],
            "negative_elements": [
                "Vague generalizations without policy references",
                "Generic compliance advice not tied to internal documentation",
                "Missing key regulatory requirements mentioned in policies"
            ]
        },
        {
            "query": "How does our data protection policy align with GDPR requirements?",
            "expected_elements": [
                "Specific GDPR principles",
                "Data subject rights",
                "Policy controls for GDPR compliance",
                "Internal processes for data handling"
            ],
            "negative_elements": [
                "Incorrect GDPR principles or requirements",
                "Missing key GDPR articles referenced in policies",
                "Generic advice not tied to internal policies"
            ]
        },
        {
            "query": "What controls should we implement to meet ISO 27001 requirements?",
            "expected_elements": [
                "Specific ISO 27001 controls",
                "Control implementation status",
                "Gap analysis",
                "Reference to internal controls"
            ],
            "negative_elements": [
                "Controls not mentioned in documentation",
                "Incorrect mapping of controls to ISO requirements",
                "Missing key control domains"
            ]
        },
        {
            "query": "Explain how our organization handles risk assessment according to internal policies",
            "expected_elements": [
                "Risk assessment methodology",
                "Risk scoring approach",
                "Assessment frequency",
                "Risk treatment options"
            ],
            "negative_elements": [
                "Generic risk advice not tied to policies",
                "Incorrect risk methodology",
                "Missing risk treatment processes"
            ]
        },
        {
            "query": "What are the consequences of non-compliance with our security policies?",
            "expected_elements": [
                "Disciplinary actions",
                "Escalation procedures",
                "Reporting mechanisms",
                "Consequences for different violation types"
            ],
            "negative_elements": [
                "Incorrect penalty information",
                "Missing key consequence types",
                "Generic advice not tied to internal policies"
            ]
        }
    ],
    "risk_assessment": [
        {
            "query": "What vulnerabilities were identified in the EBIOS RM report?",
            "expected_elements": [
                "Named vulnerabilities from EBIOS report",
                "Vulnerability categories",
                "Severity ratings",
                "Affected systems or assets"
            ],
            "negative_elements": [
                "Vulnerabilities not mentioned in EBIOS",
                "Incorrect severity ratings",
                "Generic vulnerability descriptions not from documentation"
            ]
        },
        {
            "query": "What are the highest risk threats in our environment according to our documentation?",
            "expected_elements": [
                "Named high-risk threats",
                "Risk ratings or scores",
                "Threat sources",
                "Potential impacts"
            ],
            "negative_elements": [
                "Threats not mentioned in documentation",
                "Incorrect risk ratings",
                "Missing key high-risk threats"
            ]
        },
        {
            "query": "How do we address supply chain risks according to our policies?",
            "expected_elements": [
                "Supplier assessment processes",
                "Due diligence procedures",
                "Contractual requirements",
                "Monitoring mechanisms"
            ],
            "negative_elements": [
                "Procedures not mentioned in policies",
                "Generic supply chain advice",
                "Missing key supplier controls"
            ]
        },
        {
            "query": "What is our risk acceptance threshold according to documentation?",
            "expected_elements": [
                "Risk acceptance criteria",
                "Approval thresholds",
                "Risk level definitions",
                "Approval authorities"
            ],
            "negative_elements": [
                "Incorrect thresholds",
                "Missing approval processes",
                "Generic risk advice"
            ]
        },
        {
            "query": "What mitigation strategies are recommended for the top threats?",
            "expected_elements": [
                "Specific mitigation controls",
                "Control effectiveness",
                "Implementation status",
                "Residual risk levels"
            ],
            "negative_elements": [
                "Generic controls not from documentation",
                "Missing key mitigations",
                "Incorrect control mappings"
            ]
        }
    ],
    "threat_modeling": [
        {
            "query": "What are the key attack vectors identified in our documents?",
            "expected_elements": [
                "Named attack vectors",
                "Attack surfaces",
                "Threat actors",
                "Attack complexity ratings"
            ],
            "negative_elements": [
                "Attack vectors not in documentation",
                "Generic attack descriptions",
                "Missing key attack paths"
            ]
        },
        {
            "query": "How do we prioritize security threats according to our methodology?",
            "expected_elements": [
                "Threat prioritization criteria",
                "Impact assessment",
                "Likelihood assessment",
                "Prioritization framework"
            ],
            "negative_elements": [
                "Incorrect prioritization method",
                "Missing key prioritization factors",
                "Generic advice not from methodology"
            ]
        },
        {
            "query": "What countermeasures are suggested for phishing attacks?",
            "expected_elements": [
                "Specific anti-phishing controls",
                "Training requirements",
                "Technical controls",
                "Incident response procedures"
            ],
            "negative_elements": [
                "Controls not in documentation",
                "Generic phishing advice",
                "Missing key countermeasures"
            ]
        },
        {
            "query": "Explain our defense-in-depth strategy based on documentation",
            "expected_elements": [
                "Multiple security layers",
                "Complementary controls",
                "Layer-specific measures",
                "Control redundancy approach"
            ],
            "negative_elements": [
                "Incorrect layering approach",
                "Missing key security layers",
                "Generic security advice"
            ]
        },
        {
            "query": "How do we assess emerging threats according to our framework?",
            "expected_elements": [
                "Threat intelligence sources",
                "Assessment frequency",
                "Integration with risk process",
                "Emerging threat criteria"
            ],
            "negative_elements": [
                "Processes not in documentation",
                "Generic threat intelligence advice",
                "Missing key assessment steps"
            ]
        }
    ],
    "compliance_mapping": [
        {
            "query": "Map our internal security controls to NIST CSF framework",
            "expected_elements": [
                "NIST CSF functions",
                "Control mappings",
                "Coverage analysis",
                "Gap identification"
            ],
            "negative_elements": [
                "Incorrect NIST categories",
                "Controls not in documentation",
                "Missing core functions"
            ]
        },
        {
            "query": "How do our policies align with ISO 27001 controls?",
            "expected_elements": [
                "Policy to control mappings",
                "Implementation status",
                "Coverage analysis",
                "Gap identification"
            ],
            "negative_elements": [
                "Incorrect ISO controls",
                "Policies not in documentation",
                "Missing key control domains"
            ]
        },
        {
            "query": "What gaps exist between our controls and regulatory requirements?",
            "expected_elements": [
                "Identified gaps",
                "Missing controls",
                "Partial implementations",
                "Remediation priorities"
            ],
            "negative_elements": [
                "Gaps not supported by documentation",
                "Missing major gap areas",
                "Generic compliance advice"
            ]
        },
        {
            "query": "Which compliance requirements are addressed by our data backup policy?",
            "expected_elements": [
                "Specific regulations",
                "Policy controls",
                "Backup requirements",
                "Retention requirements"
            ],
            "negative_elements": [
                "Requirements not in documentation",
                "Incorrect regulatory mappings",
                "Missing key backup controls"
            ]
        },
        {
            "query": "Create a mapping between our access control policy and PCI DSS requirements",
            "expected_elements": [
                "PCI DSS requirements",
                "Policy mappings",
                "Control implementation",
                "Gap analysis"
            ],
            "negative_elements": [
                "Incorrect PCI requirements",
                "Controls not in documentation",
                "Missing key requirement areas"
            ]
        }
    ]
}

# Evaluation criteria weights - these can be adjusted to emphasize different aspects
EVALUATION_CRITERIA = {
    "relevance": {
        "weight": 0.25,
        "description": "How relevant the response is to the query"
    },
    "accuracy": {
        "weight": 0.30,
        "description": "Factual correctness based on the source documents"
    },
    "completeness": {
        "weight": 0.20,
        "description": "How comprehensive the response is"
    },
    "coherence": {
        "weight": 0.15,
        "description": "How well-structured and logical the response is"
    },
    "actionability": {
        "weight": 0.10,
        "description": "How practical and actionable the information is"
    }
}

# Factors to consider in parsing quality
PARSING_QUALITY_FACTORS = [
    "Document structure preservation",
    "Header/section identification",
    "Table extraction accuracy",
    "List formatting preservation",
    "Image/diagram text extraction",
    "Footnote handling",
    "Cross-reference preservation",
    "Special character handling",
    "Formula/equation extraction",
    "Multi-column layout handling"
]

def get_test_cases_for_category(category: str) -> List[Dict[str, Any]]:
    """Get test cases for a specific category"""
    return TEST_CASES.get(category, [])

def get_all_test_cases() -> Dict[str, List[Dict[str, Any]]]:
    """Get all test cases"""
    return TEST_CASES

def get_evaluation_criteria() -> Dict[str, Dict[str, Any]]:
    """Get evaluation criteria with weights"""
    return EVALUATION_CRITERIA 