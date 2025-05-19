"""
Predefined decision trees for different use cases.
"""
import logging
from typing import Dict, List, Any, Optional
from .tree_reasoning import DecisionNode, DecisionTree, create_default_decision_tree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Registry of available trees
_TREE_REGISTRY = {}

def register_tree(tree: DecisionTree) -> None:
    """
    Register a decision tree in the registry.
    
    Args:
        tree: DecisionTree to register
    """
    _TREE_REGISTRY[tree.id] = tree
    logger.info(f"Registered decision tree: {tree.id} ({tree.name})")

def get_tree(tree_id: str) -> Optional[DecisionTree]:
    """
    Get a decision tree by ID.
    
    Args:
        tree_id: ID of the tree to get
        
    Returns:
        DecisionTree instance or None if not found
    """
    return _TREE_REGISTRY.get(tree_id)

def get_available_trees() -> Dict[str, Dict[str, Any]]:
    """
    Get a list of all available trees.
    
    Returns:
        Dictionary of tree IDs to tree metadata
    """
    return {
        tree_id: {
            "id": tree.id,
            "name": tree.name,
            "description": tree.description,
            "version": tree.version
        }
        for tree_id, tree in _TREE_REGISTRY.items()
    }

def create_regulatory_compliance_tree() -> DecisionTree:
    """
    Create a decision tree focused on regulatory compliance.
    
    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="regulatory_compliance",
        name="Regulatory Compliance Analysis",
        description="Analyzes queries related to regulatory compliance and provides appropriate responses",
        root_node_id="root",
        nodes={
            "root": DecisionNode(
                id="root",
                description="Determine the specific compliance domain",
                prompt="Determine which compliance domain this query relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "data_privacy": "data_privacy",
                    "financial": "financial_compliance",
                    "healthcare": "healthcare_compliance",
                    "environmental": "environmental_compliance",
                    "general": "general_compliance"
                }
            ),
            "data_privacy": DecisionNode(
                id="data_privacy",
                description="Handle data privacy compliance",
                prompt="Determine which data privacy regulation this relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "gdpr": "gdpr_compliance",
                    "ccpa": "ccpa_compliance",
                    "other": "other_privacy_compliance"
                }
            ),
            "financial_compliance": DecisionNode(
                id="financial_compliance",
                description="Handle financial compliance",
                prompt="Determine which financial regulation this relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "aml": "aml_compliance",
                    "tax": "tax_compliance",
                    "reporting": "financial_reporting"
                }
            ),
            "healthcare_compliance": DecisionNode(
                id="healthcare_compliance",
                description="Handle healthcare compliance",
                prompt="Determine which healthcare regulation this relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "hipaa": "hipaa_compliance",
                    "other": "other_healthcare_compliance"
                }
            ),
            "environmental_compliance": DecisionNode(
                id="environmental_compliance",
                description="Handle environmental compliance",
                is_leaf=True,
                prompt="",
                action="provide_environmental_compliance_information",
                response_template="Regarding your environmental compliance query:\\n\\n{context}\\n\\nThis information should help you understand the environmental compliance requirements."
            ),
            "general_compliance": DecisionNode(
                id="general_compliance",
                description="Handle general compliance",
                is_leaf=True,
                prompt="",
                action="provide_general_compliance_information",
                response_template="Regarding your compliance query:\\n\\n{context}\\n\\nThis information should help you understand your compliance obligations."
            ),
            "gdpr_compliance": DecisionNode(
                id="gdpr_compliance",
                description="Provide GDPR compliance information",
                is_leaf=True,
                prompt="",
                action="provide_gdpr_compliance_information",
                response_template="Regarding your GDPR compliance query:\\n\\n{context}\\n\\nThis information should help you understand your GDPR compliance requirements."
            ),
            "ccpa_compliance": DecisionNode(
                id="ccpa_compliance",
                description="Provide CCPA compliance information",
                is_leaf=True,
                prompt="",
                action="provide_ccpa_compliance_information",
                response_template="Regarding your CCPA compliance query:\\n\\n{context}\\n\\nThis information should help you understand your CCPA compliance requirements."
            ),
            "other_privacy_compliance": DecisionNode(
                id="other_privacy_compliance",
                description="Provide other privacy compliance information",
                is_leaf=True,
                prompt="",
                action="provide_other_privacy_compliance_information",
                response_template="Regarding your privacy compliance query:\\n\\n{context}\\n\\nThis information should help you understand the privacy compliance requirements."
            ),
            "aml_compliance": DecisionNode(
                id="aml_compliance",
                description="Provide AML compliance information",
                is_leaf=True,
                prompt="",
                action="provide_aml_compliance_information",
                response_template="Regarding your anti-money laundering compliance query:\\n\\n{context}\\n\\nThis information should help you understand your AML compliance requirements."
            ),
            "tax_compliance": DecisionNode(
                id="tax_compliance",
                description="Provide tax compliance information",
                is_leaf=True,
                prompt="",
                action="provide_tax_compliance_information",
                response_template="Regarding your tax compliance query:\\n\\n{context}\\n\\nThis information should help you understand your tax compliance requirements."
            ),
            "financial_reporting": DecisionNode(
                id="financial_reporting",
                description="Provide financial reporting compliance information",
                is_leaf=True,
                prompt="",
                action="provide_financial_reporting_information",
                response_template="Regarding your financial reporting query:\\n\\n{context}\\n\\nThis information should help you understand your financial reporting requirements."
            ),
            "hipaa_compliance": DecisionNode(
                id="hipaa_compliance",
                description="Provide HIPAA compliance information",
                is_leaf=True,
                prompt="",
                action="provide_hipaa_compliance_information",
                response_template="Regarding your HIPAA compliance query:\\n\\n{context}\\n\\nThis information should help you understand your HIPAA compliance requirements."
            ),
            "other_healthcare_compliance": DecisionNode(
                id="other_healthcare_compliance",
                description="Provide other healthcare compliance information",
                is_leaf=True,
                prompt="",
                action="provide_other_healthcare_compliance_information",
                response_template="Regarding your healthcare compliance query:\\n\\n{context}\\n\\nThis information should help you understand the healthcare compliance requirements."
            )
        }
    )
    
    return tree

def create_risk_management_tree() -> DecisionTree:
    """
    Create a decision tree focused on risk management.
    
    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="risk_management",
        name="Risk Management Analysis",
        description="Analyzes queries related to risk management and provides appropriate responses",
        root_node_id="root",
        nodes={
            "root": DecisionNode(
                id="root",
                description="Determine the specific risk management domain",
                prompt="Determine which risk management area this query relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "identification": "risk_identification",
                    "assessment": "risk_assessment",
                    "mitigation": "risk_mitigation",
                    "monitoring": "risk_monitoring",
                    "general": "general_risk_management"
                }
            ),
            "risk_identification": DecisionNode(
                id="risk_identification",
                description="Handle risk identification",
                prompt="Determine which type of risk identification this relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "processes": "process_risk_identification",
                    "threats": "threat_identification",
                    "other": "other_risk_identification"
                }
            ),
            "risk_assessment": DecisionNode(
                id="risk_assessment",
                description="Handle risk assessment",
                prompt="Determine which type of risk assessment this relates to:\\n\\n{query}\\n\\n{context}",
                children={
                    "quantitative": "quantitative_assessment",
                    "qualitative": "qualitative_assessment",
                    "other": "other_risk_assessment"
                }
            ),
            "risk_mitigation": DecisionNode(
                id="risk_mitigation",
                description="Handle risk mitigation",
                is_leaf=True,
                prompt="",
                action="provide_risk_mitigation_strategies",
                response_template="For your risk mitigation query:\\n\\n{context}\\n\\nThese strategies should help you mitigate the identified risks."
            ),
            "risk_monitoring": DecisionNode(
                id="risk_monitoring",
                description="Handle risk monitoring",
                is_leaf=True,
                prompt="",
                action="provide_risk_monitoring_information",
                response_template="Regarding your risk monitoring query:\\n\\n{context}\\n\\nThis information should help you establish effective risk monitoring procedures."
            ),
            "general_risk_management": DecisionNode(
                id="general_risk_management",
                description="Handle general risk management",
                is_leaf=True,
                prompt="",
                action="provide_general_risk_management_information",
                response_template="Regarding your risk management query:\\n\\n{context}\\n\\nThis information should help you understand general risk management principles."
            ),
            "process_risk_identification": DecisionNode(
                id="process_risk_identification",
                description="Provide process risk identification information",
                is_leaf=True,
                prompt="",
                action="provide_process_risk_identification_information",
                response_template="For identifying process risks:\\n\\n{context}\\n\\nThis should help you identify risks in your processes."
            ),
            "threat_identification": DecisionNode(
                id="threat_identification",
                description="Provide threat identification information",
                is_leaf=True,
                prompt="",
                action="provide_threat_identification_information",
                response_template="For identifying threats:\\n\\n{context}\\n\\nThis should help you identify potential threats to your organization."
            ),
            "other_risk_identification": DecisionNode(
                id="other_risk_identification",
                description="Provide other risk identification information",
                is_leaf=True,
                prompt="",
                action="provide_other_risk_identification_information",
                response_template="For identifying risks:\\n\\n{context}\\n\\nThis should help you with your risk identification process."
            ),
            "quantitative_assessment": DecisionNode(
                id="quantitative_assessment",
                description="Provide quantitative risk assessment information",
                is_leaf=True,
                prompt="",
                action="provide_quantitative_assessment_information",
                response_template="For quantitative risk assessment:\\n\\n{context}\\n\\nThis should help you with your quantitative risk assessment process."
            ),
            "qualitative_assessment": DecisionNode(
                id="qualitative_assessment",
                description="Provide qualitative risk assessment information",
                is_leaf=True,
                prompt="",
                action="provide_qualitative_assessment_information",
                response_template="For qualitative risk assessment:\\n\\n{context}\\n\\nThis should help you with your qualitative risk assessment process."
            ),
            "other_risk_assessment": DecisionNode(
                id="other_risk_assessment",
                description="Provide other risk assessment information",
                is_leaf=True,
                prompt="",
                action="provide_other_risk_assessment_information",
                response_template="For risk assessment:\\n\\n{context}\\n\\nThis should help you with your risk assessment process."
            )
        }
    )
    
    return tree

def create_default_understanding_tree() -> DecisionTree:
    """
    Create a decision tree focused on general query understanding and routing.
    
    Returns:
        DecisionTree instance
    """
    tree = DecisionTree(
        id="default_understanding",
        name="Default Query Understanding",
        description="Understands the general nature of a query and routes to basic actions or information retrieval.",
        version="1.0.0",
        root_node_id="root_understand",
        nodes={
            "root_understand": DecisionNode(
                id="root_understand",
                description="Initial understanding of the query intent.",
                prompt="What is the primary intent of this query? (e.g., information, task, question):\\n\\n{query}",
                children={
                    "information_seeking": "info_retrieval_node",
                    "simple_question": "simple_answer_node",
                    "complex_task": "task_delegation_node" 
                }
            ),
            "info_retrieval_node": DecisionNode(
                id="info_retrieval_node",
                description="Retrieve information using RAG.",
                is_leaf=True,
                prompt="", # Leaf nodes typically don't need a prompt if they have a direct action
                action="retrieve_rag_context",
                response_template="I found the following information related to your query:\\n\\n{context}"
            ),
            "simple_answer_node": DecisionNode(
                id="simple_answer_node",
                description="Provide a direct answer to a simple question.",
                is_leaf=True,
                prompt="",
                action="generate_direct_answer",
                response_template="Here is an answer to your question:\\n\\n{LLM_RESPONSE}"
            ),
            "task_delegation_node": DecisionNode(
                id="task_delegation_node",
                description="Delegate a complex task to another component or explain inability.",
                is_leaf=True,
                prompt="",
                action="delegate_or_clarify_task",
                response_template="This seems like a complex task. I will try to process it or ask for clarification if needed."
            )
        }
    )
    return tree

# Initialize the registry with predefined trees
def initialize_registry():
    """Initialize the tree registry with all predefined trees."""
    register_tree(create_default_decision_tree()) # Register the actual default tree
    register_tree(create_regulatory_compliance_tree())
    register_tree(create_risk_management_tree())
    register_tree(create_default_understanding_tree()) # Register the new tree
    
    # Example: Add more trees here if needed
    # from .cybersecurity_trees import initialize_cybersecurity_trees
    # initialize_cybersecurity_trees() # If it handles its own registration via register_tree

    logger.info(f"Initialized tree registry with {len(_TREE_REGISTRY)} trees")

# Call this at module load time to populate the registry
initialize_registry() 