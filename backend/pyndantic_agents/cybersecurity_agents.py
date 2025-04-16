"""
Specialized cybersecurity agents for GRC functions.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union
import asyncio
from openai import AsyncOpenAI
import os
import re

from .base_agent import BaseAgent, AgentInput, AgentOutput
from .rag_agent import RAGAgent
from .tree_reasoning import TreeReasoningAgent
from llamaIndex_rag.rag import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VulnerabilityAssessmentOutput(AgentOutput):
    """Enhanced output for vulnerability assessment"""
    cve_ids: List[str] = []
    severity_scores: Dict[str, float] = {}
    affected_systems: List[str] = []
    remediation_steps: List[str] = []

class ComplianceMappingOutput(AgentOutput):
    """Enhanced output for compliance mapping"""
    framework_mappings: Dict[str, List[str]] = {}
    control_gaps: List[str] = []
    implementation_status: Dict[str, str] = {}

class ThreatModelOutput(AgentOutput):
    """Enhanced output for threat modeling"""
    identified_threats: List[Dict[str, Any]] = []
    attack_vectors: List[str] = []
    mitigations: List[Dict[str, Any]] = []

class CybersecurityBaseAgent(BaseAgent):
    """
    Base class for cybersecurity-focused agents with common functionality.
    """
    
    def __init__(
        self, 
        rag_system: RAGSystem, 
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4",
        **kwargs
    ):
        """Initialize cybersecurity base agent"""
        super().__init__(**kwargs)
        self.rag_system = rag_system
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.openai_api_key)
        logger.info(f"Initialized cybersecurity agent with model {self.model}")
    
    async def extract_cybersecurity_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract cybersecurity-specific entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types to lists of entities
        """
        system_prompt = """You are a cybersecurity entity extraction specialist. 
Extract the following types of entities from the provided text:
- vulnerabilities (including CVE IDs if present)
- systems (software, hardware, cloud services)
- threats
- controls
- compliance_frameworks (like ISO27001, NIST CSF, etc.)

Respond ONLY with a JSON object with the following structure:
{
  "vulnerabilities": ["entity1", "entity2"],
  "systems": ["entity1", "entity2"],
  "threats": ["entity1", "entity2"],
  "controls": ["entity1", "entity2"],
  "compliance_frameworks": ["entity1", "entity2"]
}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            entities = json.loads(content)
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting cybersecurity entities: {str(e)}")
            return {
                "vulnerabilities": [],
                "systems": [],
                "threats": [],
                "controls": [],
                "compliance_frameworks": []
            }
    
    async def extract_cve_details(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract CVE details from text.
        
        Args:
            text: Text containing CVE information
            
        Returns:
            List of CVE details
        """
        # First extract CVE IDs using regex
        cve_pattern = r"CVE-\d{4}-\d{4,7}"
        cve_ids = re.findall(cve_pattern, text)
        
        if not cve_ids:
            return []
        
        # Get details about each CVE
        system_prompt = """You are a vulnerability assessment specialist.
For each CVE ID provided, extract the following information from the context:
- CVE ID
- Description
- Severity (if mentioned)
- Affected systems/software
- Potential impact
- Recommended mitigations

Respond ONLY with a JSON array of objects with the following structure:
[
  {
    "cve_id": "CVE-XXXX-XXXXX",
    "description": "...",
    "severity": "...",
    "affected_systems": ["..."],
    "impact": "...",
    "mitigations": ["..."]
  }
]
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CVE IDs found: {', '.join(cve_ids)}\n\nContext: {text}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            cve_details = json.loads(content)
            return cve_details
            
        except Exception as e:
            logger.error(f"Error extracting CVE details: {str(e)}")
            return [{"cve_id": cve_id, "description": "Details unavailable"} for cve_id in cve_ids]

class VulnerabilityAssessmentAgent(CybersecurityBaseAgent):
    """
    Agent for vulnerability assessment and management.
    """
    
    async def process(self, input_data: AgentInput) -> VulnerabilityAssessmentOutput:
        """
        Process a vulnerability assessment request.
        
        Args:
            input_data: Input data with query and context
            
        Returns:
            VulnerabilityAssessmentOutput with vulnerability assessment
        """
        self._log_processing(input_data)
        query = input_data.query
        
        try:
            # Step 1: Get context from RAG system
            rag_agent = RAGAgent(rag_system=self.rag_system, openai_api_key=self.openai_api_key)
            understanding = await rag_agent.understand_query(query)
            
            # Add vulnerability-specific terms to improve search
            if "reformulated_query" in understanding:
                search_query = understanding["reformulated_query"] + " vulnerability CVE security risk"
            else:
                search_query = query + " vulnerability CVE security risk"
            
            # Get context based on enhanced query
            context_results = await rag_agent.retrieve_context(
                query=search_query,
                understanding=understanding,
                top_k=input_data.parameters.get("top_k", 7) if input_data.parameters else 7
            )
            
            # Step 2: Extract CVEs and vulnerability information from context
            all_context_text = "\n\n".join([item.get("text", "") for item in context_results])
            cve_details = await self.extract_cve_details(all_context_text)
            
            # Extract other cybersecurity entities
            cybersec_entities = await self.extract_cybersecurity_entities(all_context_text)
            
            # Step 3: Generate a comprehensive vulnerability assessment
            system_prompt = """You are a vulnerability assessment specialist.
Based on the provided context and identified vulnerabilities, create a comprehensive vulnerability assessment that includes:

1. A summary of identified vulnerabilities
2. Severity assessment for each vulnerability
3. Affected systems or components
4. Potential impact of exploitation
5. Recommended remediation steps in priority order
6. Timeline recommendations for addressing each vulnerability

Your response should be thorough, actionable, and prioritized based on risk.
"""
            
            assessment_context = json.dumps({
                "query": query,
                "cve_details": cve_details,
                "entities": cybersec_entities,
                "context": [item.get("text", "") for item in context_results]
            }, indent=2)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Vulnerability Assessment Request: {query}\n\nAnalysis Context: {assessment_context}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            assessment_text = response.choices[0].message.content
            
            # Step 4: Extract structured data for the output
            cve_ids = [detail.get("cve_id", "") for detail in cve_details if "cve_id" in detail]
            severity_scores = {}
            for detail in cve_details:
                if "cve_id" in detail and "severity" in detail:
                    severity_scores[detail["cve_id"]] = self._parse_severity(detail["severity"])
            
            affected_systems = list(set(sum([detail.get("affected_systems", []) for detail in cve_details], [])))
            
            # Extract remediation steps
            remediation_steps = []
            for detail in cve_details:
                if "mitigations" in detail:
                    remediation_steps.extend(detail["mitigations"])
            
            # Ensure no duplicates in remediation steps
            remediation_steps = list(set(remediation_steps))
            
            return VulnerabilityAssessmentOutput(
                response=assessment_text,
                context_used=context_results,
                confidence=0.8 if cve_details else 0.5,
                reasoning=f"Analyzed {len(cve_details)} CVEs across {len(affected_systems)} systems",
                cve_ids=cve_ids,
                severity_scores=severity_scores,
                affected_systems=affected_systems,
                remediation_steps=remediation_steps,
                additional_data={
                    "cve_details": cve_details,
                    "cybersec_entities": cybersec_entities
                }
            )
            
        except Exception as e:
            logger.error(f"Error in vulnerability assessment: {str(e)}")
            return VulnerabilityAssessmentOutput(
                response="I encountered an error while analyzing vulnerabilities.",
                context_used=[],
                confidence=0.1,
                reasoning="Error during processing",
                cve_ids=[],
                severity_scores={},
                affected_systems=[],
                remediation_steps=[]
            )
    
    def _parse_severity(self, severity_text: str) -> float:
        """
        Parse severity text into a numeric score.
        
        Args:
            severity_text: Text description of severity
            
        Returns:
            Numeric severity score between 0-10
        """
        severity_text = severity_text.lower()
        
        # Check for CVSS score pattern (e.g., "7.5", "9.8/10")
        cvss_pattern = r"(\d+\.\d+)"
        cvss_match = re.search(cvss_pattern, severity_text)
        if cvss_match:
            return float(cvss_match.group(1))
        
        # Map textual ratings to scores
        if "critical" in severity_text:
            return 9.5
        elif "high" in severity_text:
            return 8.0
        elif "medium" in severity_text or "moderate" in severity_text:
            return 5.0
        elif "low" in severity_text:
            return 3.0
        else:
            return 1.0

class ComplianceMappingAgent(CybersecurityBaseAgent):
    """
    Agent for mapping controls across different compliance frameworks.
    """
    
    async def process(self, input_data: AgentInput) -> ComplianceMappingOutput:
        """
        Process a compliance mapping request.
        
        Args:
            input_data: Input data with query and context
            
        Returns:
            ComplianceMappingOutput with compliance mapping
        """
        self._log_processing(input_data)
        query = input_data.query
        
        try:
            # Step 1: Extract the compliance frameworks being mapped
            system_prompt = """You are a compliance mapping specialist.
Extract the compliance frameworks mentioned in the query.
Also determine if the user is asking about specific controls or the entire framework.

Respond ONLY with a JSON object with the following structure:
{
  "frameworks": ["framework1", "framework2"],
  "specific_controls": ["control1", "control2"],
  "mapping_type": "full_framework" or "specific_controls",
  "target_framework": "the framework to map to (if applicable)"
}
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            mapping_request = json.loads(response.choices[0].message.content)
            
            # Step 2: Get context from RAG system
            rag_agent = RAGAgent(rag_system=self.rag_system, openai_api_key=self.openai_api_key)
            
            # Create a search query focusing on compliance frameworks
            frameworks_str = " ".join(mapping_request["frameworks"])
            controls_str = " ".join(mapping_request.get("specific_controls", []))
            
            search_query = f"compliance mapping {frameworks_str} {controls_str}"
            
            context_results = await rag_agent.retrieve_context(
                query=search_query,
                understanding={"domain": "compliance", "query_type": "mapping"},
                top_k=input_data.parameters.get("top_k", 7) if input_data.parameters else 7
            )
            
            # Step 3: Generate a compliance mapping
            context_text = "\n\n".join([item.get("text", "") for item in context_results])
            
            mapping_prompt = """You are a compliance mapping specialist.
Based on the provided context, create a detailed mapping between the compliance frameworks mentioned.

Your mapping should include:
1. A table or structured mapping showing how controls from one framework map to controls in the other framework(s)
2. Identification of any gaps where one framework has controls not covered by the other
3. Implementation guidance for addressing gaps
4. Notes on any differences in implementation requirements

Be specific and cite the control identifiers where possible.
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": mapping_prompt},
                    {"role": "user", "content": f"Compliance Mapping Request: {query}\n\nFrameworks: {frameworks_str}\n\nContext Information:\n{context_text}"}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            mapping_text = response.choices[0].message.content
            
            # Step 4: Extract structured mapping data
            extraction_prompt = """Extract the framework mappings into a structured format.

Respond ONLY with a JSON object with the following structure:
{
  "framework_mappings": {
    "framework1": ["mapped_control1", "mapped_control2"],
    "framework2": ["mapped_control1", "mapped_control2"]
  },
  "control_gaps": ["gap1", "gap2"],
  "implementation_status": {
    "control1": "implemented/partial/not implemented",
    "control2": "implemented/partial/not implemented"
  }
}
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": mapping_text}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            mapping_data = json.loads(response.choices[0].message.content)
            
            return ComplianceMappingOutput(
                response=mapping_text,
                context_used=context_results,
                confidence=0.75,
                reasoning=f"Mapped controls across {len(mapping_data.get('framework_mappings', {}))} frameworks",
                framework_mappings=mapping_data.get("framework_mappings", {}),
                control_gaps=mapping_data.get("control_gaps", []),
                implementation_status=mapping_data.get("implementation_status", {}),
                additional_data={
                    "mapping_request": mapping_request,
                    "raw_mapping_data": mapping_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error in compliance mapping: {str(e)}")
            return ComplianceMappingOutput(
                response="I encountered an error while mapping compliance frameworks.",
                context_used=[],
                confidence=0.1,
                reasoning="Error during processing",
                framework_mappings={},
                control_gaps=[],
                implementation_status={}
            )

class ThreatModelingAgent(CybersecurityBaseAgent):
    """
    Agent for creating threat models for systems and applications.
    """
    
    async def process(self, input_data: AgentInput) -> ThreatModelOutput:
        """
        Process a threat modeling request.
        
        Args:
            input_data: Input data with query and context
            
        Returns:
            ThreatModelOutput with threat model
        """
        self._log_processing(input_data)
        query = input_data.query
        
        try:
            # Step 1: Extract system information and determine what to model
            system_prompt = """You are a threat modeling specialist.
Extract information about the system or application to be threat modeled.

Respond ONLY with a JSON object with the following structure:
{
  "system_name": "name of the system",
  "system_type": "web app/network/cloud/etc.",
  "components": ["component1", "component2"],
  "data_assets": ["asset1", "asset2"],
  "existing_controls": ["control1", "control2"],
  "modeling_scope": "scope of the modeling exercise"
}
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            system_info = json.loads(response.choices[0].message.content)
            
            # Step 2: Get context from RAG system
            rag_agent = RAGAgent(rag_system=self.rag_system, openai_api_key=self.openai_api_key)
            
            # Create search query based on system information
            system_type = system_info.get("system_type", "")
            components = " ".join(system_info.get("components", []))
            
            search_query = f"threat model {system_type} {components} security vulnerabilities attack vectors"
            
            context_results = await rag_agent.retrieve_context(
                query=search_query,
                understanding={"domain": "threat modeling", "query_type": "analysis"},
                top_k=input_data.parameters.get("top_k", 7) if input_data.parameters else 7
            )
            
            # Step 3: Generate a comprehensive threat model
            context_text = "\n\n".join([item.get("text", "") for item in context_results])
            
            model_prompt = """You are a threat modeling specialist using the STRIDE methodology.
Based on the provided system information and context, create a comprehensive threat model that includes:

1. System overview and trust boundaries
2. Identification of threats categorized by STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege)
3. Attack vectors and scenarios for each threat
4. Impact and likelihood assessment
5. Mitigations and security controls for each identified threat
6. Prioritization of threats based on risk

Your threat model should be actionable, realistic, and tailored to the specific system and its components.
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": model_prompt},
                    {"role": "user", "content": f"Threat Modeling Request: {query}\n\nSystem Information: {json.dumps(system_info, indent=2)}\n\nContext Information:\n{context_text}"}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            threat_model_text = response.choices[0].message.content
            
            # Step 4: Extract structured threat data
            extraction_prompt = """Extract the threats and mitigations into a structured format.

Respond ONLY with a JSON object with the following structure:
{
  "identified_threats": [
    {
      "category": "STRIDE category",
      "description": "threat description",
      "likelihood": "high/medium/low",
      "impact": "high/medium/low"
    }
  ],
  "attack_vectors": ["vector1", "vector2"],
  "mitigations": [
    {
      "threat_id": "index of the threat in the identified_threats array",
      "description": "mitigation description",
      "priority": "high/medium/low"
    }
  ]
}
"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": threat_model_text}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            threat_data = json.loads(response.choices[0].message.content)
            
            return ThreatModelOutput(
                response=threat_model_text,
                context_used=context_results,
                confidence=0.8,
                reasoning=f"Created threat model with {len(threat_data.get('identified_threats', []))} threats and {len(threat_data.get('mitigations', []))} mitigations",
                identified_threats=threat_data.get("identified_threats", []),
                attack_vectors=threat_data.get("attack_vectors", []),
                mitigations=threat_data.get("mitigations", []),
                additional_data={
                    "system_info": system_info,
                    "raw_threat_data": threat_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error in threat modeling: {str(e)}")
            return ThreatModelOutput(
                response="I encountered an error while creating the threat model.",
                context_used=[],
                confidence=0.1,
                reasoning="Error during processing",
                identified_threats=[],
                attack_vectors=[],
                mitigations=[]
            ) 