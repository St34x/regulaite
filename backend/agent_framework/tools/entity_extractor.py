"""
Entity Extractor - Outil d'extraction et de catégorisation d'entités GRC.
Extrait les contrôles, risques, observations, exigences et actifs des documents.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime

from ..integrations.llm_integration import get_llm_client

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Types d'entités GRC."""
    CONTROL = "control"
    RISK = "risk"
    FINDING = "finding"
    REQUIREMENT = "requirement"
    ASSET = "asset"
    POLICY = "policy"
    PROCEDURE = "procedure"

class ControlType(Enum):
    """Types de contrôles."""
    TECHNICAL = "technical"
    ADMINISTRATIVE = "administrative"
    PHYSICAL = "physical"
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"

class RiskLevel(Enum):
    """Niveaux de risque."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

@dataclass
class Control:
    """Entité Contrôle."""
    id: str
    name: str
    description: str
    control_type: ControlType
    framework: Optional[str] = None
    effectiveness: Optional[str] = None
    implementation_status: Optional[str] = None
    owner: Optional[str] = None
    frequency: Optional[str] = None
    evidence_required: List[str] = None
    related_risks: List[str] = None
    
    def __post_init__(self):
        if self.evidence_required is None:
            self.evidence_required = []
        if self.related_risks is None:
            self.related_risks = []

@dataclass
class Risk:
    """Entité Risque."""
    id: str
    name: str
    description: str
    category: Optional[str] = None
    likelihood: Optional[RiskLevel] = None
    impact: Optional[RiskLevel] = None
    overall_risk: Optional[RiskLevel] = None
    owner: Optional[str] = None
    status: Optional[str] = None
    treatment_strategy: Optional[str] = None
    controls: List[str] = None
    affected_assets: List[str] = None
    
    def __post_init__(self):
        if self.controls is None:
            self.controls = []
        if self.affected_assets is None:
            self.affected_assets = []

@dataclass
class Finding:
    """Entité Observation/Vulnérabilité."""
    id: str
    title: str
    description: str
    severity: Optional[str] = None
    cvss_score: Optional[float] = None
    category: Optional[str] = None
    affected_assets: List[str] = None
    remediation: Optional[str] = None
    status: Optional[str] = None
    discovered_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.affected_assets is None:
            self.affected_assets = []

@dataclass
class Requirement:
    """Entité Exigence."""
    id: str
    framework: str
    clause: str
    description: str
    applicability: Optional[str] = None
    implementation_guidance: Optional[str] = None
    controls: List[str] = None
    evidence_required: List[str] = None
    
    def __post_init__(self):
        if self.controls is None:
            self.controls = []
        if self.evidence_required is None:
            self.evidence_required = []

@dataclass
class Asset:
    """Entité Actif."""
    id: str
    name: str
    asset_type: str
    criticality: Optional[str] = None
    owner: Optional[str] = None
    location: Optional[str] = None
    value: Optional[str] = None
    data_classification: Optional[str] = None
    risks: List[str] = None
    controls: List[str] = None
    
    def __post_init__(self):
        if self.risks is None:
            self.risks = []
        if self.controls is None:
            self.controls = []

class EntityExtractor:
    """
    Extracteur d'entités GRC spécialisé.
    """
    
    def __init__(self):
        self.llm_client = get_llm_client()
        
        # Patterns de reconnaissance pour extraction rapide
        self.control_patterns = [
            r"contrôle\s+([A-Z0-9\.-]+)",
            r"control\s+([A-Z0-9\.-]+)",
            r"mesure\s+de\s+sécurité\s+([A-Z0-9\.-]+)",
            r"([A-Z]{2,}\.[0-9]+(?:\.[0-9]+)*)",  # ISO patterns
        ]
        
        self.risk_patterns = [
            r"risque\s+([A-Z0-9\.-]+)",
            r"risk\s+([A-Z0-9\.-]+)",
            r"R-([0-9]+)",
            r"risque\s+de\s+([^\.]+)",
        ]
        
        self.finding_patterns = [
            r"vulnérabilité\s+([A-Z0-9\.-]+)",
            r"CVE-([0-9]{4}-[0-9]+)",
            r"observation\s+([A-Z0-9\.-]+)",
            r"finding\s+([A-Z0-9\.-]+)",
        ]
        
        # Framework mappings
        self.framework_mappings = {
            "ISO27001": {
                "A.5": "Politiques de sécurité",
                "A.6": "Organisation de la sécurité",
                "A.7": "Sécurité des ressources humaines",
                "A.8": "Gestion des actifs",
                "A.9": "Contrôle d'accès",
                "A.10": "Cryptographie",
                "A.11": "Sécurité physique",
                "A.12": "Sécurité d'exploitation",
                "A.13": "Sécurité des communications",
                "A.14": "Acquisition, développement et maintenance",
                "A.15": "Relations avec les fournisseurs",
                "A.16": "Gestion des incidents",
                "A.17": "Continuité d'activité",
                "A.18": "Conformité"
            },
            "RGPD": {
                "Art.5": "Principes de traitement",
                "Art.6": "Licéité du traitement",
                "Art.25": "Protection des données dès la conception",
                "Art.32": "Sécurité du traitement",
                "Art.33": "Notification de violation",
                "Art.35": "Analyse d'impact"
            }
        }

    async def extract_entities(
        self, 
        content: str, 
        entity_types: List[EntityType] = None,
        framework_context: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extrait les entités GRC d'un contenu.
        
        Args:
            content: Contenu à analyser
            entity_types: Types d'entités à extraire
            framework_context: Contexte de framework (ISO27001, RGPD, etc.)
            
        Returns:
            Dictionnaire des entités extraites par type
        """
        if entity_types is None:
            entity_types = list(EntityType)
            
        logger.info(f"Extraction d'entités: {[e.value for e in entity_types]}")
        
        # 1. Extraction rapide par patterns
        pattern_results = self._extract_by_patterns(content, entity_types)
        
        # 2. Extraction avancée par LLM
        llm_results = await self._extract_by_llm(
            content, entity_types, framework_context
        )
        
        # 3. Fusion et validation des résultats
        final_results = self._merge_extraction_results(
            pattern_results, llm_results, entity_types
        )
        
        # 4. Enrichissement avec contexte framework
        if framework_context:
            final_results = self._enrich_with_framework_context(
                final_results, framework_context
            )
        
        return final_results

    async def extract_controls(
        self, 
        content: str, 
        framework: str = None
    ) -> List[Control]:
        """
        Extrait spécifiquement les contrôles d'un document.
        """
        logger.info(f"Extraction de contrôles (framework: {framework})")
        
        prompt = f"""
Extrait tous les contrôles de sécurité du texte suivant.

FRAMEWORK CONTEXTE: {framework or 'Général'}

TEXTE:
{content[:4000]}

Retourne un JSON avec une liste de contrôles:
{{
    "controls": [
        {{
            "id": "identifiant_controle",
            "name": "nom_du_controle",
            "description": "description_detaillee",
            "control_type": "technical|administrative|physical",
            "framework": "{framework or ''}",
            "effectiveness": "effective|ineffective|partially_effective",
            "implementation_status": "implemented|planned|not_implemented",
            "owner": "responsable",
            "frequency": "continuous|monthly|quarterly|annually",
            "evidence_required": ["liste", "des", "preuves"],
            "related_risks": ["risque1", "risque2"]
        }}
    ]
}}
"""

        try:
            response = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4.1",
                temperature=0.1
            )
            
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            
            data = json.loads(json_content)
            
            controls = []
            for ctrl_data in data.get("controls", []):
                control = Control(
                    id=ctrl_data.get("id", ""),
                    name=ctrl_data.get("name", ""),
                    description=ctrl_data.get("description", ""),
                    control_type=ControlType(ctrl_data.get("control_type", "administrative")),
                    framework=ctrl_data.get("framework"),
                    effectiveness=ctrl_data.get("effectiveness"),
                    implementation_status=ctrl_data.get("implementation_status"),
                    owner=ctrl_data.get("owner"),
                    frequency=ctrl_data.get("frequency"),
                    evidence_required=ctrl_data.get("evidence_required", []),
                    related_risks=ctrl_data.get("related_risks", [])
                )
                controls.append(control)
            
            logger.info(f"Extracted {len(controls)} controls")
            return controls
            
        except Exception as e:
            logger.error(f"Erreur extraction contrôles: {str(e)}")
            return []

    async def extract_risks(self, content: str) -> List[Risk]:
        """
        Extrait spécifiquement les risques d'un document.
        """
        logger.info("Extraction de risques")
        
        prompt = f"""
Extrait tous les risques du texte suivant.

TEXTE:
{content[:4000]}

Retourne un JSON avec une liste de risques:
{{
    "risks": [
        {{
            "id": "identifiant_risque",
            "name": "nom_du_risque",
            "description": "description_detaillee",
            "category": "operational|technical|legal|strategic",
            "likelihood": "very_low|low|medium|high|very_high|critical",
            "impact": "very_low|low|medium|high|very_high|critical",
            "overall_risk": "very_low|low|medium|high|very_high|critical",
            "owner": "responsable",
            "status": "identified|assessed|treated|monitored",
            "treatment_strategy": "accept|mitigate|transfer|avoid",
            "controls": ["controle1", "controle2"],
            "affected_assets": ["actif1", "actif2"]
        }}
    ]
}}
"""

        try:
            response = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4.1",
                temperature=0.1
            )
            
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            
            data = json.loads(json_content)
            
            risks = []
            for risk_data in data.get("risks", []):
                risk = Risk(
                    id=risk_data.get("id", ""),
                    name=risk_data.get("name", ""),
                    description=risk_data.get("description", ""),
                    category=risk_data.get("category"),
                    likelihood=RiskLevel(risk_data.get("likelihood", "medium")) if risk_data.get("likelihood") else None,
                    impact=RiskLevel(risk_data.get("impact", "medium")) if risk_data.get("impact") else None,
                    overall_risk=RiskLevel(risk_data.get("overall_risk", "medium")) if risk_data.get("overall_risk") else None,
                    owner=risk_data.get("owner"),
                    status=risk_data.get("status"),
                    treatment_strategy=risk_data.get("treatment_strategy"),
                    controls=risk_data.get("controls", []),
                    affected_assets=risk_data.get("affected_assets", [])
                )
                risks.append(risk)
            
            logger.info(f"Extracted {len(risks)} risks")
            return risks
            
        except Exception as e:
            logger.error(f"Erreur extraction risques: {str(e)}")
            return []

    def _extract_by_patterns(
        self, 
        content: str, 
        entity_types: List[EntityType]
    ) -> Dict[str, List[str]]:
        """
        Extraction rapide par patterns regex.
        """
        results = {}
        
        for entity_type in entity_types:
            results[entity_type.value] = []
            
            if entity_type == EntityType.CONTROL:
                for pattern in self.control_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    results[entity_type.value].extend(matches)
                    
            elif entity_type == EntityType.RISK:
                for pattern in self.risk_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    results[entity_type.value].extend(matches)
                    
            elif entity_type == EntityType.FINDING:
                for pattern in self.finding_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    results[entity_type.value].extend(matches)
        
        return results

    async def _extract_by_llm(
        self, 
        content: str, 
        entity_types: List[EntityType],
        framework_context: str = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extraction avancée par LLM.
        """
        entity_types_str = ", ".join([e.value for e in entity_types])
        
        prompt = f"""
Extrait les entités GRC suivantes du texte: {entity_types_str}

CONTEXTE FRAMEWORK: {framework_context or 'Général'}

TEXTE:
{content[:3000]}

Retourne un JSON structuré avec les entités trouvées.
"""

        try:
            response = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4.1",
                temperature=0.1
            )
            
            # Parse JSON response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            
            return json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Erreur extraction LLM: {str(e)}")
            return {}

    def _merge_extraction_results(
        self,
        pattern_results: Dict[str, List[str]],
        llm_results: Dict[str, List[Dict[str, Any]]],
        entity_types: List[EntityType]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fusionne les résultats d'extraction pattern et LLM.
        """
        merged = {}
        
        for entity_type in entity_types:
            type_name = entity_type.value
            merged[type_name] = []
            
            # Add LLM results
            if type_name in llm_results:
                merged[type_name].extend(llm_results[type_name])
            
            # Add pattern results as simple entities
            if type_name in pattern_results:
                for pattern_match in pattern_results[type_name]:
                    simple_entity = {
                        "id": pattern_match,
                        "name": pattern_match,
                        "description": f"Entité détectée par pattern: {pattern_match}",
                        "source": "pattern_recognition"
                    }
                    merged[type_name].append(simple_entity)
        
        return merged

    def _enrich_with_framework_context(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        framework: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enrichit les résultats avec le contexte du framework.
        """
        if framework in self.framework_mappings:
            framework_map = self.framework_mappings[framework]
            
            for entity_type, entities in results.items():
                for entity in entities:
                    entity_id = entity.get("id", "")
                    
                    # Check if entity ID matches framework pattern
                    for framework_id, description in framework_map.items():
                        if framework_id in entity_id:
                            entity["framework"] = framework
                            entity["framework_description"] = description
                            break
        
        return results


# Tool function for agent integration
async def entity_extractor_tool(
    content: str,
    entity_types: List[str] = None,
    framework: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Outil d'extraction d'entités pour les agents.
    """
    extractor = EntityExtractor()
    
    if entity_types:
        types = [EntityType(t) for t in entity_types if t in [e.value for e in EntityType]]
    else:
        types = None
    
    try:
        results = await extractor.extract_entities(
            content=content,
            entity_types=types,
            framework_context=framework
        )
        
        return {
            "success": True,
            "results": results,
            "summary": {
                "total_entities": sum(len(entities) for entities in results.values()),
                "types_found": list(results.keys()),
                "framework_context": framework
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur dans entity_extractor_tool: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": {}
        }


def get_entity_extractor():
    """Factory function pour obtenir une instance d'EntityExtractor."""
    return EntityExtractor() 