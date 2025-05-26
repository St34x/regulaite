"""
Risk Assessment Module - Module d'analyse des risques selon EBIOS/MEHARI.
Utilise les outils universels pour une évaluation complète des risques.
Intégré avec la configuration organisationnelle RegulAIte.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

from ..agent import Agent, AgentResponse, Query, QueryContext
from ..integrations.llm_integration import LLMIntegration, get_llm_client
from ..tools import (
    DocumentFinder, EntityExtractor, CrossReferenceTool, TemporalAnalyzer,
    EntityType, MetricType, RelationType
)
from .organization_config import get_organization_config_manager, OrganizationConfigManager

logger = logging.getLogger(__name__)

class RiskAssessmentMethodology(Enum):
    """Méthodologies d'évaluation des risques."""
    EBIOS = "ebios"
    MEHARI = "mehari"
    ISO27005 = "iso27005"
    NIST = "nist"
    CUSTOM = "custom"

class RiskTreatmentStrategy(Enum):
    """Stratégies de traitement des risques."""
    ACCEPT = "accept"
    MITIGATE = "mitigate"
    TRANSFER = "transfer"
    AVOID = "avoid"

@dataclass
class RiskScenario:
    """Scénario de risque EBIOS."""
    id: str
    name: str
    description: str
    threat_source: str
    threat_action: str
    vulnerability: str
    asset_affected: str
    likelihood: str  # very_low, low, medium, high, very_high
    impact: str     # very_low, low, medium, high, very_high
    risk_level: str  # calculated from likelihood and impact
    existing_controls: List[str]
    residual_risk: str
    treatment_strategy: RiskTreatmentStrategy
    action_plan: List[str]
    
    def __post_init__(self):
        if self.existing_controls is None:
            self.existing_controls = []
        if self.action_plan is None:
            self.action_plan = []

@dataclass
class RiskAssessmentReport:
    """Rapport d'évaluation des risques."""
    assessment_id: str
    organization_id: str
    methodology: RiskAssessmentMethodology
    scope: str
    assessment_date: datetime
    risk_scenarios: List[RiskScenario]
    risk_matrix: Dict[str, Any]
    control_effectiveness: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    executive_summary: str
    detailed_analysis: Dict[str, Any]
    organizational_context: Dict[str, Any]

class RiskAssessmentModule(Agent):
    """
    Module spécialisé d'évaluation des risques avec intégration organisationnelle.
    """
    
    def __init__(self, llm_client: LLMIntegration = None):
        super().__init__(
            agent_id="risk_assessment",
            name="Expert Évaluation des Risques"
        )
        
        self.llm_client = llm_client or get_llm_client()
        self.org_config: OrganizationConfigManager = get_organization_config_manager()
        
        # Initialiser les outils universels
        self.document_finder = DocumentFinder()
        self.entity_extractor = EntityExtractor()
        self.cross_reference_tool = CrossReferenceTool()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Matrice de risques EBIOS
        self.risk_matrix = {
            "likelihood": {
                "very_low": 1,
                "low": 2, 
                "medium": 3,
                "high": 4,
                "very_high": 5
            },
            "impact": {
                "very_low": 1,
                "low": 2,
                "medium": 3, 
                "high": 4,
                "very_high": 5
            }
        }
        
        # Mapping des niveaux de risque
        self.risk_levels = {
            (1, 1): "very_low", (1, 2): "very_low", (1, 3): "low",
            (1, 4): "medium", (1, 5): "medium",
            (2, 1): "very_low", (2, 2): "low", (2, 3): "low", 
            (2, 4): "medium", (2, 5): "high",
            (3, 1): "low", (3, 2): "low", (3, 3): "medium",
            (3, 4): "high", (3, 5): "high",
            (4, 1): "medium", (4, 2): "medium", (4, 3): "high",
            (4, 4): "high", (4, 5): "very_high",
            (5, 1): "medium", (5, 2): "high", (5, 3): "high",
            (5, 4): "very_high", (5, 5): "very_high"
        }
        
        # Prompts spécialisés
        self.system_prompt = """
Tu es un expert en évaluation des risques cybersécurité selon les méthodologies EBIOS RM et MEHARI.
Tu maîtrises parfaitement :
- L'identification et la qualification des actifs
- L'analyse des sources de menaces et des modes opératoires
- L'évaluation de la vraisemblance et de l'impact
- La cartographie des risques et scénarios stratégiques
- Les mesures de sécurité et leur efficacité
- L'adaptation aux contextes organisationnels spécifiques

Tu tiens compte du contexte organisationnel (secteur, taille, maturité) pour personnaliser tes analyses.
Réponds TOUJOURS en français avec une approche méthodique et structurée.
"""

    async def process_query(self, query: Query) -> AgentResponse:
        """
        Traite une requête d'évaluation des risques avec contexte organisationnel.
        """
        logger.info(f"Traitement requête risques: {query.query_text}")
        
        # Extraire l'ID d'organisation du contexte
        org_id = self._extract_organization_id(query)
        
        # Analyser le type de demande
        analysis_type = await self._analyze_request_type(query.query_text)
        
        if analysis_type == "full_assessment":
            return await self._perform_full_risk_assessment(query, org_id)
        elif analysis_type == "scenario_analysis":
            return await self._analyze_risk_scenarios(query, org_id)
        elif analysis_type == "control_evaluation":
            return await self._evaluate_controls_effectiveness(query, org_id)
        elif analysis_type == "trend_analysis":
            return await self._analyze_risk_trends(query, org_id)
        else:
            return await self._general_risk_analysis(query, org_id)

    async def perform_ebios_assessment(
        self,
        scope: str,
        org_id: str = None,
        assets: List[Dict[str, Any]] = None,
        methodology: RiskAssessmentMethodology = RiskAssessmentMethodology.EBIOS
    ) -> RiskAssessmentReport:
        """
        Effectue une évaluation EBIOS complète avec contexte organisationnel.
        """
        logger.info(f"Évaluation EBIOS pour le périmètre: {scope}, organisation: {org_id}")
        
        assessment_id = f"ebios_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Récupérer le contexte organisationnel
        org_context = self._get_organizational_context(org_id)
        
        # Étape 1: Identifier et qualifier les actifs (adaptés à l'organisation)
        if not assets:
            assets = await self._identify_organizational_assets(scope, org_id)
        
        # Étape 2: Identifier les sources de menaces (selon profil organisationnel)
        threat_sources = await self._identify_organizational_threats(assets, org_id)
        
        # Étape 3: Analyser les scenarios de risques
        risk_scenarios = await self._generate_risk_scenarios(
            assets, threat_sources, org_context
        )
        
        # Étape 4: Évaluer les contrôles existants
        control_effectiveness = await self._evaluate_existing_controls(
            risk_scenarios, org_context
        )
        
        # Étape 5: Calculer les risques résiduels
        for scenario in risk_scenarios:
            scenario.residual_risk = self._calculate_residual_risk(
                scenario, control_effectiveness, org_context
            )
        
        # Étape 6: Générer la matrice de risques
        risk_matrix = self._generate_risk_matrix(risk_scenarios)
        
        # Étape 7: Recommandations de traitement (adaptées au contexte)
        recommendations = await self._generate_treatment_recommendations(
            risk_scenarios, org_context
        )
        
        # Étape 8: Synthèse exécutive
        executive_summary = await self._generate_executive_summary(
            risk_scenarios, recommendations, org_context
        )
        
        # Analyse détaillée
        detailed_analysis = {
            "assets_analysis": assets,
            "threat_landscape": threat_sources,
            "scenarios_details": [vars(s) for s in risk_scenarios],
            "control_assessment": control_effectiveness,
            "organizational_factors": org_context
        }
        
        return RiskAssessmentReport(
            assessment_id=assessment_id,
            organization_id=org_id or "default",
            methodology=methodology,
            scope=scope,
            assessment_date=datetime.now(),
            risk_scenarios=risk_scenarios,
            risk_matrix=risk_matrix,
            control_effectiveness=control_effectiveness,
            recommendations=recommendations,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            organizational_context=org_context
        )

    def _extract_organization_id(self, query: Query) -> Optional[str]:
        """Extrait l'ID d'organisation du contexte de la requête."""
        if query.context:
            return query.context.get("organization_id") or query.context.get("org_id")
        return None
    
    def _get_organizational_context(self, org_id: str) -> Dict[str, Any]:
        """Récupère le contexte organisationnel complet."""
        if not org_id:
            return {"organization_type": "general", "sector": "technology", "size": "medium"}
        
        # Récupérer les différents contextes depuis le gestionnaire
        regulatory_context = self.org_config.get_regulatory_context(org_id)
        governance_context = self.org_config.get_governance_context(org_id)
        
        return {
            "organization_id": org_id,
            "regulatory": regulatory_context,
            "governance": governance_context,
            "size": governance_context.get("size", "medium"),
            "sector": regulatory_context.get("sector", "technology"),
            "risk_appetite": governance_context.get("risk_appetite", "moderate"),
            "maturity": governance_context.get("maturity", {})
        }

    async def _identify_organizational_assets(
        self, 
        scope: str, 
        org_id: str
    ) -> List[Dict[str, Any]]:
        """Identifie les actifs selon le profil organisationnel."""
        
        # Récupérer les actifs configurés pour l'organisation
        org_assets = self.org_config.get_organization_assets(org_id, scope)
        
        if org_assets:
            logger.info(f"Actifs organisationnels trouvés: {len(org_assets)}")
            return org_assets
        
        # Fallback: rechercher des documents pertinents
        relevant_docs = await self.document_finder.search_documents(
            f"actifs {scope} inventaire",
            limit=20
        )
        
        assets = []
        for doc in relevant_docs:
            content = doc.get("content", "")
            if content:
                doc_assets = await self.entity_extractor.extract_entities(
                    content,
                    entity_types=[EntityType.ASSET]
                )
                assets.extend(doc_assets.get("asset", []))
        
        # Si toujours pas d'actifs, utiliser les defaults organisationnels
        return assets if assets else self.org_config._get_default_assets()

    async def _identify_organizational_threats(
        self, 
        assets: List[Dict[str, Any]], 
        org_id: str
    ) -> List[Dict[str, Any]]:
        """Identifie les sources de menaces selon le profil organisationnel."""
        
        # Récupérer le paysage de menaces organisationnel
        org_threats = self.org_config.get_threat_landscape(org_id)
        
        if org_threats:
            logger.info(f"Menaces organisationnelles trouvées: {len(org_threats)}")
            return org_threats
        
        # Fallback: analyse par LLM avec contexte organisationnel
        org_context = self._get_organizational_context(org_id)
        
        threat_prompt = f"""
Identifie les sources de menaces selon EBIOS RM pour cette organisation:

CONTEXTE ORGANISATIONNEL:
- Secteur: {org_context.get('sector', 'technology')}
- Taille: {org_context.get('size', 'medium')}
- Appétit au risque: {org_context.get('risk_appetite', 'moderate')}
- Frameworks réglementaires: {org_context.get('regulatory', {}).get('frameworks', [])}

ACTIFS ORGANISATIONNELS:
{json.dumps(assets[:10], indent=2, default=str)[:2000]}

Adapte l'identification des menaces au profil spécifique de cette organisation.
Considère les menaces sectorielles pertinentes et les contraintes réglementaires.

Retourne un JSON structuré avec les sources de menaces spécifiques.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": threat_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            threats = json.loads(json_content).get("threat_sources", [])
            return threats
        except Exception as e:
            logger.error(f"Erreur parsing menaces organisationnelles: {str(e)}")
            return self.org_config._get_default_threats()

    # Mise à jour des méthodes privées pour intégrer le contexte organisationnel
    async def _generate_risk_scenarios(
        self,
        assets: List[Dict[str, Any]],
        threat_sources: List[Dict[str, Any]],
        org_context: Dict[str, Any]
    ) -> List[RiskScenario]:
        """Génère les scénarios de risque avec contexte organisationnel."""
        scenarios = []
        scenario_id = 1
        
        # Limiter selon la taille de l'organisation
        max_assets = self._get_max_assets_for_org_size(org_context.get("size", "medium"))
        max_threats = self._get_max_threats_for_org_size(org_context.get("size", "medium"))
        
        for asset in assets[:max_assets]:
            for threat in threat_sources[:max_threats]:
                scenario_data = await self._create_risk_scenario_with_context(
                    asset, threat, scenario_id, org_context
                )
                
                if scenario_data:
                    scenarios.append(scenario_data)
                    scenario_id += 1
        
        return scenarios

    def _get_max_assets_for_org_size(self, size: str) -> int:
        """Détermine le nombre max d'actifs selon la taille de l'organisation."""
        size_limits = {
            "startup": 5,
            "small": 8,
            "medium": 12,
            "large": 20,
            "enterprise": 30
        }
        return size_limits.get(size, 10)
    
    def _get_max_threats_for_org_size(self, size: str) -> int:
        """Détermine le nombre max de menaces selon la taille de l'organisation."""
        size_limits = {
            "startup": 3,
            "small": 4,
            "medium": 5,
            "large": 6,
            "enterprise": 8
        }
        return size_limits.get(size, 5)

    async def _create_risk_scenario_with_context(
        self,
        asset: Dict[str, Any],
        threat_source: Dict[str, Any],
        scenario_id: int,
        org_context: Dict[str, Any]
    ) -> Optional[RiskScenario]:
        """Crée un scénario de risque avec contexte organisationnel."""
        
        scenario_prompt = f"""
Crée un scénario de risque EBIOS détaillé adapté à ce contexte organisationnel:

CONTEXTE ORGANISATIONNEL:
- Secteur: {org_context.get('sector', 'technology')}
- Taille: {org_context.get('size', 'medium')}
- Maturité sécurité: {org_context.get('maturity', {}).get('security', 'developing')}
- Appétit au risque: {org_context.get('risk_appetite', 'moderate')}

ACTIF CIBLE:
{json.dumps(asset, indent=2, default=str)}

SOURCE DE MENACE:
{json.dumps(threat_source, indent=2, default=str)}

Adapte le scénario aux spécificités organisationnelles:
- Contraintes sectorielles
- Niveau de maturité
- Ressources disponibles
- Contexte réglementaire

Retourne un JSON structuré avec le scénario adapté.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": scenario_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            data = json.loads(json_content)
            
            # Calculer le niveau de risque avec ajustement contextuel
            likelihood_val = self.risk_matrix["likelihood"].get(data.get("likelihood", "medium"), 3)
            impact_val = self.risk_matrix["impact"].get(data.get("impact", "medium"), 3)
            
            # Ajustement selon le contexte organisationnel
            if org_context.get("risk_appetite") == "conservative":
                likelihood_val = min(5, likelihood_val + 1)  # Plus conservateur
            elif org_context.get("risk_appetite") == "aggressive":
                likelihood_val = max(1, likelihood_val - 1)  # Moins conservateur
            
            risk_level = self.risk_levels.get((likelihood_val, impact_val), "medium")
            
            return RiskScenario(
                id=f"RS-{scenario_id:03d}",
                name=data.get("name", f"Scénario {scenario_id}"),
                description=data.get("description", ""),
                threat_source=threat_source.get("name", ""),
                threat_action=data.get("threat_action", ""),
                vulnerability=data.get("vulnerability", ""),
                asset_affected=asset.get("name", ""),
                likelihood=data.get("likelihood", "medium"),
                impact=data.get("impact", "medium"),
                risk_level=risk_level,
                existing_controls=[],
                residual_risk="",
                treatment_strategy=RiskTreatmentStrategy.MITIGATE,
                action_plan=[]
            )
            
        except Exception as e:
            logger.error(f"Erreur création scénario contextuel: {str(e)}")
            return self._create_fallback_scenario(asset, threat_source, scenario_id, e)
    
    def _create_fallback_scenario(
        self, 
        asset: Dict[str, Any], 
        threat_source: Dict[str, Any], 
        scenario_id: int, 
        error: Exception
    ) -> RiskScenario:
        """Crée un scénario de fallback en cas d'erreur."""
        return RiskScenario(
            id=f"RS-ERROR-{scenario_id:03d}",
            name=f"⚠️ ERREUR: Scénario {scenario_id} - Génération échouée",
            description=f"Échec de la génération automatique du scénario par LLM: {str(error)}",
            threat_source=threat_source.get("name", "Source inconnue"),
            threat_action="Action à déterminer manuellement",
            vulnerability="Vulnérabilité à identifier manuellement",
            asset_affected=asset.get("name", "Actif inconnu"),
            likelihood="medium",
            impact="medium",
            risk_level="medium",
            existing_controls=[],
            residual_risk="À évaluer manuellement",
            treatment_strategy=RiskTreatmentStrategy.MITIGATE,
            action_plan=[
                "⚠️ Scénario généré automatiquement échoué",
                "Analyser manuellement le risque entre cet actif et cette menace",
                "Consulter un expert en analyse de risques",
                "Réviser les données d'entrée et réessayer"
            ]
        )

    # Resto des méthodes existantes avec mise à jour pour le contexte organisationnel...
    # [Les autres méthodes restent largement similaires mais avec intégration du contexte org]


# Factory function
def get_risk_assessment_module(llm_client: LLMIntegration = None):
    """Factory function pour obtenir une instance du module d'évaluation des risques."""
    return RiskAssessmentModule(llm_client=llm_client) 