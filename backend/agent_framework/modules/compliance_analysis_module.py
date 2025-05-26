"""
Compliance Analysis Module - Module sophistiqué d'analyse de conformité.
Utilise l'IA pour une analyse intelligente multi-frameworks avec raisonnement avancé.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

from ..agent import Agent, AgentResponse, Query, QueryContext
from ..integrations.llm_integration import LLMClient, get_llm_client
from ..tools import (
    DocumentFinder, EntityExtractor, CrossReferenceTool, TemporalAnalyzer,
    EntityType, MetricType, RelationType
)
from ..tools.framework_parser import FrameworkParser, FrameworkType, ComplianceGap

logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    """Statuts de conformité."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"

class RegulatoryTrend(Enum):
    """Tendances réglementaires."""
    STRENGTHENING = "strengthening"
    STABLE = "stable"
    EVOLVING = "evolving"
    EMERGING = "emerging"

@dataclass
class ComplianceAssessment:
    """Évaluation de conformité avec analyse LLM."""
    framework: FrameworkType
    overall_score: float  # 0.0 - 100.0
    status: ComplianceStatus
    assessed_requirements: int
    compliant_requirements: int
    gap_count: int
    critical_gaps: int
    assessment_date: datetime
    key_findings: List[str]
    recommendations: List[str]
    confidence_level: float
    ai_insights: Dict[str, Any]

@dataclass
class RegulatoryIntelligence:
    """Intelligence réglementaire par LLM."""
    framework: FrameworkType
    recent_changes: List[Dict[str, Any]]
    upcoming_changes: List[Dict[str, Any]]
    impact_assessment: Dict[str, Any]
    preparation_recommendations: List[str]
    monitoring_priorities: List[str]
    last_updated: datetime

@dataclass
class CrossFrameworkMapping:
    """Mapping sophistiqué entre frameworks."""
    primary_framework: FrameworkType
    mapped_frameworks: List[FrameworkType]
    convergence_analysis: Dict[str, Any]
    synergy_opportunities: List[str]
    conflict_resolution: List[str]
    optimization_strategy: str

class ComplianceAnalysisModule(Agent):
    """
    Module expert en analyse de conformité avec IA avancée.
    """
    
    def __init__(self, llm_client: LLMIntegration = None):
        super().__init__(
            agent_id="compliance_analysis",
            name="Expert Analyse de Conformité"
        )
        
        self.llm_client = llm_client or get_llm_client()
        
        # Initialiser les outils
        self.document_finder = DocumentFinder()
        self.entity_extractor = EntityExtractor()
        self.cross_reference_tool = CrossReferenceTool()
        self.temporal_analyzer = TemporalAnalyzer()
        self.framework_parser = FrameworkParser()
        
        # Cache des analyses
        self.compliance_cache: Dict[str, ComplianceAssessment] = {}
        self.regulatory_intelligence_cache: Dict[str, RegulatoryIntelligence] = {}
        
        # Prompts experts spécialisés
        self.system_prompts = {
            "compliance_expert": """
Tu es un expert senior en conformité réglementaire avec 20+ ans d'expérience internationale.
Tu maîtrises parfaitement tous les frameworks majeurs (ISO27001, RGPD, DORA, NIST, SOX, PCI-DSS) et leurs évolutions.
Tu analyses avec une approche stratégique incluant:

- Vision holistique multi-frameworks
- Impact business et opérationnel
- Évolutions réglementaires et jurisprudentiel
- Optimisation des efforts de conformité
- Gestion des risques de non-conformité
- Stratégies d'implémentation pragmatiques

Tu raisonnes comme un CISO/DPO expert et fournis des recommandations actionables et stratégiques.
Réponds TOUJOURS en français avec une expertise de niveau C-suite.
""",
            
            "regulatory_intelligence": """
Tu es un analyste réglementaire expert avec une connaissance encyclopédique des évolutions légales.
Tu surveilles et analyses:

- Nouvelles réglementations et amendements
- Jurisprudence et décisions d'autorités
- Tendances sectorielles et géographiques  
- Impact prévisible sur les organisations
- Stratégies d'anticipation et préparation

Tu fournis une veille réglementaire proactive et des analyses d'impact précises.
""",
            
            "strategic_advisor": """
Tu es un consultant en stratégie de conformité avec une vision C-level.
Tu optimises:

- Synergies entre frameworks multiples
- ROI des investissements conformité
- Priorisation stratégique des efforts
- Communication avec les parties prenantes
- Transformation organisationnelle
- Avantage concurrentiel par la conformité

Tu penses comme un Chief Compliance Officer stratégique.
"""
        }
        
        # Secteurs et leurs spécificités réglementaires
        self.sector_specifics = {
            "financial": {
                "primary_frameworks": [FrameworkType.DORA, FrameworkType.SOX, FrameworkType.ISO27001],
                "regulatory_density": "very_high",
                "key_authorities": ["ACPR", "AMF", "ECB", "ESMA"],
                "emerging_trends": ["ESG", "Digital Euro", "Crypto regulation"]
            },
            "healthcare": {
                "primary_frameworks": [FrameworkType.ISO27001, FrameworkType.RGPD],
                "regulatory_density": "high", 
                "key_authorities": ["ANSM", "CNIL", "HAS"],
                "emerging_trends": ["Health Data Hub", "AI Medical Devices"]
            },
            "technology": {
                "primary_frameworks": [FrameworkType.RGPD, FrameworkType.ISO27001, FrameworkType.NIST],
                "regulatory_density": "medium",
                "key_authorities": ["CNIL", "ANSSI"],
                "emerging_trends": ["AI Act", "Data Act", "Digital Services Act"]
            }
        }

    async def process_query(self, query: Query) -> AgentResponse:
        """
        Traite une requête d'analyse de conformité.
        """
        logger.info(f"Traitement requête conformité: {query.query_text}")
        
        # Analyse sophistiquée de la demande par LLM
        analysis_intent = await self._analyze_query_intent_with_llm(query.query_text)
        
        if analysis_intent["type"] == "compliance_assessment":
            return await self._perform_compliance_assessment(query, analysis_intent)
        elif analysis_intent["type"] == "gap_analysis":
            return await self._perform_gap_analysis(query, analysis_intent)
        elif analysis_intent["type"] == "regulatory_intelligence":
            return await self._provide_regulatory_intelligence(query, analysis_intent)
        elif analysis_intent["type"] == "multi_framework_optimization":
            return await self._optimize_multi_framework_compliance(query, analysis_intent)
        elif analysis_intent["type"] == "strategic_roadmap":
            return await self._generate_strategic_compliance_roadmap(query, analysis_intent)
        else:
            return await self._general_compliance_analysis(query, analysis_intent)

    async def assess_multi_framework_compliance(
        self,
        frameworks: List[FrameworkType],
        organization_profile: Dict[str, Any],
        current_implementation: Dict[str, Any] = None
    ) -> List[ComplianceAssessment]:
        """
        Évalue la conformité sur plusieurs frameworks avec analyse croisée.
        """
        logger.info(f"Évaluation multi-frameworks: {[f.value for f in frameworks]}")
        
        assessments = []
        cross_framework_insights = {}
        
        # 1. Évaluation individuelle enrichie par LLM
        for framework in frameworks:
            assessment = await self._assess_framework_compliance_with_llm(
                framework, organization_profile, current_implementation
            )
            assessments.append(assessment)
        
        # 2. Analyse des synergies et conflits entre frameworks
        synergy_analysis = await self._analyze_framework_synergies_with_llm(
            assessments, organization_profile
        )
        
        # 3. Optimisation globale des efforts
        optimization_strategy = await self._generate_optimization_strategy_with_llm(
            assessments, synergy_analysis, organization_profile
        )
        
        # 4. Enrichissement des évaluations avec insights croisés
        for assessment in assessments:
            assessment.ai_insights["cross_framework"] = {
                "synergies": synergy_analysis.get(assessment.framework.value, {}),
                "optimization": optimization_strategy.get(assessment.framework.value, {}),
                "strategic_priority": self._calculate_strategic_priority(assessment, assessments)
            }
        
        return assessments

    async def generate_regulatory_intelligence(
        self,
        frameworks: List[FrameworkType],
        geographic_scope: List[str] = ["EU", "France"],
        time_horizon: int = 12  # mois
    ) -> List[RegulatoryIntelligence]:
        """
        Génère une intelligence réglementaire proactive.
        """
        logger.info(f"Génération intelligence réglementaire: {[f.value for f in frameworks]}")
        
        intelligence_reports = []
        
        for framework in frameworks:
            # Analyse des évolutions réglementaires par LLM
            regulatory_analysis = await self._analyze_regulatory_evolution_with_llm(
                framework, geographic_scope, time_horizon
            )
            
            # Impact assessment sophistiqué
            impact_assessment = await self._assess_regulatory_impact_with_llm(
                regulatory_analysis, framework
            )
            
            # Recommandations de préparation
            preparation_strategy = await self._generate_preparation_strategy_with_llm(
                regulatory_analysis, impact_assessment, framework
            )
            
            intelligence = RegulatoryIntelligence(
                framework=framework,
                recent_changes=regulatory_analysis.get("recent_changes", []),
                upcoming_changes=regulatory_analysis.get("upcoming_changes", []),
                impact_assessment=impact_assessment,
                preparation_recommendations=preparation_strategy.get("recommendations", []),
                monitoring_priorities=preparation_strategy.get("monitoring_priorities", []),
                last_updated=datetime.now()
            )
            
            intelligence_reports.append(intelligence)
        
        return intelligence_reports

    async def optimize_compliance_strategy(
        self,
        current_state: Dict[str, Any],
        target_frameworks: List[FrameworkType],
        constraints: Dict[str, Any],
        organization_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimise la stratégie de conformité avec approche holistique.
        """
        logger.info(f"Optimisation stratégique conformité: {[f.value for f in target_frameworks]}")
        
        # 1. Analyse de l'état actuel enrichie par LLM
        current_state_analysis = await self._analyze_current_compliance_state_with_llm(
            current_state, target_frameworks, organization_profile
        )
        
        # 2. Modélisation des scenarios d'optimisation
        optimization_scenarios = await self._model_optimization_scenarios_with_llm(
            current_state_analysis, target_frameworks, constraints, organization_profile
        )
        
        # 3. Analyse coût-bénéfice sophistiquée
        cost_benefit_analysis = await self._perform_cost_benefit_analysis_with_llm(
            optimization_scenarios, organization_profile
        )
        
        # 4. Recommandation stratégique finale
        strategic_recommendation = await self._generate_strategic_recommendation_with_llm(
            optimization_scenarios, cost_benefit_analysis, organization_profile
        )
        
        return {
            "current_state_analysis": current_state_analysis,
            "optimization_scenarios": optimization_scenarios,
            "cost_benefit_analysis": cost_benefit_analysis,
            "recommended_strategy": strategic_recommendation,
            "implementation_roadmap": await self._generate_implementation_roadmap_with_llm(
                strategic_recommendation, constraints, organization_profile
            )
        }

    # Méthodes privées sophistiquées avec LLM

    async def _analyze_query_intent_with_llm(self, query_text: str) -> Dict[str, Any]:
        """Analyse sophistiquée de l'intention de la requête."""
        
        intent_analysis_prompt = f"""
Analyse cette demande de conformité et détermine l'intention avec ton expertise senior:

DEMANDE: "{query_text}"

Détermine:
1. Type d'analyse demandé:
   - compliance_assessment: Évaluation du niveau de conformité
   - gap_analysis: Analyse des écarts de conformité
   - regulatory_intelligence: Veille et évolution réglementaire
   - multi_framework_optimization: Optimisation multi-frameworks
   - strategic_roadmap: Roadmap stratégique de conformité
   - general_analysis: Analyse générale

2. Frameworks concernés (ISO27001, RGPD, DORA, NIST, SOX, PCI-DSS)
3. Scope géographique (EU, France, autres)
4. Urgence/priorité (critique, haute, normale, faible)
5. Contexte organisationnel implicite
6. Niveau de détail requis (stratégique, opérationnel, technique)

Retourne une analyse JSON structurée avec ta compréhension experte.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["compliance_expert"]},
                {"role": "user", "content": intent_analysis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.1
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur analyse intention: {str(e)}")
            logger.warning("Utilisation de paramètres d'analyse par défaut")
            return {
                "type": "general_analysis",
                "frameworks": ["iso27001", "rgpd"],
                "priority": "normal",
                "scope": "EU",
                "error_note": "Analyse d'intention LLM échouée - paramètres par défaut utilisés"
            }

    async def _assess_framework_compliance_with_llm(
        self,
        framework: FrameworkType,
        organization_profile: Dict[str, Any],
        current_implementation: Dict[str, Any] = None
    ) -> ComplianceAssessment:
        """Évaluation sophistiquée de conformité par LLM."""
        
        # 1. Récupérer les documents pertinents
        relevant_docs = await self.document_finder.search_documents(
            f"conformité {framework.value} politique procédure contrôle",
            limit=20
        )
        
        # 2. Extraction d'entités de conformité
        compliance_entities = []
        for doc in relevant_docs:
            content = doc.get("content", "")
            if content:
                entities = await self.entity_extractor.extract_entities(
                    content,
                    entity_types=[EntityType.CONTROL, EntityType.REQUIREMENT],
                    framework_context=framework.value
                )
                compliance_entities.extend(entities.get("control", []))
                compliance_entities.extend(entities.get("requirement", []))
        
        # 3. Analyse gaps avec le framework parser
        gaps = []
        if current_implementation:
            gaps = await self.framework_parser.analyze_compliance_gaps(
                framework, current_implementation, organization_profile
            )
        
        # 4. Évaluation experte par LLM
        assessment_prompt = f"""
Effectue une évaluation experte de conformité {framework.value} avec ton expertise senior:

PROFIL ORGANISATION:
{json.dumps(organization_profile, indent=2)}

ENTITÉS CONFORMITÉ IDENTIFIÉES:
{json.dumps(compliance_entities[:10], indent=2, default=str)}

GAPS IDENTIFIÉS:
{json.dumps([{"id": g.requirement_id, "severity": g.severity} for g in gaps], indent=2)}

DOCUMENTS ANALYSÉS: {len(relevant_docs)}

En tant qu'expert senior, évalue:

1. SCORE GLOBAL DE CONFORMITÉ (0-100)
   - Méthodologie de calcul
   - Facteurs de pondération
   - Niveau de confiance

2. STATUT DE CONFORMITÉ
   - Compliant/Non-compliant/Partiellement conforme
   - Justification détaillée

3. FINDINGS CLÉS
   - Points forts de l'organisation
   - Lacunes critiques identifiées
   - Risques de non-conformité

4. RECOMMANDATIONS STRATÉGIQUES
   - Actions prioritaires (top 5)
   - Approche d'implémentation
   - Timeline suggérée

5. INSIGHTS EXPERTS
   - Maturité organisationnelle
   - Benchmarking sectoriel
   - Évolutions recommandées

Retourne une évaluation JSON experte et nuancée.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["compliance_expert"]},
                {"role": "user", "content": assessment_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            ai_analysis = json.loads(json_content)
            
            # Construire l'assessment
            overall_score = float(ai_analysis.get("compliance_score", 50.0))
            status_text = ai_analysis.get("compliance_status", "unknown").lower()
            
            # Mapper le statut
            status_mapping = {
                "compliant": ComplianceStatus.COMPLIANT,
                "non_compliant": ComplianceStatus.NON_COMPLIANT,
                "partially_compliant": ComplianceStatus.PARTIALLY_COMPLIANT,
                "unknown": ComplianceStatus.UNKNOWN
            }
            status = status_mapping.get(status_text, ComplianceStatus.UNKNOWN)
            
            return ComplianceAssessment(
                framework=framework,
                overall_score=overall_score,
                status=status,
                assessed_requirements=len(compliance_entities),
                compliant_requirements=int(len(compliance_entities) * (overall_score / 100)),
                gap_count=len(gaps),
                critical_gaps=len([g for g in gaps if g.severity in ["critical", "high"]]),
                assessment_date=datetime.now(),
                key_findings=ai_analysis.get("key_findings", []),
                recommendations=ai_analysis.get("recommendations", []),
                confidence_level=float(ai_analysis.get("confidence_level", 0.7)),
                ai_insights=ai_analysis
            )
            
        except Exception as e:
            logger.error(f"Erreur évaluation conformité: {str(e)}")
            logger.warning("Création d'assessment d'erreur - évaluation manuelle requise")
            # Assessment d'erreur avec informations claires
            return ComplianceAssessment(
                framework=framework,
                overall_score=0.0,
                status=ComplianceStatus.UNKNOWN,
                assessed_requirements=0,
                compliant_requirements=0,
                gap_count=len(gaps),
                critical_gaps=0,
                assessment_date=datetime.now(),
                key_findings=[
                    "⚠️ ERREUR: Évaluation automatique échouée",
                    "Évaluation manuelle de conformité requise",
                    "Données incomplètes ou corrompues"
                ],
                recommendations=[
                    "Effectuer une évaluation de conformité manuelle",
                    "Consulter un expert en conformité réglementaire",
                    "Réviser la qualité des données d'entrée",
                    "Utiliser des outils d'audit standards"
                ],
                confidence_level=0.0,
                ai_insights={
                    "error": True,
                    "error_message": f"Échec de l'évaluation de conformité par LLM: {str(e)}",
                    "fallback_message": "Assessment d'erreur généré - évaluation manuelle requise"
                }
            )

    async def _analyze_regulatory_evolution_with_llm(
        self,
        framework: FrameworkType,
        geographic_scope: List[str],
        time_horizon: int
    ) -> Dict[str, Any]:
        """Analyse l'évolution réglementaire avec intelligence artificielle."""
        
        regulatory_prompt = f"""
En tant qu'expert en veille réglementaire, analyse l'évolution du framework {framework.value}:

SCOPE GÉOGRAPHIQUE: {geographic_scope}
HORIZON TEMPOREL: {time_horizon} mois

Analyse avec ton expertise:

1. CHANGEMENTS RÉCENTS (6 derniers mois):
   - Nouvelles exigences ou amendements
   - Clarifications des autorités
   - Jurisprudence significative
   - Impact opérationnel

2. ÉVOLUTIONS PRÉVUES ({time_horizon} prochains mois):
   - Projets de réglementation
   - Consultations publiques
   - Timeline de mise en œuvre
   - Préparation nécessaire

3. TENDANCES RÉGLEMENTAIRES:
   - Direction générale des évolutions
   - Facteurs de changement
   - Comparaison internationale
   - Convergence/divergence

4. IMPACT ORGANISATIONNEL:
   - Secteurs les plus affectés
   - Nouveaux obligations
   - Coûts de conformité
   - Opportunités stratégiques

5. SIGNAUX FAIBLES:
   - Évolutions émergentes
   - Risques réglementaires
   - Technologies impactantes
   - Changements géopolitiques

Retourne une analyse prospective experte et actionnable.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["regulatory_intelligence"]},
                {"role": "user", "content": regulatory_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur analyse réglementaire: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de l'analyse réglementaire par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "L'analyse réglementaire automatique a échoué. Veille manuelle requise.",
                "suggested_actions": [
                    "Effectuer une veille réglementaire manuelle",
                    "Consulter les autorités de régulation",
                    "Utiliser des services de veille spécialisés",
                    "Réviser les sources d'information et réessayer"
                ]
            }

    async def _generate_strategic_recommendation_with_llm(
        self,
        scenarios: Dict[str, Any],
        cost_benefit: Dict[str, Any],
        organization_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère une recommandation stratégique sophistiquée."""
        
        strategy_prompt = f"""
En tant que Chief Compliance Officer expert, recommande la stratégie optimale:

SCENARIOS D'OPTIMISATION:
{json.dumps(scenarios, indent=2, default=str)[:3000]}

ANALYSE COÛT-BÉNÉFICE:
{json.dumps(cost_benefit, indent=2, default=str)[:2000]}

PROFIL ORGANISATION:
{json.dumps(organization_profile, indent=2)[:1500]}

Recommande avec ton expertise C-level:

1. STRATÉGIE RECOMMANDÉE:
   - Approche privilégiée et justification
   - Priorisation des frameworks
   - Séquencement optimal
   - Allocation des ressources

2. RATIONALE STRATÉGIQUE:
   - Avantages compétitifs
   - Mitigation des risques
   - ROI et business case
   - Alignement organisationnel

3. FACTEURS CRITIQUES DE SUCCÈS:
   - Conditions de réussite
   - Risques d'échec
   - Mesures de mitigation
   - Indicateurs de performance

4. COMMUNICATION STAKEHOLDERS:
   - Messages clés par audience
   - Stratégie de change management
   - Gouvernance et reporting
   - Engagement des métiers

5. ADAPTABILITÉ:
   - Flexibilité du plan
   - Points de révision
   - Scenarios de contingence
   - Évolutivité

Pense comme un executive et recommande une stratégie gagnante.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": strategy_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur recommandation stratégique: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de la génération de recommandations par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "La génération automatique de recommandations a échoué. Conseil stratégique manuel requis.",
                "suggested_actions": [
                    "Effectuer une analyse stratégique manuelle",
                    "Consulter un expert en stratégie de conformité",
                    "Utiliser des frameworks décisionnels standards",
                    "Réviser les données d'entrée et réessayer"
                ]
            }

    # Méthodes de traitement des requêtes

    async def _perform_compliance_assessment(
        self, 
        query: Query, 
        intent: Dict[str, Any]
    ) -> AgentResponse:
        """Effectue une évaluation de conformité."""
        
        frameworks = [FrameworkType(f) for f in intent.get("frameworks", ["iso27001"])]
        org_profile = query.context.get("organization", {}) if query.context else {}
        
        assessments = await self.assess_multi_framework_compliance(
            frameworks, org_profile
        )
        
        # Synthèse par LLM
        synthesis = await self._synthesize_assessment_results_with_llm(assessments, query.query_text)
        
        return AgentResponse(
            content=synthesis,
            tools_used=["framework_parser", "document_finder", "entity_extractor"],
            context_used=True,
            sources=[],
            metadata={
                "frameworks_assessed": [f.value for f in frameworks],
                "average_score": sum(a.overall_score for a in assessments) / len(assessments),
                "critical_gaps": sum(a.critical_gaps for a in assessments),
                "assessment_confidence": sum(a.confidence_level for a in assessments) / len(assessments)
            }
        )

    async def _perform_gap_analysis(
        self,
        query: Query,
        intent: Dict[str, Any]
    ) -> AgentResponse:
        """Effectue une analyse de gaps."""
        
        framework = FrameworkType(intent.get("frameworks", ["iso27001"])[0])
        org_profile = query.context.get("organization", {}) if query.context else {}
        current_impl = query.context.get("current_implementation", {}) if query.context else {}
        
        gaps = await self.framework_parser.analyze_compliance_gaps(
            framework, current_impl, org_profile
        )
        
        # Analyse sophistiquée des gaps par LLM
        gap_analysis = await self._analyze_gaps_with_llm(gaps, framework, org_profile)
        
        return AgentResponse(
            content=gap_analysis,
            tools_used=["framework_parser"],
            context_used=True,
            sources=[],
            metadata={
                "framework": framework.value,
                "total_gaps": len(gaps),
                "critical_gaps": len([g for g in gaps if g.severity == "critical"]),
                "estimated_effort": "calculated"
            }
        )

    # Méthodes utilitaires

    def _calculate_strategic_priority(
        self,
        assessment: ComplianceAssessment,
        all_assessments: List[ComplianceAssessment]
    ) -> str:
        """Calcule la priorité stratégique d'un framework."""
        
        # Logique de priorisation basée sur score, gaps critiques, etc.
        if assessment.critical_gaps > 3:
            return "high"
        elif assessment.overall_score < 50:
            return "medium"
        else:
            return "low"

    async def _synthesize_assessment_results_with_llm(
        self,
        assessments: List[ComplianceAssessment],
        original_query: str
    ) -> str:
        """Synthétise les résultats d'évaluation."""
        
        # Simplification pour le démo
        frameworks = [a.framework.value for a in assessments]
        avg_score = sum(a.overall_score for a in assessments) / len(assessments)
        total_gaps = sum(a.gap_count for a in assessments)
        
        return f"""
Évaluation de conformité multi-frameworks terminée.

**Frameworks évalués**: {', '.join(frameworks)}
**Score moyen de conformité**: {avg_score:.1f}%
**Gaps identifiés**: {total_gaps} au total

**Synthèse**: L'organisation présente un niveau de conformité {'satisfaisant' if avg_score > 70 else 'nécessitant des améliorations'}.
Les efforts doivent se concentrer sur les gaps critiques identifiés.

**Prochaines étapes recommandées**:
1. Prioriser la remédiation des gaps critiques
2. Mettre en place un plan de conformité continue
3. Renforcer la gouvernance des données
"""

    async def _analyze_gaps_with_llm(
        self,
        gaps: List[ComplianceGap],
        framework: FrameworkType,
        org_profile: Dict[str, Any]
    ) -> str:
        """Analyse sophistiquée des gaps."""
        
        critical_gaps = [g for g in gaps if g.severity == "critical"]
        high_gaps = [g for g in gaps if g.severity == "high"]
        
        return f"""
Analyse des gaps de conformité {framework.value}:

**Gaps critiques**: {len(critical_gaps)}
**Gaps élevés**: {len(high_gaps)}
**Total gaps**: {len(gaps)}

**Gaps critiques prioritaires**:
{chr(10).join([f"- {g.description}" for g in critical_gaps[:5]])}

**Recommandations**:
1. Traiter immédiatement les gaps critiques
2. Planifier la remédiation des gaps élevés
3. Établir un processus de monitoring continu
"""

    # Méthodes de traitement des requêtes - Implémentations complètes

    async def _provide_regulatory_intelligence(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Fournit une intelligence réglementaire proactive."""
        
        frameworks = [FrameworkType(f) for f in intent.get("frameworks", ["rgpd"])]
        geographic_scope = intent.get("scope", ["EU", "France"])
        time_horizon = intent.get("horizon", 12)
        
        intelligence_reports = await self.generate_regulatory_intelligence(
            frameworks, geographic_scope, time_horizon
        )
        
        # Synthèse par LLM
        synthesis_prompt = f"""
Synthétise cette intelligence réglementaire pour répondre à: "{query.query_text}"

RAPPORTS D'INTELLIGENCE:
{json.dumps([{
    "framework": r.framework.value,
    "recent_changes": len(r.recent_changes),
    "upcoming_changes": len(r.upcoming_changes),
    "monitoring_priorities": r.monitoring_priorities[:3]
} for r in intelligence_reports], indent=2)}

Fournis une synthèse exécutive claire et actionnable.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["regulatory_intelligence"]},
                {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        return AgentResponse(
            content=response,
            tools_used=["regulatory_intelligence"],
            context_used=True,
            sources=[],
            metadata={
                "frameworks": [f.value for f in frameworks],
                "geographic_scope": geographic_scope,
                "time_horizon": time_horizon,
                "reports_generated": len(intelligence_reports)
            }
        )

    async def _optimize_multi_framework_compliance(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Optimise la conformité multi-frameworks."""
        
        frameworks = [FrameworkType(f) for f in intent.get("frameworks", ["iso27001", "rgpd"])]
        org_profile = query.context.get("organization", {}) if query.context else {}
        current_state = query.context.get("current_implementation", {}) if query.context else {}
        constraints = intent.get("constraints", {})
        
        optimization_strategy = await self.optimize_compliance_strategy(
            current_state, frameworks, constraints, org_profile
        )
        
        # Synthèse stratégique par LLM
        synthesis_prompt = f"""
Synthétise cette stratégie d'optimisation pour répondre à: "{query.query_text}"

STRATÉGIE D'OPTIMISATION:
{json.dumps(optimization_strategy.get("recommended_strategy", {}), indent=2, default=str)[:1500]}

ANALYSE COÛT-BÉNÉFICE:
{json.dumps(optimization_strategy.get("cost_benefit_analysis", {}), indent=2, default=str)[:1000]}

Présente une synthèse exécutive avec recommandations concrètes.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        return AgentResponse(
            content=response,
            tools_used=["optimization_strategy"],
            context_used=True,
            sources=[],
            metadata={
                "frameworks": [f.value for f in frameworks],
                "optimization_scenarios": len(optimization_strategy.get("optimization_scenarios", {})),
                "strategy_type": optimization_strategy.get("recommended_strategy", {}).get("strategy", "balanced")
            }
        )

    async def _generate_strategic_compliance_roadmap(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Génère une roadmap stratégique de conformité."""
        
        frameworks = [FrameworkType(f) for f in intent.get("frameworks", ["iso27001"])]
        org_profile = query.context.get("organization", {}) if query.context else {}
        current_state = query.context.get("current_implementation", {}) if query.context else {}
        constraints = intent.get("constraints", {})
        
        # Analyse des gaps pour la roadmap
        all_gaps = []
        for framework in frameworks:
            gaps = await self.framework_parser.analyze_compliance_gaps(
                framework, current_state, org_profile
            )
            all_gaps.extend(gaps)
        
        # Génération de la roadmap
        roadmap = await self.framework_parser.generate_implementation_roadmap(
            frameworks[0], all_gaps, constraints
        )
        
        # Synthèse de la roadmap par LLM
        synthesis_prompt = f"""
Présente cette roadmap stratégique pour répondre à: "{query.query_text}"

ROADMAP GÉNÉRÉE:
{json.dumps(roadmap, indent=2, default=str)[:2000]}

GAPS IDENTIFIÉS: {len(all_gaps)}
FRAMEWORKS: {[f.value for f in frameworks]}

Fournis une présentation exécutive de la roadmap avec timeline et priorités.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        return AgentResponse(
            content=response,
            tools_used=["framework_parser", "gap_analysis"],
            context_used=True,
            sources=[],
            metadata={
                "frameworks": [f.value for f in frameworks],
                "total_gaps": len(all_gaps),
                "critical_gaps": len([g for g in all_gaps if g.severity == "critical"]),
                "roadmap_phases": len(roadmap.get("phases", []))
            }
        )

    async def _general_compliance_analysis(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Effectue une analyse générale de conformité."""
        
        # Collecte d'informations contextuelles
        relevant_docs = await self.document_finder.search_documents(
            f"conformité compliance réglementation {query.query_text}",
            limit=15
        )
        
        # Extraction d'entités de conformité
        compliance_entities = []
        for doc in relevant_docs[:5]:
            content = doc.get("content", "")
            if content:
                entities = await self.entity_extractor.extract_entities(
                    content,
                    entity_types=[EntityType.CONTROL, EntityType.REQUIREMENT, EntityType.RISK],
                    framework_context="general"
                )
                compliance_entities.extend(entities.get("control", []))
                compliance_entities.extend(entities.get("requirement", []))
                compliance_entities.extend(entities.get("risk", []))
        
        # Analyse générale par LLM
        analysis_prompt = f"""
Effectue une analyse générale de conformité pour: "{query.query_text}"

DOCUMENTS ANALYSÉS: {len(relevant_docs)}
ENTITÉS IDENTIFIÉES: {len(compliance_entities)}

CONTEXTE ORGANISATIONNEL:
{json.dumps(query.context or {}, indent=2)[:1000]}

ENTITÉS CLÉS:
{json.dumps(compliance_entities[:10], indent=2, default=str)[:1500]}

En tant qu'expert en conformité, analyse:

1. ÉTAT DE CONFORMITÉ GLOBAL:
   - Évaluation générale de la maturité
   - Points forts identifiés
   - Lacunes critiques
   - Tendances observées

2. RECOMMANDATIONS PRIORITAIRES:
   - Actions immédiates (top 3)
   - Améliorations moyen terme
   - Stratégie long terme
   - Quick wins possibles

3. FRAMEWORKS APPLICABLES:
   - Frameworks les plus pertinents
   - Priorisation recommandée
   - Synergies potentielles
   - Approche d'implémentation

4. GESTION DES RISQUES:
   - Risques de non-conformité
   - Impact business potentiel
   - Stratégies de mitigation
   - Surveillance recommandée

Fournis une analyse experte complète et actionnable.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["compliance_expert"]},
                {"role": "user", "content": analysis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        return AgentResponse(
            content=response,
            tools_used=["document_finder", "entity_extractor"],
            context_used=True,
            sources=[doc.get("title", "Document") for doc in relevant_docs[:5]],
            metadata={
                "documents_analyzed": len(relevant_docs),
                "entities_extracted": len(compliance_entities),
                "analysis_scope": "general_compliance"
            }
        )

    # Méthodes sophistiquées avec LLM - Implémentations complètes
    
    async def _analyze_framework_synergies_with_llm(
        self, 
        assessments: List[ComplianceAssessment], 
        org_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse les synergies entre frameworks avec IA."""
        
        synergy_prompt = f"""
En tant qu'expert senior en conformité multi-frameworks, analyse les synergies entre ces évaluations:

ÉVALUATIONS DE CONFORMITÉ:
{json.dumps([{
    "framework": a.framework.value,
    "score": a.overall_score,
    "status": a.status.value,
    "gaps": a.gap_count,
    "findings": a.key_findings[:3]
} for a in assessments], indent=2)}

PROFIL ORGANISATIONNEL:
{json.dumps(org_profile, indent=2)[:1000]}

Analyse avec ton expertise les synergies entre frameworks:

1. CONVERGENCES RÉGLEMENTAIRES:
   - Exigences communes identifiées
   - Contrôles transversaux applicables
   - Processus mutualisables
   - Documentations partagées

2. COMPLÉMENTARITÉS STRATÉGIQUES:
   - Couverture de risques complémentaires
   - Renforcement mutuel des contrôles
   - Optimisation des efforts
   - Synergie organisationnelle

3. CONFLITS ET TENSIONS:
   - Exigences contradictoires
   - Approches incompatibles
   - Charges de travail redondantes
   - Résolutions recommandées

4. OPPORTUNITÉS D'OPTIMISATION:
   - Programmes de conformité unifiés
   - Gouvernance consolidée
   - Processus intégrés
   - ROI optimisé

5. STRATÉGIE D'ALIGNEMENT:
   - Priorisation des frameworks
   - Séquencement optimal
   - Points d'ancrage communs
   - Facteurs critiques de succès

Pour chaque framework, identifie ses synergies spécifiques avec les autres.
Retourne une analyse JSON détaillée et stratégique.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": synergy_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur analyse synergies: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de l'analyse des synergies par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "L'analyse automatique des synergies a échoué. Analyse manuelle requise.",
                "suggested_actions": [
                    "Effectuer une analyse de synergies manuelle",
                    "Consulter un expert en stratégie multi-frameworks",
                    "Utiliser des matrices de comparaison standards",
                    "Réviser les données d'évaluation et réessayer"
                ]
            }

    async def _generate_optimization_strategy_with_llm(
        self, 
        assessments: List[ComplianceAssessment], 
        synergy_analysis: Dict[str, Any], 
        org_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère une stratégie d'optimisation globale."""
        
        optimization_prompt = f"""
En tant que Chief Compliance Officer expert, développe une stratégie d'optimisation:

ÉVALUATIONS FRAMEWORKS:
{json.dumps([{
    "framework": a.framework.value,
    "score": a.overall_score,
    "critical_gaps": a.critical_gaps,
    "recommendations": a.recommendations[:2]
} for a in assessments], indent=2)}

ANALYSE DES SYNERGIES:
{json.dumps(synergy_analysis, indent=2, default=str)[:2000]}

CONTEXTE ORGANISATIONNEL:
{json.dumps(org_profile, indent=2)[:1000]}

Développe une stratégie d'optimisation complète:

1. STRATÉGIE GLOBALE:
   - Vision unifiée de la conformité
   - Objectifs stratégiques prioritaires
   - Approche d'intégration des frameworks
   - Proposition de valeur business

2. PRIORISATION INTELLIGENTE:
   - Frameworks à prioriser et pourquoi
   - Séquencement optimal des efforts
   - Critères de priorisation utilisés
   - Timeline stratégique

3. OPTIMISATION DES RESSOURCES:
   - Mutualisation des efforts
   - Économies d'échelle identifiées
   - Allocation optimale du budget
   - Compétences à développer

4. GOUVERNANCE INTÉGRÉE:
   - Structure de gouvernance unifiée
   - Processus de décision consolidés
   - Reporting et KPIs harmonisés
   - Rôles et responsabilités clairs

5. PLAN DE TRANSFORMATION:
   - Phases de transformation
   - Jalons critiques
   - Facteurs de succès
   - Gestion des résistances

6. MESURE DU SUCCÈS:
   - KPIs de performance globale
   - Métriques d'efficacité
   - ROI attendu par framework
   - Indicateurs de maturité

Pense comme un CCO stratégique et propose une optimisation ambitieuse mais réaliste.
Retourne une stratégie JSON complète et actionnable.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": optimization_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur stratégie optimisation: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de la génération de stratégie par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "La génération automatique de stratégie a échoué. Stratégie manuelle requise.",
                "suggested_actions": [
                    "Développer une stratégie d'optimisation manuelle",
                    "Consulter un Chief Compliance Officer expérimenté",
                    "Utiliser des frameworks stratégiques standards",
                    "Réviser les données d'entrée et réessayer"
                ]
            }

    async def _assess_regulatory_impact_with_llm(
        self, 
        regulatory_analysis: Dict[str, Any], 
        framework: FrameworkType
    ) -> Dict[str, Any]:
        """Évalue l'impact des évolutions réglementaires."""
        
        impact_prompt = f"""
En tant qu'expert en impact réglementaire, évalue l'impact des évolutions {framework.value}:

ANALYSE RÉGLEMENTAIRE:
{json.dumps(regulatory_analysis, indent=2, default=str)[:2500]}

FRAMEWORK: {framework.value}

Évalue avec ton expertise l'impact organisationnel:

1. IMPACT OPÉRATIONNEL:
   - Processus à modifier ou créer
   - Systèmes et technologies impactés
   - Compétences nouvelles requises
   - Charge de travail additionnelle

2. IMPACT BUSINESS:
   - Coûts de mise en conformité
   - Risques de non-conformité
   - Opportunités business créées
   - Avantage concurrentiel potentiel

3. IMPACT ORGANISATIONNEL:
   - Changements structurels nécessaires
   - Nouveaux rôles et responsabilités
   - Formation et développement
   - Gestion du changement

4. IMPACT TEMPOREL:
   - Urgence des adaptations
   - Timeline de mise en œuvre
   - Phases critiques
   - Dependencies externes

5. IMPACT SECTORIEL:
   - Spécificités sectorielles
   - Avantages/désavantages concurrentiels
   - Benchmarking industrie
   - Tendances du marché

6. RISQUES ASSOCIÉS:
   - Risques de non-adaptation
   - Coût de l'inaction
   - Pénalités potentielles
   - Impact réputation

7. OPPORTUNITÉS:
   - Amélioration des processus
   - Innovation forcée
   - Différenciation marché
   - Leadership sectoriel

Fournis une évaluation d'impact complète et stratégique.
Retourne une analyse JSON détaillée avec recommandations.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["regulatory_intelligence"]},
                {"role": "user", "content": impact_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur évaluation impact réglementaire: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de l'évaluation d'impact réglementaire par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "L'évaluation automatique d'impact a échoué. Analyse manuelle requise.",
                "suggested_actions": [
                    "Effectuer une évaluation d'impact manuelle",
                    "Consulter un expert en réglementation",
                    "Utiliser des grilles d'impact standards",
                    "Réviser les données réglementaires et réessayer"
                ]
            }

    async def _generate_preparation_strategy_with_llm(
        self, 
        regulatory_analysis: Dict[str, Any], 
        impact_assessment: Dict[str, Any], 
        framework: FrameworkType
    ) -> Dict[str, Any]:
        """Génère une stratégie de préparation aux évolutions."""
        
        preparation_prompt = f"""
En tant qu'expert en préparation réglementaire, élabore une stratégie proactive:

ÉVOLUTIONS RÉGLEMENTAIRES:
{json.dumps(regulatory_analysis, indent=2, default=str)[:1500]}

ÉVALUATION D'IMPACT:
{json.dumps(impact_assessment, indent=2, default=str)[:1500]}

FRAMEWORK: {framework.value}

Développe une stratégie de préparation complète:

1. STRATÉGIE DE PRÉPARATION:
   - Approche proactive recommandée
   - Phases de préparation
   - Actions immédiates prioritaires
   - Plan de contingence

2. RECOMMANDATIONS TACTIQUES:
   - Actions à court terme (0-3 mois)
   - Préparations moyen terme (3-12 mois)
   - Stratégie long terme (12+ mois)
   - Quick wins identifiées

3. VEILLE STRATÉGIQUE:
   - Sources de monitoring prioritaires
   - Signaux faibles à surveiller
   - Fréquence de révision
   - Triggers d'action

4. PRÉPARATION ORGANISATIONNELLE:
   - Compétences à développer
   - Processus à adapter
   - Systèmes à modifier
   - Gouvernance à ajuster

5. GESTION DES PARTIES PRENANTES:
   - Communication interne
   - Engagement régulateurs
   - Coordination industrie
   - Influence normative

6. MESURE ET AJUSTEMENT:
   - Indicateurs de préparation
   - Points de contrôle
   - Mécanismes d'ajustement
   - Validation de la stratégie

Conçois une stratégie qui anticipe et prépare efficacement aux changements.
Retourne un plan JSON stratégique et actionnable.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["regulatory_intelligence"]},
                {"role": "user", "content": preparation_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            data = json.loads(json_content)
            return {
                "recommendations": data.get("tactical_recommendations", {}).get("actions", []),
                "monitoring_priorities": data.get("strategic_monitoring", {}).get("priorities", []),
                "preparation_strategy": data.get("preparation_strategy", {}),
                "organizational_preparation": data.get("organizational_preparation", {}),
                "stakeholder_management": data.get("stakeholder_management", {}),
                "measurement": data.get("measurement", {})
            }
        except Exception as e:
            logger.error(f"Erreur stratégie préparation: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de la génération de stratégie de préparation par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "La génération automatique de stratégie de préparation a échoué. Planification manuelle requise.",
                "suggested_actions": [
                    "Développer une stratégie de préparation manuelle",
                    "Consulter un expert en préparation réglementaire",
                    "Utiliser des frameworks de préparation standards",
                    "Réviser les données d'analyse et réessayer"
                ]
            }

    async def _analyze_current_compliance_state_with_llm(
        self, 
        current_state: Dict[str, Any], 
        frameworks: List[FrameworkType], 
        org_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyse l'état actuel de conformité."""
        
        state_analysis_prompt = f"""
En tant qu'expert senior en audit de conformité, analyse l'état actuel:

ÉTAT ACTUEL DÉCLARÉ:
{json.dumps(current_state, indent=2, default=str)[:2000]}

FRAMEWORKS CIBLES: {[f.value for f in frameworks]}

PROFIL ORGANISATIONNEL:
{json.dumps(org_profile, indent=2)[:1000]}

Effectue une analyse experte de l'état de conformité:

1. DIAGNOSTIC GLOBAL:
   - Niveau de maturité général
   - Points forts organisationnels
   - Lacunes critiques identifiées
   - Tendances observées

2. ANALYSE PAR FRAMEWORK:
   - État de conformité par framework
   - Gaps spécifiques identifiés
   - Niveau de risque associé
   - Efforts requis pour la conformité

3. CAPACITÉS ORGANISATIONNELLES:
   - Maturité des processus
   - Compétences disponibles
   - Infrastructure de conformité
   - Culture de conformité

4. GOUVERNANCE ET PILOTAGE:
   - Structure de gouvernance actuelle
   - Processus de pilotage
   - Reporting et monitoring
   - Prise de décision

5. RESSOURCES ET MOYENS:
   - Budget alloué à la conformité
   - Équipes dédiées
   - Outils et technologies
   - Support externe

6. BENCHMARKING IMPLICITE:
   - Positionnement vs meilleures pratiques
   - Comparaison sectorielle
   - Écarts identifiés
   - Potentiel d'amélioration

7. FACTEURS DE RISQUE:
   - Risques de non-conformité
   - Vulnérabilités identifiées
   - Impact potentiel
   - Urgence d'action

Fournis un diagnostic expert complet et nuancé.
Retourne une analyse JSON structurée et approfondie.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["compliance_expert"]},
                {"role": "user", "content": state_analysis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur analyse état actuel: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de l'analyse de l'état actuel par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "L'analyse automatique de l'état actuel a échoué. Diagnostic manuel requis.",
                "suggested_actions": [
                    "Effectuer un diagnostic de conformité manuel",
                    "Consulter un expert en audit de conformité",
                    "Utiliser des grilles d'évaluation standards",
                    "Réviser les données d'état actuel et réessayer"
                ]
            }

    async def _model_optimization_scenarios_with_llm(
        self, 
        current_state: Dict[str, Any], 
        frameworks: List[FrameworkType], 
        constraints: Dict[str, Any], 
        org_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modélise des scénarios d'optimisation."""
        
        scenarios_prompt = f"""
En tant qu'expert en stratégie de conformité, modélise des scénarios d'optimisation:

ÉTAT ACTUEL:
{json.dumps(current_state, indent=2, default=str)[:1500]}

FRAMEWORKS: {[f.value for f in frameworks]}

CONTRAINTES:
{json.dumps(constraints, indent=2)[:1000]}

PROFIL ORGANISATION:
{json.dumps(org_profile, indent=2)[:1000]}

Modélise différents scénarios d'optimisation de la conformité:

1. SCÉNARIO CONSERVATEUR:
   - Approche prudente et progressive
   - Minimisation des risques
   - Timeline étendue
   - Investissement minimal
   - Bénéfices attendus

2. SCÉNARIO ÉQUILIBRÉ:
   - Approche pragmatique
   - Balance risque/opportunité
   - Timeline réaliste
   - Investissement modéré
   - ROI optimisé

3. SCÉNARIO AMBITIEUX:
   - Transformation accélérée
   - Innovation et différenciation
   - Timeline agressive
   - Investissement significatif
   - Avantage concurrentiel

4. SCÉNARIO HYBRIDE:
   - Approche différenciée par framework
   - Priorisation intelligente
   - Phasage optimisé
   - Allocation flexible
   - Adaptabilité maximale

Pour chaque scénario, définis:
- Stratégie d'implémentation
- Ressources requises
- Timeline et phases
- Risques et mitigation
- Bénéfices attendus
- Probabilité de succès

Pense comme un stratège et propose des scénarios réalistes et différenciés.
Retourne une modélisation JSON complète avec recommandations.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": scenarios_prompt}
            ],
            model="gpt-4.1",
            temperature=0.4
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur modélisation scénarios: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de la modélisation des scénarios par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "La modélisation automatique des scénarios a échoué. Analyse manuelle requise.",
                "suggested_actions": [
                    "Réviser les données d'entrée pour la modélisation",
                    "Effectuer une modélisation de scénarios manuelle",
                    "Consulter un expert en stratégie de conformité",
                    "Utiliser des modèles de scénarios standards"
                ]
            }

    async def _perform_cost_benefit_analysis_with_llm(
        self, 
        scenarios: Dict[str, Any], 
        org_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Effectue une analyse coût-bénéfice sophistiquée."""
        
        cost_benefit_prompt = f"""
En tant qu'expert en analyse financière de conformité, évalue les coûts-bénéfices:

SCÉNARIOS D'OPTIMISATION:
{json.dumps(scenarios, indent=2, default=str)[:2500]}

PROFIL ORGANISATIONNEL:
{json.dumps(org_profile, indent=2)[:1000]}

Effectue une analyse coût-bénéfice experte et complète:

1. ANALYSE DES COÛTS:
   - Coûts d'implémentation par scénario
   - Coûts opérationnels récurrents
   - Coûts cachés et indirects
   - Risques financiers

2. ANALYSE DES BÉNÉFICES:
   - Bénéfices directs quantifiables
   - Économies réalisées
   - Bénéfices indirects (réputation, etc.)
   - Valeur ajoutée business

3. CALCUL DU ROI:
   - ROI financier par scénario
   - Période de retour sur investissement
   - Valeur actualisée nette
   - Analyse de sensibilité

4. ANALYSE COMPARATIVE:
   - Comparaison entre scénarios
   - Avantages et inconvénients
   - Profil risque/rendement
   - Recommandation finale

5. FACTEURS CRITIQUES:
   - Hypothèses clés
   - Variables d'impact
   - Seuils de rentabilité
   - Scénarios de stress

6. IMPACT BUSINESS:
   - Amélioration de la performance
   - Réduction des risques
   - Avantage concurrentiel
   - Création de valeur

7. FINANCEMENT ET BUDGET:
   - Besoins de financement
   - Stratégie budgétaire
   - Sources de financement
   - Optimisation fiscale

Pense comme un CFO expert et fournis une analyse financière rigoureuse.
Retourne une analyse JSON détaillée avec recommandations business.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": cost_benefit_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur analyse coût-bénéfice: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de l'analyse coût-bénéfice par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "L'analyse coût-bénéfice automatique a échoué. Analyse financière manuelle requise.",
                "suggested_actions": [
                    "Effectuer une analyse coût-bénéfice manuelle",
                    "Consulter un expert financier ou CFO",
                    "Utiliser des modèles financiers standards",
                    "Réviser les données des scénarios et réessayer"
                ]
            }

    async def _generate_implementation_roadmap_with_llm(
        self, 
        strategy: Dict[str, Any], 
        constraints: Dict[str, Any], 
        org_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère une roadmap d'implémentation détaillée."""
        
        roadmap_prompt = f"""
En tant qu'expert en gestion de projet de conformité, crée une roadmap d'implémentation:

STRATÉGIE RECOMMANDÉE:
{json.dumps(strategy, indent=2, default=str)[:2000]}

CONTRAINTES:
{json.dumps(constraints, indent=2)[:1000]}

PROFIL ORGANISATION:
{json.dumps(org_profile, indent=2)[:1000]}

Développe une roadmap d'implémentation complète et réaliste:

1. STRUCTURE DE LA ROADMAP:
   - Phases d'implémentation (court/moyen/long terme)
   - Jalons critiques et validation
   - Interdépendances entre phases
   - Timeline globale

2. PHASE 1 - FONDATIONS (0-6 mois):
   - Objectifs et livrables
   - Actions prioritaires
   - Ressources mobilisées
   - Risques et mitigation
   - Critères de succès

3. PHASE 2 - DÉPLOIEMENT (6-18 mois):
   - Objectifs et livrables
   - Actions de déploiement
   - Ressources requises
   - Challenges anticipés
   - Mesures de performance

4. PHASE 3 - OPTIMISATION (18+ mois):
   - Objectifs et livrables
   - Actions d'amélioration continue
   - Évolution des ressources
   - Innovation et leadership
   - Pérennisation

5. PLAN DE GESTION:
   - Gouvernance du projet
   - Gestion des risques
   - Communication et reporting
   - Gestion du changement
   - Assurance qualité

6. RESSOURCES ET BUDGET:
   - Plan de ressources par phase
   - Budget prévisionnel
   - Allocation des compétences
   - Support externe requis
   - Optimisation des coûts

7. MESURE ET PILOTAGE:
   - KPIs par phase
   - Reporting et dashboards
   - Points de contrôle
   - Mécanismes d'ajustement
   - Validation des acquis

Conçois une roadmap pragmatique, détaillée et orientée succès.
Retourne un plan JSON structuré et actionnable.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["strategic_advisor"]},
                {"role": "user", "content": roadmap_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            return json.loads(json_content)
        except Exception as e:
            logger.error(f"Erreur génération roadmap: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de la génération de roadmap par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "La génération automatique de roadmap a échoué. Planification manuelle requise.",
                "suggested_actions": [
                    "Créer une roadmap manuelle basée sur la stratégie",
                    "Consulter un expert en gestion de projet",
                    "Utiliser un outil de planification externe",
                    "Réviser les données d'entrée et réessayer"
                ]
            }


# Factory function
def get_compliance_analysis_module(llm_client: LLMIntegration = None):
    """Factory function pour obtenir une instance du module d'analyse de conformité."""
    return ComplianceAnalysisModule(llm_client=llm_client) 