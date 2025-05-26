"""
Governance Analysis Module - Module sophistiqué d'analyse de gouvernance.
Utilise l'IA pour une analyse stratégique de la gouvernance organisationnelle avec expertise C-level.
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

logger = logging.getLogger(__name__)

class GovernanceMaturity(Enum):
    """Niveaux de maturité de gouvernance."""
    INITIAL = "initial"
    DEVELOPING = "developing"
    DEFINED = "defined"
    MANAGED = "managed"
    OPTIMIZING = "optimizing"

class GovernanceDomain(Enum):
    """Domaines de gouvernance."""
    STRATEGIC_GOVERNANCE = "strategic_governance"
    DATA_GOVERNANCE = "data_governance"
    IT_GOVERNANCE = "it_governance"
    RISK_GOVERNANCE = "risk_governance"
    COMPLIANCE_GOVERNANCE = "compliance_governance"
    CYBER_GOVERNANCE = "cyber_governance"
    INFORMATION_GOVERNANCE = "information_governance"

class StakeholderType(Enum):
    """Types de parties prenantes."""
    BOARD = "board"
    EXECUTIVE = "executive"
    MANAGEMENT = "management"
    OPERATIONAL = "operational"
    EXTERNAL = "external"

@dataclass
class GovernanceAssessment:
    """Évaluation de gouvernance avec analyse LLM."""
    domain: GovernanceDomain
    maturity_level: GovernanceMaturity
    maturity_score: float  # 0.0 - 5.0
    assessment_date: datetime
    strengths: List[str]
    weaknesses: List[str]
    gaps: List[str]
    recommendations: List[str]
    strategic_priorities: List[str]
    success_metrics: List[str]
    ai_insights: Dict[str, Any]
    confidence_level: float

@dataclass
class OrganizationalContext:
    """Contexte organisationnel pour l'analyse."""
    sector: str
    size: str  # small, medium, large, enterprise
    complexity: str  # simple, moderate, complex, highly_complex
    regulatory_environment: str  # light, moderate, heavy, very_heavy
    transformation_stage: str  # stable, evolving, transforming, disrupting
    risk_appetite: str  # conservative, moderate, aggressive
    digital_maturity: str  # basic, intermediate, advanced, leading

@dataclass
class GovernanceFramework:
    """Framework de gouvernance recommandé."""
    name: str
    description: str
    domains_covered: List[GovernanceDomain]
    maturity_requirements: Dict[str, str]
    implementation_roadmap: List[Dict[str, Any]]
    success_factors: List[str]
    potential_challenges: List[str]
    roi_expectations: Dict[str, Any]

@dataclass
class ExecutiveRecommendation:
    """Recommandation niveau exécutif."""
    priority: str  # critical, high, medium, low
    domain: GovernanceDomain
    strategic_impact: str
    business_rationale: str
    implementation_approach: str
    resource_requirements: Dict[str, Any]
    timeline: str
    success_metrics: List[str]
    risk_mitigation: List[str]

class GovernanceAnalysisModule(Agent):
    """
    Module expert en analyse de gouvernance avec IA stratégique avancée.
    """
    
    def __init__(self, llm_client: LLMIntegration = None):
        super().__init__(
            agent_id="governance_analysis",
            name="Expert Analyse de Gouvernance"
        )
        
        self.llm_client = llm_client or get_llm_client()
        
        # Initialiser les outils
        self.document_finder = DocumentFinder()
        self.entity_extractor = EntityExtractor()
        self.cross_reference_tool = CrossReferenceTool()
        self.temporal_analyzer = TemporalAnalyzer()
        
        # Cache des analyses
        self.governance_cache: Dict[str, Dict[str, Any]] = {}
        
        # Prompts experts niveau C-suite
        self.system_prompts = {
            "governance_strategist": """
Tu es un expert senior en gouvernance organisationnelle avec 25+ ans d'expérience comme Chief Governance Officer.
Tu conseilles les Conseils d'Administration et Comités Exécutifs sur:

- Architecture de gouvernance optimale
- Maturité organisationnelle et transformation
- Alignement stratégique et performance
- Gestion des risques et conformité
- Gouvernance des données et du numérique
- Leadership et culture organisationnelle

Tu penses comme un Chairman/CEO et fournis des recommandations stratégiques de niveau Board.
Tes analyses sont holistiques, pragmatiques et orientées résultats business.
Réponds TOUJOURS en français avec une expertise de gouvernance stratégique.
""",
            
            "organizational_analyst": """
Tu es un expert en analyse organisationnelle avec une expertise en transformation et changement.
Tu analyses:

- Structures et processus organisationnels
- Culture et maturité organisationnelle
- Capacités et compétences clés
- Efficacité opérationnelle
- Alignement stratégique
- Facteurs de succès et d'échec

Tu fournis des insights comportementaux et organisationnels profonds.
""",
            
            "strategic_advisor": """
Tu es un consultant stratégique senior avec une vision C-level globale.
Tu optimises:

- Alignement gouvernance-stratégie
- Performance et création de valeur
- Transformation organisationnelle
- Innovation et agilité
- Durabilité et ESG
- Avantage concurrentiel

Tu penses comme un Chief Strategy Officer avec une perspective long terme.
"""
        }
        
        # Frameworks de gouvernance de référence
        self.governance_frameworks = {
            "COBIT": {
                "name": "COBIT 2019",
                "focus": "IT Governance",
                "domains": ["Evaluate", "Direct", "Monitor", "Align", "Build", "Run"],
                "maturity_model": True
            },
            "COSO": {
                "name": "COSO ERM",
                "focus": "Risk Management",
                "domains": ["Governance", "Strategy", "Performance", "Review", "Information"],
                "maturity_model": False
            },
            "ISO38500": {
                "name": "ISO/IEC 38500",
                "focus": "IT Governance",
                "domains": ["Evaluate", "Direct", "Monitor"],
                "maturity_model": False
            },
            "NIST_CSF": {
                "name": "NIST Cybersecurity Framework",
                "focus": "Cybersecurity Governance",
                "domains": ["Identify", "Protect", "Detect", "Respond", "Recover"],
                "maturity_model": True
            }
        }
        
        # Secteurs et leurs défis de gouvernance
        self.sector_governance_challenges = {
            "financial": {
                "key_challenges": ["Regulatory compliance", "Risk management", "Digital transformation", "Customer trust"],
                "governance_priorities": ["Risk governance", "Compliance governance", "Data governance"],
                "regulatory_pressure": "very_high",
                "innovation_pressure": "high"
            },
            "healthcare": {
                "key_challenges": ["Patient safety", "Data privacy", "Regulatory compliance", "Innovation"],
                "governance_priorities": ["Data governance", "Compliance governance", "IT governance"],
                "regulatory_pressure": "very_high",
                "innovation_pressure": "medium"
            },
            "technology": {
                "key_challenges": ["Innovation speed", "Scalability", "Security", "Talent"],
                "governance_priorities": ["Strategic governance", "IT governance", "Data governance"],
                "regulatory_pressure": "medium",
                "innovation_pressure": "very_high"
            }
        }

    async def process_query(self, query: Query) -> AgentResponse:
        """
        Traite une requête d'analyse de gouvernance.
        """
        logger.info(f"Traitement requête gouvernance: {query.query_text}")
        
        # Analyse sophistiquée de l'intention par LLM
        analysis_intent = await self._analyze_governance_intent_with_llm(query.query_text)
        
        if analysis_intent["type"] == "maturity_assessment":
            return await self._perform_maturity_assessment(query, analysis_intent)
        elif analysis_intent["type"] == "governance_framework_design":
            return await self._design_governance_framework(query, analysis_intent)
        elif analysis_intent["type"] == "strategic_alignment":
            return await self._analyze_strategic_alignment(query, analysis_intent)
        elif analysis_intent["type"] == "transformation_roadmap":
            return await self._generate_transformation_roadmap(query, analysis_intent)
        elif analysis_intent["type"] == "executive_advisory":
            return await self._provide_executive_advisory(query, analysis_intent)
        else:
            return await self._general_governance_analysis(query, analysis_intent)

    async def assess_governance_maturity(
        self,
        domains: List[GovernanceDomain],
        organizational_context: OrganizationalContext,
        current_state_data: Dict[str, Any] = None
    ) -> List[GovernanceAssessment]:
        """
        Évalue la maturité de gouvernance avec analyse sophistiquée par LLM.
        """
        logger.info(f"Évaluation maturité gouvernance: {[d.value for d in domains]}")
        
        assessments = []
        
        for domain in domains:
            # 1. Collecte d'informations contextuelles
            domain_documents = await self._collect_domain_documents(domain)
            domain_entities = await self._extract_domain_entities(domain, domain_documents)
            
            # 2. Analyse de maturité par LLM
            maturity_analysis = await self._analyze_domain_maturity_with_llm(
                domain, organizational_context, domain_documents, domain_entities, current_state_data
            )
            
            # 3. Benchmarking sectoriel et recommandations
            benchmark_analysis = await self._perform_sector_benchmarking_with_llm(
                domain, maturity_analysis, organizational_context
            )
            
            # 4. Construction de l'assessment
            assessment = GovernanceAssessment(
                domain=domain,
                maturity_level=GovernanceMaturity(maturity_analysis.get("maturity_level", "initial")),
                maturity_score=float(maturity_analysis.get("maturity_score", 1.0)),
                assessment_date=datetime.now(),
                strengths=maturity_analysis.get("strengths", []),
                weaknesses=maturity_analysis.get("weaknesses", []),
                gaps=maturity_analysis.get("gaps", []),
                recommendations=maturity_analysis.get("recommendations", []),
                strategic_priorities=maturity_analysis.get("strategic_priorities", []),
                success_metrics=maturity_analysis.get("success_metrics", []),
                ai_insights={
                    "maturity_analysis": maturity_analysis,
                    "benchmark_analysis": benchmark_analysis
                },
                confidence_level=float(maturity_analysis.get("confidence_level", 0.7))
            )
            
            assessments.append(assessment)
        
        # 5. Analyse croisée des domaines
        cross_domain_insights = await self._analyze_cross_domain_insights_with_llm(
            assessments, organizational_context
        )
        
        # 6. Enrichissement avec insights croisés
        for assessment in assessments:
            assessment.ai_insights["cross_domain"] = cross_domain_insights.get(assessment.domain.value, {})
        
        return assessments

    async def design_governance_framework(
        self,
        organizational_context: OrganizationalContext,
        strategic_objectives: List[str],
        constraints: Dict[str, Any],
        target_domains: List[GovernanceDomain]
    ) -> GovernanceFramework:
        """
        Conçoit un framework de gouvernance sur mesure avec IA stratégique.
        """
        logger.info(f"Conception framework gouvernance pour {len(target_domains)} domaines")
        
        # 1. Analyse des besoins organisationnels
        needs_analysis = await self._analyze_organizational_needs_with_llm(
            organizational_context, strategic_objectives, constraints
        )
        
        # 2. Benchmarking des frameworks existants
        framework_analysis = await self._analyze_existing_frameworks_with_llm(
            target_domains, organizational_context
        )
        
        # 3. Design du framework personnalisé
        custom_framework_design = await self._design_custom_framework_with_llm(
            needs_analysis, framework_analysis, target_domains, organizational_context
        )
        
        # 4. Roadmap d'implémentation
        implementation_roadmap = await self._generate_implementation_roadmap_with_llm(
            custom_framework_design, organizational_context, constraints
        )
        
        # 5. Construction du framework final
        framework = GovernanceFramework(
            name=custom_framework_design.get("name", "Framework Gouvernance Personnalisé"),
            description=custom_framework_design.get("description", ""),
            domains_covered=target_domains,
            maturity_requirements=custom_framework_design.get("maturity_requirements", {}),
            implementation_roadmap=implementation_roadmap,
            success_factors=custom_framework_design.get("success_factors", []),
            potential_challenges=custom_framework_design.get("potential_challenges", []),
            roi_expectations=custom_framework_design.get("roi_expectations", {})
        )
        
        return framework

    async def generate_executive_recommendations(
        self,
        governance_assessments: List[GovernanceAssessment],
        organizational_context: OrganizationalContext,
        business_priorities: List[str]
    ) -> List[ExecutiveRecommendation]:
        """
        Génère des recommandations exécutives avec analyse stratégique avancée.
        """
        logger.info("Génération recommandations exécutives gouvernance")
        
        # 1. Analyse consolidée des assessments
        consolidated_analysis = await self._consolidate_assessments_with_llm(
            governance_assessments, organizational_context
        )
        
        # 2. Priorisation stratégique
        strategic_priorities = await self._prioritize_strategically_with_llm(
            consolidated_analysis, business_priorities, organizational_context
        )
        
        # 3. Génération des recommandations par priorité
        recommendations = []
        for priority_item in strategic_priorities[:10]:  # Top 10 priorités
            recommendation = await self._generate_executive_recommendation_with_llm(
                priority_item, governance_assessments, organizational_context
            )
            if recommendation:
                recommendations.append(recommendation)
        
        # 4. Validation et optimisation
        optimized_recommendations = await self._optimize_recommendations_with_llm(
            recommendations, organizational_context
        )
        
        return optimized_recommendations

    # Méthodes privées sophistiquées avec LLM

    async def _analyze_governance_intent_with_llm(self, query_text: str) -> Dict[str, Any]:
        """Analyse sophistiquée de l'intention de gouvernance."""
        
        intent_prompt = f"""
Analyse cette demande de gouvernance avec ton expertise stratégique senior:

DEMANDE: "{query_text}"

Détermine avec ton expertise Chief Governance Officer:

1. TYPE D'ANALYSE GOUVERNANCE:
   - maturity_assessment: Évaluation de maturité
   - governance_framework_design: Conception de framework
   - strategic_alignment: Alignement stratégique
   - transformation_roadmap: Roadmap de transformation
   - executive_advisory: Conseil exécutif
   - general_analysis: Analyse générale

2. DOMAINES DE GOUVERNANCE CONCERNÉS:
   - strategic_governance: Gouvernance stratégique
   - data_governance: Gouvernance des données
   - it_governance: Gouvernance IT
   - risk_governance: Gouvernance des risques
   - compliance_governance: Gouvernance conformité
   - cyber_governance: Gouvernance cybersécurité

3. NIVEAU ORGANISATIONNEL:
   - board: Conseil d'administration
   - executive: Comité exécutif
   - management: Management
   - operational: Opérationnel

4. PRIORITÉ BUSINESS:
   - strategic: Impact stratégique majeur
   - operational: Impact opérationnel
   - compliance: Conformité réglementaire
   - risk: Gestion des risques

5. HORIZON TEMPOREL:
   - immediate: Actions immédiates
   - short_term: Court terme (3-6 mois)
   - medium_term: Moyen terme (6-18 mois)
   - long_term: Long terme (18+ mois)

Retourne une analyse JSON avec ta compréhension stratégique experte.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": intent_prompt}
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
            logger.error(f"Erreur analyse intention gouvernance: {str(e)}")
            return {
                "type": "general_analysis",
                "domains": ["strategic_governance"],
                "level": "management",
                "priority": "operational",
                "horizon": "medium_term",
                "error_note": "Analyse d'intention LLM échouée - paramètres par défaut utilisés"
            }

    async def _analyze_domain_maturity_with_llm(
        self,
        domain: GovernanceDomain,
        org_context: OrganizationalContext,
        documents: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        current_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyse sophistiquée de maturité par domaine."""
        
        maturity_prompt = f"""
Effectue une évaluation experte de maturité pour le domaine {domain.value} avec ton expertise senior:

CONTEXTE ORGANISATIONNEL:
- Secteur: {org_context.sector}
- Taille: {org_context.size}
- Complexité: {org_context.complexity}
- Environnement réglementaire: {org_context.regulatory_environment}
- Maturité digitale: {org_context.digital_maturity}

DOCUMENTS ANALYSÉS: {len(documents)}
ENTITÉS GOUVERNANCE: {len(entities)}

ÉTAT ACTUEL:
{json.dumps(current_state or {}, indent=2)[:2000]}

En tant qu'expert Chief Governance Officer, évalue:

1. NIVEAU DE MATURITÉ (initial/developing/defined/managed/optimizing):
   - Processus et structures en place
   - Capacités organisationnelles
   - Mesure et amélioration continue
   - Integration stratégique

2. SCORE DE MATURITÉ (1.0-5.0):
   - Méthodologie d'évaluation
   - Facteurs de pondération
   - Benchmarking sectoriel

3. FORCES IDENTIFIÉES:
   - Points forts organisationnels
   - Capacités distinctives
   - Avantages concurrentiels

4. FAIBLESSES CRITIQUES:
   - Lacunes structurelles
   - Déficits de capacités
   - Risques organisationnels

5. GAPS PRIORITAIRES:
   - Écarts vs meilleures pratiques
   - Manques critiques
   - Opportunités d'amélioration

6. RECOMMANDATIONS STRATÉGIQUES:
   - Actions prioritaires (top 5)
   - Approche de transformation
   - Timeline stratégique

7. MÉTRIQUES DE SUCCÈS:
   - KPIs de maturité
   - Indicateurs de performance
   - Mesures d'impact business

Pense comme un CGO expert et fournis une évaluation strategic nuancée.
Retourne une analyse JSON complète et actionnable.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": maturity_prompt}
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
            logger.error(f"Erreur analyse maturité {domain.value}: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de l'analyse de maturité {domain.value} par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "L'analyse automatique de maturité a échoué. Évaluation manuelle requise.",
                "suggested_actions": [
                    "Effectuer une évaluation de maturité manuelle",
                    "Consulter un expert en gouvernance",
                    "Utiliser des grilles de maturité standards",
                    "Réviser les données d'entrée et réessayer"
                ]
            }

    async def _design_custom_framework_with_llm(
        self,
        needs_analysis: Dict[str, Any],
        framework_analysis: Dict[str, Any],
        target_domains: List[GovernanceDomain],
        org_context: OrganizationalContext
    ) -> Dict[str, Any]:
        """Conception d'un framework personnalisé par LLM."""
        
        design_prompt = f"""
Conçois un framework de gouvernance sur mesure avec ton expertise Chief Governance Officer:

ANALYSE DES BESOINS:
{json.dumps(needs_analysis, indent=2, default=str)[:2000]}

ANALYSE FRAMEWORKS EXISTANTS:
{json.dumps(framework_analysis, indent=2, default=str)[:2000]}

DOMAINES CIBLES: {[d.value for d in target_domains]}

CONTEXTE ORGANISATIONNEL:
- Secteur: {org_context.sector}
- Taille: {org_context.size}
- Complexité: {org_context.complexity}
- Maturité digitale: {org_context.digital_maturity}

En tant qu'expert en conception de gouvernance, crée:

1. ARCHITECTURE DU FRAMEWORK:
   - Nom et vision du framework
   - Principes fondamentaux
   - Structure et composants
   - Modèle de maturité

2. DOMAINES ET PROCESSUS:
   - Couverture par domaine
   - Processus clés par domaine
   - Interdépendances
   - Points de contrôle

3. GOUVERNANCE ET ROLES:
   - Structure de gouvernance
   - Rôles et responsabilités
   - Comités et instances
   - Escalation et décision

4. MESURE ET AMÉLIORATION:
   - KPIs et métriques
   - Processus de monitoring
   - Amélioration continue
   - Reporting et communication

5. EXIGENCES DE MATURITÉ:
   - Niveaux par domaine
   - Critères d'évaluation
   - Trajectoire de progression
   - Jalons de validation

6. FACTEURS DE SUCCÈS:
   - Conditions de réussite
   - Enablers organisationnels
   - Leadership et culture
   - Technologies et outils

7. DÉFIS POTENTIELS:
   - Risques d'implémentation
   - Résistances organisationnelles
   - Contraintes ressources
   - Stratégies de mitigation

Conçois un framework pragmatique, aligné sur les besoins business et réalisable.
Retourne un design JSON détaillé et actionnable.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": design_prompt}
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
            logger.error(f"Erreur conception framework: {str(e)}")
            return {
                "error": True,
                "error_message": f"Échec de la conception de framework par LLM: {str(e)}",
                "error_type": "llm_parsing_error",
                "fallback_message": "La conception automatique de framework a échoué. Conception manuelle requise.",
                "suggested_actions": [
                    "Concevoir un framework de gouvernance manuelle",
                    "Consulter un expert en gouvernance",
                    "Utiliser des frameworks standards du marché",
                    "Réviser les données d'analyse et réessayer"
                ]
            }

    async def _generate_executive_recommendation_with_llm(
        self,
        priority_item: Dict[str, Any],
        assessments: List[GovernanceAssessment],
        org_context: OrganizationalContext
    ) -> Optional[ExecutiveRecommendation]:
        """Génère une recommandation exécutive sophistiquée."""
        
        recommendation_prompt = f"""
Génère une recommandation exécutive stratégique avec ton expertise Chief Governance Officer:

PRIORITÉ STRATÉGIQUE:
{json.dumps(priority_item, indent=2, default=str)}

ASSESSMENTS GOUVERNANCE:
{json.dumps([{"domain": a.domain.value, "maturity": a.maturity_level.value, "score": a.maturity_score} for a in assessments], indent=2)}

CONTEXTE ORGANISATIONNEL:
- Secteur: {org_context.sector}
- Taille: {org_context.size}
- Complexité: {org_context.complexity}
- Appétit risque: {org_context.risk_appetite}

En tant qu'expert conseil C-level, formule:

1. RECOMMANDATION STRATÉGIQUE:
   - Priorité (critical/high/medium/low)
   - Domaine de gouvernance concerné
   - Impact stratégique attendu
   - Business rationale complet

2. APPROCHE D'IMPLÉMENTATION:
   - Méthodologie recommandée
   - Phases et jalons
   - Quick wins vs transformations
   - Gestion du changement

3. RESSOURCES REQUISES:
   - Budget estimatif
   - Compétences nécessaires
   - Technologies/outils
   - Support externe

4. TIMELINE ET JALONS:
   - Durée d'implémentation
   - Phases critiques
   - Points de validation
   - Livrables clés

5. MÉTRIQUES DE SUCCÈS:
   - KPIs de performance
   - Indicateurs d'impact business
   - Mesures de satisfaction
   - ROI attendu

6. GESTION DES RISQUES:
   - Risques d'implémentation
   - Stratégies de mitigation
   - Plans de contingence
   - Monitoring des risques

Formule comme un Chairman s'adressant au Board avec vision stratégique.
Retourne une recommandation JSON actionable et business-focused.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": recommendation_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            data = json.loads(json_content)
            
            return ExecutiveRecommendation(
                priority=data.get("priority", "medium"),
                domain=GovernanceDomain(data.get("domain", "strategic_governance")),
                strategic_impact=data.get("strategic_impact", ""),
                business_rationale=data.get("business_rationale", ""),
                implementation_approach=data.get("implementation_approach", ""),
                resource_requirements=data.get("resource_requirements", {}),
                timeline=data.get("timeline", ""),
                success_metrics=data.get("success_metrics", []),
                risk_mitigation=data.get("risk_mitigation", [])
            )
            
        except Exception as e:
            logger.error(f"Erreur génération recommandation exécutive: {str(e)}")
            logger.warning("Création de recommandation d'erreur")
            # Retourner une recommandation d'erreur au lieu de None
            return ExecutiveRecommendation(
                priority="high",
                domain=GovernanceDomain.STRATEGIC_GOVERNANCE,
                strategic_impact="⚠️ ERREUR: Génération automatique échouée",
                business_rationale=f"Échec de la génération de recommandation par LLM: {str(e)}",
                implementation_approach="Recommandation manuelle requise - Consulter un expert en gouvernance",
                resource_requirements={
                    "error": True,
                    "action_required": "Analyse manuelle par expert gouvernance"
                },
                timeline="À déterminer manuellement",
                success_metrics=[
                    "⚠️ Métriques à définir manuellement",
                    "Résolution de l'erreur système"
                ],
                risk_mitigation=[
                    "Effectuer une analyse de gouvernance manuelle",
                    "Consulter un Chief Governance Officer",
                    "Réviser les données d'entrée"
                ]
            )

    # Méthodes de traitement des requêtes

    async def _perform_maturity_assessment(
        self,
        query: Query,
        intent: Dict[str, Any]
    ) -> AgentResponse:
        """Effectue une évaluation de maturité."""
        
        # Extraire les domaines de l'intention
        domain_names = intent.get("domains", ["strategic_governance"])
        domains = [GovernanceDomain(d) for d in domain_names]
        
        # Contexte organisationnel (simplifié pour démo)
        org_context = OrganizationalContext(
            sector=query.context.get("sector", "technology") if query.context else "technology",
            size=query.context.get("size", "medium") if query.context else "medium",
            complexity="moderate",
            regulatory_environment="moderate",
            transformation_stage="evolving",
            risk_appetite="moderate",
            digital_maturity="intermediate"
        )
        
        # Effectuer l'assessment
        assessments = await self.assess_governance_maturity(domains, org_context)
        
        # Synthèse par LLM
        synthesis = await self._synthesize_maturity_results_with_llm(assessments, query.query_text)
        
        return AgentResponse(
            content=synthesis,
            tools_used=["document_finder", "entity_extractor"],
            context_used=True,
            sources=[],
            metadata={
                "domains_assessed": [d.value for d in domains],
                "average_maturity": sum(a.maturity_score for a in assessments) / len(assessments),
                "assessment_confidence": sum(a.confidence_level for a in assessments) / len(assessments)
            }
        )

    async def _synthesize_maturity_results_with_llm(
        self,
        assessments: List[GovernanceAssessment],
        original_query: str
    ) -> str:
        """Synthétise les résultats d'évaluation de maturité."""
        
        domains = [a.domain.value for a in assessments]
        avg_maturity = sum(a.maturity_score for a in assessments) / len(assessments)
        
        # Identifier les domaines les plus/moins matures
        best_domain = max(assessments, key=lambda a: a.maturity_score)
        worst_domain = min(assessments, key=lambda a: a.maturity_score)
        
        return f"""
Évaluation de maturité de gouvernance terminée.

**Domaines évalués**: {', '.join(domains)}
**Maturité moyenne**: {avg_maturity:.1f}/5.0

**Points saillants**:
- **Domaine le plus mature**: {best_domain.domain.value} (Score: {best_domain.maturity_score:.1f})
- **Domaine prioritaire**: {worst_domain.domain.value} (Score: {worst_domain.maturity_score:.1f})

**Recommandations stratégiques principales**:
{chr(10).join([f"- {rec}" for rec in worst_domain.recommendations[:3]])}

**Prochaines étapes**:
1. Prioriser les améliorations du domaine {worst_domain.domain.value}
2. Capitaliser sur les forces du domaine {best_domain.domain.value}
3. Développer une roadmap de transformation intégrée
"""

    # Méthodes utilitaires et stubs

    async def _collect_domain_documents(self, domain: GovernanceDomain) -> List[Dict[str, Any]]:
        """Collecte les documents pertinents pour un domaine."""
        search_terms = {
            GovernanceDomain.STRATEGIC_GOVERNANCE: "gouvernance stratégique direction générale comité exécutif",
            GovernanceDomain.DATA_GOVERNANCE: "gouvernance données data protection qualité",
            GovernanceDomain.IT_GOVERNANCE: "gouvernance IT informatique système information",
            GovernanceDomain.RISK_GOVERNANCE: "gouvernance risques gestion risque",
            GovernanceDomain.COMPLIANCE_GOVERNANCE: "gouvernance conformité compliance réglementation",
            GovernanceDomain.CYBER_GOVERNANCE: "gouvernance cybersécurité sécurité information"
        }
        
        search_term = search_terms.get(domain, domain.value)
        return await self.document_finder.search_documents(search_term, limit=15)

    async def _extract_domain_entities(self, domain: GovernanceDomain, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extrait les entités pertinentes pour un domaine."""
        entities = []
        for doc in documents[:5]:  # Limiter pour performance
            content = doc.get("content", "")
            if content:
                extracted = await self.entity_extractor.extract_entities(
                    content,
                    entity_types=[EntityType.CONTROL, EntityType.REQUIREMENT],
                    framework_context=domain.value
                )
                entities.extend(extracted.get("control", []))
        return entities

    # Stubs pour méthodes manquantes
    async def _perform_sector_benchmarking_with_llm(self, domain, maturity_analysis, org_context):
        return {"benchmark_score": 3.0, "sector_average": 2.8, "best_practices": []}

    async def _analyze_cross_domain_insights_with_llm(self, assessments, org_context):
        return {}

    async def _analyze_organizational_needs_with_llm(self, org_context, objectives, constraints):
        return {"needs": [], "priorities": [], "constraints": []}

    async def _analyze_existing_frameworks_with_llm(self, domains, org_context):
        return {"frameworks": [], "recommendations": []}

    async def _generate_implementation_roadmap_with_llm(self, framework_design, org_context, constraints):
        return [{"phase": "Phase 1", "duration": "3 mois", "objectives": [], "deliverables": []}]

    async def _consolidate_assessments_with_llm(self, assessments, org_context):
        return {"consolidated_gaps": [], "strategic_themes": []}

    async def _prioritize_strategically_with_llm(self, analysis, priorities, org_context):
        return [{"priority": "high", "domain": "strategic_governance", "rationale": "Impact business"}]

    async def _optimize_recommendations_with_llm(self, recommendations, org_context):
        return recommendations

    # Méthodes de traitement des requêtes - Implémentations complètes

    async def _design_governance_framework(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Conçoit un framework de gouvernance personnalisé."""
        
        # Extraire les paramètres de l'intention
        target_domains = [GovernanceDomain(d) for d in intent.get("domains", ["strategic_governance"])]
        org_size = intent.get("organization_size", "medium")
        sector = intent.get("sector", "technology")
        
        # Contexte organisationnel
        org_context = OrganizationalContext(
            sector=sector,
            size=org_size,
            complexity="moderate",
            regulatory_environment="moderate",
            transformation_stage="evolving",
            risk_appetite="moderate",
            digital_maturity="intermediate"
        )
        
        # Objectifs stratégiques (simulés)
        strategic_objectives = [
            "Améliorer la gouvernance des données",
            "Renforcer la gestion des risques",
            "Optimiser la prise de décision",
            "Assurer la conformité réglementaire"
        ]
        
        constraints = {"budget": "modéré", "timeline": "12-18 mois", "resources": "limitées"}
        
        # Conception du framework
        framework = await self.design_governance_framework(
            org_context, strategic_objectives, constraints, target_domains
        )
        
        # Synthèse par LLM
        synthesis_prompt = f"""
Présente ce framework de gouvernance pour répondre à: "{query.query_text}"

FRAMEWORK CONÇU:
- Nom: {framework.name}
- Description: {framework.description}
- Domaines couverts: {[d.value for d in framework.domains_covered]}
- Phases d'implémentation: {len(framework.implementation_roadmap)}

CONTEXTE:
- Secteur: {org_context.sector}
- Taille: {org_context.size}
- Complexité: {org_context.complexity}

Présente une synthèse exécutive du framework avec approche d'implémentation.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        return AgentResponse(
            content=response,
            tools_used=["framework_design"],
            context_used=True,
            sources=[],
            metadata={
                "framework_name": framework.name,
                "domains_covered": [d.value for d in framework.domains_covered],
                "implementation_phases": len(framework.implementation_roadmap),
                "organization_context": {
                    "sector": org_context.sector,
                    "size": org_context.size
                }
            }
        )

    async def _analyze_strategic_alignment(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Analyse l'alignement stratégique de la gouvernance."""
        
        # Collecte d'informations contextuelles
        relevant_docs = await self.document_finder.search_documents(
            f"gouvernance stratégie alignement {query.query_text}",
            limit=15
        )
        
        # Extraction d'entités de gouvernance
        governance_entities = []
        for doc in relevant_docs[:5]:
            content = doc.get("content", "")
            if content:
                entities = await self.entity_extractor.extract_entities(
                    content,
                    entity_types=[EntityType.CONTROL, EntityType.REQUIREMENT],
                    framework_context="governance"
                )
                governance_entities.extend(entities.get("control", []))
                governance_entities.extend(entities.get("requirement", []))
        
        # Analyse d'alignement par LLM
        alignment_prompt = f"""
Analyse l'alignement stratégique de la gouvernance pour: "{query.query_text}"

DOCUMENTS ANALYSÉS: {len(relevant_docs)}
ENTITÉS GOUVERNANCE: {len(governance_entities)}

CONTEXTE ORGANISATIONNEL:
{json.dumps(query.context or {}, indent=2)[:1000]}

En tant qu'expert Chief Governance Officer, analyse:

1. ALIGNEMENT STRATÉGIQUE ACTUEL:
   - Cohérence gouvernance-stratégie
   - Mécanismes d'alignement en place
   - Gaps d'alignement identifiés
   - Efficacité des processus de décision

2. GOUVERNANCE STRATÉGIQUE:
   - Structure de gouvernance du changement
   - Pilotage des initiatives stratégiques
   - Mesure de la performance stratégique
   - Communication et engagement

3. RECOMMANDATIONS D'AMÉLIORATION:
   - Optimisation de l'alignement
   - Renforcement des mécanismes
   - Amélioration des processus
   - Formation et développement

4. PLAN D'ACTION:
   - Actions prioritaires (top 3)
   - Timeline d'implémentation
   - Ressources requises
   - Mesures de succès

Fournis une analyse stratégique complète avec recommandations C-level.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": alignment_prompt}
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
                "governance_entities": len(governance_entities),
                "analysis_type": "strategic_alignment"
            }
        )

    async def _generate_transformation_roadmap(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Génère une roadmap de transformation de gouvernance."""
        
        # Paramètres de l'intention
        target_domains = [GovernanceDomain(d) for d in intent.get("domains", ["strategic_governance"])]
        org_profile = query.context.get("organization", {}) if query.context else {}
        
        # Contexte organisationnel
        org_context = OrganizationalContext(
            sector=org_profile.get("sector", "technology"),
            size=org_profile.get("size", "medium"),
            complexity=org_profile.get("complexity", "moderate"),
            regulatory_environment=org_profile.get("regulatory_environment", "moderate"),
            transformation_stage=org_profile.get("transformation_stage", "evolving"),
            risk_appetite=org_profile.get("risk_appetite", "moderate"),
            digital_maturity=org_profile.get("digital_maturity", "intermediate")
        )
        
        # Évaluation de maturité actuelle
        assessments = await self.assess_governance_maturity(target_domains, org_context)
        
        # Génération de recommandations exécutives
        recommendations = await self.generate_executive_recommendations(
            assessments, org_context, ["transformation", "efficiency", "compliance"]
        )
        
        # Synthèse de la roadmap par LLM
        roadmap_prompt = f"""
Génère une roadmap de transformation de gouvernance pour: "{query.query_text}"

ÉVALUATIONS MATURITÉ:
{json.dumps([{
    "domain": a.domain.value,
    "maturity": a.maturity_level.value,
    "score": a.maturity_score,
    "gaps": len(a.gaps)
} for a in assessments], indent=2)}

RECOMMANDATIONS EXÉCUTIVES: {len(recommendations)}

CONTEXTE ORGANISATION:
- Secteur: {org_context.sector}
- Taille: {org_context.size}
- Stage transformation: {org_context.transformation_stage}

En tant qu'expert en transformation organisationnelle, crée:

1. ROADMAP DE TRANSFORMATION:
   - Vision et objectifs de transformation
   - Phases de transformation (court/moyen/long terme)
   - Jalons critiques et livrables
   - Timeline stratégique

2. PRIORISATION STRATÉGIQUE:
   - Domaines prioritaires
   - Quick wins identifiées
   - Initiatives structurantes
   - Séquencement optimal

3. PLAN DE CONDUITE DU CHANGEMENT:
   - Stratégie de communication
   - Gestion des résistances
   - Formation et développement
   - Engagement des parties prenantes

4. MESURE ET PILOTAGE:
   - KPIs de transformation
   - Tableaux de bord
   - Points de contrôle
   - Ajustements nécessaires

Présente une roadmap stratégique et actionnable.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": roadmap_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        return AgentResponse(
            content=response,
            tools_used=["maturity_assessment", "executive_recommendations"],
            context_used=True,
            sources=[],
            metadata={
                "domains_assessed": [d.value for d in target_domains],
                "average_maturity": sum(a.maturity_score for a in assessments) / len(assessments),
                "executive_recommendations": len(recommendations),
                "transformation_stage": org_context.transformation_stage
            }
        )

    async def _provide_executive_advisory(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Fournit un conseil exécutif en gouvernance."""
        
        # Niveau organisationnel visé
        target_level = intent.get("level", "executive")
        priority = intent.get("priority", "strategic")
        
        # Collecte d'informations contextuelles
        relevant_docs = await self.document_finder.search_documents(
            f"gouvernance executive direction {query.query_text}",
            limit=10
        )
        
        # Analyse temporelle des tendances
        trends = await self.temporal_analyzer.analyze_trends(
            [doc.get("content", "") for doc in relevant_docs[:3]],
            MetricType.GOVERNANCE,
            time_window_months=12
        )
        
        # Conseil exécutif par LLM
        advisory_prompt = f"""
Fournis un conseil exécutif expert pour: "{query.query_text}"

NIVEAU CIBLE: {target_level}
PRIORITÉ: {priority}

DOCUMENTS ANALYSÉS: {len(relevant_docs)}
TENDANCES IDENTIFIÉES: {len(trends.data_points) if trends else 0}

CONTEXTE ORGANISATIONNEL:
{json.dumps(query.context or {}, indent=2)[:1000]}

En tant que Chairman/CEO advisor avec 25+ ans d'expérience, fournis:

1. DIAGNOSTIC EXÉCUTIF:
   - Situation actuelle de la gouvernance
   - Enjeux stratégiques critiques
   - Opportunités et menaces
   - Positionnement concurrentiel

2. RECOMMANDATIONS STRATÉGIQUES:
   - Actions immédiates (Board level)
   - Initiatives stratégiques
   - Transformations nécessaires
   - Investissements prioritaires

3. IMPLICATIONS BUSINESS:
   - Impact sur la performance
   - Risques organisationnels
   - Avantages concurrentiels
   - ROI attendu

4. PLAN EXÉCUTIF:
   - Decisions clés à prendre
   - Timeline de mise en œuvre
   - Governance et pilotage
   - Communication stakeholders

5. FACTEURS CRITIQUES:
   - Conditions de réussite
   - Risques d'échec
   - Mitigation des risques
   - Surveillance recommandée

Pense comme un Chairman s'adressant au Board avec vision stratégique.
Fournis des conseils actionables et orientés résultats business.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": advisory_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
        return AgentResponse(
            content=response,
            tools_used=["document_finder", "temporal_analyzer"],
            context_used=True,
            sources=[doc.get("title", "Document") for doc in relevant_docs[:5]],
            metadata={
                "advisory_level": target_level,
                "priority": priority,
                "documents_analyzed": len(relevant_docs),
                "trends_analyzed": len(trends.data_points) if trends else 0
            }
        )

    async def _general_governance_analysis(self, query: Query, intent: Dict[str, Any]) -> AgentResponse:
        """Effectue une analyse générale de gouvernance."""
        
        # Collecte d'informations multiples
        relevant_docs = await self.document_finder.search_documents(
            f"gouvernance management direction {query.query_text}",
            limit=15
        )
        
        # Extraction d'entités de gouvernance
        governance_entities = []
        for doc in relevant_docs[:5]:
            content = doc.get("content", "")
            if content:
                entities = await self.entity_extractor.extract_entities(
                    content,
                    entity_types=[EntityType.CONTROL, EntityType.REQUIREMENT, EntityType.RISK],
                    framework_context="governance"
                )
                governance_entities.extend(entities.get("control", []))
                governance_entities.extend(entities.get("requirement", []))
                governance_entities.extend(entities.get("risk", []))
        
        # Analyse croisée
        cross_refs = await self.cross_reference_tool.analyze_relationships(
            [doc.get("content", "") for doc in relevant_docs[:3]],
            relation_types=[RelationType.CONTROLS_RISK, RelationType.SUPPORTS]
        )
        
        # Analyse générale par LLM
        analysis_prompt = f"""
Effectue une analyse générale de gouvernance pour: "{query.query_text}"

DOCUMENTS ANALYSÉS: {len(relevant_docs)}
ENTITÉS GOUVERNANCE: {len(governance_entities)}
RELATIONS IDENTIFIÉES: {len(cross_refs.relationships)}

CONTEXTE ORGANISATIONNEL:
{json.dumps(query.context or {}, indent=2)[:1000]}

En tant qu'expert Chief Governance Officer, analyse:

1. ÉTAT DE LA GOUVERNANCE:
   - Maturité organisationnelle générale
   - Structures de gouvernance en place
   - Processus de prise de décision
   - Culture de gouvernance

2. DOMAINES D'AMÉLIORATION:
   - Lacunes identifiées
   - Opportunités d'optimisation
   - Risques de gouvernance
   - Priorités d'action

3. RECOMMANDATIONS STRATÉGIQUES:
   - Améliorations immédiates
   - Transformations moyen terme
   - Vision long terme
   - Approche d'implémentation

4. BENCHMARKING ET BONNES PRATIQUES:
   - Positionnement sectoriel
   - Meilleures pratiques applicables
   - Innovations en gouvernance
   - Tendances émergentes

5. PLAN D'ACTION:
   - Actions prioritaires (top 5)
   - Timeline recommandée
   - Ressources nécessaires
   - Mesures de succès

Fournis une analyse experte complète avec perspective C-level.
"""
        
        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["governance_strategist"]},
                {"role": "user", "content": analysis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.3
        )
        
        return AgentResponse(
            content=response,
            tools_used=["document_finder", "entity_extractor", "cross_reference_tool"],
            context_used=True,
            sources=[doc.get("title", "Document") for doc in relevant_docs[:5]],
            metadata={
                "documents_analyzed": len(relevant_docs),
                "governance_entities": len(governance_entities),
                "relationships_found": len(cross_refs.relationships),
                "analysis_scope": "general_governance"
            }
        )


# Factory function
def get_governance_analysis_module(llm_client: LLMIntegration = None):
    """Factory function pour obtenir une instance du module d'analyse de gouvernance."""
    return GovernanceAnalysisModule(llm_client=llm_client) 