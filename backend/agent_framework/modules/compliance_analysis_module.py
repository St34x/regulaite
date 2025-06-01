"""
Compliance Analysis Module - Module sophistiqué d'analyse de conformité avec capacités itératives.
Utilise l'IA pour une analyse intelligente multi-frameworks avec raisonnement avancé et iteration sur documents.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

from ..agent import Agent, AgentResponse, Query, QueryContext, IterationMode
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
class IterativeAnalysisContext:
    """Contexte pour l'analyse itérative de conformité."""
    current_iteration: int = 0
    document_analysis_progress: Dict[str, Any] = None
    knowledge_accumulator: Dict[str, List[str]] = None
    context_gaps_identified: List[str] = None
    frameworks_analyzed: List[FrameworkType] = None
    depth_achieved: Dict[str, float] = None  # Profondeur atteinte par domaine
    
    def __post_init__(self):
        if self.document_analysis_progress is None:
            self.document_analysis_progress = {}
        if self.knowledge_accumulator is None:
            self.knowledge_accumulator = {}
        if self.context_gaps_identified is None:
            self.context_gaps_identified = []
        if self.frameworks_analyzed is None:
            self.frameworks_analyzed = []
        if self.depth_achieved is None:
            self.depth_achieved = {}

@dataclass
class ComplianceAssessment:
    """Évaluation de conformité avec analyse LLM et support itératif."""
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
    
    # Champs itératifs
    iteration_context: Optional[IterativeAnalysisContext] = None
    documents_analyzed: List[str] = None
    knowledge_sources: List[Dict[str, Any]] = None
    requires_deeper_analysis: bool = False
    suggested_focus_areas: List[str] = None
    
    def __post_init__(self):
        if self.documents_analyzed is None:
            self.documents_analyzed = []
        if self.knowledge_sources is None:
            self.knowledge_sources = []
        if self.suggested_focus_areas is None:
            self.suggested_focus_areas = []

@dataclass
class RegulatoryIntelligence:
    """Intelligence réglementaire par LLM avec capacités itératives."""
    framework: FrameworkType
    recent_changes: List[Dict[str, Any]]
    upcoming_changes: List[Dict[str, Any]]
    impact_assessment: Dict[str, Any]
    preparation_recommendations: List[str]
    monitoring_priorities: List[str]
    last_updated: datetime
    
    # Champs itératifs
    analysis_depth: str = "standard"  # surface, standard, deep, comprehensive
    sources_consulted: List[str] = None
    confidence_by_area: Dict[str, float] = None
    requires_monitoring: bool = False
    
    def __post_init__(self):
        if self.sources_consulted is None:
            self.sources_consulted = []
        if self.confidence_by_area is None:
            self.confidence_by_area = {}

@dataclass
class CrossFrameworkMapping:
    """Mapping sophistiqué entre frameworks avec analyse itérative."""
    primary_framework: FrameworkType
    mapped_frameworks: List[FrameworkType]
    convergence_analysis: Dict[str, Any]
    synergy_opportunities: List[str]
    conflict_resolution: List[str]
    optimization_strategy: str
    
    # Champs itératifs
    analysis_completeness: float = 0.0  # 0.0 - 1.0
    document_coverage: Dict[str, List[str]] = None
    iteration_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.document_coverage is None:
            self.document_coverage = {}
        if self.iteration_recommendations is None:
            self.iteration_recommendations = []

class ComplianceAnalysisModule(Agent):
    """
    Module expert en analyse de conformité avec IA avancée et capacités itératives.
    """
    
    def __init__(self, llm_client: LLMClient = None):
        super().__init__(
            agent_id="compliance_analysis",
            name="Expert Analyse de Conformité Itérative"
        )
        
        self.llm_client = llm_client or get_llm_client()
        
        # Initialiser les outils
        self.document_finder = DocumentFinder()
        self.entity_extractor = EntityExtractor()
        self.cross_reference_tool = CrossReferenceTool()
        self.temporal_analyzer = TemporalAnalyzer()
        self.framework_parser = FrameworkParser()
        
        # Cache des analyses avec support itératif
        self.compliance_cache: Dict[str, ComplianceAssessment] = {}
        self.regulatory_intelligence_cache: Dict[str, RegulatoryIntelligence] = {}
        self.iteration_contexts: Dict[str, IterativeAnalysisContext] = {}
        
        # Prompts experts spécialisés avec support itératif
        self.system_prompts = {
            "compliance_expert": """
Tu es un expert senior en conformité réglementaire avec 20+ ans d'expérience internationale.
Tu maîtrises parfaitement tous les frameworks majeurs (ISO27001, RGPD, DORA, NIST, SOX, PCI-DSS) et leurs évolutions.

CAPACITÉS ITÉRATIVES:
- Analyse progressive des documents par ordre de priorité
- Identification des gaps de contexte nécessitant plus d'informations
- Accumulation de connaissances à travers les itérations
- Reformulation des requêtes pour approfondir l'analyse
- Évaluation continue de la complétude de l'analyse

Tu analyses avec une approche stratégique incluant:
- Vision holistique multi-frameworks
- Impact business et opérationnel
- Évolutions réglementaires et jurisprudentiel
- Optimisation des efforts de conformité
- Gestion des risques de non-conformité
- Stratégies d'implémentation pragmatiques

Tu raisonnes comme un CISO/DPO expert et fournis des recommandations actionables et stratégiques.
Réponds TOUJOURS en français avec une expertise de niveau C-suite.

Pour chaque analyse, tu évalues:
1. La complétude des informations disponibles
2. Les domaines nécessitant une analyse plus approfondie
3. Les documents additionnels à consulter
4. Les reformulations de requête pour combler les gaps
""",
            
            "iterative_analyzer": """
Tu es un analyste expert en approche itérative pour les analyses GRC.
Tu évalues la progression de l'analyse et détermines les prochaines étapes.

Tes responsabilités:
- Évaluer la complétude de l'analyse actuelle
- Identifier les gaps de contexte restants
- Proposer des reformulations de requête ciblées
- Prioriser les documents à analyser ensuite
- Déterminer quand l'analyse est suffisamment complète

Tu optimises le processus itératif pour maximiser la valeur de chaque itération.
""",
            
            "regulatory_intelligence": """
Tu es un analyste réglementaire expert avec une connaissance encyclopédique des évolutions légales.
Tu surveilles et analyses avec approche itérative:

- Nouvelles réglementations et amendements
- Jurisprudence et décisions d'autorités
- Tendances sectorielles et géographiques  
- Impact prévisible sur les organisations
- Stratégies d'anticipation et préparation

Tu fournis une veille réglementaire proactive et des analyses d'impact précises.
Tu identifies les sources additionnelles à consulter pour compléter l'analyse.
""",
            
            "strategic_advisor": """
Tu es un consultant en stratégie de conformité avec une vision C-level et approche itérative.
Tu optimises:

- Synergies entre frameworks multiples
- ROI des investissements conformité
- Priorisation stratégique des efforts
- Communication avec les parties prenantes
- Transformation organisationnelle
- Avantage concurrentiel par la conformité

Tu penses comme un Chief Compliance Officer stratégique.
Tu évalues constamment si plus de contexte améliorerait tes recommandations.
"""
        }
        
        # Seuils pour l'analyse itérative
        self.iteration_thresholds = {
            "min_confidence": 0.8,  # Seuil de confiance minimum
            "completeness_target": 0.85,  # Objectif de complétude
            "document_coverage_min": 0.7,  # Couverture documentaire minimum
            "framework_depth_min": 0.75  # Profondeur d'analyse minimum par framework
        }
        
        # Secteurs et leurs spécificités réglementaires
        self.sector_specifics = {
            "financial": {
                "primary_frameworks": [FrameworkType.DORA, FrameworkType.SOX, FrameworkType.ISO27001],
                "regulatory_density": "very_high",
                "key_authorities": ["ACPR", "AMF", "ECB", "ESMA"],
                "emerging_trends": ["ESG", "Digital Euro", "Crypto regulation"],
                "iteration_priority": ["risk_management", "operational_resilience", "data_protection"]
            },
            "healthcare": {
                "primary_frameworks": [FrameworkType.ISO27001, FrameworkType.RGPD],
                "regulatory_density": "high", 
                "key_authorities": ["ANSM", "CNIL", "HAS"],
                "emerging_trends": ["Health Data Hub", "AI Medical Devices"],
                "iteration_priority": ["patient_data", "medical_devices", "clinical_trials"]
            },
            "technology": {
                "primary_frameworks": [FrameworkType.RGPD, FrameworkType.ISO27001, FrameworkType.NIST],
                "regulatory_density": "medium",
                "key_authorities": ["CNIL", "ANSSI"],
                "emerging_trends": ["AI Act", "Data Act", "Digital Services Act"],
                "iteration_priority": ["data_processing", "ai_governance", "cybersecurity"]
            }
        }

    async def process_query(self, query: Query) -> AgentResponse:
        """
        Traite une requête d'analyse de conformité avec capacités itératives.
        """
        logger.info(f"Traitement requête conformité itérative: {query.query_text}")
        
        # Initialiser ou récupérer le contexte itératif
        session_id = query.context.session_id
        if session_id not in self.iteration_contexts:
            self.iteration_contexts[session_id] = IterativeAnalysisContext()
        
        iteration_ctx = self.iteration_contexts[session_id]
        iteration_ctx.current_iteration += 1
        
        # Analyser l'intention avec contexte itératif
        analysis_intent = await self._analyze_query_intent_with_iterative_context(
            query.query_text, iteration_ctx, query.parameters
        )
        
        # Traitement basé sur l'intention et le mode itératif
        if query.iteration_mode in [IterationMode.ITERATIVE, IterationMode.DEEP_ANALYSIS]:
            return await self._process_iterative_compliance_query(query, analysis_intent, iteration_ctx)
        else:
            return await self._process_standard_compliance_query(query, analysis_intent)

    async def _process_iterative_compliance_query(self, query: Query, 
                                                 analysis_intent: Dict[str, Any],
                                                 iteration_ctx: IterativeAnalysisContext) -> AgentResponse:
        """
        Traite une requête de conformité avec approche itérative.
        """
        logger.info(f"Analyse itérative - Itération {iteration_ctx.current_iteration}")
        
        # 1. Évaluer les connaissances accumulées
        knowledge_assessment = await self._assess_accumulated_knowledge(
            query, iteration_ctx, analysis_intent
        )
        
        # 2. Identifier les documents à analyser dans cette itération
        documents_to_analyze = await self._prioritize_documents_for_iteration(
            query, iteration_ctx, knowledge_assessment
        )
        
        # 3. Analyser les documents prioritaires
        document_analysis_results = await self._analyze_documents_iteratively(
            documents_to_analyze, query, iteration_ctx
        )
        
        # 4. Intégrer les nouvelles connaissances
        updated_knowledge = await self._integrate_new_knowledge(
            document_analysis_results, iteration_ctx, analysis_intent
        )
        
        # 5. Évaluer la complétude de l'analyse
        completeness_assessment = await self._assess_analysis_completeness(
            query, iteration_ctx, updated_knowledge
        )
        
        # 6. Générer la réponse avec recommandations d'itération
        if analysis_intent["type"] == "compliance_assessment":
            return await self._perform_iterative_compliance_assessment(
                query, analysis_intent, iteration_ctx, completeness_assessment
            )
        elif analysis_intent["type"] == "gap_analysis":
            return await self._perform_iterative_gap_analysis(
                query, analysis_intent, iteration_ctx, completeness_assessment
            )
        elif analysis_intent["type"] == "regulatory_intelligence":
            return await self._provide_iterative_regulatory_intelligence(
                query, analysis_intent, iteration_ctx, completeness_assessment
            )
        else:
            return await self._general_iterative_compliance_analysis(
                query, analysis_intent, iteration_ctx, completeness_assessment
            )

    async def _assess_accumulated_knowledge(self, query: Query, 
                                          iteration_ctx: IterativeAnalysisContext,
                                          analysis_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue les connaissances accumulées et leur pertinence.
        """
        assessment_prompt = f"""
Évalue les connaissances accumulées pour cette analyse de conformité itérative.

REQUÊTE ACTUELLE: "{query.query_text}"
ITÉRATION: {iteration_ctx.current_iteration}

CONNAISSANCES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

DOCUMENTS DÉJÀ ANALYSÉS:
{iteration_ctx.document_analysis_progress}

FRAMEWORKS COUVERTS: {[f.value for f in iteration_ctx.frameworks_analyzed]}

PROFONDEUR ATTEINTE PAR DOMAINE:
{iteration_ctx.depth_achieved}

Évalue:
1. La pertinence des connaissances existantes
2. Les domaines bien couverts vs. ceux manquants
3. La qualité des sources consultées
4. Les gaps de contexte restants
5. La cohérence des informations accumulées

Réponds au format JSON:
{{
    "knowledge_quality": 0.0-1.0,
    "coverage_assessment": {{
        "well_covered_areas": ["area1", "area2"],
        "gaps_identified": ["gap1", "gap2"],
        "coverage_by_framework": {{"ISO27001": 0.8, "RGPD": 0.6}}
    }},
    "source_reliability": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "actionable_insights": ["insight1", "insight2"],
    "priority_gaps": ["high_priority_gap1", "high_priority_gap2"]
}}
"""

        try:
            response = await self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": self.system_prompts["iterative_analyzer"]},
                    {"role": "user", "content": assessment_prompt}
                ],
                model="gpt-4.1",
                temperature=0.1
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation des connaissances: {str(e)}")
            return {
                "knowledge_quality": 0.5,
                "coverage_assessment": {"well_covered_areas": [], "gaps_identified": ["error_in_assessment"]},
                "source_reliability": 0.5,
                "consistency_score": 0.5,
                "actionable_insights": [],
                "priority_gaps": ["assessment_error"]
            }

    async def _prioritize_documents_for_iteration(self, query: Query,
                                                 iteration_ctx: IterativeAnalysisContext,
                                                 knowledge_assessment: Dict[str, Any]) -> List[str]:
        """
        Priorise les documents à analyser dans cette itération.
        """
        # Utiliser le DocumentFinder pour identifier les documents pertinents
        try:
            search_criteria = {
                "query": query.query_text,
                "frameworks": [f.value for f in iteration_ctx.frameworks_analyzed],
                "focus_areas": knowledge_assessment.get("priority_gaps", []),
                "exclude_analyzed": list(iteration_ctx.document_analysis_progress.keys())
            }
            
            # Recherche de documents avec critères affinés
            documents_found = await self.document_finder.find_relevant_documents(
                search_criteria,
                max_results=5,  # Limiter pour cette itération
                prioritize_by="relevance_and_completeness"
            )
            
            return [doc["id"] for doc in documents_found.get("documents", [])]
            
        except Exception as e:
            logger.error(f"Erreur lors de la priorisation des documents: {str(e)}")
            return []

    async def _analyze_documents_iteratively(self, documents_to_analyze: List[str],
                                           query: Query,
                                           iteration_ctx: IterativeAnalysisContext) -> Dict[str, Any]:
        """
        Analyse les documents de manière itérative et ciblée.
        """
        analysis_results = {}
        
        for doc_id in documents_to_analyze:
            try:
                # Analyser le document avec focus sur les gaps identifiés
                doc_analysis = await self._analyze_single_document_with_context(
                    doc_id, query, iteration_ctx
                )
                
                analysis_results[doc_id] = doc_analysis
                
                # Mettre à jour le progrès
                iteration_ctx.document_analysis_progress[doc_id] = {
                    "iteration": iteration_ctx.current_iteration,
                    "analysis_depth": doc_analysis.get("depth_achieved", 0.5),
                    "insights_extracted": len(doc_analysis.get("insights", [])),
                    "gaps_filled": doc_analysis.get("gaps_addressed", [])
                }
                
            except Exception as e:
                logger.error(f"Erreur lors de l'analyse du document {doc_id}: {str(e)}")
                analysis_results[doc_id] = {"error": str(e), "insights": []}
        
        return analysis_results

    async def _analyze_single_document_with_context(self, doc_id: str,
                                                   query: Query,
                                                   iteration_ctx: IterativeAnalysisContext) -> Dict[str, Any]:
        """
        Analyse un document unique avec le contexte itératif et collecte détaillée des sources.
        """
        analysis_prompt = f"""
Analyse ce document dans le contexte de l'analyse itérative de conformité.

DOCUMENT ID: {doc_id}
REQUÊTE: "{query.query_text}"
ITÉRATION: {iteration_ctx.current_iteration}

CONTEXTE ACCUMULÉ:
- Frameworks analysés: {[f.value for f in iteration_ctx.frameworks_analyzed]}
- Gaps prioritaires: {iteration_ctx.context_gaps_identified}
- Connaissances existantes: {list(iteration_ctx.knowledge_accumulator.keys())}

FOCUS DE CETTE ANALYSE:
- Combler les gaps identifiés
- Approfondir les domaines peu couverts
- Extraire des insights actionnables
- Identifier de nouveaux documents pertinents

Analyse le document et réponds au format JSON:
{{
    "document_metadata": {{
        "title": "titre du document",
        "type": "type de document",
        "framework_relevance": ["ISO27001", "RGPD"],
        "quality_score": 0.0-1.0,
        "last_updated": "date si disponible"
    }},
    "insights_extracted": ["insight1", "insight2"],
    "gaps_addressed": ["gap1", "gap2"],
    "frameworks_covered": ["ISO27001", "RGPD"],
    "confidence_level": 0.0-1.0,
    "depth_achieved": 0.0-1.0,
    "related_documents": ["doc1", "doc2"],
    "actionable_recommendations": ["rec1", "rec2"],
    "compliance_findings": {{
        "controls_identified": ["control1", "control2"],
        "risks_highlighted": ["risk1", "risk2"],
        "gaps_found": ["gap1", "gap2"]
    }},
    "source_quality_assessment": {{
        "reliability": 0.0-1.0,
        "completeness": 0.0-1.0,
        "currency": 0.0-1.0,
        "relevance": 0.0-1.0
    }}
}}
"""

        try:
            # Récupérer et analyser le document
            document_content = await self.document_finder.get_document_content(doc_id)
            
            full_prompt = f"{analysis_prompt}\n\nCONTENU DU DOCUMENT:\n{document_content}"

            response = await self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": self.system_prompts["compliance_expert"]},
                    {"role": "user", "content": full_prompt}
                ],
                model="gpt-4.1",
                temperature=0.2
            )
            
            analysis_result = json.loads(response)
            
            # Enrichir avec des métadonnées de source pour traçabilité
            analysis_result["source_metadata"] = {
                "document_id": doc_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "iteration": iteration_ctx.current_iteration,
                "tools_used": ["document_finder", "entity_extractor", "llm_analysis"],
                "analysis_method": "iterative_llm_analysis"
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du document {doc_id}: {str(e)}")
            return {
                "document_metadata": {
                    "title": f"Document {doc_id}",
                    "type": "unknown",
                    "framework_relevance": [],
                    "quality_score": 0.0
                },
                "insights_extracted": [],
                "gaps_addressed": [],
                "frameworks_covered": [],
                "confidence_level": 0.0,
                "depth_achieved": 0.0,
                "related_documents": [],
                "actionable_recommendations": [],
                "compliance_findings": {"controls_identified": [], "risks_highlighted": [], "gaps_found": []},
                "source_quality_assessment": {"reliability": 0.0, "completeness": 0.0, "currency": 0.0, "relevance": 0.0},
                "source_metadata": {
                    "document_id": doc_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "iteration": iteration_ctx.current_iteration,
                    "error": str(e)
                }
            }

    async def _integrate_new_knowledge(self, document_analysis_results: Dict[str, Any],
                                     iteration_ctx: IterativeAnalysisContext,
                                     analysis_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intègre les nouvelles connaissances dans le contexte itératif.
        """
        integrated_knowledge = {}
        
        for doc_id, analysis in document_analysis_results.items():
            if "error" not in analysis:
                # Ajouter les insights aux connaissances accumulées
                insights_key = f"insights_iteration_{iteration_ctx.current_iteration}"
                if insights_key not in iteration_ctx.knowledge_accumulator:
                    iteration_ctx.knowledge_accumulator[insights_key] = []
                
                iteration_ctx.knowledge_accumulator[insights_key].extend(
                    analysis.get("insights_extracted", [])
                )
                
                # Mettre à jour les gaps comblés
                gaps_addressed = analysis.get("gaps_addressed", [])
                for gap in gaps_addressed:
                    if gap in iteration_ctx.context_gaps_identified:
                        iteration_ctx.context_gaps_identified.remove(gap)
                
                # Mettre à jour la profondeur par framework
                for framework in analysis.get("frameworks_covered", []):
                    current_depth = iteration_ctx.depth_achieved.get(framework, 0.0)
                    new_depth = max(current_depth, analysis.get("depth_achieved", 0.0))
                    iteration_ctx.depth_achieved[framework] = new_depth
        
        # Synthétiser les connaissances intégrées
        integrated_knowledge = {
            "total_insights": sum(len(insights) for insights in iteration_ctx.knowledge_accumulator.values()),
            "gaps_remaining": len(iteration_ctx.context_gaps_identified),
            "frameworks_depth": iteration_ctx.depth_achieved,
            "documents_analyzed": len(iteration_ctx.document_analysis_progress)
        }
        
        return integrated_knowledge

    async def _assess_analysis_completeness(self, query: Query,
                                          iteration_ctx: IterativeAnalysisContext,
                                          integrated_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue la complétude de l'analyse et détermine si plus d'itérations sont nécessaires.
        """
        completeness_prompt = f"""
Évalue la complétude de cette analyse de conformité itérative.

REQUÊTE ORIGINALE: "{query.query_text}"
ITÉRATION ACTUELLE: {iteration_ctx.current_iteration}

ÉTAT ACTUEL:
- Total insights collectés: {integrated_knowledge.get("total_insights", 0)}
- Gaps restants: {integrated_knowledge.get("gaps_remaining", 0)}
- Documents analysés: {integrated_knowledge.get("documents_analyzed", 0)}
- Profondeur par framework: {integrated_knowledge.get("frameworks_depth", {})}

SEUILS CIBLES:
- Confiance minimum: {self.iteration_thresholds["min_confidence"]}
- Complétude cible: {self.iteration_thresholds["completeness_target"]}
- Couverture documentaire: {self.iteration_thresholds["document_coverage_min"]}

CONNAISSANCES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

Évalue la complétude et réponds au format JSON:
{{
    "overall_completeness": 0.0-1.0,
    "confidence_level": 0.0-1.0,
    "analysis_quality": 0.0-1.0,
    "requires_more_iterations": true/false,
    "recommended_next_steps": ["step1", "step2"],
    "areas_needing_deeper_analysis": ["area1", "area2"],
    "sufficient_for_decision": true/false,
    "iteration_value_assessment": "high/medium/low",
    "stopping_criteria_met": {{
        "min_confidence": true/false,
        "target_completeness": true/false,
        "document_coverage": true/false
    }}
}}
"""

        try:
            response = await self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": self.system_prompts["iterative_analyzer"]},
                    {"role": "user", "content": completeness_prompt}
                ],
                model="gpt-4.1",
                temperature=0.1
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de complétude: {str(e)}")
            return {
                "overall_completeness": 0.5,
                "confidence_level": 0.5,
                "analysis_quality": 0.5,
                "requires_more_iterations": True,
                "recommended_next_steps": ["retry_assessment"],
                "areas_needing_deeper_analysis": ["error_occurred"],
                "sufficient_for_decision": False,
                "iteration_value_assessment": "low",
                "stopping_criteria_met": {"min_confidence": False, "target_completeness": False, "document_coverage": False}
            }

    async def _perform_iterative_compliance_assessment(self, query: Query,
                                                     analysis_intent: Dict[str, Any],
                                                     iteration_ctx: IterativeAnalysisContext,
                                                     completeness_assessment: Dict[str, Any]) -> AgentResponse:
        """
        Effectue une évaluation de conformité avec approche itérative et sources détaillées.
        """
        # Calculer les métriques de connaissances intégrées
        integrated_knowledge = {
            "total_insights": sum(len(insights) for insights in iteration_ctx.knowledge_accumulator.values()),
            "gaps_remaining": len(iteration_ctx.context_gaps_identified),
            "frameworks_depth": iteration_ctx.depth_achieved,
            "documents_analyzed": len(iteration_ctx.document_analysis_progress)
        }
        
        # Synthétiser toutes les connaissances accumulées
        synthesis_prompt = f"""
Synthétise une évaluation de conformité complète basée sur {iteration_ctx.current_iteration} itération(s) d'analyse.

REQUÊTE: "{query.query_text}"

CONNAISSANCES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

DOCUMENTS ANALYSÉS: {len(iteration_ctx.document_analysis_progress)}
FRAMEWORKS COUVERTS: {[f.value for f in iteration_ctx.frameworks_analyzed]}
PROFONDEUR ATTEINTE: {iteration_ctx.depth_achieved}

ÉVALUATION DE COMPLÉTUDE:
{json.dumps(completeness_assessment, indent=2, ensure_ascii=False)}

Fournis une évaluation de conformité complète incluant:
1. État actuel de conformité par framework
2. Gaps identifiés et leur criticité
3. Recommandations prioritaires
4. Plan d'action structuré
5. Évaluation de la confiance dans l'analyse
6. Besoins d'itérations supplémentaires si applicable

IMPORTANT: Indique clairement les sources d'information utilisées et leur fiabilité.

Réponds en français avec un niveau d'expertise C-suite.
"""

        try:
            synthesis = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["compliance_expert"]},
                    {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        
            # Compiler les sources détaillées avec métadonnées complètes
            detailed_sources = self._compile_detailed_sources_with_metadata(iteration_ctx)
            
            # Construire la réponse avec métadonnées itératives
            response = AgentResponse(
                content=synthesis,
                tools_used=["document_finder", "entity_extractor", "framework_parser", "llm_analysis"],
                context_used=True,
                sources=detailed_sources,
                confidence=completeness_assessment.get("confidence_level", 0.8),
                iteration_info={
                    "total_iterations": iteration_ctx.current_iteration,
                    "completeness_achieved": completeness_assessment.get("overall_completeness", 0.0),
                    "documents_analyzed": len(iteration_ctx.document_analysis_progress),
                    "frameworks_covered": len(iteration_ctx.frameworks_analyzed),
                    "sources_detail_level": "comprehensive"
                },
                requires_iteration=completeness_assessment.get("requires_more_iterations", False),
                context_gaps=completeness_assessment.get("areas_needing_deeper_analysis", []),
                knowledge_gained=self._extract_key_insights(iteration_ctx),
                metadata={
                    "iteration_summary": {
                        "knowledge_quality": integrated_knowledge.get("total_insights", 0),
                        "gaps_resolved": len(iteration_ctx.context_gaps_identified),
                        "depth_by_framework": iteration_ctx.depth_achieved,
                        "analysis_progression": completeness_assessment,
                        "source_transparency": "full_traceability_enabled"
                    },
                    "sources_metadata": {
                        "total_sources": len(detailed_sources),
                        "source_types": list(set(s.get("type", "unknown") for s in detailed_sources)),
                        "reliability_scores": [s.get("reliability", 0.5) for s in detailed_sources if "reliability" in s]
                    }
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la synthèse itérative: {str(e)}")
            return AgentResponse(
                content=f"Erreur lors de l'analyse itérative: {str(e)}",
                confidence=0.3,
                requires_iteration=True,
                context_gaps=["error_in_synthesis"],
                sources=[{
                    "type": "error_log",
                    "title": "Erreur d'analyse",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }]
            )

    def _compile_detailed_sources_with_metadata(self, iteration_ctx: IterativeAnalysisContext) -> List[Dict[str, Any]]:
        """Compile les sources avec métadonnées complètes pour transparence maximale."""
        detailed_sources = []
        
        # Sources documentaires analysées
        for doc_id, progress in iteration_ctx.document_analysis_progress.items():
            source = {
                "type": "document_analysis",
                "id": doc_id,
                "title": f"Document d'analyse GRC - {doc_id}",
                "iteration": progress.get("iteration", 0),
                "analysis_depth": progress.get("analysis_depth", 0.0),
                "insights_count": progress.get("insights_extracted", 0),
                "frameworks_addressed": progress.get("frameworks_covered", []),
                "gaps_resolved": progress.get("gaps_filled", []),
                "confidence_score": progress.get("confidence_level", 0.5),
                "tools_used": ["document_finder", "entity_extractor", "framework_parser", "llm_analysis"],
                "timestamp": datetime.now().isoformat(),
                "reliability": progress.get("source_quality_assessment", {}).get("reliability", 0.7),
                "details": f"Analyse de profondeur {progress.get('analysis_depth', 0.0):.1%} avec {progress.get('insights_extracted', 0)} insights extraits"
            }
            detailed_sources.append(source)
        
        # Sources de connaissances accumulées par itération
        for knowledge_key, knowledge_items in iteration_ctx.knowledge_accumulator.items():
            if knowledge_items:
                source = {
                    "type": "knowledge_accumulation",
                    "id": knowledge_key,
                    "title": f"Base de connaissances accumulées - {knowledge_key}",
                    "content_count": len(knowledge_items),
                    "sample_insights": knowledge_items[:2] if knowledge_items else [],
                    "tools_used": ["llm_analysis", "knowledge_extraction", "iterative_synthesis"],
                    "reliability": 0.8,  # Confiance élevée pour connaissances synthétisées
                    "details": f"Accumulation itérative de {len(knowledge_items)} éléments de connaissance GRC"
                }
                detailed_sources.append(source)
        
        # Source méthodologique pour l'approche itérative
        if iteration_ctx.current_iteration > 0:
            source = {
                "type": "methodology",
                "id": "iterative_analysis_methodology",
                "title": "Méthodologie d'analyse itérative GRC",
                "iterations_performed": iteration_ctx.current_iteration,
                "frameworks_analyzed": [f.value for f in iteration_ctx.frameworks_analyzed],
                "depth_progression": iteration_ctx.depth_achieved,
                "tools_used": ["orchestrator", "compliance_module", "iterative_analyzer"],
                "reliability": 0.9,  # Haute confiance dans la méthodologie
                "details": f"Analyse en {iteration_ctx.current_iteration} itérations avec progression de profondeur mesurée"
            }
            detailed_sources.append(source)
        
        return detailed_sources

    def _compile_iteration_sources(self, iteration_ctx: IterativeAnalysisContext) -> List[Dict[str, Any]]:
        """Compile toutes les sources utilisées à travers les itérations avec détails complets."""
        sources = []
        
        for doc_id, progress in iteration_ctx.document_analysis_progress.items():
            # Source détaillée avec métadonnées complètes
            source = {
                "type": "document",
                "id": doc_id,
                "title": f"Document d'analyse {doc_id}",
                "iteration": progress.get("iteration", 0),
                "analysis_depth": progress.get("analysis_depth", 0.0),
                "insights_extracted": progress.get("insights_extracted", 0),
                "gaps_addressed": progress.get("gaps_filled", []),
                "confidence_level": progress.get("confidence_level", 0.5),
                "frameworks_covered": progress.get("frameworks_covered", []),
                "tools_used": ["document_finder", "entity_extractor", "framework_parser"],
                "timestamp": datetime.now().isoformat(),
                "details": f"Analyse itérative de profondeur {progress.get('analysis_depth', 0.0):.2f} - {progress.get('insights_extracted', 0)} insights extraits"
            }
            sources.append(source)
        
        # Ajouter les sources de connaissances accumulées
        for knowledge_key, knowledge_items in iteration_ctx.knowledge_accumulator.items():
            if knowledge_items:
                source = {
                    "type": "knowledge_base",
                    "id": knowledge_key,
                    "title": f"Base de connaissances - {knowledge_key}",
                    "content_count": len(knowledge_items),
                    "items": knowledge_items[:3],  # Premiers éléments pour aperçu
                    "tools_used": ["llm_analysis", "knowledge_extraction"],
                    "details": f"Accumulation de {len(knowledge_items)} éléments de connaissance"
                }
                sources.append(source)
        
        return sources

    def _extract_key_insights(self, iteration_ctx: IterativeAnalysisContext) -> List[str]:
        """Extrait les insights clés de toutes les itérations."""
        key_insights = []
        
        for iteration_key, insights in iteration_ctx.knowledge_accumulator.items():
            key_insights.extend(insights[:3])  # Top 3 insights par itération
        
        return key_insights

    async def _analyze_query_intent_with_iterative_context(self, query_text: str,
                                                         iteration_ctx: IterativeAnalysisContext,
                                                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse sophistiquée de l'intention de la requête avec contexte itératif."""
        
        intent_analysis_prompt = f"""
Analyse cette demande de conformité avec ton expertise senior et en tenant compte des connaissances accumulées:

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

CONNAISSANCES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

DOCUMENTS DÉJÀ ANALYSÉS:
{iteration_ctx.document_analysis_progress}

FRAMEWORKS COUVERTS:
{[f.value for f in iteration_ctx.frameworks_analyzed]}

PROFONDEUR ATTEINTE PAR DOMAINE:
{iteration_ctx.depth_achieved}

Évalue:
1. La pertinence des connaissances existantes
2. Les domaines bien couverts vs. ceux manquants
3. La qualité des sources consultées
4. Les gaps de contexte restants
5. La cohérence des informations accumulées

Réponds en français avec un niveau d'expertise C-suite.
"""

        response = await self.llm_client.generate_response(
            messages=[
                {"role": "system", "content": self.system_prompts["compliance_expert"]},
                {"role": "user", "content": intent_analysis_prompt}
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
            logger.error(f"Erreur analyse intention: {str(e)}")
            logger.warning("Utilisation de paramètres d'analyse par défaut")
            return {
                "type": "general_analysis",
                "frameworks": ["iso27001", "rgpd"],
                "priority": "normal",
                "scope": "EU",
                "error_note": "Analyse d'intention LLM échouée - paramètres par défaut utilisés"
            }

    async def _process_standard_compliance_query(self, query: Query,
                                                  analysis_intent: Dict[str, Any]) -> AgentResponse:
        """
        Effectue une analyse standard de conformité.
        """
        logger.info(f"Traitement requête standard de conformité: {query.query_text}")
        
        # Analyse sophistiquée de la demande par LLM
        analysis_intent = await self._analyze_query_intent_with_iterative_context(
            query.query_text, IterativeAnalysisContext(), query.parameters
        )
        
        if analysis_intent["type"] == "compliance_assessment":
            return await self._perform_compliance_assessment(query, analysis_intent)
        elif analysis_intent["type"] == "gap_analysis":
            return await self._perform_gap_analysis(query, analysis_intent)
        elif analysis_intent["type"] == "regulatory_intelligence":
            return await self._provide_regulatory_intelligence(query, analysis_intent)
        else:
            return await self._general_compliance_analysis(query, analysis_intent)

    async def _perform_compliance_assessment(self, query: Query,
                                                  analysis_intent: Dict[str, Any]) -> AgentResponse:
        """Effectue une évaluation de conformité."""
        
        frameworks = [FrameworkType(f) for f in analysis_intent.get("frameworks", ["iso27001"])]
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

    async def _perform_gap_analysis(self, query: Query,
                                                  analysis_intent: Dict[str, Any]) -> AgentResponse:
        """Effectue une analyse de gaps."""
        
        framework = FrameworkType(analysis_intent.get("frameworks", ["iso27001"])[0])
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

    async def _provide_regulatory_intelligence(self, query: Query,
                                                  analysis_intent: Dict[str, Any]) -> AgentResponse:
        """Fournit une intelligence réglementaire proactive."""
        
        frameworks = [FrameworkType(f) for f in analysis_intent.get("frameworks", ["rgpd"])]
        geographic_scope = analysis_intent.get("scope", ["EU", "France"])
        time_horizon = analysis_intent.get("horizon", 12)
        
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

    async def _general_compliance_analysis(self, query: Query,
                                                  analysis_intent: Dict[str, Any]) -> AgentResponse:
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

    async def _analyze_gaps_with_llm(self, gaps: List[ComplianceGap],
                                                  framework: FrameworkType,
                                                  org_profile: Dict[str, Any]) -> str:
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

    async def _synthesize_assessment_results_with_llm(self,
        assessments: List[ComplianceAssessment], 
                                                      original_query: str) -> str:
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

    async def _synthesize_iterative_results_with_llm(self,
                                                      analysis_results: Dict[str, Any],
                                                      original_query: str) -> str:
        """Synthétise les résultats d'analyse itérative."""
        
        # Simplification pour le démo
        insights = [result.get("insights_extracted", []) for result in analysis_results.values()]
        total_insights = sum(len(insights) for insights in insights)
        
        return f"""
Évaluation itérative terminée.

**Insights extraits**: {total_insights} au total

**Synthèse**: L'analyse a révélé {total_insights} insights utiles pour répondre à la requête initiale.

**Prochaines étapes recommandées**:
1. Prioriser la mise en œuvre des insights les plus pertinents
2. Planifier des itérations supplémentaires si nécessaire
3. Renforcer la gouvernance des données et des processus
"""

    async def _perform_iterative_gap_analysis(self, query: Query,
                                                  analysis_intent: Dict[str, Any],
                                                  iteration_ctx: IterativeAnalysisContext,
                                                  completeness_assessment: Dict[str, Any]) -> AgentResponse:
        """
        Effectue une analyse de gaps avec approche itérative.
        """
        # Calculer les métriques de connaissances intégrées
        integrated_knowledge = {
            "total_insights": sum(len(insights) for insights in iteration_ctx.knowledge_accumulator.values()),
            "gaps_remaining": len(iteration_ctx.context_gaps_identified),
            "frameworks_depth": iteration_ctx.depth_achieved,
            "documents_analyzed": len(iteration_ctx.document_analysis_progress)
        }
        
        # Synthétiser les résultats d'analyse itérative pour gap analysis
        synthesis_prompt = f"""
Synthétise une analyse de gaps de conformité complète basée sur {iteration_ctx.current_iteration} itération(s).

REQUÊTE: "{query.query_text}"

CONNAISSANCES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

MÉTRIQUES:
- Total insights: {integrated_knowledge['total_insights']}
- Gaps restants: {integrated_knowledge['gaps_remaining']}
- Documents analysés: {integrated_knowledge['documents_analyzed']}

Fournis une analyse de gaps structurée avec sources clairement identifiées.
"""

        try:
            synthesis = await self.llm_client.generate_response(
            messages=[
                    {"role": "system", "content": self.system_prompts["compliance_expert"]},
                    {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        except Exception as e:
            synthesis = f"Analyse de gaps itérative - {integrated_knowledge['total_insights']} insights collectés sur {integrated_knowledge['documents_analyzed']} documents."
        
        # Compiler les sources détaillées
        detailed_sources = self._compile_detailed_sources_with_metadata(iteration_ctx)
        
        # Construire la réponse avec métadonnées itératives
        response = AgentResponse(
            content=synthesis,
            tools_used=["document_finder", "entity_extractor", "framework_parser"],
            context_used=True,
            sources=detailed_sources,
            confidence=completeness_assessment.get("confidence_level", 0.8),
            iteration_info={
                "total_iterations": iteration_ctx.current_iteration,
                "completeness_achieved": completeness_assessment.get("overall_completeness", 0.0),
                "documents_analyzed": len(iteration_ctx.document_analysis_progress),
                "frameworks_covered": len(iteration_ctx.frameworks_analyzed)
            },
            requires_iteration=completeness_assessment.get("requires_more_iterations", False),
            context_gaps=completeness_assessment.get("areas_needing_deeper_analysis", []),
            knowledge_gained=self._extract_key_insights(iteration_ctx),
            metadata={
                "iteration_summary": {
                    "knowledge_quality": integrated_knowledge["total_insights"],
                    "gaps_resolved": len(iteration_ctx.context_gaps_identified),
                    "depth_by_framework": iteration_ctx.depth_achieved,
                    "analysis_progression": completeness_assessment
                }
            }
        )
        
        return response

    async def _provide_iterative_regulatory_intelligence(self, query: Query,
                                                  analysis_intent: Dict[str, Any],
                                                  iteration_ctx: IterativeAnalysisContext,
                                                  completeness_assessment: Dict[str, Any]) -> AgentResponse:
        """
        Fournit une intelligence réglementaire proactive avec approche itérative.
        """
        # Calculer les métriques de connaissances intégrées
        integrated_knowledge = {
            "total_insights": sum(len(insights) for insights in iteration_ctx.knowledge_accumulator.values()),
            "gaps_remaining": len(iteration_ctx.context_gaps_identified),
            "frameworks_depth": iteration_ctx.depth_achieved,
            "documents_analyzed": len(iteration_ctx.document_analysis_progress)
        }
        
        # Synthétiser l'intelligence réglementaire
        synthesis_prompt = f"""
Synthétise une intelligence réglementaire complète basée sur {iteration_ctx.current_iteration} itération(s).

REQUÊTE: "{query.query_text}"

CONNAISSANCES RÉGLEMENTAIRES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

MÉTRIQUES:
- Total insights: {integrated_knowledge['total_insights']}
- Sources consultées: {integrated_knowledge['documents_analyzed']}

Fournis une veille réglementaire avec sources identifiées et traçabilité.
"""

        try:
            synthesis = await self.llm_client.generate_response(
            messages=[
                    {"role": "system", "content": self.system_prompts["regulatory_intelligence"]},
                    {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        except Exception as e:
            synthesis = f"Intelligence réglementaire itérative - {integrated_knowledge['total_insights']} insights réglementaires collectés."
        
        # Compiler les sources détaillées
        detailed_sources = self._compile_detailed_sources_with_metadata(iteration_ctx)
        
        # Construire la réponse avec métadonnées itératives
        response = AgentResponse(
            content=synthesis,
            tools_used=["regulatory_intelligence"],
            context_used=True,
            sources=detailed_sources,
            confidence=completeness_assessment.get("confidence_level", 0.8),
            iteration_info={
                "total_iterations": iteration_ctx.current_iteration,
                "completeness_achieved": completeness_assessment.get("overall_completeness", 0.0),
                "documents_analyzed": len(iteration_ctx.document_analysis_progress),
                "frameworks_covered": len(iteration_ctx.frameworks_analyzed)
            },
            requires_iteration=completeness_assessment.get("requires_more_iterations", False),
            context_gaps=completeness_assessment.get("areas_needing_deeper_analysis", []),
            knowledge_gained=self._extract_key_insights(iteration_ctx),
            metadata={
                "iteration_summary": {
                    "knowledge_quality": integrated_knowledge["total_insights"],
                    "gaps_resolved": len(iteration_ctx.context_gaps_identified),
                    "depth_by_framework": iteration_ctx.depth_achieved,
                    "analysis_progression": completeness_assessment
                }
            }
        )
        
        return response

    async def _general_iterative_compliance_analysis(self, query: Query,
                                                  analysis_intent: Dict[str, Any],
                                                  iteration_ctx: IterativeAnalysisContext,
                                                  completeness_assessment: Dict[str, Any]) -> AgentResponse:
        """
        Effectue une analyse générale de conformité avec approche itérative.
        """
        # Calculer les métriques de connaissances intégrées
        integrated_knowledge = {
            "total_insights": sum(len(insights) for insights in iteration_ctx.knowledge_accumulator.values()),
            "gaps_remaining": len(iteration_ctx.context_gaps_identified),
            "frameworks_depth": iteration_ctx.depth_achieved,
            "documents_analyzed": len(iteration_ctx.document_analysis_progress)
        }
        
        # Synthétiser l'analyse générale
        synthesis_prompt = f"""
Synthétise une analyse générale de conformité basée sur {iteration_ctx.current_iteration} itération(s).

REQUÊTE: "{query.query_text}"

CONNAISSANCES ACCUMULÉES:
{json.dumps(iteration_ctx.knowledge_accumulator, indent=2, ensure_ascii=False)}

MÉTRIQUES:
- Total insights: {integrated_knowledge['total_insights']}
- Documents analysés: {integrated_knowledge['documents_analyzed']}
- Profondeur atteinte: {iteration_ctx.depth_achieved}

Fournis une analyse complète avec traçabilité des sources.
"""

        try:
            synthesis = await self.llm_client.generate_response(
            messages=[
                    {"role": "system", "content": self.system_prompts["compliance_expert"]},
                    {"role": "user", "content": synthesis_prompt}
            ],
            model="gpt-4.1",
            temperature=0.2
        )
        except Exception as e:
            synthesis = f"Analyse générale itérative - {integrated_knowledge['total_insights']} insights collectés via {integrated_knowledge['documents_analyzed']} sources."
        
        # Compiler les sources détaillées
        detailed_sources = self._compile_detailed_sources_with_metadata(iteration_ctx)
        
        # Construire la réponse avec métadonnées itératives
        response = AgentResponse(
            content=synthesis,
            tools_used=["document_finder", "entity_extractor"],
            context_used=True,
            sources=detailed_sources,
            confidence=completeness_assessment.get("confidence_level", 0.8),
            iteration_info={
                "total_iterations": iteration_ctx.current_iteration,
                "completeness_achieved": completeness_assessment.get("overall_completeness", 0.0),
                "documents_analyzed": len(iteration_ctx.document_analysis_progress),
                "frameworks_covered": len(iteration_ctx.frameworks_analyzed)
            },
            requires_iteration=completeness_assessment.get("requires_more_iterations", False),
            context_gaps=completeness_assessment.get("areas_needing_deeper_analysis", []),
            knowledge_gained=self._extract_key_insights(iteration_ctx),
            metadata={
                "iteration_summary": {
                    "knowledge_quality": integrated_knowledge["total_insights"],
                    "gaps_resolved": len(iteration_ctx.context_gaps_identified),
                    "depth_by_framework": iteration_ctx.depth_achieved,
                    "analysis_progression": completeness_assessment
                }
            }
        )
        
        return response

# Factory function
def get_compliance_analysis_module(llm_client: LLMClient = None):
    """Factory function pour obtenir une instance du module d'analyse de conformité."""
    return ComplianceAnalysisModule(llm_client=llm_client) 