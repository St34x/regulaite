"""
Agent Orchestrateur Principal pour RegulAIte.
Analyse les requêtes en français et délègue aux agents spécialisés.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from .agent import Agent, AgentResponse, Query, QueryContext
from .integrations.llm_integration import LLMIntegration

# Backward compatibility alias
LLMClient = LLMIntegration

logger = logging.getLogger(__name__)

class OrchestratorAgent(Agent):
    """
    Agent orchestrateur principal qui analyse les requêtes et coordonne 
    les agents spécialisés pour les analyses GRC.
    """
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(
            agent_id="orchestrator", 
            name="Orchestrateur Principal GRC"
        )
        
        self.llm_client = llm_client
        self.specialized_agents: Dict[str, Agent] = {}
        
        # Prompts spécialisés pour l'orchestration
        self.system_prompt = """
Tu es l'orchestrateur principal d'un système d'analyse GRC (Gouvernance, Risque, Conformité).
Tu communiques en français et analyses les demandes pour déterminer quels agents spécialisés mobiliser.

Agents disponibles:
1. risk_assessment - Analyse des risques (identification, évaluation, traitement)
2. compliance_analysis - Analyse de conformité (RGPD, ISO 27001, DORA)  
3. governance_analysis - Analyse de gouvernance (politiques, procédures, contrôles)

Outils universels disponibles:
- document_finder - Recherche intelligente de documents
- entity_extractor - Extraction d'entités GRC
- cross_reference - Liens entre documents/contrôles
- temporal_analyzer - Analyse des tendances temporelles

Tu dois:
1. Analyser la demande utilisateur
2. Déterminer quels agents/outils mobiliser
3. Planifier l'ordre d'exécution
4. Coordonner la collaboration entre agents
5. Synthétiser les résultats

Réponds TOUJOURS en français avec un plan d'action structuré.
"""

    def register_agent(self, agent_id: str, agent: Agent):
        """Enregistre un agent spécialisé."""
        self.specialized_agents[agent_id] = agent
        logger.info(f"Agent {agent_id} enregistré dans l'orchestrateur")

    async def process_query(self, query: Union[str, Query]) -> AgentResponse:
        """
        Traite une requête utilisateur et orchestre les agents appropriés.
        """
        if isinstance(query, str):
            query = Query(query_text=query)
            
        logger.info(f"Orchestrateur - Analyse de la requête: {query.query_text}")
        
        # 1. Analyser la requête pour déterminer le plan d'action
        analysis_result = await self._analyze_request(query.query_text)
        
        # 2. Exécuter le plan d'action
        execution_results = await self._execute_plan(analysis_result, query)
        
        # 3. Synthétiser les résultats
        final_response = await self._synthesize_results(
            query.query_text, 
            analysis_result, 
            execution_results
        )
        
        return AgentResponse(
            content=final_response,
            tools_used=analysis_result.get("tools_used", []),
            context_used=True,
            sources=execution_results.get("all_sources", []),
            metadata={
                "orchestration_plan": analysis_result,
                "execution_summary": execution_results.get("summary", {}),
                "agents_involved": list(execution_results.get("agent_results", {}).keys())
            }
        )

    async def _analyze_request(self, user_query: str) -> Dict[str, Any]:
        """
        Analyse la requête utilisateur pour déterminer le plan d'action.
        """
        analysis_prompt = f"""
Analyse cette demande utilisateur et crée un plan d'action:

DEMANDE: "{user_query}"

Réponds au format JSON avec:
{{
    "type_analyse": "risk_assessment|compliance_analysis|governance_analysis|mixed",
    "frameworks_concernes": ["ISO27001", "RGPD", "DORA", "autres"],
    "agents_requis": [
        {{
            "agent_id": "nom_agent",
            "priorite": 1-3,
            "objectif": "description_objectif",
            "outils_necessaires": ["document_finder", "entity_extractor", ...]
        }}
    ],
    "collaboration_requise": true/false,
    "sequence_execution": [
        {{
            "etape": 1,
            "action": "description",
            "agent": "agent_id",
            "dependances": ["etape_precedente"]
        }}
    ],
    "objectif_final": "description_synthese_attendue"
}}
"""

        try:
            response = await self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                model="gpt-4.1",
                temperature=0.1
            )
            
            # Extraire le JSON de la réponse
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_content = response[json_start:json_end]
            
            return json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la requête: {str(e)}")
            # Plan de fallback
            return {
                "type_analyse": "mixed",
                "frameworks_concernes": ["ISO27001", "RGPD"],
                "agents_requis": [
                    {
                        "agent_id": "compliance_analysis",
                        "priorite": 1,
                        "objectif": "Analyse générale",
                        "outils_necessaires": ["document_finder", "entity_extractor"]
                    }
                ],
                "collaboration_requise": False,
                "sequence_execution": [
                    {
                        "etape": 1,
                        "action": "Analyse de conformité",
                        "agent": "compliance_analysis",
                        "dependances": []
                    }
                ],
                "objectif_final": "Analyse GRC générale"
            }

    async def _execute_plan(self, analysis: Dict[str, Any], original_query: Query) -> Dict[str, Any]:
        """
        Exécute le plan d'action en orchestrant les agents appropriés.
        """
        results = {
            "agent_results": {},
            "all_sources": [],
            "summary": {},
            "errors": []
        }
        
        try:
            # Trier les étapes par ordre d'exécution
            execution_sequence = sorted(
                analysis.get("sequence_execution", []),
                key=lambda x: x.get("etape", 0)
            )
            
            for step in execution_sequence:
                agent_id = step.get("agent")
                action = step.get("action")
                
                logger.info(f"Exécution étape {step.get('etape')}: {action} avec {agent_id}")
                
                if agent_id in self.specialized_agents:
                    try:
                        # Préparer le contexte pour l'agent spécialisé
                        context = QueryContext(
                            session_id=original_query.context.session_id,
                            metadata={
                                "orchestrator_plan": analysis,
                                "step_info": step,
                                "previous_results": results["agent_results"]
                            }
                        )
                        
                        # Créer une requête enrichie pour l'agent
                        enriched_query = Query(
                            query_text=original_query.query_text,
                            context=context,
                            parameters={
                                "specific_objective": step.get("objectif", action),
                                "required_tools": step.get("outils_necessaires", []),
                                "frameworks": analysis.get("frameworks_concernes", [])
                            }
                        )
                        
                        # Exécuter l'agent spécialisé
                        agent_response = await self.specialized_agents[agent_id].process_query(enriched_query)
                        
                        results["agent_results"][agent_id] = {
                            "response": agent_response.content,
                            "tools_used": agent_response.tools_used,
                            "sources": agent_response.sources,
                            "confidence": agent_response.confidence,
                            "step_info": step
                        }
                        
                        # Agréger les sources
                        results["all_sources"].extend(agent_response.sources)
                        
                    except Exception as e:
                        error_msg = f"Erreur lors de l'exécution de {agent_id}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
                else:
                    error_msg = f"Agent {agent_id} non trouvé"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)
            
            # Résumé de l'exécution
            results["summary"] = {
                "total_steps": len(execution_sequence),
                "successful_agents": len(results["agent_results"]),
                "total_sources": len(results["all_sources"]),
                "errors_count": len(results["errors"])
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du plan: {str(e)}")
            results["errors"].append(f"Erreur d'orchestration: {str(e)}")
            
        return results

    async def _synthesize_results(self, original_query: str, analysis: Dict[str, Any], 
                                 execution_results: Dict[str, Any]) -> str:
        """
        Synthétise les résultats de tous les agents en une réponse cohérente.
        """
        synthesis_prompt = f"""
Tu dois synthétiser les résultats d'une analyse GRC multi-agents.

DEMANDE ORIGINALE: "{original_query}"

PLAN D'ANALYSE:
{json.dumps(analysis, indent=2, ensure_ascii=False)}

RÉSULTATS DES AGENTS:
{json.dumps(execution_results.get("agent_results", {}), indent=2, ensure_ascii=False)}

ERREURS RENCONTRÉES:
{execution_results.get("errors", [])}

Crée une synthèse complète et structurée en français qui:
1. Répond directement à la demande utilisateur
2. Intègre les analyses de tous les agents
3. Présente les findings principaux
4. Identifie les gaps ou recommandations
5. Structure la réponse de manière claire et actionnable

La réponse doit être professionnelle et adaptée à un contexte GRC.
"""

        try:
            synthesis = await self.llm_client.generate_response(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                model="gpt-4.1",
                temperature=0.2
            )
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Erreur lors de la synthèse: {str(e)}")
            
            # Synthèse de fallback
            fallback_response = f"""
# Analyse GRC - Synthèse

## Demande analysée
{original_query}

## Résultats obtenus
"""
            
            for agent_id, result in execution_results.get("agent_results", {}).items():
                fallback_response += f"""
### {agent_id.replace('_', ' ').title()}
{result.get('response', 'Aucun résultat')}
"""
            
            if execution_results.get("errors"):
                fallback_response += f"""
## Erreurs rencontrées
{chr(10).join(execution_results["errors"])}
"""
            
            return fallback_response 