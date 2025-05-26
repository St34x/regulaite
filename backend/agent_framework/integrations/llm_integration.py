"""
LLM Integration for the RegulAIte Agent Framework.

This module provides integration with LLM services with specialized support for
French GRC (Governance, Risk, Compliance) analysis.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import json
import os
import sys
from pathlib import Path
import asyncio

# Set up logging
logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not found, falling back to HTTP requests")
    OPENAI_AVAILABLE = False

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    Returns language code (en, fr, es, etc.)
    """
    # Simple language detection based on common words and patterns
    text_lower = text.lower()
    
    # French indicators - enhanced with more common French words
    french_indicators = [
        # Articles and basic words
        'le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'et ', 'est ', 'un ', 'une ',
        'dans ', 'pour ', 'avec ', 'sur ', 'par ', 'ce ', 'qui ', 'que ', 'comment ',
        'où ', 'quand ', 'pourquoi ', 'qu\'', 'c\'', 'd\'', 'l\'', 'n\'', 'tion ',
        'ment ', 'ées ', 'ent ', 'sont ', 'ont', 'était', 'avait', 'sera',
        # Common French words often missed
        'bonjour', 'salut', 'bonsoir', 'au revoir', 'merci', 'oui', 'non',
        'je ', 'tu ', 'il ', 'elle ', 'nous ', 'vous ', 'ils ', 'elles ',
        'me ', 'te ', 'se ', 'lui ', 'leur ', 'mes ', 'tes ', 'ses ', 'nos ', 'vos ',
        'mon ', 'ton ', 'son ', 'ma ', 'ta ', 'sa ', 'notre ', 'votre ',
        'peux', 'peut', 'peuvent', 'pouvoir', 'veux', 'veut', 'vouloir',
        'suis', 'es', 'sommes', 'êtes', 'être', 'avoir', 'ai', 'as', 'a', 'avons', 'avez',
        'faire', 'fais', 'fait', 'faisons', 'faites', 'font',
        'dire', 'dis', 'dit', 'disons', 'dites', 'disent',
        'aller', 'vais', 'va', 'allons', 'allez', 'vont',
        'savoir', 'sais', 'sait', 'savons', 'savez', 'savent',
        'voir', 'vois', 'voit', 'voyons', 'voyez', 'voient',
        'donner', 'donne', 'donnes', 'donnons', 'donnez', 'donnent',
        'présenter', 'présente', 'présentes', 'présentons', 'présentez', 'présentent',
        'capacité', 'capacités', 'fonctionnalité', 'fonctionnalités',
        'système', 'service', 'aide', 'aider', 'assistance', 'question', 'réponse',
        # Professional/technical terms
        'sécurité', 'réseau', 'conformité', 'réglementation', 'politique', 'gestion', 'contrôle',
        'analyse', 'rapport', 'document', 'fichier', 'données', 'information'
    ]
    
    # Spanish indicators
    spanish_indicators = [
        'el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'y ', 'es ', 'un ', 'una ',
        'en ', 'con ', 'por ', 'para ', 'que ', 'como ', 'donde ', 'cuando ',
        'por qué ', 'cómo ', 'ción ', 'mente ', 'ado ', 'ida ', 'son ', 'han',
        'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'adiós', 'gracias', 'sí', 'no',
        'yo ', 'tú ', 'él ', 'ella ', 'nosotros ', 'vosotros ', 'ellos ', 'ellas ',
        'me ', 'te ', 'se ', 'le ', 'les ', 'mis ', 'tus ', 'sus ', 'nuestros ', 'vuestros ',
        'seguridad', 'red', 'cumplimiento'
    ]
    
    # English is default, but check for specific patterns
    english_indicators = [
        'the ', 'and ', 'is ', 'are ', 'was ', 'were ', 'a ', 'an ', 'in ', 'on ',
        'at ', 'by ', 'for ', 'with ', 'to ', 'of ', 'that ', 'this ', 'what ',
        'how ', 'when ', 'where ', 'why ', 'tion ', 'ment ', 'ing ', 'ed ',
        'hello', 'hi', 'good morning', 'good afternoon', 'good evening', 'goodbye', 'thanks', 'yes', 'no',
        'i ', 'you ', 'he ', 'she ', 'we ', 'they ', 'me ', 'him ', 'her ', 'us ', 'them ',
        'my ', 'your ', 'his ', 'her ', 'our ', 'their ', 'mine ', 'yours ',
        'can', 'could', 'will', 'would', 'should', 'must', 'have', 'has', 'had',
        'do', 'does', 'did', 'be', 'am', 'is', 'are', 'was', 'were',
        'go', 'goes', 'went', 'come', 'comes', 'came', 'get', 'gets', 'got',
        'say', 'says', 'said', 'tell', 'tells', 'told', 'know', 'knows', 'knew',
        'present', 'show', 'display', 'capability', 'capabilities', 'feature', 'features',
        'system', 'service', 'help', 'assist', 'assistance', 'question', 'answer',
        'security', 'network', 'compliance', 'regulation', 'policy', 'management'
    ]
    
    # Count indicators for each language
    french_score = sum(1 for indicator in french_indicators if indicator in text_lower)
    spanish_score = sum(1 for indicator in spanish_indicators if indicator in text_lower)
    english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
    
    # Debug logging to help troubleshoot detection issues
    logger.debug(f"Language detection scores - French: {french_score}, Spanish: {spanish_score}, English: {english_score}")
    logger.debug(f"Text being analyzed: '{text[:100]}...' (truncated)")
    
    # Determine language based on highest score
    if french_score > english_score and french_score > spanish_score:
        return 'fr'
    elif spanish_score > english_score and spanish_score > french_score:
        return 'es'
    else:
        return 'en'  # Default to English

def get_language_instruction(language: str) -> str:
    """
    Get language-specific instruction for the LLM.
    """
    instructions = {
        'fr': """IMPORTANT: Vous devez TOUJOURS répondre en français, même si des informations en anglais sont fournies dans le contexte. Traduisez et adaptez le contenu en français naturel.""",
        'es': """IMPORTANTE: Debes responder SIEMPRE en español, incluso si se proporciona información en inglés en el contexto. Traduce y adapta el contenido al español natural.""",
        'en': """IMPORTANT: Always respond in English, even if information in other languages is provided in the context."""
    }
    return instructions.get(language, instructions['en'])

class LLMIntegration:
    """
    Integration with LLM services with specialized GRC analysis capabilities.
    
    This class provides a bridge between the Agent Framework and various
    LLM services, with specialized support for French GRC analysis.
    """
    
    def __init__(self, 
                provider: str = "openai", 
                model: str = "gpt-4.1", 
                api_key: Optional[str] = None,
                max_tokens: int = 4000,
                temperature: float = 0.2):
        """
        Initialize the LLM integration.
        
        Args:
            provider: The LLM provider to use
            model: The model to use
            api_key: API key for the provider
            max_tokens: Maximum tokens in the response
            temperature: Temperature for generation
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set up API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            self.api_key = os.environ.get("OPENAI_API_KEY")
            
        # Initialize the client
        self._client = None
        self._initialize_client()
        
        # Configuration par défaut pour les analyses GRC
        self.default_config = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # Prompts système spécialisés par domaine GRC
        self.system_prompts = {
            "risk_assessment": """
Tu es un expert en évaluation des risques informatiques et de sécurité.
Tu maîtrises les méthodologies d'analyse de risque (EBIOS, MEHARI, ISO 27005).
Tu communiques en français et structures tes analyses selon les standards GRC.
Tu identifies les risques, évalues leur probabilité et impact, et proposes des mesures de traitement.
""",
            "compliance_analysis": """
Tu es un expert en conformité réglementaire, spécialisé en RGPD, ISO 27001, et DORA.
Tu analyses les documents pour identifier les exigences, gaps de conformité, et recommandations.
Tu communiques en français et structures tes analyses selon les référentiels applicables.
Tu maîtrises la cartographie des contrôles et l'évaluation de maturité.
""",
            "governance_analysis": """
Tu es un expert en gouvernance IT et sécurité des systèmes d'information.
Tu analyses les politiques, procédures, et structures organisationnelles.
Tu communiques en français et évalues l'efficacité des dispositifs de gouvernance.
Tu identifies les gaps organisationnels et proposes des améliorations structurelles.
""",
            "document_finder": """
Tu es un expert en recherche et classification de documents GRC.
Tu identifies les types de documents, leur pertinence, et leurs interconnexions.
Tu communiques en français et organises l'information de manière structurée.
Tu maîtrises la taxonomie des documents GRC (politiques, procédures, audits, etc.).
""",
            "entity_extractor": """
Tu es un expert en extraction d'entités GRC depuis des documents techniques.
Tu identifies les contrôles, risques, actifs, exigences réglementaires.
Tu communiques en français et structures les entités selon les standards GRC.
Tu maîtrises les référentiels de sécurité et les frameworks de conformité.
""",
            "cross_reference": """
Tu es un expert en analyse des relations et dépendances dans les systèmes GRC.
Tu identifies les liens entre contrôles, risques, exigences, et documents.
Tu communiques en français et mappes les correspondances entre référentiels.
Tu maîtrises la traçabilité et la cohérence des dispositifs GRC.
""",
            "temporal_analyzer": """
Tu es un expert en analyse temporelle et évolution des dispositifs GRC.
Tu identifies les tendances, évolutions, et patterns dans les données historiques.
Tu communiques en français et analyses les améliorations/dégradations.
Tu maîtrises l'analyse de maturité et les indicateurs de performance GRC.
"""
        }
        
    def _initialize_client(self):
        """Initialize the appropriate client for the selected provider."""
        if self.provider == "openai":
            if OPENAI_AVAILABLE and self.api_key:
                try:
                    self._client = AsyncOpenAI(api_key=self.api_key)
                    logger.info(f"Initialized OpenAI client with model {self.model}")
                except Exception as e:
                    logger.error(f"Error initializing OpenAI client: {str(e)}")
                    self._client = None
            else:
                logger.warning("OpenAI integration not available")
                self._client = None
        else:
            logger.warning(f"Unsupported LLM provider: {self.provider}")
            self._client = None

    async def generate(self, prompt: str, agent_type: str = None, **kwargs) -> str:
        """
        Generate text using the LLM with optional GRC specialization.
        
        Args:
            prompt: The prompt to generate from
            agent_type: Type of agent for specialized GRC prompts
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated text
        """
        if self._client is None:
            logger.error("Cannot generate: LLM client not initialized")
            return "Je suis désolé, mais je ne peux pas accéder au modèle de langage pour le moment."
            
        try:
            logger.info(f"Generating text with {self.provider} model {self.model}")
            
            # Detect language from the prompt (unless explicitly disabled)
            if kwargs.get("auto_language_detection", True):
                detected_language = detect_language(prompt)
                logger.info(f"Agent framework detected language: {detected_language} for prompt")
                
                # Add language instruction to system message if not already provided
                existing_system = kwargs.get("system_message", "")
                language_instruction = get_language_instruction(detected_language)
                
                if existing_system:
                    kwargs["system_message"] = f"{language_instruction}\n\n{existing_system}"
                else:
                    kwargs["system_message"] = language_instruction
            
            # Add GRC specialized system prompt if specified
            if agent_type and agent_type in self.system_prompts:
                grc_prompt = self.system_prompts[agent_type]
                existing_system = kwargs.get("system_message", "")
                if existing_system:
                    kwargs["system_message"] = f"{existing_system}\n\n{grc_prompt}"
                else:
                    kwargs["system_message"] = grc_prompt
            
            if self.provider == "openai":
                return await self._generate_openai(prompt, **kwargs)
            else:
                logger.error(f"Unsupported LLM provider: {self.provider}")
                return "Je suis désolé, mais le modèle de langage demandé n'est pas supporté."
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"J'ai rencontré une erreur lors de la génération de la réponse: {str(e)}"

    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        agent_type: str = None,
        **kwargs
    ) -> str:
        """
        Génère une réponse via GPT-4.1 avec support des conversations.
        
        Args:
            messages: Liste des messages de conversation
            agent_type: Type d'agent pour sélectionner le prompt système approprié
            **kwargs: Paramètres supplémentaires pour l'API
        """
        try:
            # Fusionner la configuration par défaut avec les paramètres fournis
            config = {**self.default_config, **kwargs}
            
            # Ajouter le prompt système spécialisé si spécifié
            if agent_type and agent_type in self.system_prompts:
                # Vérifier si un message système existe déjà
                has_system = any(msg.get("role") == "system" for msg in messages)
                if not has_system:
                    messages = [
                        {"role": "system", "content": self.system_prompts[agent_type]}
                    ] + messages
            
            # Appel à l'API OpenAI
            response = await self._client.chat.completions.create(
                messages=messages,
                **config
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération LLM: {str(e)}")
            raise

    async def analyze_with_structured_output(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        agent_type: str = None
    ) -> Dict[str, Any]:
        """
        Génère une réponse structurée selon un schéma JSON.
        
        Args:
            prompt: Prompt d'analyse
            schema: Schéma JSON attendu
            agent_type: Type d'agent pour le prompt système
        """
        try:
            structured_prompt = f"""
{prompt}

Réponds UNIQUEMENT au format JSON suivant:
{json.dumps(schema, indent=2, ensure_ascii=False)}

Assure-toi que la réponse soit un JSON valide et complet.
"""
            
            messages = [{"role": "user", "content": structured_prompt}]
            
            response = await self.generate_response(
                messages=messages,
                agent_type=agent_type,
                temperature=0.1  # Plus déterministe pour les sorties structurées
            )
            
            # Extraire et valider le JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("Aucun JSON trouvé dans la réponse")
                
            json_content = response[json_start:json_end]
            return json.loads(json_content)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse structurée: {str(e)}")
            raise

    async def extract_entities(
        self, 
        text: str, 
        entity_types: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Extrait des entités GRC depuis un texte.
        
        Args:
            text: Texte à analyser
            entity_types: Types d'entités à extraire
        """
        if not entity_types:
            entity_types = [
                "controles", "risques", "exigences", "actifs", 
                "processus", "procedures", "politiques", "frameworks"
            ]
        
        schema = {
            entity_type: [] for entity_type in entity_types
        }
        
        prompt = f"""
Analyse le texte suivant et extrait toutes les entités GRC pertinentes:

TEXTE:
{text}

Pour chaque type d'entité, liste tous les éléments identifiés avec leurs caractéristiques principales.
"""
        
        return await self.analyze_with_structured_output(
            prompt=prompt,
            schema=schema,
            agent_type="entity_extractor"
        )

    async def assess_compliance_gap(
        self, 
        documents: List[str], 
        framework: str = "ISO27001"
    ) -> Dict[str, Any]:
        """
        Évalue les gaps de conformité par rapport à un framework.
        
        Args:
            documents: Liste des documents à analyser
            framework: Framework de référence (ISO27001, RGPD, etc.)
        """
        combined_text = "\n\n".join(documents)
        
        schema = {
            "framework_analyse": framework,
            "exigences_couvertes": [],
            "gaps_identifies": [],
            "niveau_maturite": "",
            "recommandations": [],
            "score_conformite": 0
        }
        
        prompt = f"""
Analyse de conformité {framework}.

DOCUMENTS À ANALYSER:
{combined_text}

Évalue la conformité aux exigences {framework} et identifie les gaps.
"""
        
        return await self.analyze_with_structured_output(
            prompt=prompt,
            schema=schema,
            agent_type="compliance_analysis"
        )

    async def perform_risk_assessment(
        self, 
        context: str, 
        risk_methodology: str = "EBIOS"
    ) -> Dict[str, Any]:
        """
        Effectue une évaluation de risque.
        
        Args:
            context: Contexte/documents pour l'évaluation
            risk_methodology: Méthodologie d'évaluation (EBIOS, MEHARI, etc.)
        """
        schema = {
            "methodologie": risk_methodology,
            "risques_identifies": [],
            "evaluation_impact": {},
            "evaluation_probabilite": {},
            "niveau_risque_global": "",
            "mesures_recommandees": [],
            "priorites_traitement": []
        }
        
        prompt = f"""
Analyse de risque selon la méthodologie {risk_methodology}.

CONTEXTE:
{context}

Identifie et évalue tous les risques pertinents avec leur impact et probabilité.
"""
        
        return await self.analyze_with_structured_output(
            prompt=prompt,
            schema=schema,
            agent_type="risk_assessment"
        )
            
    async def _generate_openai(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated text
        """
        if not self._client:
            logger.error("OpenAI client not initialized")
            return "Je suis désolé, mais je ne peux pas accéder au service OpenAI pour le moment."
            
        try:
            # Merge kwargs with defaults
            params = {
                "model": kwargs.get("model", self.model),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            
            # Create messages format for OpenAI
            messages = [{"role": "user", "content": prompt}]
            if "system_message" in kwargs:
                messages.insert(0, {"role": "system", "content": kwargs["system_message"]})
                
            # If streaming is requested
            if kwargs.get("stream", False):
                response_chunks = []
                async for chunk in await self._client.chat.completions.create(
                    messages=messages,
                    stream=True,
                    **params
                ):
                    if chunk.choices[0].delta.content:
                        response_chunks.append(chunk.choices[0].delta.content)
                response_text = "".join(response_chunks)
            else:
                # Standard non-streaming request
                response = await self._client.chat.completions.create(
                    messages=messages,
                    **params
                )
                response_text = response.choices[0].message.content
                
            return response_text
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise
            
    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embeddings
        """
        if self._client is None:
            logger.error("Cannot embed: LLM client not initialized")
            return []
            
        try:
            logger.info(f"Generating embeddings with {self.provider}")
            
            if self.provider == "openai":
                return await self._embed_openai(text)
            else:
                logger.error(f"Unsupported embedding provider: {self.provider}")
                return []
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
            
    async def _embed_openai(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI.
        
        Args:
            text: The text to embed
            
        Returns:
            The embeddings
        """
        if not self._client:
            logger.error("OpenAI client not initialized")
            return []
            
        try:
            # If text is a string, convert to list
            if isinstance(text, str):
                text = [text]
                
            response = await self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur le modèle utilisé."""
        return {
            "model": self.model,
            "provider": self.provider,
            "language_optimized": "French",
            "specialization": "GRC Analysis",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

# Backward compatibility class alias
LLMClient = LLMIntegration

# Singleton instance
_llm_integration = None

def get_llm_integration(provider: str = "openai", model: str = "gpt-4.1") -> LLMIntegration:
    """
    Get the LLM integration instance.
    
    Args:
        provider: The LLM provider to use
        model: The model to use
        
    Returns:
        The LLM integration instance
    """
    global _llm_integration
    
    if _llm_integration is None:
        _llm_integration = LLMIntegration(provider=provider, model=model)
        
    return _llm_integration

def get_llm_client() -> LLMIntegration:
    """
    Retourne une instance singleton du client LLM (backward compatibility).
    """
    return get_llm_integration()

async def test_llm_connection():
    """
    Teste la connexion LLM avec une requête simple.
    """
    try:
        client = get_llm_integration()
        response = await client.generate(
            "Dis bonjour en français pour tester la connexion"
        )
        logger.info(f"Test LLM réussi: {response}")
        return True
    except Exception as e:
        logger.error(f"Échec du test LLM: {str(e)}")
        return False 