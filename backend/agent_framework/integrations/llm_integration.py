"""
LLM Integration for the RegulAIte Agent Framework.

This module provides integration with LLM services.
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
    
    # French indicators
    french_indicators = [
        'le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'et ', 'est ', 'un ', 'une ',
        'dans ', 'pour ', 'avec ', 'sur ', 'par ', 'ce ', 'qui ', 'que ', 'comment ',
        'où ', 'quand ', 'pourquoi ', 'qu\'', 'c\'', 'd\'', 'l\'', 'n\'', 'tion ',
        'ment ', 'ées ', 'ent ', 'sont ', 'ont', 'était', 'avait', 'sera', 'sécurité',
        'réseau', 'conformité', 'réglementation', 'politique', 'gestion', 'contrôle'
    ]
    
    # Spanish indicators
    spanish_indicators = [
        'el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'y ', 'es ', 'un ', 'una ',
        'en ', 'con ', 'por ', 'para ', 'que ', 'como ', 'donde ', 'cuando ',
        'por qué ', 'cómo ', 'ción ', 'mente ', 'ado ', 'ida ', 'son ', 'han',
        'seguridad', 'red', 'cumplimiento'
    ]
    
    # English is default, but check for specific patterns
    english_indicators = [
        'the ', 'and ', 'is ', 'are ', 'was ', 'were ', 'a ', 'an ', 'in ', 'on ',
        'at ', 'by ', 'for ', 'with ', 'to ', 'of ', 'that ', 'this ', 'what ',
        'how ', 'when ', 'where ', 'why ', 'tion ', 'ment ', 'ing ', 'ed ',
        'security', 'network', 'compliance', 'regulation', 'policy', 'management'
    ]
    
    # Count indicators for each language
    french_score = sum(1 for indicator in french_indicators if indicator in text_lower)
    spanish_score = sum(1 for indicator in spanish_indicators if indicator in text_lower)
    english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
    
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
    Integration with LLM services.
    
    This class provides a bridge between the Agent Framework and various
    LLM services.
    """
    
    def __init__(self, 
                provider: str = "openai", 
                model: str = "gpt-4", 
                api_key: Optional[str] = None,
                max_tokens: int = 1024,
                temperature: float = 0.7):
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
            
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated text
        """
        if self._client is None:
            logger.error("Cannot generate: LLM client not initialized")
            return "I'm sorry, but I cannot access the language model at the moment."
            
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
            
            if self.provider == "openai":
                return await self._generate_openai(prompt, **kwargs)
            else:
                logger.error(f"Unsupported LLM provider: {self.provider}")
                return "I'm sorry, but the requested language model is not supported."
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"I encountered an error while generating a response: {str(e)}"
            
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
            return "I'm sorry, but I cannot access the OpenAI service at the moment."
            
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

# Singleton instance
_llm_integration = None

def get_llm_integration(provider: str = "openai", model: str = "gpt-4"):
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