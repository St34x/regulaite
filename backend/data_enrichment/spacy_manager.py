# plugins/regul_aite/backend/data_enrichment/spacy_manager.py
import logging
import spacy
from typing import Dict, Optional, List, Any
import os
from spacy.language import Language

logger = logging.getLogger(__name__)

class MultilingualSpacyManager:
    """
    Manages multiple spaCy models for different languages.
    Loads models on demand and caches them for reuse.
    """

    def __init__(
        self,
        max_models: int = 3,
        default_language: str = 'en',
        default_model: str = 'en_core_web_sm'
    ):
        """
        Initialize the spaCy model manager.

        Args:
            max_models: Maximum number of models to keep in memory
            default_language: Default language code
            default_model: Default model name
        """
        self.max_models = max_models
        self.default_language = default_language
        self.default_model = default_model

        # Cache of loaded models
        self.models = {}

        # Track model usage for LRU cache
        self.model_usage = []

        # Initialize default model
        self._load_model(default_model)

        logger.info(f"Multilingual spaCy manager initialized with default model: {default_model}")

    def _load_model(self, model_name: str) -> Optional[Language]:
        """
        Load a spaCy model.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded spaCy model or None if loading fails
        """
        if model_name in self.models:
            # Update usage tracking
            if model_name in self.model_usage:
                self.model_usage.remove(model_name)
            self.model_usage.append(model_name)

            return self.models[model_name]

        try:
            logger.info(f"Loading spaCy model: {model_name}")

            # Try to load the model
            try:
                nlp = spacy.load(model_name)
            except OSError:
                # Model not found, try to download it
                logger.info(f"Model {model_name} not found. Trying to download...")
                os.system(f"python -m spacy download {model_name}")
                nlp = spacy.load(model_name)

            # If we have too many models loaded, remove the least recently used
            if len(self.models) >= self.max_models and self.model_usage:
                oldest_model = self.model_usage.pop(0)
                if oldest_model != self.default_model:  # Don't remove default model
                    logger.info(f"Removing least recently used model: {oldest_model}")
                    del self.models[oldest_model]

            # Add the model to cache
            self.models[model_name] = nlp
            self.model_usage.append(model_name)

            logger.info(f"Successfully loaded spaCy model: {model_name}")
            return nlp

        except Exception as e:
            logger.error(f"Failed to load spaCy model {model_name}: {str(e)}")

            # Try to use default model as fallback
            if model_name != self.default_model and self.default_model in self.models:
                logger.info(f"Using default model {self.default_model} instead")
                return self.models[self.default_model]

            return None

    def get_model(self, lang_code: str = None, model_name: str = None) -> Optional[Language]:
        """
        Get a spaCy model for the specified language.

        Args:
            lang_code: Language code
            model_name: Specific model name (overrides lang_code if provided)

        Returns:
            spaCy model for the language or None if not available
        """
        if model_name:
            return self._load_model(model_name)

        if not lang_code:
            lang_code = self.default_language

        # Map language code to model name
        from .language_detector import LanguageDetector
        model_map = LanguageDetector.LANGUAGE_MODEL_MAPPING

        model_name = model_map.get(lang_code, self.default_model)
        return self._load_model(model_name)

    def process_text(self, text: str, lang_code: str = None, model_name: str = None) -> Optional[Any]:
        """
        Process text using the appropriate spaCy model.

        Args:
            text: Text to process
            lang_code: Language code
            model_name: Specific model name (overrides lang_code if provided)

        Returns:
            Processed spaCy Doc object or None if processing fails
        """
        nlp = self.get_model(lang_code, model_name)

        if nlp:
            try:
                return nlp(text)
            except Exception as e:
                logger.error(f"Error processing text with spaCy: {str(e)}")

        return None
