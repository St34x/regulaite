# plugins/regul_aite/backend/data_enrichment/language_detector.py
import logging
import re
from typing import Optional, Dict, List, Any
from langdetect import detect, LangDetectException
import langid

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    Detects the language of text using multiple detection methods.
    """

    # Mapping from language codes to spaCy model names
    LANGUAGE_MODEL_MAPPING = {
        'en': 'en_core_web_sm',  # English
        'de': 'de_core_news_sm',  # German
        'es': 'es_core_news_sm',  # Spanish
        'fr': 'fr_core_news_sm',  # French
        'it': 'it_core_news_sm',  # Italian
        'nl': 'nl_core_news_sm',  # Dutch
        'pt': 'pt_core_news_sm',  # Portuguese
        'xx': 'xx_ent_wiki_sm',   # Multi-language fallback
    }

    # Languages with good spaCy support
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'de': 'German',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'nl': 'Dutch',
        'pt': 'Portuguese'
    }

    def __init__(self, fallback_language: str = 'en'):
        """
        Initialize the language detector.

        Args:
            fallback_language: Language code to use when detection fails
        """
        self.fallback_language = fallback_language

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with language code, confidence, and model name
        """
        if not text:
            logger.warning("Empty text provided for language detection")
            return self._get_language_info(self.fallback_language, 1.0)

        # For very short texts, use langid which works better for small samples
        if len(text) < 100:
            return self._detect_with_langid(text)

        # For longer texts, try multiple methods for better accuracy
        try:
            # First attempt with langdetect
            lang_code = detect(text)

            # Double-check with langid for more robustness
            langid_code, confidence = langid.classify(text)

            # If they agree, we're more confident
            if lang_code == langid_code:
                logger.info(f"Language detected with high confidence: {lang_code}")
                return self._get_language_info(lang_code, confidence)

            # If they disagree, use heuristics to decide
            # Check for specific language markers in text
            if self._has_language_markers(text, lang_code):
                logger.info(f"Language detected with markers: {lang_code}")
                return self._get_language_info(lang_code, 0.8)

            # Otherwise use langid result which has confidence score
            logger.info(f"Language detected with langid: {langid_code} (confidence: {confidence:.2f})")
            return self._get_language_info(langid_code, confidence)

        except LangDetectException as e:
            logger.warning(f"Language detection failed: {str(e)}")
            # Fall back to langid
            return self._detect_with_langid(text)
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {str(e)}")
            return self._get_language_info(self.fallback_language, 0.5)

    def _detect_with_langid(self, text: str) -> Dict[str, Any]:
        """Use langid for language detection"""
        try:
            lang_code, confidence = langid.classify(text)
            logger.info(f"Language detected with langid: {lang_code} (confidence: {confidence:.2f})")
            return self._get_language_info(lang_code, confidence)
        except Exception as e:
            logger.error(f"langid detection failed: {str(e)}")
            return self._get_language_info(self.fallback_language, 0.5)

    def _has_language_markers(self, text: str, lang_code: str) -> bool:
        """Check for specific language markers"""
        # Language-specific markers (common words, patterns)
        markers = {
            'en': [r'\bthe\b', r'\band\b', r'\bof\b', r'\bin\b', r'\bto\b'],
            'es': [r'\bel\b', r'\bla\b', r'\blos\b', r'\blas\b', r'\by\b', r'\bde\b'],
            'fr': [r'\ble\b', r'\bla\b', r'\bles\b', r'\bdes\b', r'\bet\b', r'\bde\b'],
            'de': [r'\bder\b', r'\bdie\b', r'\bdas\b', r'\bund\b', r'\bin\b', r'\bzu\b'],
            'it': [r'\bil\b', r'\bla\b', r'\bi\b', r'\ble\b', r'\be\b', r'\bdi\b'],
            'pt': [r'\bo\b', r'\ba\b', r'\bos\b', r'\bas\b', r'\be\b', r'\bde\b'],
            'nl': [r'\bde\b', r'\bhet\b', r'\been\b', r'\ben\b', r'\bin\b', r'\bvan\b']
        }

        if lang_code not in markers:
            return False

        # Count matches for the detected language
        match_count = 0
        for pattern in markers[lang_code]:
            match_count += len(re.findall(pattern, text.lower()))

        # Set a threshold based on text length
        text_length = len(text)
        if text_length < 500:
            threshold = 3
        elif text_length < 2000:
            threshold = 5
        else:
            threshold = 10

        return match_count >= threshold

    def _get_language_info(self, lang_code: str, confidence: float) -> Dict[str, Any]:
        """Get full language information"""
        # Check if language is supported with a spaCy model
        is_supported = lang_code in self.SUPPORTED_LANGUAGES

        # Get the appropriate spaCy model name
        model_name = self.LANGUAGE_MODEL_MAPPING.get(lang_code, None)

        # If no specific model, use fallback options
        if not model_name:
            if confidence > 0.8:
                # For high confidence, try the multi-language model
                model_name = 'xx_ent_wiki_sm'
            else:
                # Otherwise default to English
                model_name = self.LANGUAGE_MODEL_MAPPING[self.fallback_language]
                lang_code = self.fallback_language

        return {
            'language_code': lang_code,
            'language_name': self.SUPPORTED_LANGUAGES.get(lang_code, f"Language ({lang_code})"),
            'confidence': confidence,
            'is_supported': is_supported,
            'model_name': model_name
        }
