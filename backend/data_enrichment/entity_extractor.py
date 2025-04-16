# plugins/regul_aite/backend/data_enrichment/entity_extractor.py
import logging
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import EntityRuler
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import re
import os
import json
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Enhanced entity extraction using spaCy with specialized regulatory entity detection.
    Supports multilingual documents by dynamically loading appropriate language models.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        custom_entities: bool = True,
        regulatory_patterns: bool = True,
        multilingual: bool = True,
        max_models: int = 3
    ):
        """
        Initialize the entity extractor.

        Args:
            spacy_model: Name of default spaCy model to use
            custom_entities: Whether to enable custom entity detection
            regulatory_patterns: Whether to add regulatory-specific patterns
            multilingual: Whether to enable multilingual support
            max_models: Maximum number of language models to keep in memory
        """
        self.spacy_model_name = spacy_model
        self.custom_entities = custom_entities
        self.regulatory_patterns = regulatory_patterns
        self.multilingual = multilingual

        # Initialize language detector
        self.language_detector = LanguageDetector(fallback_language='en')

        # Dictionary to store NLP models by language
        self.models = {}

        # Define mappings from spaCy entity types to our schema
        self.entity_type_mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",
            "LOC": "location",
            "DATE": "date",
            "TIME": "date",
            "LAW": "regulation",
            "REGULATION": "regulation",
            "STANDARD": "standard",
            "NORP": "group",  # Nationalities, religious or political groups
            "MONEY": "financial",
            "PERCENT": "metric",
            "PRODUCT": "product",
            "EVENT": "event",
            "WORK_OF_ART": "document",
            "LANGUAGE": "language",
            "QUANTITY": "metric",
            "REGULATORY_REF": "regulation",
            "COMPLIANCE": "compliance",
            "AUTHORITY": "authority",
            "DEADLINE": "deadline",
            "FAC": "facility",
            "CARDINAL": "number",
            "ORDINAL": "number"
        }

        # Define regulatory entity patterns for each supported language
        self.regulatory_patterns_by_language = {
            'en': self._get_english_patterns(),
            'es': self._get_spanish_patterns(),
            'fr': self._get_french_patterns(),
            'de': self._get_german_patterns(),
            'it': self._get_italian_patterns(),
            'pt': self._get_portuguese_patterns(),
            'nl': self._get_dutch_patterns()
        }

        # Load default model
        if multilingual:
            self.nlp = self._load_model_for_language('en', spacy_model)
        else:
            # Load single spaCy model
            try:
                logger.info(f"Loading spaCy model: {spacy_model}")
                self.nlp = spacy.load(spacy_model)

                # Add custom components if requested
                if self.nlp and custom_entities:
                    self._setup_custom_entity_components(self.nlp)

                # Add regulatory patterns if requested
                if self.nlp and regulatory_patterns:
                    self._add_regulatory_patterns(self.nlp, 'en')

                logger.info(f"Successfully loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {str(e)}")
                logger.warning("Continuing without spaCy NLP capabilities")
                self.nlp = None

    def _load_model_for_language(self, lang_code: str, model_name: Optional[str] = None) -> Optional[Language]:
        """
        Load a spaCy model for the specified language.

        Args:
            lang_code: Two-letter language code
            model_name: Specific model name (overrides language mapping if provided)

        Returns:
            Loaded spaCy Language model or None if loading fails
        """
        # If we already have a model for this language, return it
        if lang_code in self.models:
            return self.models[lang_code]

        # Map language code to model name if not provided
        if not model_name:
            model_mapping = {
                'en': 'en_core_web_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm',
                'it': 'it_core_news_sm',
                'nl': 'nl_core_news_sm',
                'pt': 'pt_core_news_sm',
            }
            model_name = model_mapping.get(lang_code, 'xx_ent_wiki_sm')

        try:
            # Try to load the model
            logger.info(f"Loading spaCy model {model_name} for language {lang_code}")

            try:
                nlp = spacy.load(model_name)
            except OSError:
                # Try to download the model if not found
                logger.info(f"Model {model_name} not found. Trying to download...")
                os.system(f"python -m spacy download {model_name}")
                nlp = spacy.load(model_name)

            # Add custom components if requested
            if self.custom_entities:
                self._setup_custom_entity_components(nlp)

            # Add regulatory patterns if requested
            if self.regulatory_patterns:
                self._add_regulatory_patterns(nlp, lang_code)

            # Store the model
            self.models[lang_code] = nlp

            # If we have too many models, remove the oldest ones
            if len(self.models) > 3:  # Keep at most 3 models
                # Don't remove the default English model
                languages_to_check = [lang for lang in self.models.keys() if lang != 'en']
                if languages_to_check:
                    # Remove the first language (oldest)
                    lang_to_remove = languages_to_check[0]
                    del self.models[lang_to_remove]
                    logger.info(f"Removed model for language {lang_to_remove} to conserve memory")

            logger.info(f"Successfully loaded model {model_name} for language {lang_code}")
            return nlp

        except Exception as e:
            logger.error(f"Failed to load model for language {lang_code}: {str(e)}")

            # If this is not English, try to fall back to English model
            if lang_code != 'en' and 'en' in self.models:
                logger.info("Falling back to English model")
                return self.models['en']

            # If we don't have an English model either, try to load it
            if lang_code != 'en':
                logger.info("Attempting to load English model as fallback")
                return self._load_model_for_language('en', 'en_core_web_sm')

            return None

    def _setup_custom_entity_components(self, nlp: Language) -> None:
        """
        Set up custom entity components for spaCy pipeline.

        Args:
            nlp: spaCy Language model to modify
        """
        # Add entity merging component if not already present
        if "merge_entities" not in nlp.pipe_names:
            @Language.component("merge_entities")
            def merge_entities(doc: Doc) -> Doc:
                """Merge entity spans into single tokens"""
                with doc.retokenize() as retokenizer:
                    for ent in doc.ents:
                        retokenizer.merge(doc[ent.start:ent.end])
                return doc

            nlp.add_pipe("merge_entities", after="ner")
            logger.info("Added entity merging component")

        # Add custom component for regulatory entity detection
        if "regulatory_ner" not in nlp.pipe_names:
            @Language.component("regulatory_ner")
            def regulatory_ner(doc: Doc) -> Doc:
                """Custom NER component for regulatory entities"""
                new_ents = []

                # Regular expressions for regulatory entities
                patterns = {
                    "REGULATORY_REF": [
                        r"(?:Regulation|Directive|Decision|Framework|Recommendation)\s+\((?:EU|EC)\)\s+\d{4}/\d+",
                        r"(?:Regulation|Directive|Decision|Framework|Recommendation)\s+\d{4}/\d+/(?:EU|EC)",
                        r"Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?",
                        r"Section\s+\d+\.\d+",
                        r"Paragraph\s+\d+(?:\.\d+)?"
                    ],
                    "COMPLIANCE": [
                        r"(?:mandatory|obligatory|required|compulsory|necessary)\s+(?:requirement|compliance|adherence|conformity)",
                        r"(?:compliance|regulatory)\s+(?:obligation|requirement|standard|threshold|limit)",
                        r"(?:non-)?compliant with",
                        r"in accordance with"
                    ],
                    "AUTHORITY": [
                        r"(?:European|National|Federal|State)\s+(?:Authority|Agency|Commission|Regulator|Body)",
                        r"(?:EBA|ESMA|ECB|FCA|SEC|FINRA|CFTC|FDIC|OCC)"
                    ],
                    "DEADLINE": [
                        r"(?:deadline|due date|time limit|cutoff date|submission date)\s+(?:of|for|by)?\s+\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
                        r"(?:deadline|due date|time limit|cutoff date|submission date)\s+(?:of|for|by)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
                        r"by\s+(?:the\s+end\s+of|)\s+Q[1-4]\s+\d{4}"
                    ]
                }

                # Search for patterns in text
                for label, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                            start_char = match.start()
                            end_char = match.end()

                            # Find tokens that correspond to matched span
                            start_token = None
                            end_token = None

                            for i, token in enumerate(doc):
                                if token.idx <= start_char < token.idx + len(token.text):
                                    start_token = i
                                if token.idx <= end_char <= token.idx + len(token.text):
                                    end_token = i + 1
                                    break

                            if start_token is not None and end_token is not None:
                                ent = Span(doc, start_token, end_token, label=label)
                                new_ents.append(ent)

                # Add new entities, but prioritize existing ones in case of overlap
                if doc.ents:
                    doc.ents = list(doc.ents) + [e for e in new_ents if not any(
                        existing.start <= e.start < existing.end or
                        existing.start < e.end <= existing.end
                        for existing in doc.ents
                    )]
                else:
                    doc.ents = new_ents

                return doc

            nlp.add_pipe("regulatory_ner", after="ner")
            logger.info("Added regulatory NER component")

    def _get_english_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for English language regulatory entities"""
        return [
            # Regulations and directives
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["regulation", "directive"]}}, {"TEXT": "(EU)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["regulation", "directive"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(EU)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["article", "section", "paragraph"]}}, {"IS_DIGIT": True}, {"LOWER": "of", "OP": "?"}, {"LOWER": "the", "OP": "?"}, {"LOWER": {"IN": ["regulation", "directive", "act", "law"]}, "OP": "?"}]},
            {"label": "LAW", "pattern": [{"LOWER": "law"}, {"OP": "?"}, {"IS_DIGIT": True}, {"LOWER": "of"}, {"IS_DIGIT": True}]},

            # Standards
            {"label": "STANDARD", "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 2, "<=": 10}}, {"TEXT": "-", "OP": "?"}, {"IS_DIGIT": True, "LENGTH": {">=": 1, "<=": 6}}]},
            {"label": "STANDARD", "pattern": [{"TEXT": {"REGEX": "ISO|IEC|EN|BS|ASTM|IEEE|NIST"}}, {"IS_DIGIT": True}, {"TEXT": ":", "OP": "?"}, {"IS_DIGIT": True, "OP": "?"}]},

            # Regulatory authorities
            {"label": "ORG", "pattern": [{"LOWER": {"IN": ["eba", "esma", "ecb", "fca", "sec", "finra", "cftc", "fdic", "occ"]}}]},
            {"label": "ORG", "pattern": [{"LOWER": {"IN": ["european", "financial"]}}, {"LOWER": {"IN": ["banking", "securities", "markets"]}}, {"LOWER": {"IN": ["authority", "agency", "commission"]}}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["compliance", "regulatory"]}}, {"LOWER": {"IN": ["requirement", "obligation", "standard", "threshold", "limit"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "comply"}, {"LOWER": "with"}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "in"}, {"LOWER": "accordance"}, {"LOWER": "with"}]},
        ]

    def _get_spanish_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for Spanish language regulatory entities"""
        return [
            # Regulations and laws
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["reglamento", "directiva"]}}, {"TEXT": "(UE)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["reglamento", "directiva"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(UE)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["artículo", "sección", "párrafo"]}}, {"IS_DIGIT": True}]},
            {"label": "LAW", "pattern": [{"LOWER": "ley"}, {"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["cumplimiento", "regulatorio"]}}, {"LOWER": {"IN": ["requisito", "obligación", "estándar", "umbral", "límite"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "cumplir"}, {"LOWER": "con"}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "de"}, {"LOWER": "conformidad"}, {"LOWER": "con"}]},

            # Authorities
            {"label": "AUTHORITY", "pattern": [{"LOWER": {"IN": ["autoridad", "comisión"]}}, {"LOWER": {"IN": ["europea", "nacional", "bancaria", "financiera"]}}]},
        ]

    def _get_french_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for French language regulatory entities"""
        return [
            # Regulations and laws
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["règlement", "directive"]}}, {"TEXT": "(UE)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["règlement", "directive"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(UE)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["article", "section", "paragraphe"]}}, {"IS_DIGIT": True}]},
            {"label": "LAW", "pattern": [{"LOWER": "loi"}, {"IS_DIGIT": True}, {"LOWER": "du"}, {"IS_DIGIT": True}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["conformité", "réglementaire"]}}, {"LOWER": {"IN": ["exigence", "obligation", "standard", "seuil", "limite"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "conformément"}, {"LOWER": "à"}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "en"}, {"LOWER": "conformité"}, {"LOWER": "avec"}]},

            # Authorities
            {"label": "AUTHORITY", "pattern": [{"LOWER": {"IN": ["autorité", "commission"]}}, {"LOWER": {"IN": ["européenne", "nationale", "bancaire", "financière"]}}]},
        ]

    def _get_german_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for German language regulatory entities"""
        return [
            # Regulations and laws
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["verordnung", "richtlinie"]}}, {"TEXT": "(EU)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["verordnung", "richtlinie"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(EU)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["artikel", "abschnitt", "absatz"]}}, {"IS_DIGIT": True}]},
            {"label": "LAW", "pattern": [{"LOWER": "gesetz"}, {"IS_DIGIT": True}, {"LOWER": "vom"}, {"IS_DIGIT": True}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["konformität", "regulatorisch"]}}, {"LOWER": {"IN": ["anforderung", "verpflichtung", "standard", "schwellenwert", "grenzwert"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "gemäß"}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "in"}, {"LOWER": "übereinstimmung"}, {"LOWER": "mit"}]},

            # Authorities
            {"label": "AUTHORITY", "pattern": [{"LOWER": {"IN": ["behörde", "kommission"]}}, {"LOWER": {"IN": ["europäische", "nationale", "bank", "finanz"]}}]},
        ]

    def _get_italian_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for Italian language regulatory entities"""
        return [
            # Regulations and laws
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["regolamento", "direttiva"]}}, {"TEXT": "(UE)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["regolamento", "direttiva"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(UE)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["articolo", "sezione", "paragrafo"]}}, {"IS_DIGIT": True}]},
            {"label": "LAW", "pattern": [{"LOWER": "legge"}, {"IS_DIGIT": True}, {"LOWER": "del"}, {"IS_DIGIT": True}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["conformità", "regolamentare"]}}, {"LOWER": {"IN": ["requisito", "obbligo", "standard", "soglia", "limite"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "conformemente"}, {"LOWER": "a"}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "in"}, {"LOWER": "conformità"}, {"LOWER": "con"}]},

            # Authorities
            {"label": "AUTHORITY", "pattern": [{"LOWER": {"IN": ["autorità", "commissione"]}}, {"LOWER": {"IN": ["europea", "nazionale", "bancaria", "finanziaria"]}}]},
        ]

    def _get_portuguese_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for Portuguese language regulatory entities"""
        return [
            # Regulations and laws
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["regulamento", "diretiva"]}}, {"TEXT": "(UE)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["regulamento", "diretiva"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(UE)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["artigo", "seção", "parágrafo"]}}, {"IS_DIGIT": True}]},
            {"label": "LAW", "pattern": [{"LOWER": "lei"}, {"IS_DIGIT": True}, {"LOWER": "de"}, {"IS_DIGIT": True}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["conformidade", "regulatório"]}}, {"LOWER": {"IN": ["requisito", "obrigação", "padrão", "limite"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "em"}, {"LOWER": "conformidade"}, {"LOWER": "com"}]},

            # Authorities
            {"label": "AUTHORITY", "pattern": [{"LOWER": {"IN": ["autoridade", "comissão"]}}, {"LOWER": {"IN": ["europeia", "nacional", "bancária", "financeira"]}}]},
        ]

    def _get_dutch_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns for Dutch language regulatory entities"""
        return [
            # Regulations and laws
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["verordening", "richtlijn"]}}, {"TEXT": "(EU)"}, {"SHAPE": "dddd/dddd"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["verordening", "richtlijn"]}}, {"SHAPE": "dddd/dddd"}, {"TEXT": "(EU)"}]},
            {"label": "LAW", "pattern": [{"LOWER": {"IN": ["artikel", "sectie", "paragraaf"]}}, {"IS_DIGIT": True}]},
            {"label": "LAW", "pattern": [{"LOWER": "wet"}, {"IS_DIGIT": True}, {"LOWER": "van"}, {"IS_DIGIT": True}]},

            # Compliance terms
            {"label": "COMPLIANCE", "pattern": [{"LOWER": {"IN": ["naleving", "regelgeving"]}}, {"LOWER": {"IN": ["vereiste", "verplichting", "standaard", "drempel", "limiet"]}}]},
            {"label": "COMPLIANCE", "pattern": [{"LOWER": "in"}, {"LOWER": "overeenstemming"}, {"LOWER": "met"}]},

            # Authorities
            {"label": "AUTHORITY", "pattern": [{"LOWER": {"IN": ["autoriteit", "commissie"]}}, {"LOWER": {"IN": ["europese", "nationale", "bank", "financieel"]}}]},
        ]

    def _add_regulatory_patterns(self, nlp: Language, lang_code: str = 'en') -> None:
        """
        Add regulatory-specific patterns to spaCy's entity recognition for a specific language.

        Args:
            nlp: spaCy Language model to modify
            lang_code: Language code for patterns
        """
        try:
            # Get patterns for this language
            if lang_code in self.regulatory_patterns_by_language:
                patterns = self.regulatory_patterns_by_language[lang_code]
            else:
                # Fall back to English if language not supported
                patterns = self.regulatory_patterns_by_language['en']

            # Initialize entity ruler if not present
            if "entity_ruler" not in nlp.pipe_names:
                ruler = nlp.add_pipe("entity_ruler", before="ner")
            else:
                ruler = nlp.get_pipe("entity_ruler")

            # Add patterns to the entity ruler
            ruler.add_patterns(patterns)
            logger.info(f"Added regulatory patterns for language: {lang_code}")

        except Exception as e:
            logger.error(f"Failed to add regulatory patterns for language {lang_code}: {str(e)}")

    def extract_entities(self, text: str, lang_code: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text using spaCy.

        Args:
            text: Text to extract entities from
            lang_code: Language code (auto-detected if not provided)

        Returns:
            Dictionary mapping entity types to lists of entity data
        """
        if not text:
            return {}

        # Detect language if not provided and multilingual is enabled
        language_info = None
        if self.multilingual:
            if not lang_code:
                language_info = self.language_detector.detect_language(text)
                lang_code = language_info['language_code']
                logger.info(f"Detected language: {language_info['language_name']} ({lang_code})")

        # Get appropriate NLP model
        nlp = None
        if self.multilingual:
            nlp = self._load_model_for_language(lang_code)
        else:
            nlp = self.nlp

        # If we don't have an NLP model, fall back to regex
        if not nlp:
            return self._extract_entities_fallback(text)

        try:
            # Process text with spaCy
            doc = nlp(text)

            # Extract entities
            entities = {}

            for ent in doc.ents:
                # Map spaCy entity type to our schema
                entity_type = self.entity_type_mapping.get(ent.label_, "other")

                if entity_type not in entities:
                    entities[entity_type] = []

                # Prepare entity data
                entity_data = {
                    "text": ent.text,
                    "normalized": ent.text.lower() if entity_type not in ["person", "organization", "location"] else ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "original_type": ent.label_,
                    "confidence": 1.0,  # Default confidence for spaCy entities
                    "lemma": ent.lemma_ if hasattr(ent, "lemma_") else ent.text,
                    "language": lang_code if lang_code else "en"
                }

                # Check if entity is already in the list (avoid duplicates)
                if not any(e["text"] == ent.text for e in entities[entity_type]):
                    entities[entity_type].append(entity_data)

            # If no entities were found, try fallback method
            if not entities:
                fallback_entities = self._extract_entities_fallback(text)

                # Merge fallback entities with lower confidence
                for entity_type, entity_list in fallback_entities.items():
                    if entity_type not in entities:
                        entities[entity_type] = []

                    for entity in entity_list:
                        entity["confidence"] = 0.7  # Lower confidence for regex-based entities
                        entity["language"] = lang_code if lang_code else "en"
                        if not any(e["text"] == entity["text"] for e in entities[entity_type]):
                            entities[entity_type].append(entity)

            return entities

        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {str(e)}")
            return self._extract_entities_fallback(text)

    def _extract_entities_fallback(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fallback entity extraction using regex patterns when spaCy is unavailable.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary mapping entity types to lists of entity data
        """
        entity_patterns = {
            'person': r'(?:[A-Z][a-z]+ ){1,2}[A-Z][a-z]+',
            'organization': r'(?:[A-Z][a-z]* ){1,4}(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Company|Foundation|Association)',
            'location': r'(?:[A-Z][a-z]+ ){0,1}[A-Z][a-z]+,? (?:[A-Z][a-z]+|[A-Z]{2})',
            'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'regulation': r'\b(?:Act|Law|Regulation|Directive|Code|Rule|Statute|Guideline)[s]? (?:of|on|for) .{5,50}',
            'standard': r'\b[A-Z]{2,10} \d{1,5}(?:[-.]\d+){0,3}\b',
        }

        entities = {}

        for entity_type, pattern in entity_patterns.items():
            entities[entity_type] = []
            matches = re.finditer(pattern, text)

            for match in matches:
                entity_text = match.group(0).strip()

                # Skip if already added
                if entity_type in entities and any(e["text"] == entity_text for e in entities[entity_type]):
                    continue

                entity_data = {
                    "text": entity_text,
                    "normalized": entity_text.lower() if entity_type not in ["person", "organization", "location"] else entity_text,
                    "start": match.start(),
                    "end": match.end(),
                    "original_type": entity_type,
                    "confidence": 0.6,  # Lower confidence for regex matches
                    "lemma": entity_text.lower()
                }

                if entity_type not in entities:
                    entities[entity_type] = []

                entities[entity_type].append(entity_data)

        return entities

    def get_entity_relations(self, text: str, lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Extract relations between entities in text.

        Args:
            text: Text to extract relations from
            lang_code: Language code (auto-detected if not provided)

        Returns:
            List of relation data dictionaries
        """
        # Detect language if not provided and multilingual is enabled
        if self.multilingual and not lang_code:
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']
            logger.info(f"Detected language for relation extraction: {language_info['language_name']} ({lang_code})")

        # Get appropriate NLP model
        nlp = None
        if self.multilingual:
            nlp = self._load_model_for_language(lang_code)
        else:
            nlp = self.nlp

        # If we don't have an NLP model, return empty list
        if not nlp:
            return []

        try:
            # Process text with spaCy
            doc = nlp(text)

            relations = []

            # Find potential relations based on syntactic dependencies
            for token in doc:
                # Look for subject-verb-object patterns
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    # Find subject
                    subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]

                    # Find object
                    objects = [child for child in token.children if child.dep_ in ["dobj", "pobj", "attr"]]

                    # Create relations if both subject and object are present
                    for subj in subjects:
                        for obj in objects:
                            # Check if subject or object are entities
                            subj_ent = None
                            obj_ent = None

                            for ent in doc.ents:
                                if subj.i >= ent.start and subj.i < ent.end:
                                    subj_ent = ent
                                if obj.i >= ent.start and obj.i < ent.end:
                                    obj_ent = ent

                            # If at least one entity is involved, create relation
                            if subj_ent or obj_ent:
                                relation = {
                                    "source": subj_ent.text if subj_ent else subj.text,
                                    "source_type": subj_ent.label_ if subj_ent else None,
                                    "relation": token.lemma_,
                                    "target": obj_ent.text if obj_ent else obj.text,
                                    "target_type": obj_ent.label_ if obj_ent else None,
                                    "sentence": doc[doc[subj.i].sent.start:doc[obj.i].sent.end].text,
                                    "confidence": 0.8,
                                    "language": lang_code if lang_code else "en"
                                }
                                relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"Error extracting entity relations: {str(e)}")
            return []
