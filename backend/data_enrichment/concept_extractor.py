# plugins/regul_aite/backend/data_enrichment/concept_extractor.py
import logging
import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from typing import List, Dict, Any, Set, Optional, Union
import re
import os
import json
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class ConceptExtractor:
    """
    Extracts key concepts, phrases, and terminology from text, with a focus on regulatory content.
    Uses spaCy for linguistic analysis and custom matchers for domain-specific concept extraction.
    Supports multilingual documents through language detection and appropriate model selection.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        regulatory_domain: bool = True,
        custom_glossary: Optional[str] = None,
        multilingual: bool = True,
        max_models: int = 3
    ):
        """
        Initialize the concept extractor.

        Args:
            spacy_model: Name of default spaCy model to use
            regulatory_domain: Whether to include regulatory domain knowledge
            custom_glossary: Path to custom glossary JSON file
            multilingual: Whether to enable multilingual support
            max_models: Maximum number of language models to keep in memory
        """
        self.spacy_model_name = spacy_model
        self.regulatory_domain = regulatory_domain
        self.multilingual = multilingual

        # Initialize language detector
        self.language_detector = LanguageDetector(fallback_language='en')

        # Dictionary to store NLP models by language
        self.models = {}
        self.matchers = {}  # Store phrase matchers by language

        # Load default spaCy model
        if multilingual:
            self.nlp = self._load_model_for_language('en', spacy_model)
        else:
            # Load single spaCy model
            try:
                logger.info(f"Loading spaCy model: {spacy_model}")
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Successfully loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {str(e)}")
                self.nlp = None

        # Load regulatory domain knowledge if requested
        if regulatory_domain:
            self._load_regulatory_knowledge()

        # Load custom glossary if provided
        self.custom_glossary = None
        if custom_glossary and os.path.exists(custom_glossary):
            try:
                with open(custom_glossary, 'r', encoding='utf-8') as f:
                    self.custom_glossary = json.load(f)
                logger.info(f"Loaded custom glossary from {custom_glossary}")
            except Exception as e:
                logger.error(f"Failed to load custom glossary: {str(e)}")

        # Set up phrase matcher for concepts if NLP is available
        if self.nlp:
            self._setup_phrase_matcher('en')

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

            # Set up regulatory components for this language model
            if self.regulatory_domain:
                self._load_regulatory_knowledge_for_language(nlp, lang_code)

            # Set up phrase matcher for this language
            self._setup_phrase_matcher(lang_code, nlp)

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

    def _load_regulatory_knowledge(self):
        """Load regulatory domain knowledge for multiple languages"""
        # Define common regulatory terminology for each language
        self.regulatory_terms = {
            'en': {
                "compliance": ["compliance", "regulatory compliance", "adherence", "conformity"],
                "risk": ["risk", "risk management", "risk assessment", "risk analysis", "risk mitigation"],
                "governance": ["governance", "corporate governance", "governance framework"],
                "reporting": ["reporting", "disclosure", "notification", "report", "statement"],
                "audit": ["audit", "inspection", "examination", "review", "assessment"],
                "sanction": ["sanction", "penalty", "fine", "enforcement action"],
                "requirement": ["requirement", "obligation", "mandate", "duty", "necessity"],
                "procedure": ["procedure", "process", "protocol", "method", "guideline"],
                "documentation": ["documentation", "record", "evidence", "paperwork"],
                "policy": ["policy", "standard", "rule", "regulation", "law", "directive"]
            },
            'es': {
                "compliance": ["cumplimiento", "cumplimiento normativo", "adherencia", "conformidad"],
                "risk": ["riesgo", "gestión de riesgos", "evaluación de riesgos", "análisis de riesgos"],
                "governance": ["gobernanza", "gobierno corporativo", "marco de gobernanza"],
                "reporting": ["informes", "divulgación", "notificación", "reporte", "declaración"],
                "audit": ["auditoría", "inspección", "examen", "revisión", "evaluación"],
                "sanction": ["sanción", "penalización", "multa", "acción de ejecución"],
                "requirement": ["requisito", "obligación", "mandato", "deber", "necesidad"],
                "procedure": ["procedimiento", "proceso", "protocolo", "método", "directriz"],
                "documentation": ["documentación", "registro", "evidencia", "papeleo"],
                "policy": ["política", "estándar", "regla", "reglamento", "ley", "directiva"]
            },
            'fr': {
                "compliance": ["conformité", "conformité réglementaire", "adhésion", "conformité"],
                "risk": ["risque", "gestion des risques", "évaluation des risques", "analyse des risques"],
                "governance": ["gouvernance", "gouvernance d'entreprise", "cadre de gouvernance"],
                "reporting": ["reporting", "divulgation", "notification", "rapport", "déclaration"],
                "audit": ["audit", "inspection", "examen", "révision", "évaluation"],
                "sanction": ["sanction", "pénalité", "amende", "mesure d'application"],
                "requirement": ["exigence", "obligation", "mandat", "devoir", "nécessité"],
                "procedure": ["procédure", "processus", "protocole", "méthode", "directive"],
                "documentation": ["documentation", "enregistrement", "preuve", "paperasse"],
                "policy": ["politique", "norme", "règle", "règlement", "loi", "directive"]
            },
            'de': {
                "compliance": ["Compliance", "Regelkonformität", "Einhaltung", "Konformität"],
                "risk": ["Risiko", "Risikomanagement", "Risikobewertung", "Risikoanalyse"],
                "governance": ["Governance", "Unternehmensführung", "Governance-Rahmen"],
                "reporting": ["Berichterstattung", "Offenlegung", "Meldung", "Bericht", "Erklärung"],
                "audit": ["Prüfung", "Inspektion", "Untersuchung", "Überprüfung", "Bewertung"],
                "sanction": ["Sanktion", "Strafe", "Geldbuße", "Durchsetzungsmaßnahme"],
                "requirement": ["Anforderung", "Verpflichtung", "Mandat", "Pflicht", "Notwendigkeit"],
                "procedure": ["Verfahren", "Prozess", "Protokoll", "Methode", "Richtlinie"],
                "documentation": ["Dokumentation", "Aufzeichnung", "Nachweis", "Papierkram"],
                "policy": ["Richtlinie", "Standard", "Regel", "Verordnung", "Gesetz", "Direktive"]
            },
            'it': {
                "compliance": ["conformità", "conformità normativa", "aderenza", "conformità"],
                "risk": ["rischio", "gestione dei rischi", "valutazione dei rischi", "analisi dei rischi"],
                "governance": ["governance", "corporate governance", "quadro di governance"],
                "reporting": ["reportistica", "divulgazione", "notifica", "rapporto", "dichiarazione"],
                "audit": ["audit", "ispezione", "esame", "revisione", "valutazione"],
                "sanction": ["sanzione", "penalità", "multa", "azione di applicazione"],
                "requirement": ["requisito", "obbligo", "mandato", "dovere", "necessità"],
                "procedure": ["procedura", "processo", "protocollo", "metodo", "linea guida"],
                "documentation": ["documentazione", "registrazione", "evidenza", "scartoffie"],
                "policy": ["politica", "standard", "regola", "regolamento", "legge", "direttiva"]
            }
        }

        # Load for default language (English)
        self._load_regulatory_knowledge_for_language(self.nlp, 'en')

    def _load_regulatory_knowledge_for_language(self, nlp: Language, lang_code: str):
        """
        Load regulatory domain knowledge for a specific language.

        Args:
            nlp: spaCy Language model to modify
            lang_code: Language code for terminology
        """
        if not nlp:
            return

        # Get terminology for this language or fall back to English
        terms = self.regulatory_terms.get(lang_code, self.regulatory_terms.get('en', {}))

        # Add custom component for concept detection if needed
        if "regulatory_concept_detector" not in nlp.pipe_names:
            @Language.component("regulatory_concept_detector")
            def regulatory_concept_detector(doc: Doc) -> Doc:
                """Custom component to detect regulatory concepts"""
                doc.user_data["regulatory_concepts"] = []

                # Find regulatory concepts in text
                for concept_type, term_list in terms.items():
                    for term in term_list:
                        if term.lower() in doc.text.lower():
                            # Try to find the exact position
                            pos = doc.text.lower().find(term.lower())
                            if pos >= 0:
                                doc.user_data["regulatory_concepts"].append({
                                    "type": concept_type,
                                    "term": term,
                                    "position": pos,
                                    "length": len(term)
                                })

                return doc

            nlp.add_pipe("regulatory_concept_detector", last=True)
            logger.info(f"Added regulatory concept detector for language: {lang_code}")

    def _setup_phrase_matcher(self, lang_code: str, nlp: Optional[Language] = None):
        """
        Set up phrase matcher for concept extraction for a specific language.

        Args:
            lang_code: Language code
            nlp: spaCy Language model (uses class attribute if None)
        """
        if nlp is None:
            nlp = self.nlp if lang_code == 'en' else self.models.get(lang_code)

        if not nlp:
            logger.warning(f"Cannot set up phrase matcher for language {lang_code}: no NLP model available")
            return

        # Create a new matcher for this language
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        # Add regulatory terminology if available
        if hasattr(self, 'regulatory_terms') and lang_code in self.regulatory_terms:
            for concept_type, terms in self.regulatory_terms[lang_code].items():
                patterns = [nlp.make_doc(term) for term in terms]
                matcher.add(concept_type, patterns)

        # Add custom glossary terms if available
        if self.custom_glossary:
            for concept_type, terms in self.custom_glossary.items():
                patterns = [nlp.make_doc(term) for term in terms]
                matcher.add(concept_type, patterns)

        # Store the matcher
        self.matchers[lang_code] = matcher
        logger.info(f"Set up phrase matcher for language: {lang_code}")

    def extract_concepts(self, text: str, lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Extract key concepts from text.

        Args:
            text: Text to extract concepts from
            lang_code: Language code (auto-detected if not provided)

        Returns:
            List of concept data dictionaries
        """
        if not text:
            return []

        # Detect language if not provided and multilingual is enabled
        language_info = None
        if self.multilingual:
            if not lang_code:
                language_info = self.language_detector.detect_language(text)
                lang_code = language_info['language_code']
                logger.info(f"Detected language for concept extraction: {language_info['language_name']} ({lang_code})")

        # Get appropriate NLP model and matcher
        nlp = None
        matcher = None
        if self.multilingual:
            nlp = self._load_model_for_language(lang_code)
            matcher = self.matchers.get(lang_code)

            # If we don't have a matcher for this language, create one
            if nlp and not matcher:
                self._setup_phrase_matcher(lang_code, nlp)
                matcher = self.matchers.get(lang_code)
        else:
            nlp = self.nlp
            matcher = self.matchers.get('en')

        # If we don't have NLP or multilingual support, fall back to regex
        if not nlp:
            return self._extract_concepts_fallback(text, lang_code)

        try:
            # Process text with spaCy
            doc = nlp(text)

            # Initialize concept list
            concepts = []

            # 1. Extract noun chunks as potential concepts
            for chunk in doc.noun_chunks:
                # Filter out common pronouns and short chunks
                if (len(chunk) > 1 and
                    not chunk.text.lower() in ["i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"] and
                    not chunk.root.is_stop):

                    concept_data = {
                        "text": chunk.text,
                        "lemma": chunk.root.lemma_,
                        "type": "noun_phrase",
                        "span": [chunk.start_char, chunk.end_char],
                        "confidence": 0.7,
                        "sentence": chunk.sent.text if hasattr(chunk, "sent") else "",
                        "language": lang_code if lang_code else "en"
                    }

                    # Add if not already in the list
                    if not any(c["text"].lower() == chunk.text.lower() for c in concepts):
                        concepts.append(concept_data)

            # 2. Extract concepts using the phrase matcher
            if matcher:
                matches = matcher(doc)
                for match_id, start, end in matches:
                    match_type = nlp.vocab.strings[match_id]
                    span = doc[start:end]

                    concept_data = {
                        "text": span.text,
                        "lemma": span.lemma_ if hasattr(span, "lemma_") else span.text,
                        "type": match_type,
                        "span": [span.start_char, span.end_char],
                        "confidence": 0.9,  # Higher confidence for glossary matches
                        "sentence": span.sent.text if hasattr(span, "sent") else "",
                        "language": lang_code if lang_code else "en"
                    }

                    # Add if not already in the list
                    if not any(c["text"].lower() == span.text.lower() for c in concepts):
                        concepts.append(concept_data)

            # 3. Look for regulatory concepts in user_data
            if hasattr(doc, "user_data") and "regulatory_concepts" in doc.user_data:
                for reg_concept in doc.user_data["regulatory_concepts"]:
                    # Find the concept in the text
                    concept_text = reg_concept["term"]
                    concept_pos = reg_concept.get("position", text.lower().find(concept_text.lower()))

                    if concept_pos >= 0:
                        concept_data = {
                            "text": concept_text,
                            "lemma": concept_text.lower(),
                            "type": reg_concept["type"],
                            "span": [concept_pos, concept_pos + len(concept_text)],
                            "confidence": 0.85,
                            "sentence": "",  # We don't have the sentence context here
                            "language": lang_code if lang_code else "en"
                        }

                        # Add if not already in the list
                        if not any(c["text"].lower() == concept_text.lower() for c in concepts):
                            concepts.append(concept_data)

            # 4. Look for important compound nouns or subject/objects
            for token in doc:
                # Check for nouns with compound modifiers or nouns that are subjects/objects
                if (token.pos_ == "NOUN" and
                    (any(child.dep_ == "compound" for child in token.children) or
                     token.dep_ in ["nsubj", "dobj", "pobj"]) and
                    not token.is_stop):

                    # Get the full phrase (token with its compound modifiers)
                    phrase_tokens = [child for child in token.children if child.dep_ == "compound"]
                    phrase_tokens.append(token)
                    phrase_tokens.sort(key=lambda t: t.i)  # Sort by position in doc

                    phrase_text = " ".join([t.text for t in phrase_tokens])

                    concept_data = {
                        "text": phrase_text,
                        "lemma": token.lemma_,
                        "type": "compound_noun",
                        "span": [phrase_tokens[0].idx, phrase_tokens[-1].idx + len(phrase_tokens[-1].text)],
                        "confidence": 0.75,
                        "sentence": token.sent.text if hasattr(token, "sent") else "",
                        "language": lang_code if lang_code else "en"
                    }

                    # Add if not already in the list and if it's a substantial phrase
                    if (len(phrase_text) > 3 and  # Longer than 3 chars
                        not any(c["text"].lower() == phrase_text.lower() for c in concepts)):
                        concepts.append(concept_data)

            # Filter concepts to remove duplicates and ensure quality
            filtered_concepts = self._filter_concepts(concepts, lang_code)

            # Add language information to all concepts
            for concept in filtered_concepts:
                if "language" not in concept:
                    concept["language"] = lang_code if lang_code else "en"

            return filtered_concepts

        except Exception as e:
            logger.error(f"Error in spaCy concept extraction: {str(e)}")
            return self._extract_concepts_fallback(text, lang_code)

    def _extract_concepts_fallback(self, text: str, lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Fallback concept extraction using regex patterns.

        Args:
            text: Text to extract concepts from
            lang_code: Language code

        Returns:
            List of concept data dictionaries
        """
        concepts = []

        # Simple patterns for concepts
        concept_patterns = {
            "noun_phrase": r'\b(?:[A-Z][a-z]+ ){1,3}(?:[A-Z][a-z]+)\b',
            "technical_term": r'\b(?:[A-Za-z]+-?)+(?:ing|ion|ment|ance|ence|ity|ness|ship)\b',
            "regulatory_term": r'\b(?:compliance|regulatory|governance|reporting|audit|sanction|requirement|procedure|documentation|policy)\b'
        }

        # Extract concepts using patterns
        for concept_type, pattern in concept_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                concept_text = match.group(0).strip()

                # Skip very short concepts
                if len(concept_text) < 4:
                    continue

                # Skip if only stopwords
                if concept_text.lower() in ["this", "that", "these", "those", "they", "them", "their", "there"]:
                    continue

                # Create concept data
                concept_data = {
                    "text": concept_text,
                    "lemma": concept_text.lower(),
                    "type": concept_type,
                    "span": [match.start(), match.end()],
                    "confidence": 0.6,  # Lower confidence for regex matches
                    "sentence": "",  # We don't have sentence context in regex
                    "language": lang_code if lang_code else "en"
                }

                # Add if not already in the list
                if not any(c["text"].lower() == concept_text.lower() for c in concepts):
                    concepts.append(concept_data)

        # Add regulatory terms if applicable
        if self.regulatory_domain:
            # Get terms for this language or fall back to English
            terms = {}
            if lang_code and lang_code in self.regulatory_terms:
                terms = self.regulatory_terms[lang_code]
            elif 'en' in self.regulatory_terms:
                terms = self.regulatory_terms['en']

            # Check for each term in the text
            for category, term_list in terms.items():
                for term in term_list:
                    if term.lower() in text.lower():
                        pos = text.lower().find(term.lower())

                        concept_data = {
                            "text": term,
                            "lemma": term.lower(),
                            "type": "regulatory",
                            "category": category,
                            "span": [pos, pos + len(term)],
                            "confidence": 0.75,
                            "sentence": "",
                            "language": lang_code if lang_code else "en"
                        }

                        # Add if not already in the list
                        if not any(c["text"].lower() == term.lower() for c in concepts):
                            concepts.append(concept_data)

        return self._filter_concepts(concepts, lang_code)

    def _filter_concepts(self, concepts: List[Dict[str, Any]], lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Filter and normalize concepts.

        Args:
            concepts: List of concept data dictionaries
            lang_code: Language code

        Returns:
            Filtered list of concept data dictionaries
        """
        filtered_concepts = []
        seen_concepts = set()

        # Build language-specific stopword list
        stopwords = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "to", "of", "for", "with", "by"}

        # Add language-specific stopwords
        if lang_code == 'es':
            stopwords.update({"el", "la", "los", "las", "un", "una", "y", "o", "pero", "si", "cuando", "de", "para", "con", "por"})
        elif lang_code == 'fr':
            stopwords.update({"le", "la", "les", "un", "une", "et", "ou", "mais", "si", "quand", "de", "pour", "avec", "par"})
        elif lang_code == 'de':
            stopwords.update({"der", "die", "das", "ein", "eine", "und", "oder", "aber", "wenn", "dann", "zu", "von", "für", "mit", "durch"})

        for concept in concepts:
            # Normalize text for deduplication
            normalized_text = concept["text"].lower().strip()

            # Skip very short concepts
            if len(normalized_text) < 3:
                continue

            # Skip concepts that are just stopwords
            if normalized_text in stopwords:
                continue

            # Skip if all words are stopwords
            if all(word.lower() in stopwords for word in normalized_text.split()):
                continue

            # Skip if already seen
            if normalized_text in seen_concepts:
                continue

            seen_concepts.add(normalized_text)

            # Add language if not present
            if "language" not in concept:
                concept["language"] = lang_code if lang_code else "en"

            filtered_concepts.append(concept)

        return filtered_concepts

    def extract_key_phrases(self, text: str, lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Extract key phrases from text, similar to keywords.

        Args:
            text: Text to extract key phrases from
            lang_code: Language code (auto-detected if not provided)

        Returns:
            List of key phrase data dictionaries
        """
        if not text:
            return []

        # Detect language if not provided and multilingual is enabled
        if self.multilingual and not lang_code:
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']
            logger.info(f"Detected language for key phrase extraction: {language_info['language_name']} ({lang_code})")

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

            # Initialize key phrases list
            key_phrases = []

            # Extract sentences
            sentences = list(doc.sents)

            for sent in sentences:
                # Skip very short sentences
                if len(sent) < 5:
                    continue

                # Skip sentences without a main verb
                if not any(token.pos_ == "VERB" for token in sent):
                    continue

                # Find the root of the sentence
                root = None
                for token in sent:
                    if token.dep_ == "ROOT":
                        root = token
                        break

                if not root:
                    continue

                # Extract subject-verb-object phrases
                subjects = [child for child in root.children if child.dep_ in ["nsubj", "nsubjpass"]]
                objects = [child for child in root.children if child.dep_ in ["dobj", "pobj", "attr"]]

                for subj in subjects:
                    # Get full subject phrase
                    subj_phrase = self._get_full_phrase(subj)

                    for obj in objects:
                        # Get full object phrase
                        obj_phrase = self._get_full_phrase(obj)

                        # Create key phrase
                        key_phrase = {
                            "text": f"{subj_phrase} {root.text} {obj_phrase}",
                            "subject": subj_phrase,
                            "verb": root.text,
                            "object": obj_phrase,
                            "sentence": sent.text,
                            "language": lang_code if lang_code else "en"
                        }

                        key_phrases.append(key_phrase)

            return key_phrases

        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            return []

    def _get_full_phrase(self, token) -> str:
        """
        Get the full phrase for a token, including its dependents.

        Args:
            token: spaCy token

        Returns:
            String containing the full phrase
        """
        phrase_tokens = [token]

        # Get all dependents
        for child in token.children:
            if child.dep_ in ["compound", "amod", "det", "prep", "poss"]:
                # Recursively get the child's phrase
                child_phrase = self._get_full_phrase(child)
                phrase_tokens.append(child)

        # Sort tokens by position
        phrase_tokens.sort(key=lambda t: t.i)

        # Combine tokens
        return " ".join([t.text for t in phrase_tokens])

    def extract_domain_concepts(self, text: str, domain: str, lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Extract domain-specific concepts based on a specified domain.

        Args:
            text: Text to extract concepts from
            domain: Domain for concept extraction (regulatory, financial, legal, etc.)
            lang_code: Language code (auto-detected if not provided)

        Returns:
            List of domain-specific concept data
        """
        # Extract general concepts first
        concepts = self.extract_concepts(text, lang_code)

        # Filter for domain-specific concepts
        domain_concepts = []

        # Define domain-specific keywords for filtering
        domain_keywords = {
            "regulatory": [
                "regulation", "compliance", "requirement", "standard", "law", "directive",
                "obligation", "mandate", "rule", "governance", "policy", "procedure"
            ],
            "financial": [
                "finance", "financial", "capital", "investment", "fund", "asset", "liability",
                "revenue", "expense", "profit", "loss", "budget", "transaction"
            ],
            "legal": [
                "legal", "law", "statute", "regulation", "court", "judge", "lawsuit", "plaintiff",
                "defendant", "contract", "agreement", "clause", "jurisdiction"
            ]
        }

        # Get keywords for the specified domain
        keywords = domain_keywords.get(domain.lower(), [])

        if keywords:
            # Filter concepts by domain keywords
            for concept in concepts:
                concept_text = concept["text"].lower()

                # Check if concept contains any domain keywords
                if any(keyword in concept_text for keyword in keywords):
                    # Mark as domain concept
                    concept["domain"] = domain
                    domain_concepts.append(concept)

        return domain_concepts
