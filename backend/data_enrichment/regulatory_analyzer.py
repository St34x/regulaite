# plugins/regul_aite/backend/data_enrichment/regulatory_analyzer.py
import logging
import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
import datetime
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class RegulatoryAnalyzer:
    """
    Analyzes regulatory content to extract specialized metadata and relationships.
    Supports multiple languages for regulatory term extraction and analysis.
    """

    def __init__(self, multilingual: bool = True):
        """
        Initialize the regulatory analyzer.

        Args:
            multilingual: Whether to enable multilingual support
        """
        self.multilingual = multilingual

        # Initialize language detector if multilingual is enabled
        if multilingual:
            self.language_detector = LanguageDetector()

        # Patterns for various regulatory elements by language
        self.patterns = {
            'en': self._get_english_patterns(),
            'es': self._get_spanish_patterns(),
            'fr': self._get_french_patterns(),
            'de': self._get_german_patterns(),
            'it': self._get_italian_patterns(),
            'pt': self._get_portuguese_patterns(),
            'nl': self._get_dutch_patterns()
        }

    def _get_english_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for English language regulatory elements"""
        return {
            "eu_legislation": [
                # EU regulations, directives, decisions
                r"(?:Regulation|Directive|Decision)\s+(?:\(EU\)|\(EC\))\s+(\d{4}/\d+)",
                r"(?:Regulation|Directive|Decision)\s+(\d{4}/\d+/(?:EU|EC))"
            ],
            "us_legislation": [
                r"(?:Public Law|P\.L\.) (\d{1,3}-\d{1,3})",
                r"(?:\d{1,3}) U\.S\.C\. (?:\d{1,4})(?:\([a-z]\))?"
            ],
            "deadlines": [
                # Dates
                r"(?:by|before|until|not later than)\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
                r"deadline\s+(?:of|for|on)?\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
                r"(?:due|submission)\s+date\s+(?:of|for|on)?\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
                # Quarters
                r"(?:by|before|until|not later than)\s+(?:the end of\s+)?(?:Q|Quarter\s+)([1-4])\s+(\d{4})",
                # General days/months
                r"within\s+(\d+)\s+(days|months|years)"
            ],
            "requirements": [
                r"(?:shall|must|is required to|are required to)\s+([^,.;:]+)",
                r"requirement\s+to\s+([^,.;:]+)",
                r"required\s+to\s+([^,.;:]+)",
                r"mandatory\s+(?:to|for)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"threshold\s+of\s+([^,.;:]+)",
                r"limit\s+of\s+([^,.;:]+)",
                r"at least\s+(\d+(?:\.\d+)?%?)",
                r"maximum\s+(?:of|level)?\s+(\d+(?:\.\d+)?%?)",
                r"minimum\s+(?:of|level)?\s+(\d+(?:\.\d+)?%?)"
            ],
            "authorities": [
                r"(?:report|submit|notify|inform)\s+(?:to|with)?\s+(?:the)?\s+([A-Z][a-zA-Z ]+Authority|[A-Z][a-zA-Z ]+Commission|[A-Z][a-zA-Z ]+Agency)",
                r"(?:approval|authorization|permission)\s+(?:from|by|of)?\s+(?:the)?\s+([A-Z][a-zA-Z ]+Authority|[A-Z][a-zA-Z ]+Commission|[A-Z][a-zA-Z ]+Agency)"
            ],
            "articles": [
                r"Article\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Section\s+(\d+(?:\.\d+)?)",
                r"Paragraph\s+(\d+(?:\.\d+)?)"
            ]
        }

    def _get_spanish_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for Spanish language regulatory elements"""
        return {
            "eu_legislation": [
                r"(?:Reglamento|Directiva|Decisión)\s+(?:\(UE\)|\(CE\))\s+(\d{4}/\d+)",
                r"(?:Reglamento|Directiva|Decisión)\s+(\d{4}/\d+/(?:UE|CE))"
            ],
            "deadlines": [
                # Dates
                r"(?:antes del|hasta el|no más tarde del)\s+(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4})",
                r"plazo\s+(?:límite|final)?\s+(?:del|el)?\s+(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4})",
                r"fecha\s+(?:de|límite)\s+(?:presentación|entrega)\s+(?:el)?\s+(\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4})",
                # Quarters
                r"(?:antes del|hasta el|para el)\s+(?:final del)?\s+(?:T|trimestre)\s+([1-4])\s+(?:de|del)?\s+(\d{4})",
                # General days/months
                r"en un plazo de\s+(\d+)\s+(días|meses|años)"
            ],
            "requirements": [
                r"(?:deberá|deben|debe|tendrá que|tienen que)\s+([^,.;:]+)",
                r"requisito\s+(?:de|para)\s+([^,.;:]+)",
                r"obligatorio\s+(?:para)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"umbral\s+de\s+([^,.;:]+)",
                r"límite\s+de\s+([^,.;:]+)",
                r"al menos\s+(\d+(?:\.\d+)?%?)",
                r"máximo\s+(?:de)?\s+(\d+(?:\.\d+)?%?)",
                r"mínimo\s+(?:de)?\s+(\d+(?:\.\d+)?%?)"
            ],
            "authorities": [
                r"(?:informar|notificar|comunicar)\s+(?:a|al|con)?\s+(?:la|el)?\s+([A-Z][a-zA-Z ]+Autoridad|[A-Z][a-zA-Z ]+Comisión|[A-Z][a-zA-Z ]+Agencia)",
                r"(?:aprobación|autorización|permiso)\s+(?:de|por|del)?\s+(?:la|el)?\s+([A-Z][a-zA-Z ]+Autoridad|[A-Z][a-zA-Z ]+Comisión|[A-Z][a-zA-Z ]+Agencia)"
            ],
            "articles": [
                r"Artículo\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Sección\s+(\d+(?:\.\d+)?)",
                r"Párrafo\s+(\d+(?:\.\d+)?)"
            ]
        }

    def _get_french_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for French language regulatory elements"""
        return {
            "eu_legislation": [
                r"(?:Règlement|Directive|Décision)\s+(?:\(UE\)|\(CE\))\s+(\d{4}/\d+)",
                r"(?:Règlement|Directive|Décision)\s+(\d{4}/\d+/(?:UE|CE))"
            ],
            "deadlines": [
                # Dates
                r"(?:avant le|jusqu'au|au plus tard le)\s+(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})",
                r"date\s+(?:limite|d'échéance|butoir)\s+(?:du|le)?\s+(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})",
                # Quarters
                r"(?:avant|jusqu'à|pour) la fin du\s+(?:T|trimestre)\s+([1-4])\s+(\d{4})",
                # General days/months
                r"dans un délai de\s+(\d+)\s+(jours|mois|ans|années)"
            ],
            "requirements": [
                r"(?:doit|doivent|devra|devront)\s+([^,.;:]+)",
                r"obligation\s+de\s+([^,.;:]+)",
                r"exigence\s+de\s+([^,.;:]+)",
                r"obligatoire\s+(?:de|pour)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"seuil\s+de\s+([^,.;:]+)",
                r"limite\s+de\s+([^,.;:]+)",
                r"au moins\s+(\d+(?:,\d+)?%?)",
                r"maximum\s+(?:de)?\s+(\d+(?:,\d+)?%?)",
                r"minimum\s+(?:de)?\s+(\d+(?:,\d+)?%?)"
            ],
            "authorities": [
                r"(?:déclarer|notifier|informer)\s+(?:à|auprès de)?\s+(?:l'|la|le)?\s+([A-Z][a-zA-Z ]+Autorité|[A-Z][a-zA-Z ]+Commission|[A-Z][a-zA-Z ]+Agence)",
                r"(?:approbation|autorisation|permission)\s+(?:de|par|de la)?\s+(?:l'|la|le)?\s+([A-Z][a-zA-Z ]+Autorité|[A-Z][a-zA-Z ]+Commission|[A-Z][a-zA-Z ]+Agence)"
            ],
            "articles": [
                r"Article\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Section\s+(\d+(?:\.\d+)?)",
                r"Paragraphe\s+(\d+(?:\.\d+)?)"
            ]
        }

    def _get_german_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for German language regulatory elements"""
        return {
            "eu_legislation": [
                r"(?:Verordnung|Richtlinie|Beschluss)\s+(?:\(EU\)|\(EG\))\s+(\d{4}/\d+)",
                r"(?:Verordnung|Richtlinie|Beschluss)\s+(\d{4}/\d+/(?:EU|EG))"
            ],
            "deadlines": [
                # Dates
                r"(?:bis|spätestens am|vor dem)\s+(\d{1,2}\.\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4})",
                r"(?:Termin|Frist|Stichtag)\s+(?:am|zum)?\s+(\d{1,2}\.\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4})",
                # Quarters
                r"(?:bis|vor|zum) Ende des\s+(?:Q|Quartal)\s+([1-4])\s+(\d{4})",
                # General days/months
                r"innerhalb von\s+(\d+)\s+(Tagen|Monaten|Jahren)"
            ],
            "requirements": [
                r"(?:muss|müssen|soll|sollen|ist verpflichtet|sind verpflichtet)\s+([^,.;:]+)",
                r"Anforderung\s+(?:an|für|zur)\s+([^,.;:]+)",
                r"erforderlich\s+(?:für|zu)?\s+([^,.;:]+)",
                r"verpflichtend\s+(?:für|zu)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"Schwellenwert\s+(?:von|für)\s+([^,.;:]+)",
                r"Grenzwert\s+(?:von|für)\s+([^,.;:]+)",
                r"mindestens\s+(\d+(?:,\d+)?%?)",
                r"höchstens\s+(\d+(?:,\d+)?%?)",
                r"maximal\s+(\d+(?:,\d+)?%?)",
                r"minimal\s+(\d+(?:,\d+)?%?)"
            ],
            "authorities": [
                r"(?:melden|mitteilen|informieren)\s+(?:an|bei)?\s+(?:die|der)?\s+([A-Z][a-zA-Z ]+Behörde|[A-Z][a-zA-Z ]+Kommission|[A-Z][a-zA-Z ]+Agentur)",
                r"(?:Genehmigung|Zulassung|Erlaubnis)\s+(?:von|durch|der)?\s+(?:die|der)?\s+([A-Z][a-zA-Z ]+Behörde|[A-Z][a-zA-Z ]+Kommission|[A-Z][a-zA-Z ]+Agentur)"
            ],
            "articles": [
                r"Artikel\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Abschnitt\s+(\d+(?:\.\d+)?)",
                r"Absatz\s+(\d+(?:\.\d+)?)"
            ]
        }

    def _get_italian_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for Italian language regulatory elements"""
        return {
            "eu_legislation": [
                r"(?:Regolamento|Direttiva|Decisione)\s+(?:\(UE\)|\(CE\))\s+(\d{4}/\d+)",
                r"(?:Regolamento|Direttiva|Decisione)\s+(\d{4}/\d+/(?:UE|CE))"
            ],
            "deadlines": [
                # Dates
                r"(?:entro il|fino al|non oltre il)\s+(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
                r"scadenza\s+(?:del|il)?\s+(\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4})",
                # Quarters
                r"(?:entro|fino alla) fine del\s+(?:T|trimestre)\s+([1-4])\s+(?:del)?\s+(\d{4})",
                # General days/months
                r"entro\s+(\d+)\s+(giorni|mesi|anni)"
            ],
            "requirements": [
                r"(?:deve|devono|è tenuto a|sono tenuti a)\s+([^,.;:]+)",
                r"requisito\s+(?:di|per)\s+([^,.;:]+)",
                r"obbligatorio\s+(?:per)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"soglia\s+di\s+([^,.;:]+)",
                r"limite\s+di\s+([^,.;:]+)",
                r"almeno\s+(\d+(?:,\d+)?%?)",
                r"massimo\s+(?:di)?\s+(\d+(?:,\d+)?%?)",
                r"minimo\s+(?:di)?\s+(\d+(?:,\d+)?%?)"
            ],
            "authorities": [
                r"(?:informare|notificare|comunicare)\s+(?:a|all')?\s+(?:la|l')?\s+([A-Z][a-zA-Z ]+Autorità|[A-Z][a-zA-Z ]+Commissione|[A-Z][a-zA-Z ]+Agenzia)",
                r"(?:approvazione|autorizzazione|permesso)\s+(?:da|dell')?\s+(?:la|l')?\s+([A-Z][a-zA-Z ]+Autorità|[A-Z][a-zA-Z ]+Commissione|[A-Z][a-zA-Z ]+Agenzia)"
            ],
            "articles": [
                r"Articolo\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Sezione\s+(\d+(?:\.\d+)?)",
                r"Paragrafo\s+(\d+(?:\.\d+)?)"
            ]
        }

    def _get_portuguese_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for Portuguese language regulatory elements"""
        return {
            "eu_legislation": [
                r"(?:Regulamento|Diretiva|Decisão)\s+(?:\(UE\)|\(CE\))\s+(\d{4}/\d+)",
                r"(?:Regulamento|Diretiva|Decisão)\s+(\d{4}/\d+/(?:UE|CE))"
            ],
            "deadlines": [
                # Dates
                r"(?:até|antes de|não depois de)\s+(\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4})",
                r"prazo\s+(?:até|limite)?\s+(?:de|a)?\s+(\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4})",
                # Quarters
                r"(?:até|antes do) fim do\s+(?:T|trimestre)\s+([1-4])\s+de\s+(\d{4})",
                # General days/months
                r"no prazo de\s+(\d+)\s+(dias|meses|anos)"
            ],
            "requirements": [
                r"(?:deve|devem|deverá|deverão)\s+([^,.;:]+)",
                r"requisito\s+(?:de|para)\s+([^,.;:]+)",
                r"obrigatório\s+(?:para)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"limiar\s+de\s+([^,.;:]+)",
                r"limite\s+de\s+([^,.;:]+)",
                r"pelo menos\s+(\d+(?:,\d+)?%?)",
                r"máximo\s+(?:de)?\s+(\d+(?:,\d+)?%?)",
                r"mínimo\s+(?:de)?\s+(\d+(?:,\d+)?%?)"
            ],
            "authorities": [
                r"(?:informar|notificar|comunicar)\s+(?:a|ao|à)?\s+(?:a|o)?\s+([A-Z][a-zA-Z ]+Autoridade|[A-Z][a-zA-Z ]+Comissão|[A-Z][a-zA-Z ]+Agência)",
                r"(?:aprovação|autorização|permissão)\s+(?:de|da|do)?\s+(?:a|o)?\s+([A-Z][a-zA-Z ]+Autoridade|[A-Z][a-zA-Z ]+Comissão|[A-Z][a-zA-Z ]+Agência)"
            ],
            "articles": [
                r"Artigo\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Seção\s+(\d+(?:\.\d+)?)",
                r"Parágrafo\s+(\d+(?:\.\d+)?)"
            ]
        }

    def _get_dutch_patterns(self) -> Dict[str, List[str]]:
        """Get patterns for Dutch language regulatory elements"""
        return {
            "eu_legislation": [
                r"(?:Verordening|Richtlijn|Besluit)\s+(?:\(EU\)|\(EG\))\s+(\d{4}/\d+)",
                r"(?:Verordening|Richtlijn|Besluit)\s+(\d{4}/\d+/(?:EU|EG))"
            ],
            "deadlines": [
                # Dates
                r"(?:vóór|uiterlijk op|niet later dan)\s+(\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+\d{4})",
                r"(?:deadline|uiterste datum)\s+(?:van|op)?\s+(\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+\d{4})",
                # Quarters
                r"(?:vóór|uiterlijk) eind\s+(?:Q|kwartaal)\s+([1-4])\s+(?:van)?\s+(\d{4})",
                # General days/months
                r"binnen\s+(\d+)\s+(dagen|maanden|jaar|jaren)"
            ],
            "requirements": [
                r"(?:moet|moeten|dient|dienen)\s+([^,.;:]+)",
                r"vereiste\s+(?:om|voor|tot)\s+([^,.;:]+)",
                r"verplicht\s+(?:om|voor|tot)?\s+([^,.;:]+)"
            ],
            "thresholds": [
                r"drempel\s+van\s+([^,.;:]+)",
                r"limiet\s+van\s+([^,.;:]+)",
                r"ten minste\s+(\d+(?:,\d+)?%?)",
                r"maximaal\s+(\d+(?:,\d+)?%?)",
                r"minimaal\s+(\d+(?:,\d+)?%?)"
            ],
            "authorities": [
                r"(?:melden|informeren|kennis geven)\s+(?:aan|bij)?\s+(?:de)?\s+([A-Z][a-zA-Z ]+Autoriteit|[A-Z][a-zA-Z ]+Commissie|[A-Z][a-zA-Z ]+Agentschap)",
                r"(?:goedkeuring|toestemming|vergunning)\s+(?:van|door|uit)?\s+(?:de)?\s+([A-Z][a-zA-Z ]+Autoriteit|[A-Z][a-zA-Z ]+Commissie|[A-Z][a-zA-Z ]+Agentschap)"
            ],
            "articles": [
                r"Artikel\s+(\d+(?:\.\d+)?(?:\([a-z]\))?)",
                r"Afdeling\s+(\d+(?:\.\d+)?)",
                r"Paragraaf\s+(\d+(?:\.\d+)?)"
            ]
        }

    def analyze_text(self, text: str, lang_code: str = None) -> Dict[str, Any]:
        """
        Analyze text for regulatory elements.

        Args:
            text: Text to analyze
            lang_code: Language code (auto-detected if not provided)

        Returns:
            Dictionary of extracted regulatory metadata
        """
        if not text:
            return {
                "legislation_references": [],
                "deadlines": [],
                "requirements": [],
                "thresholds": [],
                "authorities": [],
                "article_references": [],
                "summary": {
                    "has_regulatory_content": False,
                    "regulatory_score": 0
                }
            }

        # Detect language if not provided and multilingual is enabled
        if self.multilingual and not lang_code:
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']
            logger.info(f"Detected language for regulatory analysis: {language_info['language_name']} ({lang_code})")

        # If language is not supported, fall back to English
        if not lang_code or lang_code not in self.patterns:
            lang_code = 'en'

        # Get patterns for this language
        patterns = self.patterns[lang_code]

        results = {
            "legislation_references": self._extract_legislation(text, lang_code, patterns),
            "deadlines": self._extract_deadlines(text, lang_code, patterns),
            "requirements": self._extract_requirements(text, lang_code, patterns),
            "thresholds": self._extract_thresholds(text, lang_code, patterns),
            "authorities": self._extract_authorities(text, lang_code, patterns),
            "article_references": self._extract_articles(text, lang_code, patterns),
            "language": lang_code
        }

        # Generate summary
        results["summary"] = self._generate_summary(results)

        return results

    def _extract_legislation(self, text: str, lang_code: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract legislation references from text.

        Args:
            text: Text to analyze
            lang_code: Language code
            patterns: Patterns dictionary for the language

        Returns:
            List of legislation reference dictionaries
        """
        legislation = []

        # Process EU legislation patterns
        for pattern in patterns.get("eu_legislation", []):
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                full_match = match.group(0)
                reference = match.group(1)

                # Skip if already found
                if any(leg["reference"] == reference for leg in legislation):
                    continue

                # Try to determine the type
                leg_type = "Unknown"
                if re.search(r'regulation|verordening|règlement|verordnung|regolamento|regulamento', full_match.lower()):
                    leg_type = "Regulation"
                elif re.search(r'directive|richtlijn|directiva|richtlinie|direttiva|diretiva', full_match.lower()):
                    leg_type = "Directive"
                elif re.search(r'decision|besluit|décision|beschluss|decisione|decisão', full_match.lower()):
                    leg_type = "Decision"

                legislation.append({
                    "type": leg_type,
                    "reference": reference,
                    "text": full_match,
                    "span": [match.start(), match.end()],
                    "language": lang_code
                })

        # Process US legislation patterns for English
        if lang_code == 'en' and "us_legislation" in patterns:
            for pattern in patterns["us_legislation"]:
                matches = re.finditer(pattern, text, re.IGNORECASE)

                for match in matches:
                    full_match = match.group(0)
                    reference = match.group(1) if match.groups() else full_match

                    # Skip if already found
                    if any(leg["reference"] == reference for leg in legislation):
                        continue

                    # Determine the type based on pattern
                    leg_type = "US Law"
                    if "U.S.C." in full_match:
                        leg_type = "US Code"

                    legislation.append({
                        "type": leg_type,
                        "reference": reference,
                        "text": full_match,
                        "span": [match.start(), match.end()],
                        "language": lang_code
                    })

        return legislation

    def _extract_deadlines(self, text: str, lang_code: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract deadline references from text.

        Args:
            text: Text to analyze
            lang_code: Language code
            patterns: Patterns dictionary for the language

        Returns:
            List of deadline dictionaries
        """
        deadlines = []

        for pattern in patterns.get("deadlines", []):
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                full_match = match.group(0)

                try:
                    # For specific dates
                    if any(month in full_match.lower() for month in [
                        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                        "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
                        "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre",
                        "januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember",
                        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
                        "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
                        "januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"
                    ]):
                        # Extract the date component - this varies by language
                        date_str = match.group(1)

                        # Remove ordinal suffixes for English
                        if lang_code == 'en':
                            date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)

                        # Standardize format - store raw text with language info
                        date_formatted = date_str
                        deadline_type = "specific_date"

                    # For quarters
                    elif "Q" in full_match or "quarter" in full_match.lower() or "trimestre" in full_match.lower() or "quartal" in full_match.lower() or "kwartaal" in full_match.lower():
                        quarter = match.group(1) if match.groups() else "1"
                        year = match.group(2) if len(match.groups()) > 1 else "2023"

                        # Format the date as YYYY-QN
                        date_formatted = f"{year}-Q{quarter}"
                        deadline_type = "quarter_end"

                    # For relative periods
                    elif any(term in full_match.lower() for term in ["within", "binnen", "dentro", "dans", "innerhalb", "entro", "no prazo"]):
                        amount = match.group(1) if match.groups() else "30"
                        unit = match.group(2) if len(match.groups()) > 1 else "days"
                        date_formatted = f"Within {amount} {unit}"
                        deadline_type = "relative_period"

                    else:
                        date_formatted = full_match
                        deadline_type = "text"

                    deadlines.append({
                        "text": full_match,
                        "date": date_formatted,
                        "type": deadline_type,
                        "span": [match.start(), match.end()],
                        "language": lang_code
                    })

                except Exception as e:
                    logger.error(f"Error parsing deadline: {full_match} - {str(e)}")
                    # Add it anyway with raw text
                    deadlines.append({
                        "text": full_match,
                        "date": full_match,
                        "type": "text",
                        "span": [match.start(), match.end()],
                        "language": lang_code
                    })

        return deadlines

    def _extract_requirements(self, text: str, lang_code: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract requirement statements from text.

        Args:
            text: Text to analyze
            lang_code: Language code
            patterns: Patterns dictionary for the language

        Returns:
            List of requirement dictionaries
        """
        requirements = []

        for pattern in patterns.get("requirements", []):
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                full_match = match.group(0)
                requirement_text = match.group(1).strip() if match.groups() else full_match

                # Skip if too short
                if len(requirement_text) < 5:
                    continue

                # Skip if just a common verb/pronoun
                if requirement_text.lower() in ["be", "do", "have", "it", "they", "them", "ser", "estar", "fazer", "ter", "être", "faire", "avoir", "sein", "haben", "tun", "essere", "fare", "avere", "zijn", "doen", "hebben"]:
                    continue

                requirements.append({
                    "text": full_match,
                    "requirement": requirement_text,
                    "span": [match.start(), match.end()],
                    "language": lang_code
                })

        return requirements

    def _extract_thresholds(self, text: str, lang_code: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract threshold values from text.

        Args:
            text: Text to analyze
            lang_code: Language code
            patterns: Patterns dictionary for the language

        Returns:
            List of threshold dictionaries
        """
        thresholds = []

        for pattern in patterns.get("thresholds", []):
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                full_match = match.group(0)
                value = match.group(1).strip() if match.groups() else full_match

                # Try to determine the threshold type
                threshold_type = "general"

                # Check for maximum in different languages
                if any(word in full_match.lower() for word in ["maximum", "max", "máximo", "maximaal", "maximal", "massimo"]):
                    threshold_type = "maximum"
                # Check for minimum in different languages
                elif any(word in full_match.lower() for word in ["minimum", "min", "mínimo", "minimaal", "minimal", "minimo", "at least", "al menos", "au moins", "mindestens", "almeno", "pelo menos", "ten minste"]):
                    threshold_type = "minimum"

                thresholds.append({
                    "text": full_match,
                    "value": value,
                    "type": threshold_type,
                    "span": [match.start(), match.end()],
                    "language": lang_code
                })

        return thresholds

    def _extract_authorities(self, text: str, lang_code: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract references to regulatory authorities.

        Args:
            text: Text to analyze
            lang_code: Language code
            patterns: Patterns dictionary for the language

        Returns:
            List of authority dictionaries
        """
        authorities = []

        for pattern in patterns.get("authorities", []):
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                full_match = match.group(0)
                authority = match.group(1).strip() if match.groups() else full_match

                # Skip if already found
                if any(auth["name"] == authority for auth in authorities):
                    continue

                # Try to determine the context
                context = "general"

                # Context detection in multiple languages
                reporting_terms = ["report", "submit", "notify", "inform", "informar", "presentar", "notificar", "déclarer", "soumettre", "informer", "melden", "einreichen", "informieren", "segnalare", "presentare", "informare", "relatar", "apresentar", "informar", "melden", "indienen", "informeren"]
                approval_terms = ["approval", "authorization", "permission", "aprobación", "autorización", "permiso", "approbation", "autorisation", "permission", "genehmigung", "zulassung", "erlaubnis", "approvazione", "autorizzazione", "permesso", "aprovação", "autorização", "permissão", "goedkeuring", "autorisatie", "toestemming"]

                if any(term in full_match.lower() for term in reporting_terms):
                    context = "reporting"
                elif any(term in full_match.lower() for term in approval_terms):
                    context = "approval"

                authorities.append({
                    "text": full_match,
                    "name": authority,
                    "context": context,
                    "span": [match.start(), match.end()],
                    "language": lang_code
                })

        return authorities

    def _extract_articles(self, text: str, lang_code: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Extract article references from text.

        Args:
            text: Text to analyze
            lang_code: Language code
            patterns: Patterns dictionary for the language

        Returns:
            List of article reference dictionaries
        """
        articles = []

        for pattern in patterns.get("articles", []):
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                full_match = match.group(0)
                reference = match.group(1).strip() if match.groups() else full_match

                # Determine reference type based on language
                ref_type = "article"

                # Article type detection in multiple languages
                if any(word in full_match.lower() for word in ["article", "artículo", "artikel", "articolo", "artigo"]):
                    ref_type = "article"
                elif any(word in full_match.lower() for word in ["section", "sección", "abschnitt", "sezione", "seção", "afdeling"]):
                    ref_type = "section"
                elif any(word in full_match.lower() for word in ["paragraph", "párrafo", "absatz", "paragrafo", "parágrafo", "paragraaf"]):
                    ref_type = "paragraph"

                articles.append({
                    "text": full_match,
                    "reference": reference,
                    "type": ref_type,
                    "span": [match.start(), match.end()],
                    "language": lang_code
                })

        return articles

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of regulatory findings.

        Args:
            results: Results dictionary from analysis

        Returns:
            Summary dictionary
        """
        # Count occurrences of various patterns
        legislation_count = len(results.get("legislation_references", []))
        deadline_count = len(results.get("deadlines", []))
        requirement_count = len(results.get("requirements", []))
        threshold_count = len(results.get("thresholds", []))
        authority_count = len(results.get("authorities", []))
        article_count = len(results.get("article_references", []))

        # Determine if text has significant regulatory content
        regulatory_score = legislation_count * 3 + deadline_count * 2 + requirement_count + threshold_count + authority_count + article_count

        has_regulatory_content = regulatory_score > 5

        # Get most significant deadline if any
        deadlines = results.get("deadlines", [])
        nearest_deadline = None
        if deadlines:
            for deadline in deadlines:
                if deadline["type"] == "specific_date":
                    if nearest_deadline is None:
                        nearest_deadline = deadline
                    elif deadline["date"] < nearest_deadline["date"]:
                        nearest_deadline = deadline

        return {
            "has_regulatory_content": has_regulatory_content,
            "regulatory_score": regulatory_score,
            "legislation_count": legislation_count,
            "requirement_count": requirement_count,
            "deadline_count": deadline_count,
            "nearest_deadline": nearest_deadline["date"] if nearest_deadline else None,
            "authority_mentions": authority_count,
            "language": results.get("language", "en")
        }

    def extract_compliance_requirements(self, text: str, lang_code: str = None) -> List[Dict[str, Any]]:
        """
        Extract compliance requirements from text.

        Args:
            text: Text to analyze
            lang_code: Language code (auto-detected if not provided)

        Returns:
            List of compliance requirement dictionaries
        """
        # Detect language if not provided and multilingual is enabled
        if self.multilingual and not lang_code:
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']
            logger.info(f"Detected language for compliance requirement extraction: {language_info['language_name']} ({lang_code})")

        # If language is not supported, fall back to English
        if not lang_code or lang_code not in self.patterns:
            lang_code = 'en'

        # Get patterns for this language
        patterns = self.patterns[lang_code]

        # Analyze text to get components
        legislation = self._extract_legislation(text, lang_code, patterns)
        req_matches = self._extract_requirements(text, lang_code, patterns)
        deadlines = self._extract_deadlines(text, lang_code, patterns)

        # Combine requirements with deadlines and legislation
        requirements = []

        for req in req_matches:
            # Find nearby legislation (within 500 characters)
            req_pos = req["span"][0]
            related_legislation = []

            for leg in legislation:
                leg_pos = leg["span"][0]
                if abs(leg_pos - req_pos) < 500:
                    related_legislation.append(leg)

            # Find nearby deadlines (within 300 characters)
            related_deadlines = []

            for deadline in deadlines:
                deadline_pos = deadline["span"][0]
                if abs(deadline_pos - req_pos) < 300:
                    related_deadlines.append(deadline)

            # Create compliance requirement
            compliance_req = {
                "requirement": req["requirement"],
                "full_text": req["text"],
                "related_legislation": related_legislation,
                "deadlines": related_deadlines,
                "span": req["span"],
                "language": lang_code
            }

            requirements.append(compliance_req)

        return requirements

    def classify_document_type(self, text: str, lang_code: str = None) -> Dict[str, Any]:
        """
        Classify the type of regulatory document based on content analysis.

        Args:
            text: Text to analyze
            lang_code: Language code (auto-detected if not provided)

        Returns:
            Document classification result
        """
        # Detect language if not provided and multilingual is enabled
        if self.multilingual and not lang_code:
            language_info = self.language_detector.detect_language(text)
            lang_code = language_info['language_code']

        # Full analysis
        analysis = self.analyze_text(text, lang_code)

        # Define document types with scoring criteria
        doc_types = {
            "legislation": {
                "score": len(analysis["legislation_references"]) * 3 + len(analysis["article_references"]) * 2,
                "threshold": 5,
                "description": "Legislative or regulatory text"
            },
            "compliance_guidance": {
                "score": len(analysis["requirements"]) * 2 + len(analysis["thresholds"]),
                "threshold": 4,
                "description": "Compliance guide or manual"
            },
            "deadline_notification": {
                "score": len(analysis["deadlines"]) * 3,
                "threshold": 3,
                "description": "Deadline notification or calendar"
            },
            "authority_communication": {
                "score": len(analysis["authorities"]) * 3,
                "threshold": 3,
                "description": "Communication from a regulatory authority"
            }
        }

        # Determine document type
        top_score = 0
        doc_type = "general"
        doc_description = "General document"

        for type_key, type_data in doc_types.items():
            if type_data["score"] > type_data["threshold"] and type_data["score"] > top_score:
                top_score = type_data["score"]
                doc_type = type_key
                doc_description = type_data["description"]

        return {
            "document_type": doc_type,
            "description": doc_description,
            "confidence": min(1.0, top_score / 15),  # Scale confidence
            "has_regulatory_content": analysis["summary"]["has_regulatory_content"],
            "language": lang_code
        }
