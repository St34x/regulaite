"""
Query Parser for the RegulAIte Agent Framework.

This module provides functionality for parsing, validating, and classifying 
user queries using Pydantic models for type safety and LLM-based intelligence.
"""
from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field, model_validator
import re
import logging
import json
from enum import Enum

from .agent import Query, IntentType, QueryContext

# Set up logging
logger = logging.getLogger(__name__)

class QueryCategory(str, Enum):
    """Categories of queries for more specific handling."""
    LEGAL = "legal"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    DOCUMENT = "document"
    GENERAL = "general"
    SYSTEM = "system"

class ParsedQuery(Query):
    """
    Extended Query model with additional parsing information.
    """
    category: QueryCategory = Field(default=QueryCategory.GENERAL)
    entities: Dict[str, Any] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    parsed_parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

class QueryParser:
    """
    Parser for extracting structured information from user queries using LLM intelligence.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the query parser with optional LLM client."""
        self.llm_client = llm_client
        
        # Initialize LLM client if not provided
        if self.llm_client is None:
            try:
                from .integrations.llm_integration import get_llm_integration
                self.llm_client = get_llm_integration()
                logger.info("Initialized LLM client for query parser")
            except Exception as e:
                logger.warning(f"Could not initialize LLM client: {str(e)}")
                self.llm_client = None
        
        # Fallback keyword patterns for when LLM is unavailable
        self.keyword_patterns = {
            QueryCategory.LEGAL: [
                r"legal", r"law", r"regulation", r"compliance", r"statute", 
                r"directive", r"ordinance", r"article", r"section"
            ],
            QueryCategory.FINANCIAL: [
                r"financial", r"finance", r"money", r"payment", r"transaction",
                r"bank", r"account", r"deposit", r"withdraw", r"transfer"
            ],
            QueryCategory.COMPLIANCE: [
                r"compliance", r"conform", r"adhere", r"standard", r"requirement",
                r"guideline", r"policy", r"procedure", r"protocol"
            ],
            QueryCategory.DOCUMENT: [
                r"document", r"file", r"pdf", r"spreadsheet", r"contract",
                r"agreement", r"upload", r"download", r"text", r"content"
            ],
            QueryCategory.SYSTEM: [
                r"system", r"login", r"account", r"password", r"settings",
                r"configure", r"setup", r"profile", r"preferences"
            ]
        }
        
        # Common named entities we want to extract (fallback patterns)
        self.entity_patterns = {
            "date": r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "url": r"https?://[^\s]+",
            "number": r"\b\d+\b",
            "money": r"[\$€£¥]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:dollars|euros|pounds|yen)",
            "percentage": r"\d+(?:\.\d+)?%",
            "article_ref": r"article\s+\d+|section\s+\d+|§\s*\d+"
        }
        
    async def parse(self, query: Union[str, Query]) -> ParsedQuery:
        """
        Parse a query into a structured form using LLM intelligence.
        
        Args:
            query: The query text or Query object to parse
            
        Returns:
            A ParsedQuery object with structured information
        """
        # Convert string to Query if needed
        if isinstance(query, str):
            query = Query(query_text=query)
            
        # Create a ParsedQuery from the Query
        parsed = ParsedQuery(
            query_text=query.query_text,
            intent=query.intent,
            context=query.context,
            parameters=query.parameters
        )
        
        # Use LLM for intelligent parsing if available
        if self.llm_client:
            try:
                await self._parse_with_llm(parsed)
            except Exception as e:
                logger.warning(f"LLM parsing failed, falling back to regex: {str(e)}")
                await self._parse_with_regex(parsed)
        else:
            # Fallback to regex-based parsing
            await self._parse_with_regex(parsed)
        
        # Log the parsing result
        logger.info(f"Parsed query: {parsed.query_text}")
        logger.info(f"Intent: {parsed.intent}, Category: {parsed.category}")
        logger.info(f"Keywords: {parsed.keywords}")
        logger.info(f"Entities: {parsed.entities}")
        
        return parsed
        
    async def _parse_with_llm(self, parsed_query: ParsedQuery):
        """
        Parse query using LLM for intelligent extraction.
        
        Args:
            parsed_query: The ParsedQuery object to populate
        """
        prompt = f"""
Analyze the following query and extract structured information. Return your response as a JSON object with the following structure:

{{
    "keywords": ["keyword1", "keyword2", "..."],
    "entities": {{
        "dates": ["extracted dates"],
        "organizations": ["company names, org names"],
        "people": ["person names"],
        "locations": ["place names, addresses"],
        "regulations": ["regulation names, law references"],
        "financial_terms": ["financial concepts, amounts"],
        "document_types": ["document types mentioned"]
    }},
    "category": "legal|financial|compliance|document|general|system",
    "intent": "question|command|clarification|information|unknown",
    "parameters": {{
        "key": "value for any key-value pairs found"
    }},
    "confidence": 0.0-1.0
}}

Focus on extracting:
1. Important keywords that describe the main concepts
2. Named entities like dates, organizations, people, locations
3. Regulatory or legal references
4. Financial terms or amounts
5. Document types or file references
6. The overall category and intent of the query

Query: "{parsed_query.query_text}"

Respond only with valid JSON:
"""
        
        try:
            response = await self.llm_client.generate(prompt, temperature=0.1)
            
            # Try to parse the JSON response
            try:
                result = json.loads(response)
                
                # Extract keywords
                parsed_query.keywords = result.get("keywords", [])
                
                # Extract entities
                parsed_query.entities = result.get("entities", {})
                
                # Set category
                category_str = result.get("category", "general")
                try:
                    parsed_query.category = QueryCategory(category_str)
                except ValueError:
                    parsed_query.category = QueryCategory.GENERAL
                
                # Set intent
                intent_str = result.get("intent", "unknown")
                try:
                    parsed_query.intent = IntentType(intent_str.upper())
                except ValueError:
                    parsed_query.intent = IntentType.UNKNOWN
                
                # Extract parameters
                parsed_query.parsed_parameters = result.get("parameters", {})
                
                # Set confidence
                parsed_query.confidence = float(result.get("confidence", 0.8))
                
                logger.info("Successfully parsed query with LLM")
                
            except json.JSONDecodeError as e:
                logger.warning(f"LLM response is not valid JSON: {str(e)}")
                # Try to extract information from the text response
                await self._extract_from_text_response(parsed_query, response)
                
        except Exception as e:
            logger.error(f"Error in LLM parsing: {str(e)}")
            raise
            
    async def _extract_from_text_response(self, parsed_query: ParsedQuery, response: str):
        """
        Extract information from a non-JSON LLM response.
        
        Args:
            parsed_query: The ParsedQuery object to populate
            response: The text response from LLM
        """
        # Try to extract keywords from the response
        keywords_match = re.search(r'keywords?["\']?\s*:\s*\[([^\]]+)\]', response, re.IGNORECASE)
        if keywords_match:
            keywords_str = keywords_match.group(1)
            keywords = [kw.strip(' "\'') for kw in keywords_str.split(',')]
            parsed_query.keywords = [kw for kw in keywords if kw]
        
        # Try to extract category
        category_match = re.search(r'category["\']?\s*:\s*["\']?(\w+)["\']?', response, re.IGNORECASE)
        if category_match:
            category_str = category_match.group(1).lower()
            try:
                parsed_query.category = QueryCategory(category_str)
            except ValueError:
                parsed_query.category = QueryCategory.GENERAL
        
        # Set a lower confidence since we couldn't parse JSON
        parsed_query.confidence = 0.6
        
    async def _parse_with_regex(self, parsed_query: ParsedQuery):
        """
        Parse query using regex patterns as fallback.
        
        Args:
            parsed_query: The ParsedQuery object to populate
        """
        # Extract entities
        entities = self._extract_entities(parsed_query.query_text)
        parsed_query.entities = entities
        
        # Extract keywords
        keywords = self._extract_keywords(parsed_query.query_text)
        parsed_query.keywords = keywords
        
        # Classify category
        category = self._classify_category(parsed_query.query_text, keywords)
        parsed_query.category = category
        
        # Parse parameters from the query
        params = self._extract_parameters(parsed_query.query_text)
        parsed_query.parsed_parameters = params
        
        # Set lower confidence for regex parsing
        parsed_query.confidence = 0.7
        
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from the query text using regex patterns.
        
        Args:
            text: The query text
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # Apply entity patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                # Some patterns return tuples, flatten to strings
                clean_matches = []
                for match in matches:
                    if isinstance(match, tuple):
                        # Take first non-empty group
                        clean_match = next((m for m in match if m), "")
                    else:
                        clean_match = match
                    clean_matches.append(clean_match)
                
                entities[entity_type] = clean_matches
                
        return entities
        
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from the query text using regex patterns.
        
        Args:
            text: The query text
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Apply all keyword patterns
        for category, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                matches = re.findall(rf"\b{pattern}\b", text, re.IGNORECASE)
                keywords.extend(matches)
                
        # Remove duplicates and convert to lowercase
        keywords = list(set([kw.lower() for kw in keywords]))
        
        return keywords
        
    def _classify_category(self, text: str, keywords: List[str]) -> QueryCategory:
        """
        Classify the query into a category using keyword matching.
        
        Args:
            text: The query text
            keywords: Extracted keywords
            
        Returns:
            The query category
        """
        # Count keywords by category
        category_counts = {category: 0 for category in QueryCategory}
        
        for kw in keywords:
            for category, patterns in self.keyword_patterns.items():
                if any(re.search(rf"\b{pattern}\b", kw, re.IGNORECASE) for pattern in patterns):
                    category_counts[category] += 1
        
        # Find category with highest keyword count
        max_count = 0
        max_category = QueryCategory.GENERAL
        
        for category, count in category_counts.items():
            if count > max_count:
                max_count = count
                max_category = category
                
        return max_category
        
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters from the query text using regex patterns.
        
        Args:
            text: The query text
            
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        
        # Look for key-value patterns like "key: value" or "key=value"
        kv_patterns = [
            r'(\w+):\s*"([^"]+)"',  # key: "value"
            r"(\w+):\s*'([^']+)'",  # key: 'value'
            r'(\w+):\s*([^\s,;]+)',  # key: value
            r'(\w+)=\s*"([^"]+)"',  # key="value"
            r"(\w+)=\s*'([^']+)'",  # key='value'
            r'(\w+)=\s*([^\s,;]+)'   # key=value
        ]
        
        for pattern in kv_patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                # Try to convert value to appropriate type
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                elif value.isdigit():
                    params[key] = int(value)
                elif re.match(r'^-?\d+(\.\d+)?$', value):
                    params[key] = float(value)
                else:
                    params[key] = value
                    
        return params 