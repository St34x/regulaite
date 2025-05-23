"""
Search-related tools for the RegulAIte Agent Framework.

This module provides tools for searching and retrieving information.
"""
from typing import Dict, List, Optional, Any
import logging
import re

from ..tool_registry import tool

# Set up logging
logger = logging.getLogger(__name__)

@tool(
    id="query_reformulation",
    name="Query Reformulation",
    description="Reformulate a query to improve retrieval results",
    tags=["search", "retrieval", "query"],
    requires_context=False
)
async def query_reformulation(query: str, strategy: str = "expand") -> Dict[str, Any]:
    """
    Reformulate a query to improve retrieval results.
    
    Args:
        query: The original query
        strategy: The reformulation strategy (expand, specify, simplify)
        
    Returns:
        Dictionary with reformulated queries
    """
    logger.info(f"Reformulating query: {query} using strategy: {strategy}")
    
    # Simple implementations of reformulation strategies
    reformulations = []
    
    if strategy == "expand":
        # Add synonyms or related terms
        # This is a simple implementation - would be more sophisticated in practice
        reformulations = [
            query,
            f"information about {query}",
            f"details regarding {query}",
            f"explanation of {query}"
        ]
    elif strategy == "specify":
        # Make the query more specific
        reformulations = [
            query,
            f"specific information about {query}",
            f"detailed explanation of {query}",
            f"{query} detailed information"
        ]
    elif strategy == "simplify":
        # Simplify the query
        # Remove stop words, focus on key terms
        simple_query = re.sub(r'\b(the|a|an|in|on|at|to|for|with|by|about|of)\b', '', query, flags=re.IGNORECASE)
        simple_query = re.sub(r'\s+', ' ', simple_query).strip()
        
        reformulations = [
            query,
            simple_query,
            " ".join([word for word in query.split() if len(word) > 3])
        ]
    else:
        # Default strategy
        reformulations = [query]
    
    return {
        "original_query": query,
        "strategy": strategy,
        "reformulations": reformulations
    }

@tool(
    id="filter_search",
    name="Filter Search",
    description="Apply filters to a search query",
    tags=["search", "filter", "retrieval"],
    requires_context=False
)
async def filter_search(query: str, 
                       filters: Optional[Dict[str, Any]] = None, 
                       date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Apply filters to a search query.
    
    Args:
        query: The search query
        filters: Dictionary of filter criteria
        date_range: Date range for filtering
        
    Returns:
        Dictionary with filter information
    """
    logger.info(f"Applying filters to query: {query}")
    
    # Initialize filters if not provided
    if filters is None:
        filters = {}
    
    # Initialize date range if not provided
    if date_range is None:
        date_range = {}
    
    return {
        "query": query,
        "filters": filters,
        "date_range": date_range,
        "filter_query": f"{query} {' '.join([f'{k}:{v}' for k, v in filters.items()])}"
    }

@tool(
    id="extract_search_entities",
    name="Extract Search Entities",
    description="Extract entities from a search query using LLM intelligence",
    tags=["search", "entity", "extraction", "nlp"],
    requires_context=False
)
async def extract_search_entities(query: str) -> Dict[str, List[str]]:
    """
    Extract entities from a search query using LLM intelligence.
    
    Args:
        query: The search query
        
    Returns:
        Dictionary of extracted entities by type
    """
    logger.info(f"Extracting entities from query: {query}")
    
    # Try to get LLM client for intelligent entity extraction
    try:
        from ..integrations.llm_integration import get_llm_integration
        llm_client = get_llm_integration()
        
        if llm_client:
            return await _extract_entities_with_llm(query, llm_client)
        else:
            logger.warning("LLM client not available, falling back to regex extraction")
            return _extract_entities_with_regex(query)
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {str(e)}, falling back to regex")
        return _extract_entities_with_regex(query)

async def _extract_entities_with_llm(query: str, llm_client) -> Dict[str, List[str]]:
    """
    Extract entities using LLM for intelligent processing.
    
    Args:
        query: The search query
        llm_client: The LLM client to use
        
    Returns:
        Dictionary of extracted entities by type
    """
    import json
    
    prompt = f"""
Extract named entities from the following query and return them as a JSON object. Focus on identifying:

1. **dates**: Any date references (absolute or relative)
2. **organizations**: Company names, institutions, agencies, departments
3. **people**: Person names, titles, roles
4. **locations**: Places, addresses, countries, regions
5. **regulations**: Laws, regulations, directives, standards, codes
6. **financial_terms**: Financial amounts, currencies, financial concepts
7. **document_types**: Types of documents mentioned (contracts, reports, etc.)
8. **keywords**: Important domain-specific terms and concepts

Return the result as JSON in this exact format:
{{
    "dates": ["date1", "date2"],
    "organizations": ["org1", "org2"],
    "people": ["person1", "person2"],
    "locations": ["location1", "location2"],
    "regulations": ["regulation1", "regulation2"],
    "financial_terms": ["term1", "term2"],
    "document_types": ["type1", "type2"],
    "keywords": ["keyword1", "keyword2"]
}}

Query: "{query}"

JSON Response:
"""
    
    try:
        response = await llm_client.generate(prompt, temperature=0.1)
        
        # Try to parse JSON response
        try:
            entities = json.loads(response)
            
            # Ensure all expected keys are present
            expected_keys = ["dates", "organizations", "people", "locations", 
                           "regulations", "financial_terms", "document_types", "keywords"]
            
            for key in expected_keys:
                if key not in entities:
                    entities[key] = []
                    
            logger.info(f"Successfully extracted {sum(len(v) for v in entities.values())} entities with LLM")
            return entities
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM response is not valid JSON: {str(e)}")
            # Try to extract from text response
            return _parse_entities_from_text(response, query)
            
    except Exception as e:
        logger.error(f"Error in LLM entity extraction: {str(e)}")
        raise

def _parse_entities_from_text(response: str, query: str) -> Dict[str, List[str]]:
    """
    Parse entities from a non-JSON LLM response.
    
    Args:
        response: The text response from LLM
        query: Original query for fallback extraction
        
    Returns:
        Dictionary of extracted entities by type
    """
    import re
    
    entities = {
        "dates": [],
        "organizations": [],
        "people": [],
        "locations": [],
        "regulations": [],
        "financial_terms": [],
        "document_types": [],
        "keywords": []
    }
    
    # Try to extract lists from the response
    for entity_type in entities.keys():
        # Look for patterns like "dates": ["item1", "item2"]
        pattern = rf'{entity_type}["\']?\s*:\s*\[([^\]]+)\]'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            items_str = match.group(1)
            items = [item.strip(' "\'') for item in items_str.split(',')]
            entities[entity_type] = [item for item in items if item]
    
    # If no entities were found, fall back to regex on original query
    if not any(entities.values()):
        return _extract_entities_with_regex(query)
        
    return entities

def _extract_entities_with_regex(query: str) -> Dict[str, List[str]]:
    """
    Extract entities using regex patterns (fallback method).
    
    Args:
        query: The search query
        
    Returns:
        Dictionary of extracted entities by type
    """
    entities = {
        "dates": [],
        "organizations": [],
        "people": [],
        "locations": [],
        "regulations": [],
        "financial_terms": [],
        "document_types": [],
        "keywords": []
    }
    
    # Extract dates (simple pattern matching)
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, query, re.IGNORECASE)
        entities["dates"].extend(dates)
    
    # Extract organizations (simple pattern matching)
    org_patterns = [
        r"\b[A-Z][a-z]+ (Inc|Corp|Corporation|Company|Co|Ltd)\b",
        r"\b[A-Z][A-Za-z]+ (Inc|Corp|Corporation|Company|Co|Ltd)\b",
        r"\b[A-Z]{2,}\b"  # All caps words often organizations
    ]
    
    for pattern in org_patterns:
        orgs = re.findall(pattern, query)
        entities["organizations"].extend(orgs)
    
    # Extract financial terms
    financial_patterns = [
        r"[\$€£¥]\s?\d+(?:\.\d+)?",
        r"\d+(?:\.\d+)?\s?(?:dollars|euros|pounds|yen|USD|EUR|GBP|JPY)",
        r"\b(?:profit|loss|revenue|income|expense|budget|cost|price|fee|tax)\b"
    ]
    
    for pattern in financial_patterns:
        terms = re.findall(pattern, query, re.IGNORECASE)
        entities["financial_terms"].extend(terms)
    
    # Extract regulation references
    regulation_patterns = [
        r"article\s+\d+",
        r"section\s+\d+",
        r"§\s*\d+",
        r"\b(?:regulation|directive|law|act|code|standard)\s+\w+",
        r"\b(?:GDPR|SOX|HIPAA|PCI|ISO\s*\d+)\b"
    ]
    
    for pattern in regulation_patterns:
        regs = re.findall(pattern, query, re.IGNORECASE)
        entities["regulations"].extend(regs)
    
    # Extract document types
    doc_patterns = [
        r"\b(?:contract|agreement|report|document|file|pdf|spreadsheet|presentation|form|application)\b"
    ]
    
    for pattern in doc_patterns:
        docs = re.findall(pattern, query, re.IGNORECASE)
        entities["document_types"].extend(docs)
    
    # Extract keywords (simple approach - words longer than 4 chars)
    keywords = [word for word in query.split() if len(word) > 4 and word.isalpha()]
    entities["keywords"] = keywords
    
    return entities 