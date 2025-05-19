# plugins/regul_aite/backend/unstructured_parser/document_parser.py
import os
import requests
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, BinaryIO, Callable, Literal
from datetime import datetime as py_datetime  # Rename to avoid conflict with Neo4j's datetime()
import datetime  # Import the full module too
import tempfile
from neo4j import GraphDatabase
from data_enrichment.language_detector import LanguageDetector
from data_enrichment.enrichment_pipeline import EnrichmentPipeline
from data_enrichment.metadata_parser import MetadataParser
import time
import math
import re

# Import LangChain TokenTextSplitter with fallback
try:
    from langchain_text_splitters import TokenTextSplitter
    HAS_TOKEN_SPLITTER = True
except ImportError:
    HAS_TOKEN_SPLITTER = False
    logging.warning("langchain_text_splitters package not found. Token-based chunking will fall back to fixed size chunking.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define chunking strategies
ChunkingStrategy = Literal["fixed", "recursive", "semantic", "hierarchical", "token"]

class DocumentParser:
    """
    Parser for documents using Unstructured API.
    Extracts text and metadata from documents and stores in Neo4j.
    """

    def __init__(
          self,
          neo4j_uri: str,
          neo4j_user: str,
          neo4j_password: str,
          unstructured_api_url: Optional[str] = None,
          unstructured_api_key: Optional[str] = None,
          chunk_size: int = 1000,
          chunk_overlap: int = 200,
          use_enrichment: bool = True,
          chunking_strategy: ChunkingStrategy = "fixed",
          extract_tables: bool = True,
          extract_metadata: bool = True,
          extract_images: bool = False,
          is_cloud: bool = False
      ):
      """
      Initialize the document parser.

      Args:
          neo4j_uri: URI for Neo4j database
          neo4j_user: Username for Neo4j
          neo4j_password: Password for Neo4j
          unstructured_api_url: URL for Unstructured API (defaults to env var)
          unstructured_api_key: API key for Unstructured API (defaults to env var)
          chunk_size: Maximum size of text chunks in characters
          chunk_overlap: Number of characters to overlap between chunks
          use_enrichment: Whether to enrich documents by default
          chunking_strategy: Strategy for chunking text ("fixed", "recursive", "semantic", "hierarchical", "token")
          extract_tables: Whether to extract tables from documents
          extract_metadata: Whether to extract detailed metadata
          extract_images: Whether to extract and process images
          is_cloud: Whether to use the cloud version of Unstructured API
      """
      self.neo4j_uri = neo4j_uri
      self.neo4j_user = neo4j_user
      self.neo4j_password = neo4j_password
      self.is_cloud = is_cloud

      # Get Unstructured API credentials from environment if not provided
      if is_cloud:
          # Use cloud-specific environment variables
          self.unstructured_api_url = unstructured_api_url or os.getenv(
              "UNSTRUCTURED_CLOUD_API_URL", "https://api.unstructured.io/general/v0/general"
          )
          self.unstructured_api_key = unstructured_api_key or os.getenv(
              "UNSTRUCTURED_CLOUD_API_KEY", ""
          )
          if not self.unstructured_api_key:
              logger.warning("No API key provided for Unstructured Cloud API")
      else:
          # Use local-specific environment variables
          self.unstructured_api_url = unstructured_api_url or os.getenv(
              "UNSTRUCTURED_API_URL", "http://unstructured:8000/general/v0/general"
          )
          self.unstructured_api_key = unstructured_api_key or os.getenv(
              "UNSTRUCTURED_API_KEY", ""
          )

      # Chunking parameters
      self.chunk_size = chunk_size
      self.chunk_overlap = chunk_overlap
      self.chunking_strategy = chunking_strategy

      # Extraction options
      self.extract_tables = extract_tables
      self.extract_metadata = extract_metadata
      self.extract_images = extract_images

      # Initialize enrichment variables
      self.use_enrichment = use_enrichment

      # Initialize Enrichment Pipeline if enabled
      if use_enrichment:
          try:
              self.enrichment_pipeline = EnrichmentPipeline(
                  neo4j_uri=neo4j_uri,
                  neo4j_user=neo4j_user,
                  neo4j_password=neo4j_password,
                  multilingual=True
              )
              logger.info("Enrichment pipeline initialized successfully in DocumentParser.")
          except Exception as e:
              logger.error(f"CRITICAL: DocumentParser failed to initialize EnrichmentPipeline. Enrichment will be DISABLED. Error: {str(e)}", exc_info=True)
              self.enrichment_pipeline = None
              self.use_enrichment = False # Ensure this is set if pipeline fails
      else:
          logger.info("DocumentParser: Enrichment is explicitly disabled via use_enrichment=False.")
          self.enrichment_pipeline = None
          self.use_enrichment = False # Explicitly ensure this

      # Initialize Neo4j connection
      self.driver = None
      self._connect_to_neo4j()

      # Initialize metadata parser
      self.metadata_parser = MetadataParser()

      # Log initialization
      api_type = "cloud" if is_cloud else "local"
      logger.info(f"Initialized DocumentParser with {api_type} Unstructured API at {self.unstructured_api_url}")

    def _connect_to_neo4j(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=3600
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected to Neo4j' AS message")
                for record in result:
                    logger.info(record["message"])
            logger.info(f"Document parser connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def _call_unstructured_api(self, file_content: bytes, file_name: str) -> List[Dict[str, Any]]:
      """
      Call the Unstructured API to extract text from a file.

      Args:
          file_content: Binary content of the file
          file_name: Name of the file

      Returns:
          List of elements extracted from the document
      """
      logger.info(f"Calling Unstructured API for document: {file_name}")

      # Create a temporary file to send to the API
      with tempfile.NamedTemporaryFile(delete=False) as temp_file:
          temp_file.write(file_content)
          temp_file_path = temp_file.name

      try:
          headers = {
              "Accept": "application/json",
              "unstructured-api-key": self.unstructured_api_key
          }

          with open(temp_file_path, "rb") as f:
              files = {"files": (file_name, f)}

              # Enhanced parameters for better extraction
              data = {
                  "strategy": "auto",
                  "ocr_enabled": "true",
                  "languages": "auto",  # Use languages instead of ocr_languages
                  "include_page_breaks": "true",  # Preserve page break information
                  "hierarchical_pdf": "true" if self.chunking_strategy == "hierarchical" else "false",
                  "extract_images": "true" if self.extract_images else "false",
                  "extract_tables": "true" if self.extract_tables else "false",
                  "include_metadata": "true" if self.extract_metadata else "false"
              }

              logger.info(f"Calling Unstructured API with parameters: {data}")

              try:
                  response = requests.post(
                      self.unstructured_api_url,
                      headers=headers,
                      files=files,
                      data=data,
                      timeout=300  # 5-minute timeout for large documents
                  )

                  if response.status_code != 200:
                      logger.error(f"Unstructured API error: {response.status_code} - {response.text}")
                      # Don't raise here, use fallback
                      return self._create_fallback_elements(file_name, file_content)

                  # Parse and return the response
                  try:
                      elements = response.json()
                      if not elements or len(elements) == 0:
                          logger.warning(f"Unstructured API returned empty elements list for {file_name}")
                          return self._create_fallback_elements(file_name, file_content)
                      
                      logger.info(f"Extracted {len(elements)} elements from document")

                      # Log element types to help with debugging
                      element_types = {}
                      text_count = 0
                      image_count = 0
                      for element in elements:
                          element_type = element.get("type", "unknown")
                          has_text = bool(element.get("text", "").strip())
                          if has_text:
                              text_count += 1
                              
                          if element_type == 'Image':
                              image_count += 1

                          if element_type in element_types:
                              element_types[element_type] += 1
                          else:
                              element_types[element_type] = 1
                              
                      logger.info(f"Element types: {element_types}, {text_count} with text, {image_count} images")
                      
                      # If no text elements, use fallback
                      if text_count == 0:
                          logger.warning(f"No text elements found in API response for {file_name}, using fallback")
                          return self._create_fallback_elements(file_name, file_content)
                      
                      # Return the parsed elements
                      return elements
                  except json.JSONDecodeError as je:
                      logger.error(f"Failed to parse Unstructured API response as JSON: {str(je)}")
                      logger.error(f"Response content: {response.text[:500]}...")
                      return self._create_fallback_elements(file_name, file_content)
              except requests.exceptions.RequestException as re:
                  logger.error(f"Request to Unstructured API failed: {str(re)}")
                  return self._create_fallback_elements(file_name, file_content)

      except Exception as e:
          logger.error(f"Error calling Unstructured API: {str(e)}", exc_info=True)
          return self._create_fallback_elements(file_name, file_content)
      finally:
          try:
              os.unlink(temp_file_path)
          except Exception as e:
              logger.warning(f"Failed to delete temp file: {str(e)}")

    def _create_fallback_elements(self, file_name: str, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Create fallback document elements when the Unstructured API fails.
        
        Args:
            file_name: Name of the file
            file_content: Binary content of the file
            
        Returns:
            List with at least one element containing basic file information
        """
        logger.info(f"Creating fallback document elements for {file_name}")
        
        # Try to extract some text content depending on file type
        text_content = ""
        file_ext = os.path.splitext(file_name)[1].lower()
        
        try:
            # For text-based files, try to decode the content
            if file_ext in ['.txt', '.csv', '.md', '.json', '.xml', '.html']:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'windows-1252']:
                    try:
                        text_content = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                        
                if not text_content:
                    text_content = f"Failed to extract text from {file_name}"
            else:
                text_content = f"Document: {file_name} (failed to extract content)"
        except Exception as e:
            logger.warning(f"Error creating fallback text: {str(e)}")
            text_content = f"Document: {file_name} (extraction error)"
            
        # Create at least one fallback element
        fallback_element = {
            "type": "Text",
            "text": text_content or f"Document: {file_name} (no extractable text)",
            "metadata": {
                "filename": file_name,
                "filetype": file_ext.lstrip('.') if file_ext else "unknown",
                "is_fallback": True,
                "extraction_failed": True
            }
        }
        
        return [fallback_element]

    def _process_table_elements(self, elements: List[Dict[str, Any]]) -> None:
        """
        Process table elements to make them more usable in Neo4j and RAG.

        Args:
            elements: List of elements to process
        """
        if not self.extract_tables:
            return

        for i, element in enumerate(elements):
            if element.get("type") == "Table":
                # Convert table to markdown for better text representation
                if "metadata" in element and "text_as_html" in element.get("metadata", {}):
                    try:
                        html_table = element["metadata"]["text_as_html"]
                        # Add table representation as markdown for better text retrieval
                        element["text"] = f"{element.get('text', '')}\nTable content:\n{html_table}"
                    except Exception as e:
                        logger.warning(f"Error processing table element: {e}")

    def _enhance_metadata(self, elements: List[Dict[str, Any]], file_name: str) -> None:
        """
        Enhance metadata for all elements to improve Neo4j relationships.

        Args:
            elements: List of elements to enhance
            file_name: Name of the file being processed
        """
        if not self.extract_metadata:
            return

        # Extract file extension
        _, file_ext = os.path.splitext(file_name.lower())
        file_ext = file_ext.lstrip('.')

        # Get document level metadata
        doc_metadata = {}
        for element in elements:
            if element.get("metadata") and element.get("metadata", {}).get("filename"):
                # Create a copy of metadata to prevent circular references
                doc_metadata = self._safe_copy_metadata(element.get("metadata", {}))
                break

        # Add section hierarchy information
        current_section = {}
        section_stack = []

        for i, element in enumerate(elements):
            # Ensure metadata exists
            if "metadata" not in element:
                element["metadata"] = {}

            # Add file metadata (ensure it's a safe copy to prevent circular references)
            element["metadata"]["file_extension"] = file_ext
            element["metadata"]["filetype"] = file_ext
            element["metadata"]["file_type"] = file_ext  # Add both property names for compatibility
            
            # Store document metadata as a safe copy
            element["metadata"]["document_metadata"] = self._safe_copy_metadata(doc_metadata)

            # Track section hierarchy for better context
            if element.get("type") in ["Title", "NarrativeText"] and element.get("metadata", {}).get("category") == "Header":
                header_text = element.get("text", "").strip()
                header_level = element.get("metadata", {}).get("header_level", 1)

                # Pop higher level sections
                while section_stack and section_stack[-1]["level"] >= header_level:
                    section_stack.pop()

                current_section = {
                    "text": header_text,
                    "level": header_level
                }
                section_stack.append(current_section)

            # Add section hierarchy to element metadata as a clean copy
            if section_stack:
                element["metadata"]["section_hierarchy"] = [
                    {"level": s["level"], "title": s["text"]}
                    for s in section_stack
                ]

            # Add sequential position for preserving document order
            element["metadata"]["sequence_num"] = i

    def _safe_copy_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a safe copy of metadata to prevent circular references.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Clean copy of metadata with no circular references
        """
        try:
            # First try a quick deep copy - handles most cases
            return json.loads(json.dumps(metadata, default=str))
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            # If we hit errors with JSON serialization, try a manual clean copy
            logger.warning(f"JSON serialization error in metadata: {str(e)}")
            return self._manual_clean_copy(metadata)

    def _manual_clean_copy(self, obj: Any, depth: int = 0, max_depth: int = 5) -> Any:
        """
        Manually create a clean copy of an object, handling circular references.

        Args:
            obj: Object to copy
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            Clean copy of the object
        """
        # Guard against recursion depth
        if depth > max_depth:
            return "[max depth exceeded]"

        # Handle simple types directly
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Handle lists
        if isinstance(obj, list):
            return [self._manual_clean_copy(item, depth + 1, max_depth) for item in obj]

        # Handle dictionaries
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Skip complex nested structures that might cause issues
                if isinstance(value, (dict, list)) and depth > 2:
                    result[key] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                else:
                    result[key] = self._manual_clean_copy(value, depth + 1, max_depth)
            return result

        # Other types - convert to string
        return str(obj)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on the configured chunking strategy.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Choose chunking strategy
        if self.chunking_strategy == "fixed":
            return self._fixed_size_chunking(text)
        elif self.chunking_strategy == "recursive":
            return self._recursive_chunking(text)
        elif self.chunking_strategy == "semantic":
            return self._semantic_chunking(text)
        elif self.chunking_strategy == "hierarchical":
            return self._hierarchical_chunking(text)
        elif self.chunking_strategy == "token":
            # Check if token-based chunking is available and enabled
            if HAS_TOKEN_SPLITTER and os.environ.get("ENABLE_TOKEN_CHUNKING", "true").lower() != "false":
                return self._token_chunking(text)
            else:
                logger.warning(f"Token-based chunking requested but not available or disabled, using fixed size chunking")
                return self._fixed_size_chunking(text)
        else:
            # Default to fixed size chunking
            logger.warning(f"Unknown chunking strategy '{self.chunking_strategy}', using fixed size chunking")
            return self._fixed_size_chunking(text)

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks of fixed size with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                chunk = text[start:text_len]
            else:
                # Try to find a sentence boundary
                sentence_end = text.rfind(". ", start, end) + 1

                if sentence_end > start:
                    end = sentence_end
                else:
                    # If no sentence boundary, try a word boundary
                    word_end = text.rfind(" ", start, end) + 1
                    if word_end > start:
                        end = word_end

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return chunks

    def _recursive_chunking(self, text: str) -> List[str]:
        """
        Recursively split text based on structural elements like paragraphs and sections.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # First split by double newlines (paragraphs)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # If current chunk is too large, first add it to results
                if current_chunk:
                    chunks.append(current_chunk)

                # If paragraph itself exceeds chunk size, use fixed size chunking
                if len(paragraph) > self.chunk_size:
                    paragraph_chunks = self._fixed_size_chunking(paragraph)
                    chunks.extend(paragraph_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph

        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Split text based on semantic boundaries using section detection.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        # This is a simplified version - ideally would use ML to detect semantic boundaries
        # For now, use a heuristic approach to identify common section markers
        import re

        # Define section header patterns
        section_patterns = [
            r'\n#+\s+[A-Z]',  # Markdown headers
            r'\n[A-Z][A-Za-z\s]{2,40}\n-{3,}',  # Underlined headers
            r'\n\d+\.\d*\s+[A-Z]',  # Numbered sections
            r'\n[A-Z][A-Z\s]{4,40}[:\n]',  # ALL CAPS headers
            r'\nArticle \d+',  # Legal document sections
            r'\nSection \d+',  # Legal document sections
        ]

        pattern = '|'.join(section_patterns)
        sections = re.split(pattern, text)

        # If we didn't find any sections, fall back to fixed chunking
        if len(sections) <= 1:
            return self._fixed_size_chunking(text)

        # Process each section
        chunks = []
        for section in sections:
            if not section.strip():
                continue

            if len(section) <= self.chunk_size:
                chunks.append(section.strip())
            else:
                # If section is too large, recursively chunk it
                section_chunks = self._recursive_chunking(section)
                chunks.extend(section_chunks)

        return chunks

    def _hierarchical_chunking(self, text: str) -> List[str]:
        """
        Create hierarchical chunks based on document structure, maintaining parent-child relationships.
        This is particularly useful for structured documents.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks with hierarchy information in metadata
        """
        # For simplicity, this implementation is similar to semantic chunking
        # In a production environment, this would track parent-child relationships
        # and store them in Neo4j for hierarchical retrieval
        return self._semantic_chunking(text)

    def _token_chunking(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count rather than character count.
        This uses LangChain's TokenTextSplitter which is more aware of token boundaries
        and is better aligned with how LLMs process text.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks split by token count
        """
        if not text or len(text.strip()) == 0:
            return []

        # If TokenTextSplitter is not available, fall back to fixed size chunking
        if not HAS_TOKEN_SPLITTER:
            logger.warning("Token-based chunking not available - falling back to fixed size chunking")
            return self._fixed_size_chunking(text)

        try:
            # Convert character sizes to approximate token counts
            # Rule of thumb: ~4 chars per token for English text
            chars_per_token = 4
            token_size = max(1, self.chunk_size // chars_per_token)
            token_overlap = max(1, self.chunk_overlap // chars_per_token)
            
            # Create TokenTextSplitter with appropriate settings
            splitter = TokenTextSplitter(
                chunk_size=token_size,
                chunk_overlap=token_overlap
            )
            
            # Split the text
            chunks = splitter.split_text(text)
            logger.info(f"Token-based chunking created {len(chunks)} chunks from text of length {len(text)}")
            
            return chunks
        except Exception as e:
            logger.error(f"Error in token chunking: {str(e)}")
            # Fall back to fixed size chunking if token chunking fails
            logger.warning("Falling back to fixed size chunking")
            return self._fixed_size_chunking(text)

    def _hierarchical_chunking_from_elements(self, elements: List[Dict[str, Any]], doc_id: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create hierarchical chunks from document elements, maintaining parent-child relationships.
        
        Args:
            elements: Document elements extracted from parser
            doc_id: Document ID for creating chunk IDs
            
        Returns:
            Tuple of (chunks, sections) where chunks is a list of chunk dictionaries 
            and sections is a list of section dictionaries
        """
        # This is a simplified implementation
        chunks = []
        sections = []
        
        # Track section hierarchy
        current_section = {"name": "Document", "level": 0, "parent": None}
        sections.append(current_section)
        
        # First pass: identify sections
        for idx, element in enumerate(elements):
            element_type = element.get("type", "unknown")
            element_text = element.get("text", "").strip()
            
            # Skip empty elements
            if not element_text:
                continue
                
            # Create a basic chunk
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "text": element_text,
                "index": idx,
                "element_type": element_type,
                "section": current_section["name"],
                "metadata": element.get("metadata", {})
            }
            
            # Add page number if available in metadata
            if "metadata" in element and "page_number" in element["metadata"]:
                chunk["page_num"] = element["metadata"]["page_number"]
            else:
                chunk["page_num"] = 0  # Default page number
                
            # Add order_index (same as index) for consistent sorting
            chunk["order_index"] = idx
            
            # Check if this is a heading/title that should start a new section
            if element_type in ["Title", "Header", "Heading", "h1", "h2", "h3"]:
                # Determine heading level
                level = 1
                if "metadata" in element and "header_level" in element["metadata"]:
                    level = element["metadata"]["header_level"]
                elif element_type == "h1":
                    level = 1
                elif element_type == "h2":
                    level = 2
                elif element_type == "h3":
                    level = 3
                
                # Create a new section
                section = {
                    "name": element_text[:100],  # Truncate long titles
                    "level": level,
                    "parent": current_section["name"] if level > current_section["level"] else None
                }
                
                sections.append(section)
                current_section = section
                
                # Update the chunk with section info
                chunk["section"] = section["name"]
                
            chunks.append(chunk)
            
        return chunks, sections
    
    def _fixed_chunking_from_elements(self, elements: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """
        Create fixed-size chunks from document elements.
        
        Args:
            elements: Document elements extracted from parser
            doc_id: Document ID for creating chunk IDs
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunk_index = 0
        current_chunk_text = ""
        current_section = "Document"
        current_element_types = []
        current_metadata = {}
        current_page_num = 0  # Track current page number
        
        # Process each element
        for element in elements:
            element_type = element.get("type", "unknown")
            element_text = element.get("text", "").strip()
            
            # Skip empty elements
            if not element_text:
                continue
                
            # Check if this element has page information
            if "metadata" in element and "page_number" in element["metadata"]:
                current_page_num = element["metadata"]["page_number"]
                
            # Check if this is a title/heading that might indicate a section
            if element_type in ["Title", "Header", "Heading", "h1", "h2", "h3"]:
                # If we have accumulated text, create a chunk
                if current_chunk_text:
                    chunk = {
                        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                        "text": current_chunk_text,
                        "index": chunk_index,
                        "element_type": ", ".join(current_element_types),
                        "section": current_section,
                        "metadata": current_metadata.copy() if current_metadata else {},
                        "page_num": current_page_num,  # Add page number
                        "order_index": chunk_index  # Add order index
                    }
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Reset for next chunk
                    current_chunk_text = ""
                    current_element_types = []
                    current_metadata = {}
                
                # Update section name based on heading
                current_section = element_text[:100]  # Truncate long titles
            
            # Check if adding this element would exceed chunk size
            if len(current_chunk_text) + len(element_text) + 2 > self.chunk_size:
                # If we have accumulated text, create a chunk
                if current_chunk_text:
                    chunk = {
                        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                        "text": current_chunk_text,
                        "index": chunk_index,
                        "element_type": ", ".join(current_element_types),
                        "section": current_section,
                        "metadata": current_metadata.copy() if current_metadata else {},
                        "page_num": current_page_num,  # Add page number
                        "order_index": chunk_index  # Add order index
                    }
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Reset for next chunk
                    current_chunk_text = ""
                    current_element_types = []
                    current_metadata = {}
            
            # Add element text to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n" + element_text
            else:
                current_chunk_text = element_text
                
            # Track element type
            if element_type not in current_element_types:
                current_element_types.append(element_type)
                
            # Merge metadata
            if "metadata" in element and isinstance(element["metadata"], dict):
                for key, value in element["metadata"].items():
                    current_metadata[key] = value
        
        # Add final chunk if not empty
        if current_chunk_text:
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                "text": current_chunk_text,
                "index": chunk_index,
                "element_type": ", ".join(current_element_types),
                "section": current_section,
                "metadata": current_metadata.copy() if current_metadata else {},
                "page_num": current_page_num,  # Add page number
                "order_index": chunk_index  # Add order index
            }
            chunks.append(chunk)
        
        return chunks

    def _save_chunks_to_neo4j(self, chunks: List[Dict[str, Any]], doc_id: str) -> None:
        """
        Save chunks to Neo4j database.
        
        Args:
            chunks: List of chunk dictionaries
            doc_id: Document ID
        """
        logger.info(f"Saving {len(chunks)} chunks to Neo4j for document {doc_id}")
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        if not chunks:
            logger.warning(f"No chunks to save for document {doc_id}")
            return
        
        # Process in smaller batches to avoid overwhelming Neo4j
        batch_size = 50
        chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(f"Processing {len(chunk_batches)} batches of chunks")
        
        saved_count = 0
        
        for batch_index, batch in enumerate(chunk_batches):
            logger.info(f"Processing batch {batch_index+1}/{len(chunk_batches)} with {len(batch)} chunks")
            
            for chunk in batch:
                # Prepare metadata for storage
                chunk_props = chunk.copy()
                
                # If metadata is a dictionary, convert to JSON string
                if "metadata" in chunk_props and isinstance(chunk_props["metadata"], dict):
                    try:
                        chunk_props["metadata"] = json.dumps(chunk_props["metadata"])
                    except Exception as e:
                        logger.warning(f"Failed to JSON encode metadata, using str instead: {str(e)}")
                        chunk_props["metadata"] = str(chunk_props["metadata"])
                
                # Try to save with retries
                for attempt in range(max_retries):
                    try:
                        with self.driver.session() as session:
                            # Create chunk node and link to document
                            session.run(
                                """
                                MATCH (d:Document {doc_id: $doc_id})
                                MERGE (c:Chunk {chunk_id: $chunk_id})
                                SET c += $properties
                                MERGE (d)-[:CONTAINS]->(c)
                                """,
                                doc_id=doc_id,
                                chunk_id=chunk["chunk_id"],
                                properties=chunk_props
                            )
                        saved_count += 1
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Error saving chunk (attempt {attempt+1}/{max_retries}): {str(e)}")
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        else:
                            logger.error(f"Failed to save chunk after {max_retries} attempts: {str(e)}")
            
            # Sleep briefly between batches to avoid overwhelming Neo4j
            if batch_index < len(chunk_batches) - 1:
                time.sleep(0.5)
                
        logger.info(f"Successfully saved {saved_count}/{len(chunks)} chunks for document {doc_id}")
    
    def _save_sections_to_neo4j(self, sections: List[Dict[str, Any]], doc_id: str) -> None:
        """
        Save sections to Neo4j database with hierarchical relationships.
        
        Args:
            sections: List of section dictionaries
            doc_id: Document ID
        """
        logger.info(f"Saving {len(sections)} sections to Neo4j for document {doc_id}")
        
        with self.driver.session() as session:
            # First pass: Create all section nodes
            for idx, section in enumerate(sections):
                section_id = f"{doc_id}_section_{idx}"
                
                # Create section properties
                section_props = {
                    "section_id": section_id,
                    "name": section.get("name", "Unknown"),
                    "level": section.get("level", 0),
                    "index": idx
                }
                
                # Create section node and link to document
                session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (s:Section {section_id: $section_id})
                    SET s += $properties
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    doc_id=doc_id,
                    section_id=section_id,
                    properties=section_props
                )
            
            # Second pass: Create parent-child relationships
            for idx, section in enumerate(sections):
                if section.get("parent"):
                    # Find parent section node
                    parent_name = section["parent"]
                    
                    # Create relationship between parent and child sections
                    session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        MATCH (d)-[:HAS_SECTION]->(parent:Section)
                        WHERE parent.name = $parent_name
                        MATCH (d)-[:HAS_SECTION]->(child:Section {section_id: $section_id})
                        MERGE (parent)-[:CONTAINS]->(child)
                        """,
                        doc_id=doc_id,
                        parent_name=parent_name,
                        section_id=f"{doc_id}_section_{idx}"
                    )
                    
        logger.info(f"Successfully saved {len(sections)} sections for document {doc_id}")
        
    def _enrich_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enrich a document with NLP processing to extract entities, concepts, etc.

        Args:
            doc_id: Document ID
            chunks: Document chunks

        Returns:
            A dictionary containing lists of extracted entities and concepts.
            e.g., {"entities": [], "concepts": []}
        """
        if not self.enrichment_pipeline:
            logger.warning("Enrichment pipeline not initialized, skipping enrichment")
            return {"entities": [], "concepts": []}
            
        try:
            # Get all text from chunks for processing
            all_text = "\n\n".join([chunk["text"] for chunk in chunks if "text" in chunk])
            
            # Use the enrichment pipeline to process the document
            result = self.enrichment_pipeline.enrich_text(all_text, {"doc_id": doc_id})
            
            entities = []
            if result and "entities" in result:
                entities = result["entities"]
                logger.info(f"Extracted {len(entities)} entities from document {doc_id}")
            else:
                logger.warning(f"Enrichment produced no entities for document {doc_id}")

            concepts = []
            if result and "concepts" in result:
                concepts = result["concepts"]
                logger.info(f"Extracted {len(concepts)} concepts from document {doc_id}")
            else:
                logger.warning(f"Enrichment produced no concepts for document {doc_id}")
                
            return {"entities": entities, "concepts": concepts}
            
        except Exception as e:
            logger.error(f"Error enriching document {doc_id}: {str(e)}", exc_info=True)
            return {"entities": [], "concepts": []}

    def _store_document_in_neo4j(
          self,
          doc_id: str,
          file_name: str,
          elements: List[Dict[str, Any]],
          metadata: Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Store the parsed document and its elements in Neo4j.

        Args:
            doc_id: Document ID
            file_name: Original file name
            elements: Elements extracted from the document
            metadata: Additional document metadata

        Returns:
            Dictionary with document ID and statistics
        """
        logger.info(f"Storing document {doc_id} in Neo4j with {len(elements)} elements")

        # Extract file extension
        file_ext = ""
        if "." in file_name:
            file_ext = file_name.split(".")[-1].lower()

        # Ensure file size is a valid number
        file_size = metadata.get("size", 0)
        if not isinstance(file_size, (int, float)) or math.isnan(file_size):
            file_size = 0
            
        # Ensure title is valid
        title = metadata.get("title", file_name)
        if not title or not isinstance(title, str):
            title = file_name
            
        # Ensure we have a valid creation timestamp
        try:
            created_at = py_datetime.now().isoformat()
            if metadata.get("created_at") and isinstance(metadata.get("created_at"), str):
                # Try to parse the existing timestamp
                py_datetime.fromisoformat(metadata.get("created_at").replace('Z', '+00:00'))
                created_at = metadata.get("created_at")
        except (ValueError, TypeError):
            created_at = py_datetime.now().isoformat()
            
        # Ensure language is valid
        language = metadata.get("language", "en")
        if not language or not isinstance(language, str):
            language = "en"
            
        # Ensure category is valid
        category = metadata.get("category", "Uncategorized")
        if not category or not isinstance(category, str):
            category = "Uncategorized"
            
        # Ensure author is valid
        author = metadata.get("author", "")
        if not author or not isinstance(author, str):
            author = ""

        # Prepare document metadata with all required fields and validated values
        doc_metadata = {
            "doc_id": doc_id,
            "name": file_name,
            "title": title,
            "created_at": created_at,
            "uploaded_at": py_datetime.now().isoformat(),
            "file_type": file_ext or "Unknown",
            "size": file_size,
            "size_kb": round(file_size / 1024, 2) if file_size > 0 else 0,
            "language": language,
            "category": category,
            "author": author,
            "is_indexed": False
        }

        # Add additional metadata (with circular reference protection)
        safe_metadata = self._safe_copy_metadata(metadata)
        # Only add metadata fields that are not already in doc_metadata
        for key, value in safe_metadata.items():
            if key not in doc_metadata and value is not None:
                # Ensure we don't add None values or empty strings
                if isinstance(value, str) and not value.strip():
                    continue
                doc_metadata[key] = value

        # Store document node and chunks
        with self.driver.session() as session:
            # Create document node first with minimal properties
            # Extract file extension
            file_ext = ""
            if "." in file_name:
                file_ext = file_name.split(".")[-1].lower()
                
            # Calculate file size in KB for display
            size_kb = round(len(file_content) / 1024, 2)
            
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.name = $name,
                    d.title = $title,
                    d.created_at = datetime(),
                    d.uploaded_at = datetime(),
                    d.file_name = $name,
                    d.file_type = $file_type,
                    d.size = $size,
                    d.size_kb = $size_kb,
                    d.author = $author,
                    d.category = $category,
                    d.language = $language,
                    d.is_indexed = false,
                    d.processing_status = 'processing'
                """,
                doc_id=doc_id,
                name=file_name,
                title=doc_metadata.get("title", os.path.splitext(os.path.basename(file_name))[0]),
                file_type=file_ext or "Unknown",
                size=len(file_content),
                size_kb=size_kb,
                author=doc_metadata.get("author", "N/A"),
                category=doc_metadata.get("category", "Uncategorized"),
                language=doc_metadata.get("language", "en")
            )

            # Log the elements structure to debug
            logger.info(f"Elements structure sample: {str(elements[:2])[:500]}")

            # Count elements with text
            text_elements = [e for e in elements if e.get("text", "").strip()]
            logger.info(f"Elements with text: {len(text_elements)} out of {len(elements)}")

            # Process elements and create chunk nodes
            chunk_index = 0
            sections_created = set()  # Track created sections

            # Create at least one chunk even if no elements have text
            if not text_elements:
                logger.warning(f"No text elements found for document {doc_id}, creating placeholder chunk")

                chunk_id = f"{doc_id}_chunk_0"
                section_name = "Document"

                chunk_props = {
                    "chunk_id": chunk_id,
                    "text": f"Document: {file_name} (No extractable text found)",
                    "index": 0,
                    "section": section_name,
                    "section_index": 0,
                    "element_type": "placeholder",
                }

                # Create chunk node and link to document
                session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c += $properties
                    MERGE (d)-[:CONTAINS]->(c)
                    """,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    properties=chunk_props
                )

                chunk_index = 1
                sections_created.add(section_name)
            else:
                # Process each element with text
                for element in text_elements:
                    element_type = element.get("type", "unknown")
                    element_text = element.get("text", "").strip()

                    # Determine section based on element type or metadata
                    section_name = "Default"
                    if element_type in ["Title", "Header", "Heading", "h1", "h2", "h3"]:
                        section_name = element_text[:100]  # Truncate long titles
                    elif "metadata" in element and isinstance(element["metadata"], dict):
                        if "section" in element["metadata"]:
                            section_name = element["metadata"]["section"]
                        elif "section_name" in element["metadata"]:
                            section_name = element["metadata"]["section_name"]

                    # Add to tracked sections
                    sections_created.add(section_name)

                    # For longer text elements, split into chunks
                    if len(element_text) > self.chunk_size:
                        chunks = self._chunk_text(element_text)
                    else:
                        chunks = [element_text]

                    # Create chunks from this element
                    for i, chunk_text in enumerate(chunks):
                        chunk_id = f"{doc_id}_chunk_{chunk_index}"

                        # Create metadata
                        chunk_props = {
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "index": chunk_index,
                            "section": section_name,
                            "section_index": i,
                            "element_type": element_type,
                        }

                        # Add element metadata if available
                        if "metadata" in element and isinstance(element["metadata"], dict):
                            chunk_props["metadata"] = json.dumps(element["metadata"])

                        # Create chunk node and link to document
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            MERGE (c:Chunk {chunk_id: $chunk_id})
                            SET c += $properties
                            MERGE (d)-[:CONTAINS]->(c)
                            """,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            properties=chunk_props
                        )

                        chunk_index += 1

            # Set the document statistics
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                SET d.chunk_count = $chunk_count,
                    d.section_count = $section_count
                """,
                doc_id=doc_id,
                chunk_count=chunk_index,
                section_count=len(sections_created)
            )

            logger.info(f"Document {doc_id} stored in Neo4j with {chunk_index} chunks and {len(sections_created)} sections")

            # Return document ID and statistics
            return {
                "doc_id": doc_id,
                "chunk_count": chunk_index,
                "section_count": len(sections_created)
            }

    def process_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        enrich: Optional[bool] = None,
        detect_language: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document and extract text, structure, and metadata.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID
            doc_metadata: Optional document metadata
            enrich: Whether to enrich the document (defaults to class setting)
            detect_language: Whether to detect document language
            **kwargs: Additional parser-specific arguments

        Returns:
            Dict with document ID and processing details
        """
        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4()}"

        # Initialize metadata if not provided
        if doc_metadata is None:
            doc_metadata = {}
            
        # Validate and sanitize metadata
        self._sanitize_metadata(doc_metadata, file_name, file_content)
        
        # Ensure is_indexed is set to False
        if "is_indexed" not in doc_metadata:
            doc_metadata["is_indexed"] = False
            
        # Apply parser settings from metadata if provided
        if "parser_settings" in doc_metadata and isinstance(doc_metadata["parser_settings"], dict):
            settings = doc_metadata["parser_settings"]
            
            # Apply extract_images setting if provided
            if "extract_images" in settings:
                self.extract_images = settings["extract_images"]
                logger.info(f"Setting extract_images to {self.extract_images} from document metadata")
                
            # Apply extract_tables setting if provided
            if "extract_tables" in settings:
                self.extract_tables = settings["extract_tables"]
                logger.info(f"Setting extract_tables to {self.extract_tables} from document metadata")
                
            # Apply extract_metadata setting if provided
            if "extract_metadata" in settings:
                self.extract_metadata = settings["extract_metadata"]
                logger.info(f"Setting extract_metadata to {self.extract_metadata} from document metadata")
                
            # Apply chunking_strategy if provided
            if "chunking_strategy" in settings:
                self.chunking_strategy = settings["chunking_strategy"]
                logger.info(f"Setting chunking_strategy to {self.chunking_strategy} from document metadata")
                
            # Apply chunk_size if provided
            if "chunk_size" in settings and isinstance(settings["chunk_size"], int):
                self.chunk_size = settings["chunk_size"]
                logger.info(f"Setting chunk_size to {self.chunk_size} from document metadata")
                
            # Apply chunk_overlap if provided
            if "chunk_overlap" in settings and isinstance(settings["chunk_overlap"], int):
                self.chunk_overlap = settings["chunk_overlap"]
                logger.info(f"Setting chunk_overlap to {self.chunk_overlap} from document metadata")

        # Set the enrichment flag
        if enrich is None:
            enrich = self.use_enrichment
            
        # Check if file size exceeds threshold for special processing
        file_size = len(file_content)
        max_regular_size = 20 * 1024 * 1024  # 20 MB
        
        if file_size > max_regular_size:
            logger.info(f"Document size ({file_size/1024/1024:.2f} MB) exceeds threshold, using large document processing")
            
            # Use file extension to determine processing strategy
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if file_ext in ['.txt', '.md', '.csv', '.tsv']:
                # For text files, split by content
                return self._process_large_text_document(
                    file_content, file_name, doc_id, doc_metadata, enrich, detect_language
                )
            else:
                # For binary files like PDFs, use a different approach
                return self._process_large_binary_document(
                    file_content, file_name, doc_id, doc_metadata, enrich, detect_language,
                    max_size_per_chunk=10 * 1024 * 1024  # 10 MB per chunk
                )

        # For normal sized documents
        try:
            logger.info(f"Processing document: {file_name} (ID: {doc_id}, Size: {file_size/1024:.1f} KB)")
            
            # First, ensure the document exists in Neo4j, regardless of what happens later
            try:
                with self.driver.session() as session:
                    # Create document node first with minimal properties
                    # Extract file extension
                    file_ext = ""
                    if "." in file_name:
                        file_ext = file_name.split(".")[-1].lower()
                        
                    # Calculate file size in KB for display
                    size_kb = round(len(file_content) / 1024, 2)
                    
                    session.run(
                        """
                        MERGE (d:Document {doc_id: $doc_id})
                        SET d.name = $name,
                            d.title = $title,
                            d.created_at = datetime(),
                            d.uploaded_at = datetime(),
                            d.file_name = $name,
                            d.file_type = $file_type,
                            d.size = $size,
                            d.size_kb = $size_kb,
                            d.author = $author,
                            d.category = $category,
                            d.language = $language,
                            d.is_indexed = false,
                            d.processing_status = 'processing'
                        """,
                        doc_id=doc_id,
                        name=file_name,
                        title=doc_metadata.get("title", os.path.splitext(os.path.basename(file_name))[0]),
                        file_type=file_ext or "Unknown",
                        size=len(file_content),
                        size_kb=size_kb,
                        author=doc_metadata.get("author", "N/A"),
                        category=doc_metadata.get("category", "Uncategorized"),
                        language=doc_metadata.get("language", "en")
                    )
                logger.info(f"Successfully created document node in Neo4j with ID: {doc_id}")
            except Exception as ne:
                logger.error(f"Failed to create initial document node in Neo4j: {str(ne)}", exc_info=True)
                # We'll continue anyway to try the rest of the processing
            
            # Call Unstructured API to extract elements
            try:
                elements = self._call_unstructured_api(file_content, file_name)
            except Exception as e:
                logger.error(f"Error calling Unstructured API: {str(e)}", exc_info=True)
                # Create a fallback element to allow processing to continue
                elements = self._create_fallback_elements(file_name, file_content)
            
            # Process extracted elements (tables, images, etc.)
            try:
                self._process_table_elements(elements)
                self._enhance_metadata(elements, file_name)
            except Exception as e:
                logger.error(f"Error processing elements: {str(e)}", exc_info=True)
                # Continue with original elements
            
            # Count the different element types
            image_count = 0
            table_count = 0
            
            for element in elements:
                element_type = element.get("type", "unknown")
                if element_type == "Image":
                    image_count += 1
                elif element_type == "Table":
                    table_count += 1
            
            logger.info(f"Extracted {len(elements)} elements, including {image_count} images and {table_count} tables")
            
            # Detect language if requested
            language_code = None
            language_name = None
            
            if detect_language and not doc_metadata.get("language"):
                # Get a sample of text for language detection
                sample_bytes = file_content[:min(5000, len(file_content))]
                sample_text_for_detection = "" # Initialize

                if sample_bytes:
                    try:
                        sample_text_for_detection = sample_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.warning("Failed to decode sample text as UTF-8 for language detection. Trying with 'latin-1'.")
                        try:
                            sample_text_for_detection = sample_bytes.decode('latin-1') # Common fallback
                        except UnicodeDecodeError:
                            logger.warning("Failed to decode sample text with 'latin-1'. Using lossy UTF-8 decoding.")
                            sample_text_for_detection = sample_bytes.decode('utf-8', errors='replace')
                
                if sample_text_for_detection:
                    from data_enrichment.language_detector import LanguageDetector
                    language_detector = LanguageDetector()
                    language_info = language_detector.detect_language(sample_text_for_detection)
                    
                    language_code = language_info.get("language_code", "en")
                    language_name = language_info.get("language_name", "English")
                    confidence = language_info.get("confidence", 0.0)
                    
                    logger.info(f"Language detected: {language_name} ({language_code})")
                    
                    # Add language info to metadata
                    doc_metadata["language"] = language_code
                    doc_metadata["language_name"] = language_name
                    doc_metadata["language_confidence"] = confidence
                    
                    # Update document with language info
                    with self.driver.session() as session:
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            SET d.language = $language,
                                d.language_name = $language_name,
                                d.language_confidence = $confidence
                            """,
                            doc_id=doc_id,
                            language=language_code,
                            language_name=language_name,
                            confidence=confidence
                        )
            
            # Count elements with text
            text_elements = [e for e in elements if e.get("text", "").strip()]
            logger.info(f"Elements with text: {len(text_elements)} out of {len(elements)}")

            # Create Neo4j document and extract document structure
            with self.driver.session() as session:
                # Create document node first
                result = session.run(
                    """
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d += $properties
                    RETURN d
                    """,
                    doc_id=doc_id,
                    properties=doc_metadata
                )

            # Parse elements to chunks
            chunks = []
            # entities = [] # Will be populated by enrichment
            # concepts = [] # Will be populated by enrichment
            sections = []
            
            if self.chunking_strategy == "hierarchical":
                logger.info("Using hierarchical chunking strategy")
                chunks, sections = self._hierarchical_chunking_from_elements(elements, doc_id)
            else:
                # Default fixed chunking
                logger.info(f"Using fixed chunking strategy (size={self.chunk_size}, overlap={self.chunk_overlap})")
                chunks = self._fixed_chunking_from_elements(elements, doc_id)
            
            # Save chunks to Neo4j
            self._save_chunks_to_neo4j(chunks, doc_id)
            
            # If we have sections, save them too
            if sections:
                self._save_sections_to_neo4j(sections, doc_id)
            
            # Do enrichment if requested
            # Initialize with empty lists in case enrichment is skipped
            entities = []
            concepts = []

            if enrich:
                enrichment_output = self._enrich_document(doc_id, chunks)
                entities = enrichment_output.get("entities", [])
                concepts = enrichment_output.get("concepts", [])
                
            # Update document status in Neo4j
            try:
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        SET d.processed = true,
                            d.processed_at = datetime(),
                            d.chunk_count = $chunk_count,
                            d.section_count = $section_count,
                            d.is_indexed = false,
                            d.processing_status = 'completed'
                        """,
                        doc_id=doc_id,
                        chunk_count=len(chunks),
                        section_count=len(sections)
                    )
                    
                    # If language was detected, update document
                    if detect_language and "language" in doc_metadata:
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            SET d.language = $language,
                                d.language_name = $language_name,
                                d.language_confidence = $confidence
                            """,
                            doc_id=doc_id,
                            language=doc_metadata.get("language"),
                            language_name=doc_metadata.get("language_name"),
                            confidence=doc_metadata.get("language_confidence", 0.0)
                        )
            except Exception as e:
                logger.error(f"Error updating document status in Neo4j: {str(e)}", exc_info=True)
                # We'll continue to return success as document and chunks are created
            
            return {
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "section_count": len(sections),
                "entity_count": len(entities),
                "concept_count": len(concepts), # Added concept_count
                "language": doc_metadata.get("language"),
                "language_name": doc_metadata.get("language_name"),
                "image_count": image_count,
                "table_count": table_count
            }
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise

    def delete_document(self, doc_id: str, purge_orphans: bool = False) -> Dict[str, Any]:
        """
        Delete a document and all its related data from Neo4j.

        Args:
            doc_id: Document ID to delete
            purge_orphans: Whether to purge orphaned nodes after deletion

        Returns:
            Dictionary with deletion results
        """
        logger.info(f"Deleting document {doc_id} from Neo4j with purge_orphans={purge_orphans}")

        deletion_stats = {
            "document_deleted": False,
            "chunks_deleted": 0,
            "relationships_deleted": 0,
            "entities_deleted": 0,
            "concepts_deleted": 0,
            "legislation_deleted": 0,
            "requirements_deleted": 0,
            "deadlines_deleted": 0
        }

        try:
            with self.driver.session() as session:
                # Check if document exists
                check_result = session.run(
                    "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) as count, d.name as doc_name",
                    doc_id=doc_id
                )
                record = check_result.single()

                if record["count"] == 0:
                    logger.warning(f"Document {doc_id} not found in Neo4j")
                    return {"status": "error", "message": "Document not found"}

                doc_name = record["doc_name"]

                # First check which relationship types exist in the database to avoid warnings
                schema_result = session.run(
                    """
                    CALL db.relationshipTypes() YIELD relationshipType
                    RETURN collect(relationshipType) as types
                    """
                )
                rel_types = schema_result.single()["types"]

                # Similarly check which node labels exist
                labels_result = session.run(
                    """
                    CALL db.labels() YIELD label
                    RETURN collect(label) as labels
                    """
                )
                node_labels = labels_result.single()["labels"]

                logger.info(f"Found relationship types: {rel_types}")
                logger.info(f"Found node labels: {node_labels}")

                # First count all chunks for accurate statistics
                chunk_count_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                    RETURN count(c) as chunk_count
                    """,
                    doc_id=doc_id
                )
                
                chunk_count_record = chunk_count_result.single()
                chunk_count = chunk_count_record["chunk_count"] if chunk_count_record else 0
                
                if chunk_count is None:
                    chunk_count = 0
                    
                logger.info(f"Found {chunk_count} chunks to delete for document {doc_id}")

                # Get relationships count in a separate query to avoid null aggregation warnings
                rel_count_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[r]-()
                    RETURN count(r) as rel_count
                    """,
                    doc_id=doc_id
                )
                
                relationship_count = rel_count_result.single()["rel_count"]
                if relationship_count is None:
                    relationship_count = 0

                # Start transaction for atomic deletion
                tx = session.begin_transaction()
                try:
                    # Count specific entity types only if they exist in schema
                    entity_count = 0
                    if "HAS_ENTITY" in rel_types and "Entity" in node_labels:
                        # First update doc_names for entities
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_ENTITY]->(e:Entity)
                            WHERE e.doc_names IS NOT NULL AND $doc_name IN e.doc_names
                            SET e.doc_names = [name IN e.doc_names WHERE name <> $doc_name]
                            """,
                            doc_id=doc_id,
                            doc_name=doc_name
                        )

                        # Then count entities that will be deleted (those with empty doc_names)
                        entity_result = tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_ENTITY]->(e:Entity)
                            WHERE e.doc_names IS NULL OR size(e.doc_names) = 0
                            RETURN count(e) as count
                            """,
                            doc_id=doc_id
                        )
                        entity_count = entity_result.single()["count"]

                    concept_count = 0
                    if "HAS_CONCEPT" in rel_types and "Concept" in node_labels:
                        # First update doc_names for concepts
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_CONCEPT]->(c:Concept)
                            WHERE c.doc_names IS NOT NULL AND $doc_name IN c.doc_names
                            SET c.doc_names = [name IN c.doc_names WHERE name <> $doc_name]
                            """,
                            doc_id=doc_id,
                            doc_name=doc_name
                        )

                        # Then count concepts that will be deleted (those with empty doc_names)
                        concept_result = tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_CONCEPT]->(c:Concept)
                            WHERE c.doc_names IS NULL OR size(c.doc_names) = 0
                            RETURN count(c) as count
                            """,
                            doc_id=doc_id
                        )
                        concept_count = concept_result.single()["count"]

                    legislation_count = 0
                    if "REFERENCES_LEGISLATION" in rel_types and "Legislation" in node_labels:
                        # First update doc_names for legislation
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:REFERENCES_LEGISLATION]->(l:Legislation)
                            WHERE l.doc_names IS NOT NULL AND $doc_name IN l.doc_names
                            SET l.doc_names = [name IN l.doc_names WHERE name <> $doc_name]
                            """,
                            doc_id=doc_id,
                            doc_name=doc_name
                        )

                        # Then count legislation that will be deleted (those with empty doc_names)
                        legislation_result = tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:REFERENCES_LEGISLATION]->(l:Legislation)
                            WHERE l.doc_names IS NULL OR size(l.doc_names) = 0
                            RETURN count(l) as count
                            """,
                            doc_id=doc_id
                        )
                        legislation_count = legislation_result.single()["count"]

                    requirement_count = 0
                    if "HAS_REQUIREMENT" in rel_types and "Requirement" in node_labels:
                        # First update doc_names for requirements
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_REQUIREMENT]->(r:Requirement)
                            WHERE r.doc_names IS NOT NULL AND $doc_name IN r.doc_names
                            SET r.doc_names = [name IN r.doc_names WHERE name <> $doc_name]
                            """,
                            doc_id=doc_id,
                            doc_name=doc_name
                        )

                        # Then count requirements that will be deleted (those with empty doc_names)
                        requirement_result = tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_REQUIREMENT]->(r:Requirement)
                            WHERE r.doc_names IS NULL OR size(r.doc_names) = 0
                            RETURN count(r) as count
                            """,
                            doc_id=doc_id
                        )
                        requirement_count = requirement_result.single()["count"]

                    deadline_count = 0
                    if "HAS_DEADLINE" in rel_types and "Deadline" in node_labels:
                        # First update doc_names for deadlines
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_DEADLINE]->(dl:Deadline)
                            WHERE dl.doc_names IS NOT NULL AND $doc_name IN dl.doc_names
                            SET dl.doc_names = [name IN dl.doc_names WHERE name <> $doc_name]
                            """,
                            doc_id=doc_id,
                            doc_name=doc_name
                        )

                        # Then count deadlines that will be deleted (those with empty doc_names)
                        deadline_result = tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:HAS_DEADLINE]->(dl:Deadline)
                            WHERE dl.doc_names IS NULL OR size(dl.doc_names) = 0
                            RETURN count(dl) as count
                            """,
                            doc_id=doc_id
                        )
                        deadline_count = deadline_result.single()["count"]

                    # Also check for Text nodes (used in some parsers)
                    text_count = 0
                    if "Text" in node_labels:
                        text_result = tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(t:Text)
                            RETURN count(t) as count
                            """,
                            doc_id=doc_id
                        )
                        text_count = text_result.single()["count"]
                        
                        # Delete Text nodes if they exist
                        if text_count > 0:
                            logger.info(f"Found {text_count} Text nodes to delete")
                            tx.run(
                                """
                                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(t:Text)
                                OPTIONAL MATCH (t)-[tr]-()
                                DELETE tr
                                """,
                                doc_id=doc_id
                            )
                            
                            tx.run(
                                """
                                MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(t:Text)
                                DETACH DELETE t
                                """,
                                doc_id=doc_id
                            )

                    # Step 1: Delete relationships between Chunks and other nodes
                    if chunk_count > 0:
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                            OPTIONAL MATCH (c)-[cr]-()
                            DELETE cr
                            """,
                            doc_id=doc_id
                        )

                        # Step 2: Delete Chunk nodes
                        tx.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                            DETACH DELETE c
                            """,
                            doc_id=doc_id
                        )

                    # Step 3: Delete document relationships
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        OPTIONAL MATCH (d)-[r]-()
                        DELETE r
                        """,
                        doc_id=doc_id
                    )

                    # Step 4: Delete document node
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        DELETE d
                        """,
                        doc_id=doc_id
                    )

                    # Step 5: If purge_orphans is true, clean up any orphaned nodes
                    if purge_orphans:
                        logger.info("Purging orphaned nodes...")
                        # Delete nodes that have no relationships
                        if "Entity" in node_labels:
                            tx.run(
                                """
                                MATCH (e:Entity) 
                                WHERE NOT exists((e)--()) 
                                DELETE e
                                """
                            )
                        
                        if "Concept" in node_labels:
                            tx.run(
                                """
                                MATCH (c:Concept) 
                                WHERE NOT exists((c)--()) 
                                DELETE c
                                """
                            )
                        
                        if "Legislation" in node_labels:
                            tx.run(
                                """
                                MATCH (l:Legislation) 
                                WHERE NOT exists((l)--()) 
                                DELETE l
                                """
                            )
                        
                        if "Requirement" in node_labels:
                            tx.run(
                                """
                                MATCH (r:Requirement) 
                                WHERE NOT exists((r)--()) 
                                DELETE r
                                """
                            )
                        
                        if "Deadline" in node_labels:
                            tx.run(
                                """
                                MATCH (d:Deadline) 
                                WHERE NOT exists((d)--()) 
                                DELETE d
                                """
                            )
                    
                    tx.commit()
                    logger.info(f"Document {doc_id} deleted successfully with {chunk_count} chunks")
                except Exception as e:
                    tx.rollback()
                    logger.error(f"Error in document deletion transaction: {str(e)}")
                    raise
                
                # Verify deletion was successful with a separate transaction
                verify_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id}) 
                    RETURN count(d) as doc_count
                    """,
                    doc_id=doc_id
                )
                
                doc_count = verify_result.single()["doc_count"]
                if doc_count > 0:
                    logger.error(f"Document {doc_id} still exists after deletion attempt")
                    return {
                        "status": "error",
                        "message": f"Document {doc_id} still exists after deletion attempt"
                    }
                
                # Also verify if chunks were deleted properly
                chunk_verify_result = session.run(
                    """
                    MATCH (c:Chunk {doc_id: $doc_id}) 
                    RETURN count(c) as chunk_count
                    """,
                    doc_id=doc_id
                )
                
                orphaned_chunks = chunk_verify_result.single()["chunk_count"]
                if orphaned_chunks > 0:
                    logger.warning(f"Found {orphaned_chunks} orphaned chunks after document deletion")
                    
                    if purge_orphans:
                        # Delete orphaned chunks in a new transaction
                        cleanup_tx = session.begin_transaction()
                        try:
                            cleanup_tx.run(
                                """
                                MATCH (c:Chunk {doc_id: $doc_id})
                                DETACH DELETE c
                                """,
                                doc_id=doc_id
                            )
                            cleanup_tx.commit()
                            logger.info(f"Cleaned up {orphaned_chunks} orphaned chunks")
                        except Exception as cleanup_error:
                            cleanup_tx.rollback()
                            logger.error(f"Error cleaning up orphaned chunks: {str(cleanup_error)}")

            deletion_stats["document_deleted"] = True
            deletion_stats["chunks_deleted"] = chunk_count
            deletion_stats["relationships_deleted"] = relationship_count
            deletion_stats["entities_deleted"] = entity_count
            deletion_stats["concepts_deleted"] = concept_count
            deletion_stats["legislation_deleted"] = legislation_count
            deletion_stats["requirements_deleted"] = requirement_count
            deletion_stats["deadlines_deleted"] = deadline_count
            if "Text" in node_labels:
                deletion_stats["text_nodes_deleted"] = text_count

            return deletion_stats

        except Exception as e:
            logger.error(f"Error during deletion: {str(e)}")
            raise

    def process_large_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        enrich: Optional[bool] = None,
        detect_language: bool = True,
        max_size_per_chunk: int = 5 * 1024 * 1024,  # 5MB chunks
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a large document by chunking it into smaller pieces.

        For the Unstructured API implementation, we simply delegate to the regular
        process_document method as it already handles large documents appropriately.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID
            doc_metadata: Optional document metadata
            enrich: Whether to enrich the document (defaults to class setting)
            detect_language: Whether to detect document language
            max_size_per_chunk: Maximum size per chunk in bytes
            **kwargs: Additional parser-specific arguments

        Returns:
            Dict with document ID and processing details
        """
        return self.process_document(
            file_content=file_content,
            file_name=file_name,
            doc_id=doc_id,
            doc_metadata=doc_metadata,
            enrich=enrich,
            detect_language=detect_language,
            **kwargs
        )

    def close(self):
        """
        Close the document parser and release resources.
        """
        self.driver.close()

    def _sanitize_metadata(self, metadata: Dict[str, Any], file_name: str, file_content: bytes) -> None:
        """
        Sanitize metadata to ensure valid values.
        
        Args:
            metadata: Metadata dictionary to sanitize (modified in place)
            file_name: Name of the file
            file_content: Binary content of the file
        """
        # Ensure basic metadata fields exist with valid values
        
        # File size
        if "size" not in metadata or not isinstance(metadata["size"], (int, float)) or math.isnan(metadata["size"]):
            metadata["size"] = len(file_content)
            
        # Calculate size in KB for display
        metadata["size_kb"] = round(metadata["size"] / 1024, 2) if metadata["size"] > 0 else 0
            
        # File type
        if "file_type" not in metadata or not metadata["file_type"]:
            ext = os.path.splitext(file_name)[1].lower().lstrip('.')
            metadata["file_type"] = ext if ext else "Unknown"
            
        # Title
        if "title" not in metadata or not metadata["title"]:
            # Use filename without extension as title
            base_name = os.path.basename(file_name)
            title = os.path.splitext(base_name)[0]
            # Convert CamelCase or snake_case to spaces for better readability
            title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)  # CamelCase to spaces
            title = re.sub(r'_+', ' ', title)  # snake_case to spaces
            title = re.sub(r'\s+', ' ', title).strip()  # Normalize spaces
            metadata["title"] = title if title else file_name
            
        # Author
        if "author" not in metadata or not isinstance(metadata["author"], str):
            metadata["author"] = "N/A"
            
        # Category
        if "category" not in metadata or not isinstance(metadata["category"], str):
            metadata["category"] = "Uncategorized"
            
        # Timestamps
        current_time = py_datetime.now().isoformat()
        
        if "created_at" not in metadata or not metadata["created_at"]:
            metadata["created_at"] = current_time
            
        if "uploaded_at" not in metadata:
            metadata["uploaded_at"] = current_time
            
        # Language (this will be further processed during language detection)
        if "language" not in metadata or not isinstance(metadata["language"], str):
            metadata["language"] = "en"
            
        # Clean any None values
        for key in list(metadata.keys()):
            if metadata[key] is None:
                if key in ["title", "author", "category"]:
                    # Replace important None fields with defaults
                    if key == "title":
                        metadata[key] = os.path.basename(file_name)
                    elif key == "author":
                        metadata[key] = "N/A"
                    elif key == "category":
                        metadata[key] = "Uncategorized"
                else:
                    # Remove other None fields
                    del metadata[key]
