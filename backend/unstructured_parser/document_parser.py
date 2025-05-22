# plugins/regul_aite/backend/unstructured_parser/document_parser.py
import os
import requests
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, BinaryIO, Callable, Literal
from datetime import datetime as py_datetime  
import datetime  # Import the full module too
import tempfile
import time
import math
import re
import asyncio

# Import Qdrant client
from qdrant_client import QdrantClient, models as qdrant_models

# Import MetadataParser
from data_enrichment.metadata_parser import MetadataParser

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
    """

    def __init__(
          self,
          unstructured_api_url: Optional[str] = None,
          unstructured_api_key: Optional[str] = None,
          qdrant_url: Optional[str] = None,
          qdrant_collection_name: str = "regulaite_docs",
          qdrant_metadata_collection_name: str = "regulaite_metadata",
          embedding_dim: int = 384,  # Added embedding_dim
          chunk_size: int = 1000,
          chunk_overlap: int = 200,
          chunking_strategy: ChunkingStrategy = "fixed",
          extract_tables: bool = True,
          extract_metadata: bool = True,
          extract_images: bool = False,
          is_cloud: bool = False
      ):
      """
      Initialize the document parser.

      Args:
          unstructured_api_url: URL for Unstructured API (defaults to env var)
          unstructured_api_key: API key for Unstructured API (defaults to env var)
          qdrant_url: URL for Qdrant service
          qdrant_collection_name: Name of the Qdrant collection for document chunks
          qdrant_metadata_collection_name: Name of the Qdrant collection for document metadata
          embedding_dim: Dimension of the embeddings to be used for dummy vectors
          chunk_size: Maximum size of text chunks in characters
          chunk_overlap: Number of characters to overlap between chunks
          chunking_strategy: Strategy for chunking text ("fixed", "recursive", "semantic", "hierarchical", "token")
          extract_tables: Whether to extract tables from documents
          extract_metadata: Whether to extract detailed metadata
          extract_images: Whether to extract and process images
          is_cloud: Whether to use the cloud version of Unstructured API
      """
      self.is_cloud = is_cloud
      self.embedding_dim = embedding_dim # Store embedding_dim

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

      # Initialize Qdrant client
      self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://qdrant:6333")
      self.qdrant_collection_name = qdrant_collection_name
      self.qdrant_metadata_collection_name = qdrant_metadata_collection_name
      try:
          self.qdrant_client = QdrantClient(url=self.qdrant_url)
          logger.info(f"Successfully connected to Qdrant at {self.qdrant_url}")
      except Exception as e:
          logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {e}")
          self.qdrant_client = None

      # Initialize metadata parser
      self.metadata_parser = MetadataParser()
      
      # Log initialization
      api_type = "cloud" if is_cloud else "local"
      logger.info(f"Initialized DocumentParser with {api_type} Unstructured API at {self.unstructured_api_url}")

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
        """Process table elements to enhance their data."""
        for element in elements:
            if element.get("type") == "Table":
                # Add any table-specific processing here
                pass
    
    def _enhance_metadata(self, elements: List[Dict[str, Any]], file_name: str) -> None:
        """
        Enhance element metadata with additional information.
        
        Args:
            elements: List of document elements to enhance
            file_name: Original filename for reference
        """
        try:
            # Extract basename from file
            basename = os.path.basename(file_name)
            
            # Add global metadata to all elements
            for element in elements:
                if "metadata" not in element:
                    element["metadata"] = {}
                
                # Add source file info
                element["metadata"]["source_file"] = basename
                
                # Add position context if available
                if "position" in element and isinstance(element["position"], dict):
                    # Copy relevant position info to metadata
                    for key in ["page_number", "section", "paragraph"]:
                        if key in element["position"]:
                            element["metadata"][key] = element["position"][key]
        
        except Exception as e:
            logger.error(f"Error enhancing element metadata: {str(e)}")

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
                
            # Check if this is a title element to mark a new section
            if element_type == "Title":
                # If we have text in the current chunk, save it before starting a new section
                if current_chunk_text:
                    chunk = {
                        "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                        "text": current_chunk_text,
                        "index": chunk_index,
                        "element_type": ", ".join(current_element_types),
                        "section": current_section,
                        "metadata": current_metadata.copy() if current_metadata else {},
                        "page_num": current_page_num,  # Add page number
                        "order_index": chunk_index,  # Add order index
                        "doc_id": doc_id  # Add doc_id directly in chunk
                    }
                    
                    # Make sure doc_id is also in metadata
                    if "doc_id" not in chunk["metadata"]:
                        chunk["metadata"]["doc_id"] = doc_id
                        
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_text = ""
                    current_element_types = []
                    
                # Update current section to the title text
                current_section = element_text
                
            # Add element to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n"
            current_chunk_text += element_text
            
            # Keep track of element types in this chunk
            if element_type not in current_element_types:
                current_element_types.append(element_type)
                
            # Copy metadata from element if available
            if "metadata" in element and isinstance(element["metadata"], dict):
                for key, value in element["metadata"].items():
                    current_metadata[key] = value
                    
            # If we've reached the max chunk size, finalize this chunk and start a new one
            if len(current_chunk_text) >= self.chunk_size:
                chunk = {
                    "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                    "text": current_chunk_text,
                    "index": chunk_index,
                    "element_type": ", ".join(current_element_types),
                    "section": current_section,
                    "metadata": current_metadata.copy() if current_metadata else {},
                    "page_num": current_page_num,  # Add page number
                    "order_index": chunk_index,  # Add order index
                    "doc_id": doc_id  # Add doc_id directly in chunk
                }
                
                # Make sure doc_id is also in metadata
                if "doc_id" not in chunk["metadata"]:
                    chunk["metadata"]["doc_id"] = doc_id
                    
                chunks.append(chunk)
                chunk_index += 1
                
                # Start a new chunk with overlap
                words = current_chunk_text.split()
                overlap_word_count = min(len(words), int(self.chunk_overlap / 5))  # Approximate words for overlap
                overlap_text = " ".join(words[-overlap_word_count:])
                
                current_chunk_text = overlap_text
                current_element_types = [element_type]  # Reset to current element type
        
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
                "order_index": chunk_index,  # Add order index
                "doc_id": doc_id  # Add doc_id directly in chunk
            }
            
            # Make sure doc_id is also in metadata
            if "doc_id" not in chunk["metadata"]:
                chunk["metadata"]["doc_id"] = doc_id
                
            chunks.append(chunk)
        
        return chunks
           
    def _enrich_document(self, doc_id: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        This method has been modified to no longer perform entity extraction.
        It now always returns an empty list to maintain compatibility with the rest of the codebase.

        Args:
            doc_id: Document ID
            chunks: Document chunks

        Returns:
            Empty list (no entities)
        """
        logger.info(f"Entity extraction feature has been disabled for document {doc_id}")
        return []

    def process_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
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
        
        # Ensure is_indexed is set to False initially in the main metadata
        doc_metadata["is_indexed"] = False
        doc_metadata["status"] = "processing" # Indicate it's being processed

        # Persist initial document metadata (including size, title etc.) to Qdrant metadata collection
        if self.qdrant_client:
            try:
                metadata_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                # Ensure all expected fields for DocumentMetadata model are present or have defaults
                # This helps prevent issues if RAGSystem's _get_document_metadata expects certain fields
                
                # Default values for fields often expected by RAGSystem or frontend
                # based on DocumentMetadata Pydantic model and RAGSystem._create_default_metadata
                payload_to_store = {
                    "doc_id": doc_id,
                    "title": doc_metadata.get("title", f"Document {doc_id}"),
                    "name": doc_metadata.get("original_filename", file_name), # Use original_filename for display
                    "is_indexed": doc_metadata.get("is_indexed", False),
                    "file_type": doc_metadata.get("file_type", "unknown"),
                    "description": doc_metadata.get("description", ""),
                    "language": doc_metadata.get("language", "en"),
                    "size": doc_metadata.get("size", 0),
                    "page_count": doc_metadata.get("page_count", 0),
                    "chunk_count": doc_metadata.get("chunk_count", 0), # Will be 0 initially
                    "created_at": doc_metadata.get("created_at", py_datetime.now().isoformat()),
                    "tags": doc_metadata.get("tags", []),
                    "category": doc_metadata.get("category", "Uncategorized"),
                    "author": doc_metadata.get("author", "N/A"),
                    "status": doc_metadata.get("status", "active"),
                    # Add any other fields from doc_metadata that are not explicitly listed
                    **doc_metadata 
                }


                self.qdrant_client.upsert(
                    collection_name=self.qdrant_metadata_collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=metadata_point_id,
                            vector=[1.0] * self.embedding_dim, # Use embedding_dim consistent with collection
                            payload=payload_to_store
                        )
                    ]
                )
                logger.info(f"Stored initial metadata for document {doc_id} in '{self.qdrant_metadata_collection_name}'")
            except Exception as e:
                logger.error(f"Error storing initial metadata for document {doc_id} in Qdrant: {e}", exc_info=True)
        else:
            logger.error("Qdrant client not available, cannot store initial document metadata.")

            
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
                    file_content, file_name, doc_id, doc_metadata
                )
            else:
                # For binary files like PDFs, use a different approach
                return self._process_large_binary_document(
                    file_content, file_name, doc_id, doc_metadata,
                    max_size_per_chunk=10 * 1024 * 1024  # 10 MB per chunk
                )

        # For normal sized documents
        try:
            logger.info(f"Processing document: {file_name} (ID: {doc_id}, Size: {file_size/1024:.1f} KB)")
            
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
                sample_text = file_content[:min(5000, len(file_content))]
                
                if sample_text:
                    from data_enrichment.language_detector import LanguageDetector
                    language_detector = LanguageDetector()
                    language_info = language_detector.detect_language(sample_text)
                    
                    language_code = language_info.get("language_code", "en")
                    language_name = language_info.get("language_name", "English")
                    confidence = language_info.get("confidence", 0.0)
                    
                    logger.info(f"Language detected: {language_name} ({language_code})")
                    
                    # Add language info to metadata
                    doc_metadata["language"] = language_code
                    doc_metadata["language_name"] = language_name
                    doc_metadata["language_confidence"] = confidence
                    
                    # Update document metadata in Qdrant with language info
                    if self.qdrant_client:
                        try:
                            metadata_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                            # Prepare payload with updated language and potentially other fields
                            # It's important to merge with existing metadata in Qdrant or ensure full update
                            # For simplicity, we update specific fields here. A more robust solution
                            # would fetch existing payload, merge, then upsert.
                            # However, RAGSystem._update_document_metadata does a full upsert.
                            # So we should also provide the full metadata again.
                            
                            # Fetch current doc_metadata again as it might have been updated by sanitize_metadata
                            # or other previous steps.
                            current_payload = doc_metadata.copy() # Start with what we have
                            current_payload["language"] = language_code
                            current_payload["language_name"] = language_name
                            current_payload["language_confidence"] = confidence
                            
                            # Ensure all necessary fields are present
                            final_payload_for_update = {
                                "doc_id": doc_id,
                                "title": current_payload.get("title", f"Document {doc_id}"),
                                "name": current_payload.get("original_filename", file_name),
                                "is_indexed": current_payload.get("is_indexed", False),
                                "file_type": current_payload.get("file_type", "unknown"),
                                "description": current_payload.get("description", ""),
                                "language": current_payload.get("language", "en"),
                                "size": current_payload.get("size", 0),
                                "page_count": current_payload.get("page_count", 0),
                                "chunk_count": current_payload.get("chunk_count", 0), 
                                "created_at": current_payload.get("created_at", py_datetime.now().isoformat()),
                                "tags": current_payload.get("tags", []),
                                "category": current_payload.get("category", "Uncategorized"),
                                "author": current_payload.get("author", "N/A"),
                                "status": current_payload.get("status", "active"),
                                 **current_payload # Add any other fields
                            }


                            self.qdrant_client.upsert(
                                collection_name=self.qdrant_metadata_collection_name,
                                points=[
                                    qdrant_models.PointStruct(
                                        id=metadata_point_id,
                                        vector=[1.0] * self.embedding_dim, # Use embedding_dim consistent with collection
                                        payload=final_payload_for_update
                                    )
                                ]
                            )
                            logger.info(f"Updated metadata for document {doc_id} with language info.")
                        except Exception as e:
                            logger.error(f"Error updating metadata with language info for {doc_id}: {e}", exc_info=True)
            
            # Parse elements to chunks
            chunks = []
            sections = []
            
            if self.chunking_strategy == "hierarchical":
                logger.info("Using hierarchical chunking strategy")
                chunks, sections = self._hierarchical_chunking_from_elements(elements, doc_id)
            else:
                # Default fixed chunking
                logger.info(f"Using fixed chunking strategy (size={self.chunk_size}, overlap={self.chunk_overlap})")
                chunks = self._fixed_chunking_from_elements(elements, doc_id)
            
            # Store chunks in Qdrant
            if self.qdrant_client and chunks:
                try:
                    points_to_upsert = []
                    skipped_chunks = 0
                    for chunk_idx, chunk_data in enumerate(chunks):
                        # Ensure basic fields are present
                        payload_chunk_id = chunk_data.get("chunk_id", f"{doc_id}_chunk_{chunk_idx}")
                        qdrant_point_id = str(uuid.uuid4()) # Generate UUID for Qdrant point ID
                        text_content = chunk_data.get("text", "")
                        
                        # Skip chunks with empty text content
                        if not text_content or text_content.strip() == "":
                            skipped_chunks += 1
                            continue
                            
                        page_num = chunk_data.get("page_num", 0)

                        # Create payload for Qdrant
                        payload = {
                            "doc_id": doc_id,  # Add doc_id at root level
                            "chunk_id": payload_chunk_id, # Use the original chunk_id style here
                            "text": text_content,
                            "page_number": page_num,
                            "metadata": chunk_data.get("metadata", {}),
                            "element_type": chunk_data.get("element_type", "unknown"),
                            "order_index": chunk_data.get("order_index", chunk_idx)
                        }
                        
                        # Ensure doc_id is also in metadata
                        if "metadata" in payload and isinstance(payload["metadata"], dict):
                            payload["metadata"]["doc_id"] = doc_id
                        
                        # Add a dummy vector (e.g., all ones) - RAG will create real embeddings later
                        # Use the embedding_dim passed to the constructor
                        dummy_vector = [1.0] * self.embedding_dim
                        
                        points_to_upsert.append(
                            qdrant_models.PointStruct(
                                id=qdrant_point_id,
                                vector=dummy_vector,
                                payload=payload
                            )
                        )
                    
                    # Batch upsert all points
                    if points_to_upsert:
                        # Use wait=True to ensure operation completes before continuing
                        self.qdrant_client.upsert(
                            collection_name=self.qdrant_collection_name,
                            points=points_to_upsert,
                            wait=True
                        )
                        logger.info(f"Stored {len(points_to_upsert)} chunks for document {doc_id} in Qdrant collection '{self.qdrant_collection_name}'")
                        
                        # Verify the chunks were stored properly
                        try:
                            verification_response = self.qdrant_client.scroll(
                                collection_name=self.qdrant_collection_name,
                                scroll_filter=qdrant_models.Filter(
                                    must=[
                                        qdrant_models.FieldCondition(
                                            key="doc_id",
                                            match=qdrant_models.MatchValue(value=doc_id)
                                        )
                                    ]
                                ),
                                limit=1,
                                with_payload=True
                            )
                            
                            if verification_response and len(verification_response[0]) > 0:
                                logger.info(f"Verified chunks for document {doc_id} are stored in Qdrant")
                            else:
                                logger.warning(f"Failed to verify chunks for document {doc_id} in Qdrant")
                        except Exception as verify_e:
                            logger.warning(f"Error verifying stored chunks: {str(verify_e)}")
                        
                    # Update document metadata with chunk count
                    if doc_metadata is not None and "chunk_count" not in doc_metadata:
                        doc_metadata["chunk_count"] = len(points_to_upsert)

                    # Update document status to processed
                    try:
                        metadata_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                        
                        # Get existing metadata
                        try:
                            metadata_response = self.qdrant_client.retrieve(
                                collection_name=self.qdrant_metadata_collection_name,
                                ids=[metadata_point_id],
                                with_payload=True
                            )
                            
                            if metadata_response and len(metadata_response) > 0:
                                existing_metadata = metadata_response[0].payload
                            else:
                                existing_metadata = {}
                        except Exception as retrieve_error:
                            logger.warning(f"Error retrieving existing metadata for {doc_id}: {str(retrieve_error)}")
                            existing_metadata = {}
                        
                        # Merge with existing metadata
                        final_payload_for_update = existing_metadata.copy() if existing_metadata else {}
                        final_payload_for_update.update({
                            "doc_id": doc_id,
                            "status": "processed",
                            "chunk_count": len(points_to_upsert),
                            "is_indexed": doc_metadata.get("is_indexed", False)
                        })
                        
                        # Update metadata in Qdrant
                        self.qdrant_client.upsert(
                            collection_name=self.qdrant_metadata_collection_name,
                            points=[
                                qdrant_models.PointStruct(
                                    id=metadata_point_id,
                                    vector=[1.0] * self.embedding_dim,
                                    payload=final_payload_for_update
                                )
                            ],
                            wait=True
                        )
                        logger.info(f"Updated metadata for document {doc_id} to status 'processed' with chunk_count {len(points_to_upsert)}")
                    except Exception as e:
                        logger.error(f"Error updating document metadata: {str(e)}")
                except Exception as e:
                    logger.error(f"Error storing chunks in Qdrant: {str(e)}")
                    # Continue processing - we'll still return the extracted information

            elif not self.qdrant_client:
                logger.error(f"Qdrant client not initialized. Cannot store chunks for document {doc_id}")
            
            # After successful chunking and storage, update metadata status to "processed" or "pending_indexing"
            if self.qdrant_client:
                try:
                    metadata_point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                    # Update status to 'processed', indicating chunks are stored, ready for indexing by RAGSystem
                    # The RAGSystem will later update it to is_indexed=True and status='active' or 'indexed'
                    processed_payload = doc_metadata.copy()
                    processed_payload["status"] = "processed" 
                    processed_payload["chunk_count"] = len(chunks) # Update chunk_count here

                    final_processed_payload = {
                        "doc_id": doc_id,
                        "title": processed_payload.get("title", f"Document {doc_id}"),
                        "name": processed_payload.get("original_filename", file_name),
                        "is_indexed": processed_payload.get("is_indexed", False), # Should still be False
                        "file_type": processed_payload.get("file_type", "unknown"),
                        "description": processed_payload.get("description", ""),
                        "language": processed_payload.get("language", "en"),
                        "size": processed_payload.get("size", 0),
                        "page_count": processed_payload.get("page_count", 0), 
                        "chunk_count": processed_payload.get("chunk_count", 0), 
                        "created_at": processed_payload.get("created_at", py_datetime.now().isoformat()),
                        "tags": processed_payload.get("tags", []),
                        "category": processed_payload.get("category", "Uncategorized"),
                        "author": processed_payload.get("author", "N/A"),
                        "status": "processed", # Explicitly set status
                         **processed_payload # Add any other fields
                    }
                    
                    self.qdrant_client.upsert(
                        collection_name=self.qdrant_metadata_collection_name,
                        points=[
                            qdrant_models.PointStruct(
                                id=metadata_point_id,
                                vector=[1.0] * self.embedding_dim, # Use embedding_dim consistent with collection
                                payload=final_processed_payload
                            )
                        ]
                    )
                    logger.info(f"Updated metadata for document {doc_id} to status 'processed' with chunk_count {len(chunks)}.")
                except Exception as e:
                    logger.error(f"Error updating metadata status to processed for {doc_id}: {e}", exc_info=True)

            return {
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "section_count": len(sections),
                "image_count": image_count,
                "table_count": table_count
            }
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            raise

    def process_large_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
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
            detect_language=True,
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

    async def reprocess_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Reprocess a document by re-parsing and re-chunking it.
        
        Args:
            doc_id: Document ID to reprocess
            
        Returns:
            Dict with operation status
        """
        logger.info(f"Reprocessing document: {doc_id}")
        
        try:
            # First, check if document exists in the metadata store
            doc_metadata = await self.get_document_metadata(doc_id)
            
            if not doc_metadata:
                logger.error(f"Document {doc_id} not found in metadata store")
                return {
                    "status": "error",
                    "message": f"Document {doc_id} not found",
                    "doc_id": doc_id
                }
                
            # Delete all existing chunks for the document
            try:
                await self.delete_document_chunks(doc_id)
                logger.info(f"Deleted existing chunks for document {doc_id}")
            except Exception as del_error:
                logger.error(f"Error deleting chunks for document {doc_id}: {str(del_error)}")
                
            # Check if the document file still exists
            original_path = doc_metadata.get("original_path")
            if not original_path or not os.path.exists(original_path):
                logger.error(f"Original file for document {doc_id} not found at {original_path}")
                return {
                    "status": "error",
                    "message": f"Original file for document {doc_id} not found",
                    "doc_id": doc_id
                }
                
            # Reprocess the document using the original path
            try:
                # Update metadata to indicate reprocessing
                doc_metadata["status"] = "reprocessing"
                doc_metadata["is_indexed"] = False
                doc_metadata["updated_at"] = datetime.now().isoformat()
                await self.update_document_metadata(doc_id, doc_metadata)
                
                # Start reprocessing
                reprocess_task = asyncio.create_task(
                    self.process_document_file(
                        file_path=original_path,
                        doc_id=doc_id,
                        file_extension=doc_metadata.get("file_type", ""),
                        language=doc_metadata.get("language", "auto"),
                        metadata=doc_metadata
                    )
                )
                
                return {
                    "status": "success",
                    "message": f"Document {doc_id} reprocessing started",
                    "doc_id": doc_id
                }
                
            except Exception as proc_error:
                logger.error(f"Error starting reprocessing for document {doc_id}: {str(proc_error)}")
                
                # Update metadata to indicate error
                doc_metadata["status"] = "error"
                doc_metadata["error_message"] = f"Reprocessing failed: {str(proc_error)}"
                doc_metadata["updated_at"] = datetime.now().isoformat()
                await self.update_document_metadata(doc_id, doc_metadata)
                
                return {
                    "status": "error",
                    "message": f"Error reprocessing document: {str(proc_error)}",
                    "doc_id": doc_id
                }
                
        except Exception as e:
            logger.error(f"Error in document reprocessing: {str(e)}")
            return {
                "status": "error",
                "message": f"Error reprocessing document: {str(e)}",
                "doc_id": doc_id
            }
            
    async def delete_document_chunks(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a document from Qdrant.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict with deletion status
        """
        try:
            # Delete from Qdrant
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="doc_id",
                            match=qdrant_models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            
            logger.info(f"Deleted chunks for document {doc_id} from Qdrant")
            
            return {
                "status": "success",
                "message": f"Deleted chunks for document {doc_id}",
                "doc_id": doc_id
            }
            
        except Exception as e:
            logger.error(f"Error deleting chunks for document {doc_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error deleting chunks: {str(e)}",
                "doc_id": doc_id
            }

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document from Qdrant.
        
        Args:
            doc_id: Document ID to retrieve chunks for
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not self.qdrant_client:
            logger.error("Qdrant client not available, cannot retrieve document chunks")
            return []
        
        try:
            # Query Qdrant for chunks with this doc_id
            response = self.qdrant_client.scroll(
                                collection_name=self.qdrant_collection_name,
                                scroll_filter=qdrant_models.Filter(
                                    must=[
                                        qdrant_models.FieldCondition(
                                            key="doc_id",
                                            match=qdrant_models.MatchValue(value=doc_id)
                                        )
                                    ]
                                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            if response and len(response[0]) > 0:
                for point in response[0]:
                    payload = point.payload
                    # Create a chunk object with standard fields
                    chunk = {
                        "text": payload.get("text", ""),
                        "content": payload.get("text", ""),  # Use 'text' as 'content' for compatibility
                        "metadata": {
                            "doc_id": doc_id,
                            "chunk_id": payload.get("chunk_id", ""),
                            "page_num": payload.get("page_number", 0),
                            "element_type": payload.get("element_type", "")
                        }
                    }
                    
                    # Add additional metadata if available
                    if "metadata" in payload and isinstance(payload["metadata"], dict):
                        for key, value in payload["metadata"].items():
                            chunk["metadata"][key] = value
                    
                    chunks.append(chunk)
                
                logger.info(f"Retrieved {len(chunks)} chunks for document {doc_id} from Qdrant")
                return chunks
            
            # If no chunks found with 'doc_id' field, try with 'metadata.doc_id'
            response = self.qdrant_client.scroll(
                collection_name=self.qdrant_collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="metadata.doc_id",
                            match=qdrant_models.MatchValue(value=doc_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            if response and len(response[0]) > 0:
                for point in response[0]:
                    payload = point.payload
                    # Create a chunk object with standard fields
                    chunk = {
                        "text": payload.get("text", ""),
                        "content": payload.get("text", ""),  # Use 'text' as 'content' for compatibility
                        "metadata": payload.get("metadata", {})
                    }
                    
                    # Ensure doc_id is in metadata
                    if "metadata" in chunk and "doc_id" not in chunk["metadata"]:
                        chunk["metadata"]["doc_id"] = doc_id
                    
                    chunks.append(chunk)
                
                logger.info(f"Retrieved {len(chunks)} chunks for document {doc_id} from Qdrant using metadata.doc_id")
                return chunks
            
            logger.warning(f"No chunks found for document {doc_id} in Qdrant")
            return []
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {doc_id}: {str(e)}")
            return []
