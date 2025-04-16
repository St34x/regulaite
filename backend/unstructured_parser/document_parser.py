# plugins/regul_aite/backend/unstructured_parser/document_parser.py
import os
import requests
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, BinaryIO, Callable, Literal
from datetime import datetime
import tempfile
from neo4j import GraphDatabase
from data_enrichment.language_detector import LanguageDetector
from data_enrichment.enrichment_pipeline import EnrichmentPipeline
from data_enrichment.metadata_parser import MetadataParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define chunking strategies
ChunkingStrategy = Literal["fixed", "recursive", "semantic", "hierarchical"]

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
          chunking_strategy: Strategy for chunking text ("fixed", "recursive", "semantic", "hierarchical")
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
              logger.info("Enrichment pipeline initialized")
          except Exception as e:
              logger.error(f"Failed to initialize enrichment pipeline: {str(e)}")
              self.enrichment_pipeline = None
              self.use_enrichment = False
      else:
          self.enrichment_pipeline = None

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

              response = requests.post(
                  self.unstructured_api_url,
                  headers=headers,
                  files=files,
                  data=data
              )

              if response.status_code != 200:
                  logger.error(f"Unstructured API error: {response.status_code} - {response.text}")
                  raise Exception(f"Unstructured API error: {response.status_code}")

              # Parse and return the response
              elements = response.json()
              logger.info(f"Extracted {len(elements)} elements from document")

              # Log element types to help with debugging
              element_types = {}
              text_count = 0
              for element in elements:
                  element_type = element.get("type", "unknown")
                  has_text = bool(element.get("text", "").strip())
                  if has_text:
                      text_count += 1

                  if element_type in element_types:
                      element_types[element_type] += 1
                  else:
                      element_types[element_type] = 1

              logger.info(f"Element types found: {element_types}")
              logger.info(f"Elements with text: {text_count} out of {len(elements)}")

              # Enhanced post-processing for specific element types
              self._process_table_elements(elements)
              self._enhance_metadata(elements, file_name)

              # Return the extracted elements
              return elements

      except Exception as e:
          logger.error(f"Error calling Unstructured API: {str(e)}")
          raise
      finally:
          # Remove the temporary file
          if os.path.exists(temp_file_path):
              os.remove(temp_file_path)

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

        # Prepare document metadata
        doc_metadata = {
            "doc_id": doc_id,
            "name": file_name,
            "created": datetime.now().isoformat(),
            "indexed": False,
        }

        # Add additional metadata (with circular reference protection)
        doc_metadata.update(self._safe_copy_metadata(metadata))

        # Store document node and chunks
        with self.driver.session() as session:
            # Create document node
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d += $properties
                RETURN d
                """,
                doc_id=doc_id,
                properties=doc_metadata
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
        detect_language: bool = True
    ) -> Dict[str, Any]:
        """
        Process a document using Unstructured API and store in Neo4j.
        For large documents, uses partial processing to handle them efficiently.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID (generated if not provided)
            doc_metadata: Optional document metadata
            enrich: Whether to apply enrichment
            detect_language: Whether to detect document language

        Returns:
            Dictionary with document ID and processing details
        """
        try:
            # Check file size to determine processing approach
            file_size = len(file_content)

            # Define size thresholds - adjust as needed
            LARGE_DOCUMENT_THRESHOLD = 5 * 1024 * 1024  # 5MB

            # For large documents, use partial processing
            if file_size > LARGE_DOCUMENT_THRESHOLD:
                logger.info(f"Large document detected ({file_size} bytes). Using partial processing.")
                return self.process_large_document(
                    file_content=file_content,
                    file_name=file_name,
                    doc_id=doc_id,
                    doc_metadata=doc_metadata,
                    enrich=enrich,
                    detect_language=detect_language
                )

            # Original processing for smaller documents
            # Generate document ID if not provided
            if not doc_id:
                doc_id = f"doc_{uuid.uuid4()}"

            # Initialize metadata if not provided
            if not doc_metadata:
                doc_metadata = {}

            # Add processing metadata
            doc_metadata["processing_time"] = datetime.now().isoformat()
            # Specify which version of Unstructured API was used
            doc_metadata["processor"] = "unstructured-api-cloud" if self.is_cloud else "unstructured-api-local"
            doc_metadata["unstructured_api_url"] = self.unstructured_api_url

            # Call Unstructured API to extract elements
            elements = self._call_unstructured_api(file_content, file_name)

            # Detect document language if requested
            if detect_language:
                # Get text sample from elements
                sample_text = ""
                for element in elements[:10]:  # Use first 10 elements as sample
                    if "text" in element:
                        sample_text += element["text"] + "\n\n"
                        if len(sample_text) > 1000:  # Limit sample size
                            break

                if sample_text:
                    # Detect language
                    language_detector = LanguageDetector()
                    language_info = language_detector.detect_language(sample_text)

                    # Add language info to metadata
                    doc_metadata["language"] = language_info["language_code"]
                    doc_metadata["language_name"] = language_info["language_name"]
                    doc_metadata["language_confidence"] = language_info["confidence"]

                    logger.info(f"Detected document language: {language_info['language_name']} ({language_info['language_code']}) with confidence {language_info['confidence']:.2f}")

            # Clean metadata before storing
            cleaned_metadata = doc_metadata.copy() if doc_metadata else {}

            # Parse any existing metadata from the document
            if doc_metadata and "metadata" in doc_metadata and doc_metadata["metadata"]:
                try:
                    parsed_metadata = self.metadata_parser.parse(doc_metadata["metadata"])
                    cleaned_metadata.update(parsed_metadata)
                except Exception as e:
                    logger.warning(f"Error parsing document metadata: {str(e)}")

            # Use cleaned metadata instead of original
            doc_metadata = cleaned_metadata

            # Store document and chunks in Neo4j - now returns statistics
            result = self._store_document_in_neo4j(
                doc_id=doc_id,
                file_name=file_name,
                elements=elements,
                metadata=doc_metadata
            )

            # Get the stored document ID
            stored_doc_id = result["doc_id"]

            # Initialize statistics from document storage
            stats = {
                "chunk_count": result["chunk_count"],
                "section_count": result["section_count"],
                "entity_count": 0,
                "concept_count": 0,
                "requirement_count": 0,
                "has_regulatory_content": False
            }

            # Apply enrichment if requested
            should_enrich = enrich if enrich is not None else getattr(self, 'use_enrichment', False)

            if should_enrich and hasattr(self, 'enrichment_pipeline') and self.enrichment_pipeline:
                try:
                    logger.info(f"Enriching document {stored_doc_id}")
                    enrichment_result = self.enrichment_pipeline.enrich_document(stored_doc_id)

                    if enrichment_result["status"] == "success":
                        logger.info(f"Successfully enriched document {stored_doc_id}")
                        # Update statistics with enrichment results
                        stats.update({
                            "entity_count": enrichment_result.get("entities", 0),
                            "concept_count": enrichment_result.get("concepts", 0),
                            "requirement_count": enrichment_result.get("requirements", 0),
                            "has_regulatory_content": enrichment_result.get("has_regulatory_content", False)
                        })
                    else:
                        logger.warning(f"Enrichment failed: {enrichment_result['message']}")
                except Exception as e:
                    logger.error(f"Error during document enrichment: {str(e)}")
                    # Continue anyway, as the document is already processed

            # Fix circular import issue by moving the RAG indexing to a separate function
            self._index_document_in_rag_system(stored_doc_id)

            # Return the document ID and statistics
            return {
                "doc_id": stored_doc_id,
                **stats,
                "language": doc_metadata.get("language"),
                "language_name": doc_metadata.get("language_name")
            }

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
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
        logger.info(f"Deleting document {doc_id} from Neo4j")

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

                # Get deletion statistics before deleting
                stats_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    OPTIONAL MATCH (d)-[r]->()
                    WITH d, count(r) as rel_count
                    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                    RETURN rel_count, count(c) as chunk_count
                    """,
                    doc_id=doc_id
                )

                stats = stats_result.single()
                chunk_count = stats["chunk_count"]
                relationship_count = stats["rel_count"]

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

                    # Delete relationships
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        OPTIONAL MATCH (d)-[r]-()
                        DELETE r
                        """,
                        doc_id=doc_id
                    )

                    # Delete entities with empty doc_names
                    if "Entity" in node_labels:
                        tx.run(
                            """
                            MATCH (e:Entity)
                            WHERE e.doc_names IS NULL OR size(e.doc_names) = 0
                            DETACH DELETE e
                            """,
                            doc_id=doc_id
                        )

                    # Delete concepts with empty doc_names
                    if "Concept" in node_labels:
                        tx.run(
                            """
                            MATCH (c:Concept)
                            WHERE c.doc_names IS NULL OR size(c.doc_names) = 0
                            DETACH DELETE c
                            """,
                            doc_id=doc_id
                        )

                    # Delete legislation with empty doc_names
                    if "Legislation" in node_labels:
                        tx.run(
                            """
                            MATCH (l:Legislation)
                            WHERE l.doc_names IS NULL OR size(l.doc_names) = 0
                            DETACH DELETE l
                            """,
                            doc_id=doc_id
                        )

                    # Delete requirements with empty doc_names
                    if "Requirement" in node_labels:
                        tx.run(
                            """
                            MATCH (r:Requirement)
                            WHERE r.doc_names IS NULL OR size(r.doc_names) = 0
                            DETACH DELETE r
                            """,
                            doc_id=doc_id
                        )

                    # Delete deadlines with empty doc_names
                    if "Deadline" in node_labels:
                        tx.run(
                            """
                            MATCH (dl:Deadline)
                            WHERE dl.doc_names IS NULL OR size(dl.doc_names) = 0
                            DETACH DELETE dl
                            """,
                            doc_id=doc_id
                        )

                    # Delete all chunks
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                        DETACH DELETE c
                        """,
                        doc_id=doc_id
                    )

                    # Delete the document itself
                    tx.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        DETACH DELETE d
                        """,
                        doc_id=doc_id
                    )

                    # Delete all orghaned chunks remain
                    tx.run(
                        """
                        MATCH (c:Chunk)
                        WHERE NOT EXISTS((c)<-[:CONTAINS]-())
                        DETACH DELETE c
                        """
                    )

                    # Commit the transaction
                    tx.commit()

                    deletion_stats["document_deleted"] = True
                    deletion_stats["chunks_deleted"] = chunk_count
                    deletion_stats["relationships_deleted"] = relationship_count
                    deletion_stats["entities_deleted"] = entity_count
                    deletion_stats["concepts_deleted"] = concept_count
                    deletion_stats["legislation_deleted"] = legislation_count
                    deletion_stats["requirements_deleted"] = requirement_count
                    deletion_stats["deadlines_deleted"] = deadline_count

                    logger.info(f"Document {doc_id} successfully deleted with {chunk_count} chunks, {relationship_count} relationships")
                    logger.info(f"Deleted {entity_count} entities, {concept_count} concepts, {legislation_count} legislation references, {requirement_count} requirements, and {deadline_count} deadlines")

                    return {
                        "status": "success",
                        "message": f"Document {doc_id} deleted successfully",
                        "stats": deletion_stats
                    }

                except Exception as tx_error:
                    # Roll back the transaction on error
                    tx.rollback()
                    logger.error(f"Transaction error: {str(tx_error)}")
                    raise

        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from Neo4j: {str(e)}")
            return {
                "status": "error",
                "message": f"Error deleting document: {str(e)}",
                "stats": deletion_stats
            }

    def process_large_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        enrich: Optional[bool] = None,
        detect_language: bool = True,
        max_size_per_chunk: int = 5 * 1024 * 1024  # 5MB chunks
    ) -> Dict[str, Any]:
        """
        Process a large document by splitting it into smaller chunks.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID (generated if not provided)
            doc_metadata: Optional document metadata
            enrich: Whether to apply enrichment
            detect_language: Whether to detect document language
            max_size_per_chunk: Maximum size of each chunk in bytes

        Returns:
            Dictionary with document ID and processing details
        """
        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4()}"

        # Initialize metadata if not provided
        if not doc_metadata:
            doc_metadata = {}

        # Add processing metadata
        doc_metadata["processing_time"] = datetime.now().isoformat()
        doc_metadata["processor"] = "unstructured-api-large-doc"
        doc_metadata["total_size"] = len(file_content)
        doc_metadata["chunked_processing"] = True

        file_size = len(file_content)
        logger.info(f"Large document processing: {file_name} ({file_size} bytes) with ID {doc_id}")

        # Determine file type to choose appropriate splitting strategy
        file_ext = os.path.splitext(file_name)[1].lower()

        # Process based on file type
        if file_ext in ['.txt', '.csv', '.md', '.json', '.xml', '.html']:
            # For text-based files, split by character count
            return self._process_large_text_document(
                file_content, file_name, doc_id, doc_metadata, enrich, detect_language
            )
        elif file_ext in ['.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.xls']:
            # For binary files, split using the unstructured API's partition capabilities
            return self._process_large_binary_document(
                file_content, file_name, doc_id, doc_metadata, enrich, detect_language, max_size_per_chunk
            )
        else:
            # For unknown file types, try general approach
            logger.info(f"Unknown file type {file_ext}, trying general large document processing")
            return self._process_large_binary_document(
                file_content, file_name, doc_id, doc_metadata, enrich, detect_language, max_size_per_chunk
            )

    def _process_large_text_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: str,
        doc_metadata: Dict[str, Any],
        enrich: Optional[bool],
        detect_language: bool
    ) -> Dict[str, Any]:
        """Process a large text document by splitting it into manageable chunks."""
        try:
            # Convert bytes to string for text-based files
            text_content = file_content.decode('utf-8', errors='replace')

            # Document statistics
            total_length = len(text_content)
            max_chunk_size = 500000  # Characters per chunk (well under spaCy's 1M limit)

            # Split into chunks
            text_chunks = []
            for i in range(0, total_length, max_chunk_size):
                end = min(i + max_chunk_size, total_length)
                # Try to find a good break point (paragraph or sentence)
                if end < total_length:
                    # Look for paragraph break
                    break_pos = text_content.rfind('\n\n', i, end)
                    if break_pos == -1 or break_pos < i + max_chunk_size // 2:
                        # Try sentence break
                        for sep in ['. ', '! ', '? ', '\n']:
                            break_pos = text_content.rfind(sep, i, end)
                            if break_pos != -1 and break_pos > i + max_chunk_size // 2:
                                break_pos += len(sep)
                                break

                    if break_pos != -1 and break_pos > i + max_chunk_size // 2:
                        end = break_pos

                text_chunks.append(text_content[i:end])

            logger.info(f"Split large text document into {len(text_chunks)} chunks")

            # Create initial document in Neo4j
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d += $properties
                    RETURN d
                    """,
                    doc_id=doc_id,
                    properties=doc_metadata
                )

            # Process each chunk
            chunk_results = []
            all_elements = []
            chunk_index = 0

            # Detect language from first chunk if requested
            language_detected = False
            detected_language = None

            for i, chunk in enumerate(text_chunks):
                logger.info(f"Processing text chunk {i+1}/{len(text_chunks)}")

                # Create a temporary file for this chunk
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_name)[1], delete=False) as temp_file:
                    temp_file.write(chunk.encode('utf-8'))
                    temp_file_path = temp_file.name

                try:
                    # Process with Unstructured API
                    with open(temp_file_path, "rb") as f:
                        elements = self._call_unstructured_api(f.read(), f"{file_name}.part{i+1}")

                    # Detect language from first chunk if needed
                    if detect_language and not language_detected and elements:
                        sample_text = ""
                        for element in elements[:10]:
                            if "text" in element:
                                sample_text += element["text"] + "\n\n"
                                if len(sample_text) > 1000:
                                    break

                        if sample_text:
                            language_detector = LanguageDetector()
                            language_info = language_detector.detect_language(sample_text)
                            detected_language = language_info["language_code"]

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
                                    language=language_info["language_code"],
                                    language_name=language_info["language_name"],
                                    confidence=language_info["confidence"]
                                )

                            language_detected = True

                    # Add chunk offset to make section indices unique
                    for element in elements:
                        if "metadata" not in element:
                            element["metadata"] = {}
                        element["metadata"]["chunk_part"] = i + 1

                    all_elements.extend(elements)
                    chunk_index += len(elements)
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            # Store all elements in Neo4j
            result = self._store_document_in_neo4j(
                doc_id=doc_id,
                file_name=file_name,
                elements=all_elements,
                metadata=doc_metadata
            )

            # Apply enrichment if requested
            should_enrich = enrich if enrich is not None else getattr(self, 'use_enrichment', False)
            entity_count = 0
            concept_count = 0
            requirement_count = 0
            has_regulatory_content = False

            if should_enrich and hasattr(self, 'enrichment_pipeline') and self.enrichment_pipeline:
                # Process enrichment in smaller batches
                max_chunks_per_batch = 10  # Process 10 chunks at a time for enrichment

                # Calculate total number of batches
                total_batches = (len(text_chunks) + max_chunks_per_batch - 1) // max_chunks_per_batch

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * max_chunks_per_batch
                    end_idx = min((batch_idx + 1) * max_chunks_per_batch, len(text_chunks))

                    batch_chunks = text_chunks[start_idx:end_idx]
                    batch_text = "\n\n".join(batch_chunks)

                    # If batch is still too large, skip enrichment
                    if len(batch_text) > 900000:  # Stay well under spaCy's limit
                        logger.warning(f"Batch {batch_idx+1} too large for enrichment, skipping")
                        continue

                    logger.info(f"Enriching batch {batch_idx+1}/{total_batches}")

                    try:
                        batch_context = {
                            "doc_id": doc_id,
                            "batch": batch_idx + 1,
                            "total_batches": total_batches
                        }

                        batch_result = self.enrichment_pipeline.enrich_text(
                            batch_text, context=batch_context
                        )

                        # Update stats
                        entity_count += batch_result["stats"]["entity_count"]
                        concept_count += batch_result["stats"]["concept_count"]
                        requirement_count += batch_result["stats"]["requirement_count"]

                        if batch_result["stats"]["has_regulatory_content"]:
                            has_regulatory_content = True

                        # Store enrichment results
                        self.enrichment_pipeline._store_entities(doc_id, batch_result["entities"])
                        self.enrichment_pipeline._store_concepts(doc_id, batch_result["concepts"])
                        self.enrichment_pipeline._store_regulatory_items(doc_id, batch_result["regulatory_analysis"])

                    except Exception as e:
                        logger.error(f"Error during batch enrichment: {str(e)}")

                # Update document with final enrichment statistics
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        SET d.entity_count = $entity_count,
                            d.concept_count = $concept_count,
                            d.requirement_count = $requirement_count,
                            d.has_regulatory_content = $has_regulatory_content,
                            d.enriched = true,
                            d.enriched_at = datetime()
                        """,
                        doc_id=doc_id,
                        entity_count=entity_count,
                        concept_count=concept_count,
                        requirement_count=requirement_count,
                        has_regulatory_content=has_regulatory_content
                    )

            # Fix circular import issue by moving the RAG indexing to a separate function
            self._index_document_in_rag_system(doc_id)

            return {
                "doc_id": doc_id,
                "chunk_count": result["chunk_count"],
                "section_count": result["section_count"],
                "entity_count": entity_count,
                "concept_count": concept_count,
                "requirement_count": requirement_count,
                "has_regulatory_content": has_regulatory_content,
                "language": detected_language
            }

        except Exception as e:
            logger.error(f"Error processing large text document: {str(e)}")
            raise

    def _process_large_binary_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: str,
        doc_metadata: Dict[str, Any],
        enrich: Optional[bool],
        detect_language: bool,
        max_size_per_chunk: int
    ) -> Dict[str, Any]:
        """Process a large binary document (PDF, DOCX, etc.)."""
        try:
            # Save the file temporarily
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_name)[1], delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            # Create initial document in Neo4j
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d += $properties
                    RETURN d
                    """,
                    doc_id=doc_id,
                    properties=doc_metadata
                )

            # For binary files like PDFs, we'll process page by page or in small batches
            # We'll use the unstructured API differently
            file_size = len(file_content)
            file_ext = os.path.splitext(file_name)[1].lower()

            all_elements = []

            # Approach depends on file type
            try:
                if file_ext == '.pdf':
                    # For PDFs, process by page ranges
                    # First, get total page count using a fast strategy
                    headers = {
                        "Accept": "application/json",
                        "unstructured-api-key": self.unstructured_api_key
                    }

                    with open(temp_file_path, "rb") as f:
                        files = {"files": (file_name, f)}
                        data = {
                            "strategy": "fast",
                            "ocr_enabled": "false",
                            "include_page_breaks": "true",
                            "include_metadata": "true"
                        }

                        response = requests.post(
                            self.unstructured_api_url,
                            headers=headers,
                            files=files,
                            data=data
                        )

                        if response.status_code != 200:
                            raise Exception(f"Failed to get PDF metadata: {response.status_code}")

                        elements = response.json()

                        # Try to determine page count
                        page_count = 0
                        for element in elements:
                            if "metadata" in element and "page_number" in element["metadata"]:
                                page_num = element["metadata"]["page_number"]
                                if isinstance(page_num, int) and page_num > page_count:
                                    page_count = page_num

                    if page_count == 0:
                        # Couldn't determine page count, use a default
                        page_count = 20
                        logger.warning(f"Couldn't determine page count, assuming {page_count} pages")
                    else:
                        logger.info(f"Detected {page_count} pages in PDF")

                    # Now process in batches of pages
                    batch_size = 10  # Process 10 pages at a time
                    for start_page in range(1, page_count + 1, batch_size):
                        end_page = min(start_page + batch_size - 1, page_count)

                        logger.info(f"Processing PDF pages {start_page}-{end_page} of {page_count}")

                        with open(temp_file_path, "rb") as f:
                            files = {"files": (file_name, f)}
                            data = {
                                "strategy": "hi_res",
                                "ocr_enabled": "true",
                                "start_page": str(start_page),
                                "end_page": str(end_page)
                            }

                            response = requests.post(
                                self.unstructured_api_url,
                                headers=headers,
                                files=files,
                                data=data
                            )

                            if response.status_code != 200:
                                logger.error(f"Error processing PDF pages {start_page}-{end_page}: {response.status_code}")
                                continue

                            batch_elements = response.json()
                            all_elements.extend(batch_elements)
                else:
                    # For other binary files, use the standard approach but with safety limits
                    elements = self._call_unstructured_api(file_content, file_name)
                    all_elements = elements
            except Exception as e:
                logger.error(f"Error processing file with Unstructured API: {str(e)}")
                # Create a placeholder element
                all_elements = [{
                    "type": "text",
                    "text": f"Error processing document: {str(e)}",
                    "metadata": {
                        "filename": file_name,
                        "filetype": file_ext,
                        "error": str(e)
                    }
                }]

            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            # Detect language from first few elements if requested
            if detect_language and all_elements:
                sample_text = ""
                for element in all_elements[:10]:
                    if "text" in element:
                        sample_text += element["text"] + "\n\n"
                        if len(sample_text) > 1000:
                            break

                if sample_text:
                    language_detector = LanguageDetector()
                    language_info = language_detector.detect_language(sample_text)
                    detected_language = language_info["language_code"]

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
                            language=language_info["language_code"],
                            language_name=language_info["language_name"],
                            confidence=language_info["confidence"]
                        )
                else:
                    detected_language = None
            else:
                detected_language = None

            # Store document elements in Neo4j
            result = self._store_document_in_neo4j(
                doc_id=doc_id,
                file_name=file_name,
                elements=all_elements,
                metadata=doc_metadata
            )

            # Apply enrichment if requested
            should_enrich = enrich if enrich is not None else getattr(self, 'use_enrichment', False)
            entity_count = 0
            concept_count = 0
            requirement_count = 0
            has_regulatory_content = False

            if should_enrich and hasattr(self, 'enrichment_pipeline') and self.enrichment_pipeline:
                logger.info(f"Running enrichment for document {doc_id}")

                # Get chunks of text for enrichment
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                        RETURN c.chunk_id AS chunk_id, c.text AS text, c.section AS section
                        ORDER BY c.index
                        """,
                        doc_id=doc_id
                    )

                    chunks = [{"chunk_id": record["chunk_id"], "text": record["text"], "section": record["section"]}
                            for record in result]

                # Process enrichment in smaller batches
                max_chunks_per_batch = 10
                total_batches = (len(chunks) + max_chunks_per_batch - 1) // max_chunks_per_batch

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * max_chunks_per_batch
                    end_idx = min((batch_idx + 1) * max_chunks_per_batch, len(chunks))

                    batch_chunks = chunks[start_idx:end_idx]
                    batch_text = "\n\n".join([chunk["text"] for chunk in batch_chunks])

                    # If batch is still too large, skip enrichment
                    if len(batch_text) > 900000:  # Stay well under spaCy's limit
                        logger.warning(f"Batch {batch_idx+1} too large for enrichment, skipping")
                        continue

                    logger.info(f"Enriching batch {batch_idx+1}/{total_batches}")

                    try:
                        batch_context = {
                            "doc_id": doc_id,
                            "batch": batch_idx + 1,
                            "total_batches": total_batches
                        }

                        batch_result = self.enrichment_pipeline.enrich_text(
                            batch_text, context=batch_context
                        )

                        # Update stats
                        entity_count += batch_result["stats"]["entity_count"]
                        concept_count += batch_result["stats"]["concept_count"]
                        requirement_count += batch_result["stats"]["requirement_count"]

                        if batch_result["stats"]["has_regulatory_content"]:
                            has_regulatory_content = True

                        # Store enrichment results
                        self.enrichment_pipeline._store_entities(doc_id, batch_result["entities"])
                        self.enrichment_pipeline._store_concepts(doc_id, batch_result["concepts"])
                        self.enrichment_pipeline._store_regulatory_items(doc_id, batch_result["regulatory_analysis"])

                    except Exception as e:
                        logger.error(f"Error during batch enrichment: {str(e)}")

                # Update document with final enrichment statistics
                with self.driver.session() as session:
                    session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        SET d.entity_count = $entity_count,
                            d.concept_count = $concept_count,
                            d.requirement_count = $requirement_count,
                            d.has_regulatory_content = $has_regulatory_content,
                            d.enriched = true,
                            d.enriched_at = datetime()
                        """,
                        doc_id=doc_id,
                        entity_count=entity_count,
                        concept_count=concept_count,
                        requirement_count=requirement_count,
                        has_regulatory_content=has_regulatory_content
                    )

            # Fix circular import issue by moving the RAG indexing to a separate function
            self._index_document_in_rag_system(doc_id)

            return {
                "doc_id": doc_id,
                "chunk_count": result["chunk_count"],
                "section_count": result["section_count"],
                "entity_count": entity_count,
                "concept_count": concept_count,
                "requirement_count": requirement_count,
                "has_regulatory_content": has_regulatory_content,
                "language": detected_language
            }

        except Exception as e:
            logger.error(f"Error processing large binary document: {str(e)}")
            raise

    def _index_document_in_rag_system(self, doc_id: str) -> bool:
        """
        Index a document in the RAG system.
        This is a helper function to avoid circular imports.

        Args:
            doc_id: Document ID to index

        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            # Use dynamic import to avoid circular import issues
            import importlib
            rag_module = importlib.import_module("llamaIndex_rag.rag")
            RAGSystem = getattr(rag_module, "RAGSystem")

            # Initialize RAG system with Neo4j credentials
            rag_system = RAGSystem(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
                hybrid_search=True,  # Enable hybrid search
                vector_weight=0.7,   # Set vector search weight
                keyword_weight=0.3   # Set keyword search weight
            )

            # Index the document
            indexed = rag_system.index_document(doc_id)

            if indexed:
                logger.info(f"Document {doc_id} indexed in RAG system with hybrid search")
            else:
                logger.warning(f"Failed to index document {doc_id} in RAG system")

            # Clean up
            rag_system.close()
            return indexed

        except Exception as e:
            logger.error(f"Error indexing document in RAG system: {str(e)}")
            # Continue without failing
            return False

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Document parser Neo4j connection closed")
