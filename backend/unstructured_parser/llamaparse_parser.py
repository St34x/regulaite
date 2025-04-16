# plugins/regul_aite/backend/unstructured_parser/llamaparse_parser.py
"""
LlamaParse API document parser for RegulAite.
This parser uses the LlamaParse API to extract text, structure, and metadata from documents.
"""

import os
import requests
import logging
import json
import uuid
import tempfile
from typing import Dict, List, Any, Optional, BinaryIO
from datetime import datetime
from neo4j import GraphDatabase

from .base_parser import BaseParser
from data_enrichment.language_detector import LanguageDetector
from data_enrichment.enrichment_pipeline import EnrichmentPipeline
from data_enrichment.metadata_parser import MetadataParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LlamaParseParser(BaseParser):
    """
    Parser for documents using LlamaParse API.
    Extracts text and metadata from documents and stores in Neo4j.
    """

    def __init__(
          self,
          neo4j_uri: str,
          neo4j_user: str,
          neo4j_password: str,
          llamaparse_api_url: Optional[str] = None,
          llamaparse_api_key: Optional[str] = None,
          chunk_size: int = 1000,
          chunk_overlap: int = 200,
          use_enrichment: bool = True,
          extract_tables: bool = True,
          extract_metadata: bool = True
      ):
        """
        Initialize the LlamaParse document parser.

        Args:
            neo4j_uri: URI for Neo4j database
            neo4j_user: Username for Neo4j
            neo4j_password: Password for Neo4j
            llamaparse_api_url: URL for LlamaParse API (defaults to env var)
            llamaparse_api_key: API key for LlamaParse API (defaults to env var)
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            use_enrichment: Whether to enrich documents by default
            extract_tables: Whether to extract tables from documents
            extract_metadata: Whether to extract detailed metadata
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Get LlamaParse API credentials from environment if not provided
        self.llamaparse_api_url = llamaparse_api_url or os.getenv(
            "LLAMAPARSE_API_URL", "https://api.llamaindex.ai/v1/parsing"
        )
        self.llamaparse_api_key = llamaparse_api_key or os.getenv(
            "LLAMAPARSE_API_KEY", ""
        )

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Extraction options
        self.extract_tables = extract_tables
        self.extract_metadata = extract_metadata

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
            logger.info(f"LlamaParse parser connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def _call_llamaparse_api(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """
        Call the LlamaParse API to extract text and structure from a file.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file

        Returns:
            Dict containing parsed document elements
        """
        logger.info(f"Calling LlamaParse API for document: {file_name}")

        # Create a temporary file to send to the API
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            headers = {
                "Accept": "application/json",
                "X-API-Key": self.llamaparse_api_key,
                "Content-Type": "application/json"
            }

            # LlamaParse API typically expects a base64 encoded file
            import base64
            encoded_file = base64.b64encode(file_content).decode('utf-8')

            # Prepare the payload
            payload = {
                "file": encoded_file,
                "file_name": file_name,
                "parsing_type": "full",  # or "simple" for faster, less detailed parsing
                "result_type": "markdown" if not self.extract_tables else "elements"
            }

            logger.info(f"Calling LlamaParse API with parsing_type: {payload['parsing_type']}")

            response = requests.post(
                self.llamaparse_api_url,
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"LlamaParse API error: {response.status_code} - {response.text}")
                raise Exception(f"LlamaParse API error: {response.status_code}")

            # Parse the response
            result = response.json()
            logger.info(f"Successfully parsed document with LlamaParse")

            # Convert LlamaParse's response to our internal format
            elements = self._convert_llamaparse_response(result)

            return elements

        except Exception as e:
            logger.error(f"Error calling LlamaParse API: {str(e)}")
            raise
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _convert_llamaparse_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert LlamaParse API response to our internal format (similar to Unstructured API).

        Args:
            response: The LlamaParse API response

        Returns:
            List of elements in our internal format
        """
        elements = []

        # Check if we got markdown or elements format
        if "elements" in response:
            # Process elements format (structured)
            for idx, element in enumerate(response.get("elements", [])):
                el_type = element.get("type", "text")

                if el_type == "table":
                    # Convert table to text representation
                    text = self._table_to_text(element.get("data", []))
                else:
                    text = element.get("text", "")

                if text:
                    element_obj = {
                        "type": el_type,
                        "text": text,
                        "metadata": {
                            "page_number": element.get("page_number", 1),
                            "section_idx": idx,
                        }
                    }

                    # Add additional metadata if available
                    if "metadata" in element:
                        element_obj["metadata"].update(element["metadata"])

                    elements.append(element_obj)

        elif "markdown" in response:
            # Process markdown format (simpler)
            # Split markdown into sections based on headers
            markdown_text = response.get("markdown", "")

            # Very simple markdown splitting - in a real implementation you'd want more sophisticated parsing
            import re
            sections = re.split(r'(#+\s+.*)', markdown_text)
            current_section = None

            for i, section in enumerate(sections):
                if section.strip():
                    if re.match(r'#+\s+.*', section):
                        # This is a header
                        current_section = section
                        elements.append({
                            "type": "Title",
                            "text": section.strip(),
                            "metadata": {
                                "section_idx": i,
                                "is_header": True,
                                "header_level": len(section) - len(section.lstrip('#'))
                            }
                        })
                    else:
                        # This is content
                        elements.append({
                            "type": "NarrativeText",
                            "text": section.strip(),
                            "metadata": {
                                "section_idx": i,
                                "parent_header": current_section if current_section else None
                            }
                        })

        # Add document metadata if available
        if "metadata" in response and elements:
            elements[0]["metadata"]["file_metadata"] = response["metadata"]

        return elements

    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert a table to a text representation."""
        if not table_data:
            return ""

        # Format as a simple markdown table
        result = []

        # Add header row
        result.append(" | ".join(str(cell) for cell in table_data[0]))

        # Add separator
        result.append(" | ".join(["---"] * len(table_data[0])))

        # Add data rows
        for row in table_data[1:]:
            result.append(" | ".join(str(cell) for cell in row))

        return "\n".join(result)

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
        Process a document using LlamaParse API and store in Neo4j.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID (generated if not provided)
            doc_metadata: Optional document metadata
            enrich: Whether to apply enrichment (overrides instance setting)
            detect_language: Whether to detect document language

        Returns:
            Dict with document ID and processing details
        """
        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4()}"

        # Initialize metadata if not provided
        if not doc_metadata:
            doc_metadata = {}

        # Add basic metadata
        doc_metadata.update({
            "file_name": file_name,
            "parser": "llamaparse",
            "processing_time": datetime.now().isoformat(),
        })

        try:
            # Call LlamaParse API to parse the document
            elements = self._call_llamaparse_api(file_content, file_name)

            # Determine whether to use enrichment
            should_enrich = self.use_enrichment if enrich is None else enrich

            # Detect language if requested
            if detect_language:
                lang_detector = LanguageDetector()
                all_text = " ".join([el.get("text", "") for el in elements if el.get("text")])
                detected_language = lang_detector.detect_language(all_text)
                doc_metadata["detected_language"] = detected_language
                doc_metadata["language_confidence"] = lang_detector.get_confidence()

            # Store document in Neo4j
            result = self._store_document_in_neo4j(doc_id, file_name, elements, doc_metadata)

            # Apply enrichment if requested
            if should_enrich and self.enrichment_pipeline:
                try:
                    self.enrichment_pipeline.enrich_document(doc_id)
                    result["enriched"] = True
                except Exception as e:
                    logger.error(f"Error enriching document: {str(e)}")
                    result["enriched"] = False

            return result

        except Exception as e:
            logger.error(f"Error processing document with LlamaParse: {str(e)}")
            return {
                "error": str(e),
                "doc_id": doc_id,
                "status": "failed"
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
        Process a large document using LlamaParse.

        LlamaParse handles large documents well, so we simply call the standard process_document.
        """
        return self.process_document(
            file_content=file_content,
            file_name=file_name,
            doc_id=doc_id,
            doc_metadata=doc_metadata,
            enrich=enrich,
            detect_language=detect_language
        )

    def _store_document_in_neo4j(
        self,
        doc_id: str,
        file_name: str,
        elements: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store document elements in Neo4j.
        This is a simplified version - in a real implementation, you would adapt this
        to match your Neo4j schema and how you want to store documents.
        """
        if not self.driver:
            raise Exception("Neo4j connection not established")

        try:
            with self.driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.file_name = $file_name,
                        d.created_at = datetime(),
                        d.metadata = $metadata
                    """,
                    doc_id=doc_id,
                    file_name=file_name,
                    metadata=metadata
                )

                # Create text nodes for each element with hierarchical structure
                previous_headers = {}
                current_parent_id = None

                for i, element in enumerate(elements):
                    if element.get("text"):
                        element_id = f"{doc_id}_text_{i}"
                        element_type = element.get("type", "text")

                        # Create the text node
                        session.run(
                            """
                            MATCH (d:Document {doc_id: $doc_id})
                            CREATE (t:Text {
                                text_id: $text_id,
                                content: $content,
                                type: $type,
                                metadata: $metadata
                            })
                            CREATE (d)-[:CONTAINS]->(t)
                            """,
                            doc_id=doc_id,
                            text_id=element_id,
                            content=element.get("text"),
                            type=element_type,
                            metadata=element.get("metadata", {})
                        )

                        # If this is a header, store it for later hierarchical connections
                        if element_type == "Title" or element.get("metadata", {}).get("is_header", False):
                            header_level = element.get("metadata", {}).get("header_level", 1)
                            previous_headers[header_level] = element_id

                            # Clear lower level headers (a new h2 means we're done with h3, h4, etc.)
                            for level in list(previous_headers.keys()):
                                if level > header_level:
                                    previous_headers.pop(level)

                            # Set this as current parent for subsequent content
                            current_parent_id = element_id

                            # Connect to parent header if there is one
                            parent_level = header_level - 1
                            if parent_level in previous_headers:
                                parent_id = previous_headers[parent_level]
                                session.run(
                                    """
                                    MATCH (parent:Text {text_id: $parent_id})
                                    MATCH (child:Text {text_id: $child_id})
                                    CREATE (parent)-[:CONTAINS]->(child)
                                    """,
                                    parent_id=parent_id,
                                    child_id=element_id
                                )

                        # If this is content and we have a current parent, connect them
                        elif current_parent_id and element_type != "Title":
                            session.run(
                                """
                                MATCH (parent:Text {text_id: $parent_id})
                                MATCH (child:Text {text_id: $child_id})
                                CREATE (parent)-[:CONTAINS]->(child)
                                """,
                                parent_id=current_parent_id,
                                child_id=element_id
                            )

            return {
                "doc_id": doc_id,
                "status": "success",
                "elements_count": len(elements),
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error storing document in Neo4j: {str(e)}")
            return {
                "doc_id": doc_id,
                "status": "error",
                "message": str(e)
            }

    def delete_document(self, doc_id: str, purge_orphans: bool = False) -> Dict[str, Any]:
        """
        Delete a document from Neo4j.

        Args:
            doc_id: Document ID to delete
            purge_orphans: Whether to purge orphaned nodes

        Returns:
            Dict with deletion status
        """
        if not self.driver:
            raise Exception("Neo4j connection not established")

        try:
            with self.driver.session() as session:
                # Delete the document and all its relationships and connected nodes
                if purge_orphans:
                    result = session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        OPTIONAL MATCH (d)-[:CONTAINS]->(t)
                        DETACH DELETE d, t
                        """,
                        doc_id=doc_id
                    )
                else:
                    result = session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        DETACH DELETE d
                        """,
                        doc_id=doc_id
                    )

                summary = result.consume()

                return {
                    "doc_id": doc_id,
                    "status": "success",
                    "nodes_deleted": summary.counters.nodes_deleted,
                    "relationships_deleted": summary.counters.relationships_deleted
                }

        except Exception as e:
            logger.error(f"Error deleting document from Neo4j: {str(e)}")
            return {
                "doc_id": doc_id,
                "status": "error",
                "message": str(e)
            }

    def close(self):
        """Close any open resources."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")
