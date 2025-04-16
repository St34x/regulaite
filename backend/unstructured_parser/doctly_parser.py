# plugins/regul_aite/backend/unstructured_parser/doctly_parser.py
"""
Doctly API document parser for RegulAite.
This parser uses the Doctly API to extract text, structure, and metadata from documents.
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


class DoctlyParser(BaseParser):
    """
    Parser for documents using Doctly API.
    Extracts text and metadata from documents and stores in Neo4j.
    """

    def __init__(
          self,
          neo4j_uri: str,
          neo4j_user: str,
          neo4j_password: str,
          doctly_api_url: Optional[str] = None,
          doctly_api_key: Optional[str] = None,
          chunk_size: int = 1000,
          chunk_overlap: int = 200,
          use_enrichment: bool = True,
          extract_tables: bool = True,
          extract_metadata: bool = True,
          extract_images: bool = False
      ):
        """
        Initialize the Doctly document parser.

        Args:
            neo4j_uri: URI for Neo4j database
            neo4j_user: Username for Neo4j
            neo4j_password: Password for Neo4j
            doctly_api_url: URL for Doctly API (defaults to env var)
            doctly_api_key: API key for Doctly API (defaults to env var)
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            use_enrichment: Whether to enrich documents by default
            extract_tables: Whether to extract tables from documents
            extract_metadata: Whether to extract detailed metadata
            extract_images: Whether to extract and process images
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Get Doctly API credentials from environment if not provided
        self.doctly_api_url = doctly_api_url or os.getenv(
            "DOCTLY_API_URL", "https://api.doctly.dev/v1/parse"
        )
        self.doctly_api_key = doctly_api_key or os.getenv(
            "DOCTLY_API_KEY", ""
        )

        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
            logger.info(f"Doctly parser connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def _call_doctly_api(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """
        Call the Doctly API to extract text and structure from a file.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file

        Returns:
            Dict containing parsed document elements
        """
        logger.info(f"Calling Doctly API for document: {file_name}")

        # Create a temporary file to send to the API
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            headers = {
                "Accept": "application/json",
                "X-API-Key": self.doctly_api_key
            }

            with open(temp_file_path, "rb") as f:
                files = {"file": (file_name, f)}

                # Set parameters for Doctly API
                data = {
                    "extract_metadata": str(self.extract_metadata).lower(),
                    "extract_tables": str(self.extract_tables).lower(),
                    "extract_images": str(self.extract_images).lower()
                }

                logger.info(f"Calling Doctly API with parameters: {data}")

                response = requests.post(
                    self.doctly_api_url,
                    headers=headers,
                    files=files,
                    data=data
                )

            if response.status_code != 200:
                logger.error(f"Doctly API error: {response.status_code} - {response.text}")
                raise Exception(f"Doctly API error: {response.status_code}")

            # Parse the response
            result = response.json()
            logger.info(f"Successfully parsed document with Doctly")

            # Convert Doctly's response to our internal format
            elements = self._convert_doctly_response(result)

            return elements

        except Exception as e:
            logger.error(f"Error calling Doctly API: {str(e)}")
            raise
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _convert_doctly_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert Doctly API response to our internal format (similar to Unstructured API).

        Args:
            response: The Doctly API response

        Returns:
            List of elements in our internal format
        """
        elements = []

        # Process document sections based on Doctly's output format
        # This is a simplified example and should be adapted based on actual Doctly response format
        if "content" in response:
            for idx, section in enumerate(response.get("content", [])):
                element = {
                    "type": section.get("type", "NarrativeText"),
                    "text": section.get("text", ""),
                    "metadata": {
                        "page_number": section.get("page", 1),
                        "section_idx": idx,
                    }
                }

                # Add coordinates if available
                if "bbox" in section:
                    element["metadata"]["coordinates"] = section["bbox"]

                elements.append(element)

        # Process metadata
        doc_metadata = response.get("metadata", {})
        if doc_metadata and elements:
            # Attach document metadata to the first element
            elements[0]["metadata"].update({
                "file_metadata": doc_metadata
            })

        return elements

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
        Process a document using Doctly API and store in Neo4j.

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
            "parser": "doctly",
            "processing_time": datetime.now().isoformat(),
        })

        try:
            # Call Doctly API to parse the document
            elements = self._call_doctly_api(file_content, file_name)

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
            logger.error(f"Error processing document with Doctly: {str(e)}")
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
        Process a large document by chunking it into smaller pieces.

        This is a simplified implementation and should be enhanced based on
        Doctly's capabilities for handling large documents.
        """
        # For now, we'll just call the standard process_document method
        # In a real implementation, you might need to split large files
        # or use specific Doctly API capabilities for large documents
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

                # Create text nodes for each element
                for i, element in enumerate(elements):
                    if element.get("text"):
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
                            text_id=f"{doc_id}_text_{i}",
                            content=element.get("text"),
                            type=element.get("type", "text"),
                            metadata=element.get("metadata", {})
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
