"""
Base parser interface for document parsing in RegulAite.
This defines the common interface that all document parsers must implement.
"""

import abc
from typing import Dict, List, Any, Optional, BinaryIO, Callable
from datetime import datetime
from enum import Enum, auto


class ParserType(str, Enum):
    """Enum for the different types of document parsers available."""
    UNSTRUCTURED = "unstructured"
    UNSTRUCTURED_CLOUD = "unstructured_cloud"
    DOCTLY = "doctly"
    LLAMAPARSE = "llamaparse"


class BaseParser(abc.ABC):
    """Abstract base class for document parsers."""

    @abc.abstractmethod
    def process_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document and extract text, structure, and metadata.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID
            doc_metadata: Optional document metadata
            **kwargs: Additional parser-specific arguments

        Returns:
            Dict with document ID and processing details
        """
        pass

    @abc.abstractmethod
    def process_large_document(
        self,
        file_content: bytes,
        file_name: str,
        doc_id: Optional[str] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a large document by chunking and extract text, structure, and metadata.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            doc_id: Optional document ID
            doc_metadata: Optional document metadata
            **kwargs: Additional parser-specific arguments

        Returns:
            Dict with document ID and processing details
        """
        pass

    @abc.abstractmethod
    def delete_document(self, doc_id: str, purge_orphans: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Delete a document from the storage.

        Args:
            doc_id: Document ID to delete
            purge_orphans: Whether to purge orphaned nodes after deletion
            **kwargs: Additional parser-specific arguments

        Returns:
            Dict with deletion status
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Close any open resources.
        """
        pass

    @staticmethod
    def get_parser(
        parser_type: ParserType,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        **kwargs
    ) -> 'BaseParser':
        """
        Factory method to get a parser instance based on the type.

        Args:
            parser_type: Type of parser to create
            neo4j_uri: URI for Neo4j database
            neo4j_user: Username for Neo4j
            neo4j_password: Password for Neo4j
            **kwargs: Additional parser-specific initialization parameters

        Returns:
            A parser instance
        """
        from .document_parser import DocumentParser
        from .doctly_parser import DoctlyParser
        from .llamaparse_parser import LlamaParseParser

        if parser_type == ParserType.UNSTRUCTURED:
            return DocumentParser(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                **kwargs
            )
        elif parser_type == ParserType.UNSTRUCTURED_CLOUD:
            # For cloud version, we use the same parser but with different API URL
            # The API key and dedicated cloud URL must be provided in kwargs or environment variables
            return DocumentParser(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                is_cloud=True,
                **kwargs
            )
        elif parser_type == ParserType.DOCTLY:
            return DoctlyParser(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                **kwargs
            )
        elif parser_type == ParserType.LLAMAPARSE:
            return LlamaParseParser(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown parser type: {parser_type}")
