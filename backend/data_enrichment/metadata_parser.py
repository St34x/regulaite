# plugins/regul_aite/backend/data_enrichment/metadata_parser.py

import json
import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class MetadataParser:
    """
    Parser for extracting relevant fields from document metadata.
    Handles various document types and standardizes the output format.
    """

    def __init__(self, max_field_length: int = 500):
        """
        Initialize the metadata parser.

        Args:
            max_field_length: Maximum length for metadata field values
        """
        self.max_field_length = max_field_length

        # Common metadata fields to extract across document types
        self.common_fields = [
            "filename",
            "filetype",
            "page_count",
            "page_number",
            "page_name",
            "title",
            "author",
            "creation_date",
            "last_modified_date",
            "subject",
            "keywords",
            "languages"
        ]

        # Document type specific parsers
        self.document_type_parsers = {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": self._parse_excel_metadata,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._parse_word_metadata,
            "application/pdf": self._parse_pdf_metadata,
            "text/plain": self._parse_text_metadata,
            "text/html": self._parse_html_metadata,
            "application/json": self._parse_json_metadata
        }

    def parse(self, metadata: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse metadata and extract relevant fields.

        Args:
            metadata: Raw metadata as string or dictionary

        Returns:
            Dictionary of cleaned and standardized metadata
        """
        try:
            # Convert string to dictionary if needed
            if isinstance(metadata, str):
                metadata_dict = json.loads(metadata)
            else:
                metadata_dict = metadata

            # Get document type
            doc_type = metadata_dict.get("filetype", "")

            # Extract common fields first
            result = self._extract_common_fields(metadata_dict)

            # Apply document-specific parsing if available
            if doc_type in self.document_type_parsers:
                doc_specific = self.document_type_parsers[doc_type](metadata_dict)
                result.update(doc_specific)
            else:
                # For unknown document types, try a generic approach
                result.update(self._parse_generic_metadata(metadata_dict))

            # Add processing timestamp
            result["processed_at"] = datetime.now().isoformat()

            # Clean and validate all fields
            result = self._clean_metadata(result)

            return result

        except Exception as e:
            logger.error(f"Error parsing metadata: {str(e)}")
            # Return minimal metadata in case of error
            return {
                "error": f"Failed to parse metadata: {str(e)}",
                "processed_at": datetime.now().isoformat()
            }

    def _extract_common_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract common fields from metadata"""
        result = {}

        for field in self.common_fields:
            if field in metadata:
                result[field] = metadata[field]

        return result

    def _parse_excel_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata specific to Excel documents"""
        result = {}

        # For regulatory documents like SoA, extract control information
        if "text_as_html" in metadata:
            # Extract table structure if present
            table_content = self._extract_table_structure(metadata.get("text_as_html", ""))
            if table_content:
                result["table_structure"] = table_content

                # For SOA documents specifically, extract controls
                if "SOA" in metadata.get("page_name", "") or "Statement of Applicability" in metadata.get("page_name", ""):
                    controls = self._extract_control_information(table_content)
                    if controls:
                        result["controls"] = controls

        return result

    def _parse_word_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata specific to Word documents"""
        # Extract sections, headers, paragraphs count if available
        result = {}

        if "section_count" in metadata:
            result["section_count"] = metadata["section_count"]

        if "header_count" in metadata:
            result["header_count"] = metadata["header_count"]

        if "paragraph_count" in metadata:
            result["paragraph_count"] = metadata["paragraph_count"]

        # Extract document category if it appears to be a policy or procedure
        document_type = "unknown"
        title = metadata.get("title", "").lower()

        if any(term in title for term in ["policy", "procedure", "standard", "guideline"]):
            document_type = "governance_document"
        elif any(term in title for term in ["report", "assessment", "analysis"]):
            document_type = "report"

        result["document_category"] = document_type

        return result

    def _parse_pdf_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata specific to PDF documents"""
        result = {}

        # Extract PDF-specific fields if available
        pdf_fields = ["producer", "creator", "encrypted", "page_count"]

        for field in pdf_fields:
            if field in metadata:
                result[field] = metadata[field]

        return result

    def _parse_text_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata specific to text documents"""
        # For plain text, there's usually minimal metadata
        return {}

    def _parse_html_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata specific to HTML documents"""
        result = {}

        # Extract HTML specific fields
        if "meta_tags" in metadata:
            result["meta_tags"] = metadata["meta_tags"]

        return result

    def _parse_json_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata specific to JSON documents"""
        result = {}

        # For JSON, extract schema information if available
        if "schema" in metadata:
            result["schema"] = metadata["schema"]

        return result

    def _parse_generic_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generic metadata parsing for unknown document types"""
        result = {}

        # Include useful-looking fields
        for key, value in metadata.items():
            # Skip very large nested objects or arrays
            if isinstance(value, (dict, list)) and len(str(value)) > 1000:
                continue

            # Skip very long string values
            if isinstance(value, str) and len(value) > self.max_field_length:
                continue

            # Include potentially useful fields
            if key not in self.common_fields and not key.startswith("_"):
                result[key] = value

        return result

    def _extract_table_structure(self, html_content: str) -> Optional[Dict[str, Any]]:
        """Extract table structure from HTML content"""
        if not html_content or "<table>" not in html_content:
            return None

        # Simple extraction of headers and row count
        result = {}

        # Count rows
        row_count = html_content.count("<tr>")
        result["row_count"] = row_count

        # Extract headers
        headers = []
        header_match = re.search(r"<tr>(.+?)</tr>", html_content)
        if header_match:
            header_row = header_match.group(1)
            header_cells = re.findall(r"<td>(.+?)</td>", header_row)
            headers = [h for h in header_cells if h]

        result["headers"] = headers

        return result

    def _extract_control_information(self, table_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Extract control information from table data"""
        if not table_data or "headers" not in table_data:
            return None

        # For SoA documents, extract control ID and status
        # This is a simplified implementation - would need to be adapted for specific SoA formats
        headers = table_data.get("headers", [])

        # Check if this looks like a control table
        control_headers = ["Control", "Topic", "Applicable", "Status"]
        if not any(header in headers for header in control_headers):
            return None

        # Return simplified control information
        return {
            "has_control_data": True,
            "control_count": table_data.get("row_count", 0) - 1  # Subtract header row
        }

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate metadata values"""
        cleaned = {}

        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Truncate long string values
            if isinstance(value, str) and len(value) > self.max_field_length:
                cleaned[key] = value[:self.max_field_length] + "..."
            # Handle nested dicts recursively
            elif isinstance(value, dict):
                cleaned[key] = self._clean_metadata(value)
            # Handle lists of dicts recursively
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                cleaned[key] = [self._clean_metadata(item) if isinstance(item, dict) else item for item in value]
            # For other list types, keep if reasonably sized
            elif isinstance(value, list) and len(str(value)) <= self.max_field_length * 2:
                cleaned[key] = value
            # Keep simple values
            elif not isinstance(value, (dict, list)):
                cleaned[key] = value

        return cleaned
