# plugins/regul_aite/backend/tests/test_document_parser.py
import os
import pytest
from unstructured_parser.document_parser import DocumentParser

@pytest.fixture
def document_parser():
    # Use testing credentials from environment or mock
    neo4j_uri = os.getenv("TEST_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("TEST_NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("TEST_NEO4J_PASSWORD", "password")

    parser = DocumentParser(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    yield parser
    parser.close()

def test_document_parsing(document_parser):
    # Create a sample document
    sample_text = "This is a test document.\n\nIt has multiple paragraphs.\n\nThis is for testing."
    sample_bytes = sample_text.encode('utf-8')

    # Process the document
    doc_id = document_parser.process_document(
        file_content=sample_bytes,
        file_name="test_document.txt",
        doc_metadata={"test": True}
    )

    assert doc_id is not None
