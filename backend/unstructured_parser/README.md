# Document Parsers for RegulAite

This module provides document parsing capabilities for the RegulAite plugin, supporting multiple document parsing APIs:

- **Unstructured** (Local): Self-hosted Unstructured API for general document parsing
- **Unstructured Cloud**: Cloud-hosted version of Unstructured API for higher throughput
- **Doctly**: Specialized API for forms, contracts, and legal documents
- **LlamaParse**: Advanced document parsing with hierarchical structure

## Configuration

### Environment Variables

Set these environment variables to configure the parsers:

```bash
# Local Unstructured API (self-hosted)
UNSTRUCTURED_API_URL=http://unstructured:8000/general/v0/general
UNSTRUCTURED_API_KEY=your_optional_local_api_key

# Cloud Unstructured API
UNSTRUCTURED_CLOUD_API_URL=https://api.unstructured.io/general/v0/general
UNSTRUCTURED_CLOUD_API_KEY=your_cloud_api_key

# Doctly API
DOCTLY_API_URL=https://api.doctly.dev/v1/parse
DOCTLY_API_KEY=your_doctly_api_key

# LlamaParse API
LLAMAPARSE_API_URL=https://api.llamaindex.ai/v1/parsing
LLAMAPARSE_API_KEY=your_llamaparse_api_key

# Neo4j for document storage
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## API Usage

When uploading documents through the API, you can specify which parser to use:

```bash
# Using the Local Unstructured API (default)
curl -X POST "http://localhost:8000/api/tasks/documents/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "parser_type=unstructured"

# Using the Cloud Unstructured API
curl -X POST "http://localhost:8000/api/tasks/documents/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "parser_type=unstructured_cloud"

# Using Doctly
curl -X POST "http://localhost:8000/api/tasks/documents/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "parser_type=doctly"

# Using LlamaParse
curl -X POST "http://localhost:8000/api/tasks/documents/process" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "parser_type=llamaparse"
```

## Choosing Between Local and Cloud Unstructured

### Local Unstructured API
- **Pros**: Full control, no external dependencies, data privacy
- **Cons**: Requires more resources, limited by local hardware
- **Best for**: Development, smaller deployments, sensitive data

### Cloud Unstructured API
- **Pros**: Higher throughput, better scalability, enhanced features
- **Cons**: Requires API key, external dependency, potential costs
- **Best for**: Production, high-volume processing, complex documents

## Code Example

```python
from unstructured_parser.base_parser import BaseParser, ParserType

# Initialize a cloud parser
cloud_parser = BaseParser.get_parser(
    parser_type=ParserType.UNSTRUCTURED_CLOUD,
    neo4j_uri="bolt://neo4j:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    unstructured_api_key="your_cloud_api_key"
)

# Process a document
with open("document.pdf", "rb") as file:
    result = cloud_parser.process_document(
        file_content=file.read(),
        file_name="document.pdf",
        enrich=True
    )
    
print(f"Document processed: {result['doc_id']}")
```

## Parser Selection Guide

| Document Type | Recommended Parser |
|---------------|-------------------|
| General PDFs, Word docs | `unstructured` (local) |
| Scientific papers, complex docs | `unstructured_cloud` |
| Forms, receipts, contracts | `doctly` |
| Research papers, technical docs | `llamaparse` |

## Expanding Parser Support

To add support for additional document parsing APIs:

1. Create a new parser class that implements the `BaseParser` interface
2. Add the parser type to the `ParserType` enum
3. Update the `get_parser` factory method in `BaseParser`
4. Add the necessary configurations to your environment 