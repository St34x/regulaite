# Direct Ingestion of Preparsed Documents in RegulAIte

This document explains how to ingest preparsed documents directly into Neo4j and Qdrant in the RegulAIte system, bypassing the standard document processing pipeline.

## Overview

The standard document processing pipeline in RegulAIte involves:
1. Document upload
2. Parsing using various APIs (Doctly, LlamaParse, Unstructured)
3. Text extraction and chunking
4. Vector embeddings generation
5. Storage in Neo4j and Qdrant

This guide focuses on directly ingesting already parsed documents, which is useful when:
- You have documents that have been preprocessed externally
- You want to reduce processing time and API costs
- You need to import a large corpus of documents efficiently

## Prerequisites

- Access to RegulAIte backend environment
- Preparsed documents in JSON format (see format requirements below)
- Neo4j and Qdrant services running in Docker containers
- Docker containers connected to `regulaite_network`
- (Optional) FastEmbed for local embedding generation

## Format Requirements for Preparsed Documents

Your preparsed documents should be in JSON format with the following structure:

```json
{
  "document_id": "unique_document_identifier",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "date": "2023-06-15",
    "source": "Document Source",
    "document_type": "regulation|policy|standard|guideline",
    "version": "1.0",
    "custom_field1": "value1",
    "custom_field2": "value2"
  },
  "chunks": [
    {
      "chunk_id": "chunk_unique_id_1",
      "content": "The text content of this chunk...",
      "embedding": [0.1, 0.2, 0.3, ...],
      "metadata": {
        "page_number": 1,
        "section": "Introduction",
        "subsection": "Background"
      }
    },
    {
      "chunk_id": "chunk_unique_id_2",
      "content": "More content from the document...",
      "embedding": [0.4, 0.5, 0.6, ...],
      "metadata": {
        "page_number": 2,
        "section": "Main Requirements",
        "subsection": "Technical Controls"
      }
    }
  ],
  "relationships": [
    {
      "source_chunk_id": "chunk_unique_id_1",
      "target_chunk_id": "chunk_unique_id_2", 
      "relationship_type": "REFERENCED_BY"
    }
  ]
}
```

### Important Notes:
- If your documents don't have embeddings, you can provide only the content and the ingestion script will generate embeddings using FastEmbed locally
- The relationships field is optional but recommended for enhancing the knowledge graph in Neo4j

## Ingestion Process

### 1. Place Preparsed Documents

Place your preparsed documents in the designated folder:

```bash
mkdir -p /path/to/regulaite/backend/data/preparsed_documents
cp your_preparsed_documents/*.json /path/to/regulaite/backend/data/preparsed_documents/
```

### 2. Run the Direct Ingestion Script

Use the provided ingestion script:

```bash
cd /path/to/regulaite/backend
python scripts/ingest_preparsed_documents.py --input-dir data/preparsed_documents --generate-missing-embeddings
```

#### Command Line Arguments:

- `--input-dir`: Directory containing preparsed document JSON files
- `--generate-missing-embeddings`: Generate embeddings for chunks that don't have them
- `--embedding-model`: Embedding model to use (default: "BAAI/bge-small-en-v1.5")
- `--use-gpu`: Use GPU for embedding generation if available
- `--batch-size`: Number of documents to process in each batch (default: 10)
- `--skip-neo4j`: Skip ingestion into Neo4j
- `--skip-qdrant`: Skip ingestion into Qdrant
- `--dry-run`: Validate documents without ingesting

### 3. Verify Ingestion

Check that your documents were properly ingested:

```bash
# Check Neo4j
python scripts/verify_ingestion.py --storage neo4j --document-id your_document_id

# Check Qdrant
python scripts/verify_ingestion.py --storage qdrant --document-id your_document_id
```

## Local Embedding Generation with FastEmbed

RegulAIte now supports local embedding generation using FastEmbed to eliminate the need for external API calls. This is particularly useful for:

1. Offline environments
2. Reducing API costs
3. Protecting sensitive data
4. Improving processing speed

### Available Models

FastEmbed supports several embedding models, including:

- `BAAI/bge-small-en-v1.5` (default): Fast and efficient model with 384 dimensions
- `BAAI/bge-base-en-v1.5`: Better quality with 768 dimensions
- `BAAI/bge-large-en-v1.5`: Highest quality with 1024 dimensions
- `sentence-transformers/all-MiniLM-L6-v2`: Compact and fast model
- `intfloat/e5-small-v2`: Specialized for retrieval tasks

To specify a model during ingestion:

```bash
python scripts/ingest_preparsed_documents.py --input-dir data/preparsed_documents --generate-missing-embeddings --embedding-model "BAAI/bge-base-en-v1.5"
```

### GPU Acceleration

For faster processing, you can utilize GPU acceleration if available:

```bash
python scripts/ingest_preparsed_documents.py --input-dir data/preparsed_documents --generate-missing-embeddings --use-gpu
```

## How It Works

The ingestion script performs the following steps:

1. **Validation**: Validates the format of each preparsed document
2. **Embedding Generation**: For chunks without embeddings, generates embeddings using FastEmbed
3. **Neo4j Ingestion**:
   - Creates document nodes with metadata
   - Creates chunk nodes with content and metadata
   - Establishes relationships between document and chunks
   - Establishes inter-chunk relationships if provided
4. **Qdrant Ingestion**:
   - Creates collection if it doesn't exist
   - Uploads chunk vectors with payload containing document and chunk metadata
   - Configures proper indexing for efficient retrieval

## Integration with Existing Documents

The ingested preparsed documents are fully compatible with the RegulAIte agent system. After ingestion, they will be:

1. Available for retrieval by all agents
2. Included in regulatory compliance analysis
3. Used in threat modeling and vulnerability assessments
4. Incorporated into compliance mapping processes

## Converting from Other Formats

The system includes a utility to convert documents from other formats to the preparsed JSON format:

```bash
python scripts/convert_to_preparsed_format.py --input-file my_document.xml --output-file data/preparsed_documents/converted_document.json --format xml --generate-embeddings
```

Supported formats:
- XML
- CSV
- Excel Spreadsheets
- Markdown Tables

### Conversion Command Line Arguments:

- `--input-file`: Input file to convert
- `--output-file`: Output JSON file path
- `--format`: Format of the input file (xml, csv, excel, markdown)
- `--doc-id`: Document ID (optional, will be generated if not provided)
- `--title`: Document title
- `--author`: Document author
- `--date`: Document date
- `--doc-type`: Document type
- `--language`: Document language (default: en)
- `--generate-embeddings`: Generate embeddings for chunks
- `--embedding-model`: FastEmbed model to use
- `--use-gpu`: Use GPU for embedding generation if available
- `--chunk-size`: Size of text chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)

## Common Issues and Troubleshooting

### Connection Issues

If you encounter connection issues:

```bash
# Check that containers are running
docker ps | grep -E 'neo4j|qdrant'

# Verify they're on the same network
docker network inspect regulaite_network
```

### Embedding Generation Failures

If embedding generation fails:

```bash
# Check if FastEmbed is properly installed
pip install fastembed

# Test embedding generation
python -c "from fastembed import TextEmbedding; model = TextEmbedding(); result = list(model.embed(['Test'])); print(len(result[0]))"
```

### Document Validation Errors

If documents fail validation:

```bash
# Use validation-only mode
python scripts/ingest_preparsed_documents.py --input-dir data/preparsed_documents --validate-only --verbose
```

## Best Practices

1. **Batch Processing**: For large document sets, process in batches of 10-20 documents
2. **Embedding Model Consistency**: Use the same embedding model as your regular pipeline
3. **Document Versioning**: Include version information in document metadata
4. **Relationship Enrichment**: Define as many inter-chunk relationships as possible
5. **Regular Verification**: Periodically verify that documents are retrievable
6. **GPU Utilization**: Use GPU acceleration for large document sets if available

## Security Considerations

- Ensure your preparsed documents don't contain sensitive information not meant for storage
- Use secure channels when transferring preparsed documents to the ingestion directory
- Set appropriate file permissions on the preparsed documents directory
- Using local embedding generation keeps sensitive text data within your infrastructure

## Performance Considerations

Direct ingestion of preparsed documents offers significant performance benefits:
- Bypasses CPU-intensive parsing operations
- Reduces API calls to external parsing services
- Allows for parallel ingestion of multiple documents
- Enables precise control over chunk boundaries and relationships
- Local embedding generation eliminates network latency

For a corpus of 100 typical regulatory documents, direct ingestion with local embedding generation can reduce processing time from hours to minutes. 