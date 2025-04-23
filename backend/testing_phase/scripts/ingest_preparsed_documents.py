#!/usr/bin/env python3
# ingest_preparsed_documents.py
import argparse
import json
import os
import logging
import sys
from typing import Dict, List, Any, Optional
import time
from tqdm import tqdm
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from fastembed import TextEmbedding
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preparsed_ingestion.log")
    ]
)
logger = logging.getLogger(__name__)

class PreparsedDocumentIngester:
    """Tool for ingesting preparsed documents directly into Neo4j and Qdrant"""
    
    def __init__(
        self,
        neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        neo4j_user: str = os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password"),
        qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333"),
        qdrant_collection_prefix: str = "regulaite_docs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        use_gpu: bool = False,
        embedding_dimensions: int = None  # Will be determined from the model
    ):
        """Initialize the ingester with connection parameters"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_url = qdrant_url
        self.qdrant_collection_prefix = qdrant_collection_prefix
        self.embedding_model = embedding_model
        self.use_gpu = use_gpu
        self.embedding_dimensions = embedding_dimensions
        
        # Initialize connections
        self._init_neo4j()
        self._init_qdrant()
        self._init_embedding()
        
    def _init_neo4j(self):
        """Initialize connection to Neo4j"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=3600
            )
            
            # Verify connection
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 'Connection successful' as status")
                for record in result:
                    logger.info(f"Neo4j connection: {record['status']}")
            
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def _init_qdrant(self):
        """Initialize connection to Qdrant"""
        try:
            self.qdrant_client = QdrantClient(url=self.qdrant_url)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def _init_embedding(self):
        """Initialize FastEmbed for embeddings"""
        try:
            self.embedding_model_instance = TextEmbedding(
                model_name=self.embedding_model,
                max_length=512,
                gpu=self.use_gpu
            )
            
            # Test the model and determine embedding dimensions
            test_embedding = list(self.embedding_model_instance.embed(["Test embedding generation"]))[0]
            self.embedding_dimensions = len(test_embedding)
            
            logger.info(f"FastEmbed initialized with model {self.embedding_model} "
                       f"(dimensions: {self.embedding_dimensions})")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed: {str(e)}")
            self.embedding_model_instance = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using FastEmbed"""
        if not self.embedding_model_instance:
            raise ValueError("FastEmbed model not initialized. Cannot generate embeddings.")
        
        try:
            embeddings = list(self.embedding_model_instance.embed([text]))
            if not embeddings:
                raise ValueError("Failed to generate embedding: empty result")
            return embeddings[0].tolist()  # Convert numpy array to list
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def validate_document(self, doc_data: Dict[str, Any]) -> bool:
        """Validate document format"""
        # Check required fields
        if "document_id" not in doc_data:
            logger.error("Missing required field: document_id")
            return False
        
        if "metadata" not in doc_data or not isinstance(doc_data["metadata"], dict):
            logger.error("Missing or invalid metadata field")
            return False
        
        if "chunks" not in doc_data or not isinstance(doc_data["chunks"], list):
            logger.error("Missing or invalid chunks field")
            return False
        
        # Check chunks
        for i, chunk in enumerate(doc_data["chunks"]):
            if "chunk_id" not in chunk:
                logger.error(f"Chunk {i} missing chunk_id")
                return False
            
            if "content" not in chunk:
                logger.error(f"Chunk {i} missing content")
                return False
        
        return True
    
    def ensure_collection_exists(self, language: str = "en") -> str:
        """Ensure the Qdrant collection exists for the given language"""
        collection_name = f"{self.qdrant_collection_prefix}_{language}"
        
        try:
            # Check if collection exists
            try:
                self.qdrant_client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
            except Exception:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.embedding_dimensions,
                        distance=rest.Distance.COSINE,
                        on_disk=True
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(
                        indexing_threshold=20000
                    )
                )
                logger.info(f"Created collection {collection_name}")
            
            return collection_name
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def ingest_document_to_neo4j(self, doc_data: Dict[str, Any]) -> bool:
        """Ingest document into Neo4j"""
        doc_id = doc_data["document_id"]
        metadata = doc_data["metadata"]
        chunks = doc_data["chunks"]
        relationships = doc_data.get("relationships", [])
        
        try:
            with self.neo4j_driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d += $metadata,
                        d.created_at = datetime(),
                        d.ingestion_method = 'preparsed'
                    RETURN d
                    """,
                    doc_id=doc_id,
                    metadata=metadata
                )
                
                # Create chunks
                for chunk in chunks:
                    session.run(
                        """
                        MATCH (d:Document {doc_id: $doc_id})
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.content = $content,
                            c.metadata = $metadata,
                            c.doc_id = $doc_id
                        MERGE (d)-[:CONTAINS]->(c)
                        """,
                        doc_id=doc_id,
                        chunk_id=chunk["chunk_id"],
                        content=chunk["content"],
                        metadata=chunk.get("metadata", {})
                    )
                
                # Create relationships between chunks
                for rel in relationships:
                    session.run(
                        """
                        MATCH (c1:Chunk {chunk_id: $source_id})
                        MATCH (c2:Chunk {chunk_id: $target_id})
                        MERGE (c1)-[:RELATES_TO {type: $rel_type}]->(c2)
                        """,
                        source_id=rel["source_chunk_id"],
                        target_id=rel["target_chunk_id"],
                        rel_type=rel["relationship_type"]
                    )
                
                logger.info(f"Ingested document {doc_id} into Neo4j with {len(chunks)} chunks")
                return True
        except Exception as e:
            logger.error(f"Error ingesting document to Neo4j: {str(e)}")
            return False
    
    def ingest_document_to_qdrant(self, doc_data: Dict[str, Any], generate_embeddings: bool = False) -> bool:
        """Ingest document into Qdrant"""
        doc_id = doc_data["document_id"]
        metadata = doc_data["metadata"]
        chunks = doc_data["chunks"]
        language = metadata.get("language", "en")
        
        # Ensure collection exists
        collection_name = self.ensure_collection_exists(language)
        
        try:
            points = []
            
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                
                # Check if embedding is present or needs to be generated
                if "embedding" not in chunk and generate_embeddings:
                    if not self.embedding_model_instance:
                        logger.error("Cannot generate embeddings: FastEmbed model not initialized")
                        return False
                    
                    embedding = self.generate_embedding(chunk["content"])
                elif "embedding" in chunk:
                    embedding = chunk["embedding"]
                else:
                    logger.error(f"Chunk {chunk_id} has no embedding and generate_embeddings is False")
                    return False
                
                # Create payload
                payload = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "content": chunk["content"],
                    "metadata": {**metadata, **(chunk.get("metadata", {}))}
                }
                
                # Add to points
                points.append(rest.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                ))
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant")
            
            logger.info(f"Ingested document {doc_id} into Qdrant with {len(chunks)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error ingesting document to Qdrant: {str(e)}")
            return False
    
    def ingest_document(self, doc_data: Dict[str, Any], ingest_neo4j: bool = True, 
                        ingest_qdrant: bool = True, generate_embeddings: bool = False) -> Dict[str, bool]:
        """Ingest document into both Neo4j and Qdrant"""
        doc_id = doc_data["document_id"]
        
        # Validate document format
        if not self.validate_document(doc_data):
            logger.error(f"Document {doc_id} failed validation")
            return {"neo4j": False, "qdrant": False}
        
        results = {}
        
        # Ingest to Neo4j
        if ingest_neo4j:
            neo4j_result = self.ingest_document_to_neo4j(doc_data)
            results["neo4j"] = neo4j_result
        else:
            results["neo4j"] = None
        
        # Ingest to Qdrant
        if ingest_qdrant:
            qdrant_result = self.ingest_document_to_qdrant(doc_data, generate_embeddings)
            results["qdrant"] = qdrant_result
        else:
            results["qdrant"] = None
        
        return results
    
    def ingest_documents_from_directory(self, input_dir: str, ingest_neo4j: bool = True,
                                       ingest_qdrant: bool = True, generate_embeddings: bool = False,
                                       batch_size: int = 10, dry_run: bool = False) -> Dict[str, Any]:
        """Ingest all documents from a directory"""
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory {input_dir} does not exist")
            return {"status": "error", "message": f"Directory {input_dir} not found"}
        
        json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
        logger.info(f"Found {len(json_files)} JSON files in {input_dir}")
        
        results = {
            "total": len(json_files),
            "successful_neo4j": 0,
            "failed_neo4j": 0,
            "successful_qdrant": 0,
            "failed_qdrant": 0,
            "skipped_neo4j": 0,
            "skipped_qdrant": 0,
            "failed_documents": []
        }
        
        for i, json_file in enumerate(tqdm(json_files, desc="Ingesting documents")):
            file_path = os.path.join(input_dir, json_file)
            
            try:
                with open(file_path, "r") as f:
                    doc_data = json.load(f)
                
                # Log only in dry run, otherwise it will be too verbose
                if dry_run:
                    logger.info(f"Processing document {i+1}/{len(json_files)}: {doc_data.get('document_id', 'unknown')}")
                
                if dry_run:
                    valid = self.validate_document(doc_data)
                    if valid:
                        logger.info(f"Document {doc_data['document_id']} is valid")
                    else:
                        logger.error(f"Document {doc_data.get('document_id', 'unknown')} is invalid")
                    continue
                
                # Process the document
                doc_results = self.ingest_document(
                    doc_data,
                    ingest_neo4j=ingest_neo4j,
                    ingest_qdrant=ingest_qdrant,
                    generate_embeddings=generate_embeddings
                )
                
                # Update results
                if doc_results["neo4j"] is True:
                    results["successful_neo4j"] += 1
                elif doc_results["neo4j"] is False:
                    results["failed_neo4j"] += 1
                    results["failed_documents"].append(doc_data.get("document_id", f"unknown_{i}"))
                else:
                    results["skipped_neo4j"] += 1
                
                if doc_results["qdrant"] is True:
                    results["successful_qdrant"] += 1
                elif doc_results["qdrant"] is False:
                    results["failed_qdrant"] += 1
                    if doc_data.get("document_id") not in results["failed_documents"]:
                        results["failed_documents"].append(doc_data.get("document_id", f"unknown_{i}"))
                else:
                    results["skipped_qdrant"] += 1
                
                # Sleep a bit to avoid overwhelming the services
                if (i + 1) % batch_size == 0 and i < len(json_files) - 1:
                    time.sleep(2)
            
            except Exception as e:
                logger.error(f"Error processing {json_file}: {str(e)}")
                results["failed_neo4j"] += int(ingest_neo4j)
                results["failed_qdrant"] += int(ingest_qdrant)
                results["failed_documents"].append(json_file)
        
        return results
    
    def close(self):
        """Close connections"""
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()
        
        logger.info("Closed connections")


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Ingest preparsed documents into Neo4j and Qdrant")
    
    parser.add_argument("--input-dir", required=True, help="Directory containing preparsed document JSON files")
    parser.add_argument("--generate-missing-embeddings", action="store_true", help="Generate embeddings for chunks that don't have them")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="FastEmbed model to use")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for embedding generation if available")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to process in each batch")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip ingestion into Neo4j")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip ingestion into Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Validate documents without ingesting")
    parser.add_argument("--validate-only", action="store_true", help="Only validate documents, don't ingest")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        ingester = PreparsedDocumentIngester(
            embedding_model=args.embedding_model,
            use_gpu=args.use_gpu
        )
        
        results = ingester.ingest_documents_from_directory(
            input_dir=args.input_dir,
            ingest_neo4j=not args.skip_neo4j and not args.validate_only,
            ingest_qdrant=not args.skip_qdrant and not args.validate_only,
            generate_embeddings=args.generate_missing_embeddings,
            batch_size=args.batch_size,
            dry_run=args.dry_run or args.validate_only
        )
        
        ingester.close()
        
        if args.dry_run or args.validate_only:
            logger.info("Dry run completed. No documents were ingested.")
            return
        
        # Print summary
        logger.info("===== Ingestion Summary =====")
        logger.info(f"Total documents processed: {results['total']}")
        logger.info(f"Neo4j: {results['successful_neo4j']} successful, {results['failed_neo4j']} failed, {results['skipped_neo4j']} skipped")
        logger.info(f"Qdrant: {results['successful_qdrant']} successful, {results['failed_qdrant']} failed, {results['skipped_qdrant']} skipped")
        
        if results["failed_documents"]:
            logger.info(f"Failed documents ({len(results['failed_documents'])}): {', '.join(results['failed_documents'][:10])}")
            if len(results["failed_documents"]) > 10:
                logger.info(f"... and {len(results['failed_documents']) - 10} more")
        
        # Return exit code based on success
        if results["failed_neo4j"] > 0 or results["failed_qdrant"] > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 