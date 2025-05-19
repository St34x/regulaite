#!/usr/bin/env python3
# verify_ingestion.py
import argparse
import logging
import os
import sys
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class IngestionVerifier:
    """Tool for verifying document ingestion in Neo4j and Qdrant"""
    
    def __init__(
        self,
        neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        neo4j_user: str = os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password"),
        qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333"),
        qdrant_collection_prefix: str = "regulaite_docs"
    ):
        """Initialize the verifier with connection parameters"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.qdrant_url = qdrant_url
        self.qdrant_collection_prefix = qdrant_collection_prefix
        
        # Initialize connections
        self._init_neo4j()
        self._init_qdrant()
    
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
    
    def verify_neo4j(self, doc_id: str) -> Dict[str, Any]:
        """Verify document existence and structure in Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                # Check document existence
                doc_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})
                    RETURN d
                    """,
                    doc_id=doc_id
                )
                
                doc_record = doc_result.single()
                if not doc_record:
                    return {
                        "exists": False,
                        "message": f"Document {doc_id} not found in Neo4j"
                    }
                
                # Get document properties
                doc_node = doc_record["d"]
                
                # Count chunks
                chunks_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c:Chunk)
                    RETURN count(c) as chunk_count
                    """,
                    doc_id=doc_id
                )
                
                chunk_count = chunks_result.single()["chunk_count"]
                
                # Count relationships between chunks
                relationships_result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(c1:Chunk)
                    MATCH (c1)-[r:RELATES_TO]->(c2:Chunk)
                    WHERE (d)-[:CONTAINS]->(c2)
                    RETURN count(r) as relationship_count
                    """,
                    doc_id=doc_id
                )
                
                relationship_count = relationships_result.single()["relationship_count"]
                
                return {
                    "exists": True,
                    "properties": dict(doc_node),
                    "chunk_count": chunk_count,
                    "relationship_count": relationship_count,
                    "message": f"Document {doc_id} found in Neo4j with {chunk_count} chunks and {relationship_count} relationships"
                }
        except Exception as e:
            logger.error(f"Error verifying document in Neo4j: {str(e)}")
            return {
                "exists": False,
                "error": str(e),
                "message": f"Error verifying document {doc_id} in Neo4j: {str(e)}"
            }
    
    def verify_qdrant(self, doc_id: str) -> Dict[str, Any]:
        """Verify document existence in Qdrant"""
        try:
            # Get all collections
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # Filter for collections with our prefix
            regulaite_collections = [c for c in collection_names if c.startswith(self.qdrant_collection_prefix)]
            
            if not regulaite_collections:
                return {
                    "exists": False,
                    "message": f"No collections found with prefix {self.qdrant_collection_prefix}"
                }
            
            # Check each collection for the document
            total_points = 0
            found_collections = []
            
            for collection_name in regulaite_collections:
                # Search for points with the document ID
                search_result = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_filter={
                        "must": [
                            {
                                "key": "doc_id",
                                "match": {
                                    "value": doc_id
                                }
                            }
                        ]
                    },
                    limit=1  # We just need to check existence
                )
                
                if search_result:
                    # Document exists in this collection, get count
                    count_result = self.qdrant_client.count(
                        collection_name=collection_name,
                        count_filter={
                            "must": [
                                {
                                    "key": "doc_id",
                                    "match": {
                                        "value": doc_id
                                    }
                                }
                            ]
                        }
                    )
                    
                    count = count_result.count
                    if count > 0:
                        total_points += count
                        found_collections.append({
                            "name": collection_name,
                            "points": count
                        })
            
            if total_points > 0:
                return {
                    "exists": True,
                    "total_points": total_points,
                    "collections": found_collections,
                    "message": f"Document {doc_id} found in Qdrant with {total_points} vectors across {len(found_collections)} collections"
                }
            else:
                return {
                    "exists": False,
                    "message": f"Document {doc_id} not found in any Qdrant collection"
                }
        except Exception as e:
            logger.error(f"Error verifying document in Qdrant: {str(e)}")
            return {
                "exists": False,
                "error": str(e),
                "message": f"Error verifying document {doc_id} in Qdrant: {str(e)}"
            }
    
    def verify_document(self, doc_id: str, storage: str = "both") -> Dict[str, Any]:
        """Verify document existence in specified storage(s)"""
        results = {}
        
        if storage.lower() in ["both", "neo4j"]:
            results["neo4j"] = self.verify_neo4j(doc_id)
        
        if storage.lower() in ["both", "qdrant"]:
            results["qdrant"] = self.verify_qdrant(doc_id)
        
        # Determine overall status
        if storage.lower() == "both":
            results["exists"] = results["neo4j"]["exists"] and results["qdrant"]["exists"]
        elif storage.lower() == "neo4j":
            results["exists"] = results["neo4j"]["exists"]
        elif storage.lower() == "qdrant":
            results["exists"] = results["qdrant"]["exists"]
        
        return results
    
    def close(self):
        """Close connections"""
        if hasattr(self, "neo4j_driver"):
            self.neo4j_driver.close()
        
        logger.info("Closed connections")


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Verify document ingestion in Neo4j and Qdrant")
    
    parser.add_argument("--document-id", required=True, help="ID of the document to verify")
    parser.add_argument("--storage", choices=["both", "neo4j", "qdrant"], default="both", 
                       help="Storage system to verify (default: both)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        verifier = IngestionVerifier()
        
        results = verifier.verify_document(args.document_id, args.storage)
        
        verifier.close()
        
        # Print results
        print("\n===== Verification Results =====")
        
        if args.storage.lower() in ["both", "neo4j"]:
            neo4j_result = results["neo4j"]
            print(f"\nNeo4j: {neo4j_result['message']}")
            
            if neo4j_result["exists"] and args.verbose:
                print("\nDocument Properties:")
                for key, value in neo4j_result["properties"].items():
                    print(f"  {key}: {value}")
        
        if args.storage.lower() in ["both", "qdrant"]:
            qdrant_result = results["qdrant"]
            print(f"\nQdrant: {qdrant_result['message']}")
            
            if qdrant_result["exists"] and args.verbose:
                print("\nVectors by Collection:")
                for collection in qdrant_result["collections"]:
                    print(f"  {collection['name']}: {collection['points']} vectors")
        
        # Return exit code based on existence
        sys.exit(0 if results["exists"] else 1)
        
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 