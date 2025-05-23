#!/usr/bin/env python3
"""
RegulAIte Qdrant Database Dump Script

This script dumps the entire Qdrant database including:
- All collections and their configurations
- All points (vectors and payloads) from each collection
- Collection statistics and metadata

The dump is saved as timestamped JSON files for easy backup and restoration.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter
except ImportError:
    print("Error: qdrant-client not installed. Please install it with: pip install qdrant-client")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QdrantDumper:
    """Utility class to dump Qdrant database entirely"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", output_dir: str = "qdrant_dumps"):
        """
        Initialize the Qdrant dumper
        
        Args:
            qdrant_url: URL of the Qdrant server
            output_dir: Directory to save dump files
        """
        self.qdrant_url = qdrant_url
        self.output_dir = Path(output_dir)
        self.client = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this dump session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"dump_{self.timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dump session directory: {self.session_dir}")
    
    def connect(self) -> bool:
        """Connect to Qdrant server"""
        try:
            logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            return False
    
    def get_all_collections(self) -> List[str]:
        """Get list of all collection names"""
        try:
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            logger.info(f"Found collections: {collection_names}")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to get collections: {str(e)}")
            return []
    
    def dump_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Dump collection configuration and statistics"""
        try:
            logger.info(f"Dumping collection info for: {collection_name}")
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            
            # Get collection statistics
            collection_count = self.client.count(collection_name=collection_name)
            
            # Convert to serializable format
            collection_data = {
                "name": collection_name,
                "config": {
                    "params": {
                        "vectors": {
                            "size": collection_info.config.params.vectors.size,
                            "distance": collection_info.config.params.vectors.distance.value
                        },
                        "shard_number": collection_info.config.params.shard_number,
                        "replication_factor": collection_info.config.params.replication_factor,
                        "write_consistency_factor": collection_info.config.params.write_consistency_factor,
                        "on_disk_payload": collection_info.config.params.on_disk_payload,
                    },
                    "hnsw_config": {
                        "m": collection_info.config.hnsw_config.m,
                        "ef_construct": collection_info.config.hnsw_config.ef_construct,
                        "full_scan_threshold": collection_info.config.hnsw_config.full_scan_threshold,
                        "max_indexing_threads": collection_info.config.hnsw_config.max_indexing_threads,
                        "on_disk": collection_info.config.hnsw_config.on_disk,
                        "payload_m": collection_info.config.hnsw_config.payload_m,
                    },
                    "optimizer_config": {
                        "deleted_threshold": collection_info.config.optimizer_config.deleted_threshold,
                        "vacuum_min_vector_number": collection_info.config.optimizer_config.vacuum_min_vector_number,
                        "default_segment_number": collection_info.config.optimizer_config.default_segment_number,
                        "max_segment_size": collection_info.config.optimizer_config.max_segment_size,
                        "memmap_threshold": collection_info.config.optimizer_config.memmap_threshold,
                        "indexing_threshold": collection_info.config.optimizer_config.indexing_threshold,
                        "flush_interval_sec": collection_info.config.optimizer_config.flush_interval_sec,
                        "max_optimization_threads": collection_info.config.optimizer_config.max_optimization_threads,
                    },
                    "wal_config": {
                        "wal_capacity_mb": collection_info.config.wal_config.wal_capacity_mb,
                        "wal_segments_ahead": collection_info.config.wal_config.wal_segments_ahead,
                    }
                },
                "status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status.status.value,
                "points_count": collection_count.count,
                "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', None),
                "segments_count": len(collection_info.segments) if collection_info.segments else 0,
            }
            
            # Save collection info
            info_file = self.session_dir / f"{collection_name}_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(collection_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collection {collection_name} info saved to {info_file}")
            logger.info(f"Collection {collection_name} contains {collection_count.count} points")
            
            return collection_data
            
        except Exception as e:
            logger.error(f"Failed to dump collection info for {collection_name}: {str(e)}")
            return {}
    
    def dump_collection_points(self, collection_name: str, batch_size: int = 1000) -> int:
        """Dump all points from a collection"""
        try:
            logger.info(f"Dumping points from collection: {collection_name}")
            
            # Initialize scroll
            total_points = 0
            batch_number = 0
            next_page_offset = None
            
            points_file = self.session_dir / f"{collection_name}_points.json"
            
            # Open file for writing
            with open(points_file, 'w', encoding='utf-8') as f:
                f.write('[\n')  # Start JSON array
                first_point = True
                
                while True:
                    # Scroll through points
                    scroll_result = self.client.scroll(
                        collection_name=collection_name,
                        limit=batch_size,
                        offset=next_page_offset,
                        with_payload=True,
                        with_vectors=True
                    )
                    
                    points = scroll_result[0]
                    next_page_offset = scroll_result[1]
                    
                    if not points:
                        break
                    
                    batch_number += 1
                    logger.info(f"Processing batch {batch_number} with {len(points)} points")
                    
                    # Convert points to serializable format
                    for point in points:
                        if not first_point:
                            f.write(',\n')
                        else:
                            first_point = False
                        
                        point_data = {
                            "id": point.id,
                            "vector": point.vector,
                            "payload": point.payload
                        }
                        
                        json.dump(point_data, f, ensure_ascii=False)
                        total_points += 1
                    
                    # Break if no more points
                    if next_page_offset is None:
                        break
                
                f.write('\n]')  # Close JSON array
            
            logger.info(f"Successfully dumped {total_points} points from {collection_name} to {points_file}")
            return total_points
            
        except Exception as e:
            logger.error(f"Failed to dump points from {collection_name}: {str(e)}")
            return 0
    
    def create_dump_summary(self, collections_data: Dict[str, Dict[str, Any]]) -> None:
        """Create a summary of the dump session"""
        try:
            summary = {
                "dump_info": {
                    "timestamp": self.timestamp,
                    "qdrant_url": self.qdrant_url,
                    "total_collections": len(collections_data),
                    "dump_directory": str(self.session_dir)
                },
                "collections": {}
            }
            
            total_points = 0
            for collection_name, collection_info in collections_data.items():
                points_count = collection_info.get('points_count', 0)
                total_points += points_count
                
                summary["collections"][collection_name] = {
                    "points_count": points_count,
                    "vector_size": collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size'),
                    "distance_metric": collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('distance'),
                    "status": collection_info.get('status'),
                    "files": {
                        "info_file": f"{collection_name}_info.json",
                        "points_file": f"{collection_name}_points.json"
                    }
                }
            
            summary["dump_info"]["total_points"] = total_points
            
            # Save summary
            summary_file = self.session_dir / "dump_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dump summary saved to {summary_file}")
            
            # Also create a README
            readme_content = f"""# RegulAIte Qdrant Database Dump
            
## Dump Information
- **Timestamp**: {self.timestamp}
- **Source**: {self.qdrant_url}
- **Total Collections**: {len(collections_data)}
- **Total Points**: {total_points:,}

## Collections Dumped
"""
            
            for collection_name, collection_info in collections_data.items():
                points_count = collection_info.get('points_count', 0)
                vector_size = collection_info.get('config', {}).get('params', {}).get('vectors', {}).get('size')
                readme_content += f"""
### {collection_name}
- **Points**: {points_count:,}
- **Vector Size**: {vector_size}
- **Files**: `{collection_name}_info.json`, `{collection_name}_points.json`
"""
            
            readme_content += f"""
## Files Structure
- `dump_summary.json` - Overview of the entire dump
- `{collection_name}_info.json` - Collection configuration and metadata
- `{collection_name}_points.json` - All points (vectors + payloads) from the collection
- `README.md` - This file

## Restoration
To restore this dump to a Qdrant instance, you can use the restore script or manually:
1. Create collections using the configuration from `*_info.json` files
2. Import points from `*_points.json` files using batch upsert operations

## Generated by RegulAIte Qdrant Dump Script
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            
            readme_file = self.session_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            logger.info(f"README saved to {readme_file}")
            
        except Exception as e:
            logger.error(f"Failed to create dump summary: {str(e)}")
    
    def dump_all(self, batch_size: int = 1000) -> bool:
        """Dump entire Qdrant database"""
        try:
            if not self.connect():
                return False
            
            logger.info("Starting complete Qdrant database dump")
            
            # Get all collections
            collection_names = self.get_all_collections()
            if not collection_names:
                logger.warning("No collections found in Qdrant database")
                return True
            
            collections_data = {}
            
            # Dump each collection
            for collection_name in collection_names:
                logger.info(f"Processing collection: {collection_name}")
                
                # Dump collection info
                collection_info = self.dump_collection_info(collection_name)
                if collection_info:
                    collections_data[collection_name] = collection_info
                
                # Dump collection points
                points_dumped = self.dump_collection_points(collection_name, batch_size)
                if collection_name in collections_data:
                    collections_data[collection_name]['points_dumped'] = points_dumped
            
            # Create summary
            self.create_dump_summary(collections_data)
            
            logger.info(f"✅ Complete database dump finished successfully!")
            logger.info(f"📁 Dump location: {self.session_dir}")
            logger.info(f"📊 Collections dumped: {len(collections_data)}")
            
            total_points = sum(info.get('points_count', 0) for info in collections_data.values())
            logger.info(f"📈 Total points dumped: {total_points:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to dump database: {str(e)}")
            return False
        finally:
            if self.client:
                self.client.close()

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Dump RegulAIte Qdrant database entirely",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dump with default settings (localhost:6333)
  python dump_qdrant_db.py
  
  # Dump from Docker container (when running from host)
  python dump_qdrant_db.py --qdrant-url http://localhost:6333
  
  # Dump with custom output directory and batch size
  python dump_qdrant_db.py --output-dir /backup/qdrant --batch-size 500
  
  # For RegulAIte Docker setup
  python dump_qdrant_db.py --qdrant-url http://localhost:6333 --output-dir ./backups
        """
    )
    
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="URL of the Qdrant server (default: http://localhost:6333)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="qdrant_dumps",
        help="Directory to save dump files (default: qdrant_dumps)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for scrolling through points (default: 1000)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create dumper and run
    dumper = QdrantDumper(
        qdrant_url=args.qdrant_url,
        output_dir=args.output_dir
    )
    
    success = dumper.dump_all(batch_size=args.batch_size)
    
    if success:
        print("\n🎉 Database dump completed successfully!")
        print(f"📁 Check the dump files in: {dumper.session_dir}")
    else:
        print("\n❌ Database dump failed. Check the logs for details.")
        exit(1)

if __name__ == "__main__":
    main() 