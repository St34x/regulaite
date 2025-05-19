#!/usr/bin/env python3
# convert_to_preparsed_format.py
import argparse
import json
import os
import logging
import sys
import uuid
import csv
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from fastembed import TextEmbedding

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DocumentConverter:
    """Tool for converting documents from various formats to preparsed JSON format"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_gpu: bool = False
    ):
        """Initialize the converter with parameters"""
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_gpu = use_gpu
        
        # Initialize FastEmbed
        self._init_fastembed()
    
    def _init_fastembed(self):
        """Initialize FastEmbed for generating embeddings"""
        try:
            # Initialize embedding model with specified parameters
            self.embedding = TextEmbedding(
                model_name=self.embedding_model,
                max_length=512,
                gpu=self.use_gpu
            )
            logger.info(f"FastEmbed initialized with model {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed: {str(e)}")
            self.embedding = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using FastEmbed"""
        if not self.embedding:
            raise ValueError("FastEmbed not initialized. Cannot generate embeddings.")
        
        try:
            # FastEmbed returns an iterator, so we need to get the first item
            embeddings = list(self.embedding.embed([text]))
            if not embeddings:
                raise ValueError("Failed to generate embedding: empty result")
            return embeddings[0].tolist()  # Convert numpy array to list
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def create_chunks(self, text: str, generate_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Split text into chunks with optional overlap and embeddings"""
        if not text:
            return []
        
        # Split text into chunks
        chunks = []
        
        # Simple chunking by character count
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            if i > 0:
                i = i - self.chunk_overlap
            
            chunk_text = text[i:i + self.chunk_size]
            if len(chunk_text) < 50:  # Skip very small chunks
                continue
            
            chunk_id = str(uuid.uuid4())
            chunk = {
                "chunk_id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    "start_pos": i,
                    "end_pos": min(i + self.chunk_size, len(text))
                }
            }
            
            # Generate embedding if requested
            if generate_embeddings and self.embedding:
                try:
                    chunk["embedding"] = self.generate_embedding(chunk_text)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk: {str(e)}")
            
            chunks.append(chunk)
        
        return chunks
    
    def convert_xml(self, xml_file: str, doc_id: str = None, metadata: Dict[str, Any] = None,
                    generate_embeddings: bool = False) -> Dict[str, Any]:
        """Convert XML file to preparsed document format"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Create document ID if not provided
            if not doc_id:
                doc_id = f"xml_{os.path.basename(xml_file)}_{str(uuid.uuid4())[:8]}"
            
            # Extract basic metadata
            if not metadata:
                metadata = {
                    "title": os.path.basename(xml_file),
                    "source": "xml_conversion",
                    "document_type": "xml",
                    "language": "en"
                }
                
                # Try to extract more metadata from XML
                for child in root:
                    if child.tag.lower() in ["title", "author", "date", "version"]:
                        metadata[child.tag.lower()] = child.text
            
            # Extract text content from XML elements
            chunks = []
            relationships = []
            
            def process_element(element, parent_id=None, depth=0, path=""):
                """Process XML elements recursively"""
                element_id = str(uuid.uuid4())
                current_path = f"{path}/{element.tag}" if path else element.tag
                
                # Get text content
                text = element.text or ""
                if text.strip():
                    # Process this element's text
                    element_chunks = self.create_chunks(
                        text.strip(),
                        generate_embeddings=generate_embeddings
                    )
                    
                    # Add metadata to chunks
                    for chunk in element_chunks:
                        chunk["metadata"]["element_tag"] = element.tag
                        chunk["metadata"]["element_path"] = current_path
                        chunk["metadata"]["element_depth"] = depth
                        
                        # Add attributes as metadata
                        for attr_name, attr_value in element.attrib.items():
                            chunk["metadata"][f"attr_{attr_name}"] = attr_value
                    
                    # Add to chunks list
                    chunks.extend(element_chunks)
                    
                    # Create relationship with parent
                    if parent_id and element_chunks:
                        relationships.append({
                            "source_chunk_id": parent_id,
                            "target_chunk_id": element_chunks[0]["chunk_id"],
                            "relationship_type": "PARENT_OF"
                        })
                
                # Process child elements
                for child in element:
                    if isinstance(child, ET.Element):
                        process_element(
                            child,
                            parent_id=element_chunks[0]["chunk_id"] if text.strip() and element_chunks else None,
                            depth=depth + 1,
                            path=current_path
                        )
            
            # Process the root element
            process_element(root)
            
            # Create document
            document = {
                "document_id": doc_id,
                "metadata": metadata,
                "chunks": chunks,
                "relationships": relationships
            }
            
            return document
        except Exception as e:
            logger.error(f"Error converting XML file {xml_file}: {str(e)}")
            raise
    
    def convert_csv(self, csv_file: str, doc_id: str = None, metadata: Dict[str, Any] = None,
                    generate_embeddings: bool = False) -> Dict[str, Any]:
        """Convert CSV file to preparsed document format"""
        try:
            # Read CSV file
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)
                rows = list(reader)
            
            # Create document ID if not provided
            if not doc_id:
                doc_id = f"csv_{os.path.basename(csv_file)}_{str(uuid.uuid4())[:8]}"
            
            # Extract basic metadata
            if not metadata:
                metadata = {
                    "title": os.path.basename(csv_file),
                    "source": "csv_conversion",
                    "document_type": "csv",
                    "language": "en",
                    "columns": headers,
                    "row_count": len(rows)
                }
            
            # Create chunks for each row
            chunks = []
            relationships = []
            
            # Create a chunk for the headers
            headers_text = ", ".join(headers)
            headers_chunk = {
                "chunk_id": f"headers_{str(uuid.uuid4())}",
                "content": f"CSV Headers: {headers_text}",
                "metadata": {
                    "element_type": "headers",
                    "columns": headers
                }
            }
            
            # Generate embedding if requested
            if generate_embeddings and self.embedding:
                try:
                    headers_chunk["embedding"] = self.generate_embedding(headers_chunk["content"])
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for headers: {str(e)}")
            
            chunks.append(headers_chunk)
            
            # Process each row
            for row_idx, row in enumerate(rows):
                # Skip empty rows
                if not any(cell.strip() for cell in row):
                    continue
                
                # Create a string representation of the row
                row_text = ""
                for col_idx, cell in enumerate(row):
                    if cell.strip():
                        header = headers[col_idx] if col_idx < len(headers) else f"Column{col_idx}"
                        row_text += f"{header}: {cell.strip()}\n"
                
                if not row_text:
                    continue
                
                # Create chunks for this row
                row_chunks = self.create_chunks(
                    row_text,
                    generate_embeddings=generate_embeddings
                )
                
                # Add metadata to chunks
                for chunk in row_chunks:
                    chunk["metadata"]["element_type"] = "row"
                    chunk["metadata"]["row_index"] = row_idx
                
                # Add to chunks list
                chunks.extend(row_chunks)
                
                # Create relationship with headers
                if row_chunks:
                    relationships.append({
                        "source_chunk_id": headers_chunk["chunk_id"],
                        "target_chunk_id": row_chunks[0]["chunk_id"],
                        "relationship_type": "HEADER_OF"
                    })
            
            # Create document
            document = {
                "document_id": doc_id,
                "metadata": metadata,
                "chunks": chunks,
                "relationships": relationships
            }
            
            return document
        except Exception as e:
            logger.error(f"Error converting CSV file {csv_file}: {str(e)}")
            raise
    
    def convert_excel(self, excel_file: str, doc_id: str = None, metadata: Dict[str, Any] = None,
                     generate_embeddings: bool = False) -> Dict[str, Any]:
        """Convert Excel file to preparsed document format"""
        try:
            # Read Excel file
            xl = pd.ExcelFile(excel_file)
            sheet_names = xl.sheet_names
            
            # Create document ID if not provided
            if not doc_id:
                doc_id = f"excel_{os.path.basename(excel_file)}_{str(uuid.uuid4())[:8]}"
            
            # Extract basic metadata
            if not metadata:
                metadata = {
                    "title": os.path.basename(excel_file),
                    "source": "excel_conversion",
                    "document_type": "excel",
                    "language": "en",
                    "sheets": sheet_names
                }
            
            # Create chunks for each sheet
            all_chunks = []
            all_relationships = []
            
            # Create a chunk for the file overview
            overview_text = f"Excel file containing {len(sheet_names)} sheets: {', '.join(sheet_names)}"
            overview_chunk = {
                "chunk_id": f"overview_{str(uuid.uuid4())}",
                "content": overview_text,
                "metadata": {
                    "element_type": "overview",
                    "sheets": sheet_names
                }
            }
            
            # Generate embedding if requested
            if generate_embeddings and self.embedding:
                try:
                    overview_chunk["embedding"] = self.generate_embedding(overview_chunk["content"])
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for overview: {str(e)}")
            
            all_chunks.append(overview_chunk)
            
            # Process each sheet
            for sheet_name in sheet_names:
                # Read sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Create a chunk for the sheet headers
                headers = df.columns.tolist()
                headers_text = f"Sheet: {sheet_name}, Headers: {', '.join(str(h) for h in headers)}"
                headers_chunk = {
                    "chunk_id": f"headers_{sheet_name}_{str(uuid.uuid4())}",
                    "content": headers_text,
                    "metadata": {
                        "element_type": "sheet_headers",
                        "sheet_name": sheet_name,
                        "columns": [str(h) for h in headers]
                    }
                }
                
                # Generate embedding if requested
                if generate_embeddings and self.embedding:
                    try:
                        headers_chunk["embedding"] = self.generate_embedding(headers_chunk["content"])
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for sheet headers: {str(e)}")
                
                all_chunks.append(headers_chunk)
                
                # Create relationship with overview
                all_relationships.append({
                    "source_chunk_id": overview_chunk["chunk_id"],
                    "target_chunk_id": headers_chunk["chunk_id"],
                    "relationship_type": "CONTAINS"
                })
                
                # Convert dataframe to text chunks
                for row_idx, row in df.iterrows():
                    # Skip empty rows
                    if row.isna().all():
                        continue
                    
                    # Create a string representation of the row
                    row_text = f"Sheet: {sheet_name}, Row: {row_idx}\n"
                    for col_idx, header in enumerate(headers):
                        if col_idx < len(row) and not pd.isna(row[col_idx]):
                            row_text += f"{header}: {row[col_idx]}\n"
                    
                    if len(row_text) <= len(f"Sheet: {sheet_name}, Row: {row_idx}\n"):
                        continue
                    
                    # Create chunks for this row
                    row_chunks = self.create_chunks(
                        row_text,
                        generate_embeddings=generate_embeddings
                    )
                    
                    # Add metadata to chunks
                    for chunk in row_chunks:
                        chunk["metadata"]["element_type"] = "row"
                        chunk["metadata"]["sheet_name"] = sheet_name
                        chunk["metadata"]["row_index"] = row_idx
                    
                    # Add to chunks list
                    all_chunks.extend(row_chunks)
                    
                    # Create relationship with sheet headers
                    if row_chunks:
                        all_relationships.append({
                            "source_chunk_id": headers_chunk["chunk_id"],
                            "target_chunk_id": row_chunks[0]["chunk_id"],
                            "relationship_type": "HEADER_OF"
                        })
            
            # Create document
            document = {
                "document_id": doc_id,
                "metadata": metadata,
                "chunks": all_chunks,
                "relationships": all_relationships
            }
            
            return document
        except Exception as e:
            logger.error(f"Error converting Excel file {excel_file}: {str(e)}")
            raise
    
    def convert_markdown_tables(self, md_file: str, doc_id: str = None, metadata: Dict[str, Any] = None,
                               generate_embeddings: bool = False) -> Dict[str, Any]:
        """Convert Markdown file with tables to preparsed document format"""
        try:
            # Read Markdown file
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create document ID if not provided
            if not doc_id:
                doc_id = f"md_{os.path.basename(md_file)}_{str(uuid.uuid4())[:8]}"
            
            # Extract basic metadata
            if not metadata:
                metadata = {
                    "title": os.path.basename(md_file),
                    "source": "markdown_conversion",
                    "document_type": "markdown",
                    "language": "en"
                }
            
            # Find tables in Markdown
            lines = content.split("\n")
            tables = []
            current_table = []
            in_table = False
            
            for line in lines:
                line = line.strip()
                if line.startswith("|") and line.endswith("|"):
                    if not in_table:
                        in_table = True
                        current_table = []
                    current_table.append(line)
                elif in_table and not line:
                    if len(current_table) >= 2:  # Need at least header and separator
                        tables.append(current_table[:])
                    current_table = []
                    in_table = False
            
            # Catch the last table if it doesn't end with an empty line
            if in_table and len(current_table) >= 2:
                tables.append(current_table)
            
            # Create chunks
            all_chunks = []
            all_relationships = []
            
            # Create a chunk for the document overview
            overview_text = f"Markdown document with {len(tables)} tables"
            overview_chunk = {
                "chunk_id": f"overview_{str(uuid.uuid4())}",
                "content": overview_text,
                "metadata": {
                    "element_type": "overview",
                    "table_count": len(tables)
                }
            }
            
            # Generate embedding if requested
            if generate_embeddings and self.embedding:
                try:
                    overview_chunk["embedding"] = self.generate_embedding(overview_text)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for overview: {str(e)}")
            
            all_chunks.append(overview_chunk)
            
            # Process each table
            for table_idx, table in enumerate(tables):
                if len(table) < 2:
                    continue
                
                # Parse header
                header_line = table[0]
                header_cells = [cell.strip() for cell in header_line.split("|")[1:-1]]
                
                # Create a chunk for the table headers
                headers_text = f"Table {table_idx + 1} Headers: {', '.join(header_cells)}"
                headers_chunk = {
                    "chunk_id": f"table{table_idx}_headers_{str(uuid.uuid4())}",
                    "content": headers_text,
                    "metadata": {
                        "element_type": "table_headers",
                        "table_index": table_idx,
                        "columns": header_cells
                    }
                }
                
                # Generate embedding if requested
                if generate_embeddings and self.embedding:
                    try:
                        headers_chunk["embedding"] = self.generate_embedding(headers_text)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for table headers: {str(e)}")
                
                all_chunks.append(headers_chunk)
                
                # Create relationship with overview
                all_relationships.append({
                    "source_chunk_id": overview_chunk["chunk_id"],
                    "target_chunk_id": headers_chunk["chunk_id"],
                    "relationship_type": "CONTAINS"
                })
                
                # Skip the separator line
                row_start = 2
                
                # Process each row
                for row_idx, row_line in enumerate(table[row_start:]):
                    row_cells = [cell.strip() for cell in row_line.split("|")[1:-1]]
                    
                    # Create a string representation of the row
                    row_text = f"Table {table_idx + 1}, Row {row_idx + 1}:\n"
                    for col_idx, header in enumerate(header_cells):
                        if col_idx < len(row_cells):
                            row_text += f"{header}: {row_cells[col_idx]}\n"
                    
                    # Create chunk for this row
                    row_chunk = {
                        "chunk_id": f"table{table_idx}_row{row_idx}_{str(uuid.uuid4())}",
                        "content": row_text,
                        "metadata": {
                            "element_type": "table_row",
                            "table_index": table_idx,
                            "row_index": row_idx
                        }
                    }
                    
                    # Generate embedding if requested
                    if generate_embeddings and self.embedding:
                        try:
                            row_chunk["embedding"] = self.generate_embedding(row_text)
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for table row: {str(e)}")
                    
                    all_chunks.append(row_chunk)
                    
                    # Create relationship with headers
                    all_relationships.append({
                        "source_chunk_id": headers_chunk["chunk_id"],
                        "target_chunk_id": row_chunk["chunk_id"],
                        "relationship_type": "HEADER_OF"
                    })
            
            # Also create chunks for regular text
            text_chunks = []
            current_text = ""
            
            for line in lines:
                # Skip table lines
                if line.strip().startswith("|") and line.strip().endswith("|"):
                    if current_text.strip():
                        text_chunks.extend(self.create_chunks(
                            current_text.strip(),
                            generate_embeddings=generate_embeddings
                        ))
                        current_text = ""
                    continue
                
                current_text += line + "\n"
            
            # Add remaining text
            if current_text.strip():
                text_chunks.extend(self.create_chunks(
                    current_text.strip(),
                    generate_embeddings=generate_embeddings
                ))
            
            # Add metadata to text chunks
            for chunk in text_chunks:
                chunk["metadata"]["element_type"] = "text"
                
                # Create relationship with overview
                all_relationships.append({
                    "source_chunk_id": overview_chunk["chunk_id"],
                    "target_chunk_id": chunk["chunk_id"],
                    "relationship_type": "CONTAINS"
                })
            
            all_chunks.extend(text_chunks)
            
            # Create document
            document = {
                "document_id": doc_id,
                "metadata": metadata,
                "chunks": all_chunks,
                "relationships": all_relationships
            }
            
            return document
        except Exception as e:
            logger.error(f"Error converting Markdown file {md_file}: {str(e)}")
            raise
    
    def convert_file(self, input_file: str, output_file: str, format_type: str,
                    doc_id: str = None, metadata: Dict[str, Any] = None,
                    generate_embeddings: bool = False) -> bool:
        """Convert a file to preparsed document format and save to output file"""
        try:
            # Create document based on format
            document = None
            
            if format_type.lower() == "xml":
                document = self.convert_xml(input_file, doc_id, metadata, generate_embeddings)
            elif format_type.lower() == "csv":
                document = self.convert_csv(input_file, doc_id, metadata, generate_embeddings)
            elif format_type.lower() == "excel":
                document = self.convert_excel(input_file, doc_id, metadata, generate_embeddings)
            elif format_type.lower() == "markdown":
                document = self.convert_markdown_tables(input_file, doc_id, metadata, generate_embeddings)
            else:
                logger.error(f"Unsupported format: {format_type}")
                return False
            
            # Save to output file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Converted {input_file} to {output_file} with {len(document['chunks'])} chunks")
            return True
        except Exception as e:
            logger.error(f"Error converting file {input_file}: {str(e)}")
            return False


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Convert documents to preparsed JSON format")
    
    parser.add_argument("--input-file", required=True, help="Input file to convert")
    parser.add_argument("--output-file", required=True, help="Output JSON file path")
    parser.add_argument("--format", required=True, choices=["xml", "csv", "excel", "markdown"],
                        help="Format of the input file")
    parser.add_argument("--doc-id", help="Document ID (optional, will be generated if not provided)")
    parser.add_argument("--title", help="Document title")
    parser.add_argument("--author", help="Document author")
    parser.add_argument("--date", help="Document date")
    parser.add_argument("--doc-type", help="Document type")
    parser.add_argument("--language", default="en", help="Document language (default: en)")
    parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings for chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", 
                       help="FastEmbed model to use (default: BAAI/bge-small-en-v1.5)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for embedding generation if available")
    
    args = parser.parse_args()
    
    # Create metadata from arguments
    metadata = {
        "language": args.language
    }
    
    if args.title:
        metadata["title"] = args.title
    
    if args.author:
        metadata["author"] = args.author
    
    if args.date:
        metadata["date"] = args.date
    
    if args.doc_type:
        metadata["document_type"] = args.doc_type
    
    # Create converter
    converter = DocumentConverter(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_gpu=args.use_gpu
    )
    
    # Convert file
    success = converter.convert_file(
        input_file=args.input_file,
        output_file=args.output_file,
        format_type=args.format,
        doc_id=args.doc_id,
        metadata=metadata,
        generate_embeddings=args.generate_embeddings
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 