# plugins/regul_aite/backend/llamaIndex_rag/__init__.py
"""
LlamaIndex RAG integration module for RegulAite.
Provides document indexing and retrieval capabilities using Neo4j and Qdrant with FastEmbed.
"""

from .rag import RAGSystem

__all__ = ["RAGSystem"]
