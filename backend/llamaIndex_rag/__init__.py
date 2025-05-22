"""
Production-ready LlamaIndex RAG implementation with hallucination prevention and detection.

This package provides a reliable RAG (Retrieval-Augmented Generation) implementation
using LlamaIndex with techniques to minimize and detect hallucinations.
"""

from llamaIndex_rag.rag import RAGSystem
from llamaIndex_rag.query_engine import RAGQueryEngine

__all__ = ["RAGSystem", "RAGQueryEngine"] 