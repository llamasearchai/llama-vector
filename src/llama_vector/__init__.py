"""
LlamaVector: Enterprise-grade vector index with hybrid HNSW+IVF capabilities.

This package provides a high-performance vector indexing solution with advanced
features such as hybrid indexing, hardware acceleration, multi-modal embedding
support, dynamic quantization, ACID-compliant snapshots, and quality control.
"""

__version__ = "0.1.0"

from llama_vector.config import LlamaVectorConfig
from llama_vector.embeddings import EmbeddingModel, EmbeddingModelRegistry
from llama_vector.index import VectorIndex
from llama_vector.quality import DriftDetector

__all__ = [
    "VectorIndex",
    "EmbeddingModelRegistry",
    "EmbeddingModel",
    "DriftDetector",
    "LlamaVectorConfig",
]
