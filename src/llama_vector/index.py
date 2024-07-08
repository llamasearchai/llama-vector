"""
Vector index implementation with hybrid HNSW+IVF capabilities.

This module provides the core VectorIndex class, which supports various index types,
hardware acceleration, quantization, and versioned snapshots.
"""

import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm

from llama_vector.acceleration import (
    enable_cuda,
    normalize_vectors,
)
from llama_vector.config import LlamaVectorConfig
from llama_vector.config import config as global_config
from llama_vector.utils import Timer

# Try to import MLX for acceleration on Apple Silicon
try:
    if global_config.use_mlx:
        import mlx.core as mx

        MLX_AVAILABLE = True
    else:
        MLX_AVAILABLE = False
except ImportError:
    MLX_AVAILABLE = False


class VectorIndex:
    """
    Enterprise-grade vector index with hybrid HNSW+IVF capabilities.

    This class provides a high-performance vector indexing solution with support for:
    - Multiple index types (flat, HNSW, IVF, hybrid HNSW+IVF)
    - Hardware acceleration (MLX, CUDA, CPU)
    - Dynamic quantization
    - ACID-compliant versioned snapshots

    Attributes:
        dimension (int): Dimensionality of vectors in the index
        index_type (str): Type of index ("flat", "hnsw", "ivf", or "hybrid")
        index (faiss.Index): The underlying FAISS index
        metadata (dict): Metadata about the index and its contents
        config (LlamaVectorConfig): Configuration for the index
    """

    def __init__(
        self,
        dimension: int,
        index_type: Literal["flat", "hnsw", "ivf", "hybrid"] = "hybrid",
        use_gpu: bool = False,
        config: Optional[LlamaVectorConfig] = None,
        hnsw_params: Optional[Dict[str, Any]] = None,
        ivf_params: Optional[Dict[str, Any]] = None,
        quantization: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new VectorIndex.

        Args:
            dimension: Dimensionality of vectors to be indexed
            index_type: Type of index to create
            use_gpu: Whether to use GPU acceleration if available
            config: Configuration for the index (defaults to global config)
            hnsw_params: Parameters for HNSW index
            ivf_params: Parameters for IVF index
            quantization: Quantization parameters
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu

        # Use provided config or fall back to global config
        self.config = config or global_config

        # Set up parameters
        self.hnsw_params = hnsw_params or self.config.hnsw_params.model_dump()
        self.ivf_params = ivf_params or self.config.ivf_params.model_dump()
        self.quantization_params = quantization or self.config.quantization.model_dump()

        # Initialize metadata
        self.metadata = {
            "id": str(uuid.uuid4()),
            "created_at": time.time(),
            "dimension": dimension,
            "index_type": index_type,
            "hnsw_params": self.hnsw_params,
            "ivf_params": self.ivf_params,
            "quantization": self.quantization_params,
            "count": 0,
            "versions": [],
            "items": {},  # Dict mapping IDs to metadata
        }

        # Set up mutex for thread safety
        self._mutex = threading.RLock()

        # Initialize the index
        self._create_index()

        if use_gpu and global_config.use_cuda:
            self._move_to_gpu()

    @classmethod
    def create(
        cls,
        dimension: int,
        index_type: Literal["flat", "hnsw", "ivf", "hybrid"] = "hybrid",
        use_gpu: bool = False,
        config: Optional[LlamaVectorConfig] = None,
        hnsw_params: Optional[Dict[str, Any]] = None,
        ivf_params: Optional[Dict[str, Any]] = None,
        quantization: Optional[Dict[str, Any]] = None,
    ) -> "VectorIndex":
        """
        Create a new VectorIndex with the specified parameters.

        This is the recommended factory method for creating a VectorIndex instance.

        Args:
            dimension: Dimensionality of vectors to be indexed
            index_type: Type of index to create
            use_gpu: Whether to use GPU acceleration if available
            config: Configuration for the index (defaults to global config)
            hnsw_params: Parameters for HNSW index
            ivf_params: Parameters for IVF index
            quantization: Quantization parameters

        Returns:
            VectorIndex: A new vector index instance
        """
        return cls(
            dimension=dimension,
            index_type=index_type,
            use_gpu=use_gpu,
            config=config,
            hnsw_params=hnsw_params,
            ivf_params=ivf_params,
            quantization=quantization,
        )

    def _create_index(self) -> None:
        """
        Create the underlying FAISS index based on the configured index type.

        This method initializes the appropriate index structure based on the
        index_type parameter and sets up quantization if enabled.
        """
        with self._mutex:
            # Start with a base vector description
            index_factory_str = "IDMap2,Normalize"
            quantize = self.quantization_params.get("enabled", False)

            if self.index_type == "flat":
                index_factory_str += ",Flat"

            elif self.index_type == "hnsw":
                m = self.hnsw_params.get("M", 16)
                ef_construction = self.hnsw_params.get("efConstruction", 200)
                index_factory_str += f",HNSW{m}"

            elif self.index_type == "ivf":
                nlist = self.ivf_params.get("nlist", 100)
                if quantize:
                    pq_m = self.quantization_params.get("pq_m", 8)
                    pq_nbits = self.quantization_params.get("pq_nbits", 8)
                    index_factory_str += f",IVF{nlist},PQ{pq_m}x{pq_nbits}"
                else:
                    index_factory_str += f",IVF{nlist},Flat"

            elif self.index_type == "hybrid":
                # Hybrid HNSW+IVF index
                m = self.hnsw_params.get("M", 16)
                nlist = self.ivf_params.get("nlist", 100)
                if quantize:
                    pq_m = self.quantization_params.get("pq_m", 8)
                    pq_nbits = self.quantization_params.get("pq_nbits", 8)
                    index_factory_str += f",HNSW{m}_IVF{nlist},PQ{pq_m}x{pq_nbits}"
                else:
                    index_factory_str += f",HNSW{m}_IVF{nlist},Flat"

            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

            logger.info(f"Creating index with factory string: {index_factory_str}")
            self.index = faiss.index_factory(
                self.dimension, index_factory_str, faiss.METRIC_INNER_PRODUCT
            )

            # Set HNSW parameters if applicable
            if self.index_type in ["hnsw", "hybrid"]:
                hnsw_index = (
                    faiss.extract_index_ivf(self.index)
                    if self.index_type == "hybrid"
                    else self.index
                )
                hnsw_index.hnsw.efConstruction = self.hnsw_params.get("efConstruction", 200)
                hnsw_index.hnsw.efSearch = self.hnsw_params.get("efSearch", 128)

            # Set IVF parameters if applicable
            if self.index_type in ["ivf", "hybrid"]:
                ivf_index = faiss.extract_index_ivf(self.index)
                ivf_index.nprobe = self.ivf_params.get("nprobe", 10)

    def _move_to_gpu(self) -> None:
        """
        Move the index to GPU if CUDA is available.

        This method attempts to move the index to the GPU for faster search.
        If CUDA is not available, it logs a warning and keeps the index on CPU.
        """
        try:
            if not enable_cuda():
                logger.warning("CUDA acceleration requested but not available")
                return

            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            logger.info("Successfully moved index to GPU")
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Add vectors to the index.

        Args:
            vectors: Numpy array of vectors to add (shape: [n, dimension])
            ids: Optional list of IDs for the vectors (auto-generated if None)
            metadata: Optional list of metadata dictionaries for each vector

        Returns:
            List[int]: List of IDs for the added vectors
        """
        with self._mutex:
            with Timer() as timer:
                n = vectors.shape[0]

                # Normalize vectors to unit length for cosine similarity
                vectors = normalize_vectors(vectors)

                # Generate IDs if not provided
                if ids is None:
                    next_id = self.metadata["count"]
                    ids = list(range(next_id, next_id + n))

                # Convert IDs to int64
                ids_array = np.array(ids, dtype=np.int64)

                # Add vectors to the index
                self.index.add_with_ids(vectors, ids_array)

                # Update metadata
                for i, id_val in enumerate(ids):
                    item_metadata = metadata[i] if metadata else {}
                    self.metadata["items"][str(id_val)] = item_metadata

                self.metadata["count"] += n
                self.metadata["updated_at"] = time.time()

            logger.debug(f"Added {n} vectors in {timer.elapsed:.4f}s")
            return ids

    def add_texts(
        self,
        texts: List[str],
        model: Any,  # EmbeddingModel, avoiding circular import
        ids: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
    ) -> List[int]:
        """
        Add texts to the index by first embedding them.

        Args:
            texts: List of text strings to embed and add
            model: Embedding model to use
            ids: Optional list of IDs for the vectors (auto-generated if None)
            metadata: Optional list of metadata dictionaries for each vector
            batch_size: Batch size for embedding

        Returns:
            List[int]: List of IDs for the added vectors
        """
        # Embed the texts in batches
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = model.embed(batch_texts)
            all_embeddings.append(batch_embeddings)

        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)

        # Add metadata for texts if not provided
        if metadata is None:
            metadata = [{"text": text} for text in texts]
        else:
            # Add text to existing metadata
            for i, text in enumerate(texts):
                if metadata[i] is None:
                    metadata[i] = {"text": text}
                else:
                    metadata[i]["text"] = text

        # Add to index
        return self.add(embeddings, ids, metadata)

    def search(
        self,
        query: Union[np.ndarray, str],
        model: Optional[Any] = None,  # EmbeddingModel, avoiding circular import
        top_k: int = 10,
        return_embeddings: bool = False,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the index for the nearest neighbors of the query vector.

        Args:
            query: Query vector or text string
            model: Embedding model (required if query is a string)
            top_k: Number of results to return
            return_embeddings: Whether to include embeddings in the results

        Returns:
            List[Tuple[float, Dict]]: List of tuples with (score, item_metadata)
        """
        if isinstance(query, str):
            if model is None:
                raise ValueError("Embedding model must be provided for text queries")
            query_vector = model.embed([query])[0]
        else:
            query_vector = query

        # Ensure query vector is normalized
        query_vector = normalize_vectors(query_vector.reshape(1, -1))[0]

        with self._mutex:
            # Set search parameters
            if self.index_type in ["ivf", "hybrid"]:
                ivf_index = faiss.extract_index_ivf(self.index)
                ivf_index.nprobe = self.ivf_params.get("nprobe", 10)

            if self.index_type in ["hnsw", "hybrid"]:
                hns
