"""
In-memory vector store implementation for LlamaVector.
"""
import numpy as np
import os
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple

from .embedding import Embedding
from .index import Index
# from .query import Query # Query class is defined but not currently used by search


class VectorStore:
    """
    An in-memory storage and retrieval engine for vector embeddings.

    Manages vector embeddings and associated metadata, providing methods for
    adding, searching (via an Index object), deleting, and persisting data.

    Attributes:
        dimension (int): The dimensionality of the vectors stored.
        index_type (str): The type of index used (currently informational, as 
                          Index uses brute-force search).
        embeddings (Dict[str, np.ndarray]): Dictionary mapping vector IDs to numpy arrays.
        metadata (Dict[str, Dict[str, Any]]): Dictionary mapping vector IDs to metadata dicts.
        index (Index): The index object used for searching.
    """
    
    def __init__(self, dimension: int = 768, index_type: str = "hnsw"):
        """
        Initializes the VectorStore.

        Args:
            dimension: The expected dimension of the vectors.
            index_type: The type of index to use (e.g., "hnsw", "flat"). 
                        Note: Currently only affects metadata; search is brute-force.
        """
        self.dimension: int = dimension
        self.index_type: str = index_type
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        # The Index class handles the actual search implementation
        self.index: Index = Index(dimension, index_type) 
        
    def add(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a single vector embedding and its optional metadata to the store.

        Args:
            id: A unique identifier for the vector.
            vector: The vector embedding as a list of floats.
            metadata: An optional dictionary of metadata associated with the vector.

        Raises:
            ValueError: If the provided vector's dimension does not match the store's dimension.
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")
            
        # Store embeddings as float32 numpy arrays for consistency
        numpy_vector = np.array(vector, dtype=np.float32)
        self.embeddings[id] = numpy_vector
        if metadata:
            self.metadata[id] = metadata
        else:
             # Ensure metadata entry exists even if None provided
            self.metadata[id] = {}
            
        # Add to the index for searching
        self.index.add(id, numpy_vector)
        
    def add_batch(self, ids: List[str], vectors: List[List[float]], metadatas: Optional[List[Optional[Dict[str, Any]]]] = None) -> None:
        """
        Adds a batch of vector embeddings and their optional metadata.

        Args:
            ids: A list of unique identifiers for the vectors.
            vectors: A list of vector embeddings.
            metadatas: An optional list of metadata dictionaries. If provided, its
                       length must match the ids and vectors lists.

        Raises:
            ValueError: If the lengths of ids, vectors, and metadatas (if provided)
                        do not match.
            ValueError: If any vector's dimension does not match the store's dimension.
        """
        if len(ids) != len(vectors):
            raise ValueError("Number of IDs and vectors must match")
            
        if metadatas and len(ids) != len(metadatas):
            raise ValueError("Number of IDs and metadata items must match")
            
        # Validate dimensions first
        for vector in vectors:
             if len(vector) != self.dimension:
                 raise ValueError(f"Vector dimension mismatch in batch. Expected {self.dimension}, got {len(vector)}")

        # Add items one by one (can be optimized later if Index supports batch add)
        for i, (id, vector) in enumerate(zip(ids, vectors)):
            metadata_item = metadatas[i] if metadatas else None
            numpy_vector = np.array(vector, dtype=np.float32)
            
            self.embeddings[id] = numpy_vector
            self.metadata[id] = metadata_item or {}
            self.index.add(id, numpy_vector)
            
    def search(self, query_vector: List[float], k: int = 10, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Searches the index for the k vectors most similar to the query vector.

        Note: Currently uses brute-force cosine similarity search via the Index class.
              Metadata filtering is not yet implemented in this method.

        Args:
            query_vector: The vector embedding to search for.
            k: The number of nearest neighbors to return.
            include_metadata: If True, includes the metadata associated with each
                              resulting vector.

        Returns:
            A list of dictionaries, each containing the 'id', 'score', and 
            optionally 'metadata' of a similar vector, sorted by score descending.
        
        Raises:
            ValueError: If the query vector's dimension does not match the store's dimension.
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch. Expected {self.dimension}, got {len(query_vector)}")
            
        query_vec_np = np.array(query_vector, dtype=np.float32)
        # Perform search using the index
        results: List[Tuple[str, float]] = self.index.search(query_vec_np, k)
        
        # Format results
        formatted_results: List[Dict[str, Any]] = []
        if include_metadata:
            for id, score in results:
                formatted_results.append({
                    "id": id,
                    "score": score,
                    "metadata": self.metadata.get(id, {}) # Safely get metadata
                })
        else:
            for id, score in results:
                 formatted_results.append({
                    "id": id,
                    "score": score
                })
                
        return formatted_results
            
    def delete(self, id: str) -> bool:
        """
        Deletes a vector and its metadata from the store and index by ID.

        Args:
            id: The unique identifier of the vector to delete.

        Returns:
            True if the vector was found and deleted, False otherwise.
        """
        if id in self.embeddings:
            del self.embeddings[id]
            if id in self.metadata:
                del self.metadata[id] # Also remove metadata
                
            # Remove from the index
            deleted_from_index = self.index.delete(id)
            # Log a warning if deletion from index failed unexpectedly
            if not deleted_from_index:
                 # Consider adding logging here
                 pass 
            return True
        return False
        
    def save(self, path: str) -> None:
        """
        Saves the current state of the vector store to a file using pickle.

        Args:
            path: The file path where the store should be saved.
        """
        # Ensure the directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            
        # Data to save
        data_to_save = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'embeddings': self.embeddings, # Dict[str, np.ndarray]
            'metadata': self.metadata      # Dict[str, Dict]
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(data_to_save, f)
        except Exception as e:
             # Consider adding logging here
            raise IOError(f"Failed to save vector store to {path}: {e}") from e
            
    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """
        Loads a vector store from a file previously saved using the .save() method.

        Args:
            path: The file path from which to load the store.

        Returns:
            A new VectorStore instance populated with the loaded data.
        
        Raises:
            IOError: If the file cannot be loaded or data is corrupt.
        """
        try:
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
        except FileNotFoundError:
            raise IOError(f"Vector store file not found: {path}")
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            raise IOError(f"Failed to load vector store from {path}: {e}") from e
        
        # Basic validation of loaded data structure
        required_keys = ['dimension', 'index_type', 'embeddings', 'metadata']
        if not all(key in loaded_data for key in required_keys):
            raise IOError(f"Invalid data format in vector store file: {path}")

        # Create a new store instance
        store = cls(dimension=loaded_data['dimension'], index_type=loaded_data['index_type'])
        store.embeddings = loaded_data['embeddings']
        store.metadata = loaded_data['metadata']
        
        # Rebuild the index from loaded embeddings
        # This is necessary because the Index object itself isn't saved
        if not isinstance(store.embeddings, dict):
             raise IOError(f"Invalid embeddings format in vector store file: {path}")

        for id, vector in store.embeddings.items():
            if isinstance(vector, np.ndarray):
                store.index.add(id, vector)
            else:
                 # Handle potential loading inconsistencies if needed
                 store.index.add(id, np.array(vector, dtype=np.float32))
            
        return store
