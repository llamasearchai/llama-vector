"""
Vector index implementation
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import heapq


class Index:
    """
    A simple in-memory vector index for similarity search.

    Currently implements brute-force search by calculating cosine similarity 
    against all stored vectors for each query. This is suitable for small datasets
    but does not scale well.

    Future improvements could include integrating optimized indexing libraries
    like Faiss or HNSWlib based on the `index_type`.

    Attributes:
        dimension (int): The dimensionality of the vectors.
        index_type (str): The type of index specified (e.g., "hnsw"). 
                          Currently informational only.
        vectors (Dict[str, np.ndarray]): A dictionary storing vectors keyed by ID.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initializes the Index.

        Args:
            dimension: The expected dimension of the vectors to be stored.
            index_type: The type of index strategy (e.g., "flat", "hnsw"). 
                        Currently informational, search defaults to flat/brute-force.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.vectors = {}  # id -> vector mapping
        
        # Placeholder for future: Initialize specific index based on index_type
        # if self.index_type == "hnsw":
        #     self._init_hnsw()
        # elif self.index_type == "faiss":
        #     self._init_faiss()
        # else: # Default to flat/brute-force
        #     pass
    
    def add(self, id: str, vector: np.ndarray) -> None:
        """
        Adds or updates a vector in the index.

        Args:
            id: The unique identifier for the vector.
            vector: The vector as a numpy array.

        Raises:
            ValueError: If the vector's dimension does not match the index dimension.
        """
        if vector.shape == (self.dimension,):
            self.vectors[id] = vector
            # If using a real index library, add vector to it here
            # e.g., self.hnsw_index.add_items(vector.reshape(1, -1), [int_id_representation])
        elif vector.shape == (1, self.dimension):
             # Accept (1, D) shape as well
             self.vectors[id] = vector.reshape(self.dimension,)
        else:
             raise ValueError(
                 f"Vector dimension mismatch. Expected ({self.dimension},) or (1, {self.dimension}), "
                 f"got {vector.shape}"
             )
            
    def delete(self, id: str) -> bool:
        """
        Deletes a vector from the index by its ID.

        Args:
            id: The ID of the vector to delete.

        Returns:
            True if the vector was found and deleted, False otherwise.
        """
        if id in self.vectors:
            del self.vectors[id]
            # If using a real index library, remove/mark vector as deleted here
            # e.g., self.hnsw_index.mark_deleted(int_id_representation)
            return True
        return False
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Finds the k nearest neighbors to the query vector using cosine similarity.

        NOTE: This performs a brute-force search by comparing the query vector
              to every vector currently in the index. It is not efficient for
              large numbers of vectors.

        Args:
            query_vector: The query vector as a numpy array.
            k: The number of nearest neighbors to return.

        Returns:
            A list of tuples, where each tuple contains the ID and the 
            cosine similarity score of a neighbor, sorted by score descending.
            Returns an empty list if the index is empty.
        
        Raises:
            ValueError: If the query vector's dimension does not match the index dimension.
        """
        if len(self.vectors) == 0:
            return []
            
        # Validate query vector dimension
        if query_vector.shape != (self.dimension,):
            if query_vector.shape == (1, self.dimension):
                query_vector = query_vector.reshape(self.dimension,) # Reshape if needed
            else:
                 raise ValueError(
                     f"Query vector dimension mismatch. Expected ({self.dimension},), "
                     f"got {query_vector.shape}"
                 )
            
        # --- Brute-Force Cosine Similarity Search --- 
        results_heap: List[Tuple[float, str]] = [] # Use min-heap (score, id)
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return [] # Cannot compute similarity with zero vector

        for vec_id, vector in self.vectors.items():
            vec_norm = np.linalg.norm(vector)
            if vec_norm == 0:
                 similarity = 0.0
            else:
                # Calculate cosine similarity: dot(q, v) / (norm(q) * norm(v))
                similarity = float(np.dot(query_vector, vector) / (query_norm * vec_norm))
                similarity = np.clip(similarity, -1.0, 1.0) # Clip for safety
            
            # Use a min-heap to keep track of the top k largest scores
            if len(results_heap) < k:
                heapq.heappush(results_heap, (similarity, vec_id))
            elif similarity > results_heap[0][0]: # If current sim > smallest in heap
                 heapq.heapreplace(results_heap, (similarity, vec_id))
                 
        # Convert heap to sorted list (descending score)
        # Heap elements are (score, id), we want (id, score)
        sorted_results = sorted([(id, score) for score, id in results_heap], key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def get_nearest_neighbors(self, id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Finds the k nearest neighbors for a vector already present in the index.

        Excludes the vector itself from the results.
        Uses the same brute-force search mechanism as `search()`.

        Args:
            id: The ID of the vector within the index to find neighbors for.
            k: The number of nearest neighbors to return.

        Returns:
            A list of tuples (neighbor_id, score), sorted by score descending.
        
        Raises:
            ValueError: If the provided ID is not found in the index.
        """
        if id not in self.vectors:
            raise ValueError(f"Vector with ID '{id}' not found in index")
            
        query_vector = self.vectors[id]
        
        # Perform search, asking for k+1 results initially
        results = self.search(query_vector, k + 1)
        
        # Filter out the query vector itself from the results
        filtered_results = [(res_id, score) for res_id, score in results if res_id != id]
        
        # Return the top k from the filtered list
        return filtered_results[:k]
