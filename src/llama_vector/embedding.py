"""
Embedding utility functions for vector operations.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
import hashlib
import json


class Embedding:
    """
    Provides static utility methods for common vector operations like normalization,
    similarity/distance calculation, and hashing.
    """
    
    @staticmethod
    def normalize(vector: List[float]) -> List[float]:
        """
        Normalizes a vector to unit length (L2 norm).

        Args:
            vector: The vector as a list of floats.

        Returns:
            The normalized vector as a list of floats. If the input vector 
            has zero magnitude, the original vector is returned.
        """
        # Convert to numpy array for efficient calculation
        np_vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(np_vector)
        if norm == 0:
            # Return the original list if norm is zero
            return vector
        # Perform normalization and convert back to list
        return (np_vector / norm).tolist()
    
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Cosine similarity measures the cosine of the angle between two non-zero
        vectors, indicating their orientation similarity.

        Args:
            vector1: The first vector as a list of floats.
            vector2: The second vector as a list of floats.

        Returns:
            The cosine similarity score, ranging from -1 (opposite direction)
            to 1 (same direction). Returns 0.0 if either vector has zero magnitude.
        
        Raises:
            ValueError: If the input vectors have different dimensions.
        """
        if len(vector1) != len(vector2):
             raise ValueError("Vectors must have the same dimension for cosine similarity")

        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        
        # Calculate norms
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # Handle zero vectors to avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity: dot(v1, v2) / (norm(v1) * norm(v2))
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        # Clip value to handle potential floating point inaccuracies near -1 or 1
        return float(np.clip(similarity, -1.0, 1.0))
    
    @staticmethod
    def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
        """
        Calculates the Euclidean distance (L2 distance) between two vectors.

        Args:
            vector1: The first vector as a list of floats.
            vector2: The second vector as a list of floats.

        Returns:
            The Euclidean distance between the two vectors.
        
        Raises:
            ValueError: If the input vectors have different dimensions.
        """
        if len(vector1) != len(vector2):
             raise ValueError("Vectors must have the same dimension for Euclidean distance")
             
        v1 = np.array(vector1, dtype=np.float32)
        v2 = np.array(vector2, dtype=np.float32)
        # Calculate the norm of the difference vector
        return float(np.linalg.norm(v1 - v2))
    
    @staticmethod
    def hash_vector(vector: List[float], precision: int = 6) -> str:
        """
        Creates a stable SHA256 hash of a vector, useful for deduplication.

        Rounds the vector components to a specified precision before hashing
        to mitigate floating-point inaccuracies.

        Args:
            vector: The vector as a list of floats.
            precision: The number of decimal places to round each component to
                       before hashing. Defaults to 6.

        Returns:
            A hexadecimal string representing the SHA256 hash of the rounded vector.
        """
        # Round to reduce floating point differences affecting the hash
        try:
            rounded = [round(v, precision) for v in vector]
            # Convert rounded list to a JSON string for consistent serialization
            vector_bytes = json.dumps(rounded, sort_keys=True).encode('utf-8') # sort_keys for added stability
            return hashlib.sha256(vector_bytes).hexdigest()
        except TypeError as e:
             raise TypeError(f"Could not hash vector due to invalid data type: {e}") from e
