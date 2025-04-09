"""
Query representation and filtering utilities for LlamaVector.
"""
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np


class Query:
    """
    Represents a query to be performed on the VectorStore.

    Encapsulates a query vector and metadata filters.
    Provides methods for building the query and checking if metadata matches 
    the defined filters.

    Note: The filter matching logic in `matches_filters` is implemented, 
          but it needs to be integrated into the `VectorStore.search` method 
          to actually perform filtered searches.
    """
    
    def __init__(self, vector: Optional[List[float]] = None, filters: Optional[Dict[str, Any]] = None):
        """
        Initializes a Query object.

        Args:
            vector: The optional query vector embedding.
            filters: An optional dictionary defining metadata filters. 
                     Keys are metadata fields, values are the expected values 
                     (or lists of values, or callables for custom logic).
        """
        self.vector: Optional[List[float]] = vector
        self.filters: Dict[str, Any] = filters or {}
        
    def set_vector(self, vector: List[float]) -> 'Query':
        """
        Sets or updates the query vector.

        Args:
            vector: The query vector embedding.

        Returns:
            The Query instance for method chaining.
        """
        self.vector = vector
        return self
        
    def set_filter(self, key: str, value: Any) -> 'Query':
        """
        Adds or updates a single metadata filter condition.

        Args:
            key: The metadata field key to filter on.
            value: The value to filter by. Can be:
                   - A single value for exact match.
                   - A list of values for an "in" match.
                   - A callable (function) that takes the metadata value and 
                     returns True (match) or False (no match).

        Returns:
            The Query instance for method chaining.
        """
        self.filters[key] = value
        return self
        
    def set_filters(self, filters: Dict[str, Any]) -> 'Query':
        """
        Adds or updates multiple metadata filters.

        Args:
            filters: A dictionary of filter conditions.

        Returns:
            The Query instance for method chaining.
        """
        self.filters.update(filters)
        return self
        
    def clear_filters(self) -> 'Query':
        """
        Removes all currently defined metadata filters.

        Returns:
            The Query instance for method chaining.
        """
        self.filters = {}
        return self
        
    def matches_filters(self, metadata: Optional[Dict[str, Any]]) -> bool:
        """
        Checks if the provided metadata dictionary matches all defined filters.

        Args:
            metadata: The metadata dictionary of a vector to check.

        Returns:
            True if the metadata matches all filters (or if no filters are set),
            False otherwise.
        """
        # If no filters are set, it's always a match
        if not self.filters:
            return True
        
        # If metadata is None or empty, it cannot match any filters (unless filters are also empty)
        if not metadata:
             return False
            
        # Check each filter condition
        for key, filter_value in self.filters.items():
            # If the required key is missing in the metadata, it's not a match
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # Handle different filter value types
            if isinstance(filter_value, list):
                # Filter by list of allowed values ("in" condition)
                if metadata_value not in filter_value:
                    return False
            elif callable(filter_value):
                # Filter by a custom function
                try:
                    if not filter_value(metadata_value):
                        return False
                except Exception:
                     # If the callable raises an error, treat as non-match
                     return False 
            else:
                # Filter by exact match
                if metadata_value != filter_value:
                    return False
                    
        # If all filter conditions passed, it's a match
        return True
