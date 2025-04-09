"""Tests for vector store functionality"""
import pytest
import numpy as np
import os
import tempfile
import pickle
from pathlib import Path
from llama_vector.vector_store import VectorStore
from llama_vector.embedding import Embedding

DIMENSION = 3 # Define dimension for consistency in tests

@pytest.fixture
def sample_vectors() -> List[List[float]]:
    """Sample vectors for testing."""
    return [
        [1.0, 0.0, 0.0],  # First basis vector
        [0.0, 1.0, 0.0],  # Second basis vector
        [0.0, 0.0, 1.0],  # Third basis vector
        [0.7, 0.7, 0.0],  # Vector in first quadrant (normalized approx)
    ]

@pytest.fixture
def sample_store(sample_vectors) -> VectorStore:
    """Sample vector store populated with vectors for testing."""
    store = VectorStore(dimension=DIMENSION)
    
    # Add vectors with metadata
    store.add("vec1", sample_vectors[0], {"name": "First vector", "category": "basis", "index": 1})
    store.add("vec2", sample_vectors[1], {"name": "Second vector", "category": "basis", "index": 2})
    store.add("vec3", sample_vectors[2], {"name": "Third vector", "category": "basis", "index": 3})
    store.add("vec4", sample_vectors[3], {"name": "Fourth vector", "category": "derived", "index": 4})
    
    return store

@pytest.fixture
def empty_store() -> VectorStore:
    """An empty vector store."""
    return VectorStore(dimension=DIMENSION)

@pytest.fixture
def temp_file_path() -> Path:
    """Create a temporary file path and ensure cleanup."""
    # Use NamedTemporaryFile to get a unique path, delete it initially,
    # so VectorStore can create it during save.
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        path = Path(temp_file.name)
    yield path
    # Cleanup after test
    if path.exists():
        path.unlink()

def test_vector_store_initialization():
    """Test vector store initialization with default and custom index type."""
    store_default = VectorStore(dimension=128)
    assert store_default.dimension == 128
    assert store_default.index_type == "flat" # Check default index type
    assert len(store_default.embeddings) == 0
    assert len(store_default.metadata) == 0
    assert store_default.index is not None
    assert store_default.index.dimension == 128

    store_custom = VectorStore(dimension=DIMENSION, index_type="custom_test")
    assert store_custom.dimension == DIMENSION
    assert store_custom.index_type == "custom_test" # Check custom index type storage

def test_add_vector_success(empty_store, sample_vectors):
    """Test successfully adding a single vector."""
    vector_id = "test_vec"
    vector_data = sample_vectors[0]
    metadata = {"test": "metadata", "value": 123}
    
    empty_store.add(vector_id, vector_data, metadata)
    
    assert vector_id in empty_store.embeddings
    assert vector_id in empty_store.metadata
    assert vector_id in empty_store.index.vectors # Check index inclusion
    assert empty_store.metadata[vector_id] == metadata
    np.testing.assert_array_almost_equal(empty_store.embeddings[vector_id], np.array(vector_data, dtype=np.float32))
    np.testing.assert_array_almost_equal(empty_store.index.vectors[vector_id], np.array(vector_data, dtype=np.float32))

def test_add_vector_no_metadata(empty_store, sample_vectors):
    """Test adding a vector without metadata."""
    vector_id = "no_meta_vec"
    vector_data = sample_vectors[1]
    
    empty_store.add(vector_id, vector_data, None)
    
    assert vector_id in empty_store.embeddings
    assert vector_id in empty_store.metadata # Should still have an entry
    assert empty_store.metadata[vector_id] == {} # Should be an empty dict
    assert vector_id in empty_store.index.vectors

def test_add_vector_dimension_mismatch(empty_store):
    """Test adding a vector with the wrong dimension raises ValueError."""
    wrong_dim_vector = [1.0, 2.0] # Expecting DIMENSION (3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        empty_store.add("wrong_dim", wrong_dim_vector)

def test_add_batch_success(empty_store, sample_vectors):
    """Test adding vectors successfully in a batch."""
    ids = ["batch_vec1", "batch_vec2", "batch_vec3"]
    vectors = sample_vectors[:3]
    metadatas = [
        {"name": "First Batch"},
        None, # Test with None metadata in batch
        {"name": "Third Batch"}
    ]
    
    empty_store.add_batch(ids, vectors, metadatas)
    
    assert len(empty_store.embeddings) == 3
    assert len(empty_store.metadata) == 3
    assert len(empty_store.index.vectors) == 3
    assert "batch_vec1" in empty_store.embeddings
    assert "batch_vec2" in empty_store.metadata
    assert empty_store.metadata["batch_vec2"] == {} # Check None metadata handling
    assert "batch_vec3" in empty_store.index.vectors
    np.testing.assert_array_almost_equal(empty_store.embeddings["batch_vec1"], np.array(vectors[0], dtype=np.float32))

def test_add_batch_mismatched_lengths(empty_store, sample_vectors):
    """Test add_batch raises ValueError if list lengths don't match."""
    ids = ["id1", "id2"]
    vectors = sample_vectors[:3] # Length 3
    with pytest.raises(ValueError, match="IDs and vectors must match"):
        empty_store.add_batch(ids, vectors)
        
    vectors_correct = sample_vectors[:2] # Length 2
    metadatas_wrong = [{}] # Length 1
    with pytest.raises(ValueError, match="IDs and metadata items must match"):
        empty_store.add_batch(ids, vectors_correct, metadatas_wrong)

def test_add_batch_dimension_mismatch(empty_store, sample_vectors):
    """Test add_batch raises ValueError if a vector has the wrong dimension."""
    ids = ["id1", "id2"]
    vectors_wrong_dim = [
        sample_vectors[0], # Correct dimension
        [1.0, 2.0]       # Incorrect dimension
    ]
    with pytest.raises(ValueError, match="dimension mismatch in batch"):
        empty_store.add_batch(ids, vectors_wrong_dim)

def test_search_success_with_metadata(sample_store):
    """Test vector search returns correct results with metadata."""
    # Query vector close to vec1 [1.0, 0.0, 0.0]
    query = [0.9, 0.1, 0.1]
    k = 2
    results = sample_store.search(query, k=k, include_metadata=True)
    
    assert len(results) == k
    # Check sorting (highest score first)
    assert results[0]["score"] >= results[1]["score"]
    # Check expected top result
    assert results[0]["id"] == "vec1" 
    assert results[0]["score"] > 0.9 # Cosine similarity should be high
    # Check metadata inclusion
    assert "metadata" in results[0]
    assert isinstance(results[0]["metadata"], dict)
    assert results[0]["metadata"] == {"name": "First vector", "category": "basis", "index": 1}
    assert "metadata" in results[1]

def test_search_success_without_metadata(sample_store):
    """Test vector search returns correct results without metadata."""
    query = [0.1, 0.9, 0.1] # Close to vec2
    k = 1
    results = sample_store.search(query, k=k, include_metadata=False)
    
    assert len(results) == k
    assert results[0]["id"] == "vec2" 
    assert results[0]["score"] > 0.9
    # Check metadata is NOT included
    assert "metadata" not in results[0]

def test_search_empty_store(empty_store):
    """Test searching an empty store returns an empty list."""
    query = [0.1, 0.2, 0.3]
    results = empty_store.search(query, k=5)
    assert isinstance(results, list)
    assert len(results) == 0

def test_search_dimension_mismatch(sample_store):
    """Test searching with a query vector of wrong dimension raises ValueError."""
    wrong_dim_query = [1.0, 2.0]
    with pytest.raises(ValueError, match="Query vector dimension mismatch"):
        sample_store.search(wrong_dim_query)

def test_search_k_greater_than_store_size(sample_store):
    """Test searching with k larger than the number of items in the store."""
    query = [0.5, 0.5, 0.5]
    k = 10 # Store only has 4 items
    results = sample_store.search(query, k=k)
    assert len(results) == 4 # Should return all items
    # Check sorting
    assert all(results[i]["score"] >= results[i+1]["score"] for i in range(len(results)-1))

def test_delete_success(sample_store):
    """Test deleting an existing vector returns True and removes it."""
    vector_id_to_delete = "vec1"
    assert vector_id_to_delete in sample_store.embeddings
    assert vector_id_to_delete in sample_store.index.vectors
    
    delete_result = sample_store.delete(vector_id_to_delete)
    
    assert delete_result is True
    assert vector_id_to_delete not in sample_store.embeddings
    assert vector_id_to_delete not in sample_store.metadata
    assert vector_id_to_delete not in sample_store.index.vectors # Check removal from index

def test_delete_non_existent(sample_store):
    """Test deleting a non-existent vector returns False."""
    non_existent_id = "non_existent_vec"
    assert non_existent_id not in sample_store.embeddings
    
    delete_result = sample_store.delete(non_existent_id)
    
    assert delete_result is False

def test_save_load_cycle(sample_store, temp_file_path):
    """Test that saving and loading preserves the store state."""
    original_embeddings = sample_store.embeddings.copy()
    original_metadata = sample_store.metadata.copy()
    original_dimension = sample_store.dimension
    original_index_type = sample_store.index_type

    # Save the store
    sample_store.save(str(temp_file_path))
    assert temp_file_path.exists()
    
    # Load into a new store instance
    loaded_store = VectorStore.load(str(temp_file_path))
    
    # Check loaded attributes
    assert loaded_store.dimension == original_dimension
    assert loaded_store.index_type == original_index_type
    assert loaded_store.embeddings.keys() == original_embeddings.keys()
    assert loaded_store.metadata.keys() == original_metadata.keys()
    assert loaded_store.index.vectors.keys() == original_embeddings.keys() # Index rebuilt
    
    # Check content integrity
    for vec_id in original_embeddings:
        np.testing.assert_array_almost_equal(
            loaded_store.embeddings[vec_id],
            original_embeddings[vec_id]
        )
        assert loaded_store.metadata[vec_id] == original_metadata[vec_id]
        np.testing.assert_array_almost_equal(
            loaded_store.index.vectors[vec_id], # Check index content
            original_embeddings[vec_id]
        )

    # Check search functionality after loading
    query = [0.9, 0.1, 0.1]
    original_results = sample_store.search(query, k=2)
    loaded_results = loaded_store.search(query, k=2)
    assert original_results == loaded_results

def test_load_file_not_found(temp_file_path):
    """Test loading from a non-existent file raises IOError."""
    # Ensure file does not exist initially
    if temp_file_path.exists():
        temp_file_path.unlink()
        
    with pytest.raises(IOError, match="not found"):
        _ = VectorStore.load(str(temp_file_path))

def test_load_corrupt_file(temp_file_path):
    """Test loading from a corrupt/invalid pickle file raises IOError."""
    # Write invalid data to the file
    with open(temp_file_path, "wb") as f:
        f.write(b"this is not pickle data")
        
    with pytest.raises(IOError, match="Failed to load"):
        _ = VectorStore.load(str(temp_file_path))

def test_load_incomplete_data(temp_file_path):
    """Test loading a pickle file with missing keys raises IOError."""
    # Save incomplete data
    incomplete_data = {
        'dimension': DIMENSION,
        # Missing 'index_type', 'embeddings', 'metadata'
    }
    with open(temp_file_path, "wb") as f:
        pickle.dump(incomplete_data, f)
        
    with pytest.raises(IOError, match="Invalid data format"):
        _ = VectorStore.load(str(temp_file_path))
