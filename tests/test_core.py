"""Tests for core functionality"""
import pytest
from llama_vector.core import Client


def test_client_initialization():
    """Test that client initializes properly"""
    client = Client()
    assert client is not None
    assert client.config == {}
    
    # Test with config
    config = {"timeout": 60}
    client = Client(config=config)
    assert client.config == config


def test_client_process():
    """Test processing functionality"""
    client = Client()
    result = client.process("test data")
    assert isinstance(result, dict)
    assert "result" in result
    assert "input" in result
    assert result["input"] == "test data"


def test_client_status():
    """Test status functionality"""
    client = Client()
    status = client.get_status()
    assert isinstance(status, dict)
    assert "status" in status


def test_add_vector():
    """Test adding a vector"""
    client = Client()
    vector = [0.1, 0.2, 0.3]
    vector_id = client.add_vector(vector)
    assert isinstance(vector_id, str)
    
    # Test with metadata
    vector_id = client.add_vector(vector, {"name": "test"})
    assert isinstance(vector_id, str)
    
    
def test_search_similar():
    """Test searching for similar vectors"""
    client = Client()
    vector = [0.1, 0.2, 0.3]
    results = client.search_similar(vector, k=5)
    assert isinstance(results, list)
    assert len(results) == 5
    assert "id" in results[0]
    assert "score" in results[0]
