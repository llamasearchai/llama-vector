"""Pytest configuration for testing"""
import pytest


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing"""
    return {"timeout": 30, "retries": 3, "log_level": "debug"}


@pytest.fixture
def sample_data():
    """Provide sample data for testing"""
    return {"key": "value", "nested": {"inner": "data"}}


@pytest.fixture
def sample_vectors():
    """Provide sample vectors for testing"""
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0]
    ]
