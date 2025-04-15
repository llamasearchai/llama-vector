"""Basic tests for the llamavector package."""

import pytest


def test_import():
    """Test that the main package can be imported."""
    try:
        import llamavector
    except ImportError as e:
        pytest.fail(f"Failed to import llamavector: {e}")


def test_version():
    """Test that the package has a version attribute."""
    import llamavector

    assert hasattr(llamavector, "__version__")
    assert isinstance(llamavector.__version__, str)


# Add more core tests later, e.g.:
# - Test embedding model loading (mocked)
# - Test vector store adapter instantiation (mocked)
# - Test basic add/search operations (mocked or with in-memory stores like FAISS/Chroma)
