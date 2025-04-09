# LlamaVector - Python Vector Store Library

[![PyPI version](https://img.shields.io/pypi/v/llama_vector.svg)](https://pypi.org/project/llama_vector/)
[![License](https://img.shields.io/github/license/llamasearchai/llama-vector)](https://github.com/llamasearchai/llama-vector/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/llama_vector.svg)](https://pypi.org/project/llama_vector/)

**LlamaVector** is a Python library providing tools for managing and searching vector embeddings. It offers a simple interface for adding, searching, deleting, and persisting vector data along with associated metadata.

**Note:** The current version provides a functional in-memory vector store with basic persistence via `pickle`. The core similarity search (`Index.search`) uses a brute-force calculation, suitable for smaller datasets or demonstration purposes. Optimized indexing (like HNSW, FAISS) is planned for future versions.

## Features

- **Vector Storage (`VectorStore`):**
    - Add single vectors or batches with associated metadata.
    - Search for the `k` nearest neighbors using cosine similarity.
    - Delete vectors by ID.
    - Persist the entire store to disk using `pickle` (`save`/`load`).
- **Embedding Utilities (`Embedding`):**
    - Calculate cosine similarity and Euclidean distance.
    - Normalize vectors.
    - Generate vector hashes (for potential deduplication).
- **Basic Indexing (`Index`):**
    - In-memory storage of vectors.
    - Brute-force cosine similarity search.
- **Query Abstraction (`Query`):**
    - Define queries with optional metadata filters (Note: Filtering logic needs integration with `VectorStore.search`).

## Installation

```bash
# Install from PyPI (once published)
# pip install llama-vector

# Or install directly from GitHub for the latest version:
pip install git+https://github.com/llamasearchai/llama-vector.git
```

## Quick Start

```python
import numpy as np
from llama_vector import VectorStore, Embedding

# 1. Initialize a VectorStore (e.g., for 128-dimensional vectors)
store = VectorStore(dimension=128)

# 2. Add vectors with IDs and metadata
vectors_to_add = {
    \"doc_a\": {\"vector\": list(np.random.rand(128)), \"metadata\": {\"title\": \"Document A\", \"category\": \"news\"}},
    \"doc_b\": {\"vector\": list(np.random.rand(128)), \"metadata\": {\"title\": \"Document B\", \"category\": \"tech\"}},
    \"doc_c\": {\"vector\": list(np.random.rand(128)), \"metadata\": {\"title\": \"Document C\", \"category\": \"news\"}}
}

for vec_id, data in vectors_to_add.items():
    store.add(id=vec_id, vector=data[\"vector\"], metadata=data[\"metadata\"])

print(f\"Store size: {len(store.embeddings)} vectors\")

# 3. Search for similar vectors
query_vector = list(np.random.rand(128))
print(\"\nSearching for vectors similar to query vector...\")
top_k = 2
results = store.search(query_vector=query_vector, k=top_k)

print(f\"Top {top_k} results:\")
for result in results:
    print(f\"  ID: {result['id']}, Score: {result['score']:.4f}, Metadata: {result['metadata']}\")

# 4. Use Embedding utilities
vec1 = vectors_to_add[\"doc_a\"][\"vector\"]
vec2 = vectors_to_add[\"doc_b\"][\"vector\"]
similarity = Embedding.cosine_similarity(vec1, vec2)
print(f\"\nCosine similarity between doc_a and doc_b: {similarity:.4f}\")

# 5. Save and Load
store_path = \"./my_vector_store.pkl\"
print(f\"\nSaving store to {store_path}...\")
store.save(store_path)

print(\"Loading store from disk...\")
loaded_store = VectorStore.load(store_path)
print(f\"Loaded store size: {len(loaded_store.embeddings)} vectors\")

# Verify loaded data
results_loaded = loaded_store.search(query_vector=query_vector, k=top_k)
assert len(results_loaded) == len(results)
print(\"Search results from loaded store match.\")

# Clean up saved file (optional)
import os
os.remove(store_path)
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llama-vector.git
cd llama-vector

# Recommended: Create and activate a virtual environment
# python -m venv venv
# source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`

# Install in editable mode with development dependencies
# Requires numpy for core functionality, add dev tools as needed
pip install -e \".[dev]\"
# Or using uv (faster):
# uv pip install -e \".[dev]\"
```

### Testing

Tests are located in the `tests/` directory and use `pytest`.

```bash
# Run tests
pytest tests/
# Or using uv:
# uv run pytest tests/
```

### Linting and Formatting

We use `ruff` for linting and formatting.

```bash
# Check for linting errors
ruff check .
# Or using uv:
# uv run ruff check .

# Format code
ruff format .
# Or using uv:
# uv run ruff format .
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Consider opening an issue first to discuss potential changes, especially regarding adding optimized indexing methods or integrating metadata filtering into the search.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by lalamasearhc.*
