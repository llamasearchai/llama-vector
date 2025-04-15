# Llama Vector

High-performance vector database for AI embeddings with similarity search capabilities

[![GitHub](https://img.shields.io/github/license/llamasearchai/llama-vector)](https://github.com/llamasearchai/llama-vector/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llama_vector.svg)](https://pypi.org/project/llama_vector/)

## Overview


High-performance vector database for AI embeddings with similarity search capabilities. This library provides a comprehensive set of tools and utilities for
working with vector tasks in AI and data processing workflows.
It's designed to be easy to use while offering powerful capabilities for complex scenarios.


## Features


- **High-Performance Vector Storage**: Efficiently store and retrieve high-dimensional vectors
- **Similarity Search**: Fast nearest-neighbor search for recommendation and retrieval
- **Multiple Distance Metrics**: Support for cosine, Euclidean, and other distance functions
- **Batched Operations**: Process large sets of vectors efficiently
- **Persistence Options**: Save and load your vector database as needed


## Installation

```bash
pip install llama_vector
```

## Usage

```python

from llama_vector import VectorStore

# Create a vector store
store = VectorStore(dimension=768)

# Add vectors
store.add("doc1", [0.1, 0.2, ...], metadata={"title": "Document 1"})
store.add("doc2", [0.3, 0.4, ...], metadata={"title": "Document 2"})

# Search for similar vectors
results = store.search([0.1, 0.3, ...], k=5)
for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}")

```

## Documentation

For more detailed documentation, see the [docs](docs/) directory.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
