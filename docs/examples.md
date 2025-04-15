# Examples

This page provides examples of using Llama Vector for various tasks.

## Basic Usage

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

## Advanced Usage

```python
from llama_vector import VectorStore
import numpy as np

# Create a vector store with custom settings
store = VectorStore(
    dimension=128,
    metric="cosine",
    index_type="hnsw",
    ef_construction=200,
    M=16
)

# Generate some random vectors
num_vectors = 1000
vectors = np.random.rand(num_vectors, 128).astype(np.float32)

# Add vectors in batch
ids = [f"doc{i}" for i in range(num_vectors)]
metadata = [{
    "title": f"Document {i}",
    "category": f"Category {i % 5}"
} for i in range(num_vectors)]

store.add_batch(ids, vectors, metadata)

# Perform search with filtering
query_vector = np.random.rand(128).astype(np.float32)
results = store.search(
    query_vector,
    k=10,
    filter_criteria={"category": "Category 3"}
)

# Print results
for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}, Title: {result['metadata']['title']}")
```

For more examples, check out the [examples directory](https://github.com/llamasearchai/llama-vector/tree/main/examples) in the GitHub repository.
