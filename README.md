# llama-vector

[![PyPI version](https://badge.fury.io/py/llamavector.svg)](https://badge.fury.io/py/llamavector)
[![Python Version](https://img.shields.io/pypi/pyversions/llamavector.svg)](https://pypi.org/project/llamavector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yourusername/llamavector-pkg/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/llamavector-pkg/actions/workflows/ci.yml)

## Installation

```bash
pip install -e .
```

## Usage

```python
from llama_vector import LlamaVectorClient

# Initialize the client
client = LlamaVectorClient(api_key="your-api-key")
result = client.query("your query")
print(result)
```

## Features

- Fast and efficient
- Easy to use API
- Comprehensive documentation

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/nikjois/llama-vector.git
cd llama-vector

# Install development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

## License

MIT

## Author

Nik Jois (nikjois@llamasearch.ai)
