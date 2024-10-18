#!/usr/bin/env python3
"""
Llama Vector - Python package for LlamaSearch AI
"""

from setuptools import setup, find_packages

setup(
    name="llama_vector",
    version="0.1.0",
    description="[![PyPI version](https://badge.fury.io/py/llamavector.svg)](https://badge.fury.io/py/llamavector)
[![Python Version](https://img.shields.io/pypi/pyversions/llamavector.svg)](https://pypi.org/project/llamavector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yourusername/llamavector-pkg/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/llamavector-pkg/actions/workflows/ci.yml)",
    long_description="""# llama-vector

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
""",
    long_description_content_type="text/markdown",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    url="https://github.com/llamasearchai/llama-vector",
    project_urls={
        "Documentation": "https://github.com/llamasearchai/llama-vector",
        "Bug Tracker": "https://github.com/llamasearchai/llama-vector/issues",
        "Source Code": "https://github.com/llamasearchai/llama-vector",
    },
    packages='src.llama_vector',
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)
