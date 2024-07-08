"""
Configuration module for LlamaVector package.

This module provides configuration management for the LlamaVector package,
including environment variable loading, default settings, and runtime configuration.
"""

import json
import os
from pathlib import Path
from typing import Literal, Optional, Union

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class HNSWParams(BaseModel):
    """Parameters for HNSW index configuration."""

    M: int = Field(default=16, description="Number of connections per element in the index")
    efConstruction: int = Field(
        default=200, description="Size of the dynamic list for the nearest neighbors"
    )
    efSearch: int = Field(
        default=128,
        description="Size of the dynamic list for the nearest neighbors during search",
    )


class IVFParams(BaseModel):
    """Parameters for IVF index configuration."""

    nlist: int = Field(default=100, description="Number of clusters/cells")
    nprobe: int = Field(default=10, description="Number of clusters to visit during search")


class QuantizationParams(BaseModel):
    """Parameters for vector quantization."""

    enabled: bool = Field(default=False, description="Whether to enable product quantization")
    pq_m: int = Field(default=8, description="Number of sub-quantizers for PQ")
    pq_nbits: int = Field(default=8, description="Number of bits per sub-quantizer (usually 8)")


class LlamaVectorConfig(BaseModel):
    """
    Configuration for LlamaVector.

    This class manages all configuration settings for the LlamaVector package,
    with support for loading from environment variables and configuration files.
    """

    # Hardware acceleration settings
    use_mlx: bool = Field(
        default_factory=lambda: os.getenv("USE_MLX", "").lower() == "true",
        description="Whether to use MLX acceleration (Apple Silicon)",
    )
    use_cuda: bool = Field(
        default_factory=lambda: os.getenv("USE_CUDA", "").lower() == "true",
        description="Whether to use CUDA acceleration (NVIDIA GPUs)",
    )

    # Index settings
    default_index_type: Literal["flat", "hnsw", "ivf", "hybrid"] = Field(
        default_factory=lambda: os.getenv("DEFAULT_INDEX_TYPE", "hybrid"),
        description="Default index type to use",
    )

    # HNSW parameters
    hnsw_params: HNSWParams = Field(
        default_factory=lambda: HNSWParams(
            M=int(os.getenv("DEFAULT_HNSW_M", "16")),
            efConstruction=int(os.getenv("DEFAULT_HNSW_EFCONSTRUCTION", "200")),
            efSearch=int(os.getenv("DEFAULT_HNSW_EFSEARCH", "128")),
        ),
        description="HNSW index parameters",
    )

    # IVF parameters
    ivf_params: IVFParams = Field(
        default_factory=lambda: IVFParams(
            nlist=int(os.getenv("DEFAULT_IVF_NLIST", "100")),
            nprobe=int(os.getenv("DEFAULT_IVF_NPROBE", "10")),
        ),
        description="IVF index parameters",
    )

    # Quantization parameters
    quantization: QuantizationParams = Field(
        default_factory=lambda: QuantizationParams(
            enabled=os.getenv("DEFAULT_QUANTIZATION_ENABLED", "").lower() == "true",
            pq_m=int(os.getenv("DEFAULT_PQ_M", "8")),
            pq_nbits=int(os.getenv("DEFAULT_PQ_NBITS", "8")),
        ),
        description="Quantization parameters",
    )

    # API keys
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key for OpenAI models",
    )
    hf_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("HF_API_KEY", ""),
        description="HuggingFace API key for HuggingFace models",
    )

    # Storage settings
    snapshot_dir: str = Field(
        default_factory=lambda: os.getenv("SNAPSHOT_DIR", "./snapshots"),
        description="Directory for storing index snapshots",
    )
    versioning_enabled: bool = Field(
        default_factory=lambda: os.getenv("VERSIONING_ENABLED", "").lower() == "true",
        description="Whether to enable versioned snapshots",
    )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "LlamaVectorConfig":
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to the configuration JSON file

        Returns:
            LlamaVectorConfig: Loaded configuration object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Configuration file {file_path} not found, using defaults")
            return cls()

        try:
            with open(file_path, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return cls()

    def to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.

        Args:
            file_path: Path to save the configuration JSON file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

        logger.info(f"Configuration saved to {file_path}")


# Global configuration instance
config = LlamaVectorConfig()
