"""
Quality control module for vector embeddings.

This module provides tools for detecting embedding drift, validating
embedding quality, and bridging semantic gaps between different models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from loguru import logger

from llama_vector.acceleration import normalize_vectors


class DriftDetector:
    """
    Detector for embedding drift over time.

    This class helps identify when embeddings start to drift from their
    original distribution, which can indicate model quality issues or
    changes in the data distribution.
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        reference_name: Optional[str] = None,
        threshold_mean: float = 0.05,
        threshold_variance: float = 0.1,
        alert_on_drift: bool = True,
    ):
        """
        Initialize a drift detector.

        Args:
            reference_embeddings: Baseline embeddings to compare against
            reference_name: Name for this reference set
            threshold_mean: Threshold for mean shift to trigger drift alert
            threshold_variance: Threshold for variance change to trigger drift alert
            alert_on_drift: Whether to log alerts when drift is detected
        """
        # Store reference statistics
        self.reference_name = (
            reference_name or f"reference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.reference_stats = self._compute_statistics(reference_embeddings)
        self.threshold_mean = threshold_mean
        self.threshold_variance = threshold_variance
        self.alert_on_drift = alert_on_drift

        # Store history of drift checks
        self.drift_history = []

    def _compute_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistical properties of embeddings.

        Args:
            embeddings: Array of embeddings to analyze

        Returns:
            Dict[str, Any]: Statistics including mean, variance, etc.
        """
        # Normalize embeddings if they aren't already
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
            embeddings = normalize_vectors(embeddings)

        # Compute basic statistics
        stats = {
            "mean": embeddings.mean(axis=0),
            "variance": embeddings.var(axis=0),
            "norm_mean": np.linalg.norm(embeddings, axis=1).mean(),
            "norm_std": np.linalg.norm(embeddings, axis=1).std(),
            "dimension": embeddings.shape[1],
            "count": embeddings.shape[0],
            # Add singular values for a more detailed distribution characterization
            "singular_values": np.linalg.svd(embeddings, full_matrices=False)[1][:10].tolist(),
            "timestamp": datetime.now().isoformat(),
        }

        return stats

    def detect_drift(self, current_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in current embeddings compared to reference.

        Args:
            current_embeddings: Embeddings to check for drift

        Returns:
            Dict[str, Any]: Drift metrics and status
        """
        # Compute statistics for current embeddings
        current_stats = self._compute_statistics(current_embeddings)

        # Calculate drift metrics
        mean_distance = np.linalg.norm(current_stats["mean"] - self.reference_stats["mean"])
        variance_ratio = np.mean(
            current_stats["variance"] / (self.reference_stats["variance"] + 1e-8)
        )

        # Determine if drift is significant
        mean_drift = mean_distance > self.threshold_mean
        variance_drift = abs(variance_ratio - 1.0) > self.threshold_variance
        has_drift = mean_drift or variance_drift

        # Compile drift report
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "has_drift": has_drift,
            "mean_shift": float(mean_distance),
            "variance_ratio": float(variance_ratio),
            "mean_drift_detected": mean_drift,
            "variance_drift_detected": variance_drift,
            "sample_size": current_stats["count"],
            "reference_name": self.reference_name,
        }

        # Log alert if requested
        if has_drift and self.alert_on_drift:
            logger.warning(
                f"Embedding drift detected! Mean shift: {mean_distance:.4f}, "
                f"Variance ratio: {variance_ratio:.4f}"
            )

        # Store in history
        self.drift_history.append(drift_report)

        return drift_report

    def save(self, path: Union[str, Path]) -> None:
        """
        Save drift detector state to disk.

        Args:
            path: Path to save the state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = {
            "reference_name": self.reference_name,
            "threshold_mean": self.threshold_mean,
            "threshold_variance": self.threshold_variance,
            "alert_on_drift": self.alert_on_drift,
            "reference_stats": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.reference_stats.items()
            },
            "drift_history": self.drift_history,
        }

        # Save to file
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved drift detector to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DriftDetector":
        """
        Load drift detector state from disk.

        Args:
            path: Path to the saved state

        Returns:
            DriftDetector: Loaded drift detector instance
        """
        path = Path(path)

        # Load from file
        with open(path, "r") as f:
            data = json.load(f)

        # Convert arrays back to numpy
        reference_stats = data["reference_stats"]
        for k, v in reference_stats.items():
            if isinstance(v, list):
                reference_stats[k] = np.array(v)

        # Create instance
        instance = cls.__new__(cls)
        instance.reference_name = data["reference_name"]
        instance.threshold_mean = data["threshold_mean"]
        instance.threshold_variance = data["threshold_variance"]
        instance.alert_on_drift = data["alert_on_drift"]
        instance.reference_stats = reference_stats
        instance.drift_history = data.get("drift_history", [])

        return instance


class SemanticBridge:
    """
    Bridge between different embedding spaces.

    This class helps align embeddings from different models or versions
    by learning a projection between their vector spaces.
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        projection_method: Literal["linear", "mlp"] = "linear",
    ):
        """
        Initialize a semantic bridge.

        Args:
            source_dim: Dimensionality of source embeddings
            target_dim: Dimensionality of target embeddings
            projection_method: Method for projecting between spaces
        """
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.projection_method = projection_method
        self.projection_matrix = None
        self.is_fitted = False

    def fit(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> None:
        """
        Fit the projection between source and target embedding spaces.

        Args:
            source_embeddings: Embeddings from source model
            target_embeddings: Corresponding embeddings from target model
        """
        if len(source_embeddings) != len(target_embeddings):
            raise ValueError(
                f"Source and target must have same number of embeddings, "
                f"got {len(source_embeddings)} and {len(target_embeddings)}"
            )

        # Normalize embeddings
        source_embeddings = normalize_vectors(source_embeddings)
        target_embeddings = normalize_vectors(target_embeddings)

        if self.projection_method == "linear":
            # Solve for projection matrix: min ||XP - Y||
            # Using least squares solution: P = (X^T X)^-1 X^T Y
            self.projection_matrix = np.linalg.lstsq(
                source_embeddings, target_embeddings, rcond=None
            )[0]

        elif self.projection_method == "mlp":
            # For MLP method, we'd need a neural network library
            # This is a simplified placeholder using a 2-layer projection
            # In a real implementation, you'd use PyTorch or TensorFlow

            # Simple 2-layer projection simulation
            hidden_dim = (self.source_dim + self.target_dim) // 2

            # Random initialization (in a real impl, these would be trained)
            np.random.seed(42)  # for reproducibility
            W1 = np.random.randn(self.source_dim, hidden_dim) * 0.1
            W2 = np.random.randn(hidden_dim, self.target_dim) * 0.1

            # Store both matrices
            self.projection_matrix = (W1, W2)

        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")

        self.is_fitted = True

    def transform(self, source_embeddings: np.ndarray) -> np.ndarray:
        """
        Transform source embeddings to target space.

        Args:
            source_embeddings: Embeddings to transform

        Returns:
            np.ndarray: Transformed embeddings in target space
        """
        if not self.is_fitted:
            raise RuntimeError("Semantic bridge must be fitted before transform")

        # Normalize source embeddings
        source_embeddings = normalize_vectors(source_embeddings)

        if self.projection_method == "linear":
            # Apply linear projection
            projected = source_embeddings @ self.projection_matrix

        elif self.projection_method == "mlp":
            # Apply 2-layer projection
            W1, W2 = self.projection_matrix
            hidden = np.maximum(0, source_embeddings @ W1)  # ReLU activation
            projected = hidden @ W2

        # Normalize result
        return normalize_vectors(projected)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the semantic bridge to disk.

        Args:
            path: Path to save the bridge
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for serialization
        data = {
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "projection_method": self.projection_method,
            "is_fitted": self.is_fitted,
        }

        # Add projection matrix
        if self.is_fitted:
            if self.projection_method == "linear":
                data["projection_matrix"] = self.projection_matrix.tolist()
            elif self.projection_method == "mlp":
                W1, W2 = self.projection_matrix
                data["projection_matrix_W1"] = W1.tolist()
                data["projection_matrix_W2"] = W2.tolist()

        # Save to file
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved semantic bridge to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SemanticBridge":
        """
        Load a semantic bridge from disk.

        Args:
            path: Path to the saved bridge

        Returns:
            SemanticBridge: Loaded semantic bridge instance
        """
        path = Path(path)

        # Load from file
        with open(path, "r") as f:
            data = json.load(f)

        # Create instance
        instance = cls(
            source_dim=data["source_dim"],
            target_dim=data["target_dim"],
            projection_method=data["projection_method"],
        )

        # Load projection matrix
        if data["is_fitted"]:
            instance.is_fitted = True

            if instance.projection_method == "linear":
                instance.projection_matrix = np.array(data["projection_matrix"])
            elif instance.projection_method == "mlp":
                W1 = np.array(data["projection_matrix_W1"])
                W2 = np.array(data["projection_matrix_W2"])
                instance.projection_matrix = (W1, W2)

        return instance
