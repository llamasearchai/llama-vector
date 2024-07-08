"""
Utility functions for the LlamaVector package.

This module provides various utility functions and helper classes
for the LlamaVector package.
"""

import contextlib
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger


class Timer:
    """
    Context manager for timing code execution.

    Example:
    ```python
    with Timer() as timer:
        # Code to time
        result = slow_function()

    print(f"Execution took {timer.elapsed:.2f} seconds")
    ```
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize a timer.

        Args:
            name: Optional name for the timer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if self.name:
            logger.debug(f"Timer {self.name}: {self.elapsed:.4f}s")


def create_temp_file(target_path: Path) -> str:
    """
    Create a temporary file for atomic writing.

    This function creates a temporary file in the same directory as the target path,
    which can be used for atomic file operations by writing to the temporary file
    and then renaming it to the target path.

    Args:
        target_path: Target path for the file

    Returns:
        str: Path to the temporary file
    """
    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a unique temporary filename in the same directory
    temp_filename = f"{target_path.stem}_{uuid.uuid4().hex}{target_path.suffix}.tmp"
    temp_path = target_path.parent / temp_filename

    return str(temp_path)


def approximate_kmeans(
    vectors: List[List[float]],
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[int]]:
    """
    Approximate k-means clustering for high-dimensional vectors.

    This function implements a simplified k-means algorithm optimized for
    high-dimensional vectors, which can be used for implementing product
    quantization and other quantization methods.

    Args:
        vectors: List of vectors to cluster
        k: Number of clusters
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        random_seed: Random seed for reproducibility

    Returns:
        Tuple[List[List[float]], List[int]]: Centroids and cluster assignments
    """
    import numpy as np

    # Convert to numpy array
    vectors = np.array(vectors)

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly initialize centroids
    n_samples, n_features = vectors.shape
    centroid_indices = np.random.choice(n_samples, k, replace=False)
    centroids = vectors[centroid_indices].copy()

    # Iterate
    for iteration in range(max_iterations):
        # Assign vectors to nearest centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            # Compute squared Euclidean distance
            diff = vectors - centroids[i]
            distances[:, i] = np.sum(diff * diff, axis=1)

        # Assign to closest centroid
        assignments = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_vectors = vectors[assignments == i]
            if len(cluster_vectors) > 0:
                new_centroids[i] = np.mean(cluster_vectors, axis=0)
            else:
                # If a cluster is empty, keep the old centroid
                new_centroids[i] = centroids[i]

        # Check convergence
        centroid_shift = np.sum((new_centroids - centroids) ** 2)
        centroids = new_centroids

        if centroid_shift < tolerance:
            logger.debug(f"K-means converged after {iteration + 1} iterations")
            break

    return centroids.tolist(), assignments.tolist()


@contextlib.contextmanager
def atomic_write(path: Union[str, Path], mode: str = "w", **kwargs):
    """
    Context manager for atomic file writing.

    This context manager writes to a temporary file and then renames it
    to the target path upon successful completion, ensuring atomicity.

    Args:
        path: Target path to write to
        mode: File mode ("w", "wb", etc.)
        **kwargs: Additional arguments to pass to open()

    Yields:
        file: File object for writing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = create_temp_file(path)

    try:
        with open(temp_path, mode, **kwargs) as f:
            yield f

        # On successful completion, rename the temp file to the target path
        os.replace(temp_path, path)
    except Exception:
        # On error, remove the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def get_cpu_count() -> int:
    """
    Get the number of available CPU cores.

    Returns:
        int: Number of CPU cores
    """
    import multiprocessing

    return multiprocessing.cpu_count()


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size (e.g., "10.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0 or unit == "TB":
            break
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} {unit}"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        a: Numerator
        b: Denominator
        default: Default value to return if b is zero

    Returns:
        float: Result of division or default value
    """
    return a / b if b != 0 else default
