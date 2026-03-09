"""
Dataset profiling for zero-shot XGBoost parameter selection.

Computes all statistics about X and y needed to choose hyperparameters
without any search or cross-validation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class DatasetProfile:
    # Shape
    n_samples: int
    n_features: int

    # Feature characteristics
    sparsity: float          # fraction of zeros in X (0.0–1.0)
    is_sparse_counts: bool   # True for fingerprint-like data (integer, sparse, small values)

    # Task
    task: str                # "binary_classification" or "regression"

    # Binary classification
    imbalance_ratio: float = 1.0    # neg_count / pos_count; 1.0 if balanced or regression

    # Regression
    y_skewness: float = 0.0
    y_all_positive: bool = False

    def __repr__(self):
        lines = [
            f"DatasetProfile(",
            f"  n_samples={self.n_samples}, n_features={self.n_features}",
            f"  sparsity={self.sparsity:.3f}, is_sparse_counts={self.is_sparse_counts}",
            f"  task={self.task!r}",
        ]
        if self.task == "binary_classification":
            lines.append(f"  imbalance_ratio={self.imbalance_ratio:.2f}")
        else:
            lines.append(f"  y_skewness={self.y_skewness:.3f}, y_all_positive={self.y_all_positive}")
        lines.append(")")
        return "\n".join(lines)


def _compute_sparsity(X) -> float:
    """Fraction of zero entries in X."""
    if hasattr(X, "nnz"):
        # scipy sparse matrix
        n_total = X.shape[0] * X.shape[1]
        return 1.0 - X.nnz / n_total
    arr = np.asarray(X)
    return float((arr == 0).mean())


def _detect_sparse_counts(X, sparsity: float) -> bool:
    """
    Returns True if X looks like Morgan count fingerprints:
      - Majority of values are zero (sparsity > 0.5)
      - Non-zero values are integer-like
      - Max non-zero value is small (≤ 10), typical of count fingerprints
    """
    if sparsity < 0.5:
        return False

    # Sample up to 5000 rows for efficiency
    n_sample = min(5000, X.shape[0])
    if hasattr(X, "toarray"):
        sample = X[:n_sample].toarray()
    else:
        sample = np.asarray(X[:n_sample])

    nonzero_vals = sample[sample != 0]
    if nonzero_vals.size == 0:
        return False

    is_integer_like = float((nonzero_vals == np.floor(nonzero_vals)).mean()) > 0.95
    max_val = float(nonzero_vals.max())

    return is_integer_like and max_val <= 10


def _detect_task(y: np.ndarray) -> str:
    """
    Auto-detects task from y.
    Returns "binary_classification" if y contains only two unique integer values,
    else "regression".
    """
    unique = np.unique(y)
    if len(unique) == 2 and set(unique).issubset({0, 1}):
        return "binary_classification"
    return "regression"


def inspect(X, y, task: Optional[str] = None) -> DatasetProfile:
    """
    Profile the dataset and return a DatasetProfile.

    Parameters
    ----------
    X : array-like or scipy sparse matrix, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    task : str or None
        "binary_classification", "regression", or None for auto-detection.

    Returns
    -------
    DatasetProfile
    """
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    if task is None:
        task = _detect_task(y)
    if task not in ("binary_classification", "regression"):
        raise ValueError(f"task must be 'binary_classification' or 'regression', got {task!r}")

    sparsity = _compute_sparsity(X)
    is_sparse_counts = _detect_sparse_counts(X, sparsity)

    if task == "binary_classification":
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise ValueError(f"binary_classification requires exactly 2 classes, found {len(unique)}")
        # Identify positive (minority or label=1) and negative
        label_counts = dict(zip(unique, counts))
        pos_count = label_counts.get(1, counts.min())
        neg_count = label_counts.get(0, counts.max())
        imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            sparsity=sparsity,
            is_sparse_counts=is_sparse_counts,
            task=task,
            imbalance_ratio=imbalance_ratio,
        )

    else:  # regression
        y_skewness = float(stats.skew(y))
        y_all_positive = bool((y > 0).all())
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            sparsity=sparsity,
            is_sparse_counts=is_sparse_counts,
            task=task,
            y_skewness=y_skewness,
            y_all_positive=y_all_positive,
        )
