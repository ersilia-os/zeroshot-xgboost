"""
Dataset profiling for zero-shot XGBoost parameter selection.

Computes all statistics about X and y needed to choose hyperparameters
without any search or cross-validation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class DatasetProfile:
    # Shape
    n_samples: int
    n_features: int

    # Sample-to-feature ratio (samples per feature)
    n_p_ratio: float

    # Feature characteristics
    sparsity: float          # fraction of zeros in X (0.0–1.0)
    is_sparse_counts: bool   # True for fingerprint-like data (integer, sparse, small values)
    binary_feature_fraction: float  # fraction of features that only take {0, 1} values
    feature_signal_strength: float  # mean |Pearson corr| between features and target (sampled)
    feature_signal_p90: float       # 90th-percentile |Pearson corr|; captures top-feature signal

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
            f"  n_samples={self.n_samples}, n_features={self.n_features}, n_p_ratio={self.n_p_ratio:.2f}",
            f"  sparsity={self.sparsity:.3f}, is_sparse_counts={self.is_sparse_counts}",
            f"  binary_feature_fraction={self.binary_feature_fraction:.3f}, "
            f"feature_signal_strength={self.feature_signal_strength:.3f}",
            f"  task={self.task!r}",
        ]
        if self.task == "binary_classification":
            lines.append(f"  imbalance_ratio={self.imbalance_ratio:.2f}")
        else:
            lines.append(f"  y_skewness={self.y_skewness:.3f}, y_all_positive={self.y_all_positive}")
        lines.append(
            f"  feature_signal_p90={self.feature_signal_p90:.3f}"
        )
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
    Returns True if X looks like count fingerprints (Morgan, ECFP, etc.):
      - Majority of values are zero (sparsity > 0.5)
      - Non-zero values are integer-like
      - Max non-zero value is small, OR sparsity is very high (≥ 0.85)

    The max_val <= 10 check distinguishes count fingerprints from general
    integer features (e.g. ring counts, atom counts) at moderate sparsity.
    For very sparse data (≥ 0.85 zeros), the high sparsity itself is
    sufficient evidence of a fingerprint-like representation — some Morgan
    count fingerprints have occasional counts > 10 for fused ring systems
    but are still sparse fingerprints and benefit from the sparse code path.
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
    if not is_integer_like:
        return False

    # Very high sparsity (≥ 0.85) → fingerprint-like regardless of max count.
    # Moderate sparsity → require small max value to distinguish from general
    # integer features (ring counts, atom counts, etc.).
    max_val = float(nonzero_vals.max())
    return sparsity >= 0.85 or max_val <= 10


def _compute_binary_feature_fraction(X, n_sample: int = 5000) -> float:
    """
    Fraction of features that take only {0, 1} values (e.g. one-hot encoded
    or binary indicator features).  Estimated from the first n_sample rows.
    """
    n_s = min(n_sample, X.shape[0])
    if hasattr(X, "toarray"):
        sample = X[:n_s].toarray()
    else:
        sample = np.asarray(X[:n_s])
    is_binary = ((sample == 0) | (sample == 1)).all(axis=0)
    return float(is_binary.mean())


def _estimate_feature_signal(X, y: np.ndarray, n_sample: int = 5000,
                              p_sample: int = 500):
    """
    Estimate |Pearson| correlation distribution between features and target.

    Uses a random subsample of up to n_sample rows and p_sample columns.
    Constant features are excluded.

    Returns (mean_signal, p90_signal) where:
    - mean_signal: mean |Pearson| across sampled features — overall noise level.
    - p90_signal: 90th-percentile |Pearson| — captures the strength of the
      top 10% of features, which matters more than the average when most
      features are uninformative (e.g. ECFP fingerprints where ~93% of bits
      are zero and only a handful drive the prediction).
    """
    n, p = X.shape
    n_s = min(n_sample, n)
    rng = np.random.RandomState(42)

    # Random row sampling (avoids bias from sorted/ordered datasets)
    row_idx = rng.choice(n, n_s, replace=False) if n > n_s else np.arange(n_s)
    if hasattr(X, "toarray"):
        X_s = X[row_idx].toarray().astype(float)
    else:
        X_s = np.asarray(X)[row_idx].astype(float)
    y_s = y[row_idx].astype(float)

    # Random column subsample for speed
    if p > p_sample:
        col_idx = rng.choice(p, p_sample, replace=False)
        X_s = X_s[:, col_idx]

    # Drop constant features; check for constant target
    x_std = X_s.std(axis=0)
    X_s = X_s[:, x_std > 0]
    y_std = float(y_s.std())
    if X_s.shape[1] == 0 or y_std == 0.0:
        return 0.0, 0.0

    # Vectorized Pearson correlation of each feature with the target
    X_c = X_s - X_s.mean(axis=0)      # (n_s, p_s)
    y_c = y_s - y_s.mean()            # (n_s,)
    cov = (X_c * y_c[:, None]).mean(axis=0)   # (p_s,)
    x_stds = X_c.std(axis=0)
    mask = x_stds > 0
    corrs = np.clip(np.abs(cov[mask] / (x_stds[mask] * y_std)), 0.0, 1.0)
    if corrs.size == 0:
        return 0.0, 0.0
    return float(corrs.mean()), float(np.percentile(corrs, 90))


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
    binary_feature_fraction = _compute_binary_feature_fraction(X)
    feature_signal_strength, feature_signal_p90 = _estimate_feature_signal(X, y)
    n_p_ratio = float(n_samples) / n_features

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
            n_p_ratio=n_p_ratio,
            sparsity=sparsity,
            is_sparse_counts=is_sparse_counts,
            binary_feature_fraction=binary_feature_fraction,
            feature_signal_strength=feature_signal_strength,
            feature_signal_p90=feature_signal_p90,
            task=task,
            imbalance_ratio=imbalance_ratio,
        )

    else:  # regression
        y_skewness = float(stats.skew(y))
        y_all_positive = bool((y > 0).all())
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_p_ratio=n_p_ratio,
            sparsity=sparsity,
            is_sparse_counts=is_sparse_counts,
            binary_feature_fraction=binary_feature_fraction,
            feature_signal_strength=feature_signal_strength,
            feature_signal_p90=feature_signal_p90,
            task=task,
            y_skewness=y_skewness,
            y_all_positive=y_all_positive,
        )
