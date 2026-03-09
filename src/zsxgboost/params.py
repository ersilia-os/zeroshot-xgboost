"""
Zero-shot XGBoost hyperparameter selection.

All rules are derived from:
  - Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"
  - Chen & Guestrin (2016) XGBoost paper (arXiv:1603.02754)
  - XGBoost official documentation
  - Community heuristics from Kaggle and AnalyticsVidhya

No search, no cross-validation. Parameters are chosen purely from dataset
statistics captured in a DatasetProfile.
"""

import os
from typing import Dict, Any

from .inspector import DatasetProfile


def get_params(profile: DatasetProfile, device: str = "cpu") -> Dict[str, Any]:
    """
    Return a dict of XGBoost parameters for the given dataset profile.

    Parameters
    ----------
    profile : DatasetProfile
        Output of zsxgboost.inspect(X, y).
    device : str
        "cpu" or "gpu".

    Returns
    -------
    dict
        Ready to unpack into xgb.XGBClassifier(**params) or XGBRegressor(**params).
        Includes early_stopping_rounds; the caller must supply an eval_set when
        calling .fit().
    """
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

    n = profile.n_samples
    p = profile.n_features
    params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tree method and device
    # ------------------------------------------------------------------
    params["tree_method"] = "hist"
    params["device"] = "cuda" if device == "gpu" else "cpu"

    # ------------------------------------------------------------------
    # Learning rate
    # Slower learning on larger datasets: more samples → each tree is
    # already informative, so large steps overshoot.
    # ------------------------------------------------------------------
    if n < 10_000:
        params["learning_rate"] = 0.1
    elif n < 100_000:
        params["learning_rate"] = 0.05
    else:
        params["learning_rate"] = 0.02

    # ------------------------------------------------------------------
    # n_estimators and early stopping
    # Set a high ceiling; early stopping will find the optimal round.
    # The caller must pass eval_set to .fit().
    # ------------------------------------------------------------------
    params["n_estimators"] = 1000
    params["early_stopping_rounds"] = 50

    # ------------------------------------------------------------------
    # max_depth
    # Larger datasets can support deeper trees without overfitting.
    # Sparse count data (fingerprints) benefits from +1 depth because
    # useful splits need to combine several mostly-zero features.
    # ------------------------------------------------------------------
    if n < 1_000:
        max_depth = 3
    elif n < 10_000:
        max_depth = 4
    elif n < 100_000:
        max_depth = 5
    else:
        max_depth = 6

    if profile.is_sparse_counts:
        max_depth = min(max_depth + 1, 8)

    params["max_depth"] = max_depth

    # ------------------------------------------------------------------
    # min_child_weight
    # Scales with dataset size to prevent leaves supported by very few
    # samples. Halved for severely imbalanced classification so the
    # minority class can form leaves at all.
    # ------------------------------------------------------------------
    mcw = max(1, n // 1000)
    mcw = min(mcw, 20)

    if (
        profile.task == "binary_classification"
        and profile.imbalance_ratio > 10
    ):
        mcw = max(1, mcw // 2)

    params["min_child_weight"] = mcw

    # ------------------------------------------------------------------
    # subsample
    # Stochastic gradient boosting: Friedman showed that subsample in
    # [0.3, 0.8] almost universally helps. Reduce further for very large
    # datasets to save memory.
    # ------------------------------------------------------------------
    if n >= 1_000_000:
        params["subsample"] = 0.6
    else:
        params["subsample"] = 0.8

    # ------------------------------------------------------------------
    # colsample_bytree
    # Scales down with feature count to reduce variance and memory.
    # For sparse count features the effective information density is low,
    # so we allow sampling more columns (capped at 1.0).
    # ------------------------------------------------------------------
    if p <= 50:
        cst = 1.0
    elif p <= 200:
        cst = 0.8
    elif p <= 500:
        cst = 0.7
    elif p <= 2000:
        cst = 0.5
    else:
        cst = max(0.3, 500 / p)

    if profile.is_sparse_counts:
        cst = min(1.0, cst * 1.5)

    params["colsample_bytree"] = round(cst, 2)

    # ------------------------------------------------------------------
    # Regularization
    # L1 (reg_alpha) is better for high-dimensional sparse data; it drives
    # uninformative leaf weights toward zero.
    # L2 (reg_lambda) is the XGBoost default and suits dense continuous data.
    # Increase both for very small datasets where data alone cannot
    # regularize.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        params["reg_alpha"] = 0.1
        params["reg_lambda"] = 1.0
    else:
        params["reg_alpha"] = 0.0
        params["reg_lambda"] = 1.0

    if n < 1_000:
        params["reg_lambda"] = 5.0
        params["reg_alpha"] = max(params["reg_alpha"], 0.5)

    # ------------------------------------------------------------------
    # max_bin (histogram granularity)
    # Reducing max_bin halves histogram memory proportionally.
    # Critical for datasets with many features or very many rows.
    # ------------------------------------------------------------------
    if n > 1_000_000 or p > 500:
        params["max_bin"] = 128
    else:
        params["max_bin"] = 256

    # ------------------------------------------------------------------
    # Parallelism (CPU only; GPU manages its own threads)
    # ------------------------------------------------------------------
    if device == "cpu":
        params["nthread"] = os.cpu_count() or 1

    # ------------------------------------------------------------------
    # Task-specific parameters
    # ------------------------------------------------------------------
    if profile.task == "binary_classification":
        _set_classification_params(params, profile)
    else:
        _set_regression_params(params, profile)

    return params


def _set_classification_params(params: Dict[str, Any], profile: DatasetProfile) -> None:
    params["objective"] = "binary:logistic"

    ratio = profile.imbalance_ratio

    # scale_pos_weight: XGBoost docs recommend neg/pos for imbalanced data.
    # Only apply when ratio is meaningfully > 1.
    if ratio > 1.5:
        params["scale_pos_weight"] = round(ratio, 4)

    # max_delta_step = 1 stabilises logistic regression gradient updates
    # when imbalance is extreme (XGBoost docs recommendation).
    if ratio > 100:
        params["max_delta_step"] = 1

    # AUC-PR is a more informative metric than AUC-ROC for imbalanced data
    # because it focuses on the minority (positive) class.
    if ratio > 10:
        params["eval_metric"] = "aucpr"
    else:
        params["eval_metric"] = "auc"


def _set_regression_params(params: Dict[str, Any], profile: DatasetProfile) -> None:
    skew = profile.y_skewness
    abs_skew = abs(skew)

    if abs_skew < 1.0:
        # Approximately symmetric: standard MSE loss is appropriate.
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"

    elif abs_skew < 2.0:
        # Moderate skew with positive y: Tweedie loss handles right-skewed,
        # non-negative distributions (e.g. count data, insurance losses).
        # For negative or mixed-sign y fall back to pseudoHuber which is
        # robust to outliers without requiring positivity.
        if profile.y_all_positive:
            params["objective"] = "reg:tweedie"
            params["tweedie_variance_power"] = 1.5
            params["eval_metric"] = "tweedie-nloglik@1.5"
        else:
            params["objective"] = "reg:pseudohubererror"
            params["eval_metric"] = "mae"

    else:
        # Severe skew: pseudoHuber is robust to the heavy tail regardless
        # of sign. For positive-only targets Tweedie is also valid but
        # pseudoHuber makes no distributional assumptions.
        if profile.y_all_positive:
            params["objective"] = "reg:tweedie"
            params["tweedie_variance_power"] = 1.5
            params["eval_metric"] = "tweedie-nloglik@1.5"
        else:
            params["objective"] = "reg:pseudohubererror"
            params["eval_metric"] = "mae"
