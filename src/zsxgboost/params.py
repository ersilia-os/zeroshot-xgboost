"""
Zero-shot XGBoost hyperparameter selection.

All rules are derived from:
  - Friedman (2001) "Greedy Function Approximation: A Gradient Boosting Machine"
  - Chen & Guestrin (2016) XGBoost paper (arXiv:1603.02754)
  - XGBoost official documentation and parameter tuning notes
  - Winkelmolen et al. (2020) "Practical and Sample Efficient Zero-Shot HPO"
    (arXiv:2007.13382)
  - Lima Marinho et al. (2024) "Optimization on selecting XGBoost hyperparameters
    using meta-learning" (Expert Systems)
  - Sommer et al. (2019) "Learning to Tune XGBoost with XGBoost"
    (arXiv:1909.07218)
  - Hutter et al. (2014) fANOVA hyperparameter importance analysis (ICML)
  - Community heuristics from Kaggle and AnalyticsVidhya

No search, no cross-validation. Parameters are chosen purely from dataset
statistics captured in a DatasetProfile.

Key design decisions
--------------------
* early_stopping_rounds scales with 1/learning_rate so slower learners have
  enough patience to reach their optimum (50 rounds at lr=0.1; 250 at lr=0.02).
* The n/p ratio (samples per feature) drives max_depth and regularization:
  underdetermined problems (n/p < 5) need shallower trees and stronger L1/L2
  to avoid fitting noise.
* gamma (min_split_loss) is set > 0 only when n/p < 5 — a regime where
  post-split pruning provides measurable benefit beyond weight regularisation.
* Binary-feature-rich inputs (one-hot style) get one depth level reduced,
  since each binary split carries roughly half the information of a continuous
  split.
* Sparse count data (Morgan fingerprints etc.) uses max_bin=64; integer values
  ≤ 10 are perfectly captured by 64 histogram bins, halving memory versus 128.
* A small random forest component (num_parallel_tree=3) is injected for
  datasets with n < 1000 to reduce variance via within-round bagging —
  following XGBoost's own RF tutorial and practitioner guidance.
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
    # early_stopping_rounds scales with 1/lr: slower learners need more
    # patience before improvement plateaus.  Formula:
    #   patience ≈ 50 × (0.1 / lr)
    # giving 50 rounds at lr=0.1, 100 at lr=0.05, 250 at lr=0.02.
    # ------------------------------------------------------------------
    params["n_estimators"] = 1000
    params["early_stopping_rounds"] = max(
        20, int(round(50 * (0.1 / params["learning_rate"])))
    )

    # ------------------------------------------------------------------
    # max_depth
    # Larger datasets can support deeper trees without overfitting.
    # Sparse count data (fingerprints) benefits from +1 depth because
    # useful splits need to combine several mostly-zero features.
    #
    # n/p ratio cap: when the data is underdetermined (few samples per
    # feature), deep trees trivially overfit — cap depth at 3 (n/p < 2)
    # or 4 (n/p < 5) regardless of n.
    #
    # Binary features: one-hot / indicator inputs carry roughly half the
    # information per split of continuous features; one fewer depth level
    # gives similar expressiveness with less overfitting risk.
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

    # n/p ratio cap (applied after sparse-counts bump)
    if profile.n_p_ratio < 2:
        max_depth = min(max_depth, 3)
    elif profile.n_p_ratio < 5:
        max_depth = min(max_depth, 4)

    # Binary-feature-rich inputs: reduce depth by one (floor at 3)
    if profile.binary_feature_fraction > 0.8:
        max_depth = max(3, max_depth - 1)

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
    # [0.3, 0.8] almost universally helps.  For tiny datasets subsampling
    # introduces more variance than it removes — use all rows instead.
    # Reduce further for very large datasets to save memory.
    # ------------------------------------------------------------------
    if n < 200:
        params["subsample"] = 1.0
    elif n >= 1_000_000:
        params["subsample"] = 0.6
    else:
        params["subsample"] = 0.8

    # ------------------------------------------------------------------
    # num_parallel_tree  (random-forest style within-round bagging)
    # For small datasets, training a small forest at each boosting step
    # reduces variance via bagging diversity without cross-validation.
    # subsample < 1 (0.8, set above) ensures the parallel trees see
    # different row subsets within each round.
    # ------------------------------------------------------------------
    if 200 <= n < 1_000:
        params["num_parallel_tree"] = 3

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
    # Regularization (base rules)
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
    # n/p ratio regularization multiplier
    # Underdetermined problems have many more model parameters than
    # training constraints.  Scale up L2 and raise the L1 floor to
    # prevent the model from fitting noise that correlates with y by
    # chance.  Applied multiplicatively on top of the base rules above.
    # ------------------------------------------------------------------
    if profile.n_p_ratio < 2:
        params["reg_lambda"] = round(params["reg_lambda"] * 2.0, 4)
        params["reg_alpha"] = max(params["reg_alpha"], 0.5)
    elif profile.n_p_ratio < 5:
        params["reg_lambda"] = round(params["reg_lambda"] * 1.5, 4)
        params["reg_alpha"] = max(params["reg_alpha"], 0.1)

    # ------------------------------------------------------------------
    # gamma (min_split_loss)
    # Pre-pruning regularizer: a split is accepted only if it reduces the
    # loss by at least gamma.  Most effective when the data is
    # underdetermined (n/p < 5) and depthwise growth would otherwise
    # produce many small-gain splits deep in the tree.
    # For well-determined data, other regularization handles overfitting
    # and gamma=0 (default) is preferable to avoid pruning real signal.
    # ------------------------------------------------------------------
    if profile.n_p_ratio < 5:
        params["gamma"] = 0.1

    # ------------------------------------------------------------------
    # max_bin (histogram granularity)
    # Sparse count features (integer values ≤ 10) are perfectly captured
    # by 64 bins — using more is wasteful and slows histogram construction.
    # For large or high-dimensional dense data, 128 balances accuracy and
    # memory.  The default 256 is used otherwise.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        params["max_bin"] = 64
    elif n > 1_000_000 or p > 500:
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
