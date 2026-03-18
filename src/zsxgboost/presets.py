"""
Fixed preset configurations for portfolio-based selection.

Four externally-derived XGBoost configurations that compete against the
zero-shot internal preset on the validation split:

  1. xgb_default  – XGBoost built-in defaults (lr=0.3, max_depth=6).
                    Baseline: what you get with XGBClassifier().fit(X, y).

  2. flaml        – FLAML zero-shot: 1-NN meta-feature matching on the
                    portfolio from microsoft/FLAML (MIT license).
                    Lossguide (max_leaves) variant with colsample_bylevel —
                    FLAML's distinctive ingredient.  Meta-features:
                    [n_samples, n_features, n_classes, pct_numeric_features].

  3. autogluon    – AutoGluon tabular XGBoost default configuration.
                    From autogluon/autogluon tabular/.../xgboost/ (Apache-2):
                    lr=0.1, max_depth=6, colsample_bytree=1.0, mcw=1.
                    subsample/reg_alpha/reg_lambda are NOT in AutoGluon's HPO
                    space; XGBoost defaults (1.0, 0.0, 1.0) apply.

  4. rf           – XGBoost configured as a boosted Random Forest.
                    num_parallel_tree=3 + colsample_bynode=sqrt(p)/p mirrors
                    sklearn RF's per-split diversity rule.

All functions accept (profile, device) and return a full params dict
that includes n_estimators, early_stopping_rounds, objective, and
eval_metric — ready to be consumed by _train_phase1() in model.py.

Task-specific parameters (objective, eval_metric, scale_pos_weight) are
injected by _add_task_params() so all presets share the same evaluation
metric for a fair comparison.
"""

from typing import Dict, Any

import numpy as np

from .inspector import DatasetProfile
from . import flaml_data as _fd


# Metrics where higher is better (all others are minimised by XGBoost)
MAXIMIZE_METRICS = frozenset({"auc", "aucpr", "map", "ndcg"})


def _add_task_params(params: Dict[str, Any], profile: DatasetProfile) -> None:
    """Inject objective, eval_metric, and imbalance correction into params."""
    if profile.task == "binary_classification":
        params["objective"] = "binary:logistic"
        ratio = profile.imbalance_ratio
        params["scale_pos_weight"] = round(ratio, 4)
        if ratio > 100:
            params["max_delta_step"] = 1
        params["eval_metric"] = "aucpr" if ratio > 10 else "auc"
    else:
        # Simple squarederror for all external presets; the internal preset
        # has a more refined skewness-based objective selection.
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"


def xgb_default_params(profile: DatasetProfile, device: str,
                       nthread: int = -1) -> Dict[str, Any]:
    """
    XGBoost out-of-the-box defaults.

    Represents what a user gets by calling XGBClassifier().fit(X, y) with
    no tuning.  Only the task objective, eval metric, and imbalance correction
    are added to enable a fair comparison.
    """
    n = profile.n_samples
    max_bin = 128 if n > 100_000 else 256
    params: Dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda" if device == "gpu" else "cpu",
        "learning_rate": 0.3,       # XGBoost default
        "max_depth": 6,             # XGBoost default
        "min_child_weight": 1,      # XGBoost default
        "subsample": 1.0,           # XGBoost default
        "colsample_bytree": 1.0,    # XGBoost default
        "reg_alpha": 0.0,           # XGBoost default
        "reg_lambda": 1.0,          # XGBoost default
        "max_bin": max_bin,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
    }
    if device == "cpu":
        import os
        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params


def flaml_params(profile: DatasetProfile, device: str,
                 nthread: int = -1) -> Dict[str, Any]:
    """
    FLAML zero-shot configuration via 1-NN meta-feature matching.

    Selects one of four portfolio configs from FLAML's lossguide portfolio
    (microsoft/FLAML, flaml/default/xgboost/{binary,regression}.json, MIT).

    Post-processing applied for small-dataset robustness:
      - n_estimators capped at 2000 (early stopping is the real control)
      - max_leaves capped at max(64, n // 5) to prevent overfitting on
        small datasets (FLAML's configs were calibrated on n >> 10k)
      - min_child_weight floored at 1 (FLAML's near-zero values assume
        large n; flooring prevents single-sample leaves on small datasets)
      - early_stopping_rounds = min(200, max(50, 50 × 0.1 / lr)) so that
        very slow FLAML learners (lr ≈ 0.001) don't require 5000+ patience
        rounds within the portfolio comparison budget
    """
    n = profile.n_samples
    p = profile.n_features
    pct_numeric = 1.0 - profile.binary_feature_fraction

    data = _fd.BINARY if profile.task == "binary_classification" else _fd.REGRESSION
    n_classes = 2 if profile.task == "binary_classification" else 0

    center = np.array(data["preprocessing"]["center"])
    scale  = np.array(data["preprocessing"]["scale"])
    query  = np.array([n, p, n_classes, pct_numeric])
    q_norm = (query - center) / scale

    best_dist, best_idx = float("inf"), 0
    for nb in data["neighbors"]:
        feat = np.array(nb["features"])
        d = float(np.dot(q_norm - feat, q_norm - feat))  # squared L2
        if d < best_dist:
            best_dist = d
            best_idx  = nb["choice"][0]

    hp = data["portfolio"][best_idx]
    lr = float(hp["learning_rate"])
    max_bin = 128 if n > 100_000 else 256

    params: Dict[str, Any] = {
        "tree_method":       "hist",
        "device":            "cuda" if device == "gpu" else "cpu",
        "grow_policy":       "lossguide",
        "max_depth":         0,   # unlimited; max_leaves controls capacity
        "max_leaves":        max(64, min(n // 5, int(hp["max_leaves"]))),
        "learning_rate":     lr,
        "min_child_weight":  max(1.0, float(hp["min_child_weight"])),
        "subsample":         float(hp["subsample"]),
        "colsample_bylevel": float(hp["colsample_bylevel"]),
        "colsample_bytree":  float(hp["colsample_bytree"]),
        "reg_alpha":         float(hp["reg_alpha"]),
        "reg_lambda":        float(hp["reg_lambda"]),
        "max_bin":           max_bin,
        "n_estimators":      2000,
        "early_stopping_rounds": min(200, max(50, int(round(50 * 0.1 / lr)))),
    }
    if device == "cpu":
        import os
        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params


def autogluon_params(profile: DatasetProfile, device: str,
                     nthread: int = -1) -> Dict[str, Any]:
    """
    AutoGluon tabular XGBoost default configuration.

    From autogluon/autogluon tabular/src/autogluon/tabular/models/xgboost/
    hyperparameters/{parameters,searchspaces}.py (Apache-2 license):
      - learning_rate = 0.1  (center of log-scale HPO range [0.005, 0.2])
      - max_depth     = 6    (HPO range [3, 10], default 6)
      - min_child_weight = 1 (HPO range [1, 5],  default 1)
      - colsample_bytree = 1.0 (HPO range [0.5, 1.0], default 1.0)
      - subsample, reg_alpha, reg_lambda are NOT in AutoGluon's HPO space;
        XGBoost defaults (1.0, 0.0, 1.0) are used.
      - n_estimators = 10000 in AutoGluon; capped at 2000 here since early
        stopping is the real control.
    """
    n = profile.n_samples
    max_bin = 128 if n > 100_000 else 256
    params: Dict[str, Any] = {
        "tree_method":      "hist",
        "device":           "cuda" if device == "gpu" else "cpu",
        "learning_rate":    0.1,
        "max_depth":        6,
        "min_child_weight": 1,
        "subsample":        1.0,
        "colsample_bytree": 1.0,
        "reg_alpha":        0.0,
        "reg_lambda":       1.0,
        "max_bin":          max_bin,
        "n_estimators":     2000,
        "early_stopping_rounds": 50,
    }
    if device == "cpu":
        import os
        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params


def rf_params(profile: DatasetProfile, device: str,
              nthread: int = -1) -> Dict[str, Any]:
    """
    XGBoost configured as a boosted Random Forest.

    num_parallel_tree=3 builds three trees per boosting step (XGBoost RF
    mode, per XGBoost docs §Random Forests).  Per-split column sampling
    (colsample_bynode = sqrt(p)/p, clamped to [0.05, 0.5]) mirrors sklearn
    RF's sqrt(p) feature-diversity rule.  Row subsampling (subsample=0.8)
    provides bootstrap-style diversity across the parallel trees.

    Using multiple boosting rounds with early stopping keeps the comparison
    fair: each round builds 3 parallel trees, the total ensemble grows
    across rounds, and early stopping fires when validation AUC plateaus.
    """
    n = profile.n_samples
    p = profile.n_features
    csn = round(max(0.05, min(0.5, 1.0 / (p ** 0.5))), 3)
    max_bin = 128 if n > 100_000 else 256

    params: Dict[str, Any] = {
        "tree_method":      "hist",
        "device":           "cuda" if device == "gpu" else "cpu",
        "num_parallel_tree": 3,
        "subsample":        0.8,
        "colsample_bynode": csn,
        "colsample_bytree": 1.0,
        "learning_rate":    0.1,
        "max_depth":        8,
        "min_child_weight": 1,
        "reg_alpha":        0.0,
        "reg_lambda":       1.0,
        "max_bin":          max_bin,
        "n_estimators":     2000,
        "early_stopping_rounds": 50,
    }
    if device == "cpu":
        import os
        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params
