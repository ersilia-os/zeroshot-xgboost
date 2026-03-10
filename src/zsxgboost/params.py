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
  - Grinsztajn et al. (2022) "Why do tree-based models still outperform deep
    learning on tabular data?" NeurIPS (arXiv:2207.08815) — search space priors
  - Probst et al. (2023) "Practical guidelines for the use of gradient boosting
    for molecular property prediction" J. Cheminformatics — QSAR-specific priors;
    ranked learning_rate, scale_pos_weight, gamma as the three most impactful
    parameters; recommends lr 0.01-0.05 and min_child_weight 5-20 for fingerprints
  - Sheridan et al. (2016) "Extreme Gradient Boosting as a Method for QSAR"
    JCIM — found max_depth=4, lr=0.1 needed to match RF on ECFP features
  - AutoGluon tabular XGBoost search space (n_estimators=10000, lr 0.005-0.2,
    max_depth 3-10, colsample_bytree 0.5-1.0, min_child_weight 1-5)
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
* gamma (min_split_loss) scales with dataset difficulty: 0.1 for n/p < 5
  (underdetermined), raised to 0.5 for sparse-count (fingerprint) data where
  bit-level noise is highest.  Probst et al. 2023 ranks gamma #3 in importance
  for QSAR; values 0.1-5 consistently prune spurious splits on noisy fingerprint
  bits.
* min_child_weight is raised for sparse-count data: requiring 3-10 samples per
  leaf effectively filters structural fragments that appear in fewer than that
  many training molecules, preventing the model from overfitting rare ECFP bits.
  Probst et al. 2023 recommends exploring 5-20 for fingerprint QSAR.
* learning_rate is reduced to 0.05 for sparse-count small datasets (Probst et
  al. 2023 found 0.01-0.05 consistently outperforms 0.1 for molecular property
  prediction).
* Binary-only features that are NOT sparse fingerprint data (pure one-hot style)
  get one depth level reduced, since each binary split carries roughly half the
  information of a continuous split.  ECFP fingerprints are exempt from this
  reduction because they are already handled by the sparse-counts depth bump.
* The n/p ratio depth cap is relaxed for sparse-count data: colsample_bynode,
  min_child_weight, gamma, and L1 regularization together constrain capacity more
  precisely than a hard depth cap.  Fingerprint data with n/p < 2 is allowed
  max_depth=4 instead of 3 so the model can capture multi-bit interactions.
* Sparse count data (Morgan fingerprints etc.) uses max_bin=64; integer values
  ≤ 10 are perfectly captured by 64 histogram bins, halving memory versus 128.
* A small random forest component (num_parallel_tree=3) is injected for
  datasets with n < 2000 to reduce variance via within-round bagging —
  following XGBoost's own RF tutorial and practitioner guidance.  The range
  extends to n < 2000 (from the previous n < 1000) because drug datasets in
  the 1000-2000 range are still small enough to benefit from the variance
  reduction, and subsample=0.8 ensures per-tree diversity.
* colsample_bynode (per-split column sampling) is set for high-dimensional
  binary/sparse data (p > 200, binary or fingerprint features), mirroring
  Random Forest's sqrt(p) per-split diversity.  This is the primary mechanism
  that makes RF competitive on ECFP fingerprints and is absent from
  colsample_bytree alone.  The formula sqrt(p)/p gives the RF-equivalent
  fraction; we clamp to [0.05, 0.3] to prevent too few features per split
  while still injecting meaningful diversity on 1024-2048 bit fingerprints.
* colsample_bytree for ECFP/sparse data is kept >= 0.6 so that the tree-level
  sampling does not additionally starve the per-node sampling budget.
* For ECFP/Morgan fingerprints with n/p ratio typically 1-5 (underdetermined),
  L1 regularization (reg_alpha) is substantially more effective than L2 alone
  because it drives many uninformative leaf weights exactly to zero, effectively
  performing implicit feature selection within each tree.  We therefore use a
  stronger L1 prior (reg_alpha >= 0.1 for sparse, >= 1.0 for underdetermined
  sparse) while keeping reg_lambda at a moderate value.
* feature_signal_strength (mean |Pearson| between features and target) adjusts
  the final regularization: very weak signal (< 0.02) tightens L1/L2; strong
  signal (> 0.05) eases L2 so the model can exploit available structure.
* n_estimators is set to 2000 (increased from 1000) with a correspondingly
  wider early-stopping window so that slow learners on small datasets can
  always reach their optimum before the ceiling is hit.
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
    # For sparse-count (fingerprint) data on small datasets, Probst et al.
    # 2023 found lr 0.01-0.05 consistently outperforms 0.1: noisy bit-level
    # features benefit from smaller steps so the model doesn't commit to
    # spurious early trees.
    # ------------------------------------------------------------------
    if n < 10_000:
        params["learning_rate"] = 0.05 if profile.is_sparse_counts else 0.1
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
    # Ceiling raised to 2000 so slow learners on small fingerprint
    # datasets (lr=0.1, ~50 patience) still have room to converge.
    # ------------------------------------------------------------------
    params["n_estimators"] = 2000
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

    # n/p ratio cap (applied after sparse-counts bump).
    # For sparse-count (fingerprint) data, colsample_bynode + min_child_weight
    # + gamma + L1 together constrain capacity more precisely than a hard depth
    # cap; allowing one extra depth level lets the model capture multi-bit
    # interactions (e.g. two rare ECFP bits co-occurring → active molecule).
    if profile.n_p_ratio < 2:
        max_depth = min(max_depth, 4 if profile.is_sparse_counts else 3)
    elif profile.n_p_ratio < 5:
        max_depth = min(max_depth, 5 if profile.is_sparse_counts else 4)

    # Binary-feature-rich inputs: reduce depth by one (floor at 3).
    # EXEMPTION: sparse count data (ECFP / Morgan fingerprints) already
    # received a depth bump above that accounts for the low per-feature
    # information density.  Applying the binary penalty on top would cancel
    # that bump and leave fingerprint data with the same depth as dense
    # continuous data — the opposite of what we want.
    if profile.binary_feature_fraction > 0.8 and not profile.is_sparse_counts:
        max_depth = max(3, max_depth - 1)

    params["max_depth"] = max_depth

    # ------------------------------------------------------------------
    # min_child_weight
    # Scales with dataset size to prevent leaves supported by very few
    # samples. Halved for severely imbalanced classification so the
    # minority class can form leaves at all.
    # For sparse-count (fingerprint) data, Probst et al. 2023 recommends
    # 5-20: requiring N molecules to share a structural fragment before it
    # becomes a split criterion filters rare ECFP bits that correlate with
    # the target by chance.  We use max(3, n//500) which gives 3-10 in
    # the typical drug dataset range (1000-5000 molecules).
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        mcw = max(3, n // 500)
    else:
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
    # Range extended to n < 2000 because drug datasets in the 1000-2000
    # range are still underdetermined relative to fingerprint dimensionality
    # (n/p ratio typically 1-2) and benefit from the additional variance
    # reduction.  num_parallel_tree=3 follows the XGBoost RF tutorial
    # recommendation and Kaggle practitioner consensus for boosted RF.
    # ------------------------------------------------------------------
    if 200 <= n < 2_000:
        params["num_parallel_tree"] = 3

    # ------------------------------------------------------------------
    # colsample_bytree
    # Scales down with feature count to reduce variance and memory.
    # For sparse count features (ECFP), the tree-level sampling must not
    # be set too low because colsample_bynode will further restrict the
    # per-split candidate pool; a floor of 0.6 ensures the combined
    # sampling does not starve individual splits.
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
        # For ECFP fingerprints, colsample_bynode already provides per-split
        # diversity; colsample_bytree should stay reasonably high (>= 0.6)
        # so that the tree-level pool is not too narrow.
        cst = max(0.6, cst)

    params["colsample_bytree"] = round(cst, 2)

    # ------------------------------------------------------------------
    # colsample_bynode  (per-split column sampling)
    # For high-dimensional binary / sparse data (e.g. ECFP fingerprints),
    # sampling columns at each split — rather than only per tree — injects
    # the same per-node diversity that makes Random Forests strong on these
    # inputs.  The fraction mirrors RF's sqrt(p)/p rule:
    #   csn = sqrt(p) / p = 1 / sqrt(p)
    # Examples: p=512 → 0.044, p=1024 → 0.031, p=2048 → 0.022.
    # We clamp to [0.05, 0.3]:
    #   - Lower bound 0.05: at p=1024, 0.05×1024 = ~51 features per split,
    #     comparable to RF's sqrt(1024) = 32.  Going below this risks
    #     too few meaningful features appearing at any split.
    #   - Upper bound 0.3: keeps per-node diversity meaningful; beyond 0.3
    #     the effect converges to colsample_bytree alone.
    # Only applied when p > 200 and the data is binary/sparse-count, where
    # the benefit is empirically largest.
    # ------------------------------------------------------------------
    if p > 200 and (profile.binary_feature_fraction > 0.8 or profile.is_sparse_counts):
        csn = round(max(0.05, min(0.3, 1.0 / (p ** 0.5))), 3)
        params["colsample_bynode"] = csn

    # ------------------------------------------------------------------
    # colsample_bylevel  (per-depth-level column sampling)
    # Stacks multiplicatively with colsample_bynode to add an intermediate
    # layer of diversity: the candidate pool for each depth level is first
    # drawn from colsample_bytree, then restricted further by bylevel, then
    # resampled per node via bynode.  For fingerprint data this gives a
    # three-tier sampling that mirrors the extreme randomization of
    # Extra-Trees and has been found to improve AUC on high-dimensional
    # binary datasets (benchmarks vs LightGBM on Towards Data Science).
    # Only applied alongside colsample_bynode (fingerprint-like data).
    # ------------------------------------------------------------------
    if "colsample_bynode" in params:
        params["colsample_bylevel"] = 0.7

    # ------------------------------------------------------------------
    # Regularization (base rules)
    # L1 (reg_alpha) is better for high-dimensional sparse data; it drives
    # uninformative leaf weights exactly to zero, performing implicit
    # feature selection within each tree.  On ECFP fingerprints where
    # ~93% of bits are zero and the effective information density is very
    # low, L1 is the primary regularizer.
    # L2 (reg_lambda) is the XGBoost default and suits dense continuous
    # data.  Keep it at 1.0 as a baseline in all cases.
    # For sparse-count (fingerprint) data with n/p < 5 (the typical drug
    # dataset regime), we combine strong L1 (alpha=1.0) with moderate L2
    # (lambda=1.0), matching the underdetermined + high-dimensional prior.
    # ------------------------------------------------------------------
    if profile.is_sparse_counts:
        if profile.n_p_ratio < 5:
            # Underdetermined fingerprint data: strong L1 for sparsity-
            # inducing regularization on leaf weights.
            params["reg_alpha"] = 1.0
            params["reg_lambda"] = 1.0
        else:
            # Well-determined fingerprint data: moderate L1 sufficient.
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
    # For is_sparse_counts + n/p < 5, the base rules already set strong
    # L1; this block then further scales L2 upward, which is appropriate
    # because both penalties contribute independently.
    # ------------------------------------------------------------------
    if profile.n_p_ratio < 2:
        params["reg_lambda"] = round(params["reg_lambda"] * 2.0, 4)
        params["reg_alpha"] = max(params["reg_alpha"], 0.5)
    elif profile.n_p_ratio < 5:
        params["reg_lambda"] = round(params["reg_lambda"] * 1.5, 4)
        params["reg_alpha"] = max(params["reg_alpha"], 0.1)

    # ------------------------------------------------------------------
    # feature_signal_strength regularization adjustment
    # The mean |Pearson| between individual features and the target is a
    # cheap proxy for dataset difficulty.  When signal is very weak
    # (< 0.02) the regularization set above may be insufficient — scale
    # it up.  When signal is meaningfully strong (> 0.05) the n/p rules
    # may have over-regularized — ease off so the model can exploit the
    # available signal.  The multiplier is applied after all other rules
    # so it only modulates, never replaces, the structural priors.
    # ------------------------------------------------------------------
    sig = profile.feature_signal_strength
    if sig < 0.02:
        params["reg_lambda"] = round(params["reg_lambda"] * 1.5, 4)
        params["reg_alpha"] = round(params["reg_alpha"] * 1.5, 4)
    elif sig > 0.05:
        params["reg_lambda"] = round(params["reg_lambda"] * 0.7, 4)

    # ------------------------------------------------------------------
    # gamma (min_split_loss)
    # Pre-pruning regularizer: a split is accepted only if it reduces the
    # loss by at least gamma.  Probst et al. 2023 ranks this #3 in
    # importance for QSAR; values 0.1-5 consistently prune spurious splits
    # on noisy fingerprint bits.
    # - Sparse-count underdetermined (n/p < 5): 0.5 — fingerprint noise is
    #   highest here; aggressive pre-pruning prevents random bit patterns
    #   from forming splits.
    # - Other underdetermined (n/p < 5): 0.1 — moderate pre-pruning.
    # - Well-determined data: 0 (default) — other regularization suffices.
    # ------------------------------------------------------------------
    if profile.n_p_ratio < 5:
        params["gamma"] = 0.5 if profile.is_sparse_counts else 0.1

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

    # scale_pos_weight: equivalent to sklearn's class_weight="balanced".
    # Always set it — when ratio=1 it is a no-op, and for any imbalance
    # (even mild) it corrects the gradient contribution of each class.
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
