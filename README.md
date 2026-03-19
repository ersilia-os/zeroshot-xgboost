# Zero-shot XGBoost

Zero-shot XGBoost (`zsxgboost`) automatically selects XGBoost hyperparameters for binary classification and regression — no grid search, no cross-validation, no tuning required.

Given a dataset `(X, y)`, it profiles the data (size, dimensionality, sparsity, class imbalance, target skew) and picks parameters grounded in published research and practical guidelines. Optionally it runs a fast multi-preset portfolio comparison and selects whichever configuration validates best on a held-out split.

## Why?

XGBoost works well out of the box, but its defaults are not optimal for every dataset:
- A dataset with 200 samples and 1 024 binary features needs very different regularisation than one with 50 000 samples and 10 continuous features.
- Severely imbalanced classes need `scale_pos_weight`, a patience that scales with the learning rate, and AUC-PR as the stopping metric.
- Sparse integer count data (e.g. Morgan fingerprints) benefit from `max_bin=64`, `grow_policy="lossguide"`, and per-split feature subsampling.
- A skewed regression target should use Tweedie loss, not squared error.

`zsxgboost` encodes these decisions as a set of rules derived from the XGBoost paper, FLAML, AutoGluon, and QSAR modelling guidelines — so you get reasonable parameters without running a single tuning job.

## How it works

### Layer 1 — Zero-shot rules

`inspect(X, y)` computes a `DatasetProfile` (sample count, feature count, sparsity, binary-feature fraction, signal strength, class imbalance, target skew). `get_params(profile)` maps that profile to a full XGBoost parameter dictionary using a table of rules:

| Condition | Parameter effect |
|---|---|
| `n_samples` < 10k / 100k / ≥ 100k | `learning_rate` = 0.1 / 0.05 / 0.02 |
| Any | `early_stopping_rounds` = `50 × (0.1 / lr)` — patience scales with learning rate |
| `n_samples` < 1 000 | `subsample` = 1.0 (no row sampling on tiny datasets) |
| `n_p_ratio` < 2 | `max_depth` ≤ 3, `reg_lambda` ×2, `reg_alpha` ≥ 0.5, `gamma` = 0.1 |
| `n_p_ratio` < 5 | `max_depth` ≤ 4, `reg_lambda` ×1.5, `reg_alpha` ≥ 0.1, `gamma` = 0.1 |
| `binary_feature_fraction` > 0.8 | `max_depth` − 1 (floor 3) |
| `n_features` > 500 | `colsample_bytree` ≤ 0.5 |
| Sparse count data (fingerprints) | lossguide growth, `max_bin=64`, `colsample_bynode=sqrt(p)/p`, L1 reg |
| Imbalance ratio > 1 | `scale_pos_weight` = neg/pos |
| Imbalance ratio > 10 | `eval_metric` = `aucpr`, `max_delta_step` = 1 |
| Regression \|skew\| < 1 | `objective` = `reg:squarederror` |
| Regression \|skew\| ≥ 1, y > 0 | `objective` = `reg:tweedie` |
| Regression \|skew\| ≥ 1, y mixed | `objective` = `reg:pseudohubererror` |

### Layer 2 — Portfolio selection

`ZeroShotXGBClassifier(portfolio=True)` (the default) trains five preset configurations in parallel on a 90/10 validation split and selects the best:

| Preset | Description |
|---|---|
| `internal` | Zero-shot rules from Layer 1 |
| `default` | XGBoost out-of-the-box defaults (lr=0.3, max_depth=6) |
| `flaml` | FLAML zero-shot: 1-NN meta-feature matching on the FLAML portfolio |
| `autogluon` | AutoGluon tabular XGBoost defaults, selected by dataset size and feature type |
| `rf_like` | XGBoost configured as a Random Forest (colsample_bynode=√p/p, subsample=0.632) |

The portfolio comparison is fast: each preset runs for at most 300 rounds with 30-round patience. A preset wins only if its validation score exceeds the baseline (`default`) by a noise-aware threshold (larger for small datasets where single-split AUC is noisy). The winning preset is then retrained on 100% of the data for the calibrated number of rounds.

## Benchmark

Evaluated on 18 ADMET binary classification datasets from the [Therapeutics Data Commons](https://tdcommons.ai/) using 1024-bit ECFP4 Morgan fingerprints, single 80/20 stratified split:

| Model | Mean ROC-AUC rank | Mean PR-AUC rank |
|---|---|---|
| **ZS-XGBoost (portfolio)** | **2.00** | **1.83** |
| Default RF | 1.94 | 2.17 |
| ZS-XGBoost (default only) | 2.61 | 2.39 |
| Default XGBoost | 3.06 | 2.78 |
| Logistic Regression | 3.44 | 3.61 |

Lower rank = better. `zsxgboost` with portfolio selection is **#1 on PR-AUC** (the right metric for imbalanced drug datasets) and statistically tied with Random Forest on ROC-AUC, while being strictly better than plain XGBoost defaults.

## Installation

```bash
pip install git+https://github.com/ersilia-os/zeroshot-xgboost.git
```

Requires Python ≥ 3.10. Core dependencies: `xgboost>=2.0`, `scikit-learn>=1.0`, `scipy>=1.7`, `numpy>=1.21`, `loguru>=0.6`, `rich>=12.0`.

For ONNX export support:
```bash
pip install "zsxgboost[onnx]"
```

## Quick start

```python
from zsxgboost import ZeroShotXGBClassifier, ZeroShotXGBRegressor

# Binary classification — portfolio=True is the default
clf = ZeroShotXGBClassifier()
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]

# Regression
reg = ZeroShotXGBRegressor()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```

Both estimators are sklearn-compatible and work inside `Pipeline`, `cross_val_score`, etc.

## Inspecting the chosen parameters

```python
clf = ZeroShotXGBClassifier()
clf.fit(X_train, y_train)

print(clf.profile_)        # dataset statistics used to select parameters
print(clf.params_)         # full XGBoost parameter dictionary
print(clf.preset_name_)    # which preset won: "internal", "default", "flaml", etc.
print(clf.portfolio_scores_)  # validation AUC of every competing preset
```

Example output for a 5 000-sample, 100-feature dataset with 97:3 class imbalance:

```
DatasetProfile(
  n_samples=5000, n_features=100, n_p_ratio=50.00
  sparsity=0.000, is_sparse_counts=False
  binary_feature_fraction=0.000, feature_signal_strength=0.142
  task='binary_classification'
  imbalance_ratio=28.07
)

{
  'tree_method': 'hist', 'device': 'cpu',
  'learning_rate': 0.05, 'n_estimators': 2000, 'early_stopping_rounds': 100,
  'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8,
  'colsample_bytree': 0.8, 'reg_alpha': 0.0, 'reg_lambda': 1.0,
  'max_bin': 256, 'nthread': 10,
  'objective': 'binary:logistic', 'scale_pos_weight': 28.07, 'eval_metric': 'aucpr'
}

preset_name_: 'internal'
```

## Portfolio selection without full tuning

Set `portfolio=False` to skip preset comparison and use only the zero-shot internal rules. This is faster but gives up the adaptive selection:

```python
clf = ZeroShotXGBClassifier(portfolio=False)
clf.fit(X_train, y_train)
```

## Low-level API

You can use the dataset profiler and parameter selector independently:

```python
from zsxgboost import inspect, get_params

profile = inspect(X_train, y_train, task="binary_classification")
params  = get_params(profile, device="cpu")

import xgboost as xgb
booster = xgb.train(params, xgb.DMatrix(X_train, y_train), num_boost_round=100)
```

`inspect()` accepts dense numpy arrays, pandas DataFrames, and scipy sparse matrices.

## GPU support

```python
clf = ZeroShotXGBClassifier(device="gpu")   # explicit GPU
clf = ZeroShotXGBClassifier(device="auto")  # use GPU if CUDA is available
```

Everything else is identical; `device="cuda"` is set internally in the XGBoost parameter dict.

## ONNX export

After fitting, export to ONNX for deployment in any ONNX-compatible runtime:

```python
clf.to_onnx("classifier.onnx")
reg.to_onnx("regressor.onnx")
```

Run inference with `onnxruntime`:

```python
import onnxruntime as ort
import numpy as np

# Classifier: returns (label int64, probabilities float32 n×2)
sess  = ort.InferenceSession("classifier.onnx")
label, proba = sess.run(None, {"float_input": X_test.astype(np.float32)})

# Regressor: returns (predictions float32 n×1)
sess  = ort.InferenceSession("regressor.onnx")
preds = sess.run(None, {"float_input": X_test.astype(np.float32)})[0].ravel()
```

## API reference

### `ZeroShotXGBClassifier(device="cpu", verbose=False, portfolio=True, nthread=-1)`

| Parameter | Description |
|---|---|
| `device` | `"cpu"`, `"gpu"`, or `"auto"` |
| `verbose` | Print training progress and preset selection details |
| `portfolio` | Run the five-preset portfolio comparison (default `True`) |
| `nthread` | CPU threads for XGBoost; `-1` uses all available cores |

**Attributes after `.fit()`:**

| Attribute | Type | Description |
|---|---|---|
| `profile_` | `DatasetProfile` | Dataset statistics used for parameter selection |
| `params_` | `dict` | Full XGBoost parameter dict of the winning preset |
| `preset_name_` | `str` | Winning preset: `"internal"`, `"default"`, `"flaml"`, `"autogluon"`, or `"rf_like"` |
| `portfolio_scores_` | `dict` | Validation AUC (or −RMSE) for every preset evaluated |
| `best_iteration_` | `int` | Number of boosting rounds in the final model |
| `booster_` | `xgb.Booster` | Trained XGBoost booster |

### `ZeroShotXGBRegressor(device="cpu", verbose=False, portfolio=True, nthread=-1)`

Same interface as the classifier. The loss function is chosen automatically from the target distribution: `reg:squarederror` (|skew| < 1), `reg:tweedie` (|skew| ≥ 1 and y > 0), or `reg:pseudohubererror` (|skew| ≥ 1 with mixed-sign targets).

### `inspect(X, y, task=None) → DatasetProfile`

Profile a dataset without training. `task` can be `"binary_classification"`, `"regression"`, or `None` (auto-detected). Returns a `DatasetProfile` dataclass with all statistics used by `get_params`.

### `get_params(profile, device="cpu", nthread=-1) → dict`

Return the zero-shot XGBoost parameter dictionary for a given profile. Useful for integrating with the low-level `xgb.train()` API.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fuelling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
