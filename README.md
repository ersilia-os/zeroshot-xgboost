# Zero-shot XGBoost

`zsxgboost` automatically selects XGBoost hyperparameters for classification and regression — no grid search, no cross-validation, no tuning required.

Given a dataset `(X, y)`, it profiles the data (size, dimensionality, sparsity, class imbalance, target skew) and selects parameters grounded in published research and practical guidelines. Optionally it runs a fast multi-preset portfolio comparison and picks whichever configuration validates best on a held-out split.

## Installation

```bash
pip install git+https://github.com/ersilia-os/zeroshot-xgboost.git
```

Requires Python ≥ 3.10.

## Quick start

```python
from zsxgboost import ZeroShotXGBClassifier, ZeroShotXGBRegressor

# Classification
clf = ZeroShotXGBClassifier()
clf.fit(X_train, y_train)
preds  = clf.predict(X_test)          # binary labels: 0 or 1
proba  = clf.predict_proba(X_test)    # shape (n, 2) — [P(0), P(1)]

# Regression
reg = ZeroShotXGBRegressor()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)           # shape (n,)
```

Both estimators are sklearn-compatible and work inside `Pipeline`, `cross_val_score`, etc.

## Saving and loading models

### Save

```python
clf.save("my_model/")            # default: ONNX format
clf.save("my_model/", onnx=True) # explicit ONNX
clf.save("my_model/", onnx=False) # joblib format
```

`save()` always writes two files:
- `xgboost.onnx` (or `xgboost.joblib`) — the serialised model
- `xgboost.json` — fit metadata (parameters, dataset profile, preset scores, best iteration)

### Load and run inference

Use `XGBArtifact` to load a saved model without needing to re-fit:

```python
from zsxgboost import XGBArtifact

artifact = XGBArtifact.load("my_model/")
out = artifact.run(X_test)
# Classification: shape (n, 2) — [P(class=0), P(class=1)]
# Regression:     shape (n,)   — predicted values
```

`load()` automatically detects whether the saved format is ONNX or joblib. The fit metadata is available at `artifact.metadata`.

## Inspecting the chosen parameters

```python
clf.fit(X_train, y_train)

print(clf.profile_)           # dataset statistics
print(clf.params_)            # full XGBoost parameter dict
print(clf.preset_name_)       # winning preset: "internal", "default", "flaml", etc.
print(clf.portfolio_scores_)  # validation score per preset
print(clf.best_iteration_)    # boosting rounds in the final model
```

## How it works

### Layer 1 — Zero-shot rules

`inspect(X, y)` computes a `DatasetProfile` (sample count, feature count, sparsity, signal strength, class imbalance, target skew). `get_params(profile)` maps that profile to a full XGBoost parameter dictionary. Key decisions:

| Condition | Effect |
|---|---|
| `n_samples` < 10k / 100k / ≥ 100k | `learning_rate` = 0.1 / 0.05 / 0.02 |
| Any | `early_stopping_rounds` = `50 × (0.1 / lr)` |
| `n_samples` < 1 000 | `subsample` = 1.0 |
| `n_p_ratio` < 2 or < 5 | Shallower trees, heavier regularisation, `gamma` > 0 |
| Sparse count data (fingerprints) | Lossguide growth, `max_bin=64`, per-split column sampling |
| Imbalance ratio > 1 | `scale_pos_weight` = neg/pos |
| Imbalance ratio > 10 | `eval_metric` = `aucpr`, `max_delta_step` = 1 |
| Regression \|skew\| < 1 | `reg:squarederror` |
| Regression \|skew\| ≥ 1, y > 0 | `reg:tweedie` |
| Regression \|skew\| ≥ 1, mixed y | `reg:pseudohubererror` |

### Layer 2 — Portfolio selection

With `portfolio=True` (the default), five preset configurations are trained in parallel on a 90/10 validation split and the best is selected:

| Preset | Description |
|---|---|
| `internal` | Zero-shot rules from Layer 1 |
| `default` | XGBoost out-of-the-box defaults (lr=0.3, max_depth=6) |
| `flaml` | FLAML zero-shot: 1-NN meta-feature matching on the FLAML portfolio |
| `autogluon` | AutoGluon tabular XGBoost defaults, selected by dataset size and feature type |
| `rf_like` | XGBoost configured as a Random Forest (colsample_bynode=√p/p, subsample=0.632) |

Set `portfolio=False` to skip this step and use only the zero-shot rules (faster, useful as a baseline).

## Benchmark

Evaluated on 18 ADMET binary classification datasets from the [Therapeutics Data Commons](https://tdcommons.ai/) using 1024-bit ECFP4 Morgan fingerprints, single 80/20 stratified split:

| Model | Mean ROC-AUC rank | Mean PR-AUC rank |
|---|---|---|
| **ZS-XGBoost (portfolio)** | **2.00** | **1.83** |
| Default RF | 1.94 | 2.17 |
| ZS-XGBoost (default only) | 2.61 | 2.39 |
| Default XGBoost | 3.06 | 2.78 |
| Logistic Regression | 3.44 | 3.61 |

Lower rank = better.

## API reference

### `ZeroShotXGBClassifier`

```python
ZeroShotXGBClassifier(device="cpu", verbose=False, portfolio=True, nthread=-1)
```

| Method | Description |
|---|---|
| `.fit(X, y)` | Train the model |
| `.predict(X)` | Binary labels: 0 or 1 |
| `.predict_proba(X)` | Class probabilities, shape `(n, 2)` |
| `.save(directory, onnx=True)` | Save model and metadata to a directory |

Attributes after `.fit()`: `profile_`, `params_`, `preset_name_`, `portfolio_scores_`, `best_iteration_`, `booster_`.

### `ZeroShotXGBRegressor`

```python
ZeroShotXGBRegressor(device="cpu", verbose=False, portfolio=True, nthread=-1)
```

| Method | Description |
|---|---|
| `.fit(X, y)` | Train the model |
| `.predict(X)` | Predicted values, shape `(n,)` |
| `.save(directory, onnx=True)` | Save model and metadata to a directory |

Loss function is chosen automatically: `reg:squarederror`, `reg:tweedie`, or `reg:pseudohubererror`.

### `XGBArtifact`

```python
artifact = XGBArtifact.load(directory)  # load a saved model
out = artifact.run(X)                   # run inference
```

| Attribute | Description |
|---|---|
| `artifact.task` | `"classification"` or `"regression"` |
| `artifact.metadata` | Full contents of `xgboost.json` |

`run()` returns shape `(n, 2)` for classification and `(n,)` for regression.

### Low-level API

```python
from zsxgboost import inspect, get_params

profile = inspect(X, y, task="classification")  # or "regression", or None for auto
params  = get_params(profile, device="cpu")

import xgboost as xgb
booster = xgb.train(params, xgb.DMatrix(X, y), num_boost_round=100)
```

## GPU support

```python
clf = ZeroShotXGBClassifier(device="gpu")   # explicit GPU
clf = ZeroShotXGBClassifier(device="auto")  # use GPU if CUDA is available
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fuelling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
