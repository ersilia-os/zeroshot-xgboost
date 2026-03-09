# Zero-shot XGBoost

Reasonable zero-shot XGBoost configuration for binary classification and regression.

Given a dataset `(X, y)`, `zsxgboost` selects reasonable XGBoost hyperparameters automatically — no grid search, no cross-validation, no tuning. Parameters are derived entirely from dataset statistics: size, feature sparsity, class imbalance, and target distribution.

## Why?

XGBoost works well out of the box, but its defaults are not optimal for every dataset. Choosing parameters manually requires experience and experimentation. `zsxgboost` encodes that experience as a set of rules:

- Large datasets get shallower learning rates and smaller subsampling
- Sparse count data (e.g. Morgan fingerprints) get L1 regularisation and deeper trees
- Severely imbalanced classes get `scale_pos_weight` and AUC-PR as the eval metric
- Skewed regression targets get Tweedie or pseudo-Huber loss instead of MSE
- Memory-constrained regimes (many rows, many columns) get reduced `max_bin` and column subsampling

## Installation

```bash
pip install git+https://github.com/ersilia-os/zeroshot-xgboost.git
```

Requires Python ≥ 3.10. Dependencies: `xgboost>=2.0`, `scikit-learn>=1.0`, `scipy>=1.7`, `numpy>=1.21`.

## Quick start

```python
from zsxgboost import ZeroShotXGBClassifier, ZeroShotXGBRegressor

# Binary classification
clf = ZeroShotXGBClassifier()
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]

# Regression
reg = ZeroShotXGBRegressor()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
```

Both estimators are sklearn-compatible and can be used inside `Pipeline`, `cross_val_score`, etc.

## Inspecting the chosen parameters

```python
clf = ZeroShotXGBClassifier()
clf.fit(X_train, y_train)

print(clf.profile_)   # dataset statistics used to make the decision
print(clf.params_)    # the full XGBoost param dict
```

Example output for a 5 000-sample, 100-feature dataset with 97:3 class imbalance:

```
DatasetProfile(
  n_samples=5000, n_features=100
  sparsity=0.000, is_sparse_counts=False
  task='binary_classification'
  imbalance_ratio=28.07
)

{
  'tree_method': 'hist', 'device': 'cpu',
  'learning_rate': 0.05, 'n_estimators': 1000, 'early_stopping_rounds': 50,
  'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8,
  'colsample_bytree': 0.8, 'reg_alpha': 0.0, 'reg_lambda': 1.0,
  'max_bin': 256, 'nthread': 10,
  'objective': 'binary:logistic', 'scale_pos_weight': 28.07, 'eval_metric': 'aucpr'
}
```

## Low-level API

If you only want the parameter dict and will handle model training yourself:

```python
from zsxgboost import inspect, get_params
import xgboost as xgb

profile = inspect(X, y, task="binary_classification")
params = get_params(profile, device="cpu")

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

`task` can be `"binary_classification"` or `"regression"`. If omitted, it is inferred from `y` (two unique 0/1 values → classification).

## GPU support

```python
clf = ZeroShotXGBClassifier(device="gpu")
```

Sets `device="cuda"` internally. Everything else is identical.

## ONNX export

Trained models can be exported to ONNX for deployment in any ONNX-compatible runtime.

Install the optional dependencies first:

```bash
pip install zsxgboost[onnx]
```

Then export after fitting:

```python
clf.to_onnx("classifier.onnx")
reg.to_onnx("regressor.onnx")
```

Run inference with `onnxruntime`:

```python
import onnxruntime as ort
import numpy as np

# Classifier: outputs label (int64) and probabilities (float32, n×2)
sess = ort.InferenceSession("classifier.onnx")
label, proba = sess.run(None, {"float_input": X_test.astype(np.float32)})

# Regressor: outputs variable (float32, n×1)
sess = ort.InferenceSession("regressor.onnx")
preds = sess.run(None, {"float_input": X_test.astype(np.float32)})[0].ravel()
```

## Parameter selection rules

| Condition | Effect |
|-----------|--------|
| `n_samples` < 10k / 100k / ≥ 100k | `learning_rate` = 0.1 / 0.05 / 0.02 |
| `n_samples` ≥ 1M | `subsample` = 0.6, `max_bin` = 128 |
| `n_features` > 500 | `max_bin` = 128, `colsample_bytree` ≤ 0.5 |
| Sparse count features | `reg_alpha` = 0.1, `max_depth` +1, `colsample_bytree` ×1.5 |
| Imbalance ratio > 1.5 | `scale_pos_weight` = neg/pos |
| Imbalance ratio > 10 | `eval_metric` = `aucpr` |
| Imbalance ratio > 100 | `max_delta_step` = 1 |
| Regression \|skew\| < 1 | `objective` = `reg:squarederror` |
| Regression \|skew\| ≥ 1, y > 0 | `objective` = `reg:tweedie` |
| Regression \|skew\| ≥ 1, y mixed | `objective` = `reg:pseudohubererror` |

Early stopping is always active (`n_estimators=1000`, `early_stopping_rounds=50`). A 10 % stratified validation split is held out internally during `.fit()`.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
