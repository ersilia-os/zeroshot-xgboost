# Zero-shot XGBoost

Reasonable zero-shot XGBoost configuration for binary classification and regression.

Given a dataset `(X, y)`, `zsxgboost` selects reasonable XGBoost hyperparameters automatically — no grid search, no cross-validation, no tuning. Parameters are derived entirely from dataset statistics: size, dimensionality, feature types, signal strength, class imbalance, and target distribution.

## Why?

XGBoost works well out of the box, but its defaults are not optimal for every dataset. Choosing parameters manually requires experience and experimentation. `zsxgboost` encodes that experience as a set of rules:

- Large datasets get shallower learning rates; `early_stopping_rounds` scales with `1/lr` so slower learners get appropriate patience
- Underdetermined data (few samples per feature) gets capped tree depth, stronger L1/L2 regularisation, and a non-zero `gamma`
- Small datasets (n < 1 000) get a 3-tree random-forest component per boosting round to reduce variance without cross-validation
- Sparse count data (e.g. Morgan fingerprints) get L1 regularisation, deeper trees, and `max_bin=64` (sufficient for integer values ≤ 10)
- Binary/one-hot features get one depth level removed, since each binary split carries less information than a continuous one
- Severely imbalanced classes get `scale_pos_weight` and AUC-PR as the eval metric
- Skewed regression targets get Tweedie or pseudo-Huber loss instead of MSE

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
  n_samples=5000, n_features=100, n_p_ratio=50.00
  sparsity=0.000, is_sparse_counts=False
  binary_feature_fraction=0.000, feature_signal_strength=0.142
  task='binary_classification'
  imbalance_ratio=28.07
)

{
  'tree_method': 'hist', 'device': 'cpu',
  'learning_rate': 0.05, 'n_estimators': 1000, 'early_stopping_rounds': 100,
  'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8,
  'colsample_bytree': 0.8, 'reg_alpha': 0.0, 'reg_lambda': 1.0,
  'max_bin': 256, 'nthread': 10,
  'objective': 'binary:logistic', 'scale_pos_weight': 28.07, 'eval_metric': 'aucpr'
}
```

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
| Any | `early_stopping_rounds` = `50 × (0.1 / lr)` — scales with learning rate |
| `n_samples` < 200 | `subsample` = 1.0 (no row sampling on tiny datasets) |
| 200 ≤ `n_samples` < 1 000 | `num_parallel_tree` = 3 (RF-style within-round bagging) |
| `n_samples` ≥ 1M | `subsample` = 0.6 |
| `n_p_ratio` < 2 | `max_depth` ≤ 3, `reg_lambda` ×2, `reg_alpha` ≥ 0.5, `gamma` = 0.1 |
| `n_p_ratio` < 5 | `max_depth` ≤ 4, `reg_lambda` ×1.5, `reg_alpha` ≥ 0.1, `gamma` = 0.1 |
| `binary_feature_fraction` > 0.8 | `max_depth` −1 (floor 3) |
| `n_features` > 500 | `colsample_bytree` ≤ 0.5 |
| Sparse count features | `reg_alpha` = 0.1, `max_depth` +1, `colsample_bytree` ×1.5, `max_bin` = 64 |
| Imbalance ratio > 1.5 | `scale_pos_weight` = neg/pos |
| Imbalance ratio > 10 | `eval_metric` = `aucpr` |
| Imbalance ratio > 100 | `max_delta_step` = 1 |
| Regression \|skew\| < 1 | `objective` = `reg:squarederror` |
| Regression \|skew\| ≥ 1, y > 0 | `objective` = `reg:tweedie` |
| Regression \|skew\| ≥ 1, y mixed | `objective` = `reg:pseudohubererror` |

Early stopping is always active (`n_estimators=1000`, patience scales with `learning_rate`). A 10 % stratified validation split is held out internally during `.fit()`.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
