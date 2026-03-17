"""
Sklearn-compatible estimators backed by zero-shot XGBoost parameter selection.

Training always uses the low-level xgb.train() API with QuantileDMatrix so
that quantile boundaries are computed once on the training split and reused
for the validation split via the ref= parameter. This is the fastest path
regardless of dataset size.

Both ZeroShotXGBClassifier and ZeroShotXGBRegressor follow the standard
sklearn fit/predict API. On .fit(), they:
  1. Profile the dataset via inspect()
  2. Select hyperparameters via get_params()
  3. Split off a 10% validation set (stratified for classification)
  4. Build QuantileDMatrix for train and val (shared quantile boundaries)
  5. Train via xgb.train() with early stopping

The chosen parameters are accessible via .params_ after fitting.
The best boosting round is in .best_iteration_.
"""

import os
import tempfile

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import xgboost as xgb

from .inspector import inspect as _inspect, DatasetProfile
from .params import get_params as _get_params
from .utils.logging import logger


_VAL_FRACTION = 0.1
_VAL_MIN_ROWS = 200
_VAL_MIN_MINORITY = 15   # minimum minority-class samples in the validation split
_RANDOM_STATE = 42


class ZeroShotXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary classifier with automatically selected XGBoost hyperparameters.

    Parameters
    ----------
    device : str
        "cpu" or "gpu".
    verbose : bool
        If True, print the chosen parameters before training and XGBoost's
        per-round eval log.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict
    booster_ : xgb.Booster
    best_iteration_ : int
    classes_ : ndarray
    """

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        logger.set_verbosity(self.verbose)
        y = np.asarray(y).ravel()
        profile = _inspect(X, y, task="binary_classification")
        params = _get_params(profile, device=self.device)
        self.profile_ = profile
        self.params_ = params

        logger.info(
            f"Binary classification | n={profile.n_samples}, p={profile.n_features} | "
            f"imbalance_ratio={profile.imbalance_ratio:.2f} | "
            f"sparse_counts={profile.is_sparse_counts}"
        )
        logger.debug(
            f"objective={params['objective']} | eval_metric={params['eval_metric']} | "
            f"lr={params['learning_rate']} | max_depth={params['max_depth']} | "
            f"colsample_bytree={params['colsample_bytree']}"
        )

        X_train, X_val, y_train, y_val, did_split = _validation_split(
            X, y, profile, stratify=True
        )
        logger.debug(
            f"Train split: {len(y_train)} rows | Val split: {len(y_val)} rows"
        )
        self.booster_, self.best_iteration_ = _train(
            X_train, y_train, X_val, y_val, params, self.verbose,
            X_full=X if did_split else None,
            y_full=y if did_split else None,
        )
        logger.success(
            f"Training complete | best_iteration={self.best_iteration_}"
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Return class probabilities, shape (n_samples, 2)."""
        check_is_fitted(self, "booster_")
        dtest = xgb.DMatrix(X)
        prob_pos = self.booster_.predict(
            dtest, iteration_range=(0, self.best_iteration_ + 1)
        )
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X):
        """Return binary predictions (0 or 1)."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to an ONNX file.

        Requires the optional ``onnx`` extras:
            pip install zsxgboost[onnx]

        The exported model accepts a float32 input named ``"float_input"``
        with shape ``(n_samples, n_features)`` and produces two outputs:
          - ``"label"``         int64  (n_samples,)   — predicted class
          - ``"probabilities"`` float32 (n_samples, 2) — [P(0), P(1)]

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        check_is_fitted(self, "booster_")
        wrapper = _booster_to_sklearn_wrapper(
            self.booster_, task="binary_classification"
        )
        _export_onnx(wrapper, path, self.profile_.n_features)


class ZeroShotXGBRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor with automatically selected XGBoost hyperparameters.

    Handles skewed and non-negative targets by selecting an appropriate
    objective function (squarederror, tweedie, or pseudohubererror).

    Parameters
    ----------
    device : str
        "cpu" or "gpu".
    verbose : bool
        If True, print the chosen parameters before training and XGBoost's
        per-round eval log.

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict
    booster_ : xgb.Booster
    best_iteration_ : int
    """

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        logger.set_verbosity(self.verbose)
        y = np.asarray(y).ravel()
        profile = _inspect(X, y, task="regression")
        params = _get_params(profile, device=self.device)
        self.profile_ = profile
        self.params_ = params

        logger.info(
            f"Regression | n={profile.n_samples}, p={profile.n_features} | "
            f"y_skewness={profile.y_skewness:.3f} | "
            f"sparse_counts={profile.is_sparse_counts}"
        )
        logger.debug(
            f"objective={params['objective']} | eval_metric={params['eval_metric']} | "
            f"lr={params['learning_rate']} | max_depth={params['max_depth']} | "
            f"colsample_bytree={params['colsample_bytree']}"
        )

        X_train, X_val, y_train, y_val, did_split = _validation_split(
            X, y, profile, stratify=False
        )
        logger.debug(
            f"Train split: {len(y_train)} rows | Val split: {len(y_val)} rows"
        )
        self.booster_, self.best_iteration_ = _train(
            X_train, y_train, X_val, y_val, params, self.verbose,
            X_full=X if did_split else None,
            y_full=y if did_split else None,
        )
        logger.success(
            f"Training complete | best_iteration={self.best_iteration_}"
        )
        return self

    def predict(self, X):
        """Return continuous predictions."""
        check_is_fitted(self, "booster_")
        dtest = xgb.DMatrix(X)
        return self.booster_.predict(
            dtest, iteration_range=(0, self.best_iteration_ + 1)
        )

    def to_onnx(self, path: str) -> None:
        """
        Export the trained model to an ONNX file.

        Requires the optional ``onnx`` extras:
            pip install zsxgboost[onnx]

        The exported model accepts a float32 input named ``"float_input"``
        with shape ``(n_samples, n_features)`` and produces one output:
          - ``"variable"`` float32 (n_samples, 1) — predicted values

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model.onnx"``.
        """
        check_is_fitted(self, "booster_")
        wrapper = _booster_to_sklearn_wrapper(self.booster_, task="regression")
        _export_onnx(wrapper, path, self.profile_.n_features)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _train(X_train, y_train, X_val, y_val, params: dict, verbose: bool,
           X_full=None, y_full=None):
    """
    Two-phase training to avoid discarding the validation split.

    Phase 1: train on X_train/y_train with early stopping against X_val/y_val
             to find best_iteration.
    Phase 2: retrain on the full dataset (X_full/y_full, which is X_train+X_val
             if provided) for exactly best_iteration rounds with no early stopping.

    The final booster therefore sees 100% of the data at the round count
    calibrated on 90%.  When X_full is None (e.g. the dataset was too small
    to split), only Phase 1 runs.

    Returns (booster, best_iteration).
    """
    max_bin = params.get("max_bin", 256)
    num_boost_round = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]

    # Keys consumed here; not passed to xgb.train() params dict
    _skip = {"n_estimators", "early_stopping_rounds"}
    xgb_params = {k: v for k, v in params.items() if k not in _skip}

    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=max_bin)
    dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain, max_bin=max_bin)

    # Phase 1: find best_iteration via early stopping
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
    )
    best_iter = booster.best_iteration

    # Phase 2: retrain on full data for best_iter rounds (no early stopping)
    if X_full is not None and best_iter > 0:
        dfull = xgb.QuantileDMatrix(X_full, label=y_full, max_bin=max_bin)
        booster = xgb.train(
            xgb_params,
            dfull,
            num_boost_round=best_iter + 1,
            verbose_eval=False,
        )

    return booster, best_iter


def _booster_to_sklearn_wrapper(booster: xgb.Booster, task: str):
    """
    Load a raw Booster into an sklearn wrapper for onnxmltools export.
    Saves to a temp file and reloads via the sklearn API.
    """
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        tmpfile = f.name
    try:
        booster.save_model(tmpfile)
        if task == "binary_classification":
            wrapper = xgb.XGBClassifier()
            wrapper.load_model(tmpfile)
            wrapper.n_classes_ = 2
        else:
            wrapper = xgb.XGBRegressor()
            wrapper.load_model(tmpfile)
    finally:
        os.unlink(tmpfile)
    return wrapper


def _export_onnx(model, path: str, n_features: int) -> None:
    """Convert an XGBoost sklearn estimator to ONNX and write to path."""
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "ONNX export requires optional dependencies. "
            "Install them with:  pip install zsxgboost[onnx]"
        )
    onnx_model = convert_xgboost(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
    )
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def _validation_split(X, y, profile: DatasetProfile, stratify: bool):
    """
    Split off a small validation set for early stopping.
    Falls back to reusing the full set when n_samples is very small.

    For binary classification, the validation fraction is dynamically raised
    when the minority class is small, ensuring at least _VAL_MIN_MINORITY
    minority samples in the validation set.  With too few minority samples,
    AUC estimates are noisy and early stopping can trigger at a suboptimal
    round (e.g. HIA_Hou has ~62 minority samples in training; 10% val gives
    only 6, making each rank-swap change AUC by ~0.017 — far too noisy).

    Returns (X_train, X_val, y_train, y_val, did_split).
    did_split=False means the dataset was too small to split; train==full.
    """
    if profile.n_samples < _VAL_MIN_ROWS:
        return X, X, y, y, False

    val_fraction = _VAL_FRACTION
    if stratify and profile.task == "binary_classification":
        # Estimate minority count in the full fitting set from the profile.
        ratio = profile.imbalance_ratio        # neg / pos
        minority_count = int(profile.n_samples * min(ratio, 1.0) / (1.0 + ratio))
        if minority_count > 0:
            needed = _VAL_MIN_MINORITY / minority_count
            val_fraction = max(_VAL_FRACTION, min(0.25, needed))

    strat = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        random_state=_RANDOM_STATE,
        stratify=strat,
    )
    return X_train, X_val, y_train, y_val, True
