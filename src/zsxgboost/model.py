"""
Sklearn-compatible estimators backed by zero-shot XGBoost parameter selection.

Both ZeroShotXGBClassifier and ZeroShotXGBRegressor follow the standard
sklearn fit/predict API. On .fit(), they:
  1. Profile the dataset via inspect()
  2. Select hyperparameters via get_params()
  3. Split off a validation set for early stopping
  4. Train an XGBoost model

The chosen parameters are accessible via the .params_ property after fitting.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import xgboost as xgb

from .inspector import inspect as _inspect, DatasetProfile
from .params import get_params as _get_params


_VAL_FRACTION = 0.1   # fraction of training data used as early-stopping validation set
_VAL_MIN_ROWS = 200   # minimum rows to bother with a validation split
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
        training log.
    """

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        """
        Profile X and y, select parameters, and train.

        Parameters
        ----------
        X : array-like or scipy sparse, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)  — binary labels (0/1)

        Returns
        -------
        self
        """
        y = np.asarray(y).ravel()
        profile = _inspect(X, y, task="binary_classification")
        params = _get_params(profile, device=self.device)
        self.profile_ = profile
        self.params_ = params

        if self.verbose:
            print(profile)
            print("Selected parameters:", params)

        X_train, X_val, y_train, y_val = _validation_split(
            X, y, profile, stratify=True
        )

        xgb_params = {k: v for k, v in params.items()
                      if k not in ("early_stopping_rounds", "eval_metric")}
        self.model_ = xgb.XGBClassifier(
            **xgb_params,
            early_stopping_rounds=params["early_stopping_rounds"],
            eval_metric=params["eval_metric"],
            verbosity=1 if self.verbose else 0,
        )
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.verbose,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Return class probabilities, shape (n_samples, 2)."""
        check_is_fitted(self, "model_")
        prob_pos = self.model_.predict_proba(X)[:, 1]
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, X):
        """Return binary predictions (0 or 1)."""
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

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
        check_is_fitted(self, "model_")
        _export_onnx(self.model_, path, self.profile_.n_features)


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
        training log.
    """

    def __init__(self, device: str = "cpu", verbose: bool = False):
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        """
        Profile X and y, select parameters, and train.

        Parameters
        ----------
        X : array-like or scipy sparse, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)  — continuous target

        Returns
        -------
        self
        """
        y = np.asarray(y).ravel()
        profile = _inspect(X, y, task="regression")
        params = _get_params(profile, device=self.device)
        self.profile_ = profile
        self.params_ = params

        if self.verbose:
            print(profile)
            print("Selected parameters:", params)

        X_train, X_val, y_train, y_val = _validation_split(
            X, y, profile, stratify=False
        )

        xgb_params = {k: v for k, v in params.items()
                      if k not in ("early_stopping_rounds", "eval_metric")}
        self.model_ = xgb.XGBRegressor(
            **xgb_params,
            early_stopping_rounds=params["early_stopping_rounds"],
            eval_metric=params["eval_metric"],
            verbosity=1 if self.verbose else 0,
        )
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        """Return continuous predictions."""
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

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
        check_is_fitted(self, "model_")
        _export_onnx(self.model_, path, self.profile_.n_features)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    Falls back to a copy of the training set when n_samples is too small.
    """
    n = profile.n_samples
    if n < _VAL_MIN_ROWS:
        # Too few rows to split: reuse training data for eval (slight overfit
        # is acceptable at this scale compared to losing training samples).
        return X, X, y, y

    strat = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=_VAL_FRACTION,
        random_state=_RANDOM_STATE,
        stratify=strat,
    )
    return X_train, X_val, y_train, y_val
