"""
Sklearn-compatible estimators backed by zero-shot XGBoost parameter selection.

Training uses the low-level xgb.train() API with QuantileDMatrix so that
quantile boundaries are computed once on the training split and reused for
the validation split via the ref= parameter.

On .fit(), both estimators run a two-phase portfolio process:

  Phase 0 – Portfolio selection (when a genuine validation split exists):
    Five preset configurations are trained in parallel on the 90% training
    split with early stopping against the 10% validation split:
      1. internal   – zero-shot rules from params.py (dataset-profiling based)
      2. default    – XGBoost out-of-the-box defaults (lr=0.3, max_depth=6)
      3. flaml      – FLAML zero-shot: 1-NN portfolio selection on meta-features
                      (microsoft/FLAML, flaml/default/xgboost/*.json, MIT license)
      4. autogluon  – AutoGluon tabular XGBoost defaults (lr=0.1, max_depth=6)
                      (autogluon/autogluon tabular/.../xgboost, Apache-2 license)
      5. rf         – XGBoost as boosted Random Forest (num_parallel_tree=3,
                      colsample_bynode=sqrt(p)/p)
    The preset with the highest validation metric score wins.

  Phase 1 – Retrain winner on 100% of the data for exactly best_iteration
    rounds (no early stopping), so the final model sees all training samples
    at the round count calibrated on 90%.

  Fallback – When the dataset is too small to split (n < 200), the internal
    preset is used directly (no portfolio comparison).

The winning preset name is stored in .preset_name_ after .fit().
The chosen parameters are accessible via .params_.
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
from .presets import (
    xgb_default_params, flaml_params, autogluon_params, rf_params,
    MAXIMIZE_METRICS,
)
from .utils.logging import logger


_VAL_FRACTION    = 0.1
_VAL_MIN_ROWS    = 200
_VAL_MIN_MINORITY = 15   # minimum minority-class samples in the validation split
_RANDOM_STATE    = 42

# Keys that guide training but are not native XGBoost parameters
_META_KEYS = frozenset({"n_estimators", "early_stopping_rounds"})


class ZeroShotXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary classifier with automatically selected XGBoost hyperparameters.

    Parameters
    ----------
    device : str
        "cpu" or "gpu".
    verbose : bool
        If True, log chosen parameters and winning preset name.
    portfolio : bool
        If True (default), train all five preset configurations on a validation
        split and select the best.  If False, use the internal zero-shot preset
        only (faster; useful as a baseline or when n is very small).

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict        — hyperparameters of the winning preset
    preset_name_ : str    — which of the 5 presets won ("internal", "default",
                            "flaml", "autogluon", or "rf")
    portfolio_scores_ : dict — val scores for every preset (empty when
                               portfolio=False or dataset too small to split)
    booster_ : xgb.Booster
    best_iteration_ : int
    classes_ : ndarray
    """

    def __init__(self, device: str = "cpu", verbose: bool = False,
                 portfolio: bool = True):
        self.device = device
        self.verbose = verbose
        self.portfolio = portfolio

    def fit(self, X, y):
        logger.set_verbosity(self.verbose)
        y = np.asarray(y).ravel()
        profile = _inspect(X, y, task="binary_classification")
        self.profile_ = profile

        logger.info(
            f"Binary classification | n={profile.n_samples}, p={profile.n_features} | "
            f"imbalance_ratio={profile.imbalance_ratio:.2f} | "
            f"sparse_counts={profile.is_sparse_counts}"
        )

        X_train, X_val, y_train, y_val, did_split = _validation_split(
            X, y, profile, stratify=True
        )
        logger.debug(
            f"Train split: {len(y_train)} rows | Val split: {len(y_val)} rows"
        )

        if did_split:
            if self.portfolio:
                best_name, best_params, best_iter, scores = _portfolio_select(
                    X_train, y_train, X_val, y_val, profile, self.device
                )
                self.preset_name_ = best_name
                self.params_ = best_params
                self.portfolio_scores_ = scores
            else:
                best_params = _get_params(profile, device=self.device)
                _, best_iter, _ = _train_phase1(
                    X_train, y_train, X_val, y_val, best_params, verbose=self.verbose
                )
                self.preset_name_ = "internal"
                self.params_ = best_params
                self.portfolio_scores_ = {}
            logger.debug(
                f"objective={best_params['objective']} | "
                f"eval_metric={best_params['eval_metric']} | "
                f"lr={best_params['learning_rate']} | "
                f"colsample_bytree={best_params['colsample_bytree']}"
            )
            final_booster = _train_phase2(X, y, best_params, best_iter)
        else:
            # Dataset too small to split: use internal preset only
            params = _get_params(profile, device=self.device)
            self.params_ = params
            self.preset_name_ = "internal"
            self.portfolio_scores_ = {}
            final_booster, best_iter, _ = _train_phase1(
                X, y, X, y, params, verbose=self.verbose
            )

        self.booster_ = final_booster
        self.best_iteration_ = best_iter
        logger.success(
            f"Training complete | preset={self.preset_name_} | "
            f"best_iteration={self.best_iteration_}"
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
    objective function (squarederror, tweedie, or pseudohubererror) for the
    internal preset.  External presets use squarederror for simplicity.

    Parameters
    ----------
    device : str
        "cpu" or "gpu".
    verbose : bool
        If True, log chosen parameters and winning preset name.
    portfolio : bool
        If True (default), train all five preset configurations on a validation
        split and select the best.  If False, use the internal zero-shot preset
        only (faster; useful as a baseline or when n is very small).

    Attributes (after .fit())
    --------------------------
    profile_ : DatasetProfile
    params_ : dict
    preset_name_ : str
    portfolio_scores_ : dict
    booster_ : xgb.Booster
    best_iteration_ : int
    """

    def __init__(self, device: str = "cpu", verbose: bool = False,
                 portfolio: bool = True):
        self.device = device
        self.verbose = verbose
        self.portfolio = portfolio

    def fit(self, X, y):
        logger.set_verbosity(self.verbose)
        y = np.asarray(y).ravel()
        profile = _inspect(X, y, task="regression")
        self.profile_ = profile

        logger.info(
            f"Regression | n={profile.n_samples}, p={profile.n_features} | "
            f"y_skewness={profile.y_skewness:.3f} | "
            f"sparse_counts={profile.is_sparse_counts}"
        )

        X_train, X_val, y_train, y_val, did_split = _validation_split(
            X, y, profile, stratify=False
        )
        logger.debug(
            f"Train split: {len(y_train)} rows | Val split: {len(y_val)} rows"
        )

        if did_split:
            if self.portfolio:
                best_name, best_params, best_iter, scores = _portfolio_select(
                    X_train, y_train, X_val, y_val, profile, self.device
                )
                self.preset_name_ = best_name
                self.params_ = best_params
                self.portfolio_scores_ = scores
            else:
                best_params = _get_params(profile, device=self.device)
                _, best_iter, _ = _train_phase1(
                    X_train, y_train, X_val, y_val, best_params, verbose=self.verbose
                )
                self.preset_name_ = "internal"
                self.params_ = best_params
                self.portfolio_scores_ = {}
            logger.debug(
                f"objective={best_params['objective']} | "
                f"eval_metric={best_params['eval_metric']} | "
                f"lr={best_params['learning_rate']} | "
                f"colsample_bytree={best_params['colsample_bytree']}"
            )
            final_booster = _train_phase2(X, y, best_params, best_iter)
        else:
            params = _get_params(profile, device=self.device)
            self.params_ = params
            self.preset_name_ = "internal"
            self.portfolio_scores_ = {}
            final_booster, best_iter, _ = _train_phase1(
                X, y, X, y, params, verbose=self.verbose
            )

        self.booster_ = final_booster
        self.best_iteration_ = best_iter
        logger.success(
            f"Training complete | preset={self.preset_name_} | "
            f"best_iteration={self.best_iteration_}"
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

def _train_phase1(X_train, y_train, X_val, y_val, params: dict, verbose: bool):
    """
    Phase 1: train on X_train/y_train with early stopping against X_val/y_val.

    Returns (booster, best_iteration, comparable_score) where comparable_score
    is normalised so that higher is always better (AUC/AUCPR are returned
    as-is; RMSE and other minimisation metrics are negated).
    """
    max_bin = params.get("max_bin", 256)
    num_boost_round      = params["n_estimators"]
    early_stopping_rounds = params["early_stopping_rounds"]

    xgb_params = {k: v for k, v in params.items() if k not in _META_KEYS}

    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=max_bin)
    dval   = xgb.QuantileDMatrix(X_val,   label=y_val,   ref=dtrain, max_bin=max_bin)

    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
    )
    best_iter = booster.best_iteration
    metric    = xgb_params.get("eval_metric", "rmse")
    score     = booster.best_score
    if metric not in MAXIMIZE_METRICS:
        score = -score  # normalise: higher is always better for comparison

    return booster, best_iter, score


def _train_phase2(X_full, y_full, params: dict, best_iter: int):
    """
    Phase 2: retrain on the full dataset for exactly best_iter+1 rounds.

    No early stopping — the round count was calibrated in phase 1.
    The final model therefore sees 100% of the data.
    """
    max_bin    = params.get("max_bin", 256)
    xgb_params = {k: v for k, v in params.items() if k not in _META_KEYS}

    dfull = xgb.QuantileDMatrix(X_full, label=y_full, max_bin=max_bin)
    booster = xgb.train(
        xgb_params,
        dfull,
        num_boost_round=max(1, best_iter + 1),
        verbose_eval=False,
    )
    return booster


def _portfolio_select(X_train, y_train, X_val, y_val,
                      profile: DatasetProfile, device: str):
    """
    Train all five presets on (X_train, X_val) with early stopping.
    Return (best_preset_name, best_params, best_iteration, scores_dict).

    scores_dict maps preset name → comparable validation score
    (higher is always better; minimisation metrics are negated).

    All presets use verbose=False during portfolio comparison; only the
    winning preset name and score are logged at INFO level.
    """
    candidates = [
        ("internal",  _get_params(profile, device)),
        ("default",   xgb_default_params(profile, device)),
        ("flaml",     flaml_params(profile, device)),
        ("autogluon", autogluon_params(profile, device)),
        ("rf",        rf_params(profile, device)),
    ]

    scores     = {}
    best_score = float("-inf")
    best_name  = None
    best_params = None
    best_iter   = 0

    for name, params in candidates:
        try:
            _, b_iter, score = _train_phase1(
                X_train, y_train, X_val, y_val, params, verbose=False
            )
            scores[name] = score
            logger.debug(
                f"[portfolio] {name:10s}: score={score:+.4f}  iter={b_iter}"
            )
            if score > best_score:
                best_score  = score
                best_name   = name
                best_params = params
                best_iter   = b_iter
        except Exception as exc:
            logger.debug(f"[portfolio] {name} failed: {exc}")
            scores[name] = float("nan")

    if best_name is None:
        # All presets failed (extremely unusual) — fall back to internal
        logger.debug("[portfolio] all presets failed; using internal preset")
        params = _get_params(profile, device)
        _, best_iter, best_score = _train_phase1(
            X_train, y_train, X_val, y_val, params, verbose=False
        )
        best_name   = "internal"
        best_params = params
        scores["internal"] = best_score

    logger.info(
        f"Portfolio winner: {best_name}  (val_score={best_score:+.4f})"
    )
    return best_name, best_params, best_iter, scores


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
        ratio = profile.imbalance_ratio
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
