"""
zsxgboost — Zero-shot XGBoost parameter selection.

Public API
----------
inspect(X, y, task=None) -> DatasetProfile
    Profile a dataset and return its characteristics.

get_params(profile, device="cpu") -> dict
    Select XGBoost hyperparameters from a DatasetProfile.

ZeroShotXGBClassifier(device="cpu", verbose=False)
    Sklearn-compatible binary classifier with auto-selected parameters.

ZeroShotXGBRegressor(device="cpu", verbose=False)
    Sklearn-compatible regressor with auto-selected parameters.

Typical usage
-------------
    from zsxgboost import ZeroShotXGBClassifier, ZeroShotXGBRegressor

    clf = ZeroShotXGBClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(clf.params_)   # inspect the chosen hyperparameters

    reg = ZeroShotXGBRegressor()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

Low-level usage
---------------
    from zsxgboost import inspect, get_params

    profile = inspect(X, y, task="binary_classification")
    params = get_params(profile, device="gpu")
    # params is a plain dict — pass directly to xgb.XGBClassifier(**params)
"""

from .inspector import inspect, DatasetProfile
from .params import get_params
from .model import ZeroShotXGBClassifier, ZeroShotXGBRegressor

__all__ = [
    "inspect",
    "DatasetProfile",
    "get_params",
    "ZeroShotXGBClassifier",
    "ZeroShotXGBRegressor",
]
