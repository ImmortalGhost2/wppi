"""Model factory mapping method names to scikit-learn regressors.

A single ``make_regressor`` entry point keeps the training pipeline agnostic to
the concrete estimator: callers pass a method name from the config and receive a
ready-to-fit regressor with the project's default hyperparameters (overridable
per family via ``params``).
"""
from typing import Any, Dict

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


def make_regressor(name: str, params: Dict[str, Any] | None = None, seed: int = 42):
    """Build a regressor for ``name`` with project defaults applied.

    The random-forest family covers both the direct path-loss controls
    (``direct_rf`` and its ``*_all_features`` / ``*_geometry`` /
    ``*_sparse_anchor`` variants) and the WallPath-PI residual learner
    ``wallpath_rf``; they share the same estimator and differ only in the
    features and target supplied by the pipeline. The
    Extra Trees family covers ``wallpath_extra``, the calibrated correction
    learner ``wallpath_calibrated``, and the all-feature direct Extra Trees
    control ``direct_extra_all_features``, all configured via
    ``params["extra_trees"]``.
    Hyperparameters may be overridden through ``params[<family>]``. Raises
    ``ValueError`` for an unrecognized name.
    """
    params = params or {}
    name = str(name).lower().strip()
    if name in {"wallpath_rf", "direct_rf", "direct_rf_all_features", "all_feature_rf", "direct_rf_geometry", "direct_rf_sparse_anchor", "random_forest", "rf"}:
        p = dict(n_estimators=120, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=int(seed))
        p.update(params.get("random_forest", params))
        p["random_state"] = int(seed)
        return RandomForestRegressor(**p)
    if name in {"wallpath_extra", "wallpath_calibrated", "direct_extra_all_features", "all_feature_extra", "extra_trees", "extratrees"}:
        p = dict(n_estimators=160, max_depth=24, min_samples_leaf=2, n_jobs=-1, random_state=int(seed))
        p.update(params.get("extra_trees", params))
        p["random_state"] = int(seed)
        return ExtraTreesRegressor(**p)
    if name in {"direct_mlp", "wallpath_mlp", "mlp"}:
        p = dict(hidden_layer_sizes=(64, 64), activation="relu", alpha=1.0e-4, max_iter=400, random_state=int(seed), early_stopping=True)
        p.update(params.get("mlp", params))
        p["random_state"] = int(seed)
        return Pipeline([("scale", StandardScaler()), ("mlp", MLPRegressor(**p))])
    raise ValueError(f"Unsupported regressor name: {name}")


def feature_importance(model, feature_names: list[str]) -> Dict[str, float]:
    """Map feature names to tree-based importances.

    Returns an empty dict for estimators without ``feature_importances_`` (for
    example the MLP pipeline), so callers can treat importance reporting as
    optional.
    """
    raw = model
    if hasattr(model, "named_steps") and "mlp" in model.named_steps:
        return {}
    if hasattr(raw, "feature_importances_"):
        vals = raw.feature_importances_
        return {str(k): float(v) for k, v in zip(feature_names, vals)}
    return {}
