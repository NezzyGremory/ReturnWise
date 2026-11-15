"""Leakage-safe model registry, cross-validation benchmark, and bounded tuning."""

from __future__ import annotations

import importlib.util
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from returnwise.features import ReturnWiseFeatureEngineer

SCORING = ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1")


class CatBoostFrameCleaner(BaseEstimator, TransformerMixin):
    """Keep feature names and normalize missing values for native CatBoost categories."""

    def __init__(self, categorical_columns):
        self.categorical_columns = list(categorical_columns)

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        frame = ReturnWiseFeatureEngineer().transform(X)
        for col in frame.columns:
            if col in self.categorical_columns or not pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)
            else:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return frame


class SoftVotingModel(BaseEstimator, ClassifierMixin):
    """Equal-weight probability blend fitted from the supplied training rows."""

    def __init__(self, model_names, mode="quick", seed=42, parameters=None):
        self.model_names = model_names
        self.mode = mode
        self.seed = seed
        self.parameters = parameters

    def fit(self, X, y):
        candidates, skipped = model_registry(pd.DataFrame(X), self.mode, self.seed)
        fitted = []
        for name in self.model_names:
            if name not in candidates:
                raise ValueError(f"Blend member `{name}` is unavailable: {skipped.get(name, 'not in model registry')}")
            model = clone_pipeline(candidates[name])
            params = (self.parameters or {}).get(name)
            if params:
                model.set_params(**params)
            model.fit(X, y)
            fitted.append(model)
        self.models_ = fitted
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not hasattr(self, "models_"):
            raise RuntimeError("Fit the probability blend before requesting predictions.")
        probabilities = np.mean([_probabilities(model, X) for model in self.models_], axis=0)
        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def resolve_candidate(X: pd.DataFrame, model_name: str, mode: str = "quick", seed: int = 42, parameters: dict | None = None):  # noqa: N803
    prefix = "Soft voting: "
    if model_name.startswith(prefix):
        members = tuple(part.strip() for part in model_name[len(prefix) :].split(" + "))
        if len(members) < 2:
            raise ValueError("A soft-voting candidate requires at least two named members.")
        return SoftVotingModel(members, mode=mode, seed=seed, parameters=parameters)
    candidates, skipped = model_registry(X, mode, seed)
    if model_name not in candidates:
        raise ValueError(f"Model `{model_name}` is unavailable: {skipped.get(model_name, 'not in model registry')}")
    model = clone_pipeline(candidates[model_name])
    if parameters:
        model.set_params(**parameters)
    return model


def _feature_groups(X: pd.DataFrame):  # noqa: N803
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, dense: bool = False, scale: bool = False):  # noqa: N803
    numeric, categorical = _feature_groups(X)
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler(with_mean=False)))
    numeric_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=not dense)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", cat_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.3 if not dense else 0.0,
        verbose_feature_names_out=False,
    )


def _pipeline(X: pd.DataFrame, estimator, *, scale: bool = False, dense: bool = False):  # noqa: N803
    return Pipeline(
        [
            ("features", ReturnWiseFeatureEngineer()),
            ("preprocess", build_preprocessor(X, dense=dense, scale=scale)),
            ("model", estimator),
        ]
    )


def _hist_pipeline(X: pd.DataFrame, seed: int, max_iter: int):  # noqa: N803
    engineered = ReturnWiseFeatureEngineer().fit_transform(X)
    numeric, categorical = _feature_groups(engineered)
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        [
            ("features", ReturnWiseFeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, random_state=seed)),
        ]
    )


def model_registry(X: pd.DataFrame, mode: str = "quick", seed: int = 42) -> tuple[dict[str, object], dict[str, str]]:  # noqa: N803
    """Create available candidate pipelines; optional families fail independently."""
    quick = mode.lower() == "quick"
    trees = 32 if quick else 300
    iterations = 48 if quick else 250
    candidates: dict[str, object] = {
        "Dummy (prior)": _pipeline(X, DummyClassifier(strategy="prior")),
        "Logistic regression": _pipeline(X, LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
        "Random forest": _pipeline(
            X,
            RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
        ),
        "Extra trees": _pipeline(
            X,
            ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed),
        ),
        "HistGradientBoosting": _hist_pipeline(X, seed, iterations),
    }
    skipped: dict[str, str] = {}

    if importlib.util.find_spec("catboost"):
        try:
            from catboost import CatBoostClassifier

            engineered = ReturnWiseFeatureEngineer().fit_transform(X)
            cats = [c for c in engineered.columns if not pd.api.types.is_numeric_dtype(engineered[c])]
            candidates["CatBoost"] = Pipeline(
                [
                    ("features", CatBoostFrameCleaner(cats)),
                    (
                        "model",
                        CatBoostClassifier(
                            iterations=iterations,
                            depth=7,
                            learning_rate=0.06,
                            verbose=False,
                            allow_writing_files=False,
                            random_seed=seed,
                            thread_count=1,
                        ),
                    ),
                ]
            )
        except Exception as exc:  # dependency can be installed but binary-incompatible
            skipped["CatBoost"] = f"Unavailable at import: {exc.__class__.__name__}: {exc}"
    else:
        skipped["CatBoost"] = "Optional dependency `catboost` is not installed."

    if importlib.util.find_spec("lightgbm"):
        try:
            from lightgbm import LGBMClassifier

            candidates["LightGBM"] = _pipeline(
                X,
                LGBMClassifier(
                    n_estimators=iterations,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    verbosity=-1,
                    n_jobs=1,
                    random_state=seed,
                ),
            )
        except Exception as exc:
            skipped["LightGBM"] = f"Unavailable at import: {exc.__class__.__name__}: {exc}"
    else:
        skipped["LightGBM"] = "Optional dependency `lightgbm` is not installed."

    if importlib.util.find_spec("xgboost"):
        try:
            from xgboost import XGBClassifier

            candidates["XGBoost"] = _pipeline(
                X,
                XGBClassifier(
                    n_estimators=iterations,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    n_jobs=1,
                    random_state=seed,
                ),
            )
        except Exception as exc:
            skipped["XGBoost"] = f"Unavailable at import: {exc.__class__.__name__}: {exc}"
    else:
        skipped["XGBoost"] = "Optional dependency `xgboost` is not installed."

    if importlib.util.find_spec("tabpfn"):
        rows = len(X)
        estimated_features = sum(X[c].nunique(dropna=True) if c in _feature_groups(X)[1] else 1 for c in X.columns)
        if rows <= 10_000 and estimated_features <= 100:
            try:
                from tabpfn import TabPFNClassifier

                candidates["TabPFN"] = _pipeline(X, TabPFNClassifier(), dense=True)
            except Exception as exc:
                skipped["TabPFN"] = f"Unavailable at import: {exc.__class__.__name__}: {exc}"
        else:
            skipped["TabPFN"] = f"Skipped by conservative eligibility gate ({rows:,} rows; about {estimated_features} raw/encoded features; limits are 10,000 rows and 100 features)."
    else:
        skipped["TabPFN"] = "Optional dependency `tabpfn` is not installed."

    return candidates, skipped


def _probabilities(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    return 1 / (1 + np.exp(-np.asarray(scores)))


def _metrics(y, p):
    pred = (p >= 0.5).astype(int)
    unique = np.unique(y)
    return {
        "roc_auc": roc_auc_score(y, p) if len(unique) == 2 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(unique) == 2 else np.nan,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tn_fp_fn_tp": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }


def benchmark_models(
    X: pd.DataFrame,  # noqa: N803
    y,
    mode: str = "quick",
    folds: int | None = None,
    seed: int = 42,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run consistent stratified CV and report discrimination, cost-independent metrics, and time."""
    if len(X) != len(y):
        raise ValueError("X and y row counts do not match.")
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    min_class = int(y.value_counts().min())
    n_splits = min(folds or (3 if mode.lower() == "quick" else 5), min_class)
    if n_splits < 2:
        raise ValueError("At least two observations per class are required for stratified validation.")
    candidates, skipped = model_registry(X, mode, seed)
    if model_names is not None:
        for name in model_names:
            if name not in candidates and name not in skipped:
                skipped[name] = "Requested model name is not in the registry."
        candidates = {name: model for name, model in candidates.items() if name in model_names}
    if not candidates:
        raise ValueError("No requested model family is available. " + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    process = psutil.Process()
    rows, fold_rows = [], []

    for name, pipeline in candidates.items():
        fit_seconds, predict_seconds, memory_deltas = [], [], []
        per_fold = []
        failure = None
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            try:
                model = clone_pipeline(pipeline)
                train_x, valid_x = X.iloc[train_idx], X.iloc[valid_idx]
                train_y, valid_y = y.iloc[train_idx], y.iloc[valid_idx]
                rss_before = process.memory_info().rss
                started = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train_x, train_y)
                fit_time = time.perf_counter() - started
                rss_after = process.memory_info().rss
                started = time.perf_counter()
                probabilities = _probabilities(model, valid_x)
                predict_time = time.perf_counter() - started
                values = _metrics(valid_y, probabilities)
                per_fold.append(values)
                fit_seconds.append(fit_time)
                predict_seconds.append(predict_time)
                memory_deltas.append(max(0, rss_after - rss_before) / (1024**2))
                fold_rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        **{k: v for k, v in values.items() if k != "tn_fp_fn_tp"},
                        "confusion_matrix_tn_fp_fn_tp": values["tn_fp_fn_tp"],
                        "fit_seconds": fit_time,
                        "inference_seconds": predict_time,
                        "rss_delta_mb": memory_deltas[-1],
                        "status": "completed",
                    }
                )
            except Exception as exc:
                failure = f"{exc.__class__.__name__}: {exc}"
                skipped[name] = f"Benchmark failed at fold {fold}: {failure}"
                break
        if failure:
            rows.append({"model": name, "folds": n_splits, "status": "failed", "error": failure})
            continue
        summary = {metric: float(np.nanmean([fold[metric] for fold in per_fold])) for metric in SCORING}
        summary.update(
            {
                f"{metric}_std": float(np.nanstd([fold[metric] for fold in per_fold], ddof=1)) if len(per_fold) > 1 else 0.0
                for metric in SCORING
            }
        )
        rows.append(
            {
                "model": name,
                "folds": n_splits,
                **summary,
                "mean_fit_seconds": float(np.mean(fit_seconds)),
                "mean_inference_seconds": float(np.mean(predict_seconds)),
                "mean_rss_delta_mb": float(np.mean(memory_deltas)),
                "status": "completed",
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True), pd.DataFrame(fold_rows), skipped


def clone_pipeline(pipeline):
    from sklearn.base import clone

    return clone(pipeline)


def fit_best_model(
    X: pd.DataFrame,
    y,
    model_name: str,
    mode: str = "quick",
    seed: int = 42,
    parameters: dict | None = None,
):  # noqa: N803
    model = resolve_candidate(X, model_name, mode, seed, parameters)
    model.fit(X, y)
    return model


def tune_with_optuna(X: pd.DataFrame, y, model_name: str, mode: str = "quick", seed: int = 42):  # noqa: N803
    """Bounded Optuna search on training data only; unsupported optional families are rejected."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed. Install it from requirements.txt.") from exc
    candidates, _ = model_registry(X, mode, seed)
    if model_name not in {"CatBoost", "LightGBM", "XGBoost"} or model_name not in candidates:
        raise ValueError("Optuna tuning is available for installed CatBoost, LightGBM, or XGBoost candidates.")
    trials = 8 if mode.lower() == "quick" else 40
    cv = StratifiedKFold(n_splits=3 if mode.lower() == "quick" else 5, shuffle=True, random_state=seed)
    base = candidates[model_name]

    def objective(trial):
        if model_name == "CatBoost":
            params = {
                "depth": trial.suggest_int("depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "iterations": trial.suggest_int("iterations", 100, 500 if mode.lower() == "full" else 200),
            }
        elif model_name == "LightGBM":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 700 if mode.lower() == "full" else 250),
            }
        else:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 700 if mode.lower() == "full" else 250),
            }
        scores = []
        target = np.asarray(y, dtype=int)
        for train_idx, valid_idx in cv.split(X, target):
            candidate = clone_pipeline(base)
            estimator = candidate.named_steps["model"].set_params(**params)
            train_x, valid_x = X.iloc[train_idx], X.iloc[valid_idx]
            train_y, valid_y = target[train_idx], target[valid_idx]
            feature_step = candidate.named_steps["features"]
            if model_name == "CatBoost":
                train_features = feature_step.fit_transform(train_x)
                valid_features = feature_step.transform(valid_x)
                estimator.fit(
                    train_features,
                    train_y,
                    eval_set=(valid_features, valid_y),
                    early_stopping_rounds=30,
                    verbose=False,
                )
            else:
                preprocessor = candidate.named_steps["preprocess"]
                train_engineered = feature_step.fit_transform(train_x)
                valid_engineered = feature_step.transform(valid_x)
                train_features = preprocessor.fit_transform(train_engineered, train_y)
                valid_features = preprocessor.transform(valid_engineered)
                if model_name == "LightGBM":
                    from lightgbm import early_stopping, log_evaluation

                    estimator.fit(
                        train_features,
                        train_y,
                        eval_set=[(valid_features, valid_y)],
                        callbacks=[early_stopping(30, verbose=False), log_evaluation(period=0)],
                    )
                else:
                    estimator.set_params(early_stopping_rounds=30)
                    estimator.fit(train_features, train_y, eval_set=[(valid_features, valid_y)], verbose=False)
            scores.append(roc_auc_score(valid_y, estimator.predict_proba(valid_features)[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    started = time.perf_counter()
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {
        "model": model_name,
        "best_score_roc_auc": study.best_value,
        "best_parameters": study.best_params,
        "pipeline_parameters": {f"model__{key}": value for key, value in study.best_params.items()},
        "duration_seconds": time.perf_counter() - started,
        "trials": [{"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)} for t in study.trials],
    }


def evaluate_probability_blend(X: pd.DataFrame, y, model_names: list[str], mode: str = "quick", seed: int = 42):  # noqa: N803
    """OOF probability blending; each meta-probability is built from held-out folds."""
    candidates, _ = model_registry(X, mode, seed)
    missing = [name for name in model_names if name not in candidates]
    if missing or len(model_names) < 2:
        raise ValueError(f"Select at least two available model families; unavailable: {missing}")
    y = np.asarray(y, dtype=int)
    splitter = StratifiedKFold(n_splits=3 if mode.lower() == "quick" else 5, shuffle=True, random_state=seed)
    oof = {name: np.zeros(len(y), dtype=float) for name in model_names}
    timing = {name: {"fit_seconds_total": 0.0, "inference_seconds_total": 0.0, "rss_delta_mb_total": 0.0} for name in model_names}
    process = psutil.Process()
    for train_idx, valid_idx in splitter.split(X, y):
        for name in model_names:
            model = clone_pipeline(candidates[name])
            rss_before = process.memory_info().rss
            start = time.perf_counter()
            model.fit(X.iloc[train_idx], y[train_idx])
            timing[name]["fit_seconds_total"] += time.perf_counter() - start
            rss_after = process.memory_info().rss
            timing[name]["rss_delta_mb_total"] += max(0, rss_after - rss_before) / (1024**2)
            start = time.perf_counter()
            oof[name][valid_idx] = _probabilities(model, X.iloc[valid_idx])
            timing[name]["inference_seconds_total"] += time.perf_counter() - start
    rows = []
    for left_idx, left in enumerate(model_names):
        rows.append({"blend": left, "roc_auc": roc_auc_score(y, oof[left]), "pr_auc": average_precision_score(y, oof[left]), **timing[left]})
        for right in model_names[left_idx + 1 :]:
            blended = (oof[left] + oof[right]) / 2
            rows.append(
                {
                    "blend": f"OOF mean: {left} + {right}",
                    "roc_auc": roc_auc_score(y, blended),
                    "pr_auc": average_precision_score(y, blended),
                    "fit_seconds_total": timing[left]["fit_seconds_total"] + timing[right]["fit_seconds_total"],
                    "inference_seconds_total": timing[left]["inference_seconds_total"] + timing[right]["inference_seconds_total"],
                    "rss_delta_mb_total": timing[left]["rss_delta_mb_total"] + timing[right]["rss_delta_mb_total"],
                }
            )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True), oof
