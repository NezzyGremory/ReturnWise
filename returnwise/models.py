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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_selection import SelectFromModel

from returnwise.features import ReturnWiseFeatureEngineer
from returnwise.paths import EXPERIMENT_DIR

SCORING = ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1")


class CatBoostFrameCleaner(BaseEstimator, TransformerMixin):
    """Keep feature names and normalize missing values for native CatBoost categories."""

    def __init__(self, categorical_columns, feature_engineering=False):
        self.categorical_columns = list(categorical_columns)
        self.feature_engineering = feature_engineering

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        frame = ReturnWiseFeatureEngineer().transform(X) if self.feature_engineering else pd.DataFrame(X).copy()
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


def _pipeline(
    X: pd.DataFrame,
    estimator,
    *,
    scale: bool = False,
    dense: bool = False,
    feature_engineering: bool = False,
    feature_selection: str | None = None,
):  # noqa: N803
    preprocessing_frame = ReturnWiseFeatureEngineer().fit_transform(X) if feature_engineering else pd.DataFrame(X)
    steps = []
    if feature_engineering:
        steps.append(("features", ReturnWiseFeatureEngineer()))
    steps.extend(
        [
            ("preprocess", build_preprocessor(preprocessing_frame, dense=dense, scale=scale)),
        ]
    )
    if feature_selection == "l1":
        selector = LogisticRegression(
            C=0.1,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        steps.append(("feature_selection", SelectFromModel(selector, threshold="median")))
    elif feature_selection is not None:
        raise ValueError("Supported fold-local feature_selection is None or 'l1'.")
    steps.append(("model", estimator))
    return Pipeline(steps)


def _hist_pipeline(X: pd.DataFrame, seed: int, max_iter: int, feature_engineering: bool = False):  # noqa: N803
    preprocessing_frame = ReturnWiseFeatureEngineer().fit_transform(X) if feature_engineering else pd.DataFrame(X)
    numeric, categorical = _feature_groups(preprocessing_frame)
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
    steps = []
    if feature_engineering:
        steps.append(("features", ReturnWiseFeatureEngineer()))
    steps.extend(
        [
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, random_state=seed)),
        ]
    )
    return Pipeline(steps)


def model_registry(
    X: pd.DataFrame,
    mode: str = "quick",
    seed: int = 42,
    feature_engineering: bool = False,
    feature_selection: str | None = None,
) -> tuple[dict[str, object], dict[str, str]]:  # noqa: N803
    """Create available candidate pipelines; optional families fail independently."""
    quick = mode.lower() == "quick"
    trees = 32 if quick else 300
    iterations = 48 if quick else 250
    candidates: dict[str, object] = {
        "Dummy (prior)": _pipeline(
            X, DummyClassifier(strategy="prior"), feature_engineering=feature_engineering
        ),
        "Logistic regression": _pipeline(
            X,
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            scale=True,
            feature_engineering=feature_engineering,
            feature_selection=feature_selection,
        ),
        "Random forest": _pipeline(
            X,
            RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
            feature_engineering=feature_engineering,
            feature_selection=feature_selection,
        ),
        "Extra trees": _pipeline(
            X,
            ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed),
            feature_engineering=feature_engineering,
            feature_selection=feature_selection,
        ),
        "HistGradientBoosting": _hist_pipeline(X, seed, iterations, feature_engineering=feature_engineering),
    }
    skipped: dict[str, str] = {}

    if importlib.util.find_spec("catboost"):
        try:
            from catboost import CatBoostClassifier

            engineered = ReturnWiseFeatureEngineer().fit_transform(X)
            cats = [c for c in engineered.columns if not pd.api.types.is_numeric_dtype(engineered[c])]
            candidates["CatBoost"] = Pipeline(
                [
                    ("features", CatBoostFrameCleaner(cats, feature_engineering=feature_engineering)),
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
                feature_engineering=feature_engineering,
                feature_selection=feature_selection,
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
                feature_engineering=feature_engineering,
                feature_selection=feature_selection,
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

                candidates["TabPFN"] = _pipeline(
                    X, TabPFNClassifier(), dense=True, feature_engineering=feature_engineering
                )
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
    feature_engineering: bool = False,
    feature_selection: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run consistent stratified CV and report metrics, optionally disabling deterministic interactions."""
    if feature_selection is not None and (
        feature_selection != "l1" or model_names is None or len(model_names) != 1 or model_names[0] != "Logistic regression"
    ):
        raise ValueError("The current fold-local L1 selection comparison is defined for Logistic regression only.")
    if len(X) != len(y):
        raise ValueError("X and y row counts do not match.")
    y = pd.Series(y).astype(int).reset_index(drop=True)
    X = pd.DataFrame(X).reset_index(drop=True)
    min_class = int(y.value_counts().min())
    n_splits = min(folds or (3 if mode.lower() == "quick" else 5), min_class)
    if n_splits < 2:
        raise ValueError("At least two observations per class are required for stratified validation.")
    candidates, skipped = model_registry(
        X,
        mode,
        seed,
        feature_engineering=feature_engineering,
        feature_selection=feature_selection,
    )
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
                selected_features = None
                selector = model.named_steps.get("feature_selection") if hasattr(model, "named_steps") else None
                if selector is not None:
                    support = selector.get_support()
                    try:
                        feature_names = model.named_steps["preprocess"].get_feature_names_out()
                        selected_features = np.asarray(feature_names)[support].tolist()
                    except (AttributeError, ValueError):
                        selected_features = [int(index) for index in np.flatnonzero(support)]
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
                        "selected_features": selected_features,
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


def tune_with_optuna(
    X: pd.DataFrame,
    y,
    model_name: str,
    mode: str = "quick",
    seed: int = 42,
    n_trials: int | None = None,
    study_path: str | None = None,
    study_name: str | None = None,
    timeout: int | None = None,
):  # noqa: N803
    """Run resumable TPE tuning on training data with inner-fold early stopping."""
    try:
        import optuna
        from sqlalchemy.engine import URL
    except ImportError as exc:
        raise RuntimeError("Optuna and its SQLAlchemy storage dependency must be installed.") from exc

    mode = mode.lower()
    candidates, skipped = model_registry(X, mode, seed)
    if model_name not in {"CatBoost", "LightGBM", "XGBoost"} or model_name not in candidates:
        raise ValueError("Optuna tuning is available for installed CatBoost, LightGBM, or XGBoost candidates.")
    trial_budget = n_trials if n_trials is not None else (8 if mode == "quick" else 40)
    if trial_budget < 1:
        raise ValueError("n_trials must be a positive integer.")
    cv = StratifiedKFold(n_splits=3 if mode == "quick" else 5, shuffle=True, random_state=seed)
    base = candidates[model_name]
    target = np.asarray(y, dtype=int)
    if len(X) != len(target):
        raise ValueError("X and y row counts do not match.")
    iterations_high = 250 if mode == "quick" else 700
    default_name = f"returnwise-{model_name.lower().replace(' ', '-')}-{mode}-seed-{seed}"
    study_name = study_name or default_name
    sqlite_path = Path(study_path) if study_path else EXPERIMENT_DIR / "optuna_studies.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_path = sqlite_path.resolve()
    storage_url = URL.create("sqlite", database=str(sqlite_path))

    if model_name == "CatBoost":
        from catboost import CatBoostError

        caught_errors = (ValueError, TypeError, RuntimeError, CatBoostError)
    elif model_name == "LightGBM":
        from lightgbm.basic import LightGBMError

        caught_errors = (ValueError, TypeError, RuntimeError, LightGBMError)
    else:
        from xgboost.core import XGBoostError

        caught_errors = (ValueError, TypeError, RuntimeError, XGBoostError)

    def objective(trial):
        if model_name == "CatBoost":
            params = {
                "depth": trial.suggest_int("depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "iterations": trial.suggest_int("iterations", 100, iterations_high),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
            }
        elif model_name == "LightGBM":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, iterations_high),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 100),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        else:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, iterations_high),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

        scores = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, target), start=1):
            candidate = clone_pipeline(base)
            estimator = candidate.named_steps["model"].set_params(**params)
            train_idx, early_idx = train_test_split(
                train_idx,
                test_size=0.15,
                random_state=seed + 1000 * trial.number + fold,
                stratify=target[train_idx],
            )
            fit_x, early_x, valid_x = X.iloc[train_idx], X.iloc[early_idx], X.iloc[valid_idx]
            fit_y, early_y, valid_y = target[train_idx], target[early_idx], target[valid_idx]
            feature_step = candidate.named_steps.get("features")
            if feature_step is not None:
                fit_frame = feature_step.fit_transform(fit_x, fit_y)
                early_frame = feature_step.transform(early_x)
                valid_frame = feature_step.transform(valid_x)
            else:
                fit_frame, early_frame, valid_frame = fit_x, early_x, valid_x

            preprocessor = candidate.named_steps.get("preprocess")
            if preprocessor is not None:
                fit_features = preprocessor.fit_transform(fit_frame, fit_y)
                early_features = preprocessor.transform(early_frame)
                valid_features = preprocessor.transform(valid_frame)
            else:
                fit_features, early_features, valid_features = fit_frame, early_frame, valid_frame

            if model_name == "CatBoost":
                estimator.fit(
                    fit_features,
                    fit_y,
                    eval_set=(early_features, early_y),
                    early_stopping_rounds=30,
                    verbose=False,
                )
            elif model_name == "LightGBM":
                from lightgbm import early_stopping, log_evaluation

                estimator.fit(
                    fit_features,
                    fit_y,
                    eval_set=[(early_features, early_y)],
                    callbacks=[early_stopping(30, verbose=False), log_evaluation(period=0)],
                )
            else:
                estimator.set_params(early_stopping_rounds=30)
                estimator.fit(fit_features, fit_y, eval_set=[(early_features, early_y)], verbose=False)

            scores.append(roc_auc_score(valid_y, estimator.predict_proba(valid_features)[:, 1]))
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url.render_as_string(hide_password=False),
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    started = time.perf_counter()
    study.optimize(
        objective,
        n_trials=trial_budget,
        timeout=timeout,
        show_progress_bar=False,
        gc_after_trial=True,
        catch=caught_errors,
    )
    history = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        }
        for trial in study.trials
    ]
    return {
        "model": model_name,
        "mode": mode,
        "folds": 3 if mode == "quick" else 5,
        "seed": seed,
        "best_score_roc_auc": study.best_value,
        "best_parameters": study.best_params,
        "pipeline_parameters": {f"model__{key}": value for key, value in study.best_params.items()},
        "duration_seconds": time.perf_counter() - started,
        "study_name": study.study_name,
        "study_path": str(sqlite_path),
        "trial_count_total": len(study.trials),
        "completed_trials": sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials),
        "pruned_trials": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
        "failed_trials": sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials),
        "trials": history,
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
