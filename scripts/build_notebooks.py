"""Generate clean, output-free Jupyter research notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
KERNEL = {"display_name": "Python (ReturnWise)", "language": "python", "name": "returnwise"}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def py(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def create_notebook(filename, title, purpose, method, limitations, references, code):
    notebook = {
        "cells": [
            md(
                f"# {title}\n\n"
                f"## Purpose\n\n{purpose}\n\n"
                f"## Method\n\n{method}\n\n"
                "## Interpretation\n\n"
                "Use the displayed tables, figures, and saved metrics produced by this run. Do not fill this section with assumed outcomes. "
                "For every result, distinguish observed evidence from interpretation.\n\n"
                f"## Limitations\n\n{limitations}\n\n"
                f"## References\n\n{references}\n"
            ),
            py(
                "from pathlib import Path\n"
                "import sys\n"
                "ROOT = Path.cwd().resolve()\n"
                "while not (ROOT / 'returnwise').is_dir() and ROOT != ROOT.parent:\n"
                "    ROOT = ROOT.parent\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n"
                "from IPython.display import display\n"
                "from returnwise.research_state import load_state, save_state\n"
                "state = load_state()\n"
                "print('Project:', ROOT)\n"
                "print('Notebook:', '" + title + "')\n"
            ),
            py(code),
        ],
        "metadata": {
            "kernelspec": KERNEL,
            "language_info": {"name": "python", "version": "3.11+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    (NOTEBOOK_DIR / filename).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


SPECS = [
    (
        "00_environment_setup.ipynb",
        "00 · Environment setup",
        "Confirm the local Python environment and the dependencies needed by the ReturnWise research sequence.",
        "Inspect versions and optional package availability without installing packages or accessing credentials from inside the notebook. Set up the named kernel using the documented terminal command.",
        "Dependency presence does not prove that a later model can train on this machine. Resource and compatibility checks happen again at experiment time.",
        "See the environment setup steps in README.md.",
        """import importlib.metadata as metadata
import importlib.util
import matplotlib.pyplot as plt
import pandas as pd

packages = ['streamlit', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'kaggle', 'ipykernel',
            'catboost', 'lightgbm', 'xgboost', 'optuna', 'shap', 'tabpfn']
rows = []
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = 'not installed'
    module_name = {'scikit-learn': 'sklearn'}.get(package, package.replace('-', '_'))
    rows.append({'package': package, 'version': version, 'importable': importlib.util.find_spec(module_name) is not None})
environment = pd.DataFrame(rows)
display(environment)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(environment['package'], environment['importable'].astype(int), color='#126E63')
ax.set_xlim(0, 1.1)
ax.set_xlabel('Importable (1 = yes)')
ax.set_title('ReturnWise environment dependency check')
plt.tight_layout()
plt.show()
print('Kernel install command (run in the activated project environment if needed):')
print('python -m ipykernel install --user --name returnwise --display-name \"Python (ReturnWise)\"')
EXPERIMENT_MODE = 'quick'  # Change to 'full' for five-fold runs and the larger Optuna budget.
state['experiment_mode'] = EXPERIMENT_MODE
save_state(state)
print('Experiment mode saved for the notebook sequence:', EXPERIMENT_MODE)
"""
    ),
    (
        "01_kaggle_dataset_acquisition.ipynb",
        "01 · Kaggle dataset acquisition",
        "Acquire the specified competition files through the official Kaggle CLI, or deliberately use a CSV already imported in Manual mode.",
        "Check CLI access and competition file names before download. Download, safely extract, validate train/test/submission roles, fingerprint the files, and select the training file. No other dataset is substituted.",
        "This step requires a Kaggle account with competition access. If access fails, authenticate locally with kaggle auth login or select Manual mode; do not share tokens.",
        "Official CLI: https://github.com/Kaggle/kaggle-cli · Competition: https://www.kaggle.com/competitions/retail-return-risk-modeling",
        """from returnwise.data.kaggle import sync_competition
from returnwise.storage import load_active_dataset, use_kaggle_training_data
import matplotlib.pyplot as plt
import pandas as pd

USE_MANUAL_DATASET = False  # Change to True only after importing a CSV in the application.
if USE_MANUAL_DATASET:
    frame, manifest = load_active_dataset()
    if frame is None:
        raise RuntimeError('No active manual CSV. Import a CSV in the app, then rerun this notebook.')
    print('Explicit manual dataset selected:', manifest)
else:
    acquisition = sync_competition(force=False)
    print(acquisition.message)
    if acquisition.details:
        print(acquisition.details)
    if not acquisition.ok:
        raise RuntimeError('Kaggle acquisition stopped safely. Authenticate and verify access, or explicitly select a manual dataset.')
    manifest = use_kaggle_training_data()
    frame = None  # The active training manifest is already set; avoid loading train.csv twice.
    print('Acquisition metadata:')
    display(pd.Series(acquisition.metadata or {}))

if acquisition.ok if not USE_MANUAL_DATASET else True:
    sizes = acquisition.metadata.get('file_bytes', {}) if not USE_MANUAL_DATASET else {'imported CSV': frame.memory_usage(deep=True).sum()}
    size_frame = pd.DataFrame({'file': list(sizes), 'bytes': list(sizes.values())})
    display(size_frame)
    ax = size_frame.set_index('file')['bytes'].plot.bar(figsize=(7, 3), color='#126E63', title='Downloaded or selected data size')
    ax.set_ylabel('Bytes')
    plt.tight_layout()
    plt.show()
from returnwise.experiments import log_experiment
log_experiment('Dataset acquisition', None, {'success': bool(not USE_MANUAL_DATASET and acquisition.ok) if not USE_MANUAL_DATASET else True},
               {'source': 'manual' if USE_MANUAL_DATASET else 'kaggle', 'metadata': acquisition.metadata if not USE_MANUAL_DATASET else manifest})
from returnwise.storage import active_dataset_fingerprint
fingerprint = active_dataset_fingerprint()
experiment_mode = state.get('experiment_mode', 'quick')
if state.get('dataset_fingerprint') != fingerprint:
    state = {'experiment_mode': experiment_mode}
state['dataset_fingerprint'] = fingerprint
state['dataset_manifest'] = manifest
save_state(state)
"""
    ),
    (
        "02_dataset_understanding.ipynb",
        "02 · Dataset understanding",
        "Inspect the actual active CSV schema, values, dimensions, target, identifiers, and available competition test/submission schemas.",
        "Generate a data dictionary from observed CSV values and attach descriptions only for documented competition fields. Compare train/test feature schemas without evaluating unlabeled competition test rows.",
        "The competition test split has no labels. Semantic descriptions for undocumented columns remain explicitly unknown until confirmed from the source documentation.",
        "Competition data page: https://www.kaggle.com/competitions/retail-return-risk-modeling/data",
        """from returnwise.data.profile import data_dictionary
from returnwise.paths import KAGGLE_DIR
from returnwise.storage import load_active_dataset
import pandas as pd

frame, manifest = load_active_dataset()
if frame is None:
    raise RuntimeError('Select an imported or downloaded dataset in notebook 01 first.')
print('Source:', manifest)
print('Dimensions:', frame.shape)
print('Target present:', 'returned' in frame.columns)
print('Data types and unique values:')
display(pd.DataFrame({'dtype': frame.dtypes.astype(str), 'unique': frame.nunique(dropna=True), 'missing': frame.isna().sum()}))
print('Automatic data dictionary:')
dictionary = data_dictionary(frame)
display(dictionary)

test_path = KAGGLE_DIR / 'test.csv'
sample_path = KAGGLE_DIR / 'sample_submission.csv'
if manifest.get('source') == 'kaggle' and test_path.exists():
    competition_test = pd.read_csv(test_path)
    train_features = [c for c in frame.columns if c != 'returned']
    print('Competition test shape (labels are unavailable):', competition_test.shape)
    print('Train/test feature schema identical:', train_features == list(competition_test.columns))
    print('Columns only in active train:', sorted(set(train_features) - set(competition_test.columns)))
    print('Columns only in competition test:', sorted(set(competition_test.columns) - set(train_features)))
    if sample_path.exists():
        sample_submission = pd.read_csv(sample_path)
        print('Sample submission schema:', list(sample_submission.columns))
        display(sample_submission.head())
else:
    print('No competition test split is present; this is a manually imported dataset.')
"""
    ),
    (
        "03_data_quality_assessment.ipynb",
        "03 · Data quality assessment",
        "Measure missingness, duplicate rows and identifiers, numeric outlier flags, schema roles, and class balance from the actual active CSV.",
        "Create a quality report. Outliers use the 1.5×IQR screening rule only; data is not deleted automatically. Record target imbalance only when the target exists.",
        "IQR flags are review indicators, not proof that a value is invalid. Leakage and feature reliability require documentation beyond automated name and type checks.",
        "Quality choices and leakage limitations are documented in docs/implementation_plan.md.",
        """from returnwise.data.profile import quality_report
from returnwise.storage import load_active_dataset
import matplotlib.pyplot as plt

frame, manifest = load_active_dataset()
if frame is None:
    raise RuntimeError('No active dataset. Complete notebook 01 first.')
quality = quality_report(frame)
print({key: value for key, value in quality.items() if key not in {'missing', 'schema', 'target_counts'}})
display(quality['missing'])
display(quality['schema'])
if quality['target_counts'] is not None:
    display(quality['target_counts'])
missing = quality['missing'].head(20)
if not missing.empty:
    ax = missing.sort_values('missing_percent').plot.barh(x='column', y='missing_percent', figsize=(8, max(3, len(missing) * 0.25)), legend=False, color='#B56B33', title='Missingness by feature')
    ax.set_xlabel('Missing values (%)')
    plt.tight_layout()
    plt.show()
else:
    print('No missing values were observed in the active file.')
"""
    ),
    (
        "04_exploratory_data_analysis.ipynb",
        "04 · Exploratory data analysis",
        "Describe target balance and observed numeric and categorical feature distributions before modeling.",
        "Use the actual active CSV, show target proportions, numeric summaries, and available grouped distributions. Treat observed associations as descriptive.",
        "A single file without transaction dates cannot show temporal drift. Category groups with few observations should be interpreted cautiously.",
        "See the competition field notes in returnwise.data.profile.KNOWN_DESCRIPTIONS.",
        """from returnwise.data.profile import eda_summaries
from returnwise.storage import load_active_dataset
import matplotlib.pyplot as plt
import pandas as pd

frame, manifest = load_active_dataset()
if frame is None:
    raise RuntimeError('No active dataset. Complete notebook 01 first.')
eda = eda_summaries(frame)
display(eda['numeric_summary'])
display(eda['categorical_summary'])
if eda['target_distribution'] is not None:
    display(eda['target_distribution'])
    ax = frame['returned'].value_counts(dropna=False).sort_index().plot.bar(figsize=(5, 3), color=['#72817D', '#126E63'], title='Observed returned target distribution')
    ax.set_xlabel('returned')
    ax.set_ylabel('Transactions')
    plt.tight_layout()
    plt.show()
    numeric = frame.select_dtypes(include='number').columns.drop('returned', errors='ignore')
    for feature in list(numeric[:4]):
        ax = frame.boxplot(column=feature, by='returned', figsize=(5, 3))
        ax.set_title(f'{feature} by returned status')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()
    categorical = [c for c in frame.columns if c != 'returned' and not pd.api.types.is_numeric_dtype(frame[c])]
    for feature in categorical[:4]:
        grouped = frame.groupby(feature, dropna=False)['returned'].mean().sort_values(ascending=False).head(12)
        ax = grouped.plot.bar(figsize=(7, 3), color='#126E63', title=f'Observed return proportion by {feature} (top groups)')
        ax.set_ylabel('Return proportion')
        plt.tight_layout()
        plt.show()
else:
    print('No return target is present; supervised EDA is not available.')
"""
    ),
    (
        "05_statistical_analysis.ipynb",
        "05 · Statistical analysis",
        "Test feature-target associations using methods suited to numeric and categorical fields, and report effect sizes and adjusted p-values.",
        "Compare numeric distributions with two-sided Mann–Whitney U plus rank-biserial effect size and a bootstrap interval for median difference. Compare categorical features with Pearson chi-square and Cramér’s V. Adjust across tests using Benjamini–Hochberg.",
        "Tests address association, not causality. Small or highly sparse category tables may violate chi-square approximations and should be merged or analyzed with a justified alternative.",
        "SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html",
        """from returnwise.analysis.statistics import association_tests
from returnwise.features import select_features
from returnwise.storage import load_active_dataset
import matplotlib.pyplot as plt

frame, manifest = load_active_dataset()
if frame is None or 'returned' not in frame:
    raise RuntimeError('A labeled active CSV with returned is required for supervised statistical analysis.')
candidate_features = select_features(frame, scenario='pre_shipment')
analysis_frame = frame[candidate_features + ['returned']]
tests = association_tests(analysis_frame)
display(tests)
if not tests.empty:
    plot_data = tests.sort_values('effect_size', key=lambda s: s.abs()).tail(15)
    ax = plot_data.set_index('feature')['effect_size'].plot.barh(figsize=(8, max(3, len(plot_data) * 0.3)), color='#126E63', title='Association effect sizes (not causal effects)')
    ax.set_xlabel('Rank-biserial correlation or Cramér’s V')
    plt.tight_layout()
    plt.show()
    print('Features passing BH-adjusted p < 0.05:', tests.loc[tests['significant_fdr_0_05'], 'feature'].tolist())
    numeric = analysis_frame.select_dtypes(include='number').drop(columns=['returned'], errors='ignore')
    if numeric.shape[1] >= 2:
        correlations = numeric.corr(method='spearman')
        display(correlations)
        fig, ax = plt.subplots(figsize=(max(5, len(correlations) * 0.55), max(4, len(correlations) * 0.45)))
        image = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(correlations)), labels=correlations.columns, rotation=60, ha='right')
        ax.set_yticks(range(len(correlations)), labels=correlations.index)
        ax.set_title('Spearman correlation among numeric predictors')
        fig.colorbar(image, ax=ax, label='Spearman rho')
        plt.tight_layout()
        plt.show()
"""
    ),
    (
        "06_problem_identification.ipynb",
        "06 · Problem identification",
        "Turn computed data-quality and association outputs into evidence-based research questions, without assuming that the dataset must show an effect.",
        "Summarize observed return prevalence, imbalance, significant adjusted associations, data limitations, and the prediction-time policy. Print findings directly from the active data and test table.",
        "The dataset is observational and may be synthetic or competition-generated. Its associations do not establish customer or product causes.",
        "See docs/references.md for the method map and novelty boundary.",
        """from returnwise.analysis.statistics import association_tests
from returnwise.data.profile import quality_report
from returnwise.features import select_features, temporal_feature_audit
from returnwise.storage import load_active_dataset
import matplotlib.pyplot as plt

frame, manifest = load_active_dataset()
if frame is None:
    raise RuntimeError('No active dataset. Complete notebook 01 first.')
quality = quality_report(frame)
print(f'Observed rows: {len(frame):,}; columns: {len(frame.columns):,}; exact duplicate rows: {quality[\"duplicate_rows\"]:,}.')
if 'returned' in frame:
    print('Observed target proportions:')
    display(frame['returned'].value_counts(normalize=True, dropna=False).rename('proportion').to_frame())
    candidate_features = select_features(frame, scenario='pre_shipment')
    tests = association_tests(frame[candidate_features + ['returned']])
    if not tests.empty:
        supported = tests[tests['significant_fdr_0_05']]
        print(f'Features with BH-adjusted p < 0.05: {len(supported)} of {len(tests)} tested.')
        display(supported[['feature', 'test', 'effect_size', 'effect_label', 'p_adjusted_bh']])
        chart = tests.set_index('feature')['p_adjusted_bh'].sort_values().head(15)
        ax = chart.plot.barh(figsize=(7, 4), color='#B56B33', title='Smallest BH-adjusted p-values')
        ax.set_xlabel('Adjusted p-value')
        plt.tight_layout()
        plt.show()
    else:
        print('No eligible feature-target association tests were available.')
else:
    print('No target is available, so return-risk research questions cannot be evaluated.')
print('Prediction timing: pre-shipment. No temporal validation is claimed without transaction event-order timestamps.')
print('Candidate research questions: (1) Do available pre-shipment features distinguish return risk? (2) Are model probabilities reliable? (3) Does review prioritization improve expected outcomes under stated costs and capacity?')
print('Operational impact is a hypothesis for scenario analysis, not a measured business effect.')
display(temporal_feature_audit(list(frame.columns)))
"""
    ),
    (
        "07_data_preprocessing.ipynb",
        "07 · Data preprocessing",
        "Choose features available in the declared prediction scenario, retain an untouched internal final holdout, and establish reusable train/holdout variables.",
        "Use the conservative pre-shipment temporal audit, exclude identifiers and unverified history aggregates, and create a stratified 80/20 internal split. Competition test rows are never part of this split.",
        "Without event-order timestamps, the split is random stratification rather than temporal validation. Feature availability assumptions must be checked against real operational data.",
        "Scikit-learn pipeline guide: https://scikit-learn.org/stable/modules/compose.html",
        """from returnwise.features import select_features, temporal_feature_audit
from returnwise.storage import load_active_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

frame, manifest = load_active_dataset()
if frame is None or 'returned' not in frame:
    raise RuntimeError('A labeled dataset is required for preprocessing and supervised modeling.')
audit = temporal_feature_audit(list(frame.columns))
display(audit)
feature_columns = select_features(frame, scenario='pre_shipment')
if not feature_columns:
    raise ValueError('The temporal audit left no eligible features.')
X = frame[feature_columns].copy()
y = frame['returned'].astype(int).copy()
train_rows, final_rows = train_test_split(np.arange(len(y)), test_size=0.20, random_state=42, stratify=y)
X_train, X_final = X.iloc[train_rows].reset_index(drop=True), X.iloc[final_rows].reset_index(drop=True)
y_train, y_final = y.iloc[train_rows].reset_index(drop=True), y.iloc[final_rows].reset_index(drop=True)
print('Eligible feature columns:', feature_columns)
print('Training rows:', len(X_train), '| untouched final holdout rows:', len(X_final))
print('No temporal validation was performed.')
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(['training', 'final holdout'], [len(train_rows), len(final_rows)], color=['#126E63', '#A6B4AF'])
ax.set_ylabel('Rows')
ax.set_title('Stratified internal split (competition test excluded)')
plt.tight_layout()
plt.show()
state.update({'feature_columns': feature_columns, 'X_train': X_train, 'y_train': y_train,
              'X_final': X_final, 'y_final': y_final, 'dataset_manifest': manifest})
save_state(state)
"""
    ),
    (
        "08_feature_engineering.ipynb",
        "08 · Feature engineering",
        "Apply small, deterministic feature interactions that can be recomputed identically during model inference.",
        "Add only price-discount, price-rating, and session views-per-minute interactions when their source columns exist. Inspect the transformed schema and missing values; do not construct future-derived customer aggregates.",
        "The feature interactions are candidate representations, not known causal mechanisms. Compare them through cross-validated ablations before retaining them.",
        "Feature engineering is implemented in returnwise.features.ReturnWiseFeatureEngineer.",
        """from returnwise.features import ReturnWiseFeatureEngineer
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt

state = load_state()
if 'X_train' not in state:
    raise RuntimeError('Run notebook 07 first to create the training split.')
X_train = state['X_train']
engineered = ReturnWiseFeatureEngineer().fit_transform(X_train)
new_features = [c for c in engineered.columns if c not in X_train.columns]
print('Input shape:', X_train.shape, '| engineered shape:', engineered.shape)
print('Interactions added when source fields were available:', new_features)
display(engineered[new_features].describe().T if new_features else engineered.describe(include='all').T.head(20))
for feature in new_features[:4]:
    ax = engineered[feature].dropna().plot.hist(bins=30, figsize=(6, 3), color='#126E63', title=f'Observed engineered feature: {feature}')
    ax.set_xlabel(feature)
    plt.tight_layout()
    plt.show()
state['engineered_feature_columns'] = new_features
save_state(state)
"""
    ),
    (
        "09_baseline_machine_learning.ipynb",
        "09 · Baseline machine learning",
        "Establish dummy and logistic-regression baselines before comparing more flexible models.",
        "Use the same stratified folds, preprocessing rules, primary ROC-AUC, minority-sensitive PR-AUC, threshold metrics, runtime, and memory delta for both baseline families.",
        "Fold metrics are internal cross-validation estimates and may vary with the split. Accuracy is not used as the sole selection criterion.",
        "Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html",
        """from returnwise.models import benchmark_models
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt

state = load_state()
if 'X_train' not in state:
    raise RuntimeError('Run notebook 07 first.')
baseline_results, baseline_folds, baseline_skipped = benchmark_models(
    state['X_train'], state['y_train'], mode=state.get('experiment_mode', 'quick'), model_names=['Dummy (prior)', 'Logistic regression']
)
display(baseline_results)
display(baseline_folds)
print('Unavailable or failed:', baseline_skipped)
ax = baseline_results.plot.bar(x='model', y=['roc_auc', 'pr_auc'], figsize=(7, 4), title='Baseline cross-validation metrics')
ax.set_ylabel('Score')
plt.tight_layout()
plt.show()
state.update({'baseline_results': baseline_results, 'baseline_folds': baseline_folds,
              'baseline_skipped': baseline_skipped})
from returnwise.experiments import log_experiment
log_experiment('Baseline machine learning benchmark', None,
               {'results': baseline_results.to_dict(orient='records')}, {'mode': state.get('experiment_mode', 'quick')})
save_state(state)
"""
    ),
    (
        "10_advanced_model_benchmarking.ipynb",
        "10 · Advanced model benchmarking",
        "Compare traditional ensembles, histogram boosting, and installed modern gradient-boosting candidates under the shared validation folds.",
        "Benchmark Random Forest, Extra Trees, HistGradientBoosting, CatBoost, LightGBM, XGBoost, and eligible TabPFN. Quick mode uses three folds, bounded estimator counts, one thread per candidate, and per-candidate checkpoints; unavailable or failed packages are reported.",
        "Runtime and process RSS changes depend on the machine. Optional model families can be omitted for installation, incompatibility, or resource reasons; no model is assumed to win.",
        "Model papers and eligibility notes: docs/references.md",
        """from returnwise.models import benchmark_models, model_registry
from returnwise.research_state import load_state, save_state
import gc
import matplotlib.pyplot as plt
import pandas as pd
import psutil

state = load_state()
if 'X_train' not in state:
    raise RuntimeError('Run notebook 07 first.')
mode = state.get('experiment_mode', 'quick')
if mode != 'quick':
    raise RuntimeError('This resource-managed run requires the verified Quick mode.')
advanced_names = ['Random forest', 'Extra trees', 'HistGradientBoosting', 'CatBoost', 'LightGBM', 'XGBoost', 'TabPFN']
available, advanced_skipped = model_registry(state['X_train'], mode=mode)
advanced_results = state.get('advanced_results', pd.DataFrame())
advanced_folds = state.get('advanced_folds', pd.DataFrame())
done = set(advanced_results.loc[advanced_results['status'].isin(['completed', 'failed']), 'model']) if not advanced_results.empty else set()
for name in advanced_names:
    if name in done:
        continue
    if name not in available:
        advanced_skipped[name] = advanced_skipped.get(name, 'Unavailable in the active model registry.')
        continue
    result, fold_result, skipped = benchmark_models(
        state['X_train'], state['y_train'], mode=mode, model_names=[name]
    )
    advanced_results = pd.concat([advanced_results, result], ignore_index=True)
    advanced_folds = pd.concat([advanced_folds, fold_result], ignore_index=True)
    advanced_skipped.update(skipped)
    state.update({'advanced_results': advanced_results, 'advanced_folds': advanced_folds,
                  'advanced_skipped': advanced_skipped})
    save_state(state)
    done.add(name)
    del result, fold_result
    gc.collect()
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f'Checkpointed {name}; available RAM: {available_gb:.2f} GiB')
    if available_gb < 0.45:
        raise RuntimeError(f'Stopping after {name}: available RAM fell below 0.45 GiB.')
display(advanced_results)
display(advanced_folds)
print('Optional candidates skipped or failed:', advanced_skipped)
completed = advanced_results[advanced_results['status'] == 'completed']
if not completed.empty:
    ax = completed.sort_values('roc_auc').plot.barh(x='model', y='roc_auc', xerr='roc_auc_std', figsize=(8, max(3, len(completed) * 0.45)), legend=False, title='Advanced model ROC-AUC by stratified CV')
    ax.set_xlabel('ROC-AUC (fold mean ± standard deviation)')
    plt.tight_layout()
    plt.show()
state.update({'advanced_results': advanced_results, 'advanced_folds': advanced_folds,
              'advanced_skipped': advanced_skipped})
from returnwise.experiments import log_experiment
log_experiment('Advanced model benchmark', None,
               {'results': advanced_results.to_dict(orient='records'), 'skipped': advanced_skipped},
               {'mode': state.get('experiment_mode', 'quick')})
save_state(state)
"""
    ),
    (
        "11_hyperparameter_optimization.ipynb",
        "11 · Hyperparameter optimization",
        "Run a reproducible, bounded Optuna search for a selected installed gradient-boosting model.",
        "Use Quick or Full mode, training folds only, a seeded TPE sampler, early stopping where supported by the installed estimator, and persist every trial and its duration.",
        "A search is meaningful only when a supported boosting family is installed. The untouched internal final holdout is not accessed by this notebook.",
        "Optuna documentation: https://optuna.readthedocs.io/en/stable/",
        """from returnwise.models import model_registry, tune_with_optuna
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt
import pandas as pd

state = load_state()
cv_parts = [x for x in [state.get('baseline_results'), state.get('advanced_results')] if x is not None]
cv_results = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
if cv_results.empty:
    raise RuntimeError('Run notebooks 09 and 10 first.')
registry, skipped = model_registry(state['X_train'], mode=state.get('experiment_mode', 'quick'))
boosting = [name for name in ['CatBoost', 'LightGBM', 'XGBoost'] if name in registry]
completed = cv_results[cv_results['status'] == 'completed'].sort_values('roc_auc', ascending=False)
best_cv_model = completed.iloc[0]['model'] if not completed.empty else None
if not boosting:
    print('No optional gradient-boosting library is available; optimization was not run.')
    print('Availability:', skipped)
    tuning_result = None
else:
    if best_cv_model not in boosting:
        print('The best CV candidate is not an installed modern boosting model; no Optuna run was performed.')
        tuning_result = None
    else:
        tuning_candidate = best_cv_model
        try:
            tuning_result = tune_with_optuna(state['X_train'], state['y_train'], tuning_candidate, mode=state.get('experiment_mode', 'quick'), seed=42)
            print('Selected search candidate:', tuning_candidate)
            print('Best training-CV ROC-AUC:', tuning_result['best_score_roc_auc'])
            print('Duration (seconds):', tuning_result['duration_seconds'])
            trials = pd.DataFrame(tuning_result['trials'])
            display(trials)
            ax = trials.sort_values('number').plot(x='number', y='value', marker='o', figsize=(7, 3), legend=False, title='Optuna trial ROC-AUC')
            ax.set_ylabel('Cross-validated ROC-AUC')
            plt.tight_layout()
            plt.show()
        except RuntimeError as error:
            print('Optuna search skipped:', error)
            tuning_result = None
state['tuning_result'] = tuning_result
from returnwise.experiments import log_experiment
if tuning_result:
    log_experiment('Optuna hyperparameter search', None,
                   {'best_score_roc_auc': tuning_result['best_score_roc_auc'], 'trials': tuning_result['trials']},
                   {'model': tuning_result['model'], 'best_parameters': tuning_result['best_parameters'],
                    'duration_seconds': tuning_result['duration_seconds']})
save_state(state)
"""
    ),
    (
        "12_ensemble_experiments.ipynb",
        "12 · Ensemble experiments",
        "Compare out-of-fold probability averaging with its individual members and retain blending only when the validation improvement is measured.",
        "Build out-of-fold predictions for the two strongest available CV candidates. Compare soft probability means on identical folds; no in-sample probabilities are used.",
        "Only pairwise mean blends are benchmarked in this baseline. Stacking is not added unless data volume and a separate nested evaluation justify it.",
        "Ensemble and validation design notes are in docs/implementation_plan.md.",
        """from returnwise.models import evaluate_probability_blend, model_registry
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt
import pandas as pd

state = load_state()
cv_parts = [x for x in [state.get('baseline_results'), state.get('advanced_results')] if x is not None]
cv_results = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
completed = cv_results[cv_results['status'] == 'completed'].sort_values('roc_auc', ascending=False)
registry, skipped = model_registry(state['X_train'], mode=state.get('experiment_mode', 'quick'))
members = [name for name in completed['model'].tolist() if name in registry][:2]
blend_candidate = None
if len(members) < 2:
    print('Fewer than two candidate families completed validation; no ensemble was evaluated.')
    blend_results, oof_probabilities = pd.DataFrame(), {}
else:
    blend_results, oof_probabilities = evaluate_probability_blend(
        state['X_train'], state['y_train'], members, mode=state.get('experiment_mode', 'quick'), seed=42
    )
    display(blend_results)
    single_rows = blend_results[~blend_results['blend'].str.startswith('OOF mean: ')]
    pair_rows = blend_results[blend_results['blend'].str.startswith('OOF mean: ')]
    baseline_best = float(single_rows['roc_auc'].max())
    best_blend_row = pair_rows.sort_values('roc_auc', ascending=False).iloc[0] if not pair_rows.empty else None
    best_blend = float(best_blend_row['roc_auc']) if best_blend_row is not None else float('-inf')
    blend_candidate = 'Soft voting: ' + best_blend_row['blend'].replace('OOF mean: ', '') if best_blend > baseline_best else None
    print('Best single-model OOF ROC-AUC:', baseline_best)
    print('Best probability blend OOF ROC-AUC:', best_blend)
    print('Blend retained for nested calibration and holdout study:', blend_candidate)
    ax = blend_results.plot.bar(x='blend', y='roc_auc', figsize=(8, 4), legend=False, title='Single candidates versus out-of-fold mean blends')
    ax.set_ylabel('Out-of-fold ROC-AUC')
    plt.tight_layout()
    plt.show()
state.update({'blend_results': blend_results, 'oof_probabilities': oof_probabilities,
              'selected_ensemble': blend_candidate})
from returnwise.experiments import log_experiment
log_experiment('OOF probability blend benchmark', None, {'results': blend_results.to_dict(orient='records')},
               {'members': members, 'selected_ensemble': blend_candidate})
save_state(state)
"""
    ),
    (
        "13_probability_calibration.ipynb",
        "13 · Probability calibration",
        "Compare reliability of uncalibrated, sigmoid, isotonic, and beta-calibrated probabilities without using the internal final holdout for selection.",
        "For each outer fold, reserve part of development data for calibration, train the base model on the remaining part, then evaluate the calibrated probabilities on the untouched outer fold. Select the model and calibrator from these CV results.",
        "Isotonic and beta maps can overfit on small calibration folds. ROC-AUC/PR-AUC measure discrimination, while Brier score, log loss, ECE, and reliability diagrams measure probability quality.",
        "Calibration references: https://proceedings.mlr.press/v54/kull17a.html · https://proceedings.mlr.press/v108/wenger20a.html",
        """from returnwise.experiments import run_calibration_cv
from returnwise.models import model_registry
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt
import pandas as pd

state = load_state()
cv_parts = [x for x in [state.get('baseline_results'), state.get('advanced_results')] if x is not None]
cv_results = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame()
completed = cv_results[cv_results['status'] == 'completed'].sort_values('roc_auc', ascending=False)
registry, skipped = model_registry(state['X_train'], mode=state.get('experiment_mode', 'quick'))
leaders = [name for name in completed['model'].tolist() if name in registry][:3]
blend_candidate = state.get('selected_ensemble')
if blend_candidate:
    leaders.append(blend_candidate)
if not leaders:
    raise RuntimeError('No validated model candidate is available; run notebooks 09 and 10.')
calibration_folds = pd.concat(
    [run_calibration_cv(
        state['X_train'], state['y_train'], name,
        mode=state.get('experiment_mode', 'quick'), seed=42,
        parameters=(state.get('tuning_result') or {}).get('pipeline_parameters') if (state.get('tuning_result') or {}).get('model') == name else None,
    ).assign(model=name) for name in leaders],
    ignore_index=True,
)
calibration_summary = calibration_folds[calibration_folds['status'] == 'completed'].groupby(['model', 'method'], as_index=False).agg(
    roc_auc=('roc_auc', 'mean'), pr_auc=('pr_auc', 'mean'), brier_score=('brier_score', 'mean'),
    log_loss=('log_loss', 'mean'), expected_calibration_error=('expected_calibration_error', 'mean')
)
display(calibration_folds)
display(calibration_summary)
uncalibrated = calibration_summary[calibration_summary['method'] == 'uncalibrated'].rename(
    columns={'roc_auc': 'cv_roc_auc', 'pr_auc': 'cv_pr_auc'}
)
stability = calibration_folds[(calibration_folds['status'] == 'completed') & (calibration_folds['method'] == 'uncalibrated')].groupby('model', as_index=False).agg(
    cv_roc_auc_std=('roc_auc', 'std'), cv_pr_auc_std=('pr_auc', 'std')
)
calibration_choice = calibration_summary.sort_values(
    ['brier_score', 'log_loss', 'expected_calibration_error']
).groupby('model', as_index=False).first().rename(columns={'method': 'selected_calibrator'})
selection_table = uncalibrated[['model', 'cv_roc_auc', 'cv_pr_auc']].merge(
    calibration_choice[['model', 'selected_calibrator', 'brier_score', 'log_loss', 'expected_calibration_error']],
    on='model', how='inner'
)
selection_table = selection_table.merge(stability, on='model', how='left')
timing_parts = [x[x['status'] == 'completed'] for x in [state.get('baseline_results'), state.get('advanced_results')] if x is not None]
timing = pd.concat(timing_parts, ignore_index=True).groupby('model', as_index=False).agg(
    mean_fit_seconds=('mean_fit_seconds', 'mean'), mean_inference_seconds=('mean_inference_seconds', 'mean'),
    mean_rss_delta_mb=('mean_rss_delta_mb', 'mean')
) if timing_parts else pd.DataFrame()
selection_table = selection_table.merge(timing, on='model', how='left')
if blend_candidate:
    blend_name = blend_candidate.replace('Soft voting: ', 'OOF mean: ')
    blend_row = state['blend_results'].loc[state['blend_results']['blend'] == blend_name].iloc[0]
    fold_count = 3 if state.get('experiment_mode', 'quick') == 'quick' else 5
    selection_table.loc[selection_table['model'] == blend_candidate, 'mean_fit_seconds'] = blend_row['fit_seconds_total'] / fold_count
    selection_table.loc[selection_table['model'] == blend_candidate, 'mean_inference_seconds'] = blend_row['inference_seconds_total'] / fold_count
    selection_table.loc[selection_table['model'] == blend_candidate, 'mean_rss_delta_mb'] = blend_row['rss_delta_mb_total'] / fold_count
for metric, ascending in [('cv_roc_auc', False), ('cv_pr_auc', False), ('brier_score', True),
                          ('log_loss', True), ('cv_roc_auc_std', True),
                          ('mean_fit_seconds', True), ('mean_rss_delta_mb', True)]:
    selection_table[f'rank_{metric}'] = selection_table[metric].rank(pct=True, ascending=ascending)
selection_table['selection_score'] = (
    0.28 * selection_table['rank_cv_roc_auc'] + 0.18 * selection_table['rank_cv_pr_auc']
    + 0.24 * selection_table['rank_brier_score'] + 0.14 * selection_table['rank_log_loss']
    + 0.07 * selection_table['rank_cv_roc_auc_std'] + 0.05 * selection_table['rank_mean_fit_seconds']
    + 0.04 * selection_table['rank_mean_rss_delta_mb']
)
selection_table = selection_table.sort_values('selection_score').reset_index(drop=True)
selected_model = selection_table.iloc[0]['model']
selected_calibrator = selection_table.iloc[0]['selected_calibrator']
print('Model selection table with discrimination, calibration, runtime, and memory measurements:')
display(selection_table)
print('CV-selected model:', selected_model)
print('CV-selected calibrator:', selected_calibrator)
print('Selection is based on training CV only; the final holdout remains untouched.')
chart = calibration_summary[calibration_summary['model'] == selected_model].set_index('method')['brier_score']
ax = chart.plot.bar(figsize=(6, 3), color='#126E63', title=f'Calibration candidate Brier scores: {selected_model}')
ax.set_ylabel('Lower is better')
plt.tight_layout()
plt.show()
state.update({'calibration_folds': calibration_folds, 'calibration_summary': calibration_summary,
              'selection_table': selection_table, 'selected_model': selected_model,
              'selected_calibrator': selected_calibrator})
from returnwise.experiments import log_experiment
log_experiment('Probability calibration and model selection', None,
               {'selection': selection_table.to_dict(orient='records'),
                'calibration': calibration_summary.to_dict(orient='records')},
               {'selected_model': selected_model, 'selected_calibrator': selected_calibrator})
save_state(state)
"""
    ),
    (
        "14_explainable_ai.ipynb",
        "14 · Explainable AI",
        "Measure global permutation importance on a validation partition and attempt SHAP global and local plots when the installed model and dependency support them.",
        "Fit the selected CV model on a training subset and evaluate permutation importance on a held-out part of the training split. SHAP uses a small sample; any failure falls back to permutation importance.",
        "Importance describes model reliance under this data distribution, not a causal effect. Correlated features can divide or mask importance.",
        "Lundberg & Lee (2017): https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
        """from returnwise.explain import permutation_importance_table, shap_explanation
from returnwise.models import fit_best_model
from returnwise.research_state import load_state, save_state
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

state = load_state()
X_train, y_train = state['X_train'], state['y_train']
x_fit, x_valid, y_fit, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)
params = (state.get('tuning_result') or {}).get('pipeline_parameters') if (state.get('tuning_result') or {}).get('model') == state['selected_model'] else None
explain_model = fit_best_model(x_fit, y_fit, state['selected_model'], mode=state.get('experiment_mode', 'quick'), seed=42, parameters=params)
importance = permutation_importance_table(explain_model, x_valid, y_valid, repeats=5, seed=42)
display(importance)
ax = importance.head(20).sort_values('importance_mean_roc_auc_decrease').plot.barh(
    x='feature', y='importance_mean_roc_auc_decrease', xerr='importance_std',
    figsize=(8, max(3, min(20, len(importance)) * 0.3)), legend=False, color='#126E63',
    title='Validation permutation importance'
)
ax.set_xlabel('Decrease in ROC-AUC after permutation')
plt.tight_layout()
plt.show()
shap_status = 'not attempted'
try:
    explanation = shap_explanation(explain_model, x_valid, max_rows=200)
    import shap
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.show()
    shap_status = 'summary plot completed'
    try:
        top_feature = importance.iloc[0]['feature']
        shap.plots.scatter(explanation[:, top_feature], color=explanation, show=False)
        plt.tight_layout()
        plt.show()
        shap_status += '; dependence scatter completed'
    except Exception as dependence_error:
        print('SHAP dependence plot unavailable:', dependence_error)
    try:
        shap.plots.waterfall(explanation[0], max_display=12, show=False)
        plt.tight_layout()
        plt.show()
        shap_status += '; one local waterfall completed'
    except Exception as local_error:
        print('Local SHAP waterfall unavailable:', local_error)
except Exception as error:
    print('SHAP unavailable for this model or environment:', error)
    print('Permutation importance remains the model-agnostic fallback.')
state.update({'permutation_importance': importance.to_dict(orient='records'), 'shap_status': shap_status})
from returnwise.experiments import log_experiment
log_experiment('Explainable AI feature importance', None,
               {'permutation_importance': importance.head(20).to_dict(orient='records')},
               {'model': state['selected_model'], 'shap_status': shap_status})
save_state(state)
"""
    ),
    (
        "15_uncertainty_analysis.ipynb",
        "15 · Uncertainty analysis",
        "Flag low-margin probabilities for additional review and assess risk discrimination on progressively more confident subsets.",
        "Create a decision-study holdout from the training partition, calibrate only on its separate calibration subset, and summarize probability margin, coverage, error, and ROC-AUC among retained confident predictions.",
        "Margin-based flags are heuristic. They are not a conformal prediction set and provide no finite-sample coverage guarantee.",
        "For this study, uncertainty flags are operational review cues; conformal methods are not enabled.",
        """from returnwise.experiments import make_holdout_predictions
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

state = load_state()
decision_model, decision_calibrators, decision_predictions, decision_metrics = make_holdout_predictions(
    state['X_train'], state['y_train'], state['selected_model'], mode=state.get('experiment_mode', 'quick'), seed=104,
    parameters=(state.get('tuning_result') or {}).get('pipeline_parameters') if (state.get('tuning_result') or {}).get('model') == state['selected_model'] else None,
)
calibrator_name = state['selected_calibrator']
if calibrator_name not in decision_predictions.columns.str.replace('probability_', '').tolist():
    calibrator_name = 'uncalibrated'
calibrated_column = f'probability_{calibrator_name}'
decision_predictions.attrs['selected_calibrator'] = calibrator_name
p = decision_predictions[calibrated_column].to_numpy()
y = decision_predictions['y_true'].to_numpy()
decision_predictions['probability_margin'] = np.abs(p - 0.5) * 2
decision_predictions['confidence'] = np.maximum(p, 1 - p)
decision_predictions['uncertainty_flag'] = decision_predictions['confidence'] < 0.60
display(decision_metrics)
print('Selected calibration method used for the decision-study holdout:', calibrator_name)
for min_confidence in [0.50, 0.60, 0.70, 0.80, 0.90]:
    mask = decision_predictions['confidence'] >= min_confidence
    score = roc_auc_score(y[mask], p[mask]) if mask.sum() > 1 and len(np.unique(y[mask])) == 2 else float('nan')
    print({'minimum_confidence': min_confidence, 'coverage': float(mask.mean()), 'rows_retained': int(mask.sum()), 'roc_auc_retained': score})
counts = decision_predictions['uncertainty_flag'].value_counts().rename(index={False: 'confident', True: 'uncertain'})
ax = counts.plot.bar(figsize=(5, 3), color=['#126E63', '#B56B33'], title='Decision-study holdout uncertainty flags')
ax.set_ylabel('Transactions')
plt.tight_layout()
plt.show()
state.update({'decision_model': decision_model, 'decision_calibrators': decision_calibrators,
              'decision_predictions': decision_predictions, 'decision_metrics': decision_metrics,
              'decision_calibrator_name': calibrator_name})
from returnwise.experiments import log_experiment
log_experiment('Uncertainty analysis', None,
               {'calibration_metrics': decision_metrics.to_dict(orient='records'),
                'uncertain_rows': int(decision_predictions['uncertainty_flag'].sum())},
               {'model': state['selected_model'], 'calibrator': calibrator_name})
save_state(state)
"""
    ),
    (
        "16_cost_sensitive_decision_analysis.ipynb",
        "16 · Cost-sensitive decision analysis",
        "Compare random, probability-only, fixed-threshold, cost-sensitive, calibration-aware, and uncertainty-aware review under explicit low, medium, and high cost assumptions.",
        "Evaluate only on the decision-study labeled holdout. Compute precision@K, recall@K, realized false positives/negatives, expected-decision-cost formula, and reviewed-case counts at 5%, 10%, 20%, and 30% capacities.",
        "Cost values are illustrative scenario inputs, not actual business measurements. Results do not demonstrate causal return prevention or profit.",
        "Decision equations and interpretation boundaries: docs/references.md",
        """from returnwise.decision import evaluate_review_strategies
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt
import pandas as pd

state = load_state()
predictions = state['decision_predictions']
calibrator = state['decision_calibrator_name']
probability = predictions[f'probability_{calibrator}'].to_numpy()
raw = predictions['probability_uncalibrated'].to_numpy()
y_true = predictions['y_true'].to_numpy()
scenarios = {
    'low': {'false_positive': 0.5, 'false_negative': 2.0, 'review': 0.1},
    'medium': {'false_positive': 1.0, 'false_negative': 5.0, 'review': 0.25},
    'high': {'false_positive': 2.0, 'false_negative': 10.0, 'review': 0.5},
}
parts = []
for scenario, costs in scenarios.items():
    result = evaluate_review_strategies(raw, y_true=y_true, calibrated_probabilities=probability, costs=costs)
    result.insert(0, 'scenario', scenario)
    parts.append(result)
decision_results = pd.concat(parts, ignore_index=True)
display(decision_results)
plot = decision_results[(decision_results['scenario'] == 'medium') & (decision_results['capacity_percent'] == 10)]
ax = plot.plot.bar(x='strategy', y='expected_decision_cost', figsize=(9, 4), legend=False, color='#126E63', title='Observed holdout decision cost at 10% capacity (medium assumptions)')
ax.set_ylabel('Simulated cost units')
plt.tight_layout()
plt.show()
state.update({'decision_results': decision_results, 'cost_scenarios': scenarios})
from returnwise.experiments import log_experiment
log_experiment('Cost-sensitive decision analysis', None,
               {'results': decision_results.to_dict(orient='records')}, {'scenarios': scenarios})
save_state(state)
"""
    ),
    (
        "17_adaptive_priority_experiments.ipynb",
        "17 · Adaptive priority experiments",
        "Measure how expected and observed prioritization metrics change across review capacity and cost assumptions.",
        "Run sensitivity analysis over 5%, 10%, 20%, and 30% capacities and three explicitly hypothetical cost profiles, retaining strategy-level precision, recall, and false-negative cost.",
        "Capacity rankings are evaluated on one decision-study holdout. They require replication across folds before a stability claim.",
        "The cost simulator and metrics are implemented in returnwise.decision.",
        """from returnwise.decision import evaluate_review_strategies
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt
import pandas as pd

state = load_state()
predictions = state['decision_predictions']
calibrator = state['decision_calibrator_name']
raw = predictions['probability_uncalibrated'].to_numpy()
calibrated = predictions[f'probability_{calibrator}'].to_numpy()
y = predictions['y_true'].to_numpy()
rows = []
for fn_cost in [2.0, 5.0, 10.0]:
    for review_cost in [0.1, 0.25, 0.5]:
        costs = {'false_positive': 1.0, 'false_negative': fn_cost, 'review': review_cost}
        part = evaluate_review_strategies(raw, y_true=y, calibrated_probabilities=calibrated, costs=costs)
        part['false_negative_cost_scenario'] = fn_cost
        part['review_cost_scenario'] = review_cost
        rows.append(part)
priority_sensitivity = pd.concat(rows, ignore_index=True)
display(priority_sensitivity)
plot_data = priority_sensitivity[(priority_sensitivity['strategy'] == 'calibration_aware_cost_sensitive') & (priority_sensitivity['review_cost_scenario'] == 0.25)]
for fn_cost, group in plot_data.groupby('false_negative_cost_scenario'):
    plt.plot(group['capacity_percent'], group['recall_at_k'], marker='o', label=f'FN cost {fn_cost:g}')
plt.xlabel('Review capacity (%)')
plt.ylabel('Observed recall@K')
plt.title('Capacity and false-negative-cost sensitivity')
plt.legend()
plt.tight_layout()
plt.show()
state['priority_sensitivity'] = priority_sensitivity
from returnwise.experiments import log_experiment
log_experiment('Adaptive priority sensitivity analysis', None,
               {'results': priority_sensitivity.to_dict(orient='records')},
               {'capacities': [5, 10, 20, 30], 'costs_are_simulated': True})
save_state(state)
"""
    ),
    (
        "18_novelty_ablation_study.ipynb",
        "18 · Ablation and contribution checks",
        "Test the proposed combination as an empirical ablation of standard components, without claiming global novelty.",
        "Compare probability-ranking, calibration, cost-sensitive selection, the combined calibrated cost strategy, and uncertainty-first review on the decision-study holdout; report AUC, PR-AUC, Brier, log loss, precision/recall@K, cost, and runtime.",
        "One holdout does not prove stability. Cost-sensitive ranking can be monotonic in probability for fixed costs, so equal rankings are a possible and informative result.",
        "Related established components and the novelty boundary are summarized in docs/references.md.",
        """from returnwise.experiments import ablation_table
from returnwise.research_state import load_state, save_state
import matplotlib.pyplot as plt

state = load_state()
predictions = state['decision_predictions']
predictions.attrs['selected_calibrator'] = state['decision_calibrator_name']
ablation = ablation_table(predictions, costs=state['cost_scenarios']['medium'])
display(ablation)
if 'expected_decision_cost' in ablation:
    plot_data = ablation[ablation['capacity_percent'] == 10]
    ax = plot_data.plot.bar(x='experiment', y='expected_decision_cost', figsize=(9, 4), legend=False, color='#126E63', title='Ablation expected cost at 10% review capacity')
    ax.set_ylabel('Simulated cost units')
    plt.tight_layout()
    plt.show()
state['ablation'] = ablation
from returnwise.experiments import log_experiment
log_experiment('Ablation study', None, {'results': ablation.to_dict(orient='records')},
               {'costs_are_simulated': True, 'decision_holdout_rows': len(predictions)})
save_state(state)
"""
    ),
    (
        "19_final_model_evaluation.ipynb",
        "19 · Final model evaluation",
        "Evaluate the CV-selected model and calibrator exactly once on the stratified internal holdout that remained untouched throughout selection.",
        "Fit the model on a development subset of the training split, fit the selected calibrator on a separate calibration subset, then compute discrimination, calibration, confusion matrix, and capacity-constrained costs on the final holdout. Compare ROC-AUC/PR-AUC against random/prevalence baselines and Brier/log loss against a constant-prevalence predictor before export.",
        "The final holdout is internal and random-stratified because event timestamps are unavailable. Do not use these results to retune the model or calibrator.",
        "Metric definitions: https://scikit-learn.org/stable/modules/model_evaluation.html",
        """from returnwise.calibration import ProbabilityCalibrator, expected_calibration_error
from returnwise.decision import evaluate_review_strategies
from returnwise.models import fit_best_model, _probabilities
from returnwise.research_state import load_state, save_state
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

state = load_state()
X_train, y_train = state['X_train'], state['y_train']
X_final, y_final = state['X_final'], state['y_final']
x_base, x_cal, y_base, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=2718, stratify=y_train)
params = (state.get('tuning_result') or {}).get('pipeline_parameters') if (state.get('tuning_result') or {}).get('model') == state['selected_model'] else None
final_model = fit_best_model(x_base, y_base, state['selected_model'], mode=state.get('experiment_mode', 'quick'), seed=42, parameters=params)
raw_cal = _probabilities(final_model, x_cal)
raw_final = _probabilities(final_model, X_final)
method = state['selected_calibrator']
final_calibrators = {'uncalibrated': None}
if method != 'uncalibrated':
    final_calibrators[method] = ProbabilityCalibrator(method).fit(raw_cal, y_cal)
p_final = raw_final if method == 'uncalibrated' else final_calibrators[method].predict(raw_final)
pred = (p_final >= 0.5).astype(int)
final_metrics = {
    'roc_auc': roc_auc_score(y_final, p_final),
    'pr_auc': average_precision_score(y_final, p_final),
    'brier_score': brier_score_loss(y_final, p_final),
    'log_loss': log_loss(y_final, p_final, labels=[0, 1]),
    'expected_calibration_error': expected_calibration_error(y_final, p_final),
    'holdout_prevalence': float(y_final.mean()),
    'constant_prevalence_brier': brier_score_loss(y_final, np.full(len(y_final), float(y_final.mean()))),
    'constant_prevalence_log_loss': log_loss(y_final, np.full(len(y_final), float(y_final.mean())), labels=[0, 1]),
    'threshold': 0.5,
    'precision_at_threshold': precision_score(y_final, pred, zero_division=0),
    'recall_at_threshold': recall_score(y_final, pred, zero_division=0),
    'f1_at_threshold': f1_score(y_final, pred, zero_division=0),
    'confusion_matrix_tn_fp_fn_tp': confusion_matrix(y_final, pred, labels=[0, 1]).ravel().tolist(),
    'holdout_rows': len(X_final),
    'selected_model': state['selected_model'],
    'selected_calibrator': method,
}
observed_frequency, mean_probability = calibration_curve(y_final, p_final, n_bins=10, strategy='quantile')
final_metrics['reliability_curve'] = [
    {'mean_predicted_probability': float(predicted), 'observed_return_rate': float(observed)}
    for predicted, observed in zip(mean_probability, observed_frequency)
]
print('One-time internal final holdout evaluation:')
display(pd.Series(final_metrics, name='value').to_frame())
final_costs = evaluate_review_strategies(raw_final, y_true=y_final, calibrated_probabilities=p_final)
display(final_costs)
observed, predicted = calibration_curve(y_final, p_final, n_bins=10, strategy='quantile')
plt.plot(predicted, observed, marker='o', label='Final holdout')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Observed return frequency')
plt.title('Final holdout reliability diagram')
plt.legend()
plt.tight_layout()
plt.show()
state.update({'final_model': final_model, 'final_calibrators': final_calibrators,
              'final_metrics': final_metrics, 'final_raw_probabilities': raw_final,
              'final_probabilities': p_final, 'final_costs': final_costs})
from returnwise.experiments import log_experiment
log_experiment('One-time internal final holdout evaluation', None,
               {'metrics': final_metrics, 'costs': final_costs.to_dict(orient='records')},
               {'selected_model': state['selected_model'], 'selected_calibrator': method,
                'holdout_rows': len(X_final), 'temporal_validation': False})
save_state(state)
"""
    ),
    (
        "20_returnwise_model_export.ipynb",
        "20 · ReturnWise model export",
        "Export the selected real model, calibration map, feature schema, dataset fingerprint, and one-time internal holdout metrics for application inference.",
        "Serialize only the model selected by training CV and calibrated using its separate calibration split, and only when the internal holdout exceeds the random/prevalence discrimination baseline and the constant-prevalence Brier/log-loss baselines. Store the temporal scenario, selected calibrator, validation metrics, available permutation importance, and editable decision defaults. Reload the artifact and require prediction consistency before marking export complete.",
        "The exported artifact is only valid for the saved feature schema and installed package versions. Re-run experiments after data or dependency changes.",
        "Model export and prediction schema: returnwise.export and returnwise.storage.",
        """from returnwise.data.kaggle import read_metadata
from returnwise.export import export_model, make_competition_submission, predict_with_artifact
from returnwise.paths import KAGGLE_DIR
from returnwise.research_state import load_state, save_state
from returnwise.storage import load_model_artifact
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

state = load_state()
if 'final_model' not in state or 'final_metrics' not in state:
    raise RuntimeError('Run notebook 19 to complete the one-time final holdout evaluation.')
manifest = state.get('dataset_manifest') or {}
kaggle_metadata = read_metadata()
fingerprint = (kaggle_metadata or {}).get('fingerprint_sha256')
if not fingerprint:
    from returnwise.storage import active_dataset_fingerprint
    fingerprint = active_dataset_fingerprint()
importance = state.get('permutation_importance', [])
metadata = export_model(
    model=state['final_model'],
    feature_columns=state['feature_columns'],
    selected_model=state['selected_model'],
    dataset_fingerprint=fingerprint,
    calibrators=state['final_calibrators'],
    selected_calibrator=state['selected_calibrator'],
    permutation_importance=importance,
    prediction_scenario='pre_shipment',
    validation_metrics=state['final_metrics'],
    decision_configuration={
        'review_capacity_percent': [5, 10, 20, 30],
        'default_capacity_percent': 10,
        'fixed_threshold': 0.5,
        'uncertainty_confidence_ceiling': 0.60,
        'cost_scenarios': state.get('cost_scenarios', {}),
        'costs_are_simulated': True,
    },
)
print('Exported artifact metadata:')
display(pd.Series(metadata))
raw, calibrated = predict_with_artifact(
    {'model': state['final_model'], 'calibrators': state['final_calibrators'],
     'feature_columns': state['feature_columns'], 'metadata': metadata},
    state['X_final'].drop(columns=[], errors='ignore')
)
assert len(raw) == len(state['X_final']) and len(calibrated) == len(raw)
persisted_payload = load_model_artifact()
if persisted_payload is None:
    raise RuntimeError('The exported model could not be loaded back from disk.')
raw_reloaded, calibrated_reloaded = predict_with_artifact(persisted_payload, state['X_final'])
np.testing.assert_allclose(raw, raw_reloaded, rtol=0, atol=1e-12)
np.testing.assert_allclose(calibrated, calibrated_reloaded, rtol=0, atol=1e-12)
print('Saved artifact reload and prediction consistency checks passed on the held-out rows; these predictions are not substituted for the measured evaluation.')
state['export_metadata'] = metadata
save_state(state)
test_path, sample_path = KAGGLE_DIR / 'test.csv', KAGGLE_DIR / 'sample_submission.csv'
if manifest.get('source') == 'kaggle' and test_path.exists() and sample_path.exists():
    competition_test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    competition_submission = make_competition_submission(
        persisted_payload,
        competition_test, sample_submission
    )
    output_path = KAGGLE_DIR / 'returnwise_submission.csv'
    competition_submission.to_csv(output_path, index=False)
    print('Competition submission file created from unlabeled test rows:', output_path)
    print('Submission schema matches sample_submission.csv:', list(competition_submission.columns) == list(sample_submission.columns))
    display(competition_submission.head())
else:
    print('The active source is manual, or competition test/submission files are unavailable; no competition submission was created.')
from returnwise.experiments import log_experiment
log_experiment('Model export', fingerprint,
               {'success': True, 'selected_model': state['selected_model'],
                'roundtrip_verified': True, 'validation_metrics': state['final_metrics']},
               {'metadata': metadata, 'decision_configuration': metadata['decision_configuration']})
save_state(state)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(['ROC-AUC', 'PR-AUC', 'Brier'], [state['final_metrics']['roc_auc'], state['final_metrics']['pr_auc'], state['final_metrics']['brier_score']], color=['#126E63', '#3E8A7D', '#B56B33'])
ax.set_title('Recorded internal holdout summary (lower Brier is better)')
plt.tight_layout()
plt.show()
"""
    ),
]


def main():
    for spec in SPECS:
        create_notebook(*spec)
    print(f"Created {len(SPECS)} clean notebooks in {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
