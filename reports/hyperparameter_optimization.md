# ReturnWise Hyperparameter Optimization

Date: 2026-11-15. Study database: `experiments/optuna_studies.db` (ignored by Git and retained locally).

## Objective and hypothesis

Test a reproducible, resumable Optuna path for LightGBM using only development data. The hypothesis was that a small search could identify a candidate scoring better than the quick defaults while preserving outer-fold separation.

## Method and parameters

The implementation uses Optuna 4.9.0, TPE sampling, a median pruner, SQLite persistence, one worker, incremental trial budgets, and `gc_after_trial`. For each outer stratified fold, early stopping uses a validation subset drawn only from that fold's training partition; the outer validation fold is scored separately. The pilot ran two trials of three-fold ROC-AUC on the 160,000-row development frame. Study name: `returnwise-lightgbm-quick-seed42`.

## Results

| Trial | State | 3-fold ROC-AUC |
|---:|---|---:|
| 0 | Complete | 0.586433 |
| 1 | Complete | **0.587598** |

The best sampled settings were `n_estimators=132`, `learning_rate=0.101984`, `num_leaves=16`, `max_depth=10`, `min_child_samples=30`, `subsample=0.755021`, `subsample_freq=2`, `colsample_bytree=0.857427`, `reg_alpha=0.00007718`, and `reg_lambda=0.00000418`. There were no pruned or failed trials. The pilot completed in about 20 seconds.

## Interpretation and conclusion

The persistent study and trial history were successfully exercised, including SQLite resume support. Two trials do not explore the search space adequately and cannot establish a tuned optimum or reliable improvement. The best score is comparable to the corrected raw LightGBM baseline, but no paired uncertainty analysis was run. Treat the parameters as the best of two samples only.

## Limits and status

CatBoost and XGBoost search spaces exist in code but were not run because those libraries are absent. Available RAM after the pilot was about 0.471 GiB, only slightly above the runner's 0.45 GiB safety floor; additional tuning was stopped under the prompt's resource rule. Larger search, repeated CV, and new holdout evaluation remain incomplete. The environment has Optuna 4.9.0 and LightGBM 4.7.0 installed; `pip check` found only the pre-existing unrelated `scikit-image` / `networkx` mismatch described in [deep_research_audit.md](deep_research_audit.md).
