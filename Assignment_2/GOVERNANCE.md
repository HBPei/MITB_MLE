# Credit default model: governance, refresh SOP and deployment

Model: probability that a new loan is **30+ days past due at month-on-book 6** (`label_def = 30dpd_6mob`).
Candidate algorithms: Logistic Regression (WoE categoricals) and XGBoost (monotone constraints).
Orchestration: two Airflow DAGs. `credit_model_training` (automatic: once at start-up, then on every approved
refresh decision) builds the stores from all history and
trains/registers the champion; `credit_model_inference` (monthly) scores new applications, monitors and applies the
refresh policy. Shared settings: `dags/pipeline_config.py`. Code: `scripts/data_pipeline.py` (data),
`scripts/utils/feature_inference.py` (features for new applications), `scripts/main.py` (model).
Every policy number below is a constant in `main.py` (`GOVERNANCE`, `MONITORING`, `CHAMPION_METRIC`), so this
document and the running pipeline cannot drift apart.

---

## 1. Roles

| Role | Responsibility |
|---|---|
| Model owner (credit risk) | Approves the refresh policy, thresholds and any feature change; signs off manual promotions/rollbacks |
| ML engineer | Operates both pipelines, triages monitoring alerts, executes the rollback SOP |
| Independent validator | Reviews each new champion's training record (registry + governance log) before it is relied on for decisions |
| Data engineering | Owns the bronze/silver/gold feature and label stores the pipelines depend on |

## 2. Model store and audit trail

| Artefact | Location | Content |
|---|---|---|
| Model artefact | `model_bank/credit_model_<algo>_<train_date>.pkl` | fitted preprocessor + model, features, hyper-parameters, split dates, data stats, results per split, decision threshold, cleaning rules, monitoring baseline (train score / feature distributions), LR coefficients or XGB monotone constraints + SHAP |
| Registry | `model_bank/model_registry.json` | every model ever trained, every training run (trigger, gates, comparison, outcome), `role_history` = which model was champion / challenger from which date |
| Governance log (gold) | `datamart/gold/model_governance/` | append-only: every monthly refresh decision, training outcome and manual promotion |
| Feature parameters | `datamart/gold/feature_store/feature_params_<train_date>.pkl` | imputation medians, percentile caps and age bins fitted on that training run's train window; applied (never re-fitted) to new applications |
| Inference features (gold) | `datamart/gold/inference_feature_store/inference_features_<yyyy_mm_dd>.parquet` | one partition per application month, built from that month's raw data with the champion's feature parameters |
| Predictions (gold) | `datamart/gold/model_predictions/<model_version>/` | one partition per month, with `model_role` (champion / challenger) |
| Monitoring (gold) | `datamart/gold/model_monitoring/`, `datamart/gold/model_feature_drift/` | one row per model per scoring month; per-feature PSI |
| Dashboard | `datamart/gold/model_reports/model_dashboard_asof_<ds>.png` | regenerated every run; one image per month = time-lapse of what was known when |

Artefacts are immutable: a refresh creates a new version, nothing is overwritten. The champion is resolved **as of a
date** from `role_history`, so re-running any past DAG run reproduces exactly the model that was live that month.

## 3. Pipelines

**Training (`credit_model_training`, automatic: runs once on unpause, then whenever the inference pipeline's
refresh decision is "train"; params `model_train_date`, `effective_from`)**
```
label_store   : check -> bronze -> silver -> gold label store           (full history, rebuilt)
feature_store : attributes / financials / clickstream: check -> bronze -> silver -> gold feature store
                (train/validation/test/OOT counted back from model_train_date; fitted imputation
                 parameters saved as feature_params_<date>.pkl)
model_training: train_logreg + train_xgboost -> select_champion (gates + champion comparison -> registry)
```

**Inference (`credit_model_inference`, monthly, logical date `ds` = application month)**
```
check_champion
new_applications : this month's attributes + financials -> bronze -> silver -> inference features
                   (training parameters applied, never re-fitted; no label needed)
label_refresh    : label store rebuilt so newly matured labels reach monitoring
model_inference  : score_champion (production) + score_challenger_shadow -> gold model_predictions
model_monitoring : monitor_champion + monitor_challenger -> gold model_monitoring / feature_drift -> dashboard
governance       : refresh_decision -> retrain_recommended (triggers training) | model_healthy
```
`wait_for_champion` (a sensor) holds each month until a champion exists and no training run is in progress.

`max_active_runs=1` and `depends_on_past=True`: month *n+1* starts only after month *n* has finished, because the
refresh decision of month *n+1* reads the monitoring written in month *n*.

## 4. When to refresh the model

| Trigger | Rule | Rationale |
|---|---|---|
| **Initial deployment** | trigger `credit_model_training` once: LR + XGB trained as of 2025-06-01 on all labelled history with the notebook's train / validation / test / OOT windows; the gated winner becomes champion from `effective_from` | the development model, validated on an untouched OOT window, is what goes to production |
| **Scheduled** | champion trained ≥ 6 months ago | 6 new matured label months is ~ +50% data at the first refresh and matches the label lag. Refreshing faster adds little new information |
| **Triggered** | latest champion monitoring = `alert` (AUC drop > 0.05 vs its test AUC on a matured month, or score / feature PSI > 0.25) | performance decay or population shift |
| **Cooldown** | no triggered refresh within 2 months of the last training | avoids retraining every month on the same alert while new labels have not arrived |
| **Manual** | data definition change, policy change, regulatory request | trigger `credit_model_training` + review, or `promote` for a rollback |

A **refresh** retrains the *approved* feature set and preprocessing on newer data. Adding or removing features,
changing encodings or the target definition is **new model development**: it goes through EDA, validation and
model-owner sign-off, not through the automatic refresh.

### How a refresh trains (`credit_model_training` with `model_train_date`)
- The gold feature store is rebuilt with windows counted back from `model_train_date`: OOT = last 2 months,
  test = 2 months before that, validation = 1 month, train = everything earlier; imputation parameters are refitted
  on that train window and saved for the inference pipeline.
- Hyper-parameters are chosen by expanding-window, time-ordered CV inside train. XGBoost's number of trees is set by
  early stopping on validation. The decision threshold is the KS point on validation.
- Built-in checks: LR coefficient-sign enforcement, XGBoost monotonicity verification (ICE), and a pickle reload
  test that must reproduce the stored OOT AUC.

## 5. Promotion: champion / challenger

1. Rank the candidates (LR, XGB) on **test Gini** → best candidate.
2. **Quality gates** (all must pass): test Gini ≥ 0.30, OOT Gini ≥ 0.30, |Gini_test − Gini_OOT| ≤ 0.10,
   monotonicity verified.
3. **Comparison on the same window**: the incumbent champion is re-scored on the candidate's OOT window (data neither
   model was trained on). Promote if candidate OOT Gini ≥ champion OOT Gini (+ uplift, default 0).
4. Outcome:
   - Promoted: the candidate becomes champion from `ds`, and the runner-up algorithm becomes the shadow challenger.
   - Not promoted: the champion stays, and the new candidate becomes the shadow challenger, so it builds a
     production track record.
5. Every outcome (gates, comparison, reason) goes to the registry and the governance log.

The challenger scores every month in **shadow mode**. Its predictions are stored and monitored but never used for
decisions. The dashboard plots champion vs challenger Gini on the same months.

## 6. Monitoring and actions

| Metric (per scoring month) | Watch | Alert | Available |
|---|---|---|---|
| AUC drop vs champion's test AUC | > 0.03 | > 0.05 | 6 months after scoring (label maturity) |
| Score PSI vs training baseline | > 0.10 | > 0.25 | immediately |
| Max feature PSI vs training baseline | > 0.10 | > 0.25 | immediately |
| Calibration: mean predicted PD vs observed default rate | reviewed on dashboard | | 6 months after scoring |

| Status | Action | Owner | Time |
|---|---|---|---|
| ok | none | | |
| watch | investigate the drifting feature / segment; note in monthly model review | ML engineer | next monthly review |
| alert | automatic triggered refresh (subject to cooldown); if the refreshed model fails gates → escalate | ML engineer → model owner | within the month |
| repeated alert or failed gates | consider rollback to previous champion or challenger; open model redevelopment | model owner | 5 business days |

Stability metrics are an early warning; performance metrics are the confirmation 6 months later.

### Refresh SOP
1. Every month `governance.refresh_decision` applies the policy above and writes the decision to the governance log.
2. On "train", `retrain_recommended` triggers `credit_model_training` automatically with
   `model_train_date` = this month and `effective_from` = next month; the next inference month waits for it.
3. The new candidates replace the champion only if they pass the quality gates AND beat the champion on the same
   OOT window; otherwise the champion stays and the best candidate becomes the shadow challenger.
4. Every outcome is in the registry and governance log for the model owner's monthly review. Automatic promotion
   is a deliberate choice for this pipeline; a regulated deployment would add a manual sign-off between steps 3 and 4
   (e.g. trigger with promotion disabled, then `--mode promote` after review).

## 7. Rollback SOP

1. Identify the target model in `model_registry.json` (previous champion or current challenger).
2. `python main.py --mode promote --modelname <model_version>.pkl --snapshotdate <effective month>`
3. The change is written to `role_history` (reason "manual override / rollback") and the governance log; the next DAG
   run scores with the restored model. Clear and re-run the affected DAG runs if past months must be re-scored.

## 8. Deployment options

| Option | How | Pros | Cons | Fit here |
|---|---|---|---|---|
| **Batch scoring (chosen)** | Airflow monthly DAG, model loaded from the model store, scores written to a gold table | matches monthly snapshot features and decision cadence; simple, auditable, reproducible, backfillable; Spark scales to full portfolios | not real-time | **yes**: features are monthly snapshots and the label is monthly |
| Real-time API | same artefact behind FastAPI/Flask in a container (Docker / Kubernetes), registry resolves the champion | instant decision at application time | needs online features with the same cleaning; latency/uptime SLAs; more ops | next step for loan origination |
| Streaming | Kafka/Spark Structured Streaming scoring events | near-real-time on event data | complex; clickstream features unusable today (100% missing outside the 2023-07..2024-12 label window) | not now |
| In-database / warehouse | export LR as SQL (WoE tables + coefficients) | no serving infra; very transparent | XGB cannot be exported as easily | possible for LR only |

Deployment hygiene for any option:
- the Docker image pins all library versions (`requirements.txt`);
- the artefact carries its own preprocessing and cleaning rules;
- the champion is always resolved from the registry, never hard-coded;
- scaling beyond the assignment means LocalExecutor/CeleryExecutor with Postgres in place of SequentialExecutor/SQLite.

## 9. Known limitations

- In `backtest` mode the champion is trained on labels up to 2024-12, so scoring and monitoring 2023-01 to 2024-06 applications is
  in-sample (their month-6 labels are in its train split); only 2024-07 to 2024-11 applications are out of sample
  (validation 2024-07, test 2024-08/09, OOT 2024-10/11). Performance monitoring of the in-sample months is a sanity check,
  not evidence of generalisation.
- The real data ends with applications in 2025-01 (their month-6 labels arrive by 2025-07, after the 2025-06-01
  training cut-off, so they are never used to train the first champion). New months from 2025-02 are simulated with
  `scripts/generate_synthetic_data.py`: labelled applications from the last year are resampled with their repayment
  history (so labels stay realistic), re-dated, and optionally drifted. It is a simulation: label behaviour is copied,
  not generated, so drift moves the inputs without changing true risk.
- Inference features are rebuilt in pandas from the saved parameters (`utils/feature_inference.py`), mirroring the
  Spark gold job. Any change to the gold job's rules must be made in both places; in backtest mode, comparing the two
  stores for the same loans is the consistency check.
- Bronze/silver for training are full-history rebuilds; at much larger volumes they should become incremental.
- 6-month label lag: performance monitoring always trails stability monitoring by 6 months.
- LR uses no class weighting, so its scores are calibrated PDs. XGBoost scores are compared on ranking and
  calibration in the dashboard.

## 10. Running it

1. `docker compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 generate_synthetic_data.py"`
   (new application months 2025-02 to 2026-05 in `data/incoming/`, gradual drift from 2025-09).
2. Open http://localhost:8080 and unpause both DAGs. Training runs once by itself; inference waits for it, then
   processes 2025-06 onward month by month (catchup), retraining automatically when the policy says so.
`dags/pipeline_config.py` switches between `production` (score only applications from the train date on, the
default) and `backtest` (replay every historical month with the champion).
