# Credit default model: ML pipeline on Airflow

Predicts, at application, the probability that a new loan will be **30+ days past due at month-on-book 6**
(`30dpd_6mob`). Two Airflow pipelines run in Docker:

- **`credit_model_training`** builds the bronze → silver → gold stores from all history, trains Logistic Regression
  and XGBoost with the development notebook's train / validation / test / OOT windows, and registers the champion.
- **`credit_model_inference`** runs monthly: builds features for that month's new applications, scores them with
  the champion (and the challenger in shadow), monitors performance and stability, draws the dashboard and decides
  whether the model should be refreshed. A "train" decision triggers the training pipeline automatically.

Policies (refresh triggers, promotion gates, monitoring thresholds, SOPs, deployment options): see
[`GOVERNANCE.md`](GOVERNANCE.md).

---

## Repository layout

```
.
├── dags/
│   ├── credit_model_training.py     training pipeline (runs once on unpause, then when triggered)
│   ├── credit_model_inference.py    monthly inference + monitoring + refresh decision
│   └── pipeline_config.py           shared settings: train date, production / backtest mode
├── scripts/                         mounted at /opt/airflow/scripts; every task runs from here
│   ├── data/                        raw source CSVs (never modified)
│   │   └── incoming/                simulated new months (aftere docker container is running, docker compose   exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 generate_synthetic_data.py")
│   │   └── feature_clickstream.csv               
│   │   └── feature_attributes.csv                
│   │   └── feature_financials.csv
│   │   └── lms_loan_daily.csv                
│   ├── utils/
│   │   ├── data_processing_bronze_table.py      (used by the notebooks)
│   │   ├── data_processing_silver_tables.py     one silver function per source
│   │   ├── data_processing_lms_gold_table.py    gold label store + gold feature store (+ saved parameters)
│   │   └── feature_inference.py                 features for new applications from saved parameters
│   ├── data_pipeline.py             CLI for every bronze / silver / gold step
│   ├── main.py                      CLI for train / select / inference / monitor / report / decide / promote
│   ├── generate_synthetic_data.py   simulates new monthly applications after the real data
│   ├── datamart/                    bronze/, silver/, gold/ (written by the pipelines)
│   ├── model_bank/                  model artefacts (.pkl) + model_registry.json
│   └── logs/                        one log file per main.py run
├── Dockerfile                       apache/airflow:2.6.1-python3.10 + Java 17 (for Spark)
├── docker-compose.yaml              airflow-init, webserver (port 8080), scheduler
├── requirements.txt
├── GOVERNANCE.md
└── README.md
```

## Data timeline

Every loan has two dates six months apart: the **application month** (when it is scored) and the **label month**
(month-on-book 6, when its outcome is known). The development model uses all labelled history, split by label month
counting back from the train date **2025-06-01**:

| Split | Label months | Same loans by application month | Used for |
|---|---|---|---|
| Train | up to 2024-12 | 2023-01 to 2024-06 | fitting, hyper-parameter CV |
| Validation | 2025-01 | 2024-07 | XGBoost early stopping, decision threshold |
| Test | 2025-02 to 2025-03 | 2024-08 to 2024-09 | choosing the champion |
| OOT | 2025-04 to 2025-05 | 2024-10 to 2024-11 | reported only, never used for a decision |

Real applications continue to 2025-01; their labels arrive after the cut-off. New months from 2025-02 are simulated
(next section), so the inference pipeline has applications the model has never seen.

## Running it

**1. Start Airflow** (from the folder with `docker-compose.yaml`):
```bash
docker compose up -d --build
docker compose exec airflow-scheduler airflow dags list-import-errors     # expect "No data found"
```
Open http://localhost:8080 (admin / admin).

**2. Simulate new data** (writes `scripts/data/incoming/`; re-running replaces it):
```bash
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 generate_synthetic_data.py"
```
Defaults: 16 months from the month after the last real application, 500 applications per month, resampled from the
last year of labelled customers with their repayment history, gradual drift from 2025-09. Options:
`--drift none|gradual|strong`, `--months`, `--per_month`, `--drift_start`, `--seed`.

**3. Unpause both DAGs** in the UI: `credit_model_training` first, then `credit_model_inference`.
- Training starts by itself (`@once`). Its last task, `model_training.select_champion`, writes
  `scripts/model_bank/model_registry.json`.
- Inference runs one month at a time from 2025-06 (catchup). Each month's `wait_for_champion` holds until a champion
  exists and no training run is in progress, so a month is always scored by the model in force for it.
- When `governance.refresh_decision` says "train" (6-month schedule or a monitoring alert), `retrain_recommended`
  triggers a training run as of that month; the new candidate replaces the champion only if it passes the gates and
  beats it on the same OOT window.

Optional speed-up: **Admin → Variables**, add `xgb_n_iter` = `10` (XGBoost random-search iterations, default 30).

**Starting over:** pause both DAGs, move `scripts/model_bank/model_registry.json` and
`scripts/datamart/gold/{model_*,inference_feature_store}` out, delete the DAG runs in the UI, unpause again.

## Configuration

| Setting | Where | Default | Effect |
|---|---|---|---|
| `MODEL_TRAIN_DATE` | `dags/pipeline_config.py` | `2025-06-01` | train date of the first champion (split windows count back from it) |
| `DEPLOYMENT_MODE` | `dags/pipeline_config.py` | `production` | `production`: score applications from the train date on (simulated new data). `backtest`: champion effective from 2023-01 and every historical month replayed (in-sample before 2024-07) |
| `xgb_n_iter` | Airflow Variable | `30` | XGBoost hyper-parameter search iterations |
| `rebuild_data` | Airflow Variable | `false` | `true` forces the inference pipeline to rebuild monthly partitions that already exist |
| training run params | Trigger DAG w/ config | `model_train_date`, `effective_from` | retrain as of a date; new champion live from `effective_from` |

## Outputs

| What | Where (under `scripts/`) |
|---|---|
| Model artefacts, registry | `model_bank/credit_model_<algo>_<train_date>.pkl`, `model_bank/model_registry.json` |
| Gold label / feature stores | `datamart/gold/label_store/`, `datamart/gold/feature_store/` (+ `feature_params_<date>.pkl`) |
| Features of new applications | `datamart/gold/inference_feature_store/inference_features_<yyyy_mm_dd>.parquet` |
| Predictions | `datamart/gold/model_predictions/<model_version>/…_predictions_<yyyy_mm_dd>.parquet` |
| Monitoring | `datamart/gold/model_monitoring/<model_version>/`, `datamart/gold/model_feature_drift/<model_version>/` |
| Dashboard (performance + stability over time) | `datamart/gold/model_reports/model_dashboard_asof_<ds>.png` |
| Governance log (decisions, training outcomes) | `datamart/gold/model_governance/` |
| Run logs | Airflow UI task logs; `logs/main_<mode>_<timestamp>.log` |

Performance monitoring of an application month appears 6 months after it is scored (labels must mature); stability
(score and feature PSI) is available immediately.

## Running steps by hand

Every task is a plain command, so any step can be re-run or debugged outside Airflow:
```bash
docker compose exec airflow-scheduler bash
cd /opt/airflow/scripts
python3 data_pipeline.py --step inference_features --snapshotdate 2025-07-01
python3 main.py --mode inference --role champion --snapshotdate 2025-07-01
python3 main.py --mode monitor --role champion --snapshotdate 2026-01-01
python3 main.py --mode report --snapshotdate 2026-01-01
python3 main.py --mode promote --modelname credit_model_logreg_2025_06_01.pkl --snapshotdate 2025-09-01   # rollback
python3 main.py                     # standalone one-shot run without Airflow (train -> backfill -> monitor)
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `apt.postgresql.org … bullseye-pgdg … 404` during build | dead apt repo in the base image; the Dockerfile removes `pgdg.list` before `apt-get update` |
| `openjdk-17 … 404 Not Found` during build | Debian 11 security archive moved; the Dockerfile copies Java 17 from `eclipse-temurin:17-jdk` instead |
| `No module named 'airflow.sdk'` / `xcom_backend` error | old Airflow 3 config in the `airflow_data` volume: `docker compose down -v`, then `up -d` |
| `ModuleNotFoundError: utils…` in a task | `utils/` and `data/` must sit inside `scripts/` (only `dags/` and `scripts/` are mounted) |
| `wait_for_champion` stays "up for reschedule" | training has not finished (or not been unpaused); it continues by itself afterwards |
| `nothing to score` in inference logs | no application rows for that month: run `generate_synthetic_data.py`, or the month is after the simulated data |
| Elasticsearch / Snowflake provider warnings in the webserver log | unused Airflow providers incompatible with NumPy 2 / pyarrow 25; harmless |
