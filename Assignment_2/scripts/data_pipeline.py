"""
Thin command-line wrapper so Airflow can run the existing bronze / silver / gold code in utils/ as separate tasks.
Paths and arguments are the ones used in Gold_Processing-Copy1.ipynb.

    python data_pipeline.py --step dep_check --table lms_loan_daily --snapshotdate 2024-01-01
    python data_pipeline.py --step bronze    --table lms_loan_daily --snapshotdate 2024-01-01
    python data_pipeline.py --step silver    --table lms_loan_daily --snapshotdate 2024-01-01
    python data_pipeline.py --step gold_label_store                 --snapshotdate 2024-01-01
    python data_pipeline.py --step gold_feature_store --model_train_date 2025-06-01 --snapshotdate 2025-06-01

Inference (one application month, no labels needed):
    python data_pipeline.py --step bronze_month --table features_financials --snapshotdate 2025-07-01
    python data_pipeline.py --step silver_month --table features_financials --snapshotdate 2025-07-01
    python data_pipeline.py --step inference_features                      --snapshotdate 2025-07-01

The *_main tables cover the full history, so they are rebuilt only when missing or with --force
(Airflow Variable rebuild_data=true); the monthly gold label partition is (re)built for every run date.
The gold feature store also saves the parameters it fitted on TRAIN (feature_params_<train date>.pkl);
inference_features applies them to new months, so new loans get exactly the training-time treatment.
"""
import json
import pickle
import argparse
import logging
import os
import shutil
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", stream=sys.stdout)
log = logging.getLogger("data_pipeline")

# ---------------------------------------------------------------- config (from Gold_Processing-Copy1.ipynb)
RAW_DIRECTORY = "data/"
INCOMING_DIRECTORY = "data/incoming/"   # new monthly data lands here (e.g. generate_synthetic_data.py)
BRONZE_DIRECTORY = "datamart/bronze/"
SOURCES = {   # table -> (raw csv in data/, bronze csv read by the silver code)
    "lms_loan_daily": ("lms_loan_daily.csv", "lms_loan_daily/lms_loan_daily_main.csv"),
    "features_attributes": ("features_attributes.csv", "features_attributes/features_attributes_main.csv"),
    "features_financials": ("features_financials.csv", "features_financials/features_financials_main.csv"),
    "feature_clickstream": ("feature_clickstream.csv", "feature_clickstream/feature_clickstream_main.csv"),
}
SILVER_DIRECTORY = "datamart/silver/"
SILVER_DIRS = {
    "features_attributes": "datamart/silver/features_attributes/",
    "features_financials": "datamart/silver/features_financials/",
    "lms_loan_daily": "datamart/silver/lms_loan_daily/",
    "feature_clickstream": "datamart/silver/feature_clickstream/",
}
SILVER_MAIN = {
    "features_attributes": "features_attributes/silver_features_attributes_main.parquet",
    "features_financials": "features_financials/silver_features_financials_main.parquet",
    "lms_loan_daily": "lms_loan_daily/silver_loan_daily_main_.parquet",
    "feature_clickstream": "feature_clickstream/silver_clickstream_main_.parquet",
}
GOLD_LABEL_DIRECTORY = "datamart/gold/label_store/"
GOLD_LABEL_MAIN = "datamart/gold/label_store/gold_label_store_main_.parquet"
GOLD_FEATURE_DIRECTORY = "datamart/gold/feature_store/"
GOLD_FEATURE_MAIN = "datamart/gold/feature_store/gold_feature_store_main_cs.parquet"
DATA_START_DATE_STR, DATA_END_DATE_STR = "2023-01-01", "2024-12-01"
TRAIN_END_DATE, UNSEEN_START_DATE = "2024-09-01", "2024-10-01"
DEFAULT_MODEL_TRAIN_DATE = "2025-06-01"
GOLD_SPLIT = {"validation_period_months": 1, "test_period_months": 2, "oot_period_months": 2}
INFERENCE_FEATURE_DIRECTORY = "datamart/gold/inference_feature_store/"
INFERENCE_TABLES = ["features_attributes", "features_financials"]   # what the model needs at application
MODEL_REGISTRY = "model_bank/model_registry.json"
LABEL_DEF = {"dpd": 30, "mob": 6}


def get_spark():
    import pyspark
    spark = pyspark.sql.SparkSession.builder.appName("data_pipeline").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def exists(path):
    return os.path.exists(path)


# ---------------------------------------------------------------- steps
def dep_check(table, ds):
    """Source data must be available: the raw csv or an already-built bronze csv."""
    raw, bronze = SOURCES[table]
    found = [p for p in (RAW_DIRECTORY + raw, BRONZE_DIRECTORY + bronze) if exists(p)]
    if not found:
        raise FileNotFoundError(f"{table}: neither {RAW_DIRECTORY + raw} nor {BRONZE_DIRECTORY + bronze} exists")
    log.info("%s source available for %s: %s", table, ds, found)


def raw_files(table):
    files = [RAW_DIRECTORY + SOURCES[table][0], INCOMING_DIRECTORY + SOURCES[table][0]]
    return [f for f in files if exists(f)]


def read_raw(table):
    """Original history + any newly arrived rows, as text (the silver layer does the typing)."""
    import pandas as pd
    return pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in raw_files(table)],
                     ignore_index=True)


def bronze(table, ds, force):
    raw, target = SOURCES[table]
    target = BRONZE_DIRECTORY + target
    if exists(target) and not force:
        log.info("bronze %s exists (%s) - skipped", table, target)
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    files = raw_files(table)
    if len(files) == 1:
        shutil.copyfile(files[0], target)
    else:
        read_raw(table).to_csv(target, index=False)
    log.info("bronze %s: %s -> %s", table, files, target)


def silver(table, ds, force):
    """One bronze table -> its own silver table (each source is a separate Airflow task)."""
    target = SILVER_DIRECTORY + SILVER_MAIN[table]
    if exists(target) and not force:
        log.info("silver %s exists (%s) - skipped", table, target)
        return
    import utils.data_processing_silver_tables as silver_tables
    build = {
        "features_attributes": silver_tables.process_silver_attributes_main,
        "features_financials": silver_tables.process_silver_financials_main,
        "lms_loan_daily": silver_tables.process_silver_loans_main,
        "feature_clickstream": silver_tables.process_silver_clickstream_main,
    }[table]
    os.makedirs(SILVER_DIRS[table], exist_ok=True)
    spark = get_spark()
    try:
        build(BRONZE_DIRECTORY, SILVER_DIRS[table], spark, SOURCES[table][1])
        log.info("silver %s -> %s", table, target)
    finally:
        spark.stop()


def gold_label_store(ds, force):
    monthly_silver = SILVER_DIRS["lms_loan_daily"] + "silver_loan_daily_" + ds.replace("-", "_") + ".parquet"
    build_monthly = exists(monthly_silver)
    build_main = force or not exists(GOLD_LABEL_MAIN)
    if not build_monthly:
        log.warning("no monthly silver loan partition for %s (%s) - monthly gold label skipped", ds, monthly_silver)
    if not build_main:
        log.info("gold label store main exists - skipped: %s", GOLD_LABEL_MAIN)
    if not (build_monthly or build_main):
        return
    import utils.data_processing_lms_gold_table as gold
    os.makedirs(GOLD_LABEL_DIRECTORY, exist_ok=True)
    spark = get_spark()
    try:
        if build_monthly:                                 # monthly gold label partition for this run date
            gold.process_labels_gold_table(ds, SILVER_DIRS["lms_loan_daily"], GOLD_LABEL_DIRECTORY, spark,
                                           dpd=LABEL_DEF["dpd"], mob=LABEL_DEF["mob"])
        if build_main:                                    # full-history label store read by the model
            gold.process_labels_gold_table_main(SILVER_DIRS["lms_loan_daily"], GOLD_LABEL_DIRECTORY, spark,
                                                dpd=LABEL_DEF["dpd"], mob=LABEL_DEF["mob"])
    finally:
        spark.stop()


def params_path(model_train_date):
    return GOLD_FEATURE_DIRECTORY + "feature_params_" + model_train_date.replace("-", "_") + ".pkl"


def gold_feature_store(ds, force, model_train_date=DEFAULT_MODEL_TRAIN_DATE):
    if exists(GOLD_FEATURE_MAIN) and exists(params_path(model_train_date)) and not force:
        log.info("gold feature store + parameters for %s exist - skipped", model_train_date)
        return
    import utils.data_processing_lms_gold_table as gold
    os.makedirs(GOLD_FEATURE_DIRECTORY, exist_ok=True)
    spark = get_spark()
    try:
        gold.process_features_gold_table_main(
            SILVER_DIRECTORY, spark, SILVER_MAIN["features_attributes"], SILVER_MAIN["features_financials"],
            GOLD_LABEL_MAIN, SILVER_MAIN["feature_clickstream"], GOLD_FEATURE_DIRECTORY,
            TRAIN_END_DATE, UNSEEN_START_DATE, DATA_START_DATE_STR, DATA_END_DATE_STR,
            model_train_date_str=model_train_date, **GOLD_SPLIT)
    finally:
        spark.stop()


# ---------------------------------------------------------------- inference: one application month
def month_tag(ds):
    return ds.replace("-", "_")


def bronze_month(table, ds, force):
    """Land only this month's rows of the raw source (new data arrives month by month)."""
    import pandas as pd
    target = BRONZE_DIRECTORY + f"{table}/{table}_{month_tag(ds)}.csv"
    if exists(target) and not force:
        log.info("bronze %s %s exists - skipped", table, ds)
        return
    if not raw_files(table):
        raise FileNotFoundError(f"no raw file for {table}")
    df = read_raw(table)
    month = df[pd.to_datetime(df["snapshot_date"], errors="coerce") == pd.Timestamp(ds)]
    if month.empty:
        log.warning("no %s rows with snapshot_date = %s in %s - nothing new this month", table, ds, raw_files(table))
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    month.to_csv(target, index=False)
    log.info("bronze %s %s: %d rows -> %s", table, ds, len(month), target)


def silver_month(table, ds, force):
    import utils.data_processing_silver_tables as silver_tables
    source = f"{table}/{table}_{month_tag(ds)}.csv"
    target = SILVER_DIRS[table] + f"silver_{table}_{month_tag(ds)}.parquet"
    if not exists(BRONZE_DIRECTORY + source):
        log.warning("no bronze %s partition for %s - skipped", table, ds)
        return
    if exists(target) and not force:
        log.info("silver %s %s exists - skipped", table, ds)
        return
    build = {"features_attributes": silver_tables.process_silver_attributes_main,
             "features_financials": silver_tables.process_silver_financials_main}[table]
    os.makedirs(SILVER_DIRS[table], exist_ok=True)
    spark = get_spark()
    try:
        build(BRONZE_DIRECTORY, SILVER_DIRS[table], spark, source)   # writes silver_<table>_<yyyy_mm_dd>.parquet
    finally:
        spark.stop()


def champion_train_date():
    """Feature parameters must match the champion's training run (its gold build)."""
    if exists(MODEL_REGISTRY):
        with open(MODEL_REGISTRY) as f:
            reg = json.load(f)
        if reg.get("champion"):
            return reg["models"][reg["champion"]]["model_train_date"]
    return DEFAULT_MODEL_TRAIN_DATE


def inference_features(ds, force):
    import pandas as pd
    from utils.feature_inference import build_inference_features
    target = INFERENCE_FEATURE_DIRECTORY + f"inference_features_{month_tag(ds)}.parquet"
    if exists(target) and not force:
        log.info("inference features for %s exist - skipped", ds)
        return
    parts = {t: SILVER_DIRS[t] + f"silver_{t}_{month_tag(ds)}.parquet" for t in INFERENCE_TABLES}
    missing = [p for p in parts.values() if not exists(p)]
    if missing:
        log.warning("no new applications for %s (missing %s) - nothing to score", ds, missing)
        return
    train_date = champion_train_date()
    with open(params_path(train_date), "rb") as f:
        params = pickle.load(f)
    fin = pd.read_parquet(parts["features_financials"])
    attr = pd.read_parquet(parts["features_attributes"])
    out = build_inference_features(fin, attr, params, ds)
    os.makedirs(INFERENCE_FEATURE_DIRECTORY, exist_ok=True)
    out.to_parquet(target, index=False)
    log.info("inference features %s: %d applications, parameters from training run %s -> %s",
             ds, len(out), train_date, target)


def main():
    p = argparse.ArgumentParser(description="bronze / silver / gold steps for Airflow")
    p.add_argument("--step", required=True,
                   choices=["dep_check", "bronze", "silver", "gold_label_store", "gold_feature_store",
                            "bronze_month", "silver_month", "inference_features"])
    p.add_argument("--snapshotdate", required=True, help="the DAG run's {{ ds }}")
    p.add_argument("--table", choices=list(SOURCES), help="dep_check / bronze / silver: source table")
    p.add_argument("--force", default="false", help="true = rebuild full-history tables")
    p.add_argument("--model_train_date", default=DEFAULT_MODEL_TRAIN_DATE,
                   help="gold_feature_store: train date that sets the train/validation/test/OOT windows")
    a = p.parse_args()
    force = str(a.force).lower() == "true"
    log.info("---starting job--- step=%s ds=%s table=%s force=%s", a.step, a.snapshotdate, a.table, force)
    if a.step in ("dep_check", "bronze", "silver", "bronze_month", "silver_month") and not a.table:
        p.error("--table is required for dep_check / bronze / silver / bronze_month / silver_month")
    {"dep_check": lambda: dep_check(a.table, a.snapshotdate),
     "bronze": lambda: bronze(a.table, a.snapshotdate, force),
     "silver": lambda: silver(a.table, a.snapshotdate, force),
     "gold_label_store": lambda: gold_label_store(a.snapshotdate, force),
     "gold_feature_store": lambda: gold_feature_store(a.snapshotdate, force, a.model_train_date),
     "bronze_month": lambda: bronze_month(a.table, a.snapshotdate, force),
     "silver_month": lambda: silver_month(a.table, a.snapshotdate, force),
     "inference_features": lambda: inference_features(a.snapshotdate, force)}[a.step]()
    log.info("---completed job---")


if __name__ == "__main__":
    main()
