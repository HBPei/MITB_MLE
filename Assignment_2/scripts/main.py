"""
Credit default model (30dpd @ 6mob): processing steps called by the Airflow DAG (dags/dag.py).

Each mode is one DAG task and takes the run's logical date (--snapshotdate "{{ ds }}"). Airflow does the
orchestration, scheduling and backfill; this script only does the work for one month.

    python main.py --mode bootstrap --snapshotdate 2023-01-01          # once: notebook model (2025-06-01) live from 2023-01
    python main.py --mode decide    --snapshotdate 2024-06-01          # refresh policy -> last line DECISION={...}
    python main.py --mode train     --snapshotdate 2024-06-01 --algo logreg --split_source dates   # model_1_automl
    python main.py --mode train     --snapshotdate 2024-06-01 --algo xgb    --split_source dates   # model_2_automl
    python main.py --mode select    --snapshotdate 2024-06-01 --split_source dates   # promotion gate -> registry
    python main.py --mode select    --snapshotdate 2025-06-01 --effective_from 2025-06-01   # live from that month
    python main.py --mode inference --snapshotdate 2024-07-01 --role champion        # / --role challenger
    python main.py --mode monitor   --snapshotdate 2024-12-01 --role champion        # / --role challenger
    python main.py --mode report    --snapshotdate 2025-06-01                        # dashboard PNG + CSV
    python main.py --mode promote   --snapshotdate 2024-09-01 --modelname credit_model_logreg_2024_06_01.pkl

    python main.py                  # standalone one-shot run without Airflow (train -> select -> backfill -> monitor)

Policies (refresh, promotion gates, monitoring thresholds): GOVERNANCE / MONITORING below and GOVERNANCE.md.
Feature / preprocessing choices come from the EDA in Gold_Processing-Copy1.
"""
import argparse
import glob
import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import sklearn
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================================================================
# CONFIG
# =====================================================================================================================
PATHS = {
    "feature_store": "datamart/gold/feature_store/gold_feature_store_main_cs.parquet",
    "inference_features": "datamart/gold/inference_feature_store/",   # one partition per new application month
    "label_store": "datamart/gold/label_store/gold_label_store_main_.parquet",
    "model_bank": "model_bank/",
    "predictions_root": "datamart/gold/model_predictions/",
    "monitoring_root": "datamart/gold/model_monitoring/",
    "feature_drift_root": "datamart/gold/model_feature_drift/",
    "registry": "model_bank/model_registry.json",
    "governance_root": "datamart/gold/model_governance/",
    "reports_root": "datamart/gold/model_reports/",
    "logs": "logs/",
}
PIPELINE_DEFAULTS = {"model_train_date": "2025-06-01", "start_date": "2023-01-01", "end_date": "2024-12-01"}
SPLIT_CONFIG = {"validation_period_months": 1, "test_period_months": 2, "oot_period_months": 2,
                "data_start_date_str": "2023-01-01", "label_definition": "30dpd_6mob"}
SPLITS = ["train", "validation", "test", "oot"]
TARGET = "label"
SCORING_DATE_COL = "financials_snapshot_date"      # features observed at loan application month
RANDOM_STATE = 42

# ---- champion selection: highest Gini on TEST (OOT stays an untouched report-only period) ----
CHAMPION_METRIC = "gini_test"

# ---- governance: when to (re)train and when a new model may replace the champion (see GOVERNANCE.md) ----
GOVERNANCE = {
    "min_label_months": 11,            # cold start: 6 train + 1 validation + 2 test + 2 OOT labelled months
    "refresh_every_months": 6,         # scheduled refresh
    "min_months_between_training": 2,  # cooldown so a triggered refresh cannot fire every month
    "min_gini_gate": 0.30,             # quality gate on candidate TEST and OOT Gini
    "max_test_oot_gini_gap": 0.10,     # stability gate: |Gini_test - Gini_oot|
    "promotion_min_uplift": 0.0,       # candidate OOT Gini must be >= champion OOT Gini (same window) + uplift
}

# ---- monitoring thresholds ----
MONITORING = {
    "psi_watch": 0.10, "psi_alert": 0.25,          # score / feature PSI vs train baseline
    "auc_drop_watch": 0.03, "auc_drop_alert": 0.05,  # AUC below the champion's TEST AUC
    "min_labelled_rows": 100,                      # below this, performance is not assessed
}

# ---- row-level cleaning (same rules as the notebook; plain data so it can be pickled into the artefact) ----
CLEANING_CONFIG = {
    "non_negative_cols": ["Num_of_Delayed_Payment", "Num_Credit_Inquiries",
                          "Delayed_Payments_per_Card", "Num_of_Loan_Cleaned"],
    "invalid_value_rules": {                        # {column: (op, threshold)} -> NaN, BEFORE quantile capping
        "Interest_Rate": (">", 50),                 # % p.a.; source has values in the thousands
        "Num_Credit_Inquiries": (">", 30),
        "Num_of_Delayed_Payment": (">", 50),
        "Monthly_Balance": ("abs>", 1e9),           # '-333333333...' placeholder
        "Amount_invested_monthly": ("==", 10000),   # '__10000__' placeholder
    },
    "redundant_cols": ["Age", "Num_Bank_Accounts", "Num_Credit_Card",
                       "Age_Bin", "Bank_Accounts_Imputed_Bin", "Credit_Cards_Imputed_Bin"],
}

# =====================================================================================================================
# FEATURE SELECTION  (evidence: Gold_Processing-Copy1 EDA / model outputs, train 2023-07..2024-12)
# ---------------------------------------------------------------------------------------------------------------------
# Excluded up-front
#   fe_* clickstream (10)            100% missing in validation/test/OOT -> unusable at scoring time
#   Credit_Utilization_Ratio, Occupation, Loan_Not_Specified           IV < 0.02
#   Outstanding_Debt, Interest_Rate  |rho| > 0.8 with InterestRate_x_Debt (IV 0.93, kept)
#   Monthly_Inhand_Salary            |rho| > 0.8 with Annual_Income
#   Bank_Accounts_imputed            |rho| > 0.8 with Num_Fin_Pdts
#   Loan_* type flags (8)            XGB mean|SHAP| = 0; each ~= a slice of Num_of_Loan_Cleaned (collinear)
#   Monthly_Balance, Changed_Credit_Limit, Total_EMI_per_month, Delayed_Payments_per_Card,
#   Num_of_Loan_Cleaned, Num_of_Delayed_Payment
#                                    XGB mean|SHAP| <= 0.0035 AND wrong-sign in LR (conditional effect flips);
#                                    Total_EMI / Delayed_Payments_per_Card also non-monotone by decile
# LR only drops
#   Num_Fin_Pdts                     VIF 10.4 (linear combination of account counts)
#   Debt_per_Credit_Account, Annual_Income   wrong coefficient sign in LR, but kept in XGB where the monotone
#                                    constraint fixes the direction (SHAP 0.037 / 0.007)
# Categoricals
#   LR  : one-hot gave wrong signs (Credit_Mix_Good vs Bad +0.60, Payment_of_Min_Amount_Yes < 0) because levels
#         overlap with the numerics. -> Weight-of-Evidence encoding: one column per variable, higher = riskier,
#         one coefficient per variable whose sign must be positive.
#   XGB : one-hot + per-level monotone constraints (verified: 0 violations, OOT AUC 0.786).
# =====================================================================================================================
NUMERIC_CORE = ["InterestRate_x_Debt", "Delay_from_due_date", "Num_Credit_Inquiries", "Credit_History_Age_Years",
                "Credit_Cards_imputed", "Amount_invested_monthly", "Age_imputed"]
CATEGORICAL = ["Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour_Imputed"]

LR_FEATURES = NUMERIC_CORE + CATEGORICAL
XGB_FEATURES = NUMERIC_CORE + ["Debt_per_Credit_Account", "Annual_Income", "Num_Fin_Pdts"] + CATEGORICAL

# +1: higher value -> higher default risk | -1: lower risk. One-hot levels: "<feature>_<level>".
EXPECTED_DIRECTION = {
    "InterestRate_x_Debt": +1, "Delay_from_due_date": +1, "Num_Credit_Inquiries": +1,
    "Credit_History_Age_Years": -1, "Credit_Cards_imputed": +1, "Amount_invested_monthly": -1,
    "Age_imputed": -1, "Debt_per_Credit_Account": +1, "Annual_Income": -1, "Num_Fin_Pdts": +1,
    "Credit_Mix_Bad": +1, "Credit_Mix_Good": -1,
    "Payment_of_Min_Amount_Yes": +1, "Payment_of_Min_Amount_No": -1,
}

# ---- hyper-parameter spaces (time-ordered CV inside train) ----
# LR: class_weight fixed to None - 'balanced' gained 0.0002 CV AUC but makes scores non-probabilities.
# solver 'saga' does not penalise the intercept ('liblinear' does: with small C it pulls every PD toward 0.5).
LR_C_GRID = np.logspace(-3, 1, 9)
XGB_PARAM_DIST = {
    "model__n_estimators": [200, 300, 500, 800],
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_child_weight": [1, 5, 10, 20],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
    "model__reg_lambda": [1, 5, 10],
    "model__reg_alpha": [0, 0.1, 1],
    "model__gamma": [0, 0.5, 1],
}

log = logging.getLogger("credit_model")


# =====================================================================================================================
# LOGGING / SPARK
# =====================================================================================================================
def setup_logging(mode):
    os.makedirs(PATHS["logs"], exist_ok=True)
    logfile = os.path.join(PATHS["logs"], f"main_{mode}_{datetime.now():%Y%m%d_%H%M%S}.log")
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("log file: %s", logfile)
    return logfile


def log_table(title, frame, digits=4):
    log.info("%s\n%s", title, frame.round(digits).to_string() if hasattr(frame, "round") else frame)


def get_spark():
    spark = pyspark.sql.SparkSession.builder.appName("credit_model").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# =====================================================================================================================
# CLEANING
# =====================================================================================================================
_OPS = {">": lambda s, v: s > v, ">=": lambda s, v: s >= v, "<": lambda s, v: s < v,
        "<=": lambda s, v: s <= v, "==": lambda s, v: s == v, "abs>": lambda s, v: s.abs() > v}


def collapse_dummies(d, prefix, new_col):
    cols = [c for c in d.columns if c.startswith(prefix)]
    if not cols:
        return d
    vals = d[cols].to_numpy()
    levels = np.array([c[len(prefix):] for c in cols], dtype=object)
    d[new_col] = pd.Series(np.where(vals.sum(axis=1) == 0, None, levels[vals.argmax(axis=1)]),
                           index=d.index).fillna("Unknown")
    return d.drop(columns=cols)


def basic_clean(d, non_negative_cols, invalid_value_rules, redundant_cols, verbose=True):
    d = d.copy()
    d = d.loc[:, ~d.columns.duplicated()]
    d["Occupation"] = d["Occupation"].replace("", np.nan).fillna("Unknown")
    for c in non_negative_cols:
        if c in d.columns:
            d[c] = d[c].clip(lower=0)
    for c, (op, value) in invalid_value_rules.items():
        if c in d.columns:
            d[c] = d[c].astype(float)
            mask = _OPS[op](d[c], value)
            if verbose:
                log.info("  invalid -> NaN  %-25s %4s %-10s %5d rows (%.2f%%)", c, op, value, mask.sum(), 100 * mask.mean())
            d[c] = d[c].mask(mask)
    d["Num_Fin_Pdts"] = d["Bank_Accounts_imputed"] + d["Credit_Cards_imputed"] + d["Num_of_Loan_Cleaned"]
    d["InterestRate_x_Debt"] = d["Interest_Rate"] * d["Outstanding_Debt"]
    d = collapse_dummies(d, "Credit_Mix_", "Credit_Mix")
    d = collapse_dummies(d, "Payment_of_Min_Amount_", "Payment_of_Min_Amount")
    for c in CATEGORICAL:
        if c in d.columns:
            d[c] = d[c].fillna("Unknown").astype(str)
    return d.drop(columns=[c for c in redundant_cols if c in d.columns])


# =====================================================================================================================
# CUSTOM TRANSFORMERS (pickled inside the artefacts; referenced as main.<Class>)
# =====================================================================================================================
class QuantileCapper(BaseEstimator, TransformerMixin):
    """Winsorise each column to its TRAIN [lower_q, upper_q] quantiles."""

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_, self.upper_ = X.quantile(self.lower_q).to_numpy(), X.quantile(self.upper_q).to_numpy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


class SkewLogTransformer(BaseEstimator, TransformerMixin):
    """Signed log1p on columns whose TRAIN |skew| > threshold (monotone, so coefficient signs are preserved)."""

    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        self.log_mask_ = (pd.DataFrame(np.asarray(X, dtype=float)).skew().abs() > self.skew_threshold).to_numpy()
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.log_mask_] = np.sign(X[:, self.log_mask_]) * np.log1p(np.abs(X[:, self.log_mask_]))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Category -> TRAIN Weight of Evidence ln(%bad/%good). Higher = riskier; unseen level -> 0 (neutral)."""

    def __init__(self, smoothing=0.5):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X).astype(str)
        y = pd.Series(np.asarray(y), index=X.index)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.woe_ = {}
        for c in X.columns:
            tab = pd.crosstab(X[c], y).reindex(columns=[0, 1], fill_value=0)
            bad = (tab[1] + self.smoothing) / (tab[1].sum() + self.smoothing)
            good = (tab[0] + self.smoothing) / (tab[0].sum() + self.smoothing)
            self.woe_[c] = np.log(bad / good).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).astype(str)
        return np.column_stack([X[c].map(self.woe_[c]).fillna(0.0).to_numpy(dtype=float)
                                for c in self.feature_names_in_])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_woe" for c in self.feature_names_in_], dtype=object)


def build_lr_preprocessor(num, cat):
    numeric = Pipeline([("cap", QuantileCapper(0.01, 0.99)),
                        ("log", SkewLogTransformer(1.0)),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("woe", WoEEncoder(0.5)), ("scale", StandardScaler())])
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)], verbose_feature_names_out=False)


def build_xgb_preprocessor(num, cat, categories):
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                            ("ohe", OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", QuantileCapper(0.01, 0.99), num), ("cat", categorical, cat)],
                             verbose_feature_names_out=False).set_output(transform="pandas")


# =====================================================================================================================
# STATS HELPERS
# =====================================================================================================================
def bin_edges(s, n_bins=10):
    e = np.unique(np.nanquantile(s.dropna(), np.linspace(0, 1, n_bins + 1)))
    if len(e) < 2:
        return None
    e[0], e[-1] = -np.inf, np.inf
    return e


def assign_bins(s, edges, is_cat):
    if is_cat or edges is None:
        return s.astype(str)
    b = pd.cut(s, bins=edges, include_lowest=True, duplicates="drop").astype(str)
    return b.where(s.notna(), "__MISSING__")


def information_value(bins, y, eps=0.5):
    tab = pd.crosstab(bins, y).reindex(columns=[0, 1], fill_value=0)
    good, bad = (tab[0] + eps) / (tab[0].sum() + eps), (tab[1] + eps) / (tab[1].sum() + eps)
    return float(((bad - good) * np.log(bad / good)).sum())


def psi(expected_bins, actual_bins, eps=1e-4):
    e, a = expected_bins.value_counts(normalize=True), actual_bins.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx, fill_value=0) + eps, a.reindex(idx, fill_value=0) + eps
    return float(((a - e) * np.log(a / e)).sum())


def vif_table(X):
    Z = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).to_numpy()
    out = {}
    for j, c in enumerate(X.columns):
        A = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
        coef, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        r2 = 1 - ((Z[:, j] - A @ coef) ** 2).sum() / max((Z[:, j] ** 2).sum(), 1e-12)
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def monthly_expanding_splits(dates, n_splits=4, val_months=2):
    months = pd.Series(pd.to_datetime(dates)).dt.to_period("M").to_numpy()
    uniq = np.sort(pd.unique(months))
    if len(uniq) < 3 + n_splits * val_months:          # short history (early refreshes): 1-month folds
        val_months, n_splits = 1, max(1, min(n_splits, len(uniq) - 3))
    splits = []
    for i in range(n_splits, 0, -1):
        cut = len(uniq) - i * val_months
        if cut < 3:
            continue
        tr, va = np.where(np.isin(months, uniq[:cut]))[0], np.where(np.isin(months, uniq[cut:cut + val_months]))[0]
        splits.append((tr, va))
        log.info("  CV fold %d: fit %s..%s (%d rows) -> score %s..%s (%d rows)", len(splits), uniq[0],
                 uniq[cut - 1], len(tr), uniq[cut], uniq[min(cut + val_months, len(uniq)) - 1], len(va))
    return splits


# =====================================================================================================================
# TRAINING
# =====================================================================================================================
def build_split_config(model_train_date_str):
    c = dict(SPLIT_CONFIG)
    c["model_train_date_str"] = model_train_date_str
    c["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    c["oot_end_date"] = c["model_train_date"] - timedelta(days=1)
    c["oot_start_date"] = c["model_train_date"] - relativedelta(months=c["oot_period_months"])
    c["test_end_date"] = c["oot_start_date"] - timedelta(days=1)
    c["test_start_date"] = c["test_end_date"] - relativedelta(months=c["test_period_months"]) + timedelta(days=1)
    c["validation_end_date"] = c["test_start_date"] - timedelta(days=1)
    c["validation_start_date"] = (c["validation_end_date"] - relativedelta(months=c["validation_period_months"])
                                  + timedelta(days=1))
    c["train_start_date"] = datetime.strptime(c["data_start_date_str"], "%Y-%m-%d")
    c["train_end_date"] = c["validation_start_date"] - timedelta(days=1)
    return c


def load_training_data(spark, config, split_source="store"):
    fs = spark.read.parquet(PATHS["feature_store"])
    labels = (spark.read.parquet(PATHS["label_store"])
              .select("loan_id", F.col("snapshot_date").alias("labels_snapshot_date"), TARGET))
    log.info("feature store rows: %d | label store rows: %d", fs.count(), labels.count())
    data = fs.join(labels, on=["loan_id", "labels_snapshot_date"], how="left").toPandas()

    n_missing = int(data[TARGET].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} feature rows have no label in the label store")
    data[TARGET] = data[TARGET].astype(int)
    for c in ["labels_snapshot_date", "attributes_snapshot_date", "financials_snapshot_date"]:
        if c in data.columns:
            data[c] = pd.to_datetime(data[c])

    if split_source == "dates":
        # as-of training (Airflow backfill): split by label date relative to this train date; labels on/after the
        # train date do not exist yet and are dropped.
        new_split = pd.Series(None, index=data.index, dtype=object)
        for s in SPLITS:
            new_split[data.labels_snapshot_date.between(config[f"{s}_start_date"], config[f"{s}_end_date"])] = s
        n_future = int((data.labels_snapshot_date > config["oot_end_date"]).sum())
        n_moved = int((new_split.notna() & (new_split != data.data_split)).sum())
        data = data.assign(data_split=new_split)[new_split.notna()].copy()
        log.info("date-based split as of %s: %d rows with labels on/after the train date excluded",
                 config["model_train_date_str"], n_future)
        if n_moved:
            log.warning("%d rows sit in a different split than the gold store's data_split: the gold job's "
                        "imputation parameters were fitted on its own train window (mild look-ahead in imputation)",
                        n_moved)
    # with split_source="store" the gold job fitted its imputations on ITS train split -> splits must be identical
    for s in SPLITS:
        lo, hi = config[f"{s}_start_date"], config[f"{s}_end_date"]
        d = data.loc[data.data_split == s, "labels_snapshot_date"]
        if d.empty or not d.between(lo, hi).all():
            raise ValueError(f"split '{s}' has no rows / does not match {lo.date()}..{hi.date()}; with "
                             f"--split_source store re-run process_features_gold_table_main with "
                             f"model_train_date_str={config['model_train_date_str']}")
        log.info("  %-10s %s -> %s | rows %5d | default rate %.4f", s, lo.date(), hi.date(), len(d),
                 data.loc[data.data_split == s, TARGET].mean())
    for c in ["attributes_snapshot_date", "financials_snapshot_date"]:
        n_future = int((data[c] > data["labels_snapshot_date"]).sum())
        (log.warning if n_future else log.info)("point-in-time check %s after label date: %d rows", c, n_future)
    return data


def feature_checks(df, features):
    """Re-validate the curated features on the current data: availability, IV, stability, VIF (LR numerics)."""
    tr, ev = df[df.data_split == "train"], df[df.data_split.isin(["validation", "test"])]
    rows = []
    for f in features:
        is_cat = f in CATEGORICAL
        e = None if is_cat else bin_edges(tr[f])
        b_tr, b_ev = assign_bins(tr[f], e, is_cat), assign_bins(ev[f], e, is_cat)
        rows.append({"feature": f, "missing_train": tr[f].isna().mean(), "missing_val_test": ev[f].isna().mean(),
                     "iv": information_value(b_tr, tr[TARGET]), "psi_train_vs_valtest": psi(b_tr, b_ev),
                     "in_lr": f in LR_FEATURES, "in_xgb": f in XGB_FEATURES})
    chk = pd.DataFrame(rows).set_index("feature")
    X = tr[[f for f in LR_FEATURES if f not in CATEGORICAL]]
    X = X.clip(X.quantile(0.01), X.quantile(0.99), axis=1).fillna(X.median())
    chk["vif_lr"] = vif_table(X)
    log_table("feature checks (train / val+test):", chk.sort_values("iv", ascending=False))
    for f, r in chk.iterrows():
        if max(r.missing_train, r.missing_val_test) > 0.5:
            log.warning("feature %s: >50%% missing", f)
        if r.iv < 0.02:
            log.warning("feature %s: IV %.4f < 0.02", f, r.iv)
        if r.psi_train_vs_valtest > 0.25:
            log.warning("feature %s: PSI %.3f > 0.25 (unstable)", f, r.psi_train_vs_valtest)
        if r.vif_lr > 10:
            log.warning("feature %s: VIF %.1f > 10 in LR", f, r.vif_lr)
    return chk


def direction_table(train, num, cat, categories):
    rows = []
    for f in num:
        rho = train[f].corr(train[TARGET], method="spearman")
        rows.append({"model_input": f, "data_direction": int(np.sign(rho)), "train_spearman": rho})
    for f, levels in zip(cat, categories):
        for lvl in levels:
            m = train[f] == lvl
            diff = train.loc[m, TARGET].mean() - train.loc[~m, TARGET].mean()
            rows.append({"model_input": f"{f}_{lvl}", "data_direction": int(np.sign(diff)), "level_minus_rest": diff})
    t = pd.DataFrame(rows).set_index("model_input")
    t["prior"] = t.index.map(EXPECTED_DIRECTION)
    t["constraint"] = t.prior.fillna(t.data_direction).astype(int)
    t["conflict"] = t.prior.notna() & (t.data_direction != 0) & (t.prior != t.data_direction)
    if t.conflict.any():
        log.warning("prior vs train-data direction conflicts (prior used): %s", t.index[t.conflict].tolist())
    return t


def train_logreg(train_sorted, cv_splits, iv):
    feats = list(LR_FEATURES)
    num, cat = [f for f in feats if f not in CATEGORICAL], [f for f in feats if f in CATEGORICAL]
    pipe = Pipeline([("pre", build_lr_preprocessor(num, cat)),
                     ("model", LogisticRegression(solver="saga", max_iter=5000, random_state=RANDOM_STATE))])
    grid = {"model__C": LR_C_GRID}
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8):
        grid["model__l1_ratio"] = [0.0, 1.0]
    else:
        grid["model__penalty"] = ["l2", "l1"]
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv_splits, n_jobs=-1, refit=True)
    search.fit(train_sorted[feats], train_sorted[TARGET])
    log.info("LR best CV AUC %.4f | params %s", search.best_score_, search.best_params_)
    model = search.best_estimator_

    # sign check: numerics vs expected direction, WoE columns must be > 0. Drop weakest offender (IV) and refit.
    dropped = []
    while True:
        names = model.named_steps["pre"].get_feature_names_out()
        coef = pd.Series(model.named_steps["model"].coef_.ravel(), index=names)
        expected = {f: EXPECTED_DIRECTION[f] for f in feats if f not in CATEGORICAL}
        expected.update({f"{c}_woe": 1 for c in feats if c in CATEGORICAL})
        bad = [(n, coef[n]) for n, e in expected.items() if np.sign(coef[n]) not in (0, e)]
        if not bad:
            break
        raw = {n: n[:-4] if n.endswith("_woe") else n for n, _ in bad}
        worst = min(raw.values(), key=lambda f: iv.get(f, 0))
        log.warning("LR wrong sign: %s -> dropping %s (lowest IV) and refitting",
                    {n: round(v, 4) for n, v in bad}, worst)
        dropped.append({"feature": worst, "coef": float(coef[[n for n in raw if raw[n] == worst][0]])})
        feats.remove(worst)
        num, cat = [f for f in feats if f not in CATEGORICAL], [f for f in feats if f in CATEGORICAL]
        model = (Pipeline([("pre", build_lr_preprocessor(num, cat)),
                           ("model", LogisticRegression(solver="saga", max_iter=5000,
                                                        random_state=RANDOM_STATE))])
                 .set_params(**search.best_params_).fit(train_sorted[feats], train_sorted[TARGET]))
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["model"].coef_.ravel()
    coefs = (pd.DataFrame({"feature": names, "coef_per_1sd": coef, "odds_ratio": np.exp(coef)})
             .sort_values("coef_per_1sd", key=abs, ascending=False).reset_index(drop=True))
    log_table(f"LR coefficients (intercept {model.named_steps['model'].intercept_[0]:.4f}):", coefs)
    hp = {k.replace("model__", ""): (v.item() if isinstance(v, np.generic) else v)
          for k, v in search.best_params_.items()}
    return model, feats, hp, {"coefficients": coefs, "lr_dropped_for_sign": dropped}


def train_xgboost(train_sorted, valid, cv_splits, n_iter):
    feats = list(XGB_FEATURES)
    num, cat = [f for f in feats if f not in CATEGORICAL], [f for f in feats if f in CATEGORICAL]
    categories = [sorted(train_sorted[c].unique()) for c in cat]
    dirs = direction_table(train_sorted, num, cat, categories)
    monotone = {k: int(v) for k, v in dirs.constraint.items()}
    log_table("XGB monotone constraints:", dirs)

    pipe = Pipeline([("pre", build_xgb_preprocessor(num, cat, categories)),
                     ("model", XGBClassifier(tree_method="hist", eval_metric="auc", monotone_constraints=monotone,
                                             random_state=RANDOM_STATE, n_jobs=-1))])
    search = RandomizedSearchCV(pipe, XGB_PARAM_DIST, n_iter=n_iter, scoring="roc_auc", cv=cv_splits,
                                n_jobs=-1, random_state=RANDOM_STATE, refit=False)
    search.fit(train_sorted[feats], train_sorted[TARGET])
    log.info("XGB best CV AUC %.4f | params %s", search.best_score_, search.best_params_)

    best = {k.replace("model__", ""): v for k, v in search.best_params_.items() if k != "model__n_estimators"}
    pre = build_xgb_preprocessor(num, cat, categories).fit(train_sorted[feats], train_sorted[TARGET])
    assert set(monotone) == set(pre.get_feature_names_out()), "constraint names must match encoded columns"
    clf = XGBClassifier(**best, n_estimators=3000, early_stopping_rounds=100, tree_method="hist", eval_metric="auc",
                        monotone_constraints=monotone, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(pre.transform(train_sorted[feats]), train_sorted[TARGET],
            eval_set=[(pre.transform(valid[feats]), valid[TARGET])], verbose=False)
    log.info("XGB early stopping on validation: best_iteration %d | validation AUC %.4f",
             clf.best_iteration, clf.best_score)
    hp = {**{k: (v.item() if isinstance(v, np.generic) else v) for k, v in best.items()},
          "n_estimators": int(clf.best_iteration + 1), "n_estimators_selected_by": "early stopping on validation"}
    return Pipeline([("pre", pre), ("model", clf)]), feats, hp, {"monotone_constraints": monotone,
                                                                  "direction_table": dirs}


def check_monotonicity(pipeline, X_raw, X_train_raw, constraints, n_rows=300, n_grid=20):
    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["model"]
    Xs = pre.transform(X_raw.sample(min(n_rows, len(X_raw)), random_state=RANDOM_STATE)).reset_index(drop=True)
    Xtr = pre.transform(X_train_raw)
    booster, it = clf.get_booster(), (0, clf.best_iteration + 1)
    rows = []
    for name, c in constraints.items():
        grid = np.unique(np.nanquantile(Xtr[name].astype(float), np.linspace(0, 1, n_grid)))
        if len(grid) < 2:
            continue
        P = np.vstack([booster.predict(xgb.DMatrix(Xs.assign(**{name: g})), output_margin=True, iteration_range=it)
                       for g in grid])
        rows.append({"model_input": name, "constraint": c,
                     "rows_violating": float((np.diff(P, axis=0) * c < -1e-6).any(axis=0).mean()),
                     "logodds_effect_min_to_max": float((P[-1] - P[0]).mean())})
    t = pd.DataFrame(rows).set_index("model_input").sort_values("logodds_effect_min_to_max", key=abs, ascending=False)
    log_table("XGB monotonicity check (ICE on OOT sample):", t)
    if (t.rows_violating > 0).any():
        raise RuntimeError(f"monotone constraint violated for {t.index[t.rows_violating > 0].tolist()}")
    log.info("XGB monotonicity verified for all %d inputs", len(t))
    return t


def xgb_shap_importance(pipeline, X_raw, monotone):
    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["model"]
    Xt = pre.transform(X_raw).reset_index(drop=True)
    contribs = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True, iteration_range=(0, clf.best_iteration + 1))
    shap = pd.DataFrame(contribs[:, :-1], columns=Xt.columns)
    t = pd.DataFrame({"mean_abs_shap": shap.abs().mean(), "constraint": pd.Series(monotone)})
    log_table("XGB mean |SHAP| on OOT:", t.sort_values("mean_abs_shap", ascending=False))
    return t.sort_values("mean_abs_shap", ascending=False)


def evaluate(model, df, feats, threshold, name):
    rows = []
    for s in SPLITS:
        d = df[df.data_split == s]
        p = model.predict_proba(d[feats])[:, 1]
        auc = roc_auc_score(d[TARGET], p)
        rows.append({"model": name, "split": s, "n": len(d), "default_rate": d[TARGET].mean(), "AUC": auc,
                     "Gini": 2 * auc - 1, "KS": ks_stat(d[TARGET], p), "PR_AUC": average_precision_score(d[TARGET], p),
                     "Brier": brier_score_loss(d[TARGET], p), "LogLoss": log_loss(d[TARGET], p),
                     "mean_pred": p.mean(), "threshold": threshold})
    m = pd.DataFrame(rows)
    log_table(f"{name} metrics by split:", m)
    s_tr = pd.Series(model.predict_proba(df.loc[df.data_split == "train", feats])[:, 1])
    e = bin_edges(s_tr)
    for s in ["validation", "test", "oot"]:
        s_ev = pd.Series(model.predict_proba(df.loc[df.data_split == s, feats])[:, 1])
        v = psi(assign_bins(s_tr, e, False), assign_bins(s_ev, e, False))
        (log.warning if v > 0.1 else log.info)("%s score PSI train -> %s: %.4f", name, s, v)
    return m


def build_artefact(algo, name, pipeline, feats, hp, metrics, threshold, df, config, checks, extra):
    pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["model"]
    num_step = pre.named_transformers_["num"]
    cat_step = pre.named_transformers_["cat"]
    m = metrics.set_index("split")
    a = {}
    a["model"] = clf
    a["model_version"] = f"credit_model_{algo}_" + config["model_train_date_str"].replace("-", "_")
    a["preprocessing_transformers"] = {
        "preprocessor": pre,
        "stdscaler": num_step.named_steps["scale"] if isinstance(num_step, Pipeline) else None,
        "quantile_capper": num_step.named_steps["cap"] if isinstance(num_step, Pipeline) else num_step,
        "woe_encoder": cat_step.named_steps.get("woe") if isinstance(cat_step, Pipeline) else None,
    }
    a["data_dates"] = config
    a["data_stats"] = {}
    for s in SPLITS:
        a["data_stats"][f"X_{s}"] = int((df.data_split == s).sum())
    for s in SPLITS:
        a["data_stats"][f"y_{s}"] = round(float(df.loc[df.data_split == s, TARGET].mean()), 2)
    a["results"] = {}
    for s in SPLITS:
        a["results"][f"auc_{s}"] = float(m.loc[s, "AUC"])
    for s in SPLITS:
        a["results"][f"gini_{s}"] = round(2 * float(m.loc[s, "AUC"]) - 1, 3)
    for s in SPLITS:
        a["results"][f"ks_{s}"] = round(float(m.loc[s, "KS"]), 3)
    a["hp_params"] = hp
    a["features"] = {"raw_input_features": list(feats), "model_input_columns": list(pre.get_feature_names_out())}
    a["decision_threshold"] = float(threshold)
    a["feature_cleaning"] = CLEANING_CONFIG
    a["feature_checks"] = checks
    a["metrics"] = metrics
    a["monitoring_baseline"] = build_monitoring_baseline(pipeline, df[df.data_split == "train"], feats,
                                                         reference_auc=float(m.loc["test", "AUC"]))
    a.update(extra)
    return a


def build_monitoring_baseline(pipeline, train, feats, reference_auc):
    """Train distributions of the score and of each raw input feature, binned once, for PSI at monitoring time."""
    scores = pd.Series(pipeline.predict_proba(train[feats])[:, 1])
    s_edges = bin_edges(scores)
    base = {"reference_auc": reference_auc, "reference_split": "test",
            "score_edges": s_edges.tolist(),
            "score_dist": assign_bins(scores, s_edges, False).value_counts(normalize=True).to_dict(),
            "feature_edges": {}, "feature_dist": {}, "feature_missing": {}}
    for f in feats:
        is_cat = f in CATEGORICAL
        e = None if is_cat else bin_edges(train[f])
        base["feature_edges"][f] = None if e is None else e.tolist()
        base["feature_dist"][f] = assign_bins(train[f], e, is_cat).value_counts(normalize=True).to_dict()
        base["feature_missing"][f] = float(train[f].isna().mean())
    return base


def psi_vs_baseline(base_dist, actual_bins, eps=1e-4):
    e = pd.Series(base_dist, dtype=float)
    a = actual_bins.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx, fill_value=0) + eps, a.reindex(idx, fill_value=0) + eps
    return float(((a - e) * np.log(a / e)).sum())


def prepare_training_frame(model_train_date_str, spark, split_source):
    config = build_split_config(model_train_date_str)
    log.info("config: %s", {k: (v.date() if isinstance(v, datetime) else v) for k, v in config.items()})
    raw = load_training_data(spark, config, split_source)
    log.info("cleaning (%d rows)", len(raw))
    return config, basic_clean(raw, **CLEANING_CONFIG)


def run_training(model_train_date_str, xgb_n_iter, spark, split_source="store", trigger="manual",
                 algos=("logreg", "xgb"), register=True):
    """Train the requested algorithms as of model_train_date_str and save their artefacts.
    register=True also runs the promotion gate (standalone use); the DAG trains each algorithm in its own task
    and runs the gate afterwards in `select`."""
    config, df = prepare_training_frame(model_train_date_str, spark, split_source)
    missing = [f for f in set(LR_FEATURES + XGB_FEATURES) if f not in df.columns]
    if missing:
        raise KeyError(f"features missing from the feature store: {missing}")

    checks = feature_checks(df, list(dict.fromkeys(XGB_FEATURES + LR_FEATURES)))
    iv = checks.iv.to_dict()

    train_sorted = df[df.data_split == "train"].sort_values("labels_snapshot_date").reset_index(drop=True)
    valid = df[df.data_split == "validation"]
    log.info("time-ordered CV folds inside train:")
    cv_splits = monthly_expanding_splits(train_sorted.labels_snapshot_date)

    oot = df[df.data_split == "oot"]
    trained = []
    if "logreg" in algos:
        log.info("=" * 25 + " Logistic Regression " + "=" * 25)
        lr_model, lr_feats, lr_hp, lr_extra = train_logreg(train_sorted, cv_splits, iv)
        trained.append(("logreg", "LogReg", lr_model, lr_feats, lr_hp, lr_extra))
    if "xgb" in algos:
        log.info("=" * 25 + " XGBoost " + "=" * 25)
        xgb_model, xgb_feats, xgb_hp, xgb_extra = train_xgboost(train_sorted, valid, cv_splits, xgb_n_iter)
        xgb_extra["monotonicity_check"] = check_monotonicity(xgb_model, oot[xgb_feats], train_sorted[xgb_feats],
                                                             xgb_extra["monotone_constraints"])
        xgb_extra["shap_importance"] = xgb_shap_importance(xgb_model, oot[xgb_feats],
                                                           xgb_extra["monotone_constraints"])
        trained.append(("xgb", "XGBoost", xgb_model, xgb_feats, xgb_hp, xgb_extra))

    artefacts = []
    for algo, name, model, feats, hp, extra in trained:
        thr = youden_threshold(valid[TARGET], model.predict_proba(valid[feats])[:, 1])
        log.info("%s decision threshold (KS point on validation): %.4f", name, thr)
        metrics = evaluate(model, df, feats, thr, name)
        artefacts.append(build_artefact(algo, name, model, feats, hp, metrics, thr, df, config, checks, extra))

    os.makedirs(PATHS["model_bank"], exist_ok=True)
    for a in artefacts:
        path = os.path.join(PATHS["model_bank"], a["model_version"] + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(a, f)
        log.info("saved artefact: %s", path)

        # reload test: the stored preprocessor + model must reproduce the stored OOT AUC
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        feats = loaded["features"]["raw_input_features"]
        p = loaded["model"].predict_proba(loaded["preprocessing_transformers"]["preprocessor"].transform(oot[feats]))[:, 1]
        auc = roc_auc_score(oot[TARGET], p)
        if not np.isclose(auc, loaded["results"]["auc_oot"]):
            raise RuntimeError(f"reloaded {path} gives OOT AUC {auc:.4f} != stored {loaded['results']['auc_oot']:.4f}")
        log.info("reload test OK: %s OOT AUC %.4f | Gini %.3f", loaded["model_version"], auc, 2 * auc - 1)

    summary = pd.DataFrame({a["model_version"]: a["results"] for a in artefacts})
    log_table("training summary:", summary)
    if register:
        register_training_run(artefacts, df, model_train_date_str, trigger)
    return artefacts


def run_selection(as_of, spark, split_source="dates", trigger=None, effective_from=None):
    """Promotion gate for the candidates trained as of `as_of` (one artefact per algorithm task)."""
    tag = as_of.replace("-", "_")
    files = sorted(glob.glob(os.path.join(PATHS["model_bank"], f"credit_model_*_{tag}.pkl")))
    if not files:
        raise FileNotFoundError(f"no candidate artefacts for {as_of} in {PATHS['model_bank']}")
    artefacts = []
    for f in files:
        with open(f, "rb") as fh:
            artefacts.append(pickle.load(fh))
    log.info("candidates for %s: %s", as_of, [a["model_version"] for a in artefacts])
    _, df = prepare_training_frame(as_of, spark, split_source)       # OOT window for the champion comparison
    trigger = trigger or (latest_decision(as_of) or {}).get("trigger", "manual")
    return register_training_run(artefacts, df, as_of, trigger, effective_from=effective_from)


# =====================================================================================================================
# MODEL STORE: champion selection + registry
# =====================================================================================================================
def _empty_registry():
    return {"selection_metric": CHAMPION_METRIC, "governance": GOVERNANCE, "models": {}, "training_runs": [],
            "role_history": [], "champion": None, "champion_file": None, "challenger": None}


def load_registry(missing_ok=False):
    if not os.path.exists(PATHS["registry"]):
        if missing_ok:
            return _empty_registry()
        raise FileNotFoundError(f"{PATHS['registry']} not found - train a model first")
    with open(PATHS["registry"]) as f:
        reg = json.load(f)
    if "role_history" not in reg:                       # registry written by an earlier main.py version
        old = reg
        reg = _empty_registry()
        for c in old.get("candidates", []):
            reg["models"][c["model_version"]] = c
        others = [c["model_version"] for c in old.get("candidates", []) if c["role"] != "champion"]
        reg["role_history"].append({"effective_from": "1900-01-01", "champion": old["champion"],
                                    "challenger": others[0] if others else None, "reason": "migrated"})
    return reg


def save_registry(reg):
    hist = sorted(reg["role_history"], key=lambda r: r["effective_from"])
    reg["champion"] = hist[-1]["champion"] if hist else None
    reg["challenger"] = hist[-1]["challenger"] if hist else None
    reg["champion_file"] = reg["champion"] + ".pkl" if reg["champion"] else None
    reg["updated_at"] = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(PATHS["registry"]), exist_ok=True)
    with open(PATHS["registry"], "w") as f:
        json.dump(reg, f, indent=2, default=str)


def resolve_roles(as_of, reg=None):
    """Champion / challenger in force on `as_of` (latest role_history entry with effective_from <= as_of)."""
    reg = reg or load_registry(missing_ok=True)
    hist = [r for r in reg["role_history"] if r["effective_from"] <= as_of]
    return max(hist, key=lambda r: r["effective_from"]) if hist else None


def resolve_models(modelnames, include_challenger=False, as_of=None):
    if modelnames:
        return modelnames
    roles = resolve_roles(as_of or "9999-12-31")
    if roles is None:
        raise FileNotFoundError("no champion in the registry - train a model first")
    names = [roles["champion"] + ".pkl"]
    if include_challenger and roles.get("challenger"):
        names.append(roles["challenger"] + ".pkl")
    log.info("models from registry as of %s: %s", as_of or "latest", names)
    return names


def log_governance_event(as_of, event, **fields):
    """Append-only audit trail -> gold table datamart/gold/model_governance/ (one small parquet per event)."""
    os.makedirs(PATHS["governance_root"], exist_ok=True)
    row = {"as_of_date": as_of, "event": event, "logged_at": datetime.now().isoformat(timespec="seconds")}
    row.update({k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in fields.items()})
    path = os.path.join(PATHS["governance_root"],
                        f"governance_{as_of.replace('-', '_')}_{event}_{datetime.now():%H%M%S%f}.parquet")
    pd.DataFrame([row]).astype(str).to_parquet(path, index=False)
    return row


def read_governance_log():
    files = sorted(glob.glob(os.path.join(PATHS["governance_root"], "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def gini_on(artefact, frame):
    feats = artefact["features"]["raw_input_features"]
    p = artefact["model"].predict_proba(artefact["preprocessing_transformers"]["preprocessor"].transform(frame[feats]))[:, 1]
    return 2 * roc_auc_score(frame[TARGET], p) - 1


def register_training_run(artefacts, df, as_of, trigger, effective_from=None):
    """Model store update + promotion gate. All candidates are stored; only a gated winner becomes champion.
    effective_from: date the resulting roles take effect (default: as_of). The bootstrap uses the first
    deployment month, so a model trained on the full history can score that history."""
    reg = load_registry(missing_ok=True)
    effective = effective_from or as_of
    # idempotent re-runs (Airflow retry / clear): replace this month's training record instead of appending
    reg["training_runs"] = [t for t in reg["training_runs"] if t["as_of"] != as_of]
    reg["role_history"] = [h for h in reg["role_history"]
                           if not (h["effective_from"] == effective and "refresh" in h.get("reason", ""))]
    for a in artefacts:
        reg["models"][a["model_version"]] = {
            "file": a["model_version"] + ".pkl", "algo": a["model_version"].split("_")[2],
            "model_train_date": a["data_dates"]["model_train_date_str"], "trained_as_of": as_of,
            "features": a["features"]["raw_input_features"], "decision_threshold": a["decision_threshold"],
            "results": {k: round(v, 4) for k, v in a["results"].items()}}
    ranked = sorted(artefacts, key=lambda a: a["results"][CHAMPION_METRIC], reverse=True)
    cand, runner_up = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    r = cand["results"]
    gates = {"min_gini_test": bool(r["gini_test"] >= GOVERNANCE["min_gini_gate"]),
             "min_gini_oot": bool(r["gini_oot"] >= GOVERNANCE["min_gini_gate"]),
             "test_oot_gap": bool(abs(r["gini_test"] - r["gini_oot"]) <= GOVERNANCE["max_test_oot_gini_gap"]),
             "monotonicity_verified": True}           # check_monotonicity raised otherwise
    incumbent = resolve_roles(as_of, reg)
    comparison = {"candidate": cand["model_version"], "candidate_gini_oot": round(r["gini_oot"], 4)}
    if incumbent is None:
        promote = all(gates.values())
        reason = "initial deployment" if promote else "initial candidate failed quality gates"
    else:
        with open(os.path.join(PATHS["model_bank"], incumbent["champion"] + ".pkl"), "rb") as f:
            champ = pickle.load(f)
        oot = df[df.data_split == "oot"]
        champ_gini = float(gini_on(champ, oot))        # incumbent on the SAME, newest out-of-time window
        comparison.update(champion=incumbent["champion"], champion_gini_same_oot=round(champ_gini, 4),
                          uplift=round(r["gini_oot"] - champ_gini, 4))
        better = bool(r["gini_oot"] >= champ_gini + GOVERNANCE["promotion_min_uplift"])
        promote = all(gates.values()) and better
        reason = ("promoted: passes gates and OOT Gini >= champion" if promote else
                  "kept champion: " + ("candidate failed quality gates" if not all(gates.values())
                                       else "candidate not better on the same OOT window"))
    if promote:
        reg["role_history"].append({"effective_from": effective, "champion": cand["model_version"],
                                    "challenger": runner_up["model_version"] if runner_up else None,
                                    "reason": f"{trigger} refresh - {reason}"})
    elif incumbent is not None:                         # shadow the freshest candidate against the champion
        reg["role_history"].append({"effective_from": effective, "champion": incumbent["champion"],
                                    "challenger": cand["model_version"], "reason": f"{trigger} refresh - {reason}"})
    reg["training_runs"].append({"as_of": as_of, "trigger": trigger, "candidates": [a["model_version"] for a in ranked],
                                 "best_candidate": cand["model_version"], "gates": gates, "comparison": comparison,
                                 "promoted": promote, "reason": reason})
    save_registry(reg)
    log_table(f"candidates ranked on {CHAMPION_METRIC}:", pd.DataFrame(
        {a["model_version"]: {k: a["results"][k] for k in ["auc_test", "gini_test", "gini_oot", "ks_test"]}
         for a in ranked}).T)
    log.info("promotion gates: %s | comparison: %s", gates, comparison)
    (log.info if promote else log.warning)("DECISION (%s): %s -> champion as of %s: %s", trigger, reason, as_of,
                                           resolve_roles(as_of, reg)["champion"] if resolve_roles(as_of, reg) else None)
    log_governance_event(as_of, "training", trigger=trigger, best_candidate=cand["model_version"], promoted=promote,
                         reason=reason, gates=gates, comparison=comparison,
                         champion_after=(resolve_roles(as_of, reg) or {}).get("champion"))
    return reg


REQUIRED_ARTEFACT_KEYS = ("model", "model_version", "preprocessing_transformers", "features", "feature_cleaning",
                          "results", "decision_threshold", "monitoring_baseline")


def bootstrap_champion(deploy_from, model_train_date, xgb_n_iter, spark):
    """One-off initial training for the DAG: the same model as the notebook / standalone run (fixed
    model_train_date, gold-store splits), registered as champion from the first deployment month.
    No-op once the registry has a champion, so every later DAG run skips straight through."""
    reg = load_registry(missing_ok=True)
    if reg["role_history"]:
        log.info("bootstrap: registry already has champion history (current champion %s) - nothing to do",
                 reg.get("champion"))
        return None
    tag = model_train_date.replace("-", "_")
    files = sorted(glob.glob(os.path.join(PATHS["model_bank"], f"credit_model_*_{tag}.pkl")))
    artefacts = []
    for path in files:
        with open(path, "rb") as fh:
            a = pickle.load(fh)
        missing = [k for k in REQUIRED_ARTEFACT_KEYS if k not in a]
        if missing:
            raise ValueError(f"{path} was not written by this main.py (missing {missing}); move it out of "
                             f"{PATHS['model_bank']} so bootstrap can train a fresh artefact")
        artefacts.append(a)
    if artefacts:
        log.info("bootstrap: reusing existing artefacts for %s: %s", model_train_date,
                 [a["model_version"] for a in artefacts])
    else:
        log.info("bootstrap: training LogReg + XGBoost as of %s on the gold-store splits", model_train_date)
        artefacts = run_training(model_train_date, xgb_n_iter, spark, split_source="store", trigger="initial",
                                 register=False)
    _, df = prepare_training_frame(model_train_date, spark, "store")
    reg = register_training_run(artefacts, df, model_train_date, "initial", effective_from=deploy_from)
    log.warning("bootstrap: champion %s / challenger %s in force from %s", reg.get("champion"),
                reg.get("challenger"), deploy_from)
    return reg


def promote_manual(modelname, as_of, reason="manual override / rollback"):
    reg = load_registry()
    mv = modelname[:-4] if modelname.endswith(".pkl") else modelname
    if mv not in reg["models"]:
        raise KeyError(f"{mv} is not in the model store")
    current = resolve_roles(as_of, reg)
    reg["role_history"].append({"effective_from": as_of, "champion": mv,
                                "challenger": current["champion"] if current else None, "reason": reason})
    save_registry(reg)
    log.warning("MANUAL PROMOTION: %s is champion from %s (%s)", mv, as_of, reason)
    log_governance_event(as_of, "manual_promotion", champion_after=mv, reason=reason)


# =====================================================================================================================
# DATA CHECKS + REFRESH DECISION (no Spark needed)
# =====================================================================================================================
def months_between(later, earlier):
    a, b = pd.Timestamp(later), pd.Timestamp(earlier)
    return (a.year - b.year) * 12 + a.month - b.month


def run_check_data(as_of, store="all"):
    """Dependency check for the gold stores. Fails only if a store is missing; empty months are a warning."""
    for name in [n for n in ["feature_store", "label_store"] if store in ("all", n.split("_")[0])]:
        if not os.path.exists(PATHS[name]):
            raise FileNotFoundError(f"{name} not found at {PATHS[name]} - run the data pipeline first")
    fs = pd.read_parquet(PATHS["feature_store"], columns=[SCORING_DATE_COL])
    lb = pd.read_parquet(PATHS["label_store"], columns=["snapshot_date"])
    n_feat = int((pd.to_datetime(fs[SCORING_DATE_COL]) == pd.Timestamp(as_of)).sum())
    lab_dates = pd.to_datetime(lb["snapshot_date"])
    n_lab = int((lab_dates == pd.Timestamp(as_of)).sum())
    n_months = int(lab_dates[lab_dates < pd.Timestamp(as_of)].dt.to_period("M").nunique())
    log.info("as of %s: %d loans to score (features observed this month) | %d labels maturing this month | "
             "%d labelled months of history", as_of, n_feat, n_lab, n_months)
    if n_feat == 0:
        log.warning("no new applications with %s = %s: inference will skip this month", SCORING_DATE_COL, as_of)
    return {"n_to_score": n_feat, "n_new_labels": n_lab, "labelled_months": n_months}


def latest_monitoring(model_version, as_of):
    folder = os.path.join(PATHS["monitoring_root"], model_version)
    if not os.path.isdir(folder):
        return pd.DataFrame()
    parts = [pd.read_parquet(f) for f in glob.glob(os.path.join(folder, "*.parquet"))]
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts, ignore_index=True)
    m["snapshot_date"] = pd.to_datetime(m["snapshot_date"])
    return m[m.snapshot_date < pd.Timestamp(as_of)].sort_values("snapshot_date")


def decide_refresh(as_of):
    """Refresh policy (GOVERNANCE): cold start -> initial; age -> scheduled; monitoring alert -> triggered."""
    reg = load_registry(missing_ok=True)
    roles = resolve_roles(as_of, reg)
    info = run_check_data(as_of)
    last_train = max([t["as_of"] for t in reg["training_runs"] if t["as_of"] <= as_of], default=None)
    if roles is None:
        if last_train and months_between(as_of, last_train) < GOVERNANCE["min_months_between_training"]:
            d = {"decision": "skip", "trigger": "cooldown", "detail": f"last attempt {last_train} failed gates"}
        elif info["labelled_months"] >= GOVERNANCE["min_label_months"]:
            d = {"decision": "train", "trigger": "initial",
                 "detail": f"{info['labelled_months']} labelled months available"}
        else:
            d = {"decision": "skip", "trigger": "cold_start",
                 "detail": f"{info['labelled_months']}/{GOVERNANCE['min_label_months']} labelled months"}
    else:
        champ = reg["models"][roles["champion"]]
        age = months_between(as_of, champ["model_train_date"])
        since_train = months_between(as_of, last_train) if last_train else 999
        mon = latest_monitoring(roles["champion"], as_of)
        alerts = []
        if not mon.empty:
            stab = mon.iloc[-1]
            if stab.stability_status == "alert":
                alerts.append(f"stability alert {stab.snapshot_date:%Y-%m} (score PSI {float(stab.score_psi):.3f}, "
                              f"max feature PSI {float(stab.max_feature_psi):.3f})")
            perf = mon[mon.performance_status.isin(["ok", "watch", "alert"])]
            if not perf.empty and perf.iloc[-1].performance_status == "alert":
                alerts.append(f"performance alert {perf.iloc[-1].snapshot_date:%Y-%m} "
                              f"(AUC {float(perf.iloc[-1].auc):.3f} vs ref {float(perf.iloc[-1].reference_auc):.3f})")
        if since_train >= GOVERNANCE["refresh_every_months"] and age >= GOVERNANCE["refresh_every_months"]:
            d = {"decision": "train", "trigger": "scheduled", "detail": f"champion trained {age} months ago"}
        elif alerts and since_train >= GOVERNANCE["min_months_between_training"]:
            d = {"decision": "train", "trigger": "triggered", "detail": "; ".join(alerts)}
        elif alerts:
            d = {"decision": "skip", "trigger": "cooldown", "detail": "; ".join(alerts) + f" (trained {since_train}m ago)"}
        else:
            d = {"decision": "skip", "trigger": "healthy", "detail": f"champion {roles['champion']} age {age}m, no alerts"}
    d.update(as_of=as_of, champion=roles["champion"] if roles else None)
    (log.warning if d["decision"] == "train" else log.info)("REFRESH DECISION %s: %s (%s) - %s", as_of,
                                                            d["decision"].upper(), d["trigger"], d["detail"])
    log_governance_event(as_of, "decision", **{k: v for k, v in d.items() if k != "as_of"})
    return d


def latest_decision(as_of):
    g = read_governance_log()
    if g.empty:
        return None
    g = g[(g.event == "decision") & (g.as_of_date == as_of)]
    return None if g.empty else g.sort_values("logged_at").iloc[-1].to_dict()


# =====================================================================================================================
# INFERENCE / BACKFILL
# =====================================================================================================================
def inference_partition(ds):
    return os.path.join(PATHS["inference_features"], f"inference_features_{ds.replace('-', '_')}.parquet")


def load_features_for_month(ds, spark):
    """New applications come from the inference feature store (built by the inference DAG with the training-time
    parameters); months only in the historical gold feature store fall back to it (backtest)."""
    path = inference_partition(ds)
    if os.path.exists(path):
        return pd.read_parquet(path), "inference_feature_store"
    sdf = spark.read.parquet(PATHS["feature_store"]).filter(F.col(SCORING_DATE_COL) == F.to_date(F.lit(ds)))
    return sdf.toPandas(), "gold_feature_store"


def load_features_for_range(dates, spark):
    gold = (spark.read.parquet(PATHS["feature_store"])
            .filter((F.col(SCORING_DATE_COL) >= F.to_date(F.lit(dates[0])))
                    & (F.col(SCORING_DATE_COL) <= F.to_date(F.lit(dates[-1])))).toPandas())
    parts = [pd.read_parquet(inference_partition(d)) for d in dates if os.path.exists(inference_partition(d))]
    if not parts:
        return gold
    inf = pd.concat(parts, ignore_index=True)
    split_of = dict(zip(gold["loan_id"], gold["data_split"])) if "data_split" in gold.columns else {}
    inf["data_split"] = inf["loan_id"].map(split_of).fillna("production")   # production = never seen in training
    return pd.concat([inf, gold[~gold["loan_id"].isin(inf["loan_id"])]], ignore_index=True)


def run_inference(snapshotdate, modelname, spark, role=None):
    config = {"snapshot_date_str": snapshotdate, "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
              "model_name": modelname, "model_artefact_filepath": os.path.join(PATHS["model_bank"], modelname)}
    log.info("inference config: %s", config)

    with open(config["model_artefact_filepath"], "rb") as f:
        artefact = pickle.load(f)
    log.info("loaded %s (trained %s, OOT AUC %.4f)", artefact["model_version"],
             artefact["data_dates"]["model_train_date_str"], artefact["results"]["auc_oot"])

    features_pdf, source = load_features_for_month(snapshotdate, spark)
    n_rows = len(features_pdf)
    log.info("extracted %d applications for %s from %s", n_rows, snapshotdate, source)
    if n_rows == 0:
        log.warning("nothing to score for %s - skipped", snapshotdate)
        return None

    clean = basic_clean(features_pdf, **artefact["feature_cleaning"], verbose=False)
    feats = artefact["features"]["raw_input_features"]
    missing_share = clean[feats].isna().mean()
    for f, v in missing_share[missing_share > 0.5].items():
        log.warning("feature %s is %.0f%% missing in this snapshot", f, 100 * v)
    X = artefact["preprocessing_transformers"]["preprocessor"].transform(clean[feats])
    y = artefact["model"].predict_proba(X)[:, 1]

    out = clean[["Customer_ID", "loan_id"]].copy()
    out["snapshot_date"] = config["snapshot_date"].date()
    out["model_name"] = modelname
    out["model_role"] = role or "adhoc"
    out["model_predictions"] = y.astype(float)
    out["decision_threshold"] = float(artefact["decision_threshold"])
    out["feature_source"] = source
    out["predicted_default"] = (y >= artefact["decision_threshold"]).astype(int)
    log.info("scores: mean %.4f | p10 %.4f | p50 %.4f | p90 %.4f | predicted default %.2f%%",
             y.mean(), *np.quantile(y, [0.1, 0.5, 0.9]), 100 * out.predicted_default.mean())

    gold_directory = os.path.join(PATHS["predictions_root"], modelname[:-4]) + "/"
    os.makedirs(gold_directory, exist_ok=True)
    filepath = gold_directory + modelname[:-4] + "_predictions_" + snapshotdate.replace("-", "_") + ".parquet"
    spark.createDataFrame(out).write.mode("overwrite").parquet(filepath)
    log.info("saved predictions: %s (%d rows)", filepath, len(out))
    return out


def run_inference_role(snapshotdate, role, spark):
    roles = resolve_roles(snapshotdate)
    mv = (roles or {}).get(role)
    if mv is None:
        log.warning("no %s model in force on %s - inference skipped", role, snapshotdate)
        return None
    return run_inference(snapshotdate, mv + ".pkl", spark, role=role)


def first_of_month_dates(start_date_str, end_date_str):
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date_str, end_date_str, freq="MS")]


def run_backfill(startdate, enddate, modelnames, spark):
    dates = first_of_month_dates(startdate, enddate)
    log.info("backfill %d months (%s..%s) for %s", len(dates), dates[0], dates[-1], modelnames)
    scored = {}
    for modelname in modelnames:
        n = 0
        for d in dates:
            out = run_inference(d, modelname, spark)
            n += 0 if out is None else len(out)
        scored[modelname] = n
        folder = os.path.join(PATHS["predictions_root"], modelname[:-4]) + "/"
        files = [folder + os.path.basename(f) for f in glob.glob(os.path.join(folder, "*"))]
        total = spark.read.parquet(*files).count() if files else 0
        log.info("backfill %s: %d rows scored this run | %d rows in datamart", modelname, n, total)
    return scored


# =====================================================================================================================
# MONITORING
# =====================================================================================================================
def _status(value, watch, alert, higher_is_worse=True):
    if value is None or pd.isna(value):
        return "n/a"
    v = value if higher_is_worse else -value
    return "alert" if v > alert else "watch" if v > watch else "ok"


def run_monitoring(startdate, enddate, modelname, spark, as_of=None, plot=True):
    """Per scoring month: performance (once labels exist) and stability (score + feature PSI vs train baseline).
    With as_of, only labels observed on/before as_of are used - what monitoring knew at that time."""
    with open(os.path.join(PATHS["model_bank"], modelname), "rb") as f:
        artefact = pickle.load(f)
    mv, base = artefact["model_version"], artefact.get("monitoring_baseline")
    if base is None:
        raise KeyError(f"{modelname} has no monitoring_baseline - retrain with this version of main.py")
    feats = artefact["features"]["raw_input_features"]
    dates = first_of_month_dates(startdate, enddate)
    log.info("monitoring %s over %d months (%s..%s); performance reference = %s AUC %.4f",
             mv, len(dates), dates[0], dates[-1], base["reference_split"], base["reference_auc"])

    pred_folder = os.path.join(PATHS["predictions_root"], mv) + "/"
    pred_files = [pred_folder + os.path.basename(f) for f in glob.glob(os.path.join(pred_folder, "*"))]
    if not pred_files:
        raise FileNotFoundError(f"no predictions in {pred_folder} - run backfill first")
    preds = spark.read.parquet(*pred_files).toPandas()
    preds["snapshot_date"] = pd.to_datetime(preds["snapshot_date"])
    if "model_role" not in preds.columns:
        preds["model_role"] = "adhoc"
    dates = [d for d in dates if (preds.snapshot_date == pd.Timestamp(d)).any()]
    if not dates:
        log.warning("monitoring %s: no predictions in range - skipped", mv)
        return None, None

    labels = (spark.read.parquet(PATHS["label_store"])
              .select("loan_id", TARGET, F.col("snapshot_date").alias("label_date")).toPandas())
    if as_of:
        labels = labels[pd.to_datetime(labels.label_date) <= pd.Timestamp(as_of)]
    features = basic_clean(load_features_for_range(dates, spark), **artefact["feature_cleaning"], verbose=False)
    features[SCORING_DATE_COL] = pd.to_datetime(features[SCORING_DATE_COL])

    data = preds.merge(labels, on="loan_id", how="left").merge(
        features[["loan_id", "data_split"] + feats], on="loan_id", how="left")

    summary_rows, drift_rows = [], []
    for d in dates:
        m = data[data.snapshot_date == pd.Timestamp(d)]
        if m.empty:
            log.warning("monitoring %s: no predictions - skipped", d)
            continue
        p = m.model_predictions.to_numpy()
        lab = m.dropna(subset=[TARGET])
        row = {"model_version": mv, "snapshot_date": pd.Timestamp(d).date(), "as_of_date": as_of or "latest",
               "model_role": m.model_role.mode().iat[0], "n_scored": len(m),
               "n_labelled": len(lab), "label_period": m.data_split.mode().iat[0] if m.data_split.notna().any() else None,
               "mean_predicted_pd": float(p.mean()), "predicted_default_rate": float(m.predicted_default.mean()),
               "observed_default_rate": float(lab[TARGET].mean()) if len(lab) else np.nan,
               "auc": np.nan, "gini": np.nan, "ks": np.nan, "brier": np.nan,
               "reference_auc": base["reference_auc"]}
        if len(lab) >= MONITORING["min_labelled_rows"] and lab[TARGET].nunique() == 2:
            y, pl = lab[TARGET].astype(int), lab.model_predictions
            row.update(auc=roc_auc_score(y, pl), ks=ks_stat(y, pl), brier=brier_score_loss(y, pl))
            row["gini"] = 2 * row["auc"] - 1
        row["auc_drop_vs_reference"] = base["reference_auc"] - row["auc"]
        row["score_psi"] = psi_vs_baseline(base["score_dist"],
                                           assign_bins(pd.Series(p), np.asarray(base["score_edges"]), False))
        for f in feats:
            e = base["feature_edges"][f]
            v = psi_vs_baseline(base["feature_dist"][f],
                                assign_bins(m[f].reset_index(drop=True), None if e is None else np.asarray(e),
                                            f in CATEGORICAL))
            drift_rows.append({"model_version": mv, "snapshot_date": row["snapshot_date"], "feature": f, "psi": v,
                               "missing_rate": float(m[f].isna().mean()),
                               "baseline_missing_rate": base["feature_missing"][f],
                               "status": _status(v, MONITORING["psi_watch"], MONITORING["psi_alert"])})
        month_drift = [r for r in drift_rows if r["snapshot_date"] == row["snapshot_date"]]
        worst = max(month_drift, key=lambda r: r["psi"])
        row.update(max_feature_psi=worst["psi"], max_feature_psi_name=worst["feature"],
                   n_features_psi_alert=sum(r["status"] == "alert" for r in month_drift))
        row["performance_status"] = _status(row["auc_drop_vs_reference"], MONITORING["auc_drop_watch"],
                                            MONITORING["auc_drop_alert"]) if not pd.isna(row["auc"]) else "no labels yet"
        row["stability_status"] = _status(max(row["score_psi"], row["max_feature_psi"]),
                                          MONITORING["psi_watch"], MONITORING["psi_alert"])
        order = ["ok", "n/a", "no labels yet", "watch", "alert"]
        row["overall_status"] = max([row["performance_status"], row["stability_status"]], key=order.index)
        summary_rows.append(row)

    summary, drift = pd.DataFrame(summary_rows), pd.DataFrame(drift_rows)
    log_table(f"monitoring summary - {mv}:", summary.drop(columns=["model_version", "reference_auc"]))
    for _, r in summary[summary.overall_status.isin(["watch", "alert"])].iterrows():
        log.warning("%s %s: %s | perf=%s (AUC %.4f vs ref %.4f) | stability=%s (score PSI %.3f, max feature PSI "
                    "%.3f on %s)", mv, r.snapshot_date, r.overall_status.upper(), r.performance_status, r.auc,
                    r.reference_auc, r.stability_status, r.score_psi, r.max_feature_psi, r.max_feature_psi_name)
    counts = summary.overall_status.value_counts().to_dict()
    log.info("monitoring status counts for %s: %s", mv, counts)

    # ---- gold tables: one partition per snapshot month (same convention as model_predictions) ----
    for root, table, name in [(PATHS["monitoring_root"], summary, "monitoring"),
                              (PATHS["feature_drift_root"], drift, "feature_drift")]:
        folder = os.path.join(root, mv) + "/"
        os.makedirs(folder, exist_ok=True)
        for d, part in table.groupby("snapshot_date"):
            path = folder + f"{mv}_{name}_{d.strftime('%Y_%m_%d')}.parquet"
            spark.createDataFrame(part.reset_index(drop=True)).write.mode("overwrite").parquet(path)
        log.info("saved gold table %s: %s (%d rows, %d partitions)", name, folder, len(table), table.snapshot_date.nunique())

    if plot:
        plot_monitoring(summary, drift, mv)
    return summary, drift


def run_monitoring_asof(as_of, spark, role=None):
    """Monitor as of `as_of`: role champion = the champion; role challenger = every other model with predictions
    (current challenger and retired models, whose labels keep maturing); no role = all models."""
    reg = load_registry(missing_ok=True)
    champion = (resolve_roles(as_of, reg) or {}).get("champion")
    if role == "champion":
        if champion is None:
            log.warning("no champion in force on %s - monitoring skipped", as_of)
            return []
        targets = [champion]
    elif role == "challenger":       # current challenger + retired models whose labels are still maturing
        targets = [mv for mv in reg["models"] if mv != champion]
    else:
        targets = list(reg["models"])
    done = []
    for mv in targets:
        folder = os.path.join(PATHS["predictions_root"], mv)
        if not glob.glob(os.path.join(folder, "*.parquet")):
            continue
        first = min(os.path.basename(f)[-18:-8].replace("_", "-") for f in glob.glob(os.path.join(folder, "*.parquet")))
        summary, _ = run_monitoring(first, as_of, mv + ".pkl", spark, as_of=as_of, plot=False)
        if summary is not None:
            done.append(mv)
    if not done:
        log.info("monitoring as of %s: no model has predictions yet", as_of)
    return done


def plot_monitoring(summary, drift, mv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = pd.to_datetime(summary.snapshot_date).dt.strftime("%Y-%m")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(x, summary.auc, marker="o", label="AUC")
    axes[0].axhline(summary.reference_auc.iat[0], color="k", ls="--", lw=0.8, label="reference (test AUC)")
    axes[0].axhline(summary.reference_auc.iat[0] - MONITORING["auc_drop_alert"], color="tab:red", ls=":", lw=0.8,
                    label="alert threshold")
    axes[0].set_title(f"{mv} - performance by scoring month (labels mature 6 months later)"); axes[0].legend()
    axes[1].plot(x, summary.score_psi, marker="o", label="score PSI")
    axes[1].plot(x, summary.max_feature_psi, marker="s", label="max feature PSI")
    for v, c in [(MONITORING["psi_watch"], "tab:orange"), (MONITORING["psi_alert"], "tab:red")]:
        axes[1].axhline(v, color=c, ls=":", lw=0.8)
    axes[1].set_title("stability vs train baseline (orange = watch, red = alert)"); axes[1].legend()
    axes[2].plot(x, summary.mean_predicted_pd, marker="o", label="mean predicted PD")
    axes[2].plot(x, summary.observed_default_rate, marker="s", label="observed default rate")
    axes[2].set_title("calibration over time"); axes[2].legend()
    axes[2].tick_params(axis="x", rotation=60)
    plt.tight_layout()
    path = os.path.join(PATHS["monitoring_root"], mv, f"{mv}_monitoring_report.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log.info("saved monitoring chart: %s", path)


# =====================================================================================================================
# DASHBOARD (visualise performance + stability across time)
# =====================================================================================================================
def _read_gold(root):
    files = glob.glob(os.path.join(root, "*", "*.parquet"))
    if not files:
        return pd.DataFrame()
    t = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    t["snapshot_date"] = pd.to_datetime(t["snapshot_date"])
    return t


def run_report(as_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mon, drift = _read_gold(PATHS["monitoring_root"]), _read_gold(PATHS["feature_drift_root"])
    reg, gov = load_registry(missing_ok=True), read_governance_log()
    os.makedirs(PATHS["reports_root"], exist_ok=True)
    if mon.empty:
        log.info("report as of %s: no monitoring data yet (cold start) - nothing to plot", as_of)
        return None
    mon = mon[mon.snapshot_date <= pd.Timestamp(as_of)]
    drift = drift[drift.snapshot_date <= pd.Timestamp(as_of)] if not drift.empty else drift
    for c in ["auc", "gini", "score_psi", "max_feature_psi", "mean_predicted_pd", "observed_default_rate",
              "reference_auc", "n_scored", "n_labelled"]:
        mon[c] = pd.to_numeric(mon[c], errors="coerce")
    prod = mon[mon.model_role == "champion"].sort_values("snapshot_date")
    shadow = mon[mon.model_role == "challenger"].sort_values("snapshot_date")

    fig, axes = plt.subplots(5, 1, figsize=(13, 22), sharex=True)
    first_month = mon.snapshot_date.min() if not mon.empty else pd.Timestamp(PIPELINE_DEFAULTS["start_date"])
    months = pd.date_range(first_month, as_of, freq="MS")      # axis starts at the first scored month
    colours = {mv: plt.cm.tab10(i % 10) for i, mv in enumerate(sorted(mon.model_version.unique()))}

    ax = axes[0]                                                   # production performance
    for mv, g in prod.groupby("model_version"):
        ax.plot(g.snapshot_date, g.auc, "o-", color=colours[mv], label=f"{mv} (champion)")
        ax.hlines(g.reference_auc.iat[0], g.snapshot_date.min(), g.snapshot_date.max(), colors=colours[mv],
                  linestyles="--", lw=0.8)
        ax.hlines(g.reference_auc.iat[0] - MONITORING["auc_drop_alert"], g.snapshot_date.min(), g.snapshot_date.max(),
                  colors="tab:red", linestyles=":", lw=0.8)
    ax.set_title(f"Production model performance by scoring month (as of {as_of}; labels mature 6 months later)\n"
                 "dashed = champion's test AUC reference, red dotted = alert threshold")
    ax.set_ylabel("AUC"); ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]                                                   # champion vs challenger (shadow)
    for (mv, role), g in mon.sort_values("snapshot_date").groupby(["model_version", "model_role"]):
        ax.plot(g.snapshot_date, g.gini, "o-" if role == "champion" else "s--", color=colours[mv],
                label=f"{mv} ({role})", alpha=1 if role == "champion" else 0.7)
    ax.set_title("Champion vs challenger (shadow) Gini on the same scoring months"); ax.set_ylabel("Gini")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]                                                   # stability
    ax.plot(prod.snapshot_date, prod.score_psi, "o-", label="score PSI (champion)")
    ax.plot(prod.snapshot_date, prod.max_feature_psi, "s-", label="max feature PSI (champion)")
    ax.axhline(MONITORING["psi_watch"], color="tab:orange", ls=":", lw=1, label="watch 0.10")
    ax.axhline(MONITORING["psi_alert"], color="tab:red", ls=":", lw=1, label="alert 0.25")
    ax.set_title("Stability vs each model's training baseline"); ax.set_ylabel("PSI"); ax.legend(fontsize=8)

    ax = axes[3]                                                   # calibration + volume
    ax.plot(prod.snapshot_date, prod.mean_predicted_pd, "o-", label="mean predicted PD")
    ax.plot(prod.snapshot_date, prod.observed_default_rate, "s-", label="observed default rate (matured)")
    ax2 = ax.twinx(); ax2.bar(prod.snapshot_date, prod.n_scored, width=20, alpha=0.15, color="gray")
    ax2.set_ylabel("loans scored")
    ax.set_title("Calibration over time (bars = volume scored)"); ax.set_ylabel("rate"); ax.legend(fontsize=8)

    ax = axes[4]                                                   # governance timeline
    ax.set_ylim(0, 4); ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["decision", "training", "champion"])
    if not gov.empty:
        g = gov[pd.to_datetime(gov.as_of_date) <= pd.Timestamp(as_of)]
        style = {"cold_start": ("lightgray", "."), "healthy": ("tab:green", "."), "cooldown": ("tab:orange", "x"),
                 "initial": ("tab:blue", "^"), "scheduled": ("tab:purple", "^"), "triggered": ("tab:red", "^")}
        for _, r in g[g.event == "decision"].iterrows():
            c, mk = style.get(r.trigger, ("k", "."))
            ax.scatter(pd.Timestamp(r.as_of_date), 1, color=c, marker=mk, s=60)
        trainings = g[g.event == "training"].sort_values("as_of_date")
        for k, (_, r) in enumerate(trainings.iterrows()):
            ax.scatter(pd.Timestamp(r.as_of_date), 2, color="tab:green" if r.promoted == "True" else "tab:red",
                       marker="o", s=70)
            ax.annotate(("promoted " if r.promoted == "True" else "rejected ") + r.best_candidate.replace("credit_model_", ""),
                        (pd.Timestamp(r.as_of_date), 2), textcoords="offset points",
                        xytext=(0, 10 if k % 2 == 0 else -18), fontsize=7, ha="center")
    hist = sorted([h for h in reg["role_history"] if h["effective_from"] <= as_of], key=lambda h: h["effective_from"])
    for i, h in enumerate(hist):
        start = max(pd.Timestamp(h["effective_from"]), months[0])
        end = pd.Timestamp(hist[i + 1]["effective_from"]) if i + 1 < len(hist) else pd.Timestamp(as_of)
        ax.barh(3, (end - start).days or 15, left=start, height=0.5, color=colours.get(h["champion"], "gray"),
                alpha=0.6)
        if i == 0 or hist[i - 1]["champion"] != h["champion"]:
            ax.text(start, 3.35 if i % 2 == 0 else 2.45, h["champion"].replace("credit_model_", ""), fontsize=7)
        for a2 in axes[:4]:
            a2.axvline(pd.Timestamp(h["effective_from"]), color="k", lw=0.6, ls="-.", alpha=0.5)
    ax.set_title("Governance timeline: refresh decisions (green=healthy, grey=cold start, triangles=retrain), "
                 "training outcomes, champion in force")
    ax.set_xlim(months[0] - pd.Timedelta(days=15), pd.Timestamp(as_of) + pd.Timedelta(days=15))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = os.path.join(PATHS["reports_root"], f"model_dashboard_asof_{as_of.replace('-', '_')}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    fig.savefig(os.path.join(PATHS["reports_root"], "model_dashboard_latest.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    if not drift.empty:                                            # feature drift heatmap for the current champion
        in_force = prod[["snapshot_date", "model_version"]].drop_duplicates()   # champion that scored each month
        d = drift.merge(in_force, on=["snapshot_date", "model_version"], how="inner")
        if not d.empty:
            d = d.assign(psi=pd.to_numeric(d.psi, errors="coerce"))
            pv = d.pivot_table(index="feature", columns=d.snapshot_date.dt.strftime("%Y-%m"), values="psi")
            fig, ax = plt.subplots(figsize=(max(7.0, 3 + 0.7 * pv.shape[1]), 0.45 * pv.shape[0] + 1.5))
            im = ax.imshow(pv.values, cmap="Reds", vmin=0, vmax=MONITORING["psi_alert"], aspect="auto")
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=60, fontsize=8)
            ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index, fontsize=8)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    if not np.isnan(pv.values[i, j]):
                        ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, fraction=0.03)
            ax.set_title("Feature PSI vs the training baseline of the champion in force each month", fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(PATHS["reports_root"], f"feature_drift_asof_{as_of.replace('-', '_')}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close(fig)

    table = mon.sort_values(["model_version", "snapshot_date"])
    table.to_csv(os.path.join(PATHS["reports_root"], f"monitoring_asof_{as_of.replace('-', '_')}.csv"), index=False)
    log_table(f"production (champion) monitoring as of {as_of}:", prod[["snapshot_date", "model_version", "n_scored",
              "n_labelled", "auc", "gini", "score_psi", "max_feature_psi", "overall_status"]])
    log.info("saved dashboard: %s", path)
    return path


# =====================================================================================================================
# PIPELINE
# =====================================================================================================================
def run_pipeline(model_train_date, startdate, enddate, xgb_n_iter, include_challenger, spark):
    log.info("=" * 30 + " 1/3 TRAIN + SELECT " + "=" * 30)
    run_training(model_train_date, xgb_n_iter, spark, split_source="store", trigger="pipeline")
    models = resolve_models(None, include_challenger)
    log.info("=" * 30 + " 2/3 INFERENCE (BACKFILL) " + "=" * 30)
    run_backfill(startdate, enddate, models, spark)
    log.info("=" * 30 + " 3/3 MONITORING " + "=" * 30)
    for m in models:
        run_monitoring(startdate, enddate, m, spark)


# =====================================================================================================================
# CLI
# =====================================================================================================================
MODES = ["pipeline", "bootstrap", "check_data", "decide", "train", "select", "inference", "monitor", "report", "promote",
         "backfill"]
NEEDS_SPARK = {"pipeline", "bootstrap", "train", "select", "inference", "monitor", "backfill"}


def cli(argv=None):
    p = argparse.ArgumentParser(description="credit model processing steps (orchestrated by Airflow dags/dag.py)")
    p.add_argument("--mode", choices=MODES, default="pipeline",
                   help="default 'pipeline' = standalone one-shot: train -> select -> backfill -> monitor")
    p.add_argument("--snapshotdate", help="the DAG run's logical date {{ ds }} (train date for train/select)")
    p.add_argument("--modelname", nargs="+", help="artefact file(s) in model_bank/; default resolved from registry")
    p.add_argument("--algo", choices=["logreg", "xgb", "all"], default="all", help="train: which algorithm")
    p.add_argument("--role", choices=["champion", "challenger"], help="inference / monitor: model role")
    p.add_argument("--store", choices=["all", "feature", "label"], default="all", help="check_data: which gold store")
    p.add_argument("--startdate", default=PIPELINE_DEFAULTS["start_date"], help="pipeline/backfill start")
    p.add_argument("--enddate", default=PIPELINE_DEFAULTS["end_date"], help="pipeline/backfill end")
    p.add_argument("--split_source", choices=["store", "dates"], default="store",
                   help="train/select: 'store' = gold data_split column, 'dates' = as-of split by label date")
    p.add_argument("--trigger", help="train/select: reason recorded in the registry (default: today's decision)")
    p.add_argument("--include_challenger", action="store_true", help="pipeline/backfill: also the challenger")
    p.add_argument("--effective_from", help="select: date the new champion goes live (default: the train date)")
    p.add_argument("--model_train_date", default=PIPELINE_DEFAULTS["model_train_date"],
                   help="bootstrap: train date of the initial model (default: the notebook's 2025-06-01)")
    p.add_argument("--xgb_n_iter", type=int, default=int(os.environ.get("XGB_N_ITER", 30)),
                   help="random-search iterations for XGBoost (env XGB_N_ITER)")
    args = p.parse_args(argv)
    if args.mode in ("pipeline", "train") and not args.snapshotdate:
        args.snapshotdate = PIPELINE_DEFAULTS["model_train_date"]
    if args.mode in ("bootstrap", "check_data", "decide", "select", "inference", "report", "promote") \
            and not args.snapshotdate:
        p.error(f"--snapshotdate is required for {args.mode}")

    setup_logging(args.mode)
    log.info("---starting job--- mode=%s | args=%s", args.mode, vars(args))
    log.info("versions: python %s | sklearn %s | xgboost %s | pyspark %s", sys.version.split()[0],
             sklearn.__version__, xgb.__version__, pyspark.__version__)
    spark = get_spark() if args.mode in NEEDS_SPARK else None
    decision = None
    try:
        if args.mode == "check_data":
            run_check_data(args.snapshotdate, args.store)
        elif args.mode == "decide":
            decision = decide_refresh(args.snapshotdate)
        elif args.mode == "train":
            algos = ("logreg", "xgb") if args.algo == "all" else (args.algo,)
            trigger = args.trigger or (latest_decision(args.snapshotdate) or {}).get("trigger", "manual")
            run_training(args.snapshotdate, args.xgb_n_iter, spark, args.split_source, trigger, algos,
                         register=args.algo == "all")
        elif args.mode == "select":
            run_selection(args.snapshotdate, spark, args.split_source, args.trigger, args.effective_from)
        elif args.mode == "inference":
            if args.modelname:
                for m in args.modelname:
                    run_inference(args.snapshotdate, m, spark, role="adhoc")
            else:
                run_inference_role(args.snapshotdate, args.role or "champion", spark)
        elif args.mode == "monitor":
            if args.snapshotdate and not args.modelname:
                run_monitoring_asof(args.snapshotdate, spark, args.role)
            else:
                for m in resolve_models(args.modelname, args.include_challenger):
                    run_monitoring(args.startdate, args.enddate, m, spark)
        elif args.mode == "report":
            run_report(args.snapshotdate)
        elif args.mode == "bootstrap":
            bootstrap_champion(args.snapshotdate, args.model_train_date, args.xgb_n_iter, spark)
        elif args.mode == "promote":
            if not args.modelname:
                p.error("--modelname is required for promote")
            promote_manual(args.modelname[0], args.snapshotdate)
        elif args.mode == "backfill":
            run_backfill(args.startdate, args.enddate, resolve_models(args.modelname, args.include_challenger), spark)
        elif args.mode == "pipeline":
            run_pipeline(args.snapshotdate, args.startdate, args.enddate, args.xgb_n_iter,
                         args.include_challenger, spark)
    except Exception:
        log.exception("job failed")
        raise
    finally:
        if spark is not None:
            spark.stop()
    log.info("---completed job---")
    if decision is not None:                       # last stdout line -> Airflow XCom for the branch operator
        for h in log.handlers:
            h.flush()
        print("DECISION=" + json.dumps({k: decision[k] for k in ("decision", "trigger", "detail")}), flush=True)


if __name__ == "__main__":
    # Re-import as module `main` so pickled classes are stored as main.QuantileCapper etc. (not __main__.*),
    # which lets notebooks load the artefacts with a plain `import main`.
    import main as _main
    _main.cli()
