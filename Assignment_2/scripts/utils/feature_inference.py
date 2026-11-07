"""
Build model-ready features for NEW applications (no label needed) by APPLYING the parameters that
process_features_gold_table_main fitted on its training split (saved as feature_params_<train_date>.pkl).

Mirrors, step by step, the transformations in utils/data_processing_lms_gold_table.py:
Changed_Credit_Limit imputation -> Age cleaning/imputation -> Age_Bin -> bank-account / credit-card cleaning
and imputation -> ratio features -> drop the same columns. Spark semantics are kept where they differ from
pandas: null join keys never match, and rounding is half-up (Spark F.round), not banker's rounding.
"""
import numpy as np
import pandas as pd


def _round_half_up(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return np.sign(s) * np.floor(np.abs(s) + 0.5)


def _lookup(keys, table, key_cols, value_col):
    """Left-join `value_col` from `table` on key_cols; rows with any null key get NaN (Spark join semantics)."""
    t = table[key_cols + [value_col]].dropna(subset=key_cols).drop_duplicates(subset=key_cols)
    out = keys[key_cols].merge(t, on=key_cols, how="left")[value_col]
    out.index = keys.index
    out[keys[key_cols].isna().any(axis=1)] = np.nan
    return out


def _bin(values, edges, labels):
    """create_age_bin_column: first bin [e0, e1], then (e_i, e_i+1]; outside -> None."""
    out = pd.Series(None, index=values.index, dtype=object)
    if edges is None or len(edges) < 2:
        return out
    v = pd.to_numeric(values, errors="coerce")
    for i, label in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        m = ((v >= lo) if i == 0 else (v > lo)) & (v <= hi) & out.isna()
        out[m] = label
    return out


def _impute_age(df, p):
    age = pd.to_numeric(df["Age"], errors="coerce")
    df["Age_clean"] = age.where(age.between(18, 100))
    chay = pd.to_numeric(df["Credit_History_Age_Years"], errors="coerce")
    inconsistent = chay.notna() & df["Age_clean"].notna() & (chay > df["Age_clean"])
    df["Age_History_Inconsistent"] = inconsistent.astype(int)
    df.loc[inconsistent, "Age_clean"] = np.nan
    df["Age_was_invalid"] = (df["Age_clean"].isna() & age.notna()).astype(int)

    income = pd.to_numeric(df["Annual_Income"], errors="coerce")
    log_income = np.log1p(income.clip(lower=0)).where(income.notna())

    # candidates: rows needing imputation x the occupation's (Age_Bin) groups from TRAIN
    need = df["Age_clean"].isna() & df["Occupation"].notna() & income.notna()
    best_age = pd.Series(np.nan, index=df.index)
    strong = pd.Series(False, index=df.index)
    lookup = p["age_income_lookup"].dropna(subset=["Occupation"])
    if need.any() and not lookup.empty:
        cand = (pd.DataFrame({"_row": df.index[need], "Occupation": df.loc[need, "Occupation"].values,
                              "_log_income": log_income[need].values})
                .merge(lookup[["Occupation", "log_income_median", "age_median"]], on="Occupation", how="inner"))
        cand["_distance"] = (cand["log_income_median"] - cand["_log_income"]).abs()
        cand = cand.sort_values(["_row", "_distance"], kind="mergesort")
        cand["_rank"] = cand.groupby("_row").cumcount() + 1
        first = cand[cand["_rank"] == 1].set_index("_row")
        second = cand[cand["_rank"] == 2].set_index("_row")["_distance"]
        gap = second.reindex(first.index) - first["_distance"]
        is_strong = second.reindex(first.index).isna() | (gap >= p["min_distance_gap"])
        best_age[first.index] = first["age_median"].astype(float)
        strong[first.index] = is_strong.values

    occ_med = _lookup(df, p["occupation_age_median"], ["Occupation"], "occupation_age_median")
    imputed = df["Age_clean"].copy()
    imputed = imputed.where(imputed.notna(), best_age.where(strong))
    imputed = imputed.where(imputed.notna(), occ_med)
    imputed = imputed.fillna(p["overall_age_median"])
    df["Age_imputed"] = _round_half_up(imputed).astype("Int64")
    df["Age_imputation_method"] = np.select(
        [df["Age_clean"].notna(), strong, occ_med.notna()],
        ["original", "occupation_income_age_bin", "occupation_median"], default="overall_median")
    df["Age_Bin"] = _bin(df["Age_imputed"].astype(float), p["age_edges_final"], p["age_labels_final"])
    return df


def _impute_accounts(df, raw_col, clean_col, imputed_col, method_col, q):
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    df[clean_col] = raw.where(raw.between(0, q["p95"]))
    df[clean_col.replace("_clean", "_Was_Invalid")] = (df[clean_col].isna() & raw.notna()).astype(int)
    grp = _lookup(df, q["group_stats"], ["Occupation", "Age_Bin"], "median")
    occ = _lookup(df, q["occupation_median"], ["Occupation"], "occupation_median")
    value = df[clean_col].combine_first(grp).combine_first(occ).fillna(q["overall_median"])
    df[imputed_col] = _round_half_up(value).astype("Int64")
    df[method_col] = np.select([df[clean_col].notna(), grp.notna(), occ.notna()],
                               ["original", "occupation_age_bin_median", "occupation_median"],
                               default="overall_median")
    return df


def build_inference_features(financials, attributes, params, snapshot_date_str):
    """financials / attributes: silver rows of ONE application month (pandas). Returns one row per application."""
    fin = financials.rename(columns={"snapshot_date": "financials_snapshot_date"}).copy()
    attr = (attributes.rename(columns={"snapshot_date": "attributes_snapshot_date"})
            [["Customer_ID", "Age", "Occupation", "attributes_snapshot_date"]])
    df = fin.merge(attr, on="Customer_ID", how="left").reset_index(drop=True)
    # same key the label store uses: <Customer_ID>_<loan start yyyy_mm_dd>
    df.insert(0, "loan_id", df["Customer_ID"] + "_" + snapshot_date_str.replace("-", "_"))

    # Changed_Credit_Limit: TRAIN medians by (Credit_Mix, Payment_of_Min_Amount)
    med = _lookup(df, params["changed_credit_limit_medians"], ["Credit_Mix", "Payment_of_Min_Amount"],
                  "Changed_Credit_Limit_Median")
    df["Changed_Credit_Limit"] = pd.to_numeric(df["Changed_Credit_Limit"], errors="coerce").combine_first(med)

    df = _impute_age(df, params)
    df = _impute_accounts(df, "Num_Bank_Accounts", "Bank_Accounts_clean", "Bank_Accounts_imputed",
                          "Bank_Accounts_imputation_method", params["bank"])
    df = _impute_accounts(df, "Num_Credit_Card", "Credit_Cards_clean", "Credit_Cards_imputed",
                          "Credit_Cards_imputation_method", params["card"])

    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    df["Num_Fin_Pdts"] = num("Num_Bank_Accounts") + num("Num_Credit_Card") + num("Num_of_Loan_Cleaned")
    accounts = df["Bank_Accounts_clean"] + df["Credit_Cards_clean"]
    df["Debt_per_Credit_Account"] = (num("Outstanding_Debt") / accounts).where(accounts != 0)
    df["Delayed_Payments_per_Card"] = (num("Num_of_Delayed_Payment") / df["Credit_Cards_clean"]).where(
        df["Credit_Cards_clean"] != 0)
    df["InterestRate_x_Debt"] = num("Interest_Rate") * num("Outstanding_Debt")

    df = df.drop(columns=[c for c in params["columns_to_drop"] if c in df.columns])
    for c in ["Age_imputed", "Bank_Accounts_imputed", "Credit_Cards_imputed"]:
        df[c] = df[c].astype(float)
    df["feature_params_date"] = params["model_train_date_str"]
    return df
