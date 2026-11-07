"""
Simulate new monthly data arriving after the historical dataset, so the inference pipeline has genuinely
unseen applications to score and the monitoring has fresh labels to evaluate.

How: for every new application month, resample real applications from the most recent year of LABELLED history
(customers whose loan reached month-on-book 6 in the data, so their outcome is known), copy their attributes,
financials and full repayment history, give them new IDs and shift every date to the new month.
Repayment behaviour is copied with the customer, so the label (30+ dpd at month 6) stays realistic. An optional,
gradually growing drift is applied to a few financial inputs from --drift_start, so stability monitoring has
something to detect.

Writes data/incoming/<source>.csv; the original files in data/ are never modified. Re-running replaces the files.

    cd /opt/airflow/scripts && python3 generate_synthetic_data.py                        # defaults below
    python3 generate_synthetic_data.py --start 2025-02-01 --months 16 --per_month 500 --drift gradual
    python3 generate_synthetic_data.py --drift none
"""
import argparse
import os

import numpy as np
import pandas as pd

RAW = "data/"
INCOMING = "data/incoming/"
FILES = {"attributes": "features_attributes.csv", "financials": "features_financials.csv",
         "loans": "lms_loan_daily.csv"}

# per month of drift k (1, 2, 3, ...): how much riskier the incoming population becomes
DRIFT = {
    "none": {},
    "gradual": {"Outstanding_Debt": ("mul", 0.04), "Interest_Rate": ("add", 0.6),
                "Num_Credit_Inquiries": ("add", 0.4), "Delay_from_due_date": ("add", 1.0)},
    "strong": {"Outstanding_Debt": ("mul", 0.10), "Interest_Rate": ("add", 1.5),
               "Num_Credit_Inquiries": ("add", 1.0), "Delay_from_due_date": ("add", 2.5)},
}


def read(name):
    return pd.read_csv(RAW + FILES[name], dtype=str, keep_default_na=False)


def months_between(a, b):
    return (a.year - b.year) * 12 + a.month - b.month


def apply_drift(fin, k, spec):
    """k-th month of drift; only values that parse as numbers are changed (raw messy text is kept)."""
    for col, (how, step) in spec.items():
        num = pd.to_numeric(fin[col].str.strip("_"), errors="coerce")
        new = num * (1 + step * k) if how == "mul" else num + step * k
        if col in ("Interest_Rate", "Num_Credit_Inquiries", "Delay_from_due_date"):
            new = new.round()
            fin[col] = fin[col].where(num.isna(), new.map(lambda v: str(int(v)) if pd.notna(v) else ""))
        else:
            fin[col] = fin[col].where(num.isna(), new.map(lambda v: f"{v:.2f}" if pd.notna(v) else ""))
    return fin


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None,
                   help="first new application month (default: the month after the last real application)")
    p.add_argument("--months", type=int, default=16, help="number of new application months")
    p.add_argument("--per_month", type=int, default=500, help="applications per month (history: ~500)")
    p.add_argument("--pool_months", type=int, default=12, help="resample from the last N labelled loan months")
    p.add_argument("--label_mob", type=int, default=6, help="month-on-book at which the label is observed")
    p.add_argument("--drift", choices=list(DRIFT), default="gradual")
    p.add_argument("--drift_start", default="2025-09-01", help="first month with drift")
    p.add_argument("--seed", type=int, default=611)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    attr, fin, loans = read("attributes"), read("financials"), read("loans")
    # only customers whose outcome is known: their loan reached month-on-book `label_mob` within the data
    mob = pd.to_numeric(loans["installment_num"], errors="coerce")
    starts = pd.to_datetime(loans.groupby("Customer_ID")["loan_start_date"].first())
    labelled = starts[mob.groupby(loans["Customer_ID"]).max().reindex(starts.index) >= a.label_mob]
    last_labelled = labelled.max()
    pool_start = last_labelled - pd.DateOffset(months=a.pool_months - 1)
    pool = labelled[labelled >= pool_start].index
    pool = pool[pool.isin(fin["Customer_ID"]) & pool.isin(attr["Customer_ID"])].to_numpy()
    print(f"repayment data: loans start up to {starts.max():%Y-%m}; outcome at month {a.label_mob} known for loans "
          f"up to {last_labelled:%Y-%m}")
    print(f"resampling {len(pool)} labelled customers whose loans started {pool_start:%Y-%m}..{last_labelled:%Y-%m}")
    last_application = max(pd.to_datetime(fin["snapshot_date"]).max(), starts.max())
    if a.start is None:      # new months follow straight on from the real data: no gap, no overlap
        a.start = (last_application + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    elif pd.Timestamp(a.start) <= last_application:
        print(f"WARNING: --start {a.start} overlaps real applications (last real month {last_application:%Y-%m})")
    print(f"last real application month {last_application:%Y-%m}; new months start {a.start}")

    attr_by, fin_by = attr.set_index("Customer_ID"), fin.set_index("Customer_ID")
    loans["_snap"] = pd.to_datetime(loans["snapshot_date"])
    loans["_start"] = pd.to_datetime(loans["loan_start_date"])
    loans_by = {c: g for c, g in loans.groupby("Customer_ID")}

    out_attr, out_fin, out_loans = [], [], []
    start, drift_start = pd.Timestamp(a.start), pd.Timestamp(a.drift_start)
    for m in range(a.months):
        month = start + pd.DateOffset(months=m)
        picks = rng.choice(pool, size=a.per_month, replace=True)
        ids = [f"SYN_{month:%Y%m}_{i:05d}" for i in range(a.per_month)]
        tag = month.strftime("%Y-%m-%d")

        at = attr_by.loc[picks].reset_index(drop=True)
        at.insert(0, "Customer_ID", ids)
        at["snapshot_date"] = tag
        fi = fin_by.loc[picks].reset_index(drop=True)
        fi.insert(0, "Customer_ID", ids)
        fi["snapshot_date"] = tag
        k = months_between(month, drift_start) + 1
        if DRIFT[a.drift] and k > 0:
            fi = apply_drift(fi, k, DRIFT[a.drift])
        out_attr.append(at)
        out_fin.append(fi)

        for new_id, src in zip(ids, picks):
            g = loans_by[src].copy()
            shift = months_between(month, g["_start"].iloc[0])
            g["Customer_ID"] = new_id
            g["loan_id"] = f"{new_id}_{month:%Y_%m_%d}"
            g["loan_start_date"] = tag
            g["snapshot_date"] = (g["_snap"] + pd.DateOffset(months=shift)).dt.strftime("%Y-%m-%d")
            out_loans.append(g.drop(columns=["_snap", "_start"]))
        print(f"  {tag}: {a.per_month} applications" + (f" | drift month {k}" if DRIFT[a.drift] and k > 0 else ""))

    os.makedirs(INCOMING, exist_ok=True)
    for name, frames, cols in [("attributes", out_attr, attr.columns), ("financials", out_fin, fin.columns),
                               ("loans", out_loans, [c for c in loans.columns if not c.startswith("_")])]:
        df = pd.concat(frames, ignore_index=True)[list(cols)]
        df.to_csv(INCOMING + FILES[name], index=False)
        print(f"wrote {INCOMING + FILES[name]}: {len(df)} rows")


if __name__ == "__main__":
    main()
