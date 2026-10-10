"""Step 1: filter the raw Lending Club file down to what the project needs.

Keeps 36-month loans issued 2012-2015 with a known outcome, adds a default
flag and a few derived features, and saves a small parquet file.
"""
import numpy as np
import pandas as pd

import config as C

RAW_COLUMNS = [
    "issue_d", "term", "loan_status", "loan_amnt", "annual_inc", "dti",
    "fico_range_low", "revol_util", "revol_bal", "inq_last_6mths",
    "delinq_2yrs", "open_acc", "total_acc", "pub_rec", "emp_length",
    "earliest_cr_line", "home_ownership", "verification_status", "purpose",
]


def emp_length_to_years(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing stays missing."""
    s = s.astype("string").str.replace("< 1", "0", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")


def main() -> None:
    years = set(C.DEV_YEARS + C.MONITOR_YEARS)
    raw = C.RAW_FILE
    if not raw.exists():   # macOS sometimes unzips the .gz automatically; accept the plain .csv too
        raw = raw.with_suffix("")
    if not raw.exists():
        raise SystemExit(f"Raw file not found. Put accepted_2007_to_2018Q4.csv.gz (or .csv) in {C.RAW_FILE.parent}")
    keep = []
    # Read in chunks so the 1.6 GB file fits in laptop memory
    for chunk in pd.read_csv(raw, usecols=RAW_COLUMNS, chunksize=200_000,
                             low_memory=False):
        chunk = chunk[chunk["term"].astype(str).str.strip() == "36 months"]
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        chunk = chunk[chunk["issue_d"].dt.year.isin(years)]
        chunk = chunk[chunk["loan_status"].isin(C.BAD_STATUSES + C.GOOD_STATUSES)]
        keep.append(chunk)
    df = pd.concat(keep, ignore_index=True)

    df["default"] = df["loan_status"].isin(C.BAD_STATUSES).astype(int)
    df["emp_length_years"] = emp_length_to_years(df["emp_length"])
    first_line = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_hist_years"] = ((df["issue_d"] - first_line).dt.days / 365.25).round(1)
    df["quarter"] = df["issue_d"].dt.year.astype(str) + "Q" + df["issue_d"].dt.quarter.astype(str)

    cols = ["issue_d", "quarter", "default"] + C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
    df = df[cols].sort_values("issue_d").reset_index(drop=True)

    C.PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.PROCESSED_FILE, index=False)

    print(f"Saved {len(df):,} loans to {C.PROCESSED_FILE.relative_to(C.ROOT)}")
    print(df.groupby(df["issue_d"].dt.year)["default"].agg(loans="size", default_rate="mean").round(4))


if __name__ == "__main__":
    main()
