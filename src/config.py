"""All settings for the pipeline live here, so every script uses the same values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
PROCESSED_FILE = ROOT / "data" / "processed" / "loans_36m_2012_2015.parquet"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"

# Time windows
DEV_YEARS = [2012, 2013]          # model development (training) window
MONITOR_YEARS = [2014, 2015]      # 8 quarters monitored after deployment

# Loan outcome: 1 = default (charged off), 0 = fully paid.
# Loans still running or late are dropped because their outcome is unknown.
BAD_STATUSES = ["Charged Off", "Default",
                "Does not meet the credit policy. Status:Charged Off"]
GOOD_STATUSES = ["Fully Paid",
                 "Does not meet the credit policy. Status:Fully Paid"]

# Application-time borrower attributes only. Lending Club's own grade and
# interest rate are left out: they are outputs of LC's risk model, so using
# them would just copy that model instead of building our own.
NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "fico_range_low", "revol_util",
    "revol_bal", "inq_last_6mths", "delinq_2yrs", "open_acc", "total_acc",
    "pub_rec", "emp_length_years", "credit_hist_years",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose"]

# WoE binning and feature selection
N_BINS = 10                # quantile bins per numeric feature
MIN_CATEGORY_SHARE = 0.01  # categories smaller than 1% are merged into "Other"
MIN_IV = 0.02              # drop features with IV below this (no predictive power)
MAX_CORR = 0.70            # drop the weaker of any two WoE features above this correlation

# Scorecard scaling: 600 points at good:bad odds of 50:1, odds double every 20 points
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20

# Red / amber / green thresholds
PSI_AMBER, PSI_RED = 0.10, 0.25          # also used for CSI
GINI_AMBER, GINI_RED = 0.90, 0.80        # Gini as a share of development Gini
CALIB_AMBER, CALIB_RED = 0.10, 0.20      # |actual / predicted - 1|
