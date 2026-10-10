# Credit Risk Model Monitoring Pipeline

A probability-of-default (PD) scorecard for Lending Club 36-month loans, with an automated
quarterly monitoring workflow that tracks stability, discrimination and calibration, flags
each check Red / Amber / Green, and feeds a Tableau dashboard.

**Dashboard:** [Tableau Public: Credit Risk Model Monitoring](https://public.tableau.com/views/CreditRiskModelMonitoring/Dashboard)
**Report:** [reports/model_monitoring_report.md](reports/model_monitoring_report.md)

## What it does
1. **Scorecard (2012-2013 loans).** Bins each borrower attribute, computes weight of evidence (WoE)
   and information value (IV), keeps features with IV of 0.02 or more, and fits a logistic regression
   on the WoE values. Probabilities are scaled to scorecard points (600 at 50:1 odds, 20 points to double the odds).
2. **Monitoring (8 quarters, 2014-2015).** Scores each quarterly vintage and computes:
   - PSI on the score distribution and CSI on each feature (stability)
   - AUC, Gini and KS (discrimination)
   - actual vs predicted default rate, overall and by decile (calibration)
3. **Red / Amber / Green status.** PSI and CSI: below 0.10 green, 0.10-0.25 amber, above 0.25 red.
   Gini: 90% or more of development Gini green, 80-90% amber, below 80% red.
   Calibration: actual within 10% of predicted green, 10-20% amber, beyond 20% red.
4. **Automation.** A GitHub Actions workflow re-runs monitoring and the report on a monthly schedule
   (and on demand) and commits the refreshed outputs.

## Key results
- 589,488 Lending Club 36-month loans; scorecard trained on 143,892 loans (2012-2013) with 8 of 16 features selected by IV.
- Hold-out Gini 0.299 (AUC 0.650, KS 0.218).
- PSI stayed below 0.01 and CSI was green for every feature in all 8 quarters, yet the model under-predicted defaults by 10-20% from 2014Q4 (actual default rate up to 15.4% vs about 12.5% predicted).
- Diagnosis: calibration drift, not population shift. Escalated to Red under the two-consecutive-Ambers rule; recommendation is to recalibrate the PD levels rather than redevelop the model.

## Run it yourself
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Raw data: Kaggle "Lending Club" (wordsforthewise), accepted_2007_to_2018Q4.csv.gz -> data/raw/
python3 src/prepare_data.py       # filter to 36-month loans 2012-2015, add default flag
python3 src/train.py              # WoE/IV, feature selection, logistic regression, scorecard
python3 src/monitor.py            # quarterly PSI, CSI, AUC/Gini, KS, calibration + RAG
python3 src/report.py             # writes reports/model_monitoring_report.md
```

## Project structure
```
src/config.py          settings: time windows, features, thresholds
src/prepare_data.py    raw data -> data/processed/loans_36m_2012_2015.parquet
src/woe.py             WoE binning and IV
src/metrics.py         PSI/CSI, KS, AUC/Gini, calibration, RAG rules
src/train.py           builds and saves the scorecard (artifacts/scorecard.joblib)
src/monitor.py         quarterly monitoring -> outputs/*.csv and outputs/monitoring.db
src/report.py          monitoring report
sql/example_queries.sql  example SQL on the monitoring database
.github/workflows/model-monitoring.yml  scheduled run
```

## Data
Lending Club accepted loans 2007-2018 (public Kaggle dataset). Only the filtered parquet file is
committed; the raw file is not.
