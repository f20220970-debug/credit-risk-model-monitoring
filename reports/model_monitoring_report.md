# Model Monitoring Report: Lending Club PD Scorecard

## 1. Purpose and scope
This report monitors a probability-of-default (PD) scorecard for 36-month unsecured
consumer loans. The model was developed on loans issued in 2012-2013
and is monitored on each quarterly vintage issued in 2014-2015.
The aim is to check that the model stays stable, keeps ranking risk correctly, and stays
calibrated, and to say what action each result requires.

## 2. Data
- Source: Lending Club accepted loans, 2007-2018 (public Kaggle dataset).
- Population: 36-month loans with a final outcome. Default = charged off; good = fully paid.
  Loans still running or late are excluded because their outcome is unknown.
- Features: borrower attributes known at application (income, DTI, FICO, credit history,
  utilisation, enquiries, delinquencies, home ownership, verification, purpose).
  Lending Club's own grade and interest rate are excluded because they are outputs of
  Lending Club's risk model.

## 3. Model methodology
1. Development loans were split 70/30 into training and hold-out samples (stratified on default).
2. Numeric features were cut into up to 10 quantile bins; categorical features kept
   categories above 1% of loans and grouped the rest as "Other";
   missing values got their own bin.
3. Weight of evidence (WoE) was computed per bin and information value (IV) per feature.
   Features with IV below 0.02 were dropped, and of any two WoE features correlated
   above 0.7, the one with lower IV was dropped.
4. A logistic regression was fitted on the WoE values. All coefficients are expected to be
   negative, since a higher WoE means a safer bin.
5. Probabilities were scaled to scorecard points: 600 points at 50:1
   good:bad odds, with odds doubling every 20 points.

Selected features:

| feature | iv | strength |
| --- | --- | --- |
| fico_range_low | 0.123 | medium |
| annual_inc | 0.09 | weak |
| dti | 0.039 | weak |
| home_ownership | 0.033 | weak |
| revol_util | 0.027 | weak |
| purpose | 0.025 | weak |
| credit_hist_years | 0.022 | weak |
| inq_last_6mths | 0.021 | weak |

Coefficients:

| feature | coefficient | p_value |
| --- | --- | --- |
| const | -1.9286 | 0.0 |
| fico_range_low | -0.7836 | 0.0 |
| annual_inc | -0.7461 | 0.0 |
| dti | -0.6378 | 0.0 |
| home_ownership | -0.4449 | 0.0 |
| revol_util | -0.2959 | 0.0 |
| purpose | -0.9818 | 0.0 |
| credit_hist_years | -0.2667 | 0.0001 |
| inq_last_6mths | -1.116 | 0.0 |

Development hold-out performance: AUC 0.649, Gini 0.299, KS 0.218.

## 4. Monitoring framework

| Check | Metric | Green | Amber | Red |
| --- | --- | --- | --- | --- |
| Population stability | PSI on score bands (development deciles) | below 0.1 | 0.1 to 0.25 | above 0.25 |
| Feature stability | CSI on each feature's WoE bins | below 0.1 | 0.1 to 0.25 | above 0.25 |
| Discrimination | Gini as a share of development Gini | 90% or more | 80% to 90% | below 80% |
| Calibration | Actual default rate vs mean predicted PD | within 10% | 10% to 20% | more than 20% apart |

KS and decile-level calibration are tracked alongside for context. A quarter's overall
status is its worst individual status.

## 5. Results

| period | loans | default_rate | mean_predicted_pd | auc | gini | ks | psi | psi_rag | gini_rag | calibration_rag | overall_rag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Development hold-out | 43168 | 12.7% | 12.67% | 0.65 | 0.299 | 0.218 | 0.0 | Green | Green | Green | Green |
| 2014Q1 | 34074 | 12.87% | 12.86% | 0.633 | 0.265 | 0.19 | 0.001 | Green | Amber | Green | Amber |
| 2014Q2 | 37881 | 13.55% | 12.96% | 0.641 | 0.282 | 0.207 | 0.004 | Green | Green | Green | Green |
| 2014Q3 | 40595 | 13.63% | 12.93% | 0.644 | 0.289 | 0.214 | 0.002 | Green | Green | Green | Green |
| 2014Q4 | 50020 | 14.53% | 13.19% | 0.638 | 0.276 | 0.204 | 0.009 | Green | Green | Amber | Amber |
| 2015Q1 | 56568 | 14.83% | 13.02% | 0.64 | 0.28 | 0.203 | 0.005 | Green | Green | Amber | Amber |
| 2015Q2 | 64222 | 15.38% | 12.85% | 0.638 | 0.277 | 0.197 | 0.001 | Green | Green | Amber | Amber |
| 2015Q3 | 73567 | 14.58% | 12.49% | 0.646 | 0.291 | 0.21 | 0.002 | Green | Green | Amber | Amber |
| 2015Q4 | 88669 | 14.82% | 12.46% | 0.64 | 0.28 | 0.205 | 0.003 | Green | Green | Amber | Amber |

Features driving instability (top two CSI per quarter, Amber or Red only):

_No feature breached the CSI thresholds._

## 6. Findings
- Quarters monitored: 8 (2014Q1 to 2015Q4).
- Overall status: 2 Green, 6 Amber, 0 Red.
- PSI ranged from 0.001 to 0.009; highest in 2014Q4.
- Gini ranged from 0.265 to 0.291 against 0.299 in development.
- Actual default rate ranged from 12.87% to 15.38%; predicted PD averaged 12.85%.

### Analyst commentary
<!-- ANALYST COMMENTARY -->
Six of the eight quarters were Amber and none breached a Red threshold on a single check. 2014Q1 was Amber on discrimination only: Gini fell to 0.265, 89% of the development Gini of 0.299, and recovered to 92-97% in every later quarter. From 2014Q4 onward the model under-predicted defaults by 10-20%: actual default rates rose from 12.9% to as high as 15.4% while the mean predicted PD stayed near 12.5-13.2%. PSI stayed below 0.01 and CSI was Green for every feature in every quarter, so the drift is not explained by a change in the applicant population; it more likely reflects factors outside the model's inputs, such as Lending Club's rapid volume growth and looser underwriting in 2014-2015. Because calibration was Amber for five consecutive quarters, the two-consecutive-Ambers rule makes this a Red escalation. Since the model still ranks risk well, I recommend recalibrating the PD levels (re-estimating the intercept on recent vintages) rather than redeveloping the model, and applying a temporary upward PD overlay until the recalibrated model is validated.
<!-- ANALYST COMMENTARY -->

## 7. Escalation steps

| Status | Action | Owner | Timeline |
| --- | --- | --- | --- |
| Green | No action. Continue routine quarterly monitoring. | Monitoring analyst | Next cycle |
| Amber | Investigate the driver (CSI by feature, segment default rates). Document the finding and watch the next quarter closely. | Monitoring analyst, model owner informed | Within the current cycle |
| Red | Escalate to the model owner and model risk. Do a root-cause analysis and decide between a temporary overlay, recalibration, or redevelopment. | Model owner, model risk | Immediately; decision within one cycle |

Two consecutive Amber quarters on the same check are treated as Red.

## 8. Limitations
- Lending Club data covers one lender's approved loans only, so the model never sees declined applicants.
- Outcomes are taken at the 2018 data snapshot; a very small number of loans may still have changed status later.
- Quarters are monitored retrospectively on historical data to simulate a live monitoring cycle.
