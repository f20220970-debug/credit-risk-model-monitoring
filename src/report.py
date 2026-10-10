"""Step 4: write reports/model_monitoring_report.md from the monitoring outputs.

The methodology, thresholds and escalation steps are fixed text; the tables and
the "Findings" facts are filled from the latest run. The "Analyst commentary"
section is yours to write by hand: explain the results in your own words.
"""
import pandas as pd

import config as C

COMMENTARY_MARKER = "<!-- ANALYST COMMENTARY -->"


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def keep_commentary(path) -> str:
    """Keep whatever you wrote in the commentary section when the report is regenerated."""
    default = ("_Write 4-6 sentences here in your own words: which quarters breached which "
               "thresholds, the most likely cause (look at the CSI drivers and default "
               "rates), and what you would recommend._")
    if not path.exists():
        return default
    text = path.read_text(encoding="utf-8")
    parts = text.split(COMMENTARY_MARKER)
    return parts[1].strip() if len(parts) == 3 else default


def main() -> None:
    perf = pd.read_csv(C.OUTPUTS / "performance.csv")
    csi = pd.read_csv(C.OUTPUTS / "csi.csv")
    iv = pd.read_csv(C.OUTPUTS / "iv_table.csv")
    coefs = pd.read_csv(C.OUTPUTS / "coefficients.csv")
    out = C.REPORTS / "model_monitoring_report.md"
    commentary = keep_commentary(out)

    dev = perf.iloc[0]
    q = perf.iloc[1:]
    perf_view = perf[["period", "loans", "default_rate", "mean_predicted_pd", "auc", "gini",
                      "ks", "psi", "psi_rag", "gini_rag", "calibration_rag", "overall_rag"]].copy()
    for c in ["default_rate", "mean_predicted_pd"]:
        perf_view[c] = (perf_view[c] * 100).round(2).astype(str) + "%"
    for c in ["auc", "gini", "ks", "psi"]:
        perf_view[c] = perf_view[c].round(3)

    top_csi = (csi.sort_values("csi", ascending=False).groupby("period").head(2)
                  .sort_values(["period", "csi"], ascending=[True, False]))
    top_csi = top_csi[top_csi["rag"] != "Green"]

    facts = [
        f"Quarters monitored: {len(q)} ({q['period'].iloc[0]} to {q['period'].iloc[-1]}).",
        f"Overall status: {(q['overall_rag'] == 'Green').sum()} Green, "
        f"{(q['overall_rag'] == 'Amber').sum()} Amber, {(q['overall_rag'] == 'Red').sum()} Red.",
        f"PSI ranged from {q['psi'].min():.3f} to {q['psi'].max():.3f}; "
        f"highest in {q.loc[q['psi'].idxmax(), 'period']}.",
        f"Gini ranged from {q['gini'].min():.3f} to {q['gini'].max():.3f} "
        f"against {dev['gini']:.3f} in development.",
        f"Actual default rate ranged from {q['default_rate'].min():.2%} to "
        f"{q['default_rate'].max():.2%}; predicted PD averaged {q['mean_predicted_pd'].mean():.2%}.",
    ]

    selected = iv[iv["selected"]][["feature", "iv", "strength"]].round(3)

    text = f"""# Model Monitoring Report: Lending Club PD Scorecard

## 1. Purpose and scope
This report monitors a probability-of-default (PD) scorecard for 36-month unsecured
consumer loans. The model was developed on loans issued in {C.DEV_YEARS[0]}-{C.DEV_YEARS[-1]}
and is monitored on each quarterly vintage issued in {C.MONITOR_YEARS[0]}-{C.MONITOR_YEARS[-1]}.
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
2. Numeric features were cut into up to {C.N_BINS} quantile bins; categorical features kept
   categories above {C.MIN_CATEGORY_SHARE:.0%} of loans and grouped the rest as "Other";
   missing values got their own bin.
3. Weight of evidence (WoE) was computed per bin and information value (IV) per feature.
   Features with IV below {C.MIN_IV} were dropped, and of any two WoE features correlated
   above {C.MAX_CORR}, the one with lower IV was dropped.
4. A logistic regression was fitted on the WoE values. All coefficients are expected to be
   negative, since a higher WoE means a safer bin.
5. Probabilities were scaled to scorecard points: {C.BASE_SCORE} points at {C.BASE_ODDS}:1
   good:bad odds, with odds doubling every {C.PDO} points.

Selected features:

{md_table(selected)}

Coefficients:

{md_table(coefs)}

Development hold-out performance: AUC {dev['auc']:.3f}, Gini {dev['gini']:.3f}, KS {dev['ks']:.3f}.

## 4. Monitoring framework

| Check | Metric | Green | Amber | Red |
| --- | --- | --- | --- | --- |
| Population stability | PSI on score bands (development deciles) | below {C.PSI_AMBER} | {C.PSI_AMBER} to {C.PSI_RED} | above {C.PSI_RED} |
| Feature stability | CSI on each feature's WoE bins | below {C.PSI_AMBER} | {C.PSI_AMBER} to {C.PSI_RED} | above {C.PSI_RED} |
| Discrimination | Gini as a share of development Gini | {C.GINI_AMBER:.0%} or more | {C.GINI_RED:.0%} to {C.GINI_AMBER:.0%} | below {C.GINI_RED:.0%} |
| Calibration | Actual default rate vs mean predicted PD | within {C.CALIB_AMBER:.0%} | {C.CALIB_AMBER:.0%} to {C.CALIB_RED:.0%} | more than {C.CALIB_RED:.0%} apart |

KS and decile-level calibration are tracked alongside for context. A quarter's overall
status is its worst individual status.

## 5. Results

{md_table(perf_view)}

Features driving instability (top two CSI per quarter, Amber or Red only):

{md_table(top_csi[["period", "feature", "csi", "rag"]].round(3)) if len(top_csi) else "_No feature breached the CSI thresholds._"}

## 6. Findings
""" + "\n".join(f"- {f}" for f in facts) + f"""

### Analyst commentary
{COMMENTARY_MARKER}
{commentary}
{COMMENTARY_MARKER}

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
"""
    C.REPORTS.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Report written to {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
