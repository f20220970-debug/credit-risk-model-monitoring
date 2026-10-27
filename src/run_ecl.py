"""Step 5: expected credit loss, staging, stress tests and portfolio aggregation.

Uses the trained scorecard (artifacts/scorecard.joblib) to score the 2015 vintage as the
"current book", then computes IFRS 9-style ECL, three macro scenarios, the effect of the
recalibration overlay, and an aggregation view. Also back-tests expected loss against realised
loss for 2014 and 2015 loans. Results: outputs/ecl_*.csv and tables in outputs/monitoring.db.
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd

import config as C
import ecl as E


def score(df, art):
    return art["model"].predict_proba(art["binner"].transform(df, art["features"]))[:, 1]


def main() -> None:
    art = joblib.load(C.ARTIFACTS / "scorecard.joblib")
    df = pd.read_parquet(C.PROCESSED_FILE)
    df["pd_life"] = score(df, art)
    df["vintage"] = df["issue_d"].dt.year
    df["ead"] = df["loan_amnt"].astype(float)

    book = df[df["vintage"] == 2015].copy()                 # the "current book"
    ref = book["pd_life"].mean()
    book["stage"] = E.stage(book["pd_life"].to_numpy(), ref)
    book["ecl_model"] = E.ecl(book["pd_life"].to_numpy(), book["ead"].to_numpy(), book["stage"].to_numpy(), E.LGD_BASE)

    # Model-risk overlay: the monitoring found the model under-predicts defaults; apply the
    # recommended +0.2 log-odds recalibration and see what it does to provisions.
    pd_recal = E.shift_pd(book["pd_life"].to_numpy(), E.RECALIBRATION_SHIFT)
    st_recal = E.stage(pd_recal, ref)
    book["ecl_recal"] = E.ecl(pd_recal, book["ead"].to_numpy(), st_recal, E.LGD_BASE)
    book["stage"] = st_recal                               # stages reported on the recalibrated PDs

    scen, weighted = E.scenario_ecl(pd_recal, book["ead"].to_numpy(), ref)
    book["ecl_weighted"] = weighted

    stages = book.groupby("stage").agg(loans=("ead", "size"), ead=("ead", "sum"), ecl=("ecl_recal", "sum")).reset_index()
    stages["coverage"] = stages["ecl"] / stages["ead"]
    agg = (book.groupby(["purpose", "home_ownership"])
               .agg(loans=("ead", "size"), ead=("ead", "sum"), avg_pd=("pd_life", "mean"),
                    stage2_share=("stage", lambda s: (s == 2).mean()), ecl=("ecl_weighted", "sum"))
               .reset_index().sort_values("ecl", ascending=False))
    agg["coverage"] = agg["ecl"] / agg["ead"]

    # Back-test: expected lifetime loss vs realised loss (defaults x LGD x EAD) by vintage
    bt = []
    for v, g in df[df["vintage"].isin([2014, 2015])].groupby("vintage"):
        exp_model = (g["pd_life"] * E.LGD_BASE * g["ead"]).sum()
        exp_recal = (E.shift_pd(g["pd_life"].to_numpy(), E.RECALIBRATION_SHIFT) * E.LGD_BASE * g["ead"]).sum()
        realised = (g["default"] * E.LGD_BASE * g["ead"]).sum()
        bt.append({"vintage": v, "loans": len(g), "expected_loss_model": exp_model,
                   "expected_loss_recalibrated": exp_recal, "realised_loss": realised,
                   "realised_vs_model": realised / exp_model, "realised_vs_recalibrated": realised / exp_recal})
    bt = pd.DataFrame(bt)

    totals = pd.DataFrame([{
        "loans": len(book), "ead": book["ead"].sum(),
        "ecl_model": book["ecl_model"].sum(), "ecl_recalibrated": book["ecl_recal"].sum(),
        "ecl_probability_weighted": book["ecl_weighted"].sum()}])
    totals["coverage_weighted"] = totals["ecl_probability_weighted"] / totals["ead"]

    out = {"ecl_totals": totals, "ecl_stages": stages, "ecl_scenarios": scen,
           "ecl_aggregation": agg, "ecl_backtest": bt}
    with sqlite3.connect(C.OUTPUTS / "monitoring.db") as con:
        for name, t in out.items():
            t.round(4).to_csv(C.OUTPUTS / f"{name}.csv", index=False)
            t.round(4).to_sql(name, con, if_exists="replace", index=False)

    t = totals.iloc[0]; s2 = stages.set_index("stage")
    print("# ECL and stress-test summary\n")
    print(f"- Current book (2015 vintage): {int(t.loans):,} loans, exposure ${t.ead/1e9:.2f}bn.")
    print(f"- Stage 2 (lifetime PD >= {E.SICR_MULTIPLE:.0f}x average): {s2.loc[2,'loans'] / t.loans:.1%} of loans "
          f"but {s2.loc[2,'ecl'] / s2['ecl'].sum():.0%} of ECL (coverage {s2.loc[2,'coverage']:.1%} vs {s2.loc[1,'coverage']:.1%} in stage 1).")
    print(f"- ECL on the model as built: ${t.ecl_model/1e6:,.0f}m; after the +0.2 log-odds recalibration overlay: "
          f"${t.ecl_recalibrated/1e6:,.0f}m (+{t.ecl_recalibrated / t.ecl_model - 1:.0%}).")
    for _, r in scen.iterrows():
        print(f"- {r.scenario.title()} scenario: ECL ${r.ecl/1e6:,.0f}m, coverage {r.coverage:.1%}, stage 2 share {r.stage2_share:.1%}.")
    print(f"- Probability-weighted ECL (60/30/10): ${t.ecl_probability_weighted/1e6:,.0f}m, coverage {t.coverage_weighted:.1%}.")
    for _, r in bt.iterrows():
        print(f"- Back-test {int(r.vintage)}: realised loss was {r.realised_vs_model:.2f}x the model's expected loss "
              f"and {r.realised_vs_recalibrated:.2f}x the recalibrated expected loss.")
    top = agg.iloc[0]
    print(f"- Largest ECL pocket: {top.purpose} / {top.home_ownership} ({int(top.loans):,} loans, coverage {top.coverage:.1%}).")


if __name__ == "__main__":
    main()
