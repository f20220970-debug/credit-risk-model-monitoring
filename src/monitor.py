"""Step 3: monitor the scorecard on every quarter of 2014-2015.

For each quarter: score the loans, then compare against the development baseline.
  Stability      PSI on the score distribution; CSI on each feature's bins
  Discrimination AUC, Gini, KS (Gini judged relative to development Gini)
  Calibration    actual default rate vs average predicted PD, overall and by decile
Each check gets Red / Amber / Green; the quarter's overall status is the worst one.
Results go to CSV files (for Tableau) and a SQLite database (for SQL queries).
"""
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import config as C
from metrics import (auc_gini, calibration_by_decile, ks, psi, rag_calibration,
                     rag_gini, rag_psi, shares, worst)
from train import to_score


def score_bands(scores, edges) -> pd.Series:
    labels = [f"Band {i:02d}" for i in range(1, len(edges))]   # Band 01 = riskiest
    return pd.Series(pd.cut(scores, bins=edges, labels=labels, include_lowest=True)).astype(str)


def main() -> None:
    art = joblib.load(C.ARTIFACTS / "scorecard.joblib")
    binner, model, features = art["binner"], art["model"], art["features"]
    dev = art["dev_performance"]

    df = pd.read_parquet(C.PROCESSED_FILE)
    dev_df = df[df["issue_d"].dt.year.isin(C.DEV_YEARS)]
    dev_scores = to_score(model.predict_proba(binner.transform(dev_df, features))[:, 1])
    base_band_share = shares(score_bands(dev_scores, art["score_edges"]))
    dev_bins = binner.bin(dev_df)
    base_bin_shares = {f: shares(dev_bins[f]) for f in features}   # baseline for CSI

    mon = df[df["issue_d"].dt.year.isin(C.MONITOR_YEARS)]
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    perf_rows, csi_rows, calib_rows, dist_rows = [], [], [], []

    for band, share in base_band_share.sort_index().items():
        dist_rows.append({"period": "Development", "score_band": band, "share": share})

    for quarter, q in mon.groupby("quarter"):
        y = q["default"]
        pd_hat = model.predict_proba(binner.transform(q, features))[:, 1]
        scores = to_score(pd_hat)

        # Stability
        band_share = shares(score_bands(scores, art["score_edges"]))
        psi_value = psi(base_band_share, band_share)
        for band, share in band_share.sort_index().items():
            dist_rows.append({"period": quarter, "score_band": band, "share": share})
        q_bins = binner.bin(q)
        for f in features:
            c = psi(base_bin_shares[f], shares(q_bins[f]))
            csi_rows.append({"period": quarter, "feature": f, "csi": c, "rag": rag_psi(c)})

        # Discrimination
        auc, gini = auc_gini(y, pd_hat)
        ks_value = ks(y, pd_hat)

        # Calibration
        actual, predicted = y.mean(), pd_hat.mean()
        cal = calibration_by_decile(y, pd_hat)
        cal.insert(0, "period", quarter)
        calib_rows.append(cal)

        psi_rag = rag_psi(psi_value)
        gini_rag = rag_gini(gini, dev["gini"])
        calib_rag = rag_calibration(actual, predicted)
        perf_rows.append({
            "period": quarter, "loans": len(q), "default_rate": actual,
            "mean_predicted_pd": predicted, "actual_to_predicted": actual / predicted,
            "auc": auc, "gini": gini, "gini_vs_dev": gini / dev["gini"], "ks": ks_value,
            "psi": psi_value, "psi_rag": psi_rag, "gini_rag": gini_rag,
            "calibration_rag": calib_rag, "overall_rag": worst(psi_rag, gini_rag, calib_rag),
            "run_at": run_at,
        })

    dev_row = {**dev, "actual_to_predicted": dev["default_rate"] / dev["mean_predicted_pd"],
               "gini_vs_dev": 1.0, "psi": 0.0, "psi_rag": "Green", "gini_rag": "Green",
               "calibration_rag": "Green", "overall_rag": "Green", "run_at": run_at}
    tables = {
        "performance": pd.DataFrame([dev_row] + perf_rows).round(4),
        "csi": pd.DataFrame(csi_rows).round(4),
        "calibration": pd.concat(calib_rows, ignore_index=True).round(4),
        "score_distribution": pd.DataFrame(dist_rows).round(4),
    }

    C.OUTPUTS.mkdir(exist_ok=True)
    with sqlite3.connect(C.OUTPUTS / "monitoring.db") as con:
        for name, t in tables.items():
            t.to_csv(C.OUTPUTS / f"{name}.csv", index=False)
            t.to_sql(name, con, if_exists="replace", index=False)

    cols = ["period", "loans", "default_rate", "actual_to_predicted", "gini", "ks", "psi", "overall_rag"]
    print(tables["performance"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
