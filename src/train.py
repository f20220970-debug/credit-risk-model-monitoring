"""Step 2: build the PD scorecard on 2012-2013 loans.

1. Split development data 70/30 (train / hold-out).
2. Bin every feature and compute WoE and IV on the training part.
3. Keep features with IV >= 0.02, then drop one of any highly correlated pair.
4. Fit logistic regression on the WoE values.
5. Turn the model into scorecard points and save everything for monitoring.
"""
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config as C
from metrics import auc_gini, calibration_by_decile, ks
from woe import WoEBinner


def select_features(binner: WoEBinner, woe_train: pd.DataFrame) -> list:
    iv = binner.iv_table()
    candidates = iv.loc[iv["iv"] >= C.MIN_IV, "feature"].tolist()   # sorted by IV, strongest first
    selected = []
    for f in candidates:
        corr = [abs(woe_train[f].corr(woe_train[s])) for s in selected]
        if all(c <= C.MAX_CORR for c in corr):
            selected.append(f)
    return selected


def to_score(pd_hat: np.ndarray) -> np.ndarray:
    """Scorecard points: higher score = safer. 600 at 50:1 good:bad odds, +20 doubles the odds."""
    factor = C.PDO / np.log(2)
    offset = C.BASE_SCORE - factor * np.log(C.BASE_ODDS)
    odds_good = (1 - pd_hat) / np.clip(pd_hat, 1e-9, None)
    return offset + factor * np.log(odds_good)


def main() -> None:
    df = pd.read_parquet(C.PROCESSED_FILE)
    dev = df[df["issue_d"].dt.year.isin(C.DEV_YEARS)].copy()
    train, hold = train_test_split(dev, test_size=0.30, stratify=dev["default"], random_state=42)

    binner = WoEBinner(C.N_BINS, C.MIN_CATEGORY_SHARE).fit(
        train, train["default"], C.NUMERIC_FEATURES, C.CATEGORICAL_FEATURES)
    woe_train = binner.transform(train)
    features = select_features(binner, woe_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(woe_train[features], train["default"])

    # statsmodels gives p-values; WoE coefficients should all be negative
    # (higher WoE = safer bin = lower PD). A positive sign is a warning.
    sm_fit = sm.Logit(train["default"], sm.add_constant(woe_train[features])).fit(disp=0)
    coefs = pd.DataFrame({"feature": sm_fit.params.index, "coefficient": sm_fit.params.values,
                          "p_value": sm_fit.pvalues.values}).round(4)

    # Development performance on the hold-out sample
    pd_hold = model.predict_proba(binner.transform(hold, features))[:, 1]
    auc, gini = auc_gini(hold["default"], pd_hold)
    dev_perf = {"period": "Development hold-out", "loans": len(hold),
                "default_rate": hold["default"].mean(), "mean_predicted_pd": pd_hold.mean(),
                "auc": auc, "gini": gini, "ks": ks(hold["default"], pd_hold)}

    # Score band edges for PSI: deciles of the training scores
    pd_train = model.predict_proba(woe_train[features])[:, 1]
    score_train = to_score(pd_train)
    score_edges = np.unique(np.quantile(score_train, np.linspace(0, 1, 11)))
    score_edges[0], score_edges[-1] = -np.inf, np.inf

    C.ARTIFACTS.mkdir(exist_ok=True)
    C.OUTPUTS.mkdir(exist_ok=True)
    joblib.dump({"binner": binner, "model": model, "features": features,
                 "score_edges": score_edges,
                 "dev_performance": dev_perf}, C.ARTIFACTS / "scorecard.joblib")

    iv = binner.iv_table()
    iv["selected"] = iv["feature"].isin(features)
    iv.round(4).to_csv(C.OUTPUTS / "iv_table.csv", index=False)
    binner.bin_table(train, train["default"]).to_csv(C.OUTPUTS / "woe_bins.csv", index=False)
    coefs.to_csv(C.OUTPUTS / "coefficients.csv", index=False)
    calibration_by_decile(hold["default"], pd_hold).round(4).to_csv(
        C.OUTPUTS / "dev_calibration.csv", index=False)

    print(f"Development loans: {len(dev):,} (train {len(train):,}, hold-out {len(hold):,})")
    print("\nInformation value:\n", iv.round(4).to_string(index=False))
    print(f"\nSelected features ({len(features)}): {features}")
    print("\nCoefficients:\n", coefs.to_string(index=False))
    bad_signs = coefs.loc[(coefs["feature"] != "const") & (coefs["coefficient"] > 0), "feature"].tolist()
    if bad_signs:
        print(f"\nWARNING: positive coefficients (check these): {bad_signs}")
    print(f"\nHold-out AUC {auc:.3f} | Gini {gini:.3f} | KS {dev_perf['ks']:.3f}")


if __name__ == "__main__":
    main()
