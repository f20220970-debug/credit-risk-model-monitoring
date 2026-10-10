"""Monitoring metrics: PSI/CSI (stability), KS/AUC/Gini (discrimination), calibration."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C

EPS = 1e-6  # stops ln(0) when a bin is empty


def psi(expected_share: pd.Series, actual_share: pd.Series) -> float:
    """Population Stability Index between two distributions over the same bins.
    PSI = sum (actual% - expected%) * ln(actual% / expected%)."""
    e, a = expected_share.align(actual_share, fill_value=0)
    e, a = e.clip(lower=EPS), a.clip(lower=EPS)
    return float(((a - e) * np.log(a / e)).sum())


def shares(labels: pd.Series) -> pd.Series:
    return labels.value_counts(normalize=True)


def ks(y, pd_hat) -> float:
    """Kolmogorov-Smirnov: largest gap between the cumulative score
    distributions of defaulters and non-defaulters (0 = no separation, 1 = perfect)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)}).sort_values("p", ascending=False)
    cum_bad = d["y"].cumsum() / d["y"].sum()
    cum_good = (1 - d["y"]).cumsum() / (1 - d["y"]).sum()
    return float((cum_bad - cum_good).abs().max())


def auc_gini(y, pd_hat):
    auc = float(roc_auc_score(y, pd_hat))
    return auc, 2 * auc - 1


def calibration_by_decile(y, pd_hat) -> pd.DataFrame:
    """Average predicted PD vs actual default rate in 10 equal-sized groups."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    return (d.groupby("decile")
             .agg(loans=("y", "size"), mean_predicted_pd=("p", "mean"), actual_default_rate=("y", "mean"))
             .reset_index())


def rag_psi(value: float) -> str:
    return "Red" if value > C.PSI_RED else "Amber" if value >= C.PSI_AMBER else "Green"


def rag_gini(gini: float, dev_gini: float) -> str:
    ratio = gini / dev_gini
    return "Red" if ratio < C.GINI_RED else "Amber" if ratio < C.GINI_AMBER else "Green"


def rag_calibration(actual: float, predicted: float) -> str:
    gap = abs(actual / predicted - 1)
    return "Red" if gap > C.CALIB_RED else "Amber" if gap > C.CALIB_AMBER else "Green"


def worst(*statuses) -> str:
    order = {"Green": 0, "Amber": 1, "Red": 2}
    return max(statuses, key=order.get)
