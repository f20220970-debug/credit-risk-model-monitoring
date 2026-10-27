"""IFRS 9-style expected credit loss (ECL) and stress testing: the maths, kept separate from data
loading so it can be unit-tested.

ECL = PD x LGD x EAD
  Stage 1 (performing, no significant increase in credit risk): 12-month PD
  Stage 2 (significant increase in credit risk, SICR):           lifetime PD
The scorecard predicts default over the loan's 36-month life. Assuming a constant default
hazard over the term, the 12-month PD is 1 - (1 - PD_life)^(12/36).

Scenarios shift every loan's PD on the log-odds scale (a standard way to stress a PD model)
and raise LGD; IFRS 9 ECL is the probability-weighted average across scenarios.
"""
import numpy as np
import pandas as pd

TERM_MONTHS = 36
LGD_BASE = 0.85            # unsecured consumer loans recover little; illustrative assumption
SICR_MULTIPLE = 2.0        # stage 2 if lifetime PD >= 2x the portfolio's average PD (SICR proxy)
RECALIBRATION_SHIFT = 0.20 # log-odds intercept shift recommended by the monitoring work

SCENARIOS = pd.DataFrame([
    {"scenario": "baseline", "weight": 0.60, "logodds_shift": 0.00, "lgd_add": 0.00},
    {"scenario": "adverse",  "weight": 0.30, "logodds_shift": 0.40, "lgd_add": 0.05},
    {"scenario": "severe",   "weight": 0.10, "logodds_shift": 0.80, "lgd_add": 0.10},
])


def shift_pd(pd_: np.ndarray, shift: float) -> np.ndarray:
    pd_ = np.clip(pd_, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(pd_ / (1 - pd_)) + shift)))


def twelve_month_pd(pd_life: np.ndarray, term_months: int = TERM_MONTHS) -> np.ndarray:
    return 1 - (1 - pd_life) ** (12 / term_months)


def stage(pd_life: np.ndarray, reference_pd: float, multiple: float = SICR_MULTIPLE) -> np.ndarray:
    return np.where(pd_life >= multiple * reference_pd, 2, 1)


def ecl(pd_life: np.ndarray, ead: np.ndarray, stages: np.ndarray, lgd: float) -> np.ndarray:
    pd_used = np.where(stages == 2, pd_life, twelve_month_pd(pd_life))
    return pd_used * lgd * ead


def scenario_ecl(pd_life, ead, reference_pd: float, scenarios: pd.DataFrame = SCENARIOS, lgd_base: float = LGD_BASE):
    """ECL per scenario (stages re-assessed against a fixed reference PD) and the probability-weighted total."""
    rows, weighted = [], np.zeros(len(pd_life))
    ref = reference_pd
    for _, s in scenarios.iterrows():
        p = shift_pd(pd_life, s.logodds_shift)
        st = stage(p, ref)
        e = ecl(p, ead, st, min(1.0, lgd_base + s.lgd_add))
        weighted += s.weight * e
        rows.append({"scenario": s.scenario, "weight": s.weight, "avg_lifetime_pd": p.mean(),
                     "stage2_share": (st == 2).mean(), "ecl": e.sum(), "coverage": e.sum() / ead.sum()})
    return pd.DataFrame(rows), weighted
