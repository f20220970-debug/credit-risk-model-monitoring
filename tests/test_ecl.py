import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ecl as E  # noqa: E402


def test_twelve_month_pd_constant_hazard():
    assert E.twelve_month_pd(np.array([0.271]))[0] == pytest.approx(1 - 0.729 ** (1 / 3), abs=1e-9)  # = 0.1


def test_shift_pd_zero_is_identity_and_positive_raises():
    p = np.array([0.05, 0.2])
    assert np.allclose(E.shift_pd(p, 0), p)
    assert (E.shift_pd(p, 0.4) > p).all()


def test_stage2_uses_lifetime_pd():
    pd_life, ead = np.array([0.30, 0.30]), np.array([1000.0, 1000.0])
    e = E.ecl(pd_life, ead, np.array([1, 2]), 0.85)
    assert e[1] == pytest.approx(0.30 * 0.85 * 1000) and e[0] < e[1]


def test_scenarios_increase_and_weights_sum():
    pd_life = np.linspace(0.02, 0.4, 100); ead = np.full(100, 1000.0)
    scen, weighted = E.scenario_ecl(pd_life, ead, pd_life.mean())
    assert scen["weight"].sum() == pytest.approx(1)
    assert scen["ecl"].is_monotonic_increasing
    assert scen["ecl"].min() <= weighted.sum() <= scen["ecl"].max()
