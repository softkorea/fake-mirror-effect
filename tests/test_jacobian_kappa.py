"""Reproducibility test for the Section 3.7 local-conditioning (kappa) result.

Asserts that the shipped analysis (experiments/run_jacobian_analysis.kappa_contrasts)
reproduces the Section 3.7 PRIMARY statistics from the frozen per-seed jacobian CSV:
the 6-contrast median-statistic null (min raw p = 0.729, all Holm-adjusted p = 1.000)
and the secondary mean-statistic finding (self vs wrong-trajectory nominal raw p = 0.019
that does not survive Holm and vanishes under the robust median).
"""
import os
import pandas as pd
import pytest

from experiments.run_jacobian_analysis import kappa_contrasts

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "integration_control_jacobian.csv")


@pytest.mark.skipif(not os.path.exists(CSV), reason="frozen jacobian CSV not present")
def test_kappa_section_3_7_reproduces():
    df = pd.read_csv(CSV)
    kc = kappa_contrasts(df, n_boot=200)  # small n_boot: CIs are not asserted here

    # --- PRIMARY: per-seed median, 6-contrast Holm ---
    med = kc["median"]
    tp = med["typical_per_mode"]
    assert round(tp["self"], 1) == 13.1
    assert round(tp["clone"], 1) == 10.1
    assert round(tp["wrong_traj"], 1) == 12.5
    assert round(tp["normmatched_clone"], 1) == 14.0
    assert med["n_contrasts"] == 6
    assert abs(med["min_raw_p"] - 0.729) < 0.005
    assert med["all_holm_nonsignificant"] is True
    assert all(c["p_holm"] >= 0.999 for c in med["contrasts"].values())  # all ~1.000

    # --- SECONDARY: per-seed mean (tail-sensitive) ---
    mean_sw = kc["mean"]["contrasts"]["self_vs_wrong_traj"]
    assert abs(mean_sw["p_raw"] - 0.019) < 0.005          # one nominal contrast
    assert mean_sw["p_holm"] >= 0.05                       # does NOT survive Holm
    # the same contrast is non-significant under the robust primary (median) statistic
    assert med["contrasts"]["self_vs_wrong_traj"]["p_raw"] > 0.5
