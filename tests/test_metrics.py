"""Phase 4: Measurement tools tests.

Metrics: accuracy_t1/t3, correction_gain, recurrent_contribution_norm,
          step_delta, ECE (Expected Calibration Error)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train, evaluate_accuracy_at_timestep
from src.ablation import create_trained_network
from src.metrics import (
    compute_correction_gain,
    compute_recurrent_contribution_norm,
    compute_step_delta,
    compute_ece,
    compute_all_metrics,
)


@pytest.fixture(scope="module")
def trained_net_and_data():
    """Module scope: trained network + test data."""
    net = create_trained_network(seed=42, epochs=300, noise_level=0.5)
    X_test, y_test = generate_data(n_samples=100, noise_level=0.5, seed=99)
    return net, X_test, y_test


# ──────────────────────────────────────────────
# 1. correction_gain
# ──────────────────────────────────────────────
def test_correction_gain_range(trained_net_and_data):
    """correction_gain = acc_t3 - acc_t1. Range: [-1, 1]."""
    net, X, y = trained_net_and_data
    gain = compute_correction_gain(net, X, y)
    assert -1.0 <= gain <= 1.0


def test_correction_gain_positive(trained_net_and_data):
    """A trained recurrent network has positive correction_gain."""
    net, X, y = trained_net_and_data
    gain = compute_correction_gain(net, X, y)
    print(f"Correction gain: {gain:.3f}")
    assert gain > 0, f"Correction gain negative: {gain:.3f}"


# ──────────────────────────────────────────────
# 2. recurrent_contribution_norm
# ──────────────────────────────────────────────
def test_recurrent_contribution_norm_positive(trained_net_and_data):
    """In a trained network, the feedback contribution norm > 0."""
    net, X, y = trained_net_and_data
    r_norm = compute_recurrent_contribution_norm(net, X)
    print(f"Recurrent contribution norm: {r_norm:.4f}")
    assert r_norm > 0


def test_recurrent_contribution_norm_zero_after_ablation(trained_net_and_data):
    """After ablating recurrence, the feedback norm = 0."""
    net, X, y = trained_net_and_data
    # Copy the network, then ablate
    from src.ablation import deep_copy_weights, restore_weights, ablate_recurrent
    saved = deep_copy_weights(net)
    ablate_recurrent(net)
    r_norm = compute_recurrent_contribution_norm(net, X)
    restore_weights(net, saved)  # restore
    assert r_norm < 1e-10


# ──────────────────────────────────────────────
# 3. step_delta
# ──────────────────────────────────────────────
def test_step_delta_positive(trained_net_and_data):
    """In a trained network, step_delta > 0 (the output changes)."""
    net, X, y = trained_net_and_data
    delta = compute_step_delta(net, X)
    print(f"Step delta: {delta:.4f}")
    assert delta > 0


# ──────────────────────────────────────────────
# 4. ECE
# ──────────────────────────────────────────────
def test_ece_range(trained_net_and_data):
    """ECE is between 0 and 1."""
    net, X, y = trained_net_and_data
    ece = compute_ece(net, X, y)
    print(f"ECE: {ece:.4f}")
    assert 0.0 <= ece <= 1.0


# ──────────────────────────────────────────────
# 5. Aggregated metrics
# ──────────────────────────────────────────────
def test_compute_all_metrics(trained_net_and_data):
    """compute_all_metrics returns all required keys."""
    net, X, y = trained_net_and_data
    m = compute_all_metrics(net, X, y)

    required_keys = ['acc_t1', 'acc_t2', 'acc_t3', 'gain',
                     'ece', 'r_norm', 'delta_norm']
    for k in required_keys:
        assert k in m, f"Missing key: {k}"
        assert isinstance(m[k], float), f"{k} is not float: {type(m[k])}"

    print("All metrics:", {k: f"{v:.4f}" for k, v in m.items()})


# ──────────────────────────────────────────────
# 6. accuracy range
# ──────────────────────────────────────────────
def test_accuracy_range(trained_net_and_data):
    """Accuracy 0~1."""
    net, X, y = trained_net_and_data
    for t in [1, 2, 3]:
        acc = evaluate_accuracy_at_timestep(net, X, y, t=t)
        assert 0.0 <= acc <= 1.0, f"t={t}: acc={acc} out of range"
