"""Task 2: VN Hyperparameter Partial Sweep tests.

Verifies that VN training operates correctly across various w1/tau combinations.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import generate_data_variable_noise, train_vn
from src.metrics import compute_all_metrics_vn


class TestVNSweep:
    """VN hyperparameter sweep correctness tests."""

    def test_vn_sweep_single_config_default(self):
        """One VN training run with default params produces valid gain."""
        net = RecurrentMLP(seed=0, feedback_tau=2.0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        losses = train_vn(net, X_seq, y, epochs=50, lr=0.01,
                          time_weights=[0.0, 0.2, 1.0])
        assert len(losses) == 50
        assert all(np.isfinite(l) for l in losses)

        X_test, y_test = generate_data_variable_noise(50, 0.5, T=3, seed=999)
        metrics = compute_all_metrics_vn(net, X_test, y_test)
        assert 'correction_gain' not in metrics  # key is 'gain'
        assert 'gain' in metrics
        assert np.isfinite(metrics['gain'])

    def test_vn_sweep_tau_1(self):
        """VN training with tau=1.0 completes without error."""
        net = RecurrentMLP(seed=0, feedback_tau=1.0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        losses = train_vn(net, X_seq, y, epochs=50, lr=0.01,
                          time_weights=[0.0, 0.2, 1.0])
        assert all(np.isfinite(l) for l in losses)

    def test_vn_sweep_tau_3(self):
        """VN training with tau=3.0 completes without error."""
        net = RecurrentMLP(seed=0, feedback_tau=3.0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        losses = train_vn(net, X_seq, y, epochs=50, lr=0.01,
                          time_weights=[0.0, 0.2, 1.0])
        assert all(np.isfinite(l) for l in losses)

    def test_vn_sweep_w1_nonzero(self):
        """VN training with w1=0.3 completes and produces valid metrics."""
        net = RecurrentMLP(seed=0, feedback_tau=2.0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        losses = train_vn(net, X_seq, y, epochs=50, lr=0.01,
                          time_weights=[0.3, 0.2, 1.0])
        assert all(np.isfinite(l) for l in losses)

        X_test, y_test = generate_data_variable_noise(50, 0.5, T=3, seed=999)
        metrics = compute_all_metrics_vn(net, X_test, y_test)
        assert np.isfinite(metrics['acc_t1'])
        assert np.isfinite(metrics['acc_t3'])

    def test_vn_sweep_all_grid_corners(self):
        """All 4 corner configs of the sweep grid train without error."""
        corners = [
            (0.0, 1.0),   # w1=0, tau=1
            (0.0, 3.0),   # w1=0, tau=3
            (0.3, 1.0),   # w1=0.3, tau=1
            (0.3, 3.0),   # w1=0.3, tau=3
        ]
        X_seq, y = generate_data_variable_noise(30, 0.5, T=3, seed=42)
        for w1, tau in corners:
            net = RecurrentMLP(seed=0, feedback_tau=tau)
            losses = train_vn(net, X_seq, y, epochs=10, lr=0.01,
                              time_weights=[w1, 0.2, 1.0])
            assert all(np.isfinite(l) for l in losses), \
                f"NaN loss at w1={w1}, tau={tau}"
