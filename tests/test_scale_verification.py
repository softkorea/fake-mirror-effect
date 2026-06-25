"""Task 3: Scale Verification tests.

Verifies that RecurrentMLP operates correctly across various hidden widths.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import (
    generate_data, generate_data_variable_noise,
    train, train_vn,
)
from src.metrics import compute_all_metrics, compute_all_metrics_vn


class TestScaleVerification:
    """Training verification for hidden-width variant models."""

    def test_scale_variant_h20_trains(self):
        """h=20 variant trains without error (static)."""
        net = RecurrentMLP(input_size=10, hidden1=20, hidden2=20,
                           output_size=5, seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        losses = train(net, X, y, epochs=10, lr=0.01)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_scale_variant_h45_trains(self):
        """h=45 variant trains without error (static)."""
        net = RecurrentMLP(input_size=10, hidden1=45, hidden2=45,
                           output_size=5, seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        losses = train(net, X, y, epochs=10, lr=0.01)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_scale_variant_h245_trains(self):
        """h=245 variant trains without error (static)."""
        net = RecurrentMLP(input_size=10, hidden1=245, hidden2=245,
                           output_size=5, seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        losses = train(net, X, y, epochs=10, lr=0.01)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_scale_variant_vn_h20_trains(self):
        """h=20 variant trains with VN without error."""
        net = RecurrentMLP(input_size=10, hidden1=20, hidden2=20,
                           output_size=5, seed=0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        losses = train_vn(net, X_seq, y, epochs=10, lr=0.01)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_scale_variant_vn_h45_trains(self):
        """h=45 variant trains with VN without error."""
        net = RecurrentMLP(input_size=10, hidden1=45, hidden2=45,
                           output_size=5, seed=0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        losses = train_vn(net, X_seq, y, epochs=10, lr=0.01)
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_scale_variant_metrics_static(self):
        """compute_all_metrics works with h=20 model."""
        net = RecurrentMLP(input_size=10, hidden1=20, hidden2=20,
                           output_size=5, seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        train(net, X, y, epochs=10, lr=0.01)
        metrics = compute_all_metrics(net, X, y)
        assert 'gain' in metrics
        assert np.isfinite(metrics['gain'])

    def test_scale_variant_metrics_vn(self):
        """compute_all_metrics_vn works with h=20 model."""
        net = RecurrentMLP(input_size=10, hidden1=20, hidden2=20,
                           output_size=5, seed=0)
        X_seq, y = generate_data_variable_noise(50, 0.5, T=3, seed=42)
        train_vn(net, X_seq, y, epochs=10, lr=0.01)
        metrics = compute_all_metrics_vn(net, X_seq, y)
        assert 'gain' in metrics
        assert np.isfinite(metrics['gain'])

    def test_scale_variant_param_counts(self):
        """Parameter counts increase with hidden width."""
        counts = {}
        for h in [10, 20, 45, 245]:
            net = RecurrentMLP(input_size=10, hidden1=h, hidden2=h,
                               output_size=5, seed=0)
            counts[h] = net.count_params()
        assert counts[10] < counts[20] < counts[45] < counts[245]

    def test_scale_ablation_group_a(self):
        """Group A ablation works at non-default hidden width."""
        net = RecurrentMLP(input_size=10, hidden1=20, hidden2=20,
                           output_size=5, seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        train(net, X, y, epochs=10, lr=0.01)

        # Disable recurrent
        net.disable_recurrent_loop()
        outputs1, _ = net.forward_sequence(X[0], T=3)
        outputs2, _ = net.forward_sequence(X[0], T=3)
        # Should be deterministic
        for t in range(3):
            np.testing.assert_array_equal(outputs1[t], outputs2[t])
        # t=2 and t=3 should equal t=1 (no recurrence)
        np.testing.assert_array_equal(outputs1[0], outputs1[1])
        np.testing.assert_array_equal(outputs1[0], outputs1[2])
        net.enable_recurrent_loop()
