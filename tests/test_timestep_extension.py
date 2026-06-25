"""Task 1: Timestep Extension Sweep tests.

Verifies that the forward pass operates correctly at T>3.
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
from src.ablation import forward_sequence_with_clone, forward_sequence_with_clone_vn


class TestTimestepExtensionStatic:
    """Static input: extended forward pass at T>3."""

    def test_forward_sequence_T10(self):
        """forward_sequence produces 10 valid outputs at T=10."""
        net = RecurrentMLP(seed=0)
        x = np.random.RandomState(0).randn(10)
        outputs, caches = net.forward_sequence(x, T=10)
        assert len(outputs) == 10
        assert len(caches) == 10
        for o in outputs:
            assert o.shape == (5,)
            assert np.all(np.isfinite(o))

    def test_forward_sequence_T20(self):
        """forward_sequence produces 20 valid outputs at T=20."""
        net = RecurrentMLP(seed=0)
        x = np.random.RandomState(0).randn(10)
        outputs, caches = net.forward_sequence(x, T=20)
        assert len(outputs) == 20
        for o in outputs:
            assert o.shape == (5,)
            assert np.all(np.isfinite(o))

    def test_trained_model_T10_no_divergence(self):
        """Trained model outputs stay finite at T=10."""
        net = RecurrentMLP(seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        train(net, X, y, epochs=50, lr=0.01)
        x = X[0]
        outputs, _ = net.forward_sequence(x, T=10)
        for o in outputs:
            assert np.all(np.isfinite(o))
            assert np.max(np.abs(o)) < 1e6  # no blowup

    def test_disabled_recurrent_T10(self):
        """Group A: disabled recurrent still works at T=10."""
        net = RecurrentMLP(seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        train(net, X, y, epochs=50, lr=0.01)
        net.disable_recurrent_loop()
        x = X[0]
        outputs, _ = net.forward_sequence(x, T=10)
        # All outputs should be identical (no recurrence)
        for t in range(1, 10):
            np.testing.assert_array_equal(outputs[t], outputs[0])

    def test_scrambled_feedback_T10(self):
        """Group C1: scrambled feedback works at T>3."""
        net = RecurrentMLP(seed=0)
        X, y = generate_data(50, 0.5, seed=42)
        train(net, X, y, epochs=50, lr=0.01)
        net.enable_scrambled_feedback(seed=42)
        x = X[0]
        outputs, _ = net.forward_sequence(x, T=10)
        assert len(outputs) == 10
        for o in outputs:
            assert np.all(np.isfinite(o))
        net.disable_scrambled_feedback()

    def test_clone_feedback_T10(self):
        """Group C2: clone feedback works at T>3."""
        target = RecurrentMLP(seed=0)
        clone = RecurrentMLP(seed=100)
        X, y = generate_data(50, 0.5, seed=42)
        train(target, X, y, epochs=50, lr=0.01)
        train(clone, X, y, epochs=50, lr=0.01)
        x = X[0]
        outputs, _ = forward_sequence_with_clone(target, clone, x, T=10)
        assert len(outputs) == 10
        for o in outputs:
            assert np.all(np.isfinite(o))


class TestTimestepExtensionVN:
    """Variable-noise: extended forward pass at T>3."""

    def test_vn_forward_sequence_T10(self):
        """VN forward_sequence produces 10 valid outputs."""
        net = RecurrentMLP(seed=0)
        x_seq = np.random.RandomState(0).randn(10, 10)
        outputs, caches = net.forward_sequence_vn(x_seq, T=10)
        assert len(outputs) == 10
        for o in outputs:
            assert o.shape == (5,)
            assert np.all(np.isfinite(o))

    def test_vn_forward_sequence_T20(self):
        """VN forward_sequence produces 20 valid outputs."""
        net = RecurrentMLP(seed=0)
        x_seq = np.random.RandomState(0).randn(20, 10)
        outputs, caches = net.forward_sequence_vn(x_seq, T=20)
        assert len(outputs) == 20
        for o in outputs:
            assert np.all(np.isfinite(o))

    def test_vn_clone_feedback_T10(self):
        """VN C2: clone feedback at T>3."""
        target = RecurrentMLP(seed=0)
        clone = RecurrentMLP(seed=100)
        x_seq = np.random.RandomState(0).randn(10, 10)
        outputs, _ = forward_sequence_with_clone_vn(target, clone, x_seq, T=10)
        assert len(outputs) == 10
        for o in outputs:
            assert np.all(np.isfinite(o))

    def test_generate_data_variable_noise_T_gt_3(self):
        """generate_data_variable_noise works with T>3."""
        X_seq, y = generate_data_variable_noise(10, 0.5, T=10, seed=42)
        assert X_seq.shape == (10, 10, 10)
        assert y.shape == (10, 5)
