"""TDD tests for cosine divergence null baseline experiment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.network import RecurrentMLP


def test_null_divergence_computation():
    """Verify null baseline produces valid cosine divergence values."""
    net = RecurrentMLP(seed=0)
    W_rec = net.W_rec.copy()
    tau = net.feedback_tau

    # Create a "self" output vector
    y_self = np.array([1.0, -0.5, 0.3, 0.0, 0.8])

    # Self W_rec contribution
    self_contrib = np.tanh(y_self / tau) @ W_rec

    # Same vector should give divergence = 0
    same_contrib = np.tanh(y_self / tau) @ W_rec
    cos_sim = np.dot(self_contrib, same_contrib) / (
        np.linalg.norm(self_contrib) * np.linalg.norm(same_contrib) + 1e-12)
    cos_div = 1 - cos_sim
    assert abs(cos_div) < 1e-10, f"Self-vs-self divergence should be 0, got {cos_div}"

    # Random vector should give non-zero divergence
    rng = np.random.RandomState(42)
    z = rng.randn(5)
    z = z * np.linalg.norm(y_self) / (np.linalg.norm(z) + 1e-12)
    rand_contrib = np.tanh(z / tau) @ W_rec
    cos_sim_rand = np.dot(self_contrib, rand_contrib) / (
        np.linalg.norm(self_contrib) * np.linalg.norm(rand_contrib) + 1e-12)
    cos_div_rand = 1 - cos_sim_rand
    assert 0 <= cos_div_rand <= 2, f"Divergence should be in [0,2], got {cos_div_rand}"

    # Mean over many random draws should be > 0
    divs = []
    for _ in range(100):
        z = rng.randn(5)
        z = z * np.linalg.norm(y_self) / (np.linalg.norm(z) + 1e-12)
        rc = np.tanh(z / tau) @ W_rec
        cs = np.dot(self_contrib, rc) / (
            np.linalg.norm(self_contrib) * np.linalg.norm(rc) + 1e-12)
        divs.append(1 - cs)
    mean_div = np.mean(divs)
    assert mean_div > 0.1, f"Mean random divergence should be substantially > 0, got {mean_div}"


def test_divergence_range():
    """Cosine divergence values must be in [0, 2]."""
    rng = np.random.RandomState(0)
    for _ in range(50):
        a = rng.randn(10)
        b = rng.randn(10)
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        cos_div = 1 - cos_sim
        assert -0.01 <= cos_div <= 2.01, f"Divergence out of range: {cos_div}"
