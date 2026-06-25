"""Phase 3+: Group C2 (Clone Feedback) tests.

Clone Feedback: replace each target model's feedback with the output of an
independently trained donor model.
Same distribution, same structure, different "self" -- blocks the
out-of-distribution criticism.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import create_trained_network, forward_sequence_with_clone


@pytest.fixture
def two_trained_nets():
    """Two networks trained with two different seeds."""
    net_a = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    net_b = create_trained_network(seed=1, epochs=200, noise_level=0.3)
    return net_a, net_b


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_clone_feedback_uses_different_model_output(two_trained_nets):
    """Verify that clone feedback actually uses another model's output.

    The target's own forward_sequence result and the result using
    clone feedback must differ.
    """
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    # Normal forward
    normal_outputs, _ = target.forward_sequence(x, T=3)

    # Clone feedback forward
    clone_outputs, _ = forward_sequence_with_clone(target, clone, x, T=3)

    # t=1 must be identical (no feedback)
    assert np.allclose(normal_outputs[0], clone_outputs[0]), \
        "t=1 output changed -- it was modified at a step with no feedback"

    # t=2 onward must differ (uses another model's feedback)
    assert not np.allclose(normal_outputs[1], clone_outputs[1]), \
        "Clone feedback did not change the t=2 output"


def test_clone_feedback_preserves_original_weights(two_trained_nets):
    """After running clone feedback, the target model's weights must not change."""
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    w_before = {k: v.copy() for k, v in target.get_all_weights().items()}

    forward_sequence_with_clone(target, clone, x, T=3)

    w_after = target.get_all_weights()
    for k in w_before:
        assert np.allclose(w_before[k], w_after[k]), \
            f"Clone feedback modified the {k} weights"


def test_clone_output_is_valid(two_trained_nets):
    """Whether the clone model's output is valid -- non-zero and in a reasonable range."""
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    clone.reset_state()
    y = clone.forward(x)

    assert not np.allclose(y, 0), "Clone output is all zeros"
    assert np.all(np.isfinite(y)), "Clone output contains inf/nan"
    assert np.linalg.norm(y) > 0.01, "Clone output norm is too small"


def test_clone_feedback_deterministic(two_trained_nets):
    """Same input, same model pair -> same result."""
    target, clone = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    out1, _ = forward_sequence_with_clone(target, clone, x, T=3)
    out2, _ = forward_sequence_with_clone(target, clone, x, T=3)

    for t in range(3):
        assert np.allclose(out1[t], out2[t]), \
            f"Result is non-deterministic at t={t+1}"


def test_clone_vs_self_feedback():
    """Using itself as the clone must be identical to a normal forward."""
    net = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    x = np.random.RandomState(42).randn(10)

    # Normal
    normal_outputs, _ = net.forward_sequence(x, T=3)

    # Self-clone (same model as both target and clone)
    # Need a second copy since forward modifies state
    net2 = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    clone_outputs, _ = forward_sequence_with_clone(net, net2, x, T=3)

    for t in range(3):
        assert np.allclose(normal_outputs[t], clone_outputs[t], atol=1e-10), \
            f"Self-clone differs from normal forward at t={t+1}"
