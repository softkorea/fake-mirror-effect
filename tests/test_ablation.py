"""Phase 3: Ablation tests.

Groups: A (recurrent ablation), B1 (random ablation), B2 (structural ablation),
      C1 (permutation feedback), C2 (batch-shuffle feedback),
      D (feedforward), D' (param-matched FF)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import (
    ablate_recurrent,
    ablate_random,
    ablate_structural,
    count_zeroed_weights,
    create_trained_network,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
@pytest.fixture
def trained_net():
    """Trained network (seed=42)."""
    return create_trained_network(seed=42, epochs=200, noise_level=0.3)


# ──────────────────────────────────────────────
# Group A: recurrent ablation
# ──────────────────────────────────────────────
def test_ablate_recurrent_zeroes_weights(trained_net):
    """Whether the recurrent weights actually become 0."""
    assert np.any(trained_net.get_all_weights()['recurrent'] != 0)
    ablate_recurrent(trained_net)
    assert np.all(trained_net.get_all_weights()['recurrent'] == 0)


def test_ablate_recurrent_preserves_other_weights(trained_net):
    """Recurrent ablation does not affect other weights."""
    w_before = {k: v.copy() for k, v in trained_net.get_all_weights().items()}
    ablate_recurrent(trained_net)
    w_after = trained_net.get_all_weights()
    for k in ['input_to_h1', 'h1_to_h2', 'h2_to_output']:
        assert np.allclose(w_before[k], w_after[k])


# ──────────────────────────────────────────────
# Group B1: random ablation
# ──────────────────────────────────────────────
def test_ablate_random_correct_count(trained_net):
    """Whether random ablation severs the exact number of connections."""
    n_target = 50  # same as the number of recurrent weights
    w_before = {k: v.copy() for k, v in trained_net.get_all_weights().items()}
    ablate_random(trained_net, n_connections=n_target, seed=42)

    zeroed = 0
    w_after = trained_net.get_all_weights()
    for k in w_before:
        # Count entries that were non-zero before but are 0 now
        zeroed += np.sum((w_before[k] != 0) & (w_after[k] == 0))
    assert zeroed == n_target, f"Expected {n_target} zeroed, got {zeroed}"


def test_ablate_random_deterministic(trained_net):
    """Same seed -> same ablation result."""
    net2 = create_trained_network(seed=42, epochs=200, noise_level=0.3)

    ablate_random(trained_net, n_connections=20, seed=7)
    ablate_random(net2, n_connections=20, seed=7)

    for k in trained_net.get_all_weights():
        assert np.allclose(
            trained_net.get_all_weights()[k],
            net2.get_all_weights()[k]
        ), f"same seed=7 but {k} ablation result differs"


# ──────────────────────────────────────────────
# Group B2: structural ablation
# ──────────────────────────────────────────────
def test_ablate_structural():
    """Ablate an entire specific layer."""
    net = create_trained_network(seed=42, epochs=200, noise_level=0.3)
    ablate_structural(net, layer='h1_to_h2')
    assert np.all(net.get_all_weights()['h1_to_h2'] == 0)
    # Other layers are preserved
    assert np.any(net.get_all_weights()['input_to_h1'] != 0)


# ──────────────────────────────────────────────
# Group C: Scrambled Feedback (built-in network feature)
# ──────────────────────────────────────────────
def test_scrambled_feedback_preserves_weights(trained_net):
    """Scrambled feedback: weights unchanged."""
    w_before = {k: v.copy() for k, v in trained_net.get_all_weights().items()}
    trained_net.enable_scrambled_feedback(seed=42)
    w_after = trained_net.get_all_weights()
    for k in w_before:
        assert np.allclose(w_before[k], w_after[k]), \
            f"Scrambled feedback modified the {k} weights"


def test_scrambled_feedback_changes_output(trained_net):
    """When scrambled feedback is enabled, the t>=2 output changes."""
    x = np.random.RandomState(0).randn(10)

    trained_net.reset_state()
    trained_net.forward(x)  # t=1
    y_normal = trained_net.forward(x)  # t=2 (normal feedback)

    trained_net.reset_state()
    trained_net.enable_scrambled_feedback(seed=42)
    trained_net.forward(x)  # t=1
    y_scrambled = trained_net.forward(x)  # t=2 (scrambled feedback)

    assert not np.allclose(y_normal, y_scrambled), \
        "Scrambled feedback did not change the output"


def test_scrambled_feedback_t1_unchanged(trained_net):
    """Scrambled feedback has no effect on the t=1 output (there is no feedback)."""
    x = np.random.RandomState(0).randn(10)

    trained_net.reset_state()
    y_normal_t1 = trained_net.forward(x)

    trained_net.reset_state()
    trained_net.enable_scrambled_feedback(seed=42)
    y_scrambled_t1 = trained_net.forward(x)

    trained_net.disable_scrambled_feedback()
    assert np.allclose(y_normal_t1, y_scrambled_t1), \
        "Scrambled feedback modified the t=1 output (bug)"


# ──────────────────────────────────────────────
# Multi-model verification
# ──────────────────────────────────────────────
def test_multi_seed_models_differ():
    """Whether models trained with different seeds are actually different."""
    net1 = create_trained_network(seed=1, epochs=100, noise_level=0.3)
    net2 = create_trained_network(seed=2, epochs=100, noise_level=0.3)
    w1 = net1.get_all_weights()['input_to_h1']
    w2 = net2.get_all_weights()['input_to_h1']
    assert not np.allclose(w1, w2), "different seeds but identical weights after training"


# ──────────────────────────────────────────────
# Ablation-count statistics
# ──────────────────────────────────────────────
def test_count_zeroed_weights(trained_net):
    """Zeroed-weight counting function."""
    n_before = count_zeroed_weights(trained_net)
    ablate_recurrent(trained_net)
    n_after = count_zeroed_weights(trained_net)
    assert n_after > n_before
    assert n_after - n_before == 50  # W_rec is 5×10
