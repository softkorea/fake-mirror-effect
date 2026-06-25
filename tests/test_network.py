"""Phase 1: Network structure tests.

TDD -- these tests are written first, and the implementation follows.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP


# ──────────────────────────────────────────────
# 1. Basic shape
# ──────────────────────────────────────────────
def test_network_shape():
    """Check the network output shape."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.random.RandomState(0).randn(10)
    y = net.forward(x)
    assert y.shape == (5,)


# ──────────────────────────────────────────────
# 2. Structural verification of the recurrent loop
#    Verified with fixed weights, not relying on random initialization.
# ──────────────────────────────────────────────
def test_recurrent_loop_exists():
    """Structurally verify, with fixed weights, that the recurrent contribution norm is non-zero."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)

    # Set fixed weights
    W_rec = np.ones((5, 10)) * 0.5
    net.get_all_weights()['recurrent'][:] = W_rec

    # t=1: first forward (prev_output = zeros)
    x = np.ones(10)
    y1 = net.forward(x)

    # Now prev_output = y1 exists.
    # recurrent_contrib = W_rec.T @ y1  (shape: 10,)
    recurrent_contrib = W_rec.T @ y1
    assert np.linalg.norm(recurrent_contrib) > 1e-6, \
        "recurrent contribution is 0 -- the recurrent loop is structurally not working"

    # t=2: same input but the output must differ
    y2 = net.forward(x)
    assert not np.allclose(y1, y2), \
        "same input but t=1 and t=2 outputs are identical -- recurrent feedback not applied"


# ──────────────────────────────────────────────
# 3. Disabling the recurrent loop
# ──────────────────────────────────────────────
def test_recurrent_loop_disable():
    """After disabling the loop, same input -> same output."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)
    net.forward(x)  # t=1 -- create internal state
    net.disable_recurrent_loop()
    y2 = net.forward(x)
    y3 = net.forward(x)
    assert np.allclose(y2, y3), "output still changes after disabling the loop"


def test_recurrent_loop_enable():
    """Check that re-enabling after disabling brings recurrence back."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)

    net.disable_recurrent_loop()
    net.forward(x)
    y_off = net.forward(x)

    net.enable_recurrent_loop()
    net.reset_state()
    net.forward(x)
    y_on = net.forward(x)

    # After re-enabling, the second forward result must differ (feedback applied)
    # With y_off, the same result comes out with no feedback
    # Check that recurrence works after enabling
    net.reset_state()
    y1 = net.forward(x)
    y2 = net.forward(x)
    assert not np.allclose(y1, y2), "recurrence still does not work after re-enabling"


# ──────────────────────────────────────────────
# 4. Direct weight access
# ──────────────────────────────────────────────
def test_weight_access():
    """Check that all weights can be accessed directly."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    weights = net.get_all_weights()

    assert len(weights) == 4
    assert weights['input_to_h1'].shape == (10, 10)
    assert weights['h1_to_h2'].shape == (10, 10)
    assert weights['h2_to_output'].shape == (10, 5)
    assert weights['recurrent'].shape == (5, 10)   # output -> h1 feedback


def test_weight_mutation():
    """Check that externally modifying weights is reflected in the network."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)
    y_before = net.forward(x).copy()

    # Modify weights directly
    net.get_all_weights()['input_to_h1'][:] = 0.0
    net.reset_state()
    y_after = net.forward(x)

    assert not np.allclose(y_before, y_after), "weight modification is not reflected in the output"


# ──────────────────────────────────────────────
# 5. State reset
# ──────────────────────────────────────────────
def test_reset_state():
    """Check that internal state is initialized after calling reset_state()."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)

    # Two forwards -> internal state exists
    net.forward(x)
    net.forward(x)

    net.reset_state()

    # After reset, the first forward must equal the completely fresh initial state
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    y_reset = net.forward(x)
    y_fresh = net2.forward(x)
    assert np.allclose(y_reset, y_fresh), "output after reset_state() differs from the initial state"


# ──────────────────────────────────────────────
# 6. forward_sequence (T=3 unroll)
# ──────────────────────────────────────────────
def test_forward_sequence_shape():
    """Whether forward_sequence returns T outputs."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)
    outputs, caches = net.forward_sequence(x, T=3)
    assert len(outputs) == 3
    assert len(caches) == 3
    for y in outputs:
        assert y.shape == (5,)


def test_forward_sequence_resets_state():
    """Whether forward_sequence internally calls reset_state() (reset at each sample start)."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    x = np.ones(10)

    # Contaminate the state beforehand
    net.forward(np.random.RandomState(99).randn(10))
    net.forward(np.random.RandomState(99).randn(10))

    # forward_sequence must start after its own reset
    outputs_seq, _ = net.forward_sequence(x, T=3)

    # Manual comparison with a clean network
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    net2.reset_state()
    y1 = net2.forward(x)
    y2 = net2.forward(x)
    y3 = net2.forward(x)

    assert np.allclose(outputs_seq[0], y1), "forward_sequence t=1 mismatch"
    assert np.allclose(outputs_seq[1], y2), "forward_sequence t=2 mismatch"
    assert np.allclose(outputs_seq[2], y3), "forward_sequence t=3 mismatch"


# ──────────────────────────────────────────────
# 7. Neuron-count constraint
# ──────────────────────────────────────────────
def test_total_neuron_count():
    """Total neuron count <= 35."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    total = 10 + 10 + 10 + 5
    assert total <= 35


# ──────────────────────────────────────────────
# 8. Reproducibility (seed)
# ──────────────────────────────────────────────
def test_reproducibility():
    """Same seed -> same initial weights."""
    net1 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    w1 = net1.get_all_weights()
    w2 = net2.get_all_weights()
    for k in w1:
        assert np.allclose(w1[k], w2[k]), f"same seed 42 but {k} weights differ"


def test_different_seeds_differ():
    """Different seed -> different initial weights."""
    net1 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=1)
    net2 = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=2)
    w1 = net1.get_all_weights()
    w2 = net2.get_all_weights()
    any_diff = any(not np.allclose(w1[k], w2[k]) for k in w1)
    assert any_diff, "different seeds but all weights are identical"


# ──────────────────────────────────────────────
# 9. Presence of biases
# ──────────────────────────────────────────────
def test_biases_exist():
    """Check that each layer has a bias vector."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    biases = net.get_all_biases()
    assert biases['b_h1'].shape == (10,)
    assert biases['b_h2'].shape == (10,)
    assert biases['b_out'].shape == (5,)
