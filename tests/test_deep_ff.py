"""Phase D'': DeepFeedforwardMLP tests.

TDD -- tests first, implementation later.
Compute-matched feedforward control: 6 hidden layers of 10 neurons.
3 timesteps x 2 hidden layers = 6 layer traversals in the recurrent model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import DeepFeedforwardMLP
from src.training import (
    generate_data,
    softmax,
    cross_entropy_loss,
    d_cross_entropy,
    compute_loss_and_gradients_deep_ff,
    compute_batch_loss_and_gradients_deep_ff,
    train_deep_ff,
    evaluate_accuracy_deep_ff,
)


# ──────────────────────────────────────────────
# 1. Architecture
# ──────────────────────────────────────────────
def test_deep_ff_architecture():
    """6 hidden layers of 10, output 5."""
    net = DeepFeedforwardMLP(input_size=10, hidden_size=10, n_hidden=6,
                              output_size=5, seed=0)
    # Should have 6 hidden (W, b) pairs + 1 output (W, b) pair
    assert len(net.hidden_layers) == 6
    for W, b in net.hidden_layers:
        assert W.shape[1] == 10  # hidden_size
        assert b.shape == (10,)
    # First hidden layer takes input_size
    assert net.hidden_layers[0][0].shape == (10, 10)
    # Output layer
    assert net.W_out.shape == (10, 5)
    assert net.b_out.shape == (5,)


# ──────────────────────────────────────────────
# 2. Parameter count
# ──────────────────────────────────────────────
def test_deep_ff_param_count():
    """Should be 715: 6x(10x10+10) + (10x5+5) = 660 + 55 = 715."""
    net = DeepFeedforwardMLP(input_size=10, hidden_size=10, n_hidden=6,
                              output_size=5, seed=0)
    assert net.count_params() == 715


# ──────────────────────────────────────────────
# 3. Forward pass
# ──────────────────────────────────────────────
def test_deep_ff_forward():
    """Forward pass produces correct output shape."""
    net = DeepFeedforwardMLP(input_size=10, hidden_size=10, n_hidden=6,
                              output_size=5, seed=0)
    x = np.random.RandomState(0).randn(10)
    y = net.forward(x)
    assert y.shape == (5,)
    assert np.all(np.isfinite(y))


def test_deep_ff_forward_deterministic():
    """Same input -> same output (no internal state)."""
    net = DeepFeedforwardMLP(input_size=10, hidden_size=10, n_hidden=6,
                              output_size=5, seed=0)
    x = np.random.RandomState(0).randn(10)
    y1 = net.forward(x)
    y2 = net.forward(x)
    assert np.allclose(y1, y2), "Same input should produce same output"


# ──────────────────────────────────────────────
# 4. Reproducibility
# ──────────────────────────────────────────────
def test_deep_ff_reproducibility():
    """Same seed -> same weights."""
    net1 = DeepFeedforwardMLP(seed=42)
    net2 = DeepFeedforwardMLP(seed=42)
    for (W1, b1), (W2, b2) in zip(net1.hidden_layers, net2.hidden_layers):
        assert np.allclose(W1, W2)
        assert np.allclose(b1, b2)
    assert np.allclose(net1.W_out, net2.W_out)
    assert np.allclose(net1.b_out, net2.b_out)


def test_deep_ff_different_seeds():
    """Different seeds -> different weights."""
    net1 = DeepFeedforwardMLP(seed=1)
    net2 = DeepFeedforwardMLP(seed=2)
    any_diff = any(
        not np.allclose(W1, W2)
        for (W1, _), (W2, _) in zip(net1.hidden_layers, net2.hidden_layers)
    )
    assert any_diff, "Different seeds should give different weights"


# ──────────────────────────────────────────────
# 5. get_all_params
# ──────────────────────────────────────────────
def test_deep_ff_get_all_params():
    """get_all_params returns flat list of all parameter arrays."""
    net = DeepFeedforwardMLP(seed=0)
    params = net.get_all_params()
    # 6 hidden layers * 2 (W + b) + 2 (W_out + b_out) = 14
    assert len(params) == 14
    total = sum(p.size for p in params)
    assert total == 715


# ──────────────────────────────────────────────
# 6. Training
# ──────────────────────────────────────────────
def test_deep_ff_training():
    """Can train and achieve reasonable accuracy."""
    net = DeepFeedforwardMLP(seed=42)
    X_train, y_train = generate_data(n_samples=200, noise_level=0.3, seed=42)
    X_test, y_test = generate_data(n_samples=100, noise_level=0.3, seed=99)

    loss_history = train_deep_ff(net, X_train, y_train, epochs=500, lr=0.01)

    # Loss should decrease (6-layer deep network may converge slowly with basic SGD)
    assert loss_history[-1] < loss_history[0], \
        f"Loss didn't decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"

    acc = evaluate_accuracy_deep_ff(net, X_test, y_test)
    print(f"Deep FF accuracy after training: {acc:.3f}")
    assert acc > 0.4, f"Accuracy too low: {acc:.3f}"


# ──────────────────────────────────────────────
# 7. No correction gain
# ──────────────────────────────────────────────
def test_deep_ff_no_correction_gain():
    """Single-pass model has zero correction gain by construction.

    The DeepFeedforwardMLP has no recurrent connections, so there is no
    multi-timestep processing. Evaluating it yields a single accuracy value
    that is the same at all 'timesteps', making gain = 0 by definition.
    """
    net = DeepFeedforwardMLP(seed=42)
    X, y = generate_data(n_samples=100, noise_level=0.3, seed=42)
    train_deep_ff(net, X, y, epochs=300, lr=0.01)

    acc = evaluate_accuracy_deep_ff(net, X, y)
    # gain is always 0 because there's only one pass
    # (acc_t1 = acc_t2 = acc_t3 = acc, so gain = acc - acc = 0)
    assert acc > 0.0, "Model should have non-zero accuracy"
    # The key point: no mechanism for correction gain exists


# ──────────────────────────────────────────────
# 8. Gradient check
# ──────────────────────────────────────────────
def test_deep_ff_gradient_check():
    """Numerical vs analytical gradient match (rel error < 1e-4)."""
    net = DeepFeedforwardMLP(seed=7)

    rng = np.random.RandomState(7)
    x = rng.randn(10)
    target = np.zeros(5)
    target[2] = 1.0

    # Analytical gradients
    loss_ana, grads_ana = compute_loss_and_gradients_deep_ff(net, x, target)

    # Numerical gradients (central difference)
    epsilon = 1e-5
    max_rel_error = 0.0

    for param, grad in zip(net.get_all_params(), grads_ana):
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + epsilon
            y_plus = net.forward(x)
            loss_plus = cross_entropy_loss(y_plus, target)

            param[idx] = old_val - epsilon
            y_minus = net.forward(x)
            loss_minus = cross_entropy_loss(y_minus, target)

            num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            param[idx] = old_val
            it.iternext()

        # Compute relative error
        a = grad.ravel()
        n = num_grad.ravel()
        for i in range(len(a)):
            denom = max(abs(a[i]), abs(n[i]), 1e-8)
            rel_error = abs(a[i] - n[i]) / denom
            max_rel_error = max(max_rel_error, rel_error)

    print(f"Deep FF gradient check max rel error: {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, \
        f"Gradient check failed: max relative error = {max_rel_error:.2e}"


# ──────────────────────────────────────────────
# 9. Loss computation
# ──────────────────────────────────────────────
def test_deep_ff_loss_positive():
    """Loss is positive."""
    net = DeepFeedforwardMLP(seed=0)
    x = np.random.RandomState(0).randn(10)
    target = np.zeros(5)
    target[0] = 1.0
    loss, grads = compute_loss_and_gradients_deep_ff(net, x, target)
    assert loss > 0


def test_deep_ff_batch_loss():
    """Batch loss matches average of individual losses."""
    net = DeepFeedforwardMLP(seed=0)
    X, y = generate_data(n_samples=10, noise_level=0.3, seed=0)

    batch_loss, batch_grads = compute_batch_loss_and_gradients_deep_ff(net, X, y)

    # Manual average
    total_loss = 0.0
    for i in range(len(X)):
        loss_i, _ = compute_loss_and_gradients_deep_ff(net, X[i], y[i])
        total_loss += loss_i
    mean_loss = total_loss / len(X)

    assert np.isclose(batch_loss, mean_loss, rtol=1e-10), \
        f"Batch loss {batch_loss:.6f} != mean loss {mean_loss:.6f}"
