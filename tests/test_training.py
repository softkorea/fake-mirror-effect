"""Phase 2: Training tests.

TDD -- tests first, implementation later.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import (
    generate_data,
    softmax,
    cross_entropy_loss,
    compute_loss_and_gradients,
    compute_batch_loss_and_gradients,
    gradient_check,
    train,
    evaluate_accuracy_at_timestep,
)


# ──────────────────────────────────────────────
# 1. Data generation
# ──────────────────────────────────────────────
def test_generate_data_shape():
    """Check data shape and one-hot targets."""
    X, y = generate_data(n_samples=50, noise_level=0.3, seed=0)
    assert X.shape == (50, 10)
    assert y.shape == (50, 5)
    # Each target is one-hot
    assert np.allclose(y.sum(axis=1), 1.0)
    assert np.all((y == 0) | (y == 1))


def test_generate_data_reproducibility():
    """Same seed -> same data."""
    X1, y1 = generate_data(50, 0.3, seed=42)
    X2, y2 = generate_data(50, 0.3, seed=42)
    assert np.allclose(X1, X2)
    assert np.allclose(y1, y2)


def test_generate_data_class_balance():
    """Class distribution is roughly uniform."""
    X, y = generate_data(500, 0.3, seed=0)
    counts = y.sum(axis=0)
    # Each class in the 50~150 range (expected value 100)
    assert np.all(counts >= 50) and np.all(counts <= 150)


# ──────────────────────────────────────────────
# 2. Softmax / loss function
# ──────────────────────────────────────────────
def test_softmax_valid_distribution():
    """softmax -> valid probability distribution."""
    p = softmax(np.array([2.0, 1.0, 0.1, -1.0, 3.5]))
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


def test_softmax_numerical_stability():
    """softmax is stable even for large values."""
    p = softmax(np.array([1000.0, 1001.0, 999.0, 1000.5, 998.0]))
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)


def test_cross_entropy_loss_positive():
    """CE loss is positive."""
    output = np.array([2.0, 1.0, 0.1, -1.0, 3.5])
    target = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    loss = cross_entropy_loss(output, target)
    assert loss > 0


def test_cross_entropy_perfect_prediction():
    """CE loss is minimized for a correct prediction."""
    # Large logit on the correct class
    output_good = np.array([10.0, -10.0, -10.0, -10.0, -10.0])
    output_bad = np.array([-10.0, 10.0, -10.0, -10.0, -10.0])
    target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert cross_entropy_loss(output_good, target) < cross_entropy_loss(output_bad, target)


# ──────────────────────────────────────────────
# 3. Gradient Check
# ──────────────────────────────────────────────
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("skip_conn", [False, True])
def test_gradient_check(recurrent, skip_conn):
    """Numerical-gradient verification for all architecture variants."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5,
                       seed=7, skip_connection=skip_conn)
    if not recurrent:
        net.disable_recurrent_loop()

    rng = np.random.RandomState(7)
    x = rng.randn(10)
    target = np.zeros(5)
    target[2] = 1.0

    max_rel_error = gradient_check(net, x, target, T=3)
    print(f"Gradient check (rec={recurrent}, skip={skip_conn}): {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, \
        f"Gradient check failed (rec={recurrent}, skip={skip_conn}): {max_rel_error:.2e}"


# ──────────────────────────────────────────────
# 4. Training
# ──────────────────────────────────────────────
def test_training_reduces_loss():
    """Whether training reduces the loss by at least 50%."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    X, y = generate_data(n_samples=100, noise_level=0.3, seed=42)

    loss_before, _ = compute_batch_loss_and_gradients(net, X, y, T=3)
    train(net, X, y, epochs=300, lr=0.01, T=3)
    loss_after, _ = compute_batch_loss_and_gradients(net, X, y, T=3)

    print(f"Loss: {loss_before:.4f} → {loss_after:.4f} ({loss_after/loss_before:.1%})")
    assert loss_after < loss_before * 0.5


# ──────────────────────────────────────────────
# 5. Self-Correction (core hypothesis premise)
# ──────────────────────────────────────────────
def test_self_correction_occurs():
    """After training, acc_t3 > acc_t1 -- self-correction occurs."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    X_train, y_train = generate_data(n_samples=200, noise_level=0.5, seed=42)
    X_test, y_test = generate_data(n_samples=100, noise_level=0.5, seed=99)

    train(net, X_train, y_train, epochs=500, lr=0.01, T=3)

    acc_t1 = evaluate_accuracy_at_timestep(net, X_test, y_test, t=1)
    acc_t3 = evaluate_accuracy_at_timestep(net, X_test, y_test, t=3)
    gain = acc_t3 - acc_t1

    print(f"t=1 acc: {acc_t1:.3f}, t=3 acc: {acc_t3:.3f}, gain: {gain:.3f}")
    assert acc_t3 > acc_t1, f"No self-correction: t=1={acc_t1:.3f}, t=3={acc_t3:.3f}"


def test_state_isolation_between_samples():
    """Check that there is no state leakage between samples."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=42)
    X, y = generate_data(n_samples=10, noise_level=0.3, seed=42)

    # Process in order
    outputs_forward = []
    for i in range(len(X)):
        outs, _ = net.forward_sequence(X[i], T=3)
        outputs_forward.append([o.copy() for o in outs])

    # Process in reverse order -- forward_sequence resets, so results must be identical
    outputs_reverse = []
    for i in reversed(range(len(X))):
        outs, _ = net.forward_sequence(X[i], T=3)
        outputs_reverse.append([o.copy() for o in outs])
    outputs_reverse.reverse()

    for i in range(len(X)):
        for t in range(3):
            assert np.allclose(outputs_forward[i][t], outputs_reverse[i][t]), \
                f"sample {i}, t={t+1}: output differs by processing order -- state leakage"
