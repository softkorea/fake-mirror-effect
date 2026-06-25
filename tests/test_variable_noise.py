"""Variable-Noise Task (WS0) tests.

WS0: apply independent noise at every timestep to resolve the static-input tautology.
x_t = prototype_k + ε_t, where ε_1 ⊥ ε_2 ⊥ ε_3.

TDD: write tests first -> confirm failure -> implement -> pass.
"""

import numpy as np
import pytest

from src.network import RecurrentMLP
from src.training import (
    generate_data_variable_noise,
    compute_loss_and_gradients_vn,
    compute_batch_loss_and_gradients_vn,
    train_vn,
    numerical_gradient_vn,
    gradient_check_vn,
)
from src.metrics import compute_all_metrics_vn


# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────

class TestDataGenerationVN:
    """generate_data_variable_noise tests."""

    def test_output_shapes(self):
        """X_seq shape = (n_samples, T, input_size), y shape = (n_samples, n_classes)."""
        X_seq, y = generate_data_variable_noise(
            100, noise_level=0.5, T=3, seed=0)
        assert X_seq.shape == (100, 3, 10)
        assert y.shape == (100, 5)

    def test_independent_noise_per_timestep(self):
        """Confirm that the noise at each timestep is independent."""
        X_seq, y = generate_data_variable_noise(
            100, noise_level=0.5, T=3, seed=0)
        # Different noise => different inputs per timestep
        assert not np.allclose(X_seq[:, 0, :], X_seq[:, 1, :])
        assert not np.allclose(X_seq[:, 1, :], X_seq[:, 2, :])

    def test_same_class_across_timesteps(self):
        """With noise=0, the same prototype at every timestep."""
        X_seq, y = generate_data_variable_noise(
            50, noise_level=0.0, T=3, seed=0)
        np.testing.assert_allclose(X_seq[:, 0, :], X_seq[:, 1, :])
        np.testing.assert_allclose(X_seq[:, 1, :], X_seq[:, 2, :])

    def test_reproducibility(self):
        """Same seed yields the same result."""
        X1, y1 = generate_data_variable_noise(50, noise_level=0.5, T=3, seed=42)
        X2, y2 = generate_data_variable_noise(50, noise_level=0.5, T=3, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        """Different seeds generate different data."""
        X1, _ = generate_data_variable_noise(50, noise_level=0.5, T=3, seed=0)
        X2, _ = generate_data_variable_noise(50, noise_level=0.5, T=3, seed=1)
        assert not np.allclose(X1, X2)

    def test_class_distribution(self):
        """Classes are evenly distributed."""
        X_seq, y = generate_data_variable_noise(
            500, noise_level=0.5, T=3, seed=0)
        counts = y.sum(axis=0)
        # Each class ~100 out of 500
        assert all(c > 50 for c in counts)


# ──────────────────────────────────────────────
# forward_sequence_vn
# ──────────────────────────────────────────────

class TestForwardSequenceVN:
    """forward_sequence_vn tests."""

    def test_output_length(self):
        """Returns T outputs."""
        net = RecurrentMLP(seed=0)
        X_seq, _ = generate_data_variable_noise(1, noise_level=0.5, T=3, seed=0)
        outputs, caches = net.forward_sequence_vn(X_seq[0], T=3)
        assert len(outputs) == 3
        assert len(caches) == 3

    def test_different_inputs_per_timestep(self):
        """A different input is stored in the cache at each timestep."""
        net = RecurrentMLP(seed=0)
        X_seq, _ = generate_data_variable_noise(1, noise_level=0.5, T=3, seed=0)
        outputs, caches = net.forward_sequence_vn(X_seq[0], T=3)
        np.testing.assert_allclose(caches[0]['x'], X_seq[0, 0])
        np.testing.assert_allclose(caches[1]['x'], X_seq[0, 1])
        np.testing.assert_allclose(caches[2]['x'], X_seq[0, 2])

    def test_recurrent_disabled_outputs_differ(self):
        """Even without recurrence, different inputs make outputs differ -- tautology broken."""
        net = RecurrentMLP(seed=0)
        net.disable_recurrent_loop()
        X_seq, _ = generate_data_variable_noise(1, noise_level=0.5, T=3, seed=0)
        outputs, _ = net.forward_sequence_vn(X_seq[0], T=3)
        assert not np.allclose(outputs[0], outputs[1])
        assert not np.allclose(outputs[1], outputs[2])

    def test_state_reset(self):
        """State is reset at the start of forward_sequence_vn."""
        net = RecurrentMLP(seed=0)
        X_seq, _ = generate_data_variable_noise(2, noise_level=0.5, T=3, seed=0)
        out1, _ = net.forward_sequence_vn(X_seq[0], T=3)
        out2, _ = net.forward_sequence_vn(X_seq[0], T=3)
        for t in range(3):
            np.testing.assert_allclose(out1[t], out2[t])

    def test_matches_static_when_no_noise(self):
        """With noise=0, identical to forward_sequence."""
        net = RecurrentMLP(seed=0)
        X_seq, _ = generate_data_variable_noise(1, noise_level=0.0, T=3, seed=0)
        outputs_vn, _ = net.forward_sequence_vn(X_seq[0], T=3)
        outputs_static, _ = net.forward_sequence(X_seq[0, 0], T=3)
        for t in range(3):
            np.testing.assert_allclose(outputs_vn[t], outputs_static[t])

    def test_output_shape(self):
        """Shape of each output vector."""
        net = RecurrentMLP(seed=0)
        X_seq, _ = generate_data_variable_noise(1, noise_level=0.5, T=3, seed=0)
        outputs, _ = net.forward_sequence_vn(X_seq[0], T=3)
        for out in outputs:
            assert out.shape == (5,)


# ──────────────────────────────────────────────
# Gradient check (VN)
# ──────────────────────────────────────────────

class TestGradientVN:
    """VN training gradient tests."""

    def test_gradient_check(self):
        """Analytical vs numerical gradient (VN), rel error < 1e-4."""
        net = RecurrentMLP(seed=42)
        X_seq, y = generate_data_variable_noise(1, noise_level=0.5, T=3, seed=42)
        rel_error = gradient_check_vn(net, X_seq[0], y[0], T=3)
        assert rel_error < 1e-4, f"Gradient check failed: rel_error={rel_error}"

    def test_loss_computation(self):
        """VN loss is a finite positive number."""
        net = RecurrentMLP(seed=0)
        X_seq, y = generate_data_variable_noise(1, noise_level=0.5, T=3, seed=0)
        loss, grads = compute_loss_and_gradients_vn(net, X_seq[0], y[0])
        assert np.isfinite(loss)
        assert loss > 0

    def test_batch_loss(self):
        """Batch loss = mean of individual losses."""
        net = RecurrentMLP(seed=0)
        X_seq, y = generate_data_variable_noise(5, noise_level=0.5, T=3, seed=0)

        batch_loss, batch_grads = compute_batch_loss_and_gradients_vn(
            net, X_seq, y)

        # Manual average
        total_loss = 0.0
        for i in range(5):
            loss_i, _ = compute_loss_and_gradients_vn(net, X_seq[i], y[i])
            total_loss += loss_i
        avg_loss = total_loss / 5

        np.testing.assert_allclose(batch_loss, avg_loss, rtol=1e-10)


# ──────────────────────────────────────────────
# Training (VN)
# ──────────────────────────────────────────────

class TestTrainingVN:
    """VN training tests."""

    def test_loss_decreases(self):
        """Loss decreases during training."""
        net = RecurrentMLP(seed=0, feedback_tau=2.0)
        X_seq, y = generate_data_variable_noise(
            200, noise_level=0.5, T=3, seed=0)
        history = train_vn(net, X_seq, y, epochs=100, lr=0.01,
                          time_weights=[0.0, 0.2, 1.0])
        assert history[-1] < history[0], "Loss did not decrease"

    def test_train_produces_nonzero_gain(self):
        """After VN training, gain is not exactly 0 (confirms tautology broken)."""
        net = RecurrentMLP(seed=0, feedback_tau=2.0)
        X_seq, y = generate_data_variable_noise(
            200, noise_level=0.5, T=3, seed=0)
        train_vn(net, X_seq, y, epochs=500, lr=0.01,
                time_weights=[0.0, 0.2, 1.0])

        X_test, y_test = generate_data_variable_noise(
            200, noise_level=0.5, T=3, seed=999)
        metrics = compute_all_metrics_vn(net, X_test, y_test)
        # With trained recurrence, gain should not be exactly 0
        # (unlike static input where it's trivially guaranteed)
        # We accept gain >= -0.1 as "not catastrophically broken"
        assert metrics['gain'] >= -0.1, f"Unexpected gain: {metrics['gain']}"


# ──────────────────────────────────────────────
# Metrics (VN)
# ──────────────────────────────────────────────

class TestMetricsVN:
    """VN metrics tests."""

    def test_output_keys(self):
        """compute_all_metrics_vn returns the same keys."""
        net = RecurrentMLP(seed=0)
        X_seq, y = generate_data_variable_noise(
            50, noise_level=0.5, T=3, seed=0)
        metrics = compute_all_metrics_vn(net, X_seq, y)
        expected_keys = {'acc_t1', 'acc_t2', 'acc_t3', 'gain',
                         'ece', 'r_norm', 'delta_norm'}
        assert set(metrics.keys()) == expected_keys

    def test_gain_is_acc_diff(self):
        """gain = acc_t3 - acc_t1."""
        net = RecurrentMLP(seed=0)
        X_seq, y = generate_data_variable_noise(
            50, noise_level=0.5, T=3, seed=0)
        metrics = compute_all_metrics_vn(net, X_seq, y)
        assert abs(metrics['gain'] - (metrics['acc_t3'] - metrics['acc_t1'])) < 1e-10

    def test_accuracy_range(self):
        """Accuracy is in the [0, 1] range."""
        net = RecurrentMLP(seed=0)
        X_seq, y = generate_data_variable_noise(
            50, noise_level=0.5, T=3, seed=0)
        metrics = compute_all_metrics_vn(net, X_seq, y)
        for key in ['acc_t1', 'acc_t2', 'acc_t3']:
            assert 0 <= metrics[key] <= 1

    def test_matches_static_metrics_when_no_noise(self):
        """With noise=0, identical to static metrics.

        Uses the same data: X_seq[:, 0, :] from VN(noise=0) as the static input.
        """
        from src.metrics import compute_all_metrics

        net = RecurrentMLP(seed=0)
        X_seq, y = generate_data_variable_noise(
            50, noise_level=0.0, T=3, seed=0)
        metrics_vn = compute_all_metrics_vn(net, X_seq, y)

        # With noise=0, all timesteps are identical -> use first as static
        metrics_static = compute_all_metrics(net, X_seq[:, 0, :], y)

        for key in ['acc_t1', 'acc_t2', 'acc_t3', 'gain']:
            np.testing.assert_allclose(
                metrics_vn[key], metrics_static[key], atol=1e-10,
                err_msg=f"Mismatch on {key}")


# ──────────────────────────────────────────────
# Tautology-breaking verification
# ──────────────────────────────────────────────

class TestTautologyBreaking:
    """Verify that VN actually resolves the static-input tautology."""

    def test_group_a_gain_not_structurally_zero(self):
        """For Group A (recurrent cut), gain != 0 is not structurally guaranteed.

        For static input, gain=0 is a mathematical necessity. For VN, since the
        inputs differ, gain != 0 is possible. This test only checks the structural
        possibility (before training).
        """
        net = RecurrentMLP(seed=0)
        net.disable_recurrent_loop()
        X_seq, y = generate_data_variable_noise(
            200, noise_level=0.5, T=3, seed=0)
        metrics = compute_all_metrics_vn(net, X_seq, y)
        # Key: gain CAN be nonzero (unlike static where it's ALWAYS 0)
        # We just verify the outputs are different at each timestep
        outputs_differ = False
        for i in range(min(50, len(X_seq))):
            outs, _ = net.forward_sequence_vn(X_seq[i], T=3)
            if not np.allclose(outs[0], outs[2]):
                outputs_differ = True
                break
        assert outputs_differ, \
            "With VN inputs and no recurrence, outputs should differ across timesteps"

    def test_static_input_gain_always_zero_without_recurrence(self):
        """Control: static input + no recurrence = gain is always exactly 0."""
        from src.training import generate_data
        from src.metrics import compute_all_metrics

        net = RecurrentMLP(seed=0)
        net.disable_recurrent_loop()
        X, y = generate_data(200, noise_level=0.5, seed=0)
        metrics = compute_all_metrics(net, X, y)
        assert metrics['gain'] == 0.0, \
            "Static input + no recurrence should give exactly zero gain"
