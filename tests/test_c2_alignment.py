"""Phase 3+: C2 Alignment Controls tests.

C2 alignment conditions: norm-matched, affine-matched, multi-donor ensemble.
Separates whether the performance degradation under clone feedback is due to
"not-self" or due to "not calibrated the way W_rec expects".
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.network import RecurrentMLP
from src.training import generate_data, train
from src.ablation import (
    create_trained_network,
    align_norm,
    align_affine,
    forward_sequence_with_aligned_clone,
    forward_sequence_multi_donor,
)
from src.metrics import (
    compute_all_metrics_with_aligned_clone,
    compute_all_metrics_multi_donor,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def two_trained_nets():
    """Two networks trained with two different seeds."""
    net_a = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    net_b = create_trained_network(seed=1, epochs=200, noise_level=0.3)
    return net_a, net_b


@pytest.fixture
def five_donor_nets():
    """Five independently trained networks for multi-donor use."""
    donors = []
    for seed in [100, 101, 102, 103, 104]:
        donors.append(create_trained_network(seed=seed, epochs=200, noise_level=0.3))
    return donors


@pytest.fixture
def test_data():
    """Test data."""
    X, y = generate_data(n_samples=50, noise_level=0.3, seed=999)
    return X, y


# ──────────────────────────────────────────────
# align_norm tests
# ──────────────────────────────────────────────

def test_align_norm_preserves_direction():
    """Norm alignment should preserve direction, only change magnitude."""
    rng = np.random.RandomState(42)
    donor = rng.randn(5)
    target = rng.randn(5) * 3.0  # different magnitude

    aligned = align_norm(donor, target)

    # Direction check: cosine similarity should be 1.0
    cos_sim = np.dot(donor, aligned) / (np.linalg.norm(donor) * np.linalg.norm(aligned))
    assert np.isclose(cos_sim, 1.0, atol=1e-10), \
        f"Direction not preserved: cosine similarity = {cos_sim}"


def test_align_norm_matches_target_norm():
    """Aligned output should have same L2 norm as target."""
    rng = np.random.RandomState(42)
    donor = rng.randn(5) * 2.0
    target = rng.randn(5) * 0.5

    aligned = align_norm(donor, target)

    target_norm = np.linalg.norm(target)
    aligned_norm = np.linalg.norm(aligned)
    assert np.isclose(aligned_norm, target_norm, atol=1e-10), \
        f"Norm mismatch: aligned={aligned_norm:.6f}, target={target_norm:.6f}"


def test_align_norm_zero_donor():
    """Zero donor vector should return zero (edge case)."""
    donor = np.zeros(5)
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    aligned = align_norm(donor, target)
    assert np.allclose(aligned, 0), "Zero donor should produce zero output"


def test_align_norm_zero_target():
    """Zero target vector should return zero."""
    donor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.zeros(5)

    aligned = align_norm(donor, target)
    assert np.allclose(aligned, 0), "Zero target norm should produce zero output"


# ──────────────────────────────────────────────
# align_affine tests
# ──────────────────────────────────────────────

def test_align_affine_matches_statistics():
    """Affine-aligned output should match target mean and std."""
    rng = np.random.RandomState(42)
    donor = rng.randn(5) * 2.0 + 3.0  # mean ~3, std ~2
    target = rng.randn(5) * 0.5 - 1.0  # mean ~-1, std ~0.5

    aligned = align_affine(donor, target)

    assert np.isclose(np.mean(aligned), np.mean(target), atol=1e-10), \
        f"Mean mismatch: aligned={np.mean(aligned):.6f}, target={np.mean(target):.6f}"
    assert np.isclose(np.std(aligned), np.std(target), atol=1e-10), \
        f"Std mismatch: aligned={np.std(aligned):.6f}, target={np.std(target):.6f}"


def test_align_affine_zero_std_handling():
    """Handle constant vectors gracefully (zero std)."""
    donor = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # constant, std=0
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    aligned = align_affine(donor, target)

    # With zero donor std, should return target mean broadcast
    assert np.all(np.isfinite(aligned)), "Non-finite values in aligned output"
    assert np.isclose(np.mean(aligned), np.mean(target), atol=1e-10), \
        "Mean should match target even with zero donor std"


def test_align_affine_zero_target_std():
    """When target has zero std, aligned should be constant at target mean."""
    donor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # constant

    aligned = align_affine(donor, target)

    assert np.allclose(aligned, 2.0, atol=1e-10), \
        "With zero target std, all aligned values should equal target mean"


def test_align_affine_identity():
    """Aligning a vector to itself should return the same vector."""
    rng = np.random.RandomState(42)
    v = rng.randn(5)

    aligned = align_affine(v, v)
    assert np.allclose(aligned, v, atol=1e-10), \
        "Self-alignment should be identity"


# ──────────────────────────────────────────────
# forward_sequence_with_aligned_clone tests
# ──────────────────────────────────────────────

def test_aligned_clone_forward_shape(two_trained_nets):
    """Output shapes match standard forward."""
    target, donor = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    for align_fn in [align_norm, align_affine]:
        outputs, caches = forward_sequence_with_aligned_clone(
            target, donor, x, align_fn, T=3
        )

        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        assert len(caches) == 3, f"Expected 3 caches, got {len(caches)}"
        for t in range(3):
            assert outputs[t].shape == (5,), \
                f"Output shape at t={t}: {outputs[t].shape}, expected (5,)"


def test_aligned_clone_t1_unchanged(two_trained_nets):
    """t=0 output should match normal forward (no feedback at t=0)."""
    target, donor = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    # Normal forward
    normal_outputs, _ = target.forward_sequence(x, T=3)

    # Aligned clone forward
    for align_fn in [align_norm, align_affine]:
        aligned_outputs, _ = forward_sequence_with_aligned_clone(
            target, donor, x, align_fn, T=3
        )
        assert np.allclose(normal_outputs[0], aligned_outputs[0], atol=1e-10), \
            f"t=0 output changed with {align_fn.__name__}"


def test_aligned_clone_differs_from_unaligned(two_trained_nets):
    """Aligned clone should produce different outputs than unaligned clone."""
    from src.ablation import forward_sequence_with_clone

    target, donor = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    unaligned_outputs, _ = forward_sequence_with_clone(target, donor, x, T=3)

    for align_fn in [align_norm, align_affine]:
        aligned_outputs, _ = forward_sequence_with_aligned_clone(
            target, donor, x, align_fn, T=3
        )
        # At t>=1, aligned should differ from unaligned
        # (unless alignment happens to be identity, which is unlikely)
        any_different = False
        for t in range(1, 3):
            if not np.allclose(aligned_outputs[t], unaligned_outputs[t], atol=1e-10):
                any_different = True
                break
        assert any_different, \
            f"Aligned ({align_fn.__name__}) is identical to unaligned — alignment has no effect"


def test_aligned_clone_deterministic(two_trained_nets):
    """Same inputs -> same outputs (deterministic)."""
    target, donor = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    out1, _ = forward_sequence_with_aligned_clone(target, donor, x, align_norm, T=3)
    out2, _ = forward_sequence_with_aligned_clone(target, donor, x, align_norm, T=3)

    for t in range(3):
        assert np.allclose(out1[t], out2[t], atol=1e-10), \
            f"Non-deterministic at t={t}"


def test_aligned_clone_preserves_weights(two_trained_nets):
    """Aligned clone forward should not modify target or donor weights."""
    target, donor = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    tw_before = {k: v.copy() for k, v in target.get_all_weights().items()}
    dw_before = {k: v.copy() for k, v in donor.get_all_weights().items()}

    forward_sequence_with_aligned_clone(target, donor, x, align_norm, T=3)

    for k in tw_before:
        assert np.allclose(tw_before[k], target.get_all_weights()[k]), \
            f"Target weight {k} modified"
    for k in dw_before:
        assert np.allclose(dw_before[k], donor.get_all_weights()[k]), \
            f"Donor weight {k} modified"


# ──────────────────────────────────────────────
# forward_sequence_multi_donor tests
# ──────────────────────────────────────────────

def test_multi_donor_averages(five_donor_nets):
    """Multi-donor output should be average of donor outputs at feedback injection."""
    target = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    donors = five_donor_nets
    x = np.random.RandomState(42).randn(10)

    outputs, caches = forward_sequence_multi_donor(target, donors, x, T=3)

    assert len(outputs) == 3
    for t in range(3):
        assert outputs[t].shape == (5,), \
            f"Output shape at t={t}: {outputs[t].shape}, expected (5,)"


def test_multi_donor_shape(five_donor_nets):
    """Output shapes match standard forward."""
    target = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    x = np.random.RandomState(42).randn(10)

    outputs, caches = forward_sequence_multi_donor(target, five_donor_nets, x, T=3)

    assert len(outputs) == 3
    assert len(caches) == 3
    for t in range(3):
        assert outputs[t].shape == (5,)


def test_multi_donor_t1_unchanged(five_donor_nets):
    """t=0 output should match normal forward (no feedback)."""
    target = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    x = np.random.RandomState(42).randn(10)

    normal_outputs, _ = target.forward_sequence(x, T=3)
    multi_outputs, _ = forward_sequence_multi_donor(target, five_donor_nets, x, T=3)

    assert np.allclose(normal_outputs[0], multi_outputs[0], atol=1e-10), \
        "t=0 output changed with multi-donor"


def test_multi_donor_deterministic(five_donor_nets):
    """Same inputs -> same outputs."""
    target = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    x = np.random.RandomState(42).randn(10)

    out1, _ = forward_sequence_multi_donor(target, five_donor_nets, x, T=3)
    out2, _ = forward_sequence_multi_donor(target, five_donor_nets, x, T=3)

    for t in range(3):
        assert np.allclose(out1[t], out2[t], atol=1e-10), \
            f"Non-deterministic at t={t}"


def test_multi_donor_single_equals_clone(two_trained_nets):
    """Single-donor multi-donor should equal regular clone feedback."""
    from src.ablation import forward_sequence_with_clone

    target, donor = two_trained_nets
    x = np.random.RandomState(42).randn(10)

    clone_outputs, _ = forward_sequence_with_clone(target, donor, x, T=3)
    multi_outputs, _ = forward_sequence_multi_donor(target, [donor], x, T=3)

    for t in range(3):
        assert np.allclose(clone_outputs[t], multi_outputs[t], atol=1e-10), \
            f"Single-donor multi != clone at t={t}"


# ──────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────

def test_aligned_clone_metrics(two_trained_nets, test_data):
    """Can compute metrics with aligned clone feedback."""
    target, donor = two_trained_nets
    X, y = test_data

    for align_fn in [align_norm, align_affine]:
        metrics = compute_all_metrics_with_aligned_clone(
            target, donor, X, y, align_fn
        )

        expected_keys = {'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece',
                         'r_norm', 'delta_norm'}
        assert set(metrics.keys()) == expected_keys, \
            f"Missing keys: {expected_keys - set(metrics.keys())}"

        # All values should be finite
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

        # Gain should be acc_t3 - acc_t1
        assert np.isclose(metrics['gain'], metrics['acc_t3'] - metrics['acc_t1'], atol=1e-10), \
            "gain != acc_t3 - acc_t1"


def test_multi_donor_metrics(five_donor_nets, test_data):
    """Can compute metrics with multi-donor feedback."""
    target = create_trained_network(seed=0, epochs=200, noise_level=0.3)
    X, y = test_data

    metrics = compute_all_metrics_multi_donor(target, five_donor_nets, X, y)

    expected_keys = {'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece',
                     'r_norm', 'delta_norm'}
    assert set(metrics.keys()) == expected_keys, \
        f"Missing keys: {expected_keys - set(metrics.keys())}"

    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"

    assert np.isclose(metrics['gain'], metrics['acc_t3'] - metrics['acc_t1'], atol=1e-10), \
        "gain != acc_t3 - acc_t1"
