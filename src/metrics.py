"""Metrics module - experiment metric computations.

compute_all_metrics is a single-pass implementation (O(N) instead of the original O(6N)).
Also includes neuron-importance analysis helpers.
"""

import numpy as np
from src.training import softmax, evaluate_accuracy_at_timestep


def compute_correction_gain(net, X, y):
    """correction_gain = acc_t3 - acc_t1."""
    acc_t1 = evaluate_accuracy_at_timestep(net, X, y, t=1)
    acc_t3 = evaluate_accuracy_at_timestep(net, X, y, t=3)
    return acc_t3 - acc_t1


def compute_recurrent_contribution_norm(net, X):
    """Mean ||feedback @ W_rec|| (feedback contribution magnitude at t=2, t=3)."""
    norms = []
    for i in range(len(X)):
        outputs, caches = net.forward_sequence(X[i], T=3)
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ net.W_rec
            norms.append(np.linalg.norm(contrib))
    return float(np.mean(norms))


def compute_step_delta(net, X):
    """Mean ||y_t - y_{t-1}|| (output change magnitude at t=2, t=3)."""
    deltas = []
    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        for t in range(1, 3):
            delta = np.linalg.norm(outputs[t] - outputs[t - 1])
            deltas.append(delta)
    return float(np.mean(deltas))


def compute_ece(net, X, y, n_bins=10):
    """Expected Calibration Error.

    Uses max softmax probability as the confidence.
    """
    confidences = []
    accuracies = []

    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        probs = softmax(outputs[2])
        conf = np.max(probs)
        pred = np.argmax(probs)
        true = np.argmax(y[i])
        confidences.append(conf)
        accuracies.append(float(pred == true))

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += mask.sum() / len(X) * abs(avg_acc - avg_conf)

    return float(ece)


def compute_all_metrics(net, X, y):
    """Compute all metrics in a single pass.

    Calls forward_sequence once per sample (down from 6x in the previous implementation).

    Returns:
        dict with keys: acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm
    """
    n = len(X)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = net.forward_sequence(X[i], T=3)
        true_cls = np.argmax(y[i])

        # accuracy at each timestep
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        # recurrent contribution norm (t=2, t=3)
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        # step delta (t=2, t=3)
        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        # ECE data (t=3)
        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    # ECE
    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)),
        'delta_norm': float(np.mean(deltas)),
    }


def compute_all_metrics_vn(net, X_seq, y):
    """Compute variable-noise metrics in a single pass.

    Args:
        net: RecurrentMLP
        X_seq: (n_samples, T, input_size) per-timestep inputs
        y: (n_samples, n_classes) one-hot

    Returns:
        dict with keys: acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm
    """
    n = len(X_seq)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = net.forward_sequence_vn(X_seq[i], T=3)
        true_cls = np.argmax(y[i])

        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(
            accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)),
        'delta_norm': float(np.mean(deltas)),
    }


def compute_all_metrics_with_clone_vn(target_net, clone_net, X_seq, y):
    """Compute VN clone-feedback metrics.

    Args:
        target_net: model under evaluation
        clone_net: model providing the feedback signal
        X_seq: (n_samples, T, input_size) per-timestep inputs
        y: (n_samples, n_classes) one-hot

    Returns:
        dict with keys: acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm
    """
    from src.ablation import forward_sequence_with_clone_vn

    n = len(X_seq)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = forward_sequence_with_clone_vn(
            target_net, clone_net, X_seq[i], T=3)
        true_cls = np.argmax(y[i])

        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ target_net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(
            accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)) if r_norms else 0.0,
        'delta_norm': float(np.mean(deltas)) if deltas else 0.0,
    }


def compute_all_metrics_with_clone(target_net, clone_net, X, y):
    """Compute metrics with clone feedback.

    Replaces target_net's feedback with clone_net's output.
    Returns the same dict shape as compute_all_metrics.
    """
    from src.ablation import forward_sequence_with_clone

    n = len(X)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = forward_sequence_with_clone(target_net, clone_net, X[i], T=3)
        true_cls = np.argmax(y[i])

        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        # recurrent contribution norm (t=2, t=3) - uses per-timestep caches
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ target_net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        # step delta
        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        # ECE data (t=3)
        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)) if r_norms else 0.0,
        'delta_norm': float(np.mean(deltas)) if deltas else 0.0,
    }


def compute_all_metrics_with_aligned_clone(target_net, donor_net, X, y, align_fn):
    """Compute metrics with aligned clone feedback.

    Replaces target_net's feedback with donor_net's output transformed by align_fn.
    Returns the same dict shape as compute_all_metrics.

    Args:
        target_net: model under evaluation
        donor_net: model providing the feedback signal
        X: test inputs (n_samples, input_size)
        y: test labels (n_samples, n_classes) one-hot
        align_fn: alignment function (align_norm or align_affine)

    Returns:
        dict with keys: acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm
    """
    from src.ablation import forward_sequence_with_aligned_clone

    n = len(X)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = forward_sequence_with_aligned_clone(
            target_net, donor_net, X[i], align_fn, T=3
        )
        true_cls = np.argmax(y[i])

        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        # recurrent contribution norm (t=2, t=3)
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ target_net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        # step delta
        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        # ECE data (t=3)
        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)) if r_norms else 0.0,
        'delta_norm': float(np.mean(deltas)) if deltas else 0.0,
    }


def compute_all_metrics_multi_donor(target_net, donor_nets, X, y):
    """Compute metrics with multi-donor ensemble feedback.

    Replaces target_net's feedback with the mean of several donor models' outputs.
    Returns the same dict shape as compute_all_metrics.

    Args:
        target_net: model under evaluation
        donor_nets: list of donor models
        X: test inputs (n_samples, input_size)
        y: test labels (n_samples, n_classes) one-hot

    Returns:
        dict with keys: acc_t1, acc_t2, acc_t3, gain, ece, r_norm, delta_norm
    """
    from src.ablation import forward_sequence_multi_donor

    n = len(X)
    correct_t1 = 0
    correct_t2 = 0
    correct_t3 = 0
    r_norms = []
    deltas = []
    confidences = []
    accuracies_for_ece = []

    for i in range(n):
        outputs, caches = forward_sequence_multi_donor(
            target_net, donor_nets, X[i], T=3
        )
        true_cls = np.argmax(y[i])

        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[1]) == true_cls:
            correct_t2 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

        # recurrent contribution norm (t=2, t=3)
        for t in [1, 2]:
            feedback = caches[t]['feedback']
            contrib = feedback @ target_net.W_rec
            r_norms.append(np.linalg.norm(contrib))

        # step delta
        for t in range(1, 3):
            deltas.append(np.linalg.norm(outputs[t] - outputs[t - 1]))

        # ECE data (t=3)
        probs = softmax(outputs[2])
        confidences.append(np.max(probs))
        accuracies_for_ece.append(float(np.argmax(outputs[2]) == true_cls))

    acc_t1 = correct_t1 / n
    acc_t2 = correct_t2 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    confidences = np.array(confidences)
    accuracies_for_ece = np.array(accuracies_for_ece)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for b in range(10):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(accuracies_for_ece[mask].mean() - confidences[mask].mean())

    return {
        'acc_t1': float(acc_t1),
        'acc_t2': float(acc_t2),
        'acc_t3': float(acc_t3),
        'gain': float(gain),
        'ece': float(ece),
        'r_norm': float(np.mean(r_norms)) if r_norms else 0.0,
        'delta_norm': float(np.mean(deltas)) if deltas else 0.0,
    }


# ----------------------------------------------
# Wilcoxon Signed-Rank Exact Test (scipy-free)
# ----------------------------------------------

def wilcoxon_exact(x, y, alternative='two-sided'):
    """Wilcoxon signed-rank exact test (paired).

    For N <= 25 the exact p-value is computed by enumerating all 2^N
    sign assignments. Verified to match scipy.stats.wilcoxon for tie-free
    data; under tied |differences| this implementation enumerates the exact
    sign-flip distribution conditional on the observed midranks (average
    ranks), whereas scipy's forced-exact mode uses arbitrarily tie-broken
    integer ranks, so p-values can differ slightly in the tied case.

    Args:
        x, y: 1-D array-like paired samples (same length)
        alternative: 'two-sided' (default), 'less', or 'greater'.
          'less'    : H1 is median(x - y) < 0   (small W+ supports H1)
          'greater' : H1 is median(x - y) > 0   (large W+ supports H1)
          Matches scipy.stats.wilcoxon convention.

    Returns:
        (T, p_value) where T = min(T+, T-) for two-sided; T = T+ for one-sided.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x - y

    # Remove zero differences (with float tolerance for accuracy-derived values)
    d = d[np.abs(d) > 1e-9]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    if n > 25:
        raise ValueError(f"wilcoxon_exact: n={n} too large for exact enumeration (2^{n}). "
                         "Use scipy.stats.wilcoxon for n>25.")

    # Rank absolute differences
    abs_d = np.abs(d)
    order = np.argsort(abs_d, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    # Handle ties: assign average rank (with float tolerance)
    sorted_abs = abs_d[order]
    i = 0
    while i < n:
        j = i
        while j < n and abs(sorted_abs[j] - sorted_abs[i]) < 1e-9:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[order[i:j]])
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j

    # T+ and T-
    T_plus = float(np.sum(ranks[d > 0]))
    T_minus = float(np.sum(ranks[d < 0]))
    rank_sum_total = float(ranks.sum())

    # Build the per-permutation T+ distribution (sum of ranks at the set bits)
    # under each sign assignment, reused across all three alternatives.
    # Vectorized in memory-bounded chunks (replaces an O(2^n * n) Python double
    # loop). Bit-identical to the loop: every partial sum is a multiple of 0.5
    # and <= n(n+1)/2, hence exactly representable in float64 regardless of
    # summation order. The 1e-9 tolerance / alternative / p-value logic below is
    # unchanged.
    n_perms = 1 << n  # 2^n
    bits = 1 << np.arange(n, dtype=np.int64)            # (n,)
    tplus_vals = np.empty(n_perms, dtype=np.float64)
    chunk = 1 << 16
    for start in range(0, n_perms, chunk):
        stop = min(start + chunk, n_perms)
        masks = np.arange(start, stop, dtype=np.int64)[:, None]   # (m, 1)
        member = ((masks & bits) > 0).astype(np.float64)          # (m, n)
        tplus_vals[start:stop] = member @ ranks                   # (m,)

    if alternative == 'two-sided':
        T = min(T_plus, T_minus)
        t_min_vals = np.minimum(tplus_vals, rank_sum_total - tplus_vals)
        p = float(np.sum(t_min_vals <= T + 1e-9) / n_perms)
        return float(T), p
    elif alternative == 'less':
        # H1: median(d) < 0  ->  small W+ supports H1
        p = float(np.sum(tplus_vals <= T_plus + 1e-9) / n_perms)
        return float(T_plus), p
    elif alternative == 'greater':
        # H1: median(d) > 0  ->  large W+ supports H1
        p = float(np.sum(tplus_vals >= T_plus - 1e-9) / n_perms)
        return float(T_plus), p
    else:
        raise ValueError(f"alternative must be 'two-sided', 'less', or 'greater'; got {alternative!r}")


# ----------------------------------------------
# Neuron Importance (data for heatmap)
# ----------------------------------------------

def compute_neuron_importance(net, X, y):
    """Measure intelligence / self-correction importance per hidden neuron.

    H1 neurons: decoupled ablation to avoid intelligence->correction confound.
      - Intelligence: ablate feedforward input (W_ih1 + bias) only, measure deltaacc_t1
      - Correction: ablate recurrent input (W_rec) only, measure deltagain
    H2 neurons: full knockout (W_h1h2 + bias), since H2 has no direct W_rec input.
      - Note: H2 correction importance may be confounded with intelligence importance.

    Returns:
        intelligence: dict {neuron_id: importance}
        correction: dict {neuron_id: importance}
    """
    baseline = compute_all_metrics(net, X, y)
    baseline_acc_t1 = baseline['acc_t1']
    baseline_gain = baseline['gain']

    intelligence = {}
    correction = {}

    # Hidden1 neurons (0-9) - decoupled ablation
    for idx in range(net.hidden1):
        # 1. Intelligence: ablate feedforward input only (W_ih1 + bias)
        col_ih1 = net.W_ih1[:, idx].copy()
        b_h1_val = net.b_h1[idx]
        net.W_ih1[:, idx] = 0.0
        net.b_h1[idx] = 0.0

        m_intel = compute_all_metrics(net, X, y)
        intelligence[f'h1_{idx}'] = baseline_acc_t1 - m_intel['acc_t1']

        # Restore feedforward
        net.W_ih1[:, idx] = col_ih1
        net.b_h1[idx] = b_h1_val

        # 2. Correction: ablate recurrent input only (W_rec)
        col_rec = net.W_rec[:, idx].copy()
        net.W_rec[:, idx] = 0.0

        m_corr = compute_all_metrics(net, X, y)
        correction[f'h1_{idx}'] = baseline_gain - m_corr['gain']

        # Restore recurrent
        net.W_rec[:, idx] = col_rec

    # Hidden2 neurons (0-9) - full knockout (no direct W_rec input)
    for idx in range(net.hidden2):
        col_h1h2 = net.W_h1h2[:, idx].copy()
        b_h2_val = net.b_h2[idx]

        net.W_h1h2[:, idx] = 0.0
        net.b_h2[idx] = 0.0

        m = compute_all_metrics(net, X, y)
        intelligence[f'h2_{idx}'] = baseline_acc_t1 - m['acc_t1']
        correction[f'h2_{idx}'] = baseline_gain - m['gain']

        net.W_h1h2[:, idx] = col_h1h2
        net.b_h2[idx] = b_h2_val

    return intelligence, correction
