"""Ablation module - selective connection cutting.

Group A: ablate_recurrent - zero out all recurrent weights
Group B1: ablate_random - zero out n random connections
Group B2: ablate_structural - zero out an entire layer
Group C1: use network.enable_scrambled_feedback()
Group C2: forward_sequence_with_clone() - inject another model's output as feedback
Group C2-norm/affine/multi: aligned clone-feedback controls
Group D: train a RecurrentMLP with the recurrent loop disabled
Group D': param-matched feedforward (skip connection)
"""

import numpy as np
from src.network import RecurrentMLP
from src.training import generate_data, train


# ----------------------------------------------
# Helper: create a trained network
# ----------------------------------------------

def create_trained_network(seed=42, epochs=200, noise_level=0.3,
                           n_samples=200, lr=0.01, T=3):
    """Return a trained RecurrentMLP.

    Test/debug fixture only -- NOT the paper protocol. The small defaults
    (epochs=200, noise_level=0.3) keep the unit tests in tests/ fast; every
    caller is a test that passes its own values. The paper/experiment pipeline
    trains via src.training.train(..., epochs=1000, lr=0.01, noise_level=0.5)
    in the experiments/ scripts, not through this helper.
    """
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed)
    X, y = generate_data(n_samples=n_samples, noise_level=noise_level, seed=seed)
    train(net, X, y, epochs=epochs, lr=lr, T=T)
    return net


# ----------------------------------------------
# Group A: cut the recurrent path
# ----------------------------------------------

def ablate_recurrent(net):
    """Zero out all recurrent weights (W_rec)."""
    net.W_rec[:] = 0.0


# ----------------------------------------------
# Group B1: random ablation
# ----------------------------------------------

def ablate_random(net, n_connections, seed=42):
    """Zero out n_connections random weights from across the network.

    Skips weights that are already zero; exactly n_connections non-zero
    weights are zeroed.
    """
    rng = np.random.RandomState(seed)

    # Collect every non-zero weight by flat index. Include the skip weight
    # (Group D') in the candidate pool when the network has one; the recurrent
    # Baseline (the only net B1 is applied to) has no skip, so this is a
    # correctness/future-proofing fix with no effect on the reported B1 results.
    weights = net.get_all_weights()
    weight_keys = [k for k in ('input_to_h1', 'h1_to_h2', 'h2_to_output',
                               'recurrent', 'skip') if k in weights]

    candidates = []  # (key, flat_index)
    for k in weight_keys:
        w = weights[k]
        nonzero_idx = np.flatnonzero(w)
        for idx in nonzero_idx:
            candidates.append((k, idx))

    assert len(candidates) >= n_connections, \
        f"Non-zero weights ({len(candidates)}) fewer than requested ({n_connections})"

    chosen = rng.choice(len(candidates), size=n_connections, replace=False)
    for c in chosen:
        k, idx = candidates[c]
        weights[k].flat[idx] = 0.0


# ----------------------------------------------
# Group B2: structural ablation
# ----------------------------------------------

def ablate_structural(net, layer):
    """Zero out an entire layer's weights.

    Args:
        layer: 'input_to_h1', 'h1_to_h2', 'h2_to_output', or 'recurrent'
    """
    weights = net.get_all_weights()
    assert layer in weights, f"Unknown layer: {layer}"
    weights[layer][:] = 0.0


# ----------------------------------------------
# Group C2: Clone Feedback
# ----------------------------------------------

def forward_sequence_interpolated(target_net, x, alpha, interp_type='zero',
                                  clone_net=None, shuffle_seed=42, T=3):
    """Interpolated feedback: feedback = alpha*y_self + (1-alpha)*y_other.

    WS2: Feedback Interpolation experiment.

    Args:
        target_net: model under evaluation
        x: input vector (static)
        alpha: interpolation coefficient (0=fully other, 1=fully self)
        interp_type: 'zero' | 'shuffle' | 'clone'
        clone_net: clone model (required when interp_type='clone')
        shuffle_seed: RNG seed for shuffle type
        T: number of timesteps

    Returns:
        outputs: list of T output vectors
        caches: list of T cache dicts
    """
    if interp_type == 'clone' and clone_net is None:
        raise ValueError(
            "forward_sequence_interpolated requires clone_net when interp_type='clone'")
    x = np.asarray(x, dtype=np.float64)
    target_net.reset_state()
    if clone_net is not None:
        clone_net.reset_state()

    shuffle_rng = np.random.RandomState(shuffle_seed)

    target_outputs = []
    target_caches = []
    clone_outputs = []

    for t in range(T):
        # Run clone independently (its own self-feedback loop)
        if interp_type == 'clone' and clone_net is not None:
            clone_y = clone_net.forward(x)
            clone_outputs.append(clone_y.copy())

        if t > 0:
            y_self = target_outputs[t - 1]

            if interp_type == 'zero':
                y_other = np.zeros_like(y_self)
            elif interp_type == 'shuffle':
                y_other = y_self.copy()
                shuffle_rng.shuffle(y_other)
            elif interp_type == 'clone':
                y_other = clone_outputs[t - 1]
            else:
                raise ValueError(f"Unknown interp_type: {interp_type}")

            interpolated = alpha * y_self + (1 - alpha) * y_other
            target_net._prev_output = interpolated.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


def forward_sequence_with_clone_vn(target_net, clone_net, x_seq, T=3):
    """VN clone feedback: per-timestep independent input + clone-sourced feedback.

    Args:
        target_net: model under evaluation
        clone_net: model providing the feedback signal
        x_seq: (T, input_size) per-timestep inputs
        T: number of timesteps

    Returns:
        target_outputs, target_caches
    """
    x_seq = np.asarray(x_seq, dtype=np.float64)
    target_net.reset_state()
    clone_net.reset_state()

    target_outputs = []
    target_caches = []
    clone_outputs = []

    for t in range(T):
        clone_y = clone_net.forward(x_seq[t])
        clone_outputs.append(clone_y.copy())

        if t > 0:
            target_net._prev_output = clone_outputs[t - 1].copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x_seq[t])
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


def forward_sequence_with_clone(target_net, clone_net, x, T=3):
    """Forward target_net but replace its feedback with clone_net's output.

    t=1: both models forward independently (no feedback)
    t=2,3: target's _prev_output is overwritten with clone's previous output

    Args:
        target_net: model under evaluation
        clone_net: model providing the feedback signal (same architecture, different seed)
        x: input vector
        T: number of timesteps

    Returns:
        target_outputs: list of T output vectors from target_net
        target_caches: list of T cache dicts from target_net
    """
    target_net.reset_state()
    clone_net.reset_state()

    target_outputs = []
    target_caches = []
    clone_outputs = []

    for t in range(T):
        clone_y = clone_net.forward(x)
        clone_outputs.append(clone_y.copy())

        if t > 0:
            target_net._prev_output = clone_outputs[t - 1].copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


# ----------------------------------------------
# C2 Per-Sample Alignment (oracle-based, NOT used in paper)
# Paper uses fit_learned_affine / fit_learned_affine_vn instead.
# ----------------------------------------------

def align_norm(donor_output, target_output):
    """Norm-matched alignment: scale donor to match target's L2 norm.

    aligned = donor * (||target|| / ||donor||)
    Preserves direction of donor, matches magnitude of target.

    Args:
        donor_output: donor model's output vector
        target_output: target model's own output vector (alignment reference)

    Returns:
        aligned output vector with same direction as donor, same norm as target
    """
    donor_output = np.asarray(donor_output, dtype=np.float64)
    target_output = np.asarray(target_output, dtype=np.float64)

    donor_norm = np.linalg.norm(donor_output)
    target_norm = np.linalg.norm(target_output)

    if donor_norm < 1e-12 or target_norm < 1e-12:
        return np.zeros_like(donor_output)

    return donor_output * (target_norm / donor_norm)


def align_affine(donor_output, target_output):
    """Affine alignment: match mean and std element-wise.

    aligned = (donor - mean(donor)) / std(donor) * std(target) + mean(target)
    Transforms donor to have same mean and std as target.

    Args:
        donor_output: donor model's output vector
        target_output: target model's own output vector (alignment reference)

    Returns:
        aligned output vector matching target's mean and std
    """
    donor_output = np.asarray(donor_output, dtype=np.float64)
    target_output = np.asarray(target_output, dtype=np.float64)

    donor_mean = np.mean(donor_output)
    donor_std = np.std(donor_output)
    target_mean = np.mean(target_output)
    target_std = np.std(target_output)

    if donor_std < 1e-12:
        # Constant donor: return target mean broadcast
        return np.full_like(donor_output, target_mean)

    # Standardize donor, then rescale to target statistics
    aligned = (donor_output - donor_mean) / donor_std * target_std + target_mean
    return aligned


# ----------------------------------------------
# C2 Aligned Clone Forward
# ----------------------------------------------

def forward_sequence_with_aligned_clone(target_net, donor_net, x, align_fn, T=3):
    """Forward target_net but replace its feedback with donor output transformed by align_fn.

    Runs target's own forward pass in parallel to provide an alignment reference.
    t=0: both models forward independently (no feedback)
    t>0:
      - target_ref = target's own forward t-1 output (alignment reference)
      - donor_prev = donor's t-1 output
      - aligned = align_fn(donor_prev, target_ref)
      - target._prev_output = aligned
      - target forward

    Args:
        target_net: model under evaluation
        donor_net: model providing the feedback signal
        x: input vector
        align_fn: alignment function (align_norm or align_affine)
        T: number of timesteps

    Returns:
        target_outputs: list of T output vectors from target_net
        target_caches: list of T cache dicts from target_net
    """
    x = np.asarray(x, dtype=np.float64)

    # Step 1: Run target normally to get reference outputs for alignment
    target_net.reset_state()
    target_ref_outputs = []
    for t in range(T):
        ref_y = target_net.forward(x)
        target_ref_outputs.append(ref_y.copy())

    # Step 2: Run donor normally to get donor outputs
    donor_net.reset_state()
    donor_outputs = []
    for t in range(T):
        donor_y = donor_net.forward(x)
        donor_outputs.append(donor_y.copy())

    # Step 3: Run target with aligned donor feedback
    target_net.reset_state()
    target_outputs = []
    target_caches = []

    for t in range(T):
        if t > 0:
            # Align donor's t-1 output to match target's t-1 reference output
            aligned = align_fn(donor_outputs[t - 1], target_ref_outputs[t - 1])
            target_net._prev_output = aligned.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


def forward_sequence_multi_donor(target_net, donor_nets, x, T=3):
    """Multi-donor ensemble: average output of multiple donors before injection.

    t=0: All forward independently (no feedback)
    t>0: target._prev_output = mean of all donors' t-1 outputs

    Args:
        target_net: model under evaluation
        donor_nets: list of donor models
        x: input vector
        T: number of timesteps

    Returns:
        target_outputs: list of T output vectors from target_net
        target_caches: list of T cache dicts from target_net
    """
    x = np.asarray(x, dtype=np.float64)

    # Run all donors normally to collect their outputs
    all_donor_outputs = []
    for donor in donor_nets:
        donor.reset_state()
        donor_outputs = []
        for t in range(T):
            donor_y = donor.forward(x)
            donor_outputs.append(donor_y.copy())
        all_donor_outputs.append(donor_outputs)

    # Run target with averaged donor feedback
    target_net.reset_state()
    target_outputs = []
    target_caches = []

    for t in range(T):
        if t > 0:
            # Average of all donors' t-1 outputs
            avg_output = np.mean(
                [all_donor_outputs[d][t - 1] for d in range(len(donor_nets))],
                axis=0
            )
            target_net._prev_output = avg_output.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


# ----------------------------------------------
# C2 Learned Affine Alignment (Calibration-based)
# ----------------------------------------------

def fit_learned_affine(target_net, donor_net, X_calib, T=3):
    """Fit a donor->target logit linear regression on a calibration set.

    Run both models with self-feedback to collect (donor_output, target_output) pairs,
    then fit target ~ donor @ W + b by least squares.

    Args:
        target_net: target model
        donor_net: donor model
        X_calib: (n, input_size) calibration inputs (e.g., training data)
        T: timesteps

    Returns:
        W: (output_size, output_size) alignment weight matrix
        b: (output_size,) alignment bias vector
    """
    donor_all = []
    target_all = []

    for i in range(len(X_calib)):
        target_net.reset_state()
        donor_net.reset_state()

        for t in range(T):
            target_y = target_net.forward(X_calib[i])
            donor_y = donor_net.forward(X_calib[i])
            donor_all.append(donor_y.copy())
            target_all.append(target_y.copy())

    D = np.array(donor_all)      # (m, output_size)
    T_ref = np.array(target_all)  # (m, output_size)

    # Augment with bias column
    D_aug = np.hstack([D, np.ones((len(D), 1))])  # (m, output_size+1)
    params, _, _, _ = np.linalg.lstsq(D_aug, T_ref, rcond=None)
    # params: (output_size+1, output_size)
    out_dim = D.shape[1]
    W = params[:out_dim]  # (output_size, output_size)
    b = params[out_dim]   # (output_size,)

    return W, b


def fit_learned_affine_vn(target_net, donor_net, X_seq_calib, T=3):
    """VN calibration: fit donor->target logit linear regression.

    Args:
        target_net: target model
        donor_net: donor model
        X_seq_calib: (n, T, input_size) per-timestep calibration inputs
        T: timesteps

    Returns:
        W: (output_size, output_size) alignment weight matrix
        b: (output_size,) alignment bias vector
    """
    donor_all = []
    target_all = []

    for i in range(len(X_seq_calib)):
        target_net.reset_state()
        donor_net.reset_state()

        for t in range(T):
            target_y = target_net.forward(X_seq_calib[i, t])
            donor_y = donor_net.forward(X_seq_calib[i, t])
            donor_all.append(donor_y.copy())
            target_all.append(target_y.copy())

    D = np.array(donor_all)
    T_ref = np.array(target_all)

    D_aug = np.hstack([D, np.ones((len(D), 1))])
    params, _, _, _ = np.linalg.lstsq(D_aug, T_ref, rcond=None)
    out_dim = D.shape[1]
    W = params[:out_dim]
    b = params[out_dim]

    return W, b


def forward_sequence_with_learned_affine_clone(target_net, donor_net, x,
                                                W_align, b_align, T=3):
    """Learned affine aligned clone feedback (static input).

    Run donor with self-feedback as usual; transform donor outputs by
    donor_output @ W_align + b_align before injecting into target.

    Args:
        target_net: target model
        donor_net: donor model
        x: input vector
        W_align: (output_size, output_size) learned alignment weights
        b_align: (output_size,) learned alignment bias
        T: timesteps

    Returns:
        target_outputs, target_caches
    """
    x = np.asarray(x, dtype=np.float64)
    target_net.reset_state()
    donor_net.reset_state()

    target_outputs = []
    target_caches = []
    donor_outputs = []

    for t in range(T):
        donor_y = donor_net.forward(x)
        donor_outputs.append(donor_y.copy())

        if t > 0:
            aligned = donor_outputs[t - 1] @ W_align + b_align
            target_net._prev_output = aligned.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x)
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


def forward_sequence_with_learned_affine_clone_vn(target_net, donor_net, x_seq,
                                                   W_align, b_align, T=3):
    """Learned affine aligned clone feedback (variable noise).

    Args:
        target_net: target model
        donor_net: donor model
        x_seq: (T, input_size) per-timestep inputs
        W_align, b_align: learned alignment parameters
        T: timesteps

    Returns:
        target_outputs, target_caches
    """
    x_seq = np.asarray(x_seq, dtype=np.float64)
    target_net.reset_state()
    donor_net.reset_state()

    target_outputs = []
    target_caches = []
    donor_outputs = []

    for t in range(T):
        donor_y = donor_net.forward(x_seq[t])
        donor_outputs.append(donor_y.copy())

        if t > 0:
            aligned = donor_outputs[t - 1] @ W_align + b_align
            target_net._prev_output = aligned.copy()
            target_net._has_feedback = True

        target_y = target_net.forward(x_seq[t])
        target_outputs.append(target_y.copy())
        target_caches.append(target_net._cache.copy())

    return target_outputs, target_caches


# ----------------------------------------------
# Utilities
# ----------------------------------------------

def count_zeroed_weights(net):
    """Count zero-valued weights across the entire model."""
    total = 0
    for w in net.get_all_weights().values():
        total += np.sum(w == 0)
    return int(total)


def deep_copy_weights(net):
    """Return a deep copy of all weights."""
    return {k: v.copy() for k, v in net.get_all_weights().items()}


def restore_weights(net, saved):
    """Restore saved weights into the network."""
    weights = net.get_all_weights()
    for k in saved:
        weights[k][:] = saved[k]
