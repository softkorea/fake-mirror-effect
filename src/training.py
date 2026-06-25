"""Training module - data generation, 3-step unroll backprop, training loop.

Constraints:
- forward_sequence calls reset_state() per sample.
- 3-step unroll backprop. Numerical differentiation is for gradient-check only (rel error < 1e-4).
- tanh bounded feedback -> backward path includes the tanh derivative.
- Inter-class ambiguity baked into the data.
- W_skip (Group D') gradients handled.
"""

import numpy as np
from src.network import RecurrentMLP


# ----------------------------------------------
# Utilities: softmax, cross-entropy
# ----------------------------------------------

def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cross_entropy_loss(output, target):
    """Cross-entropy loss (with softmax)."""
    p = softmax(output)
    return -np.sum(target * np.log(p + 1e-12))


def d_cross_entropy(output, target):
    """dCE/doutput = softmax(output) - target."""
    return softmax(output) - target


# ----------------------------------------------
# Data generation
# ----------------------------------------------

def generate_data(n_samples, noise_level, n_classes=5, input_size=10, seed=0):
    """Generate static pattern-classification data.

    Five base prototypes: each has two adjacent dimensions strongly activated
    (amplitude 1.0) and adjacent-class dimensions partially activated (0.3) -
    encoding inter-class ambiguity.

    Args:
        n_samples: number of samples
        noise_level: Gaussian-noise standard deviation
        n_classes: number of classes (default 5, matches output_size)
        input_size: input dimension (default 10)
        seed: RNG seed

    Returns:
        X: (n_samples, input_size)
        y: (n_samples, n_classes) one-hot
    """
    assert input_size >= 2 * n_classes, "input_size must be at least 2 * n_classes"
    rng = np.random.RandomState(seed)

    base_patterns = np.zeros((n_classes, input_size))
    for k in range(n_classes):
        base_patterns[k, 2 * k: 2 * k + 2] = 1.0
        # inter-class ambiguity: weak signal in adjacent-class dimensions
        base_patterns[k, (2 * k + 2) % input_size] = 0.3
        base_patterns[k, (2 * k - 1) % input_size] = 0.3

    X = np.zeros((n_samples, input_size))
    y = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        cls = rng.randint(n_classes)
        X[i] = base_patterns[cls] + noise_level * rng.randn(input_size)
        y[i, cls] = 1.0

    return X, y


def generate_data_variable_noise(n_samples, noise_level, T=3, n_classes=5,
                                  input_size=10, seed=0):
    """Generate variable-noise data: independent per-timestep noise.

    x_t = prototype_k + epsilon_t, where epsilon_t ~ N(0, sigma^2I), epsilon_1 perp epsilon_2 perp epsilon_3.
    Resolves the static-input tautology while preserving memory-correction separability.

    Args:
        n_samples: number of samples
        noise_level: Gaussian-noise standard deviation
        T: number of timesteps
        n_classes: number of classes
        input_size: input dimension
        seed: RNG seed

    Returns:
        X_seq: (n_samples, T, input_size)
        y: (n_samples, n_classes) one-hot
    """
    assert input_size >= 2 * n_classes, "input_size must be at least 2 * n_classes"
    rng = np.random.RandomState(seed)

    base_patterns = np.zeros((n_classes, input_size))
    for k in range(n_classes):
        base_patterns[k, 2 * k: 2 * k + 2] = 1.0
        base_patterns[k, (2 * k + 2) % input_size] = 0.3
        base_patterns[k, (2 * k - 1) % input_size] = 0.3

    X_seq = np.zeros((n_samples, T, input_size))
    y = np.zeros((n_samples, n_classes))

    for i in range(n_samples):
        cls = rng.randint(n_classes)
        y[i, cls] = 1.0
        for t in range(T):
            X_seq[i, t] = base_patterns[cls] + noise_level * rng.randn(input_size)

    return X_seq, y


# ----------------------------------------------
# 3-step unroll backprop
# ----------------------------------------------

def compute_loss_and_gradients(net, x, target, T=3, time_weights=None):
    """Per-sample loss + analytical gradients (3-step unroll).

    Time-weighted loss: t=1 carries no penalty, weight concentrates on t=3.
    Pushes the network to learn self-correction through feedback.
    feedback = tanh(prev_output / tau).
    """
    if time_weights is None:
        time_weights = [0.0, 0.2, 1.0]
    sw = float(sum(time_weights))  # normalize to a weighted AVERAGE
    if sw <= 0:
        raise ValueError(
            f"time_weights must sum to a positive value; got {time_weights} (sum={sw})")
    if len(time_weights) != T:
        raise ValueError(
            f"time_weights has {len(time_weights)} entries but T={T}; "
            "pass explicit weights when training at a non-default horizon")
    # The backward pass does not invert the C1 permutation; training with
    # scrambled feedback enabled would produce wrong gradients (C1 is an
    # evaluation-only condition).
    if getattr(net, "_scrambled_feedback", False):
        raise RuntimeError("compute_loss_and_gradients: scrambled feedback (C1) "
                           "must not be enabled during training")

    outputs, caches = net.forward_sequence(x, T=T)

    # Total loss (weighted average over timesteps)
    total_loss = 0.0
    for t in range(T):
        total_loss += time_weights[t] * cross_entropy_loss(outputs[t], target)
    total_loss /= sw

    # Initialize gradients
    grads = {
        'W_ih1': np.zeros_like(net.W_ih1),
        'b_h1': np.zeros_like(net.b_h1),
        'W_h1h2': np.zeros_like(net.W_h1h2),
        'b_h2': np.zeros_like(net.b_h2),
        'W_h2o': np.zeros_like(net.W_h2o),
        'b_out': np.zeros_like(net.b_out),
        'W_rec': np.zeros_like(net.W_rec),
    }
    if net.W_skip is not None:
        grads['W_skip'] = np.zeros_like(net.W_skip)

    # backward: iterate from the last timestep to the first
    d_output_future = np.zeros(net.output_size)

    for t in reversed(range(T)):
        cache = caches[t]

        # loss gradient at this timestep (time-weighted average)
        d_out = time_weights[t] / sw * d_cross_entropy(outputs[t], target)

        # + gradient from future timestep via recurrent
        d_out = d_out + d_output_future

        # -- skip connection (Group D') --
        if net.W_skip is not None:
            grads['W_skip'] += np.outer(cache['x'], d_out)

        # -- output layer (linear) --
        grads['W_h2o'] += np.outer(cache['a_h2'], d_out)
        grads['b_out'] += d_out

        d_a_h2 = net.W_h2o @ d_out
        d_z_h2 = d_a_h2 * (cache['z_h2'] > 0).astype(np.float64)

        # -- hidden 2 --
        grads['W_h1h2'] += np.outer(cache['a_h1'], d_z_h2)
        grads['b_h2'] += d_z_h2

        d_a_h1 = net.W_h1h2 @ d_z_h2
        d_z_h1 = d_a_h1 * (cache['z_h1'] > 0).astype(np.float64)

        # -- hidden 1 --
        grads['W_ih1'] += np.outer(cache['x'], d_z_h1)
        grads['b_h1'] += d_z_h1
        grads['W_rec'] += np.outer(cache['feedback'], d_z_h1)

        # propagate gradient to previous timestep's output
        # feedback = tanh(prev_output / tau) -> d/d(prev) = (1 - feedback^2) / tau
        # When the recurrent loop is disabled, block future->past gradient flow
        if net._recurrent_enabled:
            d_feedback = net.W_rec @ d_z_h1
            d_output_future = d_feedback * (1.0 - cache['feedback'] ** 2) / net.feedback_tau
        else:
            d_output_future = np.zeros(net.output_size)

    return total_loss, grads


def compute_batch_loss_and_gradients(net, X, Y, T=3, time_weights=None):
    """Batch-averaged loss + gradients."""
    n = len(X)
    total_loss = 0.0
    batch_grads = None

    for i in range(n):
        loss, grads = compute_loss_and_gradients(net, X[i], Y[i], T, time_weights)
        total_loss += loss
        if batch_grads is None:
            batch_grads = {k: v.copy() for k, v in grads.items()}
        else:
            for k in grads:
                batch_grads[k] += grads[k]

    total_loss /= n
    for k in batch_grads:
        batch_grads[k] /= n

    return total_loss, batch_grads


# ----------------------------------------------
# Gradient check (numerical differentiation)
# ----------------------------------------------

def numerical_gradient(net, x, target, T=3, epsilon=1e-5):
    """Compute numerical gradients via central differences."""
    num_grads = {}
    time_weights = [0.0, 0.2, 1.0]
    sw = float(sum(time_weights))  # weighted-average loss

    params = [
        ('W_ih1', net.W_ih1),
        ('b_h1', net.b_h1),
        ('W_h1h2', net.W_h1h2),
        ('b_h2', net.b_h2),
        ('W_h2o', net.W_h2o),
        ('b_out', net.b_out),
        ('W_rec', net.W_rec),
    ]
    if net.W_skip is not None:
        params.append(('W_skip', net.W_skip))

    for name, param in params:
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + epsilon
            outputs_plus, _ = net.forward_sequence(x, T)
            loss_plus = sum(
                time_weights[t] * cross_entropy_loss(outputs_plus[t], target)
                for t in range(T)
            )

            param[idx] = old_val - epsilon
            outputs_minus, _ = net.forward_sequence(x, T)
            loss_minus = sum(
                time_weights[t] * cross_entropy_loss(outputs_minus[t], target)
                for t in range(T)
            )

            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon) / sw
            param[idx] = old_val

            it.iternext()

        num_grads[name] = grad

    return num_grads


def gradient_check(net, x, target, T=3, epsilon=1e-5):
    """Compare analytical vs numerical gradients. Returns max relative error."""
    _, ana_grads = compute_loss_and_gradients(net, x, target, T)
    num_grads = numerical_gradient(net, x, target, T, epsilon)

    max_rel_error = 0.0
    for k in ana_grads:
        a = ana_grads[k].ravel()
        n = num_grads[k].ravel()
        for i in range(len(a)):
            denom = max(abs(a[i]), abs(n[i]), 1e-8)
            rel_error = abs(a[i] - n[i]) / denom
            max_rel_error = max(max_rel_error, rel_error)

    return max_rel_error


# ----------------------------------------------
# Training loop
# ----------------------------------------------

def train(net, X, y, epochs=100, lr=0.01, T=3, verbose=False, time_weights=None):
    """Train with full-batch SGD.

    Returns:
        loss_history: list of per-epoch losses
    """
    loss_history = []

    for epoch in range(epochs):
        loss, grads = compute_batch_loss_and_gradients(net, X, y, T, time_weights)
        loss_history.append(loss)

        # SGD update
        net.W_ih1  -= lr * grads['W_ih1']
        net.b_h1   -= lr * grads['b_h1']
        net.W_h1h2 -= lr * grads['W_h1h2']
        net.b_h2   -= lr * grads['b_h2']
        net.W_h2o  -= lr * grads['W_h2o']
        net.b_out  -= lr * grads['b_out']
        net.W_rec  -= lr * grads['W_rec']
        if 'W_skip' in grads:
            net.W_skip -= lr * grads['W_skip']

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss:.4f}")

    return loss_history


# ----------------------------------------------
# Evaluation
# ----------------------------------------------

def evaluate_accuracy_at_timestep(net, X, y, t):
    """Classification accuracy at a specific timestep.

    Args:
        t: timestep (1-indexed). t=1 = initial prediction, t=3 = final.
    """
    correct = 0
    for i in range(len(X)):
        outputs, _ = net.forward_sequence(X[i], T=3)
        pred = np.argmax(outputs[t - 1])
        true = np.argmax(y[i])
        if pred == true:
            correct += 1
    return correct / len(X)


# ----------------------------------------------
# DeepFeedforwardMLP (Group D'') - training support
# ----------------------------------------------

def compute_loss_and_gradients_deep_ff(net, x, target):
    """Single-sample loss + gradients for DeepFeedforwardMLP.

    Standard backprop through 6 hidden layers. No BPTT, no time weighting.
    Cross-entropy loss with softmax.

    Args:
        net: DeepFeedforwardMLP instance
        x: input vector (input_size,)
        target: one-hot target vector (output_size,)

    Returns:
        (loss, grad_list) where grad_list is a list of gradient arrays
        in the same order as net.get_all_params():
        [dW_h1, db_h1, dW_h2, db_h2, ..., dW_h6, db_h6, dW_out, db_out]
    """
    output = net.forward(x)
    loss = cross_entropy_loss(output, target)
    cache = net._cache

    # -- Backward pass --

    # Output layer gradient: dL/dz_out = softmax(output) - target
    d_out = d_cross_entropy(output, target)

    # Output layer parameter gradients
    a_last = cache['a_list'][-1]  # last hidden activation
    dW_out = np.outer(a_last, d_out)
    db_out = d_out.copy()

    # Propagate to last hidden layer
    d_a = net.W_out @ d_out

    # Hidden layers (reverse order)
    grad_list_hidden = []
    for i in reversed(range(net.n_hidden)):
        z = cache['z_list'][i]
        a_prev = cache['a_list'][i]  # activation feeding into this layer

        # ReLU derivative
        d_z = d_a * (z > 0).astype(np.float64)

        W, b = net.hidden_layers[i]
        dW = np.outer(a_prev, d_z)
        db = d_z.copy()
        grad_list_hidden.append((dW, db))

        # Propagate to previous layer
        d_a = W @ d_z

    # Reverse to get correct order (layer 0 first)
    grad_list_hidden.reverse()

    # Assemble into flat list matching get_all_params() order
    grad_list = []
    for dW, db in grad_list_hidden:
        grad_list.append(dW)
        grad_list.append(db)
    grad_list.append(dW_out)
    grad_list.append(db_out)

    return loss, grad_list


def compute_batch_loss_and_gradients_deep_ff(net, X, Y):
    """Batch average loss + gradients for DeepFeedforwardMLP.

    Args:
        net: DeepFeedforwardMLP instance
        X: (n_samples, input_size)
        Y: (n_samples, output_size) one-hot

    Returns:
        (mean_loss, mean_grad_list)
    """
    n = len(X)
    total_loss = 0.0
    batch_grads = None

    for i in range(n):
        loss, grads = compute_loss_and_gradients_deep_ff(net, X[i], Y[i])
        total_loss += loss
        if batch_grads is None:
            batch_grads = [g.copy() for g in grads]
        else:
            for j in range(len(grads)):
                batch_grads[j] += grads[j]

    total_loss /= n
    for j in range(len(batch_grads)):
        batch_grads[j] /= n

    return total_loss, batch_grads


def train_deep_ff(net, X, y, epochs=1000, lr=0.01, verbose=False):
    """Training loop for DeepFeedforwardMLP.

    Full-batch SGD.

    Args:
        net: DeepFeedforwardMLP instance
        X: training inputs
        y: training targets (one-hot)
        epochs: number of training epochs
        lr: learning rate
        verbose: print loss every 50 epochs

    Returns:
        loss_history: list of per-epoch losses
    """
    loss_history = []

    for epoch in range(epochs):
        loss, grads = compute_batch_loss_and_gradients_deep_ff(net, X, y)
        loss_history.append(loss)

        # SGD update
        params = net.get_all_params()
        for p, g in zip(params, grads):
            p -= lr * g

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss:.4f}")

    return loss_history


def evaluate_accuracy_deep_ff(net, X, y):
    """Single-pass classification accuracy for DeepFeedforwardMLP.

    Args:
        net: DeepFeedforwardMLP instance
        X: (n_samples, input_size)
        y: (n_samples, output_size) one-hot

    Returns:
        accuracy (float)
    """
    correct = 0
    for i in range(len(X)):
        output = net.forward(X[i])
        pred = np.argmax(output)
        true = np.argmax(y[i])
        if pred == true:
            correct += 1
    return correct / len(X)


# ----------------------------------------------
# Variable-Noise (VN) - 3-step unroll backprop
# ----------------------------------------------

def compute_loss_and_gradients_vn(net, x_seq, target, T=3, time_weights=None):
    """VN per-sample loss + gradients. Different input at each timestep.

    Backward pass is identical to the static version - cache['x'] already stores
    the per-timestep input.

    Args:
        net: RecurrentMLP
        x_seq: (T, input_size) per-timestep inputs
        target: one-hot target
        T: number of timesteps
        time_weights: per-timestep weights (default [0.0, 0.2, 1.0])
    """
    if time_weights is None:
        time_weights = [0.0, 0.2, 1.0]
    sw = float(sum(time_weights))  # normalize to a weighted AVERAGE
    if sw <= 0:
        raise ValueError(
            f"time_weights must sum to a positive value; got {time_weights} (sum={sw})")
    if len(time_weights) != T:
        raise ValueError(
            f"time_weights has {len(time_weights)} entries but T={T}; "
            "pass explicit weights when training at a non-default horizon")
    # See compute_loss_and_gradients: C1 scrambling is evaluation-only.
    if getattr(net, "_scrambled_feedback", False):
        raise RuntimeError("compute_loss_and_gradients_vn: scrambled feedback (C1) "
                           "must not be enabled during training")

    outputs, caches = net.forward_sequence_vn(x_seq, T=T)

    total_loss = 0.0
    for t in range(T):
        total_loss += time_weights[t] * cross_entropy_loss(outputs[t], target)
    total_loss /= sw

    grads = {
        'W_ih1': np.zeros_like(net.W_ih1),
        'b_h1': np.zeros_like(net.b_h1),
        'W_h1h2': np.zeros_like(net.W_h1h2),
        'b_h2': np.zeros_like(net.b_h2),
        'W_h2o': np.zeros_like(net.W_h2o),
        'b_out': np.zeros_like(net.b_out),
        'W_rec': np.zeros_like(net.W_rec),
    }
    if net.W_skip is not None:
        grads['W_skip'] = np.zeros_like(net.W_skip)

    d_output_future = np.zeros(net.output_size)

    for t in reversed(range(T)):
        cache = caches[t]
        d_out = time_weights[t] / sw * d_cross_entropy(outputs[t], target)
        d_out = d_out + d_output_future

        if net.W_skip is not None:
            grads['W_skip'] += np.outer(cache['x'], d_out)

        grads['W_h2o'] += np.outer(cache['a_h2'], d_out)
        grads['b_out'] += d_out

        d_a_h2 = net.W_h2o @ d_out
        d_z_h2 = d_a_h2 * (cache['z_h2'] > 0).astype(np.float64)

        grads['W_h1h2'] += np.outer(cache['a_h1'], d_z_h2)
        grads['b_h2'] += d_z_h2

        d_a_h1 = net.W_h1h2 @ d_z_h2
        d_z_h1 = d_a_h1 * (cache['z_h1'] > 0).astype(np.float64)

        grads['W_ih1'] += np.outer(cache['x'], d_z_h1)
        grads['b_h1'] += d_z_h1
        grads['W_rec'] += np.outer(cache['feedback'], d_z_h1)

        if net._recurrent_enabled:
            d_feedback = net.W_rec @ d_z_h1
            d_output_future = d_feedback * (1.0 - cache['feedback'] ** 2) / net.feedback_tau
        else:
            d_output_future = np.zeros(net.output_size)

    return total_loss, grads


def compute_batch_loss_and_gradients_vn(net, X_seq, Y, T=3, time_weights=None):
    """VN batch-averaged loss + gradients.

    Args:
        X_seq: (n_samples, T, input_size)
        Y: (n_samples, n_classes) one-hot
    """
    n = len(X_seq)
    total_loss = 0.0
    batch_grads = None

    for i in range(n):
        loss, grads = compute_loss_and_gradients_vn(
            net, X_seq[i], Y[i], T, time_weights)
        total_loss += loss
        if batch_grads is None:
            batch_grads = {k: v.copy() for k, v in grads.items()}
        else:
            for k in grads:
                batch_grads[k] += grads[k]

    total_loss /= n
    for k in batch_grads:
        batch_grads[k] /= n

    return total_loss, batch_grads


def numerical_gradient_vn(net, x_seq, target, T=3, epsilon=1e-5):
    """VN numerical gradient (central differences)."""
    num_grads = {}
    time_weights = [0.0, 0.2, 1.0]
    sw = float(sum(time_weights))  # weighted-average loss

    params = [
        ('W_ih1', net.W_ih1),
        ('b_h1', net.b_h1),
        ('W_h1h2', net.W_h1h2),
        ('b_h2', net.b_h2),
        ('W_h2o', net.W_h2o),
        ('b_out', net.b_out),
        ('W_rec', net.W_rec),
    ]
    if net.W_skip is not None:
        params.append(('W_skip', net.W_skip))

    for name, param in params:
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]

            param[idx] = old_val + epsilon
            outputs_plus, _ = net.forward_sequence_vn(x_seq, T)
            loss_plus = sum(
                time_weights[t] * cross_entropy_loss(outputs_plus[t], target)
                for t in range(T)
            )

            param[idx] = old_val - epsilon
            outputs_minus, _ = net.forward_sequence_vn(x_seq, T)
            loss_minus = sum(
                time_weights[t] * cross_entropy_loss(outputs_minus[t], target)
                for t in range(T)
            )

            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon) / sw
            param[idx] = old_val

            it.iternext()

        num_grads[name] = grad

    return num_grads


def gradient_check_vn(net, x_seq, target, T=3, epsilon=1e-5):
    """Compare VN analytical vs numerical gradients. Returns max relative error."""
    _, ana_grads = compute_loss_and_gradients_vn(net, x_seq, target, T)
    num_grads = numerical_gradient_vn(net, x_seq, target, T, epsilon)

    max_rel_error = 0.0
    for k in ana_grads:
        a = ana_grads[k].ravel()
        n = num_grads[k].ravel()
        for i in range(len(a)):
            denom = max(abs(a[i]), abs(n[i]), 1e-8)
            rel_error = abs(a[i] - n[i]) / denom
            max_rel_error = max(max_rel_error, rel_error)

    return max_rel_error


def train_vn(net, X_seq, y, epochs=100, lr=0.01, T=3, verbose=False,
             time_weights=None):
    """VN full-batch SGD training.

    Args:
        X_seq: (n_samples, T, input_size)
        y: (n_samples, n_classes) one-hot

    Returns:
        loss_history
    """
    loss_history = []

    for epoch in range(epochs):
        loss, grads = compute_batch_loss_and_gradients_vn(
            net, X_seq, y, T, time_weights)
        loss_history.append(loss)

        net.W_ih1  -= lr * grads['W_ih1']
        net.b_h1   -= lr * grads['b_h1']
        net.W_h1h2 -= lr * grads['W_h1h2']
        net.b_h2   -= lr * grads['b_h2']
        net.W_h2o  -= lr * grads['W_h2o']
        net.b_out  -= lr * grads['b_out']
        net.W_rec  -= lr * grads['W_rec']
        if 'W_skip' in grads:
            net.W_skip -= lr * grads['W_skip']

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}: loss = {loss:.4f}")

    return loss_history
