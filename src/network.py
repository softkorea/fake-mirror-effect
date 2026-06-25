"""RecurrentMLP - recurrent network for self-correction experiments.

Architecture:
    Input(10) -> Hidden1(10) -> Hidden2(10) -> Output(5)
                  ^                              |
                  +---- recurrent feedback -------+

35 neurons total - small enough for humans to inspect every connection.
Activations: ReLU (hidden), linear (output).
Feedback: tanh(prev_output / feedback_tau) - bounded to [-1, 1].
"""

import numpy as np


class RecurrentMLP:
    def __init__(self, input_size=10, hidden1=10, hidden2=10, output_size=5,
                 seed=0, skip_connection=False, feedback_tau=2.0):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.feedback_tau = feedback_tau

        rng = np.random.RandomState(seed)

        # He initialization (suitable for ReLU)
        self.W_ih1 = rng.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b_h1 = np.zeros(hidden1)

        self.W_h1h2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b_h2 = np.zeros(hidden2)

        self.W_h2o = rng.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
        self.b_out = np.zeros(output_size)

        # Recurrent: output(5) -> hidden1(10) feedback
        self.W_rec = rng.randn(output_size, hidden1) * np.sqrt(2.0 / output_size)

        # Skip connection: input(10) -> output(5) - Group D' param-matched FF
        if skip_connection:
            self.W_skip = rng.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.W_skip = None

        # Internal state
        self._prev_output = np.zeros(output_size)
        self._has_feedback = False  # False at t=1
        self._recurrent_enabled = True
        self._scrambled_feedback = False
        self._scramble_rng = None

        # Cache of forward-pass intermediates (used by backprop)
        self._cache = {}

    # -- forward ----------------------------------

    def forward(self, x):
        """Single-timestep forward pass.

        Args:
            x: input vector (input_size,)

        Returns:
            output vector (output_size,)
        """
        x = np.asarray(x, dtype=np.float64)

        # recurrent feedback (tanh bounded, temperature scaling)
        # Avoid tanh saturation: logit +/-5 -> tanh(5)~1.0 (gradient~0) vs tanh(2.5)~0.987 (gradient~0.026)
        if self._recurrent_enabled:
            feedback = np.tanh(self._prev_output / self.feedback_tau)
            if self._scrambled_feedback and self._has_feedback:
                # Permutation preserves the distribution (mean/variance) but destroys positional information
                feedback = feedback.copy()
                self._scramble_rng.shuffle(feedback)
            rec_contrib = feedback @ self.W_rec
        else:
            feedback = np.zeros(self.output_size)
            rec_contrib = np.zeros(self.hidden1)

        # Hidden 1: input + recurrent contribution
        z_h1 = x @ self.W_ih1 + rec_contrib + self.b_h1
        a_h1 = np.maximum(0, z_h1)  # ReLU

        # Hidden 2
        z_h2 = a_h1 @ self.W_h1h2 + self.b_h2
        a_h2 = np.maximum(0, z_h2)  # ReLU

        # Output (linear)
        z_out = a_h2 @ self.W_h2o + self.b_out
        if self.W_skip is not None:
            z_out = z_out + x @ self.W_skip
        output = z_out

        # Cache intermediates (used by backprop)
        self._cache = {
            'x': x,
            'feedback': feedback,
            'rec_contrib': rec_contrib,
            'z_h1': z_h1,
            'a_h1': a_h1,
            'z_h2': z_h2,
            'a_h2': a_h2,
            'z_out': z_out,
            'output': output,
        }

        # Update internal state
        self._prev_output = output.copy()
        self._has_feedback = True

        return output

    def forward_sequence(self, x, T=3):
        """T-timestep unroll, with per-sample state reset.

        Args:
            x: static input vector (input_size,) - identical at every timestep
            T: number of timesteps (default 3)

        Returns:
            list of T output vectors, list of T cache dicts
        """
        self.reset_state()
        outputs = []
        caches = []
        for _ in range(T):
            y = self.forward(x)
            outputs.append(y.copy())
            caches.append(self._cache.copy())
        return outputs, caches

    def forward_sequence_vn(self, x_seq, T=3):
        """Variable-noise T-step unroll. A different input is fed at each timestep.

        Resolves the static-input tautology: x_t = prototype + epsilon_t (independent noise).

        Args:
            x_seq: (T, input_size) array - per-timestep inputs
            T: number of timesteps (default 3)

        Returns:
            list of T output vectors, list of T cache dicts
        """
        x_seq = np.asarray(x_seq, dtype=np.float64)
        self.reset_state()
        outputs = []
        caches = []
        for t in range(T):
            y = self.forward(x_seq[t])
            outputs.append(y.copy())
            caches.append(self._cache.copy())
        return outputs, caches

    # -- State management ------------------------

    def reset_state(self):
        """Reset internal state (previous output)."""
        self._prev_output = np.zeros(self.output_size)
        self._has_feedback = False
        self._cache = {}

    def disable_recurrent_loop(self):
        """Disable the recurrent loop."""
        self._recurrent_enabled = False

    def enable_recurrent_loop(self):
        """Enable the recurrent loop."""
        self._recurrent_enabled = True

    def enable_scrambled_feedback(self, seed=42):
        """Scrambled-feedback mode: keep the recurrent connections, permute the feedback vector."""
        self._scrambled_feedback = True
        self._scramble_rng = np.random.RandomState(seed)

    def disable_scrambled_feedback(self):
        """Disable scrambled-feedback mode."""
        self._scrambled_feedback = False
        self._scramble_rng = None

    # -- Weight accessors ------------------------

    def get_all_weights(self):
        """Return all weight matrices as a dict (views, mutable)."""
        w = {
            'input_to_h1': self.W_ih1,
            'h1_to_h2': self.W_h1h2,
            'h2_to_output': self.W_h2o,
            'recurrent': self.W_rec,
        }
        if self.W_skip is not None:
            w['skip'] = self.W_skip
        return w

    def get_all_biases(self):
        """Return all bias vectors as a dict (views, mutable)."""
        return {
            'b_h1': self.b_h1,
            'b_h2': self.b_h2,
            'b_out': self.b_out,
        }

    def get_all_params(self):
        """Return all parameters (weights + biases) as a flat list."""
        params = [
            self.W_ih1, self.b_h1,
            self.W_h1h2, self.b_h2,
            self.W_h2o, self.b_out,
            self.W_rec,
        ]
        if self.W_skip is not None:
            params.append(self.W_skip)
        return params

    def count_params(self):
        """Total parameter count."""
        return sum(p.size for p in self.get_all_params())


class DeepFeedforwardMLP:
    """Compute-matched feedforward control (Group D'').

    6 hidden layers of 10 neurons each, matching the recurrent model's
    3-timestep x 2-hidden-layer = 6 layer traversals.

    Architecture: Input(10) -> H1(10) -> H2(10) -> H3(10) -> H4(10) -> H5(10) -> H6(10) -> Output(5)

    Activations: ReLU (hidden), Linear (output).
    Initialization: He (sqrt(2/fan_in)).
    """

    def __init__(self, input_size=10, hidden_size=10, n_hidden=6,
                 output_size=5, seed=0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.output_size = output_size

        rng = np.random.RandomState(seed)

        # Hidden layers: list of (W, b) pairs
        self.hidden_layers = []
        for i in range(n_hidden):
            fan_in = input_size if i == 0 else hidden_size
            W = rng.randn(fan_in, hidden_size) * np.sqrt(2.0 / fan_in)
            b = np.zeros(hidden_size)
            self.hidden_layers.append((W, b))

        # Output layer
        self.W_out = rng.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros(output_size)

        # Forward cache (for backprop)
        self._cache = {}

    def forward(self, x):
        """Sequential forward pass through all layers.

        Args:
            x: input vector (input_size,)

        Returns:
            output vector (output_size,)
        """
        x = np.asarray(x, dtype=np.float64)

        # Cache for backprop
        z_list = []  # pre-activation
        a_list = [x]  # post-activation (a_list[0] = input)

        a = x
        for W, b in self.hidden_layers:
            z = a @ W + b
            a = np.maximum(0, z)  # ReLU
            z_list.append(z)
            a_list.append(a)

        # Output (linear)
        z_out = a @ self.W_out + self.b_out
        output = z_out

        self._cache = {
            'z_list': z_list,
            'a_list': a_list,
            'z_out': z_out,
            'output': output,
        }

        return output

    def get_all_params(self):
        """Return flat list of all parameter arrays (for gradient check etc.)."""
        params = []
        for W, b in self.hidden_layers:
            params.append(W)
            params.append(b)
        params.append(self.W_out)
        params.append(self.b_out)
        return params

    def count_params(self):
        """Total parameter count."""
        return sum(p.size for p in self.get_all_params())
