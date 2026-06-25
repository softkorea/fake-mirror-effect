"""Closed-loop BPTT alignment module.

Trainable aligners that produce pre-tanh aligned logits A(y_donor) that are
injected into a frozen target's recurrent feedback path. Gradients flow through
the frozen target's BPTT unroll back to the aligner.

Used by experiments/run_closed_loop_alignment.py (Appendix G; establishes a
task-supervised upper bound).

Design (the pre-specified analysis plan):
  - Aligner output is PRE-TANH logits. Forward path is:
      aligned_logits = A(y_donor_prev)
      feedback       = tanh(aligned_logits / tau)
      rec_contrib    = feedback @ W_rec^target  (frozen)
  - At t=0 the aligner output is masked to zero (F1 fix) to match Baseline/C2
    structural parity (no unearned bias injection at t=1 output).
  - Loss is the receiver's time-weighted CE: L = sum_t w_t * CE(y_t^target, k).

Aligner families (matching static-aligner parameter counts):
  - affine    : 5->5 linear,                     30 params
  - MLP-small : 5->16->5 ReLU,                   181 params
  - MLP-medium: 5->64->64->5 ReLU,             4,869 params
  - MLP-large : 5->128->128->5 ReLU,          17,925 params

Plus parity-test aligners (no parameters):
  - IdentityAligner: A(y)=y.   Closed-loop with this must match C2-raw exactly.
  - ZeroAligner    : A(y)=0.   Closed-loop with this must match no-feedback exactly.

And x_t-only control aligners (parameter-matched, 10-dim input):
  - XtOnlyAffine    : 10->5,    55 params  (descriptive only at affine level)
  - XtOnlyMLPSmall  : 10->13->5,~210 params
  - XtOnlyMLPMedium : 10->60->60->5, ~4,625 params
  - XtOnlyMLPLarge  : 10->120->120->5, ~16,445 params
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Aligner classes
# ----------------------------------------------------------------------

class ClosedLoopAligner(nn.Module):
    """Donor-fed aligner: input is y_donor (5-dim), output is aligned pre-tanh logits (5-dim).

    Args:
        hidden_sizes: list of hidden widths for MLP. Empty list = affine.
        input_dim:    aligner input dimensionality (5 for donor-fed; 10 for x_t-only).
        output_dim:   aligner output dimensionality (5 = target's output dim).
        activation:   'relu' for hidden activations; output is linear (pre-tanh logits).
    """

    def __init__(self, hidden_sizes: Sequence[int] = (), input_dim: int = 5,
                 output_dim: int = 5, activation: str = 'relu'):
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = activation
        # Track configuration for downstream reporting
        self.hidden_sizes = tuple(hidden_sizes)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, y_donor: torch.Tensor) -> torch.Tensor:
        """Returns pre-tanh aligned logits."""
        h = y_donor
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    h = F.relu(h)
                else:
                    raise ValueError(f"Unknown activation {self.activation}")
        return h  # linear output (pre-tanh)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IdentityAligner(nn.Module):
    """A(y) = y. Parameter-free. Used for F1 parity test (must match C2-raw)."""

    def __init__(self, output_dim: int = 5):
        super().__init__()
        self.output_dim = output_dim
        # Register a no-op buffer so the module has a device
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(self, y_donor: torch.Tensor) -> torch.Tensor:
        return y_donor

    def count_params(self) -> int:
        return 0


class ZeroAligner(nn.Module):
    """A(y) = 0. Parameter-free. Used for F1 parity test (must match no-feedback)."""

    def __init__(self, output_dim: int = 5):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(self, y_donor: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y_donor[..., :self.output_dim]) \
            if y_donor.shape[-1] != self.output_dim else torch.zeros_like(y_donor)

    def count_params(self) -> int:
        return 0


class BiasOnlyAligner(nn.Module):
    """A(y) = constant 5-dim learnable parameter. Ignores input entirely.

    Mechanism diagnostic: if this trivial aligner achieves task gain comparable
    to donor-fed or x_t-only aligners, the closed-loop "alignment" result
    reduces to a **constant bias injection through W_rec** - there is nothing
    input-dependent or representational about the closed-loop solution. W_rec
    is a 5-dim -> 10-dim linear projection; an aligner producing a saturated
    5-dim sign-vector encodes a constant per-step bias to h1 via W_rec.
    Gradient descent on task CE finds the single constant offset that
    minimizes the time-weighted loss across all trials.

    Params: 5 (a single 5-dim vector). This is the minimal possible aligner.
    """

    def __init__(self, output_dim: int = 5):
        super().__init__()
        self.output_dim = output_dim
        # Initialise at zero so untrained behaviour = no-feedback (Group A)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, y_input: torch.Tensor) -> torch.Tensor:
        """Returns the learned bias, broadcast to the input batch.

        Input is ignored entirely - only used to determine batch size and dtype.
        """
        batch_size = y_input.shape[0]
        # Broadcast bias [output_dim] to [batch_size, output_dim]
        return self.bias.unsqueeze(0).expand(batch_size, -1)

    def count_params(self) -> int:
        return self.output_dim


# ----------------------------------------------------------------------
# Architecture catalogues (the pre-specified analysis plan, M1)
# ----------------------------------------------------------------------

# Donor-fed (5-dim input) - matches existing static aligner families exactly
DONOR_FED_FAMILIES = {
    'affine':     {'hidden_sizes': (),        'input_dim': 5},
    'MLP-small':  {'hidden_sizes': (16,),     'input_dim': 5},
    'MLP-medium': {'hidden_sizes': (64, 64),  'input_dim': 5},
    'MLP-large':  {'hidden_sizes': (128, 128),'input_dim': 5},
}

# x_t-only control (10-dim input) - parameter-matched (M1)
#   affine: 10->5 = 55 params (vs donor 30; overshoot accepted - descriptive only)
#   small : 10->13->5 = (10*13+13) + (13*5+5) = 143+70 = 213 (vs donor 181; +18%)
#   medium: 10->60->60->5 = (10*60+60)+(60*60+60)+(60*5+5) = 660+3660+305 = 4625 (vs 4869; -5%)
#   large : 10->120->120->5 = (10*120+120)+(120*120+120)+(120*5+5) = 1320+14520+605 = 16445 (vs 17925; -8%)
XT_ONLY_FAMILIES = {
    'affine':     {'hidden_sizes': (),         'input_dim': 10},
    'MLP-small':  {'hidden_sizes': (13,),      'input_dim': 10},
    'MLP-medium': {'hidden_sizes': (60, 60),   'input_dim': 10},
    'MLP-large':  {'hidden_sizes': (120, 120), 'input_dim': 10},
}


def make_aligner(family: str, kind: str = 'donor_fed') -> ClosedLoopAligner:
    """Construct an aligner from {family, kind}.

    Args:
        family: 'affine' | 'MLP-small' | 'MLP-medium' | 'MLP-large'.
        kind:   'donor_fed' (5-dim) | 'xt_only' (10-dim).
    """
    if kind == 'donor_fed':
        cfg = DONOR_FED_FAMILIES[family]
    elif kind == 'xt_only':
        cfg = XT_ONLY_FAMILIES[family]
    else:
        raise ValueError(f"Unknown aligner kind: {kind}")
    return ClosedLoopAligner(hidden_sizes=cfg['hidden_sizes'],
                             input_dim=cfg['input_dim'],
                             output_dim=5,
                             activation='relu')


# ----------------------------------------------------------------------
# Closed-loop unroll (the pre-specified analysis plan - verbatim C2 forward + aligner substitution)
# ----------------------------------------------------------------------

def closed_loop_unroll(
    target: nn.Module,
    donor: nn.Module,
    aligner: nn.Module,
    x: torch.Tensor,
    T: int = 3,
    feedback_tau: float = 2.0,
    aligner_input_kind: str = 'donor_fed',
    diagnostics: bool = False,
):
    """Forward pass with closed-loop BPTT aligner injected into target's feedback path.

    Reproduces the PyTorch RecurrentMLP's 'clone' branch but replaces the donor-to-target
    feedback with `tanh(aligner(...)/tau)` where aligner is trainable. Target and donor
    weights must already have requires_grad=False set by the caller.

    Args:
        target:  frozen target RecurrentMLP.
        donor:   frozen donor RecurrentMLP.
        aligner: trainable aligner (donor_fed or xt_only).
        x:       input batch, shape [batch, T, input_dim] (VN regime).
        T:       number of timesteps (default 3).
        feedback_tau: tanh temperature (default 2.0).
        aligner_input_kind: 'donor_fed' (input is y_donor) | 'xt_only' (input is x_t).
        diagnostics: if True, return additional per-timestep tensors for off-manifold
                     drift monitoring (aligned_logits norms, etc.).

    Returns:
        outputs: list of length T of target output tensors [batch, output_dim].
        diag: optional dict if diagnostics=True (None otherwise).
    """
    batch = x.shape[0]
    device = next(target.parameters()).device
    output_dim = target.fc_out.out_features

    donor_prev = torch.zeros(batch, output_dim, device=device, dtype=x.dtype)
    outputs = []

    diag_records = {
        'aligned_logits_norm':       [],
        'aligned_logits_max_abs':    [],
        'feedback_norm':             [],
        'tanh_saturation_fraction':  [],  # fraction of |aligned_logits| > 4.0
    } if diagnostics else None

    for t in range(T):
        x_t = x[:, t, :]

        # ---- Target feedback path (F1 mask: t=0 -> zero feedback) ----
        if t == 0:
            # No feedback at t=0 (matches Baseline / C2 / Group A exactly).
            # Use a zero recurrent contribution; do NOT invoke aligner.
            h1 = F.relu(target.fc1(x_t))
        else:
            # t >= 1: aligner produces pre-tanh aligned logits from donor's t-1 output.
            if aligner_input_kind == 'donor_fed':
                aligner_input = donor_prev
            elif aligner_input_kind == 'xt_only':
                aligner_input = x_t  # 10-dim input
            else:
                raise ValueError(f"Unknown aligner_input_kind: {aligner_input_kind}")

            aligned_logits = aligner(aligner_input)
            feedback = torch.tanh(aligned_logits / feedback_tau)
            h1 = F.relu(target.fc1(x_t) + target.W_rec(feedback))

            if diagnostics:
                with torch.no_grad():
                    diag_records['aligned_logits_norm'].append(
                        aligned_logits.norm(dim=-1).mean().item())
                    diag_records['aligned_logits_max_abs'].append(
                        aligned_logits.abs().max().item())
                    diag_records['feedback_norm'].append(
                        feedback.norm(dim=-1).mean().item())
                    diag_records['tanh_saturation_fraction'].append(
                        (aligned_logits.abs() > 4.0).float().mean().item())

        # ---- Target rest of forward (frozen) ----
        h2 = F.relu(target.fc2(h1))
        y_t = target.fc_out(h2)
        outputs.append(y_t)

        # ---- Donor parallel forward (frozen, self-feedback). Detach to block donor grads. ----
        if t == 0:
            donor_h1 = F.relu(donor.fc1(x_t))
        else:
            donor_fb = torch.tanh(donor_prev / donor.feedback_tau)
            donor_h1 = F.relu(donor.fc1(x_t) + donor.W_rec(donor_fb))
        donor_h2 = F.relu(donor.fc2(donor_h1))
        donor_prev = donor.fc_out(donor_h2).detach()

    return (outputs, diag_records) if diagnostics else (outputs, None)


# ----------------------------------------------------------------------
# Helper: prepare frozen models
# ----------------------------------------------------------------------

def freeze_model(model: nn.Module) -> nn.Module:
    """Set all parameters' requires_grad=False and eval mode."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
