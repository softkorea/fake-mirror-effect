"""F1 parity tests for closed-loop BPTT alignment (the pre-specified analysis plan).

PRECONDITION for the main run: this script must pass for all 20 seeds before
any aligner training begins.

Two parity tests:

  Test A (C2 parity):
    Closed-loop unroll with IdentityAligner (A(y)=y) must match the PyTorch
    RecurrentMLP's feedback_mode='clone' branch EXACTLY (float64, atol=1e-9).

  Test B (no-feedback parity):
    Closed-loop unroll with ZeroAligner (A(y)=0) must match the PyTorch
    RecurrentMLP's feedback_mode='ablated' branch EXACTLY (float64, atol=1e-9).

If either fails on any seed: ABORT, fix the implementation, retry.

Per the pre-specified analysis plan refinement: extended from "seed 0 only" to "all 20 seeds"
since the algorithm is seed-independent and the extra cost is trivial.

Output: results/closed_loop_parity_tests.csv (per-seed pass/fail + max diffs)
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJ_ROOT)
PYTORCH_REPO = os.path.normpath(os.path.join(PROJ_ROOT, '..', 'pytorch-35neuron-validation'))
sys.path.insert(0, PYTORCH_REPO)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import SGD

# Force float64 globally so the external RecurrentMLP's internal `torch.zeros`
# calls (which use default dtype) match our float64 weights and data.
torch.set_default_dtype(torch.float64)

from network import RecurrentMLP                     # external PyTorch repo
from src.closed_loop_aligner import (
    IdentityAligner, ZeroAligner, closed_loop_unroll, freeze_model,
)

# ----------------------------------------------------------------------
# Defaults match the existing PyTorch cross-validation protocol
# ----------------------------------------------------------------------
N_SEEDS_DEFAULT = 20
N_TRAIN         = 200
N_TEST          = 200
T               = 3
NOISE           = 0.5
TRAIN_EPOCHS    = 1000
TRAIN_LR        = 0.01
FEEDBACK_TAU    = 2.0
TIME_WEIGHTS    = (0.0, 0.2, 1.0)
ATOL            = 1e-9      # float64 parity tolerance


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_data_vn(n_samples, noise_level=0.5, T=3, n_classes=5,
                     input_size=10, seed=0):
    """Variable-noise data -- mirrors src.training.generate_data_variable_noise."""
    assert input_size >= 2 * n_classes
    rng = np.random.RandomState(seed)
    base = np.zeros((n_classes, input_size))
    for k in range(n_classes):
        base[k, 2 * k: 2 * k + 2] = 1.0
        base[k, (2 * k + 2) % input_size] = 0.3
        base[k, (2 * k - 1) % input_size] = 0.3
    X = np.zeros((n_samples, T, input_size), dtype=np.float64)   # float64 for parity
    y = np.zeros((n_samples, n_classes), dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        cls = rng.randint(n_classes)
        for t in range(T):
            X[i, t] = base[cls] + noise_level * rng.randn(input_size)
        y[i, cls] = 1.0
        labels[i] = cls
    return X, y, labels


def train_vn_model(net, X, y, epochs=1000, lr=0.01, time_weights=TIME_WEIGHTS):
    """Time-weighted CE BPTT training -- mirrors run_pytorch_cross_validation."""
    optimizer = SGD(net.parameters(), lr=lr)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    labels = torch.argmax(y_t, dim=1)
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outs = net(X_t, T=T, feedback_mode='self')
        loss = sum(w * F.cross_entropy(outs[t], labels)
                   for t, w in enumerate(time_weights)) / sum(time_weights)
        loss.backward()
        optimizer.step()


def native_forward_clone(target, donor, x, T=3):
    """Run PyTorch RecurrentMLP's native 'clone' branch for C2 parity reference."""
    target.eval()
    donor.eval()
    with torch.no_grad():
        outs = target(x, T=T, feedback_mode='clone', clone=donor)
    return outs


def native_forward_ablated(target, x, T=3):
    """Run PyTorch RecurrentMLP's native 'ablated' branch for no-feedback parity reference."""
    target.eval()
    with torch.no_grad():
        outs = target(x, T=T, feedback_mode='ablated')
    return outs


def parity_test_seed(seed: int) -> dict:
    """Run F1 parity tests for one seed. Returns per-seed pass/fail + max diffs."""
    rec = {'seed': seed}

    # ---- Train target and donor (same protocol as paper primary) ----
    set_seed(seed)
    X_tr_t, y_tr_t, _ = generate_data_vn(N_TRAIN, NOISE, T=T, seed=seed)
    X_te,   y_te,   _ = generate_data_vn(N_TEST,  NOISE, T=T, seed=seed + 500)

    set_seed(seed)
    target = RecurrentMLP(feedback_tau=FEEDBACK_TAU)
    target = target.to(torch.float64)
    train_vn_model(target, X_tr_t, y_tr_t)
    target = freeze_model(target)

    donor_seed = seed + 100  # paper convention: target seeds 0-19, donor seeds 100-119
    set_seed(donor_seed)
    X_tr_d, y_tr_d, _ = generate_data_vn(N_TRAIN, NOISE, T=T, seed=donor_seed)
    set_seed(donor_seed)
    donor = RecurrentMLP(feedback_tau=FEEDBACK_TAU)
    donor = donor.to(torch.float64)
    train_vn_model(donor, X_tr_d, y_tr_d)
    donor = freeze_model(donor)

    X_te_t = torch.from_numpy(X_te)  # already float64

    # ---- Test A: Identity aligner vs native clone ----
    identity = IdentityAligner().to(torch.float64)
    with torch.no_grad():
        outs_identity, _ = closed_loop_unroll(
            target, donor, identity, X_te_t,
            T=T, feedback_tau=FEEDBACK_TAU, aligner_input_kind='donor_fed',
        )
    outs_native_clone = native_forward_clone(target, donor, X_te_t, T=T)

    max_diff_A = 0.0
    pass_A = True
    for t in range(T):
        diff = (outs_identity[t] - outs_native_clone[t]).abs().max().item()
        max_diff_A = max(max_diff_A, diff)
        if not torch.allclose(outs_identity[t], outs_native_clone[t], rtol=0.0, atol=ATOL):
            pass_A = False
    rec['test_A_pass'] = pass_A
    rec['test_A_max_diff'] = max_diff_A

    # ---- Test B: Zero aligner vs native ablated ----
    zero = ZeroAligner().to(torch.float64)
    with torch.no_grad():
        outs_zero, _ = closed_loop_unroll(
            target, donor, zero, X_te_t,
            T=T, feedback_tau=FEEDBACK_TAU, aligner_input_kind='donor_fed',
        )
    outs_native_ablated = native_forward_ablated(target, X_te_t, T=T)

    max_diff_B = 0.0
    pass_B = True
    for t in range(T):
        diff = (outs_zero[t] - outs_native_ablated[t]).abs().max().item()
        max_diff_B = max(max_diff_B, diff)
        if not torch.allclose(outs_zero[t], outs_native_ablated[t], rtol=0.0, atol=ATOL):
            pass_B = False
    rec['test_B_pass'] = pass_B
    rec['test_B_max_diff'] = max_diff_B

    rec['both_pass'] = pass_A and pass_B
    return rec


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-seeds', type=int, default=N_SEEDS_DEFAULT)
    p.add_argument('--out-csv', default='results/closed_loop_parity_tests.csv')
    args = p.parse_args()

    os.makedirs('results', exist_ok=True)
    print(f"[PARITY] F1 parity tests, N_SEEDS={args.n_seeds}, ATOL={ATOL}")
    t0 = time.time()
    rows = []
    for s in range(args.n_seeds):
        try:
            r = parity_test_seed(s)
            rows.append(r)
            tag = "PASS" if r['both_pass'] else "**FAIL**"
            print(f"  [seed={s:2d}] {tag} "
                  f"test_A_max_diff={r['test_A_max_diff']:.2e} "
                  f"test_B_max_diff={r['test_B_max_diff']:.2e}")
        except Exception as e:
            print(f"  [seed={s}] FAILED with exception: {e}")
            import traceback; traceback.print_exc()
            rows.append({'seed': s, 'both_pass': False,
                         'test_A_pass': False, 'test_B_pass': False,
                         'test_A_max_diff': float('nan'),
                         'test_B_max_diff': float('nan')})

    df = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)

    n_pass = int(df['both_pass'].sum())
    n_total = len(df)
    print(f"\n[PARITY] {n_pass}/{n_total} seeds pass both tests")
    print(f"[PARITY] wall-clock {time.time()-t0:.1f}s")
    print(f"[PARITY] CSV written to {args.out_csv}")

    if n_pass != n_total:
        print("\n*** F1 parity FAILED -- DO NOT proceed to main run. ***")
        sys.exit(1)

    print("\nF1 parity passed. Cleared to proceed with LR pilot.")


if __name__ == '__main__':
    main()
