"""Closed-loop BPTT alignment experiment (the pre-specified analysis plan).

Establishes the open-loop alignment limit as a TASK-SUPERVISED UPPER BOUND
ablation. Closed-loop = aligner trained end-to-end through frozen target's
recurrent unroll with task-loss gradients.

Phases (run sequentially):
  1. PRECONDITION: experiments/run_closed_loop_parity_tests.py must have passed
     (results/closed_loop_parity_tests.csv shows 20/20 PASS at 0.00e+00).
  2. LR pilot (--phase lr-pilot, default seeds 0-4):
       For each (size, lr in {1e-2, 1e-3, 1e-4}), train donor-fed aligner;
       record val_loss at early stop. Lock LR = argmin mean across pilot seeds.
  3. LR sensitivity (--phase lr-sensitivity, default seeds 5-19):
       For each seed, evaluate all 3 LRs at the locked size; check chosen-LR
       val-loss <= 1.10 * best-LR val-loss for >= 13/15 seeds.
  4. delta pilot (--phase delta-pilot):
       From pilot variance of g_BL - g_closed_loop at MLP-medium / MLP-large,
       simulate TOST acceptance; pick smallest delta with power >= 0.70 in
       {0.005, 0.01, 0.015, 0.02}.
  5. Main run (--phase main, default 20 seeds x 4 sizes):
       Donor-fed aligner trained per locked LR. Reports closed-loop gain on
       paper frozen test cohort (200 samples, matched noise seeds).
  6. x_t-only control (--phase xt-only, same 20 x 4):
       Aligner takes x_t only; same protocol. M1 label-decoder check.
  7. Donor ablations (computed inside --phase main, not separately invocable):
       For each main-run aligner: shuffled-donor + zero-donor evaluation
       (no retrain). M2 decoupling diagnostics.

Outputs (results/):
  - closed_loop_alignment.csv             (main donor-fed arm)
  - closed_loop_alignment_xt_only.csv     (M1 control)
  - closed_loop_alignment_shuffled.csv    (M2 shuffled donor eval)
  - closed_loop_alignment_zero_donor.csv  (M2 zero donor eval)
  - closed_loop_alignment_lr_pilot.csv    (LR pilot raw)
  - closed_loop_alignment_lr_sensitivity.csv  (M4 sensitivity check)
  - closed_loop_alignment_delta_pilot.json    (delta + power calc)
  - closed_loop_alignment_summary.json    (aggregate, TOST, Holm, outcome)
  - closed_loop_alignment_meta.json       (RNG seed lineage)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Tuple

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
from scipy.stats import t as t_dist
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam

torch.set_default_dtype(torch.float64)  # the pre-specified analysis plan: float64 throughout

from network import RecurrentMLP
from src.closed_loop_aligner import (
    ClosedLoopAligner, BiasOnlyAligner,
    make_aligner, closed_loop_unroll, freeze_model,
    DONOR_FED_FAMILIES, XT_ONLY_FAMILIES,
)

# ----------------------------------------------------------------------
# Constants matching paper primary protocol
# ----------------------------------------------------------------------
N_SEEDS_DEFAULT     = 20
N_TRAIN             = 200       # for target/donor receiver training
N_TEST              = 200       # frozen test cohort
N_CALIBRATION       = 1000      # the pre-specified analysis plan: 1000 train+val + 200 frozen test = 1200 total
N_VAL_FRACTION      = 0.20      # 800 train / 200 val of the 1000 calibration trajectories
T                   = 3
NOISE               = 0.5
RECEIVER_EPOCHS     = 1000
RECEIVER_LR         = 0.01
ALIGNER_EPOCHS_MAX  = 2000
ALIGNER_PATIENCE    = 100
FEEDBACK_TAU        = 2.0
TIME_WEIGHTS        = (0.0, 0.2, 1.0)
FAMILY_NAMES        = ['affine', 'MLP-small', 'MLP-medium', 'MLP-large']
LR_GRID             = [1e-2, 1e-3, 1e-4]
DELTA_GRID          = [0.005, 0.01, 0.015, 0.02]
POWER_THRESHOLD     = 0.70


# ----------------------------------------------------------------------
# Utilities (mirror run_pytorch_cross_validation)
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_data_vn(n_samples, noise_level=NOISE, T=T, n_classes=5,
                     input_size=10, seed=0):
    """Variable-noise data - mirrors src.training.generate_data_variable_noise."""
    assert input_size >= 2 * n_classes
    rng = np.random.RandomState(seed)
    base = np.zeros((n_classes, input_size))
    for k in range(n_classes):
        base[k, 2 * k: 2 * k + 2] = 1.0
        base[k, (2 * k + 2) % input_size] = 0.3
        base[k, (2 * k - 1) % input_size] = 0.3
    X = np.zeros((n_samples, T, input_size), dtype=np.float64)
    y = np.zeros((n_samples, n_classes), dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        cls = rng.randint(n_classes)
        for t in range(T):
            X[i, t] = base[cls] + noise_level * rng.randn(input_size)
        y[i, cls] = 1.0
        labels[i] = cls
    return X, y, labels


def train_receiver(net, X, y, epochs=RECEIVER_EPOCHS, lr=RECEIVER_LR,
                   time_weights=TIME_WEIGHTS):
    """Train a target or donor receiver. Mirrors run_pytorch_cross_validation.train_model."""
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


def make_target_and_donor(seed: int) -> Tuple[RecurrentMLP, RecurrentMLP]:
    """Train and freeze a (target, donor) pair following the paper convention."""
    # Target
    set_seed(seed)
    X_tr_t, y_tr_t, _ = generate_data_vn(N_TRAIN, seed=seed)
    set_seed(seed)
    target = RecurrentMLP(feedback_tau=FEEDBACK_TAU)
    train_receiver(target, X_tr_t, y_tr_t)
    target = freeze_model(target)

    # Donor
    donor_seed = seed + 100
    set_seed(donor_seed)
    X_tr_d, y_tr_d, _ = generate_data_vn(N_TRAIN, seed=donor_seed)
    set_seed(donor_seed)
    donor = RecurrentMLP(feedback_tau=FEEDBACK_TAU)
    train_receiver(donor, X_tr_d, y_tr_d)
    donor = freeze_model(donor)

    return target, donor


def compute_gain(outs: List[torch.Tensor], labels: torch.Tensor) -> Tuple[float, float, float]:
    """Compute acc_t1, acc_t3, gain = acc_t3 - acc_t1 from forward outputs."""
    pred_t1 = outs[0].argmax(dim=1)
    pred_t3 = outs[-1].argmax(dim=1)
    acc_t1 = float((pred_t1 == labels).float().mean())
    acc_t3 = float((pred_t3 == labels).float().mean())
    return acc_t1, acc_t3, acc_t3 - acc_t1


# ----------------------------------------------------------------------
# Aligner training (closed-loop BPTT)
# ----------------------------------------------------------------------

def train_aligner(
    target: RecurrentMLP,
    donor: RecurrentMLP,
    aligner: ClosedLoopAligner,
    X_train: torch.Tensor, labels_train: torch.Tensor,
    X_val:   torch.Tensor, labels_val:   torch.Tensor,
    lr: float,
    epochs_max: int = ALIGNER_EPOCHS_MAX,
    patience: int = ALIGNER_PATIENCE,
    time_weights: Tuple[float, float, float] = TIME_WEIGHTS,
    aligner_input_kind: str = 'donor_fed',
) -> Dict:
    """Train aligner via BPTT through frozen target. Early-stop on val loss.

    Returns dict with final epoch, best_val_loss, train_loss_at_best,
    epochs_run, time_elapsed.
    """
    opt = Adam(aligner.parameters(), lr=lr)
    aligner.train()
    target.eval()
    donor.eval()
    t0 = time.time()

    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    wait = 0
    epochs_run = 0
    train_loss_at_best = float('nan')

    for epoch in range(epochs_max):
        epochs_run = epoch + 1
        # Train step (full-batch)
        opt.zero_grad()
        outs_tr, _ = closed_loop_unroll(
            target, donor, aligner, X_train,
            T=T, feedback_tau=FEEDBACK_TAU, aligner_input_kind=aligner_input_kind,
        )
        loss_tr = sum(w * F.cross_entropy(outs_tr[t], labels_train)
                      for t, w in enumerate(time_weights)) / sum(time_weights)
        loss_tr.backward()
        opt.step()

        # Val step
        aligner.eval()
        with torch.no_grad():
            outs_val, _ = closed_loop_unroll(
                target, donor, aligner, X_val,
                T=T, feedback_tau=FEEDBACK_TAU, aligner_input_kind=aligner_input_kind,
            )
            loss_val = sum(w * F.cross_entropy(outs_val[t], labels_val)
                           for t, w in enumerate(time_weights)) / sum(time_weights)
        aligner.train()

        if float(loss_val) < best_val_loss:
            best_val_loss = float(loss_val)
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in aligner.state_dict().items()}
            train_loss_at_best = float(loss_tr.detach())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best
    if best_state is not None:
        aligner.load_state_dict(best_state)

    return {
        'epochs_run': epochs_run,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_loss_at_best': train_loss_at_best,
        'elapsed_s': time.time() - t0,
    }


# ----------------------------------------------------------------------
# Phase 1: LR pilot (seeds 0-4, all sizes, all LRs)
# ----------------------------------------------------------------------

def phase_lr_pilot(seeds: List[int], out_csv: str,
                   aligner_kind: str = 'donor_fed') -> pd.DataFrame:
    """Run LR pilot: for each (seed, size, lr), train aligner and record val_loss."""
    rows = []
    for seed in seeds:
        print(f"  [LR-pilot seed={seed:2d}]")
        target, donor = make_target_and_donor(seed)

        # Calibration data (1000 trajectories: 800 train + 200 val)
        rng_seed = seed * 1000 + 7
        X_cal, y_cal, _ = generate_data_vn(N_CALIBRATION, seed=rng_seed)
        labels_cal = torch.from_numpy(np.argmax(y_cal, axis=1))
        X_cal_t = torch.from_numpy(X_cal)

        n_val = int(N_CALIBRATION * N_VAL_FRACTION)
        n_train = N_CALIBRATION - n_val
        # Deterministic split: first n_train for train, rest for val
        X_train = X_cal_t[:n_train]
        labels_train = labels_cal[:n_train]
        X_val = X_cal_t[n_train:]
        labels_val = labels_cal[n_train:]

        for size in FAMILY_NAMES:
            for lr in LR_GRID:
                # Fresh aligner per (size, lr)
                set_seed(seed * 100 + LR_GRID.index(lr))
                aligner = make_aligner(size, aligner_kind)
                info = train_aligner(
                    target, donor, aligner,
                    X_train, labels_train, X_val, labels_val,
                    lr=lr, aligner_input_kind=aligner_kind,
                )
                rows.append({
                    'seed': seed,
                    'aligner_size': size,
                    'lr': lr,
                    'aligner_kind': aligner_kind,
                    'best_val_loss': info['best_val_loss'],
                    'best_epoch': info['best_epoch'],
                    'epochs_run': info['epochs_run'],
                    'elapsed_s': info['elapsed_s'],
                })
                print(f"     size={size:11s} lr={lr:7.0e} best_val={info['best_val_loss']:.4f} "
                      f"epoch={info['best_epoch']:3d} elapsed={info['elapsed_s']:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n  [LR-pilot] wrote {out_csv}")
    return df


def lock_lr_from_pilot(pilot_df: pd.DataFrame) -> Dict[str, float]:
    """For each size, lock LR = argmin mean best_val_loss across pilot seeds."""
    locked = {}
    for size in FAMILY_NAMES:
        sub = pilot_df[pilot_df['aligner_size'] == size]
        mean_per_lr = sub.groupby('lr')['best_val_loss'].mean()
        best_lr = float(mean_per_lr.idxmin())
        locked[size] = best_lr
    return locked


# ----------------------------------------------------------------------
# Phase 2: LR sensitivity check (M4)
# ----------------------------------------------------------------------

def phase_lr_sensitivity(seeds: List[int], locked_lr: Dict[str, float],
                         out_csv: str, aligner_kind: str = 'donor_fed') -> pd.DataFrame:
    """For each (seed in 5..19, size), run all 3 LRs; check chosen-LR within 10% of best-LR."""
    rows = []
    for seed in seeds:
        print(f"  [LR-sens seed={seed:2d}]")
        target, donor = make_target_and_donor(seed)

        rng_seed = seed * 1000 + 7
        X_cal, y_cal, _ = generate_data_vn(N_CALIBRATION, seed=rng_seed)
        labels_cal = torch.from_numpy(np.argmax(y_cal, axis=1))
        X_cal_t = torch.from_numpy(X_cal)
        n_val = int(N_CALIBRATION * N_VAL_FRACTION)
        n_train = N_CALIBRATION - n_val
        X_train = X_cal_t[:n_train]
        labels_train = labels_cal[:n_train]
        X_val = X_cal_t[n_train:]
        labels_val = labels_cal[n_train:]

        for size in FAMILY_NAMES:
            per_lr = {}
            for lr in LR_GRID:
                set_seed(seed * 100 + LR_GRID.index(lr))
                aligner = make_aligner(size, aligner_kind)
                info = train_aligner(
                    target, donor, aligner,
                    X_train, labels_train, X_val, labels_val,
                    lr=lr, aligner_input_kind=aligner_kind,
                )
                per_lr[lr] = info['best_val_loss']

            chosen_lr = locked_lr[size]
            chosen_val = per_lr[chosen_lr]
            best_val = min(per_lr.values())
            ratio = chosen_val / best_val if best_val > 0 else float('inf')
            stable = ratio <= 1.10

            rows.append({
                'seed': seed,
                'aligner_size': size,
                'chosen_lr': chosen_lr,
                'chosen_val_loss': chosen_val,
                'best_lr': min(per_lr, key=per_lr.get),
                'best_val_loss': best_val,
                'ratio': ratio,
                'stable_within_10pct': stable,
            })
            print(f"     size={size:11s} chosen_lr={chosen_lr:.0e} "
                  f"chosen={chosen_val:.4f} best={best_val:.4f} "
                  f"ratio={ratio:.3f} stable={stable}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ----------------------------------------------------------------------
# Static-PyTorch aligner (for paired H1 comparison with closed-loop)
# ----------------------------------------------------------------------

def train_static_aligner_pytorch(
    target: RecurrentMLP, donor: RecurrentMLP,
    aligner: ClosedLoopAligner,
    X_cal: torch.Tensor,
    lr: float = 1e-2,
    epochs_max: int = ALIGNER_EPOCHS_MAX,
    patience: int = ALIGNER_PATIENCE,
    val_fraction: float = N_VAL_FRACTION,
) -> Dict:
    """Static-aligner training in PyTorch (matched harness for paired comparison).

    Generates (donor_logit, target_logit) pairs by running target and donor
    independently with self-feedback on the calibration cohort, then fits
    aligner via MSE on the pairs. This mirrors the existing NumPy
    run_stronger_alignment but in PyTorch on the same target/donor used by
    the closed-loop arm.
    """
    target.eval(); donor.eval()
    with torch.no_grad():
        donor_outs = donor(X_cal, T=T, feedback_mode='self')  # list of [batch, 5]
        target_outs = target(X_cal, T=T, feedback_mode='self')

    # Keep sequence (trajectory) structure: [batch, T, 5]
    donor_seq  = torch.stack(donor_outs, dim=1)
    target_seq = torch.stack(target_outs, dim=1)
    n_seq = donor_seq.shape[0]

    # Train/val split at the TRAJECTORY level. Sequential (deterministic) cut
    # at index `n_seq - n_val_seq` to perfectly mirror the closed-loop arm's
    # split (`X_cal_t[:800]` train, `X_cal_t[800:]` val in `phase_main`).
    # A correctness fix replaced the original row-level (batch*T) flat permutation
    # that leaked timesteps across train/val; a later fix removed the
    # within-trajectory permutation that was injecting cross-cohort variance
    # into the paired H1 comparison (the closed-loop arm uses a sequential
    # split, so the paired-comparator must match for variance reduction).
    n_val_seq = int(n_seq * val_fraction)
    train_seq = np.arange(n_seq - n_val_seq)
    val_seq   = np.arange(n_seq - n_val_seq, n_seq)

    X_train = donor_seq[train_seq].reshape(-1, 5)
    Y_train = target_seq[train_seq].reshape(-1, 5)
    X_val   = donor_seq[val_seq].reshape(-1, 5)
    Y_val   = target_seq[val_seq].reshape(-1, 5)

    opt = Adam(aligner.parameters(), lr=lr)
    aligner.train()
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    epochs_run = 0
    t0 = time.time()

    for epoch in range(epochs_max):
        epochs_run = epoch + 1
        opt.zero_grad()
        pred_train = aligner(X_train)
        loss_train = F.mse_loss(pred_train, Y_train)
        loss_train.backward()
        opt.step()

        aligner.eval()
        with torch.no_grad():
            pred_val = aligner(X_val)
            loss_val = F.mse_loss(pred_val, Y_val)
        aligner.train()

        if float(loss_val) < best_val_loss:
            best_val_loss = float(loss_val)
            best_state = {k: v.clone() for k, v in aligner.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        aligner.load_state_dict(best_state)

    return {
        'epochs_run': epochs_run,
        'best_val_loss': best_val_loss,
        'elapsed_s': time.time() - t0,
    }


def phase_static_pytorch(seeds: List[int], locked_lr: Dict[str, float],
                          out_csv: str) -> pd.DataFrame:
    """Train static-PyTorch aligners (MSE on logit pairs) for paired comparison."""
    rows = []
    for seed in seeds:
        print(f"\n  [STATIC-PT seed={seed:2d}]")
        target, donor = make_target_and_donor(seed)
        X_te, y_te, _ = generate_data_vn(N_TEST, seed=seed + 500)
        X_te_t = torch.from_numpy(X_te)
        labels_te = torch.from_numpy(np.argmax(y_te, axis=1))
        ref = evaluate_reference_bl_c2_groupa(target, donor, X_te_t, labels_te)

        X_cal, _, _ = generate_data_vn(N_CALIBRATION, seed=seed * 1000 + 7)
        X_cal_t = torch.from_numpy(X_cal)

        for size in FAMILY_NAMES:
            set_seed(seed * 100 + FAMILY_NAMES.index(size) * 17 + 31)
            aligner = make_aligner(size, 'donor_fed')
            info = train_static_aligner_pytorch(
                target, donor, aligner, X_cal_t, lr=locked_lr[size],
            )
            eval_rec = evaluate_aligner_on_test(target, donor, aligner, X_te_t, labels_te,
                                                 aligner_input_kind='donor_fed')
            rows.append({
                'seed': seed, 'aligner_size': size,
                'aligner_params': aligner.count_params(),
                'lr': locked_lr[size],
                'bl_gain': ref['bl_gain'],
                'c2_raw_gain': ref['c2_raw_gain'],
                'static_pt_gain': eval_rec['gain'],
                'static_pt_acc_t1': eval_rec['acc_t1'],
                'static_pt_acc_t3': eval_rec['acc_t3'],
                'epochs_run': info['epochs_run'],
                'best_val_loss': info['best_val_loss'],
                'elapsed_s': info['elapsed_s'],
            })
            print(f"    size={size:11s} static_PT_gain={eval_rec['gain']:+.4f} "
                  f"recovery={(eval_rec['gain']-ref['c2_raw_gain'])/(ref['bl_gain']-ref['c2_raw_gain']+1e-9):.2f} "
                  f"epochs={info['epochs_run']}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n  [STATIC-PT] wrote {out_csv}")
    return df


# ----------------------------------------------------------------------
# Evaluation utilities (gain on frozen test cohort)
# ----------------------------------------------------------------------

def evaluate_aligner_on_test(target: RecurrentMLP, donor: RecurrentMLP,
                              aligner: ClosedLoopAligner,
                              X_te: torch.Tensor, labels_te: torch.Tensor,
                              aligner_input_kind: str = 'donor_fed',
                              diagnostics: bool = False) -> Dict:
    """Compute gain + diagnostics on the frozen test cohort."""
    aligner.eval()
    with torch.no_grad():
        outs, diag = closed_loop_unroll(
            target, donor, aligner, X_te,
            T=T, feedback_tau=FEEDBACK_TAU, aligner_input_kind=aligner_input_kind,
            diagnostics=diagnostics,
        )
    acc_t1, acc_t3, gain = compute_gain(outs, labels_te)
    rec = {'acc_t1': acc_t1, 'acc_t3': acc_t3, 'gain': gain}
    if diagnostics and diag is not None:
        for key, vals in diag.items():
            # vals is a list of per-timestep scalars; average across t=2,3
            if vals:
                rec[f'{key}_t2'] = vals[0]
                rec[f'{key}_t3'] = vals[1] if len(vals) > 1 else float('nan')
    return rec


def evaluate_reference_bl_c2_groupa(target: RecurrentMLP, donor: RecurrentMLP,
                                     X_te: torch.Tensor, labels_te: torch.Tensor) -> Dict:
    """Compute per-seed Baseline (self-feedback), C2 (clone), and Group A (no-fb) gains
    using the native PyTorch RecurrentMLP forward (no aligner)."""
    target.eval(); donor.eval()
    with torch.no_grad():
        outs_bl = target(X_te, T=T, feedback_mode='self')
        outs_c2 = target(X_te, T=T, feedback_mode='clone', clone=donor)
        outs_a  = target(X_te, T=T, feedback_mode='ablated')
    _, _, bl_gain = compute_gain(outs_bl, labels_te)
    _, _, c2_gain = compute_gain(outs_c2, labels_te)
    _, _, a_gain  = compute_gain(outs_a,  labels_te)
    return {'bl_gain': bl_gain, 'c2_raw_gain': c2_gain, 'group_a_gain': a_gain}


def cosine_to_self_feedback(target: RecurrentMLP, donor: RecurrentMLP,
                             aligner: ClosedLoopAligner,
                             X_te: torch.Tensor,
                             aligner_input_kind: str = 'donor_fed') -> Tuple[float, float]:
    """Cosine similarity between aligner output and target's actual self-feedback
    on matched inputs (target running standalone with self-feedback)."""
    # Run target alone (self-feedback) to get y_self at each timestep
    target.eval(); donor.eval(); aligner.eval()
    with torch.no_grad():
        # Self-feedback trajectory
        outs_self = target(X_te, T=T, feedback_mode='self')
        # Donor trajectory (for aligner input)
        outs_donor = donor(X_te, T=T, feedback_mode='self')
        # Aligned logits at t=2, t=3 (from donor's t=1, t=2 outputs)
        if aligner_input_kind == 'donor_fed':
            aligned_t2 = aligner(outs_donor[0])  # input from donor t=1
            aligned_t3 = aligner(outs_donor[1])  # input from donor t=2
        else:
            aligned_t2 = aligner(X_te[:, 1, :])
            aligned_t3 = aligner(X_te[:, 2, :])
        self_t1 = outs_self[0]
        self_t2 = outs_self[1]
        # Compare aligned(donor_{t-1}) vs self_t-1 (the feedback signal target would have produced)
        cos_t2 = float(F.cosine_similarity(aligned_t2, self_t1, dim=1).mean())
        cos_t3 = float(F.cosine_similarity(aligned_t3, self_t2, dim=1).mean())
    return cos_t2, cos_t3


# ----------------------------------------------------------------------
# Phase: delta pilot (the pre-specified analysis plan, H1 fix)
# ----------------------------------------------------------------------

def phase_delta_pilot(out_json: str, locked_lr: Dict[str, float],
                      pilot_seeds: List[int]) -> Dict:
    """From pilot variance of (g_BL - g_closed_loop) at MLP-medium / MLP-large,
    simulate TOST acceptance for delta in {0.005, 0.01, 0.015, 0.02}; pick smallest
    delta with power >= POWER_THRESHOLD.
    """
    print(f"  [delta-pilot] pilot seeds: {pilot_seeds}")

    # Collect pilot paired diffs at medium and large
    diffs_medium = []
    diffs_large = []
    for seed in pilot_seeds:
        target, donor = make_target_and_donor(seed)
        # Test cohort
        X_te, y_te, _ = generate_data_vn(N_TEST, seed=seed + 500)
        X_te_t = torch.from_numpy(X_te)
        labels_te = torch.from_numpy(np.argmax(y_te, axis=1))
        ref = evaluate_reference_bl_c2_groupa(target, donor, X_te_t, labels_te)

        # Calibration data
        X_cal, y_cal, _ = generate_data_vn(N_CALIBRATION, seed=seed * 1000 + 7)
        labels_cal = torch.from_numpy(np.argmax(y_cal, axis=1))
        X_cal_t = torch.from_numpy(X_cal)
        X_train, X_val = X_cal_t[:800], X_cal_t[800:]
        labels_train, labels_val = labels_cal[:800], labels_cal[800:]

        for size, diffs_list in [('MLP-medium', diffs_medium), ('MLP-large', diffs_large)]:
            set_seed(seed * 100 + 999)
            aligner = make_aligner(size, 'donor_fed')
            train_aligner(target, donor, aligner, X_train, labels_train, X_val, labels_val,
                          lr=locked_lr[size])
            eval_rec = evaluate_aligner_on_test(target, donor, aligner, X_te_t, labels_te)
            diff = ref['bl_gain'] - eval_rec['gain']
            diffs_list.append(diff)
            print(f"    seed={seed:2d} size={size:11s} diff={diff:+.4f}")

    diffs_medium = np.array(diffs_medium)
    diffs_large = np.array(diffs_large)
    var_medium = float(np.var(diffs_medium, ddof=1))
    var_large = float(np.var(diffs_large, ddof=1))
    std_pooled = float(np.sqrt((var_medium + var_large) / 2))
    print(f"\n  [delta-pilot] pooled std of paired diffs ~ {std_pooled:.4f}")

    # Simulate TOST acceptance under (a) true diff = +delta (null: closed-loop worse),
    # (b) true diff = 0 (equivalence alternative)
    n_sim = 5000
    rng = np.random.RandomState(42)
    pilot_n = 20  # main run will use N=20

    power_results = {}
    for delta in DELTA_GRID:
        # Power = P(reject both nulls | true diff = 0)
        # Under true diff = 0, generate paired diffs ~ N(0, std_pooled^2), n=20
        n_accept = 0
        for _ in range(n_sim):
            d_sim = rng.normal(0, std_pooled, size=pilot_n)
            d_mean = d_sim.mean()
            d_se = d_sim.std(ddof=1) / np.sqrt(pilot_n)
            # 95% CI of the paired mean diff: d_mean +- t_{n-1, 0.975} * d_se.
            # Previously this used the asymptotic Normal quantile 1.96, which
            # is too small at pilot_n=20 (t_{19, 0.975} ~ 2.093) and
            # underestimated CI width -> overestimated TOST acceptance power.
            t_crit = float(t_dist.ppf(0.975, df=pilot_n - 1))
            ci_low = d_mean - t_crit * d_se
            ci_high = d_mean + t_crit * d_se
            if ci_low > -delta and ci_high < delta:
                n_accept += 1
        power_results[delta] = n_accept / n_sim
        # Type I error: under true diff = +delta, the test should reject the null at <= alpha
        # (less critical to simulate here for the smallest-delta-with-power selection)
        print(f"    delta={delta:.3f}: power(true=0) = {power_results[delta]:.3f}")

    # Pick smallest delta with power >= threshold
    chosen_delta = None
    for d in sorted(DELTA_GRID):
        if power_results[d] >= POWER_THRESHOLD:
            chosen_delta = d
            break
    if chosen_delta is None:
        print(f"  [delta-pilot] WARNING: no delta achieves power {POWER_THRESHOLD}; using max {max(DELTA_GRID)}")
        chosen_delta = max(DELTA_GRID)
        underpowered = True
    else:
        underpowered = False

    record = {
        'pilot_seeds': pilot_seeds,
        'pilot_diffs_medium': diffs_medium.tolist(),
        'pilot_diffs_large': diffs_large.tolist(),
        'var_medium': var_medium,
        'var_large': var_large,
        'std_pooled': std_pooled,
        'delta_grid': DELTA_GRID,
        'power_results': power_results,
        'chosen_delta': chosen_delta,
        'power_threshold': POWER_THRESHOLD,
        'underpowered': underpowered,
        'n_simulations': n_sim,
    }
    with open(out_json, 'w') as f:
        json.dump(record, f, indent=2)
    print(f"\n  [delta-pilot] chosen_delta = {chosen_delta:.3f} (power={power_results[chosen_delta]:.3f})")
    print(f"  [delta-pilot] wrote {out_json}")
    return record


# ----------------------------------------------------------------------
# Phase: main run (donor-fed primary, all 20 seeds x 4 sizes)
# ----------------------------------------------------------------------

def phase_main(seeds: List[int], locked_lr: Dict[str, float],
               out_csv: str, aligner_kind: str = 'donor_fed') -> pd.DataFrame:
    """Train donor-fed aligner per (seed, size); evaluate on frozen test cohort."""
    rows = []
    aligner_states = {}  # cache trained aligners for donor-ablation phase

    for seed in seeds:
        print(f"\n  [MAIN seed={seed:2d}, kind={aligner_kind}]")
        target, donor = make_target_and_donor(seed)

        # Test cohort (frozen) + reference
        X_te, y_te, _ = generate_data_vn(N_TEST, seed=seed + 500)
        X_te_t = torch.from_numpy(X_te)
        labels_te = torch.from_numpy(np.argmax(y_te, axis=1))
        ref = evaluate_reference_bl_c2_groupa(target, donor, X_te_t, labels_te)
        print(f"    refs: BL={ref['bl_gain']:+.4f}  C2_raw={ref['c2_raw_gain']:+.4f}  "
              f"GroupA={ref['group_a_gain']:+.4f}")

        # Calibration cohort
        X_cal, y_cal, _ = generate_data_vn(N_CALIBRATION, seed=seed * 1000 + 7)
        labels_cal = torch.from_numpy(np.argmax(y_cal, axis=1))
        X_cal_t = torch.from_numpy(X_cal)
        X_train, X_val = X_cal_t[:800], X_cal_t[800:]
        labels_train, labels_val = labels_cal[:800], labels_cal[800:]

        for size in FAMILY_NAMES:
            set_seed(seed * 100 + FAMILY_NAMES.index(size) * 17 + 31)
            aligner = make_aligner(size, aligner_kind)
            info = train_aligner(
                target, donor, aligner,
                X_train, labels_train, X_val, labels_val,
                lr=locked_lr[size],
                aligner_input_kind=aligner_kind,
            )
            eval_rec = evaluate_aligner_on_test(target, donor, aligner, X_te_t, labels_te,
                                                 aligner_input_kind=aligner_kind, diagnostics=True)
            cos_t2, cos_t3 = cosine_to_self_feedback(target, donor, aligner, X_te_t,
                                                       aligner_input_kind=aligner_kind)

            row = {
                'seed': seed, 'aligner_size': size, 'aligner_kind': aligner_kind,
                'aligner_params': aligner.count_params(),
                'lr': locked_lr[size],
                'bl_gain': ref['bl_gain'],
                'c2_raw_gain': ref['c2_raw_gain'],
                'group_a_gain': ref['group_a_gain'],
                'closed_loop_gain': eval_rec['gain'],
                'closed_loop_acc_t1': eval_rec['acc_t1'],
                'closed_loop_acc_t3': eval_rec['acc_t3'],
                'cosine_to_self_t2': cos_t2,
                'cosine_to_self_t3': cos_t3,
                'epochs_run': info['epochs_run'],
                'best_epoch': info['best_epoch'],
                'best_val_loss': info['best_val_loss'],
                'elapsed_s': info['elapsed_s'],
            }
            # Add diagnostics
            for key in ['aligned_logits_norm_t2', 'aligned_logits_norm_t3',
                        'aligned_logits_max_abs_t2', 'aligned_logits_max_abs_t3',
                        'feedback_norm_t2', 'feedback_norm_t3',
                        'tanh_saturation_fraction_t2', 'tanh_saturation_fraction_t3']:
                row[key] = eval_rec.get(key, float('nan'))
            rows.append(row)

            # Cache for donor-ablation
            aligner_states[(seed, size)] = (aligner.state_dict(),
                                              X_te_t, labels_te, target, donor, ref)

            print(f"    size={size:11s} CL_gain={eval_rec['gain']:+.4f} "
                  f"recovery={(eval_rec['gain']-ref['c2_raw_gain'])/(ref['bl_gain']-ref['c2_raw_gain']+1e-9):.2f} "
                  f"cos_t3={cos_t3:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n  [MAIN] wrote {out_csv}")
    return df, aligner_states


# ----------------------------------------------------------------------
# Phase: donor ablations (shuffled + zero donor evaluation, no retrain)
# ----------------------------------------------------------------------

def phase_donor_ablations(aligner_states: Dict, out_shuffled_csv: str,
                          out_zero_csv: str, aligner_kind: str = 'donor_fed'):
    """For each cached (seed, size) trained aligner: evaluate with shuffled donor
    trajectories and with zero donor input. No retraining."""
    shuffled_rows = []
    zero_rows = []
    for (seed, size), (aligner_state, X_te, labels_te, target, donor, ref) in aligner_states.items():
        aligner = make_aligner(size, aligner_kind)
        aligner.load_state_dict(aligner_state)
        aligner.eval()

        # Shuffled donor: permute samples in the test cohort for donor input
        # while keeping target's x_t identical. Precompute donor outputs on
        # the shuffled X, then inject via aligner.
        #
        # Average across SHUFFLED_DONOR_REPEATS independent permutations per
        # (seed, size) to remove single-shuffle noise (the prior single-perm
        # implementation was Method-B-style; this matches the Method-A
        # 30-shuffle averaging convention used by Group C1 in run_n20_full).
        # The model is the inferential unit (N=20); the within-seed average
        # is purely a within-cell variance reduction.
        SHUFFLED_DONOR_REPEATS = 30
        gain_reps = []
        acc_t1_reps = []
        acc_t3_reps = []
        donor_outs_shuffled = None  # carries forward to the zero-donor block
        for rep in range(SHUFFLED_DONOR_REPEATS):
            rng_shuffle = np.random.RandomState(seed * 7919 + 1 + rep)
            perm = rng_shuffle.permutation(X_te.shape[0])
            X_te_shuffled_donor = X_te[perm]
            with torch.no_grad():
                donor_outs_rep = donor(X_te_shuffled_donor, T=T, feedback_mode='self')
                outs_rep = _closed_loop_unroll_with_external_donor_outs(
                    target, aligner, X_te, donor_outs_rep, aligner_kind=aligner_kind,
                )
            a1, a3, g = compute_gain(outs_rep, labels_te)
            gain_reps.append(g)
            acc_t1_reps.append(a1)
            acc_t3_reps.append(a3)
            if donor_outs_shuffled is None:
                donor_outs_shuffled = donor_outs_rep  # any rep works for the zero-donor block (only shape used)
        gain = float(np.mean(gain_reps))
        acc_t1 = float(np.mean(acc_t1_reps))
        acc_t3 = float(np.mean(acc_t3_reps))
        shuffled_rows.append({
            'seed': seed, 'aligner_size': size,
            'shuffled_donor_gain': gain, 'acc_t1': acc_t1, 'acc_t3': acc_t3,
            'shuffled_donor_gain_std': float(np.std(gain_reps, ddof=1)),
            'shuffled_donor_n_repeats': SHUFFLED_DONOR_REPEATS,
            'closed_loop_gain': ref.get('closed_loop_gain', None),  # filled by aggregate
            'bl_gain': ref['bl_gain'], 'c2_raw_gain': ref['c2_raw_gain'],
        })

        # Zero donor: replace donor outputs with zeros (single eval - no
        # randomness to average over).
        zero_donor_outs = [torch.zeros_like(o) for o in donor_outs_shuffled]
        with torch.no_grad():
            outs_z = _closed_loop_unroll_with_external_donor_outs(
                target, aligner, X_te, zero_donor_outs, aligner_kind=aligner_kind,
            )
        acc_t1_z, acc_t3_z, gain_z = compute_gain(outs_z, labels_te)
        zero_rows.append({
            'seed': seed, 'aligner_size': size,
            'zero_donor_gain': gain_z, 'acc_t1': acc_t1_z, 'acc_t3': acc_t3_z,
            'bl_gain': ref['bl_gain'], 'c2_raw_gain': ref['c2_raw_gain'],
        })

    pd.DataFrame(shuffled_rows).to_csv(out_shuffled_csv, index=False)
    pd.DataFrame(zero_rows).to_csv(out_zero_csv, index=False)
    print(f"  [donor-ablations] wrote {out_shuffled_csv}, {out_zero_csv}")


def _closed_loop_unroll_with_external_donor_outs(target, aligner, X, donor_outs,
                                                  aligner_kind='donor_fed'):
    """Helper: closed-loop unroll where donor outputs come from an external precomputed list."""
    batch = X.shape[0]
    device = next(target.parameters()).device
    output_dim = target.fc_out.out_features
    outputs = []
    for t in range(T):
        x_t = X[:, t, :]
        if t == 0:
            h1 = F.relu(target.fc1(x_t))
        else:
            if aligner_kind == 'donor_fed':
                aligner_input = donor_outs[t - 1]
            else:
                aligner_input = x_t
            aligned_logits = aligner(aligner_input)
            feedback = torch.tanh(aligned_logits / FEEDBACK_TAU)
            h1 = F.relu(target.fc1(x_t) + target.W_rec(feedback))
        h2 = F.relu(target.fc2(h1))
        outputs.append(target.fc_out(h2))
    return outputs


# ----------------------------------------------------------------------
# Phase: bias-only mechanism diagnostic
# ----------------------------------------------------------------------

def phase_bias_only(seeds: List[int], out_csv: str,
                    lr: float = 1e-2, epochs_max: int = ALIGNER_EPOCHS_MAX,
                    patience: int = ALIGNER_PATIENCE) -> pd.DataFrame:
    """Train a BiasOnlyAligner (5 params, no input) per seed.

    Mechanism diagnostic: if this matches xt-only / donor-fed gain, the
    closed-loop "alignment" is just a constant-bias injection through W_rec,
    confirming that nothing input-dependent or representational is required
    for the BL-exceeding closed-loop gain.
    """
    print(f"\n  [BIAS-ONLY] {len(seeds)} seeds, lr={lr}, 5 params each")
    rows = []
    for seed in seeds:
        target, donor = make_target_and_donor(seed)
        X_te, y_te, _ = generate_data_vn(N_TEST, seed=seed + 500)
        X_te_t = torch.from_numpy(X_te)
        labels_te = torch.from_numpy(np.argmax(y_te, axis=1))
        ref = evaluate_reference_bl_c2_groupa(target, donor, X_te_t, labels_te)

        # Calibration data
        X_cal, y_cal, _ = generate_data_vn(N_CALIBRATION, seed=seed * 1000 + 7)
        labels_cal = torch.from_numpy(np.argmax(y_cal, axis=1))
        X_cal_t = torch.from_numpy(X_cal)
        X_train, X_val = X_cal_t[:800], X_cal_t[800:]
        labels_train, labels_val = labels_cal[:800], labels_cal[800:]

        set_seed(seed * 100 + 999)  # unique seed for bias-only init (irrelevant - init at 0)
        aligner = BiasOnlyAligner(output_dim=5)
        info = train_aligner(
            target, donor, aligner,
            X_train, labels_train, X_val, labels_val,
            lr=lr, epochs_max=epochs_max, patience=patience,
            aligner_input_kind='donor_fed',
        )
        eval_rec = evaluate_aligner_on_test(target, donor, aligner, X_te_t, labels_te,
                                             aligner_input_kind='donor_fed', diagnostics=True)

        # Record learned bias value for inspection
        learned_bias = aligner.bias.detach().cpu().numpy().tolist()

        row = {
            'seed': seed,
            'aligner_size': 'bias_only',
            'aligner_params': aligner.count_params(),
            'lr': lr,
            'bl_gain': ref['bl_gain'],
            'c2_raw_gain': ref['c2_raw_gain'],
            'group_a_gain': ref['group_a_gain'],
            'bias_only_gain': eval_rec['gain'],
            'bias_only_acc_t1': eval_rec['acc_t1'],
            'bias_only_acc_t3': eval_rec['acc_t3'],
            'learned_bias_0': learned_bias[0],
            'learned_bias_1': learned_bias[1],
            'learned_bias_2': learned_bias[2],
            'learned_bias_3': learned_bias[3],
            'learned_bias_4': learned_bias[4],
            'learned_bias_norm': float(np.linalg.norm(learned_bias)),
            'epochs_run': info['epochs_run'],
            'best_epoch': info['best_epoch'],
            'best_val_loss': info['best_val_loss'],
            'elapsed_s': info['elapsed_s'],
        }
        for key in ['aligned_logits_norm_t2', 'aligned_logits_norm_t3',
                    'aligned_logits_max_abs_t2', 'aligned_logits_max_abs_t3',
                    'feedback_norm_t2', 'feedback_norm_t3',
                    'tanh_saturation_fraction_t2', 'tanh_saturation_fraction_t3']:
            row[key] = eval_rec.get(key, float('nan'))
        rows.append(row)

        print(f"    seed={seed:2d}  bias-only_gain={eval_rec['gain']:+.4f}  "
              f"BL={ref['bl_gain']:+.4f}  bias_norm={row['learned_bias_norm']:.2f}  "
              f"epochs={info['epochs_run']}  elapsed={info['elapsed_s']:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n  [BIAS-ONLY] wrote {out_csv}")
    print(f"  [BIAS-ONLY] mean gain: {df['bias_only_gain'].mean():+.4f} (std {df['bias_only_gain'].std(ddof=1):.4f})")
    return df


# ----------------------------------------------------------------------
# CLI dispatcher
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--phase', required=True,
                   choices=['lr-pilot', 'lr-sensitivity', 'delta-pilot',
                            'main', 'xt-only', 'static-pytorch', 'donor-ablations',
                            'bias-only'])
    p.add_argument('--pilot-seeds', default='0,1,2,3,4',
                   help='comma-separated pilot seeds (default 0,1,2,3,4)')
    p.add_argument('--sens-seeds', default='5,6,7,8,9,10,11,12,13,14,15,16,17,18,19',
                   help='comma-separated sensitivity seeds')
    p.add_argument('--all-seeds', default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19',
                   help='all seeds for main run')
    p.add_argument('--locked-lr-json', default='results/closed_loop_alignment_locked_lr.json',
                   help='where to load/save locked LR per size')
    args = p.parse_args()

    os.makedirs('results', exist_ok=True)

    if args.phase == 'lr-pilot':
        seeds = [int(s) for s in args.pilot_seeds.split(',')]
        df = phase_lr_pilot(seeds, 'results/closed_loop_alignment_lr_pilot.csv')
        locked = lock_lr_from_pilot(df)
        with open(args.locked_lr_json, 'w') as f:
            json.dump(locked, f, indent=2)
        print(f"\n  [LR-lock] locked LRs: {locked}")
        print(f"  [LR-lock] wrote {args.locked_lr_json}")

    elif args.phase == 'lr-sensitivity':
        seeds = [int(s) for s in args.sens_seeds.split(',')]
        with open(args.locked_lr_json) as f:
            locked = json.load(f)
        df = phase_lr_sensitivity(seeds, locked, 'results/closed_loop_alignment_lr_sensitivity.csv')
        n_stable = int(df['stable_within_10pct'].sum())
        # Per size, count stable seeds
        for size in FAMILY_NAMES:
            sub = df[df['aligner_size'] == size]
            n_stable_size = int(sub['stable_within_10pct'].sum())
            n_total_size = len(sub)
            tag = "OK" if n_stable_size >= 13 else "**INSUFFICIENT**"
            print(f"  [LR-sens] {size:11s} stable {n_stable_size}/{n_total_size} {tag}")

    elif args.phase == 'delta-pilot':
        seeds = [int(s) for s in args.pilot_seeds.split(',')]
        with open(args.locked_lr_json) as f:
            locked = json.load(f)
        phase_delta_pilot('results/closed_loop_alignment_delta_pilot.json', locked, seeds)

    elif args.phase == 'main':
        seeds = [int(s) for s in args.all_seeds.split(',')]
        with open(args.locked_lr_json) as f:
            locked = json.load(f)
        df, aligner_states = phase_main(seeds, locked,
                                         'results/closed_loop_alignment.csv',
                                         aligner_kind='donor_fed')
        # Run donor ablations using cached states
        phase_donor_ablations(aligner_states,
                              'results/closed_loop_alignment_shuffled.csv',
                              'results/closed_loop_alignment_zero_donor.csv',
                              aligner_kind='donor_fed')

    elif args.phase == 'xt-only':
        seeds = [int(s) for s in args.all_seeds.split(',')]
        with open(args.locked_lr_json) as f:
            locked = json.load(f)
        df, _ = phase_main(seeds, locked,
                            'results/closed_loop_alignment_xt_only.csv',
                            aligner_kind='xt_only')

    elif args.phase == 'static-pytorch':
        seeds = [int(s) for s in args.all_seeds.split(',')]
        with open(args.locked_lr_json) as f:
            locked = json.load(f)
        phase_static_pytorch(seeds, locked,
                              'results/closed_loop_alignment_static_pt.csv')

    elif args.phase == 'bias-only':
        # Mechanism diagnostic: 5-dim learnable bias aligner, no input
        seeds = [int(s) for s in args.all_seeds.split(',')]
        phase_bias_only(seeds, 'results/closed_loop_alignment_bias_only.csv')

    elif args.phase == 'donor-ablations':
        print("Note: donor-ablations run automatically inside the 'main' phase via in-memory cached aligner states.")
        print("They are not separately invocable (states are not persisted to disk) - use 'main' instead.")
        sys.exit(1)

    else:
        raise NotImplementedError(f"Phase {args.phase} not yet implemented.")


if __name__ == '__main__':
    main()
