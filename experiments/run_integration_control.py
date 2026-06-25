"""R4 evidence-accumulation control.

Conceptual split (do NOT mix in writeups):

DIRECT R4 ANSWER - pure no-loop integration of the SAME network's outputs:
- E1 (Baseline weights, recurrence disabled at evaluation): probmean / logprod
  ensemble of the network's own per-step outputs. No additional training.
  Comparator: e1_gain_logprod  vs bl_gain  (within-network)

SUPERVISED SEQUENCE UPPER BOUNDS - what a separately-supervised
sequence model can do at the same parameter budget:
- E2 (Group D' = 325-param skip-FF) + LogisticRegressionCV combiner
  trained on 1000 additional labels. Late-fusion ceiling.
- E3 (concat-FF, 326 params, ReLU) trained end-to-end on VN. Early-fusion ceiling.
  Comparator: bl_acc_t3 vs (e2_acc_combiner, e3_acc_t3)

R5b - C2_NormMatched (per-trial L2 rescale, +1e-8 epsilon, before tanh gate)
R5a - cosine divergence of self vs raw clone AND of self vs the actually-
      injected norm-matched clone, both on tanh(y/tau) @ W_rec (paper-compatible)

Output: results/integration_control.csv             (per-seed numbers)
        results/integration_control_summary.md      (auto-generated summary)
        results/integration_control_summary.json    (machine-readable)

The frozen primary-results CSVs are unaffected; this script writes only NEW
files in `results/`.

Terminology discipline (use exactly these terms in writeups):
- gain                    = acc_t3 - acc_t1
- terminal accuracy       = acc_t3 (single number)
- integrated accuracy     = ensemble / logprod / combiner output accuracy
- residual gain           = bl_gain - e1_gain (within-network ensemble residual)
- terminal gap            = bl_acc_t3 - integrator_acc (across-model gap)
- fraction                = e1_gain.mean() / bl_gain.mean() (ratio of means)

Dependencies (not in repo's base requirements):
- numpy           (already in repo)
- pandas          (CSV summary)
- scikit-learn    (LogisticRegressionCV combiner for E2)
See `requirements_revision.txt`.

CLI: see `python run_integration_control.py --help`.
"""

import os, sys

# Force single-threaded BLAS so multiprocessing workers do not oversubscribe
# the CPU. Must be set BEFORE importing numpy / scikit-learn / scipy.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.network import RecurrentMLP
from src.training import (
    generate_data_variable_noise, train_vn,
)
from src.metrics import compute_all_metrics_vn, wilcoxon_exact

# ===========================================================
# Defaults (overridable via CLI; match run_n20_full.py)
# ===========================================================
DEFAULT_N_MODELS = 20
DEFAULT_N_TRAIN = 200
DEFAULT_N_TEST = 200
DEFAULT_N_CALIB = 1000  # held-out set for logistic combiner training
DEFAULT_T = 3
DEFAULT_NOISE = 0.5
DEFAULT_TRAIN_EPOCHS = 1000
DEFAULT_TRAIN_LR = 0.01     # for RecurrentMLP / Group D' (3-step BPTT or recurrence-disabled)
DEFAULT_E3_LR = 0.1         # ConcatFF needs higher LR (no BPTT; verified by sweep)
DEFAULT_DONOR_SEED_OFFSET = 100
DEFAULT_CALIB_SEED_OFFSET = 800  # disjoint from train (seed_model) and test (seed_model+500)
DEFAULT_WORKERS = 1         # default: deterministic single-worker for reproducibility.
                            # Multiprocessing on Windows can hit paging-file limits;
                            # use --workers > 1 only when the platform allows.
DEFAULT_OUT_CSV = 'results/integration_control.csv'
DEFAULT_OUT_SUMMARY = 'results/integration_control_summary.md'
DEFAULT_OUT_JSON = 'results/integration_control_summary.json'

# Backward-compat module-level constants (used by run_seed when called from
# tests/imports without going through main()).  main() overrides these from CLI.
N_MODELS = DEFAULT_N_MODELS
N_TRAIN = DEFAULT_N_TRAIN
N_TEST = DEFAULT_N_TEST
N_CALIB = DEFAULT_N_CALIB
T = DEFAULT_T
NOISE = DEFAULT_NOISE
TRAIN_EPOCHS = DEFAULT_TRAIN_EPOCHS
TRAIN_LR = DEFAULT_TRAIN_LR
E3_LR = DEFAULT_E3_LR
E3_EPOCHS = 500  # E3 ceiling control: sweep-tuned at (lr=0.1, 500ep); kept fixed
                 # while the recurrent/compared-model budget moves to 1000.
DONOR_SEED_OFFSET = DEFAULT_DONOR_SEED_OFFSET
CALIB_SEED_OFFSET = DEFAULT_CALIB_SEED_OFFSET


# ===========================================================
# Numerical helpers
# ===========================================================

def softmax_arr(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax_arr(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def _cos_divergence_safe(u, v, norm_floor=1e-12):
    """Cosine divergence (1 - cosine similarity) with degenerate-norm safety.

    - If either vector has near-zero norm, return 0.0 (treated as perfectly
      aligned by convention; the result is uninformative either way).
    - Clip the cosine into [-1, 1] before subtraction so floating-point
      round-off cannot push divergence outside [0, 2].
    """
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < norm_floor or nv < norm_floor:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


# ===========================================================
# E3: Concat-FF integrator (326 params, ReLU)
# ===========================================================

class ConcatFF:
    """Early-fusion concat-FF: Linear(30->7)->ReLU->Linear(7->8)->ReLU->Linear(8->5).
    Params: 30*7 + 7 + 7*8 + 8 + 8*5 + 5 = 326.
    """
    def __init__(self, seed=0, input_size=30, h1=7, h2=8, output_size=5):
        rng = np.random.RandomState(seed)
        # He init (matches RecurrentMLP convention)
        self.W1 = rng.randn(input_size, h1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(h1)
        self.W2 = rng.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.randn(h2, output_size) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(output_size)

    def forward_batch(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(0, Z2)
        Z3 = A2 @ self.W3 + self.b3
        return Z3, (Z1, A1, Z2, A2)

    def count_params(self):
        return (self.W1.size + self.b1.size + self.W2.size + self.b2.size
                + self.W3.size + self.b3.size)


def train_concat_ff(net, X, y, epochs=500, lr=0.01):
    """Full-batch SGD on ConcatFF, cross-entropy loss."""
    n = len(X)
    for _ in range(epochs):
        Z3, (Z1, A1, Z2, A2) = net.forward_batch(X)
        P = softmax_arr(Z3, axis=1)
        d_Z3 = (P - y) / n  # (n, 5)
        d_W3 = A2.T @ d_Z3
        d_b3 = d_Z3.sum(axis=0)
        d_A2 = d_Z3 @ net.W3.T
        d_Z2 = d_A2 * (Z2 > 0)
        d_W2 = A1.T @ d_Z2
        d_b2 = d_Z2.sum(axis=0)
        d_A1 = d_Z2 @ net.W2.T
        d_Z1 = d_A1 * (Z1 > 0)
        d_W1 = X.T @ d_Z1
        d_b1 = d_Z1.sum(axis=0)
        net.W3 -= lr * d_W3; net.b3 -= lr * d_b3
        net.W2 -= lr * d_W2; net.b2 -= lr * d_b2
        net.W1 -= lr * d_W1; net.b1 -= lr * d_b1


def eval_concat_ff(net, X, y):
    """Return (acc, logits matrix)."""
    Z3, _ = net.forward_batch(X)
    preds = np.argmax(Z3, axis=1)
    truth = np.argmax(y, axis=1)
    return float((preds == truth).mean()), Z3


# ===========================================================
# Ensemble inference for RecurrentMLP (E1, E2, Baseline)
# ===========================================================

def run_ensemble(net, X_seq, y, recurrent_on, n_classes=5):
    """Run forward_sequence_vn over X_seq.

    Returns dict: acc_t1, acc_t3, acc_probmean, acc_logprod, logits (n, T, k).
    """
    if recurrent_on:
        net.enable_recurrent_loop()
    else:
        net.disable_recurrent_loop()

    n = len(X_seq)
    correct_t1 = correct_t3 = correct_pm = correct_lp = 0
    all_logits = np.zeros((n, T, n_classes))

    for i in range(n):
        outputs, _ = net.forward_sequence_vn(X_seq[i], T=T)
        z = np.stack(outputs)  # (T, 5)
        all_logits[i] = z
        true_cls = int(np.argmax(y[i]))
        if int(np.argmax(z[0])) == true_cls: correct_t1 += 1
        if int(np.argmax(z[T-1])) == true_cls: correct_t3 += 1
        # probmean
        p = softmax_arr(z, axis=1)
        if int(np.argmax(p.mean(axis=0))) == true_cls: correct_pm += 1
        # logprod
        ell = log_softmax_arr(z, axis=1)
        if int(np.argmax(ell.sum(axis=0))) == true_cls: correct_lp += 1

    return {
        'acc_t1': correct_t1 / n,
        'acc_t3': correct_t3 / n,
        'acc_probmean': correct_pm / n,
        'acc_logprod': correct_lp / n,
        'logits': all_logits,
    }


# ===========================================================
# R5b: C2_NormMatched (per-trial L2 rescale, before tanh gate)
# ===========================================================

def forward_with_normmatched_clone_vn(target_net, clone_net, x_seq, T=3):
    """Like forward_sequence_with_clone_vn but L2-rescale clone_y_{t-1} per trial
    to match the L2 norm of target's own y_{t-1} (computed in a self-feedback pass).

    The rescaled clone replaces target._prev_output BEFORE target.forward() is
    called at t (so it enters the tanh(prev/2) gate at the next step).

    Returns (target_outs, clone_outs, self_outs), each a list of T arrays.
    """
    x_seq = np.asarray(x_seq, dtype=np.float64)
    n_classes = target_net.output_size

    # Pass 1: collect target's self-feedback outputs (norms reference)
    target_net.reset_state()
    self_outs = []
    for t in range(T):
        y = target_net.forward(x_seq[t])
        self_outs.append(y.copy())
    self_norms = [float(np.linalg.norm(o)) for o in self_outs]

    # Pass 2: rerun target with norm-matched clone feedback
    target_net.reset_state()
    clone_net.reset_state()
    clone_outs = []
    target_outs = []
    for t in range(T):
        clone_y = clone_net.forward(x_seq[t])
        clone_outs.append(clone_y.copy())
        if t > 0:
            cn = float(np.linalg.norm(clone_outs[t - 1]))
            scale = self_norms[t - 1] / (cn + 1e-8)
            target_net._prev_output = clone_outs[t - 1] * scale
            target_net._has_feedback = True
        target_y = target_net.forward(x_seq[t])
        target_outs.append(target_y.copy())
    return target_outs, clone_outs, self_outs


def metrics_normmatched(target_net, clone_net, X_seq, y):
    """Per-seed NormMatched C2 metrics + cosine divergence stats.

    Cosine divergence is computed on the W_rec contribution:
        contrib = tanh(y / tau) @ W_rec
    matching paper Figure 4 / `run_divergence_null.py` definition.

    We report THREE divergence quantities:
    (a) self_contrib  vs raw clone_contrib              -- paper-compat (standard C2)
    (b) self_contrib  vs norm-matched clone_contrib     -- the actually-injected
                                                          feedback under R5b. Because
                                                          tanh is nonlinear, scaling
                                                          y_clone changes both the
                                                          magnitude and the direction
                                                          of tanh(y_clone/tau) @ W_rec;
                                                          this quantity captures how
                                                          angular separation actually
                                                          shifts under the magnitude
                                                          intervention.
    (c) raw logit-level cosine (diagnostic only)
    """
    n = len(X_seq)
    tau = float(target_net.feedback_tau)
    W_rec = target_net.W_rec
    correct_t1 = correct_t3 = 0
    cos_divs_raw_clone = []     # (a) self vs raw clone (paper-compat)
    cos_divs_normmatched = []   # (b) self vs ACTUAL norm-matched clone contribution
    cos_divs_raw_logit = []     # (c) raw logit-level (diagnostic)
    for i in range(n):
        target_outs, clone_outs, self_outs = forward_with_normmatched_clone_vn(
            target_net, clone_net, X_seq[i], T=T)
        true_cls = int(np.argmax(y[i]))
        if int(np.argmax(target_outs[0])) == true_cls: correct_t1 += 1
        if int(np.argmax(target_outs[T-1])) == true_cls: correct_t3 += 1
        # cosine divergence at t=1 prior (used as feedback at t=2)
        for t in range(T - 1):
            so = self_outs[t]
            co_raw = clone_outs[t]
            # Norm matching used at injection time:
            self_norm = float(np.linalg.norm(so))
            clone_norm = float(np.linalg.norm(co_raw))
            scale = self_norm / (clone_norm + 1e-8)
            co_nm = co_raw * scale  # actually-injected feedback under R5b

            # (a) Paper-compatible: self vs raw clone contribution
            self_contrib   = np.tanh(so       / tau) @ W_rec
            clone_contrib  = np.tanh(co_raw   / tau) @ W_rec
            cos_divs_raw_clone.append(_cos_divergence_safe(self_contrib, clone_contrib))

            # (b) Norm-matched contribution (the actually-injected feedback)
            normmatched_contrib = np.tanh(co_nm / tau) @ W_rec
            cos_divs_normmatched.append(_cos_divergence_safe(self_contrib, normmatched_contrib))

            # (c) Raw logit cosine (diagnostic)
            cos_divs_raw_logit.append(_cos_divergence_safe(so, co_raw))
    return {
        'acc_t1': correct_t1 / n,
        'acc_t3': correct_t3 / n,
        'gain': (correct_t3 - correct_t1) / n,
        'cos_div_clone_self_mean':    float(np.mean(cos_divs_raw_clone)),    # (a) paper-compat
        'cos_div_normmatched_mean':   float(np.mean(cos_divs_normmatched)),  # (b) actually injected
        'cos_div_raw_mean':           float(np.mean(cos_divs_raw_logit)),    # (c) diagnostic
    }


# ===========================================================
# Per-seed worker: train + evaluate everything
# ===========================================================

def run_seed(seed_model):
    """Train Baseline (rec) + Group D' (no rec) + E3 + donor; evaluate all."""
    t0 = time.time()
    rec = {'seed': seed_model}

    # -- Training data (VN, same seeds for Baseline, D', E3) --
    X_sq_tr, y_tr = generate_data_variable_noise(
        N_TRAIN, NOISE, T=T, seed=seed_model)

    # -- Test data (shared) --
    X_sq_te, y_te = generate_data_variable_noise(
        N_TEST, NOISE, T=T, seed=seed_model + 500)

    # -- Calibration data for logistic combiner (disjoint) --
    X_sq_ca, y_ca = generate_data_variable_noise(
        N_CALIB, NOISE, T=T, seed=seed_model + CALIB_SEED_OFFSET)

    # -------------------------------------------------
    # Baseline (recurrent VN)
    # -------------------------------------------------
    target_vn = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    train_vn(target_vn, X_sq_tr, y_tr, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    bl_res = run_ensemble(target_vn, X_sq_te, y_te, recurrent_on=True)
    rec['bl_acc_t1'] = bl_res['acc_t1']
    rec['bl_acc_t3'] = bl_res['acc_t3']
    rec['bl_gain'] = bl_res['acc_t3'] - bl_res['acc_t1']

    # -------------------------------------------------
    # E1 sanity: same Baseline weights, recurrence DISABLED at eval
    # -------------------------------------------------
    e1_res = run_ensemble(target_vn, X_sq_te, y_te, recurrent_on=False)
    rec['e1_acc_t1']       = e1_res['acc_t1']
    rec['e1_acc_t3']       = e1_res['acc_t3']
    rec['e1_acc_probmean'] = e1_res['acc_probmean']
    rec['e1_acc_logprod']  = e1_res['acc_logprod']

    # restore recurrent state for any downstream
    target_vn.enable_recurrent_loop()

    # -------------------------------------------------
    # E2: Group D' (RecurrentMLP + skip, recurrence-disabled, trained VN)
    # -------------------------------------------------
    net_dp = RecurrentMLP(10, 10, 10, 5, seed=seed_model, skip_connection=True)
    net_dp.disable_recurrent_loop()
    train_vn(net_dp, X_sq_tr, y_tr, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    # Recurrence stays disabled for evaluation
    e2_res = run_ensemble(net_dp, X_sq_te, y_te, recurrent_on=False)
    rec['e2_acc_t1']       = e2_res['acc_t1']
    rec['e2_acc_t3']       = e2_res['acc_t3']
    rec['e2_acc_probmean'] = e2_res['acc_probmean']
    rec['e2_acc_logprod']  = e2_res['acc_logprod']

    # E2 with logistic-regression late-fusion combiner
    # Train on calibration set, evaluate on test set
    e2_calib = run_ensemble(net_dp, X_sq_ca, y_ca, recurrent_on=False)
    Z_ca = e2_calib['logits'].reshape(N_CALIB, -1)  # (N_CALIB, 15)
    y_ca_idx = np.argmax(y_ca, axis=1)
    Z_te = e2_res['logits'].reshape(N_TEST, -1)
    y_te_idx = np.argmax(y_te, axis=1)

    if os.environ.get('INTEGRATION_CONTROL_SKIP_COMBINER') == '1':
        rec['e2_acc_combiner'] = float('nan')
    else:
        try:
            from sklearn.linear_model import LogisticRegressionCV
            from sklearn.model_selection import StratifiedKFold
            # Shuffled CV folds. With cv=int the default StratifiedKFold is
            # NOT shuffled; if calibration data happens to be class-grouped
            # this yields severely imbalanced folds.
            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True,
                                          random_state=seed_model)
            lr_combiner = LogisticRegressionCV(
                cv=cv_splitter, max_iter=2000, n_jobs=1,
                random_state=seed_model)
            lr_combiner.fit(Z_ca, y_ca_idx)
            preds = lr_combiner.predict(Z_te)
            rec['e2_acc_combiner'] = float((preds == y_te_idx).mean())
        except Exception as ex:
            rec['e2_acc_combiner'] = float('nan')
            rec['e2_combiner_error'] = str(ex)[:200]

    # -------------------------------------------------
    # E3: Concat-FF (326 params, ReLU), trained from scratch on VN
    # Same data seeds as Baseline; reshape (B, 3, 10) -> (B, 30)
    # -------------------------------------------------
    e3 = ConcatFF(seed=seed_model)
    assert e3.count_params() == 326, f"E3 param count {e3.count_params()} != 326"
    X_tr_flat = X_sq_tr.reshape(N_TRAIN, T * 10)  # (200, 30)
    X_te_flat = X_sq_te.reshape(N_TEST, T * 10)
    # E3 is a sweep-tuned ceiling control (lr=0.1); keep its 500-epoch budget
    # rather than the model's 1000 (at lr=0.1 a 326-param FF converges by ~100ep
    # and 1000 would over-train it, distorting the early-fusion ceiling).
    train_concat_ff(e3, X_tr_flat, y_tr, epochs=E3_EPOCHS, lr=E3_LR)
    e3_acc, e3_logits = eval_concat_ff(e3, X_te_flat, y_te)
    rec['e3_acc_t3'] = e3_acc  # only "t3" (single-pass on full sequence)
    # E3 has no acc_t1 by construction (early fusion)

    # -------------------------------------------------
    # R5b: C2_NormMatched
    # -------------------------------------------------
    donor_seed = seed_model + DONOR_SEED_OFFSET
    donor_vn = RecurrentMLP(10, 10, 10, 5, seed=donor_seed)
    X_sq_d, y_d = generate_data_variable_noise(
        N_TRAIN, NOISE, T=T, seed=donor_seed)
    train_vn(donor_vn, X_sq_d, y_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    nm_res = metrics_normmatched(target_vn, donor_vn, X_sq_te, y_te)
    rec['c2_normmatched_acc_t1']    = nm_res['acc_t1']
    rec['c2_normmatched_acc_t3']    = nm_res['acc_t3']
    rec['c2_normmatched_gain']      = nm_res['gain']
    rec['cos_div_clone_self']       = nm_res['cos_div_clone_self_mean']   # (a) self vs raw clone (paper-compat W_rec contrib)
    rec['cos_div_normmatched']      = nm_res['cos_div_normmatched_mean']  # (b) self vs ACTUALLY-INJECTED norm-matched feedback
    rec['cos_div_raw']              = nm_res['cos_div_raw_mean']          # (c) raw logit (diagnostic)

    # -- E1 / Baseline gain decomposition --
    # Direct answer to "what fraction of VN +0.189 is no-loop integration?"
    # Use E1 (same Baseline weights, recurrence off, ensemble) to compute the
    # within-network integration gain. This is the most apples-to-apples
    # version: e1_gain_logprod = e1_acc_logprod - e1_acc_t1 vs bl_gain.
    rec['e1_gain_probmean'] = rec['e1_acc_probmean'] - rec['e1_acc_t1']
    rec['e1_gain_logprod']  = rec['e1_acc_logprod']  - rec['e1_acc_t1']

    # -------------------------------------------------
    # Reference: standard C2 (no norm matching) for comparison
    # -------------------------------------------------
    from src.metrics import compute_all_metrics_with_clone_vn
    c2_std = compute_all_metrics_with_clone_vn(target_vn, donor_vn, X_sq_te, y_te)
    rec['c2_std_acc_t1'] = c2_std['acc_t1']
    rec['c2_std_acc_t3'] = c2_std['acc_t3']
    rec['c2_std_gain']   = c2_std['gain']

    rec['elapsed_s'] = time.time() - t0
    return rec


# ===========================================================
# Main
# ===========================================================

def paired_bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=42):
    """Paired bootstrap CI on mean(values).

    Resamples indices with replacement and computes mean per resample.
    Returns (lo, hi) at (alpha/2, 1-alpha/2) percentiles.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.randint(0, n, n)
        boot[k] = values[idx].mean()
    boot.sort()
    lo = boot[int((alpha / 2.0) * n_resamples)]
    hi = boot[int((1.0 - alpha / 2.0) * n_resamples)]
    return float(lo), float(hi)


def paired_bootstrap_ratio_ci(num, den, n_resamples=10000, alpha=0.05, seed=42):
    """Bootstrap CI on the ratio of means: mean(num) / mean(den).

    Pairs are preserved by resampling a single index array.
    """
    rng = np.random.RandomState(seed)
    n = len(num)
    boot = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.randint(0, n, n)
        d = den[idx].mean()
        boot[k] = num[idx].mean() / d if d != 0 else float('nan')
    boot = boot[np.isfinite(boot)]
    boot.sort()
    n_eff = len(boot)
    lo = boot[int((alpha / 2.0) * n_eff)]
    hi = boot[int((1.0 - alpha / 2.0) * n_eff)]
    return float(lo), float(hi)


def run_seeds(seed_list, n_workers, run_fn=run_seed):
    """Run each seed via worker pool (or sequentially if n_workers==1).

    Sequential execution is the default - deterministic, no cross-process
    import races, and stable on Windows where parallel multiprocessing can
    hit paging-file limits.
    """
    records = []
    if n_workers <= 1:
        for s in seed_list:
            print(f"  [seed={s:2d}] starting...")
            try:
                rec = run_fn(s)
                records.append(rec)
                print(f"  [seed={s:2d}] elapsed={rec['elapsed_s']:.1f}s "
                      f"bl_t3={rec['bl_acc_t3']:.3f} "
                      f"e3_t3={rec['e3_acc_t3']:.3f} "
                      f"e2_combiner={rec.get('e2_acc_combiner', float('nan')):.3f} "
                      f"c2nm_gain={rec['c2_normmatched_gain']:+.3f}")
            except Exception as e:
                print(f"  [seed={s}] FAILED: {e}")
                import traceback
                traceback.print_exc()
        return records

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_fn, s): s for s in seed_list}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                rec = fut.result()
                records.append(rec)
                print(f"  [seed={s:2d}] elapsed={rec['elapsed_s']:.1f}s "
                      f"bl_t3={rec['bl_acc_t3']:.3f} "
                      f"e3_t3={rec['e3_acc_t3']:.3f} "
                      f"e2_combiner={rec.get('e2_acc_combiner', float('nan')):.3f} "
                      f"c2nm_gain={rec['c2_normmatched_gain']:+.3f}")
            except Exception as e:
                print(f"  [seed={s}] FAILED: {e}")
                import traceback
                traceback.print_exc()
    return records


def sanity_check(df, n_models_expected, baseline_gain_floor=0.05,
                 paper_reproduction_min_n=10):
    """Sanity asserts to defend against silent regressions.

    Verifies:
      - row count matches the requested seed count
      - no NaN in primary columns
      - E3 param count is locked at 326 (re-instantiate to verify)
      - QUALITATIVE (reset-robust): VN baseline gain is clearly positive, and the
        C2 clone gain is below the self-feedback baseline. Exact-value asserts
        (the old unnormalized@500 reference values) were removed under the standardized protocol
        because the operating point moved to normalized@1000; sign/ordering checks
        survive that and still catch gross regressions.
    """
    assert len(df) == n_models_expected, \
        f"sanity: row count {len(df)} != {n_models_expected}"

    primary_cols = ['bl_acc_t1', 'bl_acc_t3', 'bl_gain',
                    'e1_acc_logprod', 'e1_gain_logprod',
                    'e3_acc_t3', 'c2_std_gain', 'c2_normmatched_gain',
                    'cos_div_clone_self', 'cos_div_normmatched']
    for c in primary_cols:
        if c in df.columns:
            assert not df[c].isna().any(), f"sanity: NaN in primary column {c}"

    e3 = ConcatFF(seed=0)
    assert e3.count_params() == 326, \
        f"sanity: ConcatFF param count {e3.count_params()} != 326"

    print(f"  [sanity] row count = {len(df)}                                 OK")
    print(f"  [sanity] no NaN in {len(primary_cols)} primary columns                OK")
    print(f"  [sanity] ConcatFF.count_params() == 326                          OK")

    if len(df) < paper_reproduction_min_n:
        print(f"  [sanity] paper reproduction checks SKIPPED (N={len(df)} < {paper_reproduction_min_n})")
        return

    bl_g = float(df['bl_gain'].mean())
    c2_g = float(df['c2_std_gain'].mean())
    assert bl_g > baseline_gain_floor, \
        (f"sanity: VN baseline gain mean {bl_g:+.4f} is not clearly positive "
         f"(floor {baseline_gain_floor}) -- recurrent self-correction collapsed?")
    assert c2_g < bl_g, \
        (f"sanity: C2 clone gain {c2_g:+.4f} is not below the self-feedback "
         f"baseline {bl_g:+.4f} -- clone-feedback harm absent?")

    print(f"  [sanity] VN baseline gain {bl_g:+.4f} > floor {baseline_gain_floor}      OK")
    print(f"  [sanity] C2 clone gain {c2_g:+.4f} < baseline {bl_g:+.4f}            OK")


def write_summary(df, out_md, out_json):
    """Write a machine- and human-readable summary alongside the per-seed CSV.

    The MD file is the human-readable summary; the JSON file is consumed by
    downstream scripts (figure generation etc.) so they don't have to re-derive
    point estimates.
    """
    bl_t3 = df['bl_acc_t3'].astype(float).values
    bl_g  = df['bl_gain'].astype(float).values

    summary = {
        'n_models': int(len(df)),
        'baseline': {
            'acc_t1_mean': float(df['bl_acc_t1'].mean()),
            'acc_t3_mean': float(bl_t3.mean()),
            'gain_mean':   float(bl_g.mean()),
            'gain_sd':     float(df['bl_gain'].std(ddof=0)),
        },
    }

    # -- E1 within-network ensemble vs Baseline gain --
    r4 = {}
    for agg, gcol in [('logprod', 'e1_gain_logprod'),
                      ('probmean', 'e1_gain_probmean')]:
        if gcol not in df.columns:
            continue
        g = df[gcol].astype(float).values
        residual = bl_g - g
        try:
            T_w, p_w = wilcoxon_exact(bl_g, g)
        except Exception:
            T_w, p_w = float('nan'), float('nan')
        res_lo, res_hi = paired_bootstrap_ci(residual)
        ratio_lo, ratio_hi = paired_bootstrap_ratio_ci(g, bl_g)
        r4[agg] = {
            'e1_gain_mean':       float(g.mean()),
            'residual_mean':      float(residual.mean()),
            'residual_ci_95':     [res_lo, res_hi],
            'ratio_e1_over_bl':   float(g.mean() / bl_g.mean()),
            'ratio_ci_95':        [ratio_lo, ratio_hi],
            'paired_wilcoxon_T':  float(T_w),
            'paired_wilcoxon_p':  float(p_w),
        }
    summary['r4_within_network_ensemble'] = r4

    # -- Sequence upper bounds: E2 combiner, E3 (terminal accuracy) --
    upper = {}
    holm_raw_ps = {}
    for label, col in [('e3', 'e3_acc_t3'),
                       ('e2_combiner', 'e2_acc_combiner')]:
        if col not in df.columns:
            continue
        v_full = df[col].astype(float).values
        # Filter NaN seeds for this single comparison instead of dropping the
        # entire test (e.g., when LogisticRegressionCV fails on one seed).
        valid = np.isfinite(v_full)
        if valid.sum() < 2:
            upper[label] = {'note': 'too few valid seeds', 'n_valid': int(valid.sum())}
            continue
        v   = v_full[valid]
        bl  = bl_t3[valid]
        delta = bl - v
        try:
            T_w, p_w = wilcoxon_exact(bl, v)
        except Exception:
            T_w, p_w = float('nan'), float('nan')
        d_lo, d_hi = paired_bootstrap_ci(delta)
        upper[label] = {
            'n_valid':                int(valid.sum()),
            'integrator_acc_t3_mean': float(v.mean()),
            'terminal_gap_mean':      float(delta.mean()),
            'terminal_gap_ci_95':     [d_lo, d_hi],
            'ratio_integrator_over_bl_t3': float(v.mean() / bl.mean()),
            'paired_wilcoxon_T':      float(T_w),
            'paired_wilcoxon_p_raw':  float(p_w),
        }
        if np.isfinite(p_w):
            holm_raw_ps[label] = float(p_w)

    # Holm m=2 over the two primary upper-bound comparisons
    if len(holm_raw_ps) == 2:
        items = sorted(holm_raw_ps.items(), key=lambda x: x[1])
        prev = 0.0
        for i, (k, p) in enumerate(items):
            corrected = min(1.0, max(prev, p * (2 - i)))
            prev = corrected
            upper[k]['paired_wilcoxon_p_holm_m2'] = corrected
    summary['supervised_upper_bounds'] = upper

    # -- Magnitude check --
    r5 = {
        'c2_standard_gain_mean':       float(df['c2_std_gain'].mean()),
        'c2_normmatched_gain_mean':    float(df['c2_normmatched_gain'].mean()),
        'cos_div_self_vs_raw_clone':   float(df['cos_div_clone_self'].mean()),
        'cos_div_self_vs_normmatched': float(df['cos_div_normmatched'].mean())
                                       if 'cos_div_normmatched' in df.columns else None,
        'cos_div_raw_logit':           float(df['cos_div_raw'].mean())
                                       if 'cos_div_raw' in df.columns else None,
    }
    nm = df['c2_normmatched_gain'].astype(float).values
    nm_lo, nm_hi = paired_bootstrap_ci(nm)
    r5['c2_normmatched_gain_ci_95'] = [nm_lo, nm_hi]
    summary['r5_magnitude_check'] = r5

    summary['notes'] = {
        'mathematical_shield': (
            "For static input, any deterministic stateless aggregator receives "
            "identical logits at every timestep, so its gain is exactly zero. "
            "The +0.036 static gain in Baseline therefore reflects dynamic "
            "trajectory evolution within the closed loop and cannot be reproduced "
            "by no-loop temporal integration in principle."
        ),
        'e1_vs_e2_e3_split': (
            "E1 (within-network ensemble of frozen Baseline outputs) is the "
            "DIRECT R4 comparator — it uses no extra training. "
            "E2 combiner (1000-label LR fit) and E3 (separate end-to-end-trained "
            "concat-FF) are SUPERVISED SEQUENCE UPPER BOUNDS, not pure no-loop "
            "integrators. Do NOT mix the two interpretations in writeups."
        ),
        'multiple_comparisons': (
            "Holm m=2 correction is applied across the two pre-specified "
            "supervised upper bounds (e3, e2_combiner). The R4 within-network "
            "comparison (E1 logprod / probmean vs Baseline gain) is reported "
            "uncorrected as it tests a single pre-specified hypothesis. "
            "Diagnostic comparisons (raw E2 logprod / probmean) are reported "
            "for transparency only and not corrected."
        ),
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    # Human-readable MD
    md = []
    md.append(f"# Integration control summary (auto-generated, N={summary['n_models']} seeds)\n")
    md.append("## Baseline reference\n")
    bl = summary['baseline']
    md.append(f"- acc_t1 = {bl['acc_t1_mean']:+.4f}, acc_t3 = {bl['acc_t3_mean']:+.4f}, "
              f"gain = {bl['gain_mean']:+.4f} (sd {bl['gain_sd']:.4f})\n")

    md.append("\n## R4 — direct within-network ensemble (E1, no extra training)\n")
    md.append("| Aggregation | E1 gain | Residual (bl−E1) | Residual CI 95% | Ratio (E1/bl) | Ratio CI 95% | Wilcoxon p |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for agg, d in r4.items():
        md.append(f"| {agg} | {d['e1_gain_mean']:+.4f} | {d['residual_mean']:+.4f} | "
                  f"[{d['residual_ci_95'][0]:+.4f}, {d['residual_ci_95'][1]:+.4f}] | "
                  f"{d['ratio_e1_over_bl']:.3f} | "
                  f"[{d['ratio_ci_95'][0]:.3f}, {d['ratio_ci_95'][1]:.3f}] | "
                  f"{d['paired_wilcoxon_p']:.5f} |\n")

    md.append("\n## Supervised sequence upper bounds (E2 combiner, E3) — terminal accuracy\n")
    md.append("| Control | acc_t3 | Terminal gap (bl−int) | Gap CI 95% | Ratio | Wilcoxon p (Holm m=2) |\n")
    md.append("|---|---|---|---|---|---|\n")
    for label, d in upper.items():
        ph = d.get('paired_wilcoxon_p_holm_m2', d['paired_wilcoxon_p_raw'])
        md.append(f"| {label} | {d['integrator_acc_t3_mean']:.4f} | {d['terminal_gap_mean']:+.4f} | "
                  f"[{d['terminal_gap_ci_95'][0]:+.4f}, {d['terminal_gap_ci_95'][1]:+.4f}] | "
                  f"{d['ratio_integrator_over_bl_t3']:.3f} | {ph:.6f} |\n")

    md.append("\n## Magnitude check\n")
    md.append(f"- C2 standard gain mean = {r5['c2_standard_gain_mean']:+.4f}\n")
    md.append(f"- C2 NormMatched gain mean = {r5['c2_normmatched_gain_mean']:+.4f}, "
              f"95% CI = [{r5['c2_normmatched_gain_ci_95'][0]:+.4f}, {r5['c2_normmatched_gain_ci_95'][1]:+.4f}]\n")
    md.append(f"- Cosine divergence (self vs raw clone, on tanh(y/τ) @ W_rec) = "
              f"{r5['cos_div_self_vs_raw_clone']:.4f}\n")
    if r5.get('cos_div_self_vs_normmatched') is not None:
        md.append(f"- Cosine divergence (self vs **actually-injected** norm-matched clone) = "
                  f"{r5['cos_div_self_vs_normmatched']:.4f}\n")

    md.append("\n## Notes\n")
    md.append(f"- Mathematical shield: {summary['notes']['mathematical_shield']}\n")
    md.append(f"- Interpretation discipline: {summary['notes']['e1_vs_e2_e3_split']}\n")

    with open(out_md, 'w', encoding='utf-8') as f:
        f.writelines(md)
    print(f"Wrote {out_md}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-models', type=int, default=DEFAULT_N_MODELS,
                        help=f'number of seeds (default {DEFAULT_N_MODELS})')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'parallel workers; 1 = sequential/deterministic (default {DEFAULT_WORKERS})')
    parser.add_argument('--noise', type=float, default=DEFAULT_NOISE,
                        help=f'noise level σ (default {DEFAULT_NOISE})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAIN_EPOCHS,
                        help=f'training epochs (default {DEFAULT_TRAIN_EPOCHS})')
    parser.add_argument('--out-csv', default=DEFAULT_OUT_CSV)
    parser.add_argument('--out-summary-md', default=DEFAULT_OUT_SUMMARY)
    parser.add_argument('--out-summary-json', default=DEFAULT_OUT_JSON)
    parser.add_argument('--skip-combiner', action='store_true',
                        help='skip the LogisticRegressionCV combiner (no sklearn dependency)')
    args = parser.parse_args()

    # Override module-level constants used by run_seed
    global N_MODELS, NOISE, TRAIN_EPOCHS
    N_MODELS = args.n_models
    NOISE = args.noise
    TRAIN_EPOCHS = args.epochs

    if args.skip_combiner:
        # Patch run_seed at runtime to avoid sklearn import.
        os.environ['INTEGRATION_CONTROL_SKIP_COMBINER'] = '1'

    print(f"[INTEGRATION_CONTROL] N_MODELS={args.n_models} NOISE={args.noise} "
          f"EPOCHS={args.epochs} WORKERS={args.workers}")
    t0 = time.time()
    seeds = list(range(args.n_models))
    records = run_seeds(seeds, args.workers)

    df = pd.DataFrame(records).sort_values('seed').reset_index(drop=True)
    for path in (args.out_csv, args.out_summary_md, args.out_summary_json):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    print(f"Total elapsed: {time.time() - t0:.1f}s ({(time.time()-t0)/60:.1f}m)")

    # -- Sanity checks (defends against silent regressions) --
    print("\nSanity checks:")
    sanity_check(df, n_models_expected=args.n_models)

    # -- Console summary --
    print("\n" + "=" * 70)
    print("SUMMARY (mean over seeds, ddof=0)")
    print("=" * 70)
    cols = ['bl_acc_t1', 'bl_acc_t3', 'bl_gain',
            'e1_gain_probmean', 'e1_gain_logprod',
            'e2_acc_combiner', 'e3_acc_t3',
            'c2_std_gain', 'c2_normmatched_gain',
            'cos_div_clone_self', 'cos_div_normmatched']
    for c in cols:
        if c in df.columns:
            v = df[c].astype(float)
            print(f"  {c:25s}: mean={v.mean():+.4f}  sd={v.std(ddof=0):.4f}")

    # -- Auto-generated MD + JSON summary --
    summary = write_summary(df, args.out_summary_md, args.out_summary_json)

    # Quick console echo of headline result (for log readability)
    if 'logprod' in summary['r4_within_network_ensemble']:
        d = summary['r4_within_network_ensemble']['logprod']
        print(f"\n[R4 HEADLINE] bl_gain={summary['baseline']['gain_mean']:+.4f}, "
              f"e1_logprod_gain={d['e1_gain_mean']:+.4f}, ratio={d['ratio_e1_over_bl']:.3f}, "
              f"residual CI 95% = [{d['residual_ci_95'][0]:+.4f}, {d['residual_ci_95'][1]:+.4f}], "
              f"p_paired={d['paired_wilcoxon_p']:.5f}")


if __name__ == '__main__':
    main()
