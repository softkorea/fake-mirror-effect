"""Phase 2: C2 + VN + alignment (N=20).

Output: results/n20_c2_vn_alignment.csv - authoritative source for VN primary
results (Baseline, A, C1, C2, C2-affine, C2-mlp) used in Table 1 Panel B.
Static C2 also included for cross-reference.

Internal phases:
Phase 1: Static ablation (Baseline, A, B1, B2, C1, D, D', D'')
Phase 2: C2 clone feedback (static + VN)
Phase 3: VN ablation (Baseline, A, C1 under VN)
Phase 4: Affine + MLP alignment controls

Each internal phase above parallelizes its own work across (seed, noise_level) or
(seed,) worker units (this is within-script worker parallelism; the top-level
run_all.py pipeline phases themselves run sequentially).
Workers = max(1, cpu_count() - 4).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import csv
import time
import math
import multiprocessing as mp
from collections import defaultdict

from src.network import RecurrentMLP, DeepFeedforwardMLP
from src.training import (
    generate_data, generate_data_variable_noise,
    train, train_vn, train_deep_ff, evaluate_accuracy_deep_ff,
)
from src.ablation import (
    ablate_recurrent, ablate_random, ablate_structural,
    deep_copy_weights, restore_weights,
    forward_sequence_with_clone, forward_sequence_with_clone_vn,
    fit_learned_affine, fit_learned_affine_vn,
    forward_sequence_with_learned_affine_clone,
    forward_sequence_with_learned_affine_clone_vn,
)
from src.metrics import (
    compute_all_metrics, compute_all_metrics_vn,
    compute_all_metrics_with_clone, compute_all_metrics_with_clone_vn,
    wilcoxon_exact,
)

# ----------------------------------------------
# Configuration
# ----------------------------------------------

N_MODELS = 20
N_RANDOM_ABLATIONS = 30
N_SCRAMBLE_SEEDS = 30
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_EPOCHS = 1000
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3
N_REC_WEIGHTS = 50
DONOR_SEED_OFFSET = 100
MLP_HIDDEN = 16
MLP_EPOCHS = 500  # aligner-fitting budget (control hyperparameter, NOT model
                 # training); kept at 500 like the other aligner budgets even as
                 # model training moves to 1000 under the standardized protocol.
MLP_LR = 0.01


def get_n_workers():
    return max(1, mp.cpu_count() - 4)


# ==============================================
# Phase 1: Static ablation
# ==============================================

def run_static_ablation(args):
    """Single (seed, noise) static ablation."""
    seed_model, noise_level = args
    rows = []

    net = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_train, y_train = generate_data(N_TRAIN, noise_level, seed=seed_model)
    X_test, y_test = generate_data(N_TEST, noise_level, seed=seed_model + 500)
    train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    def row(group, metrics, seed_abl=0):
        return {'seed_model': seed_model, 'group': group,
                'seed_ablation': seed_abl, 'noise_level': noise_level,
                **{k: metrics[k] for k in
                   ['acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']}}

    # Baseline
    rows.append(row('Baseline', compute_all_metrics(net, X_test, y_test)))

    # Group A
    saved = deep_copy_weights(net)
    ablate_recurrent(net)
    rows.append(row('A', compute_all_metrics(net, X_test, y_test)))
    restore_weights(net, saved)

    # Group B1 (30x)
    for s in range(N_RANDOM_ABLATIONS):
        saved = deep_copy_weights(net)
        ablate_random(net, n_connections=N_REC_WEIGHTS, seed=s + 1000)
        rows.append(row('B1', compute_all_metrics(net, X_test, y_test), s + 1000))
        restore_weights(net, saved)

    # Group B2
    saved = deep_copy_weights(net)
    ablate_structural(net, layer='h2_to_output')
    rows.append(row('B2', compute_all_metrics(net, X_test, y_test)))
    restore_weights(net, saved)

    # Group C1 (30x)
    for s in range(N_SCRAMBLE_SEEDS):
        net.enable_scrambled_feedback(seed=s + 2000)
        rows.append(row('C1', compute_all_metrics(net, X_test, y_test), s + 2000))
        net.disable_scrambled_feedback()

    # Group D
    net_d = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    net_d.disable_recurrent_loop()
    train(net_d, generate_data(N_TRAIN, noise_level, seed=seed_model)[0],
          generate_data(N_TRAIN, noise_level, seed=seed_model)[1],
          epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    rows.append(row('D', compute_all_metrics(net_d, X_test, y_test)))

    # Group D'
    net_dp = RecurrentMLP(10, 10, 10, 5, seed=seed_model, skip_connection=True)
    net_dp.disable_recurrent_loop()
    train(net_dp, generate_data(N_TRAIN, noise_level, seed=seed_model)[0],
          generate_data(N_TRAIN, noise_level, seed=seed_model)[1],
          epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    rows.append(row("D'", compute_all_metrics(net_dp, X_test, y_test)))

    # Group D''
    net_dpp = DeepFeedforwardMLP(10, 10, 6, 5, seed=seed_model)
    train_deep_ff(net_dpp,
                  generate_data(N_TRAIN, noise_level, seed=seed_model)[0],
                  generate_data(N_TRAIN, noise_level, seed=seed_model)[1],
                  epochs=TRAIN_EPOCHS, lr=TRAIN_LR)
    # D'' is stateless: same static input at every timestep produces
    # identical output, so gain is mathematically zero. Evaluate t1=t3
    # explicitly to avoid hardcoding.
    acc = evaluate_accuracy_deep_ff(net_dpp, X_test, y_test)
    rows.append({'seed_model': seed_model, 'group': "D''", 'seed_ablation': 0,
                 'noise_level': noise_level, 'acc_t1': acc, 'acc_t2': acc,
                 'acc_t3': acc, 'gain': acc - acc, 'ece': '',
                 'r_norm': '', 'delta_norm': ''})

    return rows


# ==============================================
# Phase 2: C2 + VN (combined)
# ==============================================

def run_c2_and_vn(seed_model):
    """Train target+donor (static+VN), evaluate C2 + VN conditions."""
    rows = []
    noise = 0.5  # Primary noise level for C2/VN

    # -- Static models --
    target_st = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_tr_st, y_tr_st = generate_data(N_TRAIN, noise, seed=seed_model)
    X_te_st, y_te_st = generate_data(N_TEST, noise, seed=seed_model + 500)
    train(target_st, X_tr_st, y_tr_st, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    donor_seed = seed_model + DONOR_SEED_OFFSET
    donor_st = RecurrentMLP(10, 10, 10, 5, seed=donor_seed)
    train(donor_st, generate_data(N_TRAIN, noise, seed=donor_seed)[0],
          generate_data(N_TRAIN, noise, seed=donor_seed)[1],
          epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    # C2 static
    c2_st = compute_all_metrics_with_clone(target_st, donor_st, X_te_st, y_te_st)
    rows.append({'seed_model': seed_model, 'setting': 'static', 'group': 'C2',
                 'noise_level': noise, 'gain': c2_st['gain'],
                 'acc_t1': c2_st['acc_t1'], 'acc_t3': c2_st['acc_t3']})

    # Baseline static (for reference)
    bl_st = compute_all_metrics(target_st, X_te_st, y_te_st)
    rows.append({'seed_model': seed_model, 'setting': 'static', 'group': 'Baseline',
                 'noise_level': noise, 'gain': bl_st['gain'],
                 'acc_t1': bl_st['acc_t1'], 'acc_t3': bl_st['acc_t3']})

    # -- VN models --
    target_vn = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_sq_tr, y_tr_vn = generate_data_variable_noise(N_TRAIN, noise, T=T, seed=seed_model)
    train_vn(target_vn, X_sq_tr, y_tr_vn, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    donor_vn = RecurrentMLP(10, 10, 10, 5, seed=donor_seed)
    X_sq_d, y_d_vn = generate_data_variable_noise(N_TRAIN, noise, T=T, seed=donor_seed)
    train_vn(donor_vn, X_sq_d, y_d_vn, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    # Shared test data
    X_sq_te, y_te_vn = generate_data_variable_noise(N_TEST, noise, T=T, seed=seed_model + 500)

    # Baseline VN
    bl_vn = compute_all_metrics_vn(target_vn, X_sq_te, y_te_vn)
    rows.append({'seed_model': seed_model, 'setting': 'vn', 'group': 'Baseline',
                 'noise_level': noise, 'gain': bl_vn['gain'],
                 'acc_t1': bl_vn['acc_t1'], 'acc_t3': bl_vn['acc_t3']})

    # Group A VN
    saved = deep_copy_weights(target_vn)
    ablate_recurrent(target_vn)
    a_vn = compute_all_metrics_vn(target_vn, X_sq_te, y_te_vn)
    rows.append({'seed_model': seed_model, 'setting': 'vn', 'group': 'A',
                 'noise_level': noise, 'gain': a_vn['gain'],
                 'acc_t1': a_vn['acc_t1'], 'acc_t3': a_vn['acc_t3']})
    restore_weights(target_vn, saved)

    # C1 VN (30 scramble seeds, matching Phase 1)
    c1_gains = []
    c1_acc_t1s = []
    c1_acc_t3s = []
    for s in range(N_SCRAMBLE_SEEDS):
        target_vn.enable_scrambled_feedback(seed=s + 3000)
        c1_m = compute_all_metrics_vn(target_vn, X_sq_te, y_te_vn)
        c1_gains.append(c1_m['gain'])
        c1_acc_t1s.append(c1_m['acc_t1'])
        c1_acc_t3s.append(c1_m['acc_t3'])
        target_vn.disable_scrambled_feedback()
    rows.append({'seed_model': seed_model, 'setting': 'vn', 'group': 'C1',
                 'noise_level': noise, 'gain': float(np.mean(c1_gains)),
                 'acc_t1': float(np.mean(c1_acc_t1s)),
                 'acc_t3': float(np.mean(c1_acc_t3s))})

    # C2 VN
    c2_vn = compute_all_metrics_with_clone_vn(target_vn, donor_vn, X_sq_te, y_te_vn)
    rows.append({'seed_model': seed_model, 'setting': 'vn', 'group': 'C2',
                 'noise_level': noise, 'gain': c2_vn['gain'],
                 'acc_t1': c2_vn['acc_t1'], 'acc_t3': c2_vn['acc_t3']})

    # -- Affine alignment --
    # Independent calibration data (not training data) to avoid data leakage
    N_CALIB = 400
    X_cal_st, _ = generate_data(N_CALIB, noise, seed=seed_model + 2000)
    W_st, b_st = fit_learned_affine(target_st, donor_st, X_cal_st, T=T)
    X_sq_cal, _ = generate_data_variable_noise(N_CALIB, noise, T=T, seed=seed_model + 2000)
    W_vn, b_vn = fit_learned_affine_vn(target_vn, donor_vn, X_sq_cal, T=T)

    # Affine static
    n = len(X_te_st)
    c1_af = c3_af = 0
    for i in range(n):
        out, _ = forward_sequence_with_learned_affine_clone(
            target_st, donor_st, X_te_st[i], W_st, b_st, T=T)
        tc = np.argmax(y_te_st[i])
        if np.argmax(out[0]) == tc: c1_af += 1
        if np.argmax(out[2]) == tc: c3_af += 1
    rows.append({'seed_model': seed_model, 'setting': 'static', 'group': 'C2-affine',
                 'noise_level': noise, 'gain': c3_af/n - c1_af/n,
                 'acc_t1': c1_af/n, 'acc_t3': c3_af/n})

    # Affine VN
    c1_af = c3_af = 0
    for i in range(n):
        out, _ = forward_sequence_with_learned_affine_clone_vn(
            target_vn, donor_vn, X_sq_te[i], W_vn, b_vn, T=T)
        tc = np.argmax(y_te_vn[i])
        if np.argmax(out[0]) == tc: c1_af += 1
        if np.argmax(out[2]) == tc: c3_af += 1
    rows.append({'seed_model': seed_model, 'setting': 'vn', 'group': 'C2-affine',
                 'noise_level': noise, 'gain': c3_af/n - c1_af/n,
                 'acc_t1': c1_af/n, 'acc_t3': c3_af/n})

    # -- MLP alignment --
    def fit_mlp(X_data, Y_data, mlp_seed=seed_model,
                val_fraction=0.2, patience=100):
        rng = np.random.RandomState(mlp_seed)
        ind, h, outd = X_data.shape[1], MLP_HIDDEN, Y_data.shape[1]

        # Sample-level train/val split (3 timesteps per sample)
        T_per_sample = 3
        n_total = len(X_data)
        n_samples = n_total // T_per_sample if n_total % T_per_sample == 0 else n_total
        if n_total % T_per_sample == 0 and n_total > T_per_sample:
            sample_idx = rng.permutation(n_samples)
            n_val_s = max(1, int(n_samples * val_fraction))
            val_set = set(sample_idx[:n_val_s])
            tr = [i for i in range(n_total) if (i // T_per_sample) not in val_set]
            vl = [i for i in range(n_total) if (i // T_per_sample) in val_set]
        else:
            idx = rng.permutation(n_total)
            n_val = max(1, int(n_total * val_fraction))
            tr, vl = list(idx[n_val:]), list(idx[:n_val])
        X_tr, Y_tr = X_data[tr], Y_data[tr]
        X_vl, Y_vl = X_data[vl], Y_data[vl]
        n_train = len(X_tr)

        W1 = rng.randn(ind, h) * np.sqrt(2.0/ind)
        b1 = np.zeros(h)
        W2 = rng.randn(h, outd) * np.sqrt(2.0/h)
        b2 = np.zeros(outd)
        max_grad_norm = 1.0

        best_val = float('inf')
        best_W1, best_b1, best_W2, best_b2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
        wait = 0

        for _ in range(MLP_EPOCHS):
            # Forward (train)
            z1 = X_tr @ W1 + b1; a1 = np.maximum(0, z1)
            pred = a1 @ W2 + b2; diff = pred - Y_tr
            dp = 2.0 * diff / n_train
            dW2 = a1.T @ dp; db2 = dp.sum(0)
            da1 = dp @ W2.T; dz1 = da1 * (z1 > 0)
            dW1 = X_tr.T @ dz1; db1 = dz1.sum(0)
            global_norm = np.sqrt(
                np.sum(dW1**2) + np.sum(db1**2) +
                np.sum(dW2**2) + np.sum(db2**2))
            if global_norm > max_grad_norm:
                scale = max_grad_norm / global_norm
                dW1 *= scale; db1 *= scale
                dW2 *= scale; db2 *= scale
            W1 -= MLP_LR * dW1; b1 -= MLP_LR * db1
            W2 -= MLP_LR * dW2; b2 -= MLP_LR * db2

            # Val loss + best-epoch selection
            z1v = X_vl @ W1 + b1; a1v = np.maximum(0, z1v)
            predv = a1v @ W2 + b2
            val_loss = np.mean((predv - Y_vl) ** 2)
            if val_loss < best_val:
                best_val = val_loss
                best_W1, best_b1 = W1.copy(), b1.copy()
                best_W2, best_b2 = W2.copy(), b2.copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return best_W1, best_b1, best_W2, best_b2

    def collect_logits(tgt, don, X_cal, vn=False):
        dall, tall = [], []
        for i in range(len(X_cal)):
            tgt.reset_state(); don.reset_state()
            for t in range(T):
                inp = X_cal[i, t] if vn else X_cal[i]
                ty = tgt.forward(inp); dy = don.forward(inp)
                dall.append(dy.copy()); tall.append(ty.copy())
        return np.array(dall), np.array(tall)

    D_st, T_st = collect_logits(target_st, donor_st, X_cal_st, vn=False)
    mW1s, mb1s, mW2s, mb2s = fit_mlp(D_st, T_st)

    D_vn, T_vn = collect_logits(target_vn, donor_vn, X_sq_cal, vn=True)
    mW1v, mb1v, mW2v, mb2v = fit_mlp(D_vn, T_vn)

    def apply_mlp(x, W1, b1, W2, b2):
        return np.maximum(0, x @ W1 + b1) @ W2 + b2

    # MLP static
    c1_m = c3_m = 0
    for i in range(n):
        target_st.reset_state(); donor_st.reset_state()
        tout, dout = [], []
        for t in range(T):
            dy = donor_st.forward(X_te_st[i]); dout.append(dy.copy())
            if t > 0:
                target_st._prev_output = apply_mlp(dout[t-1], mW1s, mb1s, mW2s, mb2s).copy()
                target_st._has_feedback = True
            ty = target_st.forward(X_te_st[i]); tout.append(ty)
        tc = np.argmax(y_te_st[i])
        if np.argmax(tout[0]) == tc: c1_m += 1
        if np.argmax(tout[2]) == tc: c3_m += 1
    rows.append({'seed_model': seed_model, 'setting': 'static', 'group': 'C2-mlp',
                 'noise_level': noise, 'gain': c3_m/n - c1_m/n,
                 'acc_t1': c1_m/n, 'acc_t3': c3_m/n})

    # MLP VN
    c1_m = c3_m = 0
    for i in range(n):
        target_vn.reset_state(); donor_vn.reset_state()
        tout, dout = [], []
        for t in range(T):
            dy = donor_vn.forward(X_sq_te[i, t]); dout.append(dy.copy())
            if t > 0:
                target_vn._prev_output = apply_mlp(dout[t-1], mW1v, mb1v, mW2v, mb2v).copy()
                target_vn._has_feedback = True
            ty = target_vn.forward(X_sq_te[i, t]); tout.append(ty)
        tc = np.argmax(y_te_vn[i])
        if np.argmax(tout[0]) == tc: c1_m += 1
        if np.argmax(tout[2]) == tc: c3_m += 1
    rows.append({'seed_model': seed_model, 'setting': 'vn', 'group': 'C2-mlp',
                 'noise_level': noise, 'gain': c3_m/n - c1_m/n,
                 'acc_t1': c1_m/n, 'acc_t3': c3_m/n})

    return rows


# ==============================================
# Main
# ==============================================

def main():
    os.makedirs('results', exist_ok=True)
    n_workers = get_n_workers()
    print(f"[N20] CPU={mp.cpu_count()}, workers={n_workers}, N_MODELS={N_MODELS}")

    # -- Phase 1: Static ablation --
    t0 = time.time()
    tasks_p1 = [(s, nl) for nl in NOISE_LEVELS for s in range(N_MODELS)]
    print(f"\n[Phase 1] Static ablation: {len(tasks_p1)} tasks...", flush=True)

    with mp.Pool(n_workers) as pool:
        results_p1 = []
        for batch in pool.imap_unordered(run_static_ablation, tasks_p1):
            results_p1.extend(batch)

    csv1 = 'results/raw_metrics.csv'
    fields1 = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']
    with open(csv1, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields1); w.writeheader(); w.writerows(results_p1)
    print(f"  -> {len(results_p1)} rows, {time.time()-t0:.0f}s", flush=True)

    # -- Phase 2: C2 + VN + alignment --
    t1 = time.time()
    print(f"\n[Phase 2] C2/VN/alignment: {N_MODELS} model pairs...", flush=True)

    with mp.Pool(n_workers) as pool:
        results_p2 = []
        for batch in pool.imap_unordered(run_c2_and_vn, range(N_MODELS)):
            results_p2.extend(batch)

    csv2 = 'results/n20_c2_vn_alignment.csv'
    fields2 = ['seed_model', 'setting', 'group', 'noise_level',
                'gain', 'acc_t1', 'acc_t3']
    with open(csv2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields2); w.writeheader(); w.writerows(results_p2)
    print(f"  -> {len(results_p2)} rows, {time.time()-t1:.0f}s", flush=True)

    # ==============================================
    # Summary statistics
    # ==============================================
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"N=20 FULL EXPERIMENT COMPLETE ({total_time:.0f}s)")
    print(f"{'='*70}")

    # Phase 1 summary (noise=0.5)
    print(f"\n[Phase 1: Static ablation, noise=0.5]")
    agg = defaultdict(lambda: defaultdict(list))
    for r in results_p1:
        if r['noise_level'] == 0.5:
            agg[r['group']][r['seed_model']].append(r['gain'])
    model_gains = {g: [np.mean(agg[g][s]) for s in sorted(agg[g])] for g in agg}

    print(f"  {'Group':12s}  {'gain (mean+/-SD)':>18s}  {'N':>3s}")
    for g in ['Baseline', 'A', 'B1', 'B2', 'C1', 'D', "D'", "D''"]:
        if g in model_gains:
            v = model_gains[g]
            print(f"  {g:12s}  {np.mean(v):+.4f}+/-{np.std(v):.4f}  {len(v):3d}")

    # Phase 2 summary
    print(f"\n[Phase 2: C2/VN/alignment, noise=0.5]")
    for setting in ['static', 'vn']:
        print(f"\n  [{setting.upper()}]")
        for group in ['Baseline', 'A', 'C1', 'C2', 'C2-affine', 'C2-mlp']:
            vals = [r['gain'] for r in results_p2
                    if r['setting'] == setting and r['group'] == group]
            if vals:
                print(f"    {group:14s}: {np.mean(vals):+.4f}+/-{np.std(vals):.4f} (N={len(vals)})")

    # Key statistical tests
    print(f"\n{'='*70}")
    print("KEY STATISTICAL TESTS (Wilcoxon exact, N=20)")
    print(f"{'='*70}")

    def holm_bonferroni(p_dict):
        """Apply Holm step-down correction."""
        sorted_ps = sorted(p_dict.items(), key=lambda x: x[1])
        m = len(sorted_ps)
        corrected = {}
        prev_adj = 0.0
        for rank, (name, p) in enumerate(sorted_ps):
            adj = min(p * (m - rank), 1.0)
            adj = max(prev_adj, adj)
            prev_adj = adj
            corrected[name] = adj
        return corrected

    for setting in ['static', 'vn']:
        print(f"\n  [{setting.upper()}]")
        bl = [r['gain'] for r in results_p2
              if r['setting'] == setting and r['group'] == 'Baseline']

        # Raw p-values for all comparisons
        print(f"    Raw p-values:")
        for group in ['A', 'C1', 'C2', 'C2-affine', 'C2-mlp']:
            gv = [r['gain'] for r in results_p2
                  if r['setting'] == setting and r['group'] == group]
            if len(bl) == len(gv) == N_MODELS:
                T_stat, p = wilcoxon_exact(np.array(bl), np.array(gv))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"      BL vs {group:12s}: T={T_stat:.0f}, p={p:.6f} {sig}")

        # VN primary family m=3: BL vs {A, C1, C2}
        if setting == 'vn':
            primary_raw = {}
            for g in ['A', 'C1', 'C2']:
                gv = [r['gain'] for r in results_p2
                      if r['setting'] == setting and r['group'] == g]
                if len(bl) == len(gv) == N_MODELS:
                    _, p = wilcoxon_exact(np.array(bl), np.array(gv))
                    primary_raw[g] = p
            if primary_raw:
                corrected = holm_bonferroni(primary_raw)
                print(f"    Holm-Bonferroni VN primary (m=3):")
                for g in ['A', 'C1', 'C2']:
                    if g in corrected:
                        sig = "***" if corrected[g] < 0.001 else "ns"
                        print(f"      BL vs {g}: corrected={corrected[g]:.4e} {sig}")

        # Secondary family m=2: C1-vs-A, C2-vs-A
        c1 = [r['gain'] for r in results_p2
              if r['setting'] == setting and r['group'] == 'C1']
        c2 = [r['gain'] for r in results_p2
              if r['setting'] == setting and r['group'] == 'C2']
        a = [r['gain'] for r in results_p2
             if r['setting'] == setting and r['group'] == 'A']
        sec_raw = {}
        if len(c1) == len(a) == N_MODELS:
            _, p = wilcoxon_exact(np.array(c1), np.array(a))
            sec_raw['C1-vs-A'] = p
        if len(c2) == len(a) == N_MODELS:
            _, p = wilcoxon_exact(np.array(c2), np.array(a))
            sec_raw['C2-vs-A'] = p
        if sec_raw:
            corrected = holm_bonferroni(sec_raw)
            print(f"    Holm-Bonferroni secondary (m=2):")
            for name in ['C1-vs-A', 'C2-vs-A']:
                if name in corrected:
                    sig = "***" if corrected[name] < 0.001 else "**" if corrected[name] < 0.01 else "*" if corrected[name] < 0.05 else "ns"
                    print(f"      {name}: corrected={corrected[name]:.4e} {sig}")

    print(f"\n[Saved] {csv1}, {csv2}")
    print(f"[Total time] {total_time:.0f}s")


if __name__ == '__main__':
    main()
