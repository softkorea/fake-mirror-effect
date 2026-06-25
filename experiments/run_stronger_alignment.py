"""Stronger alignment experiment: test if larger MLPs close the ~20% residual.

Tests progressively stronger alignment models:
  1. Affine:     5->5 linear (30 params)    - same as run_n20_full affine
  2. MLP-small:  5->16->5    (181 params)    - same as run_n20_full MLP
  3. MLP-medium: 5->64->64->5 (~4.9k params)  - 25x more params
  4. MLP-large:  5->128->128->5 (~17.9k params) - 100x more params

Exact param counts are computed at runtime by count_params().

All use doubled calibration data (1200 logit pairs); aligner training budgets are tuned per
tier (500 epochs for affine/MLP-small, 2000 for MLP-medium/large).
Reports gain for each alignment level to see if residual closes.
"""

import os
import sys
import time
import csv
import numpy as np
import multiprocessing as mp

# CPU affinity for NumPy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import (
    generate_data, generate_data_variable_noise,
    train, train_vn,
)
from src.ablation import (
    forward_sequence_with_clone, forward_sequence_with_clone_vn,
)

N_MODELS = 20
NOISE = 0.5
T = 3
N_TRAIN = 200
N_TEST = 200
N_CAL = 400  # doubled from 200
EPOCHS_TRAIN = 1000  # target/donor MODEL training -> 1000 (convergence).
                     # The aligner capacity ladder below keeps its tuned per-tier
                     # budgets (500/2000) - those are mapping fits, not model training.
TIME_WEIGHTS = [0.0, 0.2, 1.0]

# Alignment configs to test
ALIGN_CONFIGS = [
    {'name': 'affine',     'hidden': [],         'epochs': 500,  'lr': 0.01},
    {'name': 'MLP-small',  'hidden': [16],       'epochs': 500,  'lr': 0.01},
    {'name': 'MLP-medium', 'hidden': [64, 64],   'epochs': 2000, 'lr': 0.005},
    {'name': 'MLP-large',  'hidden': [128, 128], 'epochs': 2000, 'lr': 0.003},
]


def count_params(layers):
    """Count params in a list of (W, b) tuples."""
    return sum(w.size + b.size for w, b in layers)


def fit_deep_mlp(X_data, Y_data, hidden_sizes, epochs, lr, seed=0,
                  val_fraction=0.2, patience=100):
    """Fit a multi-layer MLP from donor logits to target logits.

    Uses train/val split with early stopping to prevent overfitting.
    """
    rng = np.random.RandomState(seed)
    n_total = len(X_data)

    # Train/val split by SAMPLE ID to prevent temporal leakage.
    # Data has T rows per sample (e.g., 400 samples x 3 timesteps = 1200 rows).
    # Split at the sample level, then flatten.
    T_per_sample = 3  # timesteps per sample
    n_samples = n_total // T_per_sample if n_total % T_per_sample == 0 else n_total
    if n_total % T_per_sample == 0 and n_total > T_per_sample:
        sample_indices = rng.permutation(n_samples)
        n_val_samples = max(1, int(n_samples * val_fraction))
        val_samples = set(sample_indices[:n_val_samples])
        train_rows = [i for i in range(n_total) if (i // T_per_sample) not in val_samples]
        val_rows = [i for i in range(n_total) if (i // T_per_sample) in val_samples]
    else:
        # Fallback: random row split (for non-sequence data)
        indices = rng.permutation(n_total)
        n_val = max(1, int(n_total * val_fraction))
        train_rows = list(indices[n_val:])
        val_rows = list(indices[:n_val])
    X_train, Y_train = X_data[train_rows], Y_data[train_rows]
    n_train = len(X_train)
    X_val, Y_val = X_data[val_rows], Y_data[val_rows]

    # Build layers
    layers = []
    sizes = [X_data.shape[1]] + hidden_sizes + [Y_data.shape[1]]
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        W = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        layers.append([W, b])

    max_grad_norm = 1.0
    best_val_loss = float('inf')
    best_layers = None
    wait = 0

    def _forward(X, layers_):
        a = X
        acts = [a]
        pre_acts = []
        for i, (W, b) in enumerate(layers_):
            z = a @ W + b
            pre_acts.append(z)
            a = np.maximum(0, z) if i < len(layers_) - 1 else z
            acts.append(a)
        return acts, pre_acts

    def _mse(pred, target):
        return np.mean((pred - target) ** 2)

    for epoch in range(epochs):
        # Forward (train)
        activations, pre_activations = _forward(X_train, layers)
        pred = activations[-1]
        diff = pred - Y_train

        # Backward
        grads = []
        dp = 2.0 * diff / n_train

        for i in reversed(range(len(layers))):
            W, b = layers[i]
            a_prev = activations[i]
            dW = a_prev.T @ dp
            db = dp.sum(0)
            grads.append((dW, db))
            if i > 0:
                dp = dp @ W.T
                dp = dp * (pre_activations[i - 1] > 0)

        grads.reverse()

        # Global norm clipping
        global_norm = np.sqrt(sum(
            np.sum(dW**2) + np.sum(db**2) for dW, db in grads
        ))
        if global_norm > max_grad_norm:
            scale = max_grad_norm / global_norm
            grads = [(dW * scale, db * scale) for dW, db in grads]

        # Update
        for i, (dW, db) in enumerate(grads):
            layers[i][0] -= lr * dW
            layers[i][1] -= lr * db

        # Validation loss + early stopping
        val_acts, _ = _forward(X_val, layers)
        val_loss = _mse(val_acts[-1], Y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_layers = [(W.copy(), b.copy()) for W, b in layers]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return best_layers if best_layers is not None else layers


def apply_deep_mlp(x, layers):
    """Forward pass through a fitted deep MLP."""
    a = x
    for i, (W, b) in enumerate(layers):
        z = a @ W + b
        if i < len(layers) - 1:
            a = np.maximum(0, z)
        else:
            a = z
    return a


def collect_logits(tgt, don, X_cal, vn=False):
    """Collect (donor, target) logit pairs for calibration."""
    dall, tall = [], []
    for i in range(len(X_cal)):
        tgt.reset_state()
        don.reset_state()
        for t in range(T):
            inp = X_cal[i, t] if vn else X_cal[i]
            ty = tgt.forward(inp)
            dy = don.forward(inp)
            dall.append(dy.copy())
            tall.append(ty.copy())
    return np.array(dall), np.array(tall)


def evaluate_alignment(target, donor, X_test, y_test, layers, vn=False):
    """Evaluate aligned clone feedback."""
    n = len(X_test)
    c1 = c3 = 0
    for i in range(n):
        target.reset_state()
        donor.reset_state()
        dout = []
        tout = []
        for t in range(T):
            inp = X_test[i, t] if vn else X_test[i]
            dy = donor.forward(inp)
            dout.append(dy.copy())
            if t > 0:
                aligned = apply_deep_mlp(dout[t - 1], layers)
                target._prev_output = aligned.copy()
                target._has_feedback = True
            ty = target.forward(inp)
            tout.append(ty.copy())
        tc = np.argmax(y_test[i])
        if np.argmax(tout[0]) == tc:
            c1 += 1
        if np.argmax(tout[-1]) == tc:
            c3 += 1
    return c1 / n, c3 / n


def run_one_seed(seed_model):
    """Run all alignment configs for one model pair."""
    rows = []

    # Train target (static + VN)
    X_tr, y_tr = generate_data(N_TRAIN, NOISE, seed=seed_model)
    X_te, y_te = generate_data(N_TEST, NOISE, seed=seed_model + 500)
    target_st = RecurrentMLP(seed=seed_model)
    train(target_st, X_tr, y_tr, epochs=EPOCHS_TRAIN, lr=0.01, time_weights=TIME_WEIGHTS)

    X_sq_tr, y_sq_tr = generate_data_variable_noise(N_TRAIN, NOISE, T=T, seed=seed_model)
    X_sq_te, y_sq_te = generate_data_variable_noise(N_TEST, NOISE, T=T, seed=seed_model + 500)
    target_vn = RecurrentMLP(seed=seed_model)
    train_vn(target_vn, X_sq_tr, y_sq_tr, epochs=EPOCHS_TRAIN, lr=0.01, T=T, time_weights=TIME_WEIGHTS)

    # Train donor
    donor_seed = seed_model + 100
    X_tr_d, y_tr_d = generate_data(N_TRAIN, NOISE, seed=donor_seed)
    donor_st = RecurrentMLP(seed=donor_seed)
    train(donor_st, X_tr_d, y_tr_d, epochs=EPOCHS_TRAIN, lr=0.01, time_weights=TIME_WEIGHTS)

    X_sq_tr_d, y_sq_tr_d = generate_data_variable_noise(N_TRAIN, NOISE, T=T, seed=donor_seed)
    donor_vn = RecurrentMLP(seed=donor_seed)
    train_vn(donor_vn, X_sq_tr_d, y_sq_tr_d, epochs=EPOCHS_TRAIN, lr=0.01, T=T, time_weights=TIME_WEIGHTS)

    # Calibration data (doubled: N_CAL=400)
    X_cal_st, _ = generate_data(N_CAL, NOISE, seed=seed_model + 3000)
    X_sq_cal, _ = generate_data_variable_noise(N_CAL, NOISE, T=T, seed=seed_model + 3000)

    # Collect calibration logits
    D_st, Tgt_st = collect_logits(target_st, donor_st, X_cal_st, vn=False)
    D_vn, Tgt_vn = collect_logits(target_vn, donor_vn, X_sq_cal, vn=True)

    print(f"  seed={seed_model}: cal_pairs_st={len(D_st)}, cal_pairs_vn={len(D_vn)}")

    # Baseline and raw C2 for reference
    # Static baseline
    bl_c1 = bl_c3 = 0
    c2_c1 = c2_c3 = 0
    n = len(X_te)
    for i in range(n):
        outs_bl, _ = target_st.forward_sequence(X_te[i], T)
        tc = np.argmax(y_te[i])
        if np.argmax(outs_bl[0]) == tc: bl_c1 += 1
        if np.argmax(outs_bl[2]) == tc: bl_c3 += 1

        outs_c2, _ = forward_sequence_with_clone(target_st, donor_st, X_te[i], T)
        if np.argmax(outs_c2[0]) == tc: c2_c1 += 1
        if np.argmax(outs_c2[2]) == tc: c2_c3 += 1

    rows.append({'seed': seed_model, 'setting': 'static', 'group': 'Baseline',
                 'gain': bl_c3/n - bl_c1/n, 'params': 0})
    rows.append({'seed': seed_model, 'setting': 'static', 'group': 'C2-raw',
                 'gain': c2_c3/n - c2_c1/n, 'params': 0})

    # VN baseline and raw C2
    bl_c1 = bl_c3 = 0
    c2_c1 = c2_c3 = 0
    n_vn = len(X_sq_te)
    for i in range(n_vn):
        outs_bl, _ = target_vn.forward_sequence_vn(X_sq_te[i], T)
        tc = np.argmax(y_sq_te[i])
        if np.argmax(outs_bl[0]) == tc: bl_c1 += 1
        if np.argmax(outs_bl[2]) == tc: bl_c3 += 1

        outs_c2, _ = forward_sequence_with_clone_vn(target_vn, donor_vn, X_sq_te[i], T)
        if np.argmax(outs_c2[0]) == tc: c2_c1 += 1
        if np.argmax(outs_c2[2]) == tc: c2_c3 += 1

    rows.append({'seed': seed_model, 'setting': 'vn', 'group': 'Baseline',
                 'gain': bl_c3/n_vn - bl_c1/n_vn, 'params': 0})
    rows.append({'seed': seed_model, 'setting': 'vn', 'group': 'C2-raw',
                 'gain': c2_c3/n_vn - c2_c1/n_vn, 'params': 0})

    # Test each alignment config
    for cfg in ALIGN_CONFIGS:
        t0 = time.time()
        # Fit alignment
        layers_st = fit_deep_mlp(D_st, Tgt_st, cfg['hidden'], cfg['epochs'],
                                  cfg['lr'], seed=seed_model)
        layers_vn = fit_deep_mlp(D_vn, Tgt_vn, cfg['hidden'], cfg['epochs'],
                                  cfg['lr'], seed=seed_model)

        n_params = count_params([(W, b) for W, b in layers_st])

        # Evaluate static
        acc1_st, acc3_st = evaluate_alignment(
            target_st, donor_st, X_te, y_te, layers_st, vn=False)
        rows.append({'seed': seed_model, 'setting': 'static',
                     'group': f'C2-{cfg["name"]}',
                     'gain': acc3_st - acc1_st, 'params': n_params})

        # Evaluate VN
        acc1_vn, acc3_vn = evaluate_alignment(
            target_vn, donor_vn, X_sq_te, y_sq_te, layers_vn, vn=True)
        rows.append({'seed': seed_model, 'setting': 'vn',
                     'group': f'C2-{cfg["name"]}',
                     'gain': acc3_vn - acc1_vn, 'params': n_params})

        elapsed = time.time() - t0
        print(f"    {cfg['name']} ({n_params} params): "
              f"static={acc3_st - acc1_st:+.4f} vn={acc3_vn - acc1_vn:+.4f} ({elapsed:.1f}s)")

    return rows


def main():
    print("=" * 60)
    print("Stronger Alignment Experiment")
    print(f"  N={N_MODELS}, calibration={N_CAL} samples")
    print(f"  Configs: {[c['name'] for c in ALIGN_CONFIGS]}")
    print("=" * 60)

    os.makedirs('results', exist_ok=True)
    total_t0 = time.time()

    n_workers = max(1, mp.cpu_count() - 4)
    print(f"  Workers: {n_workers}\n")

    with mp.Pool(n_workers) as pool:
        all_results = pool.map(run_one_seed, range(N_MODELS))

    rows = [r for sublist in all_results for r in sublist]

    # Save CSV
    fieldnames = ['seed', 'setting', 'group', 'gain', 'params']
    with open('results/stronger_alignment.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    groups = sorted(set(r['group'] for r in rows))
    for setting in ['static', 'vn']:
        print(f"\n  [{setting.upper()}]")
        for group in groups:
            vals = [r['gain'] for r in rows
                    if r['setting'] == setting and r['group'] == group]
            if vals:
                params = [r['params'] for r in rows
                          if r['setting'] == setting and r['group'] == group][0]
                m = np.mean(vals)
                s = np.std(vals)
                print(f"    {group:20s}: {m:+.4f} +/- {s:.4f}  (params={params})")

        # Recovery percentages
        bl = [r['gain'] for r in rows
              if r['setting'] == setting and r['group'] == 'Baseline']
        c2 = [r['gain'] for r in rows
              if r['setting'] == setting and r['group'] == 'C2-raw']
        if bl and c2:
            gap = np.mean(bl) - np.mean(c2)
            print(f"\n    Baseline-C2 gap: {gap:+.4f}")
            if abs(gap) > 1e-8:
                for cfg in ALIGN_CONFIGS:
                    gname = f'C2-{cfg["name"]}'
                    av = [r['gain'] for r in rows
                          if r['setting'] == setting and r['group'] == gname]
                    if av:
                        recovery = (np.mean(av) - np.mean(c2)) / gap * 100
                        residual = (np.mean(bl) - np.mean(av)) / gap * 100
                        print(f"    {gname:20s}: recovery={recovery:.1f}%, "
                              f"residual={residual:.1f}%")
            else:
                print("    (gap ~ 0, skipping recovery computation)")

    # Statistical tests
    from src.metrics import wilcoxon_exact
    report_lines = []
    report_lines.append("# Stronger Alignment Experiment Results\n")

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (Wilcoxon exact, paired)")
    print("=" * 60)

    for setting in ['static', 'vn']:
        print(f"\n  [{setting.upper()}]")
        report_lines.append(f"\n## {setting.upper()}\n")

        bl = np.array([r['gain'] for r in rows
                       if r['setting'] == setting and r['group'] == 'Baseline'])
        c2 = np.array([r['gain'] for r in rows
                       if r['setting'] == setting and r['group'] == 'C2-raw'])

        # Summary table
        report_lines.append("| Group | Gain (mean±std) | Params | Recovery % | Residual % |")
        report_lines.append("|-------|----------------|--------|------------|------------|")
        gap = np.mean(bl) - np.mean(c2)
        report_lines.append(f"| Baseline | {np.mean(bl):+.4f}±{np.std(bl):.4f} | - | - | - |")
        report_lines.append(f"| C2-raw | {np.mean(c2):+.4f}±{np.std(c2):.4f} | 0 | 0.0% | 100.0% |")

        for cfg in ALIGN_CONFIGS:
            gname = f'C2-{cfg["name"]}'
            av = np.array([r['gain'] for r in rows
                          if r['setting'] == setting and r['group'] == gname])
            if len(av) == N_MODELS and abs(gap) > 1e-8:
                rec = (np.mean(av) - np.mean(c2)) / gap * 100
                res = (np.mean(bl) - np.mean(av)) / gap * 100
                params = [r['params'] for r in rows
                          if r['setting'] == setting and r['group'] == gname][0]
                report_lines.append(
                    f"| {gname} | {np.mean(av):+.4f}±{np.std(av):.4f} | {params} | {rec:.1f}% | {res:.1f}% |")

        # Baseline vs each aligned
        report_lines.append(f"\n### Baseline vs Aligned (is residual significant?)\n")
        report_lines.append("| Comparison | Diff | T | p |")
        report_lines.append("|------------|------|---|---|")
        print(f"\n    Baseline vs Aligned:")
        for cfg in ALIGN_CONFIGS:
            gname = f'C2-{cfg["name"]}'
            av = np.array([r['gain'] for r in rows
                          if r['setting'] == setting and r['group'] == gname])
            if len(bl) == len(av) == N_MODELS:
                T_stat, p = wilcoxon_exact(bl, av)
                diff = np.mean(bl) - np.mean(av)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    BL vs {gname:20s}: diff={diff:+.4f}, T={T_stat:.0f}, p={p:.6f} {sig}")
                report_lines.append(f"| BL vs {gname} | {diff:+.4f} | {T_stat:.0f} | {p:.6f} {sig} |")

        # C2-raw vs each aligned (improvement significant?)
        report_lines.append(f"\n### C2-raw vs Aligned (is improvement significant?)\n")
        report_lines.append("| Comparison | Diff | T | p |")
        report_lines.append("|------------|------|---|---|")
        print(f"\n    C2-raw vs Aligned:")
        for cfg in ALIGN_CONFIGS:
            gname = f'C2-{cfg["name"]}'
            av = np.array([r['gain'] for r in rows
                          if r['setting'] == setting and r['group'] == gname])
            if len(c2) == len(av) == N_MODELS:
                T_stat, p = wilcoxon_exact(av, c2)
                diff = np.mean(av) - np.mean(c2)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {gname:20s} vs C2-raw: diff={diff:+.4f}, T={T_stat:.0f}, p={p:.6f} {sig}")
                report_lines.append(f"| {gname} vs C2-raw | {diff:+.4f} | {T_stat:.0f} | {p:.6f} {sig} |")

    # Save report
    with open('results/REPORT_STRONGER_ALIGNMENT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')

    total_time = time.time() - total_t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"Saved: results/stronger_alignment.csv, results/REPORT_STRONGER_ALIGNMENT.md")


if __name__ == '__main__':
    main()
