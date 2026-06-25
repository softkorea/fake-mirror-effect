"""Hyperparameter Sweep - emergence robustness check.

w1  in  [0.0, 0.1, 0.2, 0.3]
w2  in  [0.1, 0.2, 0.3, 0.5]
w3 = 1.0 (fixed)
tau   in  [1.0, 1.5, 2.0, 3.0, 5.0]

80 configs x 10 models, noise=0.5.
Measures Baseline only (checks whether gain > 0).
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
import multiprocessing as mp
import itertools

from src.network import RecurrentMLP
from src.training import generate_data, train
from src.metrics import compute_all_metrics

# ----------------------------------------------
# Configuration
# ----------------------------------------------

W1_VALUES = [0.0, 0.1, 0.2, 0.3]
W2_VALUES = [0.1, 0.2, 0.3, 0.5]
W3_FIXED = 1.0
TAU_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0]

N_MODELS = 10
NOISE_LEVEL = 0.5
TRAIN_EPOCHS = 1000
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3


# ----------------------------------------------
# Worker function
# ----------------------------------------------

def run_single_config(args):
    """Process one (w1, w2, tau, seed) combination.

    Returns:
        dict with config + metrics
    """
    w1, w2, tau, seed_model = args
    time_weights = [w1, w2, W3_FIXED]

    # Train
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed_model, feedback_tau=tau)
    X_train, y_train = generate_data(N_TRAIN, NOISE_LEVEL, seed=seed_model)
    X_test, y_test = generate_data(N_TEST, NOISE_LEVEL, seed=seed_model + 500)
    train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T,
          time_weights=time_weights)

    # Evaluate
    metrics = compute_all_metrics(net, X_test, y_test)

    return {
        'w1': w1,
        'w2': w2,
        'w3': W3_FIXED,
        'tau': tau,
        'seed_model': seed_model,
        'acc_t1': metrics['acc_t1'],
        'acc_t2': metrics['acc_t2'],
        'acc_t3': metrics['acc_t3'],
        'gain': metrics['gain'],
        'r_norm': metrics['r_norm'],
        'delta_norm': metrics['delta_norm'],
    }


# ----------------------------------------------
# Main
# ----------------------------------------------

def run_sweep():
    os.makedirs('results', exist_ok=True)

    # Generate all combinations
    configs = list(itertools.product(W1_VALUES, W2_VALUES, TAU_VALUES))
    tasks = [(w1, w2, tau, seed)
             for w1, w2, tau in configs
             for seed in range(N_MODELS)]

    n_workers = min(max(1, mp.cpu_count() - 4), len(tasks))
    print(f"[SWEEP] {len(configs)} configs x {N_MODELS} models = {len(tasks)} tasks "
          f"on {n_workers} workers", flush=True)

    # Parallel execution
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for row in tqdm(pool.imap_unordered(run_single_config, tasks),
                            total=len(tasks), desc="Sweep"):
                all_rows.append(row)
        else:
            all_rows = pool.map(run_single_config, tasks)

    print(f"[SWEEP] Collected {len(all_rows)} rows", flush=True)

    # Save CSV
    csv_path = 'results/sweep_hyperparams.csv'
    csv_fields = ['w1', 'w2', 'w3', 'tau', 'seed_model',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'r_norm', 'delta_norm']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[SWEEP] Saved to {csv_path}", flush=True)

    # ------------------------------------------
    # Analysis: per-config model-level aggregation
    # ------------------------------------------
    from collections import defaultdict
    config_gains = defaultdict(list)
    config_t1 = defaultdict(list)
    config_t3 = defaultdict(list)
    for r in all_rows:
        key = (r['w1'], r['w2'], r['tau'])
        config_gains[key].append(r['gain'])
        config_t1[key].append(r['acc_t1'])
        config_t3[key].append(r['acc_t3'])

    print(f"\n{'='*80}", flush=True)
    print("HYPERPARAMETER SWEEP RESULTS (noise=0.5, N=10 models per config)", flush=True)
    print('='*80, flush=True)
    print(f"  {'w1':>4s}  {'w2':>4s}  {'tau':>4s}  |  {'acc_t1':>10s}  {'acc_t3':>10s}"
          f"  {'gain':>12s}  {'emerge':>7s}", flush=True)
    print(f"  {'-'*4}  {'-'*4}  {'-'*4}  |  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*7}", flush=True)

    emergence_count = 0
    total_configs = 0
    emergence_configs = []

    for key in sorted(config_gains.keys()):
        w1, w2, tau = key
        gains = np.array(config_gains[key])
        t1s = np.array(config_t1[key])
        t3s = np.array(config_t3[key])
        mean_gain = np.mean(gains)
        std_gain = np.std(gains)
        frac_positive = np.mean(gains > 0)
        emerged = mean_gain > 0 and frac_positive >= 0.6
        total_configs += 1
        if emerged:
            emergence_count += 1
            emergence_configs.append(key)
        tag = "  YES" if emerged else "   no"
        print(f"  {w1:4.1f}  {w2:4.1f}  {tau:4.1f}  |  "
              f"{np.mean(t1s):.3f}+/-{np.std(t1s):.3f}  "
              f"{np.mean(t3s):.3f}+/-{np.std(t3s):.3f}  "
              f"{mean_gain:+.4f}+/-{std_gain:.4f}  {tag}", flush=True)

    print(f"\n{'-'*80}", flush=True)
    print(f"Emergence: {emergence_count}/{total_configs} configs "
          f"({100*emergence_count/total_configs:.0f}%)", flush=True)

    if emergence_configs:
        w1s = sorted(set(k[0] for k in emergence_configs))
        w2s = sorted(set(k[1] for k in emergence_configs))
        taus = sorted(set(k[2] for k in emergence_configs))
        print(f"  w1 range: {w1s}", flush=True)
        print(f"  w2 range: {w2s}", flush=True)
        print(f"  tau  range: {taus}", flush=True)

    # ------------------------------------------
    # Heatmap visualisation
    # ------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for tau in TAU_VALUES:
        fig, ax = plt.subplots(figsize=(6, 5))
        matrix = np.zeros((len(W1_VALUES), len(W2_VALUES)))
        for i, w1 in enumerate(W1_VALUES):
            for j, w2 in enumerate(W2_VALUES):
                key = (w1, w2, tau)
                if key in config_gains:
                    matrix[i, j] = np.mean(config_gains[key])

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                        vmin=-0.05, vmax=0.10, origin='lower')
        ax.set_xticks(range(len(W2_VALUES)))
        ax.set_xticklabels([f'{w:.1f}' for w in W2_VALUES])
        ax.set_yticks(range(len(W1_VALUES)))
        ax.set_yticklabels([f'{w:.1f}' for w in W1_VALUES])
        ax.set_xlabel('w2 (t=2 weight)')
        ax.set_ylabel('w1 (t=1 weight)')
        ax.set_title(f'Mean Correction Gain (τ={tau})')

        for i in range(len(W1_VALUES)):
            for j in range(len(W2_VALUES)):
                val = matrix[i, j]
                color = 'white' if abs(val) > 0.04 else 'black'
                ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Correction Gain')
        plt.tight_layout()
        plt.savefig(f'results/sweep_heatmap_tau{tau}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Combined tau sweep figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for tau in TAU_VALUES:
        gains_by_tau = []
        for w1 in W1_VALUES:
            for w2 in W2_VALUES:
                key = (w1, w2, tau)
                if key in config_gains:
                    gains_by_tau.append(np.mean(config_gains[key]))
        ax.scatter([tau] * len(gains_by_tau), gains_by_tau, alpha=0.5, s=30)
        ax.errorbar(tau, np.mean(gains_by_tau), yerr=np.std(gains_by_tau),
                    fmt='D', color='black', markersize=8, capsize=5, zorder=5)

    ax.axhline(y=0, color='red', linestyle='--', lw=1)
    ax.set_xlabel('Temperature τ', fontsize=12)
    ax.set_ylabel('Mean Correction Gain', fontsize=12)
    ax.set_title('Emergence vs Temperature (all w1,w2 configs)', fontsize=13)
    ax.set_xticks(TAU_VALUES)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/sweep_tau_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[SWEEP] Plots saved to results/sweep_*.png", flush=True)

    # ------------------------------------------
    # Generate REPORT
    # ------------------------------------------
    write_report(config_gains, config_t1, config_t3,
                 emergence_count, total_configs, emergence_configs)


def write_report(config_gains, config_t1, config_t3,
                 emergence_count, total_configs, emergence_configs):
    """Generate REPORT_SWEEP.md."""
    report_path = 'results/REPORT_SWEEP.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hyperparameter Sweep — Robustness Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"We conducted **{total_configs * N_MODELS} experiments** across "
                f"{total_configs} hyperparameter combinations "
                f"(w1 x w2 x tau) x {N_MODELS} independent models.\n")
        f.write(f"Result: emergence (self-correction) was confirmed in "
                f"**{emergence_count}/{total_configs} "
                f"({100*emergence_count/total_configs:.0f}%)** combinations.\n\n")
        f.write("---\n\n")

        # Config
        f.write("## 1. Sweep Configuration\n\n")
        f.write("| Item | Value |\n|------|-------|\n")
        f.write(f"| w1 (t=1 weight) | {W1_VALUES} |\n")
        f.write(f"| w2 (t=2 weight) | {W2_VALUES} |\n")
        f.write(f"| w3 (t=3 weight) | {W3_FIXED} (fixed) |\n")
        f.write(f"| tau (feedback temperature) | {TAU_VALUES} |\n")
        f.write(f"| Total combinations | {total_configs} |\n")
        f.write(f"| Models per config | {N_MODELS} |\n")
        f.write(f"| Total experiments | {total_configs * N_MODELS} |\n")
        f.write(f"| Noise level | {NOISE_LEVEL} |\n")
        f.write(f"| Emergence criterion | mean gain > 0 AND >= 60% positive |\n\n")

        # Results table
        f.write("## 2. Per-Configuration Results\n\n")
        f.write("| w1 | w2 | tau | acc_t1 | acc_t3 | gain (mean+/-SD) | %pos | emerge |\n")
        f.write("|----|----|-----|--------|--------|------------------|------|--------|\n")

        for key in sorted(config_gains.keys()):
            w1, w2, tau = key
            gains = np.array(config_gains[key])
            t1s = np.array(config_t1[key])
            t3s = np.array(config_t3[key])
            mean_gain = np.mean(gains)
            std_gain = np.std(gains)
            frac_pos = np.mean(gains > 0)
            emerged = mean_gain > 0 and frac_pos >= 0.6
            tag = "YES" if emerged else "no"
            f.write(f"| {w1:.1f} | {w2:.1f} | {tau:.1f} | "
                    f"{np.mean(t1s):.3f}+/-{np.std(t1s):.3f} | "
                    f"{np.mean(t3s):.3f}+/-{np.std(t3s):.3f} | "
                    f"{mean_gain:+.4f}+/-{std_gain:.4f} | "
                    f"{100*frac_pos:.0f}% | {tag} |\n")

        # Summary
        f.write(f"\n## 3. Summary\n\n")
        f.write(f"- Emergence rate: {emergence_count}/{total_configs} "
                f"({100*emergence_count/total_configs:.0f}%)\n")
        if emergence_configs:
            w1s = sorted(set(k[0] for k in emergence_configs))
            w2s = sorted(set(k[1] for k in emergence_configs))
            taus = sorted(set(k[2] for k in emergence_configs))
            f.write(f"- w1 range with emergence: {w1s}\n")
            f.write(f"- w2 range with emergence: {w2s}\n")
            f.write(f"- tau range with emergence: {taus}\n")

        # Best/worst
        all_means = {k: np.mean(v) for k, v in config_gains.items()}
        best = max(all_means, key=all_means.get)
        worst = min(all_means, key=all_means.get)
        f.write(f"\n- Best config: w1={best[0]}, w2={best[1]}, tau={best[2]} "
                f"(gain={all_means[best]:+.4f})\n")
        f.write(f"- Worst config: w1={worst[0]}, w2={worst[1]}, tau={worst[2]} "
                f"(gain={all_means[worst]:+.4f})\n")

    print(f"[SWEEP] Report saved to {report_path}", flush=True)


if __name__ == '__main__':
    run_sweep()
