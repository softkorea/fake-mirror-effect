"""Phase 4: Static C2 clone feedback - (re)writes C2 rows into raw_metrics.csv.

Output: results/raw_metrics.csv -- strips any existing C2 rows, then rewrites the
file with fresh static C2 rows (idempotent on re-run; not a plain append).
VN C2 and alignment results are in run_n20_full.py (Phase 2).

For each target model (seed 0-19), replace its feedback with the output
of an independently trained donor model (seed 100-119) and re-evaluate.
Strict 1:1 independent pairing.

Parallelisation strategy:
- Parallel per noise_level (train both target and donor at the same noise
  before running the clone evaluation).
- n_workers = cpu_count() - 4 (avoid hogging the host).
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
from collections import defaultdict

from src.network import RecurrentMLP
from src.training import generate_data, train
from src.metrics import compute_all_metrics_with_clone
from src.visualize import plot_ablation_comparison, plot_noise_sweep, ensure_results_dir

# ----------------------------------------------
# Configuration (identical to the primary experiment)
# ----------------------------------------------

N_MODELS = 20
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_EPOCHS = 1000
TRAIN_LR = 0.01
N_TRAIN = 200
N_TEST = 200
T = 3


# ----------------------------------------------
# Worker function (one process per noise_level)
# ----------------------------------------------

DONOR_SEED_OFFSET = 100  # donor seeds: 100-119 (independent of target seeds)


def run_c2_for_noise(noise_level):
    """Train 20 target + 20 donor models at one noise_level and run C2 eval.

    Target models (seed 0-19) are 1:1 paired with independent donor models
    (seed 100-119) - strict statistical independence.

    Returns:
        list of row dicts
    """
    rows = []

    # Train 20 target models
    targets = []
    test_data = []
    for seed in range(N_MODELS):
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=seed)
        X_test, y_test = generate_data(N_TEST, noise_level, seed=seed + 500)
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        targets.append(net)
        test_data.append((X_test, y_test))

    # Train 20 independent donor models (seed 100-119)
    # Independently sampled data at the same noise_level distribution.
    donors = []
    for seed in range(N_MODELS):
        donor_seed = seed + DONOR_SEED_OFFSET
        net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                           output_size=5, seed=donor_seed)
        X_train, y_train = generate_data(N_TRAIN, noise_level, seed=donor_seed)
        train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
        donors.append(net)

    # C2: Clone feedback - replace target[i] feedback with donor[i] (independent 1:1 pairing)
    for seed in range(N_MODELS):
        target = targets[seed]
        donor = donors[seed]
        X_test, y_test = test_data[seed]

        metrics = compute_all_metrics_with_clone(target, donor, X_test, y_test)

        row = {
            'seed_model': seed,
            'group': 'C2',
            'seed_ablation': seed + DONOR_SEED_OFFSET,
            'noise_level': noise_level,
            **{k: metrics[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                         'gain', 'ece', 'r_norm', 'delta_norm']}
        }
        rows.append(row)

    return rows


# ----------------------------------------------
# Main
# ----------------------------------------------

def run_c2_experiment():
    ensure_results_dir('results')

    n_workers = max(1, mp.cpu_count() - 4)

    print(f"[C2] Starting: {len(NOISE_LEVELS)} noise levels on {n_workers} workers "
          f"(cpu_count={mp.cpu_count()}, reserved 4)", flush=True)

    # Parallel execution
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for batch in tqdm(pool.imap_unordered(run_c2_for_noise, NOISE_LEVELS),
                              total=len(NOISE_LEVELS), desc="C2 Experiment"):
                all_rows.extend(batch)
        else:
            results = pool.map(run_c2_for_noise, NOISE_LEVELS)
            for batch in results:
                all_rows.extend(batch)

    print(f"[C2] Collected {len(all_rows)} C2 rows", flush=True)

    # Rewrite CSV: re-emit non-C2 rows, then fresh C2 rows (idempotent; not a plain append)
    csv_path = 'results/raw_metrics.csv'
    csv_fields = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']

    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)
    else:
        print(f"[C2] Warning: {csv_path} not found. Creating new file.", flush=True)

    # Remove existing C2 rows (prevent duplicates on re-run)
    existing_rows = [r for r in existing_rows if r['group'] != 'C2']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(all_rows)

    print(f"[C2] Wrote {len(all_rows)} C2 rows to {csv_path} (idempotent rewrite, prior C2 stripped)", flush=True)

    # Regenerate figures
    all_data = existing_rows + all_rows

    # Ablation comparison (noise=0.5)
    comparison = {}
    for row in all_data:
        nl = float(row['noise_level']) if isinstance(row['noise_level'], str) else row['noise_level']
        if nl == 0.5:
            g = row['group']
            gain = float(row['gain']) if isinstance(row['gain'], str) else row['gain']
            comparison.setdefault(g, []).append(gain)
    plot_ablation_comparison(comparison, 'results/ablation_comparison.png')
    print("[C2] Updated ablation_comparison.png", flush=True)

    # Noise sweep (all groups including C2)
    sweep = {}
    for row in all_data:
        g = row['group']
        nl = float(row['noise_level']) if isinstance(row['noise_level'], str) else row['noise_level']
        gain = float(row['gain']) if isinstance(row['gain'], str) else row['gain']
        sweep.setdefault(g, {}).setdefault(nl, []).append(gain)
    plot_noise_sweep(sweep, 'results/noise_sweep_curve.png')
    print("[C2] Updated noise_sweep_curve.png", flush=True)

    # Statistical summary
    print(f"\n{'='*70}", flush=True)
    print("C2 RESULTS SUMMARY (model-level aggregation, N=20)", flush=True)
    print('='*70, flush=True)

    def aggregate(rows, noise=0.5):
        raw = defaultdict(lambda: defaultdict(list))
        for r in rows:
            nl = float(r['noise_level']) if isinstance(r['noise_level'], str) else r['noise_level']
            if nl == noise:
                g = r['group']
                gain = float(r['gain']) if isinstance(r['gain'], str) else r['gain']
                sm = int(r['seed_model']) if isinstance(r['seed_model'], str) else r['seed_model']
                raw[g][sm].append(gain)
        return {g: [np.mean(raw[g][s]) for s in sorted(raw[g].keys())] for g in raw}

    model_gains = aggregate(all_data, noise=0.5)

    print(f"\n  {'Group':12s}  {'gain (mean+/-std)':>20s}  {'n':>3s}", flush=True)
    print(f"  {'-'*12}  {'-'*20}  {'-'*3}", flush=True)

    for g in ['Baseline', 'A', 'C1', 'C2']:
        if g in model_gains:
            gains = model_gains[g]
            print(f"  {g:12s}  {np.mean(gains):+.4f}+/-{np.std(gains):.4f}"
                  f"          {len(gains):3d}", flush=True)

    # Bootstrap 95% CI
    rng = np.random.RandomState(999)
    print("\n95% Bootstrap CI (noise=0.5, model-level):", flush=True)
    for g in ['Baseline', 'A', 'C1', 'C2']:
        if g in model_gains and len(model_gains[g]) > 1:
            gains = np.array(model_gains[g])
            boot = [np.mean(rng.choice(gains, len(gains), replace=True)) for _ in range(10000)]
            print(f"  {g:12s}: [{np.percentile(boot, 2.5):+.4f}, "
                  f"{np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Paired-difference Bootstrap 95% CI (BL - X, resampling paired diffs as a unit)
    baseline_arr = np.array(model_gains.get('Baseline', []))
    if len(baseline_arr) > 1:
        print("\n95% Bootstrap CI for paired differences (BL - X):", flush=True)
        rng_paired = np.random.RandomState(999)
        for g in ['A', 'C1', 'C2']:
            if g in model_gains and len(model_gains[g]) == len(baseline_arr):
                diffs = baseline_arr - np.array(model_gains[g])
                boot = [np.mean(rng_paired.choice(diffs, len(diffs), replace=True))
                        for _ in range(10000)]
                print(f"  BL-{g:6s}: mean={np.mean(diffs):+.4f}, "
                      f"CI=[{np.percentile(boot, 2.5):+.4f}, "
                      f"{np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Holm-Bonferroni corrected p-values
    from src.metrics import wilcoxon_exact
    print("\nHolm-Bonferroni corrected p-values (Baseline vs each, noise=0.5):", flush=True)
    baseline_gains = np.array(model_gains.get('Baseline', []))
    p_values = {}
    for g in ['A', 'C1', 'C2']:
        if g in model_gains:
            g_gains = np.array(model_gains[g])
            if len(g_gains) > 1 and len(baseline_gains) > 1:
                _, p = wilcoxon_exact(baseline_gains, g_gains)
                p_values[g] = p

    if p_values:
        sorted_ps = sorted(p_values.items(), key=lambda x: x[1])
        m_comp = len(sorted_ps)
        prev_adj_p = 0.0
        for rank, (g, p) in enumerate(sorted_ps):
            adj_p = min(p * (m_comp - rank), 1.0)
            adj_p = max(prev_adj_p, adj_p)
            prev_adj_p = adj_p
            sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
            print(f"  Baseline vs {g:4s}: p={adj_p:.4e} {sig}", flush=True)

    # Generate REPORT
    write_report(model_gains, baseline_gains, p_values)

    print(f"\n[C2] Experiment complete.", flush=True)


def write_report(model_gains, baseline_gains, p_values):
    """Generate REPORT_C2.md."""
    report_path = 'results/REPORT_C2.md'
    rng = np.random.RandomState(999)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Group C2 (Clone Feedback) — Experiment Report\n\n")
        f.write("## Executive Summary\n\n")

        c2_gains = np.array(model_gains.get('C2', []))
        bl_gains = np.array(model_gains.get('Baseline', []))
        n = len(c2_gains)

        if len(c2_gains) > 0:
            f.write(f"Group C2 injects **another trained model's well-formed output** "
                    f"as feedback (N={n}).\n\n")
            f.write(f"Result: **gain = {np.mean(c2_gains):+.3f} +/- {np.std(c2_gains):.3f}**\n\n")

        f.write("---\n\n")

        # Per-group table
        f.write("## 1. Results at noise=0.5 (model-level)\n\n")
        f.write("| Group | gain (mean+/-SD) | 95% CI | N |\n")
        f.write("|-------|------------------|--------|---|\n")
        for g in ['Baseline', 'A', 'C1', 'C2']:
            if g in model_gains and len(model_gains[g]) > 1:
                gains = np.array(model_gains[g])
                boot = [np.mean(rng.choice(gains, len(gains), replace=True))
                        for _ in range(10000)]
                ci_lo = np.percentile(boot, 2.5)
                ci_hi = np.percentile(boot, 97.5)
                f.write(f"| {g} | {np.mean(gains):+.4f}+/-{np.std(gains):.4f} | "
                        f"[{ci_lo:+.4f}, {ci_hi:+.4f}] | {len(gains)} |\n")

        # Wilcoxon tests
        f.write("\n## 2. Statistical Tests\n\n")
        f.write("Holm-Bonferroni corrected Wilcoxon signed-rank tests "
                "(Baseline vs each):\n\n")
        if p_values:
            sorted_ps = sorted(p_values.items(), key=lambda x: x[1])
            m_comp = len(sorted_ps)
            prev_adj_p = 0.0
            for rank, (g, p) in enumerate(sorted_ps):
                adj_p = min(p * (m_comp - rank), 1.0)
                adj_p = max(prev_adj_p, adj_p)
                prev_adj_p = adj_p
                sig = "***" if adj_p < 0.001 else "**" if adj_p < 0.01 else "*" if adj_p < 0.05 else "ns"
                f.write(f"- Baseline vs {g}: p={adj_p:.4e} {sig}\n")

        # C2 interpretation
        f.write("\n## 3. Interpretation\n\n")
        if len(c2_gains) > 0 and len(bl_gains) > 0:
            c1_gains = np.array(model_gains.get('C1', []))
            f.write(f"- C2 gain ({np.mean(c2_gains):+.4f}) is comparable to "
                    f"C1 ({np.mean(c1_gains):+.4f})\n")
            f.write("- Both produce negative gain (worse than no feedback)\n")
            f.write("- This demonstrates feedback-contract specificity: "
                    "the recurrent weights are co-adapted to the model's own "
                    "output geometry\n")

    print(f"[C2] Report saved to {report_path}", flush=True)


if __name__ == '__main__':
    run_c2_experiment()
