"""Phase 1: Static ablation (Baseline, A, B1, B2, C1, D, D', D'').

Output: results/raw_metrics.csv (static groups, excluding C2).
C2 is added by run_c2_experiment.py (Phase 4).
VN results are in run_n20_full.py (Phase 2) -> n20_c2_vn_alignment.csv.

Top-level experiment driver (parallelised).

Parallelisation strategy:
- multiprocessing.Pool over (seed_model, noise_level) combinations
- Each worker independently trains the model and runs all groups
- Automatically scales to available CPU cores
- Reproducibility: per-worker seed is fixed
"""

# Disable NumPy internal multi-threading - avoids CPU thrashing with mp.Pool.
# Must be set BEFORE importing numpy.
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
from functools import partial

from src.network import RecurrentMLP, DeepFeedforwardMLP
from src.training import generate_data, train, train_deep_ff, evaluate_accuracy_deep_ff
from src.ablation import (
    ablate_recurrent, ablate_random, ablate_structural,
    deep_copy_weights, restore_weights,
)
from src.metrics import compute_all_metrics, compute_neuron_importance
from src.visualize import (
    plot_network_map, plot_ablation_comparison,
    plot_accuracy_distribution, plot_noise_sweep,
    plot_neuron_importance_heatmap, ensure_results_dir,
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


# ----------------------------------------------
# Worker function (one process per (seed, noise) combination)
# ----------------------------------------------

def run_single_model(args):
    """Train one model and run every group experiment.

    Args:
        args: (seed_model, noise_level)

    Returns:
        list of row dicts
    """
    seed_model, noise_level = args
    rows = []

    # Train
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed_model)
    X_train, y_train = generate_data(N_TRAIN, noise_level, seed=seed_model)
    X_test, y_test = generate_data(N_TEST, noise_level, seed=seed_model + 500)
    train(net, X_train, y_train, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    def make_row(group, metrics, seed_abl=0):
        return {
            'seed_model': seed_model,
            'group': group,
            'seed_ablation': seed_abl,
            'noise_level': noise_level,
            **{k: metrics[k] for k in ['acc_t1', 'acc_t2', 'acc_t3',
                                         'gain', 'ece', 'r_norm', 'delta_norm']}
        }

    # Baseline
    rows.append(make_row('Baseline', compute_all_metrics(net, X_test, y_test)))

    # Group A: cut the recurrent path
    saved = deep_copy_weights(net)
    ablate_recurrent(net)
    a_metrics = compute_all_metrics(net, X_test, y_test)
    # Mathematical shield (symmetric to E1 static in
    # run_integration_control_extras.py): a no-recurrence network on STATIC
    # input must produce identical predictions at every timestep, so
    # gain = acc_t3 - acc_t1 must be exactly 0.0 by construction. 1e-12
    # tolerance absorbs IEEE-754 dust; the minimum true non-zero gain is
    # 1/N_TEST (a single discrete mismatch), many orders of magnitude above
    # tolerance. Any failure indicates state leakage upstream of
    # ablate_recurrent / restore_weights.
    assert abs(a_metrics['gain']) < 1e-12, (
        f"Mathematical shield broken: Group A static gain = {a_metrics['gain']:.3e}. "
        f"A no-recurrence network on static input must yield acc_t1 = acc_t3."
    )
    rows.append(make_row('A', a_metrics))
    restore_weights(net, saved)

    # Group B1: random ablation (30 repeats)
    for seed_abl in range(N_RANDOM_ABLATIONS):
        saved = deep_copy_weights(net)
        ablate_random(net, n_connections=N_REC_WEIGHTS, seed=seed_abl + 1000)
        rows.append(make_row('B1', compute_all_metrics(net, X_test, y_test), seed_abl + 1000))
        restore_weights(net, saved)

    # Group B2: structural ablation (h2_to_output, 50 params = same count as W_rec)
    saved = deep_copy_weights(net)
    ablate_structural(net, layer='h2_to_output')
    rows.append(make_row('B2', compute_all_metrics(net, X_test, y_test)))
    restore_weights(net, saved)

    # Group C1: permuted feedback (30 repeats)
    for seed_scr in range(N_SCRAMBLE_SEEDS):
        net.enable_scrambled_feedback(seed=seed_scr + 2000)
        rows.append(make_row('C1', compute_all_metrics(net, X_test, y_test), seed_scr + 2000))
        net.disable_scrambled_feedback()

    # Group D: feedforward (trained without recurrence)
    net_d = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                         output_size=5, seed=seed_model)
    net_d.disable_recurrent_loop()
    X_train_d, y_train_d = generate_data(N_TRAIN, noise_level, seed=seed_model)
    train(net_d, X_train_d, y_train_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    rows.append(make_row('D', compute_all_metrics(net_d, X_test, y_test)))

    # Group D': Param-matched FF (skip connection)
    net_dp = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                          output_size=5, seed=seed_model, skip_connection=True)
    net_dp.disable_recurrent_loop()
    train(net_dp, X_train_d, y_train_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    rows.append(make_row("D'", compute_all_metrics(net_dp, X_test, y_test)))

    # Group D'': Compute-matched FF (6-layer deep)
    net_dpp = DeepFeedforwardMLP(input_size=10, hidden_size=10, n_hidden=6,
                                  output_size=5, seed=seed_model)
    train_deep_ff(net_dpp, X_train_d, y_train_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR)
    acc_dpp = evaluate_accuracy_deep_ff(net_dpp, X_test, y_test)
    rows.append({
        'seed_model': seed_model,
        'group': "D''",
        'seed_ablation': 0,
        'noise_level': noise_level,
        'acc_t1': acc_dpp, 'acc_t2': acc_dpp, 'acc_t3': acc_dpp,
        'gain': acc_dpp - acc_dpp, 'ece': '', 'r_norm': '', 'delta_norm': '',
    })

    return rows


# ----------------------------------------------
# Main
# ----------------------------------------------

def run_full_experiment():
    ensure_results_dir('results')

    # Generate all (seed, noise) combinations
    tasks = [(seed, nl) for nl in NOISE_LEVELS for seed in range(N_MODELS)]
    n_workers = min(max(1, mp.cpu_count() - 4), len(tasks))

    print(f"[EXP] Starting: {len(tasks)} tasks on {n_workers} workers "
          f"({N_MODELS} models x {len(NOISE_LEVELS)} noise levels)", flush=True)

    # Parallel execution (with tqdm progress bar)
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        if has_tqdm:
            for batch in tqdm(pool.imap_unordered(run_single_model, tasks),
                              total=len(tasks), desc="Experiments"):
                all_rows.extend(batch)
        else:
            results = pool.map(run_single_model, tasks)
            for batch in results:
                all_rows.extend(batch)

    print(f"[EXP] Collected {len(all_rows)} rows", flush=True)

    # Save CSV
    csv_path = 'results/raw_metrics.csv'
    csv_fields = ['seed_model', 'group', 'seed_ablation', 'noise_level',
                  'acc_t1', 'acc_t2', 'acc_t3', 'gain', 'ece', 'r_norm', 'delta_norm']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # ------------------------------------------
    # Figures
    # ------------------------------------------
    print("[EXP] Generating plots...", flush=True)

    # Ablation comparison (noise=0.5)
    comparison = {}
    for row in all_rows:
        if row['noise_level'] == 0.5:
            comparison.setdefault(row['group'], []).append(row['gain'])
    plot_ablation_comparison(comparison, 'results/ablation_comparison.png')

    # Accuracy distribution (noise=0.5)
    b1 = [r['gain'] for r in all_rows if r['group'] == 'B1' and r['noise_level'] == 0.5]
    a_g = np.mean([r['gain'] for r in all_rows if r['group'] == 'A' and r['noise_level'] == 0.5])
    c1_g = np.mean([r['gain'] for r in all_rows if r['group'] == 'C1' and r['noise_level'] == 0.5])
    if b1:
        plot_accuracy_distribution(b1, a_g, c1_g, 'results/accuracy_distribution.png')

    # Noise sweep
    sweep = {}
    for row in all_rows:
        sweep.setdefault(row['group'], {}).setdefault(row['noise_level'], []).append(row['gain'])
    plot_noise_sweep(sweep, 'results/noise_sweep_curve.png')

    # Network map + Neuron importance heatmap (noise=0.5, seed=0)
    net_hm = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5, seed=0)
    X_train_hm, y_train_hm = generate_data(N_TRAIN, 0.5, seed=0)
    X_test_hm, y_test_hm = generate_data(N_TEST, 0.5, seed=500)
    train(net_hm, X_train_hm, y_train_hm, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    plot_network_map(net_hm, 'results/network_map.png')

    print("[EXP] Computing neuron importance heatmap...", flush=True)
    intelligence, correction = compute_neuron_importance(net_hm, X_test_hm, y_test_hm)
    plot_neuron_importance_heatmap(intelligence, correction, 'results/neuron_importance_heatmap.png')

    # Save neuron-importance CSV (reproducibility)
    ni_path = 'results/neuron_importance.csv'
    with open(ni_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['neuron_id', 'intelligence_importance',
                                                'correction_importance'])
        writer.writeheader()
        for nid in sorted(intelligence.keys()):
            writer.writerow({'neuron_id': nid,
                             'intelligence_importance': intelligence[nid],
                             'correction_importance': correction[nid]})
    print(f"[EXP] Saved neuron importance to {ni_path}", flush=True)

    # ------------------------------------------
    # Model-level aggregation
    # B1/C1 have 30 repeats per model -> average to a single seed_model value, N=20
    # ------------------------------------------
    def aggregate_by_model(rows, noise=0.5):
        """Aggregate rows by seed_model within each group. {group: [model_mean_gain x 20]}"""
        from collections import defaultdict
        raw = defaultdict(lambda: defaultdict(list))
        raw_acc_t1 = defaultdict(lambda: defaultdict(list))
        raw_acc_t3 = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r['noise_level'] == noise:
                raw[r['group']][r['seed_model']].append(r['gain'])
                raw_acc_t1[r['group']][r['seed_model']].append(r['acc_t1'])
                raw_acc_t3[r['group']][r['seed_model']].append(r['acc_t3'])
        result = {}
        result_t1 = {}
        result_t3 = {}
        for g in raw:
            result[g] = [np.mean(raw[g][s]) for s in sorted(raw[g].keys())]
            result_t1[g] = [np.mean(raw_acc_t1[g][s]) for s in sorted(raw_acc_t1[g].keys())]
            result_t3[g] = [np.mean(raw_acc_t3[g][s]) for s in sorted(raw_acc_t3[g].keys())]
        return result, result_t1, result_t3

    model_gains, model_t1, model_t3 = aggregate_by_model(all_rows, noise=0.5)

    # ------------------------------------------
    # Statistical summary (model-level, N=20)
    # ------------------------------------------
    print(f"\n{'='*70}", flush=True)
    print("RESULTS SUMMARY (noise_level=0.5, model-level aggregation, N=20)", flush=True)
    print('='*70, flush=True)
    print(f"  {'Group':12s}  {'acc_t1':>10s}  {'acc_t3':>10s}  {'gain':>14s}  {'n':>3s}", flush=True)
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*3}", flush=True)

    for g in ['Baseline', 'A', 'B1', 'B2', 'C1', 'D', "D'", "D''"]:
        if g in model_gains:
            gains = model_gains[g]
            t1s = model_t1[g]
            t3s = model_t3[g]
            print(f"  {g:12s}  {np.mean(t1s):.4f}+/-{np.std(t1s):.3f}"
                  f"  {np.mean(t3s):.4f}+/-{np.std(t3s):.3f}"
                  f"  {np.mean(gains):+.4f}+/-{np.std(gains):.4f}"
                  f"  {len(gains):3d}", flush=True)

    # Bootstrap 95% CI (model-level)
    print("\n95% Bootstrap CI (noise=0.5, model-level):", flush=True)
    rng = np.random.RandomState(999)
    for g in ['Baseline', 'A', 'B1', 'C1', 'D', "D'", "D''"]:
        if g in model_gains and len(model_gains[g]) > 1:
            gains = np.array(model_gains[g])
            boot = [np.mean(rng.choice(gains, len(gains), replace=True)) for _ in range(10000)]
            print(f"  {g:12s}: [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Paired-difference Bootstrap 95% CI (BL - X, resampling paired diffs as a unit)
    baseline_arr = np.array(model_gains.get('Baseline', []))
    if len(baseline_arr) > 1:
        print("\n95% Bootstrap CI for paired differences (BL - X):", flush=True)
        rng_paired = np.random.RandomState(999)
        for g in ['A', 'B1', 'C1', 'C2', 'D', "D'", "D''"]:
            if g in model_gains and len(model_gains[g]) == len(baseline_arr):
                diffs = baseline_arr - np.array(model_gains[g])
                boot = [np.mean(rng_paired.choice(diffs, len(diffs), replace=True))
                        for _ in range(10000)]
                print(f"  BL-{g:6s}: mean={np.mean(diffs):+.4f}, "
                      f"CI=[{np.percentile(boot, 2.5):+.4f}, "
                      f"{np.percentile(boot, 97.5):+.4f}]", flush=True)

    # Holm-Bonferroni (pre-specified families per S2.6)
    from src.metrics import wilcoxon_exact

    def holm_bonferroni(p_dict):
        """Apply Holm step-down correction. Returns {name: corrected_p}."""
        sorted_ps = sorted(p_dict.items(), key=lambda x: x[1])
        m = len(sorted_ps)
        corrected = {}
        prev_adj = 0.0
        for rank, (name, p) in enumerate(sorted_ps):
            adj = min(p * (m - rank), 1.0)
            adj = max(prev_adj, adj)  # enforce monotonicity
            prev_adj = adj
            corrected[name] = adj
        return corrected

    baseline_gains = np.array(model_gains.get('Baseline', []))

    # Static primary family m=4: BL vs {A, B1, C1, C2}
    # Note: C2 is evaluated in run_c2_experiment.py; if absent here, m < 4.
    # Full m=4 correction uses combined results from both scripts.
    print("\nHolm-Bonferroni: Static primary family (m=4):", flush=True)
    primary_raw = {}
    for g in ['A', 'B1', 'C1', 'C2']:
        if g in model_gains:
            g_gains = np.array(model_gains[g])
            if len(g_gains) > 1 and len(baseline_gains) > 1:
                _, p = wilcoxon_exact(baseline_gains, g_gains)
                primary_raw[g] = p
    if primary_raw:
        corrected = holm_bonferroni(primary_raw)
        for g in ['A', 'B1', 'C1', 'C2']:
            if g in corrected:
                sig = "***" if corrected[g] < 0.001 else "**" if corrected[g] < 0.01 else "*" if corrected[g] < 0.05 else "ns"
                print(f"  Baseline vs {g:4s}: raw={primary_raw[g]:.4e}, corrected={corrected[g]:.4e} {sig}", flush=True)

    # Structural controls (excluded from Holm families, raw p only)
    print("\nRaw p-values (structural controls, excluded from Holm):", flush=True)
    for g in ['D', "D'", "D''"]:
        if g in model_gains:
            g_gains = np.array(model_gains[g])
            if len(g_gains) > 1 and len(baseline_gains) > 1:
                _, p = wilcoxon_exact(baseline_gains, g_gains)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  Baseline vs {g:4s}: raw={p:.4e} {sig}", flush=True)

    print(f"\nExperiment complete. Results in results/", flush=True)


if __name__ == '__main__':
    run_full_experiment()
