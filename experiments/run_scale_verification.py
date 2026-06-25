"""Task 3: Scale Verification - Fixed Task, Varying Width.

Hidden widths: 10 (S-35), 20 (S-55), 45 (S-105), 245 (S-505).
Both static and VN modes.
Conditions: Baseline, Group A, C1 (30 seeds avg), C2 (clone).
20 seeds each. Multiprocessing for training.

Output:
    results/scale_verification_static.csv
    results/scale_verification_vn.csv
    results/fig_scale_verification.png
    results/REPORT_SCALE.md
"""

import sys
import os
import csv
import time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Pool
from src.network import RecurrentMLP
from src.training import (
    generate_data, generate_data_variable_noise,
    train, train_vn,
)
from src.metrics import (
    compute_all_metrics, compute_all_metrics_vn,
    compute_all_metrics_with_clone, compute_all_metrics_with_clone_vn,
)
from src.ablation import deep_copy_weights, restore_weights


HIDDEN_WIDTHS = [10, 20, 45, 245]
SEEDS = list(range(20))
DONOR_SEEDS = list(range(100, 120))
N_C1_REPEATS = 30
NOISE_LEVEL = 0.5
N_TRAIN = 200
N_TEST = 200
EPOCHS = 1000
LR = 0.01
TAU = 2.0


# -- Top-level worker functions (picklable) --

def _train_static_worker(args):
    seed, h = args
    net = RecurrentMLP(input_size=10, hidden1=h, hidden2=h,
                       output_size=5, seed=seed, feedback_tau=TAU)
    X, y = generate_data(N_TRAIN, noise_level=NOISE_LEVEL, seed=seed)
    train(net, X, y, epochs=EPOCHS, lr=LR, time_weights=[0.0, 0.2, 1.0])
    weights = {k: v.copy() for k, v in net.get_all_weights().items()}
    biases = {k: v.copy() for k, v in net.get_all_biases().items()}
    return seed, h, weights, biases


def _train_vn_worker(args):
    seed, h = args
    net = RecurrentMLP(input_size=10, hidden1=h, hidden2=h,
                       output_size=5, seed=seed, feedback_tau=TAU)
    X_seq, y = generate_data_variable_noise(
        N_TRAIN, noise_level=NOISE_LEVEL, T=3, seed=seed)
    train_vn(net, X_seq, y, epochs=EPOCHS, lr=LR,
             time_weights=[0.0, 0.2, 1.0])
    weights = {k: v.copy() for k, v in net.get_all_weights().items()}
    biases = {k: v.copy() for k, v in net.get_all_biases().items()}
    return seed, h, weights, biases


def _rebuild_net(seed, h, weights, biases):
    net = RecurrentMLP(input_size=10, hidden1=h, hidden2=h,
                       output_size=5, seed=seed, feedback_tau=TAU)
    net.W_ih1[:] = weights['input_to_h1']
    net.W_h1h2[:] = weights['h1_to_h2']
    net.W_h2o[:] = weights['h2_to_output']
    net.W_rec[:] = weights['recurrent']
    net.b_h1[:] = biases['b_h1']
    net.b_h2[:] = biases['b_h2']
    net.b_out[:] = biases['b_out']
    return net


def evaluate_conditions_static(net, h, seed, X_test, y_test, clone_net=None):
    rows = []
    saved = deep_copy_weights(net)

    metrics = compute_all_metrics(net, X_test, y_test)
    rows.append({'hidden_width': h, 'seed': seed, 'condition': 'baseline',
                 'acc_t1': metrics['acc_t1'], 'acc_t3': metrics['acc_t3'],
                 'gain': metrics['gain']})

    net.disable_recurrent_loop()
    metrics = compute_all_metrics(net, X_test, y_test)
    # Mathematical shield: a no-recurrence network on static input
    # must produce identical predictions at every timestep, so gain = acc_t3 -
    # acc_t1 must be exactly 0.0 by construction. The 1e-12 tolerance absorbs
    # IEEE-754 dust; the minimum true non-zero gain is 1/N_TEST ~ 1/200 = 5e-3
    # (a single discrete mismatch), 9 orders of magnitude above tolerance.
    assert abs(metrics['gain']) < 1e-12, (
        f"Mathematical shield broken: Group A static gain = {metrics['gain']:.3e} "
        f"at (h={h}, seed={seed}). A no-recurrence network on static input must "
        f"have acc_t1 = acc_t3 exactly; any deviation indicates state leakage "
        f"upstream (likely in disable_recurrent_loop / restore_weights)."
    )
    rows.append({'hidden_width': h, 'seed': seed, 'condition': 'group_a',
                 'acc_t1': metrics['acc_t1'], 'acc_t3': metrics['acc_t3'],
                 'gain': metrics['gain']})
    net.enable_recurrent_loop()
    restore_weights(net, saved)

    c1_gains, c1_t1s, c1_t3s = [], [], []
    for rep in range(N_C1_REPEATS):
        net.enable_scrambled_feedback(seed=rep)
        metrics = compute_all_metrics(net, X_test, y_test)
        c1_gains.append(metrics['gain'])
        c1_t1s.append(metrics['acc_t1'])
        c1_t3s.append(metrics['acc_t3'])
        net.disable_scrambled_feedback()
        restore_weights(net, saved)
    rows.append({'hidden_width': h, 'seed': seed, 'condition': 'group_c1',
                 'acc_t1': np.mean(c1_t1s), 'acc_t3': np.mean(c1_t3s),
                 'gain': np.mean(c1_gains)})

    if clone_net is not None:
        metrics = compute_all_metrics_with_clone(net, clone_net, X_test, y_test)
        rows.append({'hidden_width': h, 'seed': seed, 'condition': 'group_c2',
                     'acc_t1': metrics['acc_t1'], 'acc_t3': metrics['acc_t3'],
                     'gain': metrics['gain']})

    restore_weights(net, saved)
    return rows


def evaluate_conditions_vn(net, h, seed, X_seq_test, y_test, clone_net=None):
    rows = []
    saved = deep_copy_weights(net)

    metrics = compute_all_metrics_vn(net, X_seq_test, y_test)
    rows.append({'hidden_width': h, 'seed': seed, 'condition': 'baseline',
                 'acc_t1': metrics['acc_t1'], 'acc_t3': metrics['acc_t3'],
                 'gain': metrics['gain']})

    net.disable_recurrent_loop()
    metrics = compute_all_metrics_vn(net, X_seq_test, y_test)
    rows.append({'hidden_width': h, 'seed': seed, 'condition': 'group_a',
                 'acc_t1': metrics['acc_t1'], 'acc_t3': metrics['acc_t3'],
                 'gain': metrics['gain']})
    net.enable_recurrent_loop()
    restore_weights(net, saved)

    c1_gains, c1_t1s, c1_t3s = [], [], []
    for rep in range(N_C1_REPEATS):
        net.enable_scrambled_feedback(seed=rep)
        metrics = compute_all_metrics_vn(net, X_seq_test, y_test)
        c1_gains.append(metrics['gain'])
        c1_t1s.append(metrics['acc_t1'])
        c1_t3s.append(metrics['acc_t3'])
        net.disable_scrambled_feedback()
        restore_weights(net, saved)
    rows.append({'hidden_width': h, 'seed': seed, 'condition': 'group_c1',
                 'acc_t1': np.mean(c1_t1s), 'acc_t3': np.mean(c1_t3s),
                 'gain': np.mean(c1_gains)})

    if clone_net is not None:
        metrics = compute_all_metrics_with_clone_vn(
            net, clone_net, X_seq_test, y_test)
        rows.append({'hidden_width': h, 'seed': seed, 'condition': 'group_c2',
                     'acc_t1': metrics['acc_t1'], 'acc_t3': metrics['acc_t3'],
                     'gain': metrics['gain']})

    restore_weights(net, saved)
    return rows


def main():
    n_workers = max(1, os.cpu_count() - 4)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    fieldnames = ['hidden_width', 'seed', 'condition', 'acc_t1', 'acc_t3', 'gain']

    total_start = time.time()
    print("=" * 60)
    print("Task 3: Scale Verification")
    print(f"  Hidden widths: {HIDDEN_WIDTHS}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    all_static_rows = []
    all_vn_rows = []

    for h in HIDDEN_WIDTHS:
        h_start = time.time()
        print(f"\n--- h={h} ---")

        all_seeds = SEEDS + DONOR_SEEDS  # 40 models trained per width: 20 target (analyzed; "20 models each" in paper) + 20 donor (for C2)
        train_args = [(s, h) for s in all_seeds]

        # Parallel static training
        print(f"  Training {len(all_seeds)} static models...")
        t0 = time.time()
        with Pool(processes=n_workers) as pool:
            static_results = pool.map(_train_static_worker, train_args)
        print(f"    {time.time()-t0:.0f}s")
        static_nets = {}
        for s, hh, w, b in static_results:
            static_nets[s] = _rebuild_net(s, hh, w, b)

        # Parallel VN training
        print(f"  Training {len(all_seeds)} VN models...")
        t0 = time.time()
        with Pool(processes=n_workers) as pool:
            vn_results = pool.map(_train_vn_worker, train_args)
        print(f"    {time.time()-t0:.0f}s")
        vn_nets = {}
        for s, hh, w, b in vn_results:
            vn_nets[s] = _rebuild_net(s, hh, w, b)

        # Evaluate static
        # Note: uses a shared test set (seed=999) across all models, unlike the
        # main experiment which uses per-model held-out sets. This reduces
        # across-seed variance slightly but does not bias group comparisons.
        print(f"  Evaluating static...")
        X_test, y_test = generate_data(N_TEST, noise_level=NOISE_LEVEL, seed=999)
        for s_idx, s in enumerate(SEEDS):
            clone = static_nets[DONOR_SEEDS[s_idx]]
            rows = evaluate_conditions_static(
                static_nets[s], h, s, X_test, y_test, clone_net=clone)
            all_static_rows.extend(rows)

        # Evaluate VN
        print(f"  Evaluating VN...")
        X_seq_test, y_test_vn = generate_data_variable_noise(
            N_TEST, noise_level=NOISE_LEVEL, T=3, seed=999)
        for s_idx, s in enumerate(SEEDS):
            clone = vn_nets[DONOR_SEEDS[s_idx]]
            rows = evaluate_conditions_vn(
                vn_nets[s], h, s, X_seq_test, y_test_vn, clone_net=clone)
            all_vn_rows.extend(rows)

        h_time = time.time() - h_start
        print(f"  h={h} done in {h_time/60:.1f} min")

    # Write CSVs
    csv_static = os.path.join(results_dir, 'scale_verification_static.csv')
    with open(csv_static, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_static_rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in r.items()})
    print(f"\nWritten {len(all_static_rows)} rows to {csv_static}")

    csv_vn = os.path.join(results_dir, 'scale_verification_vn.csv')
    with open(csv_vn, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_vn_rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in r.items()})
    print(f"Written {len(all_vn_rows)} rows to {csv_vn}")

    # Generate figure and report
    generate_figure(all_static_rows, all_vn_rows, results_dir)
    generate_report(all_static_rows, all_vn_rows, results_dir)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("Done!")


def generate_figure(static_rows, vn_rows, results_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conditions = ['baseline', 'group_a', 'group_c1', 'group_c2']
    labels = {'baseline': 'Baseline', 'group_a': 'Group A',
              'group_c1': 'Group C1', 'group_c2': 'Group C2'}
    # Colorblind-safer palette: Group A (no feedback) uses neutral gray to
    # distinguish from C1's warning orange; original red/orange were
    # confusable for protanopia/deuteranopia.
    colors = {'baseline': '#1f77b4', 'group_a': '#7f7f7f',
              'group_c1': '#ff7f0e', 'group_c2': '#9467bd'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for idx, (ax, rows, title) in enumerate([
        (axes[0], static_rows, 'Static'),
        (axes[1], vn_rows, 'Variable-Noise'),
    ]):
        x_pos = np.arange(len(HIDDEN_WIDTHS))
        bar_width = 0.18
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

        for c_idx, cond in enumerate(conditions):
            means, stds = [], []
            for h in HIDDEN_WIDTHS:
                gains = [r['gain'] for r in rows
                         if r['hidden_width'] == h and r['condition'] == cond]
                means.append(np.mean(gains) if gains else 0)
                stds.append(np.std(gains) if gains else 0)
            ax.bar(x_pos + offsets[c_idx], means, bar_width,
                   yerr=stds, label=labels[cond], color=colors[cond],
                   capsize=3, alpha=0.85, edgecolor='black', linewidth=0.4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(h) for h in HIDDEN_WIDTHS])
        ax.set_xlabel('Hidden width', fontsize=11)
        if idx == 0:
            ax.set_ylabel(
                r'Correction gain ($\mathrm{acc}_{t_3} - \mathrm{acc}_{t_1}$)',
                fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'fig_scale_verification.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")


def generate_report(static_rows, vn_rows, results_dir):
    lines = []
    lines.append("# Scale Verification Report\n")
    lines.append("## Configuration")
    lines.append(f"- Hidden widths: {HIDDEN_WIDTHS}")
    lines.append(f"- Seeds: 0-19 (targets), 100-119 (donors)")
    lines.append(f"- Epochs: {EPOCHS}, LR: {LR}, tau: {TAU}")
    lines.append(f"- Noise: {NOISE_LEVEL}")
    lines.append("")

    conditions = ['baseline', 'group_a', 'group_c1', 'group_c2']

    for mode, rows, title in [
        ('static', static_rows, 'Static'),
        ('vn', vn_rows, 'Variable-Noise'),
    ]:
        lines.append(f"## {title} Results\n")
        header = "| Hidden Width | " + " | ".join(
            c.replace('_', ' ').title() for c in conditions) + " |"
        sep = "|---|" + "---|" * len(conditions)
        lines.append(header)
        lines.append(sep)

        for h in HIDDEN_WIDTHS:
            row_cells = [str(h)]
            for cond in conditions:
                gains = [r['gain'] for r in rows
                         if r['hidden_width'] == h and r['condition'] == cond]
                if gains:
                    mean_g = np.mean(gains)
                    std_g = np.std(gains)
                    row_cells.append(f"{mean_g:+.4f} +/- {std_g:.4f}")
                else:
                    row_cells.append("N/A")
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    # Fake mirror effect analysis
    lines.append("## Fake Mirror Effect Persistence\n")
    for mode, rows, title in [
        ('static', static_rows, 'Static'),
        ('vn', vn_rows, 'VN'),
    ]:
        lines.append(f"### {title}\n")
        for cond in ['group_c1', 'group_c2']:
            all_negative = True
            exceptions = []
            for h in HIDDEN_WIDTHS:
                gains = [r['gain'] for r in rows
                         if r['hidden_width'] == h and r['condition'] == cond]
                mean_g = np.mean(gains) if gains else 0
                if mean_g >= 0:
                    all_negative = False
                    exceptions.append(f"h={h} (gain={mean_g:+.4f})")
            status = "YES" if all_negative else f"NO (exceptions: {', '.join(exceptions)})"
            lines.append(f"- {cond} gain < 0 at all scales? **{status}**")
        lines.append("")

    report_path = os.path.join(results_dir, 'REPORT_SCALE.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved {report_path}")


if __name__ == '__main__':
    main()
