"""Task 2: VN Hyperparameter Partial Sweep.

w1 x tau grid: w1 in {0.0, 0.1, 0.2, 0.3}, tau in {1.0, 1.5, 2.0, 3.0}.
w2=0.2 fixed. 16 configs x 10 seeds = 160 training runs.

Output:
    results/sweep_vn_hyperparams.csv
    results/sweep_vn_heatmap.png
    results/REPORT_VN_SWEEP.md
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
from src.training import generate_data_variable_noise, train_vn
from src.metrics import compute_all_metrics_vn


def run_single_config(args):
    """Train and evaluate one (seed, w1, tau) configuration."""
    seed, w1, w2, tau = args
    noise_level = 0.5
    n_train = 200
    n_test = 200
    epochs = 1000
    lr = 0.01

    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=tau)

    X_seq_train, y_train = generate_data_variable_noise(
        n_train, noise_level=noise_level, T=3, seed=seed)
    X_seq_test, y_test = generate_data_variable_noise(
        n_test, noise_level=noise_level, T=3, seed=seed + 500)

    losses = train_vn(net, X_seq_train, y_train, epochs=epochs, lr=lr,
                      time_weights=[w1, w2, 1.0])

    metrics = compute_all_metrics_vn(net, X_seq_test, y_test)

    return {
        'w1': w1,
        'w2': w2,
        'tau': tau,
        'seed': seed,
        'acc_t1': metrics['acc_t1'],
        'acc_t3': metrics['acc_t3'],
        'gain': metrics['gain'],
    }


def main():
    w1_values = [0.0, 0.1, 0.2, 0.3]
    tau_values = [1.0, 1.5, 2.0, 3.0]
    w2 = 0.2
    seeds = list(range(10))

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'sweep_vn_hyperparams.csv')

    total_start = time.time()
    print("=" * 60)
    print("Task 2: VN Hyperparameter Partial Sweep")
    print(f"  Grid: w1={w1_values} x tau={tau_values}")
    print(f"  w2={w2} fixed, 10 seeds each")
    print(f"  Total: {len(w1_values)*len(tau_values)*len(seeds)} runs")
    print("=" * 60)

    # Build all configs
    configs = []
    for w1 in w1_values:
        for tau in tau_values:
            for seed in seeds:
                configs.append((seed, w1, w2, tau))

    # Run with multiprocessing
    n_workers = max(1, os.cpu_count() - 4)
    print(f"\nRunning with {n_workers} workers...")

    results = []
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_config, configs)):
            results.append(result)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - total_start
                print(f"  {i+1}/{len(configs)} done ({elapsed:.0f}s)")

    # Sort for reproducibility
    results.sort(key=lambda r: (r['w1'], r['tau'], r['seed']))

    # Write CSV
    fieldnames = ['w1', 'w2', 'tau', 'seed', 'acc_t1', 'acc_t3', 'gain']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'w1': f"{r['w1']:.1f}",
                'w2': f"{r['w2']:.1f}",
                'tau': f"{r['tau']:.1f}",
                'seed': r['seed'],
                'acc_t1': f"{r['acc_t1']:.6f}",
                'acc_t3': f"{r['acc_t3']:.6f}",
                'gain': f"{r['gain']:.6f}",
            })
    print(f"\nWritten {len(results)} rows to {csv_path}")

    # Generate heatmap and report
    generate_heatmap(results, results_dir)
    generate_report(results, results_dir)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("Done!")


def generate_heatmap(results, results_dir):
    """Generate sweep_vn_heatmap.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w1_values = sorted(set(r['w1'] for r in results))
    tau_values = sorted(set(r['tau'] for r in results))

    # Compute cell means twice: float for the colormap, Decimal for the
    # displayed label. Float arithmetic accumulates IEEE-754 error that
    # mis-rounds near-half boundaries (e.g., 1.005/10 evaluates to
    # 0.10049999... rather than 0.1005, so rounding to 3 decimals yields
    # 0.100 instead of 0.101). Decimal
    # arithmetic from string representations preserves source precision.
    from decimal import Decimal
    gain_matrix = np.zeros((len(w1_values), len(tau_values)))
    gain_matrix_exact = [[Decimal(0)] * len(tau_values)
                         for _ in range(len(w1_values))]
    for i, w1 in enumerate(w1_values):
        for j, tau in enumerate(tau_values):
            gains = [r['gain'] for r in results
                     if r['w1'] == w1 and r['tau'] == tau]
            if gains:
                gain_matrix[i, j] = np.mean(gains)
                total = sum((Decimal(str(g)) for g in gains), Decimal(0))
                gain_matrix_exact[i][j] = total / Decimal(len(gains))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Sequential YlGn colormap: all values in this sweep are positive,
    # so a diverging palette would be misleading. YlGn is colorblind-safe.
    im = ax.imshow(gain_matrix, aspect='auto', cmap='YlGn', origin='lower')
    ax.set_xticks(range(len(tau_values)))
    ax.set_xticklabels([f"{t:.1f}" for t in tau_values])
    ax.set_yticks(range(len(w1_values)))
    ax.set_yticklabels([f"{w:.1f}" for w in w1_values])
    ax.set_xlabel(r'Feedback temperature $\tau$', fontsize=11)
    ax.set_ylabel(r'Time weight $w_1$', fontsize=11)
    ax.tick_params(labelsize=10)
    # Title removed -- caption describes the figure (paper convention).

    # Annotate cells with luminance-aware text color.
    # Round the exact Decimal mean with ROUND_HALF_UP so labels match the
    # true mathematical value (not float-arithmetic artefacts).
    from decimal import ROUND_HALF_UP
    vmin, vmax = gain_matrix.min(), gain_matrix.max()
    for i in range(len(w1_values)):
        for j in range(len(tau_values)):
            val = gain_matrix[i, j]
            normalized = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
            color = 'white' if normalized > 0.6 else 'black'
            rounded = gain_matrix_exact[i][j].quantize(
                Decimal('0.001'), rounding=ROUND_HALF_UP)
            sign = '+' if rounded >= 0 else '-'
            label = f"{sign}{abs(rounded):.3f}"
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    cbar = plt.colorbar(im,
        label=r'Mean correction gain ($\mathrm{acc}_{t_3} - \mathrm{acc}_{t_1}$)')
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'sweep_vn_heatmap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")


def generate_report(results, results_dir):
    """Generate REPORT_VN_SWEEP.md."""
    w1_values = sorted(set(r['w1'] for r in results))
    tau_values = sorted(set(r['tau'] for r in results))

    lines = []
    lines.append("# VN Hyperparameter Sweep Report\n")
    lines.append("## Configuration")
    lines.append(f"- w1: {w1_values}")
    lines.append(f"- tau: {tau_values}")
    lines.append(f"- w2: 0.2 (fixed)")
    lines.append(f"- Seeds: 0-9 (10 models per config)")
    lines.append(f"- Total runs: {len(results)}")
    lines.append("")

    # Emergence analysis
    lines.append("## Emergence Analysis")
    lines.append("")
    lines.append("Emergence criterion: mean gain > 0 AND >= 60% seeds positive.")
    lines.append("")

    emerge_count = 0
    total_configs = 0

    # Emergence matrix
    lines.append("### w1 x tau Emergence Matrix\n")
    header = "| w1 \\ tau | " + " | ".join(f"{t:.1f}" for t in tau_values) + " |"
    sep = "|" + "---|" * (len(tau_values) + 1)
    lines.append(header)
    lines.append(sep)

    per_w1_emerge = {}
    per_tau_emerge = {}

    for w1 in w1_values:
        row_cells = [f"**{w1:.1f}**"]
        per_w1_emerge[w1] = 0
        for tau in tau_values:
            if tau not in per_tau_emerge:
                per_tau_emerge[tau] = 0
            gains = [r['gain'] for r in results
                     if r['w1'] == w1 and r['tau'] == tau]
            mean_g = np.mean(gains)
            frac_pos = sum(1 for g in gains if g > 0) / len(gains)
            emerged = mean_g > 0 and frac_pos >= 0.6
            total_configs += 1
            if emerged:
                emerge_count += 1
                per_w1_emerge[w1] += 1
                per_tau_emerge[tau] += 1
                row_cells.append(f"+{mean_g:.3f} ({frac_pos*100:.0f}%)")
            else:
                row_cells.append(f"{mean_g:+.3f} ({frac_pos*100:.0f}%)")
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append(f"**Overall emergence rate: {emerge_count}/{total_configs} "
                 f"({100*emerge_count/total_configs:.0f}%)**")
    lines.append("")

    # Per-w1
    lines.append("### Per-w1 Emergence Rate\n")
    lines.append("| w1 | Emergence |")
    lines.append("|---|---|")
    for w1 in w1_values:
        n_tau = len(tau_values)
        lines.append(f"| {w1:.1f} | {per_w1_emerge[w1]}/{n_tau} |")

    lines.append("")

    # Per-tau
    lines.append("### Per-tau Emergence Rate\n")
    lines.append("| tau | Emergence |")
    lines.append("|---|---|")
    for tau in tau_values:
        n_w1 = len(w1_values)
        lines.append(f"| {tau:.1f} | {per_tau_emerge[tau]}/{n_w1} |")

    lines.append("")

    # Best config
    best = None
    best_gain = -999
    for w1 in w1_values:
        for tau in tau_values:
            gains = [r['gain'] for r in results
                     if r['w1'] == w1 and r['tau'] == tau]
            mg = np.mean(gains)
            if mg > best_gain:
                best_gain = mg
                best = (w1, tau, mg)

    if best:
        lines.append(f"**Best config:** w1={best[0]:.1f}, tau={best[1]:.1f}, "
                      f"mean gain=+{best[2]:.4f}")

    lines.append("")

    # Comparison note (static emergence recomputed from the static sweep CSV with
    # the App-D criterion: mean gain > 0 AND >= 60% of models positive)
    lines.append("## Comparison: Static vs VN\n")
    static_csv = os.path.join(results_dir, 'sweep_hyperparams.csv')
    if os.path.exists(static_csv):
        static_cfgs = {}
        with open(static_csv) as f:
            for row in csv.DictReader(f):
                key = (row['w1'], row['w2'], row['tau'])
                static_cfgs.setdefault(key, []).append(float(row['gain']))
        s_emerge = sum(
            1 for gains in static_cfgs.values()
            if sum(gains) / len(gains) > 0
            and sum(1 for g in gains if g > 0) / len(gains) >= 0.6)
        lines.append(f"Static sweep (from existing results): {s_emerge}/{len(static_cfgs)} "
                     f"configs = {100 * s_emerge / len(static_cfgs):.0f}% emergence")
    else:
        lines.append("Static sweep: sweep_hyperparams.csv not found at report time")
    lines.append(f"VN sweep (this experiment): {emerge_count}/{total_configs} configs = "
                 f"{100*emerge_count/total_configs:.0f}% emergence")

    report_path = os.path.join(results_dir, 'REPORT_VN_SWEEP.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved {report_path}")


if __name__ == '__main__':
    main()
