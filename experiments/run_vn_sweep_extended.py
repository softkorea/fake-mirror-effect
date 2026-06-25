"""Extended VN sweep: w1  in  {0.5, 0.7, 1.0} to test if self-correction
emerges even under uniform or near-uniform time weighting.

If w1=1.0 shows positive gain -> "induced, not spontaneous" limitation weakens.
"""

import sys
import os
import csv
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Pool
from src.network import RecurrentMLP
from src.training import generate_data_variable_noise, train_vn
from src.metrics import compute_all_metrics_vn


def run_single_config(args):
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
        'seed': seed,
        'w1': w1,
        'w2': w2,
        'tau': tau,
        'gain': metrics['gain'],
        'acc_t1': metrics['acc_t1'],
        'acc_t3': metrics['acc_t3'],
        'final_loss': losses[-1] if losses else float('nan'),
    }


def main():
    os.makedirs('results', exist_ok=True)

    W1_VALUES = [0.5, 0.7, 1.0]
    TAU_VALUES = [1.0, 1.5, 2.0, 3.0]
    W2_VALUES = [0.2, 1.0]  # 0.2 = paper default, 1.0 = truly uniform
    N_SEEDS = 10

    configs = []
    for w1 in W1_VALUES:
        for w2 in W2_VALUES:
            for tau in TAU_VALUES:
                for seed in range(N_SEEDS):
                    configs.append((seed, w1, w2, tau))

    print(f"[VN-Sweep-Extended] {len(configs)} configs "
          f"(w1={W1_VALUES}, w2={W2_VALUES}, tau={TAU_VALUES}, {N_SEEDS} seeds)")

    n_workers = max(1, os.cpu_count() - 4)
    with Pool(n_workers) as pool:
        results = pool.map(run_single_config, configs)

    # Save CSV
    csv_path = 'results/sweep_vn_extended.csv'
    fields = ['seed', 'w1', 'w2', 'tau', 'gain', 'acc_t1', 'acc_t3', 'final_loss']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS: Does self-correction emerge with high w1 under VN?")
    print(f"{'='*60}")
    print(f"\n  {'w1':>4s}  {'w2':>4s}  {'tau':>4s}  {'mean gain':>10s}  {'positive':>8s}  {'emerge?':>8s}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}")

    # Summarize per (w1, w2, tau) so w2=0.2 and w2=1.0 are NOT mixed (each cell
    # is N_SEEDS runs). Previously the loop filtered only (w1, tau), conflating
    # the two w2 settings and mislabeling the count as /10 when it held 2*N_SEEDS.
    for w1 in W1_VALUES:
        for w2 in W2_VALUES:
            for tau in TAU_VALUES:
                gains = [r['gain'] for r in results
                         if r['w1'] == w1 and r['w2'] == w2 and r['tau'] == tau]
                mean_g = np.mean(gains)
                n_pos = sum(1 for g in gains if g > 0)
                emerge = "YES" if mean_g > 0 and n_pos >= 6 else "no"
                print(f"  {w1:4.1f}  {w2:4.1f}  {tau:4.1f}  {mean_g:+10.4f}  {n_pos:>5d}/{N_SEEDS:<2d} {emerge:>8s}")

    # Key result: TRULY uniform weighting (w1 = w2 = 1.0) - not w1=1.0 with mixed w2.
    uniform_gains = [r['gain'] for r in results if r['w1'] == 1.0 and r['w2'] == 1.0]
    print(f"\n  w1=w2=1.0 (truly uniform): mean gain = {np.mean(uniform_gains):+.4f} "
          f"+/- {np.std(uniform_gains):.4f}")
    print(f"  {sum(1 for g in uniform_gains if g > 0)}/{len(uniform_gains)} positive")

    if np.mean(uniform_gains) > 0:
        print("\n  -> Self-correction emerges even under UNIFORM time weighting!")
        print("    'Induced, not spontaneous' limitation is substantially weakened.")
    else:
        print("\n  -> Self-correction requires time-weighted loss (w1 < 1.0).")
        print("    'Induced' limitation stands, but VN reduces sensitivity.")

    print(f"\n[Saved] {csv_path}")

    # Generate combined REPORT_VN_SWEEP.md (targeted + extended)
    generate_combined_report(os.path.dirname(csv_path))


def generate_combined_report(results_dir):
    """Combine targeted + extended VN sweep CSVs into REPORT_VN_SWEEP.md."""
    all_results = []
    for fname in ['sweep_vn_hyperparams.csv', 'sweep_vn_extended.csv']:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found, cannot generate combined report")
            return
        with open(path) as f:
            for row in csv.DictReader(f):
                all_results.append({
                    'w1': float(row['w1']),
                    'w2': float(row.get('w2', 0.2)),
                    'tau': float(row['tau']),
                    'gain': float(row['gain']),
                })

    w1_values = sorted(set(r['w1'] for r in all_results))
    w2_values = sorted(set(r['w2'] for r in all_results))
    tau_values = sorted(set(r['tau'] for r in all_results))
    n_configs = len(set((r['w1'], r['w2'], r['tau']) for r in all_results))

    lines = []
    lines.append('# VN Hyperparameter Sweep Report (Combined)')
    lines.append('')
    lines.append('## Configuration')
    lines.append(f'- w1: {w1_values}')
    lines.append(f'- w2: {w2_values}')
    lines.append(f'- tau: {tau_values}')
    lines.append(f'- Seeds per config: 10')
    lines.append(f'- Total runs: {len(all_results)}')
    lines.append(f'- Unique hyperparameter configurations: {n_configs}')
    lines.append('')

    # Emergence
    lines.append('## Emergence Analysis')
    lines.append('')
    lines.append('Emergence criterion: mean gain > 0 AND >= 60% seeds positive.')
    lines.append('')

    emerge_count = 0
    total_configs = 0
    for w1 in w1_values:
        for w2 in w2_values:
            for tau in tau_values:
                gains = [r['gain'] for r in all_results
                         if r['w1'] == w1 and r['w2'] == w2 and r['tau'] == tau]
                if not gains:
                    continue
                total_configs += 1
                if np.mean(gains) > 0 and sum(1 for g in gains if g > 0) / len(gains) >= 0.6:
                    emerge_count += 1

    lines.append(f'**Overall emergence rate: {emerge_count}/{total_configs} '
                 f'({100 * emerge_count / total_configs:.0f}%)**')
    lines.append('')

    # Targeted grid
    targeted_w1 = [w for w in w1_values if w <= 0.3]
    lines.append(f'### Targeted Grid (w2=0.2, w1 <= 0.3): '
                 f'{len(targeted_w1) * len(tau_values)} configurations')
    lines.append('')
    header = '| w1 \\\\ tau | ' + ' | '.join(f'{t:.1f}' for t in tau_values) + ' |'
    sep = '|' + '---|' * (len(tau_values) + 1)
    lines.append(header)
    lines.append(sep)
    for w1 in targeted_w1:
        cells = [f'**{w1:.1f}**']
        for tau in tau_values:
            gains = [r['gain'] for r in all_results
                     if r['w1'] == w1 and r['w2'] == 0.2 and r['tau'] == tau]
            if gains:
                mg = np.mean(gains)
                fp = sum(1 for g in gains if g > 0) / len(gains)
                cells.append(f'+{mg:.3f} ({fp * 100:.0f}%)')
            else:
                cells.append('--')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    # Extended grid
    ext_w1 = [w for w in w1_values if w > 0.3]
    n_ext = sum(1 for w1 in ext_w1 for w2 in w2_values for tau in tau_values
                if any(r['w1'] == w1 and r['w2'] == w2 and r['tau'] == tau
                       for r in all_results))
    lines.append(f'### Extended Grid (w1 >= 0.5, w2 in {{0.2, 1.0}}): '
                 f'{n_ext} configurations')
    lines.append('')
    lines.append('| w1 | w2 | tau | Mean Gain | % Positive | N |')
    lines.append('|---|---|---|---|---|---|')
    for w1 in ext_w1:
        for w2 in w2_values:
            for tau in tau_values:
                gains = [r['gain'] for r in all_results
                         if r['w1'] == w1 and r['w2'] == w2 and r['tau'] == tau]
                if gains:
                    mg = np.mean(gains)
                    fp = sum(1 for g in gains if g > 0) / len(gains)
                    lines.append(f'| {w1:.1f} | {w2:.1f} | {tau:.1f} | '
                                 f'{mg:+.4f} | {fp * 100:.0f}% | {len(gains)} |')
    lines.append('')

    # Uniform weighting
    lines.append('### Truly Uniform Time Weighting (w1=w2=1.0)')
    lines.append('')
    uniform_gains = [r['gain'] for r in all_results
                     if r['w1'] == 1.0 and r['w2'] == 1.0]
    if uniform_gains:
        lines.append(f'- N = {len(uniform_gains)} models across '
                     f'{len(tau_values)} tau values')
        lines.append(f'- Mean gain: {np.mean(uniform_gains):+.4f}')
        lines.append(f'- All positive: {all(g > 0 for g in uniform_gains)}')
        lines.append(f'- Min gain: {min(uniform_gains):+.4f}')
    lines.append('')

    # Static comparison
    from collections import defaultdict
    static_path = os.path.join(results_dir, 'sweep_hyperparams.csv')
    lines.append('## Comparison: Static vs VN')
    lines.append('')
    if os.path.exists(static_path):
        static_configs = defaultdict(list)
        with open(static_path) as f:
            for row in csv.DictReader(f):
                key = (row['w1'], row['w2'], row['tau'])
                static_configs[key].append(float(row['gain']))
        s_total = len(static_configs)
        s_emerged = sum(1 for gains in static_configs.values()
                        if np.mean(gains) > 0
                        and sum(1 for g in gains if g > 0) / len(gains) >= 0.6)
        lines.append(f'- Static sweep: {s_emerged}/{s_total} configs = '
                     f'{100 * s_emerged / s_total:.0f}% emergence')
    lines.append(f'- VN sweep (combined): {emerge_count}/{total_configs} configs = '
                 f'{100 * emerge_count / total_configs:.0f}% emergence')
    lines.append('')

    report_path = os.path.join(results_dir, 'REPORT_VN_SWEEP.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  [Saved] {report_path} ({total_configs} configs, {emerge_count} emerged)")


if __name__ == '__main__':
    main()
