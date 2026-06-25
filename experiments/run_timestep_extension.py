"""Task 1: Timestep Extension Sweep.

Evaluate models trained at T=3 by extending the unroll to T={3,5,7,10,15,20}.
Forward pass only, no retraining.

Run both Static and VN (training parallelised via multiprocessing).
4 conditions: Baseline, Group A, Group C1 (30 seeds avg), Group C2 (clone).

Output:
    results/timestep_extension_static.csv
    results/timestep_extension_vn.csv
    results/fig_timestep_extension.png
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
from src.ablation import (
    deep_copy_weights, restore_weights,
    forward_sequence_with_clone, forward_sequence_with_clone_vn,
)
from src.training import softmax


# -- Training workers (top-level for pickling) --

def _train_static(seed):
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=2.0)
    X, y = generate_data(200, noise_level=0.5, seed=seed)
    train(net, X, y, epochs=1000, lr=0.01, time_weights=[0.0, 0.2, 1.0])
    # Return weights for reconstruction
    weights = {}
    for k, v in net.get_all_weights().items():
        weights[k] = v.copy()
    biases = {}
    for k, v in net.get_all_biases().items():
        biases[k] = v.copy()
    return seed, weights, biases


def _train_vn(seed):
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=2.0)
    X_seq, y = generate_data_variable_noise(200, noise_level=0.5, T=3, seed=seed)
    train_vn(net, X_seq, y, epochs=1000, lr=0.01, time_weights=[0.0, 0.2, 1.0])
    weights = {k: v.copy() for k, v in net.get_all_weights().items()}
    biases = {k: v.copy() for k, v in net.get_all_biases().items()}
    return seed, weights, biases


def _rebuild_net(seed, weights, biases):
    """Rebuild a RecurrentMLP from saved weights/biases."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=2.0)
    net.W_ih1[:] = weights['input_to_h1']
    net.W_h1h2[:] = weights['h1_to_h2']
    net.W_h2o[:] = weights['h2_to_output']
    net.W_rec[:] = weights['recurrent']
    net.b_h1[:] = biases['b_h1']
    net.b_h2[:] = biases['b_h2']
    net.b_out[:] = biases['b_out']
    return net


# -- Evaluation helpers --

def evaluate_per_timestep_accuracy(outputs_list, y, T_max):
    n = len(outputs_list)
    acc = []
    for t in range(T_max):
        correct = sum(1 for i in range(n)
                      if np.argmax(outputs_list[i][t]) == np.argmax(y[i]))
        acc.append(correct / n)
    return acc


def run_static_evaluation(net, X_test, y_test, T_max, condition,
                          clone_net=None, n_c1_repeats=30):
    saved = deep_copy_weights(net)
    n = len(X_test)

    if condition == 'baseline':
        all_outputs = [net.forward_sequence(X_test[i], T=T_max)[0] for i in range(n)]
        accs = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)

    elif condition == 'group_a':
        net.disable_recurrent_loop()
        all_outputs = [net.forward_sequence(X_test[i], T=T_max)[0] for i in range(n)]
        accs = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)
        net.enable_recurrent_loop()

    elif condition == 'group_c1':
        accs_all = np.zeros((n_c1_repeats, T_max))
        for rep in range(n_c1_repeats):
            net.enable_scrambled_feedback(seed=rep)
            all_outputs = [net.forward_sequence(X_test[i], T=T_max)[0] for i in range(n)]
            accs_all[rep] = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)
            net.disable_scrambled_feedback()
        accs = list(np.mean(accs_all, axis=0))

    elif condition == 'group_c2':
        all_outputs = [forward_sequence_with_clone(net, clone_net, X_test[i], T=T_max)[0]
                       for i in range(n)]
        accs = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)

    restore_weights(net, saved)
    return accs


def run_vn_evaluation(net, X_seq_test, y_test, T_max, condition,
                      clone_net=None, n_c1_repeats=30):
    saved = deep_copy_weights(net)
    n = len(X_seq_test)

    if condition == 'baseline':
        all_outputs = [net.forward_sequence_vn(X_seq_test[i], T=T_max)[0] for i in range(n)]
        accs = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)

    elif condition == 'group_a':
        net.disable_recurrent_loop()
        all_outputs = [net.forward_sequence_vn(X_seq_test[i], T=T_max)[0] for i in range(n)]
        accs = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)
        net.enable_recurrent_loop()

    elif condition == 'group_c1':
        accs_all = np.zeros((n_c1_repeats, T_max))
        for rep in range(n_c1_repeats):
            net.enable_scrambled_feedback(seed=rep)
            all_outputs = [net.forward_sequence_vn(X_seq_test[i], T=T_max)[0] for i in range(n)]
            accs_all[rep] = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)
            net.disable_scrambled_feedback()
        accs = list(np.mean(accs_all, axis=0))

    elif condition == 'group_c2':
        all_outputs = [forward_sequence_with_clone_vn(net, clone_net, X_seq_test[i], T=T_max)[0]
                       for i in range(n)]
        accs = evaluate_per_timestep_accuracy(all_outputs, y_test, T_max)

    restore_weights(net, saved)
    return accs


def main():
    seeds = list(range(20))
    donor_seeds = list(range(100, 120))
    T_values = [3, 5, 7, 10, 15, 20]
    conditions = ['baseline', 'group_a', 'group_c1', 'group_c2']
    n_test = 200
    noise_level = 0.5
    n_workers = max(1, os.cpu_count() - 4)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    fieldnames = ['seed', 'condition', 'T_max', 't', 'accuracy']

    total_start = time.time()
    print("=" * 60)
    print("Task 1: Timestep Extension Sweep")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    # -- Train all models in parallel --
    all_train_seeds = seeds + donor_seeds  # 40 models total

    print(f"\n[Phase 1] Training 40 static models ({n_workers} workers)...")
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        static_results = pool.map(_train_static, all_train_seeds)
    print(f"  Done in {time.time()-t0:.0f}s")
    static_nets = {}
    for s, w, b in static_results:
        static_nets[s] = _rebuild_net(s, w, b)

    print(f"\n[Phase 2] Training 40 VN models ({n_workers} workers)...")
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        vn_results = pool.map(_train_vn, all_train_seeds)
    print(f"  Done in {time.time()-t0:.0f}s")
    vn_nets = {}
    for s, w, b in vn_results:
        vn_nets[s] = _rebuild_net(s, w, b)

    # -- Static evaluation --
    print("\n[Phase 3] Static evaluation...")
    static_rows = []
    for s_idx, s in enumerate(seeds):
        net = static_nets[s]
        clone = static_nets[donor_seeds[s_idx]]
        X_test, y_test = generate_data(n_test, noise_level=noise_level, seed=999 + s)
        for T_max in T_values:
            for cond in conditions:
                accs = run_static_evaluation(
                    net, X_test, y_test, T_max, cond,
                    clone_net=clone if cond == 'group_c2' else None)
                for t in range(T_max):
                    static_rows.append({
                        'seed': s, 'condition': cond, 'T_max': T_max,
                        't': t + 1, 'accuracy': f"{accs[t]:.6f}",
                    })
        print(f"  Model {s}: done")

    csv_static = os.path.join(results_dir, 'timestep_extension_static.csv')
    with open(csv_static, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(static_rows)
    print(f"  Written {len(static_rows)} rows to {csv_static}")

    # -- VN evaluation --
    print("\n[Phase 4] VN evaluation...")
    vn_rows = []
    for s_idx, s in enumerate(seeds):
        net = vn_nets[s]
        clone = vn_nets[donor_seeds[s_idx]]
        for T_max in T_values:
            X_seq_test, y_test = generate_data_variable_noise(
                n_test, noise_level=noise_level, T=T_max, seed=999 + s)
            for cond in conditions:
                accs = run_vn_evaluation(
                    net, X_seq_test, y_test, T_max, cond,
                    clone_net=clone if cond == 'group_c2' else None)
                for t in range(T_max):
                    vn_rows.append({
                        'seed': s, 'condition': cond, 'T_max': T_max,
                        't': t + 1, 'accuracy': f"{accs[t]:.6f}",
                    })
        print(f"  Model {s}: done")

    csv_vn = os.path.join(results_dir, 'timestep_extension_vn.csv')
    with open(csv_vn, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(vn_rows)
    print(f"  Written {len(vn_rows)} rows to {csv_vn}")

    # -- Plot --
    print("\n[Phase 5] Generating figure...")
    generate_figure(csv_static, csv_vn, results_dir)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("Done!")


def generate_figure(csv_static, csv_vn, results_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    conditions = ['baseline', 'group_a', 'group_c1', 'group_c2']
    labels = {'baseline': 'Baseline', 'group_a': 'Group A (no recurrence)',
              'group_c1': 'Group C1 (scrambled)', 'group_c2': 'Group C2 (clone)'}
    colors = {'baseline': '#2196F3', 'group_a': '#F44336',
              'group_c1': '#FF9800', 'group_c2': '#9C27B0'}

    for ax, csv_path, title in [
        (axes[0], csv_static, 'Static Input'),
        (axes[1], csv_vn, 'Variable-Noise Input'),
    ]:
        data = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['condition'], int(row['T_max']), int(row['t']))
                if key not in data:
                    data[key] = []
                data[key].append(float(row['accuracy']))

        T_max = 20
        for cond in conditions:
            ts = list(range(1, T_max + 1))
            means, stds = [], []
            for t in ts:
                vals = data.get((cond, T_max, t), [])
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals) if vals else 0)
            means, stds = np.array(means), np.array(stds)
            ax.plot(ts, means, label=labels[cond], color=colors[cond], linewidth=2)
            ax.fill_between(ts, means - stds, means + stds,
                            alpha=0.15, color=colors[cond])

        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, T_max)

    plt.tight_layout()
    fig_path = os.path.join(results_dir, 'fig_timestep_extension.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fig_path}")


if __name__ == '__main__':
    main()
