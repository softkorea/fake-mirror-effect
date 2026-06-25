"""Bonus: C2 Cross-Pairing Analysis.

20 VN targets x 20 donors = 400 forward pass evaluations.
No retraining, just clone feedback forward passes.

Output:
    results/cross_pairing_vn.csv
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
from src.metrics import compute_all_metrics_with_clone_vn


def _train_vn(seed):
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=2.0)
    X_seq, y = generate_data_variable_noise(200, noise_level=0.5, T=3, seed=seed)
    train_vn(net, X_seq, y, epochs=1000, lr=0.01, time_weights=[0.0, 0.2, 1.0])
    weights = {k: v.copy() for k, v in net.get_all_weights().items()}
    biases = {k: v.copy() for k, v in net.get_all_biases().items()}
    return seed, weights, biases


def _rebuild(seed, weights, biases):
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


def main():
    target_seeds = list(range(20))
    donor_seeds = list(range(100, 120))
    n_workers = max(1, os.cpu_count() - 4)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    total_start = time.time()
    print("=" * 60)
    print("Bonus: C2 Cross-Pairing Analysis")
    print(f"  {len(target_seeds)} targets x {len(donor_seeds)} donors = "
          f"{len(target_seeds)*len(donor_seeds)} evaluations")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    # Train all models in parallel
    all_seeds = target_seeds + donor_seeds
    print(f"\n[Phase 1] Training {len(all_seeds)} VN models...")
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(_train_vn, all_seeds)
    print(f"  Done in {time.time()-t0:.0f}s")

    nets = {}
    for s, w, b in results:
        nets[s] = _rebuild(s, w, b)

    # Test data - shared across all donor-target pairings for comparability.
    # Unlike the main experiment (per-model held-out sets), this uses a single
    # common test set to isolate donor-target identity effects from data variation.
    X_seq_test, y_test = generate_data_variable_noise(
        200, noise_level=0.5, T=3, seed=999)

    # Evaluate all cross-pairings
    print("\n[Phase 2] Evaluating 400 cross-pairings...")
    rows = []
    for ts in target_seeds:
        for ds in donor_seeds:
            metrics = compute_all_metrics_with_clone_vn(
                nets[ts], nets[ds], X_seq_test, y_test)
            rows.append({
                'target_seed': ts,
                'donor_seed': ds,
                'acc_t1': f"{metrics['acc_t1']:.6f}",
                'acc_t3': f"{metrics['acc_t3']:.6f}",
                'gain': f"{metrics['gain']:.6f}",
            })
        print(f"  target={ts}: done")

    csv_path = os.path.join(results_dir, 'cross_pairing_vn.csv')
    fieldnames = ['target_seed', 'donor_seed', 'acc_t1', 'acc_t3', 'gain']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - total_start
    print(f"\nWritten {len(rows)} rows to {csv_path}")
    print(f"Total time: {total_time/60:.1f} min")

    # Quick summary
    gains = [float(r['gain']) for r in rows]
    print(f"\nSummary:")
    print(f"  Mean gain: {np.mean(gains):+.4f} +/- {np.std(gains):.4f}")
    print(f"  Variance across donors (per target):")
    for ts in target_seeds:
        ts_gains = [float(r['gain']) for r in rows if r['target_seed'] == ts]
        print(f"    target={ts}: mean={np.mean(ts_gains):+.4f}, "
              f"std={np.std(ts_gains):.4f}")
    print("Done!")


if __name__ == '__main__':
    main()
