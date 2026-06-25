"""WS2: Feedback Interpolation Experiment.

feedback_t = alpha * y_self_{t-1} + (1-alpha) * y_other_{t-1}
alpha  in  {0.0, 0.1, ..., 1.0}, three types: zero, shuffle, clone.
20 models x 11 alphas x 3 types = 660 evaluations.

Multiprocessing for parallel evaluation.

Usage:
    python experiments/run_interpolation.py
"""

import sys
import os
import csv
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import generate_data, train, softmax
from src.ablation import forward_sequence_interpolated


# -- Training (sequential, seeded) --

def train_model(seed, noise_level=0.5, n_samples=200, epochs=1000,
                lr=0.01, tau=2.0):
    """Train a RecurrentMLP and return it."""
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=tau)
    X, y = generate_data(n_samples, noise_level=noise_level, seed=seed)
    train(net, X, y, epochs=epochs, lr=lr,
          time_weights=[0.0, 0.2, 1.0])
    return net


# -- Evaluation worker --

def evaluate_single(args):
    """Evaluate one (model_seed, alpha, interp_type) combination.

    Args is a tuple for multiprocessing compatibility:
        (target_weights, clone_weights, seed_model, alpha, interp_type,
         X_test, y_test, tau)

    Returns dict row.
    """
    (target_params, clone_params, seed_model, alpha, interp_type,
     X_test, y_test, tau) = args

    # Reconstruct models from serialized params
    target = _reconstruct_model(target_params, tau)
    clone = _reconstruct_model(clone_params, tau) if clone_params else None

    n = len(X_test)
    correct_t1 = 0
    correct_t3 = 0

    for i in range(n):
        kwargs = {}
        if interp_type == 'clone':
            kwargs['clone_net'] = clone
        if interp_type == 'shuffle':
            kwargs['shuffle_seed'] = seed_model * 1000 + i

        outputs, _ = forward_sequence_interpolated(
            target, X_test[i], alpha=alpha, interp_type=interp_type,
            **kwargs)

        true_cls = np.argmax(y_test[i])
        if np.argmax(outputs[0]) == true_cls:
            correct_t1 += 1
        if np.argmax(outputs[2]) == true_cls:
            correct_t3 += 1

    acc_t1 = correct_t1 / n
    acc_t3 = correct_t3 / n
    gain = acc_t3 - acc_t1

    return {
        'seed_model': seed_model,
        'alpha': f"{alpha:.1f}",
        'interp_type': interp_type,
        'acc_t1': f"{acc_t1:.6f}",
        'acc_t3': f"{acc_t3:.6f}",
        'gain': f"{gain:.6f}",
    }


def _serialize_model(net):
    """Extract model parameters for pickling."""
    return {
        'W_ih1': net.W_ih1.copy(),
        'b_h1': net.b_h1.copy(),
        'W_h1h2': net.W_h1h2.copy(),
        'b_h2': net.b_h2.copy(),
        'W_h2o': net.W_h2o.copy(),
        'b_out': net.b_out.copy(),
        'W_rec': net.W_rec.copy(),
    }


def _reconstruct_model(params, tau=2.0):
    """Reconstruct a RecurrentMLP from serialized parameters."""
    net = RecurrentMLP(seed=0, feedback_tau=tau)  # seed doesn't matter
    net.W_ih1[:] = params['W_ih1']
    net.b_h1[:] = params['b_h1']
    net.W_h1h2[:] = params['W_h1h2']
    net.b_h2[:] = params['b_h2']
    net.W_h2o[:] = params['W_h2o']
    net.b_out[:] = params['b_out']
    net.W_rec[:] = params['W_rec']
    return net


def main():
    seeds = list(range(20))
    donor_seeds = list(range(100, 120))
    alphas = [round(a * 0.1, 1) for a in range(11)]  # 0.0, 0.1, ..., 1.0
    interp_types = ['zero', 'shuffle', 'clone']
    noise_level = 0.5
    n_test = 200
    tau = 2.0

    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results')
    csv_path = os.path.join(results_dir, 'interpolation.csv')

    total_start = time.time()

    print("=" * 60)
    print("WS2: Feedback Interpolation Experiment")
    print(f"  {len(seeds)} models x {len(alphas)} alphas x "
          f"{len(interp_types)} types = {len(seeds)*len(alphas)*len(interp_types)} evals")
    print("=" * 60)

    # Phase 1: Train models
    print("\n[Phase 1] Training models...")
    target_models = {}
    target_params = {}
    for s in seeds:
        t0 = time.time()
        net = train_model(s, noise_level=noise_level)
        target_models[s] = net
        target_params[s] = _serialize_model(net)
        print(f"  target seed={s}: {time.time()-t0:.1f}s")

    clone_params = {}
    for ds in donor_seeds:
        t0 = time.time()
        net = train_model(ds, noise_level=noise_level)
        clone_params[ds] = _serialize_model(net)
        print(f"  donor  seed={ds}: {time.time()-t0:.1f}s")

    # Generate test data
    test_data = {}
    for s in seeds:
        X_test, y_test = generate_data(n_test, noise_level=noise_level,
                                       seed=1000 + s)
        test_data[s] = (X_test, y_test)

    # Phase 2: Build evaluation tasks
    print("\n[Phase 2] Parallel evaluation...")
    tasks = []
    for s_idx, s in enumerate(seeds):
        X_test, y_test = test_data[s]
        ds = donor_seeds[s_idx]
        for alpha in alphas:
            for itype in interp_types:
                cp = clone_params[ds] if itype == 'clone' else None
                tasks.append((
                    target_params[s], cp, s, alpha, itype,
                    X_test, y_test, tau
                ))

    n_workers = max(1, cpu_count() - 4)
    print(f"  {len(tasks)} tasks, {n_workers} workers")

    with Pool(n_workers) as pool:
        results = pool.map(evaluate_single, tasks)

    # Write CSV
    fieldnames = ['seed_model', 'alpha', 'interp_type',
                  'acc_t1', 'acc_t3', 'gain']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.time() - total_start
    print(f"\nDone! {len(results)} rows -> {csv_path}")
    print(f"Total time: {total_time:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("Gain vs alpha (mean across 20 models):")
    print(f"{'='*60}")
    print(f"{'alpha':>6s} {'zero':>10s} {'shuffle':>10s} {'clone':>10s}")
    for alpha in alphas:
        vals = {}
        for itype in interp_types:
            gains = [float(r['gain']) for r in results
                     if float(r['alpha']) == alpha and r['interp_type'] == itype]
            vals[itype] = np.mean(gains)
        print(f"{alpha:6.1f} {vals['zero']:+10.4f} "
              f"{vals['shuffle']:+10.4f} {vals['clone']:+10.4f}")


if __name__ == '__main__':
    main()
