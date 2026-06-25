"""Training Dynamics: When does the feedback contract form?

Track correction gain (Baseline, A, C1, C2) at checkpoints during training
to reveal whether feedback-contract specificity develops gradually or abruptly.

Uses multiprocessing for parallel seed evaluation.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multiprocessing import Pool
import csv

from src.network import RecurrentMLP
from src.training import (
    generate_data, generate_data_variable_noise,
    train, train_vn,
    compute_batch_loss_and_gradients, compute_batch_loss_and_gradients_vn,
)
from src.ablation import forward_sequence_with_clone


def _eval_gains(net, clone, X_test, y_test, task='static', X_test_seq=None):
    """Evaluate Baseline, A, C1, C2 gains for one model at one checkpoint."""
    n = len(X_test) if task == 'static' else len(X_test_seq)

    if task == 'static':
        # Baseline
        correct_t1_b, correct_t3_b = 0, 0
        for i in range(n):
            outputs, _ = net.forward_sequence(X_test[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_b += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_b += 1
        gain_baseline = correct_t3_b / n - correct_t1_b / n

        # Group A
        from src.ablation import ablate_recurrent, deep_copy_weights, restore_weights
        saved = deep_copy_weights(net)
        ablate_recurrent(net)
        correct_t1_a, correct_t3_a = 0, 0
        for i in range(n):
            outputs, _ = net.forward_sequence(X_test[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_a += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_a += 1
        gain_a = correct_t3_a / n - correct_t1_a / n
        restore_weights(net, saved)

        # C1
        net.enable_scrambled_feedback(seed=999)
        correct_t1_c1, correct_t3_c1 = 0, 0
        for i in range(n):
            outputs, _ = net.forward_sequence(X_test[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_c1 += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_c1 += 1
        gain_c1 = correct_t3_c1 / n - correct_t1_c1 / n
        net.disable_scrambled_feedback()

        # C2
        correct_t1_c2, correct_t3_c2 = 0, 0
        for i in range(n):
            outputs, _ = forward_sequence_with_clone(net, clone, X_test[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_c2 += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_c2 += 1
        gain_c2 = correct_t3_c2 / n - correct_t1_c2 / n

    else:  # VN
        from src.ablation import forward_sequence_with_clone_vn
        correct_t1_b, correct_t3_b = 0, 0
        for i in range(n):
            outputs, _ = net.forward_sequence_vn(X_test_seq[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_b += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_b += 1
        gain_baseline = correct_t3_b / n - correct_t1_b / n

        from src.ablation import ablate_recurrent, deep_copy_weights, restore_weights
        saved = deep_copy_weights(net)
        ablate_recurrent(net)
        correct_t1_a, correct_t3_a = 0, 0
        for i in range(n):
            outputs, _ = net.forward_sequence_vn(X_test_seq[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_a += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_a += 1
        gain_a = correct_t3_a / n - correct_t1_a / n
        restore_weights(net, saved)

        net.enable_scrambled_feedback(seed=999)
        correct_t1_c1, correct_t3_c1 = 0, 0
        for i in range(n):
            outputs, _ = net.forward_sequence_vn(X_test_seq[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_c1 += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_c1 += 1
        gain_c1 = correct_t3_c1 / n - correct_t1_c1 / n
        net.disable_scrambled_feedback()

        correct_t1_c2, correct_t3_c2 = 0, 0
        for i in range(n):
            outputs, _ = forward_sequence_with_clone_vn(net, clone, X_test_seq[i], T=3)
            if np.argmax(outputs[0]) == np.argmax(y_test[i]): correct_t1_c2 += 1
            if np.argmax(outputs[2]) == np.argmax(y_test[i]): correct_t3_c2 += 1
        gain_c2 = correct_t3_c2 / n - correct_t1_c2 / n

    return {
        'gain_baseline': gain_baseline, 'gain_a': gain_a,
        'gain_c1': gain_c1, 'gain_c2': gain_c2,
        'acc_t1': correct_t1_b / n, 'acc_t3': correct_t3_b / n,
    }


def _process_seed(args):
    """Train one seed with checkpoints, evaluate at each."""
    seed, task, checkpoints, total_epochs = args
    time_weights = [0.0, 0.2, 1.0]

    if task == 'static':
        X_train, y_train = generate_data(200, noise_level=0.5, seed=seed)
        X_test, y_test = generate_data(200, noise_level=0.5, seed=seed + 1000)
        X_train_seq, X_test_seq = None, None
    else:
        X_train_seq, y_train = generate_data_variable_noise(200, noise_level=0.5, seed=seed)
        X_test_seq, y_test = generate_data_variable_noise(200, noise_level=0.5, seed=seed + 1000)
        X_train, X_test = None, None

    # Train target
    net = RecurrentMLP(seed=seed)
    # Train clone to final state (fixed reference)
    clone = RecurrentMLP(seed=seed + 100)
    if task == 'static':
        X_clone, y_clone = generate_data(200, noise_level=0.5, seed=seed + 100)
        train(clone, X_clone, y_clone, epochs=total_epochs, lr=0.01, time_weights=time_weights)
    else:
        X_clone_seq, y_clone = generate_data_variable_noise(200, noise_level=0.5, seed=seed + 100)
        train_vn(clone, X_clone_seq, y_clone, epochs=total_epochs, lr=0.01, time_weights=time_weights)

    results = []
    prev_epoch = 0

    for cp in checkpoints:
        # Train from prev_epoch to cp
        epochs_to_train = cp - prev_epoch
        if epochs_to_train > 0:
            if task == 'static':
                train(net, X_train, y_train, epochs=epochs_to_train, lr=0.01,
                      time_weights=time_weights)
            else:
                train_vn(net, X_train_seq, y_train, epochs=epochs_to_train, lr=0.01,
                         time_weights=time_weights)
        prev_epoch = cp

        # Evaluate
        r = _eval_gains(net, clone, X_test, y_test, task=task, X_test_seq=X_test_seq)
        r['seed'] = seed
        r['epoch'] = cp
        r['task'] = task
        results.append(r)

    return results


def main():
    os.makedirs('results', exist_ok=True)

    # Checkpoints extended to the 1000-epoch budget (standardized protocol): the target net
    # is trained incrementally up to the LAST checkpoint, so this array (not
    # total_epochs alone) sets how far the trajectory is logged. Stopping at 500
    # would truncate the dynamics figure and miss the ~1000-epoch convergence peak.
    checkpoints = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500,
                   600, 700, 800, 900, 1000]
    total_epochs = 1000
    n_workers = max(1, os.cpu_count() - 4)
    print(f"Workers: {n_workers}")

    all_results = []

    for task in ['static', 'vn']:
        print(f"\n=== {task.upper()} Training Dynamics ===")
        jobs = [(seed, task, checkpoints, total_epochs) for seed in range(20)]
        with Pool(n_workers) as pool:
            seed_results = pool.map(_process_seed, jobs)

        for sr in seed_results:
            all_results.extend(sr)
            seed = sr[0]['seed']
            final = sr[-1]
            print(f"  seed={seed}: final baseline={final['gain_baseline']:+.3f} "
                  f"C1={final['gain_c1']:+.3f} C2={final['gain_c2']:+.3f}")

    # Save CSV
    fieldnames = ['task', 'seed', 'epoch', 'acc_t1', 'acc_t3',
                  'gain_baseline', 'gain_a', 'gain_c1', 'gain_c2']
    with open('results/training_dynamics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Summary report
    with open('results/REPORT_TRAINING_DYNAMICS.md', 'w', encoding='utf-8') as f:
        f.write("# Training Dynamics: When Does the Feedback Contract Form?\n\n")

        for task in ['static', 'vn']:
            f.write(f"## {task.upper()}\n\n")
            f.write("| Epoch | Baseline Gain | Group A | C1 | C2 |\n")
            f.write("|-------|---------------|---------|------|------|\n")
            for cp in checkpoints:
                rows = [r for r in all_results if r['task'] == task and r['epoch'] == cp]
                bg = np.mean([r['gain_baseline'] for r in rows])
                ag = np.mean([r['gain_a'] for r in rows])
                c1g = np.mean([r['gain_c1'] for r in rows])
                c2g = np.mean([r['gain_c2'] for r in rows])
                f.write(f"| {cp} | {bg:+.4f} | {ag:+.4f} | {c1g:+.4f} | {c2g:+.4f} |\n")
            f.write("\n")

        # Onset detection: first epoch where baseline gain > 0.01
        f.write("## Onset Detection\n\n")
        for task in ['static', 'vn']:
            for seed in range(20):
                rows = sorted([r for r in all_results if r['task'] == task and r['seed'] == seed],
                              key=lambda r: r['epoch'])
                onset = None
                for r in rows:
                    if r['gain_baseline'] > 0.01:
                        onset = r['epoch']
                        break
                f.write(f"  {task} seed={seed}: onset at epoch {onset}\n")
            f.write("\n")

    print("\nSaved: results/training_dynamics.csv, results/REPORT_TRAINING_DYNAMICS.md")


if __name__ == '__main__':
    main()
