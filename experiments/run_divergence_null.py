"""Experiment B: Cosine Divergence Null Baseline.

Provides context for the reported 0.635 cosine divergence between self and clone
W_rec contributions by comparing against:
  (a) Isotropic Gaussian null (norm-matched random vectors)
  (b) Same-class resampled self null (different trial, same class)
  (c) Clone (existing)

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
from src.training import generate_data, train
from src.ablation import forward_sequence_with_clone


def _cosine_divergence(a, b):
    """Cosine divergence = 1 - cosine_similarity."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0  # undefined, treat as maximal divergence
    return 1.0 - np.dot(a, b) / (na * nb)


def _process_seed(seed):
    """Process one model seed: compute divergences for all trials."""
    net = RecurrentMLP(seed=seed)
    X_train, y_train = generate_data(200, noise_level=0.5, seed=seed)
    X_test, y_test = generate_data(200, noise_level=0.5, seed=seed + 1000)
    train(net, X_train, y_train, epochs=1000, lr=0.01, time_weights=[0.0, 0.2, 1.0])

    clone = RecurrentMLP(seed=seed + 100)
    X_clone_train, y_clone_train = generate_data(200, noise_level=0.5, seed=seed + 100)
    train(clone, X_clone_train, y_clone_train, epochs=1000, lr=0.01,
          time_weights=[0.0, 0.2, 1.0])

    W_rec = net.W_rec.copy()
    tau = net.feedback_tau
    n_test = len(X_test)
    rng = np.random.RandomState(seed + 9000)

    # Get self outputs for all trials
    self_outputs_all = []
    true_labels = []
    for i in range(n_test):
        outputs, _ = net.forward_sequence(X_test[i], T=3)
        self_outputs_all.append(outputs)
        true_labels.append(np.argmax(y_test[i]))
    true_labels = np.array(true_labels)

    # Get clone outputs for all trials
    clone_outputs_all = []
    for i in range(n_test):
        c_outputs, _ = forward_sequence_with_clone(net, clone, X_test[i], T=3)
        # We need clone's OWN output, not target's output under clone feedback
        clone.reset_state()
        clone_y1 = clone.forward(X_test[i])
        clone_y2 = clone.forward(X_test[i])
        clone_outputs_all.append([clone_y1.copy(), clone_y2.copy()])

    # Build same-class trial groups
    class_groups = {}
    for i in range(n_test):
        c = true_labels[i]
        if c not in class_groups:
            class_groups[c] = []
        class_groups[c].append(i)

    trial_results = []
    for i in range(n_test):
        y_self_t1 = self_outputs_all[i][0]  # t=1 output
        self_contrib = np.tanh(y_self_t1 / tau) @ W_rec  # self W_rec contribution at t=2

        # (a) Isotropic Gaussian null: 100 random samples, norm-matched
        norm_self = np.linalg.norm(y_self_t1)
        random_divs = np.zeros(100)
        z_batch = rng.randn(100, len(y_self_t1))
        z_norms = np.linalg.norm(z_batch, axis=1, keepdims=True)
        z_batch = z_batch / (z_norms + 1e-12) * norm_self
        for k in range(100):
            rand_contrib = np.tanh(z_batch[k] / tau) @ W_rec
            random_divs[k] = _cosine_divergence(self_contrib, rand_contrib)
        div_random_mean = float(np.mean(random_divs))

        # (b) Same-class resampled self null
        same_class = true_labels[i]
        candidates = [j for j in class_groups[same_class] if j != i]
        if candidates:
            j = candidates[rng.randint(len(candidates))]
            y_resampled = self_outputs_all[j][0]
            resampled_contrib = np.tanh(y_resampled / tau) @ W_rec
            div_resampled = float(_cosine_divergence(self_contrib, resampled_contrib))
        else:
            div_resampled = float('nan')

        # (c) Clone divergence
        y_clone_t1 = clone_outputs_all[i][0]
        clone_contrib = np.tanh(y_clone_t1 / tau) @ W_rec
        div_clone = float(_cosine_divergence(self_contrib, clone_contrib))

        trial_results.append({
            'seed': seed,
            'trial': i,
            'divergence_random_mean': div_random_mean,
            'divergence_resampled': div_resampled,
            'divergence_clone': div_clone,
        })

    return trial_results


def write_report(all_results):
    """Write summary report."""
    # Flatten
    flat = [r for seed_results in all_results for r in seed_results]

    div_random = [r['divergence_random_mean'] for r in flat]
    div_resampled = [r['divergence_resampled'] for r in flat if not np.isnan(r['divergence_resampled'])]
    div_clone = [r['divergence_clone'] for r in flat]

    with open('results/REPORT_DIVERGENCE_NULL.md', 'w', encoding='utf-8') as f:
        f.write("# Cosine Divergence Null Baseline\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"| Condition | Mean | SD | Min | Max |\n")
        f.write(f"|-----------|------|----|-----|-----|\n")
        f.write(f"| (a) Isotropic random | {np.mean(div_random):.3f} | {np.std(div_random):.3f} | "
                f"{np.min(div_random):.3f} | {np.max(div_random):.3f} |\n")
        f.write(f"| (b) Same-class resampled | {np.mean(div_resampled):.3f} | {np.std(div_resampled):.3f} | "
                f"{np.min(div_resampled):.3f} | {np.max(div_resampled):.3f} |\n")
        f.write(f"| (c) Clone | {np.mean(div_clone):.3f} | {np.std(div_clone):.3f} | "
                f"{np.min(div_clone):.3f} | {np.max(div_clone):.3f} |\n")

        f.write(f"\n## Interpretation\n\n")

        # Per-seed means
        f.write("### Per-seed means\n\n")
        f.write("| Seed | Random | Resampled | Clone |\n")
        f.write("|------|--------|-----------|-------|\n")
        for seed_results in all_results:
            seed = seed_results[0]['seed']
            r_mean = np.mean([r['divergence_random_mean'] for r in seed_results])
            rs_vals = [r['divergence_resampled'] for r in seed_results if not np.isnan(r['divergence_resampled'])]
            rs_mean = np.mean(rs_vals) if rs_vals else float('nan')
            c_mean = np.mean([r['divergence_clone'] for r in seed_results])
            f.write(f"| {seed} | {r_mean:.3f} | {rs_mean:.3f} | {c_mean:.3f} |\n")

        # Critical ordering
        f.write("\n### Critical ordering\n\n")
        mean_r = np.mean(div_random)
        mean_rs = np.mean(div_resampled)
        mean_c = np.mean(div_clone)

        ordering = sorted([
            ('Random', mean_r), ('Resampled', mean_rs), ('Clone', mean_c)
        ], key=lambda x: x[1])
        f.write(f"Ordering: {' < '.join(f'{n} ({v:.3f})' for n, v in ordering)}\n\n")

        if mean_c < mean_r:
            f.write("**Clone divergence < Random divergence**: Clone is MORE aligned than random, "
                    "not less. The 0.635 divergence must be interpreted in this context: "
                    "it represents substantial but partial alignment, not evidence of "
                    "maximal misalignment.\n\n")
        else:
            f.write("**Clone divergence >= Random divergence**: Clone is as misaligned as or "
                    "more misaligned than random. The 0.635 divergence represents genuine "
                    "geometric misalignment.\n\n")

        if mean_rs < mean_c:
            f.write("**Same-class resampled < Clone**: Within-model same-class trials are more "
                    "aligned than cross-model trials, supporting model-specific feedback "
                    "contract beyond class-level effects.\n")
        else:
            f.write("**Same-class resampled >= Clone**: Within-model alignment advantage is "
                    "not clearly established at the same-class level.\n")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    n_workers = max(1, os.cpu_count() - 4)
    print(f"Using {n_workers} workers\n")

    print("Computing divergence null baselines...")
    with Pool(processes=n_workers) as pool:
        all_results = pool.map(_process_seed, range(20))

    for seed_results in all_results:
        seed = seed_results[0]['seed']
        r_mean = np.mean([r['divergence_random_mean'] for r in seed_results])
        rs_vals = [r['divergence_resampled'] for r in seed_results if not np.isnan(r['divergence_resampled'])]
        rs_mean = np.mean(rs_vals) if rs_vals else float('nan')
        c_mean = np.mean([r['divergence_clone'] for r in seed_results])
        print(f"  seed={seed}: random={r_mean:.3f} resampled={rs_mean:.3f} clone={c_mean:.3f}")

    # Write CSV
    fieldnames = ['seed', 'trial', 'divergence_random_mean', 'divergence_resampled', 'divergence_clone']
    flat = [r for sr in all_results for r in sr]
    with open('results/divergence_null_baseline.csv', 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)

    write_report(all_results)
    print("\nResults saved to results/divergence_null_baseline.csv and REPORT_DIVERGENCE_NULL.md")
