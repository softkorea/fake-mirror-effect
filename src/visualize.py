"""Visualisation module - generate experiment-result figures.

Five figures:
1. network_map.png - neurons + connections visualisation
2. ablation_comparison.png - per-group correction_gain bar chart
3. accuracy_distribution.png - random-ablation distribution + recurrent/scrambled markers
4. neuron_importance_heatmap.png - intelligence vs self-correction importance
5. noise_sweep_curve.png - correction_gain curve across noise levels
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def ensure_results_dir(path='results'):
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------
# 1. Network Map
# ----------------------------------------------

def plot_network_map(net, save_path='results/network_map.png'):
    """Visualise all neurons + connections. Recurrent connections in red."""
    ensure_results_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('RecurrentMLP Network Map (35 neurons)', fontsize=14)

    layers = {
        'Input': (0, 10),
        'Hidden1': (1.5, 10),
        'Hidden2': (3, 10),
        'Output': (4.5, 5),
    }

    neuron_pos = {}

    for layer_name, (x, n) in layers.items():
        for i in range(n):
            y = i * (10 / max(n - 1, 1))
            neuron_pos[(layer_name, i)] = (x, y)
            color = '#4ECDC4' if layer_name != 'Output' else '#FF6B6B'
            ax.add_patch(plt.Circle((x, y), 0.15, color=color, ec='black', lw=0.5, zorder=5))

        ax.text(x, -0.7, f'{layer_name}\n({n})', ha='center', fontsize=9)

    weights = net.get_all_weights()

    # Feedforward connections (gray, thin)
    connections = [
        ('Input', 'Hidden1', weights['input_to_h1']),
        ('Hidden1', 'Hidden2', weights['h1_to_h2']),
        ('Hidden2', 'Output', weights['h2_to_output']),
    ]

    for src_layer, dst_layer, W in connections:
        n_src = W.shape[0]
        n_dst = W.shape[1]
        max_w = max(np.abs(W).max(), 1e-6)
        for i in range(n_src):
            for j in range(n_dst):
                if abs(W[i, j]) > 0.01 * max_w:
                    x1, y1 = neuron_pos[(src_layer, i)]
                    x2, y2 = neuron_pos[(dst_layer, j)]
                    alpha = min(abs(W[i, j]) / max_w, 1.0) * 0.3
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=alpha, lw=0.3, zorder=1)

    # Recurrent connections (red)
    W_rec = weights['recurrent']
    max_wr = max(np.abs(W_rec).max(), 1e-6)
    for i in range(W_rec.shape[0]):
        for j in range(W_rec.shape[1]):
            if abs(W_rec[i, j]) > 0.01 * max_wr:
                x1, y1 = neuron_pos[('Output', i)]
                x2, y2 = neuron_pos[('Hidden1', j)]
                alpha = min(abs(W_rec[i, j]) / max_wr, 1.0) * 0.5
                ax.annotate('', xy=(x2 + 0.15, y2), xytext=(x1 - 0.15, y1),
                           arrowprops=dict(arrowstyle='->', color='red',
                                          alpha=alpha, lw=0.8),
                           zorder=3)

    ax.text(3.0, 10.7, '→ Red: Recurrent (output→h1)', color='red', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ----------------------------------------------
# 2. Ablation Comparison
# ----------------------------------------------

def plot_ablation_comparison(results_df, save_path='results/ablation_comparison.png'):
    """Per-group correction_gain bar chart.

    Args:
        results_df: dict with keys = group names,
                    values = list of gain values (across models/seeds)
    """
    ensure_results_dir(os.path.dirname(save_path))

    groups = list(results_df.keys())
    means = [np.mean(results_df[g]) for g in groups]
    stds = [np.std(results_df[g]) for g in groups]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # tab10-style palette; shared conditions (Baseline / A / C1 / C2) match
    # Figure 7's colors for cross-figure coherence.
    palette = {
        'Baseline': '#1f77b4',  # blue  - matches Figure 7
        'A':        '#7f7f7f',  # gray  - matches Figure 7 (semantically "no feedback")
        'B1':       '#8c564b',  # brown
        'B2':       '#e377c2',  # pink
        'C1':       '#ff7f0e',  # orange - matches Figure 7
        'D':        '#2ca02c',  # green
        "D'":       '#bcbd22',  # olive
        "D''":      '#17becf',  # cyan
        'C2':       '#9467bd',  # purple - matches Figure 7
    }
    bar_colors = [palette.get(g, '#bbbbbb') for g in groups]
    x = np.arange(len(groups))

    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(r'Correction gain ($\mathrm{acc}_{t_3} - \mathrm{acc}_{t_1}$)', fontsize=11)
    # Title removed - caption describes the figure (paper convention).
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(labelsize=10)

    # Display values above (or below for negative) each bar.
    # Drop the explicit sign on near-zero means so labels read "0.000" rather than "+0.000".
    for bar, m, s in zip(bars, means, stds):
        if m >= 0:
            y_pos = bar.get_height() + s + 0.005
            va = 'bottom'
        else:
            y_pos = bar.get_height() - s - 0.005
            va = 'top'
        label = f'{m:.3f}' if abs(m) < 5e-4 else f'{m:+.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                label, ha='center', va=va, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ----------------------------------------------
# 3. Accuracy Distribution
# ----------------------------------------------

def plot_accuracy_distribution(random_gains, recurrent_gain, scrambled_gain,
                                save_path='results/accuracy_distribution.png'):
    """Random-ablation distribution histogram + recurrent/scrambled markers.

    Args:
        random_gains: list of gain values from random ablation (B1)
        recurrent_gain: float, gain after recurrent ablation (A)
        scrambled_gain: float, gain after scrambled feedback (C)
    """
    ensure_results_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(random_gains, bins=20, alpha=0.7, color='#3498db',
            edgecolor='black', label='Random ablation (B1)')
    ax.axvline(recurrent_gain, color='red', linestyle='--', lw=2,
               label=f'Recurrent ablation (A): {recurrent_gain:.3f}')
    ax.axvline(scrambled_gain, color='purple', linestyle='--', lw=2,
               label=f'Scrambled feedback (C): {scrambled_gain:.3f}')

    ax.set_xlabel('Correction Gain', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Correction Gain: Random vs Targeted Ablation', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ----------------------------------------------
# 4. Neuron Importance Heatmap
# ----------------------------------------------

def plot_neuron_importance_heatmap(intelligence_importance, correction_importance,
                                    save_path='results/neuron_importance_heatmap.png'):
    """Two-panel single-neuron knockout summary (H1 decoupled, H2 full-knockout).

    Args:
        intelligence_importance: dict {neuron_id: importance_value} with h1_X and h2_X keys
        correction_importance: dict {neuron_id: importance_value} with h1_X and h2_X keys
    """
    ensure_results_dir(os.path.dirname(save_path))

    def _sort_key(n):
        return int(n.split('_')[1])

    h1_ids = sorted([n for n in intelligence_importance if n.startswith('h1_')], key=_sort_key)
    h2_ids = sorted([n for n in intelligence_importance if n.startswith('h2_')], key=_sort_key)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

    # Panel A: H1 (decoupled)
    x_a = [intelligence_importance[n] for n in h1_ids]
    y_a = [correction_importance[n] for n in h1_ids]
    ax_a.scatter(x_a, y_a, c='#2980b9', s=80, alpha=0.85, edgecolors='black', linewidth=0.5)
    for n, x, y in zip(h1_ids, x_a, y_a):
        ax_a.annotate(n, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax_a.set_xlabel(r'Intelligence Importance $\Delta\mathrm{acc}_{t_1}$', fontsize=11)
    ax_a.set_ylabel(r'Correction Importance $\Delta\mathrm{gain}$', fontsize=11)
    ax_a.set_title('(A) Hidden Layer 1 — decoupled', fontsize=12, fontweight='bold')
    ax_a.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax_a.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax_a.grid(alpha=0.2)

    # Panel B: H2 (full knockout, confounded)
    x_b = [intelligence_importance[n] for n in h2_ids]
    y_b = [correction_importance[n] for n in h2_ids]
    ax_b.scatter(x_b, y_b, c='#c0392b', s=80, alpha=0.85, edgecolors='black', linewidth=0.5)
    for n, x, y in zip(h2_ids, x_b, y_b):
        ax_b.annotate(n, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax_b.set_xlabel(r'Intelligence Importance $\Delta\mathrm{acc}_{t_1}$', fontsize=11)
    ax_b.set_ylabel(r'Correction Importance $\Delta\mathrm{gain}$', fontsize=11)
    ax_b.set_title('(B) Hidden Layer 2 — full knockout (confounded)', fontsize=12, fontweight='bold')
    ax_b.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax_b.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax_b.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ----------------------------------------------
# 5. Noise Sweep Curve
# ----------------------------------------------

def plot_noise_sweep(sweep_data, save_path='results/noise_sweep_curve.png'):
    """correction_gain curve across noise levels.

    Args:
        sweep_data: dict {group_name: {noise_level: [gain_values]}}
    """
    ensure_results_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 6))
    # Group code -> style mapping (fixes code/label mismatch)
    GROUP_STYLE = {
        'Baseline': ('#2ecc71', 'o', 'Baseline'),
        'A':        ('#e74c3c', 's', 'A (Recurrent Cut)'),
        'B1':       ('#3498db', '^', 'B1 (Random Cut)'),
        'B2':       ('#8e44ad', 'v', 'B2 (Structural Cut)'),
        'C1':       ('#9b59b6', 'D', 'C1 (Permutation)'),
        'C2':       ('#f39c12', 'P', 'C2 (Clone Feedback)'),
        'D':        ('#95a5a6', 'x', 'D (Feedforward)'),
        "D'":       ('#1abc9c', '+', "D' (Param-matched FF)"),
    }

    for group_code, noise_gains in sweep_data.items():
        noise_levels = sorted(noise_gains.keys())
        means = [np.mean(noise_gains[nl]) for nl in noise_levels]
        stds = [np.std(noise_gains[nl]) for nl in noise_levels]

        c, m, label = GROUP_STYLE.get(group_code, ('gray', 'o', group_code))
        ax.errorbar(noise_levels, means, yerr=stds, label=label,
                    color=c, marker=m, capsize=3, lw=1.5, markersize=6)

    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Correction Gain', fontsize=11)
    ax.set_title('Noise Sweep: Correction Gain by Group and Noise Level\n'
                 '(20 models, mean ± std)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', lw=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path
