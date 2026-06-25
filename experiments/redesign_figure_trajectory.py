"""Redesign of Figure 3 (now Figure 4, fig:trajectory).

Renders the figure-design explorations (options A-D and paradigm variants) for
comparison, plus the final composed figure used in the paper; same projection
basis used across panels within each option.

  Option A - Filter to changed-prediction trials (4-panel grid, PCA-H1).
  Option B - Aggregate displacement on PCA-H1 (per-class arrows + density bg).
  Option C - LDA on H2 with per-class arrows + 1-sigma covariance ellipses
             (expected to make class structure visually separable since H2
             feeds the linear readout).
  Option D / paradigm_* - further exploratory variants.

All options use a unified cosmetic style (seaborn colorblind palette,
despined axes, serif font, B&W-safe line/marker encoding).

Output: results/figure_redesign/{option_A,option_B,option_C,option_D}.png,
        paradigm_*.png, and hybrid_main.{png,pdf}
        (hybrid_main.pdf is the published Figure 4, fig:trajectory).
Cache: results/figure_redesign/traces_cache_h2.pkl (a_h1 + a_h2 + output traces)

Usage:
    python experiments/redesign_figure_trajectory.py
"""

import sys
import os
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network import RecurrentMLP
from src.training import generate_data, train, softmax
from src.ablation import forward_sequence_with_clone


# ---------------------------------------------------------------------------
# Cosmetic style (applied globally before any rendering)
# ---------------------------------------------------------------------------
def configure_style():
    import matplotlib as mpl
    # Embed fonts as TrueType (Type 42) rather than matplotlib's default Type 3
    # bitmap glyphs, so the figure PDF carries no Type 3 fonts (cleaner for
    # publication-PDF preflight; glyphs stay vector and scalable).
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman']
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.16
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['grid.linewidth'] = 0.45
    # Academic-journal typography: 1 pt smaller across the board
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8.5
    mpl.rcParams['ytick.labelsize'] = 8.5
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['legend.framealpha'] = 0.88


def palette_and_markers():
    """Return (class_colors, class_markers, type_colors).

    class_colors: 5 colorblind-safe colors for the 5 classes.
    class_markers: distinct shapes for B&W readability.
    type_colors: muted accents for trial-type encoding (corrected vs over-corrected).
    """
    try:
        import seaborn as sns
        cb = sns.color_palette('colorblind', n_colors=10)
    except ImportError:
        # Fallback: ColorBrewer-like palette baked in
        cb = [(0.00, 0.45, 0.70), (0.90, 0.62, 0.00), (0.00, 0.62, 0.45),
              (0.80, 0.47, 0.65), (0.34, 0.71, 0.91), (0.84, 0.37, 0.00),
              (0.94, 0.89, 0.26), (0.30, 0.30, 0.30), (0.55, 0.27, 0.60),
              (0.00, 0.62, 0.45)]
    class_colors = [cb[0], cb[1], cb[2], cb[3], cb[4]]
    class_markers = ['o', 's', '^', 'D', 'v']
    # Muted accents for trial-type (used in Option A only)
    type_colors = {'corrected': cb[2], 'over_corrected': cb[1],
                   'stable_correct': cb[0], 'stable_incorrect': cb[3]}
    return class_colors, class_markers, type_colors


# ---------------------------------------------------------------------------
# Model training and trace collection (with a_h2 + output for Option C)
# ---------------------------------------------------------------------------
def train_model(seed, noise_level=0.5, n_samples=200, epochs=1000,
                lr=0.01, tau=2.0):
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10,
                       output_size=5, seed=seed, feedback_tau=tau)
    X, y = generate_data(n_samples, noise_level=noise_level, seed=seed)
    train(net, X, y, epochs=epochs, lr=lr,
          time_weights=[0.0, 0.2, 1.0])
    return net


def collect_baseline_clone_traces(target_net, clone_net, X, y):
    n = len(X)
    T = 3
    traces = {
        'a_h1_self': np.zeros((n, T, 10)),
        'a_h2_self': np.zeros((n, T, 10)),
        'output_self': np.zeros((n, T, 5)),
        'a_h1_clone': np.zeros((n, T, 10)),
        'a_h2_clone': np.zeros((n, T, 10)),
        'output_clone': np.zeros((n, T, 5)),
        'true': np.zeros(n, dtype=int),
        'trial_type_self': [],
        'trial_type_clone': [],
    }
    for i in range(n):
        true_cls = np.argmax(y[i])
        traces['true'][i] = true_cls

        out_s, ca_s = target_net.forward_sequence(X[i], T=T)
        for t in range(T):
            traces['a_h1_self'][i, t] = ca_s[t]['a_h1']
            traces['a_h2_self'][i, t] = ca_s[t]['a_h2']
            traces['output_self'][i, t] = ca_s[t]['output']
        c1 = (np.argmax(ca_s[0]['output']) == true_cls)
        c3 = (np.argmax(ca_s[2]['output']) == true_cls)
        if not c1 and c3:
            traces['trial_type_self'].append('corrected')
        elif c1 and c3:
            traces['trial_type_self'].append('stable_correct')
        elif not c1 and not c3:
            traces['trial_type_self'].append('stable_incorrect')
        else:
            traces['trial_type_self'].append('over_corrected')

        out_c, ca_c = forward_sequence_with_clone(target_net, clone_net, X[i], T=T)
        for t in range(T):
            traces['a_h1_clone'][i, t] = ca_c[t]['a_h1']
            traces['a_h2_clone'][i, t] = ca_c[t]['a_h2']
            traces['output_clone'][i, t] = ca_c[t]['output']
        c1c = (np.argmax(ca_c[0]['output']) == true_cls)
        c3c = (np.argmax(ca_c[2]['output']) == true_cls)
        if not c1c and c3c:
            traces['trial_type_clone'].append('corrected')
        elif c1c and c3c:
            traces['trial_type_clone'].append('stable_correct')
        elif not c1c and not c3c:
            traces['trial_type_clone'].append('stable_incorrect')
        else:
            traces['trial_type_clone'].append('over_corrected')

    traces['trial_type_self'] = np.array(traces['trial_type_self'])
    traces['trial_type_clone'] = np.array(traces['trial_type_clone'])
    return traces


def collect_all_traces(seeds, donor_seeds, noise_level=0.5, n_test=200):
    print("[Phase 1] Training models...")
    target_models = {}
    for s in seeds:
        t0 = time.time()
        target_models[s] = train_model(s, noise_level=noise_level)
        print(f"  target seed={s}: {time.time()-t0:.1f}s")

    clone_models = {}
    for ds in donor_seeds:
        t0 = time.time()
        clone_models[ds] = train_model(ds, noise_level=noise_level)
        print(f"  donor  seed={ds}: {time.time()-t0:.1f}s")

    print("\n[Phase 2] Collecting traces...")
    all_traces = []
    for s_idx, s in enumerate(seeds):
        X_test, y_test = generate_data(n_test, noise_level=noise_level,
                                       seed=1000 + s)
        net = target_models[s]
        clone = clone_models[donor_seeds[s_idx]]
        traces = collect_baseline_clone_traces(net, clone, X_test, y_test)
        traces['seed'] = s
        traces['donor_seed'] = donor_seeds[s_idx]
        all_traces.append(traces)
        print(f"  seed={s}: collected {n_test} trials")
    return all_traces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_pca_h1(all_traces, n_components=2):
    from sklearn.decomposition import PCA
    all_t1 = np.vstack([tr['a_h1_self'][:, 0, :] for tr in all_traces])
    pca = PCA(n_components=n_components)
    pca.fit(all_t1)
    return pca


def fit_lda_h2(all_traces, n_components=2):
    """Fit LDA on baseline stable-correct t=3 H2 activations."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = []
    y = []
    for tr in all_traces:
        mask = tr['trial_type_self'] == 'stable_correct'
        if mask.any():
            X.append(tr['a_h2_self'][mask, 2, :])  # t=3
            y.append(tr['true'][mask])
    X = np.vstack(X)
    y = np.concatenate(y)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    print(f"  LDA-H2 fit on {len(y)} baseline-stable-correct t=3 samples; "
          f"per-class counts {[int((y==c).sum()) for c in range(5)]}")
    return lda


def fit_lda_logit(all_traces, n_components=2):
    """Fit LDA on baseline stable-correct t=3 output logits.
    Note: tautological in the decision-space framing (LDA on logits explicitly
    separates by class), but the most class-discriminative space available;
    included as a 4th comparison option.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = []
    y = []
    for tr in all_traces:
        mask = tr['trial_type_self'] == 'stable_correct'
        if mask.any():
            X.append(tr['output_self'][mask, 2, :])
            y.append(tr['true'][mask])
    X = np.vstack(X)
    y = np.concatenate(y)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    print(f"  LDA-logit fit on {len(y)} baseline-stable-correct t=3 samples")
    return lda


def aggregate_by_type_and_class(all_traces, key_act, key_type):
    """Return nested dict: trial_type -> class -> list of (3, D) trajectories."""
    out = {tt: {c: [] for c in range(5)} for tt in
           ['corrected', 'stable_correct', 'stable_incorrect', 'over_corrected']}
    for tr in all_traces:
        ttypes = tr[key_type]
        truths = tr['true']
        for i in range(len(ttypes)):
            out[ttypes[i]][int(truths[i])].append(tr[key_act][i])
    return out


def class_centroids_in_proj(all_traces, projector, key_act, on_t=2):
    """Class centroid in projected space, computed from baseline stable-correct."""
    cents = {}
    for cls in range(5):
        pts = []
        for tr in all_traces:
            mask = (tr['true'] == cls) & (tr['trial_type_self'] == 'stable_correct')
            if mask.any():
                pts.append(tr[key_act][mask, on_t, :])
        if pts:
            pts = np.vstack(pts)
            proj = projector.transform(pts)
            cents[cls] = proj.mean(axis=0)
    return cents


def cov_ellipse_params(points, n_sigma=1.0):
    """Return (cx, cy, width, height, angle_deg) for a 2D points cloud."""
    if len(points) < 2:
        return None
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_sigma * np.sqrt(max(eigvals[0], 1e-12))
    height = 2 * n_sigma * np.sqrt(max(eigvals[1], 1e-12))
    return points[:, 0].mean(), points[:, 1].mean(), width, height, angle


# ---------------------------------------------------------------------------
# Option A - 4-panel filter (PCA-H1)
# ---------------------------------------------------------------------------
def render_option_A(all_traces, pca, save_path):
    import matplotlib.pyplot as plt
    class_colors, class_markers, type_colors = palette_and_markers()

    bl = aggregate_by_type_and_class(all_traces, 'a_h1_self', 'trial_type_self')
    c2 = aggregate_by_type_and_class(all_traces, 'a_h1_clone', 'trial_type_clone')
    cents = class_centroids_in_proj(all_traces, pca, 'a_h1_self', on_t=2)

    def gather(by_class):
        return [t for c in range(5) for t in by_class[c]]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex=True, sharey=True)
    panels = [
        ('Baseline / Corrected (wrong->right)', gather(bl['corrected']), type_colors['corrected']),
        ('C2 / Corrected (wrong->right)',       gather(c2['corrected']), type_colors['corrected']),
        ('Baseline / Over-corrected (right->wrong)', gather(bl['over_corrected']), type_colors['over_corrected']),
        ('C2 / Over-corrected (right->wrong)',       gather(c2['over_corrected']), type_colors['over_corrected']),
    ]
    for ax, (title, trajs, line_color) in zip(axes.ravel(), panels):
        for cls, pos in cents.items():
            ax.scatter(pos[0], pos[1], marker=class_markers[cls], s=120,
                       c=[class_colors[cls]], edgecolors='black', linewidths=0.6,
                       zorder=5,
                       label=f'class {cls}' if title.startswith('Baseline / Corrected') else None)
        if trajs:
            for traj in trajs:
                proj = pca.transform(traj)
                ax.plot(proj[:, 0], proj[:, 1], '-', color=line_color,
                        alpha=0.25, linewidth=0.6)
                ax.plot(proj[2, 0], proj[2, 1], '*', color=line_color,
                        alpha=0.7, markersize=6)
        ax.set_title(f'{title}  (N={len(trajs)})', fontsize=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    axes[0, 0].legend(loc='upper right')
    plt.suptitle(
        'Option A - Filter to changed-prediction trials (PCA on H1)\n'
        'Star = t=3 endpoint of trajectory; class markers indicate canonical class regions.',
        fontsize=11, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Option B - aggregate per-class displacement (PCA-H1)
# ---------------------------------------------------------------------------
def render_option_B(all_traces, pca, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    class_colors, class_markers, _ = palette_and_markers()

    bl = aggregate_by_type_and_class(all_traces, 'a_h1_self', 'trial_type_self')
    c2 = aggregate_by_type_and_class(all_traces, 'a_h1_clone', 'trial_type_clone')
    cents = class_centroids_in_proj(all_traces, pca, 'a_h1_self', on_t=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharex=True, sharey=True)

    for panel_idx, (ax, (title, by_type), line_style) in enumerate(zip(
        axes,
        [('Baseline (self-feedback)', bl), ('C2 (clone feedback)', c2)],
        ['-', '--'],  # B&W-safe encoding for BL vs C2
    )):
        # Background: stable trial t=3 endpoints (faint)
        bg_pts = []
        for ttype in ['stable_correct', 'stable_incorrect']:
            for cls in range(5):
                for traj in by_type[ttype][cls]:
                    bg_pts.append(traj[2])
        if bg_pts:
            bg_arr = np.vstack(bg_pts)
            bg_proj = pca.transform(bg_arr)
            ax.scatter(bg_proj[:, 0], bg_proj[:, 1], s=4, c='lightgray',
                       alpha=0.30, zorder=1,
                       label='stable trials (t=3)' if panel_idx == 0 else None)

        # Class centroids
        for cls, pos in cents.items():
            ax.scatter(pos[0], pos[1], marker=class_markers[cls], s=180,
                       c=[class_colors[cls]], edgecolors='black', linewidths=0.7,
                       zorder=5,
                       label=f'class {cls}' if panel_idx == 0 else None)

        # Per-class corrected-trial mean displacement arrows
        for cls in range(5):
            trajs = by_type['corrected'][cls]
            if len(trajs) < 2:
                continue
            arr = np.array(trajs)
            mean_traj = arr.mean(axis=0)
            proj_mean = pca.transform(mean_traj)
            start, end = proj_mean[0], proj_mean[2]
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=class_colors[cls],
                                lw=2.0, mutation_scale=15, alpha=0.95,
                                linestyle=line_style),
                zorder=4,
            )
            # Origin marker (Top-3 #2 for Option B): black dot at t=1
            ax.scatter(start[0], start[1], marker='o', s=22, c='black',
                       zorder=6)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    axes[0].legend(loc='upper right')
    plt.suptitle(
        'Option B - Per-class mean displacement on corrected trials (PCA on H1)\n'
        'Solid = self-feedback, dashed = clone feedback (B&W-safe). '
        'Black dot = t=1 origin. Background: stable t=3 distribution.',
        fontsize=11, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Option C - LDA on H2
# ---------------------------------------------------------------------------
def render_option_C(all_traces, lda, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    class_colors, class_markers, _ = palette_and_markers()

    bl = aggregate_by_type_and_class(all_traces, 'a_h2_self', 'trial_type_self')
    c2 = aggregate_by_type_and_class(all_traces, 'a_h2_clone', 'trial_type_clone')
    cents = class_centroids_in_proj(all_traces, lda, 'a_h2_self', on_t=2)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharex=True, sharey=True)

    for panel_idx, (ax, (title, by_type), line_style) in enumerate(zip(
        axes,
        [('Baseline (self-feedback)', bl), ('C2 (clone feedback)', c2)],
        ['-', '--'],
    )):
        # Background: stable-correct t=3 KDE-style soft scatter per class
        # (using small alpha dots; faster than scipy KDE and OK at this density)
        for cls in range(5):
            stable_pts = []
            for traj in by_type['stable_correct'][cls]:
                stable_pts.append(traj[2])
            if stable_pts:
                arr = np.vstack(stable_pts)
                proj = lda.transform(arr)
                ax.scatter(proj[:, 0], proj[:, 1], s=3, c=[class_colors[cls]],
                           alpha=0.10, zorder=1)

        # Class centroids (large markers)
        for cls, pos in cents.items():
            ax.scatter(pos[0], pos[1], marker=class_markers[cls], s=200,
                       c=[class_colors[cls]], edgecolors='black', linewidths=0.8,
                       zorder=6,
                       label=f'class {cls}' if panel_idx == 0 else None)

        # Per-class corrected-trial mean arrows + 1-sigma t=3 ellipse
        for cls in range(5):
            trajs = by_type['corrected'][cls]
            if len(trajs) < 2:
                continue
            arr = np.array(trajs)
            mean_traj = arr.mean(axis=0)
            proj_mean = lda.transform(mean_traj)
            start, end = proj_mean[0], proj_mean[2]

            # 1-sigma ellipse on t=3 endpoint distribution
            t3_pts = arr[:, 2, :]
            t3_proj = lda.transform(t3_pts)
            ell_params = cov_ellipse_params(t3_proj, n_sigma=1.0)
            if ell_params is not None:
                cx, cy, w, h, ang = ell_params
                # Outline-emphasized ellipse: too many filled ellipses re-introduce
                # visual complexity; we favour outline at low fill alpha.
                ell = Ellipse((cx, cy), w, h, angle=ang,
                              edgecolor=class_colors[cls],
                              facecolor=class_colors[cls],
                              alpha=0.10, linewidth=1.4, zorder=3,
                              linestyle='-')
                ax.add_patch(ell)

            # Arrow t=1 -> t=3
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=class_colors[cls],
                                lw=2.0, mutation_scale=16, alpha=0.95,
                                linestyle=line_style),
                zorder=5,
            )
            # Origin marker
            ax.scatter(start[0], start[1], marker='o', s=22, c='black',
                       zorder=7)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')

    axes[0].legend(loc='upper right')
    plt.suptitle(
        'Option C - Aggregate per-class displacement of pre-decision (H2) '
        'representations\n'
        '(LDA basis fitted on baseline stable-correct t=3 H2; same axes for '
        'both panels. Single-trial trajectories: Appendix.)\n'
        'Solid = self-feedback, dashed = clone feedback. '
        'Black dot = t=1 origin. Light scatter: stable-correct t=3 per class. '
        'Ellipses: 1-sigma cov. of t=3 corrected endpoints.',
        fontsize=9.5, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Option D - LDA on output logits (decision-space clarity figure)
# ---------------------------------------------------------------------------
def render_option_D(all_traces, lda_logit, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    class_colors, class_markers, _ = palette_and_markers()

    bl = aggregate_by_type_and_class(all_traces, 'output_self', 'trial_type_self')
    c2 = aggregate_by_type_and_class(all_traces, 'output_clone', 'trial_type_clone')
    cents = class_centroids_in_proj(all_traces, lda_logit, 'output_self', on_t=2)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)

    for panel_idx, (ax, (title, by_type), line_style) in enumerate(zip(
        axes,
        [('Baseline (self-feedback)', bl), ('C2 (clone feedback)', c2)],
        ['-', '--'],
    )):
        # Background: very faint per-class stable-correct scatter
        for cls in range(5):
            stable_pts = []
            for traj in by_type['stable_correct'][cls]:
                stable_pts.append(traj[2])
            if stable_pts:
                arr = np.vstack(stable_pts)
                proj = lda_logit.transform(arr)
                ax.scatter(proj[:, 0], proj[:, 1], s=2.5, c=[class_colors[cls]],
                           alpha=0.06, zorder=1, linewidths=0)

        # Class centroids with white halo for visual pop above scatter
        for cls, pos in cents.items():
            # White halo (one ply behind centroid)
            ax.scatter(pos[0], pos[1], marker=class_markers[cls], s=320,
                       c='white', edgecolors='white', linewidths=2.5, zorder=5)
            # Class centroid marker
            ax.scatter(pos[0], pos[1], marker=class_markers[cls], s=200,
                       c=[class_colors[cls]], edgecolors='black', linewidths=0.9,
                       zorder=6,
                       label=f'class {cls}' if panel_idx == 0 else None)

        # Per-class arrows + 1-sigma ellipses for corrected trials
        for cls in range(5):
            trajs = by_type['corrected'][cls]
            if len(trajs) < 2:
                continue
            arr = np.array(trajs)
            mean_traj = arr.mean(axis=0)
            proj_mean = lda_logit.transform(mean_traj)
            start, end = proj_mean[0], proj_mean[2]

            # 1-sigma covariance ellipse - outline-emphasized, light fill
            t3_pts = arr[:, 2, :]
            t3_proj = lda_logit.transform(t3_pts)
            ell_params = cov_ellipse_params(t3_proj, n_sigma=1.0)
            if ell_params is not None:
                cx, cy, w, h, ang = ell_params
                ell = Ellipse((cx, cy), w, h, angle=ang,
                              edgecolor=class_colors[cls],
                              facecolor=class_colors[cls],
                              alpha=0.07, linewidth=1.8, zorder=3,
                              linestyle='-')
                ax.add_patch(ell)

            # Stronger arrow (thicker line, larger arrowhead)
            ax.annotate(
                '', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=class_colors[cls],
                                lw=2.4, mutation_scale=18, alpha=1.0,
                                linestyle=line_style),
                zorder=5,
            )
            # Larger origin marker with white edge for clarity
            ax.scatter(start[0], start[1], marker='o', s=44, c='black',
                       edgecolors='white', linewidths=1.0, zorder=7)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('LD1 (logit)', fontsize=10)
        if panel_idx == 0:
            ax.set_ylabel('LD2 (logit)', fontsize=10)

    # Concise paper-style legend
    axes[0].legend(loc='upper right', frameon=True, facecolor='white',
                   edgecolor='lightgray', framealpha=0.92, fontsize=8.5)

    # Single-line, paper-style title (methodological detail goes in caption)
    plt.suptitle(
        'Decision-space visualization of feedback-dependent trajectory changes',
        fontsize=11.5, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ===========================================================================
# Paradigm prototypes - native-metric, AE-safe
# ===========================================================================

def _class_centroids_h2(all_traces):
    """Class centroids in h2 space, computed from baseline stable-correct t=3."""
    cents = {}
    for cls in range(5):
        pts = []
        for tr in all_traces:
            mask = (tr['true'] == cls) & (tr['trial_type_self'] == 'stable_correct')
            if mask.any():
                pts.append(tr['a_h2_self'][mask, 2, :])
        if pts:
            cents[cls] = np.vstack(pts).mean(axis=0)
    return cents


def _class_target_directions_h2(all_traces):
    """Per-class target directions in h2 space.

    target_dir_c = centroid_c - mean(centroid_*) (centered, unit-normalized).
    Represents 'how to move toward class c relative to mean class location'.
    """
    cents = _class_centroids_h2(all_traces)
    if not cents:
        return {}
    grand = np.stack(list(cents.values())).mean(axis=0)
    dirs = {}
    for cls, c in cents.items():
        v = c - grand
        n = np.linalg.norm(v)
        dirs[cls] = v / n if n > 1e-10 else v
    return dirs


def _delta_metrics(all_traces, act_key, type_key, target_dirs):
    """For each trial: ||deltah||, cos(deltah, target_dir_for_true_class), parallel/orthogonal split."""
    mags, aligns, par_mag, orth_mag, true_classes = [], [], [], [], []
    for tr in all_traces:
        truths = tr['true']
        for i in range(len(truths)):
            cls = int(truths[i])
            if cls not in target_dirs:
                continue
            h_t1 = tr[act_key][i, 0, :]
            h_t3 = tr[act_key][i, 2, :]
            delta = h_t3 - h_t1
            mag = float(np.linalg.norm(delta))
            if mag < 1e-10:
                continue
            t = target_dirs[cls]
            cos_a = float(np.dot(delta, t) / mag)
            par = float(np.dot(delta, t))                   # signed parallel component
            orth_vec = delta - par * t
            orth = float(np.linalg.norm(orth_vec))
            mags.append(mag)
            aligns.append(cos_a)
            par_mag.append(par)
            orth_mag.append(orth)
            true_classes.append(cls)
    return (np.array(mags), np.array(aligns),
            np.array(par_mag), np.array(orth_mag),
            np.array(true_classes))


# ---------------------------------------------------------------------------
# Paradigm 1 - Magnitude x Alignment 2D Density
# ---------------------------------------------------------------------------
def render_paradigm_1_magnitude_alignment(all_traces, save_path):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        have_sns = True
    except ImportError:
        have_sns = False

    target_dirs = _class_target_directions_h2(all_traces)

    bl_m, bl_a, *_ = _delta_metrics(all_traces, 'a_h2_self', 'trial_type_self', target_dirs)
    c2_m, c2_a, *_ = _delta_metrics(all_traces, 'a_h2_clone', 'trial_type_clone', target_dirs)

    xmin = min(bl_m.min(), c2_m.min()) * 0.95
    xmax = max(bl_m.max(), c2_m.max()) * 1.02
    ymin = min(bl_a.min(), c2_a.min())
    ymax = max(bl_a.max(), c2_a.max())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    for ax, title, mags, aligns in zip(
            axes,
            ['Baseline (self-feedback)', 'C2 (clone feedback)'],
            [bl_m, c2_m], [bl_a, c2_a]):
        if have_sns:
            sns.kdeplot(x=mags, y=aligns, ax=ax, levels=10, fill=True,
                        cmap='viridis', alpha=0.88, thresh=0.02)
        else:
            ax.hexbin(mags, aligns, gridsize=40, cmap='viridis', mincnt=1)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.scatter([mags.mean()], [aligns.mean()], s=380, marker='+', c='red',
                   linewidths=3.0, zorder=10,
                   label=f'mean ({mags.mean():.2f}, {aligns.mean():.3f})')
        ax.legend(loc='upper right', fontsize=8.5)
        ax.set_title(f'{title}  (N={len(mags)})', fontsize=11.5)
        ax.set_xlabel(r'$\|\Delta h^{(2)}\|_2$  (native L2 displacement)', fontsize=10)
        ax.set_ylabel(r'$\cos(\Delta h^{(2)},\, t_c)$', fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    plt.suptitle(
        'Paradigm 1 — Magnitude × alignment density (native h₂ metrics)',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Paradigm 2 - True-Class Logit Margin ECDFs
# ---------------------------------------------------------------------------
def render_paradigm_2_margin_ecdf(all_traces, save_path):
    import matplotlib.pyplot as plt

    def margins_at(t_idx, output_key, type_key):
        """Per-trial logit margin = logit[true] - max(logit[other]) at timestep t_idx."""
        out = []
        for tr in all_traces:
            truths = tr['true']
            for i in range(len(truths)):
                cls = int(truths[i])
                logits = tr[output_key][i, t_idx, :]
                other_max = np.max(np.delete(logits, cls))
                out.append(float(logits[cls] - other_max))
        return np.array(out)

    bl_t1 = margins_at(0, 'output_self', 'trial_type_self')
    bl_t3 = margins_at(2, 'output_self', 'trial_type_self')
    c2_t1 = margins_at(0, 'output_clone', 'trial_type_clone')
    c2_t3 = margins_at(2, 'output_clone', 'trial_type_clone')

    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    cb = palette_and_markers()[0]
    bl_color, c2_color = cb[0], cb[1]

    for ax, title, bl, c2 in zip(
            axes,
            ['Initial state (t = 1)', 'Final state (t = 3)'],
            [bl_t1, bl_t3], [c2_t1, c2_t3]):
        x_bl, y_bl = ecdf(bl)
        x_c2, y_c2 = ecdf(c2)
        ax.plot(x_bl, y_bl, color=bl_color, lw=2.0, label='Baseline (self)')
        ax.plot(x_c2, y_c2, color=c2_color, lw=2.0, linestyle='--', label='C2 (clone)')
        ax.axvline(0, color='gray', linestyle=':', lw=0.9, alpha=0.6,
                   label='decision boundary')
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel(r'logit margin = $y_{\mathrm{true}} - \max_{c\neq\mathrm{true}} y_c$',
                      fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel('Cumulative fraction', fontsize=10)
        ax.legend(loc='lower right', fontsize=8.5)

    plt.suptitle(
        'Paradigm 2 — Logit margin ECDFs across timesteps',
        fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Paradigm 3 - Orthogonal Vector Decomposition (Boxplot)
# ---------------------------------------------------------------------------
def render_paradigm_3_orthogonal_boxplot(all_traces, save_path):
    import matplotlib.pyplot as plt

    target_dirs = _class_target_directions_h2(all_traces)
    _, _, bl_par, bl_orth, _ = _delta_metrics(all_traces, 'a_h2_self', 'trial_type_self', target_dirs)
    _, _, c2_par, c2_orth, _ = _delta_metrics(all_traces, 'a_h2_clone', 'trial_type_clone', target_dirs)

    cb = palette_and_markers()[0]
    bl_color, c2_color = cb[0], cb[1]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))

    data = [bl_par, c2_par, bl_orth, c2_orth]
    positions = [1, 1.7, 3.2, 3.9]
    colors = [bl_color, c2_color, bl_color, c2_color]
    labels = ['BL\nparallel', 'C2\nparallel', 'BL\northogonal', 'C2\northogonal']

    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.5))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    ax.axhline(0, color='gray', linestyle='--', lw=0.8, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'displacement component  ($h^{(2)}$ space)', fontsize=10)

    # Annotate group separation
    ax.axvline(2.45, color='lightgray', linestyle=':', lw=0.7)
    ax.text(1.35, ax.get_ylim()[1] * 0.92, 'aligned with\ntrue-class direction',
            ha='center', fontsize=9, style='italic', alpha=0.75)
    ax.text(3.55, ax.get_ylim()[1] * 0.92, 'orthogonal\n(task-irrelevant)',
            ha='center', fontsize=9, style='italic', alpha=0.75)

    plt.suptitle(
        'Paradigm 3 — Δh₂ vector decomposition (parallel vs orthogonal to target direction)',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Paradigm 4 - Logit Phase Portrait (target vs top distractor logit)
# ---------------------------------------------------------------------------
def render_paradigm_4_logit_phase_portrait(all_traces, save_path):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        have_sns = True
    except ImportError:
        have_sns = False

    def gather(t_idx, output_key):
        x_dist, y_targ = [], []
        for tr in all_traces:
            truths = tr['true']
            for i in range(len(truths)):
                cls = int(truths[i])
                logits = tr[output_key][i, t_idx, :]
                y_targ.append(float(logits[cls]))
                x_dist.append(float(np.max(np.delete(logits, cls))))
        return np.array(x_dist), np.array(y_targ)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharex=True, sharey=True)

    for ax, title, output_key in zip(
            axes,
            ['Baseline (self-feedback)', 'C2 (clone feedback)'],
            ['output_self', 'output_clone']):
        x1, y1 = gather(0, output_key)
        x3, y3 = gather(2, output_key)

        if have_sns:
            sns.kdeplot(x=x1, y=y1, ax=ax, levels=6, color='gray',
                        linewidths=0.9, alpha=0.55, linestyles='--')
            sns.kdeplot(x=x3, y=y3, ax=ax, levels=6, fill=True, cmap='viridis',
                        alpha=0.78, thresh=0.04)
        else:
            ax.hexbin(x3, y3, gridsize=40, cmap='viridis', mincnt=1)

        # Decision boundary (y == x means target == top distractor)
        lim = [min(min(x1), min(x3), min(y1), min(y3)),
               max(max(x1), max(x3), max(y1), max(y3))]
        ax.plot(lim, lim, color='red', linestyle='-.', lw=1.0, alpha=0.7,
                label='decision boundary  y=x', zorder=8)

        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel(r'$\max_{c\neq\mathrm{true}}\, y_c$  (top-distractor logit)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r'$y_{\mathrm{true}}$  (target logit)', fontsize=10)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle(
        'Paradigm 4 — Logit phase portrait (target vs top-distractor; native logit space)\n'
        'Dashed contours: t=1 distribution. Filled contours: t=3 distribution.',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Paradigm 5 - Seed-Paired Behavioral Dumbbell Plot
# ---------------------------------------------------------------------------
def render_paradigm_5_dumbbell(all_traces, save_path):
    import matplotlib.pyplot as plt

    cats = ['corrected', 'stable_correct', 'stable_incorrect', 'over_corrected']
    labels = ['Corrected\n(wrong→right)', 'Stable correct',
              'Stable incorrect', 'Over-corrected\n(right→wrong)']

    bl_pct, c2_pct = {c: [] for c in cats}, {c: [] for c in cats}
    for tr in all_traces:
        n = len(tr['trial_type_self'])
        for c in cats:
            bl_pct[c].append(100 * float((tr['trial_type_self'] == c).sum()) / n)
            c2_pct[c].append(100 * float((tr['trial_type_clone'] == c).sum()) / n)

    cb = palette_and_markers()[0]
    bl_color, c2_color = cb[0], cb[1]

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.5))

    y_positions = np.arange(len(cats))
    for i, c in enumerate(cats):
        bl_arr = np.array(bl_pct[c])
        c2_arr = np.array(c2_pct[c])
        bl_mean, c2_mean = bl_arr.mean(), c2_arr.mean()
        bl_ci = 1.96 * bl_arr.std(ddof=1) / np.sqrt(len(bl_arr))
        c2_ci = 1.96 * c2_arr.std(ddof=1) / np.sqrt(len(c2_arr))

        # Connecting line
        ax.plot([bl_mean, c2_mean], [i, i], color='gray', lw=2.0, alpha=0.6, zorder=1)
        # BL dot with CI
        ax.errorbar(bl_mean, i, xerr=bl_ci, fmt='o', color=bl_color, markersize=10,
                    elinewidth=1.5, capsize=4, zorder=4,
                    label='Baseline' if i == 0 else None,
                    markeredgecolor='black', markeredgewidth=0.6)
        # C2 dot with CI
        ax.errorbar(c2_mean, i, xerr=c2_ci, fmt='s', color=c2_color, markersize=10,
                    elinewidth=1.5, capsize=4, zorder=4,
                    label='C2 (clone)' if i == 0 else None,
                    markeredgecolor='black', markeredgewidth=0.6)
        # Difference annotation
        delta = c2_mean - bl_mean
        midpoint = (bl_mean + c2_mean) / 2
        ax.text(midpoint, i + 0.22, f'Δ = {delta:+.1f} pp',
                ha='center', va='bottom', fontsize=8.5, color='darkgray', style='italic')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(f'Percentage of trials (mean ± 95% CI across N={len(all_traces)} model seeds)',
                  fontsize=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, axis='x', alpha=0.18)

    plt.suptitle(
        'Paradigm 5 — Seed-paired behavioral dumbbell plot',
        fontsize=12, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Paradigm 6 - Logit Phase Quiver Plot (wildcard)
# ---------------------------------------------------------------------------
def render_paradigm_6_quiver(all_traces, save_path):
    import matplotlib.pyplot as plt

    def gather_states(output_key):
        """Returns (x_dist_t1, y_targ_t1, dx, dy) per trial."""
        x1, y1, dx, dy = [], [], [], []
        for tr in all_traces:
            truths = tr['true']
            for i in range(len(truths)):
                cls = int(truths[i])
                lt1 = tr[output_key][i, 0, :]
                lt3 = tr[output_key][i, 2, :]
                x_dist_1 = float(np.max(np.delete(lt1, cls)))
                y_targ_1 = float(lt1[cls])
                x_dist_3 = float(np.max(np.delete(lt3, cls)))
                y_targ_3 = float(lt3[cls])
                x1.append(x_dist_1)
                y1.append(y_targ_1)
                dx.append(x_dist_3 - x_dist_1)
                dy.append(y_targ_3 - y_targ_1)
        return (np.array(x1), np.array(y1), np.array(dx), np.array(dy))

    bl = gather_states('output_self')
    c2 = gather_states('output_clone')

    # Common grid
    all_x = np.concatenate([bl[0], c2[0]])
    all_y = np.concatenate([bl[1], c2[1]])
    nbin = 14
    x_edges = np.linspace(all_x.min(), all_x.max(), nbin + 1)
    y_edges = np.linspace(all_y.min(), all_y.max(), nbin + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    def grid_quiver(x1, y1, dx, dy):
        U = np.full((nbin, nbin), np.nan)
        V = np.full((nbin, nbin), np.nan)
        N = np.zeros((nbin, nbin))
        ix = np.digitize(x1, x_edges) - 1
        iy = np.digitize(y1, y_edges) - 1
        for k in range(len(x1)):
            i, j = ix[k], iy[k]
            if 0 <= i < nbin and 0 <= j < nbin:
                if np.isnan(U[j, i]):
                    U[j, i] = 0.0
                    V[j, i] = 0.0
                U[j, i] += dx[k]
                V[j, i] += dy[k]
                N[j, i] += 1
        with np.errstate(invalid='ignore'):
            U = np.where(N > 0, U / N, np.nan)
            V = np.where(N > 0, V / N, np.nan)
        return U, V, N

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)

    for ax, title, data in zip(
            axes,
            ['Baseline (self-feedback)', 'C2 (clone feedback)'],
            [bl, c2]):
        x1, y1, dx, dy = data
        U, V, N = grid_quiver(x1, y1, dx, dy)
        XX, YY = np.meshgrid(x_centers, y_centers)
        # Colour quiver by magnitude
        M = np.sqrt(U**2 + V**2)
        ax.quiver(XX, YY, U, V, M, angles='xy', scale_units='xy', scale=1.0,
                  cmap='viridis', alpha=0.95, width=0.005, zorder=4)
        # Light density background
        ax.scatter(x1, y1, s=1, c='lightgray', alpha=0.18, zorder=1)
        # Decision boundary
        lim = [min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())]
        ax.plot(lim, lim, color='red', linestyle='-.', lw=1.0, alpha=0.6,
                label='decision boundary y=x')

        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel(r'$\max_{c\neq\mathrm{true}}\, y_c$  (top distractor)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r'$y_{\mathrm{true}}$  (target logit)', fontsize=10)
        ax.legend(loc='lower right', fontsize=8.5)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle(
        'Paradigm 6 (wildcard) — Logit phase quiver field (mean Δ logit per grid cell)',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Paradigm 4b - Logit phase portrait with explicit centroids + migration arrows
# ---------------------------------------------------------------------------
def render_paradigm_4b_centroid_migration(all_traces, save_path):
    """Enhanced Paradigm 4: full 2D distribution + visible centroid migration.

    Adds:
      - t=1 and t=3 centroids as large markers
      - Migration arrow (t=1 -> t=3 centroid)
      - Mass-above-boundary annotation
      - Light yellow shading on the 'correct prediction' half-plane
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        have_sns = True
    except ImportError:
        have_sns = False

    cb = palette_and_markers()[0]

    def gather(t_idx, output_key):
        x_dist, y_targ = [], []
        for tr in all_traces:
            truths = tr['true']
            for i in range(len(truths)):
                cls = int(truths[i])
                logits = tr[output_key][i, t_idx, :]
                y_targ.append(float(logits[cls]))
                x_dist.append(float(np.max(np.delete(logits, cls))))
        return np.array(x_dist), np.array(y_targ)

    # Pre-compute everything for both panels
    bl_x1, bl_y1 = gather(0, 'output_self')
    bl_x3, bl_y3 = gather(2, 'output_self')
    c2_x1, c2_y1 = gather(0, 'output_clone')
    c2_x3, c2_y3 = gather(2, 'output_clone')

    bl_above_t1 = (bl_y1 > bl_x1).mean() * 100
    bl_above_t3 = (bl_y3 > bl_x3).mean() * 100
    c2_above_t1 = (c2_y1 > c2_x1).mean() * 100
    c2_above_t3 = (c2_y3 > c2_x3).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharex=True, sharey=True)

    panels = [
        ('Baseline (self-feedback)', bl_x1, bl_y1, bl_x3, bl_y3,
         bl_above_t1, bl_above_t3, cb[0]),
        ('C2 (clone feedback)',      c2_x1, c2_y1, c2_x3, c2_y3,
         c2_above_t1, c2_above_t3, cb[1]),
    ]
    zoom_min, zoom_max = -1.0, 4.5

    for ax, (title, x1, y1, x3, y3, above_t1, above_t3, color) in zip(axes, panels):
        # Light yellow shading on the "correct prediction" zone (above y = x)
        ax.fill_between([zoom_min, zoom_max], [zoom_min, zoom_max], [zoom_max, zoom_max],
                        color='gold', alpha=0.10, zorder=0)
        # Density contours: t=1 dashed (gray), t=3 filled (viridis); thresh=0.12 strips outliers
        if have_sns:
            sns.kdeplot(x=x1, y=y1, ax=ax, levels=4, color='gray',
                        linewidths=0.85, alpha=0.55, linestyles='--', thresh=0.12)
            sns.kdeplot(x=x3, y=y3, ax=ax, levels=5, fill=True, cmap='viridis',
                        alpha=0.78, thresh=0.12)
        else:
            ax.hexbin(x3, y3, gridsize=40, cmap='viridis', mincnt=1)
        # Decision boundary line
        ax.plot([zoom_min, zoom_max], [zoom_min, zoom_max], color='red',
                linestyle='-.', lw=1.0, alpha=0.7, zorder=2)
        ax.set_xlim(zoom_min, zoom_max)
        ax.set_ylim(zoom_min, zoom_max)
        # Centroids
        cx1, cy1 = x1.mean(), y1.mean()
        cx3, cy3 = x3.mean(), y3.mean()
        ax.scatter([cx1], [cy1], s=300, marker='X', c='white', edgecolors='black',
                   linewidths=1.6, zorder=8)
        ax.scatter([cx1], [cy1], s=160, marker='X', c='gray', edgecolors='black',
                   linewidths=1.0, zorder=9, label=f't=1 centroid ({cx1:.2f}, {cy1:.2f})')
        ax.scatter([cx3], [cy3], s=380, marker='*', c='white', edgecolors='black',
                   linewidths=1.6, zorder=8)
        ax.scatter([cx3], [cy3], s=240, marker='*', c=color, edgecolors='black',
                   linewidths=1.0, zorder=9,
                   label=f't=3 centroid ({cx3:.2f}, {cy3:.2f})')
        # Migration arrow (t=1 -> t=3)
        ax.annotate('', xy=(cx3, cy3), xytext=(cx1, cy1),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=2.4,
                                    mutation_scale=20),
                    zorder=10)
        # Mass-above-boundary annotation
        info_text = (f'mass above y=x:\n'
                     f'  t=1: {above_t1:4.1f}%\n'
                     f'  t=3: {above_t3:4.1f}%  (Δ {above_t3 - above_t1:+.1f} pp)')
        ax.text(0.04, 0.96, info_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='lightgray', alpha=0.92))

        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel(r'$\max_{c\neq\mathrm{true}}\, y_c$  (top-distractor logit)', fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel(r'$y_{\mathrm{true}}$  (target logit)', fontsize=10)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle(
        'Paradigm 4b — Logit phase portrait with centroid migration\n'
        'Yellow zone: above y=x → model predicts correctly. '
        'Dashed contour: t=1.  Filled contour: t=3.  Black arrow: centroid migration t=1→t=3.',
        fontsize=10.5, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# HYBRID - Paradigm 4 (logit phase portrait) + Paradigm 5 (dumbbell)
# ---------------------------------------------------------------------------
def render_hybrid_main_figure(all_traces, save_path):
    """Recommended main Figure 4: Mechanism (logit phase + centroids) + Behavior (dumbbell)."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    try:
        import seaborn as sns
        have_sns = True
    except ImportError:
        have_sns = False

    cb = palette_and_markers()[0]
    bl_color, c2_color = cb[0], cb[1]

    fig = plt.figure(figsize=(13.5, 6.6))
    # Explicit panel positioning.  No figure-level title - descriptive
    # text belongs in the LaTeX \caption{} when integrated into the paper.
    # Panels expanded to fill the freed top area.  Gap between (a)(b) widened
    # per request; gap before (c) preserved so its y-tick labels do not
    # encroach on (b).  Bottom space reserved for shared legend.
    ax_bl = fig.add_axes([0.06, 0.20, 0.18, 0.74])
    ax_c2 = fig.add_axes([0.29, 0.20, 0.18, 0.74], sharex=ax_bl, sharey=ax_bl)
    ax_db = fig.add_axes([0.59, 0.20, 0.39, 0.74])

    # ---- Panels A & B: Logit Phase Portrait + centroid migration (Paradigm 4b) ----
    def gather(t_idx, output_key):
        x_dist, y_targ = [], []
        for tr in all_traces:
            truths = tr['true']
            for i in range(len(truths)):
                cls = int(truths[i])
                logits = tr[output_key][i, t_idx, :]
                y_targ.append(float(logits[cls]))
                x_dist.append(float(np.max(np.delete(logits, cls))))
        return np.array(x_dist), np.array(y_targ)

    bl_x1, bl_y1 = gather(0, 'output_self');  bl_x3, bl_y3 = gather(2, 'output_self')
    c2_x1, c2_y1 = gather(0, 'output_clone'); c2_x3, c2_y3 = gather(2, 'output_clone')

    all_min = min(bl_x1.min(), bl_x3.min(), c2_x1.min(), c2_x3.min(),
                  bl_y1.min(), bl_y3.min(), c2_y1.min(), c2_y3.min())
    all_max = max(bl_x1.max(), bl_x3.max(), c2_x1.max(), c2_x3.max(),
                  bl_y1.max(), bl_y3.max(), c2_y1.max(), c2_y3.max())

    # Zoom to dense central region (most trials live here; outliers pulled out).
    # y_hi extended from 3.0 to 3.5 to bring (a)(b) panel height closer to
    # panel (c) without diluting data density too much.
    x_lo, x_hi = 0.0, 2.0
    y_lo, y_hi = 0.0, 3.5

    panel_specs = [
        (ax_bl, 'Baseline (self-feedback)', bl_x1, bl_y1, bl_x3, bl_y3, cb[0], 'a'),
        (ax_c2, 'C2 (clone feedback)',      c2_x1, c2_y1, c2_x3, c2_y3, cb[1], 'b'),
    ]
    for ax, title, x1, y1, x3, y3, color, letter in panel_specs:
        # Yellow shading on the "correct" half-plane - slightly stronger now that
        # the heavy viridis fill has been replaced by stroked contour lines.
        ax.fill_between([x_lo, x_hi], [x_lo, x_hi], [y_hi, y_hi],
                        color='gold', alpha=0.10, zorder=0)
        if have_sns:
            # Both timesteps as stroked contour lines on white background.
            # t=1 dashed gray, t=3 solid in panel-specific colour.
            sns.kdeplot(x=x1, y=y1, ax=ax, levels=4, color='gray',
                        linewidths=0.85, alpha=0.65, linestyles='--', thresh=0.12)
            sns.kdeplot(x=x3, y=y3, ax=ax, levels=5, color=color,
                        linewidths=1.3, alpha=0.85, thresh=0.12)
        # Decision boundary - thicker, fully opaque so it visibly bisects the space
        ax.plot([x_lo, x_hi], [x_lo, x_hi], color='red',
                linestyle='-.', lw=1.5, alpha=1.0, zorder=2)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        cx1, cy1 = x1.mean(), y1.mean()
        cx3, cy3 = x3.mean(), y3.mean()
        # White halo behind centroids
        ax.scatter([cx1], [cy1], s=220, marker='X', c='white',
                   edgecolors='white', linewidths=2.0, zorder=7)
        ax.scatter([cx3], [cy3], s=300, marker='*', c='white',
                   edgecolors='white', linewidths=2.0, zorder=7)
        ax.scatter([cx1], [cy1], s=130, marker='X', c='gray',
                   edgecolors='black', linewidths=0.8, zorder=8)
        ax.scatter([cx3], [cy3], s=190, marker='*', c=color,
                   edgecolors='black', linewidths=0.8, zorder=9)
        # Refined arrow - slimmer, smaller arrowhead so it doesn't dominate
        ax.annotate('', xy=(cx3, cy3), xytext=(cx1, cy1),
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2,
                                    mutation_scale=12), zorder=10)
        # Mass annotation: arithmetic computed from already-rounded display values
        above_t1 = (y1 > x1).mean() * 100
        above_t3 = (y3 > x3).mean() * 100
        t1_disp = round(above_t1, 1)
        t3_disp = round(above_t3, 1)
        delta_disp = round(t3_disp - t1_disp, 1)
        # Top-left placement (upper-left in axes coords) - sits cleanly inside
        # the yellow correct-zone where data density is empty; lower-right
        # placement would clip solid t=3 contour tails.
        ax.text(0.03, 0.96,
                f'mass above y=x:\n  t=1: {t1_disp:.1f}%\n  t=3: {t3_disp:.1f}% '
                f'(Δ {delta_disp:+.1f} pp)',
                transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
                multialignment='left',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='lightgray', alpha=0.95))
        ax.set_title(f'({letter})  {title}', fontsize=11, loc='left',
                     fontweight='semibold')
        ax.set_xlabel(r'$\max_{c\neq\mathrm{true}}\, y_c$  (top-distractor logit)', fontsize=9)
        ax.set_ylabel(r'$y_{\mathrm{true}}$  (target logit)', fontsize=9)
        ax.set_aspect('equal', adjustable='box')

    # Shared legend below panels (a) and (b), centred between them, with vertical
    # position aligned to the centre of panel (c)'s xlabel.  We render once to
    # determine xlabel position, then place the legend so the two visual rows
    # (legend row and xlabel row) sit at the same fig_y.
    from matplotlib.lines import Line2D
    # Single shared legend below the figure carries everything: (a)(b) glyphs
    # + (c) BL/C2 markers.  Removes the in-(c) legend that was floating over
    # data and lets the caption document contour interpretation instead of
    # cluttering the in-figure legend.
    shared_handles = [
        Line2D([0], [0], color='red', linestyle='-.', lw=1.2, label=r'decision boundary  $y=x$'),
        Line2D([0], [0], marker='X', color='gray', markersize=10, linestyle='',
               markeredgecolor='black', markeredgewidth=1.0, label='t=1 centroid'),
        Line2D([0], [0], marker='*', color='lightgray', markersize=14, linestyle='',
               markeredgecolor='black', markeredgewidth=1.0, label='t=3 centroid'),
        Line2D([0], [0], marker='o', color=bl_color, markersize=9, linestyle='',
               markeredgecolor='black', markeredgewidth=0.5, label='Baseline'),
        Line2D([0], [0], marker='s', color=c2_color, markersize=9, linestyle='',
               markeredgecolor='black', markeredgewidth=0.5, label='C2 (clone)'),
    ]
    bl_pos = ax_bl.get_position()
    c2_pos = ax_c2.get_position()
    # Centre legend across (a)(b) so it sits in the gap below those two
    # panels; expanding to all three would collide with (c)'s xlabel.
    legend_x = (bl_pos.x0 + c2_pos.x1) / 2

    # First draw to obtain xlabel position
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel_bbox = ax_db.xaxis.label.get_window_extent(renderer=renderer)
    inv = fig.transFigure.inverted()
    xlabel_y_fig = inv.transform(((xlabel_bbox.x0 + xlabel_bbox.x1) / 2,
                                  (xlabel_bbox.y0 + xlabel_bbox.y1) / 2))[1]

    # ncol=5 single row; entries are short so they fit within the (a)(b) span
    # without spilling into (c).
    fig.legend(handles=shared_handles, loc='center',
               bbox_to_anchor=(legend_x, xlabel_y_fig - 0.025),
               ncol=5, frameon=True, framealpha=0.92, fontsize=8.5,
               columnspacing=1.2, handletextpad=0.4)

    # ---- Panel C: Dumbbell (Paradigm 5) ----
    cats = ['corrected', 'stable_correct', 'stable_incorrect', 'over_corrected']
    # Math-mode rightarrows for typesetting consistency
    labels = [r'Corrected' '\n' r'(wrong $\rightarrow$ right)',
              'Stable\ncorrect',
              'Stable\nincorrect',
              r'Over-corrected' '\n' r'(right $\rightarrow$ wrong)']
    bl_pct = {c: [] for c in cats}
    c2_pct = {c: [] for c in cats}
    for tr in all_traces:
        n = len(tr['trial_type_self'])
        for c in cats:
            bl_pct[c].append(100 * float((tr['trial_type_self'] == c).sum()) / n)
            c2_pct[c].append(100 * float((tr['trial_type_clone'] == c).sum()) / n)

    # Compute paired Wilcoxon (N=20 seeds) per category, with Holm correction.
    # Exact signed-rank with float tolerance (1e-9), matching src.metrics and the
    # paper's reported tests; scipy's exact mode lacks the tie/zero tolerance.
    from src.metrics import wilcoxon_exact
    p_raw = []
    for c in cats:
        bl_arr = np.array(bl_pct[c])
        c2_arr = np.array(c2_pct[c])
        stat, p = wilcoxon_exact(bl_arr, c2_arr)
        p_raw.append(float(p))
    # Holm step-down correction across the 4 categories
    p_raw_arr = np.array(p_raw)
    order = np.argsort(p_raw_arr)
    m = len(p_raw_arr)
    p_holm = np.empty_like(p_raw_arr)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_raw_arr[idx]
        running = max(running, adj)
        p_holm[idx] = min(running, 1.0)
    # Diagnostic print so the user sees the per-category statistics
    print(f'\n[Paired Wilcoxon, BL vs C2, N={len(all_traces)} seeds, '
          f'Holm-corrected over m={len(cats)}]')
    for c, p_r, p_h in zip(cats, p_raw, p_holm):
        bl_arr = np.array(bl_pct[c]); c2_arr = np.array(c2_pct[c])
        delta_mean = c2_arr.mean() - bl_arr.mean()
        print(f'  {c:18s}  BL={bl_arr.mean():5.1f}%  C2={c2_arr.mean():5.1f}%  '
              f'Delta={delta_mean:+5.1f}pp  p_raw={p_r:.2e}  p_Holm={p_h:.2e}')

    def stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'n.s.'

    def threshold(p):
        if p < 0.001: return r'$p<0.001$'
        if p < 0.01:  return r'$p<0.01$'
        if p < 0.05:  return r'$p<0.05$'
        return r'$p\geq 0.05$'

    y_pos = np.arange(len(cats))
    # Undodged: BL and C2 sit on the same horizontal centerline so the paired
    # connector lies flat (diagonal jitter clashed with the horizontal anchor
    # gridlines). Circle vs square markers carry the BL/C2 distinction
    # without the y-offset.
    y_offset = 0.0

    # First pass: determine the right-edge maximum across all rows so the
    # delta-text column is left-aligned at a uniform x-position (cleaner than
    # per-row jagged placement).
    max_right_edge = 0.0
    cat_stats = []
    for c in cats:
        bl_arr = np.array(bl_pct[c]); c2_arr = np.array(c2_pct[c])
        bl_mean = bl_arr.mean(); c2_mean = c2_arr.mean()
        bl_ci = 1.96 * bl_arr.std(ddof=1) / np.sqrt(len(bl_arr))
        c2_ci = 1.96 * c2_arr.std(ddof=1) / np.sqrt(len(c2_arr))
        cat_stats.append((bl_mean, c2_mean, bl_ci, c2_ci))
        max_right_edge = max(max_right_edge, bl_mean + bl_ci, c2_mean + c2_ci)
    text_x_fixed = max_right_edge + 3.0  # uniform reading edge

    for i, (c, (bl_mean, c2_mean, bl_ci, c2_ci)) in enumerate(zip(cats, cat_stats)):
        bl_y = i - y_offset
        c2_y = i + y_offset
        ax_db.axhline(i, color='lightgray', lw=0.35, alpha=0.4, zorder=0)
        # Subtle paired connector (thin gray line) - encodes delta direction without
        # competing with the markers and CIs.
        ax_db.plot([bl_mean, c2_mean], [bl_y, c2_y],
                   color='lightgray', lw=0.7, alpha=0.40, zorder=1)
        ax_db.errorbar(bl_mean, bl_y, xerr=bl_ci, fmt='o', color=bl_color, markersize=10,
                       elinewidth=1.5, capsize=3.5, zorder=4,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label='Baseline' if i == 0 else None)
        ax_db.errorbar(c2_mean, c2_y, xerr=c2_ci, fmt='s', color=c2_color, markersize=10,
                       elinewidth=1.5, capsize=3.5, zorder=4,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label='C2 (clone)' if i == 0 else None)
        # delta computed strictly from rounded display means to match in-figure annotation
        bl_disp = round(bl_mean, 1); c2_disp = round(c2_mean, 1)
        delta_disp = round(c2_disp - bl_disp, 1)
        # Compact form: stars only (caption documents '** p<0.01, *** p<0.001').
        # Dropping the inline 'p<0.01' threshold text shortens the column from
        # ~28 to ~16 units so xlim stays tight without clipping.
        ax_db.text(text_x_fixed, i,
                   f'Δ = {delta_disp:+.1f} pp   {stars(p_holm[i])}',
                   ha='left', va='center', fontsize=8.5, color='gray',
                   style='italic')

    # Right buffer sized to fit the shortened "delta = +X.X pp  ***" annotation
    # (~17 units at fontsize 8.5) with a small breathing margin.
    ax_db.set_xlim(0, text_x_fixed + 18)
    # Explicit ylim adds a small top margin (negative y after invert) so the
    # "delta = C2 - Baseline" column header has clean space above row 0.
    ax_db.set_ylim(-0.55, 3.45)
    ax_db.set_yticks(y_pos)
    ax_db.set_yticklabels(labels, fontsize=9.5)
    ax_db.invert_yaxis()
    ax_db.set_xlabel(f'% of trials  (mean ± 95% CI across N={len(all_traces)} seeds)',
                     fontsize=9.5)
    # delta-column header - disambiguates delta sign convention in-figure.
    # Sits just above row 0 in the previously-empty top margin.
    ax_db.text(text_x_fixed, -0.35, r'$\Delta = \mathrm{C2} - \mathrm{Baseline}$',
               ha='left', va='center', fontsize=8.5, color='dimgray',
               style='italic')
    # Panel-(c) legend removed: BL/C2 markers now in the shared bottom legend.
    ax_db.grid(True, axis='x', alpha=0.18)
    ax_db.set_title('(c)  Trial outcome distribution', fontsize=11, loc='left',
                    fontweight='semibold')

    # No figure-level title (suptitle / figtext) - descriptive text and the
    # numeric summary belong in the LaTeX \caption{} when this figure is
    # integrated into the paper.  Save both PNG (preview) and PDF (vector).
    plt.savefig(save_path, dpi=300)
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    plt.savefig(pdf_path)
    print(f"  -> {pdf_path}")
    plt.close()
    print(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    seeds = list(range(20))
    donor_seeds = list(range(100, 120))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results', 'figure_redesign')
    os.makedirs(out_dir, exist_ok=True)

    # Separate cache file for the H2-extended schema; do not overwrite the
    # original H1-only cache.
    cache_path = os.path.join(out_dir, 'traces_cache_h2.pkl')

    # Load cache only if it has the new schema (a_h2 keys present)
    use_cache = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if cached and 'a_h2_self' in cached[0]:
                all_traces = cached
                use_cache = True
                print(f"[Cache] Loaded {len(all_traces)} seeds from {cache_path}")
        except Exception as e:
            print(f"[Cache] Skipped (load error: {e})")

    if not use_cache:
        print("[Compute] (Re-)collecting traces with extended schema (a_h1 + a_h2 + output)...")
        all_traces = collect_all_traces(seeds, donor_seeds)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_traces, f)
        print(f"  cached to {cache_path}")

    configure_style()

    # PCA on H1 (used by A and B)
    pca = fit_pca_h1(all_traces)
    print(f"\n[PCA-H1] PC1 var ratio: {pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2 var ratio: {pca.explained_variance_ratio_[1]:.3f}")

    # LDA on H2 (used by C)
    print("\n[LDA-H2 fit]")
    lda = fit_lda_h2(all_traces)

    print("\n[Render] Option A - 4-panel filter (PCA-H1)...")
    render_option_A(all_traces, pca, os.path.join(out_dir, 'option_A.png'))

    print("[Render] Option B - aggregate per-class displacement (PCA-H1)...")
    render_option_B(all_traces, pca, os.path.join(out_dir, 'option_B.png'))

    print("[Render] Option C - per-class displacement (LDA-H2)...")
    render_option_C(all_traces, lda, os.path.join(out_dir, 'option_C.png'))

    print("\n[LDA-logit fit]")
    lda_logit = fit_lda_logit(all_traces)
    print("[Render] Option D - per-class displacement (LDA on output logits, decision space)...")
    render_option_D(all_traces, lda_logit, os.path.join(out_dir, 'option_D.png'))

    print("\n=== Paradigm prototypes ===")
    print("[Render] Paradigm 1 - Magnitude x Alignment density...")
    render_paradigm_1_magnitude_alignment(all_traces, os.path.join(out_dir, 'paradigm_1.png'))
    print("[Render] Paradigm 2 - Logit margin ECDFs...")
    render_paradigm_2_margin_ecdf(all_traces, os.path.join(out_dir, 'paradigm_2.png'))
    print("[Render] Paradigm 3 - Orthogonal/parallel boxplots...")
    render_paradigm_3_orthogonal_boxplot(all_traces, os.path.join(out_dir, 'paradigm_3.png'))
    print("[Render] Paradigm 4 - Logit phase portrait...")
    render_paradigm_4_logit_phase_portrait(all_traces, os.path.join(out_dir, 'paradigm_4.png'))
    print("[Render] Paradigm 5 - Seed-paired dumbbell plot...")
    render_paradigm_5_dumbbell(all_traces, os.path.join(out_dir, 'paradigm_5.png'))
    print("[Render] Paradigm 6 - Logit phase quiver (wildcard)...")
    render_paradigm_6_quiver(all_traces, os.path.join(out_dir, 'paradigm_6.png'))

    print("\n[Render] Paradigm 4b - logit phase portrait with centroids + migration arrows...")
    render_paradigm_4b_centroid_migration(all_traces, os.path.join(out_dir, 'paradigm_4b.png'))

    print("\n=== Recommended HYBRID main figure (Paradigm 4 + 5) ===")
    print("[Render] Hybrid main figure...")
    render_hybrid_main_figure(all_traces, os.path.join(out_dir, 'hybrid_main.png'))

    print(f"\nDone. Compare options A-D, paradigms 1-6, and hybrid in {out_dir}.")


if __name__ == '__main__':
    main()
