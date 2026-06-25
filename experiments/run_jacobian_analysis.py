"""Jacobian / relative condition number analysis at the feedback receiver.

For each trained Baseline model (20 seeds) and each evaluation-time recurrent
state, computes:

    J(y) = (W_rec^T / tau) * diag(sech^2(y / tau))         shape (h1, output_size)

This is the Jacobian of the H1 pre-activation contribution from the recurrent
feedback path with respect to the previous-output vector y, with tau the
feedback-gate temperature and W_rec the (output_size, h1) recurrent weight
matrix. The factor sech^2(y/tau)/tau comes from differentiating tanh(y/tau).
We use the convention that the H1 pre-activation receives `tanh(y/tau) @ W_rec`
(shape (h1,)), so dH1/dy is W_rec^T*diag(sech^2(y/tau))/tau.

Reported per (seed, condition) summary:

    sigma_max(J)              # operator norm
    sigma_min(J)              # smallest singular value
    kappa = sigma_max/sigma_min
    frobenius_norm(J)
    nuclear_norm(J)

across four feedback conditions: self / clone / wrong-trajectory / norm-matched
clone. The point of the analysis is to test whether the local sensitivity of
the feedback-to-H1 map differs *qualitatively* across conditions when the
network actually receives those feedback signals.

Output:
  results/integration_control_jacobian.csv         per-(seed, mode) row
  results/integration_control_jacobian_summary.json  aggregated stats

This is a NEW file; the existing 18 frozen CSVs and the integration_control
artefacts are unaffected.
"""

import os, sys

os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.network import RecurrentMLP
from src.training import generate_data_variable_noise, train_vn
from src.metrics import wilcoxon_exact

N_MODELS_DEFAULT = 20
N_TRAIN          = 200
N_TEST           = 200
T                = 3
NOISE            = 0.5
TRAIN_EPOCHS     = 1000
TRAIN_LR         = 0.01
DONOR_SEED_OFFSET = 100

import itertools

KAPPA_MODES = ('self', 'clone', 'wrong_traj', 'normmatched_clone')


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values (order-preserving), clipped to [0, 1]."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate(p[order] * (n - np.arange(n)))
    adj = np.empty(n)
    adj[order] = np.clip(adj_sorted, 0.0, 1.0)
    return [float(x) for x in adj]


def kappa_contrasts(df, modes=KAPPA_MODES, n_boot=10000, seed=42):
    """Pairwise kappa contrasts reproducing the Section 3.7 PRIMARY (median) statistic.

    Reads the per-(seed, mode) jacobian table (columns kappa_median, kappa_mean) and
    computes, for BOTH the robust per-seed MEDIAN (primary) and the tail-sensitive
    per-seed MEAN (secondary): all C(len(modes), 2) pairwise paired exact Wilcoxon
    contrasts, Holm-corrected within the m=C(n,2) family, plus a bootstrap 95% CI on
    each median paired difference (descriptive). 'typical_per_mode' is the median across
    seeds of the per-seed statistic. This is the canonical derivation of the Section 3.7
    numbers from the frozen CSV (see tests/test_jacobian_kappa.py).

    kappa = sigma_max/sigma_min is a ratio whose denominator -> 0 under deep tanh
    saturation, so per-seed means are heavy-tailed; the MEDIAN is the pre-specified
    primary statistic. Per-state kappa is capped at +inf when sigma_min <= 1e-12
    (jacobian_and_spectrum); the per-seed median is robust to such tail entries.
    """
    rng = np.random.RandomState(seed)
    out = {}
    for stat, col in [('median', 'kappa_median'), ('mean', 'kappa_mean')]:
        piv = df.pivot_table(index='seed', columns='mode', values=col)
        cols = [m for m in modes if m in piv.columns]
        raw, deltas, cis = {}, {}, {}
        for a, b in itertools.combinations(cols, 2):
            sub = piv[[a, b]].dropna()
            d = sub[a].values - sub[b].values
            _, p = wilcoxon_exact(sub[a].values, sub[b].values)
            key = f"{a}_vs_{b}"
            raw[key] = float(p)
            deltas[key] = float(d.mean())
            if stat == 'median' and len(d) > 1:
                boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
                cis[key] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        keys = list(raw.keys())
        holm = holm_bonferroni([raw[k] for k in keys])
        out[stat] = {
            'statistic': 'per-seed median (PRIMARY)' if stat == 'median'
                         else 'per-seed mean (secondary, tail-sensitive)',
            'typical_per_mode': {m: float(piv[m].median()) for m in cols},
            'n_contrasts': len(keys),
            'min_raw_p': float(min(raw.values())) if raw else None,
            'all_holm_nonsignificant': bool(all(h >= 0.05 for h in holm)) if holm else None,
            'contrasts': {
                k: {'delta_mean': deltas[k], 'p_raw': raw[k], 'p_holm': holm[i],
                    **({'ci95_median_diff': cis[k]} if k in cis else {})}
                for i, k in enumerate(keys)
            },
        }
    return out


def jacobian_and_spectrum(y, W_rec, tau):
    """Compute J = (W_rec^T / tau) * diag(sech^2(y / tau)) and its singular spectrum.

    Args
    ----
    y      : previous-output vector, shape (output_size,)
    W_rec  : recurrent weight matrix, shape (output_size, h1)
    tau    : feedback-gate temperature

    Returns
    -------
    dict with sigma_max, sigma_min, kappa, frobenius_norm, nuclear_norm.
    """
    z = np.asarray(y, dtype=np.float64) / tau
    # sech^2 = 1 - tanh^2; numerically stable
    th = np.tanh(z)
    sech2 = 1.0 - th * th
    # J : shape (h1, output_size) = W_rec^T * diag(sech^2) / tau
    J = (W_rec.T * sech2[None, :]) / tau
    s = np.linalg.svd(J, compute_uv=False)
    sigma_max = float(s[0]) if s.size else 0.0
    sigma_min = float(s[-1]) if s.size else 0.0
    kappa     = (sigma_max / sigma_min) if sigma_min > 1e-12 else float('inf')
    return {
        'sigma_max':       sigma_max,
        'sigma_min':       sigma_min,
        'kappa':           kappa,
        'frobenius_norm':  float(np.linalg.norm(J, ord='fro')),
        'nuclear_norm':    float(np.sum(s)),
    }


def collect_feedback_states(target_net, X_seq, y_te, *, mode, donor_net=None,
                             rng=None, classes=None):
    """Run target on X_seq under a given feedback regime; return the list of
    previous-output vectors `y` that ENTER the tanh gate at each timestep
    transition (i.e., the feedback signals we want to take Jacobians at).

    For each test trial, returns up to T-1 vectors (the previous outputs at
    t=0..T-2 that become feedback at t=1..T-1). Vectors are taken AFTER the
    relevant substitution / rescaling so the Jacobian is evaluated at the
    actually-injected feedback.
    """
    n = len(X_seq)
    states = []
    if mode == 'self':
        for i in range(n):
            target_net.reset_state()
            outs = []
            for t in range(T):
                y = target_net.forward(X_seq[i, t])
                outs.append(y.copy())
            for t in range(T - 1):
                states.append(outs[t])
    elif mode == 'clone':
        for i in range(n):
            target_net.reset_state()
            donor_net.reset_state()
            clone_outs = []
            for t in range(T):
                clone_y = donor_net.forward(X_seq[i, t])
                clone_outs.append(clone_y.copy())
                if t > 0:
                    target_net._prev_output = clone_outs[t - 1].copy()
                    target_net._has_feedback = True
                target_net.forward(X_seq[i, t])
            for t in range(T - 1):
                states.append(clone_outs[t])
    elif mode == 'normmatched_clone':
        for i in range(n):
            # Pass 1 -- target self outputs to get reference norms
            target_net.reset_state()
            self_outs = []
            for t in range(T):
                self_outs.append(target_net.forward(X_seq[i, t]).copy())
            self_norms = [float(np.linalg.norm(s)) for s in self_outs]
            # Pass 2 -- norm-matched clone
            target_net.reset_state()
            donor_net.reset_state()
            clone_outs = []
            for t in range(T):
                clone_y = donor_net.forward(X_seq[i, t])
                clone_outs.append(clone_y.copy())
                if t > 0:
                    cn = float(np.linalg.norm(clone_outs[t - 1]))
                    scale = self_norms[t - 1] / (cn + 1e-8)
                    target_net._prev_output = clone_outs[t - 1] * scale
                    target_net._has_feedback = True
                target_net.forward(X_seq[i, t])
            # The actually-injected vector is scale * clone_outs[t-1]
            for t in range(T - 1):
                cn = float(np.linalg.norm(clone_outs[t]))
                scale = self_norms[t] / (cn + 1e-8)
                states.append(clone_outs[t] * scale)
    elif mode == 'wrong_traj':
        # Self-feedback from a different class-matched trial.
        # The wrong-trajectory intervention substitutes trial j's self-unrolled
        # outputs as feedback for trial i at every transition t in 1..T-1, so
        # the Jacobian must be evaluated at trial j's outputs at t=0..T-2
        # (the un-tanh-ed vectors that enter the tanh gate). Previously this
        # branch only collected t=0, yielding n_states=200 vs 400 elsewhere.
        self_traj = []  # self_traj[i] = list of (T-1) per-timestep outputs
        for i in range(n):
            target_net.reset_state()
            trial_outs = []
            for t in range(T):
                y = target_net.forward(X_seq[i, t])
                if t < T - 1:
                    trial_outs.append(y.copy())
            self_traj.append(trial_outs)
        if classes is None:
            classes = np.array([int(np.argmax(yi)) for yi in y_te])
        class_groups = {}
        for i in range(n):
            class_groups.setdefault(int(classes[i]), []).append(i)
        for i in range(n):
            c = int(classes[i])
            cands = [j for j in class_groups[c] if j != i]
            if not cands:
                continue
            j = cands[rng.randint(len(cands))]
            for t in range(T - 1):
                states.append(self_traj[j][t])
    else:
        raise ValueError(f"unknown mode: {mode}")
    return states


def run_seed(seed_model):
    rec_rows = []
    t0 = time.time()

    target_vn = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_sq_tr, y_tr = generate_data_variable_noise(N_TRAIN, NOISE, T=T, seed=seed_model)
    train_vn(target_vn, X_sq_tr, y_tr, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    donor_seed = seed_model + DONOR_SEED_OFFSET
    donor_vn = RecurrentMLP(10, 10, 10, 5, seed=donor_seed)
    X_sq_d, y_d = generate_data_variable_noise(N_TRAIN, NOISE, T=T, seed=donor_seed)
    train_vn(donor_vn, X_sq_d, y_d, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    X_sq_te, y_te = generate_data_variable_noise(N_TEST, NOISE, T=T, seed=seed_model + 500)
    classes = np.array([int(np.argmax(yi)) for yi in y_te])
    rng = np.random.RandomState(seed_model + 7000)

    W_rec = target_vn.W_rec
    tau   = float(target_vn.feedback_tau)

    for mode in ['self', 'clone', 'wrong_traj', 'normmatched_clone']:
        states = collect_feedback_states(
            target_vn, X_sq_te, y_te, mode=mode,
            donor_net=donor_vn if mode in ('clone', 'normmatched_clone') else None,
            rng=rng, classes=classes,
        )
        # Compute spectral stats per state, then aggregate per (seed, mode)
        kap_list, smax_list, smin_list, fro_list, nuc_list = [], [], [], [], []
        for y_state in states:
            stat = jacobian_and_spectrum(y_state, W_rec, tau)
            if np.isfinite(stat['kappa']):
                kap_list.append(stat['kappa'])
            smax_list.append(stat['sigma_max'])
            smin_list.append(stat['sigma_min'])
            fro_list.append(stat['frobenius_norm'])
            nuc_list.append(stat['nuclear_norm'])
        rec_rows.append({
            'seed':           seed_model,
            'mode':           mode,
            'n_states':       len(states),
            'sigma_max_mean': float(np.mean(smax_list)) if smax_list else float('nan'),
            'sigma_max_sd':   float(np.std(smax_list, ddof=0)) if smax_list else float('nan'),
            'sigma_min_mean': float(np.mean(smin_list)) if smin_list else float('nan'),
            'sigma_min_sd':   float(np.std(smin_list, ddof=0)) if smin_list else float('nan'),
            'kappa_mean':     float(np.mean(kap_list)) if kap_list else float('nan'),
            'kappa_median':   float(np.median(kap_list)) if kap_list else float('nan'),
            'kappa_sd':       float(np.std(kap_list, ddof=0)) if kap_list else float('nan'),
            'fro_mean':       float(np.mean(fro_list)) if fro_list else float('nan'),
            'nuc_mean':       float(np.mean(nuc_list)) if nuc_list else float('nan'),
        })
    rec_rows[0]['elapsed_s'] = time.time() - t0
    return rec_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-models', type=int, default=N_MODELS_DEFAULT)
    parser.add_argument('--workers',  type=int, default=1)
    parser.add_argument('--out-csv',  default='results/integration_control_jacobian.csv')
    parser.add_argument('--out-json', default='results/integration_control_jacobian_summary.json')
    args = parser.parse_args()

    print(f"[JACOBIAN] N_MODELS={args.n_models} WORKERS={args.workers} NOISE={NOISE}")
    t0 = time.time()
    seeds = list(range(args.n_models))

    all_rows = []
    if args.workers <= 1:
        for s in seeds:
            try:
                rows = run_seed(s)
                all_rows.extend(rows)
                t = rows[0].get('elapsed_s', 0)
                kappa_self = next(r['kappa_mean'] for r in rows if r['mode'] == 'self')
                kappa_clone = next(r['kappa_mean'] for r in rows if r['mode'] == 'clone')
                print(f"  [seed={s:2d}] elapsed={t:.1f}s  kappa(self)={kappa_self:.3f}  kappa(clone)={kappa_clone:.3f}")
            except Exception as e:
                print(f"  [seed={s}] FAILED: {e}")
                import traceback; traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_seed, s): s for s in seeds}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                    t = rows[0].get('elapsed_s', 0)
                    kappa_self = next(r['kappa_mean'] for r in rows if r['mode'] == 'self')
                    kappa_clone = next(r['kappa_mean'] for r in rows if r['mode'] == 'clone')
                    print(f"  [seed={s:2d}] elapsed={t:.1f}s  kappa(self)={kappa_self:.3f}  kappa(clone)={kappa_clone:.3f}")
                except Exception as e:
                    print(f"  [seed={s}] FAILED: {e}")
                    import traceback; traceback.print_exc()

    df = pd.DataFrame(all_rows).sort_values(['seed', 'mode']).reset_index(drop=True)
    # Fail loud: every seed must have produced its rows (no silent partial CSV).
    n_done = df['seed'].nunique() if 'seed' in df.columns else 0
    if n_done != args.n_models:
        raise SystemExit(
            f"[JACOBIAN] FATAL: only {n_done}/{args.n_models} seeds produced rows; "
            "refusing to write a partial CSV. Investigate the failed seed(s) above.")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")
    print(f"Total elapsed: {time.time() - t0:.1f}s ({(time.time()-t0)/60:.1f}m)")

    # Aggregate summary
    print("\n" + "=" * 70)
    print("kappa (relative condition number) per mode (mean over seeds, ddof=0)")
    print("=" * 70)
    summary = {'n_models': int(df['seed'].nunique()), 'modes': {}}
    pivot_kappa = df.pivot(index='seed', columns='mode', values='kappa_mean')
    pivot_smax = df.pivot(index='seed', columns='mode', values='sigma_max_mean')
    pivot_smin = df.pivot(index='seed', columns='mode', values='sigma_min_mean')
    print(f"  {'mode':>20s} | {'kappa mean':>8s} | {'kappa sd':>6s} | "
          f"{'sigma_max mean':>10s} | {'sigma_min mean':>10s}")
    print(f"  {'-'*20} | {'-'*8} | {'-'*6} | {'-'*10} | {'-'*10}")
    for mode in ['self', 'clone', 'wrong_traj', 'normmatched_clone']:
        if mode not in pivot_kappa.columns:
            continue
        k = pivot_kappa[mode].values
        smax = pivot_smax[mode].values
        smin = pivot_smin[mode].values
        print(f"  {mode:>20s} | {np.mean(k):>8.3f} | {np.std(k, ddof=0):>6.3f} | "
              f"{np.mean(smax):>10.4f} | {np.mean(smin):>10.5f}")
        summary['modes'][mode] = {
            'kappa_mean':     float(np.mean(k)),
            'kappa_sd':       float(np.std(k, ddof=0)),
            'kappa_median_typical': float(df[df['mode'] == mode]['kappa_median'].median()),
            'sigma_max_mean': float(np.mean(smax)),
            'sigma_min_mean': float(np.mean(smin)),
        }

    # Pairwise kappa contrasts: all C(4,2)=6 contrasts for BOTH the per-seed MEDIAN
    # (Section 3.7 PRIMARY, robust to the heavy kappa tails) and the per-seed MEAN
    # (secondary), each Holm-corrected within the m=6 family. Reproduces the Section 3.7
    # numbers from the per-seed table; verified by tests/test_jacobian_kappa.py.
    summary['kappa_contrasts'] = kappa_contrasts(df)
    for stat in ('median', 'mean'):
        kc = summary['kappa_contrasts'][stat]
        print(f"\nPairwise kappa -- {kc['statistic']} (Holm m={kc['n_contrasts']}; "
              f"min raw p={kc['min_raw_p']:.3f}):")
        for k, v in sorted(kc['contrasts'].items(), key=lambda x: x[1]['p_raw']):
            print(f"  {k:34s} delta={v['delta_mean']:+9.3f}  raw={v['p_raw']:.3f}  "
                  f"Holm={v['p_holm']:.3f}  {'SIG' if v['p_holm'] < 0.05 else 'n.s.'}")

    summary['interpretation_note'] = (
        "If κ(clone) and κ(self) differ qualitatively across seeds, the local "
        "sensitivity of the W_rec map at clone-feedback states is structurally "
        "different from at self-feedback states, supporting a trajectory-specific "
        "reading of the C1/C2 dissociation. If κ values are similar across modes, "
        "the hedged sensitivity reading in the paper's discussion applies: "
        "the local-linear sensitivity tool is uninformative for this geometry, and "
        "the intervention experiments (R5b NormMatched) carry the sensitivity "
        "question instead."
    )

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == '__main__':
    main()
