"""Extra analyses for the R4 evidence-accumulation control.

Three independent analyses, each writing its own NEW CSV:

P1.1 - Static-input control (mathematical shield, empirical row).
   For static input (x_1 = x_2 = x_3), any stateless aggregator
   MUST yield acc_t3 = acc_t1 by determinism. We empirically verify
   this with E1 (Baseline weights, recurrence disabled at evaluation)
   on static test data, and contrast with the recurrent Baseline's
   non-zero static gain. Output: results/integration_control_static.csv

P1.2 - Wrong-trajectory cosine divergence (R5 angular gap).
   We measure cosine divergence on `tanh(y/tau) @ W_rec` for the
   wrong-trajectory feedback condition (same model, different
   class-matched trial). This addresses R5's angular-magnitude
   concern using the same metric as the main run.
   Output: results/integration_control_wrongtraj.csv

P1.3 - Noise sweep on E1 (R4 robustness across sigma).
   At test noise sigma  in  {0.1, 0.3, 0.5, 0.7, 1.0}, compute bl_gain and
   e1_gain_logprod / e1_gain_probmean. Reports the integration
   fraction across noise levels. Models are trained at sigma=0.5 (per
   the paper); only the test noise varies.
   Output: results/integration_control_noisesweep.csv

Reuses run_ensemble and stats helpers from run_integration_control.py.
No E3 / ConcatFF early-fusion fit is performed here (these analyses train
only the recurrent Baseline via train / train_vn).
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
from src.training import (
    generate_data, generate_data_variable_noise, train, train_vn,
)
from src.metrics import wilcoxon_exact

from experiments.run_integration_control import (
    _cos_divergence_safe, run_ensemble, paired_bootstrap_ci,
)


N_MODELS_DEFAULT  = 20
N_TRAIN           = 200
N_TEST            = 200
T                 = 3
NOISE_TRAIN       = 0.5
TRAIN_EPOCHS      = 1000
TRAIN_LR          = 0.01
# (No E3_LR / E3_EPOCHS here: this script performs NO ConcatFF early-fusion
#  ceiling fit -- only recurrent Baseline training via train / train_vn. The E3
#  ceiling lives in run_integration_control.py, where it is decoupled at
#  E3_EPOCHS=500 / E3_LR=0.10.)


# ===========================================================
# P1.1: Static-input control
# ===========================================================

def static_seed(seed_model):
    """Train BL on static AND VN; evaluate both on static test.

    Reports BL_static_gain (matches paper +0.036), E1 (BL_vn, recurrence off)
    on static (must be exactly 0 by determinism), and BL_vn-recurrent on static.
    """
    rec = {'seed': seed_model}
    t0 = time.time()

    # Static-trained Baseline
    bl_st = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_tr_st, y_tr_st = generate_data(N_TRAIN, NOISE_TRAIN, seed=seed_model)
    train(bl_st, X_tr_st, y_tr_st, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    # VN-trained Baseline
    bl_vn = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_sq_tr, y_sq_tr = generate_data_variable_noise(N_TRAIN, NOISE_TRAIN, T=T, seed=seed_model)
    train_vn(bl_vn, X_sq_tr, y_sq_tr, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    # Static test data (paper convention: seed + 500)
    X_te_st, y_te_st = generate_data(N_TEST, NOISE_TRAIN, seed=seed_model + 500)
    # Reformat for forward_sequence_vn: same input at every t
    X_seq_static = np.tile(X_te_st[:, None, :], (1, T, 1))  # (N_TEST, T, 10)

    # -- BL_static recurrent on static --
    res = run_ensemble(bl_st, X_seq_static, y_te_st, recurrent_on=True)
    rec['bl_static_acc_t1']      = res['acc_t1']
    rec['bl_static_acc_t3']      = res['acc_t3']
    rec['bl_static_gain']        = res['acc_t3'] - res['acc_t1']

    # -- E1 (BL_vn, recurrence DISABLED) on static --
    res = run_ensemble(bl_vn, X_seq_static, y_te_st, recurrent_on=False)
    rec['e1_on_static_acc_t1']         = res['acc_t1']
    rec['e1_on_static_acc_t3']         = res['acc_t3']
    rec['e1_on_static_gain']           = res['acc_t3'] - res['acc_t1']
    rec['e1_on_static_gain_probmean']  = res['acc_probmean'] - res['acc_t1']
    rec['e1_on_static_gain_logprod']   = res['acc_logprod']  - res['acc_t1']

    # -- BL_vn recurrent on static (curiosity: does VN-trained recurrent help on static?) --
    res = run_ensemble(bl_vn, X_seq_static, y_te_st, recurrent_on=True)
    rec['bl_vn_on_static_acc_t1']  = res['acc_t1']
    rec['bl_vn_on_static_acc_t3']  = res['acc_t3']
    rec['bl_vn_on_static_gain']    = res['acc_t3'] - res['acc_t1']

    rec['elapsed_s'] = time.time() - t0
    return rec


# ===========================================================
# P1.2: Wrong-trajectory cosine divergence
# ===========================================================

def wrong_traj_seed(seed_model):
    """Compute cosine divergence on tanh(y/tau) @ W_rec for wrong-trajectory
    feedback (same model's output on a different class-matched trial).

    Compares to the main-run clone divergence (~0.635).
    """
    rec = {'seed': seed_model}
    t0 = time.time()

    bl_vn = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_sq_tr, y_sq_tr = generate_data_variable_noise(N_TRAIN, NOISE_TRAIN, T=T, seed=seed_model)
    train_vn(bl_vn, X_sq_tr, y_sq_tr, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    X_sq_te, y_te = generate_data_variable_noise(N_TEST, NOISE_TRAIN, T=T, seed=seed_model + 500)
    n = len(X_sq_te)
    tau = float(bl_vn.feedback_tau)
    W_rec = bl_vn.W_rec

    # 1) Self-feedback outputs at t=0 (this is the feedback that would enter t=1)
    self_outputs_t0 = []
    true_classes = []
    for i in range(n):
        bl_vn.reset_state()
        y0 = bl_vn.forward(X_sq_te[i, 0])
        self_outputs_t0.append(y0.copy())
        true_classes.append(int(np.argmax(y_te[i])))
    true_classes = np.array(true_classes)

    # 2) Class groups for wrong-trajectory partner sampling
    class_groups = {}
    for i in range(n):
        c = int(true_classes[i])
        class_groups.setdefault(c, []).append(i)

    # 3) For each trial i, find a class-matched j != i. Use j's self-feedback as
    #    "wrong-trajectory feedback" for trial i. Compute cosine divergence.
    rng = np.random.RandomState(seed_model + 7000)
    wrong_traj_divs = []
    same_class_self_divs = []   # paper's "resampled self" - same class, different trial
    clone_seed = seed_model + 100  # for direct comparison to main-run clone divergence
    clone_net = RecurrentMLP(10, 10, 10, 5, seed=clone_seed)
    X_sq_cl, y_cl = generate_data_variable_noise(N_TRAIN, NOISE_TRAIN, T=T, seed=clone_seed)
    train_vn(clone_net, X_sq_cl, y_cl, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)
    clone_outputs_t0 = []
    for i in range(n):
        clone_net.reset_state()
        y0c = clone_net.forward(X_sq_te[i, 0])
        clone_outputs_t0.append(y0c.copy())
    clone_divs = []

    for i in range(n):
        self_y = self_outputs_t0[i]
        self_contrib = np.tanh(self_y / tau) @ W_rec
        c = int(true_classes[i])
        candidates = [j for j in class_groups[c] if j != i]
        if not candidates:
            continue
        j = candidates[rng.randint(len(candidates))]
        wrong_y = self_outputs_t0[j]
        wrong_contrib = np.tanh(wrong_y / tau) @ W_rec
        wrong_traj_divs.append(_cos_divergence_safe(self_contrib, wrong_contrib))
        same_class_self_divs.append(_cos_divergence_safe(self_contrib, wrong_contrib))
        clone_y = clone_outputs_t0[i]
        clone_contrib = np.tanh(clone_y / tau) @ W_rec
        clone_divs.append(_cos_divergence_safe(self_contrib, clone_contrib))

    rec['n_pairs']                 = len(wrong_traj_divs)
    rec['cos_div_wrong_traj_mean'] = float(np.mean(wrong_traj_divs))
    rec['cos_div_wrong_traj_sd']   = float(np.std(wrong_traj_divs, ddof=0))
    rec['cos_div_clone_mean']      = float(np.mean(clone_divs))
    rec['cos_div_clone_sd']        = float(np.std(clone_divs, ddof=0))
    rec['elapsed_s'] = time.time() - t0
    return rec


# ===========================================================
# P1.3: Noise sweep on E1
# ===========================================================

NOISE_SWEEP_DEFAULT = (0.1, 0.3, 0.5, 0.7, 1.0)


def noise_sweep_seed(seed_model, noise_levels=NOISE_SWEEP_DEFAULT):
    """Train BL_vn at sigma=0.5; evaluate at multiple test-time sigma.

    For each test sigma: report bl_gain (recurrent t3 - t1) and
    e1_gain_logprod / e1_gain_probmean (within-network ensemble).
    """
    rows = []
    t0 = time.time()

    bl_vn = RecurrentMLP(10, 10, 10, 5, seed=seed_model)
    X_sq_tr, y_sq_tr = generate_data_variable_noise(N_TRAIN, NOISE_TRAIN, T=T, seed=seed_model)
    train_vn(bl_vn, X_sq_tr, y_sq_tr, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, T=T)

    for sigma in noise_levels:
        X_te, y_te = generate_data_variable_noise(N_TEST, sigma, T=T, seed=seed_model + 500)
        # Recurrent baseline
        res_bl = run_ensemble(bl_vn, X_te, y_te, recurrent_on=True)
        # E1: same weights, recurrence disabled
        res_e1 = run_ensemble(bl_vn, X_te, y_te, recurrent_on=False)
        rows.append({
            'seed': seed_model,
            'noise': float(sigma),
            'bl_acc_t1':            res_bl['acc_t1'],
            'bl_acc_t3':            res_bl['acc_t3'],
            'bl_gain':              res_bl['acc_t3'] - res_bl['acc_t1'],
            'e1_acc_t1':            res_e1['acc_t1'],
            'e1_acc_t3':            res_e1['acc_t3'],
            'e1_acc_probmean':      res_e1['acc_probmean'],
            'e1_acc_logprod':       res_e1['acc_logprod'],
            'e1_gain_t1_to_t3':     res_e1['acc_t3']       - res_e1['acc_t1'],
            'e1_gain_probmean':     res_e1['acc_probmean'] - res_e1['acc_t1'],
            'e1_gain_logprod':      res_e1['acc_logprod']  - res_e1['acc_t1'],
        })
    rows[0]['elapsed_s'] = time.time() - t0
    return rows


# ===========================================================
# Runners
# ===========================================================

def run_seeds_seq(seed_list, fn, label):
    out = []
    for s in seed_list:
        try:
            r = fn(s)
            out.append(r)
            if isinstance(r, dict):
                t = r.get('elapsed_s', 0)
                print(f"  [{label} seed={s:2d}] elapsed={t:.1f}s")
            else:
                t = r[0].get('elapsed_s', 0) if r else 0
                print(f"  [{label} seed={s:2d}] elapsed={t:.1f}s ({len(r)} rows)")
        except Exception as e:
            print(f"  [{label} seed={s}] FAILED: {e}")
            import traceback; traceback.print_exc()
    # Fail loud: a canonical run must not silently proceed with a partial set.
    if len(out) != len(seed_list):
        raise SystemExit(
            f"[{label}] FATAL: only {len(out)}/{len(seed_list)} seeds succeeded; "
            "refusing to proceed with a partial result set.")
    return out


def run_seeds_par(seed_list, fn, label, n_workers):
    out = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(fn, s): s for s in seed_list}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                r = fut.result()
                out.append(r)
                if isinstance(r, dict):
                    t = r.get('elapsed_s', 0)
                    print(f"  [{label} seed={s:2d}] elapsed={t:.1f}s")
                else:
                    t = r[0].get('elapsed_s', 0) if r else 0
                    print(f"  [{label} seed={s:2d}] elapsed={t:.1f}s ({len(r)} rows)")
            except Exception as e:
                print(f"  [{label} seed={s}] FAILED: {e}")
                import traceback; traceback.print_exc()
    return out


# ===========================================================
# Main
# ===========================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-models', type=int, default=N_MODELS_DEFAULT)
    parser.add_argument('--workers',  type=int, default=1)
    parser.add_argument('--phases',   type=str, default='static,wrong_traj,noise_sweep',
                        help='comma-separated subset of {static, wrong_traj, noise_sweep}')
    parser.add_argument('--out-dir',  type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.n_models))
    selected = set(p.strip() for p in args.phases.split(','))

    runner = run_seeds_seq if args.workers <= 1 else \
             (lambda L, fn, lab: run_seeds_par(L, fn, lab, args.workers))

    summary = {'n_models': args.n_models, 'phases': sorted(selected)}

    # -- P1.1 Static-input control --
    if 'static' in selected:
        print("\n" + "=" * 70)
        print("P1.1 Static-input control (mathematical shield empirical)")
        print("=" * 70)
        recs = runner(seeds, static_seed, 'static')
        df_static = pd.DataFrame(recs).sort_values('seed').reset_index(drop=True)
        out_path = os.path.join(args.out_dir, 'integration_control_static.csv')
        df_static.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

        # Summary
        bl_st = df_static['bl_static_gain'].values
        e1_st = df_static['e1_on_static_gain'].values
        e1_st_lp = df_static['e1_on_static_gain_logprod'].values
        e1_st_pm = df_static['e1_on_static_gain_probmean'].values
        bl_vn_st = df_static['bl_vn_on_static_gain'].values

        print(f"  BL_static gain on static                         : {bl_st.mean():+.4f} (sd {bl_st.std(ddof=0):.4f})  [paper: +0.036]")
        print(f"  E1 (BL_vn, recurrence off) gain on static (t1->t3): {e1_st.mean():+.4f}  [shield prediction: 0.0000]")
        print(f"  E1 logprod ensemble gain on static               : {e1_st_lp.mean():+.4f}")
        print(f"  E1 probmean ensemble gain on static              : {e1_st_pm.mean():+.4f}")
        print(f"  BL_vn (recurrent) gain on static (curiosity)     : {bl_vn_st.mean():+.4f}")
        max_e1_static = float(np.max(np.abs(e1_st)))
        print(f"  max |E1 on static gain| over seeds               : {max_e1_static:.6f}  (must be 0.0 exactly if shield holds)")
        # Mathematical shield: a no-recurrence ensemble on static input must
        # have zero correction gain by construction (the receiver depends only
        # on x_t which is constant across timesteps under static input). Any
        # silent regression that introduces timestep-dependent state into this
        # path would break a load-bearing paper claim; assert strictly so the
        # CI / next rerun fails loudly rather than passing with a tainted CSV.
        assert max_e1_static < 1e-12, (
            f"Mathematical shield broken: E1 static gain max-abs = "
            f"{max_e1_static:.3e} exceeds 1e-12 tolerance. A no-recurrence "
            f"ensemble on STATIC input must yield exactly zero correction "
            f"gain by construction (constant x_t across t). Investigate "
            f"upstream changes to the integration_control / E1 path before "
            f"re-running."
        )
        summary['static'] = {
            'bl_static_gain_mean':          float(bl_st.mean()),
            'e1_on_static_gain_mean':       float(e1_st.mean()),
            'e1_on_static_gain_max_abs':    max_e1_static,
            'e1_on_static_gain_logprod_mean':  float(e1_st_lp.mean()),
            'e1_on_static_gain_probmean_mean': float(e1_st_pm.mean()),
            'bl_vn_on_static_gain_mean':    float(bl_vn_st.mean()),
        }

    # -- P1.2 Wrong-trajectory cosine divergence --
    if 'wrong_traj' in selected:
        print("\n" + "=" * 70)
        print("P1.2 Wrong-trajectory cosine divergence (R5 angular gap)")
        print("=" * 70)
        recs = runner(seeds, wrong_traj_seed, 'wrongtraj')
        df_wt = pd.DataFrame(recs).sort_values('seed').reset_index(drop=True)
        out_path = os.path.join(args.out_dir, 'integration_control_wrongtraj.csv')
        df_wt.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

        wt = df_wt['cos_div_wrong_traj_mean'].values
        cl = df_wt['cos_div_clone_mean'].values
        T_w, p_w = wilcoxon_exact(wt, cl)
        delta_lo, delta_hi = paired_bootstrap_ci(cl - wt)
        print(f"  cos_div(self vs wrong-traj feedback): {wt.mean():.4f} (sd {wt.std(ddof=0):.4f})")
        print(f"  cos_div(self vs clone feedback)     : {cl.mean():.4f} (sd {cl.std(ddof=0):.4f})  [main run: 0.635]")
        print(f"  delta(clone - wrong_traj)               : {(cl - wt).mean():+.4f} [CI {delta_lo:+.4f}, {delta_hi:+.4f}]")
        print(f"  paired Wilcoxon: T={T_w:.1f}  p={p_w:.5f}")
        summary['wrong_traj'] = {
            'cos_div_wrong_traj_mean': float(wt.mean()),
            'cos_div_wrong_traj_sd':   float(wt.std(ddof=0)),
            'cos_div_clone_mean':      float(cl.mean()),
            'cos_div_clone_sd':        float(cl.std(ddof=0)),
            'delta_clone_minus_wt':    float((cl - wt).mean()),
            'delta_ci_95':             [delta_lo, delta_hi],
            'paired_wilcoxon_T':       float(T_w),
            'paired_wilcoxon_p':       float(p_w),
        }

    # -- P1.3 Noise sweep on E1 --
    if 'noise_sweep' in selected:
        print("\n" + "=" * 70)
        print("P1.3 Noise sweep on E1 (R4 robustness across sigma)")
        print("=" * 70)
        results = runner(seeds, noise_sweep_seed, 'noise')
        # results is list of list-of-rows
        flat = [row for seed_rows in results for row in seed_rows]
        df_ns = pd.DataFrame(flat).sort_values(['noise', 'seed']).reset_index(drop=True)
        out_path = os.path.join(args.out_dir, 'integration_control_noisesweep.csv')
        df_ns.to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")

        # Per-noise summary
        sweep = {}
        print(f"\n  {'noise':>6s} | {'bl_gain':>8s} | {'e1_lp_gain':>10s} | {'ratio':>6s} | {'p_paired':>8s}")
        print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*6} | {'-'*8}")
        for sigma in NOISE_SWEEP_DEFAULT:
            sub = df_ns[df_ns.noise == sigma]
            bl_g = sub['bl_gain'].values
            e1_g = sub['e1_gain_logprod'].values
            try:
                T_w, p_w = wilcoxon_exact(bl_g, e1_g)
            except Exception:
                T_w, p_w = float('nan'), float('nan')
            ratio = e1_g.mean() / bl_g.mean() if bl_g.mean() != 0 else float('nan')
            print(f"  {sigma:>6.2f} | {bl_g.mean():+.4f}  | {e1_g.mean():+.4f}    | {ratio:>5.3f}  | {p_w:>.5f}")
            sweep[f'sigma_{sigma}'] = {
                'bl_gain_mean':        float(bl_g.mean()),
                'e1_gain_logprod_mean': float(e1_g.mean()),
                'ratio_e1_over_bl':    float(ratio) if np.isfinite(ratio) else None,
                'paired_wilcoxon_p':   float(p_w),
            }
        summary['noise_sweep'] = sweep

    # -- Write summary JSON --
    out_json = os.path.join(args.out_dir, 'integration_control_extras_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == '__main__':
    main()
