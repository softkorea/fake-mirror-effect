"""Regenerate closed_loop_alignment_bl_selffeedback_magnitudes.csv (de-orphan).

Regenerates the baseline self-feedback magnitudes CSV that Appendix G.3 cites,
recomputed from the same converged targets Appendix G uses so it is reproducible
and pipeline-managed (reuses run_closed_loop_alignment.make_target_and_donor +
the seed+500 frozen test cohort).

Definition: for the recurrent BASELINE (target) on the frozen VN test cohort, the
self-feedback signal at timestep t is the model's own output logits that feed back.
y1 = output at t=0 (feeds into t=1), y2 = output at t=1 (feeds into t=2). Reported
per seed over all (sample, output-dim): mean/max |y| and the deep/shallow
saturation fractions frac|y|>4 (tanh(4/tau=2)=tanh(2)~0.96) and frac|y|>2.

Usage:
    python experiments/regen_bl_selffeedback.py            # 20 seeds -> results/
    python experiments/regen_bl_selffeedback.py --n-seeds 2 --out /tmp/x.csv   # quick test
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import experiments.run_closed_loop_alignment as cl

FIELDS = ["seed", "mean_y1", "max_y1", "mean_y2", "max_y2",
          "frac_y1_gt4", "frac_y2_gt4", "frac_y1_gt2", "frac_y2_gt2"]


def selffeedback_row(seed: int) -> dict:
    target, _ = cl.make_target_and_donor(seed)          # converged, frozen (same as App-G)
    X_te, _, _ = cl.generate_data_vn(cl.N_TEST, seed=seed + 500)  # frozen test cohort
    with torch.no_grad():
        outs = target(torch.from_numpy(X_te), T=cl.T, feedback_mode="self")
    y1 = outs[0].abs().cpu().numpy().ravel()            # feedback into t=1
    y2 = outs[1].abs().cpu().numpy().ravel()            # feedback into t=2
    return {
        "seed": seed,
        "mean_y1": float(y1.mean()), "max_y1": float(y1.max()),
        "mean_y2": float(y2.mean()), "max_y2": float(y2.max()),
        "frac_y1_gt4": float((y1 > 4).mean()), "frac_y2_gt4": float((y2 > 4).mean()),
        "frac_y1_gt2": float((y1 > 2).mean()), "frac_y2_gt2": float((y2 > 2).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--out", default="results/closed_loop_alignment_bl_selffeedback_magnitudes.csv")
    args = ap.parse_args()

    rows = []
    for s in range(args.n_seeds):
        r = selffeedback_row(s)
        rows.append(r)
        print(f"  [seed={s:2d}] mean_y1={r['mean_y1']:.2f} mean_y2={r['mean_y2']:.2f} "
              f"frac_y1>4={r['frac_y1_gt4']*100:.1f}% frac_y2>4={r['frac_y2_gt4']*100:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    agg = {k: np.mean([r[k] for r in rows]) for k in FIELDS if k != "seed"}
    print(f"\nWrote {args.out}  (N={len(rows)})")
    print(f"  mean|y1|={agg['mean_y1']:.2f}  mean|y2|={agg['mean_y2']:.2f}  "
          f"frac|y1|>4={agg['frac_y1_gt4']*100:.1f}%  frac|y2|>4={agg['frac_y2_gt4']*100:.1f}%")


if __name__ == "__main__":
    main()
