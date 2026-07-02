"""C2-inversion hyperparameter spot-check.

The "fake-mirror inversion" (clone feedback C2 worse than no-feedback Group A) is the
headline phenomenon but is reported only at the primary (tau=2.0, w1=0, T=3) training
configuration. This spot-checks whether the STATIC inversion (C2 < A, and C2 < 0) still
holds at alternative tau and w1, using the exact paper methodology (src eval functions,
35-neuron h=10, per-model held-out test seed s+500).

Configs: primary (2.0, 0.0) for reproduction + (1.5, 0.0), (3.0, 0.0), (2.0, 0.1).

Output: results/c2_inversion_hyperparam_spotcheck.csv  (+ printed per-config verdict)
Usage:  python experiments/run_c2_inversion_spotcheck.py
"""
import os, sys, csv, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.network import RecurrentMLP
from src.training import generate_data, train
from src.metrics import compute_all_metrics, compute_all_metrics_with_clone
from src.ablation import deep_copy_weights, restore_weights

SEEDS = list(range(20))
DONOR = list(range(100, 120))
NOISE = 0.5
N = 200
EPOCHS = 1000
LR = 0.01
CONFIGS = [(2.0, 0.0), (1.5, 0.0), (3.0, 0.0), (2.0, 0.1)]   # (tau, w1)
OUT = "results/c2_inversion_hyperparam_spotcheck.csv"


def train_one(seed, tau, w1):
    net = RecurrentMLP(input_size=10, hidden1=10, hidden2=10, output_size=5,
                       seed=seed, feedback_tau=tau)
    X, y = generate_data(N, noise_level=NOISE, seed=seed)
    train(net, X, y, epochs=EPOCHS, lr=LR, time_weights=[w1, 0.2, 1.0])
    return net


def eval_static(net, clone, Xte, yte):
    saved = deep_copy_weights(net)
    bl = compute_all_metrics(net, Xte, yte)["gain"]
    net.disable_recurrent_loop()
    a = compute_all_metrics(net, Xte, yte)["gain"]
    net.enable_recurrent_loop(); restore_weights(net, saved)
    c2 = compute_all_metrics_with_clone(net, clone, Xte, yte)["gain"]
    restore_weights(net, saved)
    return bl, a, c2


def main():
    rows = []
    t0 = time.time()
    for tau, w1 in CONFIGS:
        for i, s in enumerate(SEEDS):
            target = train_one(s, tau, w1)
            clone = train_one(DONOR[i], tau, w1)
            Xte, yte = generate_data(N, noise_level=NOISE, seed=s + 500)
            bl, a, c2 = eval_static(target, clone, Xte, yte)
            rows.append({"tau": tau, "w1": w1, "seed": s,
                         "bl_gain": bl, "a_gain": a, "c2_gain": c2})
        print(f"  (tau={tau}, w1={w1}) done  {time.time()-t0:.0f}s", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "w1", "seed", "bl_gain", "a_gain", "c2_gain"])
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 72)
    print("C2-INVERSION STATIC SPOT-CHECK (mean over 20 seeds)")
    print("=" * 72)
    print(f"  {'config':>16s} | {'BL':>7s} {'A':>7s} {'C2':>7s} | {'C2<A?':>6s} {'C2<0?':>6s} {'#C2<A':>6s}")
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["tau"], r["w1"])].append(r)
    for (tau, w1), rs in by.items():
        bl = np.mean([r["bl_gain"] for r in rs]); a = np.mean([r["a_gain"] for r in rs])
        c2 = np.mean([r["c2_gain"] for r in rs])
        nlt = sum(1 for r in rs if r["c2_gain"] < r["a_gain"])
        flag = "primary" if (tau, w1) == (2.0, 0.0) else ""
        print(f"  tau={tau},w1={w1} {flag:>4s} | {bl:+.3f} {a:+.3f} {c2:+.3f} | "
              f"{'YES' if c2 < a else 'no':>6s} {'YES' if c2 < 0 else 'no':>6s} {nlt:>4d}/20")
    print(f"\n[spotcheck] wrote {OUT}  ({time.time()-t0:.0f}s total)")
    print("  Inversion = (C2 < A) AND (C2 < 0). Primary should reproduce ~C2=-0.12.")


if __name__ == "__main__":
    main()
