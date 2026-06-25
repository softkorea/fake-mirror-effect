"""Master pipeline for the pure-NumPy core: run all CPU-deterministic phases.

Usage:
    python run_all.py        # (re)generate all NumPy results into results/

Single CPU, 20 seeds per phase: ~10 h. Every phase is deterministic
(numpy.random.RandomState with fixed seeds, OMP_NUM_THREADS=1), so re-execution
reproduces the shipped data: the per-seed CSV values are bit-for-bit identical (row
order can differ because of within-phase worker parallelism -- the phases themselves
run sequentially via blocking subprocess calls -- and the per-row `elapsed_s` timing
column is wall-clock). To confirm reproduction, run `verify.py` after this — it
re-derives the paper numbers from the freshly written results/ and is independent
of row order and timing. To preserve the distributed data for a raw diff, copy
results/ aside first (see README).

The PyTorch cross-implementation check, the closed-loop alignment appendix, and the
MNIST extension are not part of this core; they live in separate companion
repositories. The numbers they produce are still verifiable here against the frozen
data in results/ (see verify.py).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PHASES = [
    # (name, script relative to ROOT)
    ("Static ablation (N=20)", "experiments/run_experiment.py"),
    ("C2 + VN + alignment (N=20)", "experiments/run_n20_full.py"),
    ("Variable noise (N=20)", "experiments/run_variable_noise.py"),
    ("C2 static (N=20)", "experiments/run_c2_experiment.py"),
    ("Interpolation (N=20)", "experiments/run_interpolation.py"),
    ("Wrong-trajectory (N=20)", "experiments/run_wrong_trajectory.py"),
    ("Cross-pairing (20x20)", "experiments/run_cross_pairing.py"),
    ("VN sweep", "experiments/run_vn_sweep.py"),
    ("VN sweep extended", "experiments/run_vn_sweep_extended.py"),
    ("Mechanistic", "experiments/run_mechanistic.py"),
    ("Divergence null", "experiments/run_divergence_null.py"),
    ("Training dynamics", "experiments/run_training_dynamics.py"),
    ("Timestep extension", "experiments/run_timestep_extension.py"),
    ("Scale verification (N=20)", "experiments/run_scale_verification.py"),
    ("Hyperparameter sweep", "experiments/sweep_hyperparams.py"),
    ("Stronger alignment", "experiments/run_stronger_alignment.py"),
    ("Integration control (N=20)", "experiments/run_integration_control.py"),
    ("Integration control extras", "experiments/run_integration_control_extras.py"),
    ("Jacobian / kappa analysis (N=20)", "experiments/run_jacobian_analysis.py"),
    ("Decision-space figure", "experiments/redesign_figure_trajectory.py"),
]


def run_phase(name, script, env):
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
    t0 = time.time()
    # Run with cwd=ROOT so every phase writes to ROOT/results, regardless of whether
    # the phase resolves its output path from cwd or from its own __file__ location.
    cmd = [sys.executable, str(ROOT / script)]
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"  [{status}] {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    return elapsed, result.returncode


def main():
    (ROOT / "results" / "figure_redesign").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    print("=" * 60)
    print("PURE-NUMPY PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results -> {ROOT / 'results'}")
    print("=" * 60)

    total_start = time.time()
    results = []
    for name, script in PHASES:
        elapsed, code = run_phase(name, script, env)
        results.append((name, elapsed, code))
        if code != 0:
            print(f"\n*** PHASE FAILED: {name} ***\nContinuing with remaining phases...")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}\nRUN COMPLETE  ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{'=' * 60}")
    print(f"\n{'Phase':<40s} {'Time':>8s} {'Status':>8s}")
    print(f"{'-' * 40} {'-' * 8} {'-' * 8}")
    failed = 0
    for name, elapsed, code in results:
        status = "OK" if code == 0 else "FAILED"
        failed += code != 0
        print(f"{name:<40s} {elapsed:>6.0f}s {status:>8s}")
    print(f"{'-' * 40} {'-' * 8} {'-' * 8}")
    print(f"{'Total':<40s} {total_time:>6.0f}s")
    print(f"\n{len(results)} phases, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
