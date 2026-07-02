# Manifest — what ships in the public release

## Scope

The **pure-NumPy core** of the experiments: everything that is bit-deterministic on
a CPU. Running it reproduces the paper's primary results, figures, and CSVs
identically, and the bundled gate verifies every load-bearing number against the
shipped data.

Out of scope (separate companion repositories): the PyTorch cross-check
(reported in §3.6, tabulated in Appendix A.6) and the MNIST-scale extension. These are GPU/PyTorch and only
qualitatively reproducible, so they ship as their own snapshots; their numbers are
still **verifiable** here against frozen data (below). The closed-loop BPTT alignment
(Appendix G) is PyTorch but its scripts ARE included here under `experiments/` — it
requires `torch` plus the PyTorch companion repository (see the README).

## Contents

- `src/` (7) — `network.py`, `training.py`, `ablation.py`, `metrics.py`,
  `visualize.py`, `__init__.py` (pure NumPy), plus `closed_loop_aligner.py`
  (PyTorch, used only by the Appendix-G script).
- `experiments/` (25) — 20 NumPy phases driven by `run_all.py`, plus 4 PyTorch
  closed-loop scripts (Appendix G; run separately, not via `run_all.py`), plus the
  standalone NumPy spot-check `run_c2_inversion_spotcheck.py` (regenerates
  `results/c2_inversion_hyperparam_spotcheck.csv`, the paper's Limitations item 6).
- `results/` — frozen result data:
  - **28 NumPy CSVs** regenerated and bit-diffed by a run (Table 1 / Figures /
    sweeps / scale / integration-control).
  - **10 regen-frozen** files (NumPy, produced by side regen scripts not in the main
    loop, incl. the figure trace cache `figure_redesign/traces_cache_h2.pkl`).
  - **23 torch-frozen** files — see callout below.
- `verification/` — `verify_paper_numbers.py` (the gate), `audit_float_artifacts.py`
  (its dependency), `paper_numbers_manifest.txt` (verbatim numeric-claim extract).
- `tests/` (16) — unit/integration suite for the NumPy modules.
- `scripts/compare_results.py` — bit-identity diff tool.
- `run_all.py`, `verify.py`, `requirements.txt`.

## Frozen-data files (verified here; re-derived in companion repos or via the Appendix-G scripts)

The gate reads these to verify the PyTorch cross-check (§3.6 / Appendix A.6), the §3.6 MNIST residual,
and the Appendix-G closed-loop numbers. They are shipped as frozen data because they
are GPU/PyTorch outputs (CUDA is not bit-deterministic). The PyTorch cross-check and
MNIST residuals are re-derived in their companion repositories; the closed-loop data
can be re-derived here via the Appendix-G scripts (`torch` + the PyTorch companion
required), but not by `run_all.py`.

- `closed_loop_alignment*.{csv,json}` (14) — Appendix G closed-loop alignment.
- `closed_loop_parity_tests.csv` (1) — Appendix G parity prerequisite.
- `integration_control_pytorch_cv*.{csv,json}` (6) — PyTorch cross-check (§3.6 / Appendix A.6).
  These ship frozen (the generating torch script lives with the PyTorch work, not in this NumPy
  core). The low-noise residual is the fp64 value (`*_sweep_fp64.csv`, +0.051 at σ=0.5); the
  paired `*_sweep.csv` is single-precision, where the same σ=0.5 residual attenuates 7.8× to +0.0065 —
  i.e. this small descriptive residual is precision-sensitive (noted in §3.6).
- `integration_control_mnist_cv*.{csv,json}` (2) — §3.6 MNIST residual.

## Not shipped (and why)

- The PyTorch cross-validation script (§3.6 / Appendix A.6) and the MNIST experiment scripts — torch +
  sibling-repo / dataset dependencies; they belong to the companion repos. (The
  Appendix-G closed-loop scripts and `src/closed_loop_aligner.py` ARE shipped here under
  `experiments/` + `src/` — see Contents; they need `torch` + the PyTorch companion.)
- `RESULTS_SOURCES.md` and other internal provenance/report docs — superseded here by
  this manifest + the verification gate.
- Model checkpoints, datasets, logs, LaTeX sources — not needed to reproduce or
  verify the NumPy core.
