# Minimal reproduction package

This is the pure-NumPy, CPU-only core of the experiments. It is self-contained:
you can reproduce the primary result CSVs bit-for-bit and verify the
load-bearing numbers against that shipped data, with no other files (see **Verify**).

See `MANIFEST.md` for exactly what ships.

Paper: *The Fake Mirror Effect: Foreign Feedback Disrupts Self-Correction in Minimal
Recurrent Networks*, Transactions on Machine Learning Research (2026).
<https://openreview.net/forum?id=cwENvGCLRv>

The primary 35-neuron experiments above are pure NumPy. Two pieces additionally use
PyTorch: the PyTorch cross-implementation check (Appendix A.6) and the MNIST-scale
extension live in separate companion repositories; the closed-loop BPTT alignment
ablation (Appendix G) is included here under `experiments/` but requires PyTorch and
the PyTorch companion repository (see **Appendix G** below, and it is not part of
`run_all.py`). Their numbers are backed here by the frozen data
shipped in `results/` (see **Verify** for how each is checked — closed-loop here,
PyTorch-CV/MNIST in their companion repos), independent of re-running the PyTorch parts.

## Install

```
pip install -r requirements.txt
```

A single install covers the primary NumPy pipeline and tests: NumPy, matplotlib,
pandas, SciPy, scikit-learn, seaborn, pytest. CPU only; no GPU required. (The optional
Appendix-G closed-loop scripts additionally need `torch` — see **Appendix G**.)

## Reproduce

```
python run_all.py
```

Runs the NumPy experiment phases (~10 h, single CPU, 20 seeds/phase,
`OMP_NUM_THREADS=1`) and regenerates the CSV data + figures into `results/`,
overwriting the distributed copies in place.

## Verify: confirm the reproduction (recommended)

```
python verify.py
```

`verify.py` recomputes most shipped result numbers from the `results/` CSV data
(reading a few from a pre-aggregated summary JSON: MNIST/closed-loop/kappa/TOST)
and checks that each one appears in the recorded numeric-claims
manifest (`verification/paper_numbers_manifest.txt`) using whitespace-normalized
**token-boundary** matching (not a raw substring test, so e.g. `64` does not
spuriously match inside `+0.0064`). It is therefore an internal
**data ↔ manifest consistency check**: it confirms the shipped data reproduces the
recorded numbers (and, after `python run_all.py`, that a fresh run reproduces them),
independent of row order and timing. It is **not** an independent verification of the
paper PDF — this package ships no `main.tex`, so the gate runs in its
**NON-AUTHORITATIVE** manifest-fallback mode and prints a notice to that effect. The
authoritative paper-accuracy check is the published TMLR paper itself. (The gate also
covers the closed-loop / PyTorch-CV / MNIST numbers against the frozen data shipped in
`results/`; the PyTorch-CV and MNIST numbers are re-derived only in the companion
repos, while the closed-loop numbers can also be re-derived here via the Appendix-G
scripts.)

## Optional: raw byte-level diff

To diff a fresh run against the distributed data, keep a copy first:

```
cp -r results results_shipped
python run_all.py
python scripts/compare_results.py --new results --old results_shipped
```

Per-seed CSV values are bit-for-bit identical. Two non-substantive differences are
expected and not drift: **row order** (within each phase, parallel workers write
rows in completion order; the phases themselves run sequentially) and the per-row
**`elapsed_s`** wall-clock timing column. On a different platform/BLAS, one file —
`integration_control_jacobian.csv` (SVD / condition numbers) — may also differ by
≤1e-7 from floating-point accumulation order; this does not affect any reported number.
`verify.py` above is order- and timing-independent and is the preferred check.

## Appendix G: closed-loop BPTT alignment (PyTorch)

The closed-loop alignment ablation is implemented in PyTorch and is **not** part of
`run_all.py`. To re-run it you additionally need `torch` and the PyTorch companion
repository cloned as a sibling directory:

```
pip install torch
git clone https://github.com/softkorea/pytorch-35neuron-validation ../pytorch-35neuron-validation
python experiments/run_closed_loop_alignment.py --phase main   # plus the other --phase variants
python experiments/analyze_closed_loop_alignment.py            # -> results/closed_loop_alignment_summary.json
```

The shipped `results/closed_loop_alignment_*` files back the Appendix-G numbers and
are verifiable via `verify.py` without running this step.

## Tests

The full suite is a thorough check, not a quick smoke test: it includes
gradient-verification and short training-loop tests and takes several minutes.

```
pytest tests/                                    # full suite (several minutes)
pytest tests/test_metrics.py tests/test_network.py -q   # quick sanity check (seconds)
```

## Determinism

All randomness uses `numpy.random.RandomState(seed)` with fixed per-experiment
seeds (train `s`, held-out test `s+500`, donor `s+100`), `OMP_NUM_THREADS=1`, and
no GPU. Per-seed values are therefore reproducible bit-for-bit; only row order and
the `elapsed_s` timing column vary between runs.

## Citation

```bibtex
@article{ong2026fakemirror,
  title   = {The Fake Mirror Effect: Foreign Feedback Disrupts Self-Correction
             in Minimal Recurrent Networks},
  author  = {Sungmoon Ong},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=cwENvGCLRv}
}
```

Companion repositories: [PyTorch cross-validation](https://github.com/softkorea/pytorch-35neuron-validation) ·
[MNIST-scale extension](https://github.com/softkorea/mnist-feedback-contract)

## License

MIT — see [LICENSE](LICENSE).
