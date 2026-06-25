"""Standalone paper-number verification (internal data <-> manifest consistency).

Recomputes every registered number directly from the shipped CSV/JSON data in
results/ and checks that each appears in the recorded numeric-claims manifest
(verification/paper_numbers_manifest.txt) via whitespace-normalized token-boundary
matching. This package ships no main.tex, so the gate runs in its NON-AUTHORITATIVE
manifest-fallback mode (it prints a notice). It confirms the shipped data reproduces
the recorded numbers; it is NOT an independent verification of the paper PDF -- the
authoritative paper-accuracy check is the published TMLR paper itself. No network, no
GPU, no PyTorch, and no dependency on any other repository.

Usage:
    python verify.py              # exit 0 on GATE PASS, 1 if any number is missing
    python verify.py --coverage   # also list any unaccounted numeric tokens
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "verification"))

import verify_paper_numbers as gate  # noqa: E402  (path set above)

if __name__ == "__main__":
    # Forward args (e.g. --coverage) to the gate. We deliberately do NOT inject
    # --paper: with no main.tex shipped, the gate auto-falls-back to the manifest
    # and reports its NON-AUTHORITATIVE status.
    sys.exit(gate.main() or 0)
