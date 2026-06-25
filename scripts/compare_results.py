"""Compare current results/ against results_shipped/ (or any two directories).

Compares per-cell VALUES within 1e-12 after dropping the wall-clock `elapsed_s`
column and sorting rows, so the documented row-order / timing variation (from
within-phase worker parallelism) does not register as a false difference. This is
a numeric-tolerance value check, NOT a byte-for-byte comparison.

Usage:
    python scripts/compare_results.py                          # results/ vs results_shipped/
    python scripts/compare_results.py --new results/ --old results_shipped/
    python scripts/compare_results.py --summary-only           # skip per-cell details
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


def compare_csv(path_old, path_new, name, summary_only=False):
    """Compare two CSV files cell by cell. Return dict of findings."""
    findings = {
        'name': name,
        'status': 'unknown',
        'n_cells': 0,
        'n_diff': 0,
        'max_abs_diff': 0.0,
        'details': [],
    }

    if not os.path.exists(path_old):
        findings['status'] = 'NEW (not in old)'
        return findings
    if not os.path.exists(path_new):
        findings['status'] = 'MISSING (not in new)'
        return findings

    try:
        df_old = pd.read_csv(path_old)
        df_new = pd.read_csv(path_new)
    except Exception as e:
        findings['status'] = f'READ ERROR: {e}'
        return findings

    # Reproduction contract (README / run_all): per-seed VALUES are bit-identical, but row
    # ORDER can vary (within-phase worker parallelism) and the `elapsed_s` column is wall-clock.
    # Drop timing columns and sort by the remaining columns so the comparison tests VALUES, not
    # row position or wall-clock -- otherwise reordered/timed-but-identical CSVs read as a false
    # 'DIFFERENT' (the position-based iloc compare below would mis-align shuffled rows).
    _timing = [c for c in df_old.columns if c in ('elapsed_s', 'elapsed', 'wall_s', 'timestamp')]
    if _timing:
        df_old = df_old.drop(columns=_timing)
        df_new = df_new.drop(columns=[c for c in _timing if c in df_new.columns])
    if list(df_old.columns) == list(df_new.columns) and len(df_old.columns):
        _k = list(df_old.columns)
        df_old = df_old.sort_values(by=_k, kind='mergesort').reset_index(drop=True)
        df_new = df_new.sort_values(by=_k, kind='mergesort').reset_index(drop=True)

    # Shape check
    if df_old.shape != df_new.shape:
        findings['status'] = f'SHAPE DIFFERS: old={df_old.shape} new={df_new.shape}'
        return findings

    # Column check
    if list(df_old.columns) != list(df_new.columns):
        findings['status'] = f'COLUMNS DIFFER'
        return findings

    # Cell-by-cell comparison
    n_cells = 0
    n_diff = 0
    max_abs = 0.0
    details = []

    for col in df_old.columns:
        old_vals = df_old[col]
        new_vals = df_new[col]

        for i in range(len(old_vals)):
            ov, nv = old_vals.iloc[i], new_vals.iloc[i]
            n_cells += 1

            # Handle non-numeric
            if isinstance(ov, str) or isinstance(nv, str):
                if str(ov) != str(nv):
                    n_diff += 1
                    if not summary_only:
                        details.append(f'  row {i}, {col}: "{ov}" -> "{nv}"')
                continue

            # Handle NaN
            if pd.isna(ov) and pd.isna(nv):
                continue
            if pd.isna(ov) or pd.isna(nv):
                n_diff += 1
                if not summary_only:
                    details.append(f'  row {i}, {col}: {ov} -> {nv} (NaN mismatch)')
                continue

            # Numeric comparison
            diff = abs(float(nv) - float(ov))
            if diff > 1e-12:  # beyond float noise
                n_diff += 1
                max_abs = max(max_abs, diff)
                if not summary_only and diff > 1e-6:  # only report meaningful diffs
                    details.append(
                        f'  row {i}, {col}: {ov} -> {nv} (delta={nv-ov:+.6g})'
                    )

    findings['n_cells'] = n_cells
    findings['n_diff'] = n_diff
    findings['max_abs_diff'] = max_abs
    findings['details'] = details

    if n_diff == 0:
        findings['status'] = 'IDENTICAL'
    else:
        findings['status'] = f'{n_diff} cells differ (max |delta|={max_abs:.6g})'

    return findings


def main():
    parser = argparse.ArgumentParser(description='Compare CSV results directories')
    parser.add_argument('--new', default='results', help='New results directory')
    parser.add_argument('--old', default='results_shipped', help='Old (frozen) results directory')
    parser.add_argument('--summary-only', action='store_true', help='Skip per-cell details')
    args = parser.parse_args()

    print(f'Comparing: {args.old}/ (frozen) vs {args.new}/ (current)')
    print('=' * 70)

    # Collect all CSV files from both dirs
    old_csvs = set(f for f in os.listdir(args.old) if f.endswith('.csv'))
    new_csvs = set(f for f in os.listdir(args.new) if f.endswith('.csv'))
    all_csvs = sorted(old_csvs | new_csvs)

    results = []
    for csv_name in all_csvs:
        path_old = os.path.join(args.old, csv_name)
        path_new = os.path.join(args.new, csv_name)
        findings = compare_csv(path_old, path_new, csv_name, args.summary_only)
        results.append(findings)

    # Print results
    identical = 0
    different = 0
    missing = 0

    for f in results:
        if f['status'] == 'IDENTICAL':
            identical += 1
            marker = 'OK'
        elif 'MISSING' in f['status'] or 'NEW' in f['status']:
            missing += 1
            marker = '!!'
        else:
            different += 1
            marker = '**'

        print(f'[{marker}] {f["name"]:<45s} {f["status"]}')
        if f['details']:
            for d in f['details'][:20]:  # cap at 20 per file
                print(d)
            if len(f['details']) > 20:
                print(f'  ... and {len(f["details"]) - 20} more')
        print()

    # Summary
    print('=' * 70)
    print(f'SUMMARY: {len(results)} CSV files compared')
    print(f'  IDENTICAL: {identical}')
    print(f'  DIFFERENT: {different}')
    print(f'  MISSING/NEW: {missing}')

    if different == 0 and missing == 0:
        print('\n  >>> ALL CSVs MATCH (per-cell values within 1e-12; row order + elapsed_s ignored, per the reproduction contract). Pipeline is reproducible. <<<')
    elif different > 0:
        print(f'\n  >>> {different} CSV(s) DIFFER. Investigate before proceeding. <<<')

    return 1 if different > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
