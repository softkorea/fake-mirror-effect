"""Float-arithmetic rounding artifact audit.

For each CSV with seed-level gain/accuracy data, compute group means
two ways:
  1. np.mean() with f-string formatting (current paper display path)
  2. Decimal arithmetic from string representations (exact + ROUND_HALF_UP)

Flag any cells where the two formatted displays differ at the paper's
typical precision (.3f for gains, .4f for accuracies). These are the
locations where the displayed paper value may be a float-rounding
artifact rather than the true mathematical value.

Example artifact (already caught manually): Figure 6 cell (w1=0.2, tau=1.5)
displayed +0.135 via np.mean+f-string, but true mean is 1.355/10 = 0.1355
which rounds to +0.136 under standard rounding.
"""

from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
from pathlib import Path


def exact_mean(values, grid=Decimal('0.005')):
    """Compute exact mean.

    Each CSV value is first snapped to the nearest grid point (default 0.005,
    which is the resolution of a 200-trial test set). This recovers the true
    precision from float64-lossy CSV representations like
    "0.1999999999999999" -> 0.2.

    Then sum and divide exactly via Decimal arithmetic. The result is the
    true mathematical mean of the underlying 200-trial accuracy/gain data,
    free of both IEEE-754 accumulation error and CSV-write precision loss.

    For accuracy-derived data on a 200-trial test set, gain values are
    constrained to multiples of 0.005 (= 1/200). This holds for all the
    35-neuron experiments in this repo.
    """
    decimals = []
    for v in values:
        # Snap to nearest grid point
        n_units = (Decimal(str(v)) / grid).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        decimals.append(n_units * grid)
    return sum(decimals, Decimal(0)) / Decimal(len(decimals))


def exact_mean_raw(values):
    """Compute exact mean WITHOUT precision recovery (for non-200-trial data)."""
    decimals = [Decimal(str(v)) for v in values]
    return sum(decimals, Decimal(0)) / Decimal(len(decimals))


def float_mean(values):
    return float(np.mean(values))


def _decimal_quantum(precision):
    """Convert precision spec like '.3f' or '.4f' to Decimal quantum '0.001'/'0.0001'."""
    n_dec = int(precision.lstrip('.').rstrip('f'))
    return Decimal('1').scaleb(-n_dec), n_dec


def fmt_signed_d(d, precision):
    """Decimal d -> +X.XXX or -X.XXX format at given precision."""
    quantum, n_dec = _decimal_quantum(precision)
    rounded = d.quantize(quantum, rounding=ROUND_HALF_UP)
    sign = '+' if rounded >= 0 else '-'
    return f"{sign}{abs(rounded):.{n_dec}f}"


def fmt_signed_f(f, precision):
    """Float f -> +X.XXX or -X.XXX via Python f-string (current paper path)."""
    _, n_dec = _decimal_quantum(precision)
    return f"{f:+.{n_dec}f}"


def values_are_grid_aligned(values, grid=0.005, tol=1e-9):
    """Return True if every value is exact integer multiple of grid (within tol).

    If False, per-seed values are pre-averaged (e.g., B1 / C1 conditions
    average over multiple shuffles) and grid recovery would snap incorrectly.
    Use raw Decimal-of-string instead.
    """
    for v in values:
        snap = round(v / grid) * grid
        if abs(snap - v) > tol:
            return False
    return True


def audit_group(values, label, precision='.3f'):
    """Compare float-display vs exact-display for a single group of values.

    Uses grid-recovery (snap to 0.005 multiples) only when per-seed values
    are themselves single 200-trial measurements. For pre-averaged values
    (B1 / C1 / other multi-shuffle conditions), uses Decimal-of-CSV-string
    directly - the float artifact in that case is below the precision of
    the CSV itself.
    """
    if not values:
        return None
    f_mean = float_mean(values)
    use_grid = values_are_grid_aligned(values)
    e_mean = exact_mean(values) if use_grid else exact_mean_raw(values)
    f_disp = fmt_signed_f(f_mean, precision)
    e_disp = fmt_signed_d(e_mean, precision)
    if f_disp != e_disp:
        return {
            'group': label,
            'n': len(values),
            'grid_aligned': use_grid,
            'float_mean': f_mean,
            'exact_mean': float(e_mean),
            'float_display': f_disp,
            'exact_display': e_disp,
        }
    return None


def audit_csv(csv_path, group_cols, value_col, precision='.3f'):
    df = pd.read_csv(csv_path)
    artifacts = []
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = ', '.join(f'{c}={k}' for c, k in zip(group_cols, keys))
        vals = group[value_col].tolist()
        result = audit_group(vals, label, precision)
        if result:
            artifacts.append(result)
    return artifacts


AUDIT_PLAN = [
    # (csv_path, group_cols, value_col, precision, description)
    ('results/raw_metrics.csv', ['group', 'noise_level'], 'gain', '.3f', 'Table 1 static gains'),
    ('results/n20_c2_vn_alignment.csv', ['setting', 'group'], 'gain', '.3f', 'Table 1 VN + alignment'),
    ('results/stronger_alignment.csv', ['setting', 'group'], 'gain', '.3f', 'Table 2 alignment family'),
    ('results/variable_noise_metrics.csv', ['group', 'noise_level'], 'gain', '.3f', 'sigma sweep'),
    ('results/interpolation.csv', ['alpha', 'interp_type'], 'gain', '.3f', 'Figure 3 interpolation'),
    ('results/scale_verification_static.csv', ['hidden_width', 'condition'], 'gain', '.3f', 'Figure 7 static'),
    ('results/scale_verification_vn.csv', ['hidden_width', 'condition'], 'gain', '.3f', 'Figure 7 VN'),
    ('results/wrong_trajectory_static.csv', ['condition'], 'gain', '.3f', 'Wrong-traj static'),
    ('results/wrong_trajectory_vn.csv', ['condition'], 'gain', '.3f', 'Wrong-traj VN'),
    ('results/sweep_vn_hyperparams.csv', ['w1', 'tau'], 'gain', '.3f', 'Figure 6 sweep (already fixed)'),
    ('results/sweep_vn_extended.csv', ['w1', 'w2', 'tau'], 'gain', '.3f', 'Figure 6 extended sweep'),
    ('results/cross_pairing_vn.csv', ['target_seed'], 'gain', '.3f', 'Cross-pairing per-target'),
    ('results/integration_control.csv', None, None, '.4f', 'Integration control --handled separately'),
]


def main():
    print("=" * 70)
    print("Float-arithmetic rounding artifact audit")
    print("=" * 70)
    print()
    total_artifacts = 0
    for entry in AUDIT_PLAN:
        csv_path, group_cols, value_col, precision, description = entry
        if not Path(csv_path).exists():
            print(f"SKIP: {csv_path} not found")
            continue
        if group_cols is None:
            # Special handling for integration_control (multiple value columns)
            print(f"\n--- {csv_path} --{description} ---")
            df = pd.read_csv(csv_path)
            value_cols = ['bl_gain', 'e1_gain_probmean', 'e1_gain_logprod',
                          'c2_normmatched_gain', 'c2_std_gain']
            for vc in value_cols:
                if vc not in df.columns:
                    continue
                result = audit_group(df[vc].tolist(), f'all seeds, {vc}', precision)
                if result:
                    print(f"  ARTIFACT: {result}")
                    total_artifacts += 1
                else:
                    f_disp = fmt_signed_f(float_mean(df[vc].tolist()), precision)
                    print(f"  OK ({vc}): {f_disp} (N={len(df)})")
            continue

        print(f"\n--- {csv_path} --{description} ---")
        artifacts = audit_csv(csv_path, group_cols, value_col, precision)
        if not artifacts:
            print(f"  OK: no artifacts at {precision} precision")
        else:
            for art in artifacts:
                print(f"  ARTIFACT: {art['group']}")
                print(f"    N={art['n']}, float={art['float_mean']:.10f}, "
                      f"exact={art['exact_mean']:.10f}")
                print(f"    Display: float='{art['float_display']}' vs exact='{art['exact_display']}'")
                total_artifacts += 1

    print()
    print("=" * 70)
    print(f"Total artifacts found: {total_artifacts}")
    print("=" * 70)


if __name__ == '__main__':
    main()
