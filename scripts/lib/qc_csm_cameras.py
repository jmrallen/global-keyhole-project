#!/usr/bin/env python3
"""qc_csm_cameras.py — structural QC of jitter_solve adjusted_state.json files.

Catches the Cauchy-attenuated-outlier failure mode where Phase 7 produces cameras
with great median residuals but per-line position/quaternion spikes that make
them unusable for mapproject (200+ km position-radius spread, mean reprojection
errors of 1e+29 to 1e+132 px).

Exits 0 if every camera passes; exits 1 otherwise so the calling shell script's
`set -e` halts the pipeline before Phase 8 wastes hours on broken cameras.
"""
import argparse
import glob
import json
import math
import os
import sys


def load_csm_state(path):
    """Load a CSM adjusted_state.json, skipping the magic header line."""
    with open(path) as f:
        content = f.read()
    pos = content.find('{')
    if pos < 0:
        raise ValueError(f"No JSON object found in {path}")
    return json.loads(content[pos:])


def check_camera(path, max_pos_spread_km, max_pos_delta_km):
    state = load_csm_state(path)
    issues = []

    pos = state.get('m_positions', [])
    if len(pos) < 3 or len(pos) % 3 != 0:
        issues.append(f"m_positions length {len(pos)} not a positive multiple of 3")
        return issues, None, None, None

    n = len(pos) // 3
    radii = [math.sqrt(pos[3*i]**2 + pos[3*i+1]**2 + pos[3*i+2]**2) for i in range(n)]
    spread_km = (max(radii) - min(radii)) / 1000.0
    if spread_km > max_pos_spread_km:
        issues.append(f"ECEF |r| spread {spread_km:.1f} km > {max_pos_spread_km} km")

    deltas = []
    for i in range(1, n):
        dx = pos[3*i]   - pos[3*(i-1)]
        dy = pos[3*i+1] - pos[3*(i-1)+1]
        dz = pos[3*i+2] - pos[3*(i-1)+2]
        deltas.append(math.sqrt(dx*dx + dy*dy + dz*dz))
    max_delta_km = (max(deltas) if deltas else 0.0) / 1000.0
    if max_delta_km > max_pos_delta_km:
        issues.append(f"max successive position delta {max_delta_km:.1f} km > {max_pos_delta_km} km")

    quat = state.get('m_quaternions', [])
    if len(quat) < 4 or len(quat) % 4 != 0:
        issues.append(f"m_quaternions length {len(quat)} not a positive multiple of 4")
        return issues, spread_km, max_delta_km, None

    nq = len(quat) // 4
    qnorms = [math.sqrt(quat[4*i]**2 + quat[4*i+1]**2 + quat[4*i+2]**2 + quat[4*i+3]**2)
              for i in range(nq)]
    qnorm_min, qnorm_max = min(qnorms), max(qnorms)
    # ASP renormalizes quaternions internally, so mild non-unit stored values
    # are harmless. Only flag the cases that signal actual numerical breakdown.
    if qnorm_min < 0.5 or qnorm_max > 2.0:
        issues.append(f"quaternion norm range [{qnorm_min:.4f}, {qnorm_max:.4f}] outside [0.5, 2.0]")

    # Successive dot product on normalized quaternions; < 0.5 means a sign flip
    # or near-180-degree jump between adjacent samples. That truly breaks
    # mapproject because slerp interpolation flips orientation mid-strip.
    bad_dot = None
    for i in range(1, nq):
        a = quat[4*(i-1):4*i]
        b = quat[4*i:4*(i+1)]
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        if na == 0 or nb == 0:
            continue
        dot = abs(sum(a[j]*b[j] for j in range(4))) / (na * nb)
        if bad_dot is None or dot < bad_dot:
            bad_dot = dot
    if bad_dot is not None and bad_dot < 0.5:
        issues.append(f"min successive quaternion |dot| {bad_dot:.3f} < 0.5 (sign flip / 90+ deg jump)")

    return issues, spread_km, max_delta_km, (qnorm_min, qnorm_max)


def check_residuals(stats_path, max_mean, max_median):
    """Parse run-final_residuals_stats.txt; return list of (image, issue) tuples."""
    if not os.path.isfile(stats_path):
        return [("(missing)", f"residuals stats file not found: {stats_path}")]

    issues = []
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            img = os.path.basename(parts[0])
            try:
                mean = float(parts[1])
                median = float(parts[2])
            except ValueError:
                continue
            row_issues = []
            if not math.isfinite(mean) or mean > max_mean:
                row_issues.append(f"mean residual {mean:g} > {max_mean}")
            if not math.isfinite(median) or median > max_median:
                row_issues.append(f"median residual {median:g} > {max_median}")
            if row_issues:
                issues.append((img, "; ".join(row_issues)))
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jitter-dir", required=True,
                    help="directory containing run-*.adjusted_state.json")
    ap.add_argument("--stats-file",
                    help="path to run-final_residuals_stats.txt (optional)")
    ap.add_argument("--max-pos-spread-km", type=float, default=10.0)
    ap.add_argument("--max-pos-delta-km", type=float, default=2.0)
    ap.add_argument("--max-mean-residual", type=float, default=100.0)
    ap.add_argument("--max-median-residual", type=float, default=5.0)
    args = ap.parse_args()

    pattern = os.path.join(args.jitter_dir, "run-*.adjusted_state.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: no cameras matched {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Inspecting {len(files)} adjusted_state.json files in {args.jitter_dir}")
    print(f"  thresholds: pos-spread<={args.max_pos_spread_km}km  "
          f"pos-delta<={args.max_pos_delta_km}km  "
          f"mean-resid<={args.max_mean_residual}px  "
          f"median-resid<={args.max_median_residual}px")
    print()

    fail = 0
    print(f"  {'camera':<55}  {'spread_km':>9}  {'maxdelta_km':>11}  {'q_range':>15}  status")
    for f in files:
        name = os.path.basename(f).replace("run-", "").replace(".adjusted_state.json", "")
        issues, spread, max_delta, qr = check_camera(
            f, args.max_pos_spread_km, args.max_pos_delta_km)
        if qr is None:
            qrtxt = "n/a"
        else:
            qrtxt = f"[{qr[0]:.3f},{qr[1]:.3f}]"
        spread_txt   = f"{spread:9.2f}" if spread is not None else "      n/a"
        delta_txt    = f"{max_delta:11.2f}" if max_delta is not None else "        n/a"
        if issues:
            fail += 1
            print(f"  {name:<55}  {spread_txt}  {delta_txt}  {qrtxt:>15}  FAIL")
            for msg in issues:
                print(f"    - {msg}")
        else:
            print(f"  {name:<55}  {spread_txt}  {delta_txt}  {qrtxt:>15}  ok")

    if args.stats_file:
        print()
        print(f"Residuals stats from {args.stats_file}:")
        resid_issues = check_residuals(
            args.stats_file, args.max_mean_residual, args.max_median_residual)
        if resid_issues:
            # Residual-stats anomalies are DIAGNOSTIC, not predictive of mapproject
            # behavior. A camera that's structurally clean but has high residual
            # mean means its GCPs disagree among themselves (e.g., cloud-features
            # matched against ground reference) — the camera will still mapproject,
            # the bbox-gate in Phase 8 catches inflated footprints separately.
            # Surface as WARNINGs so they show in the log without halting.
            for img, msg in resid_issues:
                print(f"  WARN {img}: {msg}")
        else:
            print("  all cameras within thresholds")

    print()
    if fail:
        print(f"QC FAILED: {fail} structural issue(s) detected. Halt before Phase 8.")
        sys.exit(1)
    print("QC passed (residual warnings, if any, are diagnostic only).")


if __name__ == "__main__":
    main()
