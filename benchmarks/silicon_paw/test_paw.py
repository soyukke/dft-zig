#!/usr/bin/env python3
"""Silicon PAW regression test: DFT-Zig vs QE baseline.

Compares band energies (VBM-aligned) against QE baseline.
PAW uses Si.pbe-n-kjpaw_psl.1.0.0.UPF, ecut=15 Ry, 4x4x4 k-mesh.

Usage:
    python3 test_paw.py --run          # run DFT-Zig and check
    python3 test_paw.py --check-only   # check existing output
"""

import argparse
import csv
import os
import subprocess
import sys

Ry_to_meV = 13605.7  # 1 Ry = 13605.7 meV

# Number of valence bands (for VBM alignment)
N_VALENCE = 4

# MSE thresholds per band (meV^2) — DFT-Zig vs QE
# PAW at ecut=15 Ry: band gap matches QE within ~5 meV,
# but absolute offset ~0.16 Ry exists (G=0 convention difference).
# VBM-aligned comparison removes the offset.
THRESHOLDS = {
    0: 100,     # deep valence
    1: 100,
    2: 100,
    3: 100,     # VBM
    4: 500,     # CBM (more sensitive to ecut)
    5: 500,
    6: 1000,
    7: 2000,    # high conduction (less stable)
}

BAND_GAP_THRESHOLD_meV = 50.0


def run_dft_zig():
    """Run DFT-Zig calculation."""
    print("=" * 50)
    print("Running DFT-Zig...")
    print("=" * 50)
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", "dft_zig.toml"],
        capture_output=True, text=True
    )
    print(result.stderr)
    if result.returncode != 0:
        print("DFT-Zig failed!")
        sys.exit(1)
    print("DFT-Zig done.\n")


def load_band_energies(csv_path):
    """Load band energies from CSV, return (dists, bands_meV) VBM-aligned."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    nbands = len([k for k in rows[0] if k.startswith('band')])
    bands = []
    for i in range(nbands):
        bands.append([float(r[f'band{i}']) * Ry_to_meV for r in rows])

    # VBM alignment
    vbm = max(max(bands[i]) for i in range(N_VALENCE))
    for band in bands:
        for j in range(len(band)):
            band[j] -= vbm

    return bands


def check_regression(dft_path, baseline_path):
    """Compare DFT-Zig bands against QE baseline."""
    dft_bands = load_band_energies(dft_path)
    qe_bands = load_band_energies(baseline_path)

    nbands = min(len(dft_bands), len(qe_bands))
    nkpts = min(len(dft_bands[0]), len(qe_bands[0]))

    if len(dft_bands[0]) != len(qe_bands[0]):
        print(f"Warning: k-point count mismatch: DFT-Zig={len(dft_bands[0])}, QE={len(qe_bands[0])}")

    print("=" * 50)
    print("Regression Check Results (Silicon PAW vs QE)")
    print("=" * 50)
    print("MSE per band (meV^2):")

    all_pass = True
    for i in range(nbands):
        mse = sum((dft_bands[i][k] - qe_bands[i][k])**2 for k in range(nkpts)) / nkpts
        threshold = THRESHOLDS.get(i, 2000)
        passed = mse <= threshold
        status = "[OK]" if passed else "[FAIL]"
        print(f"  Band {i}: {mse:.1f} (threshold: {threshold}) {status}")
        if not passed:
            all_pass = False

    # Band gap check
    dft_cbm = min(min(dft_bands[i]) for i in range(N_VALENCE, nbands))
    qe_cbm = min(min(qe_bands[i]) for i in range(N_VALENCE, nbands))
    dft_gap = dft_cbm  # VBM is at 0
    qe_gap = qe_cbm
    gap_diff = abs(dft_gap - qe_gap)
    gap_pass = gap_diff <= BAND_GAP_THRESHOLD_meV
    status = "[OK]" if gap_pass else "[FAIL]"
    print(f"Band gap: DFT-Zig={dft_gap:.1f} meV, QE={qe_gap:.1f} meV, diff={gap_diff:.1f} meV (threshold: {BAND_GAP_THRESHOLD_meV}) {status}")
    if not gap_pass:
        all_pass = False

    if all_pass:
        print("PASS: All checks passed!")
    else:
        print("FAIL: Some checks failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Silicon PAW regression test")
    parser.add_argument("--run", action="store_true", help="Run DFT-Zig and check")
    parser.add_argument("--check-only", action="store_true", help="Check existing output")
    args = parser.parse_args()

    if args.run:
        run_dft_zig()

    dft_path = "out_dft_zig/band_energies.csv"
    baseline_path = "baseline/qe_bands.csv"

    if not os.path.exists(dft_path):
        print(f"Error: {dft_path} not found. Run with --run first.")
        sys.exit(1)
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found. Run generate_qe_baseline.py first.")
        sys.exit(1)

    check_regression(dft_path, baseline_path)


if __name__ == "__main__":
    main()
