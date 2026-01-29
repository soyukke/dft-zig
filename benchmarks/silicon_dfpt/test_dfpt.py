#!/usr/bin/env python3
"""Si DFPT phonon regression test: DFT-Zig vs ABINIT (2x2x2 k-mesh).

Runs DFT-Zig DFPT with 2x2x2 k-mesh and compares phonon frequencies
at Gamma, X, L against pre-computed ABINIT reference values.

Checks:
1. phonon_band.csv is generated
2. Phonon frequencies at Gamma, X, L match ABINIT within threshold
3. Acoustic modes at Gamma are near zero

Usage:
    python3 test_dfpt.py --run     # Run DFT-Zig and check
    python3 test_dfpt.py           # Check existing output only
"""

import subprocess
import os
import sys
import csv
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DFT_ZIG = os.path.join(PROJECT_ROOT, "zig-out", "bin", "dft_zig")

TOML_FILE = "dft_zig_dfpt_regression.toml"
OUT_DIR = "out_dfpt_regression"
PHONON_CSV = os.path.join(SCRIPT_DIR, OUT_DIR, "phonon_band.csv")
REF_CSV = os.path.join(SCRIPT_DIR, "abinit_dfpt_ref_2k.csv")

# Threshold for frequency comparison (cm^-1)
# 2x2x2 k-mesh is coarse, so we allow a moderate tolerance.
# Optical modes: 5 cm^-1; acoustic/imaginary modes: 10 cm^-1
OPTICAL_THRESHOLD = 5.0   # cm^-1
ACOUSTIC_THRESHOLD = 10.0  # cm^-1
GAMMA_ACOUSTIC_THRESHOLD = 1.0  # cm^-1, acoustic at Gamma should be ~0


# q-point index in phonon_band.csv for each HSP
# Path: Gamma(0) - X(1) - W(2) - K(3) - Gamma(4) - L(5)
QPOINT_MAP = {
    "Gamma": 0,
    "X": 1,
    "L": 5,
}


def run_dft_zig():
    """Run DFT-Zig DFPT calculation."""
    print("=" * 50)
    print("Running DFT-Zig DFPT...")
    print("=" * 50)
    result = subprocess.run(
        [DFT_ZIG, TOML_FILE],
        cwd=SCRIPT_DIR,
        capture_output=True, text=True,
        timeout=1200
    )
    if result.returncode != 0:
        print(f"DFT-Zig FAILED! Return code: {result.returncode}")
        print(f"stderr: {result.stderr[-2000:]}")
        sys.exit(1)
    print("DFT-Zig done.\n")


def load_ref_data():
    """Load ABINIT reference data from CSV."""
    ref = {}
    with open(REF_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label']
            freqs = [float(row[f'mode_{i}']) for i in range(6)]
            ref[label] = freqs
    return ref


def load_dft_zig_data():
    """Load DFT-Zig phonon_band.csv output."""
    if not os.path.exists(PHONON_CSV):
        print(f"Error: {PHONON_CSV} not found.")
        print("Run with --run flag first.")
        sys.exit(1)

    data = []
    with open(PHONON_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            freqs = [float(row[f'mode_{i}']) for i in range(6)]
            data.append(freqs)
    return data


def check_results(dz_data, ref_data):
    """Compare DFT-Zig phonon frequencies against ABINIT reference."""
    print("=" * 50)
    print("DFPT Regression Check Results")
    print("=" * 50)

    all_passed = True

    for label, ref_freqs in ref_data.items():
        idx = QPOINT_MAP.get(label)
        if idx is None:
            print(f"  WARNING: No mapping for q-point '{label}', skipping.")
            continue

        if idx >= len(dz_data):
            print(f"  FAIL: q-point index {idx} ({label}) out of range")
            all_passed = False
            continue

        dz_freqs = dz_data[idx]

        print(f"\n  {label} (q-index={idx}):")
        print(f"    {'Mode':>6s}  {'DFT-Zig':>10s}  {'ABINIT':>10s}  {'Diff':>10s}  {'Thresh':>8s}  {'Status':>6s}")

        for m in range(6):
            dz_f = dz_freqs[m]
            ab_f = ref_freqs[m]
            diff = abs(dz_f - ab_f)

            # Choose threshold based on mode type
            if label.startswith("Gamma") and m < 3:
                # Acoustic modes at Gamma: should be ~0
                thresh = GAMMA_ACOUSTIC_THRESHOLD
                # For Gamma acoustic, compare absolute value to 0
                diff = abs(dz_f)
                ab_f_display = 0.0
            elif ab_f < 0 or abs(ab_f) < 50:
                # Imaginary or low-frequency modes: looser threshold
                thresh = ACOUSTIC_THRESHOLD
            else:
                # Optical modes
                thresh = OPTICAL_THRESHOLD

            passed = diff <= thresh
            status = "OK" if passed else "FAIL"
            if not passed:
                all_passed = False

            if label.startswith("Gamma") and m < 3:
                print(f"    {m:>6d}  {dz_f:>10.1f}  {'~0':>10s}  {diff:>+10.2f}  {thresh:>8.1f}  [{status}]")
            else:
                print(f"    {m:>6d}  {dz_f:>10.1f}  {ab_f:>10.1f}  {diff:>+10.2f}  {thresh:>8.1f}  [{status}]")

    print()
    if all_passed:
        print("PASS: All DFPT checks passed!")
    else:
        print("FAIL: Some DFPT checks failed!")

    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Si DFPT phonon regression test')
    parser.add_argument('--run', action='store_true', help='Run DFT-Zig calculation')
    args = parser.parse_args()

    if args.run:
        run_dft_zig()

    # Load data
    ref_data = load_ref_data()
    dz_data = load_dft_zig_data()

    # Check
    passed = check_results(dz_data, ref_data)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
