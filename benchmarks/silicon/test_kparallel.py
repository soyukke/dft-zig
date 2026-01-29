#!/usr/bin/env python3
"""Test k-point parallel execution produces identical results to serial."""

import csv
import os
import subprocess
import sys

# Tolerance for floating point comparison (in Ry)
# Must be slightly larger than LOBPCG convergence tolerance (iterative_tol = 1e-6)
# since different execution order can lead to different convergence paths
TOLERANCE = 1e-5


def run_dft_zig(config_file):
    """Run DFT-Zig with specified config."""
    print(f"Running DFT-Zig with {config_file}...")
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", config_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"DFT-Zig failed: {result.stderr}")
        sys.exit(1)
    print("Done.\n")


def load_band_energies(out_dir):
    """Load band energies from CSV."""
    data = []
    csv_path = os.path.join(out_dir, 'band_energies.csv')
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def compare_results(serial_dir, parallel_dir):
    """Compare band energies between serial and parallel runs."""
    serial_data = load_band_energies(serial_dir)
    parallel_data = load_band_energies(parallel_dir)

    if len(serial_data) != len(parallel_data):
        print(f"FAIL: k-point count mismatch: serial={len(serial_data)}, parallel={len(parallel_data)}")
        return False

    # Get band columns
    band_cols = [k for k in serial_data[0].keys() if k.startswith('band')]
    nbands = len(band_cols)

    max_diff = 0.0
    max_diff_info = None
    total_checks = 0
    failed_checks = 0

    for i, (s_row, p_row) in enumerate(zip(serial_data, parallel_data)):
        for band_col in band_cols:
            s_val = float(s_row[band_col])
            p_val = float(p_row[band_col])
            diff = abs(s_val - p_val)

            if diff > max_diff:
                max_diff = diff
                max_diff_info = (i, band_col, s_val, p_val)

            if diff > TOLERANCE:
                failed_checks += 1

            total_checks += 1

    print("=" * 50)
    print("k-parallel vs Serial Comparison")
    print("=" * 50)
    print(f"k-points: {len(serial_data)}")
    print(f"Bands: {nbands}")
    print(f"Total checks: {total_checks}")
    print(f"Tolerance: {TOLERANCE:.0e} Ry")
    print(f"Max difference: {max_diff:.2e} Ry")

    if max_diff_info:
        kpt, band, s_val, p_val = max_diff_info
        print(f"  (at k-point {kpt}, {band}: serial={s_val:.10f}, parallel={p_val:.10f})")

    if failed_checks == 0:
        print("\nPASS: k-parallel execution produces identical results!")
        return True
    else:
        print(f"\nFAIL: {failed_checks}/{total_checks} values exceed tolerance")
        return False


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    run_serial = "--run" in sys.argv
    run_parallel = "--run" in sys.argv or "--run-parallel" in sys.argv

    # Use lightweight configs for faster testing
    serial_config = "dft_zig_kparallel_test_serial.toml"
    parallel_config = "dft_zig_kparallel_test_parallel.toml"
    serial_dir = "out_kparallel_serial"
    parallel_dir = "out_kparallel_parallel"

    if run_serial:
        print("=" * 50)
        print("Running Serial (kpoint_threads=1)")
        print("=" * 50)
        run_dft_zig(serial_config)

    if run_parallel:
        print("=" * 50)
        print("Running k-parallel (kpoint_threads=4)")
        print("=" * 50)
        run_dft_zig(parallel_config)

    # Check if output directories exist
    if not os.path.exists(os.path.join(serial_dir, 'band_energies.csv')):
        print(f"Error: Serial results not found at {serial_dir}/band_energies.csv")
        print("Run with --run to generate serial results first.")
        sys.exit(1)

    if not os.path.exists(os.path.join(parallel_dir, 'band_energies.csv')):
        print(f"Error: Parallel results not found at {parallel_dir}/band_energies.csv")
        print("Run with --run or --run-parallel to generate parallel results.")
        sys.exit(1)

    passed = compare_results(serial_dir, parallel_dir)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
