#!/usr/bin/env python3
"""Molecule SCF regression test: DFT-Zig vs PySCF baseline.

Runs df_benchmark and compares conventional and DF energies against
PySCF reference values stored in baseline.json.

Usage:
    python3 test_molecule.py --run     # run df_benchmark and check
    python3 test_molecule.py --check   # check from cached output file
"""

import argparse
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BASELINE_PATH = os.path.join(SCRIPT_DIR, "baseline.json")
CACHED_OUTPUT = os.path.join(SCRIPT_DIR, "df_benchmark_output.txt")
DF_BENCHMARK_BIN = os.path.join(PROJECT_ROOT, "zig-out", "bin", "df_benchmark")

# Threshold: 0.1 mHa = 100 uHa
THRESHOLD_MHA = 0.1


def load_baseline():
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return data["molecules"]


def run_df_benchmark():
    """Run df_benchmark binary and return stderr output."""
    if not os.path.isfile(DF_BENCHMARK_BIN):
        print(f"ERROR: {DF_BENCHMARK_BIN} not found. Run 'just build' first.")
        sys.exit(1)

    print(f"Running {DF_BENCHMARK_BIN} ...")
    result = subprocess.run(
        [DF_BENCHMARK_BIN],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"ERROR: df_benchmark exited with code {result.returncode}")
        print(result.stderr)
        sys.exit(1)

    # df_benchmark uses std.debug.print -> stderr
    output = result.stderr
    # Save for later --check runs
    with open(CACHED_OUTPUT, "w") as f:
        f.write(output)

    return output


def parse_zig_output(output):
    """Parse df_benchmark stderr output.

    Expected line format (whitespace-separated):
      H2          10      -1.1785340296     0.03     5      -1.1785495330     0.03     5    1.55e-5    1.00x
    """
    results = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-") or line.startswith("Molecule"):
            continue
        # Match molecule data lines: name nbas conv_e conv_t conv_iter df_e df_t df_iter diff speedup
        parts = line.split()
        if len(parts) < 10:
            continue
        name = parts[0]
        # Skip header-like lines
        if name in ("Molecule", "Density", "B3LYP", "Grid:", "Molecules:"):
            continue
        try:
            conv_energy = float(parts[2])
            df_energy = float(parts[5])
            results[name] = {"conv": conv_energy, "df": df_energy}
        except (ValueError, IndexError):
            continue

    return results


def check_results(zig_results, baseline):
    """Compare DFT-Zig results against PySCF baseline."""
    print()
    header = f"{'Molecule':<10} {'Conv dE (mHa)':>14} {'DF dE (mHa)':>14} {'Status':>8}"
    print(header)
    print("-" * len(header))

    all_pass = True
    tested = 0

    for name in baseline:
        if name not in zig_results:
            print(f"{name:<10} {'MISSING':>14} {'MISSING':>14} {'FAIL':>8}")
            all_pass = False
            continue

        ref = baseline[name]
        zig = zig_results[name]

        conv_de_mha = (zig["conv"] - ref["conv"]) * 1000.0
        df_de_mha = (zig["df"] - ref["df"]) * 1000.0

        conv_ok = abs(conv_de_mha) < THRESHOLD_MHA
        df_ok = abs(df_de_mha) < THRESHOLD_MHA
        status = "OK" if (conv_ok and df_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{name:<10} {conv_de_mha:>+14.3f} {df_de_mha:>+14.3f} {status:>8}")
        tested += 1

    print("-" * len(header))
    print()

    if all_pass and tested == len(baseline):
        print(f"PASS: All {tested} molecules passed (threshold: {THRESHOLD_MHA} mHa)")
    else:
        if not all_pass:
            print(f"FAIL: Some molecules exceeded threshold ({THRESHOLD_MHA} mHa)")
        if tested < len(baseline):
            missing = len(baseline) - tested
            print(f"FAIL: {missing} molecule(s) missing from DFT-Zig output")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Molecule SCF regression test")
    parser.add_argument("--run", action="store_true", help="Run df_benchmark and check")
    parser.add_argument("--check", action="store_true", help="Check from cached output")
    args = parser.parse_args()

    if not args.run and not args.check:
        parser.print_help()
        sys.exit(1)

    baseline = load_baseline()

    if args.run:
        output = run_df_benchmark()
    elif args.check:
        if not os.path.isfile(CACHED_OUTPUT):
            print(f"ERROR: {CACHED_OUTPUT} not found. Use --run first.")
            sys.exit(1)
        with open(CACHED_OUTPUT) as f:
            output = f.read()

    zig_results = parse_zig_output(output)

    if not zig_results:
        print("ERROR: No molecule results parsed from df_benchmark output.")
        print("Raw output (first 500 chars):")
        print(output[:500])
        sys.exit(1)

    check_results(zig_results, baseline)


if __name__ == "__main__":
    main()
