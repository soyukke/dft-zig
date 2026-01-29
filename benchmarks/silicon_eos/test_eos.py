#!/usr/bin/env python3
"""Si EOS regression test: DFT-Zig vs ABINIT (3 points).

Runs DFT-Zig and ABINIT at 3 scale factors (0.96, 1.00, 1.04) and checks:
1. All SCF calculations converge
2. Energy difference |dE| < threshold at each point
3. E-V curve has correct shape (minimum near s=1.00)
"""

import subprocess
import os
import re
import tempfile
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DFT_ZIG = os.path.join(PROJECT_ROOT, "zig-out", "bin", "dft_zig")
ABINIT = os.path.join(PROJECT_ROOT, "external", "abinit", "build",
                       "abinit-10.4.7", "src", "98_main", "abinit")
PSEUDO_DIR = os.path.join(PROJECT_ROOT, "pseudo")

A0 = 5.431  # Angstrom
BOHR_TO_ANG = 0.529177249
SCALE_FACTORS = [0.96, 1.00, 1.04]

# Thresholds
ENERGY_DIFF_THRESHOLD = 0.5  # mRy per point


def cell_volume_bohr(a_ang):
    a_bohr = a_ang / BOHR_TO_ANG
    return a_bohr ** 3 / 4.0


DFT_ZIG_TOML_TEMPLATE = """\
title = "silicon_eos_s{scale:.4f}"
xyz = "{xyz_path}"
out_dir = "{out_dir}"
units = "angstrom"
linalg_backend = "accelerate"
threads = 0

[[pseudopotential]]
element = "Si"
path = "{pseudo_path}"
format = "upf"

[cell]
a1 = [0.0, {half:.6f}, {half:.6f}]
a2 = [{half:.6f}, 0.0, {half:.6f}]
a3 = [{half:.6f}, {half:.6f}, 0.0]

[scf]
enabled = true
solver = "iterative"
xc = "pbe"
ecut_ry = 15.0
kmesh = [8, 8, 8]
kmesh_shift = [0.0, 0.0, 0.0]
grid = [0, 0, 0]
mixing_beta = 0.3
pulay_history = 8
pulay_start = 4
max_iter = 200
convergence = 1e-6
convergence_metric = "potential"
iterative_max_iter = 200
iterative_tol = 1e-8
use_rfft = true
fft_backend = "fftw"
quiet = true

[ewald]
tol = 1e-8
"""

ABINIT_INPUT_TEMPLATE = """\
pp_dirpath "{pseudo_dir}"
pseudos "Si.upf"

ntypat 1
znucl 14
natom 2
typat 1 1

acell 1 1 1 angstrom
rprim
  0.0      {half:.6f} {half:.6f}
  {half:.6f} 0.0      {half:.6f}
  {half:.6f} {half:.6f} 0.0

xred
  0.0   0.0   0.0
  0.25  0.25  0.25

ecut 7.5
nband 8

ndtset 1

iscf1 7
diemix1 0.3
tolvrs1 1.0d-8
nstep1 100
kptopt1 1
ngkpt1 8 8 8
nshiftk1 1
shiftk1 0.0 0.0 0.0
"""


def run_dft_zig(scale_factors):
    results = {}
    eos_dir = os.path.join(SCRIPT_DIR, "out_eos_test")
    os.makedirs(eos_dir, exist_ok=True)
    pseudo_path = os.path.join(PSEUDO_DIR, "Si.upf")

    for s in scale_factors:
        a = A0 * s
        half = a / 2.0
        quarter = a / 4.0
        out_dir = os.path.join(eos_dir, f"s_{s:.4f}")

        xyz_path = os.path.join(eos_dir, f"silicon_s{s:.4f}.xyz")
        with open(xyz_path, 'w') as f:
            f.write(f"2\nSi diamond s={s:.4f}\n")
            f.write(f"Si 0.0 0.0 0.0\n")
            f.write(f"Si {quarter:.6f} {quarter:.6f} {quarter:.6f}\n")

        toml_content = DFT_ZIG_TOML_TEMPLATE.format(
            scale=s, xyz_path=xyz_path, out_dir=out_dir,
            pseudo_path=pseudo_path, half=half
        )
        toml_path = os.path.join(eos_dir, f"dft_zig_s{s:.4f}.toml")
        with open(toml_path, 'w') as f:
            f.write(toml_content)

        print(f"  DFT-Zig s={s:.4f} ...", end=" ", flush=True)
        result = subprocess.run(
            [DFT_ZIG, toml_path],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"FAILED (rc={result.returncode})")
            continue

        status_path = os.path.join(out_dir, "status.txt")
        energy = None
        converged = False
        if os.path.exists(status_path):
            with open(status_path) as f:
                for line in f:
                    if line.startswith("scf_energy_total"):
                        energy = float(line.split("=")[1].strip())
                    if line.startswith("scf_converged"):
                        converged = line.split("=")[1].strip() == "true"

        if energy is not None:
            vol = cell_volume_bohr(a)
            results[s] = (vol, energy, converged)
            conv_str = "OK" if converged else "NOT CONVERGED"
            print(f"E={energy:.8f} Ry [{conv_str}]")
        else:
            print("FAILED (no energy)")

    return results


def run_abinit(scale_factors):
    results = {}

    for s in scale_factors:
        a = A0 * s
        half = a / 2.0
        input_content = ABINIT_INPUT_TEMPLATE.format(
            pseudo_dir=PSEUDO_DIR, half=half
        )

        print(f"  ABINIT  s={s:.4f} ...", end=" ", flush=True)
        tmpdir = tempfile.mkdtemp(prefix="abinit_eos_test_")
        try:
            input_path = os.path.join(tmpdir, "silicon_eos.in")
            with open(input_path, 'w') as f:
                f.write(input_content)

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"

            result = subprocess.run(
                [ABINIT, input_path],
                capture_output=True, text=True, timeout=600,
                cwd=tmpdir, env=env
            )

            if result.returncode != 0:
                print(f"FAILED (rc={result.returncode})")
                continue

            abo_path = os.path.join(tmpdir, "silicon_eos.abo")
            energy_ha = None
            if os.path.exists(abo_path):
                with open(abo_path) as f:
                    for line in f:
                        m = re.search(r'etotal\d*\s+([-\d.E+]+)', line)
                        if m:
                            energy_ha = float(m.group(1))

            if energy_ha is not None:
                energy_ry = energy_ha * 2.0
                vol = cell_volume_bohr(a)
                results[s] = (vol, energy_ry)
                print(f"E={energy_ry:.8f} Ry")
            else:
                print("FAILED (no etotal)")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def run_check(dft_results, abinit_results):
    print("\n==================================================")
    print("EOS Regression Check Results")
    print("==================================================")

    passed = True

    # Check all points computed
    for s in SCALE_FACTORS:
        if s not in dft_results:
            print(f"FAIL: DFT-Zig missing s={s:.2f}")
            passed = False
        if s not in abinit_results:
            print(f"FAIL: ABINIT missing s={s:.2f}")
            passed = False

    if not passed:
        return 1

    # Check SCF convergence
    print("SCF convergence:")
    for s in SCALE_FACTORS:
        conv = dft_results[s][2]
        status = "OK" if conv else "FAIL"
        print(f"  s={s:.2f}: {status}")
        if not conv:
            passed = False

    # Check energy differences
    print(f"\nEnergy difference (threshold: {ENERGY_DIFF_THRESHOLD} mRy):")
    for s in SCALE_FACTORS:
        dft_e = dft_results[s][1]
        abi_e = abinit_results[s][1]
        de_mry = (dft_e - abi_e) * 1000
        status = "OK" if abs(de_mry) < ENERGY_DIFF_THRESHOLD else "FAIL"
        print(f"  s={s:.2f}: dE = {de_mry:+.3f} mRy [{status}]")
        if abs(de_mry) >= ENERGY_DIFF_THRESHOLD:
            passed = False

    # Check E-V curve shape: E(0.96) > E(1.00) and E(1.04) > E(1.00)
    print("\nE-V curve shape:")
    e_low = dft_results[0.96][1]
    e_mid = dft_results[1.00][1]
    e_high = dft_results[1.04][1]
    shape_ok = e_low > e_mid and e_high > e_mid
    status = "OK" if shape_ok else "FAIL"
    print(f"  E(0.96)={e_low:.8f} > E(1.00)={e_mid:.8f} < E(1.04)={e_high:.8f} [{status}]")
    if not shape_ok:
        passed = False

    print()
    if passed:
        print("PASS: All EOS checks passed!")
        return 0
    else:
        print("FAIL: Some EOS checks failed!")
        return 1


def main():
    run = "--run" in sys.argv

    print("==================================================")
    print("Silicon EOS Regression Test (3 points)")
    print("==================================================")
    print(f"Scale factors: {SCALE_FACTORS}")
    print(f"ecut=15 Ry, k-mesh=8x8x8, PBE")
    print()

    if run:
        print("--- DFT-Zig ---")
        dft_results = run_dft_zig(SCALE_FACTORS)
        print()

        print("--- ABINIT ---")
        abinit_results = run_abinit(SCALE_FACTORS)
        print()
    else:
        print("Use --run to execute calculations.")
        return

    rc = run_check(dft_results, abinit_results)
    sys.exit(rc)


if __name__ == "__main__":
    main()
