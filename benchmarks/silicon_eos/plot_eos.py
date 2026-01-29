#!/usr/bin/env python3
"""Si equation of state (E-V curve): DFT-Zig vs ABINIT.

Sweeps lattice constant around a0=5.431 A, fits Birch-Murnaghan EOS,
and compares equilibrium lattice constant and bulk modulus.
"""

import subprocess
import os
import re
import tempfile
import shutil

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DFT_ZIG = os.path.join(PROJECT_ROOT, "zig-out", "bin", "dft_zig")
ABINIT = os.path.join(PROJECT_ROOT, "out", "abinit", "build",
                       "abinit-10.4.7", "src", "98_main", "abinit")
PSEUDO_DIR = os.path.join(PROJECT_ROOT, "pseudo")

# --- Si diamond structure ---
A0 = 5.431  # Angstrom, experimental lattice constant

SCALE_FACTORS = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06]

BOHR_TO_ANG = 0.529177249


def birch_murnaghan(V, E0, V0, B0, B0p):
    """3rd-order Birch-Murnaghan equation of state."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * (
        (eta - 1.0) ** 3 * B0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def cell_volume_bohr(a_ang):
    """Volume of FCC primitive cell in bohr^3."""
    a_bohr = a_ang / BOHR_TO_ANG
    return a_bohr ** 3 / 4.0


# =========================================================================
# DFT-Zig
# =========================================================================

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


def run_dft_zig(scale_factors):
    """Run DFT-Zig for each scale factor.

    Returns {scale: (volume_bohr3, energy_ry, converged)}.
    """
    results = {}
    eos_dir = os.path.join(SCRIPT_DIR, "out_eos")
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

        print(f"  DFT-Zig s={s:.4f} (a={a:.4f} A) ...", end=" ", flush=True)

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
            print(f"E = {energy:.8f} Ry, V = {vol:.4f} bohr^3 [{conv_str}]")
        else:
            print("FAILED (no energy)")

    return results


# =========================================================================
# ABINIT
# =========================================================================

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


def run_abinit(scale_factors):
    """Run ABINIT for each scale factor.

    Returns {scale: (volume_bohr3, energy_ry)}.
    """
    results = {}

    for s in scale_factors:
        a = A0 * s
        half = a / 2.0

        input_content = ABINIT_INPUT_TEMPLATE.format(
            pseudo_dir=PSEUDO_DIR, half=half
        )

        print(f"  ABINIT  s={s:.4f} (a={a:.4f} A) ...", end=" ", flush=True)

        tmpdir = tempfile.mkdtemp(prefix="abinit_eos_")
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
                print(f"E = {energy_ry:.8f} Ry")
            else:
                print("FAILED (no etotal)")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return results


# =========================================================================
# Fitting & Plotting
# =========================================================================

def fit_birch_murnaghan(volumes, energies):
    """Fit BM EOS and return (E0, V0, B0_GPa, B0p)."""
    idx_min = np.argmin(energies)
    E0_guess = energies[idx_min]
    V0_guess = volumes[idx_min]
    B0_guess = 90.0 / 14710.5
    B0p_guess = 4.0

    popt, pcov = curve_fit(
        birch_murnaghan, volumes, energies,
        p0=[E0_guess, V0_guess, B0_guess, B0p_guess],
        maxfev=10000
    )
    E0, V0, B0, B0p = popt
    B0_gpa = B0 * 14710.5
    return E0, V0, B0_gpa, B0p


def v0_to_alat(V0_bohr3):
    """Convert FCC primitive cell volume (bohr^3) to lattice constant (Angstrom)."""
    a_bohr = (4.0 * V0_bohr3) ** (1.0 / 3.0)
    return a_bohr * BOHR_TO_ANG


def main():
    print("=" * 60)
    print("Si Equation of State: DFT-Zig vs ABINIT")
    print("=" * 60)
    print(f"a0 = {A0} A, ecut = 15 Ry / 7.5 Ha, k-mesh = 8x8x8")
    print(f"Scale factors: {SCALE_FACTORS}")
    print()

    # --- Run both codes ---
    print("--- DFT-Zig ---")
    dft_results = run_dft_zig(SCALE_FACTORS)
    print()

    print("--- ABINIT ---")
    abinit_results = run_abinit(SCALE_FACTORS)
    print()

    # Check convergence
    n_unconverged = sum(1 for r in dft_results.values() if not r[2])
    if n_unconverged > 0:
        print(f"WARNING: {n_unconverged} DFT-Zig SCF calculations did not converge!")
        print()

    if len(dft_results) < 5 or len(abinit_results) < 5:
        print("ERROR: Not enough data points for fitting.")
        return

    # --- Prepare arrays ---
    common_scales = sorted(set(dft_results.keys()) & set(abinit_results.keys()))
    print(f"Common data points: {len(common_scales)}")

    dft_vols = np.array([dft_results[s][0] for s in common_scales])
    dft_energies = np.array([dft_results[s][1] for s in common_scales])
    abi_vols = np.array([abinit_results[s][0] for s in common_scales])
    abi_energies = np.array([abinit_results[s][1] for s in common_scales])

    # --- Fit BM EOS ---
    print("\n--- Birch-Murnaghan Fit ---")

    # Fit ABINIT
    abi_E0, abi_V0, abi_B0, abi_B0p = fit_birch_murnaghan(abi_vols, abi_energies)
    abi_alat = v0_to_alat(abi_V0)

    # Fit DFT-Zig
    dft_E0, dft_V0, dft_B0, dft_B0p = fit_birch_murnaghan(dft_vols, dft_energies)
    dft_alat = v0_to_alat(dft_V0)

    print(f"{'':>20s}  {'DFT-Zig':>12s}  {'ABINIT':>12s}  {'Diff':>12s}")
    print("-" * 62)
    print(f"{'E0 (Ry)':>20s}  {dft_E0:>12.6f}  {abi_E0:>12.6f}  {(dft_E0-abi_E0)*1000:>10.3f} mRy")
    print(f"{'V0 (bohr^3)':>20s}  {dft_V0:>12.4f}  {abi_V0:>12.4f}  {dft_V0-abi_V0:>12.4f}")
    print(f"{'a_eq (A)':>20s}  {dft_alat:>12.4f}  {abi_alat:>12.4f}  {dft_alat-abi_alat:>12.4f}")
    print(f"{'B0 (GPa)':>20s}  {dft_B0:>12.2f}  {abi_B0:>12.2f}  {dft_B0-abi_B0:>12.2f}")
    print(f"{'B0p':>20s}  {dft_B0p:>12.3f}  {abi_B0p:>12.3f}  {dft_B0p-abi_B0p:>12.3f}")
    print()
    print("Reference (experiment): a = 5.431 A, B0 ~ 99 GPa")
    print("Reference (PBE):        a ~ 5.47 A,  B0 ~ 89 GPa")

    # --- Energy difference ---
    delta_E = dft_energies - abi_energies
    print(f"\nEnergy difference:")
    print(f"  Max |dE| = {np.max(np.abs(delta_E))*1000:.3f} mRy")
    print(f"  Mean |dE| = {np.mean(np.abs(delta_E))*1000:.3f} mRy")

    # --- Per-point table ---
    print(f"\n{'s':>6s} {'V(bohr3)':>10s} {'DFT-Zig':>14s} {'ABINIT':>14s} {'dE(mRy)':>10s}")
    for i, s in enumerate(common_scales):
        print(f"{s:>6.2f} {dft_vols[i]:>10.2f} {dft_energies[i]:>14.8f} {abi_energies[i]:>14.8f} "
              f"{delta_E[i]*1000:>10.3f}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Smooth BM curves
    v_fine = np.linspace(dft_vols.min() * 0.99, dft_vols.max() * 1.01, 200)
    dft_fit = birch_murnaghan(v_fine, dft_E0, dft_V0, dft_B0 / 14710.5, dft_B0p)
    abi_fit = birch_murnaghan(v_fine, abi_E0, abi_V0, abi_B0 / 14710.5, abi_B0p)

    ax1.plot(dft_vols, dft_energies, 'o', color='#2196F3', markersize=8,
             label='DFT-Zig', zorder=5)
    ax1.plot(v_fine, dft_fit, '-', color='#2196F3', linewidth=1.5, alpha=0.7)
    ax1.plot(abi_vols, abi_energies, 's', color='#FF5722', markersize=8,
             label='ABINIT', zorder=5)
    ax1.plot(v_fine, abi_fit, '-', color='#FF5722', linewidth=1.5, alpha=0.7)

    ax1.axvline(dft_V0, color='#2196F3', linestyle='--', alpha=0.3)
    ax1.axvline(abi_V0, color='#FF5722', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Volume (bohr$^3$)', fontsize=12)
    ax1.set_ylabel('Total Energy (Ry)', fontsize=12)
    ax1.set_title('Si Equation of State: DFT-Zig vs ABINIT\n'
                  'ecut=15 Ry, 8x8x8 k-mesh, PBE', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    textstr = (
        f'DFT-Zig: a={dft_alat:.4f} A, B={dft_B0:.1f} GPa\n'
        f'ABINIT:  a={abi_alat:.4f} A, B={abi_B0:.1f} GPa'
    )
    ax1.text(0.02, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Energy difference subplot
    ax2.plot(dft_vols, delta_E * 1000, 'D-', color='#4CAF50',
             markersize=6, linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Volume (bohr$^3$)', fontsize=12)
    ax2.set_ylabel('$\\Delta E$ (mRy)', fontsize=12)
    ax2.set_title('Energy Difference: DFT-Zig $-$ ABINIT',
                  fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "eos_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
