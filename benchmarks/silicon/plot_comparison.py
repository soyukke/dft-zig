#!/usr/bin/env python3
"""Compare silicon band structures: DFT-Zig vs ABINIT (fair comparison)."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
import sys

Ry_to_eV = 13.6057
Ha_to_eV = 27.2114


def load_dft_zig_timing():
    """Load DFT-Zig timing from timing.txt."""
    timing_file = 'out_dft_zig_fair/timing.txt'
    if not os.path.exists(timing_file):
        return None
    with open(timing_file) as f:
        for line in f:
            if line.startswith('total_sec'):
                return float(line.split('=')[1].strip())
    return None


def load_abinit_timing():
    """Load ABINIT timing from .abo file."""
    abo_file = 'abinit/silicon_fair.abo'
    if not os.path.exists(abo_file):
        return None
    with open(abo_file) as f:
        content = f.read()
    # Look for "Overall time at end (sec) : cpu=X.X  wall=X.X"
    match = re.search(r'Overall time at end.*wall=\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    return None

def run_calculations():
    """Run both DFT-Zig and ABINIT calculations."""
    print("=" * 50)
    print("Building DFT-Zig (ReleaseFast)...")
    print("=" * 50)
    result = subprocess.run(
        ["zig", "build", "-Doptimize=ReleaseFast"],
        cwd="../..",
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Build failed:", result.stderr)
        sys.exit(1)
    print("Build done.\n")

    print("=" * 50)
    print("Running DFT-Zig...")
    print("=" * 50)
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", "dft_zig_fair.toml"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("DFT-Zig failed:", result.stderr)
        sys.exit(1)
    print("DFT-Zig done.\n")

    print("=" * 50)
    print("Running ABINIT (fair comparison)...")
    print("=" * 50)
    os.chdir("abinit")
    # Clean previous outputs
    for f in os.listdir("."):
        if f.startswith("silicon_fairo") or (f.endswith(".nc") and "fair" in f):
            try:
                os.remove(f)
            except:
                pass
    result = subprocess.run(
        ["../../../external/abinit/build/abinit-10.4.7/src/98_main/abinit", "silicon_fair.in"],
        capture_output=True, text=True
    )
    os.chdir("..")
    if result.returncode != 0:
        print("ABINIT failed:", result.stderr)
        sys.exit(1)
    print("ABINIT done.\n")

def load_dft_zig():
    """Load DFT-Zig band data."""
    dft_data = []
    with open('out_dft_zig_fair/band_energies.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dft_data.append(row)

    dft_dist = [float(row['dist']) for row in dft_data]
    dft_nbands = len([k for k in dft_data[0].keys() if k.startswith('band')])
    dft_bands = [[float(row[f'band{i}']) * Ry_to_eV for row in dft_data] for i in range(dft_nbands)]

    # Read k-point labels
    kpoints_data = []
    with open('out_dft_zig_fair/band_kpoints.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kpoints_data.append(row)

    labels = []
    label_positions = []
    for row in kpoints_data:
        if row.get('label', ''):
            labels.append(row['label'])
            label_positions.append(float(row['dist']))

    return dft_dist, dft_bands, labels, label_positions

def load_abinit():
    """Load ABINIT band data from fair comparison."""
    abinit_bands = [[] for _ in range(8)]
    abinit_kpts = []

    # Use the fair comparison output
    eig_file = 'abinit/silicon_fairo_DS2_EIG'

    with open(eig_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'kpt#' in line:
            # Parse k-point coordinates from line like "kpt#   1, nband=  8, wtk=  0.00990, kpt=  0.0000  0.0000  0.0000"
            parts = line.split('kpt=')
            if len(parts) > 1:
                kpt_str = parts[1].strip().split()
                kpt = [float(x) for x in kpt_str[:3]]
                abinit_kpts.append(kpt)

            # Parse eigenvalues on following lines
            eig_text = ''
            for j in range(1, 10):
                if i+j < len(lines) and 'kpt#' not in lines[i+j] and lines[i+j].strip():
                    eig_text += lines[i+j]
                else:
                    break
            eigs = [float(x) * Ha_to_eV for x in eig_text.split()]
            for b, e in enumerate(eigs[:8]):
                abinit_bands[b].append(e)

    return abinit_bands, abinit_kpts


def calculate_kpath_distances(kpts):
    """Calculate cumulative distances along k-path using reciprocal lattice vectors."""
    # Silicon FCC primitive cell: a/2 * [(0,1,1), (1,0,1), (1,1,0)]
    # a = 5.431 Angstrom, a/2 = 2.7155 Angstrom
    a = 2.7155
    # Real space lattice vectors
    a1 = np.array([0.0, a, a])
    a2 = np.array([a, 0.0, a])
    a3 = np.array([a, a, 0.0])

    # Volume of unit cell
    vol = np.dot(a1, np.cross(a2, a3))

    # Reciprocal lattice vectors: b_i = 2*pi/vol * (a_j x a_k)
    b1 = 2 * np.pi / vol * np.cross(a2, a3)
    b2 = 2 * np.pi / vol * np.cross(a3, a1)
    b3 = 2 * np.pi / vol * np.cross(a1, a2)

    # Calculate distances
    distances = [0.0]
    for i in range(1, len(kpts)):
        # k in Cartesian = k1*b1 + k2*b2 + k3*b3
        k_prev = kpts[i-1][0]*b1 + kpts[i-1][1]*b2 + kpts[i-1][2]*b3
        k_curr = kpts[i][0]*b1 + kpts[i][1]*b2 + kpts[i][2]*b3
        dk = np.linalg.norm(k_curr - k_prev)
        distances.append(distances[-1] + dk)

    return distances

def plot_comparison():
    """Create comparison plots."""
    # Load data
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands, abinit_kpts = load_abinit()

    # Load timing information
    dft_time = load_dft_zig_timing()
    abinit_time = load_abinit_timing()

    # Align VBM to 0
    dft_vbm = max(dft_bands[3])
    for band in dft_bands:
        for i in range(len(band)):
            band[i] -= dft_vbm

    abinit_vbm = max(abinit_bands[3])
    for band in abinit_bands:
        for i in range(len(band)):
            band[i] -= abinit_vbm

    # Both codes use the same k-points (101 points on G-X-W-K-G-L path)
    # Use DFT-Zig distances directly for ABINIT since they should match
    if len(abinit_bands[0]) == len(dft_dist):
        abinit_dist = dft_dist
        print(f"Using DFT-Zig k-path distances for ABINIT ({len(abinit_dist)} points)")
    else:
        print(f"Warning: k-point count mismatch: DFT-Zig={len(dft_dist)}, ABINIT={len(abinit_bands[0])}")
        abinit_dist = calculate_kpath_distances(abinit_kpts)

    # Calculate band gaps
    dft_cbm = min([min(b) for b in dft_bands[4:]])
    dft_gap = dft_cbm  # VBM is at 0
    abinit_cbm = min([min(b) for b in abinit_bands[4:]])
    abinit_gap = abinit_cbm  # VBM is at 0

    # ============== Plot 1: Side by side ==============
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    # DFT-Zig
    for i, band in enumerate(dft_bands):
        color = 'blue' if i < 4 else 'red'
        ax1.plot(dft_dist, band, color=color, linewidth=1.5)
    for pos in label_positions:
        ax1.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Energy (eV)')
    dft_title = f'DFT-Zig (Gap: {dft_gap:.3f} eV'
    if dft_time is not None:
        dft_title += f', {dft_time:.1f}s'
    dft_title += ')'
    ax1.set_title(dft_title)
    ax1.set_xlim(min(dft_dist), max(dft_dist))
    ax1.set_ylim(-15, 10)

    # ABINIT
    for i, band in enumerate(abinit_bands):
        color = 'blue' if i < 4 else 'red'
        ax2.plot(abinit_dist, band, color=color, linewidth=1.5)
    for pos in label_positions:
        ax2.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(label_positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Energy (eV)')
    abinit_title = f'ABINIT (Gap: {abinit_gap:.3f} eV'
    if abinit_time is not None:
        abinit_title += f', {abinit_time:.1f}s'
    abinit_title += ')'
    ax2.set_title(abinit_title)
    ax2.set_xlim(min(dft_dist), max(dft_dist))
    ax2.set_ylim(-15, 10)

    # Overlay
    for i, band in enumerate(dft_bands):
        ax3.plot(dft_dist, band, 'b-', linewidth=1.5, label='DFT-Zig' if i == 0 else None)
    for i, band in enumerate(abinit_bands):
        ax3.plot(abinit_dist, band, 'r--', linewidth=1.2, label='ABINIT' if i == 0 else None)
    for pos in label_positions:
        ax3.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(label_positions)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Energy (eV)')
    ax3.set_title('Overlay')
    ax3.set_xlim(min(dft_dist), max(dft_dist))
    ax3.set_ylim(-15, 10)
    ax3.legend(loc='upper right')

    plt.suptitle('Silicon Band Structure: DFT-Zig vs ABINIT (Fair Comparison)', fontsize=14)
    plt.tight_layout()
    plt.savefig('band_comparison_fair.png', dpi=150)
    print("Saved: band_comparison_fair.png")

    # ============== Plot 2: Difference ==============
    fig2, ax = plt.subplots(figsize=(12, 6))

    # Interpolate ABINIT to DFT-Zig k-points using numpy
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        # Use numpy interp (linear interpolation)
        abinit_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - abinit_interp) * 1000  # meV
        ax.plot(dft_dist, diff_mev, color=colors[i], linewidth=1.5, label=f'Band {i}')

    for pos in label_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Energy Difference (meV)')
    ax.set_xlabel('k-path')
    ax.set_title('DFT-Zig - ABINIT (Fair Comparison, Same Conditions)')
    ax.set_xlim(min(dft_dist), max(dft_dist))
    ax.legend(loc='upper right', ncol=2)

    plt.tight_layout()
    plt.savefig('band_difference_fair.png', dpi=150)
    print("Saved: band_difference_fair.png")

    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"DFT-Zig band gap: {dft_gap:.4f} eV", end='')
    if dft_time is not None:
        print(f"  ({dft_time:.1f}s)")
    else:
        print()
    print(f"ABINIT band gap:  {abinit_gap:.4f} eV", end='')
    if abinit_time is not None:
        print(f"  ({abinit_time:.1f}s)")
    else:
        print()
    print(f"Gap difference:   {(dft_gap - abinit_gap)*1000:.1f} meV")

def save_baseline():
    """Save ABINIT data to baseline/ directory."""
    bands, kpts = load_abinit()
    baseline_dir = 'baseline'
    os.makedirs(baseline_dir, exist_ok=True)

    with open(os.path.join(baseline_dir, 'abinit_bands.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        nbands = len(bands)
        writer.writerow([f'band{i}' for i in range(nbands)])
        for k in range(len(bands[0])):
            writer.writerow([bands[b][k] for b in range(nbands)])

    with open(os.path.join(baseline_dir, 'abinit_kpoints.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kx', 'ky', 'kz'])
        for kpt in kpts:
            writer.writerow(kpt)

    print(f"Baseline saved to {baseline_dir}/")


def load_baseline():
    """Load ABINIT baseline data from baseline/ directory."""
    baseline_dir = 'baseline'
    with open(os.path.join(baseline_dir, 'abinit_bands.csv')) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    nbands = len([k for k in rows[0].keys() if k.startswith('band')])
    bands = [[float(rows[k][f'band{b}']) for k in range(len(rows))] for b in range(nbands)]

    kpts = []
    with open(os.path.join(baseline_dir, 'abinit_kpoints.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kpts.append([float(row['kx']), float(row['ky']), float(row['kz'])])

    return bands, kpts


def run_dft_zig_only():
    """Run only DFT-Zig calculation."""
    print("=" * 50)
    print("Running DFT-Zig...")
    print("=" * 50)
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", "dft_zig_fair.toml"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("DFT-Zig failed:", result.stderr)
        sys.exit(1)
    print("DFT-Zig done.\n")


def calculate_mse(dft_bands, abinit_bands, dft_dist, abinit_dist):
    """Calculate MSE between DFT-Zig and ABINIT bands (in meV^2)."""
    mse_per_band = []
    for i in range(len(dft_bands)):
        # Interpolate ABINIT to DFT-Zig k-points
        abinit_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - abinit_interp) * 1000  # meV
        mse = np.mean(diff_mev ** 2)
        mse_per_band.append(mse)
    return mse_per_band, np.mean(mse_per_band)


def run_regression_check():
    """Run regression check and return exit code."""
    # Per-band MSE thresholds (in meV^2)
    # Based on current values with ~2x margin
    MSE_THRESHOLDS = {
        0: 1000,   # Valence band (current: 518)
        1: 1000,   # Valence band (current: 347)
        2: 1000,   # Valence band (current: 459)
        3: 1000,   # VBM (current: 419)
        4: 2000,   # CBM - important for gap (current: 853)
        5: 3000,   # Conduction band (current: 1324)
        6: 4000,   # High energy (current: 1564)
        7: 5000,   # High energy (current: 2376)
    }
    GAP_DIFF_THRESHOLD = 50  # meV - stricter threshold for band gap

    # Load data
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands, abinit_kpts = load_baseline()

    # Align VBM to 0 for gap calculation
    dft_vbm = max(dft_bands[3])
    dft_bands_aligned = [[e - dft_vbm for e in band] for band in dft_bands]

    abinit_vbm = max(abinit_bands[3])
    abinit_bands_aligned = [[e - abinit_vbm for e in band] for band in abinit_bands]

    # Use same distance for both
    abinit_dist = dft_dist if len(abinit_bands[0]) == len(dft_dist) else calculate_kpath_distances(abinit_kpts)

    # Calculate MSE
    mse_per_band, mse_total = calculate_mse(dft_bands_aligned, abinit_bands_aligned, dft_dist, abinit_dist)

    # Calculate band gaps
    dft_cbm = min([min(b) for b in dft_bands_aligned[4:]])
    dft_gap = dft_cbm  # VBM is at 0
    abinit_cbm = min([min(b) for b in abinit_bands_aligned[4:]])
    abinit_gap = abinit_cbm  # VBM is at 0
    gap_diff = abs(dft_gap - abinit_gap) * 1000  # meV

    # Report
    print("\n" + "=" * 50)
    print("Regression Check Results (Silicon)")
    print("=" * 50)
    print(f"MSE per band (meV^2):")
    passed = True
    for i, mse in enumerate(mse_per_band):
        threshold = MSE_THRESHOLDS.get(i, 5000)
        status = "OK" if mse <= threshold else "FAIL"
        print(f"  Band {i}: {mse:.1f} (threshold: {threshold}) [{status}]")
        if mse > threshold:
            passed = False

    print(f"Average MSE: {mse_total:.1f} meV^2")
    print(f"Band gap difference: {gap_diff:.1f} meV (threshold: {GAP_DIFF_THRESHOLD})")

    if gap_diff > GAP_DIFF_THRESHOLD:
        print(f"FAIL: Gap difference {gap_diff:.1f} meV exceeds threshold {GAP_DIFF_THRESHOLD}")
        passed = False

    if passed:
        print("PASS: All checks passed!")
        return 0
    else:
        print("FAIL: Some checks failed!")
        return 1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    if "--run" in sys.argv:
        run_calculations()
    elif "--run-check" in sys.argv:
        run_dft_zig_only()
        sys.exit(run_regression_check())

    if "--save-baseline" in sys.argv:
        save_baseline()
    elif "--check" not in sys.argv:
        plot_comparison()

    if "--check" in sys.argv:
        sys.exit(run_regression_check())
