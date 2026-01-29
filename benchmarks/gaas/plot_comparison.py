#!/usr/bin/env python3
"""Compare GaAs band structures: DFT-Zig vs ABINIT."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
import sys

Ry_to_eV = 13.6057
Ha_to_eV = 27.2114

NBANDS = 14
NOCC = 9  # 18 electrons / 2 = 9 occupied bands
VBM_BAND = NOCC - 1  # 0-indexed: band 8


def load_dft_zig_timing():
    """Load DFT-Zig timing from timing.txt."""
    timing_file = 'out_dft_zig/timing.txt'
    if not os.path.exists(timing_file):
        return None
    with open(timing_file) as f:
        for line in f:
            if line.startswith('total_sec'):
                return float(line.split('=')[1].strip())
    return None


def load_abinit_timing():
    """Load ABINIT timing from .abo file."""
    abo_file = 'abinit_band.abo'
    if not os.path.exists(abo_file):
        return None
    with open(abo_file) as f:
        content = f.read()
    match = re.search(r'Overall time at end.*wall=\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    return None


def run_calculations():
    """Run both DFT-Zig and ABINIT calculations."""
    print("=" * 50)
    print("Running DFT-Zig...")
    print("=" * 50)
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", "dft_zig.toml"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("DFT-Zig stdout:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("DFT-Zig failed:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)
    print("DFT-Zig done.\n")

    print("=" * 50)
    print("Running ABINIT SCF...")
    print("=" * 50)
    # Clean previous outputs
    for f in os.listdir("."):
        if f.startswith("abinit_scfo") or f.startswith("abinit_scf_tmp"):
            try:
                os.remove(f)
            except:
                pass
    result = subprocess.run(
        ["../../external/abinit/build/abinit-10.4.7/src/98_main/abinit", "abinit_scf.in"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ABINIT SCF failed:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)
    print("ABINIT SCF done.\n")

    print("=" * 50)
    print("Running ABINIT band...")
    print("=" * 50)
    # Link SCF density for band calculation
    if os.path.exists("abinit_bandi_DEN"):
        os.remove("abinit_bandi_DEN")
    os.symlink("abinit_scfo_DEN", "abinit_bandi_DEN")
    # Clean previous band outputs
    for f in os.listdir("."):
        if f.startswith("abinit_bando") or f.startswith("abinit_band_tmp"):
            try:
                os.remove(f)
            except:
                pass
    result = subprocess.run(
        ["../../external/abinit/build/abinit-10.4.7/src/98_main/abinit", "abinit_band.in"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ABINIT band failed:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)
    print("ABINIT band done.\n")


def load_dft_zig():
    """Load DFT-Zig band data."""
    dft_data = []
    with open('out_dft_zig/band_energies.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dft_data.append(row)

    dft_dist = [float(row['dist']) for row in dft_data]
    dft_nbands = len([k for k in dft_data[0].keys() if k.startswith('band')])
    dft_bands = [[float(row[f'band{i}']) * Ry_to_eV for row in dft_data] for i in range(dft_nbands)]

    # Read k-point labels
    kpoints_data = []
    with open('out_dft_zig/band_kpoints.csv') as f:
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
    """Load ABINIT band data from EIG file."""
    abinit_bands = [[] for _ in range(NBANDS)]
    abinit_kpts = []

    eig_file = 'abinit_bando_EIG'

    with open(eig_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'kpt#' in line:
            parts = line.split('kpt=')
            if len(parts) > 1:
                kpt_str = parts[1].strip().split()
                kpt = [float(x) for x in kpt_str[:3]]
                abinit_kpts.append(kpt)

            # Parse eigenvalues on following lines
            eig_text = ''
            for j in range(1, 20):
                if i+j < len(lines) and 'kpt#' not in lines[i+j] and lines[i+j].strip():
                    eig_text += lines[i+j]
                else:
                    break
            eigs = [float(x) * Ha_to_eV for x in eig_text.split()]
            for b, e in enumerate(eigs[:NBANDS]):
                abinit_bands[b].append(e)

    return abinit_bands, abinit_kpts


def calculate_kpath_distances(kpts):
    """Calculate cumulative distances along k-path using reciprocal lattice vectors."""
    # GaAs FCC primitive cell: a/2 * [(0,1,1), (1,0,1), (1,1,0)]
    # a = 5.653 Angstrom, a/2 = 2.8265 Angstrom
    a = 2.8265
    a1 = np.array([0.0, a, a])
    a2 = np.array([a, 0.0, a])
    a3 = np.array([a, a, 0.0])

    vol = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi / vol * np.cross(a2, a3)
    b2 = 2 * np.pi / vol * np.cross(a3, a1)
    b3 = 2 * np.pi / vol * np.cross(a1, a2)

    distances = [0.0]
    for i in range(1, len(kpts)):
        k_prev = kpts[i-1][0]*b1 + kpts[i-1][1]*b2 + kpts[i-1][2]*b3
        k_curr = kpts[i][0]*b1 + kpts[i][1]*b2 + kpts[i][2]*b3
        dk = np.linalg.norm(k_curr - k_prev)
        distances.append(distances[-1] + dk)

    return distances


def plot_comparison():
    """Create comparison plots."""
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands, abinit_kpts = load_abinit()

    dft_time = load_dft_zig_timing()
    abinit_time = load_abinit_timing()

    nbands = min(len(dft_bands), len(abinit_bands))

    # Align VBM to 0
    dft_vbm = max(dft_bands[VBM_BAND])
    for band in dft_bands:
        for i in range(len(band)):
            band[i] -= dft_vbm

    abinit_vbm = max(abinit_bands[VBM_BAND])
    for band in abinit_bands:
        for i in range(len(band)):
            band[i] -= abinit_vbm

    if len(abinit_bands[0]) == len(dft_dist):
        abinit_dist = dft_dist
        print(f"Using DFT-Zig k-path distances for ABINIT ({len(abinit_dist)} points)")
    else:
        print(f"Warning: k-point count mismatch: DFT-Zig={len(dft_dist)}, ABINIT={len(abinit_bands[0])}")
        abinit_dist = calculate_kpath_distances(abinit_kpts)

    # Calculate band gaps
    dft_cbm = min([min(b) for b in dft_bands[NOCC:nbands]])
    dft_gap = dft_cbm
    abinit_cbm = min([min(b) for b in abinit_bands[NOCC:nbands]])
    abinit_gap = abinit_cbm

    # ============== Plot 1: Side by side ==============
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    # DFT-Zig
    for i in range(nbands):
        color = 'blue' if i < NOCC else 'red'
        ax1.plot(dft_dist, dft_bands[i], color=color, linewidth=1.5)
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
    for i in range(nbands):
        color = 'blue' if i < NOCC else 'red'
        ax2.plot(abinit_dist, abinit_bands[i], color=color, linewidth=1.5)
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
    for i in range(nbands):
        ax3.plot(dft_dist, dft_bands[i], 'b-', linewidth=1.5, label='DFT-Zig' if i == 0 else None)
    for i in range(nbands):
        ax3.plot(abinit_dist, abinit_bands[i], 'r--', linewidth=1.2, label='ABINIT' if i == 0 else None)
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

    plt.suptitle('GaAs Band Structure: DFT-Zig vs ABINIT', fontsize=14)
    plt.tight_layout()
    plt.savefig('band_comparison.png', dpi=150)
    print("Saved: band_comparison.png")

    # ============== Plot 2: Difference ==============
    fig2, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, nbands))
    for i in range(nbands):
        abinit_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - abinit_interp) * 1000
        ax.plot(dft_dist, diff_mev, color=colors[i], linewidth=1.5, label=f'Band {i}')

    for pos in label_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Energy Difference (meV)')
    ax.set_xlabel('k-path')
    ax.set_title('DFT-Zig - ABINIT (GaAs)')
    ax.set_xlim(min(dft_dist), max(dft_dist))
    ax.legend(loc='upper right', ncol=3, fontsize=7)

    plt.tight_layout()
    plt.savefig('band_difference.png', dpi=150)
    print("Saved: band_difference.png")

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
        ["../../zig-out/bin/dft_zig", "dft_zig.toml"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("DFT-Zig stdout:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("DFT-Zig failed:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)
    print("DFT-Zig done.\n")


def calculate_mse(dft_bands, abinit_bands, dft_dist, abinit_dist):
    """Calculate MSE between DFT-Zig and ABINIT bands (in meV^2)."""
    mse_per_band = []
    nbands = min(len(dft_bands), len(abinit_bands))
    for i in range(nbands):
        abinit_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - abinit_interp) * 1000
        mse = np.mean(diff_mev ** 2)
        mse_per_band.append(mse)
    return mse_per_band, np.mean(mse_per_band)


def run_regression_check():
    """Run regression check and return exit code."""
    # Per-band MSE thresholds (in meV^2)
    # Based on measured values with ~2x margin
    MSE_THRESHOLDS = {
        0: 7000,    # Ga 3d (current: 3134)
        1: 6000,    # Ga 3d (current: 2899)
        2: 4000,    # Ga 3d (current: 1792)
        3: 20000,   # Ga 3d (current: 9570)
        4: 14000,   # Ga 3d (current: 6564)
        5: 1000,    # Valence (current: 28)
        6: 1500,    # Valence (current: 546)
        7: 1000,    # Valence (current: 47)
        8: 1000,    # VBM (current: 42)
        9: 1000,    # CBM (current: 359)
        10: 1000,   # Conduction (current: 311)
        11: 1000,   # Conduction (current: 386)
        12: 1000,   # Conduction (current: 324)
        13: 1000,   # Conduction (current: 176)
    }
    GAP_DIFF_THRESHOLD = 50  # meV

    # Load data
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands, abinit_kpts = load_baseline()

    nbands = min(len(dft_bands), len(abinit_bands))

    # Align VBM to 0 for gap calculation
    dft_vbm = max(dft_bands[VBM_BAND])
    dft_bands_aligned = [[e - dft_vbm for e in band] for band in dft_bands]

    abinit_vbm = max(abinit_bands[VBM_BAND])
    abinit_bands_aligned = [[e - abinit_vbm for e in band] for band in abinit_bands]

    abinit_dist = dft_dist if len(abinit_bands[0]) == len(dft_dist) else calculate_kpath_distances(abinit_kpts)

    # Calculate MSE
    mse_per_band, mse_total = calculate_mse(dft_bands_aligned, abinit_bands_aligned, dft_dist, abinit_dist)

    # Calculate band gaps
    dft_cbm = min([min(b) for b in dft_bands_aligned[NOCC:nbands]])
    dft_gap = dft_cbm
    abinit_cbm = min([min(b) for b in abinit_bands_aligned[NOCC:nbands]])
    abinit_gap = abinit_cbm
    gap_diff = abs(dft_gap - abinit_gap) * 1000  # meV

    # Report
    print("\n" + "=" * 50)
    print("Regression Check Results (GaAs)")
    print("=" * 50)
    print(f"MSE per band (meV^2):")
    passed = True
    for i, mse in enumerate(mse_per_band):
        threshold = MSE_THRESHOLDS.get(i, 10000)
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
