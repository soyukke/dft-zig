#!/usr/bin/env python3
"""Compare graphene band structures: DFT-Zig vs ABINIT (fair comparison)."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import time

Ry_to_eV = 13.6057
Ha_to_eV = 27.2114

# Global timing variables
dft_wall_time = None
abinit_wall_time = None


def run_calculations():
    """Run both DFT-Zig and ABINIT calculations."""
    global dft_wall_time, abinit_wall_time

    print("=" * 50)
    print("Building DFT-Zig (ReleaseFast)...")
    print("=" * 50)
    print("Running DFT-Zig...")
    print("=" * 50)
    start = time.time()
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", "test15.toml"],
        capture_output=True, text=True
    )
    dft_wall_time = time.time() - start
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
        if f.startswith("graphene_fairo") or (f.endswith(".nc") and "fair" in f):
            try:
                os.remove(f)
            except:
                pass
    start = time.time()
    result = subprocess.run(
        ["../../../external/abinit/build/abinit-10.4.7/src/98_main/abinit", "graphene_fair.in"],
        capture_output=True, text=True
    )
    abinit_wall_time = time.time() - start
    os.chdir("..")
    if result.returncode != 0:
        print("ABINIT failed:", result.stderr)
        print("ABINIT stdout:", result.stdout)
        sys.exit(1)
    print("ABINIT done.\n")


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
    """Load ABINIT band data from fair comparison."""
    abinit_bands = [[] for _ in range(8)]

    # Use the fair comparison output
    eig_file = 'abinit/graphene_fairo_DS2_EIG'

    with open(eig_file) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'kpt#' in line:
            # Parse eigenvalues on following lines
            eig_text = ''
            for j in range(1, 10):
                if i + j < len(lines) and 'kpt#' not in lines[i + j] and lines[i + j].strip():
                    eig_text += lines[i + j]
                else:
                    break
            eigs = [float(x) * Ha_to_eV for x in eig_text.split()]
            for b, e in enumerate(eigs[:8]):
                abinit_bands[b].append(e)

    return abinit_bands


def plot_comparison():
    """Create comparison plots."""
    # Load data
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands = load_abinit()

    # Find Fermi level (average of highest occupied at K point - index 20)
    # For graphene, bands 3 and 4 cross at K point (Dirac point)
    dft_fermi = (dft_bands[3][20] + dft_bands[4][20]) / 2
    for band in dft_bands:
        for i in range(len(band)):
            band[i] -= dft_fermi

    abinit_fermi = (abinit_bands[3][20] + abinit_bands[4][20]) / 2
    for band in abinit_bands:
        for i in range(len(band)):
            band[i] -= abinit_fermi

    # Use same distance array (same k-points in same order)
    nkpts_abinit = len(abinit_bands[0])
    if nkpts_abinit == len(dft_dist):
        abinit_dist = dft_dist  # Same k-points, same distances
    else:
        # Fallback if different number of k-points
        abinit_dist = np.linspace(dft_dist[0], dft_dist[-1], nkpts_abinit)

    # Calculate Dirac cone gap at K point (index 20)
    dft_gap = abs(dft_bands[4][20] - dft_bands[3][20])
    abinit_gap = abs(abinit_bands[4][20] - abinit_bands[3][20])

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
    ax1.set_title(f'DFT-Zig (Dirac gap: {dft_gap*1000:.1f} meV)')
    ax1.set_xlim(min(dft_dist), max(dft_dist))
    ax1.set_ylim(-20, 15)

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
    ax2.set_title(f'ABINIT (Dirac gap: {abinit_gap*1000:.1f} meV)')
    ax2.set_xlim(min(dft_dist), max(dft_dist))
    ax2.set_ylim(-20, 15)

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
    ax3.set_ylim(-20, 15)
    ax3.legend(loc='upper right')

    timing_str = ""
    if dft_wall_time is not None and abinit_wall_time is not None:
        timing_str = f"\nTiming: DFT-Zig {dft_wall_time:.1f}s, ABINIT {abinit_wall_time:.1f}s"
    plt.suptitle(f'Graphene Band Structure: DFT-Zig vs ABINIT (Fair Comparison){timing_str}', fontsize=14)
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
    title = 'DFT-Zig - ABINIT (Fair Comparison, Same Conditions)'
    if dft_wall_time is not None and abinit_wall_time is not None:
        title += f'\nTiming: DFT-Zig {dft_wall_time:.1f}s, ABINIT {abinit_wall_time:.1f}s'
    ax.set_title(title)
    ax.set_xlim(min(dft_dist), max(dft_dist))
    ax.legend(loc='upper right', ncol=2)

    plt.tight_layout()
    plt.savefig('band_difference_fair.png', dpi=150)
    print("Saved: band_difference_fair.png")

    # ============== Plot 3: Zoom on Dirac point ==============
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Find K-point index range (around index 20)
    k_idx_start = 15
    k_idx_end = 26

    # DFT-Zig zoom
    for i in [3, 4]:  # Only pi bands
        color = 'blue' if i == 3 else 'red'
        ax1.plot(dft_dist[k_idx_start:k_idx_end],
                 [dft_bands[i][j] for j in range(k_idx_start, k_idx_end)],
                 color=color, linewidth=2, marker='o', markersize=4,
                 label=f'Band {i}')
    ax1.axvline(x=dft_dist[20], color='gray', linestyle='--', linewidth=1, label='K point')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('k-distance')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title(f'DFT-Zig Dirac Cone (gap: {dft_gap*1000:.1f} meV)')
    ax1.legend()
    ax1.set_ylim(-1.5, 1.5)

    # ABINIT zoom
    for i in [3, 4]:
        color = 'blue' if i == 3 else 'red'
        ax2.plot(abinit_dist[k_idx_start:k_idx_end],
                 [abinit_bands[i][j] for j in range(k_idx_start, k_idx_end)],
                 color=color, linewidth=2, marker='o', markersize=4,
                 label=f'Band {i}')
    ax2.axvline(x=abinit_dist[20], color='gray', linestyle='--', linewidth=1, label='K point')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('k-distance')
    ax2.set_ylabel('Energy (eV)')
    ax2.set_title(f'ABINIT Dirac Cone (gap: {abinit_gap*1000:.1f} meV)')
    ax2.legend()
    ax2.set_ylim(-1.5, 1.5)

    plt.suptitle('Graphene Dirac Cone Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('dirac_cone_comparison.png', dpi=150)
    print("Saved: dirac_cone_comparison.png")

    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"DFT-Zig Dirac gap: {dft_gap*1000:.2f} meV")
    print(f"ABINIT Dirac gap:  {abinit_gap*1000:.2f} meV")
    print(f"Gap difference:    {abs(dft_gap - abinit_gap)*1000:.2f} meV")


def save_baseline():
    """Save ABINIT data to baseline/ directory."""
    bands = load_abinit()
    baseline_dir = 'baseline'
    os.makedirs(baseline_dir, exist_ok=True)

    with open(os.path.join(baseline_dir, 'abinit_bands.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        nbands = len(bands)
        writer.writerow([f'band{i}' for i in range(nbands)])
        for k in range(len(bands[0])):
            writer.writerow([bands[b][k] for b in range(nbands)])

    print(f"Baseline saved to {baseline_dir}/")


def load_baseline():
    """Load ABINIT baseline data from baseline/ directory."""
    baseline_dir = 'baseline'
    with open(os.path.join(baseline_dir, 'abinit_bands.csv')) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    nbands = len([k for k in rows[0].keys() if k.startswith('band')])
    bands = [[float(rows[k][f'band{b}']) for k in range(len(rows))] for b in range(nbands)]
    return bands


def run_dft_zig_only():
    """Run only DFT-Zig calculation."""
    print("=" * 50)
    print("Running DFT-Zig...")
    print("=" * 50)
    result = subprocess.run(
        ["../../zig-out/bin/dft_zig", "test15.toml"],
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
        0: 7000,    # Deep band (current: 59)
        1: 7000,    # Deep band (current: 76)
        2: 12000,   # Deep band (current: 145)
        3: 5000,    # pi band - VBM (current: 78)
        4: 3000,    # pi* band - Dirac cone (current: 1319)
        5: 7000,    # (current: 3442)
        6: 20000,   # High energy (current: 2056)
        7: 17000,   # High energy (current: 4730)
    }
    DIRAC_GAP_THRESHOLD = 10  # meV - stricter, should be ~0

    # Load data
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands = load_baseline()

    # Find Fermi level (average at K point - index 20)
    dft_fermi = (dft_bands[3][20] + dft_bands[4][20]) / 2
    dft_bands_aligned = [[e - dft_fermi for e in band] for band in dft_bands]

    abinit_fermi = (abinit_bands[3][20] + abinit_bands[4][20]) / 2
    abinit_bands_aligned = [[e - abinit_fermi for e in band] for band in abinit_bands]

    # Use same distance for both
    nkpts_abinit = len(abinit_bands[0])
    if nkpts_abinit == len(dft_dist):
        abinit_dist = dft_dist
    else:
        abinit_dist = np.linspace(dft_dist[0], dft_dist[-1], nkpts_abinit)

    # Calculate MSE
    mse_per_band, mse_total = calculate_mse(dft_bands_aligned, abinit_bands_aligned, dft_dist, abinit_dist)

    # Calculate Dirac gap at K point (index 20)
    dft_dirac_gap = abs(dft_bands_aligned[4][20] - dft_bands_aligned[3][20]) * 1000  # meV
    abinit_dirac_gap = abs(abinit_bands_aligned[4][20] - abinit_bands_aligned[3][20]) * 1000  # meV

    # Report
    print("\n" + "=" * 50)
    print("Regression Check Results (Graphene)")
    print("=" * 50)
    print(f"MSE per band (meV^2):")
    passed = True
    for i, mse in enumerate(mse_per_band):
        threshold = MSE_THRESHOLDS.get(i, 20000)
        status = "OK" if mse <= threshold else "FAIL"
        print(f"  Band {i}: {mse:.1f} (threshold: {threshold}) [{status}]")
        if mse > threshold:
            passed = False

    print(f"Average MSE: {mse_total:.1f} meV^2")
    print(f"DFT-Zig Dirac gap: {dft_dirac_gap:.2f} meV (threshold: {DIRAC_GAP_THRESHOLD})")
    print(f"ABINIT Dirac gap: {abinit_dirac_gap:.2f} meV")

    if dft_dirac_gap > DIRAC_GAP_THRESHOLD:
        print(f"FAIL: Dirac gap {dft_dirac_gap:.2f} meV exceeds threshold {DIRAC_GAP_THRESHOLD}")
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
