#!/usr/bin/env python3
"""Compare Fe BCC spin-polarized band structures: DFT-Zig vs ABINIT."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
import sys

Ry_to_eV = 13.6057
Ha_to_eV = 27.2114

NBANDS = 20
NELEC = 16


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


def load_dft_zig_magnetization():
    """Load DFT-Zig magnetization from status.txt."""
    status_file = 'out_dft_zig/status.txt'
    if not os.path.exists(status_file):
        return None
    with open(status_file) as f:
        for line in f:
            if line.startswith('magnetization'):
                return float(line.split('=')[1].strip())
    return None


def load_dft_zig_fermi():
    """Load DFT-Zig Fermi energy from status.txt."""
    status_file = 'out_dft_zig/status.txt'
    if not os.path.exists(status_file):
        return None
    with open(status_file) as f:
        for line in f:
            if line.startswith('scf_fermi_level_ry'):
                return float(line.split('=')[1].strip()) * Ry_to_eV
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
    print("DFT-Zig stderr:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    print("DFT-Zig done.\n")

    print("=" * 50)
    print("Running ABINIT SCF...")
    print("=" * 50)
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
    if os.path.exists("abinit_bandi_DEN"):
        os.remove("abinit_bandi_DEN")
    os.symlink("abinit_scfo_DEN", "abinit_bandi_DEN")
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


def load_dft_zig_spin(spin='up'):
    """Load DFT-Zig spin-polarized band data."""
    filename = f'out_dft_zig/band_energies_{spin}.csv'
    dft_data = []
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dft_data.append(row)

    dft_dist = [float(row['dist']) for row in dft_data]
    dft_nbands = len([k for k in dft_data[0].keys() if k.startswith('band')])
    dft_bands = [[float(row[f'band{i}']) * Ry_to_eV for row in dft_data] for i in range(dft_nbands)]

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


def load_abinit_spin():
    """Load ABINIT spin-polarized band data from EIG file.

    Returns bands_up, bands_down, kpts.
    For nsppol=2, ABINIT EIG file has:
      "Eigenvalues (hartree) for nkpt= N  k points, SPIN UP:"
      kpt# 1 ... eigenvalues ...
      ...
      "Eigenvalues (hartree) for nkpt= N  k points, SPIN DOWN:"
      kpt# 1 ... eigenvalues ...
    """
    eig_file = 'abinit_bando_EIG'
    bands_up = [[] for _ in range(NBANDS)]
    bands_down = [[] for _ in range(NBANDS)]
    kpts = []

    with open(eig_file) as f:
        lines = f.readlines()

    # Determine spin sections
    current_spin = 'up'
    all_kpts_up = []
    all_kpts_down = []

    for i, line in enumerate(lines):
        if 'SPIN UP' in line:
            current_spin = 'up'
            continue
        if 'SPIN DOWN' in line:
            current_spin = 'down'
            continue

        if 'kpt#' in line:
            parts = line.split('kpt=')
            if len(parts) > 1:
                # Extract k-point coordinates (before "(reduced coord)")
                kpt_str = parts[1].strip().split('(')[0].strip().split()
                kpt = [float(x) for x in kpt_str[:3]]

            eig_text = ''
            for j in range(1, 30):
                if i+j < len(lines) and 'kpt#' not in lines[i+j] and 'Eigenvalues' not in lines[i+j] and lines[i+j].strip():
                    eig_text += lines[i+j]
                else:
                    break
            eigs = [float(x) * Ha_to_eV for x in eig_text.split()]

            if current_spin == 'up':
                all_kpts_up.append(kpt)
                for b, e in enumerate(eigs[:NBANDS]):
                    bands_up[b].append(e)
            else:
                all_kpts_down.append(kpt)
                for b, e in enumerate(eigs[:NBANDS]):
                    bands_down[b].append(e)

    kpts = all_kpts_up
    return bands_up, bands_down, kpts


def load_abinit_fermi():
    """Load ABINIT Fermi energy from SCF .abo file."""
    abo_file = 'abinit_scf.abo'
    if not os.path.exists(abo_file):
        return None
    with open(abo_file) as f:
        content = f.read()
    # Try "Fermi (or HOMO) energy (hartree) =   0.25031"
    match = re.search(r'Fermi.*energy.*?=\s*(-?[\d.]+)', content)
    if match:
        return float(match.group(1)) * Ha_to_eV
    # Try "fermie : 2.503E-01"
    match = re.search(r'fermie\s*[:=]\s*(-?[\d.eE+-]+)', content)
    if match:
        return float(match.group(1)) * Ha_to_eV
    return None


def load_abinit_magnetization():
    """Load ABINIT magnetization from SCF .abo file."""
    abo_file = 'abinit_scf.abo'
    if not os.path.exists(abo_file):
        return None
    with open(abo_file) as f:
        content = f.read()
    match = re.search(r'Total magnetization \(exact up - dn\):\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    return None


def calculate_kpath_distances(kpts):
    """Calculate cumulative distances along k-path using reciprocal lattice vectors."""
    # Fe BCC primitive cell: a/2 * [(-1,1,1), (1,-1,1), (1,1,-1)]
    # a = 2.87 Angstrom, a/2 = 1.435 Angstrom
    a = 1.435
    a1 = np.array([-a, a, a])
    a2 = np.array([a, -a, a])
    a3 = np.array([a, a, -a])

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
    """Create comparison plots for spin-polarized bands."""
    dft_dist_up, dft_bands_up, labels, label_positions = load_dft_zig_spin('up')
    dft_dist_down, dft_bands_down, _, _ = load_dft_zig_spin('down')
    abinit_bands_up, abinit_bands_down, abinit_kpts = load_abinit_spin()

    dft_time = load_dft_zig_timing()
    dft_mag = load_dft_zig_magnetization()
    abinit_mag = load_abinit_magnetization()

    # Get Fermi energies for alignment
    dft_ef = load_dft_zig_fermi()
    abinit_ef = load_abinit_fermi()

    if dft_ef is None:
        dft_ef = 0.0
    if abinit_ef is None:
        abinit_ef = 0.0

    # Shift to Fermi level
    dft_bands_up = [[e - dft_ef for e in band] for band in dft_bands_up]
    dft_bands_down = [[e - dft_ef for e in band] for band in dft_bands_down]
    abinit_bands_up = [[e - abinit_ef for e in band] for band in abinit_bands_up]
    abinit_bands_down = [[e - abinit_ef for e in band] for band in abinit_bands_down]

    nbands = min(len(dft_bands_up), len(abinit_bands_up))

    if len(abinit_bands_up[0]) == len(dft_dist_up):
        abinit_dist = dft_dist_up
    else:
        abinit_dist = calculate_kpath_distances(abinit_kpts)

    # ============== Plot: Spin up comparison ==============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for spin_idx, (spin_label, dft_bands, ab_bands, dft_dist) in enumerate([
        ('Spin Up', dft_bands_up, abinit_bands_up, dft_dist_up),
        ('Spin Down', dft_bands_down, abinit_bands_down, dft_dist_down),
    ]):
        ax1, ax2, ax3 = axes[spin_idx]

        # DFT-Zig
        for i in range(nbands):
            ax1.plot(dft_dist, dft_bands[i], 'b-', linewidth=1.0)
        for pos in label_positions:
            ax1.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        ax1.axhline(y=0, color='green', linestyle='-', linewidth=0.8, label='E_F')
        ax1.set_xticks(label_positions)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'DFT-Zig {spin_label}')
        ax1.set_xlim(min(dft_dist), max(dft_dist))
        ax1.set_ylim(-15, 10)

        # ABINIT
        for i in range(nbands):
            ax2.plot(abinit_dist, ab_bands[i], 'r-', linewidth=1.0)
        for pos in label_positions:
            ax2.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        ax2.axhline(y=0, color='green', linestyle='-', linewidth=0.8, label='E_F')
        ax2.set_xticks(label_positions)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Energy (eV)')
        ax2.set_title(f'ABINIT {spin_label}')
        ax2.set_xlim(min(dft_dist), max(dft_dist))
        ax2.set_ylim(-15, 10)

        # Overlay
        for i in range(nbands):
            ax3.plot(dft_dist, dft_bands[i], 'b-', linewidth=1.5, label='DFT-Zig' if i == 0 else None)
        for i in range(nbands):
            ax3.plot(abinit_dist, ab_bands[i], 'r--', linewidth=1.0, label='ABINIT' if i == 0 else None)
        for pos in label_positions:
            ax3.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        ax3.axhline(y=0, color='green', linestyle='-', linewidth=0.8)
        ax3.set_xticks(label_positions)
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Energy (eV)')
        ax3.set_title(f'Overlay {spin_label}')
        ax3.set_xlim(min(dft_dist), max(dft_dist))
        ax3.set_ylim(-15, 10)
        ax3.legend(loc='upper right', fontsize=8)

    mag_str = ''
    if dft_mag is not None:
        mag_str += f'  DFT-Zig M={dft_mag:.3f} μ_B'
    if abinit_mag is not None:
        mag_str += f'  ABINIT M={abinit_mag:.3f} μ_B'
    plt.suptitle(f'Fe BCC Spin-Polarized Band Structure: DFT-Zig vs ABINIT{mag_str}', fontsize=13)
    plt.tight_layout()
    plt.savefig('band_comparison.png', dpi=150)
    print("Saved: band_comparison.png")

    # ============== Plot: Band difference ==============
    fig2, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(12, 8))

    colors = plt.cm.tab20(np.linspace(0, 1, nbands))
    for spin_idx, (ax, spin_label, dft_bands, ab_bands, dft_dist) in enumerate([
        (ax_up, 'Spin Up', dft_bands_up, abinit_bands_up, dft_dist_up),
        (ax_down, 'Spin Down', dft_bands_down, abinit_bands_down, dft_dist_down),
    ]):
        for i in range(nbands):
            ab_interp = np.interp(dft_dist, abinit_dist, ab_bands[i])
            diff_mev = (np.array(dft_bands[i]) - ab_interp) * 1000
            ax.plot(dft_dist, diff_mev, color=colors[i], linewidth=1.0, label=f'Band {i}')
        for pos in label_positions:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('ΔE (meV)')
        ax.set_title(f'DFT-Zig - ABINIT ({spin_label})')
        ax.set_xlim(min(dft_dist), max(dft_dist))
        ax.legend(loc='upper right', ncol=5, fontsize=6)

    plt.tight_layout()
    plt.savefig('band_difference.png', dpi=150)
    print("Saved: band_difference.png")

    # Print summary
    print("\n" + "=" * 50)
    print("Summary (Fe BCC - spin-polarized)")
    print("=" * 50)
    if dft_ef:
        print(f"DFT-Zig E_F: {dft_ef:.4f} eV")
    if abinit_ef:
        print(f"ABINIT E_F:  {abinit_ef:.4f} eV")
    if dft_mag is not None:
        print(f"DFT-Zig magnetization: {dft_mag:.4f} μ_B")
    if abinit_mag is not None:
        print(f"ABINIT magnetization:  {abinit_mag:.4f} μ_B")
    if dft_time is not None:
        print(f"DFT-Zig time: {dft_time:.1f}s")

    # MSE per spin channel
    for spin_label, dft_bands, ab_bands, dft_dist in [
        ('Spin Up', dft_bands_up, abinit_bands_up, dft_dist_up),
        ('Spin Down', dft_bands_down, abinit_bands_down, dft_dist_down),
    ]:
        print(f"\nMSE per band ({spin_label}, meV^2):")
        total_mse = 0
        for i in range(nbands):
            ab_interp = np.interp(dft_dist, abinit_dist, ab_bands[i])
            diff_mev = (np.array(dft_bands[i]) - ab_interp) * 1000
            mse = np.mean(diff_mev ** 2)
            total_mse += mse
            print(f"  Band {i}: {mse:.1f}")
        print(f"  Average: {total_mse/nbands:.1f}")


def save_baseline():
    """Save ABINIT data to baseline/ directory."""
    bands_up, bands_down, kpts = load_abinit_spin()
    fermi_eV = load_abinit_fermi()
    magnetization = load_abinit_magnetization()
    baseline_dir = 'baseline'
    os.makedirs(baseline_dir, exist_ok=True)

    for spin, bands in [('up', bands_up), ('down', bands_down)]:
        with open(os.path.join(baseline_dir, f'abinit_bands_{spin}.csv'), 'w', newline='') as f:
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

    with open(os.path.join(baseline_dir, 'abinit_info.txt'), 'w') as f:
        if fermi_eV is not None:
            f.write(f'fermi_eV={fermi_eV}\n')
        if magnetization is not None:
            f.write(f'magnetization={magnetization}\n')

    print(f"Baseline saved to {baseline_dir}/")


def load_baseline():
    """Load ABINIT baseline data from baseline/ directory.

    Returns (bands_up, bands_down, kpts, fermi_eV, magnetization).
    Bands are in eV (already converted from Hartree when saved).
    """
    baseline_dir = 'baseline'

    bands = {}
    for spin in ('up', 'down'):
        filepath = os.path.join(baseline_dir, f'abinit_bands_{spin}.csv')
        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        nbands = len([k for k in rows[0].keys() if k.startswith('band')])
        bands[spin] = [[float(rows[k][f'band{b}']) for k in range(len(rows))] for b in range(nbands)]

    kpts = []
    with open(os.path.join(baseline_dir, 'abinit_kpoints.csv')) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kpts.append([float(row['kx']), float(row['ky']), float(row['kz'])])

    fermi_eV = None
    magnetization = None
    with open(os.path.join(baseline_dir, 'abinit_info.txt')) as f:
        for line in f:
            if line.startswith('fermi_eV='):
                fermi_eV = float(line.split('=')[1].strip())
            elif line.startswith('magnetization='):
                magnetization = float(line.split('=')[1].strip())

    return bands['up'], bands['down'], kpts, fermi_eV, magnetization


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
    print("DFT-Zig stderr:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    print("DFT-Zig done.\n")


def calculate_mse_spin(dft_bands, abinit_bands, dft_dist, abinit_dist):
    """Calculate MSE between DFT-Zig and ABINIT bands (in meV^2)."""
    mse_per_band = []
    nbands = min(len(dft_bands), len(abinit_bands))
    for i in range(nbands):
        ab_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - ab_interp) * 1000
        mse = np.mean(diff_mev ** 2)
        mse_per_band.append(mse)
    return mse_per_band


def run_regression_check():
    """Run regression check against saved ABINIT baseline and return exit code."""
    MSE_THRESHOLDS = {
        0: 5, 1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5,
        9: 5, 10: 5, 11: 5,
        12: 10, 13: 10, 14: 10, 15: 10,
        16: 10, 17: 10,
        18: 500,
        19: 8000,
    }

    MAG_THRESHOLD = 0.1  # μ_B

    # Load DFT-Zig data
    dft_dist_up, dft_bands_up, labels, label_positions = load_dft_zig_spin('up')
    dft_dist_down, dft_bands_down, _, _ = load_dft_zig_spin('down')

    # Load ABINIT baseline
    abinit_bands_up, abinit_bands_down, abinit_kpts, abinit_ef, abinit_mag = load_baseline()

    # Get Fermi energies for alignment
    dft_ef = load_dft_zig_fermi()
    if dft_ef is None:
        dft_ef = 0.0
    if abinit_ef is None:
        abinit_ef = 0.0

    # Shift to Fermi level
    dft_bands_up_aligned = [[e - dft_ef for e in band] for band in dft_bands_up]
    dft_bands_down_aligned = [[e - dft_ef for e in band] for band in dft_bands_down]
    abinit_bands_up_aligned = [[e - abinit_ef for e in band] for band in abinit_bands_up]
    abinit_bands_down_aligned = [[e - abinit_ef for e in band] for band in abinit_bands_down]

    if len(abinit_bands_up[0]) == len(dft_dist_up):
        abinit_dist = dft_dist_up
    else:
        abinit_dist = calculate_kpath_distances(abinit_kpts)

    # Calculate MSE per spin channel
    mse_up = calculate_mse_spin(dft_bands_up_aligned, abinit_bands_up_aligned, dft_dist_up, abinit_dist)
    mse_down = calculate_mse_spin(dft_bands_down_aligned, abinit_bands_down_aligned, dft_dist_down, abinit_dist)

    # Report
    print("\n" + "=" * 50)
    print("Regression Check Results (Fe BCC - spin-polarized)")
    print("=" * 50)

    passed = True

    for spin_label, mse_list in [('Spin Up', mse_up), ('Spin Down', mse_down)]:
        print(f"\nMSE per band ({spin_label}, meV^2):")
        for i, mse in enumerate(mse_list):
            threshold = MSE_THRESHOLDS.get(i, 8000)
            status = "OK" if mse <= threshold else "FAIL"
            print(f"  Band {i}: {mse:.1f} (threshold: {threshold}) [{status}]")
            if mse > threshold:
                passed = False

    # Magnetization check
    dft_mag = load_dft_zig_magnetization()
    if dft_mag is not None and abinit_mag is not None:
        mag_diff = abs(dft_mag - abinit_mag)
        mag_status = "OK" if mag_diff < MAG_THRESHOLD else "FAIL"
        print(f"\nMagnetization: DFT-Zig={dft_mag:.4f}, baseline={abinit_mag:.4f}, diff={mag_diff:.4f} μ_B [{mag_status}]")
        if mag_diff >= MAG_THRESHOLD:
            passed = False

    if passed:
        print("\nPASS: All checks passed!")
        return 0
    else:
        print("\nFAIL: Some checks failed!")
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
