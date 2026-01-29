#!/usr/bin/env python3
"""Compare Al 11e band structures: DFT-Zig vs ABINIT."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import subprocess
import sys

Ry_to_eV = 13.6057
Ha_to_eV = 27.2114

DFT_OUT_DIR = "out_dft_zig_11e"
ABINIT_EIG_FILE = "abinit/aluminum_11eo_DS2_EIG"
ABINIT_ABO_FILE = "abinit/aluminum_11e.abo"


def load_dft_zig_timing():
    timing_file = f"{DFT_OUT_DIR}/timing.txt"
    if not os.path.exists(timing_file):
        return None
    with open(timing_file) as f:
        for line in f:
            if line.startswith("total_sec"):
                return float(line.split("=")[1].strip())
    return None


def load_abinit_timing():
    if not os.path.exists(ABINIT_ABO_FILE):
        return None
    with open(ABINIT_ABO_FILE) as f:
        content = f.read()
    match = re.search(r"Overall time at end.*wall=\s*([\d.]+)", content)
    if match:
        return float(match.group(1))
    return None


def load_dft_zig_fermi():
    status_file = f"{DFT_OUT_DIR}/status.txt"
    if not os.path.exists(status_file):
        return None
    with open(status_file) as f:
        for line in f:
            if line.startswith("scf_fermi_level_ry"):
                return float(line.split("=")[1].strip()) * Ry_to_eV
    return None


def load_abinit_fermi():
    if not os.path.exists(ABINIT_ABO_FILE):
        return None
    with open(ABINIT_ABO_FILE) as f:
        content = f.read()
    match = re.search(r"fermie\s*:\s*([\d.E+-]+)", content)
    if not match:
        match = re.search(r"Fermi \(or HOMO\) energy \(hartree\) =\s*([\d.E+-]+)", content)
    if match:
        return float(match.group(1)) * Ha_to_eV
    return None


def load_dft_zig():
    dft_data = []
    with open(f"{DFT_OUT_DIR}/band_energies.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dft_data.append(row)

    dft_dist = [float(row["dist"]) for row in dft_data]
    dft_nbands = len([k for k in dft_data[0].keys() if k.startswith("band")])
    dft_bands = [
        [float(row[f"band{i}"]) * Ry_to_eV for row in dft_data]
        for i in range(dft_nbands)
    ]

    kpoints_data = []
    with open(f"{DFT_OUT_DIR}/band_kpoints.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kpoints_data.append(row)

    labels = []
    label_positions = []
    for row in kpoints_data:
        if row.get("label", ""):
            labels.append(row["label"])
            label_positions.append(float(row["dist"]))

    return dft_dist, dft_bands, labels, label_positions


def load_abinit():
    abinit_bands = [[] for _ in range(8)]
    abinit_kpts = []

    with open(ABINIT_EIG_FILE) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "kpt#" in line:
            parts = line.split("kpt=")
            if len(parts) > 1:
                kpt_str = parts[1].strip().split()
                kpt = [float(x) for x in kpt_str[:3]]
                abinit_kpts.append(kpt)

            eig_text = ""
            for j in range(1, 10):
                if i + j < len(lines) and "kpt#" not in lines[i + j] and lines[i + j].strip():
                    eig_text += lines[i + j]
                else:
                    break
            eigs = [float(x) * Ha_to_eV for x in eig_text.split()]
            for b, e in enumerate(eigs[:8]):
                abinit_bands[b].append(e)

    return abinit_bands, abinit_kpts


def calculate_kpath_distances(kpts):
    a = 2.025
    a1 = np.array([0.0, a, a])
    a2 = np.array([a, 0.0, a])
    a3 = np.array([a, a, 0.0])

    vol = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi / vol * np.cross(a2, a3)
    b2 = 2 * np.pi / vol * np.cross(a3, a1)
    b3 = 2 * np.pi / vol * np.cross(a1, a2)

    distances = [0.0]
    for i in range(1, len(kpts)):
        k_prev = kpts[i - 1][0] * b1 + kpts[i - 1][1] * b2 + kpts[i - 1][2] * b3
        k_curr = kpts[i][0] * b1 + kpts[i][1] * b2 + kpts[i][2] * b3
        dk = float(np.linalg.norm(k_curr - k_prev))
        distances.append(distances[-1] + dk)

    return distances


def plot_comparison():
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands, abinit_kpts = load_abinit()

    dft_time = load_dft_zig_timing()
    abinit_time = load_abinit_timing()

    dft_fermi = load_dft_zig_fermi()
    abinit_fermi = load_abinit_fermi()

    # Align to Fermi level
    if dft_fermi is not None:
        for band in dft_bands:
            for i in range(len(band)):
                band[i] -= dft_fermi

    if abinit_fermi is not None:
        for band in abinit_bands:
            for i in range(len(band)):
                band[i] -= abinit_fermi

    if len(abinit_bands[0]) == len(dft_dist):
        abinit_dist = dft_dist
    else:
        print(f"Warning: k-point count mismatch: DFT-Zig={len(dft_dist)}, ABINIT={len(abinit_bands[0])}")
        abinit_dist = calculate_kpath_distances(abinit_kpts)

    # --- Plot 1: Band comparison ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    for i, band in enumerate(dft_bands):
        ax1.plot(dft_dist, band, color="blue", linewidth=1.5)
    for pos in label_positions:
        ax1.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Energy (eV)")
    dft_title = "DFT-Zig"
    if dft_time is not None:
        dft_title += f" ({dft_time:.1f}s)"
    ax1.set_title(dft_title)
    ax1.set_xlim(min(dft_dist), max(dft_dist))
    ax1.set_ylim(-80, 20)

    for i, band in enumerate(abinit_bands):
        ax2.plot(abinit_dist, band, color="red", linewidth=1.5)
    for pos in label_positions:
        ax2.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xticks(label_positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Energy (eV)")
    abinit_title = "ABINIT"
    if abinit_time is not None:
        abinit_title += f" ({abinit_time:.1f}s)"
    ax2.set_title(abinit_title)
    ax2.set_xlim(min(dft_dist), max(dft_dist))
    ax2.set_ylim(-80, 20)

    for i, band in enumerate(dft_bands):
        ax3.plot(dft_dist, band, "b-", linewidth=1.5, label="DFT-Zig" if i == 0 else None)
    for i, band in enumerate(abinit_bands):
        ax3.plot(abinit_dist, band, "r--", linewidth=1.2, label="ABINIT" if i == 0 else None)
    for pos in label_positions:
        ax3.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xticks(label_positions)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Energy (eV)")
    ax3.set_title("Overlay")
    ax3.set_xlim(min(dft_dist), max(dft_dist))
    ax3.set_ylim(-80, 20)
    ax3.legend(loc="upper right")

    timing_str = ""
    if dft_time is not None:
        timing_str += f"DFT-Zig: {dft_time:.1f}s"
    if abinit_time is not None:
        timing_str += f", ABINIT: {abinit_time:.1f}s"

    title = "Al 11e Band Structure: DFT-Zig vs ABINIT (ecut=15 Ry)"
    if timing_str:
        title += f"\n{timing_str}"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("band_comparison_11e.png", dpi=150)
    print("Saved: band_comparison_11e.png")

    # --- Plot 2: Energy difference ---
    fig2, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 8))
    max_diff = 0
    for i in range(min(len(dft_bands), len(abinit_bands))):
        abinit_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - abinit_interp) * 1000
        max_diff = max(max_diff, np.max(np.abs(diff_mev)))
        ax.plot(dft_dist, diff_mev, color=colors[i], linewidth=1.5, label=f"Band {i}")

    for pos in label_positions:
        ax.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy Difference (meV)")
    ax.set_xlabel("k-path")
    ax.set_title(f"DFT-Zig - ABINIT (Al 11e, ecut=15 Ry)\nMax |diff| = {max_diff:.1f} meV")
    ax.set_xlim(min(dft_dist), max(dft_dist))
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig("band_difference_11e.png", dpi=150)
    print("Saved: band_difference_11e.png")

    # --- Plot 3: Semi-core zoom ---
    fig3, (ax_z1, ax_z2, ax_z3) = plt.subplots(1, 3, figsize=(16, 6))

    for i, band in enumerate(dft_bands):
        ax_z1.plot(dft_dist, band, color="blue", linewidth=1.5)
    for pos in label_positions:
        ax_z1.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax_z1.set_xticks(label_positions)
    ax_z1.set_xticklabels(labels)
    ax_z1.set_ylabel("Energy (eV)")
    ax_z1.set_title("DFT-Zig (semi-core zoom)")
    ax_z1.set_xlim(min(dft_dist), max(dft_dist))
    ax_z1.set_ylim(-72, -55)

    for i, band in enumerate(abinit_bands):
        ax_z2.plot(abinit_dist, band, color="red", linewidth=1.5)
    for pos in label_positions:
        ax_z2.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax_z2.set_xticks(label_positions)
    ax_z2.set_xticklabels(labels)
    ax_z2.set_ylabel("Energy (eV)")
    ax_z2.set_title("ABINIT (semi-core zoom)")
    ax_z2.set_xlim(min(dft_dist), max(dft_dist))
    ax_z2.set_ylim(-72, -55)

    for i, band in enumerate(dft_bands):
        ax_z3.plot(dft_dist, band, "b-", linewidth=1.5, label="DFT-Zig" if i == 0 else None)
    for i, band in enumerate(abinit_bands):
        ax_z3.plot(abinit_dist, band, "r--", linewidth=1.2, label="ABINIT" if i == 0 else None)
    for pos in label_positions:
        ax_z3.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax_z3.set_xticks(label_positions)
    ax_z3.set_xticklabels(labels)
    ax_z3.set_ylabel("Energy (eV)")
    ax_z3.set_title("Overlay (semi-core zoom)")
    ax_z3.set_xlim(min(dft_dist), max(dft_dist))
    ax_z3.set_ylim(-72, -55)
    ax_z3.legend(loc="upper right")

    plt.suptitle("Al 11e Semi-core Bands: DFT-Zig vs ABINIT", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("band_semicore_zoom.png", dpi=150)
    print("Saved: band_semicore_zoom.png")

    # --- Plot 4: Valence band zoom (near Fermi level) ---
    fig4, (ax_v1, ax_v2, ax_v3) = plt.subplots(1, 3, figsize=(16, 6))

    for i, band in enumerate(dft_bands):
        ax_v1.plot(dft_dist, band, color="blue", linewidth=1.5)
    for pos in label_positions:
        ax_v1.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax_v1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax_v1.set_xticks(label_positions)
    ax_v1.set_xticklabels(labels)
    ax_v1.set_ylabel("Energy (eV)")
    ax_v1.set_title("DFT-Zig (valence zoom)")
    ax_v1.set_xlim(min(dft_dist), max(dft_dist))
    ax_v1.set_ylim(-15, 25)

    for i, band in enumerate(abinit_bands):
        ax_v2.plot(abinit_dist, band, color="red", linewidth=1.5)
    for pos in label_positions:
        ax_v2.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax_v2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax_v2.set_xticks(label_positions)
    ax_v2.set_xticklabels(labels)
    ax_v2.set_ylabel("Energy (eV)")
    ax_v2.set_title("ABINIT (valence zoom)")
    ax_v2.set_xlim(min(dft_dist), max(dft_dist))
    ax_v2.set_ylim(-15, 25)

    for i, band in enumerate(dft_bands):
        ax_v3.plot(dft_dist, band, "b-", linewidth=1.5, label="DFT-Zig" if i == 0 else None)
    for i, band in enumerate(abinit_bands):
        ax_v3.plot(abinit_dist, band, "r--", linewidth=1.2, label="ABINIT" if i == 0 else None)
    for pos in label_positions:
        ax_v3.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax_v3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax_v3.set_xticks(label_positions)
    ax_v3.set_xticklabels(labels)
    ax_v3.set_ylabel("Energy (eV)")
    ax_v3.set_title("Overlay (valence zoom)")
    ax_v3.set_xlim(min(dft_dist), max(dft_dist))
    ax_v3.set_ylim(-15, 25)
    ax_v3.legend(loc="upper right")

    plt.suptitle("Al 11e Valence Bands: DFT-Zig vs ABINIT", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("band_valence_zoom.png", dpi=150)
    print("Saved: band_valence_zoom.png")

    # --- Summary ---
    print(f"\nFermi levels: DFT-Zig = {dft_fermi:.4f} eV, ABINIT = {abinit_fermi:.4f} eV")
    print(f"Max |diff| = {max_diff:.1f} meV")


def save_baseline():
    """Save Fermi-aligned ABINIT data to baseline/ directory."""
    bands, kpts = load_abinit()
    fermi = load_abinit_fermi()
    if fermi is not None:
        bands = [[e - fermi for e in band] for band in bands]
    else:
        print("Warning: ABINIT Fermi level not found, saving unaligned bands")

    baseline_dir = 'baseline'
    os.makedirs(baseline_dir, exist_ok=True)

    nbands = len(bands)
    with open(os.path.join(baseline_dir, 'abinit_bands.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'band{i}' for i in range(nbands)])
        for k in range(len(bands[0])):
            writer.writerow([bands[b][k] for b in range(nbands)])

    with open(os.path.join(baseline_dir, 'abinit_kpoints.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kx', 'ky', 'kz'])
        for kpt in kpts:
            writer.writerow(kpt)

    print(f"Baseline saved to {baseline_dir}/ (Fermi-aligned)")


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
        ["../../zig-out/bin/dft_zig", "dft_zig_11e.toml"],
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
    """Run regression check against saved baseline and return exit code."""
    # Per-band MSE thresholds (in meV^2)
    # Based on measured values with ~3x margin
    # Semi-core 2p offset ~88 meV is known; valence bands match ±2 meV
    MSE_THRESHOLDS = {
        0: 50,      # semi-core 2s (current: 7.4)
        1: 50,      # semi-core 2p (current: 9.1)
        2: 50,      # semi-core 2p (current: 9.0)
        3: 50,      # semi-core 2p (current: 9.0)
        4: 100,     # valence (current: 0.1)
        5: 100,     # valence (current: 0.1)
        6: 100,     # valence (current: 0.3)
        7: 100,     # valence (current: 0.6)
    }

    # Load DFT-Zig data and Fermi-align
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    dft_fermi = load_dft_zig_fermi()
    if dft_fermi is not None:
        dft_bands = [[e - dft_fermi for e in band] for band in dft_bands]

    # Load Fermi-aligned baseline
    abinit_bands, abinit_kpts = load_baseline()

    nbands = min(len(dft_bands), len(abinit_bands))

    abinit_dist = dft_dist if len(abinit_bands[0]) == len(dft_dist) else calculate_kpath_distances(abinit_kpts)

    # Calculate MSE
    mse_per_band, mse_total = calculate_mse(dft_bands, abinit_bands, dft_dist, abinit_dist)

    # Report
    print("\n" + "=" * 50)
    print("Regression Check Results (Aluminum 11e)")
    print("=" * 50)
    print("MSE per band (meV^2):")
    passed = True
    for i, mse in enumerate(mse_per_band):
        threshold = MSE_THRESHOLDS.get(i, 1000)
        status = "OK" if mse <= threshold else "FAIL"
        print(f"  Band {i}: {mse:.1f} (threshold: {threshold}) [{status}]")
        if mse > threshold:
            passed = False

    print(f"Average MSE: {mse_total:.1f} meV^2")

    if passed:
        print("PASS: All checks passed!")
        return 0
    else:
        print("FAIL: Some checks failed!")
        return 1


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

    if "--run" in sys.argv:
        print("Building DFT-Zig...")
        subprocess.run(["just", "build"], cwd="../..", check=True)
        print("\nRunning DFT-Zig...")
        subprocess.run(["../../zig-out/bin/dft_zig", "dft_zig_11e.toml"], check=True)
    elif "--run-check" in sys.argv:
        run_dft_zig_only()
        sys.exit(run_regression_check())

    if "--save-baseline" in sys.argv:
        save_baseline()
    elif "--check" not in sys.argv:
        plot_comparison()

    if "--check" in sys.argv:
        sys.exit(run_regression_check())
