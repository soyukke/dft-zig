#!/usr/bin/env python3
"""Compare aluminum band structures: DFT-Zig vs ABINIT (fair comparison)."""

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
    timing_file = "out_dft_zig_fair/timing.txt"
    if not os.path.exists(timing_file):
        return None
    with open(timing_file) as f:
        for line in f:
            if line.startswith("total_sec"):
                return float(line.split("=")[1].strip())
    return None


def load_abinit_timing():
    """Load ABINIT timing from .abo file."""
    abo_file = "abinit/aluminum_fair.abo"
    if not os.path.exists(abo_file):
        return None
    with open(abo_file) as f:
        content = f.read()
    match = re.search(r"Overall time at end.*wall=\s*([\d.]+)", content)
    if match:
        return float(match.group(1))
    return None


def format_timing_label(dft_time, abinit_time):
    if dft_time is None and abinit_time is None:
        return None
    dft_label = "n/a" if dft_time is None else f"{dft_time:.1f}s"
    abinit_label = "n/a" if abinit_time is None else f"{abinit_time:.1f}s"
    return f"Timing: DFT-Zig {dft_label}, ABINIT {abinit_label}"


def load_dft_zig_fermi():
    """Load DFT-Zig Fermi level from status.txt (Ry)."""
    status_file = "out_dft_zig_fair/status.txt"
    if not os.path.exists(status_file):
        return None
    with open(status_file) as f:
        for line in f:
            if line.startswith("scf_fermi_level_ry"):
                return float(line.split("=")[1].strip()) * Ry_to_eV
    return None


def load_abinit_fermi():
    """Load ABINIT Fermi level from .abo file (Ha)."""
    abo_file = "abinit/aluminum_fair.abo"
    if not os.path.exists(abo_file):
        return None
    with open(abo_file) as f:
        content = f.read()
    match = re.search(r"fermie\s*:\s*([\d.E+-]+)", content)
    if not match:
        match = re.search(r"Fermi \(or HOMO\) energy \(hartree\) =\s*([\d.E+-]+)", content)
    if match:
        return float(match.group(1)) * Ha_to_eV
    return None


def run_calculations():
    """Run both DFT-Zig and ABINIT calculations."""
    print("=" * 50)
    print("Building DFT-Zig (ReleaseFast + FFTW via just build)...")
    print("=" * 50)
    result = subprocess.run(
        ["just", "build"],
        cwd="../..",
        capture_output=True,
        text=True,
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
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("DFT-Zig failed:", result.stderr)
        sys.exit(1)
    print("DFT-Zig done.\n")

    print("=" * 50)
    print("Running ABINIT (fair comparison)...")
    print("=" * 50)
    os.chdir("abinit")
    for f in os.listdir("."):
        if f.startswith("aluminum_fairo") or (f.endswith(".nc") and "fair" in f):
            try:
                os.remove(f)
            except OSError:
                pass
    result = subprocess.run(
        ["../../../external/abinit/build/abinit-10.4.7/src/98_main/abinit", "aluminum_fair.in"],
        capture_output=True,
        text=True,
    )
    os.chdir("..")
    if result.returncode != 0:
        print("ABINIT failed:", result.stderr)
        sys.exit(1)
    print("ABINIT done.\n")


def load_dft_zig():
    """Load DFT-Zig band data."""
    dft_data = []
    with open("out_dft_zig_fair/band_energies.csv") as f:
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
    with open("out_dft_zig_fair/band_kpoints.csv") as f:
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
    """Load ABINIT band data from fair comparison."""
    abinit_bands = [[] for _ in range(8)]
    abinit_kpts: list[list[float]] = []
    eig_file = "abinit/aluminum_fairo_DS2_EIG"

    with open(eig_file) as f:
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
    """Calculate cumulative distances along k-path using reciprocal lattice vectors."""
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
    """Create comparison plots."""
    dft_dist, dft_bands, labels, label_positions = load_dft_zig()
    abinit_bands, abinit_kpts = load_abinit()

    dft_time = load_dft_zig_timing()
    abinit_time = load_abinit_timing()
    timing_label = format_timing_label(dft_time, abinit_time)

    dft_fermi = load_dft_zig_fermi()
    abinit_fermi = load_abinit_fermi()

    dft_align = "fermi"
    abinit_align = "fermi"

    dft_min = min(min(band) for band in dft_bands)
    dft_max = max(max(band) for band in dft_bands)
    abinit_min = min(min(band) for band in abinit_bands)
    abinit_max = max(max(band) for band in abinit_bands)

    occupied_bands = 6
    dft_ref_label = "VBM"
    abinit_ref_label = "VBM"

    if dft_align == "fermi" and dft_fermi is not None:
        if dft_fermi < dft_min or dft_fermi > dft_max:
            print(
                "Warning: DFT-Zig Fermi is outside band range; aligning anyway."
            )
        for band in dft_bands:
            for i in range(len(band)):
                band[i] -= dft_fermi
        dft_ref_label = "Fermi"
    else:
        dft_ref = max(max(band) for band in dft_bands[:occupied_bands])
        for band in dft_bands:
            for i in range(len(band)):
                band[i] -= dft_ref
        dft_ref_label = "VBM"

    if abinit_align == "fermi" and abinit_fermi is not None:
        if abinit_fermi < abinit_min or abinit_fermi > abinit_max:
            print(
                "Warning: ABINIT Fermi is outside band range; aligning anyway."
            )
        for band in abinit_bands:
            for i in range(len(band)):
                band[i] -= abinit_fermi
        abinit_ref_label = "Fermi"
    else:
        abinit_ref = max(max(band) for band in abinit_bands[:occupied_bands])
        for band in abinit_bands:
            for i in range(len(band)):
                band[i] -= abinit_ref
        abinit_ref_label = "VBM"

    reference_label = f"DFT: {dft_ref_label}, ABINIT: {abinit_ref_label}"

    dft_min_aligned = min(min(band) for band in dft_bands)
    dft_max_aligned = max(max(band) for band in dft_bands)
    abinit_min_aligned = min(min(band) for band in abinit_bands)
    abinit_max_aligned = max(max(band) for band in abinit_bands)
    full_min = min(dft_min_aligned, abinit_min_aligned)
    full_max = max(dft_max_aligned, abinit_max_aligned)
    span = full_max - full_min
    pad = 0.05 * span if span > 0.0 else 1.0
    full_ylim = (full_min - pad, full_max + pad)

    if len(abinit_bands[0]) == len(dft_dist):
        abinit_dist = dft_dist
    else:
        print(
            f"Warning: k-point count mismatch: DFT-Zig={len(dft_dist)}, ABINIT={len(abinit_bands[0])}"
        )
        abinit_dist = calculate_kpath_distances(abinit_kpts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    for i, band in enumerate(dft_bands):
        ax1.plot(dft_dist, band, color="blue", linewidth=1.5)
    for pos in label_positions:
        ax1.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Energy (eV)")
    dft_title = f"DFT-Zig (ref: {dft_ref_label}"
    if dft_time is not None:
        dft_title += f", {dft_time:.1f}s"
    dft_title += ")"
    ax1.set_title(dft_title)
    ax1.set_xlim(min(dft_dist), max(dft_dist))
    ax1.set_ylim(-15, 15)

    for i, band in enumerate(abinit_bands):
        ax2.plot(abinit_dist, band, color="red", linewidth=1.5)
    for pos in label_positions:
        ax2.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xticks(label_positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Energy (eV)")
    abinit_title = f"ABINIT (ref: {abinit_ref_label}"
    if abinit_time is not None:
        abinit_title += f", {abinit_time:.1f}s"
    abinit_title += ")"
    ax2.set_title(abinit_title)
    ax2.set_xlim(min(dft_dist), max(dft_dist))
    ax2.set_ylim(-15, 15)

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
    ax3.set_ylim(-15, 15)
    ax3.legend(loc="upper right")

    title = "Aluminum Band Structure: DFT-Zig vs ABINIT (Fair Comparison)"
    if timing_label:
        title += f"\n{timing_label}"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("band_comparison_fair.png", dpi=150)
    print("Saved: band_comparison_fair.png")

    fig_full, (ax1_full, ax2_full, ax3_full) = plt.subplots(1, 3, figsize=(16, 6))

    for band in dft_bands:
        ax1_full.plot(dft_dist, band, color="blue", linewidth=1.5)
    for pos in label_positions:
        ax1_full.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax1_full.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1_full.set_xticks(label_positions)
    ax1_full.set_xticklabels(labels)
    ax1_full.set_ylabel("Energy (eV)")
    ax1_full.set_title(dft_title)
    ax1_full.set_xlim(min(dft_dist), max(dft_dist))
    ax1_full.set_ylim(*full_ylim)

    for band in abinit_bands:
        ax2_full.plot(abinit_dist, band, color="red", linewidth=1.5)
    for pos in label_positions:
        ax2_full.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax2_full.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2_full.set_xticks(label_positions)
    ax2_full.set_xticklabels(labels)
    ax2_full.set_ylabel("Energy (eV)")
    ax2_full.set_title(abinit_title)
    ax2_full.set_xlim(min(dft_dist), max(dft_dist))
    ax2_full.set_ylim(*full_ylim)

    for i, band in enumerate(dft_bands):
        ax3_full.plot(dft_dist, band, "b-", linewidth=1.5, label="DFT-Zig" if i == 0 else None)
    for i, band in enumerate(abinit_bands):
        ax3_full.plot(abinit_dist, band, "r--", linewidth=1.2, label="ABINIT" if i == 0 else None)
    for pos in label_positions:
        ax3_full.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax3_full.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3_full.set_xticks(label_positions)
    ax3_full.set_xticklabels(labels)
    ax3_full.set_ylabel("Energy (eV)")
    ax3_full.set_title("Overlay")
    ax3_full.set_xlim(min(dft_dist), max(dft_dist))
    ax3_full.set_ylim(*full_ylim)
    ax3_full.legend(loc="upper right")

    title_full = "Aluminum Band Structure: DFT-Zig vs ABINIT (Full Range)"
    if timing_label:
        title_full += f"\n{timing_label}"
    plt.suptitle(title_full, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig("band_comparison_full.png", dpi=150)
    print("Saved: band_comparison_full.png")

    fig2, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 8))
    for i in range(8):
        abinit_interp = np.interp(dft_dist, abinit_dist, abinit_bands[i])
        diff_mev = (np.array(dft_bands[i]) - abinit_interp) * 1000
        ax.plot(dft_dist, diff_mev, color=colors[i], linewidth=1.5, label=f"Band {i}")

    for pos in label_positions:
        ax.axvline(x=pos, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy Difference (meV)")
    ax.set_xlabel("k-path")
    diff_title = "DFT-Zig - ABINIT (Fair Comparison, Same Conditions)"
    if timing_label:
        diff_title += f"\n{timing_label}"
    ax.set_title(diff_title)
    ax.set_xlim(min(dft_dist), max(dft_dist))
    ax.legend(loc="upper right", ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig("band_difference_fair.png", dpi=150)
    print("Saved: band_difference_fair.png")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Reference alignment: {reference_label}")
    if dft_time is not None:
        print(f"DFT-Zig wall time: {dft_time:.1f}s")
    if abinit_time is not None:
        print(f"ABINIT wall time:  {abinit_time:.1f}s")


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
        ["../../zig-out/bin/dft_zig", "dft_zig_fair.toml"],
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
    # Based on measured values with ~2x margin
    MSE_THRESHOLDS = {
        0: 100,     # current: 0.0
        1: 100,     # current: 0.0
        2: 100,     # current: 0.0
        3: 100,     # current: 0.0
        4: 100,     # current: 0.0
        5: 100,     # current: 0.1
        6: 2000,    # current: 903.9 (high-energy band, sensitive)
        7: 1000,    # current: 431.0
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
    print("Regression Check Results (Aluminum)")
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
