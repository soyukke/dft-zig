#!/usr/bin/env python3
"""Parse QE band output and generate baseline CSV for PAW regression test.

Usage:
    python3 generate_qe_baseline.py
"""

import csv
import os
import re
import numpy as np

eV_to_Ry = 1.0 / 13.6057  # QE outputs in eV, DFT-Zig in Ry


def parse_qe_bands(filename):
    """Parse QE bands output and return (kpoints_cart, bands_eV)."""
    with open(filename) as f:
        content = f.read()

    # Parse k-points and eigenvalues
    pattern = r'k =\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+\(\s*\d+ PWs\)\s+bands \(ev\):\s*\n\s*([\d.\s\-]+)'
    matches = re.findall(pattern, content)

    kpoints = []
    bands = []
    for m in matches:
        kx, ky, kz = float(m[0]), float(m[1]), float(m[2])
        kpoints.append([kx, ky, kz])
        eigs = [float(x) for x in m[3].split()]
        bands.append(eigs)

    return np.array(kpoints), np.array(bands)


def compute_kpath_distances(kpoints):
    """Compute cumulative k-path distances in Cartesian coords."""
    dists = [0.0]
    for i in range(1, len(kpoints)):
        dk = np.linalg.norm(kpoints[i] - kpoints[i - 1])
        dists.append(dists[-1] + dk)
    return dists


def main():
    qe_kpts, qe_bands = parse_qe_bands("qe_band.out")
    nkpts, nbands = qe_bands.shape
    print(f"Parsed {nkpts} k-points, {nbands} bands from QE")

    # Convert eV to Ry (to match DFT-Zig convention)
    qe_bands_ry = qe_bands * eV_to_Ry

    # Compute k-path distances
    dists = compute_kpath_distances(qe_kpts)

    # Write baseline CSV in same format as DFT-Zig band_energies.csv
    os.makedirs("baseline", exist_ok=True)
    with open("baseline/qe_bands.csv", "w", newline="") as f:
        header = ["index", "dist"] + [f"band{i}" for i in range(nbands)]
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(nkpts):
            row = [i, f"{dists[i]:.10f}"] + [f"{qe_bands_ry[i, j]:.10f}" for j in range(nbands)]
            writer.writerow(row)

    print(f"Wrote baseline/qe_bands.csv ({nkpts} k-points, {nbands} bands)")

    # Also write k-points for reference
    with open("baseline/qe_kpoints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "kx", "ky", "kz", "dist"])
        for i in range(nkpts):
            writer.writerow([i, f"{qe_kpts[i, 0]:.7f}", f"{qe_kpts[i, 1]:.7f}", f"{qe_kpts[i, 2]:.7f}", f"{dists[i]:.10f}"])

    print(f"Wrote baseline/qe_kpoints.csv")


if __name__ == "__main__":
    main()
