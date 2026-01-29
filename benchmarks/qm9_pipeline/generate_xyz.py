#!/usr/bin/env python3
"""Generate XYZ files and PySCF reference energies for QM9 pipeline validation.

Uses PySCF to:
  1. Build initial molecular geometries
  2. Run B3LYP/6-31G(2df,p) single-point energy (cart=True)
  3. Optionally optimize geometry and compute optimized energy
  4. Write XYZ files and references.json

Usage:
    source ../../.venv/bin/activate
    python3 generate_xyz.py [--optimize]
"""

import json
import os
import sys
import numpy as np

try:
    from pyscf import gto, dft
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize
except ImportError:
    print("ERROR: PySCF not found. Activate venv: source ../../.venv/bin/activate")
    sys.exit(1)

# Constants
BOHR_TO_ANG = 0.52917721067
BASIS = "6-31G(2df,p)"
XC = "B3LYP"
GRID_LEVEL = 3  # PySCF grid level (corresponds roughly to 50 radial / 302 angular)

# Molecules: name -> (atom_string, charge, spin)
# Atom strings use Angstrom coordinates.
MOLECULES = {
    "H2": {
        "atoms": """
            H  0.000  0.000  0.000
            H  0.000  0.000  0.740
        """,
        "charge": 0,
        "spin": 0,
    },
    "H2O": {
        "atoms": """
            O  0.000  0.000  0.117
            H  0.000  0.757 -0.470
            H  0.000 -0.757 -0.470
        """,
        "charge": 0,
        "spin": 0,
    },
    "CH4": {
        "atoms": """
            C  0.000  0.000  0.000
            H  0.629  0.629  0.629
            H -0.629 -0.629  0.629
            H -0.629  0.629 -0.629
            H  0.629 -0.629 -0.629
        """,
        "charge": 0,
        "spin": 0,
    },
    "NH3": {
        "atoms": """
            N  0.000  0.000  0.117
            H  0.000  0.939 -0.272
            H  0.813 -0.470 -0.272
            H -0.813 -0.470 -0.272
        """,
        "charge": 0,
        "spin": 0,
    },
    "HF": {
        "atoms": """
            H  0.000  0.000  0.000
            F  0.000  0.000  0.917
        """,
        "charge": 0,
        "spin": 0,
    },
    "CH2O": {
        "atoms": """
            C  0.000  0.000 -0.529
            O  0.000  0.000  0.675
            H  0.000  0.936 -1.110
            H  0.000 -0.936 -1.110
        """,
        "charge": 0,
        "spin": 0,
    },
    "CH3OH": {
        "atoms": """
            C -0.047  0.665  0.000
            O -0.047 -0.764  0.000
            H  0.986  1.020  0.000
            H -0.558  1.020  0.892
            H -0.558  1.020 -0.892
            H  0.842 -1.069  0.000
        """,
        "charge": 0,
        "spin": 0,
    },
    "C2H2": {
        "atoms": """
            C  0.000  0.000 -0.601
            C  0.000  0.000  0.601
            H  0.000  0.000 -1.665
            H  0.000  0.000  1.665
        """,
        "charge": 0,
        "spin": 0,
    },
    "C2H4": {
        "atoms": """
            C  0.000  0.000  0.667
            C  0.000  0.000 -0.667
            H  0.000  0.923  1.238
            H  0.000 -0.923  1.238
            H  0.000  0.923 -1.238
            H  0.000 -0.923 -1.238
        """,
        "charge": 0,
        "spin": 0,
    },
    "C2H6": {
        "atoms": """
            C  0.000  0.000  0.763
            C  0.000  0.000 -0.763
            H  0.000  1.018  1.158
            H  0.882 -0.509  1.158
            H -0.882 -0.509  1.158
            H  0.000 -1.018 -1.158
            H -0.882  0.509 -1.158
            H  0.882  0.509 -1.158
        """,
        "charge": 0,
        "spin": 0,
    },
    "HCN": {
        "atoms": """
            H  0.000  0.000 -1.063
            C  0.000  0.000  0.000
            N  0.000  0.000  1.156
        """,
        "charge": 0,
        "spin": 0,
    },
    "N2": {
        "atoms": """
            N  0.000  0.000  0.000
            N  0.000  0.000  1.098
        """,
        "charge": 0,
        "spin": 0,
    },
}


def build_mol(name, mol_data, cart=True):
    """Build a PySCF molecule."""
    mol = gto.M(
        atom=mol_data["atoms"],
        basis=BASIS,
        charge=mol_data["charge"],
        spin=mol_data["spin"],
        cart=cart,
        unit="Angstrom",
        verbose=0,
    )
    return mol


def run_single_point(mol):
    """Run B3LYP single-point DFT calculation."""
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = GRID_LEVEL
    # Match DFT-Zig: 50 radial, 302 angular, no pruning
    mf.grids.atom_grid = {"H": (50, 302), "C": (50, 302), "N": (50, 302),
                          "O": (50, 302), "F": (50, 302)}
    mf.grids.prune = None
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    e = mf.kernel()
    return mf, e


def mol_to_xyz_string(mol, comment=""):
    """Convert PySCF mol to XYZ format string (Angstrom)."""
    coords_ang = mol.atom_coords(unit="Angstrom")
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    lines = [str(mol.natm), comment]
    for sym, coord in zip(symbols, coords_ang):
        lines.append(f"{sym:2s}  {coord[0]:16.10f}  {coord[1]:16.10f}  {coord[2]:16.10f}")
    return "\n".join(lines)


def main():
    do_optimize = "--optimize" in sys.argv

    xyz_dir = os.path.join(os.path.dirname(__file__), "xyz")
    os.makedirs(xyz_dir, exist_ok=True)

    references = {}

    print(f"{'='*70}")
    print(f"  QM9 Pipeline: PySCF Reference Generation")
    print(f"  Basis: {BASIS}, XC: {XC}, cart=True")
    print(f"  Grid: 50 radial, 302 angular, no pruning")
    print(f"  Optimize: {do_optimize}")
    print(f"{'='*70}")
    print()

    for name, mol_data in MOLECULES.items():
        print(f"--- {name} ---")

        mol = build_mol(name, mol_data)
        nao = mol.nao_nr()
        nelec = mol.nelectron
        print(f"  nao={nao}, nelec={nelec}, natm={mol.natm}")

        # Single-point at initial geometry
        mf, e_sp = run_single_point(mol)
        print(f"  Single-point E = {e_sp:.12f} Ha")

        # Write initial geometry XYZ
        xyz_str = mol_to_xyz_string(mol, comment=f"{name} initial geometry (Angstrom)")
        xyz_path = os.path.join(xyz_dir, f"{name}.xyz")
        with open(xyz_path, "w") as f:
            f.write(xyz_str + "\n")
        print(f"  Written: {xyz_path}")

        ref_entry = {
            "formula": name,
            "n_atoms": mol.natm,
            "n_electrons": nelec,
            "n_basis": nao,
            "charge": mol_data["charge"],
            "spin": mol_data["spin"],
            "energy_sp": e_sp,
        }

        if do_optimize:
            try:
                print(f"  Optimizing geometry...")
                mol_opt = geometric_optimize(mf)
                mf_opt, e_opt = run_single_point(mol_opt)
                print(f"  Optimized E = {e_opt:.12f} Ha")

                # Write optimized geometry XYZ
                xyz_opt = mol_to_xyz_string(mol_opt, comment=f"{name} PySCF optimized (Angstrom)")
                xyz_opt_path = os.path.join(xyz_dir, f"{name}_opt.xyz")
                with open(xyz_opt_path, "w") as f:
                    f.write(xyz_opt + "\n")
                print(f"  Written: {xyz_opt_path}")

                ref_entry["energy_opt"] = e_opt
                ref_entry["has_optimized"] = True
            except Exception as ex:
                print(f"  Optimization failed: {ex}")
                ref_entry["has_optimized"] = False
        else:
            ref_entry["has_optimized"] = False

        references[name] = ref_entry
        print()

    # Write references.json
    ref_path = os.path.join(os.path.dirname(__file__), "references.json")
    with open(ref_path, "w") as f:
        json.dump(references, f, indent=2)
    print(f"Written: {ref_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary: {len(references)} molecules processed")
    print(f"{'='*70}")
    for name, ref in references.items():
        e_str = f"E_sp = {ref['energy_sp']:.12f}"
        if ref.get("has_optimized") and "energy_opt" in ref:
            e_str += f", E_opt = {ref['energy_opt']:.12f}"
        print(f"  {name:10s}: {e_str}")
    print()


if __name__ == "__main__":
    main()
