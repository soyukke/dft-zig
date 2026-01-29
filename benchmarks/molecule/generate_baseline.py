#!/usr/bin/env python3
"""Generate PySCF baseline energies for molecule regression test.

Runs B3LYP/6-31G(2df,p) with conventional and density fitting for 9 molecules,
writes results to baseline.json.

Usage:
    source .venv/bin/activate
    python3 benchmarks/molecule/generate_baseline.py
"""

import json
import os
from pyscf import gto, dft

BASIS = "6-31G(2df,p)"
XC = "B3LYP"
AUX_BASIS = "def2-universal-jkfit"

MOLECULES = [
    ("H2",    "H 0 0 0; H 0 0 0.740"),
    ("H2O",   "O 0 0 0.117; H 0 0.757 -0.470; H 0 -0.757 -0.470"),
    ("CH4",   "C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629"),
    ("N2",    "N 0 0 0; N 0 0 1.098"),
    ("C2H2",  "C 0 0 -0.601; C 0 0 0.601; H 0 0 -1.665; H 0 0 1.665"),
    ("CH2O",  "C 0 0 -0.529; O 0 0 0.675; H 0 0.936 -1.110; H 0 -0.936 -1.110"),
    ("C2H4",  "C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238; H 0 0.923 -1.238; H 0 -0.923 -1.238"),
    ("CH3OH", "C -0.047 0.665 0; O -0.047 -0.764 0; H 0.986 1.020 0; H -0.558 1.020 0.892; H -0.558 1.020 -0.892; H 0.842 -1.069 0"),
    ("C2H6",  "C 0 0 0.763; C 0 0 -0.763; H 0 1.018 1.158; H 0.882 -0.509 1.158; H -0.882 -0.509 1.158; H 0 -1.018 -1.158; H -0.882 0.509 -1.158; H 0.882 0.509 -1.158"),
]


def run_calc(mol, use_df=False):
    if use_df:
        mf = dft.RKS(mol).density_fit(auxbasis=AUX_BASIS)
    else:
        mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.atom_grid = {s: (50, 302) for s in ("H", "C", "N", "O", "F")}
    mf.grids.prune = None
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    return mf.kernel()


def main():
    data = {
        "method": f"{XC}/{BASIS}",
        "grid": "50/302/no_prune/cart",
        "aux_basis": AUX_BASIS,
        "molecules": {},
    }

    for name, atoms in MOLECULES:
        mol = gto.M(atom=atoms, basis=BASIS, cart=True, unit="Angstrom", verbose=0)
        nbas = mol.nao_nr()
        nelec = mol.nelectron

        e_conv = run_calc(mol, use_df=False)
        e_df = run_calc(mol, use_df=True)

        data["molecules"][name] = {
            "nbas": nbas,
            "nelec": nelec,
            "conv": round(e_conv, 10),
            "df": round(e_df, 10),
        }
        print(f"  {name:<8} nbas={nbas:>3}  conv={e_conv:.10f}  df={e_df:.10f}")

    out_path = os.path.join(os.path.dirname(__file__), "baseline.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
