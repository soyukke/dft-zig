#!/usr/bin/env python3
"""Generate ABINIT DFPT reference data for regression test.

Runs ABINIT GS + DFPT at Gamma and L points with 2x2x2 k-mesh,
then saves phonon frequencies to a reference CSV.

Usage:
    python3 generate_abinit_ref.py
"""

import os
import sys
import subprocess
import re
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABINIT = os.path.join(SCRIPT_DIR, "../../external/abinit/build/abinit-10.4.7/src/98_main/abinit")

ACELL_BOHR = 5.1315  # a/sqrt(2) in Bohr for FCC primitive cell

# q-points to compute (reduced coordinates) — commensurate with 2x2x2 k-mesh
# Only Gamma, X, L are commensurate (W, K have fractional coords not on 2x2x2 grid)
from collections import OrderedDict
QPOINTS = OrderedDict([
    ("Gamma", [0.0, 0.0, 0.0]),
    ("X", [0.5, 0.0, 0.5]),
    ("L", [0.5, 0.5, 0.5]),
])

WORKDIR = os.path.join(SCRIPT_DIR, "abinit_ref_2k")


def create_gs_input():
    return f"""# ABINIT Ground State SCF for Si (2x2x2 k-mesh)
pp_dirpath "../../pseudo"
pseudos "Si.upf"

ntypat 1
znucl 14
natom 2
typat 1 1

acell 3*{ACELL_BOHR}
rprim
  0.0 1.0 1.0
  1.0 0.0 1.0
  1.0 1.0 0.0

xred
  0.00 0.00 0.00
  0.25 0.25 0.25

ecut 7.5
nband 8

ixc 1

kptopt 1
ngkpt 2 2 2
nshiftk 1
shiftk 0.0 0.0 0.0

iscf 7
diemix 0.3
tolvrs 1.0d-10
nstep 50

prtden 1
prtwf 1

outdata_prefix "{WORKDIR}/abinit_gso"
tmpdata_prefix "{WORKDIR}/abinit_gs_tmp"
"""


def create_dfpt_input(q, idx):
    return f"""# ABINIT DFPT at q = ({q[0]}, {q[1]}, {q[2]})
pp_dirpath "../../pseudo"
pseudos "Si.upf"

ntypat 1
znucl 14
natom 2
typat 1 1

acell 3*{ACELL_BOHR}
rprim
  0.0 1.0 1.0
  1.0 0.0 1.0
  1.0 1.0 0.0

xred
  0.00 0.00 0.00
  0.25 0.25 0.25

ecut 7.5
nband 8

ixc 1

kptopt 3
ngkpt 2 2 2
nshiftk 1
shiftk 0.0 0.0 0.0

irdwfk 1
irdden 1

iscf 7
tolvrs 1.0d-8
nstep 100

rfphon 1
rfatpol 1 2
rfdir 1 1 1
nqpt 1
qpt {q[0]:.10f} {q[1]:.10f} {q[2]:.10f}

outdata_prefix "{WORKDIR}/abinit_dfpt_q{idx:03d}o"
tmpdata_prefix "{WORKDIR}/abinit_dfpt_q{idx:03d}_tmp"
indata_prefix "{WORKDIR}/abinit_gso"
"""


def parse_ddb_phonon_frequencies(abo_file):
    """Parse phonon frequencies from ABINIT DFPT .abo output."""
    with open(abo_file, 'r') as f:
        content = f.read()

    # Find "Phonon wavevector" and subsequent "Phonon frequencies in cm-1"
    freqs = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Phonon frequencies in cm-1' in line:
            # Next line(s) have frequencies
            j = i + 1
            freq_line = ""
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('='):
                freq_line += " " + lines[j]
                j += 1
            # Parse
            tokens = freq_line.strip().replace('-', ' -').split()
            parsed = []
            for t in tokens:
                try:
                    parsed.append(float(t))
                except ValueError:
                    pass
            if len(parsed) == 6:
                freqs = parsed
                break  # Take the last occurrence

    return freqs


def main():
    os.makedirs(WORKDIR, exist_ok=True)

    # Step 1: GS SCF
    print("Running ABINIT Ground State (2x2x2 k-mesh)...")
    gs_file = os.path.join(WORKDIR, "abinit_gs.in")
    with open(gs_file, 'w') as f:
        f.write(create_gs_input())

    gs_log = os.path.join(WORKDIR, "abinit_gs.log")
    with open(gs_log, 'w') as logf:
        result = subprocess.run(
            [ABINIT, gs_file],
            cwd=SCRIPT_DIR,
            stdout=logf, stderr=subprocess.STDOUT,
            timeout=300
        )
    if result.returncode != 0:
        print(f"  ABINIT GS failed! Check {gs_log}")
        sys.exit(1)
    print("  GS done.")

    # Step 2: DFPT at each q-point
    ref_data = {}
    for idx, (label, q) in enumerate(QPOINTS.items()):
        print(f"  DFPT {label} q=({q[0]}, {q[1]}, {q[2]}) ...", end="", flush=True)

        dfpt_file = os.path.join(WORKDIR, f"abinit_dfpt_q{idx:03d}.in")
        with open(dfpt_file, 'w') as f:
            f.write(create_dfpt_input(q, idx))

        dfpt_log = os.path.join(WORKDIR, f"abinit_dfpt_q{idx:03d}.log")
        with open(dfpt_log, 'w') as logf:
            result = subprocess.run(
                [ABINIT, dfpt_file],
                cwd=SCRIPT_DIR,
                stdout=logf, stderr=subprocess.STDOUT,
                timeout=300
            )
        if result.returncode != 0:
            print(f" FAILED! Check {dfpt_log}")
            sys.exit(1)

        # Parse frequencies from .abo file
        abo_file = os.path.join(WORKDIR, f"abinit_dfpt_q{idx:03d}.abo")
        # ABINIT may append numbers, find the actual file
        if not os.path.exists(abo_file):
            # Try finding it
            candidates = sorted([f for f in os.listdir(WORKDIR) if f.startswith(f"abinit_dfpt_q{idx:03d}.abo")])
            if candidates:
                abo_file = os.path.join(WORKDIR, candidates[-1])

        freqs = parse_ddb_phonon_frequencies(abo_file)
        if not freqs:
            print(f" WARNING: Could not parse frequencies from {abo_file}")
        else:
            ref_data[label] = {"q": q, "frequencies_cm1": freqs}
            print(f" done. freqs={[f'{x:.1f}' for x in freqs]}")

    # Step 3: Save reference CSV
    ref_csv = os.path.join(SCRIPT_DIR, "abinit_dfpt_ref_2k.csv")
    with open(ref_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "q0", "q1", "q2", "mode_0", "mode_1", "mode_2", "mode_3", "mode_4", "mode_5"])
        for label, data in ref_data.items():
            q = data["q"]
            freqs = data["frequencies_cm1"]
            writer.writerow([label, q[0], q[1], q[2]] + freqs)

    print(f"\nReference data saved to: {ref_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABINIT DFPT Reference (2x2x2 k-mesh, LDA)")
    print("=" * 60)
    for label, data in ref_data.items():
        freqs = data["frequencies_cm1"]
        print(f"  {label:>6s}: {' '.join(f'{x:8.1f}' for x in freqs)}")


if __name__ == '__main__':
    main()
