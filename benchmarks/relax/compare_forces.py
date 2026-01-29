#!/usr/bin/env python3
"""
Compare forces from DFT-Zig and ABINIT for displaced silicon structure.
"""

import re
import sys
import subprocess
from pathlib import Path

# Conversion factors
RY_PER_HA = 2.0  # 1 Ha = 2 Ry
BOHR_PER_ANGSTROM = 1.8897259886

def parse_abinit_forces(output_file):
    """Parse forces from ABINIT output file (in Ha/Bohr)."""
    forces = []
    with open(output_file) as f:
        content = f.read()

    # Find the cartesian forces section
    # ABINIT outputs: "cartesian forces (hartree/bohr)"
    pattern = r"cartesian forces \(hartree/bohr\).*?\n((?:\s+\d+\s+[-\d.E+]+\s+[-\d.E+]+\s+[-\d.E+]+\n)+)"
    match = re.search(pattern, content, re.IGNORECASE)

    if not match:
        # Try alternative format
        pattern = r"Cartesian forces.*?\n((?:\s*[-\d.E+]+\s+[-\d.E+]+\s+[-\d.E+]+\s*\n)+)"
        match = re.search(pattern, content, re.IGNORECASE)

    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                # Extract last 3 numbers (fx, fy, fz)
                try:
                    fx = float(parts[-3])
                    fy = float(parts[-2])
                    fz = float(parts[-1])
                    forces.append((fx, fy, fz))
                except ValueError:
                    continue

    return forces

def parse_dft_zig_forces(forces_file):
    """Parse forces from DFT-Zig output (relax_forces.csv in Ry/Bohr)."""
    forces = []
    with open(forces_file) as f:
        lines = f.readlines()

    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) >= 5:
            fx = float(parts[2])  # fx_ry_bohr
            fy = float(parts[3])  # fy_ry_bohr
            fz = float(parts[4])  # fz_ry_bohr
            forces.append((fx, fy, fz))

    return forces

def run_abinit(input_file, abinit_exe):
    """Run ABINIT calculation."""
    print(f"Running ABINIT with {input_file}...")
    result = subprocess.run(
        [abinit_exe, input_file],
        capture_output=True,
        text=True,
        cwd=Path(input_file).parent
    )
    if result.returncode != 0:
        print(f"ABINIT error: {result.stderr}")
        return False
    return True

def run_dft_zig(config_file, dft_zig_exe):
    """Run DFT-Zig calculation."""
    print(f"Running DFT-Zig with {config_file}...")
    result = subprocess.run(
        [dft_zig_exe, config_file],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"DFT-Zig error: {result.stderr}")
        return False
    return True

def compare_forces(abinit_forces, dft_zig_forces, tolerance=0.01):
    """Compare forces between ABINIT (Ha/Bohr) and DFT-Zig (Ry/Bohr)."""
    if len(abinit_forces) != len(dft_zig_forces):
        print(f"ERROR: Number of atoms mismatch: ABINIT={len(abinit_forces)}, DFT-Zig={len(dft_zig_forces)}")
        return False

    print("\n" + "=" * 70)
    print("Force Comparison (Ry/Bohr)")
    print("=" * 70)
    print(f"{'Atom':>4} {'Component':>8} {'ABINIT':>12} {'DFT-Zig':>12} {'Diff':>12} {'Status':>8}")
    print("-" * 70)

    max_diff = 0.0
    all_pass = True

    for i, (abi, dft) in enumerate(zip(abinit_forces, dft_zig_forces)):
        for j, comp in enumerate(['x', 'y', 'z']):
            # Convert ABINIT from Ha/Bohr to Ry/Bohr
            abi_ry = abi[j] * RY_PER_HA
            dft_ry = dft[j]
            diff = abs(abi_ry - dft_ry)
            max_diff = max(max_diff, diff)
            status = "PASS" if diff < tolerance else "FAIL"
            if diff >= tolerance:
                all_pass = False
            print(f"{i:>4} {comp:>8} {abi_ry:>12.6f} {dft_ry:>12.6f} {diff:>12.6f} {status:>8}")

    print("-" * 70)
    print(f"Max difference: {max_diff:.6f} Ry/Bohr")
    print(f"Tolerance: {tolerance:.6f} Ry/Bohr")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    return all_pass

def main():
    # Paths
    script_dir = Path(__file__).parent
    abinit_input = script_dir / "abinit_forces.in"
    abinit_output = script_dir / "abinit_forceso.abo"
    dft_zig_config = script_dir / "silicon_relax.toml"
    dft_zig_forces = script_dir.parent.parent / "out" / "silicon_relax" / "relax_forces.csv"

    # Executables (adjust paths as needed)
    abinit_exe = str(script_dir.parent.parent / "out" / "abinit" / "build" / "abinit-10.4.7" / "src" / "98_main" / "abinit")
    dft_zig_exe = str(script_dir.parent.parent / "zig-out" / "bin" / "dft_zig")

    # Parse command line
    run_abinit_flag = "--run-abinit" in sys.argv
    run_dft_zig_flag = "--run-dft-zig" in sys.argv
    tolerance = 0.01  # Ry/Bohr

    for arg in sys.argv:
        if arg.startswith("--tolerance="):
            tolerance = float(arg.split("=")[1])

    # Run calculations if requested
    if run_abinit_flag:
        if not run_abinit(str(abinit_input), abinit_exe):
            sys.exit(1)

    if run_dft_zig_flag:
        if not run_dft_zig(str(dft_zig_config), dft_zig_exe):
            sys.exit(1)

    # Parse forces
    print("\nParsing ABINIT forces...")
    if not abinit_output.exists():
        print(f"ERROR: ABINIT output file not found: {abinit_output}")
        print("Run with --run-abinit to generate it")
        sys.exit(1)
    abinit_forces = parse_abinit_forces(abinit_output)
    print(f"Found {len(abinit_forces)} atoms in ABINIT output")

    print("\nParsing DFT-Zig forces...")
    if not dft_zig_forces.exists():
        print(f"ERROR: DFT-Zig forces file not found: {dft_zig_forces}")
        print("Run with --run-dft-zig to generate it")
        sys.exit(1)
    dft_forces = parse_dft_zig_forces(dft_zig_forces)
    print(f"Found {len(dft_forces)} atoms in DFT-Zig output")

    # Compare
    success = compare_forces(abinit_forces, dft_forces, tolerance)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
