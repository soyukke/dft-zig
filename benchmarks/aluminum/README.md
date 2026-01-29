# Aluminum Benchmark

Band structure comparison between DFT-Zig and ABINIT.

## System

- Material: Aluminum (FCC)
- Lattice constant: a = 4.05 A
- Primitive cell: 1 atom
- Valence electrons: 3 (ONCV NC, NLCC)
- XC: PBE

## Running

```bash
cd benchmarks/aluminum
source ../../.venv/bin/activate

# Run DFT-Zig and plot comparison with ABINIT baseline
python3 plot_comparison.py --run

# Plot only (use existing outputs)
python3 plot_comparison.py
```
