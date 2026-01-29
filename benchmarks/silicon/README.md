# Silicon Benchmark

Band structure comparison between DFT-Zig and ABINIT.

## System

- Material: Silicon (diamond, FCC)
- Lattice constant: a = 5.431 A
- Primitive cell: 2 atoms
- Electrons: 8
- XC: PBE
- Pseudopotential: Si ONCV (NC)

## Running

```bash
cd benchmarks/silicon
source ../../.venv/bin/activate

# Regression test (DFT-Zig vs saved ABINIT baseline)
python3 plot_comparison.py --run-check

# Plot only (use existing outputs)
python3 plot_comparison.py
```

## Band Comparison

![Silicon band comparison](band_comparison.png)

Si (NC-PP, PBE, ecut=30 Ry, 4x4x4 k-mesh) — DFT-Zig vs ABINIT.
Band energies match within a few meV across the Brillouin zone.
