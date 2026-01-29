# Graphene Benchmark

Benchmark comparison between DFT-Zig and ABINIT with identical calculation parameters.

## System

- Material: Graphene (hexagonal)
- Lattice constant: a = 2.46 A, c = 10 A (vacuum)
- Primitive cell: 2 atoms (C)
- Electrons: 8

## Calculation Parameters

### Fair Comparison Settings

| Parameter | DFT-Zig | ABINIT |
|-----------|---------|--------|
| ecut | 15 Ry | 7.5 Ha (= 15 Ry) |
| k-mesh (SCF) | 6x6x1 | 6x6x1 |
| Grid | 16x16x64 | auto |
| Mixing beta | 0.3 | 0.3 |
| SCF convergence | 1e-6 | 1e-6 |
| Bands | 8 | 8 |
| Band path | G->K->M->G | G->K->M->G |
| **Band k-points** | **61** | **61** |
| Eigenvalue solver | iterative (LOBPCG) | iterative (LOBPCG, wfoptalg=14) |
| Eigensolver tol | 1e-6 | 1e-6 |
| Smearing | - | Fermi-Dirac (0.01 Ha) |

## Files

### DFT-Zig
- `dft_zig.toml` - SCF + band calculation

## Running

```bash
cd benchmarks/graphene
source ../../.venv/bin/activate

# Regression test (DFT-Zig vs saved ABINIT baseline)
python3 plot_comparison.py --run-check

# Plot only (use existing outputs)
python3 plot_comparison.py
```

## Results

Environment: Apple Silicon (M-series), multi-threaded (16 threads)

### Performance Comparison

| Code | k-points | Solver | Wall Time | CPU Time | Ratio |
|------|----------|--------|-----------|----------|-------|
| DFT-Zig | 61 | LOBPCG | 17.2s | 64.8s | 1.9x slower |
| ABINIT | 61 | LOBPCG | 9.3s | 63.5s | **1.0x** |

**Note**: ABINIT is faster for graphene (vs silicon where DFT-Zig was 1.25x faster). This may be due to:
- Graphene's large vacuum region (c=10 A) requiring more plane-waves in z-direction
- Grid size differences (DFT-Zig: 16x16x64, ABINIT: auto-optimized)

### Dirac Cone Gap at K Point

| Code | Gap (meV) |
|------|-----------|
| DFT-Zig | 86.1 |
| ABINIT | 0.0 |
| **Difference** | **86 meV** |

Note: Ideal graphene should have zero gap at K point (Dirac cone). The 86 meV gap in DFT-Zig is due to the missing symmetry block-diagonalization (see Known Issues).

## Known Issues

### Dirac Cone Gap at K Point
- **Symptom**: Non-zero gap (86 meV) at K point Dirac cone
  - ABINIT: 0.0 meV (correct - bands degenerate due to C3v symmetry)
  - DFT-Zig: 86 meV (incorrect - degeneracy broken)
- **Cause**: C3v symmetry block-diagonalization not implemented
  - K point has C3v symmetry (3-fold rotation + 3 vertical mirrors)
  - ABINIT enforces degeneracy via symmetry
  - DFT-Zig only implements Cs (single mirror) for M-G path
- **Impact**: Artificial gap at Dirac point
- **Fix**: Implement C3v symmetry block-diagonalization for K point

### M-G Band Crossing Issue (separate from K point)
- **Symptom**: Pi band near-crossing gap on M-G path may be overestimated
- **Status**: Cs symmetry (sigma_y mirror) is implemented via `use_symmetry = true`
- **Note**: Currently disabled (`use_symmetry = false`) pending validation

## Notes

- ABINIT uses Hartree units internally (1 Ha = 2 Ry = 27.2 eV)
- DFT-Zig uses Rydberg units (same as UPF files)
- Both codes use the same C.upf pseudopotential
- Graphene is a semimetal with Dirac cone at K point
