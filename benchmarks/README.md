# Benchmarks

Validation benchmarks comparing DFT-Zig against ABINIT and Quantum ESPRESSO.

## Band Structure: Silicon

![Silicon band comparison](silicon/band_comparison.png)

Si (NC-PP, PBE, ecut=30 Ry, 4x4x4 k-mesh) — DFT-Zig vs ABINIT.

## Band Structure: Aluminum (11-electron)

![Al 11e band comparison](aluminum_11e/band_comparison_11e.png)

Al with semi-core states (11e ONCV, PBE) — valence bands match within 2 meV.

## Equation of State: Silicon

![Silicon EOS](silicon_eos/eos_comparison.png)

Total energy vs volume — Birch-Murnaghan fit comparison.

## Regression Tests

Run all tests with `just test-regression`. Each benchmark directory contains:

- `dft_zig.toml` — DFT-Zig input configuration
- `baseline/` — ABINIT/QE reference data (CSV/JSON)
- `plot_comparison.py` — Comparison script (also runs regression check via `--run-check`)

| Test | Command | System | Comparison |
|------|---------|--------|------------|
| Silicon band | `just test-silicon` | Si (diamond) | ABINIT band energies |
| Graphene band | `just test-graphene` | Graphene | ABINIT band energies |
| GaAs band | `just test-gaas` | GaAs (zincblende) | ABINIT band energies |
| Cu band | `just test-cu` | Cu (FCC) | ABINIT band energies |
| Fe band | `just test-fe` | Fe (BCC, spin) | ABINIT band energies |
| k-parallel | `just test-silicon-kparallel` | Si | serial vs parallel |
| LOBPCG parallel | `just test-lobpcg-parallel` | Si | serial vs parallel |
| EOS | `just test-eos` | Si | ABINIT total energies |
| DFPT phonon | `just test-dfpt` | Si | ABINIT phonon frequencies |
| Molecule SCF | `just test-molecule` | Small molecules | PySCF energies |

## Directory Structure

```
benchmarks/
├── silicon/          # Si NC-PP band structure
├── graphene/         # Graphene band structure
├── gaas/             # GaAs band structure
├── cu/               # Cu band structure
├── fe/               # Fe BCC spin-polarized band structure
├── aluminum/         # Al 3e band structure
├── aluminum_11e/     # Al 11e (semi-core) band structure
├── silicon_dfpt/     # DFPT phonon frequencies
├── silicon_eos/      # Equation of state
├── silicon_paw/      # PAW (QE comparison)
├── molecule/         # Molecular SCF (GTO basis)
├── relax/            # Structural relaxation / forces
├── linear_scaling/   # O(N) method tests
└── qm9_pipeline/     # QM9 molecule set
```
