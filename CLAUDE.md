# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Units

- DFT-Zig: **Rydberg** (1 Ry = 13.6057 eV)
- ABINIT: **Hartree** (1 Ha = 27.2114 eV = 2 Ry)
- UPF files: **Rydberg**

## Build & Test

```bash
nix develop                              # enter dev environment (sets FFTW/BLAS paths)
just build                               # ReleaseFast build with FFTW
just test                                # all tests (unit + regression)
just test-unit                           # unit tests only (no FFTW needed)
just test-unit-fftw                      # unit tests with FFTW
just test-regression                     # all regression tests (ABINIT/QE comparison)
just fmt                                 # zig fmt src
just clean                               # remove build artifacts
```

### Running a Single Unit Test

```bash
zig build test -Dtest-filter="spacegroup silicon"
```

### Running Individual Regression Tests

```bash
just test-silicon          # Si NC (LDA, band structure)
just test-graphene         # Graphene (Dirac point)
just test-gaas             # GaAs (PBE, semi-core d)
just test-cu               # Cu (metallic)
just test-fe               # Fe BCC (spin-polarized)
just test-aluminum         # Al
just test-aluminum-11e     # Al 11e (semi-core)
just test-paw              # Si PAW (vs QE)
just test-dfpt             # DFPT phonons (Gamma, X, L)
just test-eos              # Si equation of state
just test-molecule         # Gaussian basis molecules (vs PySCF)
just test-silicon-kparallel    # k-point parallelism
just test-lobpcg-parallel      # LOBPCG block parallelism
```

### Running a Calculation

```bash
just build && ./zig-out/bin/dft_zig examples/silicon.toml
```

## FFT / Band Solver

- Use `fft_backend = "fftw"` (other backends are too slow for production)
- Use `solver = "iterative"` (LOBPCG) for band calculations; `"dense"` is very slow

## Design Principles

- **後方互換性を入れない**: 負債になるため、古いインターフェースは完全に削除する。互換レイヤーや非推奨ラッパーは作らない。
- Band k-path は `path = "auto"` または `path = "G-X-W-K-G-L"` 形式のみ（旧 `[[band.path]]` は廃止済み）

## Architecture

### Execution Flow

```
main.zig → load TOML config → load XYZ structure → dft.run()
  dft.run():
    1. Load pseudopotentials, build species/atom data
    2. relax.run()   (if enabled — runs SCF iteratively)
    3. scf.run()     (if enabled — self-consistent field)
    4. stress        (if enabled)
    5. band          (reuses converged SCF potential)
    6. dfpt          (uses converged SCF wavefunctions + XC kernel)
```

### Key Module Layout (`src/features/`)

| Module | Role |
|--------|------|
| `dft/` | Top-level orchestrator (`dft.run()`): coordinates SCF → stress → band → DFPT pipeline |
| `scf/` | SCF loop with ~20 submodules: Hamiltonian application, band solving, density/potential building, Pulay mixing, k-point threading |
| `band/` | Band structure along k-path, reuses SCF potential |
| `dfpt/` | Density functional perturbation theory: Sternheimer equations, dynamical matrix, phonon dispersion |
| `relax/` | Structural relaxation (BFGS/CG/SD), warm-starts subsequent SCF |
| `forces/` | Hellmann-Feynman forces (local, nonlocal, Ewald, NLCC, PAW D^hat) |
| `stress/` | Stress tensor (local, nonlocal, kinetic, Ewald, GGA, NLCC, PAW augmentation) |
| `paw/` | PAW: compensation charges, D matrix, on-site energy, Lebedev angular integration |
| `hamiltonian/` | Hamiltonian construction, nonlocal context, type definitions (AtomData, PotentialGrid, SpeciesEntry) |
| `pseudopotential/` | UPF parsing, form factors, radial tables |
| `plane_wave/` | Plane wave basis set: G-vector generation, cutoff, basis management |
| `xc/` | Exchange-correlation functionals (LDA Perdew-Zunger, GGA PBE) |
| `symmetry/` | Crystal point group detection, IBZ k-point reduction |
| `ewald/` | Ewald summation for ion-ion energy and forces |
| `linalg/` | LOBPCG eigensolver, BLAS/LAPACK wrappers |
| `fft/` | FFT backend abstraction (FFTW/Zig/vDSP/Metal) |
| `grid/` | Real-space FFT grid utilities |
| `kpoints/` | k-mesh generation, Monkhorst-Pack |
| `kpath/` | High-symmetry k-path generation for band structure |
| `dos/` | Density of states / projected DOS |
| `structure/` | Crystal structure, atomic positions, lattice vectors |
| `gto_scf/` | Gaussian basis molecular SCF (HF/DFT with STO-3G, 6-31G, 6-31G**) |
| `vdw/` | DFT-D3(BJ) van der Waals correction |
| `math/` | Numerical utilities (spherical harmonics, Bessel functions, integration) |

### Parallelization

- **k-point threading**: SCF, band, DFPT distribute k-points across threads
- **LOBPCG block parallelism**: Within eigensolver via `num_workspaces > 1`
- Config: `n_threads` in `[scf]`

### Caching Strategy

- `ApplyContext` caches NonlocalContext, PwGridMap, FFT plans per k-point across SCF iterations
- `RadialTableSet` pre-computed lookup tables for radial projectors, built once and shared
- `FormFactorTable` (local potential, rhoAtom, rhoCore) with linear interpolation over 4096 points
- Relax passes converged density and caches to post-relax SCF for warm-starting

### Regression Test Infrastructure

Tests live in `benchmarks/`. Each benchmark has a TOML config and Python script that:
1. Builds DFT-Zig via `just build`
2. Runs the calculation
3. Compares against saved ABINIT/QE baseline CSV data
4. Requires Python venv at `.venv/` (created by `nix develop`)

Always run `just test-regression` after modifying SCF, Hamiltonian, band, nonlocal, or force/stress code.
