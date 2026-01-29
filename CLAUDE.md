# DFT-Zig Development Guide

## Units
- DFT-Zig: **Rydberg** (1 Ry = 13.6057 eV)
- ABINIT: **Hartree** (1 Ha = 27.2114 eV = 2 Ry)
- UPF files: **Rydberg**

## Build & Test

```bash
nix develop          # enter dev environment (sets FFTW paths)
just build           # ReleaseFast build with FFTW
just test            # all tests (unit + regression)
just test-regression # regression tests only (ABINIT/QE comparison)
just fmt             # zig fmt src
```

## FFT / Band Solver

- Use `fft_backend = "fftw"` (other backends are too slow for production)
- Use `solver = "iterative"` (LOBPCG) for band calculations; `"dense"` is very slow
