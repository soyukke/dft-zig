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

# ppgen smoke/regression test (generate Si PBE UPF, then compare bands vs ABINIT)
just test-ppgen-silicon

# Plot only (use existing outputs)
python3 plot_comparison.py
```

## Band Comparison

![Silicon band comparison](band_comparison.png)

Si (NC-PP, PBE, ecut=30 Ry, 4x4x4 k-mesh) — DFT-Zig vs ABINIT.
Band energies match within a few meV across the Brillouin zone.

## ppgen Reference Files

The local reference UPF for the main Si benchmark is `../../pseudo/Si.upf`
(ONCV PBE, norm-conserving). The saved ABINIT band baseline under
`baseline/` was generated with that pseudopotential.

`test_ppgen.py` generates `out_ppgen/Si_ppgen_PBE.upf` with ppgen using
bound s/p channels, unoccupied p/d reference projectors, a smoothed local
potential, and an AE-core-derived partial-core NLCC, runs DFT-Zig using
`dft_zig_ppgen.toml`, and compares the
VBM-aligned bands against the saved ABINIT baseline. This is a quantitative
ratchet for ppgen, not a claim that ppgen already matches ONCV quality.

Current ppgen Si PBE ratchet: band-gap difference <= 75 meV and average
VBM-aligned band MSE <= 4,900 meV^2 against the saved ABINIT baseline.
The same test also checks that the generated `PP_DIJ` coefficients remain below
2e4 in absolute value, verifies the `PP_DIJ` matrix is Hermitian to 1e-8,
writes `out_ppgen/Si_ppgen_logderiv.tsv` on a 0.2 Ry energy grid, verifies
the NLCC partial-core charge is about 0.724e, and checks the atomic scattering
diagnostic: valid log-derivative max delta <= 0.75, RMS delta <= 0.17, invalid
sample count = 0, and AE/PS pole mismatch count = 0.

`diagnose_ppgen.py` runs as part of the ppgen benchmark and writes three
diagnostic artifacts under `out_ppgen/`:

- `ppgen_diagnostics.json`: per-band MSE, channel-wise log-derivative summary,
  `PP_DIJ` block conditioning, local/projector form-factor errors, and
  high-q projector hardness. The form-factor section also reports low, mid,
  high, and solid-state q-window scores so generator changes can be compared
  before tightening pass/fail thresholds.
- `ppgen_form_factors.csv`: q-space local short-range and beta-projector
  comparisons against `../../pseudo/Si.upf`.
- `ppgen_band_errors.csv`: VBM-aligned band errors at each k point.

Generator notes and references are in `../../docs/ppgen-notes.md`.
