# ppgen Notes

## References

- Troullier and Martins, "Efficient pseudopotentials for plane-wave
  calculations", Phys. Rev. B 43, 1993 (1991),
  doi:10.1103/PhysRevB.43.1993.
- Hamann, "Optimized norm-conserving Vanderbilt pseudopotentials",
  Phys. Rev. B 88, 085117 (2013), doi:10.1103/PhysRevB.88.085117.
- van Setten et al., "The PseudoDojo: Training and grading a 85 element
  optimized norm-conserving pseudopotential table", Comput. Phys. Commun. 226,
  39-54 (2018), doi:10.1016/j.cpc.2018.01.012.

## Current Direction

ABINIT is the consumer-side reference for file formats and solid-state
validation. For generator physics, the closer reference is ONCVPSP: multi
projectors, positive-energy scattering states, and explicit transferability
checks.

The current ppgen implementation remains Troullier-Martins based, so the
immediate quality gate is not to copy ONCV wholesale. The next zero-debt step is
to make the scattering error visible before changing the coefficient solver:
compare all-electron and pseudo logarithmic derivatives at a radius outside all
cutoffs, then use the result to guide coefficient and projector changes.

## Silicon Diagnostic

`ppgen --log-deriv out.tsv` writes a tab-separated diagnostic table for the Si
generator path. Each row compares the all-electron and pseudized regular radial
solution via `u'(r_match) / u(r_match)` for one channel and reference energy.
The sampled energy grid is explicit: `--log-deriv-min-ry`,
`--log-deriv-max-ry`, and `--log-deriv-step-ry` default to the original
0.0, 0.4, ..., 1.6 Ry grid, while the Si benchmark uses a finer 0.2 Ry grid.

Rows with `status != ok` are intentionally preserved in the output rather than
silently dropped. They mark energies/channels where the current pseudo channel
does not produce a finite logarithmic derivative at the match radius. As of this
note, the 4d reference channel is the visible failure point, and the p channel
around 0.4 Ry is the largest valid mismatch.

The Si default currently uses a d-channel local potential. With d-local output,
the Si generator emits a single d reference channel by default. If
`--d2-energy-ry` is explicitly provided, the local channel is selected by
`local_n/local_l` and the second d reference remains in the UPF nonlocal block.
For the current Si ratchet this optional second d projector worsens the average
band MSE, so the benchmark leaves it disabled.

For the unoccupied p reference projector, a positive scattering energy is more
stable than the zero-energy choice. In the current Si benchmark,
`--p-ref-energy-ry 0.8` keeps the logarithmic derivative ratchet passing after
NLCC is included in the XC unscreening path. Higher values around `2.6 Ry`
improve the band MSE alone but break the scattering log-derivative check, so
they are not used. The previous `0.2 Ry` reference left a much larger
p-projector block, about `8.1e4`, so the benchmark ratchets the absolute
`PP_DIJ` limit to `2e4`.

The Si PBE generator now writes an AE-core-derived nonlinear core correction by
default: `--nlcc-charge 0.7241414335 --nlcc-radius 0.95`. This is an explicit
partial-core model, not the full frozen core charge. ppgen builds it from the
occupied AE core orbitals that are not part of the valence channel set, applies
a smooth radial suppression near the origin, and normalizes the resulting
density to the configured partial-core charge. The local UPF reference uses
`core_correction="T"` with a partial core charge of about `0.724e` and an RMS
radius of about `0.978 bohr`. After the NLCC density is included in the XC
unscreening path, this setting gives a Si band MSE of about `6040 meV^2` with a
gap difference of about `13 meV`. The parameters are therefore part of the
current Si quantitative ratchet and should be replaced only by a more physical
pseudo-core construction with equal or better regression results.

The Si local potential is also smoothed by default with
`--local-smooth-radius 1.3`. The local channel still comes from the d reference
channel outside the smoothing radius, but the inner region is blended to a
finite value with a zero-slope smootherstep. This lowers the local solid-q
form-factor RMS from about `10.9` to about `6.3` and the Si band MSE from about
`6040 meV^2` to about `4528 meV^2` while preserving the scattering
log-derivative ratchet.

For multiple projectors in the same angular-momentum channel, the KB coefficient
matrix is built from the Hermitian part of `<beta_i|phi_j>`. This keeps the
written UPF nonlocal block Hermitian, which is the solid-state Hamiltonian
contract, while preserving the exact inversion target for the symmetric
projector-overlap component. Directly inverting a nonsymmetric same-l overlap
block was tested for Si and left a slightly worse band MSE plus a much more
ill-conditioned p-projector block.

The UPF beta tables are written with each projector's channel cutoff. Extending
the written beta support to the maximum of channel and local cutoffs was tested
for d-local Si and made the solid-state band comparison much worse even when the
atomic logarithmic derivatives were unchanged. The current ratchet therefore
keeps the channel-support convention explicit and covered by a unit test.

The Si regression also counts pole mismatches in the sampled logarithmic
derivatives. A pole is counted when adjacent energies have opposite signs and at
least one side has a large absolute logarithmic derivative. This is a ghost-state
screen, not a proof: a mismatch is treated as a ghost candidate and fails the
ratchet, while a clean count means only that this finite energy grid did not
detect an AE/PS pole-count discrepancy.

`benchmarks/silicon/diagnose_ppgen.py` adds the current channel-wise comparison
layer used to decide the next generator change. It writes per-band errors,
per-l logarithmic derivative summaries, same-l `D_ij` block conditioning,
q-space local short-range form factors, beta-projector form factors, and
high-q projector hardness against `pseudo/Si.upf`. These are diagnostics rather
than new pass/fail thresholds: the ratchet still fails only on the explicit
band, gap, `D_ij`, and logarithmic-derivative limits in `test_ppgen.py`.
The form-factor diagnostic is split into low, mid, high, and `0..5 bohr^-1`
solid-state q windows. The `objective.absolute_score` field is a transparent
absolute RMS score, not a fitted physical functional; it exists to compare
candidate generator changes before promoting any new threshold.

The intended ONCV-like direction is now measurable before implementation:
minimize scattering and form-factor error per angular-momentum channel, then use
projector hardness to reject changes that merely fit the band path by making
projectors too hard in reciprocal space. The next algorithmic step should be a
proper projector optimization objective, not ad-hoc tuning of one scalar cutoff.
