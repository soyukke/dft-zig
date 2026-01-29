# Linear Scaling DFT Benchmarks

## Nonlocal Pseudopotential Benchmark

Tests the nonlocal (KB projector) implementation with s-only and sp basis types.

### Results

For 8 Si atoms (conventional cell):

| Basis | Nonlocal (Ry) | Total (Ry) | Time |
|-------|---------------|------------|------|
| s_only | +32.6 | -398.0 | ~39ms |
| sp | +8.2 | -422.4 | ~53ms |

### Reference Values

See `nonlocal_reference.json` for expected values and tolerances.
