# ADR-0007: PAW D^hat に V_loc を含める（QE newd 規約）

## Status
Accepted

## Context
PAW の D^hat 計算で V_eff をどう定義するか:
- `V_eff = V_H + V_xc` （V_loc を含めない）
- `V_eff = V_loc + V_H + V_xc` （QE の newd 規約）

V_loc を含めないと、応力テンソルが ~26 GPa ずれた。

## Decision
**QE newd 規約に従い V_eff = V_loc + V_H + V_xc** とする。

## Consequences
- D^hat force は V_HXC（V_loc なし）を使い、V_loc は self-consistent D_full を通じて取り込まれる
- local stress は augmented density (ρ̃+n̂) を使う必要がある（ρ̃ だけだと P=26 GPa、ρ̃+n̂ で P=0.24 GPa）
- augmentation stress の V_eff も V_loc を含める（D^hat と整合）
- Si PAW ecut=44 grid_scale=2: DFT-Zig P=2.61 GPa vs QE P=2.59 GPa
