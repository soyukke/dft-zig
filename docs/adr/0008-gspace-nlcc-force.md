# ADR-0008: NLCC 力を G 空間で計算

## Status
Accepted

## Context
NLCC (非線形コア補正) の力の計算方法:
- **実空間**: ∫V_xc(r) ∂ρ_core(r)/∂R dr — bandwidth-limited な V_xc と unfiltered な ρ_core 導関数のエイリアシングが発生
- **G空間**: Σ_G ρ_core_form(|G|) G_α (V_xc_r sin(GR) + V_xc_i cos(GR))

## Decision
**G 空間 NLCC 力**を採用。V_xc(r) → V_xc(G) を FFT で変換し、G 空間で総和。

## Consequences
- エイリアシングが完全に除去され、QE と 4e-5 Ry/Bohr 以内で一致
- DFPT でも同様に NLCC を G 空間で処理（Si 光学モード: 793→755 cm⁻¹）
