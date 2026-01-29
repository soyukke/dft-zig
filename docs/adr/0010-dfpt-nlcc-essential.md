# ADR-0010: DFPT で NLCC の完全な取り扱い

## Status
Accepted

## Context
Si ONCV 擬ポテンシャルは NLCC あり (core_correction="T")。NLCC を無視した DFPT フォノン計算では光学モードが 793 cm⁻¹ と ABINIT の 755 cm⁻¹ から大きくずれた。

## Decision
DFPT で NLCC を完全に取り扱う。3つの修正が必要:

1. **f_xc の評価密度**: ρ_val だけでなく ρ_val + ρ_core で評価
2. **Sternheimer V^(1)_xc**: f_xc × (ρ^(1)_val + ρ^(1)_core) — コア電荷も原子変位に応答
3. **動的行列の NLCC 項**: cross 項 (∫f_xc × ρ^(1)_total × ρ^(1)_core) + self 項 (V_xc × ρ^(2)_core)

ρ^(1)_core の公式: V^(1)_loc と同じ形で ρ_core_form を使用。

## Consequences
- Si: 754.83 cm⁻¹ (ABINIT: 754.73) — 0.01% の精度
- ASR 違反: ~8e-6 Ry（十分小さい）
- NLCC なしの擬ポテンシャルでは自動的にスキップ
