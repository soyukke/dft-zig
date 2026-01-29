# ADR-0013: 誘電テンソルに 2nd Sternheimer 式を採用

## Status
Accepted

## Context
誘電テンソル ε∞ の計算で複数の公式を試行した:

1. Gonze 2n+1 公式 → 符号エラーで不正確
2. 非 SCF 摂動論 → ABINIT と 0.03% 一致するが局所場効果 (LFE) がなく 4% 過大評価
3. 2nd Sternheimer (Baroni 16π/Ω 公式) + 自己無撞着 efield response → LFE を含み正確

途中で Y_lm 符号規約の不一致による異方性バグも発見・修正した。

## Decision
**自己無撞着 efield response による 2nd Sternheimer 式**を採用。

## Consequences
- 局所場効果を含む正確な ε∞
- ABINIT と 0.03% 一致
- ddk Sternheimer を全 k 点で解く必要がある（計算コスト増）
- 非局所 ddk は動径 + 角度微分の両方が必要
