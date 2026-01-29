# ADR-0006: PAW で m 分解 D_ij を採用

## Status
Accepted

## Context
PAW の D_ij テンソルを動径量子数 (nb×nb) で扱うか、磁気量子数 m まで分解した (mt×mt) で扱うか。

動径のみの D_ij では Gaunt 係数の m 収縮で `Σ_m G(l,m,l,m,0,0) = (2l+1)/√(4π)` という因子が現れ、l>0 ペアで不正確になる。m 収縮した D^hat を動径 D として使うと物理的に間違いとなる。

## Decision
**m 分解 D_ij (mt×mt)** をハミルトニアン・応力で使用する。動径 D_ij (nb×nb) は力の計算用に後方互換として保持。

## Consequences
- `dij_m_per_atom` が NonlocalSpecies に格納される
- `applyNonlocalPotential` は m 分解 D がある場合に分岐
- D^0 は m 対角に展開、D^hat は full Gaunt、D^xc は Lebedev 角度積分で m 分解
- `updatePawDij` が動径・m 分解の両方を出力
- 立方対称 Si では動径との差は ~0.003 GPa（小さいが原理的に正しい）
