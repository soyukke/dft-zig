# ADR-0001: Rydberg 原子単位系の採用

## Status
Accepted

## Context
DFT コードの内部単位系を決める必要がある。候補:
- **Hartree 原子単位** (ABINIT, GPAW 等)
- **Rydberg 原子単位** (Quantum ESPRESSO, UPF 形式)
- **eV/Å** (VASP)

UPF 擬ポテンシャルファイルは Rydberg 単位で記述されており、Quantum ESPRESSO も Rydberg を採用している。ABINIT は Hartree 単位のため、比較時に変換 (1 Ha = 2 Ry) が必要。

## Decision
**Rydberg 原子単位**を採用する。

## Consequences
- UPF ファイルの読み込みで単位変換が不要
- QE との比較が直接可能
- ABINIT との比較時は Ha→Ry 変換が必要（初期に多数の単位取り違えバグが発生した教訓あり）
- 運動エネルギー演算子は `-∇²`（Hartree 系では `-½∇²`）
