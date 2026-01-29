# ADR-0012: 動径テーブルによるフォームファクタキャッシング

## Status
Accepted

## Context
`radialProjector(|G+k|)` は各 (beta, G-vector) ペアに対して ~700 動径点の球 Bessel 関数を計算する。SCF の各反復で数千回呼ばれ、主要ボトルネックになっていた。

同様に `localVq`, `rhoAtomG`, `rhoCoreG` も毎回動径積分を行っていた。

## Decision
事前計算テーブルを導入し、線形補間でフォームファクタを取得する:

- **RadialTableSet**: 非局所プロジェクタ用。2048 点、一度構築して全 k 点で共有
- **LocalFormFactorTable**: 局所ポテンシャル用。4096 点
- **RadialFormFactorTable**: rhoAtom/rhoCore 用。4096 点

## Consequences
- SCF 最初の反復で ~1s 節約（RadialTableSet を SCF k 点に渡す）
- 力の計算でも再利用（NonlocalContext, 局所力, NLCC 力）
- relax ステップ間でもテーブルを再利用（原子種が変わらない限り）
- テーブル構築は一度だけ。メモリ使用量は無視できるレベル
