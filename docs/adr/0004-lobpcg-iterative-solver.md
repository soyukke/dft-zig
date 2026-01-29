# ADR-0004: LOBPCG を標準固有値ソルバとして採用

## Status
Accepted

## Context
平面波 DFT の固有値問題の解法として:
- **密行列対角化** (zheev): 確実だが O(N³) でスケールしない
- **LOBPCG**: 反復法、必要なバンド数分だけ計算
- **CG (band-by-band)**: バンドごとの共役勾配法

## Decision
**LOBPCG を標準ソルバ**とする。密行列対角化は小規模系のフォールバック、CG はオプション。

## Consequences
- サブスペースサイズは `4*nbands + 8` が必要（デフォルト `2*nbands+4` では頻繁に再起動し収束が悪い）
- `added == 0`（全個別残差 < tol）を収束として扱う必要がある（そうしないと無限再起動ループ）
- SCF での eigensolver tol は緩め（1e-4 程度）が良い。tol=1e-6 は過補正を引き起こし Pulay が不安定化（49 vs 11 反復）
- Modified Gram-Schmidt を使用（CGS は反復回数が増加: 6486→7039 apply_calls）
- 直列/並列版を実装。並列版はワークスペースプールが必要
