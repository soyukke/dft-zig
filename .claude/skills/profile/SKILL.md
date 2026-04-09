---
name: profile
description: SCFプロファイリング実行。ボトルネック分析と最適化提案
argument-hint: "<config.toml>"
allowed-tools: Bash(just *), Bash(./zig-out/*), Read, Glob, Grep
---

DFT-Zig計算をプロファイル付きで実行し、パフォーマンスボトルネックを分析する。

## 手順

1. 指定されたTOMLファイルを読み込む
2. `[scf]` セクションに `profile = true` が設定されていなければ、一時コピーを作成して追加
3. `just build && ./zig-out/bin/dft_zig <config.toml>` で実行
4. 出力からプロファイル情報を抽出:
   - SCFイテレーションごとの時間内訳
   - apply_h (local / nonlocal の比率)
   - build_potential の時間
   - FFT scatter/gather の時間
   - LOBPCG収束ステップ数
5. ボトルネックを特定し、最適化の提案を行う:
   - local FFTが支配的 → グリッドサイズ確認
   - nonlocalが重い → RadialTableSetの利用確認
   - build_potentialが重い → mixing設定確認
   - 全体が遅い → ecut/k-meshの妥当性確認

## 分析のポイント

- Si ecut=15 24^3 の典型値: local=87%, nonlocal=2%, build_potential=35%
- apply_h内: iFFT 48%, FFT 46%, scatter 1.6%, V*psi 3.7%
- FFTW使用推奨（他のバックエンドは大幅に遅い）

$ARGUMENTS
