---
name: test-unit
description: ユニットテスト実行。フィルタ指定でテスト絞り込み可能。テスト失敗時はエラー箇所を特定
argument-hint: "[filter|fftw|all]"
allowed-tools: Bash(*), Read, Grep
---

DFT-Zigのユニットテストを実行する。

## 引数の解析ルール

| 引数 | コマンド | 説明 |
|------|---------|------|
| (なし) / `all` | `just test-unit` | 全ユニットテスト（FFTWなし） |
| `fftw` | `just test-unit-fftw` | 全ユニットテスト（FFTW付き、`nix develop`環境必要） |
| その他の文字列 | `zig build test -Dtest-filter="<filter>"` | フィルタ指定テスト |

「その他の文字列」の判定: `all` でも `fftw` でもなければ全てフィルタとして扱う。
スペースを含む場合もそのまま `-Dtest-filter="spacegroup silicon"` のように渡す。

## フィルタ例

- `"spacegroup silicon"` — シリコンの空間群テスト
- `"paw"` — PAW関連テスト
- `"lobpcg"` — LOBPCG関連テスト
- `"fft"` — FFT関連テスト
- `"nonlocal"` — 非局所ポテンシャル関連テスト
- `"radial"` — 動径関数テスト
- `"pulay"` — Pulay混合テスト
- `"symmetry"` — 対称性テスト

## 手順

1. 引数を解析してコマンドを決定
2. コマンドを実行（タイムアウト: 300秒）
3. 結果を報告:
   - **成功時**: `zig build test` は成功時に出力なし（exit code 0）。「全テストPASS」と報告
   - **失敗時**: エラー出力からテスト名と失敗行を特定し報告。該当ソースコードを読んで原因を調査

## 失敗時の診断

テスト失敗の出力例:
```
src/features/scf/test_xxx.zig:123:45: error: ...
```

失敗した場合:
1. エラー出力から失敗テスト名とファイル:行番号を抽出
2. 該当ソースを Read で確認
3. 期待値と実際の値を比較
4. 原因の推測と修正案を提示

## FFTW テストの注意

`fftw` テストは `FFTW_INCLUDE` と `FFTW_LIB` 環境変数が必要。
設定されていない場合は `nix develop` で環境に入るよう案内する。

$ARGUMENTS
