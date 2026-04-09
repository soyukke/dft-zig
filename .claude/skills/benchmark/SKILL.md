---
name: benchmark
description: DFT-Zigリグレッションテスト実行。silicon, graphene, gaas, cu, fe, eos, dfpt, molecule, kparallel, lobpcg
argument-hint: "[all|silicon|graphene|gaas|cu|fe|eos|dfpt|molecule|kparallel|lobpcg|paw]"
allowed-tools: Bash(just *)
allowed-tools: Bash(just *)
---

DFT-Zigのリグレッションテストを実行する。

## 対象

| 引数 | コマンド |
|------|---------|
| `all` (デフォルト) | `just test-regression` |
| `silicon` | `just test-silicon` |
| `graphene` | `just test-graphene` |
| `gaas` | `just test-gaas` |
| `cu` | `just test-cu` |
| `fe` | `just test-fe` |
| `eos` | `just test-eos` |
| `dfpt` | `just test-dfpt` |
| `molecule` | `just test-molecule` |
| `kparallel` | `just test-silicon-kparallel` |
| `lobpcg` | `just test-lobpcg-parallel` |

## 手順

1. `just build` でビルド済みか確認（なければビルド）
2. 引数に応じたテストを実行
3. PASS/FAILを報告

$ARGUMENTS
