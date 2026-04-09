---
name: compare-abinit
description: DFT-ZigとABINITの直接比較。silicon, graphene, gaas, cu, fe, eosのバンド構造・全エネルギーを比較しプロット生成
argument-hint: "[silicon|graphene|gaas|cu|fe|eos]"
allowed-tools: Bash(*)
---

DFT-Zig の計算結果を ABINIT と直接比較する。

## 前提条件

- ABINITバイナリ: `external/abinit/build/abinit-10.4.7/src/98_main/abinit`
- Python venv: `.venv/` (matplotlib, numpy)
- DFT-Zigビルド済み: `just build`

## 実行コマンド

### silicon — Si バンド構造比較
```bash
cd benchmarks/silicon && source ../../.venv/bin/activate
python3 plot_comparison.py --run
```
条件: ecut=15Ry, 4x4x4 k-mesh, LDA-PZ, LOBPCG

### graphene — Graphene バンド構造比較
```bash
cd benchmarks/graphene && source ../../.venv/bin/activate
python3 plot_comparison.py --run
```
条件: ecut=15Ry, 6x6x1 k-mesh, LDA-PZ, LOBPCG

### gaas — GaAs バンド構造比較
```bash
cd benchmarks/gaas && source ../../.venv/bin/activate
python3 plot_comparison.py --run
```

### cu — Cu バンド構造比較
```bash
cd benchmarks/cu && source ../../.venv/bin/activate
python3 plot_comparison.py --run
```

### fe — Fe BCC スピン偏極バンド構造比較
```bash
cd benchmarks/fe && source ../../.venv/bin/activate
python3 plot_comparison.py --run
```

### eos — Si 状態方程式 (E-V curve)
```bash
cd benchmarks/silicon_eos && source ../../.venv/bin/activate
python3 test_eos.py
```
3つの格子定数 (0.96, 1.00, 1.04) × a₀ で全エネルギーを比較。

## 手順

1. ABINITバイナリの存在確認: `test -x external/abinit/build/abinit-10.4.7/src/98_main/abinit`
2. `just build` でDFT-Zigビルド確認
3. 引数に応じたベンチマークスクリプトを実行
4. 結果をサマリ表示（バンドごとのMSE、ギャップ差など）

## 出力

- `band_comparison*.png` — バンド図（並列+オーバーレイ）
- 標準出力にバンドごとのMSE (meV²)、ギャップ差を表示

$ARGUMENTS
