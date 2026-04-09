---
name: compare-qe
description: DFT-ZigとQuantum ESPRESSO (QE) のPAW計算比較。silicon-pawの応力・力・バンドを比較
argument-hint: "[silicon-paw]"
allowed-tools: Bash(*)
---

DFT-Zig の PAW 計算結果を Quantum ESPRESSO と比較する。

## 前提条件

- QEバイナリ: `external/qe/conda_env/bin/pw.x`
- バンド計算で `DYLD_LIBRARY_PATH=external/qe/conda_env/lib` が必要な場合あり
- `external/qe/qe-7.4/build/bin/pw.x` は LAPACK (ZHEGV) エラーあり — 使用不可
- Python venv: `.venv/` (numpy, matplotlib)
- DFT-Zigビルド済み: `just build`

## 実行コマンド

### silicon-paw — Si PAW バンド・応力・力比較

リグレッションテスト（DFT-Zig実行 + QEベースラインとの比較）:
```bash
just test-paw
```

QEを直接実行して比較する場合:
```bash
cd benchmarks/silicon_paw && source ../../.venv/bin/activate
QE=../../external/qe/conda_env/bin/pw.x
$QE < qe_scf.in > qe_scf.out 2>&1
DYLD_LIBRARY_PATH=../../external/qe/conda_env/lib:$DYLD_LIBRARY_PATH $QE < qe_band.in > qe_band.out 2>&1
```

QE入力ファイル: `benchmarks/silicon_paw/`
- `qe_scf.in` — SCF (ecut=44Ry, 4x4x4)
- `qe_band.in` — バンド
- `qe_force_test.in` — 力テスト

応力比較: `benchmarks/silicon_paw/qe_stress/`

## 比較ポイント

| 項目 | QE出力 | DFT-Zig出力 |
|------|--------|-------------|
| バンドギャップ | `grep "highest occupied"` | band_energies.csv |
| 応力 | `grep "total   stress"` | scf.log |
| 力 | `grep "Forces acting"` | scf.log |
| エネルギー | 絶対値はG=0規約差あり。差分で比較 |

## 注意事項

- PAWの絶対エネルギーは G=0 規約で ~0.16 Ry オフセットあり（物理バグではない）
- バンドギャップ・相対バンド位置で比較すること
- ABINITはPAW UPF2未サポート（10.4.7時点）→ PAW検証はQEのみ
- Si PAW: ecut=44Ry, grid_scale=2 が必要（UPF header の suggested cutoff）

$ARGUMENTS
