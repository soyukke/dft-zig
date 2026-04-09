---
name: update-baseline
description: リグレッションテスト用ベースラインデータの再生成。ABINITバージョンアップや計算条件変更時に使用
argument-hint: "[silicon|graphene|gaas|cu|fe|dfpt|molecule|silicon_paw]"
allowed-tools: Bash(*)
---

リグレッションテスト用のベースラインデータを更新する。

## 前提条件

- ABINITバイナリ: `external/abinit/build/abinit-10.4.7/src/98_main/abinit`
- QEバイナリ: `external/qe/conda_env/bin/pw.x`
- Python venv: `.venv/` (numpy, matplotlib, scipy)
- DFT-Zigビルド済み: `just build`
- 現行 `just test-regression` がPASSすること（事前確認）

## 実行方法

全ての系で共通の流れ: ABINIT/QE実行 → ベースライン保存 → リグレッション再検証

### silicon, graphene, gaas, cu, fe (ABINIT基準)

```bash
cd benchmarks/<system> && source ../../.venv/bin/activate
python3 plot_comparison.py --run            # ABINIT実行
python3 plot_comparison.py --save-baseline  # baseline/ にCSV保存
```

- silicon: `baseline/abinit_bands.csv`, `abinit_kpoints.csv`
- graphene: `baseline/abinit_bands.csv`
- gaas: `baseline/abinit_bands.csv`, `abinit_kpoints.csv`
- cu: `baseline/abinit_bands.csv`, `abinit_kpoints.csv`
- fe: `baseline/abinit_bands_up.csv`, `abinit_bands_down.csv`, `abinit_kpoints.csv`, `abinit_info.txt`

### dfpt (ABINIT基準)

```bash
cd benchmarks/silicon_dfpt && source ../../.venv/bin/activate
python3 generate_abinit_ref.py
# → abinit_dfpt_ref_2k.csv が自動生成される
```

### silicon_paw (QE基準)

```bash
cd benchmarks/silicon_paw && source ../../.venv/bin/activate
QE=../../external/qe/conda_env/bin/pw.x
$QE < qe_scf.in > qe_scf.out 2>&1
DYLD_LIBRARY_PATH=../../external/qe/conda_env/lib:$DYLD_LIBRARY_PATH $QE < qe_band.in > qe_band.out 2>&1
python3 generate_qe_baseline.py
# → baseline/qe_bands.csv, baseline/qe_kpoints.csv が自動生成される
```

### molecule (PySCF基準)

```bash
cd benchmarks/molecule && source ../../.venv/bin/activate
python3 generate_baseline.py
# → baseline.json が自動生成される
```

## 事後確認

```bash
just test-regression  # 全テストがPASSすること
git add benchmarks/*/baseline/
git commit -m "chore: update regression baselines"
```

## 注意

- ベースライン更新は稀（ABINITバージョン変更、計算条件変更時のみ）
- 分子ベースラインはPySCF基準（ABINITではない）
- PAWベースラインはQE基準（ABINITはUPF2 PAW非対応）

$ARGUMENTS
