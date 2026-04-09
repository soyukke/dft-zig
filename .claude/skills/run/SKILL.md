---
name: run
description: DFT-Zig計算を実行し結果をサマリ表示。TOMLファイルを指定して実行。計算の実行、結果確認、エラー診断に使う
argument-hint: "<config.toml>"
allowed-tools: Bash(*), Read, Glob, Grep
---

DFT-Zig計算を実行し、結果を整理して報告する。

## 手順

1. **入力の解決**: 引数でTOMLファイルが指定されていなければ、`examples/` と `benchmarks/*/` からTOMLファイルを一覧して選択を促す
2. **TOMLの事前確認**: Read でTOMLを読み、`out_dir` のパスを把握する（デフォルト: `output/`）
3. **ビルド**: `just build` でバイナリ確認（なければビルド）
4. **計算実行**: `./zig-out/bin/dft_zig <config.toml>` を実行。stderr出力も `2>&1` でキャプチャする（収束情報が含まれるため）
5. **結果サマリ**: 出力ファイルを解析して報告（下記参照）
6. **エラー時**: よくあるエラーを診断して対処法を提案

## 出力ファイルの解析方法

計算後、TOMLの `out_dir` 以下に出力ファイルが生成される。

### status.txt — 最も重要。まずこれを読む
Key-value形式。主要フィールド:
- `scf_converged` — 収束したか (true/false)
- `scf_iterations` — SCFイテレーション数
- `scf_energy_total` — 全エネルギー (Ry)
- `scf_potential_residual_rms` — 最終残差
- `scf_fermi_level_ry` — フェルミ準位（**smearing有効時のみ出力**。smearing=none では省略される）
- `nspin` — スピン数 (1 or 2)
- `magnetization` — 磁化（**nspin=2 の場合のみ**）

### timing.txt — 計算時間の内訳
- `setup_sec`, `scf_sec`, `band_sec`, `total_sec`, `cpu_sec`
- `band_kpoints`, `band_sec_per_kpoint`

### scf.log — 最終結果（反復データなし）
- 1行目: `# converged={bool} iterations={N}`
- 2行目: `# energy_total=... band=... hartree=... xc=... ion_ion=... psp_core=...`
- 注意: scf.log にはCSV反復データは含まれない。反復ごとの収束情報は stderr に出力される

### band_energies.csv — バンド構造（bandが有効な場合）
- `index,dist,band0,band1,...,bandN`
- バンドギャップの計算:
  - 電子数の推定: TOMLの疑似ポテンシャル z_valence × 原子数 から算出。または `pseudopotentials.csv` の `z_valence` を参照
  - 占有バンド数 = ceil(電子数 / 2)（nspin=1の場合）
  - 間接ギャップ = min(最低非占有バンド) - max(最高占有バンド)
  - 縮退がある場合（ギャップが負に見える）: 結晶対称性による縮退の可能性を報告し、正確なギャップ算出にはより詳細な解析が必要と注記

### relax_status.txt — 構造緩和（relaxが有効な場合）
- `converged`, `iterations`, `final_energy_ry`, `max_force_ry_bohr`

## サマリ表示テンプレート

```
## 計算結果: <title>
- 全エネルギー: <energy_total> Ry (<energy_total × 13.6057> eV)
- SCF: <iterations>回で収束、残差 <residual>
- フェルミ準位: <fermi_level> Ry  ※smearing有効時のみ
- バンドギャップ: <gap> eV (間接)  ※band有効時
- 計算時間: SCF <scf_sec>秒 + バンド <band_sec>秒 = 合計 <total_sec>秒
```

## よくあるエラーと対処法

| エラー | 原因 | 対処 |
|--------|------|------|
| `UnknownHighSymmetryPoint` | セルタイプとk-pathの不一致 | `path = "auto"` に変更 |
| `grid is smaller than recommended` | FFTグリッド不足 | gridを大きく / 設定を削除して自動決定 |
| `FFTW_INCLUDE and FFTW_LIB must be set` | FFTW環境未設定 | `nix develop` で環境に入る |
| SCF収束しない | mixing設定 | `/convergence` で診断 |

## 計算時間の目安

| 系 | ecut | k点数 | 原子数 | 目安 |
|----|------|-------|--------|------|
| Si primitive (2原子) | 15 Ry | 10 kpts | 2 | ~3秒 |
| Si conventional (8原子) | 40 Ry | 241 kpts | 8 | ~40分 |

`examples/silicon.toml` は conventional cell で ecut=40 のため時間がかかる。素早いテストには `benchmarks/silicon/dft_zig.toml` (primitive, ecut=15) を推奨。

$ARGUMENTS
