---
name: convergence
description: SCF収束状況を診断。scf.log/status.txt/実行出力を解析し、収束しない・遅い場合の改善策を提案
argument-hint: "[<path-to-out_dir>|<path-to-scf.log>]"
allowed-tools: Read, Glob, Grep, Bash(*)
---

SCF計算の収束状況を解析し、問題があれば改善策を提案する。

## データソース

### 1. status.txt — 最優先
Key-value形式。主要フィールド:
- `scf_converged = true/false`
- `scf_iterations = N`
- `scf_energy_total = -67.630...`
- `scf_potential_residual_rms = 4.65e-8`
- `scf_convergence_metric = density` (文字列: `density` or `potential`)
- `scf_energy_band`, `scf_energy_hartree`, `scf_energy_xc`, `scf_energy_ion_ion`, `scf_energy_psp_core`
- `scf_energy_double_counting`, `scf_energy_local_pseudo`, `scf_energy_nonlocal_pseudo`, `scf_energy_paw_onsite`
- `scf_fermi_level_ry` — **smearing有効時のみ**。半導体(smearing=none)では省略
- `nspin = 1/2`, `magnetization` — nspin=2のみ

### 2. scf.log — 最終結果のみ
```
# converged=true iterations=10
# energy_total=-67.6301 band=-1.8317 hartree=4.2542 xc=-24.6079 ion_ion=-67.1834 psp_core=4.4250
```
反復ごとのデータはscf.logに保存されない。

### 3. run_info.txt — 計算設定
以下のフィールドが含まれる:
- `ecut_ry`, `scf_solver`, `scf_xc`, `scf_smearing`, `scf_smear_ry`
- `band_nbands`, `atoms`, `cell_angstrom`
- `scf_iterative_*` (LOBPCG設定)

**注意**: `mixing_beta`, `diemac`, `pulay_history`, `kmesh` は run_info.txt に含まれない。これらの確認には元のTOMLファイルを読む必要がある（`title` フィールドからTOMLを推定）。

### 4. コマンド出力 (stderr) — 反復データ
直前の `/run` 出力に反復データがある:
```
scf iter=0 diff=1.234567 vresid=0.543210 band=-3.541200 nonlocal=1.234500
```

## 手順

### 1. データの特定
引数で out_dir パスが指定されていればそこを使う。なければ:
1. `out/*/status.txt` をGlobで探す（最新のもの）
2. `benchmarks/*/out_*/status.txt` をチェック
3. 見つからなければユーザーに確認

### 2. 情報収集
1. `status.txt` を読む → 収束フラグ、イテレーション数、最終残差
2. `scf.log` を読む → エネルギー分解
3. `run_info.txt` を読む → ecut, solver 等
4. mixing_beta/diemac/kmesh が必要な場合 → 元のTOMLファイルを探して読む

### 3. 診断

## 診断パターン

### 正常収束
- 10-20回で収束、`scf_potential_residual_rms < 1e-6`
- → 問題なし。イテレーション数とエネルギーを報告
- diemac 未設定なら追加で高速化できる可能性を示唆

### 収束失敗 (`scf_converged = false`)
iterations が max_iter に達した場合。残差の大きさで切り分け:
- **残差 > 1e-2**: mixing不安定 → `mixing_beta` を下げる (0.7 → 0.3)
- **残差 1e-2〜1e-4**: プレコンディショニング不足（下記「停滞」参照）
- **残差 < 1e-4**: max_iter を増やすだけで解決の可能性

### 停滞 (残差が横ばい)
- **半導体/絶縁体**: `diemac` を設定（Si=12, GaAs=13, diamond=5.7, GaN=9.8）。デフォルトは1.0（=無効）
- **金属**: `diemac = 1e6`, `smearing = "gaussian"`, `smearing_width = 0.01`
- `dielng = 1.0` (Bohr) も調整可能

### 遅い収束 (30回以上)
- `pulay_history = 8`（デフォルト7）
- `pulay_start = 4`（デフォルト4、最初の数回は線形mixing）
- `lobpcg_tol = 1e-4`（1e-6 は厳しすぎてSCF不安定化）

### デフォルト値参考
| パラメータ | デフォルト |
|-----------|----------|
| mixing_beta | 0.7 |
| diemac | 1.0 (無効) |
| dielng | 1.0 |
| pulay_history | 7 |
| pulay_start | 4 |
| potential_mixing | true |
| lobpcg_tol | 1e-4 |

## レポートテンプレート

```
## SCF収束診断: <title>

- 収束: Yes/No
- イテレーション: N回 / max_iter
- 最終残差 (vresid): X
- 収束判定: <density|potential>
- 全エネルギー: X Ry (X eV)
- 設定: ecut=X Ry, solver=X, mixing_beta=X, diemac=X

### エネルギー分解
| 項目 | 値 (Ry) |
|------|---------|
| band | X |
| hartree | X |
| xc | X |
| ion_ion | X |
| psp_core | X |
| double_counting | X |
| local_pseudo | X |
| nonlocal_pseudo | X |
| paw_onsite | X |

### 診断
<パターンに基づく診断>

### 推奨設定変更
<具体的なTOML設定変更>
```

$ARGUMENTS
