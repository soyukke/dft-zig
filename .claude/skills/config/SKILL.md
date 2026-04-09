---
name: config
description: TOML設定ファイルの生成・バリデーション。設定の確認、テンプレート生成、パラメータ一覧表示に使う
argument-hint: "[validate <file>|generate <system>|template]"
allowed-tools: Read, Write, Glob, Grep, Bash(*)
---

DFT-Zigの設定ファイル（TOML）の作成支援とバリデーションを行う。

## サブコマンド

### `validate <file>` — 既存設定のチェック

#### 手順

1. TOMLファイルを Read で読み込む
2. コード内蔵バリデーションを活用: ビルド済みなら `./zig-out/bin/dft_zig <file> 2>&1 | head -20` で起動時の警告/エラーを確認（config.validate() の出力を取得）。SCF自体は Ctrl-C で中断して良い
3. 以下の追加チェックを手動で実施:

#### チェック項目と判定基準

**必須フィールド**:
- `xyz` — 構造ファイルパス。`ls` でファイル存在確認
- `[cell]` の `a1`, `a2`, `a3` — 3つの格子ベクトル。体積 = |a1·(a2×a3)| > 0
- `[[pseudopotential]]` — 少なくとも1つ。各要素の `path` を `ls` で存在確認

**ecut の妥当性**:
- UPF ファイルの `wfc_cutoff` / `rho_cutoff` を Grep で取得: `grep -i "cutoff\|ecutwfc\|ecutrho" <upf_file> | head -5`
- NC: 通常 15-40 Ry で十分
- PAW: UPF推奨値（ecutwfc）以上が必須。例: Si kjpaw → ecutwfc=44 Ry
- ecut < 5 Ry → ERROR

**k-mesh の妥当性**:
- 格子定数 a_i (Bohr) を計算: Angstrom→Bohr は ×1.8897
- k-spacing = 2π / (kmesh_i × a_i) を各方向で計算
- 半導体: k-spacing < 0.15 Bohr⁻¹ が推奨（例: a=10.26 Bohr, kmesh=4 → spacing=0.153 ≈ OK）
- 金属: k-spacing < 0.08 Bohr⁻¹ が必要（kmesh=8 以上）

**FFT グリッド**:
- `grid = [0,0,0]` (自動) が最も安全 → OK
- 明示指定の場合、推奨最小サイズ: `N_i >= 2 × sqrt(ecut_ry) × a_i / (2π) × grid_scale`
  - 例: ecut=40, a=10.26 Bohr → N >= 2 × 6.32 × 10.26 / 6.28 ≈ 20.7 → 最低 24 (FFT最適サイズ)
- PAW: `grid_scale >= 2` が必要

**推奨設定チェック**:
- `fft_backend` 未設定 or `"fftw"` 以外 → 「`fft_backend = "fftw"` を推奨」(HINT)
- `solver = "dense"` → 「`solver = "iterative"` を推奨」(HINT)
- 半導体で `diemac` 未設定(デフォルト1.0) → 「`diemac = 12` で収束高速化」(HINT)
- `nspin = 2` で `spinat` 未設定 → WARNING
- `smearing != "none"` で `smear_ry = 0` → ERROR
- `mixing_mode = "density"` → 「`mixing_mode = "potential"` を推奨」(HINT)
- `pulay_history = 0` → 「Pulay DIIS 無効。線形mixingのみで収束が遅い」(WARNING)
- `iterative_tol < 1e-5` → 「厳しすぎるとSCF不安定化。1e-4 推奨」(WARNING)
- `band.nbands` < ceil(電子数/2) + 2 → WARNING

#### 出力フォーマット

```
## バリデーション結果: <file>

### ERROR (計算不可)
- [なし] or エラー内容

### WARNING (要確認)
- FFT grid [32,32,32] は ecut=40 Ry に対して不足 (推奨: [48,48,48] 以上)

### HINT (最適化の余地)
- diemac 未設定。半導体なら diemac = 12 で SCF 高速化
- fft_backend = "fftw" を明示設定推奨

### OK
- 必須フィールド: 全て存在
- ecut = 40 Ry: NC疑似ポテンシャルに対して十分
- kmesh = [6,6,6]: 半導体 conventional cell に妥当
```

### `generate <system>` — テンプレート生成

テンプレートを生成して表示する。ユーザーが保存先を指定すれば Write で保存。

#### silicon — Si FCC (NC, LDA)
```toml
title = "silicon"
xyz = "silicon.xyz"
out_dir = "out/silicon"
units = "angstrom"

[[pseudopotential]]
element = "Si"
path = "pseudo/Si.upf"
format = "upf"

[cell]
a1 = [0.0, 2.7155, 2.7155]
a2 = [2.7155, 0.0, 2.7155]
a3 = [2.7155, 2.7155, 0.0]

[scf]
enabled = true
solver = "iterative"
fft_backend = "fftw"
xc = "lda_pz"
ecut_ry = 15.0
kmesh = [4, 4, 4]
mixing_beta = 0.3
max_iter = 50
convergence = 1e-6
diemac = 12.0

[band]
path = "auto"
points = 60
nbands = 8
```

#### metal — 金属系 (smearing付き, PBE)
```toml
title = "metal"
xyz = "structure.xyz"
out_dir = "out/metal"
units = "angstrom"

[[pseudopotential]]
element = "Cu"
path = "pseudo/Cu_ONCV_PBE-1.2.upf"
format = "upf"

[cell]
# FCC primitive cell (Cu, a=3.615 A)
a1 = [0.0, 1.8075, 1.8075]
a2 = [1.8075, 0.0, 1.8075]
a3 = [1.8075, 1.8075, 0.0]

[scf]
enabled = true
solver = "iterative"
fft_backend = "fftw"
xc = "pbe"
ecut_ry = 60.0
kmesh = [8, 8, 8]
mixing_beta = 0.3
pulay_history = 8
pulay_start = 4
max_iter = 100
convergence = 1e-6
smearing = "fermi"
smear_ry = 0.02
# 注意: 金属では diemac は通常不要（スピン偏極金属では有害な場合あり）

[band]
path = "auto"
points = 20
nbands = 16
solver = "iterative"

[ewald]
tol = 1e-8
```

#### spin — スピン偏極
```toml
# [scf] に以下を追加:
nspin = 2
spinat = [2.0, -2.0]  # 原子数分（反強磁性の例）
```

#### paw — PAW計算
```toml
# [scf] の変更点:
ecut_ry = 44.0        # UPF推奨値以上
grid_scale = 2.0      # PAWは2倍必要
# grid は自動(= [0,0,0])推奨
```

#### relax — 構造緩和
```toml
[relax]
enabled = true
algorithm = "bfgs"
max_iter = 100
force_tol = 1e-4
max_step = 0.5

# [scf] に compute_stress = true を追加（cell_relax時）
```

#### dfpt — DFPT フォノン
```toml
[dfpt]
enabled = true
sternheimer_tol = 1e-8
sternheimer_max_iter = 200
scf_tol = 1e-10
scf_max_iter = 50
mixing_beta = 0.3
# 注意: smearing=none, nspin=1 が必須
```

#### band — バンド計算のみ
```toml
[band]
path = "auto"       # or "G-X-W-K-G-L" 等
points = 60
nbands = 8
solver = "iterative"
```

### `template` — 全設定項目の一覧

全TOML設定項目をセクションごとに表示する。以下を参照:

**ルート**: title, xyz, out_dir, units, linalg_backend, threads, boundary
**[cell]**: a1, a2, a3
**[[pseudopotential]]**: element, path, format
**[scf]**: enabled, solver, xc, ecut_ry, kmesh, kmesh_shift, grid, grid_scale, mixing_beta, max_iter, convergence, convergence_metric, fft_backend, use_rfft, smearing, smear_ry, nspin, spinat, symmetry, time_reversal, diemac, dielng, pulay_history, pulay_start, mixing_mode, compute_stress, profile, quiet, kpoint_threads, local_potential, enable_nonlocal, iterative_max_iter, iterative_tol, iterative_max_subspace, iterative_block_size, iterative_init_diagonal, iterative_warmup_steps, iterative_warmup_max_iter, iterative_warmup_tol, iterative_reuse_vectors
**[band]**: path, points, nbands, solver, kpoint_threads, iterative_*, use_symmetry, lobpcg_parallel
**[relax]**: enabled, algorithm, max_iter, force_tol, max_step, output_trajectory, cell_relax, stress_tol, cell_step, target_pressure
**[dfpt]**: enabled, sternheimer_tol, sternheimer_max_iter, scf_tol, scf_max_iter, mixing_beta, alpha_shift, qpath_npoints, pulay_history, pulay_start, kpoint_threads, perturbation_threads, qgrid, dos_qmesh, dos_sigma, dos_nbin, compute_dielectric
**[dos]**: enabled, sigma, npoints, emin, emax, pdos
**[ewald]**: alpha, rcut, gcut, tol
**[vdw]**: enabled, method, cutoff_radius, cn_cutoff, s6, s8, a1, a2
**[output]**: cube

## ベストプラクティス

- `fft_backend = "fftw"` を常に使用（他は大幅に遅い）
- `solver = "iterative"` (LOBPCG) を推奨
- 半導体: `diemac` 設定で SCF 大幅高速化（Si=12, GaAs=13）
- 金属: `smearing = "fermi"`, `smear_ry = 0.02`（diemac は通常不要）
- PAW: `grid_scale = 2`, ecut はUPF推奨値以上
- バンド: `path = "auto"` で格子タイプに応じた自動k-path
- grid: `[0,0,0]` (自動) が最も安全

$ARGUMENTS
