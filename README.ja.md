# DFT-Zig

[English README](README.md)

Zig言語で実装された平面波基底の密度汎関数理論（DFT）電子構造計算パッケージ。
周期系（結晶）および孤立系（分子）の電子構造計算に対応。

## 機能一覧

### 計算タイプ

| 機能 | 説明 |
|------|------|
| **SCF** | 自己無撞着場計算。全エネルギー・電子密度・フェルミ準位 |
| **バンド構造** | k経路上のバンドエネルギー計算 |
| **DFPT** | 密度汎関数摂動論によるフォノン計算（Gamma点・有限q点） |
| **構造緩和** | 原子座標・格子定数の最適化（BFGS / CG / 最急降下法） |
| **力** | Hellmann-Feynman力（局所・非局所・Ewald・NLCC・PAW D^hat） |
| **ストレス** | 応力テンソル・圧力（局所・非局所・運動エネルギー・Ewald・GGA・NLCC） |
| **分子SCF** | ガウス基底（STO-3G, 6-31G, 6-31G\*\*）によるHartree-Fock / DFT |

### 交換相関汎関数

| 汎関数 | 識別子 | 説明 |
|--------|--------|------|
| LDA | `lda_pz` | Perdew-Zunger パラメタライゼーション |
| GGA | `pbe` | Perdew-Burke-Ernzerhof |

### 擬ポテンシャル

| 形式 | 対応状況 |
|------|---------|
| UPF（Norm-Conserving） | 対応 |
| UPF（PAW） | 対応（multi-L補償電荷・m-resolved D行列・Lebedev角度積分） |

### 固有値ソルバー

| ソルバー | 識別子 | 説明 |
|---------|--------|------|
| LOBPCG | `iterative` | 反復ブロック固有値法（推奨） |
| 直接対角化 | `dense` | LAPACK zheev（小系向け） |

### SCF混合法

- **線形混合**: `mixing_beta` による単純混合
- **Pulay/DIIS混合**: 履歴ベースの加速混合（`pulay_history`, `pulay_start`）
- **ポテンシャル混合**: V_in/V_outの直接混合（デフォルト・推奨）

### スメアリング

- `none`: 固定占有
- `fermi_dirac`: Fermi-Dirac分布

### 対称性

- 結晶点群の自動検出
- k点のIBZ（既約ブリルアンゾーン）削減
- 時間反転対称性

### FFTバックエンド

| バックエンド | 説明 |
|-------------|------|
| `fftw` | FFTW3（最速・推奨） |
| `zig` | Zig純正実装 |
| `zig_parallel` | Zig並列FFT |
| `vdsp` | Apple Accelerate（macOS） |
| `metal` | Metal GPU（macOS） |

### 並列化

- k点スレッド並列（SCF・バンド・DFPT）
- LOBPCG内ブロック並列

### その他の機能

- **van der Waals補正**: DFT-D3(BJ)法
- **セル緩和**: vc-relax（応力テンソルによる格子最適化）
- **スピン分極**: collinear（nspin=2）
- **NLCC**: 非線形コア補正（SCF・DFPT・力・ストレス）

## クイックスタート

### 前提条件

- Zig 0.15+
- macOS（Accelerate）またはLinux（OpenBLAS）
- FFTW3（推奨）

### Nix環境（推奨）

```bash
nix develop
just build
```

### 手動ビルド

```bash
# FFTW なし
zig build -Doptimize=ReleaseFast

# FFTW あり
zig build -Doptimize=ReleaseFast \
  -Dfftw-include=/path/to/fftw/include \
  -Dfftw-lib=/path/to/fftw/lib
```

### 実行

```bash
./zig-out/bin/dft_zig examples/silicon.toml
```

### テスト

```bash
just test              # 全テスト
just test-unit         # ユニットテスト
just test-regression   # 回帰テスト（ABINIT比較）
```

## 設定ファイル

TOML形式で計算条件を指定する。

```toml
title = "silicon"
xyz = "examples/silicon.xyz"
out_dir = "out/silicon"
units = "angstrom"

[[pseudopotential]]
element = "Si"
path = "pseudo/Si.upf"
format = "upf"

[cell]
a1 = [5.431, 0.0, 0.0]
a2 = [0.0, 5.431, 0.0]
a3 = [0.0, 0.0, 5.431]

[scf]
enabled = true
solver = "iterative"
xc = "pbe"
ecut_ry = 40.0
kmesh = [6, 6, 6]
max_iter = 50
convergence = 1e-6
mixing_beta = 0.3
pulay_history = 8
smearing = "fermi_dirac"
smear_ry = 0.01

[band]
points = 60
nbands = 8
solver = "iterative"
path = "G-X-W-L-G"

[dfpt]
enabled = true
sternheimer_tol = 1e-8
mixing_beta = 0.7

[relax]
enabled = true
algorithm = "bfgs"
force_tol = 0.001

[vdw]
enabled = true
method = "d3bj"
```

### 主な設定セクション

| セクション | 説明 |
|-----------|------|
| `[scf]` | SCF計算パラメータ（ecut, kmesh, 混合法, 収束条件） |
| `[band]` | バンド構造計算（k経路, バンド数, ソルバー） |
| `[dfpt]` | DFPT/フォノン計算（Sternheimer, q点） |
| `[relax]` | 構造緩和（アルゴリズム, 力の収束閾値） |
| `[ewald]` | Ewaldサマレーションパラメータ |
| `[vdw]` | van der Waals補正（D3-BJ） |
| `[[pseudopotential]]` | 擬ポテンシャル指定（複数元素対応） |

### 設定バリデーション

DFT-Zigは計算開始前に設定ファイルを検証する。3段階の重要度：

- **ERROR** — 無効な設定。計算中止（ファイル不在、縮退セル、パラメータ範囲外、非互換設定など）
- **WARNING** — 精度低下のリスクがある設定（gridがecutに対して小さい、nspin=2でspinat未設定、緩い収束条件）
- **HINT** — ベンチマークに基づく性能推奨（fftw、iterativeソルバー、diemac、Pulay混合、ポテンシャル混合、対称性）

出力例：
```
[WARNING] [scf.grid] grid [8,8,8] is smaller than recommended [17,17,17] for ecut_ry=15.0; use grid = [18,18,18] or set grid = [0,0,0] for auto
[HINT] [scf.fft_backend] fft_backend = "fftw" is recommended for production calculations
```

## 出力

計算結果は `out_dir` に出力される。

| ファイル | 内容 |
|---------|------|
| `band_energies.csv` | バンドエネルギー（k点 x バンド） |
| `kpoints.csv` | k点座標 |
| `scf.log` | SCF収束履歴 |
| `run_info.txt` | 入力パラメータのサマリー |

## 検証

ABINIT 10.4.7 および Quantum ESPRESSO との比較検証を `benchmarks/` で実施。

### Si — 半導体 (NC-PP, LDA, ecut=15 Ry, 4x4x4 k-mesh)

DFT-Zig vs ABINIT — バンドギャップ差 < 1 meV

![Si バンド比較](docs/images/band_silicon.png)

### Al 11e — semi-core金属 (NC-PP, PBE, ecut=30 Ry, 6x6x6 k-mesh)

DFT-Zig vs ABINIT — 最大バンド差 3.2 meV

![Al 11e バンド比較](docs/images/band_aluminum_11e.png)

### Fe BCC — スピン偏極金属 (NC-PP, PBE, ecut=40 Ry, 8x8x8 k-mesh)

DFT-Zig vs ABINIT — スピンアップ/ダウンバンド構造

![Fe バンド比較](docs/images/band_fe.png)

### 全検証マトリクス

| システム | 条件 | 検証内容 |
|---------|------|---------|
| Si (NC) | LDA, ecut=15 Ry | バンド構造・全エネルギー・EOS |
| Si (PAW) | PBE, ecut=44 Ry | バンド構造・力・ストレス (vs QE) |
| グラフェン | LDA, ecut=15 Ry | バンド構造（Dirac点） |
| GaAs | PBE, ecut=30 Ry | バンドギャップ・semi-core d帯 |
| Cu | PBE, ecut=40 Ry | 金属バンド構造 |
| Fe BCC | PBE, ecut=40 Ry | スピン分極バンド・磁化 |
| Al | PBE, ecut=15 Ry | 金属バンド構造（Fermiアライメント） |
| Al 11e | PBE, ecut=30 Ry | semi-core + 価電子バンド |
| Si DFPT | LDA, 2x2x2 k | フォノン: 754.83 vs ABINIT 754.73 cm^-1 |
| 分子 | B3LYP/6-31G** | 全エネルギー（PySCF比較, 9分子） |

## 単位系

- 内部単位: **Rydberg** (1 Ry = 13.6057 eV)
- 入力座標: angstrom または bohr（`units` で指定）
- ストレス出力: GPa

## FFTバックエンド性能比較（Si 24x24x24グリッド）

| バックエンド | 実行時間 |
|-------------|---------|
| fftw | 3.8s |
| zig_comptime24 | 23s |
| zig_parallel | 30s |

## プロジェクト構造

```
src/
├── main.zig                  # CLIエントリーポイント
├── features/
│   ├── config/               # TOML設定パーサー
│   ├── scf/                  # SCFループ
│   ├── band/                 # バンド構造計算
│   ├── dfpt/                 # 密度汎関数摂動論
│   ├── relax/                # 構造緩和
│   ├── forces/               # 力計算
│   ├── stress/               # ストレステンソル
│   ├── paw/                  # PAW実装
│   ├── pseudopotential/      # UPF擬ポテンシャル
│   ├── hamiltonian/          # ハミルトニアン構築・適用
│   ├── symmetry/             # 点群・空間群
│   ├── xc/                   # 交換相関汎関数
│   ├── kpoints/              # k点メッシュ
│   ├── fft/                  # FFT管理
│   ├── gto_scf/              # ガウス基底分子SCF
│   ├── vdw/                  # van der Waals補正
│   └── linalg/               # 固有値ソルバー（LOBPCG等）
├── lib/
│   ├── fft/                  # FFTライブラリ
│   ├── linalg/               # BLAS/LAPACKラッパー
│   └── gpu/                  # Metal GPUブリッジ
benchmarks/                   # 検証・ベンチマーク
examples/                     # 設定ファイル例
pseudo/                       # UPF擬ポテンシャル
```

## justfileコマンド

```bash
just build            # ビルド（ReleaseFast）
just test             # 全テスト実行
just test-unit        # ユニットテストのみ
just test-regression  # 回帰テスト（ABINIT比較）
just fmt              # ソースコードフォーマット
```

## License

MIT
