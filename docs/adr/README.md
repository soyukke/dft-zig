# Architecture Decision Records (ADR)

DFT-Zig の開発過程における主要な技術的意思決定の記録。

## 一覧

| # | タイトル | 分野 |
|---|---------|------|
| [0001](0001-rydberg-unit-system.md) | Rydberg 原子単位系の採用 | 基盤 |
| [0002](0002-potential-mixing-over-density.md) | SCF でポテンシャル混合を採用 | SCF |
| [0003](0003-fftw-optional-dependency.md) | FFTW3 をオプショナル依存にする | ビルド/ライセンス |
| [0004](0004-lobpcg-iterative-solver.md) | LOBPCG を標準固有値ソルバとして採用 | 固有値ソルバ |
| [0005](0005-short-range-vloc-form-factor.md) | 局所ポテンシャルに short-range 分解を採用 | 精度 |
| [0006](0006-paw-m-resolved-dij.md) | PAW で m 分解 D_ij を採用 | PAW |
| [0007](0007-paw-dhat-includes-vloc.md) | PAW D^hat に V_loc を含める | PAW |
| [0008](0008-gspace-nlcc-force.md) | NLCC 力を G 空間で計算 | 力 |
| [0009](0009-dfpt-no-kerker.md) | DFPT SCF で Kerker 前処理を使わない | DFPT |
| [0010](0010-dfpt-nlcc-essential.md) | DFPT で NLCC の完全な取り扱い | DFPT |
| [0011](0011-stress-tensor-conventions.md) | 応力テンソルの符号・対角項の規約 | 応力 |
| [0012](0012-radial-table-caching.md) | 動径テーブルによるフォームファクタキャッシング | 性能 |
| [0013](0013-dielectric-tensor-sternheimer.md) | 誘電テンソルに 2nd Sternheimer 式を採用 | DFPT |
