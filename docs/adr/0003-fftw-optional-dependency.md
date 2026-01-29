# ADR-0003: FFTW3 をオプショナル依存にする

## Status
Accepted

## Context
FFT は DFT 計算の主要ボトルネック。FFTW3 は最速だが GPL ライセンスであり、MIT ライセンスのプロジェクトとの互換性が議論になる。

## Decision
- **FFTW3 はオプショナル**（ビルド時フラグ `-Dfftw-include`, `-Dfftw-lib` で有効化）
- Zig 純正 FFT（radix-2, mixed-radix, Bluestein）をデフォルトバックエンドとして提供
- Apple vDSP, Metal GPU もバックエンドとして選択可能

## Consequences
- FFTW なしでもビルド・実行可能（ライセンス問題を回避）
- FFTW 有効時は大幅な性能向上（実用上は FFTW 推奨）
- Zig 純正 FFT は速度で劣るが、依存ゼロで動作する
- NumPy/SciPy と同じアプローチ（自前 pocketfft + オプショナル FFTW）
