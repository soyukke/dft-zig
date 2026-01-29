# ADR-0011: 応力テンソルの符号・対角項の規約

## Status
Accepted

## Context
応力テンソルの実装で複数の符号・規約のバグが発生した。各項の正しい規約を記録する。

## Decision
以下の規約を採用（QE/ABINIT と整合）:

### 位相符号（局所・NLCC 応力）
- エネルギー: `Re[ρ̃ conj(V_G)] = ρ_r cos(GR) - ρ_i sin(GR)`
- 誤って `+ ρ_i sin` としていた → ~180 GPa の誤差

### GGA off-diagonal
- `σ^GGA_αβ = -(2/Ω) ∫ (∂f/∂σ)(∂_αρ)(∂_βρ) dΩ` — **マイナス符号**（d|∇ρ|²/dε から）
- PBE では ∂f/∂σ < 0 なので結果は正の寄与

### 非局所応力の対角項
- `-E_nl/Ω` （`-2E_nl/Ω` ではない。1/Ω は <β|ψ> の体積正規化から）

### 運動エネルギー応力
- 体積からの対角項なし。PW 正規化が展開係数 c に既に含まれている
- `σ^T_αβ = -(2spin/Ω) Σ fw Σ |c|² q_α q_β`

### NLCC 応力の対角項
- `-(∫V_xc × ρ_core/Ω) δ_αβ` — XC 応力の対角項から excess core 寄与を打ち消す

### PAW 固有
- on-site 応力 = 0（Hellmann-Feynman 定理による）
- augmentation の体積対角項なし（QE addusstress と整合）
- ecutrho 球面カットオフを全 G 空間和に適用

## Consequences
- Si PAW ecut=44: DFT-Zig P=2.61 GPa vs QE P=2.59 GPa (0.02 GPa 差)
- Si NC ecut=30: DFT-Zig -2.99 GPa, ABINIT -3.05 GPa
