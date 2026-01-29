# ADR-0002: SCF でポテンシャル混合を採用（密度混合ではなく）

## Status
Accepted

## Context
SCF ループの混合対象として密度 ρ(r) とポテンシャル V(r) のどちらを選ぶか。

密度 Pulay 混合を試したところ、大きな ecut で激しく振動し収束しなかった。ABINIT の `iscf=7` はポテンシャル混合に相当する。

## Decision
**V_in / V_out を直接混合**する（ABINIT iscf=7 相当）。密度混合はオプションとして残すが、デフォルトはポテンシャル混合。

## Consequences
- 大きな ecut でも安定して収束
- Pulay delayed start (最初の数反復は線形混合) と組み合わせて、さらに安定化 (19→11 反復)
- Kerker 前処理は K(G=0)=0 のため G=0 成分が停滞する問題があり、不採用
