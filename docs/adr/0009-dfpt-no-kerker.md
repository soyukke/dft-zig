# ADR-0009: DFPT SCF で Kerker 前処理を使わない

## Status
Accepted

## Context
基底状態 SCF と同様に、DFPT SCF でも Kerker 前処理 K(G)=|G+q|²/(|G+q|²+q_TF²) を残差に適用して収束を加速できるか検証した。

結果:
- CNT bundle で 32/60 の摂動が収束失敗（ベースラインは 0/60）
- 全体の反復回数が 2.29 倍に増加
- Z 点が 14→22 反復に悪化

原因は基底状態と同じ: K(G≈0)≈0 のため低 G 成分の V_SCF が更新されない。K_min=0.2 のフロアを設けても改善しなかった。

Model dielectric function 前処理も試したが同様に失敗し、revert した。

## Decision
DFPT SCF では**前処理なし の Pulay 混合**をそのまま使う。

## Consequences
- 安定した収束（全摂動で 100% 収束）
- 反復回数は最適ではないが、信頼性を優先
