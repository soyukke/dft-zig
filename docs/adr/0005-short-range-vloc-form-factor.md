# ADR-0005: 局所ポテンシャルに short-range 分解を採用

## Status
Accepted

## Context
V_loc(G) のフォームファクタ計算に2つの方法がある:

1. **Tail 法**: ∫V_loc(r)j₀(Gr)r²dr を有限 r_max まで積分し、r_max 以降の -2Z/r テールを解析的に補正
2. **Short-range 法**: V_loc(r)+2Z/r（滑らかで 0 に減衰する関数）を数値積分し、-8πZ/q² を解析的に引く

Tail 法では truncated integral と oscillatory cos(qr_max)/q² の相殺が悪く、G ベクトルあたり 0.3–0.7 Ry の数値誤差が発生した。

## Decision
**Short-range 分解**を採用する。

## Consequences
- Al 11e の 2p バンドオフセット: -592 meV → -88 meV に改善
- ctrap 端点補正（Newton-Cotes 5点、6次精度）と組み合わせて epsatm が ABINIT と完全一致
- 被積分関数が滑らかなため数値積分の精度が高い
