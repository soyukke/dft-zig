# CLAUDE.md

## 変更禁止事項

- 後方互換レイヤー、非推奨ラッパー、旧インターフェースを追加しない。
- band の k-path は `path = "auto"` か `"G-X-W-K-G-L"` のような明示文字列のみ許可する。
- `[[band.path]]` は廃止済みで、再導入しない。

## 実装方針

- TDD サイクルを優先する。まず失敗する test を足し、次に実装し、最後に test を通してから整える。
- Zig 実装は Tiger-style を意識し、安全性 -> 性能 -> 開発体験の順で判断する。
- hot path では hidden allocation、不要な再計算、曖昧な `@intCast` / `@truncate` を避ける。
- 物理的不変条件、配列長一致、NaN / Inf、ゼロ除算の前提は `assert` で明示する。
- `ApplyContext`、radial/form-factor table、k-point 並列など既存のキャッシュと並列化を不用意に壊さない。

## タスク後の確認

- 変更後は最低でも `just test-unit` を回す。
- SCF、Hamiltonian、band、nonlocal、forces、stress、DFPT の流れを変えたら、unit test だけで済ませず該当する regression も回す。
- config schema、examples、公開挙動、推奨設定を変えたら `README.md` も更新する。
