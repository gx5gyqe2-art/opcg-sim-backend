# PROGRESS — claude/cand-leftover-don（後の段で余ったドンを攻撃に付ける）

- [x] 器の持ち込み（cand-drawn-attackers の 5767d9eb のみ）
- [x] 式（報告 §1）をコミット
- [x] 実装（OPCG_LEFTOVER_DON・既定はビット不変: 10 局の時計の行が 1 バイト一致）
- [x] 10 局の切り出しで全器（輪を閉じた予告）→ §2 をコミット
- [ ] 全記録（実 w41・合成 w39+w42）で候補あり／なし
- [ ] 報告・RESULT.json・make test・push

## 計測の進み（scratchpad/full/*/progress.txt）
- **予告の時刻**: 輪を閉じた予告（報告 §2）は全記録の結果を見る前にコミットした——`d3a8546f`（2026-10-09 18:31:32 UTC）。全記録の最初の走り（実 off の `m2_probe`）はその後に始めた。
- 順番: 実 off（a: m2〜chain → b: tb/rl/kv）→ 実 on → 合成 off（a・b・c）→ 合成 on。重い計測は同時に 1 本。
