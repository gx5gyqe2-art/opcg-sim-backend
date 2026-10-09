# PROGRESS — claude/cand-drawn-attackers（引いた札の攻め手を守る側の計算へ）

- [x] 器の持ち込み（survival_split の m2probe/din・survival_split.py・completion_check.py のみ）
- [x] 式（報告 §1）をコミット
- [x] 実装（OPCG_DRAWN_ATTACKERS・既定はビット不変: cargo 473 green・10 局の時計の行が 1 バイト一致）
- [x] 10 局の切り出しで全器（輪を閉じた予告）→ §2 をコミット
- [ ] 全記録（実 w41・合成 w39+w42）で候補あり／なし
- [ ] 報告・RESULT.json・make test・push

## 計測の進み（scratchpad/full/*/progress.txt）
- 実 off → 実 on（背景で連続）→ 合成 off → 合成 on（別々に）
- 実 off: 済（前の報告と一致: 的中 .7051・偏り +0.027・σ_T 1.412・打ち切り 9）
- 実 on: m2〜wc 済（的中 .7164・`S`<1/2 観測÷予測 0.825＝殺す基準 (b) に当たる・打ち切り 9→10〔A→B〕）。tb/rl/kv は 2 時間の上限で止まったので温めた計画で回し直し中
- 次: 合成 off（w39・w42 を分けて）→ 合成 on
