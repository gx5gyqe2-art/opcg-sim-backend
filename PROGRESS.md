# PROGRESS — claude/cand-leftover-don（後の段で余ったドンを攻撃に付ける）

- [x] 器の持ち込み（cand-drawn-attackers の 5767d9eb のみ）
- [x] 式（報告 §1）をコミット
- [x] 実装（OPCG_LEFTOVER_DON・既定はビット不変: 10 局の時計の行が 1 バイト一致）
- [x] 10 局の切り出しで全器（輪を閉じた予告）→ §2 をコミット
- [x] 全記録（実 w41・合成 w39+w42）で候補あり／なし（合成の線形の橋は w39 の 300 局）
- [x] 報告・RESULT.json・make test（green）・push

## 計測の進み（scratchpad/full/*/progress.txt）
- **予告の時刻**: 輪を閉じた予告（報告 §2）は全記録の結果を見る前にコミットした——`d3a8546f`（2026-10-09 18:31:32 UTC）。全記録の最初の走り（実 off の `m2_probe`）はその後に始めた。
- 順番: 実 off（a: m2〜chain → b: tb/rl/kv）→ 実 on → 合成 off（a・b・c）→ 合成 on。重い計測は同時に 1 本。
- 実 off（a）済: 前の報告と一致（的中 .7051・偏り +0.354・打ち切り 9・S<1/2 0.927）。実 on（a）済: 的中 .6964・S<1/2 0.830・相対 AUC .7969＝殺す基準 (b)(c)(d) に当たる。次: 実 tb/rl/kv → 合成。
- 実 tb/rl/kv 済（off/on）: ΔG .7295→.7324・帳簿 κ .7901→.8261・kappa_vector ビット同じ。次: 合成 off（a: m2〜wc → b: chain → c: rl/kv → d: tb は w39 の 300 局だけ〔900 局では 1 本で 2 時間を超えるため〕）→ 合成 on。
- コンテナが 2 回再起動した（合成 off の survival_split の途中）。合成は survival_split／段ごとの 2×2 を w39・w42 で分けて回し行をつなぐ（引いた札の候補の報告と同じやり方）。合成 off の m2・ss 済（S<1/2 1.102・前の報告と一致）。
- 合成 off/on の m2〜wc 済: on は 的中 .6602→.6493・相対 AUC .7241→.7186（殺す基準 (c)(d)）・S<1/2 1.102→0.969・打ち切り 70→3・σ_T 2.004→1.109・絶対の腕の対数損失 .864→.655。次: 合成の段ごとの 2×2（off/on）→ rl/kv → tb（w39）。
- 04:10 UTC: 合成の段ごとの 2×2（off/on）済。いま合成の rl/kv（off→on・約 1.5 時間）→ 合成の線形の橋は w39 の 300 局だけ（off/on・各約 70 分）。残り約 3.5 時間 → 報告・RESULT.json・make test・最後の push。
- 全記録 済。報告・RESULT.json 済。最終の木で make rust-develop → make test を実行中。
- make test green（cargo 474・pytest 622 passed）。最後の push。
