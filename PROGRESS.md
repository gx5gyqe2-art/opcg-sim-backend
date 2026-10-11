# PROGRESS — 理論の目で記録を読む（`claude/theory-map`）

起点: `claude/cpu-spec-improvements-yw91jd` @ 8feb1848。指示: ユーザ決定 2026-10-11「分析を先にやってください」（司令塔経由）。PR は作らない。
報告: `docs/reports/2026-10-11_theory_map.md`。

## 状態

- [x] §1（区切り・数え方）を全記録を見る前に書いて push（リーダーは 1 種あたり最大 9 局なので、色を副の区切りに足した）
- [ ] 手の値段の表（`clock_succ.py`・並びを知らない形）: 実 w41 → 合成 w39（バックグラウンド・同時に 1 本・1 局 約 13 秒）。その後に `theory_bridge`（写し）と時計の行を w41 → w39 の順に鎖で回す
- [ ] `theory_bridge`（写しつき）と `crossing_bridge.collect` の行を集める
- [x] 分析の器（`tests/scripts/theory_map.py`・1 回きり・テストなし）を書いて 3 局で試した（写しと記録の行は 195 行全部一致）。push はゲートの後にまとめて
- [ ] 報告 §0・§2〜・次に直す候補の表・RESULT.json
- [ ] `make test`（コードを足すので 1 回）→ push
