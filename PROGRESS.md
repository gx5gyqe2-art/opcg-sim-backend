# PROGRESS — 理論の目で記録を読む（`claude/theory-map`）

起点: `claude/cpu-spec-improvements-yw91jd` @ 8feb1848。指示: ユーザ決定 2026-10-11「分析を先にやってください」（司令塔経由）。PR は作らない。
**分割（司令塔 2026-10-11 17:00 UTC）**: この担当は**実 w41 だけ**。合成 w39 は `claude/theory-map-syn` が同じ道具で回し、`logs/theory_map/syn_w39.json` を push する → 報告 1 本に並べる。締め: 23:30 UTC に合成が無ければ実だけで仕上げ「未着」と書く・00:30 UTC までに報告・RESULT.json・push。

**道具の入口**: `OPCG_LOG_SILENT=1 python tests/scripts/theory_map.py --in <記録> --out <map.json>`（表 → 写し → 時計 → 集計を 1 本で・在る段は飛ばして再開・作業の置き場は `<map.json>.work/`）。
報告: `docs/reports/2026-10-11_theory_map.md`。

## 状態

- [x] §1（区切り・数え方）を全記録を見る前に書いて push（リーダーは 1 種あたり最大 9 局なので、色を副の区切りに足した）
- [ ] 実 w41: 手の値段の表（17:02 UTC に 96/300・18:00 頃に終わる見込み）→ `theory_bridge`（写し・約 3 時間・21:15 頃）→ 時計の行 → 集計
- [ ] `theory_bridge`（写しつき）と `crossing_bridge.collect` の行を集める
- [x] 分析の器（`tests/scripts/theory_map.py`・1 回きり・テストなし）を書いて 3 局で試した（写しと記録の行は 195 行全部一致）。push はゲートの後にまとめて
- [ ] 報告 §0・§2〜・次に直す候補の表・RESULT.json
- [x] 道具を push（`make test` green: cargo 474 passed・1 ignored／pytest 622 passed・12 skipped）
- [ ] 報告の後の最終 push（道具を直したらもう 1 回ゲート）
