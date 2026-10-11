# PROGRESS — 理論の目で記録を読む（`claude/theory-map`）

起点: `claude/cpu-spec-improvements-yw91jd` @ 8feb1848。指示: ユーザ決定 2026-10-11「分析を先にやってください」（司令塔経由）。PR は作らない。
**分割（司令塔 2026-10-11 17:00 UTC）**: この担当は**実 w41 だけ**。合成 w39 は `claude/theory-map-syn` が同じ道具で回し、`logs/theory_map/syn_w39.json` を push する → 報告 1 本に並べる。締め: 23:30 UTC に合成が無ければ実だけで仕上げ「未着」と書く・00:30 UTC までに報告・RESULT.json・push。

**道具の入口**: `OPCG_LOG_SILENT=1 python tests/scripts/theory_map.py --in <記録> --out <map.json>`（表 → 写し → 時計 → 集計を 1 本で・在る段は飛ばして再開・作業の置き場は `<map.json>.work/`）。
報告: `docs/reports/2026-10-11_theory_map.md`。

## 状態（21:05 UTC）

- [x] §1 を全記録の前に push（afc26604）
- [x] 道具を push（5f85261a・ゲート green）
- [x] 実 w41: 表（300/300 一致）→ 写し（18,207 行・3 時間）→ 時計の行 → 集計（`logs/theory_map/real_w41.json`）
- [x] 報告 §0・§2〜§8・§10〜§11（実だけ）・RESULT.json
- [ ] §9 合成 w39（`claude/theory-map-syn` の `logs/theory_map/syn_w39.json` 待ち・23:30 UTC に無ければ「未着」で閉じる）
