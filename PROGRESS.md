# PROGRESS — 理論の目で記録を読む（`claude/theory-map`）

起点: `claude/cpu-spec-improvements-yw91jd` @ 8feb1848。指示: ユーザ決定 2026-10-11「分析を先にやってください」（司令塔経由）。PR は作らない。
**分割（司令塔 2026-10-11 17:00 UTC）**: この担当は**実 w41 だけ**。合成 w39 は `claude/theory-map-syn` が同じ道具で回し、`logs/theory_map/syn_w39.json` を push する → 報告 1 本に並べる。締め: 23:30 UTC に合成が無ければ実だけで仕上げ「未着」と書く・00:30 UTC までに報告・RESULT.json・push。

**道具の入口**: `OPCG_LOG_SILENT=1 python tests/scripts/theory_map.py --in <記録> --out <map.json>`（表 → 写し → 時計 → 集計を 1 本で・在る段は飛ばして再開・作業の置き場は `<map.json>.work/`）。
報告: `docs/reports/2026-10-11_theory_map.md`。

## 状態（2026-10-12・訂正を反映）

- [x] §1 を全記録の前に push（afc26604）
- [x] 道具を push（5f85261a・ゲート green）
- [x] 実 w41: 表（300/300 一致）→ 写し（18,207 行・3 時間）→ 時計の行 → 集計（`logs/theory_map/real_w41.json`）
- [x] 報告 §0・§2〜§8・§10〜§11（実だけ）・RESULT.json
- [x] §9 合成 w39: 22:54 に 260/300 局の途中の版で書き、23:18 の全 300 局の版（`claude/theory-map-syn` @ dbaadf7f）に差し替えた・§0 に合成の要点・§8 の順位を合成を見て並べ直した
- [x] 完了（00:30 UTC の締めの前）
- [x] **訂正（2026-10-12）**: 実 w41 のリーダーは `user_deck_pair(seed)`（meta の `leaders` は使われなかった対）。`theory_map.py` を直し、残っていた写し・時計の行から集計だけ回し直した。デッキを使わない数は JSON が完全一致。報告に「訂正」の節を置き、§3・§4・色の行をデッキ（4 本）と色で作り直した。

## 追加: ターン単位の数え直し（2026-10-12・`docs/reports/2026-10-12_turn_plans.md`）

- [x] §1 を数の前に push（1f3d2795）
- [x] 道具 `theory_map.py turns` を push（f750773c・ゲート green）し、合成の担当に知らせた
- [x] 実 w41 の集計（`logs/theory_map/real_w41_turns.json`）と報告 §0・§2〜§7・§9〜§10
- [x] §8 合成 w39（`claude/theory-map-syn` @ 19294474 の集計を取り込み・§0・§6・§7 も合成込みに）

## 候補: 計画のドンの上限を盤面から読む（2026-10-12・`docs/reports/2026-10-12_don_deck.md`）

- [x] §1 を数の前に push（17af4477）
- [x] 実装（切替 `OPCG_DON_ROOM=1`・既定は旧）・不変量・golden の確認（4f85e503）
- [x] 較正・的中・ドンの検算・落ちた行 → **殺す基準に当たった**（エネル −0.137 → −0.138）。判断 1 つ（§9）
- [x] §7 対面ごとの較正（d506cad3）・§8 どちらの時計が外れているか（1191381e）
- [ ] 値段の写し（候補の式・300 局）→ `turns`・問い 5 の前後を §10 に（写しは 10:30 頃に終わる見込み）
