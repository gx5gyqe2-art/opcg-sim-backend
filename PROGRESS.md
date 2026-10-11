# PROGRESS — 理論の目で記録を読む・合成 w39 担当（`claude/theory-map-syn`）

起点: `claude/cpu-spec-improvements-yw91jd` @ 8feb1848。指示: 司令塔（ユーザ決定 2026-10-11「分析を先にやってください」）。PR は作らない。
分担: 実デッキ w41 と報告は `claude/theory-map`（session_01DQ98orxkPSpFgSv4aL7sP3）。**ここは合成 w39（300 局）だけ**。
締め切り: 集計 JSON を 23:30 UTC までに push。

## 道具と手順（変えない）

- 分析の道具は `claude/theory-map` @ 5f85261a の `tests/scripts/theory_map.py` を**ファイルごと持ち込んだ（無改変）**。区切り・手順は報告 §1 のまま。
- 記録: `git archive origin/claude/sigma-wip sigma_wip/data/w39`（作業コンテナの scratchpad に展開）。
- 命令（1 本）: `OPCG_LOG_SILENT=1 python tests/scripts/theory_map.py --in <w39> --out logs/theory_map/syn_w39.json --work <scratch>/work`

## 出力の置き場（作業コンテナ内・`<scratch>`＝セッションの scratchpad）

| 段 | 置き場 | 状態 |
|---|---|---|
| (1) 後続の表（`clock_succ.py --in <w39>`・既定 `--order unknown`） | `<w39>/clock_succ/<seed>.npz` | 17:05 UTC 開始・1 局 約 10 秒 |
| (2) `theory_bridge.py`（既定の値段・`OPCG_PRICE_DUMP` の写し） | `<scratch>/work/dump.jsonl`・`bridge.json`・`bridge.log` | (1) の後に自動 |
| (3) 時計の行（`theory_map.py clocks`） | `<scratch>/work/clocks.json` | (2) の後に自動 |
| (4) 集計 | **`logs/theory_map/syn_w39.json`**（push する） | (3) の後に自動 |

## 合成で直した点

- なし（3 局の試し打ちで 4 段とも無改変で動いた。写しと記録の行の突き合わせ 149 行で不一致なし）。
