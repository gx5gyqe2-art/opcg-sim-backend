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
| (1) 後続の表（`clock_succ.py --in <w39>`・既定 `--order unknown`） | `<w39>/clock_succ/<seed>.npz` | 済（300 局・不一致 0・2971 秒） |
| (2) `theory_bridge.py`（既定の値段・`OPCG_PRICE_DUMP` の写し） | `<scratch>/work/dump.jsonl`・`bridge.json`・`bridge.log` | 済（300 局・19:18〜23:12・14,037 秒） |
| (3) 時計の行（`theory_map.py clocks`） | `<scratch>/work/clocks.json` | 済（行 3,439・先に回した） |
| (4) 集計 | **`logs/theory_map/syn_w39.json`**（push する） | **済・全 300 局版を push**（行 15,154・時計つき 14,935・突き合わせ 15,796 行で欠け 0）。22:30 の部分版（260 局）は履歴に残る |

## 合成で直した点

- なし（3 局の試し打ちで 4 段とも無改変で動いた。写しと記録の行の突き合わせ 149 行で不一致なし）。

## 経過・注意

- 待ちの監視の `pgrep -f` が自分の命令行に一致し、表の完了（17:55）から鎖の起動まで約 75 分空いた。時計の行を先に回してから写しへ進めた（段の順を入れ替えただけ・中身は同じ）。
- 22:30 に 23:30 に間に合わない見込みが出たので、指示どおり 260 局分の集計を先に push（`logs/theory_map/syn_w39.json` の `status` が PARTIAL）。
- ゲート: `cargo test` 474 passed／pytest 622 passed（このコンテナに pytest・fastapi が無かったので入れてから回した）。
- 23:14 に全 300 局の集計が出たので部分版を置き換えた（道具の出力そのまま・`status` 欄なし）。

## 退避（2026-10-12）

- 作業の置き場（`work/dump.jsonl`＝写し 300 局・`work/clocks.json`＝時計の行・`bridge.json`）をコンテナが回収されても残るよう
  `logs/theory_map/syn_w39_work.tgz`（2.1MB）に退避した。戻すときは `tar xzf logs/theory_map/syn_w39_work.tgz -C <dir>` → `<dir>/work/`。
  問い 3 の数え直しの道具が来たら、この置き場に無改変で当てる（司令塔の予告 10-12）。

## 問い 3 のターン単位の数え直し（2026-10-12）

- 道具: `claude/theory-map` @ f750773c の `tests/scripts/theory_map.py`（`turns`）を**無改変**で持ち込み、退避した置き場（`work/dump.jsonl`・`clocks.json`）に当てた。
- 出力: **`logs/theory_map/syn_w39_turns.json`**（ターン 3,739・時計なし 190・判定の行なし 160・使った 3,389・17 秒）。
