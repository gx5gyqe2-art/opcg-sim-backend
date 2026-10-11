# PROGRESS — 手の値段の既定を「打った後の 2 本の時計」に反転（`claude/adopt-clock-value`）

起点: `claude/cand-clock-value` @ 3bc6f4a7（候補の報告 `docs/reports/2026-10-10_clock_value.md`・候補のときの PROGRESS はそのブランチの履歴）。
決定: ユーザ 2026-10-11「(a) 採る（説明＝知る形／教師＝知らない形）」→ 同日の変更「**知らない形を既定にして**」（司令塔経由）。
PR は作らない。

## 1. 反転の内容

| 入口 | 反転前 | 反転後 |
|---|---|---|
| `tests/scripts/theory_bridge.py`（線形の橋・`shadow_forbid` の禁止の線と T18 の器が読む値段の正本） | 切替なし＝静的な値段（`score_candidate`）。`OPCG_CLOCK_VALUE=<表の親>` で候補 | **切替なし＝打った後の 2 本の時計・並びを知らない形**。表は記録ごとの `<記録>/clock_succ/<seed>.npz`。`OPCG_CLOCK_VALUE_ORDER=known`＝並びを知る形（上限・`<記録>/clock_succ_known/`）。**`OPCG_CLOCK_VALUE=0`＝旧い静的な値段**。`OPCG_CLOCK_VALUE=<表の親>`＝任意の表（候補のときの置き方・形は表の `order`） |
| `tests/scripts/clock_succ.py`（後続局面の表） | `--out` 必須・既定は真の盤面の複製・`--shuffle` で混ぜ直し | **既定 `--order unknown`**（`Game.fork_shuffled`）・`--order known`＝真の盤面の複製・`--shuffle` は `--order unknown` の別名。`--out` の既定は `<in>/clock_succ`（known は `<in>/clock_succ_known`）。npz に `order` を書く |
| Rust `drv_tb::game`（`theory_bridge` の局の駆動） | `in.succ` が在れば候補の値段・無い行は静的 | `in.clock_value`（Python が渡す・既定 true）が真なら表に無い行は**値段なし**（`stats.clock_rows_unpriced`・静的な値段へ黙って落ちない）。`clock_value` が無い旧い入力（`cargo` の記録の再生 `theory_rows_golden`）は 1 ビットも変わらない |
| `tests/scripts/theory_table.py`（`w̄` の測定） | — | 値段を使わない測定（`κ` は値段の前に数える）なので `OPCG_CLOCK_VALUE=0` で回す（表を要らせない） |
| `live_theory.price_candidates`・`shadow_forbid`・`t18_arena`・`theory_trace`（生の局面から読む橋） | 静的な値段 | **変えていない**（静的な値段のまま）。後続を生の局面で読む器（時計と傾きを生の局面で読む）は未作成＝docstring に明記。`OPCG_CLOCK_VALUE` はここに効かない |

- 診断の切替（`OPCG_CLOCK_VALUE_LITERAL`・`OPCG_PRICE_DUMP`）は不変。
- 混ぜ直しの乱数の種は**行から決まる決定的なもの**（候補の実装のまま・`Game.fork_shuffled(seed × 100003 + step)`・行ごとに 1 回・同じ行の候補は同じ並びで比べる・同じ行を 2 回読めば同じ値段）。
- 表の無い記録で既定を回すと**落ちる**（作る命令と `OPCG_CLOCK_VALUE=0` を示す）。求めた形と違う表（`order` が違う）は拒む。
- 変わらないもの（値段を使わない）: 時計（交点の橋）・決着前の較正・決着の宣言・帳簿の `g`・行の `κ`・守りの窓・ゲームプレイの golden。
- 新しい定数: 0。

## 2. 一致の検算（実 w41 の先頭 10 局・`git archive origin/claude/sigma-wip sigma_wip/data/w41`）【実行】

反転前＝起点 3bc6f4a7 の木（`git worktree`）・反転後＝このブランチの木。両方とも同じ表・同じ 10 局（`--limit-games 10`）・
`OPCG_PRICE_DUMP` の行ごとの写し（jsonl）も取った。比べ方: 出力 JSON は `seconds` と反転後に足した鍵（`stats.move_price`・
`stats.clock_order`・`provisional.move_price`）を除いて辞書が完全一致、jsonl は SHA-256 が一致。

| 反転後 | 反転前 | 出力 JSON | 行ごとの写し（jsonl・SHA-256 先頭 16 桁） |
|---|---|---|---|
| **既定**（切替なし＝並びを知らない形） | `OPCG_CLOCK_VALUE=<--shuffle の表の親>` | 一致 | 一致（`babc07ee13ca0aa5`） |
| `OPCG_CLOCK_VALUE_ORDER=known`（並びを知る形＝上限） | `OPCG_CLOCK_VALUE=<表の親>`（真の盤面の複製の表） | 一致 | 一致（`0cdd71bb0a6ce30b`） |
| `OPCG_CLOCK_VALUE=0`（旧い静的な値段） | 切替なし（旧既定） | 一致 | 一致（`d206bc3d89e82d36`） |

- 表: 並びを知る形は起点の木の `clock_succ.py --out … --games 10`、並びを知らない形はこのブランチの `clock_succ.py --in <記録> --games 10`
  （既定＝`--order unknown`・`<記録>/clock_succ/`）。どちらも 10/10 局が記録とビット一致（`match=1`）。
  3 つの腕の写しは互いに違う（知らない形 ≠ 知る形 ≠ 静的＝腕の取り違えは無い）。
- 既定の腕の数（10 局）: 判断の行 596・無言の行 13・表に無い行 0（`clock_rows` 596＝`atk_rows`・`clock_rows_unpriced` 0）・
  `ΔS` の AUC .88（静的 .44）・`ΔG` の AUC .68（静的 .68＝値段が入らない量・報告 §1.6 どおり不変）。
- 計算時間（10 局・この環境）: 既定（時計）243 s／知る形 236 s／静的 33 s＝約 7 倍（報告 §6 の見積もりどおり）。
  **`make test` には影響しない**——`theory_bridge.collect` を回すテストは無い（`test_theory_bridge.py` は Python の集計だけ・
  `cargo` の記録の再生は `clock_value` の無い旧い入力）。
- 落ちる経路の確認: 表の無い記録で既定を回すと `FileNotFoundError`（作る命令と `OPCG_CLOCK_VALUE=0` を示す）。
  `OPCG_CLOCK_VALUE_ORDER=known` に知らない形の表（`order=unknown`）を渡すと `ValueError` で拒む。

## 3. ゲート【実行】

- `make rust-develop` → `make test`: **green**（cargo 474 passed・0 failed・1 ignored〔候補の試験 1 本＝起点と同じ〕／pytest 622 passed・12 skipped・
  4 warnings・全体 119 s＝反転前の起点と同じ数・時間は延びていない）。`cargo` の記録の再生 `recorded_theory_bridge_replays_bit_identically` も ok
  （`clock_value` の無い旧い入力＝1 ビットも変わらない）。
- 止める条件（golden か時計の数が動く・`make test` が 5 分を超える）には当たらない。ゲームプレイの golden（`rs_goldens`）は値段を読まず、
  理論の変更はエンジンの対局・探索を 1 ビットも変えない（表を作る口 `fork`／`fork_shuffled`／`apply_search_move`／`resolve_windows` は読み取り専用・候補のまま）。
- 環境の注記: この作業環境には `maturin`・`pytest`・`pytest-xdist`・`fastapi`・`httpx` が入っておらず `pip install` した（最初の `make test` は
  `pytest` 無し、2 回目は `fastapi` 無しで 13 本が収集で落ちた＝変更とは無関係。入れた後の 3 回目が上の green）。

## 4. 残したもの（この作業の外）

- 生の局面から読む橋（`shadow_forbid`・`t18_arena`・`theory_trace`）の値段は静的なまま。既定の値段（打った後の 2 本の時計）を生の局面で読むには
  時計と傾き（`κ`・`cut`・輪郭）を生の局面で読む器が要る（未作成・docstring に明記）。
- 方策の教師の器は作っていない（入口＝並びを知らない形の表が既定・`clock_succ.py --order unknown`）。
- 既存の `docs/reports/*` は書き換えていない。報告の数（並びを知らない形＝実 (i) .7533・(ii) .6482）は `theory_overview.md` §2 の行に写した。
