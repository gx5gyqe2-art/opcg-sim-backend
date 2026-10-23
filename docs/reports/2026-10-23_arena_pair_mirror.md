# WP `arena-pair-mirror`——判定の 1 ペアを「同じデッキの席でネットだけ入れ替える」形に揃える（2026-10-23）

仕様: `docs/search_efficiency_plan.md` §7.29（本線 `claude/cpu-spec-improvements-yw91jd` @ 4f846bb・台帳 2+352）。
ブランチ `claude/arena-pair-mirror`（土台 4f846bb）。

## 1. 欠陥と直し方

`opcg_sim/loop/arena.py::play_pair_detail` は 1 局目を `build_pair(la, lb)`（p1＝候補）、2 局目を
`build_pair(lb, la)`（p1＝基準・p2＝候補）で組んでいた。**候補は 2 局とも la を握り**、入れ替わるのは先後だけ。

- `real`／`user`／`official` は seed がデッキの組を決める（`la`／`lb` を無視）ので 2 局目も同じ組
  ＝実際には既に「同じデッキの席でネットだけ入れ替え」だった（台帳の `cand_leaders: [la, la]` は誤記）。
- 固定ミラー（`--leaders fixed`）は `la = lb = None` なので 2 局目も同じ組。
- **合成デッキ（`synth*`）・`singleton` の random 対面では候補が la に偏る**——同じネット同士でも 0.5 にならない
  （`2026-10-22_lethal_root.md` §5 の off 対 off 0.5375／0.425）。

直し: 2 局目を **1 局目と同じ `ab`** で打ち、席に座るネットだけを入れ替える
（`run_game(seed, {"p1": best, "p2": cand}, *ab)`）。台帳の `cand_leaders` は候補が実際に握ったリーダー
`[ab[0][0], ab[1][0]]`（a＝p1 側・b＝p2 側）。`leaders` 欄（`leader_pair` の引き）は従来どおり。

**同じネット同士なら 2 局は同じ局**（探索の乱数は `(game_seed, turn, seat)` から決まりネットの席に依らない）
＝勝者の席が同じ＝候補は必ず 1 勝 1 敗。

## 2. 検算（手元・4 ワーカー・両席 160 回・40 ペア・seed 2300000〜2300039）

| 条件 | 候補 / 基準 | 規約 | 勝率 | ペアの分布 {0, 1, 2 勝} | void |
|---|---|---|---|---|---|
| random × real | r13d / r13b | 新 | 0.4875 | {4, 33, 3} | 0 |
| random × real | r13d / r13b | 旧 | 0.4875 | {4, 33, 3} | 0 |
| random × synth_roles2 | r13d / r13d | 新 | **0.5000** | **{0, 40, 0}** | 0 |
| random × synth_roles2 | r13d / r13d | 旧 | 0.6375 | {7, 15, 18} | 0 |
| random × real | r13d / r13d | 新 | **0.5000** | **{0, 40, 0}** | 0 |
| 固定ミラー（singleton） | r13d / r13d | 新 | **0.5000** | **{0, 40, 0}** | 0 |

- **real 系は挙動不変**: 新旧で 40 ペアの各局の勝敗・ターン数が**全部一致**（差 0 ペア）。
- **同じネット同士はどの条件でも全ペアちょうど 1 勝 1 敗**（予告どおり）。
- 旧規約の synth_roles2 では同じネット同士で 0.6375（18 ペアが 2 勝・7 ペアが 0 勝）＝この 40 seed では
  la 側のデッキが勝ち越していた分がそのまま候補の勝率に乗っていた。la／lb はランダムなので総合では不偏だが、
  **ペア内で相殺されないぶん分散が大きい**（ペア水準の CI が広がる）。新規約ではデッキ相性もペア内で相殺される。
- 台帳と検算の器: `n1_results/arena_pair_mirror/`（`verify.py`・各条件の jsonl）。

## 3. 同じ組み方をしている所の洗い出し

| 所 | 判定の器か | 対応 |
|---|---|---|
| `opcg_sim/loop/arena.py::play_pair_detail`（`arena_shard`・`gate` の `play_pair` が通る） | はい | 直した |
| `opcg_sim/loop/gate.py`（stage1/本判定は `A.play_pair`・`smoke` は 1 局） | はい | `play_pair` 経由で自動的に新規約 |
| `tests/scripts/t18_arena.py::t18_pairs` | はい（理論の介入の A/B） | 既に 2 局とも同じ `build_pair(la, lb)`・介入の席だけ入れ替え＝新規約と同じ形。変更なし |
| `opcg_sim/loop/record_gen.py` | いいえ（教材の分布） | 触らない（仕様どおり） |
| `opcg_sim/loop/driver.py::play_pair_game` | 1 局だけ | 変更なし |

`build_pair(db, lb, la, …)` の形はリポジトリ内で `play_pair_detail` にしか無かった（grep）。

## 4. 影響と運用

- 2026-10-22 以前の **synth 条件**の台帳と、以後の台帳を**同じ判定に混ぜない**（組み方が違う）。
  real・user・official・固定ミラーの台帳は新旧で同じ局なので混ぜてよい。
- 対照の「off 対 off」を打って引く運用（`lethal_root.md` §5・`n_loop_ops.md` の該当 WP）は不要になった
  ＝同じネット同士の零点が 0.5 ちょうど。
- 文書: `docs/TEST_SPEC.md` §5.05-5（新設）・§2 の索引に `tests/test_arena_pair_mirror.py`・`arena_shard` の行、
  `docs/n_loop_ops.md` §4（検証）に組み方の注記。`arena.py` の docstring。

## 5. テスト・ゲート

- `tests/test_arena_pair_mirror.py`（`cpu_infra`）: (1) 2 局目は同じデッキ・同じ席でネットだけ入れ替え・`cand_leaders`
  （配線・モック）、(2) **同じネット同士で全ペア 1 勝 1 敗**（実エンジン・random × synth_roles2・sims 8・3 ペア）。
  (2) は旧規約に戻すと落ちる（手元確認: 旧規約・sims 8 で seed 230002 が 0 勝。230001〜230010 では旧 {0:2, 1:5, 2:3}・新は 10 ペアとも 1）。
- 既存の `tests/test_arena_search_opts.py` はモック（両デッキ同形）なので変更不要で green。
- `make test` green（結果は RESULT.json）。

## 6. 留保

- 並行の WP `rs-commit-sig` が `arena.py` に候補席の旗を足している。本 WP の変更は `play_pair_detail` の 3 行と
  module docstring の 1 項目だけ＝衝突しても機械的に解ける。
- `leaders` 欄は `leader_pair` の引き（real 系ではデッキに使われない値）のまま残した＝歴代の台帳と同じ意味。
  実際の対面は `cand_leaders`（a＝p1 側・b＝p2 側＝2 局の両デッキ）で読む。
