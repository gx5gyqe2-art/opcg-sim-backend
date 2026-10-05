# Rust 化・第 1 段（2026-10-05）— 守る側の動的計画＋段ごとの状態の数え方＋地平の選び方

状態報告（ユーザの判断が残る項目は無い）。設計は `rustprep` の `rust_design.md`（E1〜E24）に従った。第 1 段のみ。

## 何を作ったか
* `rust/opcg_engine/src/theory/`: `defender.rs`（動的計画・過不足の無い札の組・予算）／`layers.rs`（段ごとの数え方・地平の選び方）／`numeric.rs`（Python の `round`・銀行家の丸め）／`table.rs`（鍵の全体比較の表・手書きのハッシュ・依存の追加なし）／`pyapi.rs`（`RdDefender`・`rd_count_layers`・`rd_fit_horizon`・`rd_kernel_version`）。`build.rs` が `src/theory/**` の原文のハッシュを埋める。
* `tests/scripts/rd_kernel.py`: 切替 `OPCG_RD_KERNEL`＝`py`（**既定**）／`rs`／`auto`／`both`／`ref`。`crossing_bridge.py` の末尾 2 行で `_rule_guard_plan_ex`・`_ex_fit_horizon`・`rule_don_solve` を `functools.wraps` で包む（解き方の関数の原文は不変＝版の指紋のテストはそのまま）。`plan_store.py`: 包みを解いて読む 1 行と、`rs` のときだけ鍵に核の版（`rs:<api>:<原文のハッシュ>`）を足す 1 行（`py`／`both` の鍵は従来どおり）。
* 古い wheel: `(入口の版, 原文のハッシュ)` を突き合わせ、`rs`／`both` は落ち、`auto` は Python へ。テストは skip でなく fail。
* 取り込みの金型 `OPCG_RD_CAPTURE`、golden の作成／作り直し `tests/scripts/rd_kernel_golden.py`（`make rd-golden`）。

## 一致の証拠（すべてビット／`repr` 一致）
* cargo: 11 本（手計算の小問題・予算の判定・覚え書きの共有・段の和 = 状態の数・記録した動的計画 152 件のビット再生）。clippy `-D warnings` クリーン。
* pytest `tests/test_rd_kernel.py`（23 本）: 乱数の動的計画 300 問 × 予算 3（辞書・状態の数・予算超えの瞬間の数）／乱数 40 問 × 予算 5 と 8 局面 × 予算 3 で rs＝py（一部は原文とも）／`both` が通り、核を 1e-15 ずらすと落ちる／段の数え方・地平の選び方（予算 5 通り）／記録した 152 解（実 w41 76・合成 w39/w42 76・予算 300000 と 150）／鍵。
* 器まるごと: `crossing_bridge` を実 20 局・合成 20 局で rs 対 py、JSON は秒数を除いて**完全一致**（指示の 5 局より多いが走らせ済み）。
* 他の 4 つの器・100 局での記録（claude/cand-base）の再現は、ユーザ決定（10-05）で**やらない**。

## 速さ（第 1 段だけ・冷たい解き・同一機械・各 3 回の最良）
* 器 `crossing_bridge`（20 局・プランの覚え書き無し）: 実 329 s → 66 s（5.0 倍）／合成 386 s → 78 s（5.0 倍）。
* 8 局面（予算 300000）: real#50 8.9×／#54 2.8×／#104 1.4×／#119 1.3×／syn#3 1.5×／#27 1.7×／#23 9.2×／#24 5.9×。予算 150（縮めあり）: 0.9〜3.3×。合計 1.64×（小さな問題の和・設計の予測 1.4〜7× の範囲）。
* 暖かい（覚え書きあり）実行は解かないので差は出ない。他の器の速さは測っていない（skipped）。

## skipped（指示により）
relative_ledger／win_calib／theory_bridge／transition_ledger の rs 対 py、20 局の他の器、100 局の記録の再現、慣らし運転（暦での）。

## 残るリスク
* 第 2 のモデルの写しが増えた（H-4 が動くたびに REF・Python・Rust の 3 か所）。`both` を測定の通しで常用して守る。
* `make test` は wheel を作り直さない。古いと `test_rd_kernel.py` が落ちる（意図）。先に `make rust-develop`。
* 再帰の深さは Python と同程度（約 数百段）で Rust のスタックには余裕。状態数 30 万で鍵の領域は約 100 MB。
* Python 3.12 以降は `sum()` が補償和になり基準が変わる（テストが 3.12 未満を表明）。
* `wheel` の入れ替えはこの機械の共有の site-packages に入る（上位互換）。

## 第 2a 段が要るもの
`rules_sched`・`walk_crossing`・計画の列挙（`visit`／`sig` の重複排除／点数の `<`）・`_mask_firsts`（根の生成）を Rust へ。`steps`（`rules_steps` の出力）は Python のまま渡す。`Defender` を計画の列挙の内側から直接呼ぶ（今は 1 呼び出しごとに Python↔Rust の往復）。`nu` の表は Python から渡す方式を維持。同じ golden に `rules_sched` の入出力（`sched`・`tau`）のビット再生を足す。
