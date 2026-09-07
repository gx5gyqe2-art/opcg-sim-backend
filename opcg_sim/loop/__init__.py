"""学習ループのオーケストレーション（生成・アリーナ・ゲート判定）。

**エンジンは Rust**（`rust/opcg_engine`・PyO3 wheel）。ここにあるのは「対局を回す段取り」だけで、
ルール・効果・探索の裁定は 1 行も持たない（`docs/rust_engine_plan.md` §17 の到達形）。

    opcg_sim/loop/
      engine.py       Rust 拡張の起動（masters／net／vocab）と席のつまみ
      decks.py        デッキ生成（合成デッキ・掘り差し込み・singleton）
      driver.py       対局ループ（Rust `Game` ＋ `Game.decide`）
      record_gen.py   自己対戦の棋譜ダンプ（`python -m opcg_sim.loop.record_gen`）
      arena.py        席入替 CRN のペア対局・ペア水準 CI・帯設計・台帳
      arena_shard.py  再開可能なシャード実行（`python -m opcg_sim.loop.arena_shard`）
      arena_merge.py  シャード台帳の結合と最終判定
      gate.py         昇格ゲート（stage1/stage2）と煙試験

以前は `tests/scripts/` に置いていた（n_record_gen.py／arena_parallel.py／arena_resume.py／
arena_merge.py／promotion_gate.py／arena_gate.py／n1_gate.py）。`tests/scripts/` は
「単体実行の実験・計測・監査 CLI」の置き場なので、役割どおりここへ移した
（ユーザ決定 2026-09-07・計画 §17）。手順の正本は `docs/n_loop_ops.md`。
"""
