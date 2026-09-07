"""カード本文 → 効果構造（**裁定を書く場所その 1**・計画 `docs/rust_engine_plan.md` §17）。

`parser.py`／`parser_v2.py`／`rules/` がカードのテキストを `EffectNode` の木へ落とし、
`matcher.py` がテキスト中の対象指定（`TargetQuery`）を読む。生成物は
`opcg_sim/tools/export_effects_json.py` が JSON にして Rust エンジンへ渡す。

エンジン（ルール・効果の実行）は Rust（`rust/opcg_engine`）。ここには**解釈だけ**が残る
（2026-09-07・第 2 段 `rs-archive-cutover` で `legacy/python_engine/core/effects/` から移した）。
"""
