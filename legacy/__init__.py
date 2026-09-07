"""凍結した旧実装（tag `py-engine-final` の時点で動くもの）。

`legacy/python_engine/` は Rust 化する前の **Python エンジン**（ルール・効果・探索）と、
それを直に叩いていたテスト・ハーネス・実験 CLI。**テストゲートの対象外**で、`opcg_sim/` から
ここを import する箇所は 0（`docs/rust_engine_plan.md` §16.3-6）。

回す手順は `docs/TEST_SPEC.md` の「legacy（tag）を回す」を参照。
"""
