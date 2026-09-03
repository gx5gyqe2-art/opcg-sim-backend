"""n_eff_feat: 効果構造符号化（N系カード表現 v2）——**正本は `opcg_sim/src/learned/n_eff.py`**
（2026-09-03 c10 採用でパッケージ側へ昇格）。

本モジュールは tests/scripts 側の互換窓口: 定数・`ability_vector`・木の歩行子を再輸出し、
`build_eff_tables()` は従来どおり**引数なし**（既定エンジンのネット付属 vocab ＋ テスト DB）で
6つ組 (STATS, AB, ABM, PWR, ISL, vocab) を返す。符号化の中身・正規化規約はパッケージ側の
docstring を正とし、ここには二重に書かない。
"""
from opcg_sim.src.learned.n_eff import (  # noqa: F401
    TRIGS, OPS, NT, NOP, MAX_AB, KEYWORDS, NK, STRUCT, FILT, ABILITY_DIM, STATS_DIM,
    _walk, _has_choice, _amt, ability_vector, build_eff_tables as _build_eff_tables)


def build_eff_tables():
    """vocab index → (STATS[n,16], AB[n,4,167], ABM[n,4], PWR[n], ISL[n], vocab)。

    vocab は既定エンジン（ネット付属 vocab_ids＝gen15 系譜で固定）のもの。N系の全世代は
    この vocab で訓練されている。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from cpu_selfplay import _load_db
    vocab = LearnedEngine().vocab
    db = _load_db()
    return (*_build_eff_tables(db, vocab), vocab)
