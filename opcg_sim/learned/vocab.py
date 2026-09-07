"""カード語彙（card_id → index）とカード DB の共有ロード（エンジン非依存）。

語彙は `encoder.build_vocab(db)`＝**カード DB から決まる**（ネットの npz からではない）ので、
Python エンジンは要らない。旧 `cpu_learned._shared_vocab_game()` の語彙部分と同じ値になる
（N 系の全世代はこの語彙で訓練されている）。

プロセスで 1 回だけ作って共有する（訓練・評価帯・カード表の生成が同じものを引く）。
"""
from typing import Any, Dict

_SHARED: Dict[str, Any] = {}


def load_db():
    """カード DB（全 card_id を解決済み）。`opcg_sim.loop.decks.load_db` と同じもの。"""
    from opcg_sim.loop.decks import load_db as _load
    return _load()


def shared_vocab() -> Dict[str, int]:
    """card_id → index。`encoder.build_vocab(db)`（決定論・DB の並びで決まる）。"""
    if "vocab" not in _SHARED:
        from opcg_sim.learned import encoder as E
        _SHARED["vocab"] = E.build_vocab(load_db())
    return _SHARED["vocab"]
