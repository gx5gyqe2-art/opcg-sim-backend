"""**Python エンジン**が要る符号化の列を差し込む口（2026-09-07・第 2 段 `rs-archive-cutover`）。

`encoder.encode(manager, ...)`／`n_rel_feat.encode_rel(manager, ...)` は「Python の
`GameManager` を受け取って符号化する」参照実装で、いくつかの列は**エンジンの実測**を要る:

| 列 | 実測の中身 | 旧 import |
|---|---|---|
| `onplay`（v7 の 3 列） | 手札の登場時効果を make/unmake で試し打ちして生死を数える | `core.cpu_ai.onplay_option_scan` |
| `lethal`（v10 の 3 列） | 台本レースでリーサル距離Δを測る | `learned.lethal.lethal_scan` |
| 条件列（`n_rel_feat`） | 静的条件が今の盤面で満たせるか | `core.effects.resolver.EffectResolver` |

serve はこの経路を通らない（符号化は Rust の `encode`／`Game.encode` が持つ）ので、
`opcg_sim/` は **Python エンジンを import しない**（計画 §16.3-6）。エンジンを持つ環境
（`legacy/python_engine`）が [`install`] で差し込み、差さっていなければ列は 0／既定値になる。

    import legacy.python_engine as LE
    LE.install_resolver_hook()      # まとめて差す
"""
from typing import Any, Callable, Optional, Tuple

#: `cpu_ai.onplay_option_scan(manager, me_name) -> (n_live, n_dead, keep_live)`
ONPLAY_SCAN: Optional[Callable[..., Tuple[float, float, float]]] = None
#: `(lethal_scan(manager, me_name) -> (d_me, d_opp, d_opp_def), MAX_TURNS)`
LETHAL_SCAN: Optional[Tuple[Callable[..., Tuple[float, float, float]], int]] = None
#: `EffectResolver(manager)`（静的条件の充足判定に使う）
RESOLVER_FACTORY: Optional[Callable[[Any], Any]] = None


def install(onplay_scan=None, lethal_scan=None, resolver_factory=None) -> None:
    """フックを差す（冪等・None の欄は据え置き）。"""
    global ONPLAY_SCAN, LETHAL_SCAN, RESOLVER_FACTORY
    if onplay_scan is not None:
        ONPLAY_SCAN = onplay_scan
    if lethal_scan is not None:
        LETHAL_SCAN = lethal_scan
    if resolver_factory is not None:
        RESOLVER_FACTORY = resolver_factory
