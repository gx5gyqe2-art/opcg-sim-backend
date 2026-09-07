"""**Python エンジン**が要る符号化の列を差し込む口（2026-09-07・第 2 段 `rs-archive-cutover`）。

`encoder.encode(manager, ...)`／`n_rel_feat.encode_rel(manager, ...)` は「Python の
`GameManager` を受け取って符号化する」参照実装で、いくつかの列は**エンジンの実測**を要る:

| 列 | 実測の中身 | 旧 import |
|---|---|---|
| `onplay`（v7 の 3 列） | 手札の登場時効果を make/unmake で試し打ちして生死を数える | `core.cpu_ai.onplay_option_scan` |
| `lethal`（v10 の 3 列） | 台本レースでリーサル距離Δを測る | `learned.lethal.lethal_scan` |
| 条件列（`n_rel_feat`） | 静的条件が今の盤面で満たせるか | `core.effects.resolver.EffectResolver` |

serve はこの経路を通らない（符号化は Rust の `encode`／`Game.encode` が持つ）ので、
`opcg_sim/` は **Python エンジンを import しない**（計画 §16.3-6。この docstring も import 文の形を
避けてある＝`grep '^\s*\(from\|import\) legacy' opcg_sim/` が 0 件になるように）。エンジンを持つ
環境（`legacy/python_engine`）が [`install`] で差し込む。

    LE = importlib.import_module("legacy.python_engine")   # エンジンを持つ側から
    LE.install_hooks()                                     # 3 か所まとめて差す

**差さっていなければ [`MissingHook`] を送出する**（ユーザ決定 2026-09-07・計画 §16.4-3）。
黙って 0 を返すと「符号化はできたが列だけ違う」ネットや評価値が出てしまい、`install_hooks()` の
呼び忘れに気づけない。これは参照実装なので、**間違った数字を出すより落ちる方が安全**。
"""
from typing import Any, Callable, Optional, Tuple


class MissingHook(RuntimeError):
    """エンジン実測のフックが差さっていない（`legacy.python_engine.install_hooks()` を呼ぶ）。"""

    def __init__(self, name: str, what: str):
        super().__init__(
            f"符号化の {name} が要る実測（{what}）のフックが差さっていない。"
            "Python エンジンを持つ環境なら "
            "`importlib.import_module('legacy.python_engine').install_hooks()` を先に呼ぶこと。"
            "serve／生成／アリーナはこの経路を通らない（符号化は Rust の `Game.encode`）ので、"
            "ここへ来たなら**参照実装を Python エンジン抜きで回している**＝設定の誤り。"
        )

#: `cpu_ai.onplay_option_scan(manager, me_name) -> (n_live, n_dead, keep_live)`
ONPLAY_SCAN: Optional[Callable[..., Tuple[float, float, float]]] = None
#: `(lethal_scan(manager, me_name) -> (d_me, d_opp, d_opp_def), MAX_TURNS)`
LETHAL_SCAN: Optional[Tuple[Callable[..., Tuple[float, float, float]], int]] = None
#: `EffectResolver(manager)`（静的条件の充足判定に使う）
RESOLVER_FACTORY: Optional[Callable[[Any], Any]] = None


def require_onplay_scan():
    """`encoder.encode` の登場時スキャン v7（差さっていなければ [`MissingHook`]）。"""
    if ONPLAY_SCAN is None:
        raise MissingHook("onplay（v7 の 3 列）", "cpu_ai.onplay_option_scan")
    return ONPLAY_SCAN


def require_lethal_scan():
    """`encoder.encode` のリーサル距離 v10（差さっていなければ [`MissingHook`]）。"""
    if LETHAL_SCAN is None:
        raise MissingHook("lethal（v10 の 3 列）", "learned.lethal.lethal_scan")
    return LETHAL_SCAN


def require_resolver_factory():
    """`n_rel_feat` の条件列（差さっていなければ [`MissingHook`]）。"""
    if RESOLVER_FACTORY is None:
        raise MissingHook("条件列", "effects.resolver.EffectResolver")
    return RESOLVER_FACTORY


def install(onplay_scan=None, lethal_scan=None, resolver_factory=None) -> None:
    """フックを差す（冪等・None の欄は据え置き）。"""
    global ONPLAY_SCAN, LETHAL_SCAN, RESOLVER_FACTORY
    if onplay_scan is not None:
        ONPLAY_SCAN = onplay_scan
    if lethal_scan is not None:
        LETHAL_SCAN = lethal_scan
    if resolver_factory is not None:
        RESOLVER_FACTORY = resolver_factory
