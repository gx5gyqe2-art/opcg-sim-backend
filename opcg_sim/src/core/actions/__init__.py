"""アクション適用のディスパッチ（旧 GameManager.apply_action_to_engine の分割先）。

`apply_action(gm, ...)` がエントリポイント。プレイヤーレベル・アクションは
`player_level` のハンドラ（レジストリ登録）が処理し、それ以外は対象ループ
（現状は gm 側 `_apply_action_target_loop`。A-2 で本パッケージへ移設予定）へ委譲する。
"""
from .registry import _GAME_HANDLERS, normalize
from . import player_level  # noqa: F401  (import 時にデコレータでハンドラを登録する)
from . import per_target    # noqa: F401  (import 時に対象ハンドラを登録する)
from .target_loop import run_target_loop


def apply_action(gm, player, action, targets, value, source_card=None) -> bool:
    if not action:
        return False
    atype = normalize(action.type)
    entry = _GAME_HANDLERS.get(atype)
    if entry is not None:
        fn, guard = entry
        if guard is None or guard(action):
            return fn(gm, player, action, targets, value, source_card)
    # プレイヤーレベル・ハンドラに該当しない（または guard 不成立）→ 対象ループへフォールスルー。
    return run_target_loop(gm, player, action, atype, targets, value, source_card)
