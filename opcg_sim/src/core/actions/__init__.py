"""アクション適用のディスパッチ（旧 GameManager.apply_action_to_engine の分割先）。

`apply_action(gm, ...)` がエントリポイント。プレイヤーレベル・アクションは
`player_level` のハンドラ（レジストリ登録）が処理し、それ以外は対象ループ
`target_loop.run_target_loop` へ委譲する。

契約（設計 docs/refactoring_gamestate.md §1-1・§2-1）:
  - 本パッケージは `gamestate` を import しない。`gm`（GameManager）は**ダックタイピング**で受け、
    その public/private メソッドを安定シームとして呼ぶ（Phase B でこれらを engine/ へ分割する際も
    デリゲートで名前を維持する）。ハンドラは**ステートレス**（モジュールレベル可変状態を持たない）。
  - `apply_action`/game_handler は `bool` を返す。resolver が戻り値で後続処理をゲートするため、
    ハンドラは**必ず明示的に `return True`**（コスト不払い等の失敗時のみ `False`）。暗黙の
    `return None` は resolver の success ゲートを無言でスキップさせるバグになる。
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
