"""アクション適用のレジストリ・ディスパッチ基盤。

`apply_action_to_engine` が enum を文字列へ戻して 45 分岐していた巨大 if チェーンを、
ActionType をキーにしたハンドラ表へ置き換える。ハンドラは import 時に登録され、以後不変。

ホットパス（make/unmake 探索）配慮のため context オブジェクトは作らず、位置引数で渡す:
    game_handler:  (gm, player, action, targets, value, source_card) -> bool
"""
from typing import Callable, Dict, Optional, Tuple
from ...models.enums import ActionType

# ActionType -> (handler, guard)。guard は action を受け bool を返す述語（None なら常に適用）。
# guard が False のアクションは対象ループへフォールスルーする（現行の条件付き分岐を保存）。
_GAME_HANDLERS: Dict[ActionType, Tuple[Callable, Optional[Callable]]] = {}

# ActionType -> 1対象への適用ハンドラ（対象ループが per-target に呼ぶ）。
_TARGET_HANDLERS: Dict[ActionType, Callable] = {}


def game_handler(*types: ActionType, when: Optional[Callable] = None):
    """プレイヤーレベル・ハンドラを登録するデコレータ。"""
    def deco(fn):
        for t in types:
            _GAME_HANDLERS[t] = (fn, when)
        return fn
    return deco


def target_handler(*types: ActionType):
    """対象ループ・ハンドラを登録するデコレータ。
    シグネチャ: (gm, player, action, target, owner, source_list, value, source_card) -> None
    """
    def deco(fn):
        for t in types:
            _TARGET_HANDLERS[t] = fn
        return fn
    return deco


def normalize(action_type) -> Optional[ActionType]:
    """`action.type` を ActionType へ正規化する。

    従来の `action.type.name if hasattr(...) else str(...)` の防御を enum 側で吸収する。
    ActionType ならそのまま。文字列名なら ActionType[name]（エイリアスも解決）。未知なら None。
    """
    if isinstance(action_type, ActionType):
        return action_type
    try:
        return ActionType[str(action_type)]
    except KeyError:
        return None
