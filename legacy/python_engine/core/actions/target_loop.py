"""対象ループのランナー（除去保護/置換ゲート・B2 退避・success 規約を一元管理）。

旧 `apply_action_to_engine` 後半の `for target in targets:` 前処理を逐語移設。各対象への実適用は
`per_target` の `@target_handler` へ委譲する。act_name 文字列比較を ActionType へ置換したのみ。
"""
from ...models.enums import ActionType
from .registry import _TARGET_HANDLERS

# 「相手の効果で場を離れない」対象になり得る除去アクション（旧 _LEAVE_ACTIONS の enum 版）。
_LEAVE_ACTIONS = frozenset({
    ActionType.KO, ActionType.DISCARD, ActionType.TRASH, ActionType.BOUNCE,
    ActionType.MOVE_TO_HAND, ActionType.MOVE, ActionType.DECK_BOTTOM,
    ActionType.DECK_TOP, ActionType.MOVE_CARD,
})


def run_target_loop(gm, player, action, atype, targets, value, source_card) -> bool:
    handler = _TARGET_HANDLERS.get(atype)   # None でもループは回す（旧 no-op 挙動）
    # 初期値 True: 対象0枚でも「何もしないことに成功した」とみなす（旧 success 規約）。
    success = True
    for target in targets:
        owner, source_list = gm._find_card_location(target)
        if not owner:
            continue
        # 相手の効果でフィールド上のカードを場から除去しようとする場合、保護/置換を確認。
        #   "LEAVE"     = あらゆる除去（場を離れない）に効く保護。
        #   "EFFECT_KO" = KO 限定の保護。KO は LEAVE か EFFECT_KO で防がれ、非KO除去は LEAVE のみ。
        if (atype in _LEAVE_ACTIONS and player.name != owner.name
                and source_list is owner.field):
            guard_statuses = ("LEAVE", "EFFECT_KO") if atype is ActionType.KO else ("LEAVE",)
            if gm._active_protection(target, guard_statuses, actor=player):
                continue
            # 置換が内側中断を提示した（B2）: この時点で active_interaction が立つ。残対象を
            # deferred フレームへ退避してループを抜ける（内側中断の解決後に再開する）。
            if gm._active_replacement(target, guard_statuses, can_suspend=True):
                if gm.active_interaction is not None:
                    remaining = targets[targets.index(target) + 1:]
                    if remaining:
                        gm._defer_removal_targets(player, action, remaining, value)
                    return success
                continue
        if handler is not None:
            handler(gm, player, action, target, owner, source_list, value, source_card)
    return success
