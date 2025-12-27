import logging
from ...models.enums import ActionType, Zone
from .matcher import get_target_cards

logger = logging.getLogger("opcg_sim")

def execute_action(game_manager, player, action, source_card):
    """EffectAction を GameState に適用する"""
    targets = []
    if action.target:
        targets = get_target_cards(game_manager, action.target, source_card)

    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)

    elif action.type == ActionType.BP_BUFF:
        for card in targets:
            card.power_buff += action.value
            logger.info(f"BP_BUFF: {card.master.name} {action.value}")

    elif action.type == ActionType.KO:
        for card in targets:
            owner, _ = game_manager._find_card_location(card)
            game_manager.move_card(card, Zone.TRASH, owner if owner else player)

    elif action.type == ActionType.MOVE_CARD:
        for card in targets:
            game_manager.move_card(card, action.dest_zone, player, action.dest_position)

    elif action.type == ActionType.REST:
        for card in targets: card.is_rest = True

    elif action.type == ActionType.ACTIVE:
        for card in targets: card.is_rest = False

    # 連鎖アクションの解決
    for then_action in action.then_actions:
        execute_action(game_manager, player, then_action, source_card)
