from ...models.enums import ActionType, Zone
from .matcher import get_target_cards
from ...utils.logger_config import log_event

def execute_action(game_manager, player, action, source_card):
    targets = []
    if action.target:
        targets = get_target_cards(game_manager, action.target, source_card)

    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)

    elif action.type == ActionType.BP_BUFF:
        for card in targets:
            card.power_buff += action.value

    elif action.type == ActionType.KO:
        for card in targets:
            owner, _ = game_manager._find_card_location(card)
            game_manager.move_card(card, Zone.TRASH, owner if owner else player)

    elif action.type == ActionType.MOVE_CARD:
        for card in targets:
            game_manager.move_card(card, action.dest_zone, player, action.dest_position)

    elif action.type == ActionType.REST:
        for card in targets: 
            card.is_rest = True

    elif action.type == ActionType.ACTIVE:
        for card in targets: 
            card.is_rest = False

    for then_action in action.then_actions:
        execute_action(game_manager, player, then_action, source_card)
