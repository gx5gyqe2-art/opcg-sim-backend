from typing import Optional, List, Any, Dict
from ...models.enums import ActionType, Zone, ConditionType, CompareOperator
from ...models.effect_types import EffectAction, Condition
from ...utils.logger_config import log_event

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..gamestate import GameManager, Player, CardInstance

def check_condition(game_manager: 'GameManager', player: 'Player', condition: Optional[Condition], source_card: 'CardInstance') -> bool:
    if not condition: return True
    from .matcher import get_target_cards
    
    res = False
    if condition.target:
        matches = get_target_cards(game_manager, condition.target, source_card)
        res = len(matches) > 0
        log_event("DEBUG", "resolver.check_condition_target", f"Target condition: {len(matches)} matches", player=player.name)
    elif condition.type == ConditionType.LIFE_COUNT:
        val = len(player.life)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    
    log_event("INFO", "resolver.condition_result", f"Condition [{condition.raw_text}]: {res}", player=player.name)
    return res

def execute_action(
    game_manager: 'GameManager', 
    player: 'Player', 
    action: EffectAction, 
    source_card: 'CardInstance', 
    effect_context: Optional[Dict[str, Any]] = None
) -> bool:
    from .matcher import get_target_cards
    effect_context = effect_context or {}

    if not check_condition(game_manager, player, action.condition, source_card):
        return True

    targets = []
    if action.target:
        if action.target.select_mode == "REFERENCE":
            last_uuid = effect_context.get("last_target_uuid")
            if last_uuid:
                ref_card = game_manager._find_card_by_uuid(last_uuid)
                if ref_card: targets = [ref_card]
            log_event("DEBUG", "resolver.resolve_reference", f"Resolved reference to: {[t.name for t in targets]}", player=player.name)
        else:
            candidates = get_target_cards(game_manager, action.target, source_card)
            selected_uuids = effect_context.get("selected_uuids")
            
            if action.target.select_mode not in ["ALL", "SOURCE", "SELF"] and len(candidates) > 0:
                if selected_uuids is None:
                    log_event("INFO", "resolver.suspend", f"Selection required for {action.type}", player=player.name)
                    game_manager.active_interaction = {
                        "player_id": player.name,
                        "action_type": "SEARCH_AND_SELECT",
                        "selectable_uuids": [c.uuid for c in candidates],
                        "continuation": {
                            "action": action,
                            "source_card_uuid": source_card.uuid,
                            "effect_context": effect_context
                        }
                    }
                    return False
                targets = [c for c in candidates if c.uuid in selected_uuids]
            else:
                targets = candidates

    if targets and action.target and action.target.tag == "last_target":
        effect_context["last_target_uuid"] = targets[0].uuid

    self_execute(game_manager, player, action, targets)

    if action.then_actions:
        for sub in action.then_actions:
            if not execute_action(game_manager, player, sub, source_card, effect_context):
                return False
    return True

def self_execute(game_manager, player, action, targets):
    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)
    elif action.type == ActionType.KO:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.TRASH, owner)
    elif action.type == ActionType.BUFF:
        for t in targets: t.power_buff += action.value
