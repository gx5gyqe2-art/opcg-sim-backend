from typing import Optional, List, Any, Dict
import random
from ...models.enums import ActionType, Zone, ConditionType, CompareOperator, TriggerType
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
    elif condition.type == ConditionType.HAND_COUNT:
        val = len(player.hand)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.DON_COUNT:
        val = len(player.don_active) + len(player.don_rested)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.TRASH_COUNT:
        val = len(player.trash)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.HAS_TRAIT:
        has_in_field = any(condition.value in c.master.traits for c in player.field)
        has_in_source = (source_card and condition.value in source_card.master.traits)
        res = has_in_field or has_in_source
    elif condition.type == ConditionType.LEADER_NAME:
        res = player.leader and player.leader.master.name == condition.value
    
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
    if effect_context is None: effect_context = {}

    if not check_condition(game_manager, player, action.condition, source_card):
        return True

    targets = []
    selected_uuids = effect_context.get("selected_uuids")

    if action.target:
        if action.target.select_mode == "REFERENCE":
            last_uuid = effect_context.get("last_target_uuid")
            if last_uuid:
                ref_card = game_manager._find_card_by_uuid(last_uuid)
                if ref_card: targets = [ref_card]
            log_event("DEBUG", "resolver.resolve_reference", f"Resolved reference to: {[t.name for t in targets]}", player=player.name)
        else:
            candidates = get_target_cards(game_manager, action.target, source_card)
            
            is_search = (action.target.zone == Zone.TEMP) or (action.source_zone == Zone.TEMP)
            
            should_interact = action.target.select_mode not in ["ALL", "SOURCE", "SELF"] and (len(candidates) > 0 or is_search)

            if should_interact:
                if selected_uuids is None:
                    log_event("INFO", "resolver.suspend", f"Selection required for {action.type}. Candidates: {len(candidates)}", player=player.name)
                    
                    display_candidates = candidates
                    if is_search:
                        display_candidates = player.temp_zone
                    
                    game_manager.active_interaction = {
                        "player_id": player.name,
                        "action_type": "SEARCH_AND_SELECT",
                        "message": action.raw_text or "対象を選択してください",
                        "candidates": display_candidates, 
                        "selectable_uuids": [c.uuid for c in candidates],
                        "can_skip": True,
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

    self_execute(game_manager, player, action, targets, source_card=source_card)

    if action.then_actions:
        for sub in action.then_actions:
            if not execute_action(game_manager, player, sub, source_card, effect_context):
                return False
                
    return True

def self_execute(game_manager, player, action, targets, source_card=None):
    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)
    elif action.type == ActionType.RAMP_DON:
        for _ in range(action.value):
            if player.don_deck:
                don = player.don_deck.pop(0)
                if 'レスト' in action.raw_text:
                    don.is_rest = True
                    player.don_rested.append(don)
                else:
                    don.is_rest = False
                    player.don_active.append(don)
        log_event("INFO", "resolver.ramp_don", f"Ramped {action.value} Don", player=player.name)
    elif action.type == ActionType.LOOK:
        moved_count = 0
        for _ in range(action.value):
            if player.deck:
                card = player.deck.pop(0)
                player.temp_zone.append(card)
                moved_count += 1
        log_event("INFO", "resolver.look", f"Moved {moved_count} cards to temp_zone", player=player.name)
    elif action.type == ActionType.KO:
        for t in targets:
            if "PREVENT_LEAVE" in t.flags:
                log_event("INFO", "resolver.prevent_leave", f"{t.master.name} is protected from leaving field", player=player.name)
                continue
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.TRASH, owner)
    elif action.type == ActionType.MOVE_TO_HAND:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.HAND, owner)
    elif action.type == ActionType.TRASH:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.TRASH, owner)
    elif action.type == ActionType.DECK_BOTTOM:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.DECK, owner, dest_position="BOTTOM")
    elif action.type == ActionType.BUFF:
        for t in targets: t.power_buff += action.value
    elif action.type == ActionType.REST:
        for t in targets: t.is_rest = True
    elif action.type == ActionType.ACTIVE:
        for t in targets: t.is_rest = False
    elif action.type == ActionType.ATTACH_DON:
        if targets and player.don_active:
            don = player.don_active.pop(0)
            target_card = targets[0]
            don.attached_to = target_card.uuid
            player.don_attached_cards.append(don)
            target_card.attached_don += 1
            
    elif action.type == ActionType.COST_CHANGE:
        for t in targets:
            t.cost_buff += action.value
            log_event("INFO", "effect.cost_change", f"{t.master.name} cost buffed by {action.value}", player=player.name)

    elif action.type == ActionType.LIFE_MANIPULATE:
        txt = action.raw_text
        if '手札' in txt and '加える' in txt:
            if targets:
                for t in targets:
                    owner, current_zone = game_manager._find_card_location(t)
                    if owner and current_zone == owner.life:
                        game_manager.move_card(t, Zone.HAND, owner)
                        log_event("INFO", "effect.life_to_hand", f"Moved {t.master.name} from Life to Hand", player=player.name)
        elif ('加える' in txt or '置く' in txt) and '手札' not in txt:
            source_list = player.deck
            if targets:
                for t in targets:
                    owner, current_zone = game_manager._find_card_location(t)
                    if owner:
                        game_manager.move_card(t, Zone.LIFE, owner, dest_position="TOP")
                        log_event("INFO", "effect.life_recover", f"Added {t.master.name} to Life", player=player.name)
            else:
                if source_list:
                    card = source_list.pop(0)
                    player.life.append(card)
                    log_event("INFO", "effect.life_recover", "Recovered 1 Life from Deck", player=player.name)
        elif '向き' in txt:
            target_lives = targets if targets else player.life
            for card in target_lives:
                if '表' in txt: card.is_face_up = True
                elif '裏' in txt: card.is_face_up = False
                log_event("INFO", "effect.life_face", f"Life {card.uuid} face changed", player=player.name)

    elif action.type == ActionType.GRANT_KEYWORD:
        keywords = {
            "速攻": "速攻",
            "ブロッカー": "ブロッカー",
            "バニッシュ": "バニッシュ",
            "ダブルアタック": "ダブルアタック",
            "突進": "突進",
            "再起動": "再起動"
        }
        found_kw = None
        for k, v in keywords.items():
            if k in action.raw_text:
                found_kw = v
                break
        if found_kw:
            for t in targets:
                t.current_keywords.add(found_kw)
                log_event("INFO", "effect.grant_keyword", f"Granted [{found_kw}] to {t.master.name}", player=player.name)

    elif action.type == ActionType.ATTACK_DISABLE:
        for t in targets:
            t.flags.add("ATTACK_DISABLE")
            log_event("INFO", "effect.attack_disable", f"{t.master.name} cannot attack", player=player.name)

    elif action.type == ActionType.NEGATE_EFFECT:
        for t in targets:
            t.negated = True
            log_event("INFO", "effect.negate", f"{t.master.name} effects negated", player=player.name)

    elif action.type == ActionType.EXECUTE_MAIN_EFFECT:
        exec_targets = targets if targets else [source_card] if source_card else []
        for t in exec_targets:
             if not t or not t.master.abilities: continue
             target_ability = next((a for a in t.master.abilities if a.trigger not in [TriggerType.TRIGGER, TriggerType.COUNTER]), None)
             if target_ability:
                 log_event("INFO", "effect.execute_main", f"Executing nested ability of {t.master.name}", player=player.name)
                 game_manager.resolve_ability(player, target_ability, source_card=t)

    elif action.type == ActionType.VICTORY:
        game_manager.winner = player.name
        log_event("INFO", "game.victory_special", f"{player.name} wins by effect!", player=player.name)

    elif action.type == ActionType.RULE_PROCESSING:
        log_event("INFO", "effect.rule", f"Static rule processed: {action.raw_text}", player=player.name)

    elif action.type == ActionType.RESTRICTION:
        log_event("INFO", "effect.restriction", f"Restriction applied: {action.raw_text}", player=player.name)

    elif action.type == ActionType.DECK_TOP:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner:
                game_manager.move_card(t, Zone.DECK, owner, dest_position="TOP")
                log_event("INFO", "effect.deck_top", f"Moved {t.master.name} to Deck Top", player=player.name)

    elif action.type == ActionType.DEAL_DAMAGE:
        target_p = player if '自分' in action.raw_text or '受ける' in action.raw_text else game_manager.opponent
        if target_p.life:
            life_card = target_p.life.pop(0)
            target_p.hand.append(life_card)
            log_event("INFO", "effect.damage", f"{target_p.name} took 1 damage from effect", player=player.name)

    elif action.type == ActionType.SELECT_OPTION:
        log_event("INFO", "effect.select_option", "Option selection required (Logic pending)", player=player.name)

    elif action.type == ActionType.SET_COST:
        log_event("INFO", "effect.set_cost", f"Cost set request: {action.raw_text}", player=player.name)

    elif action.type == ActionType.SHUFFLE:
        random.shuffle(player.deck)
        log_event("INFO", "effect.shuffle", "Deck shuffled", player=player.name)

    elif action.type == ActionType.PREVENT_LEAVE:
        for t in targets:
            t.flags.add("PREVENT_LEAVE")
            log_event("INFO", "effect.prevent_leave", f"{t.master.name} gained PREVENT_LEAVE", player=player.name)

    elif action.type == ActionType.REPLACE_EFFECT:
        log_event("INFO", "effect.replace", f"Replacement Effect: {action.raw_text}", player=player.name)

    elif action.type == ActionType.MOVE_ATTACHED_DON:
        don_source = source_card if source_card and source_card.attached_don > 0 else None
        
        if don_source and targets:
            target_card = targets[0]
            moving_don = next((d for d in player.don_attached_cards if d.attached_to == don_source.uuid), None)
            
            if moving_don:
                moving_don.attached_to = target_card.uuid
                don_source.attached_don -= 1
                target_card.attached_don += 1
                log_event("INFO", "effect.move_don", f"Moved Don from {don_source.master.name} to {target_card.master.name}", player=player.name)

    elif action.type == ActionType.MODIFY_DON_PHASE:
        log_event("INFO", "effect.modify_don_phase", f"Don Phase Modified: {action.raw_text}", player=player.name)

    elif action.type == ActionType.REDIRECT_ATTACK:
        if game_manager.active_battle and targets:
            new_target = targets[0]
            game_manager.active_battle["target"] = new_target
            log_event("INFO", "effect.redirect", f"Attack target changed to {new_target.master.name}", player=player.name)

    elif action.type == ActionType.RETURN_DON:
        target_player = game_manager.opponent if "相手" in action.raw_text else player
        count = action.value if action.value > 0 else 1
        
        returned_count = 0
        
        while returned_count < count and target_player.don_active:
            don = target_player.don_active.pop()
            target_player.don_deck.append(don)
            returned_count += 1
            
        while returned_count < count and target_player.don_rested:
            don = target_player.don_rested.pop()
            target_player.don_deck.append(don)
            returned_count += 1
            
        if returned_count < count and target_player.don_attached_cards:
            while returned_count < count and target_player.don_attached_cards:
                don = target_player.don_attached_cards.pop()
                if don.attached_to:
                    attached_card = game_manager._find_card_by_uuid(don.attached_to)
                    if attached_card:
                        attached_card.attached_don = max(0, attached_card.attached_don - 1)
                
                don.attached_to = None
                target_player.don_deck.append(don)
                returned_count += 1

        log_event("INFO", "effect.return_don", f"{target_player.name} returned {returned_count} Don to deck", player=player.name)
