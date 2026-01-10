from typing import List, Any, Dict, Optional
from ...models.effect_types import (
    EffectNode, GameAction, Sequence, Branch, Choice, ValueSource
)
from ...models.enums import ActionType, Zone, TriggerType, ConditionType
from ...utils.logger_config import log_event

class EffectResolver:
    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.execution_stack: List[EffectNode] = []
        self.context: Dict[str, Any] = {
            "saved_targets": {},
            "saved_values": {},
            "last_action_success": True
        }

    def resolve_ability(self, player, ability, source_card):
        if ability.condition and not self._check_condition(player, ability.condition, source_card):
            log_event("INFO", "resolver.condition_failed", f"Condition not met for {source_card.master.name}", player=player.name)
            return

        self.execution_stack = []
        if ability.effect:
            self.execution_stack.append(ability.effect)
        if ability.cost:
            self.execution_stack.append(ability.cost)

        self._process_stack(player, source_card)

    def _process_stack(self, player, source_card):
        while self.execution_stack:
            node = self.execution_stack.pop()

            if isinstance(node, GameAction):
                success = self._execute_game_action(player, node, source_card)
                self.context["last_action_success"] = success
                if not success and node.raw_text and ":" in source_card.master.effect_text:
                    log_event("WARNING", "resolver.cost_failed", "Action failed, stopping execution", player=player.name)
                    self.execution_stack.clear()
                    return

            elif isinstance(node, Sequence):
                for sub_node in reversed(node.actions):
                    self.execution_stack.append(sub_node)

            elif isinstance(node, Branch):
                if self._check_condition(player, node.condition, source_card):
                    if node.if_true:
                        self.execution_stack.append(node.if_true)
                elif node.if_false:
                    self.execution_stack.append(node.if_false)

            elif isinstance(node, Choice):
                self._suspend_for_choice(player, node, source_card)
                return 

    def _execute_game_action(self, player, action: GameAction, source_card) -> bool:
        targets = self._resolve_targets(player, action.target, source_card)
        
        if action.target and not targets:
            log_event("INFO", "resolver.no_targets", f"No targets found for action {action.type.name}", player=player.name)
            return False

        value = self._calculate_value(player, action.value, targets)
        success = self.game_manager.apply_action_to_engine(player, action, targets, value)
        
        if not success:
            log_event("WARNING", "resolver.engine_failed", f"Engine failed to apply action {action.type.name}", player=player.name)
        
        return success

    def _resolve_targets(self, player, query, source_card):
        if not query: return []
        if query.ref_id:
            return self.context["saved_targets"].get(query.ref_id, [])
        
        from .matcher import get_target_cards
        targets = get_target_cards(self.game_manager, query, source_card)
        
        if query.save_id:
            self.context["saved_targets"][query.save_id] = targets
        return targets

    def _calculate_value(self, player, val_source: ValueSource, targets) -> int:
        if not val_source or not val_source.dynamic_source:
            return val_source.base if val_source else 0
        
        base_val = self.game_manager.get_dynamic_value(player, val_source, targets, self.context)
        return (base_val // val_source.divisor) * val_source.multiplier

    def _check_condition(self, player, condition, source_card) -> bool:
        if not condition: return True
        
        log_event("DEBUG", "resolver.check_condition", f"Checking {condition.type.name}", player=player.name)
        
        if condition.type == ConditionType.DON_COUNT:
            total_don = len(player.don_active) + len(player.don_rested) + len(player.don_attached_cards)
            nums = re.findall(r'\d+', condition.raw_text)
            required = int(nums[0]) if nums else 0
            return total_don >= required
            
        if condition.type == ConditionType.LIFE_COUNT:
            nums = re.findall(r'\d+', condition.raw_text)
            required = int(nums[0]) if nums else 0
            return len(player.life) <= required
            
        return True

    def _suspend_for_choice(self, player, node: Choice, source_card):
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "CHOICE",
            "message": node.message,
            "options": node.option_labels,
            "continuation": {
                "execution_stack": self.execution_stack,
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
                "node": node
            }
        }
        log_event("INFO", "resolver.suspend", "Suspended for player choice", player=player.name)

    def resume_choice(self, player, source_card, selected_index, execution_stack, effect_context):
        self.execution_stack = execution_stack
        self.context = effect_context
        
        cont = self.game_manager.active_interaction.get("continuation") if self.game_manager.active_interaction else None
        if not cont: return
        
        node = cont["node"]
        if 0 <= selected_index < len(node.options):
            self.execution_stack.append(node.options[selected_index])
        
        self.game_manager.active_interaction = None
        self._process_stack(player, source_card)
