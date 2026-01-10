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
            # 中断要求があればループを抜ける
            if self.game_manager.active_interaction:
                return

            node = self.execution_stack.pop()

            if isinstance(node, GameAction):
                success = self._execute_game_action(player, node, source_card)
                
                # 中断時はここで終了（再開時はcontinuationから復元される）
                if self.game_manager.active_interaction:
                    return

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
        # 【修正】現在のアクション(action)を引数として渡す
        targets = self._resolve_targets(player, action.target, source_card, action_node=action)
        
        # Noneが返ってきた場合は「ユーザー選択待ち」のため中断
        if targets is None:
            return False

        if action.target and not targets:
            log_event("INFO", "resolver.no_targets", f"No targets found for action {action.type.name}", player=player.name)
            return False

        value = self._calculate_value(player, action.value, targets)
        success = self.game_manager.apply_action_to_engine(player, action, targets, value)
        
        if not success:
            log_event("WARNING", "resolver.engine_failed", f"Engine failed to apply action {action.type.name}", player=player.name)
        
        return success

    def _resolve_targets(self, player, query, source_card, action_node=None):
        if not query: return []
        
        # 保存されたターゲットがある場合はそれを使う（再開時）
        if query.save_id and query.save_id in self.context["saved_targets"]:
            return self.context["saved_targets"][query.save_id]
        
        from .matcher import get_target_cards
        candidates = get_target_cards(self.game_manager, query, source_card)
        
        required_count = getattr(query, 'count', 1)
        is_optional = getattr(query, 'optional', False)
        
        # 候補がない場合
        if len(candidates) == 0:
            return []
            
        # 自動解決（候補数が要求数以下、かつ select_mode="ALL" または必須）
        if (query.select_mode == "ALL") or (len(candidates) <= required_count and not is_optional):
            if query.save_id:
                self.context["saved_targets"][query.save_id] = candidates
            return candidates

        # ユーザー選択が必要なため中断
        # 【修正】action_node を渡して、保存するスタックに含めるようにする
        self._suspend_for_target_selection(player, candidates, query, source_card, action_node)
        return None

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

    def _suspend_for_target_selection(self, player, candidates, query, source_card, action_node=None):
        required_count = getattr(query, 'count', 1)
        
        # 【修正】スタックの状態を保存する際、実行中だったアクション(action_node)をスタックの末尾に戻す
        # これにより再開時にこのアクションから再試行される
        saved_stack = self.execution_stack.copy()
        if action_node:
            saved_stack.append(action_node)

        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "SELECT_TARGET",
            "message": f"対象を選択してください（最大{required_count}枚）",
            "candidates": candidates,
            "constraints": {
                "min": 1,
                "max": required_count
            },
            "continuation": {
                "execution_stack": saved_stack, # 修正後のスタックを保存
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
                "query": query
            }
        }
        log_event("INFO", "resolver.suspend", "Suspended for target selection", player=player.name)

    def resume_choice(self, player, source_card, selected_index, execution_stack, effect_context):
        self.execution_stack = execution_stack
        self.context = effect_context
        
        cont = self.game_manager.active_interaction.get("continuation") if self.game_manager.active_interaction else None
        if not cont: return
        
        node = cont.get("node")
        if node and 0 <= selected_index < len(node.options):
            self.execution_stack.append(node.options[selected_index])
        
        self.game_manager.active_interaction = None
        self._process_stack(player, source_card)

    def resume_execution(self, player, source_card, execution_stack, effect_context):
        """中断からの復帰用メソッド"""
        self.execution_stack = execution_stack
        self.context = effect_context
        # 再開時は、中断していたアクションがスタックの先頭（末尾）にあるはずなので、
        # そのままループを再開すればよい。
        # _execute_game_action -> _resolve_targets の中で save_id をチェックし、
        # 保存されたターゲットを使って即座に完了するはず。
        self._process_stack(player, source_card)
