from typing import List, Any, Dict, Optional
from ...models.effect_types import (
    EffectNode, GameAction, Sequence, Branch, Choice, ValueSource
)
from ...models.enums import ActionType, Zone, TriggerType
from ...utils.logger_config import log_event

class EffectResolver:
    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.execution_stack: List[EffectNode] = []
        self.context: Dict[str, Any] = {
            "saved_targets": {},  # save_id によるカード参照用
            "saved_values": {},   # save_count による数値保持用
            "last_action_success": True
        }

    def resolve_ability(self, player, ability, source_card):
        # 1. 発動条件のチェック
        if ability.condition and not self._check_condition(player, ability.condition, source_card):
            return

        # 2. スタックの初期化（コスト -> 効果の順に積む）
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
                if not success and "cost" in str(type(node)): # コスト失敗時は中断
                    self.execution_stack.clear()
                    return

            elif isinstance(node, Sequence):
                # 先頭から実行するため、逆順にスタックに積む
                for sub_node in reversed(node.actions):
                    self.execution_stack.append(sub_node)

            elif isinstance(node, Branch):
                if self._check_condition(player, node.condition, source_card):
                    if node.if_true:
                        self.execution_stack.append(node.if_true)
                elif node.if_false:
                    self.execution_stack.append(node.if_false)

            elif isinstance(node, Choice):
                # ユーザー選択が必要なため、スタックを保持して中断
                self._suspend_for_choice(player, node, source_card)
                return 

    def _execute_game_action(self, player, action: GameAction, source_card) -> bool:
        log_event("INFO", "resolver.action", f"Executing {action.type}", player=player.name)
        
        # ターゲットの解決
        targets = self._resolve_targets(player, action.target, source_card)
        
        # 値の解決（ValueSource の計算）
        value = self._calculate_value(player, action.value, targets)

        # 実際のゲーム状態変更（既存の self_execute 相当のロジック）
        # ここで action.save_id があれば targets を self.context["saved_targets"] に保存
        return self.game_manager.apply_action_to_engine(player, action, targets, value)

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
        if not val_source.dynamic_source:
            return val_source.base
        
        # 動的な値計算（トラッシュ枚数、対象のパワー等）
        base_val = self.game_manager.get_dynamic_value(player, val_source, targets, self.context)
        return (base_val // val_source.divisor) * val_source.multiplier

    def _suspend_for_choice(self, player, node: Choice, source_card):
        # GameManagerに現在のスタックとコンテキストを預けてAPIレスポンスを返す
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "CHOICE",
            "message": node.message,
            "options": node.option_labels,
            "continuation": {
                "stack": self.execution_stack,
                "context": self.context,
                "source_card_uuid": source_card.uuid,
                "node": node # どのChoiceを処理中か保持
            }
        }
        log_event("INFO", "resolver.suspend", "Suspended for player choice", player=player.name)

    def resume_choice(self, player, selected_index):
        # 中断していたスタックを復元して再開
        cont = self.game_manager.active_interaction["continuation"]
        self.execution_stack = cont["stack"]
        self.context = cont["context"]
        source_card = self.game_manager._find_card_by_uuid(cont["source_card_uuid"])
        node = cont["node"]

        # 選ばれた選択肢をスタックに積んで再開
        if 0 <= selected_index < len(node.options):
            self.execution_stack.append(node.options[selected_index])
        
        self.game_manager.active_interaction = None
        self._process_stack(player, source_card)
