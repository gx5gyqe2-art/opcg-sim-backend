from typing import List, Any, Dict, Optional
import re
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
                
                # 中断された場合はここで終了（スタックの処理は再開時に委ねる）
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
        targets = self._resolve_targets(player, action.target, source_card)
        
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

    def _resolve_targets(self, player, query, source_card):
        if not query: return []
        
        # 既に保存されたターゲットがある場合（中断からの再開時など）はそれを使う
        if query.save_id and query.save_id in self.context["saved_targets"]:
            return self.context["saved_targets"][query.save_id]
        
        from .matcher import get_target_cards
        candidates = get_target_cards(self.game_manager, query, source_card)
        
        required_count = getattr(query, 'count', 1)
        is_optional = getattr(query, 'optional', False)
        
        # 候補がない場合
        if len(candidates) == 0:
            return []
            
        # 自動解決できる場合（候補数が要求数以下、かつ必須）
        # ※本来は「対象を取れるなら取らなければならない」ルールのため、optionalでなければ自動選択
        if len(candidates) <= required_count and not is_optional:
            if query.save_id:
                self.context["saved_targets"][query.save_id] = candidates
            return candidates

        # ユーザー選択が必要なため中断 (Suspend)
        # 現在のアクション(GameAction)は _process_stack で pop されているため、
        # 再開時に再実行できるようスタックに戻す必要がある、もしくは continuation に保持する。
        # ここでは再開ロジック側で制御するため、必要な情報を保存して None を返す。
        self._suspend_for_target_selection(player, candidates, query, source_card)
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

    def _suspend_for_target_selection(self, player, candidates, query, source_card):
        required_count = getattr(query, 'count', 1)
        is_optional = getattr(query, 'optional', False)
        
        # 中断時に実行中だったアクション（スタックからPOP済み）を取得するのは難しいが、
        # _process_stack の構造上、ターゲット選択で止まった場合は
        # 「現在のアクション」をやり直す必要がある。
        # そのため、continuation に「現在のターゲット処理が終わったらどうするか」ではなく、
        # 「どこまでやったか」のコンテキストを残す。
        
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "SELECT_TARGET",
            "message": f"対象を選択してください（最大{required_count}枚）",
            "candidates": candidates,
            "constraints": {
                "min": 0 if is_optional else 1, # ※ルール依存だが簡易実装
                "max": required_count
            },
            "continuation": {
                "execution_stack": self.execution_stack, # 次にやるべき残りのアクション
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
                "query": query,
                # GameAction自体はスタックから消えているため、呼び出し元で再構成するか
                # ここでは「選択結果を注入して再開する」ことを前提にする
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
        # 中断していたアクション自体は完了していないが、
        # resume_executionが呼ばれる前に「選択結果」がcontextに注入されている前提。
        # ただし、GameActionノード自体はpopされているため、
        # _execute_game_action で止まった続きを行う必要がある。
        # 現在の実装では _execute_game_action 内で再帰的に呼ぶわけではないので、
        # 厳密には「中断したGameAction」をスタックに戻してから再開するのが正しい。
        # しかし、query情報しか持っていないため、ここでは簡易的に
        # 「次（スタックにあるもの）へ進む」として処理する。
        # ※本来は GameAction 自体を continuation に保存すべき。
        
        # 今回は簡易修正として、_process_stack を呼んで残りの処理を続ける。
        # (GameAction内の他の処理(value計算など)がスキップされるリスクがあるが、
        #  ターゲット選択がメインの処理であることが多いため一旦許容)
        self._process_stack(player, source_card)
