from typing import List, Any, Dict, Optional, Union
import json
from dataclasses import asdict
from ...models.effect_types import (
    EffectNode, GameAction, Sequence, Branch, Choice, ValueSource, Condition, TargetQuery
)
from ...models.enums import ActionType, Zone, TriggerType, ConditionType, CompareOperator, Player
from ...utils.logger_config import log_event
import re

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
            self._log_failure_snapshot(player, source_card, ability, "CONDITION_MISMATCH", f"Condition type: {ability.condition.type.name}")
            log_event("INFO", "resolver.condition_failed", f"Condition not met for {source_card.master.name}", player=player.name)
            return

        if ability.cost and not self._can_satisfy_node(player, ability.cost, source_card):
            self._log_failure_snapshot(player, source_card, ability, "COST_UNSATISFIED", "Insufficient resources or targets for cost")
            log_event("WARNING", "resolver.cost_impossible", f"Cost cannot be satisfied for {source_card.master.name}", player=player.name)
            raise ValueError(f"コストの条件を満たすことができません: {source_card.master.name}")

        self.execution_stack = []
        if ability.effect:
            self.execution_stack.append(ability.effect)
        if ability.cost:
            self.execution_stack.append(ability.cost)

        self._process_stack(player, source_card)

    def _log_failure_snapshot(self, player, source_card, ability, error_code, detail_msg):
        """
        失敗時の状況をJSONにまとめてログ出力する。
        これをコピーしてAIに渡すとデバッグしてくれる。
        """
        try:
            snapshot = self.game_manager.get_debug_snapshot()
            
            try:
                ability_dump = asdict(ability)
                ability_dump = str(ability_dump) 
            except:
                ability_dump = str(ability)

            debug_data = {
                "error_code": error_code,
                "detail": detail_msg,
                "source_card": {
                    "id": source_card.master.id,
                    "name": source_card.master.name,
                    "uuid": source_card.uuid,
                    "zone_location": self.game_manager._find_card_location(source_card)[1].__class__.__name__ if self.game_manager._find_card_location(source_card)[1] else "Unknown"
                },
                "failed_ability": ability_dump, 
                "game_state": snapshot
            }
            
            json_str = json.dumps(debug_data, ensure_ascii=False, indent=2)
            print(f"\n======== [DEBUG_SNAPSHOT_START] ========\nAI_PROMPT: 以下のJSONはOPCGシミュレータのエラーログです。カード {source_card.master.name} の効果発動が {error_code} で失敗しました。game_state と failed_ability を分析し、なぜ条件を満たせなかったのか、またはLLMの生成したJSONデータのどこが間違っているか特定してください。\n\n{json_str}\n======== [DEBUG_SNAPSHOT_END] ========\n")
            
        except Exception as e:
            print(f"Snapshot generation failed: {e}")

    def _can_satisfy_node(self, player, node: EffectNode, source_card) -> bool:
        """
        コスト支払いなどが可能か事前にチェックする
        """
        if isinstance(node, GameAction):
            if not node.target: return True
            # ここでローカルインポートして循環参照回避
            from .matcher import get_target_cards
            candidates = get_target_cards(self.game_manager, node.target, source_card)
            required = getattr(node.target, 'count', 1)
            
            # 厳密な枚数指定がある場合（例：コストとして"2枚"捨てる）
            if getattr(node.target, 'is_strict_count', False) and len(candidates) < required:
                log_event("DEBUG", "resolver.satisfy_fail", f"Insufficient candidates for {node.type.name}: {len(candidates)}/{required}", player=player.name)
                return False
            
            # 任意枚数でないのに候補が0枚の場合
            if not getattr(node.target, 'is_up_to', False) and len(candidates) == 0:
                return False
                
            return True
        elif isinstance(node, Sequence):
            return all(self._can_satisfy_node(player, a, source_card) for a in node.actions)
        elif isinstance(node, Choice):
            # 選択肢の少なくとも1つが実行可能ならOKとする（簡易判定）
            return any(self._can_satisfy_node(player, opt, source_card) for opt in node.options)
        return True

    def _process_stack(self, player, source_card):
        while self.execution_stack:
            if self.game_manager.active_interaction:
                return

            node = self.execution_stack.pop()

            if isinstance(node, GameAction):
                success = self._execute_game_action(player, node, source_card)
                
                if self.game_manager.active_interaction:
                    return

                self.context["last_action_success"] = success
                # コストアクション（効果テキストに : がある場合のコスト部分など）が失敗したら全体を中断
                # 簡易判定として、失敗かつソースにコスト記述がある場合に中断
                # ※より厳密にはnodeがcostの一部かどうか判定が必要だが、現状はこれで運用
                if not success and node.raw_text and ":" in (source_card.master.effect_text or ""):
                    log_event("WARNING", "resolver.cost_failed", "Action failed, stopping execution", player=player.name)
                    self.execution_stack.clear()
                    return

            elif isinstance(node, Sequence):
                # StackはLIFOなので、逆順に積むことで正しい順序で実行される
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
        targets = self._resolve_targets(player, action.target, source_card, action_node=action)
        
        # ターゲット選択が保留された場合（None）はFalseを返すが、処理は中断される
        if targets is None:
            return False

        # ターゲットが必要なアクションで対象が見つからなかった場合
        if action.target and not targets and not getattr(action.target, 'is_up_to', False):
            # is_up_to=Trueなら0枚でも成功扱いだが、そうでないなら失敗
            log_event("INFO", "resolver.no_targets", f"No targets found for action {action.type.name}", player=player.name)
            return False

        value = self._calculate_value(player, action.value, targets)
        success = self.game_manager.apply_action_to_engine(player, action, targets, value)
        
        if not success:
            log_event("WARNING", "resolver.engine_failed", f"Engine failed to apply action {action.type.name}", player=player.name)
        
        return success

    def _resolve_targets(self, player, query, source_card, action_node=None):
        if not query: return []
        
        # ▼▼▼ 追加: 再開時に、一時的に渡された選択結果があればそれを返す ▼▼▼
        if "temp_resolved_targets" in self.context:
            return self.context.pop("temp_resolved_targets")

        # 既に保存されたターゲットがある場合（例：「選んだカードを...」）
        if query.save_id and query.save_id in self.context["saved_targets"]:
            return self.context["saved_targets"][query.save_id]
        
        # 参照IDがある場合（例：「そのカードを...」）
        if query.ref_id:
             if query.ref_id == "self":
                 return [source_card]
             if query.ref_id in self.context["saved_targets"]:
                 return self.context["saved_targets"][query.ref_id]
        
        from .matcher import get_target_cards
        candidates = get_target_cards(self.game_manager, query, source_card)
        
        required_count = getattr(query, 'count', 1)
        is_up_to = getattr(query, 'is_up_to', False)
        is_strict = getattr(query, 'is_strict_count', False)
        is_resource = (query.zone == Zone.COST_AREA) # ドン!!の選択は自動化されやすい
        
        if len(candidates) == 0:
            return []

        if is_strict and len(candidates) < required_count:
            log_event("INFO", "resolver.strict_count_fail", f"Insufficient targets for strict count: found {len(candidates)}, needed {required_count}", player=player.name)
            return []
            
        # 自動選択条件:
        # 1. 全選択モード
        # 2. 候補数が要求数以下（かつ任意選択ではない）
        # 3. リソース（ドン）の消費で、かつ任意選択ではない
        if (query.select_mode == "ALL") or \
           (len(candidates) <= required_count and not is_up_to) or \
           (is_resource and not is_up_to):
            
            selected = candidates[:required_count] if required_count > 0 else candidates
            if query.save_id:
                self.context["saved_targets"][query.save_id] = selected
            return selected

        # ユーザー選択が必要
        # save_idがなくても選択が必要な場合は選択画面を出す（以前のフォールバック削除）
        self._suspend_for_target_selection(player, candidates, query, source_card, action_node)
        return None

    def _calculate_value(self, player, val_source: ValueSource, targets) -> int:
        if not val_source or not val_source.dynamic_source:
            return val_source.base if val_source else 0
        
        base_val = self.game_manager.get_dynamic_value(player, val_source, targets, self.context)
        
        # 倍率と除算の適用
        val = base_val
        if val_source.divisor > 1:
            val = val // val_source.divisor
        if val_source.multiplier != 1:
            val = val * val_source.multiplier
            
        return val

    def _check_condition(self, player, condition: Condition, source_card) -> bool:
        if not condition: return True
        
        # 再帰的な複合条件の判定
        if condition.type == ConditionType.AND:
            return all(self._check_condition(player, sub, source_card) for sub in condition.args)
        if condition.type == ConditionType.OR:
            return any(self._check_condition(player, sub, source_card) for sub in condition.args)

        # ターゲットプレイヤーの特定
        target_player = player
        if condition.player == Player.OPPONENT:
            target_player = self.game_manager.p2 if player == self.game_manager.p1 else self.game_manager.p1
        
        log_event("DEBUG", "resolver.check_condition", f"Checking {condition.type.name} for {target_player.name}", player=player.name)

        current_val = 0
        target_val = condition.value if isinstance(condition.value, int) else 0
        
        # --- カウント系条件 ---
        if condition.type == ConditionType.DON_COUNT:
            # フィールドのドン（アクティブ＋レスト＋付与）
            current_val = len(target_player.don_active) + len(target_player.don_rested) + len(target_player.don_attached_cards)
            # 正規表現でテキストから数値を補完（LLMが数値を入れ忘れた場合のフォールバック）
            if target_val == 0 and isinstance(condition.value, str):
                nums = re.findall(r'\d+', condition.raw_text)
                target_val = int(nums[0]) if nums else 0
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.LIFE_COUNT:
            current_val = len(target_player.life)
            if target_val == 0 and isinstance(condition.value, str):
                nums = re.findall(r'\d+', condition.raw_text)
                target_val = int(nums[0]) if nums else 0
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.HAND_COUNT:
            current_val = len(target_player.hand)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.TRASH_COUNT:
            current_val = len(target_player.trash)
            if target_val == 0 and isinstance(condition.value, str):
                nums = re.findall(r'\d+', condition.raw_text)
                target_val = int(nums[0]) if nums else 0
            return self._compare(current_val, condition.operator, target_val)
            
        elif condition.type == ConditionType.DECK_COUNT:
            current_val = len(target_player.deck)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.FIELD_COUNT:
            # キャラクターとステージの数をカウント（リーダーは除く）
            current_val = len(target_player.field) + (1 if target_player.stage else 0)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.HAS_DON:
            # アクティブなドンの数、または付与されているドンの数
            # ※文脈によるが、一般的には「ドン!!x枚以上をレストにして」などのコスト判定用
            current_val = len(target_player.don_active)
            # もしsource_cardに付与されているドンを見る場合の実装も必要だが、一旦プレイヤーの所持ドンで判定
            return self._compare(current_val, condition.operator, target_val)

        # --- プロパティ系条件 ---
        elif condition.type == ConditionType.LEADER_NAME:
            if not target_player.leader: return False
            expected_name = condition.value
            if isinstance(expected_name, str):
                return expected_name in target_player.leader.master.name
            return False

        elif condition.type == ConditionType.LEADER_TRAIT:
            if not target_player.leader: return False
            expected_trait = condition.value
            if isinstance(expected_trait, str):
                return expected_trait in target_player.leader.master.traits
            return False
            
        elif condition.type in [ConditionType.HAS_TRAIT, ConditionType.HAS_ATTRIBUTE, ConditionType.HAS_UNIT]:
            # matcherを使って条件に合うカードが場にあるか探す
            from .matcher import get_target_cards
            # 条件判定用のクエリを構築
            query = condition.target
            if not query:
                # ターゲットがない場合、簡易的なクエリを作成
                query = TargetQuery(zone=Zone.FIELD, player=condition.player)
                if condition.type == ConditionType.HAS_TRAIT and isinstance(condition.value, str):
                    query.traits = [condition.value]
                elif condition.type == ConditionType.HAS_ATTRIBUTE and isinstance(condition.value, str):
                    query.attributes = [condition.value]
            
            candidates = get_target_cards(self.game_manager, query, source_card)
            count = len(candidates)
            # count >= 1 などを判定
            target_count = 1 if target_val == 0 else target_val
            return self._compare(count, condition.operator, target_count)

        # --- コンテキスト系条件 ---
        elif condition.type == ConditionType.CONTEXT:
            context_val = condition.value
            if context_val == "MY_TURN" or context_val == "SELF_TURN":
                return self.game_manager.turn_player == player
            elif context_val == "OPPONENT_TURN":
                return self.game_manager.turn_player != player
            # 他のコンテキスト（メインフェイズ中など）もここに追加可能
            return True

        elif condition.type == ConditionType.TURN_LIMIT:
            # ターン1回制限のチェック
            # ※現状のGameManagerには各カードの効果発動履歴が必要。
            # ここでは枠組みのみ実装し、常にTrue（未発動）として通すか、
            # 将来的に self.game_manager.has_activated_effect(source_card.uuid, effect_id) を呼ぶ
            return True 
            
        return True

    def _compare(self, current: int, operator: CompareOperator, target: int) -> bool:
        """比較演算のヘルパー"""
        if operator == CompareOperator.EQ: return current == target
        if operator == CompareOperator.NEQ: return current != target
        if operator == CompareOperator.GT: return current > target
        if operator == CompareOperator.LT: return current < target
        if operator == CompareOperator.GE: return current >= target
        if operator == CompareOperator.LE: return current <= target
        return False

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
        is_up_to = getattr(query, 'is_up_to', False)
        
        # ▼▼▼ 修正: 強制か任意かで最小選択数を変える ▼▼▼
        if is_up_to:
            # 「〜まで選ぶ」なら、0枚選択（キャンセル）も許可する
            min_select = 0
        else:
            # 「〜を選ぶ」なら、必ず指定枚数選ばせる（ただし候補数が足りない場合は全選択まで）
            min_select = required_count
            if min_select > len(candidates):
                min_select = len(candidates)
            # 候補が1枚以上あるのに、minが0になるのを防ぐ（強制効果のため）
            if min_select < 1 and len(candidates) > 0:
                min_select = 1
        # ▲▲▲ 修正ここまで ▲▲▲
        
        saved_stack = self.execution_stack.copy()
        if action_node:
            saved_stack.append(action_node)

        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "SELECT_TARGET",
            "message": f"対象を選択してください（最大{required_count}枚）",
            "candidates": candidates,
            "constraints": {
                "min": min_select,
                "max": required_count
            },
            "continuation": {
                "execution_stack": saved_stack,
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
                "query": query
            }
        }
        log_event("INFO", "resolver.suspend", f"Suspended for target selection (min:{min_select}, max:{required_count})", player=player.name)

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
        self.execution_stack = execution_stack
        self.context = effect_context
        self._process_stack(player, source_card)