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
        # ▼▼▼ 追加: 実行履歴を記録するリスト ▼▼▼
        self.action_history: List[Dict[str, Any]] = []

    def resolve_ability(self, player, ability, source_card):
        # 1. 条件チェック
        if ability.condition and not self._check_condition(player, ability.condition, source_card):
            self._log_failure_snapshot(player, source_card, ability, "CONDITION_MISMATCH", f"Condition type: {ability.condition.type.name}")
            log_event("INFO", "resolver.condition_failed", f"Condition not met for {source_card.master.name}", player=player.name)
            return

        # 1.5 使用回数制限（【ターン1回】等）の enforce。
        #   ability_used_this_turn は reset_turn_status（毎ターン境界で呼ばれる）で
        #   クリアされるため、ターン単位の使用回数として機能する。
        turn_limit = self._turn_limit_of(ability.condition)
        limit_key = used_count = None
        if turn_limit is not None:
            limit_key = self._ability_key(source_card, ability)
            used_count = source_card.ability_used_this_turn.get(limit_key, 0)
            if used_count >= turn_limit:
                self._log_failure_snapshot(player, source_card, ability, "TURN_LIMIT_REACHED", f"Used {used_count}/{turn_limit} this turn")
                log_event("WARNING", "resolver.turn_limit", f"Ability already used {used_count}/{turn_limit} this turn: {source_card.master.name}", player=player.name)
                return

        # 2. コストチェック
        if ability.cost and not self._can_satisfy_node(player, ability.cost, source_card):
            self._log_failure_snapshot(player, source_card, ability, "COST_UNSATISFIED", "Insufficient resources or targets for cost")
            log_event("WARNING", "resolver.cost_impossible", f"Cost cannot be satisfied for {source_card.master.name}", player=player.name)
            raise ValueError(f"コストの条件を満たすことができません: {source_card.master.name}")

        # 発動成立（条件・コストを満たした）→ 使用回数を消費する。
        if turn_limit is not None:
            source_card.ability_used_this_turn[limit_key] = used_count + 1

        self.execution_stack = []
        if ability.effect:
            self.execution_stack.append(ability.effect)
        if ability.cost:
            self.execution_stack.append(ability.cost)

        # 3. 実行
        self._process_stack(player, source_card)
        
        # ▼▼▼ 追加: 処理完了時にレポートを出力（中断されていなければ） ▼▼▼
        if not self.game_manager.active_interaction:
            self._log_execution_report(player, source_card, ability)

    def _turn_limit_of(self, condition):
        """条件ツリー中の TURN_LIMIT 制限値を返す（無ければ None）。AND/OR の入れ子も探索。"""
        if condition is None:
            return None
        if condition.type == ConditionType.TURN_LIMIT:
            v = condition.value
            return v if isinstance(v, int) and v > 0 else 1
        if condition.type in (ConditionType.AND, ConditionType.OR):
            for sub in condition.args:
                v = self._turn_limit_of(sub)
                if v is not None:
                    return v
        return None

    def _ability_key(self, source_card, ability):
        """使用回数カウンタのキー。master.abilities 内の位置（同一性）で識別する。"""
        abilities = getattr(source_card.master, 'abilities', ()) or ()
        for i, ab in enumerate(abilities):
            if ab is ability:
                return i
        return id(ability)

    def _log_execution_report(self, player, source_card, ability):
        """
        効果処理の結果（何をしてどうなったか）をまとめて出力する
        """
        try:
            snapshot = self.game_manager.get_debug_snapshot()
            
            report = {
                "source_card": f"{source_card.master.name} ({source_card.uuid[:4]})",
                "player": player.name,
                "action_history": self.action_history, # 実行したアクションの履歴
                "final_game_state": snapshot
            }
            
            json_str = json.dumps(report, ensure_ascii=False, indent=2)
            print(f"\n======== [EXECUTION_REPORT_START] ========\nAI_PROMPT: 以下のJSONは効果発動の実行結果レポートです。意図通りにアクションが実行されたか、対象選択や数値計算に間違いがないか確認してください。\n\n{json_str}\n======== [EXECUTION_REPORT_END] ========\n")
        except Exception as e:
            print(f"Report generation failed: {e}")

    def _log_failure_snapshot(self, player, source_card, ability, error_code, detail_msg):
        try:
            snapshot = self.game_manager.get_debug_snapshot()
            try:
                ability_dump = str(asdict(ability)) 
            except:
                ability_dump = str(ability)

            debug_data = {
                "error_code": error_code,
                "detail": detail_msg,
                "source_card": source_card.master.name,
                "failed_ability": ability_dump, 
                "game_state": snapshot
            }
            json_str = json.dumps(debug_data, ensure_ascii=False, indent=2)
            print(f"\n======== [DEBUG_SNAPSHOT_START] ========\n{json_str}\n======== [DEBUG_SNAPSHOT_END] ========\n")
        except: pass

    def _can_satisfy_node(self, player, node: EffectNode, source_card) -> bool:
        if isinstance(node, GameAction):
            if not node.target: return True
            from .matcher import get_target_cards
            candidates = get_target_cards(self.game_manager, node.target, source_card)
            required = getattr(node.target, 'count', 1)
            if getattr(node.target, 'is_strict_count', False) and len(candidates) < required:
                log_event("DEBUG", "resolver.satisfy_fail", f"Insufficient candidates for {node.type.name}: {len(candidates)}/{required}", player=player.name)
                return False
            if not getattr(node.target, 'is_up_to', False) and len(candidates) == 0:
                return False
            return True
        elif isinstance(node, Sequence):
            return all(self._can_satisfy_node(player, a, source_card) for a in node.actions)
        elif isinstance(node, Choice):
            return any(self._can_satisfy_node(player, opt, source_card) for opt in node.options)
        return True

    def _process_stack(self, player, source_card):
        while self.execution_stack:
            if self.game_manager.active_interaction:
                return

            node = self.execution_stack.pop()

            if isinstance(node, GameAction):
                # 「このカードの【メイン】効果を発動する」: 自身の ACTIVATE_MAIN
                # 能力の効果を実行スタックに展開して再発動する（主にトリガー）。
                if node.type == ActionType.EXECUTE_MAIN_EFFECT:
                    self._expand_main_effect(source_card)
                    continue

                success = self._execute_game_action(player, node, source_card)

                if self.game_manager.active_interaction:
                    return

                self.context["last_action_success"] = success
                if not success and node.raw_text and ":" in (source_card.master.effect_text or ""):
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

    def _expand_main_effect(self, source_card):
        """source_card 自身の 【メイン】(ACTIVATE_MAIN) 能力の効果を実行スタックへ展開する。

        トリガー「このカードの【メイン】効果を発動する」用。コストは支払わず効果のみ。
        多段展開（自己参照）による無限ループを防ぐためフラグで1回に限定する。
        """
        if self.context.get("_main_expanded"):
            log_event("WARNING", "resolver.execute_main_loop", "EXECUTE_MAIN_EFFECT re-entry blocked", player=source_card.owner_id)
            return
        self.context["_main_expanded"] = True

        main_abilities = [
            ab for ab in source_card.master.abilities
            if ab.trigger == TriggerType.ACTIVATE_MAIN and ab.effect is not None
        ]
        if not main_abilities:
            log_event("WARNING", "resolver.execute_main_missing", f"No ACTIVATE_MAIN ability on {source_card.master.name}", player=source_card.owner_id)
            return

        # 既存スタックの「後」に積む = 先に実行されるよう reversed で push
        for ab in reversed(main_abilities):
            self.execution_stack.append(ab.effect)
        log_event("INFO", "resolver.execute_main", f"Expanded {len(main_abilities)} main effect(s) of {source_card.master.name}", player=source_card.owner_id)

    def _execute_game_action(self, player, action: GameAction, source_card) -> bool:
        targets = self._resolve_targets(player, action.target, source_card, action_node=action)
        
        if targets is None:
            return False

        if action.target and not targets and not getattr(action.target, 'is_up_to', False):
            log_event("INFO", "resolver.no_targets", f"No targets found for action {action.type.name}", player=player.name)
            # ▼▼▼ 追加: 失敗履歴 ▼▼▼
            self.action_history.append({
                "action": action.type.name if hasattr(action.type, 'name') else str(action.type),
                "success": False,
                "reason": "No targets found"
            })
            return False

        value = self._calculate_value(player, action.value, targets)
        success = self.game_manager.apply_action_to_engine(player, action, targets, value)
        
        # ▼▼▼ 追加: 実行履歴を記録 ▼▼▼
        target_names = [f"{t.master.name}({t.uuid[:4]})" for t in targets]
        self.action_history.append({
            "action": action.type.name if hasattr(action.type, 'name') else str(action.type),
            "success": success,
            "targets": target_names,
            "value": value
        })
        
        return success

    def _resolve_targets(self, player, query, source_card, action_node=None):
        if not query: return []
        
        if "temp_resolved_targets" in self.context:
            return self.context.pop("temp_resolved_targets")

        if query.save_id and query.save_id in self.context["saved_targets"]:
            return self.context["saved_targets"][query.save_id]
        
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
        is_resource = (query.zone == Zone.COST_AREA)
        
        if len(candidates) == 0:
            return []

        if is_strict and len(candidates) < required_count:
            log_event("INFO", "resolver.strict_count_fail", f"Insufficient targets for strict count: found {len(candidates)}, needed {required_count}", player=player.name)
            return []
            
        if (query.select_mode == "ALL") or \
           (len(candidates) <= required_count and not is_up_to) or \
           (is_resource and not is_up_to):
            
            selected = candidates[:required_count] if required_count > 0 else candidates
            if query.save_id:
                self.context["saved_targets"][query.save_id] = selected
            return selected

        self._suspend_for_target_selection(player, candidates, query, source_card, action_node)
        return None

    def _calculate_value(self, player, val_source: ValueSource, targets) -> int:
        if not val_source or not val_source.dynamic_source:
            return val_source.base if val_source else 0
        
        base_val = self.game_manager.get_dynamic_value(player, val_source, targets, self.context)
        
        val = base_val
        if val_source.divisor > 1:
            val = val // val_source.divisor
        if val_source.multiplier != 1:
            val = val * val_source.multiplier
            
        return val

    def _check_condition(self, player, condition: Condition, source_card) -> bool:
        if not condition: return True
        if condition.type == ConditionType.AND:
            return all(self._check_condition(player, sub, source_card) for sub in condition.args)
        if condition.type == ConditionType.OR:
            return any(self._check_condition(player, sub, source_card) for sub in condition.args)
        
        target_player = player
        if condition.player == Player.OPPONENT:
            target_player = self.game_manager.p2 if player == self.game_manager.p1 else self.game_manager.p1
        
        log_event("DEBUG", "resolver.check_condition", f"Checking {condition.type.name} for {target_player.name}", player=player.name)

        current_val = 0
        target_val = condition.value if isinstance(condition.value, int) else 0
        
        if condition.type == ConditionType.DON_COUNT:
            current_val = len(target_player.don_active) + len(target_player.don_rested) + len(target_player.don_attached_cards)
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
            # 盤面のキャラ枚数条件。target にフィルタ（レスト/特徴/コスト/プレイヤー）が
            # あれば matcher で実体化して数える。無ければ場全体の枚数。
            if condition.target is not None:
                from .matcher import get_target_cards
                current_val = len(get_target_cards(self.game_manager, condition.target, source_card))
            else:
                current_val = len(target_player.field) + (1 if target_player.stage else 0)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.HAS_DON:
            current_val = len(target_player.don_active)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.LEADER_NAME:
            if not target_player.leader: return False
            expected_name = condition.value
            if isinstance(expected_name, str):
                return expected_name in target_player.leader.master.name
            return False

        elif condition.type == ConditionType.LEADER_COLOR:
            if not target_player.leader: return False
            colors = target_player.leader.master.colors or []
            color_vals = [getattr(c, 'value', c) for c in colors]
            if condition.value == "多色":
                return len(colors) >= 2
            return condition.value in color_vals

        elif condition.type == ConditionType.LEADER_TRAIT:
            if not target_player.leader: return False
            expected_trait = condition.value
            if isinstance(expected_trait, str):
                return expected_trait in target_player.leader.master.traits
            return False
            
        elif condition.type in [ConditionType.HAS_TRAIT, ConditionType.HAS_ATTRIBUTE, ConditionType.HAS_UNIT]:
            from .matcher import get_target_cards
            query = condition.target
            if not query:
                query = TargetQuery(zone=Zone.FIELD, player=condition.player)
                if condition.type == ConditionType.HAS_TRAIT and isinstance(condition.value, str):
                    query.traits = [condition.value]
                elif condition.type == ConditionType.HAS_ATTRIBUTE and isinstance(condition.value, str):
                    query.attributes = [condition.value]
            candidates = get_target_cards(self.game_manager, query, source_card)
            count = len(candidates)
            target_count = 1 if target_val == 0 else target_val
            return self._compare(count, condition.operator, target_count)

        elif condition.type == ConditionType.CONTEXT:
            context_val = condition.value
            if context_val == "MY_TURN" or context_val == "SELF_TURN":
                return self.game_manager.turn_player == player
            elif context_val == "OPPONENT_TURN":
                return self.game_manager.turn_player != player
            return True

        elif condition.type == ConditionType.TURN_LIMIT:
            # 使用回数制限は resolve_ability 側で enforce する（ここでは常に通す）。
            return True

        elif condition.type == ConditionType.SOURCE_STATE:
            # このキャラ自身の状態条件（レスト/アクティブ/登場ターン/パワー）
            if source_card is None: return False
            sv = condition.value
            if sv == "IS_RESTED":
                return source_card.is_rest
            if sv == "IS_ACTIVE":
                return not source_card.is_rest
            if sv == "ENTERED_THIS_TURN":
                return getattr(source_card, 'is_newly_played', False)
            if isinstance(sv, tuple) and sv[0] == "POWER":
                is_my_turn = (player == self.game_manager.turn_player)
                power = source_card.get_power(is_my_turn)
                return self._compare(power, condition.operator, sv[1])
            log_event("WARNING", "resolver.source_state_unknown",
                      f"Unknown SOURCE_STATE subtype: {sv}", player=player.name)
            return False

        elif condition.type == ConditionType.FIELD_ALL_TRAIT:
            # 場のキャラ全員が特定の特徴を持つ（「のみ」条件）
            val = condition.value
            if not isinstance(val, tuple): return True
            trait, contains = val
            chars = target_player.field
            if not chars: return False
            if contains:
                return all(any(trait in t for t in c.master.traits) for c in chars)
            return all(any(trait == t for t in c.master.traits) for c in chars)

        elif condition.type == ConditionType.HAS_CHARACTER:
            # 特定名前のキャラが場にいる（GE=いる）/いない（EQ=いない）
            char_name = condition.value
            if not isinstance(char_name, str): return True
            count = sum(1 for c in target_player.field if char_name in c.master.name)
            if target_player.leader and char_name in target_player.leader.master.name:
                count += 1
            if condition.operator == CompareOperator.GE:
                return count >= 1
            return count == 0  # EQ = 「がいない」

        elif condition.type == ConditionType.LEADER_ATTRIBUTE:
            # リーダーの属性条件（斬/打/射/特/知）
            if not target_player.leader: return False
            attr = condition.value
            if not isinstance(attr, str): return True
            return target_player.leader.master.attribute.value == attr

        elif condition.type == ConditionType.RESTED_COUNT:
            # レスト状態のカード総数（フィールド＋リーダー＋ステージ＋ドン!!）
            count = sum(1 for c in target_player.field if c.is_rest)
            if target_player.leader and target_player.leader.is_rest: count += 1
            if target_player.stage and target_player.stage.is_rest: count += 1
            count += len(target_player.don_rested)
            return self._compare(count, condition.operator, target_val)

        # 真に解釈不能な OTHER は fail-safe に倒す（誤発動を防ぐ）。
        if condition.type == ConditionType.OTHER:
            log_event("WARNING", "resolver.condition_unhandled",
                      f"Unparseable condition treated as False (fail-safe): {condition.raw_text[:40]}",
                      player=player.name)
            return False

        # GENERIC は「実在するが未分類の条件」（例: リーダーが多色／レストのキャラが2枚以上）。
        # これらを False に倒すと多数の効果が永久に不発になり誤発動より有害なため、
        # 暫定的に許容(True)しつつ可視化する。分類拡充で個別に評価可能化していく。
        if condition.type == ConditionType.GENERIC:
            log_event("INFO", "resolver.condition_generic",
                      f"GENERIC condition permitted (unclassified): {condition.raw_text[:40]}",
                      player=player.name)
            return True

        return True

    def _compare(self, current: int, operator: CompareOperator, target: int) -> bool:
        if operator == CompareOperator.EQ: return current == target
        if operator == CompareOperator.NEQ: return current != target
        if operator == CompareOperator.GT: return current > target
        if operator == CompareOperator.LT: return current < target
        if operator == CompareOperator.GE: return current >= target
        if operator == CompareOperator.LE: return current <= target
        return False

    def _suspend_for_choice(self, player, node: Choice, source_card):
        base_msg = node.message if node.message else "選択してください"
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "CHOICE",
            "source_card_name": source_card.master.name,
            "message": f"「{source_card.master.name}」の効果: {base_msg}",
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
        
        # 強制/任意の区別
        if is_up_to:
            min_select = 0
        else:
            min_select = required_count
            if min_select > len(candidates):
                min_select = len(candidates)
            if min_select < 1 and len(candidates) > 0:
                min_select = 1
        
        saved_stack = self.execution_stack.copy()
        if action_node:
            saved_stack.append(action_node)

        up_to_str = "まで" if is_up_to else ""
        count_str = f"{required_count}枚{up_to_str}" if required_count != 1 else f"1枚{up_to_str}"
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "SELECT_TARGET",
            "source_card_name": source_card.master.name,
            "message": f"「{source_card.master.name}」の効果: 対象を選択（{count_str}）",
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
