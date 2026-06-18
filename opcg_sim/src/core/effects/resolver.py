from typing import List, Any, Dict, Optional, Union
import json
import os
from dataclasses import asdict, replace
from ...models.effect_types import (
    EffectNode, GameAction, Sequence, Branch, Choice, ValueSource, Condition, TargetQuery, _nfc
)
from ...models.enums import ActionType, Zone, TriggerType, ConditionType, CompareOperator, Player, CardType
import re

# 選択グループ分配（§7-1）で「N枚を選び」の選択集合を保存する save_id。
# atoms._SEL_GROUP_ID と一致させる。
_SEL_GROUP_ID = "_sel_group"

# context にキーが「無い」ことと「値が None/空」であることを区別するための番兵。
_UNSET = object()


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

    def resolve_ability(self, player, ability, source_card, cost_confirmed=False):
        # 1. 条件チェック
        if ability.condition and not self._check_condition(player, ability.condition, source_card):
            self._log_failure_snapshot(player, source_card, ability, "CONDITION_MISMATCH", f"Condition type: {ability.condition.type.name}")
            return

        # 1.5 使用回数制限（【ターン1回】等）の enforce。
        #   ability_used_this_turn は reset_turn_status(clear_usage=True) でクリアされる。
        #   この明示クリアはターン境界（refresh_phase）と、カードが場を離れる領域移動でのみ
        #   行われる。戦闘終了や passive 再計算など「ターン途中」の reset_turn_status では
        #   クリアされないため、ターン単位の使用回数として正しく機能する。
        turn_limit = self._turn_limit_of(ability.condition)
        limit_key = used_count = None
        if turn_limit is not None:
            limit_key = self._ability_key(source_card, ability)
            used_count = source_card.ability_used_this_turn.get(limit_key, 0)
            if used_count >= turn_limit:
                self._log_failure_snapshot(player, source_card, ability, "TURN_LIMIT_REACHED", f"Used {used_count}/{turn_limit} this turn")
                return

        # 2. コストチェック
        #   OPCG では「〜できる：」のコスト句（ability.cost）は常に任意。支払えない場合は
        #   能力が発生しないだけで、例外にはしない（旧実装は raise していたため、任意コストを
        #   払えない ON_PLAY 等を持つカードを出すとゲームが落ちていた）。
        if ability.cost and not self._can_satisfy_node(player, ability.cost, source_card):
            self._log_failure_snapshot(player, source_card, ability, "COST_UNSATISFIED", "Insufficient resources or targets for cost")
            return

        # 2.5 任意コストの使用確認（A-3）。「〜できる：効果」のコスト句は任意で、自動誘発
        #   （相手のアタック時/登場時/アタック時 等）では発動するかをプレイヤーに確認する。
        #   ACTIVATE_MAIN（起動メイン=自発起動）は起動自体が意思表示のため確認しない。
        #   未確認で中断 → resume(accept) が cost_confirmed=True で再入。decline は何もせず
        #   使用回数も消費しない（払わなければ「使った」ことにならない）。
        if (not cost_confirmed and ability.cost is not None
                and getattr(ability, "cost_optional", False)
                and ability.trigger != TriggerType.ACTIVATE_MAIN):
            self._suspend_for_ability_cost_confirm(player, ability, source_card)
            return

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
        """効果処理の結果（何をしてどうなったか）をまとめて出力する。"""
        if os.environ.get("OPCG_LOG_SILENT"):
            return
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
        if os.environ.get("OPCG_LOG_SILENT"):
            return
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
            if node.type == ActionType.REST_DON:
                cost = node.value.base if node.value else 1
                return len(player.don_active) >= cost
            if node.type == ActionType.RETURN_DON:
                # 「ドン!!-N」コスト: 自分の場のドン!!を N 枚ドン!!デッキへ返却する。
                # 場のドン!!（アクティブ＋レスト＋付与中）のいずれからでも選んで戻せるため、
                # 3 つの合計が必要枚数以上あれば支払える。
                cost = node.value.base if node.value else 1
                total = (len(player.don_active) + len(player.don_rested)
                         + len(player.don_attached_cards))
                return total >= cost
            if not node.target: return True
            from .matcher import get_target_cards
            candidates = get_target_cards(self.game_manager, node.target, source_card)
            required = getattr(node.target, 'count', 1)
            if getattr(node.target, 'is_strict_count', False) and len(candidates) < required:
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
                    # target 付き（「トラッシュの…イベントの【メイン】効果を発動する」EB03-031）:
                    # 発生源自身ではなく、選んだカードの【メイン】効果を発動する。対象選択で
                    # 中断したら return（resume 時にこのノードを再処理＝選択済み対象で続行）。
                    if node.target is not None:
                        chosen = self._resolve_targets(player, node.target, source_card, action_node=node)
                        if self.game_manager.active_interaction:
                            return
                        if chosen:
                            self._execute_selected_main(player, chosen, ref_trigger=node.status)
                            if self.game_manager.active_interaction:
                                return
                        continue
                    self._expand_main_effect(source_card, ref_trigger=node.status)
                    continue

                # C8 コスト宣言: 数値入力インタラクションへ中断（resume 時に相手デッキトップを
                # 公開して context に記録し、残りの実行スタックを再開する）。
                if node.type == ActionType.DECLARE_COST:
                    self._suspend_for_cost_declaration(player, source_card)
                    return

                # 遅延実行（「このターン終了時、〜」）: 即時実行せず end_turn フックへ予約する。
                # 既に遅延フラッシュ中（_flushing_delayed）の再実行は通常どおり実行する。
                if getattr(node, "delay", None) == "TURN_END" and not self.context.get("_flushing_delayed"):
                    self.game_manager.pending_end_of_turn.append((player, node, source_card))
                    continue

                # 任意効果（「〜してもよい」）: 発動するかを yes/no で確認する。未確認なら中断し、
                # resume(yes) 時に id(node) を context の確認済み集合へ入れて同じノードを再投入する
                # （no はスキップ）。共有ノード(CardMaster)を汚さないよう確認状態は context で持つ。
                confirmed = self.context.setdefault("_confirmed_optionals", set())
                if getattr(node, "is_optional", False) and id(node) not in confirmed:
                    self._suspend_for_optional_confirmation(player, node, source_card)
                    return

                success = self._execute_game_action(player, node, source_card)

                if self.game_manager.active_interaction:
                    return

                self.context["last_action_success"] = success
                if not success and node.raw_text and ":" in (source_card.master.effect_text or ""):
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
                else:
                    # 条件不成立で何も実行しなかった事実を記録する。
                    # 後続の「（登場）させた場合」（PREV_ACTION=SUCCEEDED）が、
                    # 不発の分岐を「成功」と誤評価しないようにする（ST13-007 等）。
                    self.context["last_action_success"] = False

            elif isinstance(node, Choice):
                self._suspend_for_choice(player, node, source_card)
                return

        # スタックを完走（中断なし）した時点で temp_zone に残ったカードを回収する。
        # 「デッキの上から1枚を公開し、〜の場合」等の REVEAL は公開カードを temp に載せて
        # 条件評価するが、公開は本来カードを動かさない（デッキトップに留まる）。後続で消費
        # されなかった temp カードはデッキトップへ戻す（TEMP リーク＝デッキ消失の防止）。
        if not self.game_manager.active_interaction:
            self._reclaim_temp_to_deck_top()

    def _reclaim_temp_to_deck_top(self):
        """解決完了時に temp_zone に取り残されたカードを元のゾーンの先頭へ戻す。

        既定はデッキトップ。LOOK_LIFE 由来（_temp_origin == "LIFE"）はライフ上へ戻す。"""
        for p in (self.game_manager.p1, self.game_manager.p2):
            if not getattr(p, "temp_zone", None):
                continue
            leftover = list(p.temp_zone)
            p.temp_zone.clear()
            # 公開順を保って上から戻す（reversed で先頭が最上段になるよう挿入）
            for card in reversed(leftover):
                if getattr(card, "_temp_origin", None) == "LIFE":
                    card._temp_origin = None
                    p.life.insert(0, card)
                else:
                    p.deck.insert(0, card)
            if leftover:
                pass

    def _expand_main_effect(self, source_card, ref_trigger=None):
        """source_card 自身の参照先トリガー能力の効果を実行スタックへ展開する。

        トリガー「このカードの【メイン】/【登場時】/【KO時】効果を発動する」用。
        ref_trigger（"ON_PLAY"/"ON_KO"/"ON_ATTACK"/"ACTIVATE_MAIN"）が指定されれば
        そのトリガーの能力を展開する（従来は常に ACTIVATE_MAIN で、【登場時】/【KO時】
        参照のトリガーが no-op だった）。コストは支払わず効果のみ。
        多段展開（自己参照）による無限ループを防ぐためフラグで1回に限定する。
        """
        if self.context.get("_main_expanded"):
            return
        self.context["_main_expanded"] = True

        ref_map = {
            "ON_PLAY": TriggerType.ON_PLAY,
            "ON_KO": TriggerType.ON_KO,
            "ON_ATTACK": TriggerType.ON_ATTACK,
            "ACTIVATE_MAIN": TriggerType.ACTIVATE_MAIN,
        }
        primary = ref_map.get(ref_trigger, TriggerType.ACTIVATE_MAIN)
        main_abilities = [
            ab for ab in source_card.master.abilities
            if ab.trigger == primary and ab.effect is not None
        ]
        # 【トリガー】(ライフ公開時に発動)は ACTIVATE_MAIN だけでなく、効果が【カウンター】に
        # 書かれたイベント(例: OP01-028/OP13-039)も発動対象。参照先能力が無ければ
        # COUNTER 能力にフォールバックする（従来は ACTIVATE_MAIN 限定で何も発動しなかった）。
        if not main_abilities:
            main_abilities = [
                ab for ab in source_card.master.abilities
                if ab.trigger == TriggerType.COUNTER and ab.effect is not None
            ]
        if not main_abilities:
            return

        # 既存スタックの「後」に積む = 先に実行されるよう reversed で push
        for ab in reversed(main_abilities):
            self.execution_stack.append(ab.effect)

    def _execute_selected_main(self, player, cards, ref_trigger=None):
        """選んだカード（トラッシュのイベント等）の【メイン】効果を、そのカードを source として発動する。

        EB03-031「トラッシュのコスト7以下のイベント1枚までの【メイン】効果を発動する」用。
        発生源（EB03-031）の効果ではなく、選んだイベント自身の効果として解決するため、
        現スタックへ展開せず resolve_ability で別コンテキストを起こす（「このカード」参照が
        正しくイベントを指す）。イベントの【メイン】は ACTIVATE_MAIN（無ければ COUNTER）に
        格納される。コスト節は持たない（再コストは発生しない）。
        """
        ref_map = {
            "ON_PLAY": TriggerType.ON_PLAY,
            "ON_KO": TriggerType.ON_KO,
            "ON_ATTACK": TriggerType.ON_ATTACK,
            "ACTIVATE_MAIN": TriggerType.ACTIVATE_MAIN,
        }
        primary = ref_map.get(ref_trigger, TriggerType.ACTIVATE_MAIN)
        for card in cards:
            mains = [ab for ab in card.master.abilities
                     if ab.trigger == primary and ab.effect is not None]
            if not mains:
                mains = [ab for ab in card.master.abilities
                         if ab.trigger == TriggerType.COUNTER and ab.effect is not None]
            if not mains:
                continue
            for ab in mains:
                self.game_manager.resolve_ability(player, ab, source_card=card)
                if self.game_manager.active_interaction:
                    return

    def _execute_game_action(self, player, action: GameAction, source_card) -> bool:
        targets = self._resolve_targets(player, action.target, source_card, action_node=action)

        if targets is None:
            return False

        # PREV_ACTION 条件評価用: ターゲットの有無を記録
        if action.target is not None:
            self.context["_last_had_targets"] = bool(targets)
        else:
            self.context["_last_had_targets"] = None

        if action.target and not targets and not getattr(action.target, 'is_up_to', False):
            # ▼▼▼ 追加: 失敗履歴 ▼▼▼
            self.action_history.append({
                "action": action.type.name if hasattr(action.type, 'name') else str(action.type),
                "success": False,
                "reason": "No targets found"
            })
            return False

        # (2a)(2b) デッキ配置/ライフ並び替えの対話化。並び替え(status=="ARRANGE")や
        # 上下選択(dest_position=="CHOOSE")を伴う自分のカード配置・ライフ並べ替えは、
        # プレイヤーに順序/位置を選ばせるため中断する。ヘッドレス(drain)は既定（現状順・
        # デッキ下）で解決されるため挙動は不変。
        if self._maybe_suspend_arrange(player, action, targets, source_card):
            return False

        # COUNT_QUERY 等の動的値計算でソースカードの所有者を解決できるようにする
        self.context["_source_card_uuid"] = source_card.uuid if source_card else None
        value = self._calculate_value(player, action.value, targets)

        # RETURN_DON（「ドン!!-N」/「場のドン!!をデッキに戻す」）: 自分の場のドン!!のうち
        # どれを戻すかをプレイヤーに選ばせる。未選択なら SELECT_RESOURCE で中断し、再開時に
        # 選択済みドン!!の uuid（context["_return_don_uuids"]）で実行する。
        if action.type == ActionType.RETURN_DON:
            pending = self.context.pop("_return_don_uuids", _UNSET)
            if pending is _UNSET:
                if self._suspend_for_don_selection(player, action, source_card, value):
                    return None  # 中断: resume 時に再実行される
                self.game_manager._return_don_selection = None  # 戻せるドンが無い等→通常実行
            else:
                self.game_manager._return_don_selection = pending

        # 除去（KO/バウンス等）に対する置換 sub_effect の内側中断は常に UI へ提示してよい。
        # 後続があっても、置換が中断したら下で外側継続を deferred へ退避して再開するため（B）。
        self.game_manager._replacement_suspended = False
        success = self.game_manager.apply_action_to_engine(player, action, targets, value, source_card=source_card)
        # 除去置換が内側中断を提示した（_replacement_suspended）かつ、このシーケンスに後続が
        # 残る場合、後続を deferred フレームへ退避して execution_stack を空にする。内側中断が
        # 解決された後に _resume_deferred_continuations が後続を再開する（B = 多段継続の対話化）。
        if self.game_manager._replacement_suspended and self.execution_stack:
            self.game_manager._defer_resolver_stack(player, source_card, self.execution_stack, self.context)
            self.execution_stack = []

        # 「このターン中、このリーダーの効果で引いていない」(OP01-062) 用: リーダー能力由来の
        # ドローをターン内イベントに記録する（次回の同条件が false になり 1ターン複数ドローを防ぐ）。
        if (success and action.type == ActionType.DRAW and source_card is not None
                and getattr(source_card.master, "type", None) == CardType.LEADER):
            self.game_manager.record_turn_event("LEADER_DREW_BY_EFFECT", value or 1)

        # REVEALED_CARD_TRAIT 条件評価用: REVEAL/LOOK 実行後に公開カードを記録。
        # LOOK はターゲット無し（デッキ上から枚数ベースで TEMP へ移動）なので、
        # 移動先 TEMP の先頭（=公開したデッキトップ）を公開カードとして記録する。
        if action.type in (ActionType.REVEAL, ActionType.LOOK, ActionType.FACE_UP_LIFE,
                           ActionType.LOOK_LIFE):
            if targets:
                self.context["last_revealed_card"] = targets[0]
            elif action.type == ActionType.LOOK and getattr(player, "temp_zone", None):
                self.context["last_revealed_card"] = player.temp_zone[0]
            elif action.type == ActionType.LOOK_LIFE and getattr(player, "temp_zone", None):
                # LOOK_LIFE は temp 末尾に append するため、公開カードは末尾
                self.context["last_revealed_card"] = player.temp_zone[-1]
        # 「デッキの上からN枚をトラッシュに置く。置いたカードが〈条件〉の場合」(OP08-096)用:
        # ミルした最後のカード（=トラッシュ末尾）を公開カードとして記録する。
        elif success and action.type == ActionType.TRASH_FROM_DECK:
            tp = player
            if getattr(action, "status", None) == "OPPONENT":
                tp = self.game_manager.p2 if player is self.game_manager.p1 else self.game_manager.p1
            if getattr(tp, "trash", None):
                self.context["last_revealed_card"] = tp.trash[-1]

        # 文脈依存スケーリング（§7-5「捨てたカード1枚につき」等）用に、直前アクションが
        # 対象にした枚数を記録する。SELECT 等のメタアクションは数えない。
        if success and action.type not in (ActionType.SELECT,):
            cnt = len(targets)
            # ドン!!の増減（REST/ACTIVE/RETURN）は targets を介さず枚数処理するため、エンジンが
            # 記録した実処理枚数を使う（「レストにしたドン!!1枚につき」OP13-001）。
            if action.type in (ActionType.REST_DON, ActionType.ACTIVE_DON, ActionType.RETURN_DON):
                cnt = getattr(self.game_manager, "_last_resource_count", cnt)
            self.context["_last_action_count"] = cnt

        # ▼▼▼ 追加: 実行履歴を記録 ▼▼▼
        # ドン!!（master 無し。CHAR_OR_DON 混在選択 等）は "DON!!" 表記にする。
        target_names = [
            f"{t.master.name}({t.uuid[:4]})" if hasattr(t, "master") else f"DON!!({t.uuid[:4]})"
            for t in targets
        ]
        self.action_history.append({
            "action": action.type.name if hasattr(action.type, 'name') else str(action.type),
            "success": success,
            "targets": target_names,
            "value": value
        })

        return success

    def _with_leader(self, query, player, selected):
        """INCLUDE_LEADER フラグ付き選択（「リーダーとキャラN枚を選ぶ」）で、対象側のリーダーを
        選択群へ常に含める。OP07-059（リーダー＋キャラを凍結）/ OP14-009（リーダー↔キャラの
        パワー入替）等で、リーダーが選択に含まれず効果が片側/不発になっていたのを是正。"""
        if "INCLUDE_LEADER" not in getattr(query, "flags", set()):
            return selected
        tp = player
        qp = getattr(query, "player", None)
        if qp is not None and getattr(qp, "name", None) == "OPPONENT":
            tp = self.game_manager.p2 if player is self.game_manager.p1 else self.game_manager.p1
        ldr = getattr(tp, "leader", None)
        out = list(selected or [])
        if ldr is not None and ldr not in out:
            out = [ldr] + out
        return out

    def _resolve_targets(self, player, query, source_card, action_node=None):
        if not query: return []

        if "temp_resolved_targets" in self.context and "_both_sides_pending" not in self.context:
            resumed = self.context.pop("temp_resolved_targets")
            resumed = self._with_leader(query, player, resumed)
            # 中断→再開で解決した対象も save_id 保存を行う（「公開したカードを…」等の
            # 後続参照が、再開経路だけ保存されず空振りしていた）。
            if query.save_id:
                self.context["saved_targets"][query.save_id] = resumed
            # 「そのキャラ/そのカード」の coreference 用に、プレイヤー選択(CHOOSE)の結果を
            # 既定キー selected_card にも保存する。明示 save_id（例:「選び」）が無い
            # ACTIVE/BUFF 等の先行選択でも、後続 ref_id=selected_card が拾えるようにする。
            if (getattr(query, "select_mode", None) == "CHOOSE" and not query.ref_id
                    and query.zone == Zone.FIELD):
                self.context["saved_targets"]["selected_card"] = resumed
            return resumed

        if query.save_id and query.save_id in self.context["saved_targets"]:
            return self.context["saved_targets"][query.save_id]
        
        # 選択グループ分配（§7-1）: 先頭 M 枚を取り、消費済みとして記録する
        # （後続の「残り」が消費分を除いて参照する）。
        if getattr(query, "select_mode", None) == "GROUP_FIRST" and query.ref_id:
            group = self.context["saved_targets"].get(query.ref_id, [])
            consumed = self.context.setdefault("_grp_consumed", {}).setdefault(query.ref_id, [])
            avail = [c for c in group if c.uuid not in consumed]
            n = query.count if query.count and query.count > 0 else 1
            picked = avail[:n]
            consumed.extend(c.uuid for c in picked)
            return picked

        if query.ref_id:
             if query.ref_id == "self":
                 return [source_card]
             if query.ref_id in self.context["saved_targets"]:
                 return self.context["saved_targets"][query.ref_id]
             # ref_id が指定されているのに保存対象が無い（先行選択が条件未達でスキップ
             # された／そもそも選択されなかった）。場全体クエリへフォールスルーすると
             # 全カードへ誤適用する（OP10-099 範囲外コスト・OP07-059 条件未達）ため対象なし。
             return []

        # 「残り」: 直前の選択グループが存在すれば、その消費済みを除いた残余を対象にする
        # （field 分配 OP08-118 等。グループが無ければ従来どおり TEMP=公開残りを参照）。
        if getattr(query, "select_mode", None) == "REMAINING":
            group = self.context["saved_targets"].get(_SEL_GROUP_ID)
            if group:
                consumed = self.context.setdefault("_grp_consumed", {}).setdefault(_SEL_GROUP_ID, [])
                return [c for c in group if c.uuid not in consumed]

        from .matcher import get_target_cards

        # 「お互いの〜」(BOTH_SIDES): 両プレイヤーへ独立・同時に適用する。各サイドで候補・枚数を
        # 個別に解決し結合する。選択を伴うサイド（候補>必要枚数の手札捨て等。OP05-058）は、その
        # サイドのプレイヤーに**順に選ばせる**（相手→自分の順で SELECT_TARGET 中断）。選択の余地が
        # 無いサイド（ALL/REMAINING・隠しゾーン=ライフ/デッキの位置確定・候補≤必要数）は非中断で
        # 確定する（OP11-102 のライフ上トラッシュ等）。中断はスタック土台＋逐次再入で実現する。
        if "BOTH_SIDES" in getattr(query, "flags", set()) and query.player == Player.ALL:
            owner_p = self.game_manager.p1 if self.game_manager.p1.name == source_card.owner_id else self.game_manager.p2
            opp_p = self.game_manager.p2 if owner_p is self.game_manager.p1 else self.game_manager.p1
            bs = self.context.setdefault("_both_sides", {})
            # 再開で来た選択を、保留していたサイドへ割り当てる。
            if "_both_sides_pending" in self.context and "temp_resolved_targets" in self.context:
                bs[self.context.pop("_both_sides_pending")] = self.context.pop("temp_resolved_targets")
            for side, side_name, side_player in ((Player.OPPONENT, "OPPONENT", opp_p),
                                                 (Player.SELF, "SELF", owner_p)):
                if side_name in bs:
                    continue  # 解決済み
                side_q = replace(query, player=side,
                                 flags=(set(getattr(query, "flags", set())) - {"BOTH_SIDES"}))
                cand = get_target_cards(self.game_manager, side_q, source_card)
                if not cand:
                    bs[side_name] = []
                    continue
                if side_q.select_mode in ("ALL", "REMAINING"):
                    bs[side_name] = list(cand)
                    continue
                req = getattr(side_q, "count", 1) or 0
                if getattr(side_q, "count_dynamic", None) == "DOWN_TO_N":
                    req = max(0, len(cand) - max(req, 0))
                if req <= 0:
                    bs[side_name] = []
                    continue
                # 隠しゾーン（ライフ/デッキ）は位置確定で自動取得（情報リーク回避・選択不可）。
                hidden = side_q.zone in (Zone.DECK, Zone.LIFE) and "REVEAL_SELECT" not in getattr(side_q, "flags", set())
                if hidden or len(cand) <= req:
                    bs[side_name] = cand[:req]
                    continue
                # 選択の余地あり → このサイドのプレイヤーに選ばせる（逐次中断）。必要枚数 req は
                # 確定済みのため、DOWN_TO_N 等を畳んだ固定 count のクエリで提示する（「N枚になる
                # ように捨てる」=ちょうど req 枚＝min=max=req の選択）。
                self.context["_both_sides_pending"] = side_name
                suspend_q = replace(side_q, count=req, count_dynamic=None, is_up_to=False)
                self._suspend_for_target_selection(side_player, cand, suspend_q, source_card, action_node)
                return None
            result = list(bs.get("OPPONENT", [])) + list(bs.get("SELF", []))
            self.context.pop("_both_sides", None)
            if query.save_id:
                self.context["saved_targets"][query.save_id] = result
            return result

        candidates = get_target_cards(self.game_manager, query, source_card)

        # 「（戻した／選んだ）キャラと異なる色の…」: selected_card と色が重なる候補を除外する
        # （OP01-002）。selected_card は直前の FIELD 選択（BOUNCE 等）で保存済み。
        if "EXCLUDE_SELECTED_COLOR" in getattr(query, "flags", set()):
            ref = self.context["saved_targets"].get("selected_card") or []
            ref_colors = set()
            for rc in ref:
                ref_colors.update(c.value for c in (rc.master.colors or []))
            if ref_colors:
                candidates = [c for c in candidates
                              if not (ref_colors & {col.value for col in (c.master.colors or [])})]

        # 「（複数枚を）パワーの合計がN以下になるように〈KO等〉」: 選択集合の合計パワー上限を尊重する
        # （OP05-007/OP09-018）。組み合わせ制約のためフロント対話に丸投げせず、ルール違反（合計超過）
        # を起こさない有効な選択を確定する＝低パワー順に上限まで貪欲に取る（最大枚数を確保）。
        psum_max = getattr(query, "power_sum_max", None)
        if psum_max is not None and candidates:
            cap_n = getattr(query, "count", 1) or len(candidates)
            ordered = sorted(candidates, key=lambda c: c.master.power or 0)
            chosen, total = [], 0
            for c in ordered:
                p = c.master.power or 0
                if len(chosen) >= cap_n:
                    break
                if total + p <= psum_max:
                    chosen.append(c)
                    total += p
            if query.save_id:
                self.context["saved_targets"][query.save_id] = chosen
            return chosen

        required_count = getattr(query, 'count', 1)
        is_up_to = getattr(query, 'is_up_to', False)
        is_strict = getattr(query, 'is_strict_count', False)
        is_resource = (query.zone == Zone.COST_AREA)

        # 「<ゾーン>がN枚になるように」: N 枚を残して残り全てを対象にする（雷迎 等）。
        if getattr(query, "count_dynamic", None) == "DOWN_TO_N":
            required_count = max(0, len(candidates) - max(required_count, 0))
            if required_count == 0:
                return []
            is_up_to = False

        if len(candidates) == 0:
            return []

        if is_strict and len(candidates) < required_count:
            return []

        # 隠しゾーン（デッキ/ライフ）の直接ターゲットは「上から」位置指定で自動取得する。
        #   デッキ/ライフは非公開かつ順序のあるゾーンで、中身を見て個別に選ぶことはできない
        #   （中身を見て選ぶ対話は LOOK→TEMP=zone TEMP 経由のみ）。直接 DECK/LIFE を選択中断
        #   させると相手/自分の隠し情報が見えて任意に選べてしまう（情報リーク）。候補は
        #   get_target_cards が上から順で返すため、上から count 枚（"まで"は available 上限）を取る。
        #   例外: 「（自分の）ライフすべてを見て、1枚を選ぶ」等は flag="REVEAL_SELECT" を持ち、
        #   自分のライフを明示的に公開して選ぶため対話選択を許可する（情報リークにならない）。
        if query.zone in (Zone.DECK, Zone.LIFE) and "REVEAL_SELECT" not in getattr(query, "flags", set()):
            n = required_count if required_count and required_count > 0 else len(candidates)
            selected = candidates[:n]
            if query.save_id:
                self.context["saved_targets"][query.save_id] = selected
            return selected

        if (query.select_mode == "ALL") or \
           (query.select_mode == "REMAINING") or \
           (len(candidates) <= required_count and not is_up_to) or \
           (is_resource and not is_up_to):
            # REMAINING（「残り」）は意味的に対象=残り全部のため選択中断しない
            # （並び替え/上下は後段の ARRANGE_DECK 対話で扱う）。
            selected = candidates[:required_count] if required_count > 0 else candidates
            selected = self._with_leader(query, player, selected)
            if query.save_id:
                self.context["saved_targets"][query.save_id] = selected
            if (query.select_mode == "CHOOSE" and not query.ref_id
                    and query.zone == Zone.FIELD):
                self.context["saved_targets"]["selected_card"] = selected
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

    def _check_condition(self, player, condition: Condition, source_card, host_card=None) -> bool:
        # host_card: 能力の保持カード（置換/除去保護では保護者=リーダー等）。HAS_DON 等の
        # 「能力保持カードの状態」条件はこちらを見る。source_card は被保護/離脱カード。
        # 通常の解決では host_card 未指定＝source_card と同一（自能力）。
        if not condition: return True
        if condition.type == ConditionType.AND:
            return all(self._check_condition(player, sub, source_card, host_card) for sub in condition.args)
        if condition.type == ConditionType.OR:
            return any(self._check_condition(player, sub, source_card, host_card) for sub in condition.args)
        if condition.type == ConditionType.NOT:
            return not self._check_condition(player, condition.args[0], source_card, host_card)

        target_player = player
        if condition.player == Player.OPPONENT:
            target_player = self.game_manager.p2 if player == self.game_manager.p1 else self.game_manager.p1

        current_val = 0
        target_val = condition.value if isinstance(condition.value, int) else 0
        
        if condition.type == ConditionType.DON_COUNT:
            # 「付与されているドン!!がある/ない/合計N枚以上」はプレイヤーの付与中ドン(attached)
            # のみを数える（OP13-076 神避 / OP12-015 / OP12-024 / OP13-112）。場のドン総数ではない。
            # ただし「選んだキャラのコストが…付与ドンの枚数と同じ」のような対象固有の動的比較
            # （パーサ未対応で DON_COUNT に退化）は別物なので除外する（OP15-031）。
            _raw = condition.raw_text or ""
            if _nfc("付与") in _raw and _nfc("同じ") not in _raw:
                current_val = len(target_player.don_attached_cards)
            else:
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

        elif condition.type == ConditionType.FIELD_COST_SUM:
            # 「（自分の）キャラのコストの合計が N 以上/以下」。場のキャラの現在コスト総和を比較する。
            current_val = sum(c.current_cost for c in target_player.field)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.LIFE_COUNT_BOTH:
            # 「お互いのライフの合計枚数が N 以上/以下」（P-088 等）。両プレイヤーのライフ合計。
            current_val = len(self.game_manager.p1.life) + len(self.game_manager.p2.life)
            if target_val == 0 and isinstance(condition.value, str):
                nums = re.findall(r'\d+', condition.raw_text)
                target_val = int(nums[0]) if nums else 0
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.LIFE_HAND_SUM:
            # 「（自分の）ライフと手札の合計枚数が N 以上/以下」（OP04-040）。
            current_val = len(target_player.life) + len(target_player.hand)
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.TURN_COUNT:
            # 「自分の第Nターン以降の場合」（OP15-058）。turn_count はゲーム全体の通し番号で
            # 先攻=1,3,5… / 後攻=2,4,6… と交互に進む（gamestate の turn 進行コメント参照）。
            # そのため turn_count を直接 N と比較すると、後攻は自分の第1ターン(turn_count=2)で
            # 「第2ターン以降」を満たし、1ターン早く撃ててしまう。手番プレイヤー自身の
            # 「第何ターンか」へ変換してから比較する。own_turn = ceil(turn_count/2) は
            # 先攻/後攻に依らず正しい（先攻 1,3,5→1,2,3 ／ 後攻 2,4,6→1,2,3）。
            own_turn = (getattr(self.game_manager, "turn_count", 0) + 1) // 2
            return self._compare(own_turn, condition.operator, target_val)

        elif condition.type == ConditionType.EVENT_THIS_TURN:
            # 「〈イベント〉した時」: このターン中に当該イベントが発生したか（value=(名前, しきい値)）。
            # 既定は「発生していなければ不発」（occurred>=しきい値）。operator が明示されれば従う
            # （「引いていない」= occurred < 1 等の否定）。OP06-042/OP07-038/OP01-062 等。
            ev_name, ev_min = (condition.value if isinstance(condition.value, tuple)
                               else (condition.value, 1))
            occurred = getattr(self.game_manager, "_turn_events", {}).get(ev_name, 0)
            return self._compare(occurred, condition.operator, ev_min)

        elif condition.type == ConditionType.LIFE_COUNT_COMPARE:
            # 「自分のライフが相手(のライフ)より(N枚以上)少ない/以下/より多い/以上」: 自分 (op) 相手。
            me = self.game_manager.p1 if player is self.game_manager.p1 else self.game_manager.p2
            opp = self.game_manager.p2 if me is self.game_manager.p1 else self.game_manager.p1
            return self._compare(len(me.life), condition.operator,
                                 self._offset_threshold(len(opp.life), condition))

        elif condition.type == ConditionType.HAND_COUNT_COMPARE:
            # 「自分の手札が相手(の手札)より(N枚以上)少ない/多い」: 自分手札 (op) 相手手札±N（OP09-092）。
            opp = self.game_manager.p2 if player is self.game_manager.p1 else self.game_manager.p1
            return self._compare(len(player.hand), condition.operator,
                                 self._offset_threshold(len(opp.hand), condition))

        elif condition.type == ConditionType.CHAR_KOED_THIS_TURN:
            # 「このターン中、（相手/自分の）キャラがKOされている場合」: 当該プレイヤーの
            # キャラがこのターンに KO された回数（gamestate が CHAR_KOED_<name> で記録）。
            occurred = getattr(self.game_manager, "_turn_events", {}).get(
                f"CHAR_KOED_{target_player.name}", 0)
            return self._compare(occurred, condition.operator, target_val or 1)

        elif condition.type == ConditionType.HAS_DON:
            # 【ドン!!×N】: 能力保持カードに付与されたドン!!が N 枚以上か。コストエリアの active ドン
            # ではなく attached_don を見る。置換/除去保護では保持カード(host=protector)を見る
            # （被保護カード source_card ではない。OP05-001: リーダーの付与ドンで判定）。
            host = host_card if host_card is not None else source_card
            current_val = getattr(host, "attached_don", 0) if host is not None else 0
            return self._compare(current_val, condition.operator, target_val)

        elif condition.type == ConditionType.LEADER_NAME:
            if not target_player.leader: return False
            expected_name = condition.value
            leader_master = target_player.leader.master
            if isinstance(expected_name, str):
                return leader_master.matches_name(expected_name, partial=True)
            if isinstance(expected_name, (list, tuple)):
                # 複数リーダー名の OR（「サボ」か「エース」か「ルフィ」: OP13-016）。
                return any(leader_master.matches_name(n, partial=True) for n in expected_name)
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
            traits = target_player.leader.master.traits
            if isinstance(expected_trait, str):
                return expected_trait in traits
            if isinstance(expected_trait, (list, tuple)):
                # 複数特徴の OR（「リーダーが特徴《X》か《Y》を持つ場合」）。
                return any(t in traits for t in expected_trait)
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
            if sv == "IN_BATTLE":
                # 「このリーダー/キャラが（相手のキャラと）バトルしている場合」(OP12-020): 進行中の
                # バトル(active_battle)に source_card が攻撃側/防御側として関与しているか。
                ab = self.game_manager.active_battle
                if not ab:
                    return False
                return source_card in (ab.get("attacker"), ab.get("target"))
            if isinstance(sv, tuple) and sv[0] == "POWER":
                is_my_turn = (player == self.game_manager.turn_player)
                power = source_card.get_power(is_my_turn)
                return self._compare(power, condition.operator, sv[1])
            if isinstance(sv, tuple) and sv[0] == "NAME":
                # 置換の対象指定（「自分の「X」がKOされる場合」OP12-061）: 離れるカードが名前 X か。
                # 本来名だけでなくルール上の別名も照合（EB04-038 ロシナンテ&ロー=トラファルガー・ロー）。
                return source_card.master.matches_name(sv[1], partial=True)
            if isinstance(sv, tuple) and sv[0] == "COST":
                # 置換の対象指定（「元々のコストN以上のキャラがKOされる」EB03-001）: 離脱カードの
                # 元々コスト（master.cost）を比較する。
                return self._compare(source_card.master.cost or 0, condition.operator, sv[1])
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
            # 特定名前のキャラが場にいる/いない（枚数指定・状態指定あり/なし）。
            # condition.target.zone が TRASH のときはトラッシュ内の存在を見る（OP08-006）。
            char_val = condition.value
            in_trash = (condition.target is not None
                        and getattr(condition.target, "zone", None) == Zone.TRASH)
            pool = list(target_player.trash) if in_trash else list(target_player.field)
            include_leader = (not in_trash) and target_player.leader is not None
            if isinstance(char_val, tuple):
                char_name, sub = char_val
                if isinstance(sub, str) and sub in ("IS_RESTED", "IS_ACTIVE"):
                    # 状態付き: 「X」がレスト/アクティブ（トラッシュには状態が無いので場のみ）
                    candidates = [c for c in pool if c.master.matches_name(char_name, partial=True)]
                    if include_leader and target_player.leader.master.matches_name(char_name, partial=True):
                        candidates.append(target_player.leader)
                    if not candidates:
                        return False
                    if sub == "IS_RESTED":
                        return any(c.is_rest for c in candidates)
                    return any(not c.is_rest for c in candidates)
                else:
                    # 枚数指定: (char_name, count_thr)
                    count_thr = sub
                    count = sum(1 for c in pool if c.master.matches_name(char_name, partial=True))
                    if include_leader and target_player.leader.master.matches_name(char_name, partial=True):
                        count += 1
                    return self._compare(count, condition.operator, count_thr)
            elif isinstance(char_val, str):
                char_name = char_val
                count = sum(1 for c in pool if c.master.matches_name(char_name, partial=True))
                if include_leader and target_player.leader.master.matches_name(char_name, partial=True):
                    count += 1
                if condition.operator == CompareOperator.GE:
                    return count >= 1
                return count == 0  # EQ = 「がいない」
            return True

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

        elif condition.type == ConditionType.PREV_ACTION:
            sv = condition.value
            success = self.context.get("last_action_success", True)
            had_targets = self.context.get("_last_had_targets")
            if sv == "SKIPPED":
                return (not success) or (had_targets is False)
            # SUCCEEDED / PLAYED_CARD どちらも「直前アクションが成立した」
            return success and had_targets is not False

        elif condition.type == ConditionType.DON_COUNT_COMPARE:
            opp = self.game_manager.p2 if player == self.game_manager.p1 else self.game_manager.p1
            my_don = len(player.don_active) + len(player.don_rested) + len(player.don_attached_cards)
            opp_don = len(opp.don_active) + len(opp.don_rested) + len(opp.don_attached_cards)
            return self._compare(my_don, condition.operator,
                                 self._offset_threshold(opp_don, condition))

        elif condition.type == ConditionType.LEADER_STATE:
            leader = target_player.leader
            if not leader: return False
            sv = condition.value
            if sv == "IS_ACTIVE": return not leader.is_rest
            if sv == "IS_RESTED": return leader.is_rest
            if isinstance(sv, tuple) and sv[0] == "POWER":
                is_my_turn = (player == self.game_manager.turn_player)
                power = leader.get_power(is_my_turn)
                return self._compare(power, condition.operator, sv[1])
            return False

        elif condition.type == ConditionType.OPPONENT_REMOVAL:
            # source_card = 除去されようとしているカード（_active_replacement から渡される）
            if source_card is None: return False
            val = condition.value
            if not isinstance(val, dict): return True
            # 元々のパワー（master.power）でフィルタ
            if "power_max" in val and source_card.master.power > val["power_max"]:
                return False
            if "power_min" in val and source_card.master.power < val["power_min"]:
                return False
            # 元々のコスト
            if "cost_max" in val and source_card.master.cost > val["cost_max"]:
                return False
            # 特徴
            if "trait" in val:
                traits = getattr(source_card.master, 'traits', []) or []
                if val["trait"] not in traits:
                    return False
            return True

        elif condition.type == ConditionType.FIELD_COUNT_COMPARE:
            opp = self.game_manager.p2 if player == self.game_manager.p1 else self.game_manager.p1
            my_count = len(player.field)
            opp_count = len(opp.field)
            return self._compare(my_count, condition.operator,
                                 self._offset_threshold(opp_count, condition))

        elif condition.type == ConditionType.DECLARED_COST_MATCH:
            # C8: 公開カードのコストが宣言コストと一致するか。
            card = self.context.get("last_revealed_card")
            declared = self.context.get("declared_cost")
            if card is None or declared is None:
                return False  # 情報が無ければ不成立（誤発動防止）
            return card.master.cost == declared

        elif condition.type == ConditionType.REVEALED_CARD_TRAIT:
            card = self.context.get("last_revealed_card")
            if card is None:
                return True  # コンテキスト未設定は permissive fallback
            val = condition.value
            if not isinstance(val, dict):
                return True
            # 特徴チェック
            if "trait" in val:
                trait = val["trait"]
                contains = val.get("trait_contains", False)
                traits = getattr(card.master, 'traits', []) or []
                if contains:
                    if not any(trait in t for t in traits):
                        return False
                else:
                    if not any(trait == t for t in traits):
                        return False
            # コストチェック
            if "cost" in val:
                cost_op = val.get("cost_op", CompareOperator.LE)
                if not self._compare(card.master.cost, cost_op, val["cost"]):
                    return False
            # パワーチェック（公開カードのパワー条件。OP04-011「パワー6000以上のキャラ」等）
            if "power" in val:
                power_op = val.get("power_op", CompareOperator.GE)
                if not self._compare(getattr(card.master, "power", 0) or 0, power_op, val["power"]):
                    return False
            # カード名チェック（本来名＋ルール上の別名）
            if "name" in val and not card.master.matches_name(val["name"]):
                return False
            # カードタイプチェック
            if "card_type" in val:
                from ...models.enums import CardType
                type_map = {
                    "キャラ": CardType.CHARACTER,
                    "イベント": CardType.EVENT,
                    "ステージ": CardType.STAGE,
                }
                expected = type_map.get(val["card_type"])
                if expected and card.master.type != expected:
                    return False
            return True

        # 真に解釈不能な OTHER は fail-safe に倒す（誤発動を防ぐ）。
        if condition.type == ConditionType.OTHER:
            return False

        # GENERIC は「実在するが未分類の条件」（例: リーダーが多色／レストのキャラが2枚以上）。
        # これらを False に倒すと多数の効果が永久に不発になり誤発動より有害なため、
        # 暫定的に許容(True)しつつ可視化する。分類拡充で個別に評価可能化していく。
        if condition.type == ConditionType.GENERIC:
            return True

        return True

    def _offset_threshold(self, opp_count: int, condition) -> int:
        """相対比較「相手より N枚以上 少ない/多い」のしきい値を相手枚数±N で返す。
        condition.value=N（オフセット、既定0）。少ない方向(LE/LT)は相手-N、多い方向(GE/GT)は相手+N。
        N=0 のとき従来の単純比較（相手枚数そのもの）に一致する。"""
        offset = condition.value if isinstance(condition.value, int) else 0
        if condition.operator in (CompareOperator.LE, CompareOperator.LT):
            return opp_count - offset
        return opp_count + offset

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

    def _suspend_for_ability_cost_confirm(self, player, ability, source_card):
        """任意コスト能力（「〜できる：効果」）の使用可否を確認するため中断する。
        accept 時は resolve_interaction が resolve_ability を cost_confirmed=True で再入する。
        FE は CONFIRM_OPTIONAL オーバーレイで描画する（専用UIを増やさない）。"""
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "CONFIRM_OPTIONAL",
            "source_card_name": source_card.master.name,
            "message": f"「{source_card.master.name}」の効果を使用しますか？（コストを払う）",
            "can_skip": True,
            "continuation": {
                "source_card_uuid": source_card.uuid,
                "confirm_ability": ability,
            },
        }

    def _suspend_for_optional_confirmation(self, player, node, source_card):
        """任意効果（「〜してもよい」）の発動可否を yes/no で確認するため中断する。
        node は execution_stack から既に pop 済みなので、continuation に退避する。"""
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "CONFIRM_OPTIONAL",
            "source_card_name": source_card.master.name,
            "source_card_uuid": source_card.uuid,
            "message": f"「{source_card.master.name}」の効果を発動しますか？",
            "can_skip": True,
            "continuation": {
                "execution_stack": self.execution_stack,
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
                "optional_node": node,
            },
        }

    def resume_optional(self, player, source_card, accepted, optional_node, execution_stack, effect_context):
        """任意効果確認からの再開。accepted=True なら確認済みにして再投入、False ならスキップ。"""
        self.execution_stack = execution_stack
        self.context = effect_context
        if accepted and optional_node is not None:
            self.context.setdefault("_confirmed_optionals", set()).add(id(optional_node))
            self.execution_stack.append(optional_node)
        self._process_stack(player, source_card)

    def _maybe_suspend_arrange(self, player, action, targets, source_card) -> bool:
        """(2a)(2b) デッキ配置/ライフ並び替えが対話を要するなら中断する（要否を bool で返す）。
        - ORDER_LIFE: ライフ全体を任意順に並べ替える（2枚以上のとき）。
        - DECK_BOTTOM + status=="ARRANGE": 配置順をプレイヤーが選ぶ（2枚以上のとき）。
        - DECK_BOTTOM + dest_position=="CHOOSE": デッキの上/下をプレイヤーが選ぶ。
        ヘッドレスでは drain が既定（現状順・デッキ下）で解決し、結果は従来と同一。"""
        gm = self.game_manager
        if action.type == ActionType.ORDER_LIFE:
            tp = player
            if getattr(action, "status", None) == "OPPONENT":
                tp = gm.p2 if player is gm.p1 else gm.p1
            cards = list(tp.life)
            if len(cards) < 2:
                return False
            self._suspend_for_arrange(player, source_card, cards, dest_kind="LIFE",
                                      dest_owner=tp, needs_reorder=True, needs_pos=False,
                                      fixed_position="TOP")
            return True
        if action.type == ActionType.DECK_BOTTOM:
            needs_reorder = (getattr(action, "status", None) == "ARRANGE" and len(targets) >= 2)
            needs_pos = (getattr(action, "dest_position", None) == "CHOOSE")
            if not targets or not (needs_reorder or needs_pos):
                return False
            fixed = "TOP" if getattr(action, "dest_position", None) == "TOP" else "BOTTOM"
            self._suspend_for_arrange(player, source_card, list(targets), dest_kind="DECK",
                                      dest_owner=None, needs_reorder=needs_reorder,
                                      needs_pos=needs_pos, fixed_position=fixed)
            return True
        return False

    def _suspend_for_arrange(self, player, source_card, cards, dest_kind, dest_owner,
                             needs_reorder, needs_pos, fixed_position):
        """並び替え/上下選択のインタラクション(ARRANGE_DECK)へ中断する。
        フロントは candidates を提示し、並び替え(DnD)と上/下トグルを返す。"""
        parts = []
        if needs_reorder:
            parts.append("順番")
        if needs_pos:
            parts.append("置く位置(上/下)")
        what = "／".join(parts) if parts else "配置"
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "ARRANGE_DECK",
            "source_card_name": source_card.master.name,
            "message": f"「{source_card.master.name}」の効果: {what}を決めてください",
            "candidates": list(cards),
            # max=-1 はフロント CardSelectModal の並び替えモード（全カード配置）を意味する。
            "constraints": {"min": 0, "max": -1 if needs_reorder else 0},
            "allow_position": needs_pos,
            "allow_reorder": needs_reorder,
            "continuation": {
                "execution_stack": self.execution_stack,
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
                "arrange_targets": list(cards),
                "dest_kind": dest_kind,
                "dest_owner": dest_owner.name if dest_owner is not None else None,
                "fixed_position": fixed_position,
            },
        }

    def _suspend_for_cost_declaration(self, player, source_card):
        """C8: 数値（コスト）の宣言を待つインタラクションへ中断する。
        resume 時に gamestate.resolve_interaction が宣言値を context に記録し、相手デッキ
        トップを公開してから残りの execution_stack を再開する。"""
        self.game_manager.active_interaction = {
            "player_id": player.name,
            "action_type": "DECLARE_COST",
            "source_card_name": source_card.master.name,
            "message": f"「{source_card.master.name}」の効果: コストを宣言してください",
            "constraints": {"min": 0, "max": 10},
            "continuation": {
                "execution_stack": self.execution_stack,
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid,
            },
        }

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

        # temp_zone からの選択（デッキサーチ）では全公開カードを表示し、
        # 条件を満たすカードだけを selectable_uuids で絞り込む
        view_candidates = candidates
        selectable_uuids = None
        if getattr(query, 'zone', None) == Zone.TEMP:
            owner_player = self.game_manager.p1 if self.game_manager.p1.name == source_card.owner_id else self.game_manager.p2
            all_temp = list(owner_player.temp_zone)
            if len(all_temp) > len(candidates):
                view_candidates = all_temp
                selectable_uuids = [c.uuid for c in candidates]

        up_to_str = "まで" if is_up_to else ""
        count_str = f"{required_count}枚{up_to_str}" if required_count != 1 else f"1枚{up_to_str}"
        # 「相手が選び」: 選択者が効果コントローラーの相手に指定されている場合は
        # 相手プレイヤーに選択させる（RC-3）。
        chooser_player = player
        if getattr(query, "chooser", None) is not None and query.chooser == Player.OPPONENT:
            gm = self.game_manager
            chooser_player = gm.p2 if player is gm.p1 else gm.p1
        interaction = {
            "player_id": chooser_player.name,
            "action_type": "SELECT_TARGET",
            "source_card_name": source_card.master.name,
            "message": f"「{source_card.master.name}」の効果: 対象を選択（{count_str}）",
            "candidates": view_candidates,
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
        if selectable_uuids is not None:
            interaction["selectable_uuids"] = selectable_uuids
        self.game_manager.active_interaction = interaction

    def _suspend_for_don_selection(self, player, action, source_card, value) -> bool:
        """RETURN_DON の対象ドン!!選択で中断する。戻せるドン!!が無ければ False を返し中断しない。

        選択者・対象は _don_pool_player が決めるプール所有者（自分／相手「は…戻す」）。
        候補は当該プレイヤーの場のドン!!全て（アクティブ＋レスト＋付与中）。
        """
        gm = self.game_manager
        tp = gm._don_pool_player(player, action)
        field_don = list(tp.don_active) + list(tp.don_rested) + list(tp.don_attached_cards)
        n = value if value and value > 0 else 1
        to_return = min(n, len(field_don))
        if to_return <= 0:
            return False

        saved_stack = self.execution_stack.copy()
        saved_stack.append(action)  # resume 時に RETURN_DON を再実行する
        gm.active_interaction = {
            "player_id": tp.name,
            "action_type": "SELECT_RESOURCE",
            "source_card_name": source_card.master.name if source_card else "",
            "message": f"ドン!!デッキに戻すドン!!を{to_return}枚選択してください",
            "candidates": field_don,
            "constraints": {"min": to_return, "max": to_return},
            "continuation": {
                "execution_stack": saved_stack,
                "effect_context": self.context,
                "source_card_uuid": source_card.uuid if source_card else None,
            },
        }
        return True

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
