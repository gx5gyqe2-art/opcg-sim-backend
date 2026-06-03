import re
from typing import List, Optional
from ...models.effect_types import (
    Ability, EffectNode, GameAction, Sequence, Branch, Choice, ValueSource, TargetQuery, Condition, _nfc
)
from ...models.enums import ActionType, TriggerType, ConditionType, Zone, CompareOperator, Player
from .matcher import parse_target
from ...utils.logger_config import log_event

class EffectParser:
    def __init__(self):
        pass

    def parse_card_text(self, text: str, as_trigger: bool = False) -> List[Ability]:
        """
        カード1枚分のテキストを受け取り、全 Ability のリストを返す。
        複数の ability は ' / ' または改行で区切られている。
        as_trigger=True の場合、生成した全 Ability の trigger を TRIGGER に上書きする。
        """
        norm = _nfc(text)
        if not norm or norm.strip() in ['なし', 'None', '']:
            return []

        # キーワードのみの能力（【ブロッカー】等）を除去してテキストだけ処理
        segments = re.split(r'\s*/\s*|\n', norm)
        segments = [s.strip() for s in segments if s.strip()]

        abilities = []
        for seg in segments:
            try:
                ab = self.parse_ability(seg)
                if ab.trigger != TriggerType.UNKNOWN or ab.effect is not None:
                    abilities.append(ab)
            except Exception as e:
                log_event("WARNING", "parser.segment_skip", f"Skipped segment: {seg[:30]} | {e}")

        if as_trigger:
            for ab in abilities:
                ab.trigger = TriggerType.TRIGGER

        return abilities

    def parse_ability(self, text: str) -> Ability:
        log_event("DEBUG", "parser.input", f"Input text: {text[:50]}")
        try:
            norm_text = _nfc(text)

            # 【ターン1回】を前処理で検出し条件として記憶
            turn_limit_cond = None
            if _nfc("【ターン1回】") in norm_text or _nfc("(ターンに1回)") in norm_text or _nfc("（ターンに1回）") in norm_text:
                turn_limit_cond = Condition(type=ConditionType.TURN_LIMIT, value=1)
                norm_text = re.sub(_nfc(r'【ターン1回】|\(ターンに1回\)|（ターンに1回）'), '', norm_text).strip()

            # 【ドン!!×N】コストタグを前処理で検出
            don_cost_value = 0
            don_match = re.search(_nfc(r'【ドン[!!‼][×x×](\d+)】'), norm_text)
            if don_match:
                don_cost_value = int(don_match.group(1))
                norm_text = norm_text.replace(don_match.group(0), '').strip()

            # トリガー検出（【 】のみ対象）
            trigger = self._detect_trigger(norm_text)
            log_event("INFO", "parser.trigger", f"Detected trigger: {trigger.name}")

            # トリガータグの除去
            clean_text = re.sub(r'【.*?】', '', norm_text).strip()

            cost_node = None
            effect_text = clean_text

            # コストの分離（全角コロン優先、なければ半角コロン）
            colon = _nfc("：") if _nfc("：") in clean_text else (_nfc(":") if _nfc(":") in clean_text else None)
            if colon and colon in clean_text:
                parts = clean_text.split(colon, 1)
                cost_node = self._parse_cost_node(parts[0])
                effect_text = parts[1]

            # ドン!!コストタグを cost_node に統合
            if don_cost_value > 0:
                don_cost_action = GameAction(
                    type=ActionType.REST_DON,
                    value=ValueSource(base=don_cost_value),
                    target=TargetQuery(zone=Zone.COST_AREA, player=Player.SELF, count=don_cost_value),
                    raw_text=_nfc(f"ドン!!{don_cost_value}枚をレストにする")
                )
                if cost_node is not None:
                    cost_node = Sequence(actions=[don_cost_action, cost_node])
                else:
                    cost_node = don_cost_action

            # 効果本体の解析
            effect_node = self._parse_to_node(effect_text)

            # 先頭のゲート条件（「〜の場合、」）を ability.condition に引き上げる
            final_condition = turn_limit_cond
            if isinstance(effect_node, Branch) and effect_node.if_false is None and effect_node.condition is not None:
                if final_condition is None:
                    final_condition = effect_node.condition
                else:
                    final_condition = Condition(
                        type=ConditionType.AND,
                        args=[final_condition, effect_node.condition]
                    )
                effect_node = effect_node.if_true

            return Ability(
                trigger=trigger,
                condition=final_condition,
                cost=cost_node,
                effect=effect_node,
                raw_text=_nfc(text)
            )
        except Exception as e:
            log_event(level_key="ERROR", action="parser.parse_ability_error", msg=f"Failed to parse: {text[:20]} | Error: {str(e)}")
            return Ability(trigger=TriggerType.UNKNOWN, effect=None, raw_text=_nfc(text))

    def _parse_cost_node(self, cost_text: str) -> Optional[EffectNode]:
        """
        コストテキストを解析する。
        「このキャラをレストにできる」「このリーダーをレストにできる」パターンを
        ref_id="self" の REST アクションとして処理する。
        """
        norm = _nfc(cost_text)

        if re.search(_nfc(r'このキャラをレスト|このリーダーをレスト|このキャラをレストにできる'), norm):
            return GameAction(
                type=ActionType.REST,
                target=TargetQuery(
                    player=Player.SELF,
                    zone=Zone.FIELD,
                    count=1,
                    is_strict_count=True,
                    ref_id="self"
                ),
                raw_text=norm
            )

        return self._parse_to_node(norm, is_cost=True)

    def _detect_trigger(self, text: str) -> TriggerType:
        norm_text = _nfc(text)
        if _nfc("【登場時】") in norm_text: return TriggerType.ON_PLAY
        if _nfc("【起動メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【アタック時】") in norm_text: return TriggerType.ON_ATTACK
        if _nfc("【ブロック時】") in norm_text: return TriggerType.ON_BLOCK
        if _nfc("【KO時】") in norm_text: return TriggerType.ON_KO
        if _nfc("【自分のターン終了時】") in norm_text: return TriggerType.TURN_END
        if _nfc("【相手のターン終了時】") in norm_text: return TriggerType.OPP_TURN_END
        if _nfc("【相手のアタック時】") in norm_text: return TriggerType.OPPONENT_ATTACK
        if _nfc("【自分のターン中】") in norm_text: return TriggerType.YOUR_TURN
        if _nfc("【相手のターン中】") in norm_text: return TriggerType.OPPONENT_TURN
        if _nfc("【カウンター】") in norm_text: return TriggerType.COUNTER
        if _nfc("【トリガー】") in norm_text: return TriggerType.TRIGGER
        if _nfc("【ゲーム開始時】") in norm_text: return TriggerType.GAME_START
        return TriggerType.UNKNOWN

    def _parse_to_node(self, text: str, is_cost: bool = False) -> EffectNode:
        norm_text = _nfc(text)

        split_pattern = _nfc(r'。|その後、|置き、|加え、|引く、|捨て、|発動できる、|させ、')

        parts = re.split(split_pattern, norm_text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > 1:
            return Sequence(actions=[self._parse_logic_block(p, is_cost) for p in parts])
        elif parts:
            return self._parse_logic_block(parts[0], is_cost)
        return None

    def _parse_logic_block(self, text: str, is_cost: bool) -> EffectNode:
        norm_text = _nfc(text)

        # 条件分岐
        match = re.search(_nfc(r'^(.+?)(?:場合|なら|することで)、(.+)$'), norm_text)
        if match:
            cond_text, rest_text = match.groups()
            return Branch(
                condition=self._parse_condition_obj(cond_text),
                if_true=self._parse_to_node(rest_text, is_cost)
            )

        # 選択肢
        if _nfc("以下から1つを選ぶ") in norm_text:
            options = self._extract_options(norm_text)
            return Choice(
                message=_nfc("効果を選択してください"),
                options=[self._parse_to_node(opt, is_cost) for opt in options],
                option_labels=options
            )

        return self._parse_atomic_action(norm_text, is_cost)

    def _parse_atomic_action(self, text: str, is_cost: bool) -> GameAction:
        norm_text = _nfc(text)
        act_type = self._detect_action_type(norm_text)
        value_src = self._parse_value(norm_text, act_type)

        if _nfc("残りを") in norm_text:
            target_query = TargetQuery(select_mode="ALL", save_id=None, zone=Zone.TEMP, player=Player.SELF)
        else:
            target_query = parse_target(norm_text)

        # destination の推定
        destination = None
        if act_type == ActionType.MOVE_CARD:
            if _nfc("手札") in norm_text and _nfc("加える") in norm_text:
                destination = Zone.HAND
            elif _nfc("ライフの上") in norm_text:
                destination = Zone.LIFE
            elif _nfc("ライフの下") in norm_text:
                destination = Zone.LIFE
            elif _nfc("トラッシュ") in norm_text:
                destination = Zone.TRASH
            elif _nfc("デッキの上") in norm_text:
                destination = Zone.DECK
                target_query.select_mode = "TOP"

        status = None
        keyword_match = re.search(r'『(.*?)』', norm_text)
        if keyword_match:
            status = keyword_match.group(1)

        if _nfc("選び") in norm_text:
            target_query.save_id = "selected_card"
        if _nfc("そのカード") in norm_text or _nfc("そのキャラ") in norm_text:
            target_query.ref_id = "selected_card"

        return GameAction(
            type=act_type,
            target=target_query,
            value=value_src,
            status=status,
            destination=destination,
            raw_text=norm_text
        )

    def _parse_value(self, text: str, act_type: ActionType) -> ValueSource:
        norm_text = _nfc(text)
        nums = re.findall(r'[+-]?\d+', norm_text)
        base_val = int(nums[0]) if nums else 0
        if _nfc("枚につき") in norm_text or _nfc("枚数につき") in norm_text:
            return ValueSource(base=0, dynamic_source="COUNT_REFERENCE", multiplier=base_val if base_val != 0 else 1)
        return ValueSource(base=base_val)

    def _detect_action_type(self, text: str) -> ActionType:
        t = _nfc(text)

        # 引く より先に明確なアクションを判定する（順序が重要）

        # ドン!!付与系（「付与する」が含まれる）
        if (_nfc("付与する") in t or _nfc("付与できる") in t) and (_nfc("ドン!!") in t or _nfc("ドン‼") in t):
            return ActionType.ATTACH_DON

        # ドン!!をアクティブで追加（RAMP_DON）
        if _nfc("アクティブで追加") in t or _nfc("アクティブで加える") in t:
            return ActionType.RAMP_DON

        # カード引く
        if _nfc("引く") in t:
            return ActionType.DRAW

        # KO
        if _nfc("KOする") in t:
            return ActionType.KO

        # ライフへの追加（HEAL = デッキ→ライフ）
        if re.search(_nfc(r'デッキの上から.{0,15}ライフ.{0,5}(加える|置く)'), t):
            return ActionType.HEAL
        if re.search(_nfc(r'デッキの上から.{0,5}(加える|置く)'), t) and _nfc("ライフ") in t:
            return ActionType.HEAL

        # ライフ→手札 または 手札→ライフ などカード移動
        if (_nfc("ライフの上") in t or _nfc("ライフの下") in t) and (_nfc("手札に加える") in t or _nfc("手札に戻す") in t):
            return ActionType.MOVE_CARD
        if (_nfc("手札") in t) and (_nfc("ライフの上に加える") in t or _nfc("ライフの下に加える") in t):
            return ActionType.MOVE_CARD

        # 登場させる
        if _nfc("登場させる") in t:
            return ActionType.PLAY_CARD

        # デッキの上に置く（DECK_TOP）
        if re.search(_nfc(r'デッキの上に置く'), t):
            return ActionType.DECK_TOP
        if re.search(_nfc(r'デッキの上から.{0,3}(目)?に置く'), t):
            return ActionType.DECK_TOP

        # デッキの下に置く（DECK_BOTTOM）
        if _nfc("デッキの下") in t and _nfc("置く") in t:
            return ActionType.DECK_BOTTOM

        # トラッシュに置く / 捨てる
        if _nfc("トラッシュに置く") in t:
            return ActionType.DISCARD
        if _nfc("捨てる") in t:
            return ActionType.DISCARD

        # 手札に戻す / 手札に加える
        if _nfc("手札に戻す") in t or _nfc("手札に加える") in t:
            return ActionType.BOUNCE

        # レストにする
        if _nfc("レストにする") in t or _nfc("レストにできる") in t:
            return ActionType.REST

        # アクティブにする（DON!!以外）
        if _nfc("アクティブにする") in t:
            return ActionType.ACTIVE

        # パワー / 得る → BUFF
        if _nfc("パワー") in t or _nfc("得る") in t:
            return ActionType.BUFF

        return ActionType.OTHER

    def _parse_condition_obj(self, text: str) -> Condition:
        norm_text = _nfc(text)

        # 比較演算子と数値を抽出
        operator = CompareOperator.EQ
        value = 0
        nums = re.findall(_nfc(r'(\d+)枚?(以上|以下|より多い|未満)?'), norm_text)
        if nums:
            value = int(nums[0][0])
            op_str = nums[0][1]
            op_map = {
                _nfc('以上'): CompareOperator.GE,
                _nfc('以下'): CompareOperator.LE,
                _nfc('より多い'): CompareOperator.GT,
                _nfc('未満'): CompareOperator.LT,
            }
            operator = op_map.get(op_str, CompareOperator.EQ)

        # 対象プレイヤーの判定
        p = Player.OPPONENT if _nfc("相手") in norm_text else Player.SELF

        if _nfc("ドン!!") in norm_text or _nfc("ドン‼") in norm_text:
            return Condition(type=ConditionType.DON_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("ライフ") in norm_text:
            return Condition(type=ConditionType.LIFE_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("手札") in norm_text:
            return Condition(type=ConditionType.HAND_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("トラッシュ") in norm_text:
            return Condition(type=ConditionType.TRASH_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("リーダーが") in norm_text or _nfc("リーダーの特徴") in norm_text:
            trait_match = re.search(_nfc(r'[《<]([^》>]+)[》>]'), norm_text)
            name_match = re.search(_nfc(r'「([^」]+)」'), norm_text)
            if trait_match:
                return Condition(type=ConditionType.LEADER_TRAIT, value=trait_match.group(1), player=p, raw_text=norm_text)
            if name_match:
                return Condition(type=ConditionType.LEADER_NAME, value=name_match.group(1), player=p, raw_text=norm_text)

        return Condition(type=ConditionType.GENERIC, raw_text=norm_text)

    def _extract_options(self, text: str) -> List[str]:
        norm_text = _nfc(text)
        lines = norm_text.split('\n')
        options = [re.sub(_nfc(r'^[・\-]\s*'), '', l).strip() for l in lines if l.strip().startswith((_nfc('・'), _nfc('-')))]
        if not options:
            parts = re.split(_nfc(r'、'), norm_text)
            options = [p.strip() for p in parts if _nfc("選ぶ") not in p and _nfc("以下から") not in p]
        return options
