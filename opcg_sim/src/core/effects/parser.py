import re
from typing import List, Optional
from ...models.effect_types import (
    Ability, EffectNode, GameAction, Sequence, Branch, Choice, ValueSource, TargetQuery, Condition
)
from ...models.enums import ActionType, TriggerType, ConditionType
from .matcher import parse_target
from ...utils.logger_config import log_event

class EffectParser:
    def __init__(self):
        pass

    def parse_ability(self, text: str) -> Ability:
        try:
            trigger = self._detect_trigger(text)
            
            cost_node = None
            effect_text = text
            if ":" in text:
                cost_part, effect_part = text.split(":", 1)
                cost_node = self._parse_to_node(cost_part, is_cost=True)
                effect_text = effect_part

            effect_node = self._parse_to_node(effect_text)

            log_event(level_key="DEBUG", action="parser.parse_ability_success", msg=f"Parsed ability: {text[:30]}...")
            return Ability(
                trigger=trigger,
                cost=cost_node,
                effect=effect_node,
                raw_text=text
            )
        except Exception as e:
            log_event(level_key="ERROR", action="parser.parse_ability_error", msg=f"Failed to parse: {text} | Error: {str(e)}")
            return Ability(trigger=TriggerType.UNKNOWN, effect=None, raw_text=text)

    def _parse_to_node(self, text: str, is_cost: bool = False) -> EffectNode:
        parts = re.split(r'。|その後、', text)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) > 1:
            return Sequence(actions=[self._parse_logic_block(p, is_cost) for p in parts])
        return self._parse_logic_block(parts[0], is_cost)

    def _parse_logic_block(self, text: str, is_cost: bool) -> EffectNode:
        match = re.search(r'^(.+?)(?:場合|なら|することで)、(.+)$', text)
        if match:
            cond_text, rest_text = match.groups()
            return Branch(
                condition=self._parse_condition_obj(cond_text),
                if_true=self._parse_to_node(rest_text, is_cost)
            )

        if "以下から1つを選ぶ" in text:
            options = self._extract_options(text)
            return Choice(
                message="効果を選択してください",
                options=[self._parse_to_node(opt, is_cost) for opt in options],
                option_labels=options
            )

        return self._parse_atomic_action(text, is_cost)

    def _parse_atomic_action(self, text: str, is_cost: bool) -> GameAction:
        act_type = self._detect_action_type(text)
        value_src = self._parse_value(text, act_type)
        target_query = parse_target(text)
        
        if "選び" in text:
            target_query.save_id = "selected_card"
        
        if "そのカード" in text or "そのキャラ" in text:
            target_query.ref_id = "selected_card"

        return GameAction(
            type=act_type,
            target=target_query,
            value=value_src,
            raw_text=text
        )

    def _parse_value(self, text: str, act_type: ActionType) -> ValueSource:
        nums = re.findall(r'[+-]?\d+', text)
        base_val = int(nums[0]) if nums else 0
        
        if "枚につき" in text or "枚数につき" in text:
            return ValueSource(
                base=0,
                dynamic_source="COUNT_REFERENCE",
                multiplier=base_val if base_val != 0 else 1
            )
            
        return ValueSource(base=base_val)

    def _detect_trigger(self, text: str) -> TriggerType:
        if "『登場時』" in text: return TriggerType.ON_PLAY
        if "『起動メイン』" in text: return TriggerType.ACTIVATE_MAIN
        if "『アタック時』" in text: return TriggerType.ON_ATTACK
        if "『相手のターン中』" in text: return TriggerType.OPPONENT_TURN
        if "『自分のターン中』" in text: return TriggerType.YOUR_TURN
        return TriggerType.UNKNOWN

    def _detect_action_type(self, text: str) -> ActionType:
        if "引く" in text: return ActionType.DRAW
        if "KOする" in text: return ActionType.KO
        if "パワー" in text: return ActionType.BUFF
        if "登場させる" in text: return ActionType.PLAY_CARD
        if "トラッシュに置く" in text: return ActionType.DISCARD
        if "手札に戻す" in text: return ActionType.BOUNCE
        if "レストにする" in text: return ActionType.REST
        return ActionType.OTHER

    def _parse_condition_obj(self, text: str) -> Condition:
        if "ドン!!" in text and "枚以上" in text:
            return Condition(type=ConditionType.DON_COUNT, raw_text=text)
        if "ライフ" in text:
            return Condition(type=ConditionType.LIFE_COUNT, raw_text=text)
        return Condition(type=ConditionType.GENERIC, raw_text=text)

    def _extract_options(self, text: str) -> List[str]:
        lines = text.split('\n')
        options = [re.sub(r'^[・\-]\s*', '', l).strip() for l in lines if l.strip().startswith(('・', '-'))]
        if not options:
            parts = re.split(r'、', text)
            options = [p.strip() for p in parts if "選ぶ" not in p]
        return options
