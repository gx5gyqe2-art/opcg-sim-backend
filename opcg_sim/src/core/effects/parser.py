from __future__ import annotations
import re
import unicodedata
from typing import List, Optional, Tuple
from ...models.effect_types import Ability, EffectAction, TargetQuery, Condition, _nfc
from ...models.enums import (
    Phase, Player, Zone, ActionType, TriggerType, 
    CompareOperator, ConditionType, ParserKeyword
)
from .matcher import parse_target

class Effect:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.abilities: List[Ability] = []
        self._parse()

    def _normalize(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFKC', text)
        replacements = {
            '[': '『', ']': '『', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            '：': ':', '。': '。', '、': '、'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        return text

    def _parse(self):
        if not self.raw_text: return
        normalized = self._normalize(self.raw_text)
        parts = [p for p in normalized.split('/') if p.strip()]
        for part in parts:
            trigger = self._detect_trigger(part)
            body_text = re.sub(r'『[^』]+』', '', part)
            costs = []
            actions = []
            if ':' in body_text:
                cost_text, effect_text = body_text.split(':', 1)
                costs = self._parse_recursive(cost_text, is_cost=True)
                actions = self._parse_recursive(effect_text)
            else:
                actions = self._parse_recursive(body_text)
            if actions or costs:
                self.abilities.append(Ability(trigger=trigger, costs=costs, actions=actions, raw_text=part))

    def _detect_trigger(self, text: str) -> TriggerType:
        if '『登場時』' in text: return TriggerType.ON_PLAY
        if '『起動メイン』' in text: return TriggerType.ACTIVATE_MAIN
        if '『アタック時』' in text: return TriggerType.ON_ATTACK
        if '『ブロック時』' in text: return TriggerType.ON_BLOCK
        if '『KO時』' in text: return TriggerType.ON_KO
        if '『ターン終了時』' in text: return TriggerType.TURN_END
        if '『相手のターン終了時』' in text: return TriggerType.OPP_TURN_END
        if '『自分のターン中』' in text: return TriggerType.PASSIVE
        if '『相手のターン中』' in text: return TriggerType.PASSIVE
        if '『カウンター』' in text: return TriggerType.COUNTER
        if '『トリガー』' in text: return TriggerType.TRIGGER
        return TriggerType.UNKNOWN

    def _parse_recursive(self, text: str, is_cost: bool = False) -> List[EffectAction]:
        if not text: return []
        sentences = [s for s in text.split('。') if s]
        root_actions = []
        last_action = None

        for sentence in sentences:
            parts = re.split(r'その後、|、その後', sentence)
            for part in parts:
                current_actions = self._parse_logic_block(part, is_cost)
                for act in current_actions:
                    if last_action:
                        last_action.then_actions.append(act)
                    else:
                        root_actions.append(act)
                    last_action = self._get_deepest_action(act)
        return root_actions

    def _get_deepest_action(self, action: EffectAction) -> EffectAction:
        if not action.then_actions:
            return action
        return self._get_deepest_action(action.then_actions[-1])

    def _parse_logic_block(self, text: str, is_cost: bool) -> List[EffectAction]:
        match = re.search(r'^(.+?)(場合|なら|することで)、(.+)$', text)
        if match:
            condition_text, _, result_text = match.groups()
            condition = self._parse_condition(condition_text)
            then_actions = self._parse_recursive(result_text, is_cost)
            return [EffectAction(
                type=ActionType.OTHER,
                condition=condition,
                then_actions=then_actions,
                raw_text=text
            )]
        return self._parse_atomic_action(text, is_cost)

    def _parse_atomic_action(self, text: str, is_cost: bool) -> List[EffectAction]:
        if '見て' in text:
            return self._handle_look_action(text)

        target = None
        if any(kw in text for kw in ['それ', 'そのカード', 'そのキャラ']):
            target = TargetQuery(select_mode="REFERENCE", raw_text="last_target")
        else:
            target = parse_target(text)
            if any(kw in text for kw in ['選び', '対象とし']):
                target.tag = "last_target"

        act_type = self._detect_action_type(text)
        val = self._extract_number(text)
        
        return [EffectAction(
            type=act_type,
            target=target,
            value=val,
            raw_text=text
        )]

    def _detect_action_type(self, text: str) -> ActionType:
        if '引く' in text: return ActionType.DRAW
        if '登場' in text: return ActionType.PLAY_CARD
        if 'KO' in text: return ActionType.KO
        if '手札' in text and ('戻す' in text or '加える' in text): return ActionType.MOVE_TO_HAND
        if 'パワー' in text: return ActionType.BUFF
        if 'レスト' in text: return ActionType.REST
        if 'アクティブ' in text: return ActionType.ACTIVE
        return ActionType.OTHER

    def _extract_number(self, text: str) -> int:
        nums = re.findall(r'(\d+)', text)
        return int(nums[0]) if nums else 0

    def _parse_condition(self, text: str) -> Optional[Condition]:
        type_ = ConditionType.NONE
        if 'ライフ' in text: type_ = ConditionType.LIFE_COUNT
        elif 'ドン' in text: type_ = ConditionType.DON_COUNT
        elif '特徴' in text or '持つ' in text:
            return Condition(type=ConditionType.OTHER, target=parse_target(text), raw_text=text)
        
        nums = re.findall(r'(\d+)', text)
        val = int(nums[0]) if nums else 0
        op = CompareOperator.EQ
        if '以上' in text: op = CompareOperator.GE
        elif '以下' in text: op = CompareOperator.LE
        
        return Condition(type=type_, operator=op, value=val, raw_text=text)

    def _handle_look_action(self, text: str) -> List[EffectAction]:
        val = self._extract_number(text)
        look = EffectAction(type=ActionType.LOOK, value=val, source_zone=Zone.DECK, dest_zone=Zone.TEMP, raw_text=text)
        move_target = parse_target(text)
        move_target.zone = Zone.TEMP
        move_target.tag = "last_target"
        move = EffectAction(type=ActionType.MOVE_TO_HAND, target=move_target, source_zone=Zone.TEMP, dest_zone=Zone.HAND)
        look.then_actions.append(move)
        return [look]
