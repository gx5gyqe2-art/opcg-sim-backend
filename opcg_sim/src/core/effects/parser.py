from __future__ import annotations
import re
import unicodedata
from typing import List, Optional, Any
from ...models.effect_types import Ability, EffectAction, TargetQuery, Condition, _nfc
from ...models.enums import (
    Phase, Player, Zone, ActionType, TriggerType, 
    CompareOperator, ConditionType, ParserKeyword
)
from .matcher import parse_target

class Effect:
    # ... (__init__, _normalize, _parse は変更なし、ただし _normalize 内の !! 置換は既に定数化済) ...
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.abilities: List[Ability] = []
        self._parse()

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        replacements = {
            '[': '『', ']': '』', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            '−': '-', '-': '-', '−': '-', '‒': '-', '–': '-',
            '!!': '!!', '!': '!', 
            '+': '+', '+': '+'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        
        text = re.sub(_nfc(ParserKeyword.DON + "!!"), _nfc(ParserKeyword.DON), text)
        text = re.sub(_nfc(ParserKeyword.DON + "！！"), _nfc(ParserKeyword.DON), text)
        return text

    def _parse(self):
        if not self.raw_text: return
        normalized = self._normalize(self.raw_text)
        parts = normalized.split('/')
        for part in parts:
            part = part.strip()
            if not part: continue
            
            trigger = TriggerType.UNKNOWN
            if _nfc(ParserKeyword.ON_PLAY) in part: trigger = TriggerType.ON_PLAY
            elif _nfc(ParserKeyword.ACTIVATE_MAIN) in part: trigger = TriggerType.ACTIVATE_MAIN
            elif _nfc(ParserKeyword.WHEN_ATTACKING) in part: trigger = TriggerType.WHEN_ATTACKING
            elif _nfc(ParserKeyword.ON_KO) in part: trigger = TriggerType.ON_KO
            elif _nfc(ParserKeyword.MY_TURN) in part: trigger = TriggerType.CONST_EFFECT
            elif _nfc(ParserKeyword.OPPONENT_TURN) in part: trigger = TriggerType.CONST_EFFECT

            costs = []
            actions_text = part
            if ':' in part:
                cost_part, actions_text = part.split(':', 1)
                costs = self._parse_actions(cost_part, is_cost=True)
            actions = self._parse_actions(actions_text)
            
            if actions or costs:
                self.abilities.append(Ability(trigger=trigger, costs=costs, actions=actions, raw_text=part))

    def _parse_actions(self, text: str, is_cost: bool = False) -> List[EffectAction]:
        actions = []
        subject = Player.SELF
        if _nfc(ParserKeyword.OPPONENT) in text:
            subject = Player.OPPONENT

        sub_parts = re.split(_nfc(r'。その後、|。その後|、その後'), text)
        for sub_text in sub_parts:
            if not sub_text: continue
            parsed = self._parse_single_action(sub_text, subject, is_cost)
            if parsed:
                actions.extend(parsed)
        return actions

    def _parse_single_action(self, text: str, subject: Player, is_cost: bool) -> List[EffectAction]:
        actions = []
        target = parse_target(text, subject)
        
        act_type = ActionType.UNKNOWN
        val = 0
        dest_zone = Zone.ANY
        dest_pos = "BOTTOM"

        nums = re.findall(r'(\d+)', text)
        if nums: val = int(nums[0])

        if _nfc(ParserKeyword.DRAW) in text: act_type = ActionType.DRAW
        elif _nfc(ParserKeyword.PLAY) in text:
            act_type = ActionType.PLAY_CARD
            if _nfc(ParserKeyword.FIELD) in text: dest_zone = Zone.FIELD
            elif _nfc(ParserKeyword.LIFE) in text: dest_zone = Zone.LIFE
        elif _nfc(ParserKeyword.KO) in text: act_type = ActionType.KO
        elif _nfc(ParserKeyword.REST) in text: act_type = ActionType.REST
        elif _nfc(ParserKeyword.ACTIVE) in text: act_type = ActionType.ACTIVE
        elif _nfc(ParserKeyword.DISCARD) in text:
            act_type = ActionType.MOVE_CARD
            dest_zone = Zone.TRASH
        elif _nfc(ParserKeyword.LOOK) in text: act_type = ActionType.LOOK
        elif _nfc(ParserKeyword.REVEAL) in text: act_type = ActionType.REVEAL
        elif _nfc(ParserKeyword.ADD_TO_HAND) in text:
            act_type = ActionType.MOVE_CARD
            dest_zone = Zone.HAND
        elif _nfc(ParserKeyword.PLACE_BOTTOM) in text:
            act_type = ActionType.MOVE_CARD
            dest_zone = Zone.DECK
            dest_pos = "BOTTOM"
        
        if _nfc(ParserKeyword.POWER) in text:
            if '+' in text: act_type = ActionType.BP_BUFF
            elif '-' in text:
                act_type = ActionType.BP_BUFF
                val = -val if val > 0 else val
            elif _nfc(ParserKeyword.SET_TO) in text:
                act_type = ActionType.SET_BP

        if _nfc(ParserKeyword.COST) in text:
            if '+' in text: act_type = ActionType.COST_BUFF
            elif '-' in text:
                act_type = ActionType.COST_BUFF
                val = -val if val > 0 else val

        if is_cost:
            if _nfc(ParserKeyword.DON) in text: act_type = ActionType.DON_COST
            elif _nfc(ParserKeyword.HAND) in text:
                act_type = ActionType.MOVE_CARD
                dest_zone = Zone.TRASH
            elif _nfc(ParserKeyword.TRASH) in text: act_type = ActionType.MOVE_CARD

        condition = None
        if _nfc(ParserKeyword.IF_COND) in text:
            condition = self._parse_condition(text)

        if act_type != ActionType.UNKNOWN:
            if act_type == ActionType.LOOK:
                return self._handle_look_action(text, subject, val)

            actions.append(EffectAction(
                type=act_type, subject=subject, target=target, condition=condition,
                value=val, source_zone=target.zone if target else Zone.ANY,
                dest_zone=dest_zone, dest_position=dest_pos, raw_text=text
            ))
        
        return actions

    def _parse_condition(self, text: str) -> Optional[Condition]:
        if _nfc(ParserKeyword.TRASH) in text:
            nums = re.findall(r'(\d+)', text)
            if nums:
                val = int(nums[0])
                op = CompareOperator.GE if _nfc(ParserKeyword.ABOVE) in text else CompareOperator.LE
                return Condition(ConditionType.TRASH_COUNT, operator=op, value=val, raw_text=text)
        
        if _nfc(ParserKeyword.LIFE) in text:
            nums = re.findall(r'(\d+)', text)
            if nums:
                val = int(nums[0])
                op = CompareOperator.LE if _nfc(ParserKeyword.BELOW) in text else CompareOperator.GE
                return Condition(ConditionType.LIFE_COUNT, operator=op, value=val, raw_text=text)

        if _nfc(ParserKeyword.LEADER) in text and _nfc(ParserKeyword.SUBJECT_GA) in text:
            name_match = re.search(r'「([^」]+)」', text)
            if name_match:
                return Condition(ConditionType.LEADER_NAME, value=name_match.group(1), raw_text=text)

        return None

    def _handle_look_action(self, text: str, subject: Player, val: int) -> List[EffectAction]:
        actions = []
        actions.append(EffectAction(
            type=ActionType.LOOK, subject=subject, value=val, source_zone=Zone.DECK, raw_text=text
        ))

        if _nfc(ParserKeyword.REVEAL) in text:
            search_text = text
            if _nfc(ParserKeyword.LOOK) in text:
                parts = text.split(_nfc(ParserKeyword.LOOK), 1)
                if len(parts) > 1:
                    search_text = parts[1]
            
            search_tgt = parse_target(search_text, subject)
            search_tgt.zone = Zone.TEMP
            
            actions.append(EffectAction(ActionType.REVEAL, subject, target=search_tgt, raw_text=search_text))

            if _nfc(ParserKeyword.ADD_TO_HAND) in text:
                actions.append(EffectAction(ActionType.MOVE_CARD, subject, target=search_tgt, source_zone=Zone.TEMP, dest_zone=Zone.HAND, raw_text=text))

            if _nfc(ParserKeyword.REMAINING) in text:
                rem_dest = Zone.DECK
                rem_pos = "BOTTOM"
                if _nfc(ParserKeyword.TRASH) in text:
                    rem_dest = Zone.TRASH
                
                actions.append(EffectAction(
                    type=ActionType.MOVE_CARD,
                    subject=subject,
                    target=TargetQuery(zone=Zone.TEMP, count=-1, select_mode="REMAINING"),
                    dest_zone=rem_dest,
                    dest_position=rem_pos,
                    raw_text=text
                ))
        return actions
