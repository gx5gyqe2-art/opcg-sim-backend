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
            '[': '『', ']': '』', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            ':': ':', '。': '。', '、': '、'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'ドン!!', 'ドン', text)
        text = re.sub(r'DON!!', 'ドン', text)
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
            
            if actions or costs or trigger != TriggerType.UNKNOWN:
                self.abilities.append(Ability(trigger=trigger, costs=costs, actions=actions, raw_text=part))

    def _detect_trigger(self, text: str) -> TriggerType:
        if '『登場時』' in text: return TriggerType.ON_PLAY
        if '『起動メイン』' in text: return TriggerType.ACTIVATE_MAIN
        if '『アタック時』' in text: return TriggerType.ON_ATTACK
        if '『ブロック時』' in text: return TriggerType.ON_BLOCK
        if '『KO時』' in text: return TriggerType.ON_KO
        if '『ターン終了時』' in text: return TriggerType.TURN_END
        if '『相手のターン終了時』' in text: return TriggerType.OPP_TURN_END
        if '『自分のターン中』' in text: return TriggerType.PASSIVE
        if '『相手のターン中』' in text: return TriggerType.PASSIVE
        if '『カウンター』' in text: return TriggerType.COUNTER
        if '『トリガー』' in text: return TriggerType.TRIGGER
        if '『ルール』' in text: return TriggerType.RULE
        if '時、' in text: return TriggerType.TRIGGER 
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
                
                is_optional = 'できる' in part
                
                for act in current_actions:
                    if is_optional:
                        if act.details is None: act.details = {}
                        act.details['optional'] = True

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
        match = re.search(r'^(.+?)(場合|なら|することで|につき)、(.+)$', text)
        if match:
            condition_text = match.group(1)
            result_text = match.group(3)
            
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
        if 'ドン' in text and '追加' in text:
            target = None
        elif any(kw in text for kw in ['それ', 'そのカード', 'そのキャラ']):
            target = TargetQuery(select_mode="REFERENCE", raw_text="last_target")
            target.tag = "last_target"
        else:
            target = parse_target(text)
            if any(kw in text for kw in ['選び', '対象とし', 'にする']):
                target.tag = "last_target"

        act_type = self._detect_action_type(text)
        val = self._extract_number(text)
        
        return [EffectAction(
            type=act_type,
            target=target,
            value=val,
            source_zone=target.zone if target else Zone.ANY,
            dest_zone=Zone.ANY,
            raw_text=text
        )]

    def _detect_action_type(self, text: str) -> ActionType:
        if '引く' in text: return ActionType.DRAW
        if 'ドン' in text and '追加' in text: return ActionType.RAMP_DON
        if '登場' in text: return ActionType.PLAY_CARD
        if 'KO' in text: return ActionType.KO
        if '手札' in text and ('戻す' in text or '加える' in text): return ActionType.MOVE_TO_HAND
        if ('手札' in text and '捨てる' in text) or ('トラッシュ' in text and '置く' in text) or 'トラッシュ' in text or '捨てる' in text: return ActionType.TRASH
        if 'ライフ' in text and '加える' in text: return ActionType.LIFE_RECOVER
        
        if 'パワー' in text:
            if 'する' in text and not any(k in text for k in ['+', '-', '＋', '−', 'プラス', 'マイナス']): return ActionType.SET_BASE_POWER
            return ActionType.BUFF
        if 'コスト' in text and ('+' in text or '-' in text): return ActionType.COST_BUFF
        
        if 'アタックできない' in text: return ActionType.LOCK
        if '無効' in text: return ActionType.NEGATE_EFFECT
        if 'レスト' in text: return ActionType.REST
        if 'アクティブ' in text: return ActionType.ACTIVE
        
        if '得る' in text: return ActionType.GRANT_EFFECT
        
        if 'ドン' in text:
            if '付与' in text or '付ける' in text: return ActionType.ATTACH_DON
            if 'レスト' in text: return ActionType.REST_DON
            if 'アクティブ' in text: return ActionType.ACTIVE_DON
            if 'デッキ' in text: return ActionType.RETURN_DON

        if 'デッキ' in text:
            if '下' in text: return ActionType.DECK_BOTTOM
            if '上' in text: return ActionType.DECK_TOP
            if '順番' in text or '並び替え' in text: return ActionType.SHUFFLE

        return ActionType.OTHER

    def _extract_number(self, text: str) -> int:
        nums = re.findall(r'(\d+)', text)
        val = int(nums[0]) if nums else 0
        if '-' in text or '−' in text or 'ダウン' in text:
            val = -val
        return val

    def _parse_condition(self, text: str) -> Optional[Condition]:
        type_ = ConditionType.NONE
        
        if 'ライフ' in text: type_ = ConditionType.LIFE_COUNT
        elif '手札' in text: type_ = ConditionType.HAND_COUNT
        elif 'トラッシュ' in text: type_ = ConditionType.TRASH_COUNT
        elif 'ドン' in text: type_ = ConditionType.DON_COUNT
        elif '場' in text or 'キャラ' in text: type_ = ConditionType.FIELD_COUNT
        elif 'リーダー' in text and '特徴' not in text: type_ = ConditionType.LEADER_NAME
        elif '特徴' in text: type_ = ConditionType.HAS_TRAIT
        elif '速攻' in text or 'ブロッカー' in text: type_ = ConditionType.HAS_UNIT
        
        target = None
        if type_ in [ConditionType.HAS_TRAIT, ConditionType.FIELD_COUNT, ConditionType.HAS_UNIT]:
            target = parse_target(text)

        nums = re.findall(r'(\d+)', text)
        val = int(nums[0]) if nums else 0
        
        str_val = val
        if type_ == ConditionType.LEADER_NAME:
            m = re.search(r'「([^」]+)」', text)
            if m: str_val = m.group(1)
        elif type_ == ConditionType.HAS_TRAIT:
            m = re.search(r'《([^》]+)》', text)
            if m: str_val = m.group(1)

        op = CompareOperator.EQ
        if '以上' in text: op = CompareOperator.GE
        elif '以下' in text: op = CompareOperator.LE
        elif '含む' in text or type_ in [ConditionType.HAS_TRAIT, ConditionType.LEADER_NAME]: op = CompareOperator.HAS
        
        return Condition(type=type_, operator=op, value=str_val, target=target, raw_text=text)

    def _handle_look_action(self, text: str) -> List[EffectAction]:
        val = self._extract_number(text)
        if val <= 0: val = 1
        
        actions = []
        actions.append(EffectAction(
            type=ActionType.LOOK, 
            value=val, 
            source_zone=Zone.DECK, 
            dest_zone=Zone.TEMP, 
            raw_text=f"デッキの上から{val}枚を見る"
        ))
        
        if '加える' in text or '公開' in text:
            target = parse_target(text)
            target.zone = Zone.TEMP
            target.tag = "last_target"
            
            actions.append(EffectAction(
                type=ActionType.MOVE_TO_HAND, 
                target=target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.HAND, 
                raw_text="選択して手札に加える"
            ))
            
        if '残り' in text or '下' in text:
            rem_target = TargetQuery(zone=Zone.TEMP, select_mode="ALL")
            actions.append(EffectAction(
                type=ActionType.DECK_BOTTOM, 
                target=rem_target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.DECK, 
                dest_position="BOTTOM",
                raw_text="残りをデッキの下に置く"
            ))
            
        return actions
