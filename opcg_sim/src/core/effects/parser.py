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
        
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'（.*?）', '', text)
        
        replacements = {
            '[': '『', ']': '』', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            '：': ':', '。': '。', '、': '、',
            '−': '-', '‒': '-', '–': '-',
            '＋': '+', '➕': '+',
            '／': '/',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'ドン!!', 'ドン', text)
        text = re.sub(r'DON!!', 'ドン', text)
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
        if '見て' in text or '公開' in text or '見る' in text:
            return self._handle_look_action(text)

        act_type = self._detect_action_type(text)
        val = self._extract_number(text)

        target = None
        NO_TARGET_ACTIONS = [
            ActionType.DRAW, 
            ActionType.RAMP_DON, 
            ActionType.SHUFFLE, 
            ActionType.LIFE_RECOVER,
            ActionType.VICTORY,
            ActionType.RULE_PROCESSING,
            ActionType.SELECT_OPTION,
            ActionType.REPLACE_EFFECT,
            ActionType.MODIFY_DON_PHASE
        ]
        
        if act_type not in NO_TARGET_ACTIONS:
            if any(kw in text for kw in ['それ', 'そのカード', 'そのキャラ']):
                target = TargetQuery(select_mode="REFERENCE", raw_text="last_target")
                if not target.tag: target.tag = "last_target"
            else:
                target = parse_target(text)
                if any(kw in text for kw in ['選び', '対象とし']):
                    target.tag = "last_target"

        return [EffectAction(
            type=act_type,
            target=target,
            value=val,
            raw_text=text
        )]

    def _detect_action_type(self, text: str) -> ActionType:
        if 'アタック' in text and '対象' in text and '変更' in text:
            return ActionType.REDIRECT_ATTACK

        if 'ドン' in text and '戻す' in text and 'ドンデッキ' in text:
            return ActionType.RETURN_DON
        
        if '付与されているドン' in text and '付与する' in text:
            return ActionType.MOVE_ATTACHED_DON

        if 'ドンフェイズ' in text:
            return ActionType.MODIFY_DON_PHASE

        if 'ダメージ' in text and ('与え' in text or '受ける' in text):
            return ActionType.DEAL_DAMAGE

        if '代わりに' in text: return ActionType.REPLACE_EFFECT
        if '選ぶ' in text and ('つ' in text or 'から' in text): return ActionType.SELECT_OPTION
        if 'シャッフル' in text: return ActionType.SHUFFLE
        if 'コスト' in text and 'にする' in text: return ActionType.SET_COST
        if '場を離れない' in text: return ActionType.PREVENT_LEAVE
        if 'デッキ' in text and '上' in text and ('置く' in text or '戻す' in text or '加える' in text): return ActionType.DECK_TOP
        if 'できない' in text or '不可' in text or '加えられない' in text: return ActionType.RESTRICTION
        if '発動する' in text and ('効果' in text or 'イベント' in text): return ActionType.EXECUTE_MAIN_EFFECT
        if '勝利する' in text and ('ゲーム' in text or '敗北' in text): return ActionType.VICTORY
        if 'としても扱う' in text or '何枚でも' in text or 'カウンター' in text: return ActionType.RULE_PROCESSING
        if 'アタック' in text and ('できない' in text or '不可' in text): return ActionType.ATTACK_DISABLE
        if '無効' in text: return ActionType.NEGATE_EFFECT
            
        if 'ライフ' in text:
            if '加える' in text: return ActionType.LIFE_MANIPULATE
            if '置く' in text or '向き' in text: return ActionType.LIFE_MANIPULATE

        if 'コスト' in text and ('-' in text or '下げる' in text or '+' in text or '上げる' in text):
             return ActionType.COST_CHANGE
        
        if '得る' in text: return ActionType.GRANT_KEYWORD
        if 'ドン' in text and '追加' in text: return ActionType.RAMP_DON
        if '引く' in text: return ActionType.DRAW
        if '登場' in text: return ActionType.PLAY_CARD
        if 'KO' in text: return ActionType.KO
        if '手札' in text and ('戻す' in text or '加える' in text): return ActionType.MOVE_TO_HAND
        if 'トラッシュ' in text or '捨てる' in text: return ActionType.TRASH
        if 'デッキ' in text and '下' in text: return ActionType.DECK_BOTTOM
        if 'パワー' in text: return ActionType.BUFF
        if 'レスト' in text: return ActionType.REST
        if 'アクティブ' in text: return ActionType.ACTIVE
        
        return ActionType.OTHER

    def _extract_number(self, text: str) -> int:
        match = re.search(r'([-\u2212\u2010\u2011\u2012\u2013\u2014\u2015\uff0d+]?)(\d+)', text)
        if match:
            sign = match.group(1)
            num = int(match.group(2))
            if sign in ['-', '\u2212', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\uff0d']:
                return -num
            return num
        return 0

    def _parse_condition(self, text: str) -> Optional[Condition]:
        type_ = ConditionType.NONE
        op = CompareOperator.EQ
        val = 0
        target_in_condition = None

        if '公開したカード' in text:
            type_ = ConditionType.CONTEXT
            if 'イベント' in text: val = "TYPE_EVENT"
            elif 'キャラ' in text: val = "TYPE_CHARACTER"
            elif '特徴' in text:
                val = "HAS_TRAIT"
                m = re.search(r'[《<]([^》>]+)[》>]', text)
                if m: target_in_condition = TargetQuery(raw_text=m.group(0), traits=[m.group(1)])
            elif 'コスト' in text:
                val = "COST_CHECK"
                nums = re.findall(r'(\d+)', text)
                if nums: target_in_condition = TargetQuery(raw_text=text, cost_min=int(nums[0]))

        elif 'そうした' in text or '登場させた' in text:
            type_ = ConditionType.CONTEXT
            val = "LAST_ACTION_SUCCESS"
        
        elif 'ライフ' in text: type_ = ConditionType.LIFE_COUNT
        elif 'ドン' in text: type_ = ConditionType.DON_COUNT
        elif '手札' in text: type_ = ConditionType.HAND_COUNT
        elif 'トラッシュ' in text: type_ = ConditionType.TRASH_COUNT
        elif 'デッキ' in text: type_ = ConditionType.DECK_COUNT
        elif '特徴' in text: type_ = ConditionType.HAS_TRAIT
        elif 'リーダー' in text: type_ = ConditionType.LEADER_NAME
        elif 'キャラ' in text or '持つ' in text: type_ = ConditionType.HAS_UNIT

        if type_ not in [ConditionType.CONTEXT, ConditionType.NONE]:
             if type_ in [ConditionType.HAS_TRAIT, ConditionType.HAS_UNIT]:
                 target_in_condition = parse_target(text)
             
             nums = re.findall(r'(\d+)', text)
             if nums: val = int(nums[0])

        if '以上' in text: op = CompareOperator.GE
        elif '以下' in text: op = CompareOperator.LE
        
        str_val = ""
        m_name = re.search(r'[「『]([^」』]+)[」』]', text)
        if m_name: 
            str_val = m_name.group(1)
            if type_ == ConditionType.NONE: type_ = ConditionType.LEADER_NAME
        
        m_trait = re.search(r'[《<]([^》>]+)[》>]', text)
        if m_trait:
            str_val = m_trait.group(1)
            if type_ != ConditionType.CONTEXT:
                type_ = ConditionType.HAS_TRAIT
                op = CompareOperator.HAS

        if type_ == ConditionType.LEADER_NAME: 
            val = str_val
            op = CompareOperator.EQ
        elif type_ == ConditionType.HAS_TRAIT and type_ != ConditionType.CONTEXT:
            val = str_val
            op = CompareOperator.HAS

        return Condition(
            type=type_, 
            operator=op, 
            value=val, 
            target=target_in_condition,
            raw_text=text
        )

    def _handle_look_action(self, text: str) -> List[EffectAction]:
        val = self._extract_number(text)
        if val <= 0: val = 1
        
        look = EffectAction(
            type=ActionType.LOOK, 
            value=val, 
            source_zone=Zone.DECK, 
            dest_zone=Zone.TEMP, 
            raw_text=f"デッキの上から{val}枚を見る"
        )
        
        if '加える' in text or '公開' in text:
            move_target = parse_target(text)
            move_target.zone = Zone.TEMP
            move_target.tag = "last_target"
            
            move = EffectAction(
                type=ActionType.MOVE_TO_HAND, 
                target=move_target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.HAND,
                raw_text="選択して手札に加える"
            )
            look.then_actions.append(move)
            
        if '残り' in text or '下' in text:
            rem_target = TargetQuery(zone=Zone.TEMP, select_mode="ALL", player=Player.SELF)
            bottom = EffectAction(
                type=ActionType.DECK_BOTTOM, 
                target=rem_target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.DECK, 
                dest_position="BOTTOM",
                raw_text="残りをデッキの下に置く"
            )
            if look.then_actions:
                look.then_actions[-1].then_actions.append(bottom)
            else:
                look.then_actions.append(bottom)

        return [look]
