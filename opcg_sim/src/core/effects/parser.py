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
            '：': ':', '。': '。', '、': '、',
            '−': '-', '‒': '-', '–': '-',
            '＋': '+', '➕': '+',
            '／': '/',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        # ドン!!の正規化
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
        # Look系は別メソッドへ
        if '見て' in text or '公開' in text:
            return self._handle_look_action(text)

        # 1. 先にアクションタイプと数値を確定
        act_type = self._detect_action_type(text)
        val = self._extract_number(text)

        # 2. ターゲット解析の実行判断
        target = None
        # ターゲット不要なアクションのリスト
        NO_TARGET_ACTIONS = [
            ActionType.DRAW, 
            ActionType.RAMP_DON, 
            ActionType.SHUFFLE, 
            ActionType.LIFE_RECOVER
        ]
        
        if act_type not in NO_TARGET_ACTIONS:
            # 指示語の判定
            if any(kw in text for kw in ['それ', 'そのカード', 'そのキャラ']):
                target = TargetQuery(select_mode="REFERENCE", raw_text="last_target")
                # 参照先タグのデフォルト設定（必要に応じて）
                if not target.tag: target.tag = "last_target"
            else:
                target = parse_target(text)
                # 選択アクションならタグ付け
                if any(kw in text for kw in ['選び', '対象とし']):
                    target.tag = "last_target"

        # 3. Action生成
        return [EffectAction(
            type=act_type,
            target=target,
            value=val,
            raw_text=text
        )]

    def _detect_action_type(self, text: str) -> ActionType:
        # ▼ 追加ロジック
        # 0. ライフ操作 (手札に加える=MOVE_TO_HAND 以外)
        if 'ライフ' in text:
            if '加える' in text and '手札' not in text: return ActionType.LIFE_MANIPULATE
            if '置く' in text or '向き' in text: return ActionType.LIFE_MANIPULATE

        # 0. コスト操作
        if 'コスト' in text and ('-' in text or '下げる' in text):
             return ActionType.COST_CHANGE
        
        # 0. 能力付与
        if '得る' in text:
            return ActionType.GRANT_KEYWORD
        # ▲ 追加ロジックここまで

        # 1. ドン加速
        if 'ドン' in text and '追加' in text: return ActionType.RAMP_DON
        
        # 2. ドロー
        if '引く' in text: return ActionType.DRAW
        
        # 3. 登場
        if '登場' in text: return ActionType.PLAY_CARD
        
        # 4. KO
        if 'KO' in text: return ActionType.KO
        
        # 5. バウンス / 回収
        if '手札' in text and ('戻す' in text or '加える' in text): return ActionType.MOVE_TO_HAND
        
        # 6. トラッシュ送り / ハンデス / コスト
        if 'トラッシュ' in text or '捨てる' in text: return ActionType.TRASH
        
        # 7. デッキ下送り
        if 'デッキ' in text and '下' in text: return ActionType.DECK_BOTTOM
        
        # 8. パワー増減 (バフ/デバフ)
        # "する" は "3000にする" 等の固定化もあり得るが、変動も含むためBUFFとする
        if 'パワー' in text: return ActionType.BUFF
        
        # 9. レスト / アクティブ
        if 'レスト' in text: return ActionType.REST
        if 'アクティブ' in text: return ActionType.ACTIVE
        
        return ActionType.OTHER

    def _extract_number(self, text: str) -> int:
        # マイナス記号(半角/全角/特殊文字) + 数字 を検索
        match = re.search(r'([-\u2212\u2010\u2011\u2012\u2013\u2014\u2015\uff0d]?)(\d+)', text)
        if match:
            sign = match.group(1)
            num = int(match.group(2))
            return -num if sign else num
        return 0

    def _parse_condition(self, text: str) -> Optional[Condition]:
        type_ = ConditionType.NONE
        op = CompareOperator.EQ
        
        if 'ライフ' in text: type_ = ConditionType.LIFE_COUNT
        elif 'ドン' in text: type_ = ConditionType.DON_COUNT
        elif '手札' in text: type_ = ConditionType.HAND_COUNT
        elif 'トラッシュ' in text: type_ = ConditionType.TRASH_COUNT
        elif '特徴' in text: type_ = ConditionType.HAS_TRAIT
        elif 'リーダー' in text: type_ = ConditionType.LEADER_NAME
        elif 'キャラ' in text or '持つ' in text: type_ = ConditionType.HAS_UNIT

        # ターゲット指定がある条件の場合 (例: 「特徴《XXX》を持つキャラがいる場合」)
        target_in_condition = None
        if type_ in [ConditionType.HAS_TRAIT, ConditionType.HAS_UNIT]:
             target_in_condition = parse_target(text)

        # 数値比較
        val = 0
        nums = re.findall(r'(\d+)', text)
        if nums: val = int(nums[0])
        
        if '以上' in text: op = CompareOperator.GE
        elif '以下' in text: op = CompareOperator.LE
        
        # 文字列条件 (特徴名、リーダー名など)
        str_val = ""
        m_name = re.search(r'[「『]([^」』]+)[」』]', text)
        if m_name: 
            str_val = m_name.group(1)
            if type_ == ConditionType.NONE: type_ = ConditionType.LEADER_NAME # 仮
        
        # 特徴抽出
        m_trait = re.search(r'[《<]([^》>]+)[》>]', text)
        if m_trait:
            str_val = m_trait.group(1)
            type_ = ConditionType.HAS_TRAIT
            op = CompareOperator.HAS

        if type_ == ConditionType.LEADER_NAME: 
            val = str_val
            op = CompareOperator.EQ
        elif type_ == ConditionType.HAS_TRAIT:
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
        
        # 1. デッキ操作
        look = EffectAction(
            type=ActionType.LOOK, 
            value=val, 
            source_zone=Zone.DECK, 
            dest_zone=Zone.TEMP, 
            raw_text=f"デッキの上から{val}枚を見る"
        )
        
        # 2. 移動・選択 (加える/公開)
        if '加える' in text or '公開' in text:
            # ターゲット解析 (条件付きターゲットなど)
            # サーチ系のターゲットは常に「選択式」かつ「タグ付け」する
            move_target = parse_target(text)
            move_target.zone = Zone.TEMP
            move_target.tag = "last_target"
            
            # 手札に加える
            move = EffectAction(
                type=ActionType.MOVE_TO_HAND, 
                target=move_target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.HAND,
                raw_text="選択して手札に加える"
            )
            look.then_actions.append(move)
            
        # 3. 残り処理 (デッキ下へ)
        if '残り' in text or '下' in text:
            # 残りすべて
            rem_target = TargetQuery(zone=Zone.TEMP, select_mode="ALL", player=Player.SELF)
            bottom = EffectAction(
                type=ActionType.DECK_BOTTOM, 
                target=rem_target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.DECK, 
                dest_position="BOTTOM",
                raw_text="残りをデッキの下に置く"
            )
            # MOVEの後に実行されるようにする (兄弟関係ではなく、MOVEの子にするか、LOOKの子として順序保証するか)
            # 現在のResolverは then_actions を順次実行するので、LOOKの子として追加でOK
            # ただしMOVEがある場合はMOVEの後にしたいなら、MOVEのthen_actionsに入れるべき
            if look.then_actions:
                look.then_actions[-1].then_actions.append(bottom)
            else:
                look.then_actions.append(bottom)

        return [look]
