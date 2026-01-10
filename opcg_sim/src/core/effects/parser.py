import re
from typing import List, Optional
from ...models.effect_types import (
    Ability, EffectNode, GameAction, Sequence, Branch, Choice, ValueSource, TargetQuery, Condition, _nfc
)
from ...models.enums import ActionType, TriggerType, ConditionType
from .matcher import parse_target
from ...utils.logger_config import log_event

class EffectParser:
    def __init__(self):
        pass

    def parse_ability(self, text: str) -> Ability:
        log_event("DEBUG", "parser.input", f"Input text: {text[:50]}")
        try:
            norm_text = _nfc(text)
            
            # 【修正】厳格なトリガー検出（【 】のみ対象）
            # 文頭だけでなく、文中にある【ドン!!x】なども考慮して解析
            trigger = self._detect_trigger(norm_text)
            log_event("INFO", "parser.trigger", f"Detected trigger: {trigger.name} from {text[:20]}")
            
            cost_node = None
            effect_text = norm_text
            
            # トリガータグの除去（【 】で囲まれた部分のみを除去）
            # 例: "【登場時】カードを引く" -> "カードを引く"
            clean_text = re.sub(r'【.*?】', '', norm_text).strip()
            
            # コストの分離（「：」で区切られている場合）
            # 例: "手札1枚を捨てる：カードを引く"
            if _nfc(":") in clean_text:
                parts = clean_text.split(_nfc(":"), 1)
                cost_text = parts[0].strip()
                effect_body = parts[1].strip()
                
                # コスト部分を解析
                cost_node = self._parse_to_node(cost_text, is_cost=True)
                effect_text = effect_body
            else:
                effect_text = clean_text

            # 効果本体の解析
            effect_node = self._parse_to_node(effect_text)
            
            return Ability(
                trigger=trigger,
                cost=cost_node,
                effect=effect_node,
                raw_text=norm_text
            )
        except Exception as e:
            log_event(level_key="ERROR", action="parser.parse_ability_error", msg=f"Failed to parse: {text[:20]} | Error: {str(e)}")
            return Ability(trigger=TriggerType.UNKNOWN, effect=None, raw_text=_nfc(text))

    def _detect_trigger(self, text: str) -> TriggerType:
        """
        テキストに含まれる【 】内のキーワードからトリガータイプを判定する。
        """
        norm_text = _nfc(text)
        
        # 優先度の高い順、または特定のキーワードをチェック
        # データ通りの表記【 】を厳格にチェック
        if _nfc("【登場時】") in norm_text: return TriggerType.ON_PLAY
        if _nfc("【起動メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【アタック時】") in norm_text: return TriggerType.ON_ATTACK
        if _nfc("【ブロック時】") in norm_text: return TriggerType.ON_BLOCK
        if _nfc("【KO時】") in norm_text: return TriggerType.ON_KO
        if _nfc("【自分のターン終了時】") in norm_text: return TriggerType.TURN_END
        if _nfc("【相手のアタック時】") in norm_text: return TriggerType.OPPONENT_ATTACK
        if _nfc("【自分のターン中】") in norm_text: return TriggerType.YOUR_TURN # 常時効果に近いがトリガー枠にある場合
        if _nfc("【相手のターン中】") in norm_text: return TriggerType.OPPONENT_TURN
        if _nfc("【カウンター】") in norm_text: return TriggerType.COUNTER
        if _nfc("【トリガー】") in norm_text: return TriggerType.TRIGGER
        
        return TriggerType.UNKNOWN

    def _parse_to_node(self, text: str, is_cost: bool = False) -> EffectNode:
        norm_text = _nfc(text)
        # 句点「。」や「その後、」で分割してSequenceにする
        parts = re.split(_nfc(r'。|その後、'), norm_text)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) > 1:
            return Sequence(actions=[self._parse_logic_block(p, is_cost) for p in parts])
        elif parts:
            return self._parse_logic_block(parts[0], is_cost)
        return None

    def _parse_logic_block(self, text: str, is_cost: bool) -> EffectNode:
        norm_text = _nfc(text)
        
        # 条件分岐「～場合、」
        match = re.search(_nfc(r'^(.+?)(?:場合|なら|することで)、(.+)$'), norm_text)
        if match:
            cond_text, rest_text = match.groups()
            return Branch(
                condition=self._parse_condition_obj(cond_text),
                if_true=self._parse_to_node(rest_text, is_cost)
            )
        
        # 選択肢「以下から1つを選ぶ」
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
        target_query = parse_target(norm_text)
        
        # キーワード能力付与（『 』で囲まれた部分）
        status = None
        keyword_match = re.search(r'『(.*?)』', norm_text)
        if keyword_match:
            status = keyword_match.group(1)

        # ターゲット選択の保存ID設定
        if _nfc("選び") in norm_text:
            target_query.save_id = "selected_card"
        if _nfc("そのカード") in norm_text or _nfc("そのキャラ") in norm_text:
            target_query.ref_id = "selected_card"
            
        return GameAction(
            type=act_type, 
            target=target_query, 
            value=value_src, 
            status=status,
            raw_text=norm_text
        )

    def _parse_value(self, text: str, act_type: ActionType) -> ValueSource:
        norm_text = _nfc(text)
        # 数値の抽出
        nums = re.findall(r'[+-]?\d+', norm_text)
        base_val = int(nums[0]) if nums else 0
        
        # 「～枚につき」のような動的な値
        if _nfc("枚につき") in norm_text or _nfc("枚数につき") in norm_text:
            return ValueSource(base=0, dynamic_source="COUNT_REFERENCE", multiplier=base_val if base_val != 0 else 1)
            
        return ValueSource(base=base_val)

    def _detect_action_type(self, text: str) -> ActionType:
        norm_text = _nfc(text)
        if _nfc("引く") in norm_text: return ActionType.DRAW
        if _nfc("KOする") in norm_text: return ActionType.KO
        if _nfc("パワー") in norm_text: return ActionType.BUFF
        if _nfc("登場させる") in norm_text: return ActionType.PLAY_CARD
        if _nfc("トラッシュに置く") in norm_text: return ActionType.DISCARD # 文脈によるが一旦DISCARD
        if _nfc("捨てる") in norm_text: return ActionType.DISCARD
        if _nfc("手札に戻す") in norm_text: return ActionType.BOUNCE
        if _nfc("レストにする") in norm_text: return ActionType.REST
        if _nfc("アクティブにする") in norm_text: return ActionType.ACTIVE
        if _nfc("ライフの上") in norm_text or _nfc("ライフの下") in norm_text: return ActionType.HEAL # 簡易判定
        if _nfc("得る") in norm_text: return ActionType.BUFF # キーワード付与など
        return ActionType.OTHER

    def _parse_condition_obj(self, text: str) -> Condition:
        norm_text = _nfc(text)
        if _nfc("ドン!!") in norm_text: # 【ドン!!】表記も含む
            return Condition(type=ConditionType.DON_COUNT, raw_text=norm_text)
        if _nfc("ライフ") in norm_text:
            return Condition(type=ConditionType.LIFE_COUNT, raw_text=norm_text)
        if _nfc("手札") in norm_text:
             return Condition(type=ConditionType.HAND_COUNT, raw_text=norm_text)
        return Condition(type=ConditionType.GENERIC, raw_text=norm_text)

    def _extract_options(self, text: str) -> List[str]:
        norm_text = _nfc(text)
        # 「・」や改行で区切られた選択肢を抽出
        lines = norm_text.split('\n')
        options = [re.sub(_nfc(r'^[・\-]\s*'), '', l).strip() for l in lines if l.strip().startswith((_nfc('・'), _nfc('-')))]
        if not options:
            # 「Aか、B」のようなパターン（簡易）
            parts = re.split(_nfc(r'、'), norm_text)
            options = [p.strip() for p in parts if _nfc("選ぶ") not in p and _nfc("以下から") not in p]
        return options
