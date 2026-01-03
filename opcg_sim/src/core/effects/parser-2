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
        """
        徹底的な正規化を行い、解析しやすいトークン列に近い状態にする
        """
        if not text: return ""
        text = unicodedata.normalize('NFKC', text)
        
        # 記号の統一
        replacements = {
            '[': '『', ']': '』', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            '−': '-', '-': '-', '−': '-', '‒': '-', '–': '-',
            '!!': '!!', '!': '!', 
            '＋': '+', '➕': '+',
            '／': '/',
            '：': ':',
            '。': '。', '、': '、'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
            
        # スペース削除
        text = re.sub(r'\s+', '', text)
        
        # キーワードの正規化
        text = re.sub(r'ドン!!', 'ドン', text)
        text = re.sub(r'DON!!', 'ドン', text)
        
        return text

    def _parse(self):
        if not self.raw_text: return
        normalized = self._normalize(self.raw_text)
        
        # 複数のアビリティが / で区切られている場合 (例: [登場時]/[トリガー])
        parts = normalized.split('/')
        for part in parts:
            part = part.strip()
            if not part: continue
            
            trigger = TriggerType.UNKNOWN
            # トリガーの検出
            # 最長一致させるためにリスト順序を工夫してもよい
            if '『登場時』' in part: trigger = TriggerType.ON_PLAY
            elif '『起動メイン』' in part: trigger = TriggerType.ACTIVATE_MAIN
            elif '『アタック時』' in part: trigger = TriggerType.ON_ATTACK
            elif '『ブロック時』' in part: trigger = TriggerType.ON_BLOCK
            elif '『KO時』' in part: trigger = TriggerType.ON_KO
            elif '『ターン終了時』' in part: trigger = TriggerType.TURN_END
            elif '『相手のターン終了時』' in part: trigger = TriggerType.OPP_TURN_END
            elif '『自分のターン中』' in part: trigger = TriggerType.PASSIVE
            elif '『相手のターン中』' in part: trigger = TriggerType.PASSIVE
            elif '『カウンター』' in part: trigger = TriggerType.COUNTER
            elif '『トリガー』' in part: trigger = TriggerType.TRIGGER
            elif '『ルール』' in part: trigger = TriggerType.RULE
            
            # 本文の抽出 (トリガーを除去)
            body_text = re.sub(r'『[^』]+』', '', part)
            
            # コストと効果の分離
            # "コスト : 効果" の形式
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

    def _parse_recursive(self, text: str, is_cost: bool = False) -> List[EffectAction]:
        """
        再帰的な構文解析を行うメイン関数
        テキストを構造（条件、順接）に従って分解し、EffectActionのリスト（ツリー）を返す
        """
        if not text: return []
        actions = []

        # 1. 文の区切り ("。") で分割して順次処理
        # ただし、カッコ内などの "。" は無視すべきだが、現状は簡易的に split
        sentences = [s for s in text.split('。') if s]
        
        for sentence in sentences:
            # "その後、" などの接続詞でさらに分割
            parts = re.split(r'その後、|、その後', sentence)
            
            current_chain = []
            for part in parts:
                parsed_actions = self._parse_logic_block(part, is_cost)
                if parsed_actions:
                    current_chain.extend(parsed_actions)
            
            # 連鎖関係の構築 (簡易版: リストに追加するだけだが、論理的には前の結果に依存する)
            actions.extend(current_chain)
            
        return actions

    def _parse_logic_block(self, text: str, is_cost: bool) -> List[EffectAction]:
        """
        1つの論理ブロック（条件＋アクション）を解析する
        例: "自分のリーダーが《特徴》を持つ場合、カードを1枚引く"
        """
        # 条件パターンの検出
        # パターン: [条件]場合、[結果]
        # パターン: [条件]なら、[結果]
        # パターン: [コスト]することで、[結果]
        
        # 正規表現で条件部と実行部を分離
        # 非貪欲マッチ (.+?) で最小の条件部を探す
        match = re.search(r'^(.+?)(場合|なら|することで)、(.+)$', text)
        
        if match:
            condition_text = match.group(1)
            marker = match.group(2)
            result_text = match.group(3)
            
            # 条件の解析
            condition = self._parse_condition(condition_text)
            
            # 結果部の再帰的解析
            # "Aする場合、Bする。そうでなければCする" のような構造はここで分岐
            then_actions = self._parse_recursive(result_text, is_cost)
            
            # 条件チェックアクションを作成
            # 条件自体を1つのActionとして表現し、成功時に then_actions を実行する構造にする
            check_action = EffectAction(
                type=ActionType.OTHER, # 条件チェック用の型があればベストだが、Resolverで condition があればチェックされる
                subject=Player.SELF,
                condition=condition,
                raw_text=f"条件確認: {condition_text}",
                then_actions=then_actions
            )
            return [check_action]
        
        # 条件がない場合、単一アクションとして解析
        return self._parse_atomic_action(text, is_cost)

    def _parse_atomic_action(self, text: str, is_cost: bool) -> List[EffectAction]:
        """
        これ以上分解できない最小単位のアクションを解析
        """
        actions = []
        subject = Player.SELF
        
        # 数値抽出
        nums = re.findall(r'(\d+)', text)
        val = int(nums[0]) if nums else 0
        if is_cost and val == 0 and ('-' in text or '−' in text):
             pass # マイナスコスト等の処理

        # 特殊: 「見て」系 (Look & Select)
        if '見て' in text or '公開' in text:
            return self._handle_look_action(text, subject, val)

        # ターゲット解析
        target = parse_target(text, subject)
        
        act_type = ActionType.OTHER
        dest_zone = Zone.ANY
        
        # --- アクションタイプの判定 (優先度順) ---
        
        # ドロー
        if '引く' in text: act_type = ActionType.DRAW
        
        # 登場/プレイ
        elif '登場' in text: 
            act_type = ActionType.PLAY_CARD
            dest_zone = Zone.FIELD
            if 'レスト' in text: target.is_rest = True
            
        # KO
        elif 'KO' in text: act_type = ActionType.KO
        
        # バウンス (手札に戻す)
        elif '手札' in text and ('戻す' in text or '加える' in text) and 'デッキ' not in text and 'ライフ' not in text:
            # "手札に加える" は回収やサーチ結果の取得で使われる
            # ここでは盤面やトラッシュからの回収を想定
            act_type = ActionType.MOVE_TO_HAND
            dest_zone = Zone.HAND
            
        # デッキ操作
        elif 'デッキ' in text and ('下' in text or '上' in text) and ('戻す' in text or '置く' in text):
            act_type = ActionType.DECK_BOTTOM # 便宜上
            dest_zone = Zone.DECK
            
        # ライフ操作
        elif 'ライフ' in text:
            if '加える' in text: 
                act_type = ActionType.LIFE_RECOVER
                dest_zone = Zone.LIFE
            elif '手札' in text: # ライフを手札に
                act_type = ActionType.MOVE_TO_HAND
                target.zone = Zone.LIFE
            elif 'トラッシュ' in text:
                act_type = ActionType.TRASH
                target.zone = Zone.LIFE

        # トラッシュ送り (ハンデス、コスト)
        elif '捨てる' in text or 'トラッシュ' in text:
            act_type = ActionType.TRASH
            dest_zone = Zone.TRASH
            
        # 状態変更
        elif 'レスト' in text and 'ドン' not in text: act_type = ActionType.REST
        elif 'アクティブ' in text and 'ドン' not in text: act_type = ActionType.ACTIVE
        
        # パワー操作
        elif 'パワー' in text:
            if 'する' in text and '+' not in text and '-' not in text:
                act_type = ActionType.SET_BASE_POWER
            else:
                act_type = ActionType.BUFF
                if '-' in text or '−' in text or 'ダウン' in text:
                    val = -abs(val) # マイナス

        # ドン操作
        elif 'ドン' in text:
            target.zone = Zone.COST_AREA
            if '付与' in text or '付ける' in text:
                act_type = ActionType.ATTACH_DON
                dest_zone = Zone.FIELD
            elif 'レスト' in text:
                act_type = ActionType.REST_DON
            elif 'アクティブ' in text:
                act_type = ActionType.ACTIVE_DON
            elif 'デッキ' in text and '戻す' in text:
                act_type = ActionType.RETURN_DON
                dest_zone = Zone.DON_DECK

        # コスト操作
        elif 'コスト' in text and ('+' in text or '-' in text):
            act_type = ActionType.COST_BUFF
            if '-' in text: val = -abs(val)

        if act_type != ActionType.OTHER:
            actions.append(EffectAction(
                type=act_type,
                subject=subject,
                target=target,
                value=val,
                source_zone=target.zone if target else Zone.ANY,
                dest_zone=dest_zone,
                raw_text=text
            ))
        
        return actions

    def _parse_condition(self, text: str) -> Optional[Condition]:
        type_ = ConditionType.NONE
        op = CompareOperator.EQ
        val = 0
        
        nums = re.findall(r'(\d+)', text)
        if nums: val = int(nums[0])
        
        if '以上' in text: op = CompareOperator.GE
        elif '以下' in text: op = CompareOperator.LE
        
        if 'ライフ' in text: type_ = ConditionType.LIFE_COUNT
        elif '手札' in text: type_ = ConditionType.HAND_COUNT
        elif 'トラッシュ' in text: type_ = ConditionType.TRASH_COUNT
        elif '場' in text or 'キャラ' in text: type_ = ConditionType.FIELD_COUNT
        elif 'ドン' in text: type_ = ConditionType.DON_COUNT
        elif 'リーダー' in text:
            type_ = ConditionType.LEADER_NAME
            m = re.search(r'リーダーが「([^」]+)」', text)
            if m:
                val = m.group(1)
                op = CompareOperator.HAS
        
        # 特徴条件 ("特徴《XXX》を持つ場合")
        elif '特徴' in text:
            type_ = ConditionType.HAS_TRAIT
            m_trait = re.search(r'[《<]([^》>]+)[》>]', text)
            if m_trait:
                val = m_trait.group(1) # 値として特徴名を保持
                op = CompareOperator.HAS

        if type_ != ConditionType.NONE:
            return Condition(type=type_, operator=op, value=val, raw_text=text)
        return None

    def _handle_look_action(self, text: str, subject: Player, val: int) -> List[EffectAction]:
        actions = []
        
        # 1. デッキ操作 (Look)
        if '上から' in text:
            look_count = val if val > 0 else 1
            actions.append(EffectAction(
                type=ActionType.LOOK,
                subject=subject,
                value=look_count,
                source_zone=Zone.DECK,
                dest_zone=Zone.TEMP,
                raw_text=text + "(サーチ)"
            ))

        # 2. 選択・移動 (Move)
        if '加える' in text or '公開' in text:
            search_target = parse_target(text, subject)
            search_target.zone = Zone.TEMP # サーチ結果から選ぶ
            
            actions.append(EffectAction(
                type=ActionType.MOVE_TO_HAND,
                subject=subject,
                target=search_target,
                source_zone=Zone.TEMP,
                dest_zone=Zone.HAND,
                raw_text="選択して手札に加える"
            ))

        # 3. 残り処理 (Bottom)
        if '残り' in text:
            rem_target = TargetQuery(zone=Zone.TEMP, select_mode="ALL", player=Player.SELF)
            actions.append(EffectAction(
                type=ActionType.DECK_BOTTOM,
                subject=subject,
                target=rem_target,
                source_zone=Zone.TEMP,
                dest_zone=Zone.DECK,
                dest_position="BOTTOM",
                raw_text="残りをデッキの下に置く"
            ))
            
        return actions
