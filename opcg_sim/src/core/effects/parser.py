import re
from typing import List, Optional, Tuple, Any
from ...models.enums import TriggerType, ActionType, Zone
from ...models.effect_types import (
    EffectNode, GameAction, Sequence, Branch, Choice, Ability, 
    TargetQuery, ValueSource
)
from ...utils.logger_config import log_event

class TextScanner:
    """
    テキスト解析の第一段階を行うクラス。
    テキストを正規化し、構造ブロック（トリガー、コスト、効果本体）に分割する。
    """
    def __init__(self, raw_text: str):
        self.raw_text = self._normalize(raw_text)
        self.cursor = 0

    def _normalize(self, text: str) -> str:
        # 表記ゆれの統一と正規化
        text = text.replace('ドン!!', 'DON!!')
        text = text.replace('：', ':') # 全角コロンを半角に
        text = text.replace('、', ',') # 読点をカンマに（内部処理用）
        text = text.replace('。', '.') # 句点をドットに
        # 余分な空白の削除
        return text.strip()

    def parse_structure(self) -> Tuple[Optional[str], Optional[str], str]:
        """
        戻り値: (trigger_text, cost_text, effect_body_text)
        """
        text = self.raw_text
        trigger = None
        cost = None
        body = text

        # 1. トリガーの抽出 [Trigger]
        # 行頭にある [~] をトリガーとして認識
        trigger_match = re.match(r'^\[(.*?)\]', body)
        if trigger_match:
            trigger = trigger_match.group(1)
            body = body[trigger_match.end():].strip()

        # 2. コストの抽出 (Cost):
        # 文中に「:」がある場合、それより前をコストとみなすことが多い
        # ただし「起動メイン」などのキーワードの後にあることが多い
        if ':' in body:
            parts = body.split(':', 1)
            # コストっぽいキーワードが含まれているか簡易チェック
            # (厳密な判定は次のフェーズで行うが、ここでは構造分解のみ)
            potential_cost = parts[0]
            # 括弧書きのコスト (1) や、テキストによるコスト "手札1枚を捨てる" など
            cost = potential_cost.strip()
            body = parts[1].strip()

        return trigger, cost, body

class EffectParser:
    def __init__(self):
        self.scanner = None

    def parse(self, card_master) -> List[Ability]:
        """
        カードのテキスト全体を解析し、Abilityのリストを返すメインメソッド。
        """
        full_text = card_master.effect_text or ""
        trigger_text = card_master.trigger_text or ""
        
        abilities = []

        # 1. 起動メイン/登場時などの通常効果の解析
        if full_text:
            # 複数の効果が含まれている場合（改行などで区切られている場合）の対応も可能だが
            # ここではまず全文を1つのアビリティとして解析を試みる
            parsed_ability = self._parse_single_text(full_text)
            if parsed_ability:
                abilities.append(parsed_ability)

        # 2. トリガー（ライフで受ける効果など）の解析
        if trigger_text:
            # トリガー効果として解析（TriggerTypeは呼び出し元で設定される想定だがここでは構造のみ）
            t_ability = self._parse_single_text(trigger_text, is_trigger=True)
            if t_ability:
                # ライフ等から発動するトリガーであることをマーク
                t_ability.trigger = TriggerType.TRIGGER 
                abilities.append(t_ability)

        return abilities

    def _parse_single_text(self, text: str, is_trigger: bool = False) -> Optional[Ability]:
        scanner = TextScanner(text)
        raw_trigger, raw_cost, raw_body = scanner.parse_structure()

        # トリガータイプの判定
        trigger_type = self._map_trigger_type(raw_trigger)
        if is_trigger:
            trigger_type = TriggerType.TRIGGER

        # コストの解析 (Sequenceとして生成)
        cost_node = self._parse_flow(raw_cost) if raw_cost else None

        # 効果本体の解析 (Sequenceとして生成)
        effect_node = self._parse_flow(raw_body) if raw_body else None

        if not effect_node and not cost_node:
            return None

        return Ability(
            trigger=trigger_type,
            cost=cost_node,
            effect=effect_node,
            condition=None # 必要であれば解析
        )

    def _map_trigger_type(self, text: Optional[str]) -> TriggerType:
        if not text:
            return TriggerType.UNKNOWN
        if "登場時" in text: return TriggerType.ON_PLAY
        if "アタック時" in text: return TriggerType.WHEN_ATTACKING
        if "起動メイン" in text: return TriggerType.ACTIVATE_MAIN
        if "ターン終了時" in text: return TriggerType.TURN_END
        if "ブロック時" in text: return TriggerType.ON_BLOCK
        if "KO時" in text: return TriggerType.ON_KO
        if "カウンター" in text: return TriggerType.COUNTER
        if "トリガー" in text: return TriggerType.TRIGGER
        return TriggerType.UNKNOWN

    def _parse_flow(self, text: str) -> Optional[EffectNode]:
        """
        テキスト（効果本体やコスト）を読み、文節ごとに分割して Sequence を生成する。
        ここが「構造分解」の肝となる部分。
        """
        if not text:
            return None

        # 1. 句読点や接続詞での分割
        # 「Aし、その後Bする」 -> [A, B]
        # 「A。B。」 -> [A, B]
        
        # まず句点(.)で分割
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        actions = []
        for sentence in sentences:
            # 読点(,)や「その後」で分割
            # ※本来は「～場合」などのBranch解析もここで行う
            
            # 簡易的な分割: カンマで区切られた部分をそれぞれアクション候補とする
            # ただし「（対象）を～し、」のような文脈依存の分割が必要
            clauses = [c.strip() for c in sentence.split(',') if c.strip()]
            
            for clause in clauses:
                # 文節をアクションに変換
                action = self._create_action_from_clause(clause)
                if action:
                    actions.append(action)

        if not actions:
            return None
            
        if len(actions) == 1:
            return actions[0]
            
        return Sequence(actions=actions)

    def _create_action_from_clause(self, clause: str) -> EffectNode:
        """
        文節（Clause）を解析し、GameAction または Branch などを返す。
        現状はプレースホルダー的な実装だが、動詞判定の入り口となる。
        """
        # 条件分岐の簡易判定
        if "場合" in clause:
            # TODO: Branch解析の実装 (B案以降で詳細化)
            pass

        # 動詞ベースのアクション判定 (Matcher)
        # ここは将来的に辞書ベースのマッパー(B案)に置き換える
        
        # 例: シャルリア宮の「残りをトラッシュに置く」
        if "残りをトラッシュ" in clause:
            return GameAction(
                type=ActionType.MOVE,
                target=TargetQuery(group="REMAINING_CARDS"), # 特殊グループ
                destination=Zone.TRASH,
                raw_text=clause
            )
            
        # 例: 手札を捨てる
        if "手札" in clause and ("捨てる" in clause or "捨て" in clause):
            return GameAction(
                type=ActionType.DISCARD,
                target=TargetQuery(target_player="SELF", zone=Zone.HAND, count=1), # 仮: 1枚と仮定
                raw_text=clause
            )
            
        # 例: 山札の上を見る
        if "山札の上から" in clause and "見る" in clause:
             match = re.search(r'(\d+)枚', clause)
             count = int(match.group(1)) if match else 1
             return GameAction(
                 type=ActionType.LOOK,
                 target=TargetQuery(zone=Zone.DECK, count=count),
                 raw_text=clause
             )

        # デフォルト: 未解析のアクションとして返す
        # これによりエラーで落ちずに「何もしないアクション」として通過する
        return GameAction(
            type=ActionType.OTHER, 
            raw_text=clause
        )

