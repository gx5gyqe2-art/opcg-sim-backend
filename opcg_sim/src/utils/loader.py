import json
import os
import unicodedata
import re
from typing import List, Dict, Any, Optional, Tuple
from ..models.models import CardMaster, CardInstance
from ..core.effects.parser import EffectParser
from ..models.effect_types import Ability

try:
    from ..core.effects.catalog import get_manual_ability as _get_manual_ability
except ImportError:
    def _get_manual_ability(card_id: str):  # type: ignore[misc]
        return []
from ..models.enums import CardType, Attribute, Color, TriggerType
from ..utils.logger_config import log_event


# カードが本来持つキーワード能力（タグ【X】が能力として記載されているもの）。
_STATIC_KEYWORDS = ["ブロッカー", "速攻", "ダブルアタック", "バニッシュ"]


def _extract_static_keywords(effect_text: str):
    """effect_text からカード本来のキーワード（【ブロッカー】等）を抽出する。
    「を得る」(条件付き付与=GRANT_KEYWORD)・「を発動できない」(無効)・「を持つ」(参照) の
    文脈は除外し、カード自身が静的に持つキーワードのみ集める。"""
    kws = set()
    if not effect_text:
        return kws
    t = unicodedata.normalize("NFC", effect_text)
    for k in _STATIC_KEYWORDS:
        for m in re.finditer(r'【' + re.escape(k) + r'】', t):
            after = t[m.end():m.end() + 4]
            if after.startswith("を得") or after.startswith("を発動") \
                    or after.startswith("を持") or after.startswith("を無効"):
                continue
            kws.add(k)
            break
    return kws


def make_parser():
    """効果パーサのファクトリ。

    既定は合成ルールレジストリ版 (EffectParserV2)。環境変数
    OPCG_PARSER=legacy を設定すると従来の EffectParser に即時ロールバックできる。
    V2 は「構造分解はレガシー流用＋原子句のみルール化、未対応はレガシーへ
    フォールバック」のため、全カード比較で退行(新規OTHER)が0であることを確認済み。
    """
    if os.environ.get("OPCG_PARSER", "v2").lower() == "legacy":
        return EffectParser()
    try:
        from ..core.effects.parser_v2 import EffectParserV2
        return EffectParserV2()
    except Exception as e:  # 念のため: V2 読み込み失敗時はレガシーへ退避
        log_event("ERROR", "loader.parser_v2_fallback", f"V2 unavailable, using legacy: {e}", player="system")
        return EffectParser()

def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)

class RawDataLoader:
    # ... (変更なし) ...
    @staticmethod
    def load_json(file_path: str) -> Any:
        log_event(level_key="DEBUG", action="loader.load_json", msg=f"Loading JSON from {file_path}...")
        encodings = ['utf-8-sig', 'utf-8', 'cp932']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        log_event(level_key="ERROR", action="loader.file_load_failed", msg=f"Failed to load {file_path}")
        return []

class DataCleaner:
    # ... (既存のメソッドはそのまま) ...
    @staticmethod
    def normalize_text(text: Any) -> str:
        if text is None: return ""
        s = str(text).strip()
        return unicodedata.normalize('NFKC', s)

    @staticmethod
    def parse_int(value: Any, default: int = 0) -> int:
        if isinstance(value, int): return value
        s_val = DataCleaner.normalize_text(value)
        if not s_val or s_val.lower() in ["nan", "-", "null", "none", _nfc("なし"), "n/a"]:
            return default
        nums = re.findall(r'-?\d+', s_val)
        return int(nums[0]) if nums else default

    @staticmethod
    def parse_traits(value: Any) -> List[str]:
        s_val = DataCleaner.normalize_text(value)
        if not s_val: return []
        return [t.strip() for t in s_val.split('/') if t.strip()]

    # Catalog導入に伴い、ここは「Parserを使う場合の補助メソッド」となります
    @staticmethod
    def parse_abilities(text: str, is_trigger: bool = False) -> List[Ability]:
        s_text = DataCleaner.normalize_text(text)
        if not s_text or s_text in [_nfc("なし"), "None", ""]: return []
        try:
            parser = make_parser()
            # Parserのメソッド呼び出し (parse_card_text または parse_ability)
            # ※ 前回の修正に合わせて parse_card_text を推奨
            if hasattr(parser, 'parse_card_text'):
                abilities = parser.parse_card_text(s_text)
            else:
                ability = parser.parse_ability(s_text)
                abilities = [ability] if ability else []
            
            if abilities and is_trigger:
                for ab in abilities:
                    ab.trigger = TriggerType.TRIGGER
            return abilities
        except Exception as e:
            log_event(level_key="ERROR", action="loader.parse_error", msg=f"Text: {s_text[:20]}... Error: {e}")
            return []

    # 変更: 単一のColorではなくList[Color]を返すメソッドに変更
    @staticmethod
    def map_colors(value: str) -> List[Color]:
        clean_str = DataCleaner.normalize_text(value)
        found_colors = []
        # 列挙体の定義順に関わらず全ての色をチェック
        target_colors = [Color.RED, Color.GREEN, Color.BLUE, Color.PURPLE, Color.BLACK, Color.YELLOW]
        for c in target_colors:
            if DataCleaner.normalize_text(str(c.value)) in clean_str or DataCleaner.normalize_text(c.name) in clean_str.upper():
                found_colors.append(c)
        
        if not found_colors:
            return [Color.UNKNOWN]
        return found_colors

    @staticmethod
    def map_card_type(type_str: str) -> CardType:
        clean_str = DataCleaner.normalize_text(type_str)
        for t in CardType:
            if DataCleaner.normalize_text(str(t.value)) in clean_str or DataCleaner.normalize_text(t.name) in clean_str.upper():
                return t
        return CardType.UNKNOWN

    @staticmethod
    def map_attribute(attr_str: str) -> Attribute:
        clean_str = DataCleaner.normalize_text(attr_str)
        for a in Attribute:
            if DataCleaner.normalize_text(str(a.value)) in clean_str or DataCleaner.normalize_text(a.name) in clean_str.upper():
                return a
        return Attribute.NONE

class CardLoader:
    # ... (DB_MAPPING, __init__, load, get_card は変更なし) ...
    DB_MAPPING = {
        "ID": ["number", "Number", _nfc("品番"), _nfc("型番"), "id"],
        "NAME": ["name", "Name", _nfc("名前"), _nfc("カード名")],
        "TYPE": [_nfc("種類"), "Type", "type"],
        "ATTRIBUTE": [_nfc("属性"), "Attribute", "attribute"],
        "COLOR": [_nfc("色"), "Color", "color"],
        "COST": [_nfc("コスト"), "Cost", "cost"],
        "POWER": [_nfc("パワー"), "Power", "power"],
        "COUNTER": [_nfc("カウンター"), "Counter", "counter"],
        "LIFE": [_nfc("ライフ"), "Life", "life"],
        "TRAITS": [_nfc("特徴"), "Traits", "traits"],
        "TEXT": [_nfc("効果(テキスト)"), _nfc("テキスト"), "Text", "text"],
        "TRIGGER": [_nfc("効果(トリガー)"), _nfc("トリガー"), "Trigger", "trigger"],
        "BLOCK_ICON": [_nfc("ブロックアイコン"), "block_icon", "blockIcon"]
    }

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.cards: Dict[str, CardMaster] = {}
        self.raw_db: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        data = RawDataLoader.load_json(self.json_path)
        raw_list = data if isinstance(data, list) else []
        for item in raw_list:
            card_id = DataCleaner.normalize_text(item.get("number", item.get("Number", "")))
            if card_id:
                self.raw_db[card_id] = item
        log_event(level_key="INFO", action="loader.db_initialized", msg=f"Database initialized with {len(self.raw_db)} entries.")

    def get_card(self, card_id: str) -> Optional[CardMaster]:
        if card_id in self.cards:
            return self.cards[card_id]
        raw_data = self.raw_db.get(card_id)
        if not raw_data:
            return None
        normalized_raw = {DataCleaner.normalize_text(k): v for k, v in raw_data.items()}
        master = self._create_card_master(normalized_raw)
        if master:
            self.cards[card_id] = master
        return master

    def _create_card_master(self, raw: Dict[str, Any]) -> Optional[CardMaster]:
        def get_val(target_keys: List[str], default=None):
            for k in target_keys:
                norm_key = DataCleaner.normalize_text(k)
                if norm_key in raw:
                    return raw[norm_key]
            return default
        M = self.DB_MAPPING
        card_id = DataCleaner.normalize_text(get_val(M["ID"], "N/A"))
        if not card_id or card_id == "N/A" or "dummy" in card_id.lower():
            return None
        
        name = DataCleaner.normalize_text(get_val(M["NAME"]))
        type_val = get_val(M["TYPE"])
        c_type = DataCleaner.map_card_type(type_val) if type_val else CardType.UNKNOWN
        attr_val = get_val(M["ATTRIBUTE"])
        attribute = DataCleaner.map_attribute(attr_val) if attr_val else Attribute.NONE
        
        # 変更: 単一のmap_colorではなくmap_colorsを使用してリストを取得
        color_val = get_val(M["COLOR"])
        colors = DataCleaner.map_colors(color_val) if color_val else [Color.UNKNOWN]
        
        cost = DataCleaner.parse_int(get_val(M["COST"]))
        power = DataCleaner.parse_int(get_val(M["POWER"]))
        counter = DataCleaner.parse_int(get_val(M["COUNTER"]))
        life = DataCleaner.parse_int(get_val(M["LIFE"]))
        traits = DataCleaner.parse_traits(get_val(M["TRAITS"]))
        effect_text = DataCleaner.normalize_text(get_val(M["TEXT"]))
        trigger_text = DataCleaner.normalize_text(get_val(M["TRIGGER"]))
        block_icon = DataCleaner.normalize_text(get_val(M["BLOCK_ICON"], "")) or ""

        # 優先順位: catalog override (手動定義) > parser.py (自動)
        manual_abilities = _get_manual_ability(card_id)
        if manual_abilities:
            combined_abilities = tuple(manual_abilities)
            log_event("DEBUG", "loader.manual_load", f"Loaded manual abilities for {card_id} ({name})")
        else:
            parser = make_parser()
            main_abilities = parser.parse_card_text(effect_text) if effect_text else []
            trigger_abilities = parser.parse_card_text(trigger_text, as_trigger=True) if trigger_text else []
            combined_abilities = tuple(main_abilities + trigger_abilities)

        # カードが本来持つキーワード（【ブロッカー】等）を effect_text から抽出する。
        # 従来 master.keywords は常に空で、has_keyword("ブロッカー") が False になり
        # ブロッカーが一切機能しなかった（has_blocker が常に False → BLOCK_STEP に入らない）。
        keywords = _extract_static_keywords(effect_text)

        # color 引数を colors に変更
        return CardMaster(
            card_id=card_id, name=name, type=c_type, colors=colors, cost=cost, power=power,
            counter=counter, attribute=attribute, traits=traits, effect_text=effect_text,
            trigger_text=trigger_text, life=life, block_icon=block_icon, abilities=combined_abilities,
            keywords=keywords
        )

class DeckLoader:
    def __init__(self, card_loader: CardLoader):
        self.card_loader = card_loader

    def load_deck(self, file_path: str, owner_id: str) -> Tuple[Optional[CardInstance], List[CardInstance]]:
        data = RawDataLoader.load_json(file_path)
        deck_data = data[0] if isinstance(data, list) and len(data) > 0 else (data if isinstance(data, dict) else {})
        leader_instance = None
        if "leader" in deck_data:
            leader_id = deck_data["leader"].get("number")
            master = self.card_loader.get_card(leader_id)
            if master: leader_instance = CardInstance(master, owner_id)
        deck_list: List[CardInstance] = []
        if "cards" in deck_data:
            for item in deck_data["cards"]:
                card_id = item.get("number")
                count = item.get("count", 0)
                master = self.card_loader.get_card(card_id)
                if master:
                    for _ in range(count): deck_list.append(CardInstance(master, owner_id))
        log_event(level_key="INFO", action="loader.deck_load_success", msg=f"Loaded Deck: Leader={leader_instance.master.name if leader_instance else 'None'}, Deck Size={len(deck_list)}", player=owner_id)
        return leader_instance, deck_list
