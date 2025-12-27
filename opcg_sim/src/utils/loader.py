import json
import unicodedata
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..models.models import CardMaster, CardInstance
from ..core.effects.parser import Effect, TriggerType
from ..core.effect_types import Ability
from ..models.enums import CardType, Attribute, Color

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def _nfc(text: str) -> str:
    """
    ソースコード上の日本語リテラルを強制的にNFC正規化するヘルパー。
    ファイル保存形式(NFD等)による文字コードの不一致を防ぐために使用。
    """
    return unicodedata.normalize('NFC', text)

# ---------------------------------------------------------
# 1. Infrastructure Layer
# ---------------------------------------------------------
class RawDataLoader:
    """
    物理的なデータ読み込みを担当 (2.1 Infrastructure Layer)
    """
    @staticmethod
    def load_json(file_path: str) -> Any:
        logger.info(f"Loading JSON from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []

# ---------------------------------------------------------
# 2. Domain Service Layer
# ---------------------------------------------------------
class DataCleaner:
    """
    データの正規化と型変換を担当するユーティリティ (2.2 Domain Service Layer)
    """
    @staticmethod
    def normalize_text(text: Any) -> str:
        if text is None:
            return ""
        # NFKC正規化で濁点などを合成済みの文字に統一する
        return unicodedata.normalize('NFKC', str(text)).strip()

    @staticmethod
    def parse_int(value: Any, default: int = 0) -> int:
        if isinstance(value, int):
            return value
        
        s_val = DataCleaner.normalize_text(value)
        # "なし" 等の日本語リテラルもNFC化して比較
        if not s_val or s_val.lower() in ["nan", "-", "null", "none", _nfc("なし"), "n/a"]:
            return default
            
        nums = re.findall(r'-?\d+', s_val)
        if nums:
            return int(nums[0])
        return default

    @staticmethod
    def parse_traits(value: Any) -> List[str]:
        s_val = DataCleaner.normalize_text(value)
        if not s_val:
            return []
        # スラッシュ区切りで分割しリスト化
        return [t.strip() for t in s_val.split('/') if t.strip()]

    @staticmethod
    def parse_abilities(text: str, is_trigger: bool = False) -> List[Ability]:
        """
        テキストを解析し、構造化データへ変換するブリッジメソッド。
        解析ロジック自体は effects.Effect クラスへ委譲する。
        """
        s_text = DataCleaner.normalize_text(text)
        # "なし" 等のチェックにもNFC適用
        if not s_text or s_text in [_nfc("なし"), "None", ""]:
            return []
        
        try:
            # Effectクラスへ解析を委譲 (Design 2.1 Parsing Delegation)
            effect_parser = Effect(s_text)
            abilities = effect_parser.abilities

            # トリガー欄からの入力であれば、TriggerTypeを上書き/補正する
            if is_trigger:
                for ability in abilities:
                    # 既に適切なトリガータイプが設定されていない場合のみ強制設定するなど
                    # 運用に合わせて調整可能だが、基本はTRIGGERとする
                    ability.trigger = TriggerType.TRIGGER
            
            return abilities

        except Exception as e:
            logger.error(f"Error parsing abilities text: '{s_text[:20]}...' -> {e}")
            return []

    @staticmethod
    def map_color(value: str) -> Color:
        """文字列からColor Enumへマッピング"""
        clean_str = DataCleaner.normalize_text(value)
        for c in Color:
            # 定義値側も正規化して比較(例: "Red" vs "RED" vs "赤")
            # Enumのvalueが日本語か英語かに依存するため、プロジェクトの定義に合わせて調整
            if DataCleaner.normalize_text(str(c.value)) in clean_str or \
               DataCleaner.normalize_text(c.name) in clean_str.upper():
                return c
        return Color.UNKNOWN  # 定義がない場合は適宜デフォルトを

    @staticmethod
    def map_card_type(type_str: str) -> CardType:
        """文字列からCardType Enumへマッピング"""
        clean_str = DataCleaner.normalize_text(type_str)
        for t in CardType:
            # 日本語Value または 英語Name(大文字) でのマッチング
            if DataCleaner.normalize_text(str(t.value)) in clean_str or \
               DataCleaner.normalize_text(t.name) in clean_str.upper():
                return t
        return CardType.UNKNOWN

    @staticmethod
    def map_attribute(attr_str: str) -> Attribute:
        """文字列からAttribute Enumへマッピング"""
        clean_str = DataCleaner.normalize_text(attr_str)
        for a in Attribute:
            # 日本語Value または 英語Name(大文字) でのマッチング
            if DataCleaner.normalize_text(str(a.value)) in clean_str or \
               DataCleaner.normalize_text(a.name) in clean_str.upper():
                return a
        return Attribute.NONE

# ---------------------------------------------------------
# 3. Repository Layer
# ---------------------------------------------------------
class CardLoader:
    """
    データロードの制御とオブジェクト管理 (2.3 Repository Layer)
    """
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.cards: Dict[str, CardMaster] = {}

    def load(self) -> None:
        """
        メインロードプロセス
        """
        raw_data = RawDataLoader.load_json(self.json_path)
        
        # 配列であることを保証
        raw_list = raw_data if isinstance(raw_data, list) else []

        success_count = 0
        for i, raw_item in enumerate(raw_list):
            # 辞書キーの正規化(揺らぎ吸収)
            normalized_item = {}
            for k, v in raw_item.items():
                norm_k = DataCleaner.normalize_text(k)
                normalized_item[norm_k] = v

            # デバッグ出力(最初の1件のみ)
            if i == 0:
                logger.info("--- [DEBUG] First Card Keys ---")
                for k in normalized_item.keys():
                    logger.info(f"  key: {k}")

            try:
                card = self._create_card_master(normalized_item, debug=(i==0))
                if card:
                    self.cards[card.card_id] = card
                    success_count += 1
            except Exception as e:
                # 個別のエラーはスキップして継続
                logger.warning(f"Skipping card index {i} due to error: {e}")
        
        logger.info(f"Loaded {success_count} cards successfully.")

    def get_card(self, card_id: str) -> Optional[CardMaster]:
        return self.cards.get(card_id)

    def _create_card_master(self, raw: Dict[str, Any], debug: bool = False) -> Optional[CardMaster]:
        """
        辞書データからCardMasterオブジェクトを生成するファクトリメソッド
        """
        
        def get_val(target_keys: List[str], default=None):
            for k in target_keys:
                norm_key = DataCleaner.normalize_text(k)
                if norm_key in raw:
                    return raw[norm_key]
            return default

        # --- 1. 基本情報の抽出 ---
        # "品番", "型番" (NFCラップ)
        card_id = DataCleaner.normalize_text(get_val(["number", "Number", _nfc("品番"), _nfc("型番"), "id"], "N/A"))
        if not card_id or card_id == "N/A" or "dummy" in card_id.lower():
            return None

        # "名前", "カード名" (NFCラップ)
        name = DataCleaner.normalize_text(get_val(["name", "Name", _nfc("名前"), _nfc("カード名")]))
        
        # --- 2. Enumマッピング ---
        # "種類" (NFCラップ)
        type_val = get_val([_nfc("種類"), "Type", "type"])
        c_type = DataCleaner.map_card_type(type_val) if type_val else CardType.UNKNOWN
        
        # "属性" (NFCラップ)
        attr_val = get_val([_nfc("属性"), "Attribute", "attribute"])
        attribute = DataCleaner.map_attribute(attr_val) if attr_val else Attribute.NONE

        # "色" (NFCラップ)
        color_val = get_val([_nfc("色"), "Color", "color"])
        color = DataCleaner.map_color(color_val) if color_val else Color.UNKNOWN
        
        # --- 3. 数値・リスト変換 ---
        # "コスト", "パワー", "カウンター", "ライフ" (NFCラップ)
        cost = DataCleaner.parse_int(get_val([_nfc("コスト"), "Cost", "cost"]))
        power = DataCleaner.parse_int(get_val([_nfc("パワー"), "Power", "power"]))
        counter = DataCleaner.parse_int(get_val([_nfc("カウンター"), "Counter", "counter"]))
        life = DataCleaner.parse_int(get_val([_nfc("ライフ"), "Life", "life"]))
        
        # "特徴" (NFCラップ)
        traits = DataCleaner.parse_traits(get_val([_nfc("特徴"), "Traits", "traits"]))
        
        # --- 4. テキスト解析とアビリティ構築 (Design 3.2 Transform) ---
        # "効果(テキスト)", "テキスト" (NFCラップ)
        effect_text = DataCleaner.normalize_text(get_val([_nfc("効果(テキスト)"), _nfc("テキスト"), "Text", "text"]))
        # "効果(トリガー)", "トリガー" (NFCラップ)
        trigger_text = DataCleaner.normalize_text(get_val([_nfc("効果(トリガー)"), _nfc("トリガー"), "Trigger", "trigger"]))
        
        # メイン効果の解析
        main_abilities = DataCleaner.parse_abilities(effect_text, is_trigger=False)
        
        # トリガー効果の解析 (is_trigger=True)
        trigger_abilities = DataCleaner.parse_abilities(trigger_text, is_trigger=True)
        
        # 両方を結合
        combined_abilities = main_abilities + trigger_abilities

        if debug:
            logger.info(f"--- [DEBUG] Card: {name} ({card_id}) ---")
            logger.info(f"    Type: {c_type}, Color: {color}")
            logger.info(f"    Abilities Count: {len(combined_abilities)} (Main: {len(main_abilities)}, Trig: {len(trigger_abilities)})")

        # --- 5. オブジェクト生成 ---
        return CardMaster(
            card_id=card_id,
            name=name,
            type=c_type,
            color=color,
            cost=cost,
            power=power,
            counter=counter,
            attribute=attribute,
            traits=traits,
            effect_text=effect_text,
            trigger_text=trigger_text,
            life=life,
            abilities=combined_abilities
        )

class DeckLoader:
    """
    デッキファイルを読み込み、CardInstanceのリストを生成するローダークラス。
    """
    def __init__(self, card_loader: CardLoader):
        self.card_loader = card_loader

    def load_deck(self, file_path: str, owner_id: str) -> Tuple[Optional[CardInstance], List[CardInstance]]:
        """
        デッキJSONを読み込み、リーダーとデッキのカードインスタンスリストを返却する。
        
        Args:
            file_path: デッキJSONファイルのパス
            owner_id: カードの所有者ID (例: "Player1")
        """
        data = RawDataLoader.load_json(file_path)
        
        # リストでラップされている場合(RawDataLoaderの仕様依存)や、辞書そのものの場合に対応
        deck_data = {}
        if isinstance(data, list) and len(data) > 0:
            deck_data = data[0]
        elif isinstance(data, dict):
            deck_data = data
        else:
            logger.error(f"Invalid deck format in {file_path}")
            return None, []

        # 1. Leader Loading
        leader_instance = None
        if "leader" in deck_data:
            leader_id = deck_data["leader"].get("number")
            if leader_id:
                leader_master = self.card_loader.get_card(leader_id)
                if leader_master:
                    # 修正: owner_id を渡す
                    leader_instance = CardInstance(leader_master, owner_id)
                else:
                    logger.warning(f"Leader card not found in DB: {leader_id}")

        # 2. Deck Cards Loading
        deck_list: List[CardInstance] = []
        if "cards" in deck_data:
            for item in deck_data["cards"]:
                card_id = item.get("number")
                count = item.get("count", 0)
                
                master = self.card_loader.get_card(card_id)
                if master:
                    for _ in range(count):
                        # 修正: owner_id を渡す
                        deck_list.append(CardInstance(master, owner_id))
                else:
                    logger.warning(f"Deck card not found in DB: {card_id}")

        logger.info(f"Loaded Deck: Leader={leader_instance.master.name if leader_instance else 'None'}, Deck Size={len(deck_list)}")
        return leader_instance, deck_list
