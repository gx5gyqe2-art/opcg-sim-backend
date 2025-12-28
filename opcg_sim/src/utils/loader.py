import json
import unicodedata
import re
from typing import List, Dict, Any, Optional, Tuple
from ..models.models import CardMaster, CardInstance
from ..core.effects.parser import Effect
from ..models.effect_types import Ability
from ..models.enums import CardType, Attribute, Color, TriggerType
from ..utils.logger_config import log_event

def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)

class RawDataLoader:
    @staticmethod
    def load_json(file_path: str) -> Any:
        log_event(level_key="DEBUG", action="loader.load_json", msg=f"Loading JSON from {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            log_event(level_key="ERROR", action="loader.file_not_found", msg=f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            log_event(level_key="ERROR", action="loader.json_decode_error", msg=str(e))
            return []

class DataCleaner:
    @staticmethod
    def normalize_text(text: Any) -> str:
        if text is None:
            return ""
        return unicodedata.normalize('NFKC', str(text)).strip()

    @staticmethod
    def parse_int(value: Any, default: int = 0) -> int:
        if isinstance(value, int):
            return value
        
        s_val = DataCleaner.normalize_text(value)
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
        return [t.strip() for t in s_val.split('/') if t.strip()]

    @staticmethod
    def parse_abilities(text: str, is_trigger: bool = False) -> List[Ability]:
        s_text = DataCleaner.normalize_text(text)
        if not s_text or s_text in [_nfc("なし"), "None", ""]:
            return []
        
        try:
            effect_parser = Effect(s_text)
            abilities = effect_parser.abilities
            if is_trigger:
                for ability in abilities:
                    ability.trigger = TriggerType.TRIGGER
            return abilities
        except Exception as e:
            log_event(level_key="DEBUG", action="loader.parse_abilities_error", msg=f"Error parsing abilities text: '{s_text[:20]}...' -> {e}")
            return []

    @staticmethod
    def map_color(value: str) -> Color:
        clean_str = DataCleaner.normalize_text(value)
        for c in Color:
            if DataCleaner.normalize_text(str(c.value)) in clean_str or \
               DataCleaner.normalize_text(c.name) in clean_str.upper():
                return c
        return Color.UNKNOWN

    @staticmethod
    def map_card_type(type_str: str) -> CardType:
        clean_str = DataCleaner.normalize_text(type_str)
        for t in CardType:
            if DataCleaner.normalize_text(str(t.value)) in clean_str or \
               DataCleaner.normalize_text(t.name) in clean_str.upper():
                return t
        return CardType.UNKNOWN

    @staticmethod
    def map_attribute(attr_str: str) -> Attribute:
        clean_str = DataCleaner.normalize_text(attr_str)
        for a in Attribute:
            if DataCleaner.normalize_text(str(a.value)) in clean_str or \
               DataCleaner.normalize_text(a.name) in clean_str.upper():
                return a
        return Attribute.NONE

class CardLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.cards: Dict[str, CardMaster] = {}
        self.raw_db: List[Dict[str, Any]] = []

    def load(self) -> None:
        data = RawDataLoader.load_json(self.json_path)
        self.raw_db = data if isinstance(data, list) else []
        log_event(level_key="INFO", action="loader.db_initialized", msg=f"Database initialized with {len(self.raw_db)} entries.")

    def get_card(self, card_id: str) -> Optional[CardMaster]:
        if card_id in self.cards:
            return self.cards[card_id]

        target_raw = None
        for item in self.raw_db:
            normalized_item = {DataCleaner.normalize_text(k): v for k, v in item.items()}
            item_id = DataCleaner.normalize_text(normalized_item.get("number", normalized_item.get("Number", "")))
            if item_id == card_id:
                target_raw = normalized_item
                break

        if not target_raw:
            log_event(level_key="ERROR", action="loader.card_not_found", msg=f"Card ID not found in database: {card_id}")
            return None

        master = self._create_card_master(target_raw)
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

        card_id = DataCleaner.normalize_text(get_val(["number", "Number", _nfc("品番"), _nfc("型番"), "id"], "N/A"))
        if not card_id or card_id == "N/A" or "dummy" in card_id.lower():
            return None

        name = DataCleaner.normalize_text(get_val(["name", "Name", _nfc("名前"), _nfc("カード名")]))
        type_val = get_val([_nfc("種類"), "Type", "type"])
        c_type = DataCleaner.map_card_type(type_val) if type_val else CardType.UNKNOWN
        attr_val = get_val([_nfc("属性"), "Attribute", "attribute"])
        attribute = DataCleaner.map_attribute(attr_val) if attr_val else Attribute.NONE
        color_val = get_val([_nfc("色"), "Color", "color"])
        color = DataCleaner.map_color(color_val) if color_val else Color.UNKNOWN
        cost = DataCleaner.parse_int(get_val([_nfc("コスト"), "Cost", "cost"]))
        power = DataCleaner.parse_int(get_val([_nfc("パワー"), "Power", "power"]))
        counter = DataCleaner.parse_int(get_val([_nfc("カウンター"), "Counter", "counter"]))
        life = DataCleaner.parse_int(get_val([_nfc("ライフ"), "Life", "life"]))
        traits = DataCleaner.parse_traits(get_val([_nfc("特徴"), "Traits", "traits"]))
        effect_text = DataCleaner.normalize_text(get_val([_nfc("効果(テキスト)"), _nfc("テキスト"), "Text", "text"]))
        trigger_text = DataCleaner.normalize_text(get_val([_nfc("効果(トリガー)"), _nfc("トリガー"), "Trigger", "trigger"]))
        main_abilities = DataCleaner.parse_abilities(effect_text, is_trigger=False)
        trigger_abilities = DataCleaner.parse_abilities(trigger_text, is_trigger=True)
        combined_abilities = tuple(main_abilities + trigger_abilities)

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
    def __init__(self, card_loader: CardLoader):
        self.card_loader = card_loader

    def load_deck(self, file_path: str, owner_id: str) -> Tuple[Optional[CardInstance], List[CardInstance]]:
        data = RawDataLoader.load_json(file_path)
        deck_data = {}
        if isinstance(data, list) and len(data) > 0:
            deck_data = data[0]
        elif isinstance(data, dict):
            deck_data = data
        else:
            log_event(level_key="ERROR", action="loader.invalid_deck", msg=f"Invalid deck format in {file_path}", player="system")
            return None, []

        leader_instance = None
        if "leader" in deck_data:
            leader_id = deck_data["leader"].get("number")
            if leader_id:
                leader_master = self.card_loader.get_card(leader_id)
                if leader_master:
                    leader_instance = CardInstance(leader_master, owner_id)

        deck_list: List[CardInstance] = []
        if "cards" in deck_data:
            for item in deck_data["cards"]:
                card_id = item.get("number")
                count = item.get("count", 0)
                master = self.card_loader.get_card(card_id)
                if master:
                    for _ in range(count):
                        deck_list.append(CardInstance(master, owner_id))

        log_event(level_key="INFO", action="loader.deck_load_success", msg=f"Loaded Deck: Leader={leader_instance.master.name if leader_instance else 'None'}, Deck Size={len(deck_list)}", player=owner_id)
        return leader_instance, deck_list
