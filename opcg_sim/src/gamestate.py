from typing import List, Optional, Dict, Any, Tuple, Set
import random
import unicodedata
import re
import logging

from .models import CardInstance, CardMaster, DonInstance
from .enums import CardType, Attribute, Color, Phase
from .effects import (
    ActionType, 
    Zone, 
    TargetQuery, 
    TriggerType, 
    ConditionType, 
    CompareOperator, 
    Player as EffectPlayer
)
from .logger_config import log_object

logger = logging.getLogger("opcg_sim")
Card = CardInstance

def _nfc(text: str) -> str:
    """文字列をNFC正規化するヘルパー関数"""
    return unicodedata.normalize('NFC', text)

class Player:
    def __init__(self, name: str, deck: List[Card], leader: Optional[Card] = None):
        self.name = name
        self.life: List[Card] = []
        self.hand: List[Card] = []
        self.field: List[Card] = []
        self.trash: List[Card] = []
        self.stage: Optional[Card] = None # v1.4: ステージ専用枠
        self.deck = deck
        self.don_deck: List[DonInstance] = [DonInstance(owner_id=name) for _ in range(10)]
        self.don_active: List[DonInstance] = []
        self.don_rested: List[DonInstance] = []
        self.don_attached_cards: List[DonInstance] = [] 
        self.leader: Optional[Card] = leader
        self.temp_zone: List[Card] = [] 

    def setup_game(self):
        """初期手札とライフのセットアップ"""
        random.shuffle(self.deck)
        if self.leader:
            life_count = self.leader.master.life
            for _ in range(life_count):
                if self.deck:
                    self.life.append(self.deck.pop(0))
        for _ in range(5):
            if self.deck:
                self.hand.append(self.deck.pop(0))

    def to_dict(self):
        """
        API v1.4 適合: フロントエンドの gameState.ts 構造に準拠
        各ゾーンを 'zones' オブジェクト内に配置し、ドンの状態をPlayer直下に配置します。
        """
        return {
            "player_id": self.name,
            "name": self.name,
            "life_count": len(self.life),
            "hand_count": len(self.hand),
            "don_deck_count": len(self.don_deck),
            # ドン配置は v1.4 仕様に基づき zones の外
            "don_active": [d.to_dict() for d in self.don_active],
            "don_rested": [d.to_dict() for d in self.don_rested],
            "leader": self.leader.to_dict() if self.leader else None,
            # ゾーン構造のネスト化 (RealGame.tsx の参照に適合)
            "zones": {
                "field": [c.to_dict() for c in self.field],
                "hand": [c.to_dict() for c in self.hand],
                "life": [c.to_dict() for c in self.life],
                "trash": [c.to_dict() for c in self.trash],
                "stage": self.stage.to_dict() if self.stage else None
            }
        }

class GameManager:
    def __init__(self, player1: Player, player2: Player):
        self.p1 = player1
        self.p2 = player2
        self.turn_player = self.p1
        self.opponent = self.p2
        self.turn_count = 0
        self.phase = Phase.SETUP
        self.winner: Optional[str] = None

    def start_game(self, first_player: Optional[Player] = None):
        """ゲーム開始処理"""
        self.log("=== Game Start ===")
        self.p1.setup_game()
        self.p2.setup_game()
        if first_player:
            self.turn_player = first_player
            self.opponent = self.p2 if first_player == self.p1 else self.p1
        else:
            self.turn_player = self.p1
            self.opponent = self.p2
        self.log(f"First Player: {self.turn_player.name}")
        self.turn_count = 1
        self.refresh_phase()

    def log(self, message: str):
        logger.info(f"[GM] {message}")

    def end_turn(self):
        """ターン終了ステップの処理"""
        self.phase = Phase.END
        all_units = [self.turn_player.leader] + self.turn_player.field
        if self.turn_player.stage:
            all_units.append(self.turn_player.stage)
        
        for card in all_units:
            if card and card.master.abilities:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.TURN_END:
                        self.resolve_ability(self.turn_player, ability, source_card=card)
        self.switch_turn()

    def switch_turn(self):
        """ターンプレイヤーの交代"""
        self.turn_player, self.opponent = self.opponent, self.turn_player
        self.turn_count += 1
        self.refresh_phase()

    def refresh_phase(self):
        """リフレッシュフェーズの処理"""
        self._reset_player_status(self.opponent)
        self.refresh_all(self.turn_player)
        self.draw_phase()

    def _reset_player_status(self, player: Player):
        """全ユニットのバフ等のリセット"""
        all_units = [player.leader] + player.field
        if player.stage: all_units.append(player.stage)
        for card in all_units:
            if card: card.reset_turn_status()

    def refresh_all(self, player: Player):
        """自軍ユニットをアクティブにし、付与ドンをコストエリアに戻す"""
        all_units = [player.leader] + player.field
        if player.stage: all_units.append(player.stage)
        for card in all_units:
            if card:
                is_frozen = "FREEZE" in card.flags
                card.reset_turn_status()
                if not is_frozen:
                    card.is_rest = False
        
        player.don_active.extend(player.don_rested)
        player.don_rested = []
        for don in player.don_attached_cards:
            don.attached_to = None
            player.don_active.append(don)
        player.don_attached_cards = [] 

    def draw_phase(self):
        """ドローフェーズ"""
        if self.turn_count > 1:
            self.draw_card(self.turn_player)
        self.don_phase()

    def don_phase(self):
        """ドン!!フェーズ"""
        cards_to_add = 1 if self.turn_count == 1 else 2
        for _ in range(cards_to_add):
            if self.turn_player.don_deck:
                don = self.turn_player.don_deck.pop(0)
                self.turn_player.don_active.append(don)
        self.main_phase()

    def main_phase(self):
        """メインフェーズの開始"""
        self.phase = Phase.MAIN

    def draw_card(self, player: Player, count: int = 1):
        """山札からドロー"""
        for _ in range(count):
            if player.deck:
                card = player.deck.pop(0)
                player.hand.append(card)

    def _find_card_location(self, card: Card) -> Tuple[Optional[Player], Optional[List[Any]]]:
        """カードの現在の所在を検索"""
        for p in [self.p1, self.p2]:
            zones = [p.hand, p.field, p.life, p.trash, p.deck, p.temp_zone]
            if p.leader == card: return p, None
            if p.stage == card: return p, None
            for zone in zones:
                if card in zone: return p, zone
        return None, None

    def move_card(self, card: Card, dest_zone: Zone, dest_player: Player, dest_position: str = "BOTTOM"):
        """
        v1.4: ゾーン間移動ロジック
        ステージカードがフィールドに送られる際、自動的にPlayer.stageへ振り分けます。
        """
        current_owner, current_list = self._find_card_location(card)
        if current_list is not None and card in current_list:
            current_list.remove(card)
        elif current_owner and current_owner.stage == card:
            current_owner.stage = None
        
        target_list = None
        # STAGEの扱い: field指定でも自動的にPlayer.stageへ配置
        if dest_zone == Zone.FIELD and card.master.type == CardType.STAGE:
            if dest_player.stage is not None:
                # 既存のステージをトラッシュへ
                self.move_card(dest_player.stage, Zone.TRASH, dest_player)
            dest_player.stage = card
        elif dest_zone == Zone.HAND: target_list = dest_player.hand
        elif dest_zone == Zone.FIELD: target_list = dest_player.field
        elif dest_zone == Zone.TRASH: target_list = dest_player.trash
        elif dest_zone == Zone.LIFE: target_list = dest_player.life
        elif dest_zone == Zone.DECK: target_list = dest_player.deck
        
        if target_list is not None:
            if dest_position == "TOP": target_list.insert(0, card)
            else: target_list.append(card)

    def play_card_action(self, player: Player, card: Card):
        """
        v1.4: カードプレイアクション
        イベントカードの場合は効果解決後にトラッシュへ、それ以外はフィールドへ移動。
        """
        if card not in player.hand: return
        
        if card.master.type == CardType.EVENT:
            for ability in card.master.abilities:
                if ability.trigger in [TriggerType.ON_PLAY, TriggerType.ACTIVATE_MAIN]:
                    self.resolve_ability(player, ability, source_card=card)
            # イベント解決後にトラッシュへ移動
            self.move_card(card, Zone.TRASH, player)
        else:
            # 既に move_card 内でステージ判定を行っているため、Zone.FIELD 指定で適切に処理
            self.move_card(card, Zone.FIELD, player)
            card.attached_don = 0 # プレイ時に付与ドンを0に初期化
            card.is_newly_played = True
            if not card.ability_disabled:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_PLAY:
                        self.resolve_ability(player, ability, source_card=card)

    def resolve_ability(self, player: Player, ability: Any, source_card: Card):
        """効果解決の基盤メソッド"""
        if source_card.negated or source_card.ability_disabled: return
        # _perform_logic は既存のエフェクト解決システム（effects.py連携）を使用
        for action in ability.actions:
            self._perform_logic(player, action, source_card)

    def _perform_logic(self, player: Player, action: Any, source_card: Card):
        """個別のエフェクトアクションを実行（スタブ：詳細ロジックは effects.py 参照）"""
        self.log(f"Resolving action {action.type} for {source_card.name}")
        # ここに KO, DRAW, BP_BUFF 等の具体的な処理を実装
        pass
