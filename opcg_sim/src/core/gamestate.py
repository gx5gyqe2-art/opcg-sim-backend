from typing import List, Optional, Dict, Any, Tuple, Set
import random
import unicodedata
import re
import traceback
from ..models.models import CardInstance, CardMaster, DonInstance
from ..models.enums import CardType, Attribute, Color, Phase, Zone, TriggerType, ConditionType, CompareOperator, Player, ActionType
from ..models.effect_types import TargetQuery
from ..utils.logger_config import log_event

Card = CardInstance

def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)

class Player:
    def __init__(self, name: str, deck: List[Card], leader: Optional[Card] = None):
        self.name = name
        self.life: List[Card] = []
        self.hand: List[Card] = []
        self.field: List[Card] = []
        self.trash: List[Card] = []
        self.stage: Optional[Card] = None
        self.deck = deck
        self.don_deck: List[DonInstance] = [DonInstance(owner_id=name) for _ in range(10)]
        self.don_active: List[DonInstance] = []
        self.don_rested: List[DonInstance] = []
        self.don_attached_cards: List[DonInstance] = [] 
        self.leader: Optional[Card] = leader
        self.temp_zone: List[Card] = [] 

    def setup_game(self):
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
        return {
            "player_id": self.name,
            "name": self.name,
            "life_count": len(self.life),
            "hand_count": len(self.hand),
            "don_deck_count": len(self.don_deck),
            "don_active": [d.to_dict() for d in self.don_active],
            "don_rested": [d.to_dict() for d in self.don_rested],
            "leader": self.leader.to_dict() if self.leader else None,
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
        log_event("INFO", "game.start", "Game initialization started")
        
        self.p1.setup_game()
        self.p2.setup_game()
        if first_player:
            self.turn_player = first_player
            self.opponent = self.p2 if first_player == self.p1 else self.p1
        else:
            self.turn_player = self.p1
            self.opponent = self.p2
        
        log_event("INFO", "game.turn_player", f"First Player: {self.turn_player.name}", player=self.turn_player.name)
        self.turn_count = 1
        self.refresh_phase()

    def end_turn(self):
        self.phase = Phase.END
        log_event("INFO", "game.phase_end", f"Turn {self.turn_count} ending", player=self.turn_player.name)
        
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
        self.turn_player, self.opponent = self.opponent, self.turn_player
        self.turn_count += 1
        log_event("INFO", "game.turn_switch", f"Switched to Player: {self.turn_player.name}", player=self.turn_player.name)
        self.refresh_phase()

    def refresh_phase(self):
        self._reset_player_status(self.opponent)
        self.refresh_all(self.turn_player)
        self.draw_phase()

    def _reset_player_status(self, player: Player):
        all_units = [player.leader] + player.field
        if player.stage: all_units.append(player.stage)
        for card in all_units:
            if card: card.reset_turn_status()

    def refresh_all(self, player: Player):
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
        if self.turn_count > 1:
            self.draw_card(self.turn_player)
        self.don_phase()

    def don_phase(self):
        cards_to_add = 1 if self.turn_count == 1 else 2
        for _ in range(cards_to_add):
            if self.turn_player.don_deck:
                don = self.turn_player.don_deck.pop(0)
                self.turn_player.don_active.append(don)
        self.main_phase()

    def main_phase(self):
        self.phase = Phase.MAIN

    def draw_card(self, player: Player, count: int = 1):
        for _ in range(count):
            if player.deck:
                card = player.deck.pop(0)
                player.hand.append(card)
                log_event("INFO", "game.draw", f"Player {player.name} drew a card", player=player.name)
        if not player.deck and not self.winner:
            self.check_victory()

    def _find_card_location(self, card: Card) -> Tuple[Optional[Player], Optional[List[Any]]]:
        for p in [self.p1, self.p2]:
            zones = [p.hand, p.field, p.life, p.trash, p.deck, p.temp_zone]
            if p.leader == card: return p, None
            if p.stage == card: return p, None
            for zone in zones:
                if card in zone: return p, zone
        return None, None

    def move_card(self, card: Card, dest_zone: Zone, dest_player: Player, dest_position: str = "BOTTOM"):
        current_owner, current_list = self._find_card_location(card)
        if current_list is not None and card in current_list:
            current_list.remove(card)
        elif current_owner and current_owner.stage == card:
            current_owner.stage = None
        
        target_list = None
        if dest_zone == Zone.FIELD and card.master.type == CardType.STAGE:
            if dest_player.stage is not None:
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

    def pay_cost(self, player: Player, cost: int):
        log_event("DEBUG", "game.pay_cost_pre", f"Active Don: {len(player.don_active)}, Cost required: {cost}", player=player.name)
        if len(player.don_active) < cost:
            raise ValueError("ドン!!が不足しています。")
        for _ in range(cost):
            don = player.don_active.pop(0)
            player.don_rested.append(don)

    def resolve_attack(self, attacker: Card, target: Card):
        attacker_owner, _ = self._find_card_location(attacker)
        target_owner, _ = self._find_card_location(target)
        
        is_my_turn = (attacker_owner == self.turn_player)
        is_target_turn = (target_owner == self.turn_player)
        
        attacker_pwr = attacker.get_power(is_my_turn)
        target_pwr = target.get_power(is_target_turn)
        
        log_event("DEBUG", "game.resolve_attack_pre", f"Attacker: {attacker.master.name}({attacker_pwr}) vs Target: {target.master.name}({target_pwr})", player=attacker_owner.name if attacker_owner else "system")
        
        attacker.is_rest = True
        
        if target == target_owner.leader:
            if target_owner.life:
                life_card = target_owner.life.pop(0)
                self.move_card(life_card, Zone.HAND, target_owner)
            else:
                self.winner = attacker_owner.name if attacker_owner else None
        else:
            if attacker_pwr >= target_pwr:
                self.move_card(target, Zone.TRASH, target_owner)
        
        self.check_victory()

    def check_victory(self):
        if not self.p1.deck:
            self.winner = self.p2.name
        elif not self.p2.deck:
            self.winner = self.p1.name

    def play_card_action(self, player: Player, card: Card):
        if card not in player.hand: return
        
        log_event("INFO", "game.play_card", f"Playing card: {card.master.name}", player=player.name, payload={"card_uuid": card.uuid})
        
        if card.master.type == CardType.EVENT:
            for ability in card.master.abilities:
                if ability.trigger in [TriggerType.ON_PLAY, TriggerType.ACTIVATE_MAIN]:
                    self.resolve_ability(player, ability, source_card=card)
            self.move_card(card, Zone.TRASH, player)
        else:
            self.move_card(card, Zone.FIELD, player)
            card.attached_don = 0
            card.is_newly_played = True
            if not card.ability_disabled:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_PLAY:
                        self.resolve_ability(player, ability, source_card=card)

    def resolve_ability(self, player: Player, ability: Any, source_card: Card):
        if source_card.negated or source_card.ability_disabled: return
        for action in ability.actions:
            self._perform_logic(player, action, source_card)

    def _perform_logic(self, player: Player, action: Any, source_card: Card):
        log_event("INFO", "game.effect", f"Resolving action {action.type} for {source_card.master.name}", player=player.name)
        from .effects.resolver import execute_action
        execute_action(self, player, action, source_card)
