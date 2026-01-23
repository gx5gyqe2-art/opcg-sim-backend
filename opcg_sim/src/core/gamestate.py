from typing import List, Optional, Any, Tuple, Dict, Set
import random
import unicodedata
import re
import traceback
import uuid
from ..models.models import CardInstance, CardMaster, DonInstance, CONST
from ..models.enums import CardType, Attribute, Color, Phase, Zone, TriggerType, ConditionType, CompareOperator, ActionType, PendingMessage
from ..models.effect_types import TargetQuery, Ability, GameAction, ValueSource
from ..utils.logger_config import log_event
from .effects.resolver import EffectResolver


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

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def place_life(self):
        if self.leader:
            life_count = self.leader.master.life
            for _ in range(life_count):
                if self.deck:
                    self.life.append(self.deck.pop(0))

    def draw_initial_hand(self):
        for _ in range(5):
            if self.deck:
                self.hand.append(self.deck.pop(0))

    def to_dict(self, is_owner: bool = True):
        player_props = CONST.get('PLAYER_PROPERTIES', {})
        leader_dict = self.leader.to_dict() if self.leader else None
        if leader_dict:
            leader_dict["is_face_up"] = True
        stage_dict = self.stage.to_dict() if self.stage else None
        if stage_dict:
            stage_dict["is_face_up"] = True
        return {
            "player_id": self.name,
            "name": self.name,
            player_props.get("LIFE_COUNT", "life_count"): len(self.life),
            "hand_count": len(self.hand),
            player_props.get("DON_DECK_COUNT", "don_deck_count"): len(self.don_deck),
            player_props.get("DON_ACTIVE", "don_active"): [d.to_dict() for d in self.don_active],
            player_props.get("DON_RESTED", "don_rested"): [d.to_dict() for d in self.don_rested],
            "leader": leader_dict,
            "stage": stage_dict,
            "zones": {
                "field": [self._format_card(c, True) for c in self.field],
                "hand": [self._format_card(c, is_owner) for c in self.hand],
                "life": [self._format_card(c, False) for c in self.life],
                "trash": [self._format_card(c, True) for c in self.trash],
                "stage": stage_dict
            }
        }

    def _format_card(self, card: Card, face_up: bool) -> dict:
        d = card.to_dict()
        d["is_face_up"] = face_up
        return d

class GameManager:
    def __init__(self, player1: Player, player2: Player):
        self.p1 = player1
        self.p2 = player2
        self.turn_player = self.p1
        self.opponent = self.p2
        self.turn_count = 0
        self.phase = Phase.SETUP
        self.winner: Optional[str] = None
        self.active_battle: Optional[Dict[str, Any]] = None
        self.active_interaction: Optional[Dict[str, Any]] = None
        self.setup_phase_pending = False

    def _find_card_by_uuid(self, uuid: str) -> Optional[CardInstance]:
        all_players = [self.p1, self.p2]
        for p in all_players:
            candidates = []
            if p.leader: candidates.append(p.leader)
            if p.stage: candidates.append(p.stage)
            candidates.extend(p.hand)
            candidates.extend(p.field)
            candidates.extend(p.trash)
            candidates.extend(p.life)
            candidates.extend(p.deck)
            candidates.extend(p.temp_zone)
            for c in candidates:
                if c.uuid == uuid:
                    return c
        return None

    def get_pending_request(self) -> Optional[Dict[str, Any]]:
        pending_props = CONST.get('PENDING_REQUEST_PROPERTIES', {})
        battle_actions = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
        KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
        KEY_ACTION = pending_props.get('ACTION', 'action')
        KEY_MSG = pending_props.get('MESSAGE', 'message')
        KEY_UUIDS = pending_props.get('SELECTABLE_UUIDS', 'selectable_uuids')
        KEY_SKIP = pending_props.get('CAN_SKIP', 'can_skip')
        KEY_CANDIDATES = pending_props.get('CANDIDATES', 'candidates')
        KEY_CONSTRAINTS = pending_props.get('CONSTRAINTS', 'constraints')
        KEY_OPTIONS = pending_props.get('OPTIONS', 'options')
        
        if self.active_interaction:
            action_type = self.active_interaction.get("action_type")
            fe_action = "SEARCH_AND_SELECT" if action_type == "SELECT_TARGET" else action_type
            
            candidates = self.active_interaction.get("candidates", [])
            candidate_dicts = [c.to_dict() for c in candidates] if candidates else []
            candidate_uuids = [c.uuid for c in candidates] if candidates else []
            
            req = {
                KEY_PID: self.active_interaction.get("player_id"),
                KEY_ACTION: fe_action,
                KEY_MSG: self.active_interaction.get("message", "選択してください"),
                KEY_UUIDS: self.active_interaction.get("selectable_uuids", candidate_uuids),
                KEY_SKIP: self.active_interaction.get("can_skip", False),
                KEY_CANDIDATES: candidate_dicts,
                KEY_CONSTRAINTS: self.active_interaction.get("constraints"),
                "options": self.active_interaction.get("options"), 
                "request_id": str(uuid.uuid4())
            }
            return req

        if not self.active_battle and self.phase in [Phase.BLOCK_STEP, Phase.BATTLE_COUNTER]:
            log_event("ERROR", "game.pending_request_error", f"Active battle missing in phase: {self.phase.name}")
            self.phase = Phase.MAIN
            
        request = None
        ACT_BLOCKER = battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
        ACT_COUNTER = battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER')
        
        if self.phase == Phase.BLOCK_STEP and self.active_battle:
            target_owner = self.active_battle["target_owner"]
            blockers = [c.uuid for c in target_owner.field if not c.is_rest and "ブロッカー" in c.current_keywords]
            request = {KEY_PID: target_owner.name, KEY_ACTION: ACT_BLOCKER, KEY_MSG: PendingMessage.SELECT_BLOCKER.value, KEY_UUIDS: blockers, KEY_SKIP: True, "request_id": str(uuid.uuid4())}
        elif self.phase == Phase.BATTLE_COUNTER and self.active_battle:
            target_owner = self.active_battle["target_owner"]
            counters = [c.uuid for c in target_owner.hand if (c.master.counter and c.master.counter > 0) or (c.master.type == CardType.EVENT and any(abil.trigger == TriggerType.COUNTER for abil in c.master.abilities))]
            request = {KEY_PID: target_owner.name, KEY_ACTION: ACT_COUNTER, KEY_MSG: PendingMessage.SELECT_COUNTER.value, KEY_UUIDS: counters, KEY_SKIP: True, "request_id": str(uuid.uuid4())}
        elif self.phase == Phase.MAIN:
            selectable = [c.uuid for c in self.turn_player.hand]
            selectable += [c.uuid for c in self.turn_player.field if not c.is_rest]
            if self.turn_player.leader and not self.turn_player.leader.is_rest:
                selectable.append(self.turn_player.leader.uuid)
            request = {KEY_PID: self.turn_player.name, KEY_ACTION: "MAIN_ACTION", KEY_MSG: PendingMessage.MAIN_ACTION.value, KEY_UUIDS: selectable, KEY_SKIP: True, "request_id": str(uuid.uuid4())}
        return request

    def resolve_interaction(self, player: Player, payload: Dict[str, Any]):
        if not self.active_interaction:
            log_event("WARNING", "game.resolve_interaction", "No active interaction found", player=player.name)
            return
            
        continuation = self.active_interaction.get("continuation")
        if not continuation:
            self.active_interaction = None
            return
            
        action_type = self.active_interaction.get("action_type")
        source_uuid = continuation["source_card_uuid"]
        source_card = self._find_card_by_uuid(source_uuid)
        if not source_card:
            log_event("ERROR", "game.resume_fail", f"Source card {source_uuid} not found")
            self.active_interaction = None
            return

        resolver = EffectResolver(self)
        
        if action_type == "SELECT_TARGET":
            log_event("INFO", "game.resume_target", f"Resuming target selection for {source_card.master.name}", player=player.name)
            selected_uuids = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
            
            selected_cards = []
            candidates = self.active_interaction.get("candidates", [])
            for uid in selected_uuids:
                card = next((c for c in candidates if c.uuid == uid), None)
                if card: selected_cards.append(card)
            
            query = continuation.get("query")
            if query and getattr(query, 'save_id', None):
                 continuation["effect_context"]["saved_targets"][query.save_id] = selected_cards
            
            self.active_interaction = None
            resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), continuation.get("effect_context", {}))
            
        elif action_type == "CHOICE":
            log_event("INFO", "game.resume_choice", f"Resuming choice for {source_card.master.name}", player=player.name)
            selected_index = payload.get("index", payload.get("selected_option_index", 0))
            
            resolver.resume_choice(player, source_card, selected_index, continuation.get("execution_stack", []), continuation.get("effect_context", {}))

        if not self.active_interaction and self.setup_phase_pending:
            self.finish_setup()
            self.setup_phase_pending = False
            log_event("INFO", "game.turn_player", f"First Player: {self.turn_player.name}", player=self.turn_player.name)
            self.turn_count = 1
            self.refresh_phase()

    def _validate_action(self, player: Player, action_type: str):
        pending = self.get_pending_request()
        if not pending: raise ValueError("現在実行可能なアクションはありません。")
        
        pending_props = CONST.get('PENDING_REQUEST_PROPERTIES', {})
        KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
        KEY_ACTION = pending_props.get('ACTION', 'action')
        
        battle_actions = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
        ACT_BLOCKER = battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
        ACT_COUNTER = battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER')
        ACT_PASS = battle_actions.get('PASS', 'PASS')
        RESOLVE_SELECTION = CONST.get('c_to_s_interface', {}).get('GAME_ACTIONS', {}).get('TYPES', {}).get('RESOLVE_EFFECT_SELECTION', 'RESOLVE_EFFECT_SELECTION')
        
        if pending[KEY_PID] != player.name: raise ValueError(f"現在は {pending[KEY_PID]} のターン/フェイズです。")
        
        expected_action = pending[KEY_ACTION]
        if expected_action in [ACT_COUNTER, ACT_BLOCKER] and action_type == ACT_PASS: return True
        if self.active_interaction and action_type == RESOLVE_SELECTION: return True
        
        if expected_action != action_type:
            raise ValueError(f"不適切なアクションです。期待されているアクション: {expected_action}")
        return True

    def start_game(self, first_player: Optional[Player] = None):
        log_event("INFO", "game.start", "Game initialization started")
        
        self.p1.shuffle_deck()
        self.p2.shuffle_deck()
        
        for p in [self.p1, self.p2]:
            if p.leader:
                for ability in p.leader.master.abilities:
                    if ability.trigger == TriggerType.GAME_START:
                        log_event("INFO", "game.trigger_gamestart", f"Resolving GAME_START for {p.leader.master.name}", player=p.name)
                        self.resolve_ability(p, ability, source_card=p.leader)
                        
                        if self.active_interaction:
                            self.setup_phase_pending = True
                            if first_player: self.turn_player = first_player; self.opponent = self.p2 if first_player == self.p1 else self.p1
                            else: self.turn_player = self.p1; self.opponent = self.p2
                            return

        self.finish_setup()
        
        if first_player: self.turn_player = first_player; self.opponent = self.p2 if first_player == self.p1 else self.p1
        else: self.turn_player = self.p1; self.opponent = self.p2
        log_event("INFO", "game.turn_player", f"First Player: {self.turn_player.name}", player=self.turn_player.name)
        self.turn_count = 1; self.refresh_phase()

    def finish_setup(self):
        log_event("INFO", "game.setup_finish", "Finishing setup (Life/Hand)", player="system")
        self.p1.place_life()
        self.p1.draw_initial_hand()
        self.p2.place_life()
        self.p2.draw_initial_hand()

    def end_turn(self):
        self._validate_action(self.turn_player, "MAIN_ACTION")
        self.phase = Phase.END
        log_event("INFO", "game.phase_end", f"Turn {self.turn_count} ending", player=self.turn_player.name)
        all_units = [self.turn_player.leader] + self.turn_player.field
        if self.turn_player.stage: all_units.append(self.turn_player.stage)
        for card in all_units:
            if card and card.master.abilities:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.TURN_END: self.resolve_ability(self.turn_player, ability, source_card=card)
        self.switch_turn()

    def switch_turn(self):
        self.turn_player, self.opponent = self.opponent, self.turn_player
        self.turn_count += 1
        log_event("INFO", "game.turn_switch_end", f"After switch: turn_player={self.turn_player.name}, count={self.turn_count}", player="system")
        self.refresh_phase()

    def refresh_phase(self):
        self._reset_player_status(self.opponent); self.refresh_all(self.turn_player); self.draw_phase()

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
                if not is_frozen: card.is_rest = False
        
        for don in player.don_rested:
            don.is_rest = False
            player.don_active.append(don)
        player.don_rested = []
        
        for don in player.don_attached_cards:
            don.is_rest = False
            don.attached_to = None
            player.don_active.append(don)
        player.don_attached_cards = []

    def draw_phase(self):
        if self.turn_count > 1: self.draw_card(self.turn_player)
        self.don_phase()

    def don_phase(self):
        cards_to_add = 1 if self.turn_count == 1 else 2
        for _ in range(cards_to_add):
            if self.turn_player.don_deck:
                don = self.turn_player.don_deck.pop(0); self.turn_player.don_active.append(don)
        self.main_phase()

    def main_phase(self): 
        self.phase = Phase.MAIN
        self._apply_passive_effects(self.turn_player)

    def _apply_passive_effects(self, player: Player):
        affected_cards = [player.leader, player.stage] + player.field + player.hand
        for c in affected_cards:
            if c:
                c.cost_buff = 0
                c.base_power_override = None

        all_units = [player.leader] + player.field
        if player.stage: all_units.append(player.stage)
        for card in all_units:
            if not card or not card.master.abilities: continue
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.YOUR_TURN:
                    log_event("DEBUG", "game.passive_trigger", f"Applying passive effect: {card.master.name}", player=player.name)
                    self.resolve_ability(player, ability, source_card=card)

    def draw_card(self, player: Player, count: int = 1):
        for _ in range(count):
            if player.deck:
                card = player.deck.pop(0); player.hand.append(card)
                log_event("INFO", "game.draw", f"Player {player.name} drew a card", player=player.name)
        if not player.deck and not self.winner: self.check_victory()

    def _find_card_location(self, card: Card) -> Tuple[Optional[Player], Optional[List[Any]]]:
        for p in [self.p1, self.p2]:
            zones = [
                p.hand, p.field, p.life, p.trash, p.deck, p.temp_zone,
                p.don_active, p.don_rested, p.don_attached_cards
            ]
            if p.leader == card: return p, None
            if p.stage == card: return p, None
            for zone in zones:
                if card in zone: return p, zone
        return None, None

    def move_card(self, card: Card, dest_zone: Zone, dest_player: Player, dest_position: str = "BOTTOM"):
        current_owner, current_list = self._find_card_location(card)
        
        # 領域移動時はステータスをリセット（特にトラッシュ/手札へ戻る場合）
        if dest_zone in [Zone.TRASH, Zone.HAND]:
            card.reset_turn_status()
            
        # フィールドから離れる場合、付与されていたドン‼をレスト状態で持ち主に返す
        if current_owner and current_list is not None and current_list is current_owner.field:
            attached_dons = [d for d in current_owner.don_attached_cards if d.attached_to == card.uuid]
            for don in attached_dons:
                current_owner.don_attached_cards.remove(don)
                don.attached_to = None
                don.is_rest = True
                current_owner.don_rested.append(don)
            card.attached_don = 0

        if current_list is not None and card in current_list: current_list.remove(card)
        elif current_owner and current_owner.stage == card: current_owner.stage = None
        
        target_list = None
        if dest_zone == Zone.FIELD and card.master.type == CardType.STAGE:
            if dest_player.stage is not None: self.move_card(dest_player.stage, Zone.TRASH, dest_player)
            dest_player.stage = card
        elif dest_zone == Zone.HAND: target_list = dest_player.hand
        elif dest_zone == Zone.FIELD: target_list = dest_player.field
        elif dest_zone == Zone.TRASH: target_list = dest_player.trash
        elif dest_zone == Zone.LIFE: target_list = dest_player.life
        elif dest_zone == Zone.DECK: target_list = dest_player.deck
        elif dest_zone == Zone.TEMP: target_list = dest_player.temp_zone
        
        if target_list is not None:
            if dest_position == "TOP": target_list.insert(0, card)
            else: target_list.append(card)

    def pay_cost(self, player: Player, cost: int, don_list: Optional[List[DonInstance]] = None):
        if don_list is not None:
            if len(don_list) < cost: raise ValueError("指定されたドン!!の数が不足しています。")
            for don in don_list:
                if don in player.don_active: player.don_active.remove(don); player.don_rested.append(don); don.is_rest = True
                elif don in player.don_attached_cards: player.don_attached_cards.remove(don); player.don_rested.append(don); don.is_rest = True; don.attached_to = None
        else:
            if len(player.don_active) < cost: raise ValueError("ドン!!が不足しています。")
            for _ in range(cost): don = player.don_active.pop(0); player.don_rested.append(don); don.is_rest = True

    def has_blocker(self, player: Player) -> bool:
        for card in player.field:
            if not card.is_rest and "ブロッカー" in card.current_keywords: return True
        return False

    def declare_attack(self, attacker: Card, target: Card):
        attacker_owner, _ = self._find_card_location(attacker)
        target_owner, _ = self._find_card_location(target)
        self._validate_action(attacker_owner, "MAIN_ACTION")
        if "ATTACK_DISABLE" in attacker.flags: raise ValueError("このカードは効果によりアタックできません。")
        if attacker.is_rest: raise ValueError("アタックするカードはアクティブ状態でなければなりません。")
        if target.master.type == CardType.CHARACTER and not target.is_rest: raise ValueError("レスト状態のキャラクターのみ攻撃可能です。")
        log_event("INFO", "game.attack_declare", f"{attacker.master.name} is attacking {target.master.name}", player=attacker_owner.name)
        attacker.is_rest = True
        self.active_battle = {"attacker": attacker, "target": target, "attacker_owner": attacker_owner, "target_owner": target_owner, "counter_buff": 0}
        
        if attacker.master.abilities:
            for ability in attacker.master.abilities:
                if ability.trigger == TriggerType.ON_ATTACK:
                    self.resolve_ability(attacker_owner, ability, source_card=attacker)

        if self.has_blocker(target_owner): self.phase = Phase.BLOCK_STEP; log_event("INFO", "game.phase_transition", f"Blockers detected. Moving to {self.phase.name}", player=target_owner.name)
        else: self.phase = Phase.BATTLE_COUNTER; log_event("INFO", "game.phase_transition", f"No blockers. Moving to {self.phase.name}", player=target_owner.name)

    def handle_block(self, blocker: Optional[Card] = None):
        if not self.active_battle: return
        target_owner = self.active_battle["target_owner"]; self._validate_action(target_owner, "SELECT_BLOCKER")
        if blocker: log_event("INFO", "game.block_execute", f"{blocker.master.name} blocks the attack", player=target_owner.name); blocker.is_rest = True; self.active_battle["target"] = blocker
        else: log_event("INFO", "game.block_skip", "No block declared", player=target_owner.name)
        self.phase = Phase.BATTLE_COUNTER; log_event("INFO", "game.phase_transition", f"Moving to {self.phase.name}", player=target_owner.name)

    def apply_counter(self, player: Player, counter_card: Optional[Card] = None, don_list: Optional[List[DonInstance]] = None):
        if not self.active_battle: return
        if counter_card is None: log_event("INFO", "game.counter_pass", "Player passed counter step", player=player.name); self.resolve_attack(); return
        self._validate_action(player, "SELECT_COUNTER")
        log_event("INFO", "game.counter_play", f"Playing counter card: {counter_card.master.name}", player=player.name)
        if counter_card.master.type == CardType.EVENT:
            self.pay_cost(player, counter_card.master.cost, don_list)
            for ability in counter_card.master.abilities:
                if ability.trigger == TriggerType.COUNTER: self.resolve_ability(player, ability, source_card=counter_card)
            self.move_card(counter_card, Zone.TRASH, player)
        else:
            counter_value = counter_card.master.counter or 0; self.active_battle["counter_buff"] += counter_value
            log_event("INFO", "game.counter_apply", f"Added {counter_value} power to target", player=player.name); self.move_card(counter_card, Zone.TRASH, player)

    def resolve_attack(self):
        if not self.active_battle: return
        attacker = self.active_battle["attacker"]; target = self.active_battle["target"]
        attacker_owner = self.active_battle["attacker_owner"]; target_owner = self.active_battle["target_owner"]
        counter_buff = self.active_battle.get("counter_buff", 0)
        is_my_turn = (attacker_owner == self.turn_player); is_target_turn = (target_owner == self.turn_player)
        attacker_pwr = attacker.get_power(is_my_turn); target_pwr = target.get_power(is_target_turn) + counter_buff
        if target == target_owner.leader:
            if attacker_pwr >= target_pwr:
                damage_amount = 2 if "ダブルアタック" in attacker.current_keywords else 1; is_banish = "バニッシュ" in attacker.current_keywords
                log_event("INFO", "game.damage_step", f"Dealing {damage_amount} damage (Banish: {is_banish})", player=attacker_owner.name)
                for _ in range(damage_amount):
                    if target_owner.life:
                        life_card = target_owner.life.pop(0); dest_zone = Zone.TRASH if is_banish else Zone.HAND
                        self.move_card(life_card, dest_zone, target_owner)
                        log_event("INFO", "game.damage_life", f"{target_owner.name} takes damage to {dest_zone.name}", player=target_owner.name)
                    else: self.winner = attacker_owner.name; log_event("INFO", "game.victory", f"{attacker_owner.name} wins the game", player=attacker_owner.name); break
        else:
            if attacker_pwr >= target_pwr: self.move_card(target, Zone.TRASH, target_owner); log_event("INFO", "game.unit_ko", f"{target.master.name} was KO'd", player=target_owner.name)
        target.reset_turn_status(); self.active_battle = None; self.phase = Phase.MAIN; self.check_victory()

    def check_victory(self):
        if not self.p1.deck: self.winner = self.p2.name
        elif not self.p2.deck: self.winner = self.p1.name

    def play_card_action(self, player: Player, card: Card):
        if card not in player.hand: return
        self._validate_action(player, "MAIN_ACTION")
        log_event("INFO", "game.play_card", f"Playing card: {card.master.name}", player=player.name, payload={"card_uuid": card.uuid})
        if card.master.type == CardType.EVENT:
            for ability in card.master.abilities:
                if ability.trigger in [TriggerType.ON_PLAY, TriggerType.ACTIVATE_MAIN]:
                    self.resolve_ability(player, ability, source_card=card)
            self.move_card(card, Zone.TRASH, player)
        else:
            self.move_card(card, Zone.FIELD, player); card.attached_don = 0; card.is_newly_played = True
            if not card.ability_disabled:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_PLAY:
                        self.resolve_ability(player, ability, source_card=card)
            self._apply_passive_effects(player)

    def resolve_ability(self, player: Player, ability: Ability, source_card: CardInstance):
        if source_card.negated or source_card.ability_disabled: return
        resolver = EffectResolver(self); resolver.resolve_ability(player, ability, source_card)

    def apply_action_to_engine(self, player: Player, action: GameAction, targets: List[CardInstance], value: int) -> bool:
        if not action: return False
        act_name = action.type.name if hasattr(action.type, 'name') else str(action.type)
        log_event("INFO", "game.apply_action", f"Applying {act_name} to {len(targets)} targets", player=player.name)
        if act_name == "DRAW":
            self.draw_card(player, value); return True
        if act_name == "SHUFFLE":
            random.shuffle(player.deck)
            log_event("INFO", "game.action_shuffle", "Deck shuffled", player=player.name)
            return True
        if act_name == "LOOK":
            count = value
            deck = player.deck
            if len(deck) < count: count = len(deck)
            log_event("INFO", "game.action_look", f"Looking at {count} cards from DECK", player=player.name)
            for _ in range(count):
                card = deck.pop(0)
                player.temp_zone.append(card)
            return True
        for target in targets:
            owner, source_list = self._find_card_location(target)
            if not owner: continue
            if act_name == "KO":
                self.move_card(target, Zone.TRASH, owner); log_event("INFO", "game.action_ko", f"{target.master.name} was KO'd by effect", player=player.name); success = True
            elif act_name in ["DISCARD", "TRASH"]:
                self.move_card(target, Zone.TRASH, owner); success = True
            elif act_name in ["BOUNCE", "MOVE_TO_HAND"]:
                self.move_card(target, Zone.HAND, owner); success = True
            elif act_name == "MOVE":
                dest_zone = action.destination or Zone.TRASH; self.move_card(target, dest_zone, owner); success = True
            elif act_name == "BUFF":
                if action.status == "POWER_OVERRIDE":
                    target.base_power_override = value
                    log_event("INFO", "game.action_override", f"{target.master.name}'s power set to {value}", player=player.name)
                elif action.status == "COST_REDUCTION":
                    if hasattr(target, 'cost_buff'):
                        target.cost_buff += value
                        log_event("INFO", "game.action_cost_reduction", f"{target.master.name}'s cost changed by {value}", player=player.name)
                else:
                    if hasattr(target, 'power_buff'):
                        target.power_buff += value
                        log_event("INFO", "game.action_buff", f"{target.master.name} gained {value} power", player=player.name)
                success = True
            elif act_name == "REST":
                target.is_rest = True
                if isinstance(target, DonInstance) and source_list is not None:
                    if source_list is not owner.don_rested:
                        if target in source_list:
                            source_list.remove(target)
                            owner.don_rested.append(target)
                            if hasattr(target, 'attached_to'): target.attached_to = None
                success = True
            elif act_name == "PLAY_CARD":
                self.move_card(target, Zone.FIELD, owner); target.is_newly_played = True
                if not target.ability_disabled:
                    for ability in target.master.abilities:
                        if ability.trigger == TriggerType.ON_PLAY:
                            self.resolve_ability(owner, ability, source_card=target)
                self._apply_passive_effects(owner)
                success = True
            elif act_name == "DECK_BOTTOM":
                self.move_card(target, Zone.DECK, owner, dest_position="BOTTOM"); success = True
        return success

    def get_dynamic_value(self, player: Player, val_source: ValueSource, targets: List[CardInstance], context: Dict) -> int:
        if not val_source: return 0
        if val_source.dynamic_source == "COUNT_REFERENCE":
            log_event("INFO", "game.get_dynamic_value", "Calculating COUNT_REFERENCE", player=player.name); return len(player.trash)
        return val_source.base
