from typing import List, Optional, Any, Tuple, Dict, Set
import random
import unicodedata
import re
import traceback
import uuid
import json
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
        from .effects.continuous import ContinuousEffectManager
        self.continuous = ContinuousEffectManager(self)

    def get_debug_snapshot(self) -> Dict[str, Any]:
        """
        現在のゲーム状態をAIデバッグ用に全ダンプする。
        """
        def _dump_zone(zone: List[Card]) -> List[str]:
            # カードID(名前) [状態] の形式で出力
            return [f"{c.uuid[:4]}:{c.master.id}({c.master.name}){'[REST]' if c.is_rest else '[ACT]'}" for c in zone]

        def _dump_player(p: Player) -> Dict[str, Any]:
            return {
                "life": len(p.life),
                "hand_count": len(p.hand),
                "hand_ids": [c.master.id for c in p.hand], # 手札の中身もIDだけ見る
                "field": _dump_zone(p.field),
                "trash_count": len(p.trash),
                "trash_top": [c.master.id for c in p.trash[-3:]], # トラッシュの最新3枚
                "leader": f"{p.leader.master.id}({p.leader.master.name})" if p.leader else None,
                "stage": f"{p.stage.master.id}({p.stage.master.name})" if p.stage else None,
                "don": {
                    "active": len(p.don_active),
                    "rested": len(p.don_rested),
                    "attached": len(p.don_attached_cards)
                }
            }

        return {
            "turn_count": self.turn_count,
            "phase": self.phase.name,
            "turn_player": self.turn_player.name,
            "p1_state": _dump_player(self.p1),
            "p2_state": _dump_player(self.p2),
            "active_interaction": str(self.active_interaction) if self.active_interaction else None
        }

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
            blockers = [c.uuid for c in target_owner.field if not c.is_rest and c.has_keyword("ブロッカー")]
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

            # ▼▼▼ 修正: save_idがなくても、一時的に選択結果を渡せるようにする ▼▼▼
            if "effect_context" in continuation:
                continuation["effect_context"]["temp_resolved_targets"] = selected_cards

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
        self.continuous.expire("TURN_END", self.turn_count)
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
        opponent = self.p2 if player == self.p1 else self.p1

        # Step 1: 両プレイヤーのバフ・一時キーワードをリセット
        for p in [player, opponent]:
            for c in ([p.leader] if p.leader else []) + p.field + ([p.stage] if p.stage else []):
                if c:
                    c.cost_buff = 0
                    c.base_power_override = None
                    c.current_keywords = c.master.keywords.copy()
            for c in p.hand:
                if c:
                    c.cost_buff = 0

        # Step 2: YOUR_TURN 効果（アクティブプレイヤーのカードのみ）
        for card in ([player.leader] if player.leader else []) + player.field:
            if not card or not card.master.abilities: continue
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.YOUR_TURN:
                    log_event("DEBUG", "game.passive_trigger", f"YOUR_TURN: {card.master.name}", player=player.name)
                    self.resolve_ability(player, ability, source_card=card)

        # Step 3: PASSIVE 効果（両プレイヤーのカードを評価）
        for p in [player, opponent]:
            for card in ([p.leader] if p.leader else []) + p.field:
                if not card or not card.master.abilities: continue
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.PASSIVE:
                        log_event("DEBUG", "game.passive_trigger", f"PASSIVE: {card.master.name}", player=p.name)
                        self.resolve_ability(p, ability, source_card=card)

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
            # 場を離れたら継続効果（timed_power/flags/keywords）を破棄する。
            self.continuous.drop_for(card.uuid)

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
            if not card.is_rest and card.has_keyword("ブロッカー") and "BLOCKER_DISABLED" not in card.flags:
                return True
        return False

    def declare_attack(self, attacker: Card, target: Card):
        attacker_owner, _ = self._find_card_location(attacker)
        target_owner, _ = self._find_card_location(target)
        self._validate_action(attacker_owner, "MAIN_ACTION")
        if "ATTACK_DISABLE" in attacker.flags or "ATTACK_DISABLE" in attacker.timed_flags: raise ValueError("このカードは効果によりアタックできません。")
        if attacker.is_rest: raise ValueError("アタックするカードはアクティブ状態でなければなりません。")
        if target.master.type == CardType.CHARACTER and not target.is_rest: raise ValueError("レスト状態のキャラクターのみ攻撃可能です。")
        log_event("INFO", "game.attack_declare", f"{attacker.master.name} is attacking {target.master.name}", player=attacker_owner.name)
        attacker.is_rest = True
        self.active_battle = {"attacker": attacker, "target": target, "attacker_owner": attacker_owner, "target_owner": target_owner, "counter_buff": 0}
        
        if attacker.master.abilities:
            for ability in attacker.master.abilities:
                if ability.trigger == TriggerType.ON_ATTACK:
                    self.resolve_ability(attacker_owner, ability, source_card=attacker)

        opp_cards = ([target_owner.leader] if target_owner.leader else []) + target_owner.field
        for card in opp_cards:
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.ON_OPP_ATTACK:
                    log_event("INFO", "game.trigger_opp_attack", f"ON_OPP_ATTACK fired for {card.master.name}", player=target_owner.name)
                    self.resolve_ability(target_owner, ability, source_card=card)

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
                damage_amount = 2 if attacker.has_keyword("ダブルアタック") else 1; is_banish = attacker.has_keyword("バニッシュ")
                log_event("INFO", "game.damage_step", f"Dealing {damage_amount} damage (Banish: {is_banish})", player=attacker_owner.name)
                for _ in range(damage_amount):
                    if target_owner.life:
                        life_card = target_owner.life.pop(0)
                        dest_zone = Zone.TRASH if is_banish else Zone.HAND
                        trigger_ability = None if is_banish else next(
                            (a for a in life_card.master.abilities if a.trigger == TriggerType.TRIGGER), None
                        )
                        self.move_card(life_card, dest_zone, target_owner)
                        log_event("INFO", "game.damage_life", f"{target_owner.name} takes damage to {dest_zone.name}", player=target_owner.name)
                        if trigger_ability:
                            log_event("INFO", "game.trigger_keyword", f"TRIGGER activated: {life_card.master.name}", player=target_owner.name)
                            self.resolve_ability(target_owner, trigger_ability, source_card=life_card)
                        self._fire_on_life_decrease(target_owner)
                    else: self.winner = attacker_owner.name; log_event("INFO", "game.victory", f"{attacker_owner.name} wins the game", player=attacker_owner.name); break
        else:
            if attacker_pwr >= target_pwr:
                if self._active_protection(target, ("BATTLE_KO",)):
                    log_event("INFO", "game.battle_ko_prevented", f"{target.master.name} is protected from battle KO", player=target_owner.name)
                elif self._active_replacement(target, ("BATTLE_KO",)):
                    log_event("INFO", "game.battle_ko_replaced", f"{target.master.name}'s battle KO was replaced by an alternative effect", player=target_owner.name)
                else:
                    self.move_card(target, Zone.TRASH, target_owner)
                    log_event("INFO", "game.unit_ko", f"{target.master.name} was KO'd", player=target_owner.name)
                    self._resolve_on_ko(target, target_owner)

        target.reset_turn_status(); self.active_battle = None; self.phase = Phase.MAIN; self.check_victory()
        self.continuous.expire("BATTLE_END", self.turn_count)
        if not self.winner:
            self._apply_passive_effects(self.turn_player)

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

    # 除去保護（PREVENT_LEAVE）の判定。除去が起こる瞬間に、対象カードの
    # PASSIVE 能力を走査し、条件（例: トラッシュ7枚以上）をライブ評価する。
    # status_values: "LEAVE"（相手の効果で場を離れない）/ "BATTLE_KO"（バトルでKOされない）
    def _active_protection(self, card: CardInstance, status_values: Tuple[str, ...]) -> bool:
        if not card or not getattr(card, "master", None) or card.negated:
            return False
        owner = self.p1 if self.p1.name == card.owner_id else self.p2
        resolver = None
        for ab in card.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            eff = ab.effect
            if not isinstance(eff, GameAction) or eff.type != ActionType.PREVENT_LEAVE:
                continue
            if eff.status not in status_values:
                continue
            if ab.condition is not None:
                if resolver is None:
                    resolver = EffectResolver(self)
                if not resolver._check_condition(owner, ab.condition, card):
                    continue
            return True
        return False

    # 置換効果（REPLACE_EFFECT）の判定。除去の瞬間に対象の PASSIVE 能力を走査し、
    # 「代わりに〜」の置換アクションを（条件・実行可能性を満たせば）実行して True を返す。
    # True の場合、呼び出し側は本来の除去を行わずスキップする。
    def _active_replacement(self, card: CardInstance, status_values: Tuple[str, ...]) -> bool:
        if not card or not getattr(card, "master", None) or card.negated:
            return False
        owner = self.p1 if self.p1.name == card.owner_id else self.p2
        for ab in card.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            eff = ab.effect
            if not isinstance(eff, GameAction) or eff.type != ActionType.REPLACE_EFFECT:
                continue
            if eff.status not in status_values:
                continue
            sub = getattr(eff, "sub_effect", None)
            if sub is None:
                continue
            resolver = EffectResolver(self)
            if ab.condition is not None and not resolver._check_condition(owner, ab.condition, card):
                continue
            # 代わりの行動が取れない（例: 捨てる手札が無い）場合は置換不成立＝本来の除去が起こる。
            if not resolver._can_satisfy_node(owner, sub, card):
                continue
            log_event("INFO", "game.replacement", f"Replacement effect activated for {card.master.name}", player=owner.name)
            resolver.execution_stack = [sub]
            resolver._process_stack(owner, card)
            return True
        return False

    def _resolve_on_ko(self, card: Card, owner: Player):
        if not card.master.abilities: return
        for ability in card.master.abilities:
            if ability.trigger == TriggerType.ON_KO:
                log_event("INFO", "game.trigger_ko", f"Resolving ON_KO for {card.master.name}", player=owner.name)
                self.resolve_ability(owner, ability, source_card=card)

    def _fire_on_life_decrease(self, player: Player):
        cards = ([player.leader] if player.leader else []) + player.field
        for card in cards:
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.ON_LIFE_DECREASE:
                    log_event("INFO", "game.trigger_life_decrease", f"ON_LIFE_DECREASE fired for {card.master.name}", player=player.name)
                    self.resolve_ability(player, ability, source_card=card)

    def _don_pool_player(self, player: Player, action: GameAction) -> Player:
        """ドン操作の対象プレイヤー。「相手は…」は status="OPPONENT"、
        対象クエリの player=OPPONENT でも相手を指す。既定は効果の実行者。"""
        opp = self.p2 if player == self.p1 else self.p1
        if getattr(action, 'status', None) == "OPPONENT":
            return opp
        tgt = getattr(action, 'target', None)
        if tgt is not None and getattr(getattr(tgt, 'player', None), 'name', '') == 'OPPONENT':
            return opp
        return player

    def apply_action_to_engine(self, player: Player, action: GameAction, targets: List[CardInstance], value: int) -> bool:
        if not action: return False
        act_name = action.type.name if hasattr(action.type, 'name') else str(action.type)
        log_event("INFO", "game.apply_action", f"Applying {act_name} to {len(targets)} targets", player=player.name)
        if act_name == "DRAW":
            target_player = player
            if action.target and getattr(action.target, 'player', None) is not None:
                if getattr(action.target.player, 'name', '') == 'OPPONENT':
                    target_player = self.p2 if player == self.p1 else self.p1
            self.draw_card(target_player, value)
            return True
        if act_name == "SHUFFLE":
            target_player = player
            if action.target and getattr(action.target, 'player', None) is not None:
                if getattr(action.target.player, 'name', '') == 'OPPONENT':
                    target_player = self.p2 if player == self.p1 else self.p1
            random.shuffle(target_player.deck)
            log_event("INFO", "game.action_shuffle", "Deck shuffled", player=target_player.name)
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
        
        if act_name in ["HEAL", "LIFE_RECOVER"]:
            for _ in range(value):
                if player.deck:
                    player.life.append(player.deck.pop(0))
                    log_event("INFO", "game.action_heal", f"{player.name} +1 life from deck top", player=player.name)
            return True
        if act_name == "RAMP_DON":
            # status=="RESTED" の場合はレスト状態でコストエリアへ（「レストで追加」）。
            add_rested = getattr(action, "status", None) == "RESTED"
            for _ in range(value):
                if player.don_deck:
                    don = player.don_deck.pop(0)
                    don.is_rest = add_rested
                    if add_rested:
                        player.don_rested.append(don)
                    else:
                        player.don_active.append(don)
                    log_event("INFO", "game.action_ramp_don", f"{player.name} ramped 1 DON!! (rested={add_rested})", player=player.name)
            return True

        if act_name == "RETURN_DON":
            # 「ドン‼-N」/「ドン!!デッキに戻す」: 場のドン!!を N 枚ドン!!デッキへ戻す。
            # 影響の小さい順（レスト→アクティブ→付与中）に戻す。
            tp = self._don_pool_player(player, action)
            returned = 0
            for _ in range(value):
                if tp.don_rested:
                    don = tp.don_rested.pop()
                elif tp.don_active:
                    don = tp.don_active.pop()
                elif tp.don_attached_cards:
                    don = tp.don_attached_cards.pop()
                else:
                    break
                don.is_rest = False
                don.attached_to = None
                tp.don_deck.append(don)
                returned += 1
            log_event("INFO", "game.action_return_don", f"{tp.name} returned {returned} DON!! to don deck", player=tp.name)
            return True

        if act_name == "REST_DON":
            # 「ドン!!N枚をレストにする」/【ドン!!×N】コスト: アクティブ→レスト。
            # ドンは均質なため枚数(value)ベースで処理する。
            tp = self._don_pool_player(player, action)
            rested = 0
            for _ in range(value):
                if not tp.don_active:
                    break
                don = tp.don_active.pop(0)
                don.is_rest = True
                tp.don_rested.append(don)
                rested += 1
            log_event("INFO", "game.action_rest_don", f"{tp.name} rested {rested} DON!!", player=tp.name)
            return True

        if act_name == "ACTIVE_DON" and not getattr(action, 'target', None):
            # 「ドン!!N枚をアクティブにする」: レスト→アクティブ（枚数ベース）。
            tp = self._don_pool_player(player, action)
            activated = 0
            for _ in range(value):
                if not tp.don_rested:
                    break
                don = tp.don_rested.pop()
                don.is_rest = False
                tp.don_active.append(don)
                activated += 1
            log_event("INFO", "game.action_active_don", f"{tp.name} activated {activated} DON!!", player=tp.name)
            return True

        # ▼▼▼ 修正: 初期値をTrueに設定（対象0枚でも「何もしないことに成功した」とみなすため） ▼▼▼
        success = True

        # 「相手の効果で場を離れない」対象になり得る除去アクション
        _LEAVE_ACTIONS = {"KO", "DISCARD", "TRASH", "BOUNCE", "MOVE_TO_HAND", "MOVE", "DECK_BOTTOM", "DECK_TOP", "MOVE_CARD"}

        for target in targets:
            owner, source_list = self._find_card_location(target)
            if not owner: continue
            # 相手の効果でフィールド上のカードを場から除去しようとする場合、保護/置換を確認
            if (act_name in _LEAVE_ACTIONS and player.name != owner.name
                    and source_list is owner.field):
                if self._active_protection(target, ("LEAVE",)):
                    log_event("INFO", "game.leave_prevented", f"{target.master.name} is protected from leaving the field by opponent's effect", player=owner.name)
                    continue
                if self._active_replacement(target, ("LEAVE",)):
                    log_event("INFO", "game.leave_replaced", f"{target.master.name}'s removal was replaced by an alternative effect", player=owner.name)
                    continue
            if act_name == "PREVENT_LEAVE":
                # 保護マーカー自体は no-op（実際の保護は除去時に _active_protection で評価）。
                success = True
            elif act_name == "KO":
                self.move_card(target, Zone.TRASH, owner)
                log_event("INFO", "game.action_ko", f"{target.master.name} was KO'd by effect", player=player.name)
                self._resolve_on_ko(target, owner)
                success = True
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
                    # 期間付き（このターン中／このバトル中 等）は継続効果(timed_cost)へ。
                    # cost_buff は _apply_passive_effects で毎回リセットされ消えるため。
                    # 期間指定なし(INSTANT)は従来どおり cost_buff（PASSIVE 再計算で再適用される）。
                    dur = getattr(action, "duration", "INSTANT")
                    if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END"):
                        expire_turn = self.turn_count + 1 if dur == "UNTIL_NEXT_TURN_END" else 0
                        self.continuous.apply(target, "COST", dur, amount=value, expire_turn=expire_turn)
                    elif hasattr(target, 'cost_buff'):
                        target.cost_buff += value
                    log_event("INFO", "game.action_cost_reduction", f"{target.master.name}'s cost changed by {value} ({dur})", player=player.name)
                elif action.status == "BLOCKER_DISABLE":
                    target.flags.add("BLOCKER_DISABLED")
                    target.current_keywords.discard("ブロッカー")
                    target.timed_keywords.discard("ブロッカー")  # 効果付与分の【ブロッカー】も無効化
                    log_event("INFO", "game.action_blocker_disable", f"{target.master.name} blocker disabled", player=player.name)
                else:
                    # 「このバトル中」のパワー増減は継続効果として管理し、バトル終了時に失効させる
                    # （従来は power_buff に直接加算され、同一ターンの後続バトルへ誤って持ち越していた）。
                    if getattr(action, "duration", "INSTANT") == "THIS_BATTLE":
                        self.continuous.apply(target, "POWER", "THIS_BATTLE", amount=value)
                    elif hasattr(target, 'power_buff'):
                        target.power_buff += value
                        log_event("INFO", "game.action_buff", f"{target.master.name} gained {value} power", player=player.name)
                success = True
            elif act_name in ["ATTACK_DISABLE", "RESTRICTION"]:
                # 「（このターン中／次の相手のターン終了時まで）アタックできない」
                dur = getattr(action, "duration", "INSTANT")
                if dur == "UNTIL_NEXT_TURN_END":
                    self.continuous.apply(target, "FLAG", "UNTIL_NEXT_TURN_END", flag="ATTACK_DISABLE", expire_turn=self.turn_count + 1)
                else:
                    self.continuous.apply(target, "FLAG", "THIS_TURN", flag="ATTACK_DISABLE")
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
            elif act_name in ["ACTIVE", "ACTIVE_DON"]:
                target.is_rest = False
                if isinstance(target, DonInstance):
                    if target in owner.don_rested:
                        owner.don_rested.remove(target)
                        owner.don_active.append(target)
                log_event("INFO", "game.action_active", f"Card activated for {owner.name}", player=player.name)
                success = True
            elif act_name == "ATTACH_DON":
                # value 枚のドン!!を付与する。status="RESTED" の場合はレストのドンを
                # レストのまま付与する（無ければもう一方のプールから補う）。
                from_rested = (action.status == "RESTED")
                n = value if value and value > 0 else 1
                attached = 0
                for _ in range(n):
                    pool = player.don_rested if from_rested else player.don_active
                    if not pool:
                        pool = player.don_active if from_rested else player.don_rested
                    if not pool:
                        break
                    don = pool.pop(0)
                    don.attached_to = target.uuid
                    don.is_rest = from_rested
                    player.don_attached_cards.append(don)
                    target.attached_don += 1
                    attached += 1
                log_event("INFO", "game.action_attach_don", f"{attached} DON!! attached to {target.master.name} (rested={from_rested})", player=player.name)
                success = True
            elif act_name == "MOVE_CARD":
                dest = action.destination if action.destination else Zone.HAND
                dest_pos = getattr(action, 'dest_position', 'BOTTOM') or 'BOTTOM'
                self.move_card(target, dest, owner, dest_position=dest_pos)
                success = True
            elif act_name == "DECK_TOP":
                self.move_card(target, Zone.DECK, owner, dest_position="TOP"); success = True
            elif act_name == "FACE_UP_LIFE":
                # 「ライフを表向き／裏向きにする」: status="DOWN" のみ裏向き、他は表向き。
                target.is_face_up = (action.status != "DOWN")
                log_event("INFO", "game.action_face_up_life",
                          f"{target.master.name} life set face_up={target.is_face_up}", player=player.name)
                success = True
            elif act_name == "GRANT_KEYWORD":
                keyword = action.status
                if not keyword and getattr(action, 'raw_text', ''):
                    import unicodedata as _ud
                    _kw = re.search(r'【([^】]+)】', _ud.normalize('NFC', action.raw_text))
                    if _kw:
                        keyword = _kw.group(1)
                if keyword:
                    # 継続効果として付与する（timed_keywords）。current_keywords へ直接
                    # 加えると _apply_passive_effects のリセットで消えてしまうため。
                    dur = getattr(action, "duration", "INSTANT")
                    cdur = dur if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END") else "PERMANENT"
                    expire_turn = self.turn_count + 1 if cdur == "UNTIL_NEXT_TURN_END" else 0
                    self.continuous.apply(target, "KEYWORD", cdur, keyword=keyword, expire_turn=expire_turn)
                    log_event("INFO", "game.action_grant_keyword", f"【{keyword}】→ {target.master.name} ({cdur})", player=player.name)
                success = True
        return success

    def get_dynamic_value(self, player: Player, val_source: ValueSource, targets: List[CardInstance], context: Dict) -> int:
        if not val_source: return 0
        if val_source.dynamic_source == "COUNT_REFERENCE":
            log_event("INFO", "game.get_dynamic_value", "Calculating COUNT_REFERENCE", player=player.name); return len(player.trash)
        return val_source.base