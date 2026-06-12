from typing import List, Optional, Any, Tuple, Dict, Set
import random
import unicodedata
import re
import traceback
import uuid
import json
from ..models.models import CardInstance, CardMaster, DonInstance, CONST
from ..models.enums import CardType, Attribute, Color, Phase, Zone, TriggerType, ConditionType, CompareOperator, ActionType, PendingMessage
from ..models.effect_types import TargetQuery, Ability, GameAction, ValueSource, Sequence, Branch, Choice
from ..utils.logger_config import log_event
from .effects.resolver import EffectResolver
from .effects.matcher import get_target_cards


Card = CardInstance

# 自己制限（self_cannot）の制限キー。parser が RULE_PROCESSING + status=これらで生成し、
# apply_action_to_engine が player.restrictions に記録、各アクション地点で enforce する。
SELF_RESTRICTION_KEYS = {
    "CANNOT_PLAY_FROM_HAND",      # 手札からカードをプレイできない
    "CANNOT_PLAY_CHARACTER",      # キャラ(カード)を登場できない（min_cost で「コストN以上」に限定可）
    "CANNOT_DRAW_BY_EFFECT",      # 自分の効果でカードを引くことができない
    "CANNOT_LIFE_TO_HAND",        # 自分の効果でライフを手札に加えられない
    "CANNOT_ATTACK_LEADER",       # リーダーにアタックできない
    "CANNOT_ACTIVATE_DON",        # キャラの効果でドン‼をアクティブにできない
}


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
        # 「相手の登場時効果は無効になる」(スコープ付き相手効果無効) の期限。
        # turn_count <= negate_onplay_until の間、このプレイヤーの ON_PLAY 解決をスキップする。
        self.negate_onplay_until: int = 0
        # 自己制限（「自分は、このターン中、…できない」= self_cannot）の保管。
        # key=制限種別(CANNOT_PLAY_CHARACTER 等) → {"expire": turn_count, "min_cost": Optional[int]}。
        # turn_count <= expire の間だけ有効（negate_onplay_until と同じ遅延失効方式）。
        self.restrictions: Dict[str, Dict[str, Any]] = {}

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
                "life": [self._format_card(c, c.is_face_up) for c in self.life],
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
        self.mulligan_done: Set[str] = set()
        from .effects.continuous import ContinuousEffectManager
        self.continuous = ContinuousEffectManager(self)
        self.action_events: List[Dict] = []  # per-request event buffer; reset in API handler
        # 「このターン終了時、〜」の遅延アクション待ち行列: (player, GameAction, source_card)。
        # resolver が積み、end_turn が解決する。
        self.pending_end_of_turn: List[tuple] = []

    def get_debug_snapshot(self) -> Dict[str, Any]:
        """
        現在のゲーム状態をAIデバッグ用に全ダンプする。
        """
        def _dump_zone(zone: List[Card]) -> List[str]:
            # カードID(名前) [状態] の形式で出力
            return [f"{c.uuid[:4]}:{c.master.card_id}({c.master.name}){'[REST]' if c.is_rest else '[ACT]'}" for c in zone]

        def _dump_player(p: Player) -> Dict[str, Any]:
            return {
                "life": len(p.life),
                "hand_count": len(p.hand),
                "hand_ids": [c.master.card_id for c in p.hand],
                "field": _dump_zone(p.field),
                "trash_count": len(p.trash),
                "trash_top": [c.master.card_id for c in p.trash[-3:]],
                "leader": f"{p.leader.master.card_id}({p.leader.master.name})" if p.leader else None,
                "stage": f"{p.stage.master.card_id}({p.stage.master.name})" if p.stage else None,
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
        
        # マリガンフェーズ中は手番の決まっていないプレイヤー順に要求
        if self.phase == Phase.MULLIGAN:
            for player in [self.p1, self.p2]:
                if player.name not in self.mulligan_done:
                    hand_candidates = [c.to_dict() for c in player.hand]
                    return {
                        KEY_PID: player.name,
                        KEY_ACTION: "MULLIGAN",
                        KEY_MSG: "マリガンするカードを選んでください（交換なし＝キープ）",
                        KEY_CANDIDATES: hand_candidates,
                        KEY_UUIDS: [c.uuid for c in player.hand],
                        KEY_CONSTRAINTS: {"min": 0, "max": len(player.hand)},
                        KEY_SKIP: True,
                        "request_id": str(uuid.uuid4()),
                    }
            return None

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
            # ARRANGE_DECK(並び替え/上下選択)はフロントの UI 切替フラグを併せて渡す。
            if action_type == "ARRANGE_DECK":
                req["allow_position"] = self.active_interaction.get("allow_position", False)
                req["allow_reorder"] = self.active_interaction.get("allow_reorder", False)
            return req

        if not self.active_battle and self.phase in [Phase.BLOCK_STEP, Phase.BATTLE_COUNTER]:
            log_event("ERROR", "game.pending_request_error", f"Active battle missing in phase: {self.phase.name}")
            self.phase = Phase.MAIN
            
        request = None
        ACT_BLOCKER = battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
        ACT_COUNTER = battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER')
        
        if self.phase == Phase.BLOCK_STEP and self.active_battle:
            target_owner = self.active_battle["target_owner"]
            blockers = [c.uuid for c in target_owner.field if not c.is_rest and c.has_keyword("ブロッカー") and "CANNOT_REST" not in c.timed_flags]
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

        elif action_type == "CONFIRM_OPTIONAL":
            # 任意効果（「〜してもよい」）の発動可否。accepted=False（パス/拒否）ならスキップ。
            accepted = payload.get("accepted")
            if accepted is None:
                # selected_uuids 非空 / index>0 / skip フラグ等から推定（既定は発動=True）
                if payload.get("skip") is True or payload.get("declined") is True:
                    accepted = False
                else:
                    accepted = payload.get("index", 0) == 0
            optional_node = continuation.get("optional_node")
            self.active_interaction = None
            resolver.resume_optional(player, source_card, bool(accepted), optional_node,
                                     continuation.get("execution_stack", []), continuation.get("effect_context", {}))

        elif action_type == "ARRANGE_DECK":
            # (2a)(2b) 並び替え/上下選択の確定。selected_uuids が配置順、position が上下。
            # ヘッドレス(drain)は selected_uuids=[] / position 無し → 現状順・fixed_position。
            ordered_uuids = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
            position = (payload.get("position") or payload.get("extra", {}).get("position")
                        or continuation.get("fixed_position", "BOTTOM"))
            position = "TOP" if str(position).upper() == "TOP" else "BOTTOM"
            cards = continuation.get("arrange_targets", [])
            if ordered_uuids:
                by_uuid = {c.uuid: c for c in cards}
                ordered = [by_uuid[u] for u in ordered_uuids if u in by_uuid]
                for c in cards:  # 指定漏れは元の順序で末尾に補う
                    if c not in ordered:
                        ordered.append(c)
            else:
                ordered = list(cards)
            dest_kind = continuation.get("dest_kind", "DECK")
            self.active_interaction = None
            if dest_kind == "LIFE":
                # ライフ並べ替え: ordered を新しいライフ順とする（life[0]=一番上）。
                owner_name = continuation.get("dest_owner")
                tp = self.p1 if (owner_name and self.p1.name == owner_name) else (self.p2 if owner_name else player)
                rest = [c for c in tp.life if c not in ordered]
                tp.life = ordered + rest
                log_event("INFO", "game.resume_order_life", f"{tp.name} reordered {len(ordered)} life card(s)", player=player.name)
            else:
                # デッキ配置: BOTTOM は順に append（先頭が上）、TOP は逆順 insert(0) で
                # ordered[0] が最上面になるようにする。
                seq = ordered if position == "BOTTOM" else list(reversed(ordered))
                for c in seq:
                    owner, _ = self._find_card_location(c)
                    if owner:
                        self.move_card(c, Zone.DECK, owner, dest_position=position)
                log_event("INFO", "game.resume_arrange_deck", f"Placed {len(ordered)} card(s) to deck {position}", player=player.name)
            resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), continuation.get("effect_context", {}))

        elif action_type == "DECLARE_COST":
            # C8: 宣言コストを記録し、相手デッキトップを公開して context に保存してから再開。
            declared = payload.get("declared_value", payload.get("index", 0))
            try:
                declared = int(declared)
            except (TypeError, ValueError):
                declared = 0
            effect_context = continuation.get("effect_context", {})
            effect_context["declared_cost"] = declared
            opponent = self.p2 if player == self.p1 else self.p1
            revealed = opponent.deck[0] if opponent.deck else None
            if revealed is not None:
                effect_context["last_revealed_card"] = revealed
                log_event("INFO", "game.declare_cost",
                          f"{source_card.master.name}: declared {declared}, revealed {revealed.master.name}(cost {revealed.master.cost})",
                          player=player.name)
            else:
                log_event("INFO", "game.declare_cost", f"{source_card.master.name}: declared {declared}, opponent deck empty", player=player.name)
            self.active_interaction = None
            resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), effect_context)

        if not self.active_interaction and self.setup_phase_pending:
            self.finish_setup()
            self.setup_phase_pending = False
            log_event("INFO", "game.turn_player", f"First Player: {self.turn_player.name}", player=self.turn_player.name)
            self.phase = Phase.MULLIGAN
            self.mulligan_done = set()
            log_event("INFO", "game.mulligan_start", "Mulligan phase started")

        # バトルトリガー(ON_ATTACK/ON_OPP_ATTACK)解決中の中断から復帰した場合:
        # バトルが進行中(active_battle あり)でまだ防御フェイズへ遷移していなければ、
        # 残りトリガーの解決＋フェイズ遷移を再開する（カウンター衝突エラーの防止）。
        if (not self.active_interaction and self.active_battle
                and self.phase not in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER)):
            self._advance_battle_triggers()

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
        # マリガンフェーズへ移行（両プレイヤーの確定後にゲーム開始）
        self.phase = Phase.MULLIGAN
        self.mulligan_done = set()
        log_event("INFO", "game.mulligan_start", "Mulligan phase started")

    def do_mulligan(self, player: 'Player') -> None:
        """手札5枚全てをデッキ底に戻してシャッフル→5枚引き直す（全交換・1回限り）"""
        if self.phase != Phase.MULLIGAN:
            raise ValueError("マリガンフェーズではありません。")
        if player.name in self.mulligan_done:
            raise ValueError("既にマリガンを実施済みです。")
        # 手札を全てデッキ底に戻す
        hand_count = len(player.hand)
        player.deck.extend(player.hand)
        player.hand.clear()
        random.shuffle(player.deck)
        for _ in range(5):
            if player.deck:
                player.hand.append(player.deck.pop(0))
        self.mulligan_done.add(player.name)
        log_event("INFO", "game.mulligan", f"Mulligan: {player.name} returned all {hand_count} cards", player=player.name)
        self._check_mulligan_complete()

    def keep_hand(self, player: 'Player') -> None:
        """手札をキープしてマリガンをスキップ"""
        if self.phase != Phase.MULLIGAN:
            raise ValueError("マリガンフェーズではありません。")
        if player.name in self.mulligan_done:
            raise ValueError("既にマリガンを実施済みです。")
        self.mulligan_done.add(player.name)
        log_event("INFO", "game.mulligan_keep", f"Keep hand: {player.name}", player=player.name)
        self._check_mulligan_complete()

    def _check_mulligan_complete(self) -> None:
        """両プレイヤーのマリガン確定後にゲーム開始"""
        if self.p1.name in self.mulligan_done and self.p2.name in self.mulligan_done:
            log_event("INFO", "game.mulligan_complete", "Both players done — starting game", player="system")
            self.turn_count = 1
            self.refresh_phase()

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
        self._fire_turn_end_triggers()
        # 「このターン終了時、〜」で予約された遅延アクションを解決する。
        self._flush_pending_end_of_turn()
        self.continuous.expire("TURN_END", self.turn_count)
        self.switch_turn()

    def _fire_turn_end_triggers(self):
        """ターン終了時トリガーを発火する。ターンプレイヤーの【自分のターン終了時】
        (TURN_END) と、非ターンプレイヤーの【相手のターン終了時】(OPP_TURN_END)。"""
        def _units(pl):
            us = [pl.leader] + pl.field
            if pl.stage: us.append(pl.stage)
            return us
        for pl, trig in ((self.turn_player, TriggerType.TURN_END),
                         (self.opponent, TriggerType.OPP_TURN_END)):
            for card in _units(pl):
                if card and card.master.abilities:
                    for ability in card.master.abilities:
                        if ability.trigger == trig:
                            self.resolve_ability(pl, ability, source_card=card)

    def _flush_pending_end_of_turn(self):
        """end_turn フックで、予約された遅延アクション（このターン終了時、〜）を解決する。"""
        if not self.pending_end_of_turn:
            return
        pending = self.pending_end_of_turn
        self.pending_end_of_turn = []
        for player, node, source_card in pending:
            # 場を離れたカードのソース由来でも、トラッシュ送り等は対象解決時に弾かれる。
            resolver = EffectResolver(self)
            resolver.context["_flushing_delayed"] = True
            resolver.execution_stack = [node]
            try:
                resolver._process_stack(player, source_card)
            except Exception as e:
                log_event("WARNING", "game.delayed_action_error", f"Deferred action failed: {e}", player=player.name)
            for ev in resolver.action_history:
                self.action_events.append({
                    "type": "EFFECT", "player": player.name,
                    "card_name": source_card.master.name,
                    "action": ev.get("action", ""), "targets": ev.get("targets", []),
                    "value": ev.get("value"), "success": ev.get("success", True),
                })

    def switch_turn(self):
        # 追加ターン（EXTRA_TURN）: 予約したプレイヤーがターンプレイヤーのまま継続する
        if getattr(self, "pending_extra_turn", None) == self.turn_player.name:
            self.pending_extra_turn = None
            self.turn_count += 1
            log_event("INFO", "game.extra_turn", f"{self.turn_player.name} takes an extra turn", player=self.turn_player.name)
            self.refresh_phase()
            return
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

    _REACTIVE_RE = re.compile(r'(された|した|受けた|なった)時、')

    def _is_reactive_passive(self, ability) -> bool:
        """無タグの反応型（「…が登場した時、」「…が戻された時、」等）でトリガー写像が
        まだ無い PASSIVE 能力か。常時効果ではないため再計算ループで実行してはならない
        （実行すると盤面操作のたびに本体効果が発動し、対話中断が他の解決を飲み込む）。"""
        first = self._find_first_action(ability.effect)
        raw = getattr(first, "raw_text", "") if first is not None else ""
        return bool(self._REACTIVE_RE.search(unicodedata.normalize("NFC", raw or "")))

    def _find_first_action(self, node):
        if node is None:
            return None
        if isinstance(node, GameAction):
            return node
        if isinstance(node, Sequence):
            for a in node.actions:
                found = self._find_first_action(a)
                if found is not None:
                    return found
        elif isinstance(node, Branch):
            return self._find_first_action(node.if_true) or self._find_first_action(node.if_false)
        elif isinstance(node, Choice):
            for o in node.options:
                found = self._find_first_action(o)
                if found is not None:
                    return found
        return None

    def _apply_passive_effects(self, player: Player):
        # 対話中断中は再計算しない。Step1 のリセットは無条件に走る一方、Step2/3 の
        # resolve_ability は active_interaction ガードで何も実行できず、リセットだけが
        # 残って PASSIVE/YOUR_TURN バフが消えてしまうため（クザンのコスト-5 等）。
        if self.active_interaction:
            return
        # YOUR_TURN 効果は常にターンプレイヤー基準で適用する（呼び出し元が owner を
        # 渡しても誤適用しない）。
        if self.turn_player is not None:
            player = self.turn_player
        opponent = self.p2 if player == self.p1 else self.p1

        # Step 1: 両プレイヤーのバフ・一時キーワードをリセット
        for p in [player, opponent]:
            for c in ([p.leader] if p.leader else []) + p.field + ([p.stage] if p.stage else []):
                if c:
                    c.cost_buff = 0
                    c.passive_power = 0
                    c.passive_power_override = None
                    c.current_keywords = c.master.keywords.copy()
            for c in p.hand:
                if c:
                    c.cost_buff = 0
                    c.passive_counter = 0

        # Step 2/3 で適用される INSTANT パワーバフは passive_power（再計算レイヤ）に
        # 載せる。power_buff に加えると _apply_passive_effects が呼ばれるたびに
        # 累積し、PASSIVE「パワー+1000」が盤面操作のたびに際限なく増えていた。
        self._in_passive_recalc = True
        try:
            # Step 2: YOUR_TURN 効果（アクティブプレイヤーのカードのみ）
            #   ステージ（player.stage）も対象に含める。聖地マリージョア(コスト軽減)・
            #   虚の玉座(リーダー+1000) 等の STAGE の YOUR_TURN 効果が従来発動していなかった。
            for card in ([player.leader] if player.leader else []) + player.field + ([player.stage] if player.stage else []):
                if not card or not card.master.abilities: continue
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.YOUR_TURN:
                        if self._is_reactive_passive(ability):
                            continue  # 「【自分のターン中】…された時」型はイベント誘発（EB02-035 等）
                        log_event("DEBUG", "game.passive_trigger", f"YOUR_TURN: {card.master.name}", player=player.name)
                        self.resolve_ability(player, ability, source_card=card)

            # Step 3: PASSIVE 効果（両プレイヤーのカードを評価）。ステージも含める。
            for p in [player, opponent]:
                for card in ([p.leader] if p.leader else []) + p.field + ([p.stage] if p.stage else []):
                    if not card or not card.master.abilities: continue
                    for ability in card.master.abilities:
                        if ability.trigger == TriggerType.PASSIVE:
                            if self._is_reactive_passive(ability):
                                continue  # 「…された時」型はイベント誘発であり再計算で実行しない
                            log_event("DEBUG", "game.passive_trigger", f"PASSIVE: {card.master.name}", player=p.name)
                            self.resolve_ability(p, ability, source_card=card)
        finally:
            self._in_passive_recalc = False

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
            if (not card.is_rest and card.has_keyword("ブロッカー")
                    and "BLOCKER_DISABLED" not in card.flags
                    and "CANNOT_REST" not in card.timed_flags):
                return True
        return False

    def declare_attack(self, attacker: Card, target: Card):
        attacker_owner, _ = self._find_card_location(attacker)
        target_owner, _ = self._find_card_location(target)
        self._validate_action(attacker_owner, "MAIN_ACTION")
        if "ATTACK_DISABLE" in attacker.flags or "ATTACK_DISABLE" in attacker.timed_flags: raise ValueError("このカードは効果によりアタックできません。")
        if "CANNOT_REST" in attacker.timed_flags: raise ValueError("このカードは効果によりレストにできないためアタックできません。")
        if attacker.is_rest: raise ValueError("アタックするカードはアクティブ状態でなければなりません。")
        # 自己制限（self_cannot）:「リーダーにアタックできない」。相手リーダーへの攻撃宣言を弾く。
        if (target.master.type == CardType.LEADER
                and attacker_owner is not None
                and self._active_restriction(attacker_owner, "CANNOT_ATTACK_LEADER")):
            raise ValueError("効果により、このターンはリーダーにアタックできません。")
        if (target.master.type == CardType.CHARACTER and not target.is_rest
                and not attacker.has_keyword("ATTACK_ACTIVE")):
            raise ValueError("レスト状態のキャラクターのみ攻撃可能です。")
        log_event("INFO", "game.attack_declare", f"{attacker.master.name} is attacking {target.master.name}", player=attacker_owner.name)
        attacker.is_rest = True
        self.active_battle = {"attacker": attacker, "target": target, "attacker_owner": attacker_owner, "target_owner": target_owner, "counter_buff": 0}

        # アタック時/相手のアタック時トリガーを順に解決する。途中でいずれかが対象選択や
        # 選択(Choice)で中断した場合、解決前にブロッカー/カウンター段階へ進むと、未解決の
        # interaction とカウンター操作が衝突する（"期待:CHOICE" エラー）。トリガーを待ち行列に
        # 積み、_advance_battle_triggers で1つずつ解決し、全て片付いてからフェイズ遷移する。
        triggers = []
        if attacker.master.abilities:
            for ability in attacker.master.abilities:
                if ability.trigger == TriggerType.ON_ATTACK:
                    triggers.append((attacker_owner, ability, attacker))
        opp_cards = ([target_owner.leader] if target_owner.leader else []) + target_owner.field
        for card in opp_cards:
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.ON_OPP_ATTACK:
                    triggers.append((target_owner, ability, card))
        self._battle_triggers = triggers
        self._advance_battle_triggers()

    def _advance_battle_triggers(self):
        """積んだバトルトリガーを順に解決し、全て解決後に防御フェイズへ遷移する。
        途中で interaction が立ったら中断（resolve_interaction が解決後に再度呼ぶ）。"""
        if not self.active_battle:
            self._battle_triggers = []
            return
        while getattr(self, "_battle_triggers", None):
            player, ability, card = self._battle_triggers.pop(0)
            log_event("INFO", "game.trigger_battle", f"Battle trigger: {card.master.name} ({ability.trigger.name})", player=player.name)
            self.resolve_ability(player, ability, source_card=card)
            if self.active_interaction:
                return  # 中断: 解決後に resolve_interaction から再開される
        # 全トリガー解決 → ブロッカー/カウンター段階へ
        target_owner = self.active_battle["target_owner"]
        if self.has_blocker(target_owner):
            self.phase = Phase.BLOCK_STEP
            log_event("INFO", "game.phase_transition", f"Blockers detected. Moving to {self.phase.name}", player=target_owner.name)
        else:
            self.phase = Phase.BATTLE_COUNTER
            log_event("INFO", "game.phase_transition", f"No blockers. Moving to {self.phase.name}", player=target_owner.name)

    def handle_block(self, blocker: Optional[Card] = None):
        if not self.active_battle: return
        target_owner = self.active_battle["target_owner"]; self._validate_action(target_owner, "SELECT_BLOCKER")
        if blocker:
            log_event("INFO", "game.block_execute", f"{blocker.master.name} blocks the attack", player=target_owner.name)
            blocker.is_rest = True
            self.active_battle["target"] = blocker
            # 【ブロック時】効果を発動する（従来は未発火＝14枚が no-op だった）。
            if blocker.master.abilities and not blocker.ability_disabled and not blocker.negated:
                for ability in blocker.master.abilities:
                    if ability.trigger == TriggerType.ON_BLOCK:
                        log_event("INFO", "game.trigger_block", f"ON_BLOCK: {blocker.master.name}", player=target_owner.name)
                        self.resolve_ability(target_owner, ability, source_card=blocker)
            if self.active_interaction:
                # ブロック時効果が対象選択等で中断した場合はここで返す（resume が継続）。
                return
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
            counter_value = getattr(counter_card, "current_counter", counter_card.master.counter or 0); self.active_battle["counter_buff"] += counter_value
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
        # デッキアウト: 通常は本人の敗北（相手の勝利）。ただし C10「自分のデッキが0枚に
        # なった場合、敗北する代わりに勝利する」(VICTORY/REPLACE_DECKOUT_LOSS) を持つ場合は
        # 本人の勝利へ置換する（OP03-040 ナミ等）。
        if not self.p1.deck:
            self.winner = self.p1.name if self._has_deckout_win_replace(self.p1) else self.p2.name
        elif not self.p2.deck:
            self.winner = self.p2.name if self._has_deckout_win_replace(self.p2) else self.p1.name

    def _has_deckout_win_replace(self, player) -> bool:
        """player がデッキアウト時の敗北→勝利の置換能力(PASSIVE)を持つか。"""
        units = [player.leader] + list(player.field)
        for card in units:
            if not card or not getattr(card, "master", None) or getattr(card, "negated", False):
                continue
            if getattr(card, "ability_disabled", False):
                continue
            for ab in card.master.abilities:
                if ab.trigger != TriggerType.PASSIVE:
                    continue
                eff = self._find_action(ab.effect, ActionType.VICTORY)
                if eff is not None and eff.status == "REPLACE_DECKOUT_LOSS":
                    return True
        return False

    def play_card_action(self, player: Player, card: Card):
        if card not in player.hand: return
        self._validate_action(player, "MAIN_ACTION")
        # 自己制限（self_cannot）: 「手札からカードをプレイできない」「キャラ（コストN以上）を登場できない」。
        if self._active_restriction(player, "CANNOT_PLAY_FROM_HAND"):
            raise ValueError("効果により、このターンは手札からカードをプレイできません。")
        if card.master.type == CardType.CHARACTER:
            char_rec = self._active_restriction(player, "CANNOT_PLAY_CHARACTER")
            if char_rec is not None:
                min_cost = char_rec.get("min_cost")
                # 「元々のコスト」= master.cost（修正前の値）で判定する。
                if min_cost is None or (card.master.cost is not None and card.master.cost >= min_cost):
                    suffix = f"コスト{min_cost}以上の" if min_cost else ""
                    raise ValueError(f"効果により、このターンは{suffix}キャラを登場できません。")
        log_event("INFO", "game.play_card", f"Playing card: {card.master.name}", player=player.name, payload={"card_uuid": card.uuid})
        if card.master.type == CardType.EVENT:
            for ability in card.master.abilities:
                if ability.trigger in [TriggerType.ON_PLAY, TriggerType.ACTIVATE_MAIN]:
                    self.resolve_ability(player, ability, source_card=card)
            self.move_card(card, Zone.TRASH, player)
        else:
            self.move_card(card, Zone.FIELD, player); card.attached_don = 0; card.is_newly_played = True
            # 登場した時点で継続効果（PASSIVE/YOUR_TURN）を適用してから ON_PLAY を解決する。
            # 例: クザン「相手のキャラすべてをコスト-5」+【登場時】コスト0のキャラをKO —
            # 自身の継続効果が ON_PLAY の対象判定に反映される必要がある。
            self._apply_passive_effects(self.turn_player)
            if self._has_rested_play(player):  # 「自分のキャラはレストで登場する」PASSIVE
                card.is_rest = True
            # 「相手の登場時効果は無効になる」(OPP_ONPLAY) 期間中はこのプレイヤーの ON_PLAY を解決しない。
            onplay_negated = getattr(player, "negate_onplay_until", 0) >= self.turn_count
            if onplay_negated:
                log_event("INFO", "game.onplay_negated",
                          f"{card.master.name}'s ON_PLAY is negated by opponent's effect", player=player.name)
            if not card.ability_disabled and not onplay_negated:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_PLAY:
                        self.resolve_ability(player, ability, source_card=card)
            self._apply_passive_effects(player)

    def _has_rested_play(self, player: Player) -> bool:
        """player が「自分のキャラはレストで登場する」PASSIVE を持つか（RESTED_PLAY マーカー）。"""
        cards = ([player.leader] if player.leader else []) + list(player.field)
        for c in cards:
            if not c or getattr(c, "ability_disabled", False) or not getattr(c, "master", None):
                continue
            for ab in c.master.abilities:
                if ab.trigger != TriggerType.PASSIVE:
                    continue
                act = self._find_action(ab.effect, ActionType.RESTRICTION)
                if act is not None and getattr(act, "status", None) == "RESTED_PLAY":
                    return True
        return False

    def _active_restriction(self, player: Player, key: str) -> Optional[Dict[str, Any]]:
        """player に有効な自己制限（self_cannot）があれば、そのパラメータ dict を返す。
        turn_count <= expire の間だけ有効。期限切れエントリは掃除して None を返す。"""
        rec = getattr(player, "restrictions", {}).get(key)
        if not rec:
            return None
        if self.turn_count <= rec.get("expire", -1):
            return rec
        # 期限切れは破棄
        player.restrictions.pop(key, None)
        return None

    def _blocks_effect_play(self, card: CardInstance) -> bool:
        """card が「手札のこのカードは効果で登場できない」PASSIVE を持つか（NO_EFFECT_PLAY）。"""
        if not card or not getattr(card, "master", None):
            return False
        for ab in card.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            act = self._find_action(ab.effect, ActionType.RESTRICTION)
            if act is not None and getattr(act, "status", None) == "NO_EFFECT_PLAY":
                return True
        return False

    def resolve_ability(self, player: Player, ability: Ability, source_card: CardInstance):
        if source_card.negated or source_card.ability_disabled: return
        resolver = EffectResolver(self)
        resolver.resolve_ability(player, ability, source_card)
        for ev in resolver.action_history:
            self.action_events.append({
                "type": "EFFECT",
                "player": player.name,
                "card_name": source_card.master.name,
                "action": ev.get("action", ""),
                "targets": ev.get("targets", []),
                "value": ev.get("value"),
                "success": ev.get("success", True),
            })

    # 除去保護（PREVENT_LEAVE）の判定。除去が起こる瞬間に、対象カードの
    # PASSIVE 能力を走査し、条件（例: トラッシュ7枚以上）をライブ評価する。
    # status_values: "LEAVE"（相手の効果で場を離れない）/ "BATTLE_KO"（バトルでKOされない）
    def _find_action(self, node, action_type: ActionType) -> Optional[GameAction]:
        """効果ツリー(GameAction/Sequence/Branch/Choice)から指定タイプの GameAction を探す。
        「場を離れず、【X】を得る」のように PREVENT_LEAVE が Sequence の一要素になる場合に対応。"""
        if node is None:
            return None
        if isinstance(node, GameAction):
            return node if node.type == action_type else None
        if isinstance(node, Sequence):
            for a in node.actions:
                found = self._find_action(a, action_type)
                if found is not None:
                    return found
        elif isinstance(node, Branch):
            return self._find_action(node.if_true, action_type) or (
                self._find_action(node.if_false, action_type) if node.if_false else None)
        elif isinstance(node, Choice):
            for o in node.options:
                found = self._find_action(o, action_type)
                if found is not None:
                    return found
        return None

    def _active_protection(self, card: CardInstance, status_values: Tuple[str, ...]) -> bool:
        if not card or not getattr(card, "master", None) or card.negated:
            return False
        owner = self.p1 if self.p1.name == card.owner_id else self.p2

        # トリガー効果が継続効果として付与した期間付き保護（timed_flags）。
        # 例: 「このキャラは、次の自分のターン開始時まで、バトルでKOされない」(ON_ATTACK)
        for s in status_values:
            if f"PREVENT_{s}" in (card.flags | card.timed_flags):
                return True

        resolver = None
        # 走査対象: 自身に加え、オーナーのリーダー/フィールド/ステージの範囲保護
        # （「自分の特徴《X》を持つキャラすべては…場を離れない」等。従来は自身のみ走査で
        #   他カードを守る保護が機能しなかった）。
        protectors = [card]
        if owner.leader and owner.leader is not card:
            protectors.append(owner.leader)
        protectors.extend(fc for fc in owner.field if fc is not card)
        if getattr(owner, "stage", None) and owner.stage is not card:
            protectors.append(owner.stage)

        for protector in protectors:
            if getattr(protector, "ability_disabled", False) or getattr(protector, "negated", False):
                continue
            for ab in protector.master.abilities:
                if ab.trigger != TriggerType.PASSIVE:
                    continue
                eff = self._find_action(ab.effect, ActionType.PREVENT_LEAVE)
                if eff is None:
                    continue
                if eff.status not in status_values:
                    continue
                # 保護対象クエリの照合: SOURCE は protector 自身のみを守る。
                # 範囲クエリは card が範囲に含まれるかを実体化して確認する。
                tgt = getattr(eff, "target", None)
                if tgt is None or getattr(tgt, "select_mode", "SOURCE") == "SOURCE":
                    if protector is not card:
                        continue
                else:
                    if card not in get_target_cards(self, tgt, protector):
                        continue
                if ab.condition is not None:
                    if resolver is None:
                        resolver = EffectResolver(self)
                    src = card if protector is card else protector
                    if not resolver._check_condition(owner, ab.condition, src):
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

        # 走査対象: 除去されるカード自身 → オーナーのリーダー → フィールドの他キャラ
        # （自身の置換効果と、他キャラを守る OPPONENT_REMOVAL 型置換効果の両方をカバー）
        candidates = [card]
        if owner.leader and owner.leader is not card:
            candidates.append(owner.leader)
        for fc in owner.field:
            if fc is not card:
                candidates.append(fc)

        for protector in candidates:
            if getattr(protector, 'ability_disabled', False):
                continue
            for ab in protector.master.abilities:
                if ab.trigger != TriggerType.PASSIVE:
                    continue
                eff = self._find_action(ab.effect, ActionType.REPLACE_EFFECT)
                if eff is None:
                    continue
                if eff.status not in status_values:
                    continue
                sub = getattr(eff, "sub_effect", None)
                if sub is None:
                    continue
                resolver = EffectResolver(self)
                # 条件チェック: source_card には「除去されるカード」を渡す（OPPONENT_REMOVAL フィルタ評価のため）
                if ab.condition is not None and not resolver._check_condition(owner, ab.condition, card):
                    continue
                # 代わりの行動が取れない場合は置換不成立
                if not resolver._can_satisfy_node(owner, sub, protector):
                    continue
                log_event("INFO", "game.replacement",
                          f"Replacement by {protector.master.name} activated for {card.master.name}",
                          player=owner.name)
                # 置換は除去の解決中（apply_action_to_engine 内）に発生する「入れ子の中断」で、
                # 現行の単一 continuation 設計ではネスト中断を表現できない（doc §7-C E14/E15）。
                # 完全な continuation スタック化は高リスクのため、置換 sub_effect が対象選択／
                # 任意確認で中断した場合は、その場で保守的に自動解決して同期完了させる
                # （任意=実行＝保護を採用、対象=有効候補を選択）。これにより置換は必ず完了し、
                # ダングリング interaction（カードが KO もされず置換も未完了の宙吊り）を防ぐ。
                outer_interaction = self.active_interaction
                self.active_interaction = None
                resolver.execution_stack = [sub]
                resolver._process_stack(owner, protector)
                self._auto_resolve_replacement(owner)
                # 置換解決で外側の interaction を壊していないことを保証（元の状態へ戻す）。
                self.active_interaction = outer_interaction
                return True
        return False

    def _auto_resolve_replacement(self, owner: Player, limit: int = 16) -> None:
        """置換 sub_effect が残した中断（任意確認／対象選択）を保守的に同期解決する。

        単一 continuation 設計ではネストした中断を UI へ伝播できないため、置換は headless で
        完結させる: 任意確認(CONFIRM_OPTIONAL)は accept（保護を実行）、対象選択(SELECT_TARGET)は
        有効候補から必要数を自動選択する。選択 UI のフロント連携は E14/E15 の将来課題。"""
        n = 0
        while self.active_interaction and n < limit:
            ia = self.active_interaction
            atype = ia.get("action_type")
            pid = ia.get("player_id")
            actor = self.p1 if self.p1.name == pid else self.p2
            if atype == "SELECT_TARGET":
                cand = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
                mx = (ia.get("constraints") or {}).get("max", 1) or 1
                payload = {"selected_uuids": cand[:mx], "index": 0}
            elif atype == "CONFIRM_OPTIONAL":
                payload = {"accepted": True}
            elif atype == "CHOICE":
                payload = {"index": 0}
            else:
                # 想定外の中断種別は安全側に倒して打ち切る（宙吊り防止のため interaction を解除）。
                self.active_interaction = None
                break
            try:
                self.resolve_interaction(actor, payload)
            except Exception as e:
                log_event("WARNING", "game.replacement_autoresolve_fail",
                          f"Auto-resolve of replacement interaction failed: {e}", player=owner.name)
                self.active_interaction = None
                break
            n += 1

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
        # 自己制限（「自分は、このターン中、…できない」= self_cannot）の登録。
        # parser が RULE_PROCESSING + status=制限キーで生成する。対象を持たないため、
        # 通常の `for target in targets` ループ前にここで処理して player に記録する。
        if act_name == "RULE_PROCESSING" and getattr(action, "status", None) in SELF_RESTRICTION_KEYS:
            rec: Dict[str, Any] = {"expire": self.turn_count}  # 「このターン中」: 現ターン内のみ有効
            if action.value and getattr(action.value, "base", None):
                rec["min_cost"] = action.value.base
            player.restrictions[action.status] = rec
            log_event("INFO", "game.self_restriction",
                      f"{player.name} restricted: {action.status}{' (cost>=' + str(rec.get('min_cost')) + ')' if rec.get('min_cost') else ''} this turn",
                      player=player.name)
            return True
        if act_name == "DRAW":
            target_player = player
            if action.target and getattr(action.target, 'player', None) is not None:
                if getattr(action.target.player, 'name', '') == 'OPPONENT':
                    target_player = self.p2 if player == self.p1 else self.p1
            # 「自分の効果でカードを引くことができない」: 効果解決による DRAW を抑止する。
            if self._active_restriction(target_player, "CANNOT_DRAW_BY_EFFECT"):
                log_event("INFO", "game.draw_restricted", f"{target_player.name} cannot draw by effect this turn", player=player.name)
                return True
            self.draw_card(target_player, value)
            return True
        if act_name in ("DEAL_DAMAGE", "DAMAGE"):
            # 「相手に N ダメージを与える」: 相手リーダーへ N ダメージ。ライフ上から N 枚を
            # 手札へ移し（【トリガー】発動・ON_LIFE_DECREASE 発火）、ライフが尽きれば勝利。
            # 従来 DEAL_DAMAGE は未実装で no-op だった（ニコ・ロビン等のダメージ効果が不発）。
            damaged = self.p2 if player == self.p1 else self.p1
            if action.target and getattr(getattr(action.target, 'player', None), 'name', '') == 'SELF':
                damaged = player
            n = value if value and value > 0 else 1
            for _ in range(n):
                if damaged.life:
                    life_card = damaged.life.pop(0)
                    trig = next((a for a in life_card.master.abilities if a.trigger == TriggerType.TRIGGER), None)
                    self.move_card(life_card, Zone.HAND, damaged)
                    log_event("INFO", "game.deal_damage", f"{damaged.name} takes 1 damage to HAND", player=player.name)
                    if trig:
                        self.resolve_ability(damaged, trig, source_card=life_card)
                    self._fire_on_life_decrease(damaged)
                else:
                    self.winner = player.name
                    log_event("INFO", "game.victory", f"{player.name} wins (effect damage)", player=player.name)
                    break
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
            if getattr(action, "status", None) == "OPPONENT":
                # 「相手のデッキの上から N 枚を見る」: 公開のみで盤面は不変（並びも変えない）。
                # 後続消費が無いため temp_zone には載せない（TEMP リーク防止）。
                opp = self.p2 if player == self.p1 else self.p1
                count = min(value if value else 1, len(opp.deck))
                log_event("INFO", "game.action_look_opp", f"Looking at {count} cards from OPPONENT DECK", player=player.name)
                return True
            count = value
            deck = player.deck
            if len(deck) < count: count = len(deck)
            log_event("INFO", "game.action_look", f"Looking at {count} cards from DECK", player=player.name)
            for _ in range(count):
                card = deck.pop(0)
                player.temp_zone.append(card)
            return True

        if act_name == "LOOK_LIFE":
            # 「（自分か相手の）ライフの上から N枚を見る」→ 対象プレイヤーのライフ上 value 枚を
            # 同プレイヤーの temp_zone へ移して公開する。後続の Choice が temp→ライフ上/下に戻す。
            # status=="OPPONENT" で相手のライフを対象（相手の temp_zone に載るため、戻し先も相手）。
            target_player = player
            if getattr(action, "status", None) == "OPPONENT":
                target_player = self.p2 if player == self.p1 else self.p1
            count = value if value else 1
            moved = 0
            for _ in range(count):
                if not target_player.life:
                    break
                card_ = target_player.life.pop(0)
                # 不発時の回収先を記録する（temp 回収はデッキトップではなくライフ上へ戻す）
                card_._temp_origin = "LIFE"
                target_player.temp_zone.append(card_)
                moved += 1
            log_event("INFO", "game.action_look_life", f"{target_player.name} revealed {moved} life card(s)", player=player.name)
            return True

        if act_name == "MOVE_ATTACHED_DON":
            # 「付与されているドン‼N枚をコストエリアにレストで戻す」: 付与中のドンを N 枚外し、
            # レスト状態で don_rested（コストエリア）へ。付与先キャラの attached_don も減算する。
            n = value if value and value > 0 else 1
            moved = 0
            for don in list(player.don_attached_cards):
                if moved >= n:
                    break
                tgt_uuid = getattr(don, "attached_to", None)
                player.don_attached_cards.remove(don)
                don.attached_to = None
                don.is_rest = True
                player.don_rested.append(don)
                if tgt_uuid:
                    tgt = next((c for c in ([player.leader] + player.field) if c and c.uuid == tgt_uuid), None)
                    if tgt is not None and getattr(tgt, "attached_don", 0) > 0:
                        tgt.attached_don -= 1
                moved += 1
            log_event("INFO", "game.move_attached_don", f"{player.name} returned {moved} attached DON!! to cost area", player=player.name)
            # コストとして使われるため、要求枚数を戻せたかを成否で返す（付与ドン不足なら不成立）。
            return moved >= n

        if act_name == "REDIRECT_ATTACK":
            # 「（選んだキャラ/このリーダー等）にアタックの対象を変更する」: 進行中バトルの
            # 対象を差し替える。targets[0] が新しい対象（多くはコントローラー側のキャラ/リーダー）。
            if self.active_battle and targets:
                new_target = targets[0]
                self.active_battle["target"] = new_target
                self.active_battle["target_owner"] = self.p1 if self.p1.name == new_target.owner_id else self.p2
                log_event("INFO", "game.redirect_attack",
                          f"Attack redirected to {new_target.master.name}", player=player.name)
            return True

        if act_name == "DISABLE_ABILITY" and getattr(action, "status", None) == "OPP_ONPLAY":
            # 「（次の相手のターン終了時まで、）相手の登場時効果は無効になる」: 相手プレイヤーに
            # ON_PLAY 無効化の期限(turn_count)を設定する。次の相手ターン(=turn_count+1)を覆う。
            opp = self.p2 if player == self.p1 else self.p1
            dur = getattr(action, "duration", "INSTANT")
            opp.negate_onplay_until = self.turn_count + (1 if dur == "UNTIL_NEXT_TURN_END" else 0)
            log_event("INFO", "game.negate_opp_onplay",
                      f"{opp.name}'s ON_PLAY negated until turn {opp.negate_onplay_until}", player=player.name)
            return True

        if act_name == "EXTRA_TURN":
            # 「このターンの後に自分のターンを追加で得る」: switch_turn が消費する
            self.pending_extra_turn = player.name
            log_event("INFO", "game.action_extra_turn", f"{player.name} will take an extra turn", player=player.name)
            return True

        if act_name == "VICTORY":
            # 「（自分は）ゲームに勝利する」: 能動勝利。即座に winner を設定する。
            # status="REPLACE_DECKOUT_LOSS" はデッキアウト敗北の置換マーカー(PASSIVE)で、
            # 直接実行されない（_has_deckout_win_replace で走査）。万一実行された場合は無視。
            if getattr(action, "status", None) == "REPLACE_DECKOUT_LOSS":
                return True
            self.winner = player.name
            log_event("INFO", "game.victory", f"{player.name} wins (effect)", player=player.name)
            return True

        if act_name == "ORDER_LIFE":
            # 「（自分/相手の）ライフすべてを見て、好きな順番で置く」: ライフを任意順に並べ替える。
            # ライフ2枚以上のときは resolver が ARRANGE_DECK 対話(dest_kind=LIFE)で先に中断し、
            # プレイヤーが順序を選ぶ。ここに来るのはライフ1枚以下（並べ替え不要）の場合で、
            # 並びを保持する（枚数不変・カード消失なし・TEMP 非汚染）。
            target_player = player
            if getattr(action, "status", None) == "OPPONENT":
                target_player = self.p2 if player == self.p1 else self.p1
            log_event("INFO", "game.action_order_life",
                      f"{target_player.name} reorders {len(target_player.life)} life card(s)", player=player.name)
            return True

        if act_name == "EXECUTE_EVENT":
            # 「自分の手札から（条件）イベント1枚までを、発動する」: 手札のイベントの効果を
            # 解決し、発動後にトラッシュへ送る。効果解決は DEAL_DAMAGE のライフトリガー解決
            # （上記）と同じく resolve_ability の再入で行う（新規実行コンテキストを生成する
            # ため既存スタックを汚さない）。targets は matcher が手札のイベントを解決済み。
            _main_trigs = (TriggerType.ACTIVATE_MAIN, TriggerType.COUNTER, TriggerType.ON_PLAY)
            for ev in targets:
                ev_ability = next((a for a in ev.master.abilities
                                   if a.effect is not None and a.trigger in _main_trigs), None)
                if ev_ability is None:
                    ev_ability = next((a for a in ev.master.abilities if a.effect is not None), None)
                if ev_ability is not None:
                    self.resolve_ability(player, ev_ability, source_card=ev)
                self.move_card(ev, Zone.TRASH, player)
                log_event("INFO", "game.execute_event", f"Activated event {ev.master.name}", player=player.name)
            return True

        if act_name == "SELECT":
            # 「（対象）を選ぶ」: 対象選択のみ（盤面は動かさない）。選択結果は
            # _resolve_targets / resolve_interaction が target.save_id="selected_card" に
            # 保存済み。後続の「選んだ／その（カード/キャラ/リーダー）」が ref_id で参照する。
            log_event("INFO", "game.action_select", f"Selected {len(targets)} card(s)", player=player.name)
            return True

        if act_name in ["HEAL", "LIFE_RECOVER"]:
            for _ in range(value):
                if player.deck:
                    player.life.append(player.deck.pop(0))
                    log_event("INFO", "game.action_heal", f"{player.name} +1 life from deck top", player=player.name)
            return True
        if act_name == "TRASH_FROM_DECK":
            # 「（自分／相手の）デッキの上からN枚をトラッシュに置く」（mill）。
            # デッキは並びが意味を持つため対象選択させず、上から value 枚を送る。
            target_player = player
            if getattr(action, "status", None) == "OPPONENT":
                target_player = self.p2 if player == self.p1 else self.p1
            milled = 0
            for _ in range(value):
                if not target_player.deck:
                    break
                target_player.trash.append(target_player.deck.pop(0))
                milled += 1
            log_event("INFO", "game.action_trash_from_deck", f"{target_player.name} milled {milled} card(s) from deck top", player=target_player.name)
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
            # 「キャラの効果でドン‼をアクティブにできない」: 効果によるアクティブ化を抑止。
            if self._active_restriction(tp, "CANNOT_ACTIVATE_DON"):
                log_event("INFO", "game.active_don_restricted", f"{tp.name} cannot activate DON!! by effect this turn", player=player.name)
                return True
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
                # PASSIVE 由来(INSTANT)はマーカーのみ（除去時に _active_protection が走査）。
                # トリガー効果の期間付き保護は継続効果フラグとして対象に付与する
                # （従来は no-op で「次の…まで、バトルでKOされない」が機能しなかった）。
                dur = getattr(action, "duration", "INSTANT")
                if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END"):
                    flag = f"PREVENT_{action.status or 'LEAVE'}"
                    expire_turn = self.turn_count + 1 if dur == "UNTIL_NEXT_TURN_END" else 0
                    self.continuous.apply(target, "FLAG", dur, flag=flag, expire_turn=expire_turn)
                    log_event("INFO", "game.action_prevent_leave",
                              f"{target.master.name} protected ({action.status}, {dur})", player=player.name)
                success = True
            elif act_name == "KO":
                self.move_card(target, Zone.TRASH, owner)
                log_event("INFO", "game.action_ko", f"{target.master.name} was KO'd by effect", player=player.name)
                self._resolve_on_ko(target, owner)
                success = True
            elif act_name in ["DISCARD", "TRASH"]:
                self.move_card(target, Zone.TRASH, owner); success = True
            elif act_name == "REVEAL":
                # 公開: 盤面は動かさず、公開した事実をログに残す（条件成立の証明等）。
                log_event("INFO", "game.action_reveal", f"{target.master.name} was revealed", player=owner.name)
                success = True
            elif act_name in ["BOUNCE", "MOVE_TO_HAND"]:
                self.move_card(target, Zone.HAND, owner); success = True
            elif act_name == "MOVE":
                dest_zone = action.destination or Zone.TRASH; self.move_card(target, dest_zone, owner); success = True
            elif act_name == "BUFF":
                if action.status == "POWER_OVERRIDE":
                    # PASSIVE 再計算由来は再計算レイヤへ（即時効果の上書きを消さない）
                    if getattr(self, "_in_passive_recalc", False):
                        target.passive_power_override = value
                    else:
                        target.base_power_override = value
                    log_event("INFO", "game.action_override", f"{target.master.name}'s power set to {value}", player=player.name)
                elif action.status == "COST_OVERRIDE":
                    # コスト絶対値セット（「このターン中、コスト0にする」等）。base_power_override
                    # と同様に reset_turn_status で失効する（passive 再計算では消えない）。
                    target.base_cost_override = value
                    log_event("INFO", "game.action_cost_override", f"{target.master.name}'s cost set to {value}", player=player.name)
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
                elif action.status == "COUNTER":
                    # 「カウンター+Nになる」: 手札カードのカウンター値修正。
                    # PASSIVE 再計算レイヤ（passive_counter）に載せる。
                    if getattr(self, "_in_passive_recalc", False):
                        target.passive_counter += value
                    else:
                        target.passive_counter += value  # 即時付与も同レイヤ（手札は recalc でリセット）
                    log_event("INFO", "game.action_counter_buff", f"{target.master.name} counter {value:+d}", player=player.name)
                elif action.status == "BLOCKER_DISABLE":
                    target.flags.add("BLOCKER_DISABLED")
                    target.current_keywords.discard("ブロッカー")
                    target.timed_keywords.discard("ブロッカー")  # 効果付与分の【ブロッカー】も無効化
                    log_event("INFO", "game.action_blocker_disable", f"{target.master.name} blocker disabled", player=player.name)
                else:
                    # 期間付きパワー増減は継続効果(timed_power)として管理する。
                    #  - THIS_BATTLE: バトル終了で失効（同一ターンの後続バトルへ持ち越さない）。
                    #  - THIS_TURN / UNTIL_NEXT_TURN_END: ターン境界の reset_turn_status で
                    #    消えると困る（例: 被攻撃リーダーの「このターン中+N」が resolve_attack の
                    #    target.reset_turn_status で battle 終了時に即消える）。継続効果に載せて存続させる。
                    dur = getattr(action, "duration", "INSTANT")
                    if dur in ("THIS_BATTLE", "THIS_TURN", "UNTIL_NEXT_TURN_END"):
                        expire_turn = self.turn_count + 1 if dur == "UNTIL_NEXT_TURN_END" else 0
                        self.continuous.apply(target, "POWER", dur, amount=value, expire_turn=expire_turn)
                    elif getattr(self, "_in_passive_recalc", False):
                        # PASSIVE/YOUR_TURN 再計算中: 再計算レイヤに載せる（累積防止）
                        target.passive_power += value
                        log_event("INFO", "game.action_buff", f"{target.master.name} passive power {value:+d}", player=player.name)
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
            elif act_name == "PREVENT_REST":
                # 「（相手の）キャラは…までレストにできない」: レスト不可＝そのキャラは
                # 自身をレストできない＝アタックもブロックもできない（どちらも本体をレストにする）。
                # ATTACK_DISABLE と同様、継続効果の timed_flags に "CANNOT_REST" を載せ、
                # declare_attack / has_blocker でこのフラグを弾く。
                dur = getattr(action, "duration", "INSTANT")
                if dur == "UNTIL_NEXT_TURN_END":
                    self.continuous.apply(target, "FLAG", "UNTIL_NEXT_TURN_END", flag="CANNOT_REST", expire_turn=self.turn_count + 1)
                else:
                    self.continuous.apply(target, "FLAG", "THIS_TURN", flag="CANNOT_REST")
                log_event("INFO", "game.action_prevent_rest", f"{target.master.name} cannot be rested ({dur})", player=player.name)
                success = True
            elif act_name == "FREEZE":
                # 「次の相手のリフレッシュフェイズでアクティブにならない」
                # refresh_all が flags["FREEZE"] を確認してからリセットするため、
                # ターン境界を跨ぐ flags に直接書き込む（timed_flags でなく flags）。
                target.flags.add("FREEZE")
                log_event("INFO", "game.action_freeze", f"{target.master.name} frozen (won't activate next refresh)", player=player.name)
                success = True
            elif act_name == "NEGATE_EFFECT":
                # 「（このターン中、）効果を無効にする」
                target.ability_disabled = True
                target._refresh_keywords()
                log_event("INFO", "game.action_negate", f"{target.master.name} ability disabled this turn", player=player.name)
                success = True
            elif act_name == "RULE_PROCESSING":
                # ルール上の注記（カード名 alias、デッキ枚数ルール等）→ エンジン no-op
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
                # 「手札のこのカードは効果で登場できない」: 手札源かつ当該 PASSIVE を持つ対象は
                # 効果による登場をスキップする（NO_EFFECT_PLAY）。
                if source_list is getattr(owner, "hand", None) and self._blocks_effect_play(target):
                    log_event("INFO", "game.play_blocked", f"{target.master.name} cannot be played by effect", player=owner.name)
                    continue
                self.move_card(target, Zone.FIELD, owner)
                target.is_newly_played = True
                # 「レストで登場させる」: フィールドに出た瞬間レスト状態にする。
                # 効果の明示 RESTED、または owner の「キャラはレストで登場する」PASSIVE のいずれか。
                if getattr(action, "status", None) == "RESTED" or self._has_rested_play(owner):
                    target.is_rest = True
                if not target.ability_disabled:
                    for ability in target.master.abilities:
                        if ability.trigger == TriggerType.ON_PLAY:
                            self.resolve_ability(owner, ability, source_card=target)
                self._apply_passive_effects(owner)
                success = True
            elif act_name == "DECK_BOTTOM":
                # 並び替え/上下選択を要する場合は resolver が ARRANGE_DECK で先に中断するため、
                # ここに来るのは位置確定（TOP/BOTTOM/未指定=BOTTOM）の配置のみ。
                _pos = "TOP" if getattr(action, "dest_position", None) == "TOP" else "BOTTOM"
                self.move_card(target, Zone.DECK, owner, dest_position=_pos); success = True
            elif act_name in ["ACTIVE", "ACTIVE_DON"]:
                target.is_rest = False
                if isinstance(target, DonInstance):
                    if target in owner.don_rested:
                        owner.don_rested.remove(target)
                        owner.don_active.append(target)
                log_event("INFO", "game.action_active", f"Card activated for {owner.name}", player=player.name)
                success = True
            elif act_name == "ATTACH_DON":
                # value 枚のドン!!を付与する。status に "RESTED" を含めばレストのドンを
                # レストのまま付与する（無ければもう一方のプールから補う）。status に "OPP" を
                # 含めば相手のドンプールから付与する（OP15-015「相手のレストのドン‼を付与」）。
                st = action.status or ""
                from_rested = ("RESTED" in st)
                from_opp = ("OPP" in st)
                don_owner = (self.p2 if player == self.p1 else self.p1) if from_opp else player
                n = value if value and value > 0 else 1
                attached = 0
                for _ in range(n):
                    pool = don_owner.don_rested if from_rested else don_owner.don_active
                    if not pool:
                        pool = don_owner.don_active if from_rested else don_owner.don_rested
                    if not pool:
                        break
                    don = pool.pop(0)
                    don.attached_to = target.uuid
                    don.is_rest = from_rested
                    don_owner.don_attached_cards.append(don)
                    target.attached_don += 1
                    attached += 1
                log_event("INFO", "game.action_attach_don", f"{attached} DON!! attached to {target.master.name} (rested={from_rested}, opp_pool={from_opp})", player=player.name)
                success = True
            elif act_name == "MOVE_CARD":
                dest = action.destination if action.destination else Zone.HAND
                # 自己制限（self_cannot）:「自分の効果でライフを手札に加えられない」。
                # 自分のライフ→自分の手札の移動のみ抑止する（相手への移動・他ゾーンは対象外）。
                if (dest == Zone.HAND and source_list is owner.life and owner is player
                        and self._active_restriction(player, "CANNOT_LIFE_TO_HAND")):
                    log_event("INFO", "game.life_to_hand_restricted",
                              f"{player.name} cannot add life to hand by effect this turn", player=player.name)
                    continue
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
        # 文脈依存「直前アクションで捨てた/戻した/KOした…カードN枚につき」（§7-5）。
        # 生の枚数を返す（divisor/multiplier は _calculate_value が適用する）。
        if val_source.dynamic_source == "PREV_ACTION_COUNT":
            return int((context or {}).get("_last_action_count", 0) or 0)
        # 「<範囲>N枚につき」の汎用カウント（RC-4）。範囲クエリを毎回実体化して数える
        # （PASSIVE 再計算で盤面に追随する）。
        if val_source.dynamic_source == "COUNT_QUERY" and getattr(val_source, "count_query", None) is not None:
            src = None
            src_uuid = (context or {}).get("_source_card_uuid")
            if src_uuid:
                src = self._find_card_by_uuid(src_uuid)
            if src is None:
                src = player.leader
            n = len(get_target_cards(self, val_source.count_query, src))
            log_event("INFO", "game.get_dynamic_value", f"COUNT_QUERY = {n}", player=player.name)
            return n
        # C9「（相手のリーダー／選んだキャラ／アタックしているキャラ）と同じパワーになる」。
        # 発動時スナップショット: 参照カードの現在パワーを固定値として返す（以後の変動に追随しない）。
        if val_source.dynamic_source == "REFERENCE_POWER":
            ref = self._resolve_power_reference(player, val_source.ref_id, context)
            if ref is None:
                return val_source.base
            ref_owner, _ = self._find_card_location(ref)
            is_ref_turn = bool(ref_owner) and ref_owner.name == self.turn_player.name
            return ref.get_power(is_ref_turn)
        # 「元々のパワーと同じ」: 参照カードの基礎値（master.power）を写す（バフ非追随）
        if val_source.dynamic_source == "REFERENCE_BASE_POWER":
            ref = self._resolve_power_reference(player, val_source.ref_id, context)
            if ref is None:
                return val_source.base
            return ref.master.power
        return val_source.base

    def _resolve_power_reference(self, player, ref_id, context):
        """C9 の同値パワー参照カードを解決する。ref_id: selected/opp_leader/attacker。"""
        opponent = self.p2 if player == self.p1 else self.p1
        if ref_id == "opp_leader":
            return opponent.leader
        if ref_id == "self_leader":
            return player.leader
        if ref_id == "attacker":
            return (self.active_battle or {}).get("attacker")
        if ref_id == "selected":
            saved = (context or {}).get("saved_targets", {})
            sel = saved.get("selected_card") or saved.get("selected")
            if isinstance(sel, list):
                return sel[0] if sel else None
            return sel
        return None