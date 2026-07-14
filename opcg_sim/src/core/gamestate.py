from typing import List, Optional, Any, Tuple, Dict, Set
import random
import re
from ..models.models import CardInstance, DonInstance, CONST
from . import journal
from .journal import JournaledList, JournaledDict, JournaledSet, record_attr
from ..models.enums import CardType, Phase, Zone, TriggerType, ActionType
from ..models.effect_types import Ability, GameAction, ValueSource, Sequence, Branch, Choice
from .effects.resolver import EffectResolver
from .actions import apply_action as _apply_action
from .engine import values as _values, guards as _guards
from .engine import interaction as _interaction
from .engine import battle as _battle, turn_flow as _turn_flow
from .engine import triggers as _triggers
from .engine import card_moves as _card_moves, passives as _passives


Card = CardInstance

# 場のキャラクター上限（公式ルール）。ステージ(owner.stage)・ドン!!は含まない。
# 6体目を登場させた場合は自分のキャラ1体を選んでトラッシュして5体に戻す（強制トラッシュ）。

# 自己制限（self_cannot）の制限キーの正本は rules_constants.py（actions と共有＝循環回避の葉）。
# ここは後方互換の再エクスポート（恒久・公開エイリアス）。新規参照は rules_constants から import する
# こと（gamestate は resolver/matcher/atoms より下流なので、それら上流から本エイリアスを import すると
# 循環する）。
from .rules_constants import SELF_RESTRICTION_KEYS, FIELD_LIMIT  # noqa: E402,F401


from .engine._helpers import _nfc, _TURN1_RE, _condition_turn_limit, _ability_turn_limit, _ability_index  # noqa: F401
# ↑ 後方互換の再エクスポート（正本は engine/_helpers.py。gamestate/engine の双方が使う葉ヘルパ）。

class Player:
    # ゾーン（list）は JournaledList を保証する。本番は __init__＋append/remove で常に JournaledList だが、
    # テスト等で `p.hand=[...]` と素 list を代入されると中断再開の make/unmake が巻き戻せない（plain list は
    # 未 journaled）。代入時に素 list を JournaledList へ昇格させて make/unmake の健全性を担保する。
    _LIST_ZONES = frozenset({"hand", "field", "life", "trash", "deck"})

    def __setattr__(self, name, value):
        # 差分巻き戻し（journal.transaction 中のみ記録）。不活性時は素通り。
        if type(value) is list and name in Player._LIST_ZONES:
            value = JournaledList(value)
        if journal._TL.active is not None:   # ホットパス: threadlocal を直接読む
            record_attr(self, name, self.__dict__)
        object.__setattr__(self, name, value)

    def __init__(self, name: str, deck: List[Card], leader: Optional[Card] = None):
        self.name = name
        self.life: List[Card] = JournaledList()
        self.hand: List[Card] = JournaledList()
        self.field: List[Card] = JournaledList()
        self.trash: List[Card] = JournaledList()
        self.stage: Optional[Card] = None
        self.deck = JournaledList(deck)
        self.don_deck: List[DonInstance] = JournaledList(DonInstance(owner_id=name) for _ in range(10))
        self.don_active: List[DonInstance] = JournaledList()
        self.don_rested: List[DonInstance] = JournaledList()
        self.don_attached_cards: List[DonInstance] = JournaledList()
        self.leader: Optional[Card] = leader
        self.temp_zone: List[Card] = JournaledList()
        # 「相手の登場時効果は無効になる」(スコープ付き相手効果無効) の期限。
        # turn_count <= negate_onplay_until の間、このプレイヤーの ON_PLAY 解決をスキップする。
        self.negate_onplay_until: int = 0
        # 自己制限（「自分は、このターン中、…できない」= self_cannot）の保管。
        # key=制限種別(CANNOT_PLAY_CHARACTER 等) → {"expire": turn_count, "min_cost": Optional[int]}。
        # turn_count <= expire の間だけ有効（negate_onplay_until と同じ遅延失効方式）。
        self.restrictions: Dict[str, Dict[str, Any]] = JournaledDict()
        # 継続付与型の置換（カウンターイベント等が「自分のキャラすべては、このターン中、
        # …の場合、代わりに〜できる」を付与する）の保管。各要素 = {"status", "sub_effect",
        # "is_optional", "expire_turn"}。turn_count <= expire_turn の間だけ有効（遅延失効）。
        # 場に残らない発生源（イベント＝即トラッシュ）の置換を、被除去キャラ側から参照するため。
        self.granted_replacements: List[Dict[str, Any]] = JournaledList()

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

    def to_dict(self, is_owner: bool = True, is_my_turn: bool = True):
        player_props = CONST.get('PLAYER_PROPERTIES', {})
        leader_dict = self.leader.to_dict(is_my_turn) if self.leader else None
        if leader_dict:
            leader_dict["is_face_up"] = True
        stage_dict = self.stage.to_dict(is_my_turn) if self.stage else None
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
                "field": [self._format_card(c, True, is_my_turn) for c in self.field],
                "hand": [self._format_card(c, is_owner, is_my_turn) for c in self.hand],
                "life": [self._format_card(c, c.is_face_up, is_my_turn) for c in self.life],
                "trash": [self._format_card(c, True, is_my_turn) for c in self.trash],
                "stage": stage_dict
            }
        }

    def _format_card(self, card: Card, face_up: bool, is_my_turn: bool = True) -> dict:
        d = card.to_dict(is_my_turn)
        d["is_face_up"] = face_up
        return d

class GameManager:
    def __setattr__(self, name, value):
        # 差分巻き戻し（journal.transaction 中のみ記録）。object.__setattr__ 経由なので
        # active_interaction 等の data descriptor（property）も従来どおり機能する。
        if journal._TL.active is not None:   # ホットパス: threadlocal を直接読む
            record_attr(self, name, self.__dict__)
        object.__setattr__(self, name, value)

    def __init__(self, player1: Player, player2: Player):
        self.p1 = player1
        self.p2 = player2
        self.turn_player = self.p1
        self.opponent = self.p2
        # リーダーの「ルール上、自分のドン!!デッキはN枚になる」(OP15-058 エネル等) を適用する。
        # RULE_PROCESSING は実行時 no-op のため、ドン!!デッキ枚数はここで初期化し直さないと
        # 既定の10枚のままになり、「ドン!!が6枚以下」シナジーが機能しなくなる。
        self._apply_leader_don_deck_rule(self.p1)
        self._apply_leader_don_deck_rule(self.p2)
        self.turn_count = 0
        # Phase2: 継続効果再計算（_apply_passive_effects）の dirty-flag。最後に再計算したときの
        # journal._mut_count を保持し、探索中(make/unmake)に入力不変なら再計算を省く。-1=未計算。
        self._passive_mc = -1
        # このターン中に発生したイベントの回数（EVENT_THIS_TURN 条件用）。ターン開始でクリア。
        # 例: "DON_RETURNED"（ドン!!デッキへ返却）/ "CHAR_LEFT_BY_OWN_EFFECT" / "NAVY_DISCARD" /
        #     "TRIGGER_CHAR_PLAYED"。各イベント発生地点で record_turn_event() を呼ぶ。
        self._turn_events: Dict[str, int] = JournaledDict()
        self.phase = Phase.SETUP
        self.winner: Optional[str] = None
        self.active_battle: Optional[Dict[str, Any]] = None
        # 中断（対話）はスタックで保持する。`active_interaction` プロパティが先頭を指す。
        # 通常は深さ≤1（単一スロット相当）で、置換ネスト等のみが push で深くする。
        self._interaction_stack: List[Dict[str, Any]] = JournaledList()
        # 除去置換の内側中断を提示した直後に立つシグナル（外側継続の退避要否を resolver が判定する）。
        self._replacement_suspended = False
        # 入れ子の除去置換が中断したとき、失われる外側の継続（後続アクション／残対象）を退避する
        # スタック。内側中断が解決された後（active_interaction が無くなった後）に LIFO で再開する。
        # これにより「除去が効果シーケンスの途中（後続あり）／複数対象」でも置換の内側選択を
        # UI へ提示できる（accepted limitation B = multi-source continuation の解消）。
        self._deferred_continuations: List[Dict[str, Any]] = JournaledList()
        self.setup_phase_pending = False
        # ターン開始時誘発（TURN_START）の解決が終わるまでリフレッシュフェイズ以降を
        # 保留するフラグ。解決完了時に resolve_interaction が refresh_phase を再開する。
        self.turn_start_pending = False
        self.mulligan_done: Set[str] = JournaledSet()
        from .effects.continuous import ContinuousEffectManager
        self.continuous = ContinuousEffectManager(self)
        self.action_events: List[Dict] = JournaledList()  # per-request event buffer; reset in API handler
        # 「このターン終了時、〜」の遅延アクション待ち行列: (player, GameAction, source_card)。
        # resolver が積み、end_turn が解決する。
        self.pending_end_of_turn: List[tuple] = JournaledList()
        # ライフ公開【トリガー】・ON_LIFE_DECREASE 等の誘発能力の待ち行列。
        # 各要素 = {"player","ability","card","optional","_confirmed"}。
        # _advance_pending_triggers が1件ずつ確認/解決し、中断時は resolve_interaction が再開する。
        # _battle_triggers と同型だが、戦闘解決の外（ダメージ後・効果ダメージ）でも使う汎用版。
        self._pending_triggers: List[Dict[str, Any]] = JournaledList()
        # RETURN_DON（ドン!!返却）でプレイヤーが選んだ戻すドン!!の uuid 一覧。
        # resolver が選択解決時にセットし、apply_action_to_engine が消費する。
        self._return_don_selection: Optional[List[str]] = None

    # --- 中断（対話）スタック ---------------------------------------------
    # `active_interaction` は「いま UI へ提示すべき中断」＝スタック先頭を指す互換プロパティ。
    # 既存コードの読み書き（getter=先頭、setter(None)=先頭を pop、setter(dict)=先頭を置換／
    # 空なら push）はすべて単一スロット時代と同一挙動（深さ≤1）。置換ネスト等のネスト中断
    # のみ `push_interaction` で深さ>1にし、resolve_interaction が先頭から順に解決する。
    @property
    def active_interaction(self) -> Optional[Dict[str, Any]]:
        return self._interaction_stack[-1] if self._interaction_stack else None

    @active_interaction.setter
    def active_interaction(self, value: Optional[Dict[str, Any]]) -> None:
        if value is None:
            if self._interaction_stack:
                self._interaction_stack.pop()
        elif self._interaction_stack:
            self._interaction_stack[-1] = value
        else:
            self._interaction_stack.append(value)

    def push_interaction(self, interaction: Dict[str, Any]) -> None:
        """既存の中断を残したまま、新たな中断を上に積む（ネスト中断用）。

        通常の `active_interaction = {...}` は先頭を置換するが、置換 sub_effect のように
        外側の中断を保ったまま内側の選択を提示したい場合はこちらを使う。"""
        self._interaction_stack.append(interaction)

    def _apply_leader_don_deck_rule(self, player: Player) -> None:
        return _card_moves._apply_leader_don_deck_rule(self, player)

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

    def clone(self) -> "GameManager":
        """現在のゲーム状態の独立した深いコピーを返す（先読み/シミュレーション用）。

        `continuous` がマネージャへの後方参照を持つが deepcopy が循環を解決する。
        一時バッファ（action_events）はコピー後にリセットする。
        本体（self）は一切変化させない（docs/SPEC.md §2.5.2）。
        """
        import copy
        snapshot = copy.deepcopy(self)
        snapshot.action_events = JournaledList()
        return snapshot

    def get_legal_actions(self, player: Optional[Player] = None) -> List[Dict[str, Any]]:
        """現在 `player`（既定=pending の要求先）が取れる合法手を列挙する。

        返り値は適用可能なアクションの dict リスト:
          - ゲームアクション: {"kind":"game", "action_type":..., "payload":{...}}
          - 戦闘アクション:   {"kind":"battle", "action_type":..., "card_uuid":...}
          - 効果対話の解決:   {"kind":"game", "action_type":"RESOLVE_EFFECT_SELECTION", "payload":{...}}

        AI（探索/方策）と自己対戦ランナーが共有する合法手の単一の真実源。
        効果対話（SELECT_TARGET/CHOICE 等）は組合せ爆発を避けるため、
        `default_interaction_payload` による「妥当な既定解決」を1手として返す（PR1）。
        生成手はすべて `_validate_action` を通過することをテストで保証する。
        """
        pending = self.get_pending_request()
        if not pending:
            return []
        pending_props = CONST.get('PENDING_REQUEST_PROPERTIES', {})
        KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
        KEY_ACTION = pending_props.get('ACTION', 'action')
        battle_actions = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
        ACT_BLOCKER = battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
        ACT_COUNTER = battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER')
        ACT_PASS = battle_actions.get('PASS', 'PASS')
        RESOLVE = CONST.get('c_to_s_interface', {}).get('GAME_ACTIONS', {}).get('TYPES', {}).get('RESOLVE_EFFECT_SELECTION', 'RESOLVE_EFFECT_SELECTION')

        req_pid = pending[KEY_PID]
        # player 未指定なら要求先プレイヤーを行動主体とする。
        if player is None:
            player = self.p1 if self.p1.name == req_pid else self.p2
        # 要求先と異なるプレイヤーは合法手なし（手番/フェイズ外）。
        if player.name != req_pid:
            return []

        action = pending[KEY_ACTION]
        moves: List[Dict[str, Any]] = []

        if action == "MULLIGAN":
            moves.append({"kind": "game", "action_type": "MULLIGAN", "payload": {}})
            moves.append({"kind": "game", "action_type": "KEEP_HAND", "payload": {}})
            return moves

        if action == ACT_BLOCKER:
            for uid in pending.get(pending_props.get('SELECTABLE_UUIDS', 'selectable_uuids'), []):
                moves.append({"kind": "battle", "action_type": ACT_BLOCKER, "card_uuid": uid})
            moves.append({"kind": "battle", "action_type": ACT_PASS, "card_uuid": None})
            return moves

        if action == ACT_COUNTER:
            for uid in pending.get(pending_props.get('SELECTABLE_UUIDS', 'selectable_uuids'), []):
                moves.append({"kind": "battle", "action_type": ACT_COUNTER, "card_uuid": uid})
            moves.append({"kind": "battle", "action_type": ACT_PASS, "card_uuid": None})
            return moves

        if action == "MAIN_ACTION":
            # プレイ可能な手札（コストを active ドン!! で支払える＋自己制限を尊重）。
            # play_card_action と同じ制限判定を行い、登場が弾かれる手を最初から除外する。
            don_active = len(player.don_active)
            cannot_play_hand = self._active_restriction(player, "CANNOT_PLAY_FROM_HAND") is not None
            char_rec = self._active_restriction(player, "CANNOT_PLAY_CHARACTER")
            for c in player.hand:
                if c.current_cost > don_active:
                    continue
                if cannot_play_hand:
                    continue
                # 【メイン】効果を持たないイベント（カウンター/トリガー専用）はメインで発動不可。
                if c.master.type == CardType.EVENT and not self._event_has_main_play(c):
                    continue
                if c.master.type == CardType.CHARACTER and char_rec is not None:
                    min_cost = char_rec.get("min_cost")
                    if min_cost is None or (c.master.cost is not None and c.master.cost >= min_cost):
                        continue  # この制限下では登場できないキャラ
                moves.append({"kind": "game", "action_type": "PLAY", "payload": {"uuid": c.uuid}})
            # アタック: アクティブな攻撃者 × 有効な対象。
            # 各プレイヤーの最初のターン(turn_count<=2)はアタックできないため攻撃者を列挙しない。
            opponent = self.p2 if player == self.p1 else self.p1
            attackers = []
            if self.turn_count > 2:
                if player.leader and not player.leader.is_rest:
                    attackers.append(player.leader)
                for c in player.field:
                    if c.is_rest:
                        continue
                    if (c.master.type == CardType.CHARACTER and c.is_newly_played
                            and not c.has_keyword("速攻")):
                        continue
                    if "ATTACK_DISABLE" in c.flags or "ATTACK_DISABLE" in c.timed_flags:
                        continue
                    attackers.append(c)
            targets = []
            if opponent.leader:
                targets.append(opponent.leader)
            for c in opponent.field:
                if c.is_rest:
                    targets.append(c)
            for atk in attackers:
                for tgt in targets:
                    moves.append({"kind": "game", "action_type": "ATTACK",
                                  "payload": {"uuid": atk.uuid, "target_ids": [tgt.uuid]}})
            # ドン!!付与（アクティブな自分のリーダー/キャラへ）
            if don_active > 0:
                for c in attackers + [c for c in player.field if c.is_rest]:
                    moves.append({"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": c.uuid}})
            # 起動メイン効果。能力を「持つ」だけでなく、実際に発動が成立する
            # （条件成立・ターン使用回数未消費・コスト充足）ものだけを合法手に積む。
            # resolve_ability と同じ三条件で判定するため、レスト済みハチノス(OP09-099)等の
            # 「撃っても何も起きない no-op 起動メイン」を CPU 探索/プレイヤー双方から除外できる
            # （従来はコスト/回数を見ずに列挙していたため、CPU が同一ステージの起動メインを
            #   連打して 1 ターンを空費していた）。
            units = ([player.leader] if player.leader else []) + list(player.field)
            if player.stage:
                units.append(player.stage)
            _am_resolver = EffectResolver(self)
            for c in units:
                if c.is_effect_negated or getattr(c, "negated", False):
                    continue
                if self._has_activatable_main(c, player, _am_resolver):
                    moves.append({"kind": "game", "action_type": "ACTIVATE_MAIN", "payload": {"uuid": c.uuid}})
            # ターン終了は常に合法
            moves.append({"kind": "game", "action_type": "TURN_END", "payload": {}})
            return moves

        # 効果対話（SEARCH_AND_SELECT / CHOICE / CONFIRM_OPTIONAL / ARRANGE_DECK /
        # DECLARE_COST / SELECT_RESOURCE 等）は既定解決を1手として返す。
        payload = self.default_interaction_payload(pending)
        moves.append({"kind": "game", "action_type": RESOLVE, "payload": payload})
        return moves

    def _has_activatable_main(self, card, player, resolver: Optional[EffectResolver] = None) -> bool:
        """card が今「実際に発動成立する」起動メイン能力を 1 つ以上持つか。

        resolve_ability(resolver.py) と同じ三条件で判定する:
          1. 条件成立（ability.condition）
          2. ターン使用回数が未消費（【ターン1回】等）
          3. コスト充足（ability.cost。自己レスト等はレスト済みだと払えない）
        いずれも満たさない起動メインは発動しても no-op になるため、合法手から除外する。
        効果側の対象有無は resolve_ability も起動時には事前判定しないので、ここでも見ない
        （正規の起動を過剰に削らないため、判定はコストまでに留める）。
        """
        if resolver is None:
            resolver = EffectResolver(self)
        for ab in card.master.abilities:
            if ab.trigger != TriggerType.ACTIVATE_MAIN:
                continue
            if ab.condition is not None and not resolver._check_condition(player, ab.condition, card):
                continue
            lim = _condition_turn_limit(getattr(ab, "condition", None))
            if lim is not None and card.ability_used_this_turn.get(_ability_index(card, ab), 0) >= lim:
                continue
            if ab.cost is not None and not resolver._can_satisfy_node(player, ab.cost, card):
                continue
            return True
        return False

    def default_interaction_payload(self, pending: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return _interaction.default_interaction_payload(self, pending)

    def _find_card_by_uuid(self, uuid: str) -> Optional[CardInstance]:
        return _card_moves._find_card_by_uuid(self, uuid)

    def get_pending_request(self) -> Optional[Dict[str, Any]]:
        return _interaction.get_pending_request(self)

    def pending_actor_action(self) -> Optional[Tuple[str, str]]:
        return _interaction.pending_actor_action(self)

    def resolve_interaction(self, player: Player, payload: Dict[str, Any]):
        return _interaction.resolve_interaction(self, player, payload)

    def refresh_passive_state(self) -> None:
        return _passives.refresh_passive_state(self)

    def _enforce_field_limit(self, owner: Player) -> None:
        return _card_moves._enforce_field_limit(self, owner)

    def _suspend_for_field_overflow(self, owner: Player) -> None:
        return _card_moves._suspend_for_field_overflow(self, owner)

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
        
        return _turn_flow.start_game(self, first_player)

    def do_mulligan(self, player: 'Player') -> None:
        return _turn_flow.do_mulligan(self, player)

    def keep_hand(self, player: 'Player') -> None:
        return _turn_flow.keep_hand(self, player)

    def _check_mulligan_complete(self) -> None:
        return _turn_flow._check_mulligan_complete(self)

    def finish_setup(self):
        return _turn_flow.finish_setup(self)

    def end_turn(self):
        return _turn_flow.end_turn(self)

    def _fire_turn_end_triggers(self):
        return _turn_flow._fire_turn_end_triggers(self)

    def _flush_pending_end_of_turn(self):
        return _turn_flow._flush_pending_end_of_turn(self)

    def _record_event_played(self, card):
        """イベントカード発動をターン内イベントとして記録する（「コストN以上のイベントを発動している」
        OP15-002）。コスト k 以上の条件が拾えるよう、1..元々コスト までのしきい値を記録する。"""
        cost = getattr(card.master, "cost", None) or 0
        self.record_turn_event("EVENT_PLAYED", 1)
        for k in range(1, cost + 1):
            self.record_turn_event(f"EVENT_PLAYED_COST_GE_{k}", 1)

    def record_turn_event(self, name: str, n: int = 1):
        """このターン中に発生したイベントを記録する（EVENT_THIS_TURN 条件で参照）。"""
        ev = getattr(self, "_turn_events", None)
        if ev is None:
            ev = self._turn_events = JournaledDict()
        ev[name] = ev.get(name, 0) + n

    def switch_turn(self):
        # ターンが切り替わる/追加ターンに入る = 新しいターン。ターン内イベント記録をクリアする。
        return _turn_flow.switch_turn(self)

    def refresh_phase(self):
        return _turn_flow.refresh_phase(self)

    def _reset_player_status(self, player: Player):
        # 相手ターン開始時に直前のターンプレイヤー(=現opponent)の一時効果を解除するが、
        # 付与ドン!!は剥がさない（持ち主の次のリフレッシュフェイズまでカードに残る）。
        return _turn_flow._reset_player_status(self, player)

    def refresh_all(self, player: Player):
        return _turn_flow.refresh_all(self, player)

    def draw_phase(self):
        return _turn_flow.draw_phase(self)

    def don_phase(self):
        return _turn_flow.don_phase(self)

    def main_phase(self): 
        return _turn_flow.main_phase(self)

    _REACTIVE_RE = re.compile(r'(された|した|受けた|なった)時、')

    def _is_reactive_passive(self, ability) -> bool:
        return _passives._is_reactive_passive(self, ability)

    def _find_first_action(self, node):
        return _passives._find_first_action(self, node)

    def _apply_passive_effects(self, player: Player):
        # 対話中断中は再計算しない。Step1 のリセットは無条件に走る一方、Step2/3 の
        # resolve_ability は active_interaction ガードで何も実行できず、リセットだけが
        # 残って PASSIVE/YOUR_TURN バフが消えてしまうため（クザンのコスト-5 等）。
        return _passives._apply_passive_effects(self, player)

    def _apply_hand_self_cost(self, player: Player, opponent: Player):
        return _passives._apply_hand_self_cost(self, player, opponent)

    def draw_card(self, player: Player, count: int = 1):
        return _card_moves.draw_card(self, player, count)

    def _find_card_location(self, card: Card) -> Tuple[Optional[Player], Optional[List[Any]]]:
        return _card_moves._find_card_location(self, card)

    def move_card(self, card: Card, dest_zone: Zone, dest_player: Player, dest_position: str = "BOTTOM"):
        return _card_moves.move_card(self, card, dest_zone, dest_player, dest_position)

    def pay_cost(self, player: Player, cost: int, don_list: Optional[List[DonInstance]] = None):
        return _card_moves.pay_cost(self, player, cost, don_list)

    def has_blocker(self, player: Player) -> bool:
        return _battle.has_blocker(self, player)

    def declare_attack(self, attacker: Card, target: Card):
        return _battle.declare_attack(self, attacker, target)

    def _advance_battle_triggers(self):
        return _battle._advance_battle_triggers(self)

    # --- 誘発能力（ライフ公開【トリガー】/ON_LIFE_DECREASE 等）の汎用待ち行列 ---
    def _enqueue_trigger(self, player: Player, ability: Ability, card: CardInstance,
                         optional: bool = False) -> None:
        return _triggers._enqueue_trigger(self, player, ability, card, optional)

    def _advance_pending_triggers(self) -> None:
        return _triggers._advance_pending_triggers(self)

    def _relocate_activated_trigger_card(self, item: Dict[str, Any]) -> None:
        return _triggers._relocate_activated_trigger_card(self, item)

    def _suspend_for_trigger_confirm(self, item: Dict[str, Any]) -> None:
        return _triggers._suspend_for_trigger_confirm(self, item)

    def handle_block(self, blocker: Optional[Card] = None):
        return _battle.handle_block(self, blocker)

    def apply_counter(self, player: Player, counter_card: Optional[Card] = None, don_list: Optional[List[DonInstance]] = None):
        return _battle.apply_counter(self, player, counter_card, don_list)

    def resolve_attack(self):
        return _battle.resolve_attack(self)

    def _finish_attack(self, target: Card, target_owner: Player, life_lost: int):
        return _battle._finish_attack(self, target, target_owner, life_lost)

    def _suspend_for_battle_ko_replacement(self, target: Card, target_owner: Player, life_lost: int):
        return _battle._suspend_for_battle_ko_replacement(self, target, target_owner, life_lost)

    def check_victory(self):
        # デッキアウト: 通常は本人の敗北（相手の勝利）。ただし C10「自分のデッキが0枚に
        # なった場合、敗北する代わりに勝利する」(VICTORY/REPLACE_DECKOUT_LOSS) を持つ場合は
        # 本人の勝利へ置換する（OP03-040 ナミ等）。
        return _battle.check_victory(self)

    def _has_deckout_win_replace(self, player) -> bool:
        return _battle._has_deckout_win_replace(self, player)

    def _event_has_main_play(self, card: Card) -> bool:
        """イベントがメインフェイズに手札から発動できるか＝【メイン】効果
        （ON_PLAY/ACTIVATE_MAIN トリガー）を1つ以上持つか。【カウンター】/【トリガー】
        のみのイベントは False（メインでは発動不可）。"""
        return any(ab.trigger in (TriggerType.ON_PLAY, TriggerType.ACTIVATE_MAIN)
                   for ab in (card.master.abilities or ()))

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
        if card.master.type == CardType.EVENT:
            # 【メイン】効果を持たないイベント（【カウンター】/【トリガー】のみ）は
            # メインフェイズに手札から発動できない（カウンターは防御時の SELECT_COUNTER、
            # トリガーはライフ公開時のみ）。従来はコストさえ払えれば列挙・実行され、
            # ゴムゴムの巨人 OP09-078（カウンター専用）等を自ターンに空撃ちできていた。
            if not self._event_has_main_play(card):
                raise ValueError("このイベントはメインフェイズに発動できません（【メイン】効果を持ちません）。")
            self._record_event_played(card)   # 「このターン中…イベントを発動」条件用（OP15-002）
            for ability in card.master.abilities:
                if ability.trigger in [TriggerType.ON_PLAY, TriggerType.ACTIVATE_MAIN]:
                    self.resolve_ability(player, ability, source_card=card)
            self.move_card(card, Zone.TRASH, player)
        else:
            self.move_card(card, Zone.FIELD, player); card.attached_don = 0; card.is_newly_played = True
            # 【トリガー】を持つキャラの登場をターン内イベントとして記録（OP13-100「自分の【トリガー】を
            # 持つキャラが登場した時」）。trigger_text 非空 または TriggerType.TRIGGER 能力を持つ。
            if (getattr(card.master, "trigger_text", "") or any(
                    ab.trigger == TriggerType.TRIGGER for ab in (card.master.abilities or ()))):
                self.record_turn_event("TRIGGER_CHAR_PLAYED", 1)
            # 登場した時点で継続効果（PASSIVE/YOUR_TURN）を適用してから ON_PLAY を解決する。
            # 例: クザン「相手のキャラすべてをコスト-5」+【登場時】コスト0のキャラをKO —
            # 自身の継続効果が ON_PLAY の対象判定に反映される必要がある。
            self._apply_passive_effects(self.turn_player)
            if self._has_rested_play(player):  # 「自分のキャラはレストで登場する」PASSIVE
                card.is_rest = True
            # 場のキャラ上限超過の押し出しは ON_PLAY 解決より前に確定する（実ルールでは
            # 6枚目のキャラは並存しない）。押し出し選択で中断した場合、ON_PLAY は誘発待ち
            # 行列へ積まれ、押し出し確定後（FIELD_OVERFLOW_TRASH 解決時）に消化される。
            self._enforce_field_limit(player)
            # 「相手の登場時効果は無効になる」(OPP_ONPLAY) 期間中はこのプレイヤーの ON_PLAY を解決しない。
            onplay_negated = getattr(player, "negate_onplay_until", 0) >= self.turn_count
            if onplay_negated:
                pass
            if not card.is_effect_negated and not onplay_negated:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_PLAY:
                        if self.active_interaction:
                            self._enqueue_trigger(player, ability, card, optional=False)
                        else:
                            self.resolve_ability(player, ability, source_card=card)
            # 他カードの「…が登場した時」リスナー（OP14-041 等）。登場時無効(OPP_ONPLAY)は
            # 登場カード自身の【登場時】のみを無効にするため、リスナーは無効化に関わらず積む。
            self._enqueue_char_played_listeners(card, player, from_zone="HAND")
            self._apply_passive_effects(player)
            # ON_PLAY がさらにキャラを登場させた場合の超過はここで拾う（中断中は no-op し、
            # 対話完了時に resolve_interaction 末尾が拾う）。
            self._enforce_field_limit(player)

    def _has_rested_play(self, player: Player) -> bool:
        return _guards._has_rested_play(self, player)

    def _active_restriction(self, player: Player, key: str) -> Optional[Dict[str, Any]]:
        return _guards._active_restriction(self, player, key)

    def _blocks_effect_play(self, card: CardInstance) -> bool:
        return _guards._blocks_effect_play(self, card)

    def resolve_ability(self, player: Player, ability: Ability, source_card: CardInstance,
                        cost_confirmed: bool = False):
        if source_card.negated or source_card.is_effect_negated: return
        resolver = EffectResolver(self)
        resolver.resolve_ability(player, ability, source_card, cost_confirmed=cost_confirmed)
        # PASSIVE 再計算（_apply_passive_effects）中の継続効果の再適用は、盤面操作のたびに
        # 同じバフを何度も載せ直す内部処理にすぎない（結果はカードの cost/power に反映済み）。
        # その都度 EFFECT イベントを積むと、ティーチ OP16-080 の【相手のターン中】コスト+1 等が
        # eventLog/リプレイを同一イベントで膨張させる（挙動は不変＝コストはスタックしない）。
        # 再計算中はイベント発行を抑制する（本物の発動＝非再計算経路は従来どおり記録）。
        if getattr(self, "_in_passive_recalc", False):
            return
        for ev in resolver.action_history:
            self.action_events.append({
                "type": "EFFECT",
                "player": player.name,
                "card_name": source_card.master.name,
                "action": ev.get("action", ""),
                "targets": ev.get("targets", []),
                "value": ev.get("value"),
                "success": ev.get("success", True),
                # 移動系の行き先（"LIFE"/"HAND" 等・無い action は省略）。フロントの
                # eventLog が「ライフに加えた」等の意味を表示するために使う（additive）。
                **({"dest": ev["dest"]} if ev.get("dest") else {}),
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

    def _active_protection(self, card: CardInstance, status_values: Tuple[str, ...], actor: Optional[Player] = None, attacker: Optional[CardInstance] = None) -> bool:
        return _guards._active_protection(self, card, status_values, actor, attacker)

    # 置換効果（REPLACE_EFFECT）の検出。除去対象に適用可能な「代わりに〜」置換を1件、
    # 条件・実行可能性・【ターン1回】の残数を満たすものに限り (protector, ability, eff, sub)
    # として返す（実行はしない）。無ければ None。検出と実行を分離することで、バトルKO置換の
    # ような「任意（〜してもよい/できる）の置換を被KO側へ確認してから実行/拒否する」経路を可能にする。
    def _find_replacement(self, card: CardInstance, status_values: Tuple[str, ...]):
        return _guards._find_replacement(self, card, status_values)

    def _register_granted_replacements(self, player: Player, source_card: Card) -> None:
        return _guards._register_granted_replacements(self, player, source_card)

    # 置換効果（REPLACE_EFFECT）の判定。除去の瞬間に対象の PASSIVE 能力を走査し、
    # 「代わりに〜」の置換アクションを（条件・実行可能性を満たせば）実行して True を返す。
    # True の場合、呼び出し側は本来の除去を行わずスキップする。
    def _active_replacement(self, card: CardInstance, status_values: Tuple[str, ...],
                            can_suspend: bool = False) -> bool:
        return _guards._active_replacement(self, card, status_values, can_suspend)

    def _auto_resolve_replacement(self, owner: Player, limit: int = 16) -> None:
        return _guards._auto_resolve_replacement(self, owner, limit)

    # ------------------------------------------------------------------
    # 多段継続（deferred continuations）— accepted limitation B の解消
    # 入れ子の除去置換が内側中断（対象選択／任意確認）を UI へ提示したとき、失われる外側の
    # 継続（後続シーケンス／除去ループの残対象）を退避し、内側中断の解決後に LIFO で再開する。
    # 退避順は「残対象（append=上）→ 後続シーケンス（insert(0)=下）」なので、pop() で
    # 残対象 → 後続シーケンスの順に正しく再開される。
    # ------------------------------------------------------------------
    def _defer_resolver_stack(self, player: Player, source_card, execution_stack, context) -> None:
        return _interaction._defer_resolver_stack(self, player, source_card, execution_stack, context)

    def _defer_removal_targets(self, player: Player, action, remaining_targets, value) -> None:
        return _interaction._defer_removal_targets(self, player, action, remaining_targets, value)

    def _resume_deferred_continuations(self, limit: int = 64) -> None:
        return _interaction._resume_deferred_continuations(self, limit)

    def _ko_trigger_matches(self, ability: Ability, owner: Player,
                            cause: str, effect_controller: Player = None) -> bool:
        return _triggers._ko_trigger_matches(self, ability, owner, cause, effect_controller)

    def _resolve_on_ko(self, card: Card, owner: Player,
                       cause: str = "EFFECT", effect_controller: Player = None):
        # このターンに当該プレイヤーのキャラが KO された事実を記録する
        # （「このターン中、相手のキャラがKOされている場合」OP16-100 の判定用）。
        return _triggers._resolve_on_ko(self, card, owner, cause, effect_controller)

    def _rest_subject_matches(self, ability: Ability, rested_card: Card, host: Card,
                              host_owner: Player, by_attack: bool,
                              effect_controller: Player = None, cause_source: Card = None) -> bool:
        return _triggers._rest_subject_matches(self, ability, rested_card, host, host_owner, by_attack, effect_controller, cause_source)

    def _fire_on_rest_triggers(self, rested_card: Card, by_attack: bool,
                               effect_controller: Player = None, cause_source: Card = None):
        return _triggers._fire_on_rest_triggers(self, rested_card, by_attack, effect_controller, cause_source)

    def _leave_subject_matches(self, ability: Ability, leaving_card: Card,
                               ability_owner: Player, leaving_owner: Player) -> bool:
        return _triggers._leave_subject_matches(self, ability, leaving_card, ability_owner, leaving_owner)

    def _enqueue_on_leave(self, leaving_card: Card, leaving_owner: Player) -> None:
        return _triggers._enqueue_on_leave(self, leaving_card, leaving_owner)

    def _played_subject_matches(self, ability: Ability, holder_owner: Player,
                                played_card: Card, played_owner: Player,
                                from_zone: str = None) -> bool:
        return _triggers._played_subject_matches(self, ability, holder_owner,
                                                 played_card, played_owner, from_zone)

    def _enqueue_char_played_listeners(self, played_card: Card, played_owner: Player,
                                       from_zone: str = None) -> None:
        return _triggers._enqueue_char_played_listeners(self, played_card, played_owner, from_zone)

    def _ko_listener_matches(self, ability: Ability, holder_owner: Player,
                             koed_card: Card, koed_owner: Player) -> bool:
        return _triggers._ko_listener_matches(self, ability, holder_owner, koed_card, koed_owner)

    def _enqueue_ko_listeners(self, koed_card: Card, koed_owner: Player) -> None:
        return _triggers._enqueue_ko_listeners(self, koed_card, koed_owner)

    def _fire_turn_start_triggers(self) -> None:
        return _turn_flow._fire_turn_start_triggers(self)

    def _enqueue_life_decrease(self, player: Player, count: int = 1) -> None:
        return _triggers._enqueue_life_decrease(self, player, count)

    def _fire_on_life_decrease(self, player: Player, count: int = 1):
        return _triggers._fire_on_life_decrease(self, player, count)

    def _return_one_don(self, tp: Player, don: DonInstance) -> bool:
        return _card_moves._return_one_don(self, tp, don)

    def _don_pool_player(self, player: Player, action: GameAction) -> Player:
        return _card_moves._don_pool_player(self, player, action)

    def apply_action_to_engine(self, player: Player, action: GameAction, targets: List[CardInstance], value: int, source_card: Optional[CardInstance] = None) -> bool:
        # アクション適用はレジストリ・ディスパッチ（core/actions）へ委譲する。プレイヤーレベル・
        # アクションは actions.player_level のハンドラ、対象ループは actions.target_loop.run_target_loop
        # が担う（公開シグネチャ・挙動は不変）。
        return _apply_action(self, player, action, targets, value, source_card)

    def get_dynamic_value(self, player: Player, val_source: ValueSource, targets: List[CardInstance], context: Dict) -> int:
        return _values.get_dynamic_value(self, player, val_source, targets, context)

    def _resolve_power_reference(self, player, ref_id, context):
        return _values._resolve_power_reference(self, player, ref_id, context)