from typing import List, Optional, Any, Tuple, Dict, Set
import random
import unicodedata
import re
import traceback
import uuid
import json
import hashlib
from ..models.models import CardInstance, CardMaster, DonInstance, CONST
from . import journal
from .journal import JournaledList, JournaledDict, JournaledSet, record_attr
from ..models.enums import CardType, Attribute, Color, Phase, Zone, TriggerType, ConditionType, CompareOperator, ActionType, PendingMessage
from ..models.effect_types import TargetQuery, Ability, GameAction, ValueSource, Sequence, Branch, Choice
from .effects.resolver import EffectResolver
from .effects.matcher import get_target_cards
from .actions import apply_action as _apply_action
from .engine import values as _values, guards as _guards
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
        """効果対話に対する「妥当な既定解決」のペイロードを構築する。

        本番（自己対戦/CPU）でも使える機械的な既定選択:
          - 必要最小数 (constraints.min) を満たすよう候補の先頭から選ぶ
          - can_skip なら 0 件選択（スキップ）も可だが、min>0 のときは min 件選ぶ
          - CHOICE/CONFIRM 系は index=0（最初の選択肢/発動する）
        AI（PR2）は本メソッドを評価関数で上書きして最良選択を選ぶ。
        """
        if pending is None:
            pending = self.get_pending_request() or {}
        pending_props = CONST.get('PENDING_REQUEST_PROPERTIES', {})
        KEY_UUIDS = pending_props.get('SELECTABLE_UUIDS', 'selectable_uuids')
        KEY_CONSTRAINTS = pending_props.get('CONSTRAINTS', 'constraints')
        uuids = list(pending.get(KEY_UUIDS, []) or [])
        constraints = pending.get(KEY_CONSTRAINTS) or {}
        try:
            min_n = int(constraints.get("min", 0))
        except (TypeError, ValueError):
            min_n = 0
        try:
            max_n = int(constraints.get("max", len(uuids)))
        except (TypeError, ValueError):
            max_n = len(uuids)
        take = max(min_n, 0)
        take = min(take, max_n, len(uuids))
        selected = uuids[:take]
        return {
            "selected_uuids": selected,
            "index": 0,
            "accepted": True,
            "position": "BOTTOM",
            "declared_value": 0,
        }

    def _find_card_by_uuid(self, uuid: str) -> Optional[CardInstance]:
        return _card_moves._find_card_by_uuid(self, uuid)

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

        def _rid(d: Dict[str, Any]) -> str:
            # request_id は「同一の要求なら安定・要求が変われば変化」する決定的ハッシュにする。
            # 従来は get のたびに uuid4 を再生成しており、フロントの『request_id 変化＝新要求』検知が
            # 毎ポーリング/WS更新で誤発火していた（機能バグ）。入力側で request_id は未使用＝安全に変更可。
            #
            # ハッシュは**要求の全内容（request_id 自身を除く）＋turn_count**を正規化 JSON で取る。
            # player_id/action/message/selectable_uuids だけでは、同一ターン内に連続する
            # 別要求（同名カード2枚が各々出す同文の確認、options だけ異なる CHOICE、
            # revealed view だけ異なる探索など）が衝突し、フロントが 2 件目を新要求と認識できず
            # モーダル再マウント/選択リセットが起きない。source_card_uuid・options・constraints・
            # candidates など識別に効く全フィールドを取り込むことでこれを防ぐ。
            # 盤面不変なら d は各 to_dict()/スカラが決定的なので同一 → rid も安定（＝元の修正意図を維持）。
            payload = {k: v for k, v in d.items() if k != "request_id"}
            key = json.dumps([self.turn_count, payload],
                             sort_keys=True, ensure_ascii=False, default=str)
            return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

        # マリガンは先行プレイヤー(turn_player)から順に要求する。
        if self.phase == Phase.MULLIGAN:
            mulligan_order = ([self.turn_player, self.opponent]
                              if self.turn_player and self.opponent else [self.p1, self.p2])
            for player in mulligan_order:
                if player.name not in self.mulligan_done:
                    hand_candidates = [c.to_dict() for c in player.hand]
                    _mreq = {
                        KEY_PID: player.name,
                        KEY_ACTION: "MULLIGAN",
                        KEY_MSG: "マリガンするカードを選んでください（交換なし＝キープ）",
                        KEY_CANDIDATES: hand_candidates,
                        KEY_UUIDS: [c.uuid for c in player.hand],
                        KEY_CONSTRAINTS: {"min": 0, "max": len(player.hand)},
                        KEY_SKIP: True,
                    }
                    _mreq["request_id"] = _rid(_mreq)
                    return _mreq
            return None

        if self.active_interaction:
            action_type = self.active_interaction.get("action_type")
            fe_action = "SEARCH_AND_SELECT" if action_type in ("SELECT_TARGET", "FIELD_OVERFLOW_TRASH") else action_type
            
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
            }
            # 効果の発生源カードを UI で表示できるよう uuid を併せて渡す。
            src_uuid = self.active_interaction.get("source_card_uuid")
            if src_uuid:
                req[pending_props.get('SOURCE_CARD_UUID', 'source_card_uuid')] = src_uuid
            # ARRANGE_DECK(並び替え/上下選択)はフロントの UI 切替フラグを併せて渡す。
            if action_type == "ARRANGE_DECK":
                req["allow_position"] = self.active_interaction.get("allow_position", False)
                req["allow_reorder"] = self.active_interaction.get("allow_reorder", False)
            req["request_id"] = _rid(req)
            return req

        if not self.active_battle and self.phase in [Phase.BLOCK_STEP, Phase.BATTLE_COUNTER]:
            self.phase = Phase.MAIN
            
        request = None
        ACT_BLOCKER = battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
        ACT_COUNTER = battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER')
        
        if self.phase == Phase.BLOCK_STEP and self.active_battle:
            target_owner = self.active_battle["target_owner"]
            blockers = [c.uuid for c in target_owner.field if not c.is_rest and c.has_keyword("ブロッカー") and "CANNOT_REST" not in c.timed_flags]
            request = {KEY_PID: target_owner.name, KEY_ACTION: ACT_BLOCKER, KEY_MSG: PendingMessage.SELECT_BLOCKER.value, KEY_UUIDS: blockers, KEY_SKIP: True}
        elif self.phase == Phase.BATTLE_COUNTER and self.active_battle:
            target_owner = self.active_battle["target_owner"]
            counters = [c.uuid for c in target_owner.hand if c.current_counter > 0 or (c.master.type == CardType.EVENT and any(abil.trigger == TriggerType.COUNTER for abil in c.master.abilities))]
            request = {KEY_PID: target_owner.name, KEY_ACTION: ACT_COUNTER, KEY_MSG: PendingMessage.SELECT_COUNTER.value, KEY_UUIDS: counters, KEY_SKIP: True}
        elif self.phase == Phase.MAIN:
            selectable = [c.uuid for c in self.turn_player.hand]
            selectable += [c.uuid for c in self.turn_player.field if not c.is_rest]
            if self.turn_player.leader and not self.turn_player.leader.is_rest:
                selectable.append(self.turn_player.leader.uuid)
            request = {KEY_PID: self.turn_player.name, KEY_ACTION: "MAIN_ACTION", KEY_MSG: PendingMessage.MAIN_ACTION.value, KEY_UUIDS: selectable, KEY_SKIP: True}
        if request is not None:
            request["request_id"] = _rid(request)
        return request

    def pending_actor_action(self) -> Optional[Tuple[str, str]]:
        """`get_pending_request()` の (player_id, action) **だけ**を安価に返す（CPU 探索の葉/手番判定用）。

        探索は各ノードでこの 2 値しか見ない（手は `get_legal_actions` から得る）一方、
        `get_pending_request` は毎回 selectable 構築・候補 to_dict・request_id ハッシュ（要求全体の
        正規化 JSON を sha1）を作るため重い（探索コストの ~12%）。本メソッドは**判定ロジックと
        副作用（BLOCK_STEP/BATTLE_COUNTER で
        active_battle が無いときの phase→MAIN 正規化）を get_pending_request と一致**させたうえで、
        重い payload を作らない。一致は `tests/test_cpu_make_unmake.py` で機械照合する。
        """
        if self.phase == Phase.MULLIGAN:
            order = ([self.turn_player, self.opponent]
                     if self.turn_player and self.opponent else [self.p1, self.p2])
            for p in order:
                if p.name not in self.mulligan_done:
                    return (p.name, "MULLIGAN")
            return None
        if self.active_interaction:
            at = self.active_interaction.get("action_type")
            fe = "SEARCH_AND_SELECT" if at in ("SELECT_TARGET", "FIELD_OVERFLOW_TRASH") else at
            return (self.active_interaction.get("player_id"), fe)
        if not self.active_battle and self.phase in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER):
            self.phase = Phase.MAIN  # get_pending_request と同じ副作用
        battle_actions = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
        if self.phase == Phase.BLOCK_STEP and self.active_battle:
            return (self.active_battle["target_owner"].name, battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER'))
        if self.phase == Phase.BATTLE_COUNTER and self.active_battle:
            return (self.active_battle["target_owner"].name, battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER'))
        if self.phase == Phase.MAIN:
            return (self.turn_player.name, "MAIN_ACTION")
        return None

    def resolve_interaction(self, player: Player, payload: Dict[str, Any]):
        if not self.active_interaction:
            return
            
        continuation = self.active_interaction.get("continuation")
        if not continuation:
            self.active_interaction = None
            return

        action_type = self.active_interaction.get("action_type")

        # 誘発能力の発動確認（CONFIRM_TRIGGER）: continuation は trigger_item のみで
        # source_card_uuid を持たないため、汎用 source 解決より先に処理する。
        if action_type == "CONFIRM_TRIGGER":
            item = continuation.get("trigger_item")
            accepted = payload.get("accepted")
            if accepted is None:
                if payload.get("skip") is True or payload.get("declined") is True:
                    accepted = False
                else:
                    accepted = payload.get("index", 0) == 0
            self.active_interaction = None
            if item is not None:
                if accepted:
                    item["_confirmed"] = True  # 先頭のまま再投入 → 解決へ
                elif item in self._pending_triggers:
                    self._pending_triggers.remove(item)
            self._advance_pending_triggers()
            if not self.active_interaction and self.active_battle \
                    and self.phase not in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER):
                self._advance_battle_triggers()
            return

        # 場のキャラ上限超過の強制トラッシュ。発生源カードを持たない（ルール処理）ため、
        # 汎用 source 解決より先に処理する。選んだキャラをトラッシュ（KOではないので
        # 「KO時」誘発は起こさない）。
        if action_type == "FIELD_OVERFLOW_TRASH":
            owner = self.p1 if self.p1.name == continuation.get("owner_name") else self.p2
            selected = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
            self.active_interaction = None
            for uid in selected:
                card = next((c for c in owner.field if c.uuid == uid), None)
                if card:
                    self.move_card(card, Zone.TRASH, owner)
            self.refresh_passive_state()
            # 複数体同時超過などでまだ超過していれば再度要求する（保険）。
            if len(owner.field) > FIELD_LIMIT:
                self._suspend_for_field_overflow(owner)
            return

        source_uuid = continuation["source_card_uuid"]
        source_card = self._find_card_by_uuid(source_uuid)
        if not source_card:
            self.active_interaction = None
            return

        resolver = EffectResolver(self)
        
        if action_type == "SELECT_TARGET":
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

        elif action_type == "SELECT_RESOURCE":
            # ドン!!返却(RETURN_DON)の対象ドン!!選択。選んだ uuid を context に載せて再開すると、
            # RETURN_DON 再実行時に当該ドン!!を戻す。
            selected_uuids = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
            effect_context = continuation.get("effect_context", {})
            effect_context["_return_don_uuids"] = selected_uuids
            self.active_interaction = None
            # RETURN_DON は効果の責任者（source_card の持ち主）視点で再実行する。
            # 「相手は自身の場のドン!!を戻す」（status=OPPONENT）では選択者＝相手だが、
            # _don_pool_player は player を基準に相手プールを引くため、応答者(相手)で再開すると
            # 相手の相手=自分のプールを指して空振りする。責任者基準なら選んだ相手ドンが正しく戻る。
            controller = self.p1 if self.p1.name == source_card.owner_id else self.p2
            resolver.resume_execution(controller, source_card, continuation.get("execution_stack", []), effect_context)

        elif action_type == "CHOICE":
            selected_index = payload.get("index", payload.get("selected_option_index", 0))

            resolver.resume_choice(player, source_card, selected_index, continuation.get("execution_stack", []), continuation.get("effect_context", {}))

        elif action_type == "CONFIRM_OPTIONAL":
            # 任意効果（「〜してもよい」）/ 任意コスト能力（「〜できる：」）の発動可否。
            # accepted=False（パス/拒否）ならスキップ。
            accepted = payload.get("accepted")
            if accepted is None:
                # selected_uuids 非空 / index>0 / skip フラグ等から推定（既定は発動=True）
                if payload.get("skip") is True or payload.get("declined") is True:
                    accepted = False
                else:
                    accepted = payload.get("index", 0) == 0
            # 任意バトルKO置換（A）の確認: accept→置換実行（本来のKOをスキップ）、
            # decline→本来のKOを実行。どちらも _finish_attack で戦闘後処理して return。
            if continuation.get("kind") == "BATTLE_KO_REPLACE":
                self.active_interaction = None
                target = source_card
                target_owner = self.p1 if self.p1.name == continuation.get("target_owner_name") else self.p2
                life_lost = continuation.get("life_lost", 0)
                if accepted and self._active_replacement(target, ("BATTLE_KO",)):
                    pass
                else:
                    # 拒否、または置換が成立しなくなった場合は本来の KO を進める。
                    self.move_card(target, Zone.TRASH, target_owner)
                    self._resolve_on_ko(target, target_owner, cause="BATTLE")
                self._finish_attack(target, target_owner, life_lost)
                return
            self.active_interaction = None
            confirm_ability = continuation.get("confirm_ability")
            if confirm_ability is not None:
                # 任意コスト能力（A-3）の使用確認: accept で cost_confirmed=True で再入。
                # decline は何もしない（使用回数も未消費）。gamestate 経由で action_events を記録。
                if accepted:
                    self.resolve_ability(player, confirm_ability, source_card, cost_confirmed=True)
            else:
                optional_node = continuation.get("optional_node")
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
                tp.life = JournaledList(ordered + rest)
            else:
                # デッキ配置: BOTTOM は順に append（先頭が上）、TOP は逆順 insert(0) で
                # ordered[0] が最上面になるようにする。
                seq = ordered if position == "BOTTOM" else list(reversed(ordered))
                for c in seq:
                    owner, _ = self._find_card_location(c)
                    if owner:
                        self.move_card(c, Zone.DECK, owner, dest_position=position)
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
            else:
                pass
            self.active_interaction = None
            resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), effect_context)

        # 再開経路（resume_execution/resume_choice/resume_optional）で実行された
        # アクションも action_events へ記録する（resolve_ability 経由と同じ扱い。
        # 記録しないと中断を挟んだ効果が「何も実行していない」ように見える）。
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

        if not self.active_interaction and self.setup_phase_pending:
            self.finish_setup()
            self.setup_phase_pending = False
            self.phase = Phase.MULLIGAN
            self.mulligan_done = JournaledSet()

        # ライフ公開【トリガー】/ON_LIFE_DECREASE 等のペンディング誘発が残っていれば消化する。
        if not self.active_interaction and self._pending_triggers:
            self._advance_pending_triggers()

        # バトルトリガー(ON_ATTACK/ON_OPP_ATTACK)解決中の中断から復帰した場合:
        # バトルが進行中(active_battle あり)でまだ防御フェイズへ遷移していなければ、
        # 残りトリガーの解決＋フェイズ遷移を再開する（カウンター衝突エラーの防止）。
        if (not self.active_interaction and self.active_battle
                and self.phase not in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER)):
            self._advance_battle_triggers()

        # 入れ子の除去置換が中断したことで退避された外側継続（後続シーケンス／残対象）を、
        # 中断が解消された後に再開する（accepted limitation B = 多段継続の対話化）。
        # フィールド上限超過の処理より前に置き、継続完了後の最終盤面で超過判定する。
        if not self.active_interaction and self._deferred_continuations:
            self._resume_deferred_continuations()

        # ON_PLAY 等の対話が片付いた後で場のキャラ上限超過が残っていれば強制トラッシュ。
        # 誘発/バトル進行（上記）を横取りしないよう最後に置き、1プレイヤーずつ逐次化する。
        if not self.active_interaction:
            for pl in (self.p1, self.p2):
                if len(pl.field) > FIELD_LIMIT:
                    self._enforce_field_limit(pl)
                    break

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
        
        self.p1.shuffle_deck()
        self.p2.shuffle_deck()
        
        for p in [self.p1, self.p2]:
            if p.leader:
                for ability in p.leader.master.abilities:
                    if ability.trigger == TriggerType.GAME_START:
                        self.resolve_ability(p, ability, source_card=p.leader)
                        
                        if self.active_interaction:
                            self.setup_phase_pending = True
                            if first_player: self.turn_player = first_player; self.opponent = self.p2 if first_player == self.p1 else self.p1
                            else: self.turn_player = self.p1; self.opponent = self.p2
                            return

        self.finish_setup()

        if first_player: self.turn_player = first_player; self.opponent = self.p2 if first_player == self.p1 else self.p1
        else: self.turn_player = self.p1; self.opponent = self.p2
        # マリガンフェーズへ移行（両プレイヤーの確定後にゲーム開始）
        self.phase = Phase.MULLIGAN
        self.mulligan_done = JournaledSet()

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
        self._check_mulligan_complete()

    def keep_hand(self, player: 'Player') -> None:
        """手札をキープしてマリガンをスキップ"""
        if self.phase != Phase.MULLIGAN:
            raise ValueError("マリガンフェーズではありません。")
        if player.name in self.mulligan_done:
            raise ValueError("既にマリガンを実施済みです。")
        self.mulligan_done.add(player.name)
        self._check_mulligan_complete()

    def _check_mulligan_complete(self) -> None:
        """両プレイヤーのマリガン確定後にゲーム開始"""
        if self.p1.name in self.mulligan_done and self.p2.name in self.mulligan_done:
            self.turn_count = 1
            self.refresh_phase()

    def finish_setup(self):
        self.p1.place_life()
        self.p1.draw_initial_hand()
        self.p2.place_life()
        self.p2.draw_initial_hand()

    def end_turn(self):
        self._validate_action(self.turn_player, "MAIN_ACTION")
        self.phase = Phase.END
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
        self.pending_end_of_turn = JournaledList()
        for player, node, source_card in pending:
            # 場を離れたカードのソース由来でも、トラッシュ送り等は対象解決時に弾かれる。
            resolver = EffectResolver(self)
            resolver.context["_flushing_delayed"] = True
            resolver.execution_stack = [node]
            try:
                resolver._process_stack(player, source_card)
            except Exception as e:
                pass
            for ev in resolver.action_history:
                self.action_events.append({
                    "type": "EFFECT", "player": player.name,
                    "card_name": source_card.master.name,
                    "action": ev.get("action", ""), "targets": ev.get("targets", []),
                    "value": ev.get("value"), "success": ev.get("success", True),
                })

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
        self._turn_events = JournaledDict()
        # 追加ターン（EXTRA_TURN）: 予約したプレイヤーがターンプレイヤーのまま継続する
        if getattr(self, "pending_extra_turn", None) == self.turn_player.name:
            self.pending_extra_turn = None
            self.turn_count += 1
            self.refresh_phase()
            return
        self.turn_player, self.opponent = self.opponent, self.turn_player
        self.turn_count += 1
        self.refresh_phase()

    def refresh_phase(self):
        self._reset_player_status(self.opponent); self.refresh_all(self.turn_player); self.draw_phase()

    def _reset_player_status(self, player: Player):
        # 相手ターン開始時に直前のターンプレイヤー(=現opponent)の一時効果を解除するが、
        # 付与ドン!!は剥がさない（持ち主の次のリフレッシュフェイズまでカードに残る）。
        all_units = [player.leader] + player.field
        if player.stage: all_units.append(player.stage)
        for card in all_units:
            # ターン境界のリセット。【ターン1回】の使用回数もここで戻す。
            if card: card.reset_turn_status(keep_don=True, clear_usage=True)

    def refresh_all(self, player: Player):
        all_units = [player.leader] + player.field
        if player.stage: all_units.append(player.stage)
        for card in all_units:
            if card:
                is_frozen = "FREEZE" in card.flags
                # ターン境界のリセット。【ターン1回】の使用回数もここで戻す。
                card.reset_turn_status(clear_usage=True)
                if not is_frozen: card.is_rest = False
        
        # フリーズ中のドン!!（FREEZE_DON / OP07-026）は今回のリフレッシュではアクティブに
        # 戻さず、レストのまま据え置いてフラグを下ろす（1回限りのフリーズ）。
        still_frozen, to_activate = [], []
        for don in player.don_rested:
            if don.is_frozen:
                don.is_frozen = False
                still_frozen.append(don)
            else:
                don.is_rest = False
                to_activate.append(don)
        player.don_active.extend(to_activate)
        player.don_rested = JournaledList(still_frozen)
        
        for don in player.don_attached_cards:
            don.is_rest = False
            don.attached_to = None
            player.don_active.append(don)
        player.don_attached_cards = JournaledList()

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
        # 先攻・後攻ともに「自分の最初のターン」はリーダー・キャラのいずれもアタックできない（公式準拠）。
        # ターンは先攻=turn_count 1、後攻=turn_count 2 と交互に進むため、turn_count <= 2 が
        # 両プレイヤーの最初のターンを覆う。
        if self.turn_count <= 2:
            raise ValueError("最初のターンはアタックできません。")
        if "ATTACK_DISABLE" in attacker.flags or "ATTACK_DISABLE" in attacker.timed_flags: raise ValueError("このカードは効果によりアタックできません。")
        if "CANNOT_REST" in attacker.timed_flags: raise ValueError("このカードは効果によりレストにできないためアタックできません。")
        if attacker.is_rest: raise ValueError("アタックするカードはアクティブ状態でなければなりません。")
        # 召喚酔い: 登場したターンのキャラは攻撃できない。ただし「速攻」を持てば可。
        # リーダーは is_newly_played=False のため影響を受けない。
        if (attacker.master.type == CardType.CHARACTER
                and attacker.is_newly_played
                and not attacker.has_keyword("速攻")):
            raise ValueError("登場したターンのキャラクターは攻撃できません（速攻を除く）。")
        # 自己制限（self_cannot）:「リーダーにアタックできない」。相手リーダーへの攻撃宣言を弾く。
        if (target.master.type == CardType.LEADER
                and attacker_owner is not None
                and self._active_restriction(attacker_owner, "CANNOT_ATTACK_LEADER")):
            raise ValueError("効果により、このターンはリーダーにアタックできません。")
        if (target.master.type == CardType.CHARACTER and not target.is_rest
                and not attacker.has_keyword("ATTACK_ACTIVE")):
            raise ValueError("レスト状態のキャラクターのみ攻撃可能です。")
        # アタック税（OP08-043「アタックする際、自身の手札N枚を捨てなければアタックできない」）。
        # 付与された ATTACK_TAX_DISCARD_N フラグがあれば、手札N枚を支払えるときのみアタック可。
        tax_flags = [f for f in (attacker.flags | attacker.timed_flags)
                     if isinstance(f, str) and f.startswith("ATTACK_TAX_DISCARD_")]
        if tax_flags:
            need = max(int(f.rsplit("_", 1)[1]) for f in tax_flags)
            if len(attacker_owner.hand) < need:
                raise ValueError(f"アタックするには手札{need}枚を捨てる必要があり、手札が足りません。")
            # コスト支払い: 手札N枚を捨てる。どの札を捨てるかは本来プレイヤー選択だが、宣言経路を
            # 中断させないため先頭からN枚を捨てる（捨て札選択の対話化は今後の課題）。
            for _ in range(need):
                attacker_owner.trash.append(attacker_owner.hand.pop(0))
        attacker.is_rest = True
        self.active_battle = JournaledDict({"attacker": attacker, "target": target, "attacker_owner": attacker_owner, "target_owner": target_owner, "counter_buff": 0})

        # アタック時/相手のアタック時トリガーを順に解決する。途中でいずれかが対象選択や
        # 選択(Choice)で中断した場合、解決前にブロッカー/カウンター段階へ進むと、未解決の
        # interaction とカウンター操作が衝突する（"期待:CHOICE" エラー）。トリガーを待ち行列に
        # 積み、_advance_battle_triggers で1つずつ解決し、全て片付いてからフェイズ遷移する。
        triggers = []
        if attacker.master.abilities:
            for ability in attacker.master.abilities:
                if ability.trigger == TriggerType.ON_ATTACK:
                    triggers.append((attacker_owner, ability, attacker))
                # 「このキャラがレストになった時」(ON_REST) はアタック宣言で自身がレストになった
                # 瞬間に誘発する（要因＝アタックなので「効果で」限定の能力は対象外）。
                # OP14-119/027/028/032/035 等。CONTEXT/ターン1回条件は resolve_ability が評価。
                elif (ability.trigger == TriggerType.ON_REST
                      and self._rest_subject_matches(ability, attacker, attacker,
                                                     attacker_owner, by_attack=True)):
                    triggers.append((attacker_owner, ability, attacker))
        opp_cards = ([target_owner.leader] if target_owner.leader else []) + target_owner.field
        for card in opp_cards:
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.ON_OPP_ATTACK:
                    triggers.append((target_owner, ability, card))
        self._battle_triggers = JournaledList(triggers)
        self._advance_battle_triggers()

    def _advance_battle_triggers(self):
        """積んだバトルトリガーを順に解決し、全て解決後に防御フェイズへ遷移する。
        途中で interaction が立ったら中断（resolve_interaction が解決後に再度呼ぶ）。"""
        if not self.active_battle:
            self._battle_triggers = JournaledList()
            return
        while getattr(self, "_battle_triggers", None):
            player, ability, card = self._battle_triggers.pop(0)
            self.resolve_ability(player, ability, source_card=card)
            if self.active_interaction:
                return  # 中断: 解決後に resolve_interaction から再開される
        # 全トリガー解決 → ブロッカー/カウンター段階へ
        target_owner = self.active_battle["target_owner"]
        if self.has_blocker(target_owner):
            self.phase = Phase.BLOCK_STEP
        else:
            self.phase = Phase.BATTLE_COUNTER

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
        if not self.active_battle: return
        target_owner = self.active_battle["target_owner"]; self._validate_action(target_owner, "SELECT_BLOCKER")
        if blocker:
            blocker.is_rest = True
            self.active_battle["target"] = blocker
            # 【ブロック時】効果を発動する（従来は未発火＝14枚が no-op だった）。
            if blocker.master.abilities and not blocker.is_effect_negated and not blocker.negated:
                for ability in blocker.master.abilities:
                    if ability.trigger == TriggerType.ON_BLOCK:
                        self.resolve_ability(target_owner, ability, source_card=blocker)
            if self.active_interaction:
                # ブロック時効果が対象選択等で中断した場合はここで返す（resume が継続）。
                return
        self.phase = Phase.BATTLE_COUNTER;

    def apply_counter(self, player: Player, counter_card: Optional[Card] = None, don_list: Optional[List[DonInstance]] = None):
        if not self.active_battle: return
        if counter_card is None: self.resolve_attack(); return
        self._validate_action(player, "SELECT_COUNTER")
        if counter_card.master.type == CardType.EVENT:
            self.pay_cost(player, counter_card.master.cost, don_list)
            for ability in counter_card.master.abilities:
                if ability.trigger == TriggerType.COUNTER: self.resolve_ability(player, ability, source_card=counter_card)
            # 「自分のキャラすべては、このターン中、…代わりに〜できる」(EB02-030) のような
            # 継続付与型の置換を登録する。イベントは即トラッシュで場に残らないため、
            # _find_replacement の場上 protector 走査では拾えない。player へ this-turn 付与する。
            self._register_granted_replacements(player, counter_card)
            self.move_card(counter_card, Zone.TRASH, player)
        else:
            counter_value = getattr(counter_card, "current_counter", counter_card.master.counter or 0); self.active_battle["counter_buff"] += counter_value
            self.move_card(counter_card, Zone.TRASH, player)

    def resolve_attack(self):
        if not self.active_battle: return
        attacker = self.active_battle["attacker"]; target = self.active_battle["target"]
        attacker_owner = self.active_battle["attacker_owner"]; target_owner = self.active_battle["target_owner"]
        counter_buff = self.active_battle.get("counter_buff", 0)
        is_my_turn = (attacker_owner == self.turn_player); is_target_turn = (target_owner == self.turn_player)
        attacker_pwr = attacker.get_power(is_my_turn); target_pwr = target.get_power(is_target_turn) + counter_buff
        life_lost = 0
        if target == target_owner.leader:
            if attacker_pwr >= target_pwr:
                damage_amount = 2 if attacker.has_keyword("ダブルアタック") else 1; is_banish = attacker.has_keyword("バニッシュ")
                for _ in range(damage_amount):
                    if target_owner.life:
                        life_card = target_owner.life.pop(0)
                        dest_zone = Zone.TRASH if is_banish else Zone.HAND
                        trigger_ability = None if is_banish else next(
                            (a for a in life_card.master.abilities if a.trigger == TriggerType.TRIGGER), None
                        )
                        self.move_card(life_card, dest_zone, target_owner)
                        life_lost += 1
                        # 【トリガー】は「発動できる」（任意）。即時解決せず確認付きで待ち行列へ。
                        # 複数枚（ダブルアタック等）でも確認/解決が中断を跨いで消失しない。
                        if trigger_ability:
                            self._enqueue_trigger(target_owner, trigger_ability, life_card, optional=True)
                    else: self.winner = attacker_owner.name; break
        else:
            if attacker_pwr >= target_pwr:
                if self._active_protection(target, ("BATTLE_KO",), attacker=attacker):
                    pass
                else:
                    repl = self._find_replacement(target, ("BATTLE_KO",))
                    if repl is not None and getattr(repl[3], "is_optional", False):
                        # 任意のバトルKO置換（「代わりに〜してもよい/できる」OP10-034 等）は、
                        # 被KO側に「代わりの効果を使うか」を確認するため戦闘を中断する。
                        # accept→置換実行（本来のKOをスキップ）、decline→本来のKOを実行。
                        # どちらの分岐も resume 時に _finish_attack で戦闘後処理を行う。
                        self._suspend_for_battle_ko_replacement(target, target_owner, life_lost)
                        return
                    elif repl is not None:
                        # 任意でない置換は従来どおり即時実行（内側選択はヘッドレス自動解決）。
                        self._active_replacement(target, ("BATTLE_KO",))
                    else:
                        self.move_card(target, Zone.TRASH, target_owner)
                        self._resolve_on_ko(target, target_owner, cause="BATTLE")

        self._finish_attack(target, target_owner, life_lost)

    def _finish_attack(self, target: Card, target_owner: Player, life_lost: int):
        """戦闘解決後の共通後処理。インラインのバトルKO判定からも、任意バトルKO置換の
        確認(CONFIRM_OPTIONAL)からの resume からも呼ばれる。"""
        target.reset_turn_status(keep_don=True); self.active_battle = None; self.phase = Phase.MAIN; self.check_victory()
        self.continuous.expire("BATTLE_END", self.turn_count)
        if not self.winner:
            self._apply_passive_effects(self.turn_player)
        # ライフが離れた回数ぶん ON_LIFE_DECREASE を待ち行列へ積み、【トリガー】と共に消化する。
        if life_lost and not self.winner:
            self._enqueue_life_decrease(target_owner, life_lost)
        self._advance_pending_triggers()

    def _suspend_for_battle_ko_replacement(self, target: Card, target_owner: Player, life_lost: int):
        """任意のバトルKO置換を被KO側へ確認するため戦闘を中断する（CONFIRM_OPTIONAL）。
        resume 時: accept→置換実行（KOスキップ）／decline→本来のKO、その後 _finish_attack。
        ヘッドレス/CPU の既定応答(index0=accept)は従来の自動採用と一致する。"""
        self.active_interaction = {
            "player_id": target_owner.name,
            "action_type": "CONFIRM_OPTIONAL",
            "source_card_name": target.master.name,
            "source_card_uuid": target.uuid,
            "message": f"「{target.master.name}」がバトルでKOされます。代わりの効果を使用しますか？",
            "can_skip": True,
            "continuation": {
                "kind": "BATTLE_KO_REPLACE",
                "source_card_uuid": target.uuid,
                "target_owner_name": target_owner.name,
                "life_lost": life_lost,
            },
        }

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
            if getattr(card, "is_effect_negated", False):
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
        if card.master.type == CardType.EVENT:
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
            # 「相手の登場時効果は無効になる」(OPP_ONPLAY) 期間中はこのプレイヤーの ON_PLAY を解決しない。
            onplay_negated = getattr(player, "negate_onplay_until", 0) >= self.turn_count
            if onplay_negated:
                pass
            if not card.is_effect_negated and not onplay_negated:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_PLAY:
                        self.resolve_ability(player, ability, source_card=card)
            self._apply_passive_effects(player)
            # 場のキャラ上限超過なら強制トラッシュ。ON_PLAY が中断中(active_interaction)の
            # ときは _enforce_field_limit が no-op し、対話完了時に resolve_interaction 末尾が拾う。
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
        """除去置換の中断で失われる外側リゾルバの後続（execution_stack）を退避する（B1）。"""
        self._deferred_continuations.insert(0, {
            "kind": "RESOLVER_STACK",
            "player_name": player.name,
            "source_card_uuid": source_card.uuid if source_card else None,
            "execution_stack": JournaledList(execution_stack),
            "effect_context": context,
        })

    def _defer_removal_targets(self, player: Player, action, remaining_targets, value) -> None:
        """複数対象除去で置換中断したとき、未処理の残対象を退避する（B2）。
        再開時に apply_action_to_engine を残対象で再実行する（uuid で解決し直す）。"""
        self._deferred_continuations.append({
            "kind": "REMOVAL_TARGETS",
            "player_name": player.name,
            "action": action,
            "remaining_target_uuids": [t.uuid for t in remaining_targets],
            "value": value,
        })

    def _resume_deferred_continuations(self, limit: int = 64) -> None:
        """中断が無くなった後、退避した外側継続を LIFO で再開する。
        再開した継続が新たな中断を生んだら、active_interaction が立ってループは止まる
        （残りは次の解決後に再びここで処理される）。"""
        n = 0
        while not self.active_interaction and self._deferred_continuations and n < limit:
            frame = self._deferred_continuations.pop()
            kind = frame.get("kind")
            player = self.p1 if self.p1.name == frame.get("player_name") else self.p2
            try:
                if kind == "RESOLVER_STACK":
                    src = frame.get("source_card_uuid")
                    source_card = self._find_card_by_uuid(src) if src else None
                    resolver = EffectResolver(self)
                    resolver.resume_execution(player, source_card,
                                              frame.get("execution_stack", []),
                                              frame.get("effect_context", {}))
                elif kind == "REMOVAL_TARGETS":
                    remaining = [c for c in (self._find_card_by_uuid(u) for u in frame.get("remaining_target_uuids", [])) if c]
                    if remaining:
                        self.apply_action_to_engine(player, frame.get("action"), remaining, frame.get("value"))
            except Exception as e:
                pass
            n += 1

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