"""Rust エンジンの `hidden`（記録 v5）→ Python の `GameManager` を復元する（P1 で作った復元器）。

`docs/rust_engine_plan.md` §9.1 の `hidden` は「盤面を完全に再構成できる内部状態」で、Rust 側は
`GameState::from_record` でこれを読み、`GameState::hidden_json` でこれを書く。本モジュールはその
**Python 側の読み手**＝同じ `hidden` から `GameManager` を組み直す。

使い道は 2 つ:

1. オラクル（`tests/harness/rs_record.py` → `tests/scripts/rs_ops_oracle.py`）: 同じ `hidden` から
   Python 側も盤面を組み直し、両者に同じ操作台本を流して盤面 dict を照合する。
2. **対戦 API の暫定 CPU 経路**（`opcg_sim/api/engine_rs.py`・§15.1）: 裁定は Rust だけで進むが、
   CPU の `decide` はまだ Python（`cpu_learned`）なので、Rust の盤面を `hidden` で取り出して
   ここで `GameManager` を組み、`decide` に読ませる。返った手は Rust へ適用する。
   P4 の `rs-p4-mcts` が入ったら Rust の `decide` に差し替える（この経路は消える）。

**限界（暫定経路の既知の制約）**: `hidden` は中断（対話）スタック・誘発待ち行列・継続効果を
**件数しか持たない**（記録 v5 の契約）。よって復元した `GameManager` は「効果の途中で止まっている」
状態を持てない。`engine_rs` は対話中の要求について、Rust が出した要求から**浅い
`active_interaction`** を組んで載せる（要求・合法手・既定解決は正しく出る／探索の中でその手を
適用しても no-op になる＝その決定点だけ CPU の読みが浅くなる）。裁定は Rust 側なので**盤面は
常に正しい**。
"""
from typing import Any, Dict, Optional

from legacy.python_engine.core.gamestate import GameManager, Player
from opcg_sim.src.models.journal import JournaledDict, JournaledList, JournaledSet
from opcg_sim.src.models.enums import Phase
from opcg_sim.src.models.models import CardInstance, DonInstance

# `tests/scripts/rs_diff_replay.py::card_record` ／ Rust `GameState::card_record` が書く
# 実行時フィールド（記録 v5）。追加したら 3 か所（記録・Rust・ここ）を同時に直す。
_CARD_RUNTIME_FIELDS = (
    "is_rest", "is_newly_played", "attached_don", "is_face_up", "power_buff", "cost_buff",
    "passive_power", "passive_power_override", "passive_counter", "base_power_override",
    "base_cost_override", "negated", "ability_disabled", "timed_power", "timed_cost",
)
_CARD_SET_FIELDS = ("current_keywords", "flags", "timed_flags", "timed_keywords")

_CARD_ZONES = ("deck", "hand", "life", "field", "trash", "temp_zone")


class RestoreError(ValueError):
    """`hidden` から盤面を組み直せない（未知の card_id・付与先が居ない 等）。"""


def _card_from_record(db, rec: dict) -> CardInstance:
    master = db.get_card(rec["card_id"])
    if master is None:
        raise RestoreError(f"unknown card_id: {rec['card_id']}")
    card = CardInstance(master, rec["owner_id"], rec["uuid"])
    for f in _CARD_RUNTIME_FIELDS:
        setattr(card, f, rec[f])
    for f in _CARD_SET_FIELDS:
        setattr(card, f, JournaledSet(rec[f]))
    card.ability_used_this_turn = JournaledDict(
        {int(k): v for k, v in rec["ability_used_this_turn"].items()}
    )
    return card


def _don_from_record(rec: dict) -> DonInstance:
    return DonInstance(
        owner_id=rec["owner_id"],
        uuid=rec["uuid"],
        is_rest=rec["is_rest"],
        attached_to=rec["attached_to"],
        is_frozen=rec["is_frozen"],
    )


def manager_from_hidden(db, hidden: dict, *, suppress_pending: bool = True,
                        names: Optional[Dict[str, str]] = None) -> GameManager:
    """記録 v5 の `hidden` から `GameManager` を復元する。

    注意点:
      - `GameManager.__init__` はリーダーの「ドン!!デッキは N 枚」ルールで `don_deck` を作り直す。
        よってマネージャを作った**後**に全ゾーンを流し込む（記録の並びが正）。
      - ゾーンは `JournaledList` で入れる（make/unmake が効く形＝本番と同じ）。

    `suppress_pending=True`（既定）は `get_pending_request()` を常に None にする（オラクル用・
    下の [`_suppress_pending_request`] を参照）。**CPU 経路は False** で呼び、要求をそのまま
    引けるようにする。`names` は席（"p1"/"p2"）→ 表示名の対応（API の `p1_name` 等）。
    """
    names = names or {}

    def _name(seat: str) -> str:
        return names.get(seat, seat)

    players = {}
    for seat in ("p1", "p2"):
        rec = hidden["players"][seat]
        leader = _card_from_record(db, rec["leader"]) if rec.get("leader") else None
        players[seat] = Player(_name(seat), [], leader)
    manager = GameManager(players["p1"], players["p2"])

    by_uuid = {}
    for seat in ("p1", "p2"):
        rec = hidden["players"][seat]
        player = players[seat]
        if player.leader is not None:
            by_uuid[player.leader.uuid] = player.leader
        for zone in _CARD_ZONES:
            cards = [_card_from_record(db, c) for c in rec[zone]]
            setattr(player, zone, JournaledList(cards))
            for c in cards:
                by_uuid[c.uuid] = c
        stage = _card_from_record(db, rec["stage"]) if rec.get("stage") else None
        player.stage = stage
        if stage is not None:
            by_uuid[stage.uuid] = stage
        dons = rec["don"]
        player.don_deck = JournaledList(_don_from_record(d) for d in dons["deck"])
        player.don_active = JournaledList(_don_from_record(d) for d in dons["active"])
        player.don_rested = JournaledList(_don_from_record(d) for d in dons["rested"])
        player.don_attached_cards = JournaledList(_don_from_record(d) for d in dons["attached"])
        player.negate_onplay_until = rec["negate_onplay_until"]
        player.restrictions = JournaledDict({k: dict(v) for k, v in rec["restrictions"].items()})

    # 付与先の存在を検査する（黙って壊れた盤面を作らない）。
    for player in players.values():
        for don in player.don_attached_cards:
            if don.attached_to is not None and don.attached_to not in by_uuid:
                raise RestoreError(f"attached don points at a missing card: {don.attached_to}")

    mrec = hidden["manager"]
    manager.turn_count = mrec["turn_count"]
    manager.phase = Phase[mrec["phase"]]
    manager.turn_player = players[mrec["turn_player"]]
    manager.opponent = players["p2" if mrec["turn_player"] == "p1" else "p1"]
    manager.winner = _name(mrec["winner"]) if mrec["winner"] else mrec["winner"]
    battle = mrec.get("active_battle")
    if battle:
        try:
            manager.active_battle = {
                "attacker": by_uuid[battle["attacker"]],
                "target": by_uuid[battle["target"]],
                "counter_buff": battle.get("counter_buff", 0),
            }
        except KeyError as e:  # 戦闘参加者が盤面に居ない＝記録が壊れている
            raise RestoreError(f"active_battle points at a missing card: {e}") from e
        # v3 以降は所在の持ち主（`_find_card_location` 基準）も記録されている。
        # `engine/interaction.py` が BLOCK_STEP/COUNTER_STEP で読むので載せる。
        for key in ("attacker_owner", "target_owner"):
            if battle.get(key):
                manager.active_battle[key] = players[battle[key]]
    else:
        manager.active_battle = None
    manager._turn_events = JournaledDict(mrec["turn_events"])
    manager.mulligan_done = JournaledSet(_name(s) for s in mrec["mulligan_done"])
    manager.setup_phase_pending = mrec["setup_phase_pending"]
    manager.turn_start_pending = mrec["turn_start_pending"]
    if suppress_pending:
        _suppress_pending_request(manager)
    return manager


def _suppress_pending_request(manager) -> None:
    """復元した盤面では `get_pending_request()` を呼べないようにする（常に None）。

    記録 v2 の `active_battle` は `{attacker, target, counter_buff}` しか持たなかったため、
    `engine/interaction.py` が BLOCK_STEP/COUNTER_STEP で読む `target_owner` が無く KeyError に
    なった（P1 の照合対象からは対話が外れているので塞いで良かった）。記録 v3 以降は
    `attacker_owner`／`target_owner` も持つので **CPU 経路では塞がない**（`suppress_pending=False`）。
    """
    manager.get_pending_request = lambda with_request_id=True: None


def find_card(manager, uuid: str) -> CardInstance:
    """uuid で盤面のカードを引く（無ければ `RestoreError`）。"""
    card = manager._find_card_by_uuid(uuid)
    if card is None:
        raise RestoreError(f"unknown card uuid: {uuid}")
    return card


def find_don(manager, uuid: str) -> DonInstance:
    """uuid で盤面のドン!!を引く（無ければ `RestoreError`）。"""
    for player in (manager.p1, manager.p2):
        for zone in (player.don_deck, player.don_active, player.don_rested,
                     player.don_attached_cards):
            for don in zone:
                if don.uuid == uuid:
                    return don
    raise RestoreError(f"unknown don uuid: {uuid}")


def attach_shallow_interaction(manager, pending: Optional[Dict[str, Any]]) -> None:
    """Rust が出した要求から**浅い** `active_interaction` を組んで載せる（暫定 CPU 経路）。

    `hidden` は継続（continuation）を持たないので、要求の見た目（候補・選択制約・選択肢）だけを
    復元する。これで `get_pending_request` / `default_interaction_payload` / `get_legal_actions` /
    `_validate_action` は Rust と同じ答えを出す。`resolve_interaction` を通すと継続が無いので
    中断を落として何もしない＝**探索の中でだけ**その手が no-op になる（本物の適用は Rust 側）。

    `pending` が対話でない（MULLIGAN／MAIN_ACTION／SELECT_BLOCKER／SELECT_COUNTER）か None なら
    何もしない。
    """
    if not pending:
        return
    action = pending.get("action")
    if action in (None, "MULLIGAN", "MAIN_ACTION", "SELECT_BLOCKER", "SELECT_COUNTER"):
        return
    uuids = list(pending.get("selectable_uuids") or [])
    candidates = []
    for uid in uuids:
        card = manager._find_card_by_uuid(uid)
        if card is not None:
            candidates.append(card)
    manager.active_interaction = {
        "action_type": action,
        "player_id": pending.get("player_id"),
        "message": pending.get("message", "選択してください"),
        "candidates": candidates,
        "selectable_uuids": uuids,
        "constraints": pending.get("constraints"),
        "can_skip": pending.get("can_skip", False),
        "options": pending.get("options"),
        "source_card_uuid": pending.get("source_card_uuid"),
        "allow_position": pending.get("allow_position", False),
        "allow_reorder": pending.get("allow_reorder", False),
        # 継続は `hidden` に無い（上の docstring の限界）。
        "continuation": None,
    }
