"""記録 v2 の `hidden` → 本物の `GameManager` を復元する（Rust オラクルの Python 側・P1）。

`docs/rust_engine_plan.md` §9.1 の `hidden` は「盤面を完全に再構成できる内部状態」で、
Rust 側は `GameState::from_record` でこれを読む。原始操作のオラクル（`tests/scripts/rs_ops_oracle.py`）
では **同じ hidden から Python 側も盤面を組み直し**、両者に同じ操作台本を流して盤面 dict を照合する。

本モジュールが持つのは 2 つ:

- [`manager_from_hidden`] … `hidden` → `GameManager`（カード実体の全フィールド・ドン!!の所在と
  付与先・マネージャ欄まで）。復元の正しさは「復元直後の `board_dict` が記録の `state` と一致」で
  Python 側だけで検査できる（Rust 不要）。
- [`apply_python_op`] … 操作台本（`docs/rust_engine_plan.md` §9.5）の 1 件を**本物の Python の
  原始操作**へ流す。Rust 側の `ops.rs` はこの各行と 1:1 に対応する。

観測専用のハーネスであり、`opcg_sim/` 側は一切変更しない（P1 は Python への追加のみ）。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core.gamestate import GameManager, Player  # noqa: E402
from opcg_sim.src.core.journal import JournaledDict, JournaledList, JournaledSet  # noqa: E402
from opcg_sim.src.models.enums import Phase, Zone  # noqa: E402
from opcg_sim.src.models.models import CardInstance, DonInstance  # noqa: E402

# `tests/scripts/rs_diff_replay.py::card_record` が書く実行時フィールド（記録 v2）。
# 追加したら両方を同時に直す（片方だけだと復元が黙って欠ける）。
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


def manager_from_hidden(db, hidden: dict) -> GameManager:
    """記録 v2 の `hidden` から `GameManager` を復元する。

    注意点:
      - `GameManager.__init__` はリーダーの「ドン!!デッキは N 枚」ルールで `don_deck` を作り直す。
        よってマネージャを作った**後**に全ゾーンを流し込む（記録の並びが正）。
      - ゾーンは `JournaledList` で入れる（make/unmake が効く形＝本番と同じ）。
    """
    players = {}
    for name in ("p1", "p2"):
        rec = hidden["players"][name]
        leader = _card_from_record(db, rec["leader"]) if rec.get("leader") else None
        players[name] = Player(name, [], leader)
    manager = GameManager(players["p1"], players["p2"])

    by_uuid = {}
    for name in ("p1", "p2"):
        rec = hidden["players"][name]
        player = players[name]
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
    manager.winner = mrec["winner"]
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
    else:
        manager.active_battle = None
    manager._turn_events = JournaledDict(mrec["turn_events"])
    manager.mulligan_done = JournaledSet(mrec["mulligan_done"])
    manager.setup_phase_pending = mrec["setup_phase_pending"]
    manager.turn_start_pending = mrec["turn_start_pending"]
    _suppress_pending_request(manager)
    return manager


def _suppress_pending_request(manager) -> None:
    """復元した盤面では `get_pending_request()` を呼べないようにする（常に None）。

    記録 v2 の `active_battle` は `{attacker, target, counter_buff}` しか持たないが、
    `engine/interaction.py` は BLOCK_STEP/COUNTER_STEP で `active_battle["target_owner"]` を読む
    （KeyError になる）。対話（pending request）は P2 の契約で記録形式に足す範囲なので、P1 の
    照合対象からは外れている＝ここでは None を返して塞ぐ（`board_dict` の他の欄は影響を受けない）。
    """
    manager.get_pending_request = lambda with_request_id=True: None


# --- 操作台本（docs/rust_engine_plan.md §9.5）を Python の原始操作へ流す ---------

def _player(manager, name: str) -> Player:
    if name == "p1":
        return manager.p1
    if name == "p2":
        return manager.p2
    raise RestoreError(f"unknown player: {name}")


def find_card(manager, uuid: str) -> CardInstance:
    card = manager._find_card_by_uuid(uuid)
    if card is None:
        raise RestoreError(f"unknown card uuid: {uuid}")
    return card


def find_don(manager, uuid: str) -> DonInstance:
    for player in (manager.p1, manager.p2):
        for zone in (player.don_deck, player.don_active, player.don_rested,
                     player.don_attached_cards):
            for don in zone:
                if don.uuid == uuid:
                    return don
    raise RestoreError(f"unknown don uuid: {uuid}")


def apply_python_op(manager, op: dict) -> None:
    """操作台本 1 件を Python 側の原始操作で適用する（Rust `ops::apply_op` と 1:1）。

    ドン!!付与・ライフ→手札・デッキ→ライフには単一の関数が無いため、**呼び出し元の
    コードをそのまま書き写す**（出典をコメントに明記）。それ以外は `GameManager` の
    メソッドをそのまま呼ぶ＝Python 版が正本であることを崩さない。
    """
    kind = op["op"]
    if kind == "move_card":
        manager.move_card(find_card(manager, op["card"]), Zone[op["to"]],
                          _player(manager, op["player"]), op.get("pos", "BOTTOM"))
    elif kind == "draw":
        manager.draw_card(_player(manager, op["player"]), int(op.get("n", 1)))
    elif kind == "pay_cost":
        dons = op.get("dons")
        don_list = None if dons is None else [find_don(manager, u) for u in dons]
        manager.pay_cost(_player(manager, op["player"]), int(op.get("cost", 0)), don_list)
    elif kind == "return_don":
        manager._return_one_don(_player(manager, op["player"]), find_don(manager, op["don"]))
    elif kind == "attach_don":
        # 出典: opcg_sim/src/core/actions/per_target.py の ATTACH_DON ハンドラ
        # （`core/action_api.py` の ACT_ATTACH_DON も同じ 5 行）。
        player = _player(manager, op["player"])
        target = find_card(manager, op["card"])
        from_rested = bool(op.get("from_rested", False))
        pool = player.don_rested if from_rested else (player.don_active or player.don_rested)
        if pool:
            don = pool.pop(0)
            don.attached_to = target.uuid
            don.is_rest = from_rested
            player.don_attached_cards.append(don)
            target.attached_don += 1
    elif kind == "reset_turn_status":
        find_card(manager, op["card"]).reset_turn_status(
            keep_don=bool(op.get("keep_don", False)),
            clear_usage=bool(op.get("clear_usage", False)),
        )
    elif kind == "life_to_hand":
        # 出典: core/engine/battle.py::resolve_attack ／ core/actions/player_level.py の
        # ダメージ処理＝**ライフから取り出してから** move_card(..., HAND) を呼ぶ。
        player = _player(manager, op["player"])
        if player.life:
            card = player.life.pop(0 if op.get("from", "TOP") == "TOP" else -1)
            manager.move_card(card, Zone.HAND, player)
    elif kind == "deck_to_life":
        # 出典: core/actions/player_level.py の HEAL（`player.life.append(player.deck.pop(0))`）。
        player = _player(manager, op["player"])
        if player.deck:
            player.life.append(player.deck.pop(0))
    elif kind == "set_rest":
        find_card(manager, op["card"]).is_rest = bool(op.get("value", True))
    elif kind == "record_turn_event":
        manager.record_turn_event(op["name"], int(op.get("n", 1)))
    else:
        raise RestoreError(f"unknown op: {kind}")
