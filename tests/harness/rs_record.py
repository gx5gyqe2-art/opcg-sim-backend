"""記録 v5 の `hidden` → 本物の `GameManager` を復元する（Rust オラクルの Python 側・P1）。

`docs/rust_engine_plan.md` §9.1 の `hidden` は「盤面を完全に再構成できる内部状態」で、Rust 側は
`GameState::from_record` でこれを読む。原始操作のオラクル（`tests/scripts/rs_ops_oracle.py`）
では **同じ hidden から Python 側も盤面を組み直し**、両者に同じ操作台本を流して盤面 dict を照合する。

**復元器の本体は `opcg_sim/src/core/rs_bridge.py` へ移した**（2026-09-07・計画 §15）——対戦 API の
暫定 CPU 経路が同じ復元器を使うため、本番コード側に置く必要があった。本モジュールは
`manager_from_hidden`／`find_card`／`find_don`／`RestoreError` を再エクスポートし、
テスト専用の [`apply_python_op`]（操作台本 1 件を**本物の Python の原始操作**へ流す。Rust 側の
`ops.rs` はこの各行と 1:1 に対応する）だけを持つ。

観測専用のハーネスであり、`opcg_sim/` 側のエンジンは一切変更しない。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core.gamestate import Player  # noqa: E402
from opcg_sim.src.core.rs_bridge import (  # noqa: E402,F401
    RestoreError, find_card, find_don,
)
from opcg_sim.src.core import rs_bridge as _rs_bridge  # noqa: E402


def manager_from_hidden(db, hidden: dict, with_pending: bool = False, **kw):
    """`rs_bridge.manager_from_hidden` の薄いラッパ（P4 のオラクル互換）。

    `with_pending=True` は `get_pending_request()` を塞がない（`rs_bridge` の
    `suppress_pending=False` と同義。記録 v3 以降は `active_battle` の所在の持ち主が入るので要求を
    組み立てられる＝`rs_encode_oracle.py` の登場時スキャン v7・`_leader_act_avail` が使う）。"""
    kw.setdefault("suppress_pending", not with_pending)
    return _rs_bridge.manager_from_hidden(db, hidden, **kw)
from opcg_sim.src.models.enums import Zone  # noqa: E402


# --- 操作台本（docs/rust_engine_plan.md §9.5）を Python の原始操作へ流す ---------

def _player(manager, name: str) -> Player:
    if name == "p1":
        return manager.p1
    if name == "p2":
        return manager.p2
    raise RestoreError(f"unknown player: {name}")


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
