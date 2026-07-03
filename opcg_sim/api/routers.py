"""API ルート定義（単一 APIRouter）。`app.py` は create_app でこれを include する。

ルートは薄く保ち、ロジックは config/resources/state/presenters/ws/services へ委譲する。
monkeypatch 対象の `load_deck_mixed`/`_deck_preview` は**サービスモジュール属性経由**で呼ぶ
（`deck_svc.load_deck_mixed(...)`）。テストは `services.decks.load_deck_mixed` を patch する。
ドメイン別（game/cpu/cards/decks/sandbox/rule）への更なる分割は容易な followup。
"""
import os
import uuid
import random
import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Body, Request, Response, WebSocket, WebSocketDisconnect

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from .schemas import BattleActionRequest
from opcg_sim.src.core.gamestate import Player, GameManager
from opcg_sim.src.core import action_api
from opcg_sim.src.core.sandbox import SandboxManager
from opcg_sim.api import decide_client
from .config import CONST, IMAGE_VERSION, REPLAY_SCHEMA, constants_hash, SCHEMA_HASH
from .resources import db, card_db, CARDS_ETAG, materialize_all_cards
from .state import GAMES, SANDBOX_GAMES, CPU_GAMES, RULE_ROOMS
from .presenters import build_game_result_hybrid, build_rule_message, _rule_room_meta
from .ws import ws_manager, game_ws_manager, broadcast_rule_state
from .services import decks as deck_svc
from .services.replay import _replay_enabled, _replay_record_action, _capture_final_winner
from .services.games import _resolve_first_player
from .services.cpu_driver import _kick_ponder, _kick_speculate, _cached_cpu_move

_logger = logging.getLogger("opcg.api")

router = APIRouter()


# ---- ルールモード対局 -------------------------------------------------------

@router.options("/api/game/create")
async def options_game_create(): return {"status": "ok"}

@router.post("/api/game/create")
async def game_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        p1_source = req.get("p1_deck", ""); p2_source = req.get("p2_deck", "")
        materialize_all_cards()
        vs_cpu = bool(req.get("vs_cpu", False))
        # CPU 対戦時は p2 を CPU とし、デッキは cpu_deck（無指定なら p2_deck）を使う。
        if vs_cpu and req.get("cpu_deck"):
            p2_source = req.get("cpu_deck")
        p1_leader, p1_cards = deck_svc.load_deck_mixed(p1_source, req.get("p1_name", "P1")); p2_leader, p2_cards = deck_svc.load_deck_mixed(p2_source, req.get("p2_name", "P2"))
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader); player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        # リプレイ種（opt-in）: cpu_trace 指定時のみ seed を固定し、コイントス＋シャッフルを再現可能にする。
        # 未指定の本番対局は seed を触らない＝従来の乱数挙動を完全に維持する。
        cpu_trace = bool(req.get("cpu_trace", False)) and vs_cpu
        replay_seed = None
        if cpu_trace:
            replay_seed = int(req.get("seed")) if req.get("seed") is not None else random.randrange(2**63)
            random.seed(replay_seed)
        # 先行プレイヤー: ソロは "p1"/"p2"、CPU は "random"（コイントス）。未指定は既定。
        first_player = _resolve_first_player(req.get("first_player"), player1, player2)
        manager = GameManager(player1, player2); manager.start_game(first_player); GAMES[game_id] = manager
        if vs_cpu:
            # CPU は **hard（α-β＋ビーム＋PIMC）** が既定。**learned**（Gen2 学習型・NN誘導MCTS）も選択可。
            difficulty = req.get("cpu_difficulty", "hard")
            if difficulty not in ("hard", "learned"):
                difficulty = "hard"
            CPU_GAMES[game_id] = {"cpu_player_id": player2.name, "difficulty": difficulty}
            if cpu_trace:
                # リプレイ種＋思考ログの器を用意する（opt-in 時のみ）。
                CPU_GAMES[game_id].update({
                    "cpu_trace": True, "seed": replay_seed,
                    "first_player": first_player.name if first_player else None,
                    "leaders": {"p1": p1_leader.master.card_id if p1_leader else None,
                                "p2": p2_leader.master.card_id if p2_leader else None},
                    "decks": {"p1": [ci.master.card_id for ci in p1_cards],
                              "p2": [ci.master.card_id for ci in p2_cards]},
                    "actions": [], "decisions": [],
                })
        return build_game_result_hybrid(manager, game_id)
    except Exception as e:
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

@router.options("/api/game/action")
async def options_game_action(): return {"status": "ok"}

@router.post("/api/game/action")
async def game_action(req: Dict[str, Any] = Body(...)):
    action_type = req.get("action") or req.get("type"); player_id = req.get("player_id", "system")
    game_id = req.get("game_id"); manager = GAMES.get(game_id); error_codes = CONST.get('ERROR_CODES', {})
    if not manager: return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    payload = req.get("payload") or req.get("full_payload") or {}
    try:
        manager.action_events = []
        # ディスパッチは action_api（CPU ドライバ・自己対戦ランナーと同一コアパス）へ委譲する。
        current_player = manager.p1 if player_id == manager.p1.name else manager.p2
        _meta = CPU_GAMES.get(game_id)
        _src = "cpu" if (_meta and player_id == _meta.get("cpu_player_id")) else "human"
        _replay_record_action(_meta, manager, _src, player_id, {"action_type": action_type, "payload": payload})
        action_api.apply_game_action(manager, current_player, action_type, payload)
        _capture_final_winner(_meta, manager)
        result = build_game_result_hybrid(manager, game_id, success=True)
        await broadcast_rule_state(game_id)
        _kick_ponder(game_id)     # ⑥-a: 制御が CPU へ移ったら次手番の計画を前倒し（既定 OFF）
        _kick_speculate(game_id)  # ⑥-b: 人間 MAIN 継続中なら「今エンドしたら」を投機（既定 OFF）
        return result
    except Exception as e:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@router.options("/api/game/state")
async def options_game_state(): return {"status": "ok"}

@router.get("/api/game/state")
async def game_state_fetch(game_id: str):
    """現在の対局状態を読み取り専用で返す（盤面は一切変更しない）。ルーム対局は build_rule_message を返す。"""
    error_codes = CONST.get('ERROR_CODES', {})
    if game_id in RULE_ROOMS:
        return build_rule_message(game_id)
    manager = GAMES.get(game_id)
    if not manager:
        return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    return build_game_result_hybrid(manager, game_id, success=True)

@router.options("/api/game/battle")
async def options_game_battle(): return {"status": "ok"}

@router.post("/api/game/battle")
async def game_battle(req: BattleActionRequest):
    game_id = req.game_id; player_id = req.player_id; action_type = req.action_type; card_uuid = req.card_uuid
    manager = GAMES.get(game_id); error_codes = CONST.get('ERROR_CODES', {})
    if not manager: return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="Game not found")
    player = manager.p1 if player_id == manager.p1.name else manager.p2
    try:
        manager.action_events = []
        _meta = CPU_GAMES.get(game_id)
        _src = "cpu" if (_meta and player_id == _meta.get("cpu_player_id")) else "human"
        _replay_record_action(_meta, manager, _src, player_id, {"action_type": action_type, "card_uuid": card_uuid})
        action_api.apply_battle_action(manager, player, action_type, card_uuid)
        _capture_final_winner(_meta, manager)
        result = build_game_result_hybrid(manager, game_id, success=True)
        await broadcast_rule_state(game_id)
        _kick_ponder(game_id)
        _kick_speculate(game_id)
        return result
    except Exception as e:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))


# ---- CPU 対戦（ポーリング駆動）／リプレイ ----------------------------------

@router.options("/api/game/cpu/step")
async def options_game_cpu_step(): return {"status": "ok"}

@router.post("/api/game/cpu/step")
async def game_cpu_step(req: Dict[str, Any] = Body(...)):
    """CPU 対戦で CPU(p2) の「次の 1 手」を適用して返す（ポーリング駆動）。"""
    game_id = req.get("game_id"); manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    error_codes = CONST.get('ERROR_CODES', {})
    if not manager:
        return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    if not meta:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg="このゲームは CPU 対戦ではありません。")

    cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "hard")
    cpu_player = manager.p1 if manager.p1.name == cpu_pid else manager.p2

    def _waiting_for() -> str:
        if manager.winner:
            return "game_over"
        pending = manager.get_pending_request()
        if pending and pending.get("player_id") == cpu_pid:
            return "cpu"
        if pending:
            return "human_decision"
        return "human"

    cpu_acted = False; cpu_event = None
    try:
        manager.action_events = []
        if not manager.winner:
            pending = manager.get_pending_request()
            if pending and pending.get("player_id") == cpu_pid:
                turn_mem = meta.setdefault("turn_mem", {})
                # ⑥-a: 先行計画（pondering）が走行中なら完了を待つ（warm な queue を使う・既定 OFF）。
                _ptask = meta.get("plan_cache", {}).get("task")
                if _ptask is not None:
                    try:
                        await _ptask
                    except Exception:
                        pass
                trace_on = _replay_enabled(meta)
                tr = {} if trace_on else None
                move = None
                # Phase 3 ① 計画キャッシュ（OPCG_PLAN_CACHE=1・本番体感最適化・既定 OFF）。
                if os.environ.get("OPCG_PLAN_CACHE", "0") == "1" and not trace_on:
                    move = _cached_cpu_move(manager, cpu_player, difficulty, meta, turn_mem)
                if move is None:
                    move = decide_client.decide(manager, cpu_player, difficulty, mem=turn_mem,
                                                trace=tr, trace_read_ahead=False)
                if move is not None:
                    if trace_on:
                        meta.setdefault("decisions", []).append(
                            {"turn": manager.turn_count, "player": cpu_pid, **tr})
                        _replay_record_action(meta, manager, "cpu", cpu_pid, {
                            "action_type": move["action_type"], "card_uuid": move.get("card_uuid"),
                            "payload": move.get("payload")})
                    if move["kind"] == "battle":
                        action_api.apply_battle_action(manager, cpu_player, move["action_type"], move.get("card_uuid"))
                    else:
                        action_api.apply_game_action(manager, cpu_player, move["action_type"], move.get("payload", {}))
                    _capture_final_winner(meta, manager)
                    cpu_acted = True
                    cpu_event = manager.action_events[0] if manager.action_events else {"action": move["action_type"]}
        result = build_game_result_hybrid(manager, game_id, success=True)
        result["cpu_acted"] = cpu_acted
        result["cpu_event"] = cpu_event
        result["waiting_for"] = _waiting_for()
        await broadcast_rule_state(game_id)
        return result
    except Exception as e:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@router.options("/api/game/{game_id}/replay")
async def options_game_replay(game_id: str): return {"status": "ok"}

@router.get("/api/game/{game_id}/replay")
async def game_replay(game_id: str):
    """traced CPU 対局の「リプレイ種＋CPU思考トレース」を返す（GCS 不要・メモリ常駐）。"""
    meta = CPU_GAMES.get(game_id)
    if not _replay_enabled(meta):
        return {"success": False, "error": {"code": "REPLAY_NOT_FOUND",
                "message": "この対局のリプレイ記録がありません（cpu_trace 未指定 or 不明なゲーム）。"}}
    descriptor = {
        "schema": REPLAY_SCHEMA, "seed": meta.get("seed"),
        "first_player": meta.get("first_player"), "difficulty": meta.get("difficulty"),
        "cpu_player_id": meta.get("cpu_player_id"),
        "leaders": meta.get("leaders"), "decks": meta.get("decks"),
        "actions": meta.get("actions", []),
    }
    return {"success": True, "game_id": game_id,
            "replay": descriptor, "decisions": meta.get("decisions", [])}


# ---- カード／アセット／ヘルス ----------------------------------------------

@router.get("/api/assets/version")
async def get_assets_version():
    """カード画像のキャッシュ版数を返す（フロントが ?v= に付与してキャッシュ無効化に使う）。"""
    return {"success": True, "v": IMAGE_VERSION}

@router.get("/api/cards")
async def get_all_cards(request: Request, response: Response):
    try:
        materialize_all_cards()
        # 内容に変化が無ければ本体を返さず 304（1.2MBの転送・再パースをスキップ）
        if request.headers.get("if-none-match") == CARDS_ETAG:
            return Response(status_code=304, headers={"ETag": CARDS_ETAG, "Cache-Control": "no-cache"})
        cards_data = [c.to_dict() for c in card_db.cards.values()]
        response.headers["ETag"] = CARDS_ETAG
        response.headers["Cache-Control"] = "no-cache"
        return {"success": True, "cards": cards_data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/health")
async def health():
    # constants_hash / schema_hash: フロントが埋め込みハッシュと照合して定数・APIスキーマの乖離
    # （同期漏れ deploy）を検出する契約照合用。schema_hash は contract/manifest.json 由来。
    return {"status": "ok", "constants_loaded": bool(CONST),
            "constants_hash": constants_hash(), "schema_hash": SCHEMA_HASH}


# ---- デッキ CRUD ------------------------------------------------------------

@router.post("/api/deck")
async def save_deck(deck_data: Dict[str, Any] = Body(...)):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        doc_ref = db.collection("decks").document(deck_data["id"]) if "id" in deck_data and deck_data["id"] else db.collection("decks").document()
        save_data = {"id": doc_ref.id, "name": deck_data.get("name", "Untitled Deck"), "leader_id": deck_data.get("leader_id"), "card_uuids": deck_data.get("card_uuids", []), "don_uuids": deck_data.get("don_uuids", []), "created_at": firestore.SERVER_TIMESTAMP}
        doc_ref.set(save_data); return {"success": True, "deck_id": doc_ref.id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.delete("/api/deck/{deck_id}")
async def delete_deck(deck_id: str):
    if not db:
        return {"success": False, "error": "Database not initialized"}
    try:
        db.collection("decks").document(deck_id).delete()
        return {"success": True, "deck_id": deck_id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/api/deck/get")
async def get_deck(id: str):
    try:
        leader, cards = deck_svc.load_deck_mixed(id, "system")
        return {
            "success": True,
            "deck": {
                "leader": [leader.master.to_dict()] if leader else [],
                "cards": [c.master.to_dict() for c in cards]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/api/deck/list")
async def list_decks():
    decks = []
    if db:
        try:
            docs = db.collection("decks").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
            for doc in docs:
                d = doc.to_dict()
                if "created_at" in d and d["created_at"]: d["created_at"] = str(d["created_at"])
                decks.append(d)
        except Exception:
            # 一覧取得失敗（資格情報・インデックス変更など）は空一覧で応答しつつ診断を残す。
            _logger.warning("デッキ一覧の取得に失敗（空一覧で応答）", exc_info=True)
    return {"success": True, "decks": decks}


# ---- フリーモード（サンドボックス） ----------------------------------------

@router.get("/api/sandbox/list")
async def sandbox_list():
    games = []
    for gid, mgr in SANDBOX_GAMES.items():
        try:
            games.append({
                "game_id": gid,
                "room_name": getattr(mgr, "room_name", "Untitled Room"),
                "p1_name": mgr.state["p1"]["name"],
                "p2_name": mgr.state["p2"]["name"],
                "turn": mgr.turn_count,
                "created_at": getattr(mgr, "created_at", "N/A"),
                "active_connections": len(ws_manager.active_connections.get(gid, [])),
                "status": mgr.status,
            })
        except Exception:
            continue
    return {"success": True, "games": games}

@router.post("/api/sandbox/create")
async def sandbox_create(req: Any = Body(...)):
    try:
        p1_name = req.get("p1_name", "P1"); p2_name = req.get("p2_name", "P2")
        manager = SandboxManager(p1_name=p1_name, p2_name=p2_name, room_name=req.get("room_name", "Custom Room"))
        SANDBOX_GAMES[manager.game_id] = manager
        return {"success": True, "game_id": manager.game_id, "game_state": manager.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/sandbox/action")
async def sandbox_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id"); manager = SANDBOX_GAMES.get(game_id)
    if not manager: return {"success": False, "error": "Sandbox game not found"}
    act_type = req.get("action_type"); pid = req.get("player_id")
    try:
        if act_type == "SET_DECK":
            deck_id = req.get("deck_id"); owner_name = manager.state[pid]["name"]
            leader, cards = deck_svc.load_deck_mixed(deck_id, owner_name)
            manager.set_player_deck(pid, cards, leader)
            manager.ready_states[pid] = True
        else: manager.process_action(req)
        new_state = manager.to_dict()
        broadcast_msg = {"type": "STATE_UPDATE", "state": new_state}
        if act_type == "KICK_PLAYER":
            broadcast_msg["kicked_player"] = req.get("target_player_id")
        await ws_manager.broadcast(game_id, broadcast_msg)
        return {"success": True, "game_id": game_id, "game_state": new_state}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.websocket("/ws/sandbox/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, game_id)


# ---- ルールモード・オンライン対戦（ロビー / ルーム） ------------------------

@router.options("/api/rule/create")
async def options_rule_create(): return {"status": "ok"}

@router.post("/api/rule/create")
async def rule_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        room = {
            "game_id": game_id,
            "room_name": req.get("room_name", "Rule Room"),
            "created_at": datetime.now().isoformat(),
            "status": "WAITING",
            "ready": {"p1": False, "p2": False},
            "decks": {"p1": None, "p2": None},
            "deck_preview": {"p1": None, "p2": None},
        }
        RULE_ROOMS[game_id] = room
        return {"success": True, "game_id": game_id, **_rule_room_meta(game_id), "game_state": None}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/api/rule/list")
async def rule_list():
    rooms = []
    for gid, room in RULE_ROOMS.items():
        try:
            rooms.append({
                "game_id": gid,
                "room_name": room.get("room_name", "Rule Room"),
                "p1_name": "P1",
                "p2_name": "P2",
                "turn": GAMES[gid].turn_count if gid in GAMES else 0,
                "created_at": room.get("created_at", "N/A"),
                "active_connections": game_ws_manager.count(gid),
                "status": room.get("status", "WAITING"),
                "ready_states": room.get("ready", {"p1": False, "p2": False}),
            })
        except Exception:
            continue
    return {"success": True, "games": rooms}

@router.options("/api/rule/action")
async def options_rule_action(): return {"status": "ok"}

@router.post("/api/rule/action")
async def rule_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id"); room = RULE_ROOMS.get(game_id)
    if not room:
        return {"success": False, "error": "Rule room not found"}
    act = req.get("action_type"); pid = req.get("player_id")
    try:
        if act == "SET_DECK":
            if room["status"] != "WAITING":
                return {"success": False, "error": "Game already started"}
            if pid not in ("p1", "p2"):
                return {"success": False, "error": "Invalid player_id"}
            deck_id = req.get("deck_id")
            room["decks"][pid] = deck_id
            room["deck_preview"][pid] = deck_svc._deck_preview(deck_id, pid)
            room["ready"][pid] = bool(deck_id)
        elif act == "KICK_PLAYER":
            target = req.get("target_player_id")
            if room["status"] == "WAITING" and target in ("p1", "p2"):
                room["decks"][target] = None
                room["deck_preview"][target] = None
                room["ready"][target] = False
        elif act == "START":
            if room["status"] != "WAITING":
                return {"success": False, "error": "Game already started"}
            if not (room["ready"]["p1"] and room["ready"]["p2"]):
                return {"success": False, "error": "Both players must be ready"}
            materialize_all_cards()
            p1_leader, p1_cards = deck_svc.load_deck_mixed(room["decks"]["p1"], "p1")
            p2_leader, p2_cards = deck_svc.load_deck_mixed(room["decks"]["p2"], "p2")
            player1 = Player("p1", p1_cards, p1_leader); player2 = Player("p2", p2_cards, p2_leader)
            # 対戦モードの先行はランダム（コイントス）。結果は turn_info で両クライアントへ broadcast。
            first_player = random.choice([player1, player2])
            manager = GameManager(player1, player2); manager.start_game(first_player)
            GAMES[game_id] = manager
            room["status"] = "PLAYING"
        else:
            return {"success": False, "error": f"Unknown rule action: {act}"}

        await game_ws_manager.broadcast(game_id, build_rule_message(game_id))
        return {"success": True, "game_id": game_id, **build_rule_message(game_id)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.websocket("/ws/game/{game_id}")
async def game_websocket_endpoint(websocket: WebSocket, game_id: str):
    await game_ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        game_ws_manager.disconnect(websocket, game_id)
