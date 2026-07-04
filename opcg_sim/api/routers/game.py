"""API ルート: ルールモード対局（作成/アクション/状態/バトル）（ドメイン別 APIRouter）。

`routers/__init__.py` が全ドメインを束ねて app が include する。ロジックは
config/resources/state/presenters/ws/services へ委譲する。monkeypatch 対象の
`load_deck_mixed`/`_deck_preview` はサービスモジュール属性経由で呼ぶ（`deck_svc.*`）。
"""
import uuid
import random
from typing import Any, Dict

from fastapi import APIRouter, Body

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from ..schemas import BattleActionRequest
from opcg_sim.src.core.gamestate import Player, GameManager
from opcg_sim.src.core import action_api
from ..config import CONST
from ..resources import materialize_all_cards
from ..state import GAMES, CPU_GAMES, RULE_ROOMS
from ..presenters import build_game_result_hybrid, build_rule_message
from ..ws import broadcast_rule_state
from ..services import decks as deck_svc
from ..services.replay import _replay_record_action, _capture_final_winner
from ..services.games import _resolve_first_player
from ..services.cpu_driver import _kick_ponder, _kick_speculate

router = APIRouter()


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
