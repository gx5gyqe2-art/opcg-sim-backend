"""対戦 API のエンジン層（Rust `opcg_engine.Game` のラッパ・`docs/rust_engine_plan.md` §15）。

API から見た `GameManager` の置き換え。**裁定は全て Rust 側**で、ここがやるのは 4 つだけ:

1. **対局の生成**（デッキ card_id 列 → `opcg_engine.Game`）。シャッフルとコイントスは
   Python の `random`（`random.getrandbits` を Rust へ渡す＝`Rng::Host`）で回すので、
   `random.seed(seed)` を張った対局は**従来どおり**「種＋思考トレース」から再現できる
   （`tests/harness/replay_runner.py`）。
2. **`request_id` の計算**（`_rid`）。フロントの「新しい要求」検知に使う sha1 で、Rust は出さない
   （§15.1）。Python 版（`engine/interaction.py::get_pending_request._rid`）と同じ式。
3. **イベントログの受け渡し**（`action_events`）。Rust が Python と同じ dict を同じ順序で積む。
4. **暫定 CPU 経路**。CPU の `decide` はまだ Python なので、Rust の盤面を記録 v5 の `hidden` で
   取り出し、`opcg_sim/src/core/rs_bridge.py::manager_from_hidden` で `GameManager` を組んで
   読ませる。返った手は Rust へ適用する（裁定は Rust 側だけで進む）。P4 の `rs-p4-mcts` が
   入ったら Rust の `decide` に差し替える。

`SandboxManager`（フリーモードの自由配置）はルールエンジンではないので Python のまま。
"""
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("opcg.api")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 効果構造 JSON（`opcg_sim/tools/export_effects_json.py` の生成物・git 管理外・約 8MB）。
# Rust はこれを起動時に 1 度だけ読んで `CardMaster` 表を作る。
EFFECTS_PATH = os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")

_masters_lock = threading.Lock()
_masters_loaded = False


def load_engine():
    """Rust 拡張を import し、カード定義表をプロセスで 1 度だけ読み込む（冪等）。"""
    global _masters_loaded
    import opcg_engine  # 遅延 import（拡張が無い環境で API 以外を壊さない）

    if _masters_loaded:
        return opcg_engine
    with _masters_lock:
        if _masters_loaded:
            return opcg_engine
        if not os.path.exists(EFFECTS_PATH):
            # 生成物なので配布物に無いことがある。その場でエクスポータを回す（起動時 1 回）。
            _logger.info("効果構造 JSON が無いので生成する: %s", EFFECTS_PATH)
            subprocess.run([sys.executable, "-m", "opcg_sim.tools.export_effects_json",
                            "--out", EFFECTS_PATH],
                           cwd=_REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
        opcg_engine.load_masters(EFFECTS_PATH)
        _masters_loaded = True
    return opcg_engine


class _Phase:
    """`manager.phase.name` を読む既存コードのための薄い持ち物。"""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


def _rid(request: Dict[str, Any], turn_count: int) -> str:
    """`pending_request.request_id`（Python `engine/interaction.py::_rid` と同じ式）。

    「同一の要求なら安定・要求が変われば変化」する決定的ハッシュ＝**要求の全内容
    （`request_id` 自身を除く）＋`turn_count`** の正規化 JSON の sha1（先頭 16 桁）。
    フロントはこの変化を「新しい要求」の合図に使う（入力側では未使用）。
    """
    payload = {k: v for k, v in request.items() if k != "request_id"}
    key = json.dumps([turn_count, payload], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class RsGame:
    """1 対局（Rust `opcg_engine.Game` のラッパ）。

    API から見た表面は `GameManager` の部分集合（`winner`／`turn_count`／`phase`／
    `get_pending_request`／`get_legal_actions`／`default_interaction_payload`／`action_events`）＋
    盤面 dict（[`board`]）と暫定 CPU 経路（[`py_manager`]）。
    """

    def __init__(self, game, p1_name: str, p2_name: str, card_db=None):
        self._game = game
        self.p1_name = p1_name
        self.p2_name = p2_name
        self._card_db = card_db
        # 1 要求ぶんのイベントログ（Rust から受け取った直近の結果）。
        self.action_events: List[Dict[str, Any]] = []

    # --- 生成 ---------------------------------------------------------------

    @classmethod
    def create(cls, p1_name: str, p2_name: str, p1_leader, p1_cards, p2_leader, p2_cards,
               first_player: Optional[str] = None, card_db=None) -> "RsGame":
        """デッキ（`load_deck_mixed` の戻り値）から対局を作る。

        `first_player` は "p1"／"p2"／None（既定＝p1）。**"random"（コイントス）は呼び出し側で
        解決してから渡す**——Python 版と乱数の消費位置を揃えるため
        （`services.games._resolve_first_player_seat`）。
        """
        engine = load_engine()
        game = engine.Game(
            p1_name, p2_name,
            p1_leader.master.card_id if p1_leader else None,
            [c.master.card_id for c in p1_cards],
            p2_leader.master.card_id if p2_leader else None,
            [c.master.card_id for c in p2_cards],
            first_player, None,
            # シャッフル／コイントスは CPython の `random` の出目で回す（種からの再現性を保つ）。
            random.getrandbits,
        )
        return cls(game, p1_name, p2_name, card_db=card_db)

    # --- 盤面・要求 ---------------------------------------------------------

    @property
    def winner(self) -> Optional[str]:
        return self._game.winner()

    @property
    def turn_count(self) -> int:
        return self._game.turn_count

    @property
    def phase(self) -> _Phase:
        return _Phase(self._game.phase)

    @property
    def turn_player_name(self) -> str:
        return self._game.turn_player()

    def board(self) -> Dict[str, Any]:
        """盤面 dict（`turn_info`／`players`／`active_battle`）。"""
        return json.loads(self._game.board_json())

    def hidden(self) -> Dict[str, Any]:
        """完全な内部状態（記録 v5 の `hidden`）。"""
        return json.loads(self._game.hidden_json())

    def get_pending_request(self, with_request_id: bool = True) -> Optional[Dict[str, Any]]:
        """現在の要求（`request_id` はここで付ける）。"""
        pending = json.loads(self._game.pending_json())
        if pending is None:
            return None
        if with_request_id:
            pending["request_id"] = _rid(pending, self.turn_count)
        return pending

    def get_legal_actions(self, player_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """合法手（`player_id` 省略時は要求先プレイヤー）。"""
        if player_id is None:
            pending = json.loads(self._game.pending_json())
            if not pending:
                return []
            player_id = pending["player_id"]
        return json.loads(self._game.legal_json(player_id))

    def default_interaction_payload(self, pending: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """効果対話の「妥当な既定解決」（Python `default_interaction_payload`）。

        `pending` 引数は Python 版との互換のために受けるだけで、Rust は常に現在の要求から作る。
        """
        return json.loads(self._game.default_payload_json())

    # --- 行動の適用 ---------------------------------------------------------

    def apply_game_action(self, player_id: str, action_type: str,
                          payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ゲームアクション（不正な行動は Python と同じ文言の `ValueError`）。"""
        try:
            raw = self._game.apply_game_action(
                player_id, action_type,
                json.dumps(payload or {}, ensure_ascii=False, default=str))
        except Exception:
            # Python も「例外で中断した時点までに積んだイベント」をレスポンスに載せる
            # （`action_api` は append してから raise する経路がある）＝同じにする。
            self.action_events = json.loads(self._game.events_json())
            raise
        self.action_events = json.loads(raw)
        return self.action_events

    def apply_battle_action(self, player_id: str, action_type: str,
                            card_uuid: Optional[str] = None) -> List[Dict[str, Any]]:
        """戦闘アクション（ブロック／カウンター／パス）。"""
        try:
            raw = self._game.apply_battle_action(player_id, action_type, card_uuid)
        except Exception:
            self.action_events = json.loads(self._game.events_json())
            raise
        self.action_events = json.loads(raw)
        return self.action_events

    def apply_move(self, player_id: str, move: Dict[str, Any]) -> List[Dict[str, Any]]:
        """`get_legal_actions` が返す形の 1 手を適用する（CPU ドライバ用）。"""
        if move.get("kind") == "battle":
            return self.apply_battle_action(player_id, move["action_type"], move.get("card_uuid"))
        return self.apply_game_action(player_id, move["action_type"], move.get("payload", {}))

    # --- 暫定 CPU 経路（P4 の rs-p4-mcts が入るまで）-------------------------

    def py_manager(self):
        """Rust の盤面から Python の `GameManager` を組み直す（§15.1 の暫定経路）。

        CPU の `decide`・思考トレースの記述（`cpu_ai._describe_move`）・リプレイフレームは
        まだ Python のオブジェクトを要求するので、要求のたびにここで組む（**読むだけ**＝
        この manager を進めても Rust の盤面は変わらない）。

        中断（対話）は `hidden` に継続を持てないため、Rust が出した要求から**浅い**
        `active_interaction` を載せる（`rs_bridge.attach_shallow_interaction` の docstring に限界）。
        """
        from opcg_sim.src.core import rs_bridge

        db = self._card_db
        if db is None:
            from opcg_sim.api.resources import card_db as _db
            db = _db
        manager = rs_bridge.manager_from_hidden(
            db, self.hidden(), suppress_pending=False,
            names={"p1": self.p1_name, "p2": self.p2_name})
        rs_bridge.attach_shallow_interaction(manager, self.get_pending_request(False))
        manager.action_events = list(self.action_events)
        return manager
