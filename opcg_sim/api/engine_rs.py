"""対戦 API のエンジン層（Rust `opcg_engine.Game` のラッパ・`docs/rust_engine_plan.md` §15・§16.3）。

API から見た `GameManager` の置き換え。**裁定も思考も全て Rust 側**で、ここがやるのは 4 つだけ:

1. **対局の生成**（デッキ card_id 列 → `opcg_engine.Game`）。シャッフルとコイントスは
   Python の `random`（`random.getrandbits` を Rust へ渡す＝`Rng::Host`）で回すので、
   `random.seed(seed)` を張った対局は**従来どおり**「種＋思考トレース」から再現できる。
2. **`request_id` の計算**（`_rid`）。フロントの「新しい要求」検知に使う sha1 で、Rust は出さない
   （§15.1）。旧 Python 版（`engine/interaction.py::get_pending_request._rid`）と同じ式。
3. **イベントログの受け渡し**（`action_events`）。Rust が旧 Python と同じ dict を同じ順序で積む。
4. **CPU の思考**（[`RsGame.decide`]）。`opcg_engine.Game.decide` を**生の盤面**に対して呼ぶ
   （2026-09-07・第 2 段 `rs-archive-cutover`）。それまでの暫定経路——`hidden` から
   `GameManager` を組み直して Python の `decide` に読ませる——は撤去した。旧経路は
   中断（対話）スタックを `hidden` に持てず**浅い**要求しか見せられなかったので、
   差し替えは同時に対話中の読みの質も直している。

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


def _net_path() -> str:
    """serve の CPU が使うネット（`opcg_sim.loop.engine.DEFAULT_NET`＝出荷既定）。

    物差しを 1 本に保つため、アリーナ・生成・serve は同じ既定を指す（別々に持たない）。
    """
    from opcg_sim.loop.engine import DEFAULT_NET
    return DEFAULT_NET


def _search_seed(base: int, turn: int, seat: str) -> int:
    """探索乱数の seed（対局の基点・ターン・席から決まる・pure）。

    `opcg_sim.loop.engine.search_seed` と同じ式＝serve と生成／アリーナで同じ意味論にする。
    """
    from opcg_sim.loop.engine import search_seed
    return search_seed(base, turn, seat)


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
        # CPU の思考（`decide`）が持つ状態。基点は生成時に 1 度だけ引く＝`random.seed()` を
        # 張った traced 対局では毎回同じ値になる（種からの再現）。
        self._search_base = random.getrandbits(63)
        self._carry_key = None
        self._carry: Dict[str, Any] = {}
        self._net_loaded = False

    # --- 生成 ---------------------------------------------------------------

    @classmethod
    def create(cls, p1_name: str, p2_name: str, p1_leader, p1_cards, p2_leader, p2_cards,
               first_player: Optional[str] = None, card_db=None) -> "RsGame":
        """デッキ（`load_deck_mixed` の戻り値）から対局を作る。

        `first_player` は "p1"／"p2"／None（既定＝p1）。**"random"（コイントス）は呼び出し側で
        解決してから渡す**——Python 版と乱数の消費位置を揃えるため
        （`services.games._resolve_first_player_seat`）。
        """
        return cls.create_from_ids(
            p1_name, p2_name,
            p1_leader.master.card_id if p1_leader else None,
            [c.master.card_id for c in p1_cards],
            p2_leader.master.card_id if p2_leader else None,
            [c.master.card_id for c in p2_cards],
            first_player, card_db=card_db)

    @classmethod
    def create_from_ids(cls, p1_name: str, p2_name: str,
                        p1_leader: Optional[str], p1_deck: List[str],
                        p2_leader: Optional[str], p2_deck: List[str],
                        first_player: Optional[str] = None, card_db=None) -> "RsGame":
        """card_id の列から直接作る（録画の再生＝`tests/harness/rs_replay.py` が使う）。"""
        engine = load_engine()
        game = engine.Game(
            p1_name, p2_name, p1_leader, list(p1_deck), p2_leader, list(p2_deck),
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

    # --- CPU の思考（Rust `decide`）-----------------------------------------

    def decide(self, player_id: str, trace: Optional[Dict[str, Any]] = None):
        """[`RsGame._decide`] の薄いラッパ（`trace` を渡すと思考の内訳を書き込む）。"""
        move, tr = self._decide(player_id)
        if trace is not None and move is not None:
            trace.update(tr)
        return move

    def _decide(self, player_id: str):
        """`player_id` の 1 手を Rust の探索で決める（`opcg_engine.Game.decide`）。

        **生の盤面**（中断スタックを持ったまま）に対して決めるので、対話の途中でも正しく読める。
        探索のつまみは serve 既定（`learned/config.py` の共有既定＝Rust 側 `DecideOptions` の
        既定）。decide をまたぐ状態（箱コミットの残り手順・残り起動の待ち）は `(ターン, 席)` が
        変わるまで持ち越す（Python `LearnedEngine._commits` と同じ鍵）。

        探索の乱数は seed から決まる（`search_seed`）。**同じ (対局, ターン, 席) には同じ seed** を
        渡す＝ターン内 sticky 世界線（`_world_rng` と同じ意味論）。対局ごとの基点 `_search_base`
        は生成時に 1 度だけ引く（`random` 由来＝`random.seed()` を張った traced 対局は再現する）。
        """
        engine = load_engine()
        if not self._net_loaded:
            engine.load_net(_net_path())
            self._net_loaded = True
        turn = self.turn_count
        seat = "p1" if player_id == self.p1_name else "p2"
        carry = self._carry if self._carry_key == (turn, seat) else {}
        opts = {"search_seed": _search_seed(self._search_base, turn, seat),
                "commit": carry.get("commit") or [],
                "resact_pending": bool(carry.get("resact_pending"))}
        out = json.loads(self._game.decide(player_id, json.dumps(opts)))
        self._carry_key = (turn, seat)
        self._carry = {"commit": out.get("commit") or [],
                       "resact_pending": bool(out.get("resact_pending"))}
        return out.get("move"), self._trace(out)

    def _trace(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """思考の内訳（旧 `cpu_learned._fill_trace` の欄のうち、Rust から出せるもの）。

        `chosen`／`dialog`／`candidates`（等価手マージ後の訪問上位・visit%・行動価値 Q）／
        `value`。**decide が返した後の盤面を読まない**（決定は盤面を動かさない）ので、記述は
        決定時点のもの。旧版にあった L1 の第二意見（`readout` の一部）は L1 廃止で無くなった。
        """
        move = out.get("move")
        tr: Dict[str, Any] = {"difficulty": "learned", "turn": self.turn_count,
                              "chosen": self.describe_move(move) if move else None,
                              "readout": out.get("kind")}
        pending = json.loads(self._game.pending_json())
        if pending and pending.get("action"):
            tr["dialog"] = pending["action"]
        legal = (out.get("stats") or {}).get("legal") or []
        groups = out.get("groups") or []
        if legal and groups:
            tot = sum(float(g["n"]) for g in groups) or 1.0
            tr["candidates"] = [{
                "move": self.describe_move(legal[g["rep"]]),
                "visit_pct": round(100.0 * float(g["n"]) / tot, 1),
                "q": round(float(g["q"]), 3),
                **({"copies": len(g["idxs"])} if len(g["idxs"]) > 1 else {}),
            } for g in groups[:5]]
            sig = out.get("sig")
            for g in groups:
                # 選んだ手の Q（所属グループの加重平均）。`sig` は箱レベルの鍵＝候補と同じ粒度。
                if sig is not None and self._sig(legal[g["rep"]]) == sig:
                    tr["value"] = round(float(g["q"]), 3)
                    break
        return tr

    @staticmethod
    def _sig(mv: Dict[str, Any]):
        """`learned/plan.move_sig` と同じ鍵（Rust の `sig` と JSON で突き合わせる）。"""
        p = mv.get("payload") or {}
        return [mv.get("action_type") or p.get("action_type"), p.get("uuid"),
                list(p.get("target_ids") or ()), list(p.get("selected_uuids") or ()),
                p.get("accepted")]

    # --- 思考トレース／リプレイの記録（opt-in・観測専用）---------------------

    def describe_move(self, move: Dict[str, Any]) -> Dict[str, Any]:
        """手を card_id 基準の記述 dict へ（旧 `cpu_ai._describe_move`）。**適用前**に呼ぶ。"""
        return json.loads(self._game.describe_move_json(
            json.dumps(move, ensure_ascii=False, default=str)))

    def deck_counts(self) -> Dict[str, int]:
        """山札の残り枚数（`{"p1": n, "p2": n}`）。リプレイフレームだけが使う。"""
        return json.loads(self._game.deck_counts_json())
