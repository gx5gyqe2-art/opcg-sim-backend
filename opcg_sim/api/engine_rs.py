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
# 符号化の語彙（`opcg_engine.set_vocab`）。棋譜ダンプ・§20.4 の `RsGame.attribution`（`Game.encode`）
# が要る。**最初に読んだネットの語彙**をプロセスの語彙にする（`opcg_sim.loop.engine.load_net` と
# 同じ「1 本目が既定」規約）。
_vocab_lock = threading.Lock()
_vocab_loaded = False


def _ensure_vocab(engine, summary: Dict[str, Any]) -> None:
    """ネットの要約 JSON（`load_net` の戻り値）の `vocab_ids` を、プロセスで 1 度だけ設定する。"""
    global _vocab_loaded
    if _vocab_loaded:
        return
    with _vocab_lock:
        if _vocab_loaded:
            return
        engine.set_vocab(json.dumps(summary.get("vocab_ids") or []))
        _vocab_loaded = True


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

    def __init__(self, game, p1_name: str, p2_name: str, card_db=None, net: Optional[str] = None):
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
        # 箱コミットを焼き込んだ決定の action_index（§20.4 の思考ログ・`trace["commit_from"]`）。
        # `_carry_key` と同じ (turn, seat) の間だけ有効＝キーが変われば自然に無効になる。
        self._commit_from_index: Optional[int] = None
        # このインスタンスの既定ネット（省略時は `_net_path()`＝出荷既定）。`decide` は呼び出し
        # ごとに `net` で上書きできる（`tests/scripts/rs_scenario_play.py` が席ごとに別ネットで
        # 打たせるのに使う・§20.1）。複数ネットを同一プロセスで読める（`opcg_sim.loop.engine`）ので
        # 1 インスタンスが両方の鍵を使い分けても問題ない。
        self._net = net
        self._loaded_nets: set = set()

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

    @classmethod
    def from_hidden(cls, hidden: Dict[str, Any], p1_name: str = "p1", p2_name: str = "p2",
                    seed: Optional[int] = None, card_db=None, net: Optional[str] = None) -> "RsGame":
        """記録 v5 の `hidden` から対局を組み直す（`docs/rust_engine_plan.md` §20.1・`rs-scenario`）。

        `opcg_engine.Game.from_hidden` の薄い口。**中断（対話）スタック・誘発待ち行列は
        `hidden` に無い**ので失われる（記録 v5 の契約・`hidden` 側の docstring 参照）。
        """
        engine = load_engine()
        game = engine.Game.from_hidden(
            json.dumps(hidden, ensure_ascii=False, default=str), p1_name, p2_name, seed)
        return cls(game, p1_name, p2_name, card_db=card_db, net=net)

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

    def decide(self, player_id: str, trace: Optional[Dict[str, Any]] = None,
              net: Optional[str] = None, sims: Optional[int] = None,
              action_index: Optional[int] = None, select_rule: Optional[str] = None,
              q_min_frac: Optional[float] = None, root_prior_temp: Optional[float] = None,
              worlds: Optional[int] = None):
        """[`RsGame._decide`] の薄いラッパ（`trace` を渡すと思考の内訳を書き込む）。

        `net`／`sims`（省略可）はこの 1 回だけ serve 既定を上書きする（席ごとに別ネット・
        軽い探索数で打たせたいとき・`tests/scripts/rs_scenario_play.py`・§20.1）。
        `action_index`（省略可）は呼び出し側の決定番号（`tests/scripts/rs_scenario_play.py` の
        `len(actions)`）＝箱コミットを焼き込んだ決定を覚えておくための鍵（`trace["commit_from"]`・
        §20.4）。省略すると `commit_from` は出さない。

        `select_rule`／`q_min_frac`／`root_prior_temp`（省略可・§20.5）は探索の設定の上書き:
        `select_rule`＝根で出す手の選び方（"visits"＝訪問数最多〔既定〕／"q_min_n"＝訪問数が
        `max(1, floor(sims * q_min_frac))` 以上の手の中で Q 最大）・`q_min_frac`＝その割合
        （既定 0.125＝sims/8）・`root_prior_temp`＝根の事前分布を `P^(1/t)` へ丸めて正規化
        （既定 1.0＝何もしない・2.0 で平坦化）。**省略すれば serve 既定と 1 bit も変わらない**。

        `worlds`（省略可・§20.7.1）＝1 回の決定で引く世界サンプルの本数（既定 1）。2 以上なら
        Rust が K 本の世界で同じ sims の木を**並列**に回して根の統計を束ねる（N は和・Q は N
        重みの平均）。trace に `worlds`／`world_used`／`per_world`（世界ごとの根の N/Q と最善手）
        が入る。**省略すれば 1 本＝今までと同じ**（trace の欄も増えない）。
        """
        move, tr = self._decide(player_id, net=net, sims=sims, action_index=action_index,
                                select_rule=select_rule, q_min_frac=q_min_frac,
                                root_prior_temp=root_prior_temp, worlds=worlds)
        if trace is not None and move is not None:
            trace.update(tr)
        return move

    def _decide(self, player_id: str, net: Optional[str] = None, sims: Optional[int] = None,
               action_index: Optional[int] = None, select_rule: Optional[str] = None,
               q_min_frac: Optional[float] = None, root_prior_temp: Optional[float] = None,
               worlds: Optional[int] = None):
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
        net_path = net or self._net or _net_path()
        if net_path not in self._loaded_nets:
            summary = json.loads(engine.load_net(net_path))
            self._loaded_nets.add(net_path)
            _ensure_vocab(engine, summary)
        turn = self.turn_count
        seat = "p1" if player_id == self.p1_name else "p2"
        carry = self._carry if self._carry_key == (turn, seat) else {}
        had_commit = bool(carry.get("commit"))
        opts = {"net": net_path,
                "search_seed": _search_seed(self._search_base, turn, seat),
                "commit": carry.get("commit") or [],
                "resact_pending": bool(carry.get("resact_pending"))}
        if sims is not None:
            opts["sims"] = int(sims)
        # §20.5 の 3 つ（None＝渡さない＝Rust 側の既定）。
        if select_rule is not None:
            opts["select_rule"] = str(select_rule)
        if q_min_frac is not None:
            opts["q_min_frac"] = float(q_min_frac)
        if root_prior_temp is not None:
            opts["root_prior_temp"] = float(root_prior_temp)
        # §20.7.1（None／1＝渡さない＝Rust 側の既定＝単一世界）。
        if worlds is not None and int(worlds) > 1:
            opts["worlds"] = int(worlds)
        out = json.loads(self._game.decide(player_id, json.dumps(opts)))
        self._carry_key = (turn, seat)
        self._carry = {"commit": out.get("commit") or [],
                       "resact_pending": bool(out.get("resact_pending"))}
        # commit_from（§20.4）: このターン/席で箱コミットを初めて焼き込んだ決定の action_index を
        # 覚えておく。既に焼き込み済み（had_commit）なら覚えたままの鍵を使う＝焼き込んだ決定を指す。
        if had_commit:
            commit_from = self._commit_from_index
        else:
            commit_from = None
            self._commit_from_index = action_index if out.get("commit") else None
        return out.get("move"), self._trace(out, commit_from=commit_from)

    def _trace(self, out: Dict[str, Any], commit_from: Optional[int] = None) -> Dict[str, Any]:
        """思考の内訳（旧 `cpu_learned._fill_trace` の欄のうち、Rust から出せるもの）。

        `chosen`／`dialog`／`candidates`（等価手マージ後の訪問上位・visit%・行動価値 Q）／
        `value`。**decide が返した後の盤面を読まない**（決定は盤面を動かさない）ので、記述は
        決定時点のもの。旧版にあった L1 の第二意見（`readout` の一部）は L1 廃止で無くなった。

        §20.4（思考ログ）で足した欄: `kind`（`readout` と同じ値・欄名を揃えただけ）・
        `pv`（主変化・record["pv"] を describe_move で記述子化）・`legal_stats`（全合法手の
        P/N/Q・stats をそのまま並べる）・`commit_from`（kind=="commit" のときだけ）。
        """
        move = out.get("move")
        kind = out.get("kind")
        tr: Dict[str, Any] = {"difficulty": "learned", "turn": self.turn_count,
                              "chosen": self.describe_move(move) if move else None,
                              "readout": kind, "kind": kind}
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
        # 全合法手の P・N・Q（箱化後の候補＝探索が見た手・訪問 0 も含む）。
        stats = out.get("stats") or {}
        ns, qs, ps = stats.get("N") or [], stats.get("Q") or [], stats.get("P")
        if legal:
            tr["legal_stats"] = [{
                "move": self.describe_move(mv),
                "p": (float(ps[i]) if ps is not None and i < len(ps) else None),
                "n": (int(ns[i]) if i < len(ns) else None),
                "q": (float(qs[i]) if i < len(qs) else None),
            } for i, mv in enumerate(legal)]
        # PV（主変化）: 記述子（card_id 基準）に直す。相手の伏せ手札など、根の世界サンプル
        # にしか無いカードの uuid は実盤面で引けない＝describe_move はその uuid をそのまま返す
        # （§20.4 の注記）。
        pv = out.get("pv") or []
        if pv:
            tr["pv"] = [{
                "move": self.describe_move(p.get("move") or {}),
                "seat": p.get("seat"),
                "n": (int(p["n"]) if p.get("n") is not None else None),
                "q": (float(p["q"]) if p.get("q") is not None else None),
            } for p in pv]
        # 複数世界（§20.7.1）: `worlds>1` のときだけ Rust が出す欄。世界ごとの根の N/Q は
        # **世界 0 の legal の並び**なので `legal_stats` と添字が揃う。
        if out.get("worlds"):
            tr["worlds"] = int(out["worlds"])
            tr["world_used"] = out.get("world_used")
            tr["per_world"] = [{
                "seed": w.get("seed"),
                "best": (self.describe_move(legal[w["best"]])
                         if (w.get("best") is not None and w["best"] < len(legal)) else None),
                "best_n": (float(w["N"][w["best"]])
                           if (w.get("best") is not None and w["best"] < len(w.get("N") or []))
                           else None),
                "best_q": (round(float(w["Q"][w["best"]]), 3)
                           if (w.get("best") is not None and w["best"] < len(w.get("Q") or []))
                           else None),
                "N": [float(x) for x in (w.get("N") or [])],
                "Q": [float(x) for x in (w.get("Q") or [])],
                "unmapped": w.get("unmapped"),
                "p_differs": w.get("p_differs"),
            } for w in (out.get("per_world") or [])]
        if kind == "commit":
            tr["commit_from"] = commit_from
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

    #: 22 枠のゾーン名（`n_rel_feat._slots`／`docs/rust_engine_plan.md` §20.4 の並びと同じ:
    #: 自L・相L・自場5・相場5・手札10）。
    _ZONE_LABELS = (["own_leader", "opp_leader"] + ["own_field"] * 5 + ["opp_field"] * 5
                    + ["hand"] * 10)

    def _slot_labels(self, player_id: str) -> List[str]:
        """22 枠 → `"<ゾーン>[<枠>]:<card_id>"`（`dump_index_json` の枠→uuid→card_id を組む・
        空枠や uuid が引けない枠はゾーン名＋枠番号のみ）。
        """
        idx = json.loads(self._game.dump_index_json(player_id))
        slot_to_uuid = {v: k for k, v in (idx.get("slots") or {}).items()}
        cids = idx.get("cids") or {}
        out = []
        for i, zone in enumerate(self._ZONE_LABELS):
            uuid = slot_to_uuid.get(i)
            card_id = cids.get(uuid) if uuid else None
            out.append(f"{zone}[{i}]:{card_id}" if card_id else f"{zone}[{i}]")
        return out

    def attribution(self, player_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """V の帰属（§20.4 の 3・scenario の play にだけ呼ぶ・serve には付けない）。

        根の符号化（`self._game.encode`＝**実盤面**。decide 内部の世界サンプルではない）の
        22 枠を 1 つずつ PAD（`tok` を 0 埋め・`card_idx` を 0）へ潰し、`opcg_engine.net_eval`
        で value を取り直す（決定ごとに 23 回の forward＝軽い）。`dv = v0 - v_masked`（その枠を
        消すと value がどれだけ動くか）の絶対値上位 `top_k` を返す。相関であって理由ではない
        （§20.4 冒頭の注記）。

        **制約**: `net_eval` はプロセスの既定ネット（最初に `load_net` した npz。`opcg_engine`
        の「1 本目が既定」規約）を使う——`decide` が席ごとに別ネットを選べるのと違い、
        帰属は席ごとのネット切替を追わない。両席とも同じネットで打たせる運用（本 WP の
        9 シナリオ）ではこの差は出ない。
        """
        engine = load_engine()
        enc = json.loads(self._game.encode(player_id))
        enc["tok"] = enc.pop("tokens", None) or []
        tok = list(enc["tok"])
        card_idx = list(enc.get("card_idx") or [])
        n_tok = len(self._ZONE_LABELS)
        s_dim = (len(tok) // n_tok) if n_tok else 0
        base = json.loads(engine.net_eval(json.dumps(enc), "[]"))
        v0 = float(base["value"])
        labels = self._slot_labels(player_id)
        out = []
        for i in range(n_tok):
            t = list(tok)
            for k in range(s_dim):
                t[i * s_dim + k] = 0.0
            ci = list(card_idx)
            if i < len(ci):
                ci[i] = 0
            mod = dict(enc, tok=t, card_idx=ci)
            r = json.loads(engine.net_eval(json.dumps(mod), "[]"))
            out.append({"slot": i, "label": labels[i], "dv": v0 - float(r["value"])})
        out.sort(key=lambda d: abs(d["dv"]), reverse=True)
        return out[:top_k]

    def deck_counts(self) -> Dict[str, int]:
        """山札の残り枚数（`{"p1": n, "p2": n}`）。リプレイフレームだけが使う。"""
        return json.loads(self._game.deck_counts_json())
