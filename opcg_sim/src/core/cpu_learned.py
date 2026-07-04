"""学習型CPU（Gen2 value+policy + NN誘導MCTS）の本番エントリ。

docs/reports/cpu_rl_pilot_p3_results_20260630.md。P3本走で得た Gen2 ネット（自己対戦2世代・
製品L1+α-βに 0.925・かつ製品より高速）を実ゲームに配線する。返り値は `decide_guarded` と同一契約
（単一 move 辞書 or None）＝decide 経路にドロップイン可能。

- 葉価値 = 学習 value ネット、事前確率 = 学習 pointer policy、探索 = ノード型 PUCT MCTS。
- 不完全情報は探索ごとに1世界へ決定化（PIMC・チート防止）。net/vocab はプロセス内で1回だけロード。

**ネットの持ち方（perf計画 A3）**: `LearnedEngine` が1つのネットを**明示ハンドル**で保持する
（arena の net-vs-net＝新Gen vs 凍結Gen2 を同一プロセスで戦わせるため）。本番既定 CPU が通る
`decide_learned` は既定ネットの**プロセス共有シングルトン**（`_default_engine()`）を使う薄いラッパ
＝**挙動不変**（vocab/game はネット非依存なので複数エンジンで共有ロード可能）。
"""
import os
from typing import Any, Dict, Optional

import numpy as np

from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned.value_net import ValueNet
from opcg_sim.src.learned.policy import PolicyScorer, state_context
from opcg_sim.src.learned.action import legal_action_matrix
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.learned.mcts import TreeMCTS
from opcg_sim.src.utils.loader import CardLoader

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "learned")
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")

_DEFAULT_VALUE = os.path.join(_MODELS, "gen2_value.npz")
_DEFAULT_POLICY = os.path.join(_MODELS, "gen2_policy.npz")

# vocab（カード語彙）と game（アダプタ）はネット非依存＝プロセス内で1回だけ作り全エンジンで共有する。
_SHARED: Dict[str, Any] = {}


def _shared_vocab_game():
    if not _SHARED:
        db = CardLoader(os.path.join(_DATA, "opcg_cards.json"))
        db.load()
        for cid in list(db.raw_db.keys()):
            db.get_card(cid)
        _SHARED["vocab"] = E.build_vocab(db)
        _SHARED["game"] = OPCGGame()
    return _SHARED["vocab"], _SHARED["game"]


def available() -> bool:
    """モデル重みが同梱されているか（未同梱環境ではフォールバックさせる）。"""
    return os.path.exists(_DEFAULT_VALUE)


def _value_fn(vnet, vocab):
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(vnet.predict(batch)[0])
    return value


def _priors_fn(pnet, vocab):
    if pnet is None:
        return None
    def priors(state, legal):
        me = state.pending_actor_action()[0]
        ctx = state_context(state, me, vocab)
        am = legal_action_matrix(state, legal, me)
        p = pnet.priors(ctx, am)
        return p if p.shape[0] == len(legal) else None
    return priors


class LearnedEngine:
    """1つの Gen2 ネット（value+policy）を明示ハンドルで保持し 1 手を決める。

    net-vs-net（arena で新Gen vs 凍結Gen2）用に、ネットを**席ごとに別インスタンス**で持てるようにする。
    `value_path`/`policy_path` 省略時は出荷 Gen2（`gen2_*.npz`）＝本番既定 CPU と同一。vocab/game は
    ネット非依存なので既定では共有ロード（`_shared_vocab_game`）を使う。
    """

    def __init__(self, value_path: Optional[str] = None, policy_path: Optional[str] = None,
                 vocab=None, game=None):
        if vocab is None or game is None:
            svocab, sgame = _shared_vocab_game()
            vocab = vocab if vocab is not None else svocab
            game = game if game is not None else sgame
        self.vocab = vocab
        self.game = game
        self.vnet = ValueNet.load(value_path or _DEFAULT_VALUE)
        pp = policy_path or _DEFAULT_POLICY
        self.pnet = PolicyScorer.load(pp) if os.path.exists(pp) else None

    def decide(self, manager, player, sims: int = 160, c_puct: float = 1.5,
               rng=None, trace=None) -> Optional[Dict[str, Any]]:
        """このエンジンのネットで 1 手決定する（`decide_learned` と同一契約・同一探索）。"""
        name = player.name
        # numpy rng の種を **global random** から引く＝リプレイ種（routers が cpu_trace 時に random.seed）で
        # learned 対局も決定論再生できる。通常対局は global random 未 seed（プロセス由来）＝実質ランダム。
        if not isinstance(rng, np.random.Generator):
            import random as _random
            rng = np.random.default_rng(_random.getrandbits(64))
        mcts = TreeMCTS(self.game, value_fn=_value_fn(self.vnet, self.vocab),
                        priors_fn=_priors_fn(self.pnet, self.vocab),
                        c_puct=c_puct, n_sims=sims,
                        determinize_fn=lambda s, r: self.game.determinize(s, name, r), rng=rng)
        move, _, legal = mcts.run(manager)
        # 同名カードの別実体（手札の複製等）は探索木で別 edge になり訪問数が分裂する。
        # 素の argmax(N) は分裂した等価手を系統的に不利にする（例: EB03-053×2 のカウンターが
        # 30.6%+30.6% に割れ、38.8% の PASS に負ける）ため、等価キーで訪問数を合算した
        # グループの多数決で選ぶ。探索（TreeMCTS）自体は不変＝ルートの読み出しのみ補正。
        stats = getattr(mcts, "last_stats", None)
        if stats and stats.get("legal"):
            groups = _merge_root_stats(manager, stats["legal"], stats["N"], stats["Q"])
            if groups:
                move = stats["legal"][groups[0]["rep"]]
        if move is None:
            move = legal[0] if legal else None
        if trace is not None:
            try:
                _fill_trace(trace, manager, player, move, getattr(mcts, "last_stats", None))
            except Exception:
                pass   # 分析失敗で対局を止めない
        return move


_DEFAULT_ENGINE: Optional[LearnedEngine] = None


def _default_engine() -> LearnedEngine:
    """本番既定 CPU（出荷 Gen2）のプロセス共有シングルトンエンジン。"""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = LearnedEngine()
    return _DEFAULT_ENGINE


def _lazy_init():
    """後方互換のウォームアップ（既定エンジンを1回ロード）。perf 計測等が初回ロードを計測から外すのに使う。"""
    _default_engine()


def decide_learned(manager, player, sims: int = 160, c_puct: float = 1.5,
                   rng=None, trace=None) -> Optional[Dict[str, Any]]:
    """学習型CPUの1手決定（本番既定 CPU 経路）。返り値は `decide_guarded` 互換（move 辞書 or None）。

    出荷 Gen2 のシングルトンエンジンへ委譲する薄いラッパ＝A3 のリファクタで**挙動不変**。
    `trace`（dict）が渡された時（cpu_trace ON）は、その手の分析を書き込む＝変な手の検証用ログ:
      chosen（選んだ手）/ value（選手の行動価値Q）/ candidates（訪問上位・visit%・Q）/
      l1_move（独立評価器L1の推奨手）/ l1_disagrees（L1と食い違うか）。
    分析は挙動に影響しない（例外は握り潰し、手は必ず返す）。
    """
    return _default_engine().decide(manager, player, sims=sims, c_puct=c_puct, rng=rng, trace=trace)


def _merge_root_stats(manager, legal, N, Q):
    """ルート合法手を挙動等価キー（`cpu_ai._move_equiv_key`）でグループ化し訪問数を合算する。

    返り値: [{"rep": 代表index(グループ内N最大), "idxs": [...], "n": N合算, "q": N加重平均Q}]
    を n 降順（同数は legal 列挙順＝安定）で。等価手が無い局面では全グループが単独＝
    先頭グループの rep が従来の argmax(N) と一致し**挙動不変**。

    等価判定は card_id 基準＝リプレイ逆写像（`replay_runner._key`）と同じ同一視。場の複製
    （同名キャラで付与ドン数が違う等）は厳密には非等価だが、その残差はリプレイ側と同じ
    許容（R0 §5）に揃える。
    """
    from opcg_sim.src.core import cpu_ai
    order, groups = [], {}
    for i, mv in enumerate(legal):
        k = cpu_ai._move_equiv_key(manager, mv)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)
    out = []
    for k in order:
        idxs = groups[k]
        n = float(sum(float(N[i]) for i in idxs))
        q = (sum(float(N[i]) * float(Q[i]) for i in idxs) / n) if n > 0 else 0.0
        rep = max(idxs, key=lambda i: float(N[i]))
        out.append({"rep": rep, "idxs": idxs, "n": n, "q": q})
    out.sort(key=lambda g: -g["n"])   # sort は安定＝同数なら列挙順を保つ
    return out


def _fill_trace(trace, manager, player, chosen, stats):
    """トレース dict に learned の意思決定分析を書き込む（cpu_trace 時のみ呼ばれる）。"""
    from opcg_sim.src.core import cpu_ai
    import random as _random
    trace["difficulty"] = "learned"
    trace["turn"] = getattr(manager, "turn_count", None)
    trace["chosen"] = cpu_ai._describe_move(manager, chosen) if chosen else None
    # 対話種別（SEARCH_AND_SELECT / ARRANGE_DECK / CONFIRM_OPTIONAL 等）。無いと
    # 「ライフ追加の選択」か「底送りの順番」かがトレースから読めない。
    pend = manager.get_pending_request() or {}
    if pend.get("action"):
        trace["dialog"] = pend.get("action")
    # ① 自分の探索の内訳（等価手マージ後の訪問上位・visit%・行動価値Q）。decide の選択と
    #    同じ集計（`_merge_root_stats`）で出す＝「分裂した同名手が別行に出て PASS に負けて
    #    見える」ログ上の錯覚も消す。copies>1 は複製がマージされた印。
    if stats and stats.get("legal"):
        legal, N, Q = stats["legal"], stats["N"], stats["Q"]
        tot = float(N.sum()) or 1.0
        groups = _merge_root_stats(manager, legal, N, Q)
        trace["candidates"] = [{
            "move": cpu_ai._describe_move(manager, legal[g["rep"]]),
            "visit_pct": round(100.0 * g["n"] / tot, 1),
            "q": round(g["q"], 3),
            **({"copies": len(g["idxs"])} if len(g["idxs"]) > 1 else {}),
        } for g in groups[:5]]
        # 選んだ手の Q（＝net が見込む行動価値・所属グループの加重平均）。
        for g in groups:
            if any(legal[i] is chosen for i in g["idxs"]):
                trace["value"] = round(g["q"], 3)
                break
    # ② 独立評価器 L1 の第二意見（分布外での net 系統誤差を拾う・evalは信じ過ぎない）。
    try:
        clone = manager.clone()
        cp = clone.p1 if clone.p1.name == player.name else clone.p2
        l1 = cpu_ai.decide_guarded(clone, cp, "hard", _random.Random(0), {}, pimc_worlds=1)
        trace["l1_move"] = cpu_ai._describe_move(clone, l1) if l1 else None
        trace["l1_disagrees"] = bool(l1 and chosen and
                                     l1.get("action_type") != chosen.get("action_type"))
    except Exception:
        pass
