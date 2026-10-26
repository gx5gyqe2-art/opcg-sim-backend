"""Rust エンジンの起動と席のつまみ（`opcg_sim.loop` の土台）。

プロセスで 1 度だけやること:

1. `opcg_engine.load_masters(opcg_effects.json)` … カード定義表（効果構造 JSON は
   `opcg_sim/tools/export_effects_json.py` の生成物・無ければその場で作る）
2. `opcg_engine.load_net(npz)` … NRel の重み。**席ごとに別のネットを読める**
   （アリーナは候補と基準を 1 プロセスで持つ）。鍵は npz のパスで、`Game.decide` の
   `opts["net"]` が選ぶ。最初に読んだものが既定。
3. `opcg_engine.set_vocab(...)` … 符号化の語彙（棋譜ダンプに要る。ネット付属の `vocab_ids`）

**乱数の規約**（計画 §8.17 の申告 (2)）: 探索の乱数は `search/rng.rs` の `Pcg32SearchRng` で、
`Game.decide` は毎回 `opts["search_seed"]` から作り直す。同じ (対局, ターン, 席) には同じ seed を
渡す＝ターン内 sticky 世界線（Python `LearnedEngine._world_rng` と同じ意味論）。seed の導出は
[`search_seed`]（対局 seed とターンと席から決まる＝プロセス数や実行順に依存しない）。
"""
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_REPO_ROOT, "opcg_sim", "data")
EFFECTS_PATH = os.path.join(DATA, "opcg_effects.json")
MODELS = os.path.join(DATA, "learned")
CARDS_PATH = os.path.join(DATA, "opcg_cards.json")

#: 出荷既定のネット（生成・アリーナ・serve が同じものを使う＝物差しを 1 本に保つ）。
#: 前の既定 r23（2026-10-25 採用・`docs/reports/r23_adoption_20261025.md`・r22 が打った波 52 の π と波 52＋51 の価値で r22 から育てた＝era13）。
#: **r24**（2026-10-25 採用・`docs/reports/r24_adoption_20261025.md`・r23 が打った波 53 の π と波 53＋52 の価値で r23 から育てた＝era14）。
#: **r26**（2026-10-26 採用・`docs/reports/r26_adoption_20261026.md`・r24 を学ぶ側に相手の集団〔r24／r23／r22／r13d〕と打った
#: 波 55L の π〔学ぶ側の行だけ〕と波 55L＋54 の価値で r24 から育てた＝era15・本番の設定で r24 に +18.9）。
#: **r27**（2026-10-26 採用・`docs/reports/r27_adoption_20261026.md`・r26 の自己対戦の 2 波〔波 56＋57〕の π と価値で r26 から育てた
#: ＝era16・価値の教師は今の生成役の 2 波だけ〔2026-10-26 の決まり〕・本番の設定〔世界 4 本〕で r26 に +9.2）。
#: ロールバック先は r26（同梱 `nrel_r26.npz`）、その先は r24（`nrel_r24.npz`）、r23（`nrel_r23.npz`）、r22（`nrel_r22.npz`）、r13d（`nrel_r13d.npz`）。
DEFAULT_NET = os.path.join(MODELS, "nrel_r27.npz")

#: serve 既定（`opcg_sim/learned/config.py` の共有既定と同じ値。Rust 側 `DecideOptions`
#: の既定と一致するので、通常はここを触らない）。
SERVE_SIMS = 160
#: 対戦 API（`/api/game/cpu/step`）の CPU **だけ**の探索。8 倍の探索＋戦闘箱の枝予算なし
#: （2026-10-18 ユーザ決定・主条件 2,622 ペアで出荷 160 回に +46.5 Elo・1 手平均 0.8 秒・
#: 最大 19 秒・`docs/reports/2026-10-18_serve_sims.md`）。生成・アリーナ・golden の物差しは
#: `SERVE_SIMS`／既定の予算のまま（ここを触っても学習側は変わらない）。0＝予算なし。
PLAY_SIMS = 1280
PLAY_BUDGET = 0
#: 対戦 API の CPU が 1 回の決定で引く世界（相手の手札・山札の並びの引き直し）の本数。
#: 2026-10-21 ユーザ決定「b でお願いします」: 世界 2 本 × 1,280 回（同じ seed 帯の 1,438 ペアで +18.1 Elo・
#: 世界 4 本／8 本・5,120 回／10,240 回とは組の差が有意でない＝計算 ×2 で頭打ち・1 コアの 1 手平均 0.24 秒・最大 2.2 秒。
#: `docs/reports/2026-10-21_phase5_curve.md`・`2026-10-21_phase6_materials.md`）。生成・アリーナの物差しは 1 本のまま。
#: 2026-10-26 ユーザ決定「4 本にする」: 世界 4 本（今の 2 本に +13.2 Elo [+6.6, +19.8]・`docs/reports/2026-10-25_worlds4_prod_arena.md`）。
#: 1 コアの機械では世界を順に読むので 1 手平均 0.31 → 0.61 秒・最悪 3.5 → 17 秒（`2026-10-25_world_parallel.md`）。
#: 長い思考はフロントに経過秒を出して見せる（`api/services/cpu_think.py`）。
PLAY_WORLDS = 4
#: 対戦 API の「迷う局面だけ世界 K 本で読み直す」（`docs/search_efficiency_plan.md` §7.4・WP
#: `play-rethink`）。根の 2 位の訪問数 ÷ 1 位の訪問数が `PLAY_RETHINK_RATIO` 以上なら、同じ
#: `search_seed` で世界 `PLAY_RETHINK_WORLDS` 本の読みをやり直してその手に差し替える。
#: **None＝off（既定）**＝1 本の読みのまま 1 bit も変わらない。線は Phase 0 の結果で決める。
PLAY_RETHINK_RATIO: Optional[float] = None
PLAY_RETHINK_WORLDS = 4
#: 対戦 API の「詰みの証明」（`docs/search_efficiency_plan.md` §7.21・WP `rs-lethal-root`）。"off"／"root"／
#: "root_reply"。root＝自分のメインでこのターンに届きうるとき、公平な配り方（世界の引き直し 6 通り＋守りの強い
#: 2 通り）の全部で勝てる最初の 1 手が在ればそれを打つ（節点 20,000 で打ち切り・証明できなければ今の探索）。
#: root_reply＝加えて根の直後の相手の応手の節点でも証明する。**"off"（既定）**＝1 bit も変わらない。
#: **対戦 API の既定は "root"**（2026-10-22 ユーザ決定「入れる」・本番アリーナ +5.2 Elo［+3.1, +7.3］・1 手 +37%・
#: `docs/reports/2026-10-22_lethal_prod_arena.md`）。生成・アリーナは各自の旗のまま（既定 off）。
PLAY_LETHAL_PROOF = "root"
GEN_PRUNE_FUTILE = False   # 生成は枝刈りを外す（v6 柱⑤: 刈った枝は学習データに現れない）

_STATE: Dict[str, Any] = {}


def engine():
    """`opcg_engine` を import し、カード定義表をプロセスで 1 度だけ読む（冪等）。"""
    import opcg_engine

    if not _STATE.get("masters"):
        if not os.path.exists(EFFECTS_PATH):
            # 生成物（git 管理外・約 8MB）。無ければその場でエクスポータを回す。**プールの
            # ワーカーが同時に来ても 1 本だけが作り、他は待つ**（`ensure`・2026-09-13 の実害:
            # 4 ワーカーが同じ tmp へ書いて壊れた JSON を公開し、ほぼ全員が落ちた）。
            from opcg_sim.tools import export_effects_json as EEJ
            EEJ.ensure(EFFECTS_PATH)
        opcg_engine.load_masters(EFFECTS_PATH)
        _STATE["masters"] = True
    return opcg_engine


def resolve_net(spec: Optional[str]) -> str:
    """ネット指定 → npz のパス。

    受ける形（歴代の台帳・指示書がそのまま通るようにする）:
      ""／None … 出荷既定（[`DEFAULT_NET`]）
      "neff:xxx.npz"／"n1:xxx.npz" … 接頭辞は N 系エンジンの注入経路の名残（Rust では
        ネットの種別は npz の鍵で決まる）＝**接頭辞を落としてパスとして扱う**
      "v.npz,p.npz" … G 系の value/policy ペア。**Rust は G 系を載せない**ので拒否する
        （物差しを 1 本に保つ・計画 §8.17 の設計 1）
      それ以外 … そのままパス
    """
    if not spec:
        return DEFAULT_NET
    for prefix in ("neff:", "n1:", "nrel:"):
        if spec.startswith(prefix):
            spec = spec[len(prefix):]
            break
    if "," in spec:
        raise ValueError(
            f"G 系の value/policy ペアは Rust エンジンに載らない（{spec}）。"
            "N 系（NRel／NEff）の npz 1 本を指定すること")
    return spec


def load_net(spec: Optional[str] = None) -> str:
    """ネットを読み、その鍵（`Game.decide` の `opts["net"]`）を返す。冪等。"""
    eng = engine()
    path = resolve_net(spec)
    if path not in _STATE.setdefault("nets", {}):
        summary = json.loads(eng.load_net(path))
        _STATE["nets"][path] = summary
        # 符号化の語彙はネット付属の vocab_ids（棋譜ダンプが使う）。**最初に読んだネットの
        # 語彙**をプロセスの語彙にする（アリーナは符号化しないので競合しない）。
        if not _STATE.get("vocab"):
            eng.set_vocab(json.dumps(summary["vocab_ids"]))
            _STATE["vocab"] = summary["vocab_ids"]
    return path


def vocab_ids() -> List[str]:
    """`set_vocab` したカード語彙（棋譜ダンプの `enc_version`／列の意味に対応）。"""
    if not _STATE.get("vocab"):
        raise RuntimeError("load_net() が先に要る（語彙はネット付属の vocab_ids）")
    return _STATE["vocab"]


def search_seed(game_seed: int, turn: int, seat: str) -> int:
    """探索乱数の seed（対局 seed・ターン・席から決まる・pure）。

    同じ (対局, ターン, 席) には同じ値を返す＝ターン内のどの decide も同じ世界線から始まる
    （Python `LearnedEngine._world_rng` は初回 decide の rng から seed を引いてターン内で
    使い回していた。こちらは**引数から決める**ので、判断点の順序や並列度に依存しない）。
    SplitMix64 で混ぜる（Rust の `Pcg32SearchRng::new` は 64bit の seed を取る）。
    """
    x = (game_seed * 0x9E3779B97F4A7C15
         + turn * 0xBF58476D1CE4E5B9
         + (1 if seat == "p2" else 0) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def rethink_ratio(out: Dict[str, Any]) -> Optional[float]:
    """根の迷いの尺度 r＝2 位の訪問数 ÷ 1 位の訪問数（pure）。

    読むのは**等価手マージ後のグループ**（`out["groups"]`・出す手もここから選ばれる）。木を
    回した決定（`kind == "main"`）で候補が 2 つ以上のときだけ値を返す。戦闘窓・対話窓の根畳み
    （`window`）・箱コミットの機械実行（`commit`）・候補 1 つの決定は None＝読み直さない
    （窓・コミットは世界の本数を見ないので、読み直しても同じ手になる）。
    """
    if out.get("kind") != "main":
        return None
    ns = sorted((float(g.get("n") or 0.0) for g in (out.get("groups") or [])), reverse=True)
    if len(ns) < 2 or ns[0] <= 0.0:
        return None
    return ns[1] / ns[0]


def decide_with_rethink(decide: Callable[[str], str], opts_json: str,
                        ratio: Optional[float] = None,
                        worlds: int = PLAY_RETHINK_WORLDS) -> Dict[str, Any]:
    """1 回の `Game.decide`（`decide(opts_json) -> out JSON`）＋迷うときだけの読み直し。

    対戦 API（`RsGame._decide`）とアリーナ・生成の席（`driver.run_game`）が**同じ関数**を
    通る＝物差しは 1 本（§7.4 の 2）。

    - `ratio is None`（既定）: `decide(opts_json)` を 1 回呼んだ結果をそのまま返す（off＝1 bit
      も変わらない・戻り値に鍵も足さない）。
    - それ以外: 1 本目の結果の [`rethink_ratio`] が `ratio` 以上なら、**同じ opts（同じ
      `search_seed`・同じ持ち越し）の `worlds` だけを K 本にして**読み直し、その結果に丸ごと
      差し替える（手・統計・持ち越し）。`decide` は盤面を動かさないので 2 回呼んでよい。
      opts が既に K 本以上の世界を持つときは読み直さない（同じ読みの繰り返しになる）。
      戻り値に `rethink: {r, fired, first_move, final_move}` を足す（発動率・入れ替わり率を
      後で数える・`r` は読み直さない決定では None）。
    """
    out = json.loads(decide(opts_json))
    if ratio is None:
        return out
    r = rethink_ratio(out)
    first = out.get("move")
    info: Dict[str, Any] = {"r": r, "fired": False, "first_move": first, "final_move": first}
    opts = json.loads(opts_json)
    if r is not None and r >= float(ratio) and int(worlds) > int(opts.get("worlds") or 1):
        opts["worlds"] = int(worlds)
        out = json.loads(decide(json.dumps(opts, ensure_ascii=False)))
        info["fired"] = True
        info["final_move"] = out.get("move")
    out["rethink"] = info
    return out


class SeatSpec:
    """1 席の思考（ネット＋探索のつまみ）。`Game.decide` の `opts` を組む。

    つまみは Rust の `DecideOptions`／`SearchOptions` の欄と同名（省略した欄は serve 既定）。
    Python の `LearnedEngine(**kw)` に渡していた席別 seam（`macro_moves`／`defense_box`／
    `box_dialog`／`box_commit`／`residual_dig`／`residual_activate`／`don_margin`）はそのまま
    通る＝アリーナの `--cand-*` フラグは規約を変えずに動く。
    """

    def __init__(self, net: Optional[str] = None, sims: int = SERVE_SIMS,
                 dirichlet_eps: float = 0.0, temp_turns: int = 0,
                 eps_play: float = 0.0, eps_hold: float = 0.0,
                 eps_target: float = 0.0, rethink_ratio: Optional[float] = None,
                 rethink_worlds: Optional[int] = None, **kw):
        self.net = load_net(net)
        #: 迷う局面だけ世界 K 本で読み直す（§7.4・[`decide_with_rethink`]）。**decide の opts には
        #: 入れない**——1 本目の結果を見てから 2 回目を呼ぶ Python 側のつまみ。既定 None＝off。
        self.rethink_ratio = None if rethink_ratio is None else float(rethink_ratio)
        self.rethink_worlds = int(rethink_worlds or PLAY_RETHINK_WORLDS)
        #: 両方向 ε 探索（§20.8.5・生成の対照づくり）。**decide の opts には入れない**——
        #: 木も π も変えず、「decide が返した後で打つ手だけを差し替える」Python 側のつまみ
        #: （差し替えの実体は `record_gen.eps_swap`・入口は `driver.run_game(swap=…)`）。
        #: 既定 0.0＝1 bit も変わらない。
        self.eps_play = float(eps_play or 0.0)
        self.eps_hold = float(eps_hold or 0.0)
        #: ε の**対象**ランダム化（§20.9 の D）。木が選んだ除去の直後の対象選択を確率
        #: `eps_target` で一様に引く（`forced=1` の直後は確率に依らず必ず引く）。既定 0.0。
        self.eps_target = float(eps_target or 0.0)
        self.opts: Dict[str, Any] = {"net": self.net, "sims": int(sims)}
        if dirichlet_eps:
            self.opts["dirichlet_eps"] = float(dirichlet_eps)
        if temp_turns:
            self.opts["temp_turns"] = int(temp_turns)
        for key, value in (kw or {}).items():
            if value is not None:
                self.opts[key] = value

    def decide_opts(self, game_seed: int, turn: int, seat: str,
                    carry: Optional[Dict[str, Any]] = None) -> str:
        """1 回の `Game.decide` に渡す opts JSON（seat 固有の乱数 seed と持ち越しを載せる）。"""
        opts = dict(self.opts)
        opts["search_seed"] = search_seed(game_seed, turn, seat)
        if carry:
            opts["commit"] = carry.get("commit") or []
            opts["resact_pending"] = bool(carry.get("resact_pending"))
        return json.dumps(opts, ensure_ascii=False)
