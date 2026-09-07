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
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_REPO_ROOT, "opcg_sim", "data")
EFFECTS_PATH = os.path.join(DATA, "opcg_effects.json")
MODELS = os.path.join(DATA, "learned")
CARDS_PATH = os.path.join(DATA, "opcg_cards.json")

#: 出荷既定のネット（`cpu_learned._DEFAULT_VALUE` と同じもの＝物差しを 1 本に保つ）。
DEFAULT_NET = os.path.join(MODELS, "nrel_a1.npz")

#: serve 既定（`opcg_sim/learned/config.py` の共有既定と同じ値。Rust 側 `DecideOptions`
#: の既定と一致するので、通常はここを触らない）。
SERVE_SIMS = 160
GEN_PRUNE_FUTILE = False   # 生成は枝刈りを外す（v6 柱⑤: 刈った枝は学習データに現れない）

_STATE: Dict[str, Any] = {}


def engine():
    """`opcg_engine` を import し、カード定義表をプロセスで 1 度だけ読む（冪等）。"""
    import opcg_engine

    if not _STATE.get("masters"):
        if not os.path.exists(EFFECTS_PATH):
            # 生成物（git 管理外・約 8MB）。無ければその場でエクスポータを回す。
            subprocess.run([sys.executable, "-m", "opcg_sim.tools.export_effects_json",
                            "--out", EFFECTS_PATH],
                           cwd=_REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
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


class SeatSpec:
    """1 席の思考（ネット＋探索のつまみ）。`Game.decide` の `opts` を組む。

    つまみは Rust の `DecideOptions`／`SearchOptions` の欄と同名（省略した欄は serve 既定）。
    Python の `LearnedEngine(**kw)` に渡していた席別 seam（`macro_moves`／`defense_box`／
    `box_dialog`／`box_commit`／`residual_dig`／`residual_activate`／`don_margin`）はそのまま
    通る＝アリーナの `--cand-*` フラグは規約を変えずに動く。
    """

    def __init__(self, net: Optional[str] = None, sims: int = SERVE_SIMS,
                 dirichlet_eps: float = 0.0, temp_turns: int = 0, **kw):
        self.net = load_net(net)
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
