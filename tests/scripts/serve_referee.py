"""本番仕様レフェリー（v48・2026-08-10・読み取り専用）。

**問い**: この決定点の各候補手は、**出荷している CPU の現実**でどちらが勝つか。

ユーザ指示（2026-08-10）: 「選択手を測る時は本番仕様で測るべき。本番の価値ネットと
ソースコードをそのまま使って検証して欲しい」。既存の `counterfactual_referee` は教師を
gen5 に**固定**する設計（学習で漂流しない外部の錨＝教師ラベル用としては正しい）だが、
その分「いまの出荷 CPU（gen13）が実際に打つゲーム」とは別の物差しになる。裁定を
**製品の挙動**で確定したいときは本器を使う。

**本番仕様の中身**（すべて serve と同一・つまみ無し）:
  - エンジン: `LearnedEngine()`（同梱既定ネット＝gen13）・`decide(sims=SERVE_SIMS=160)`
  - 両席とも同じ本番 decide で終局まで打つ（def_temp / opp_temp のような測定用温度は無い）
  - 枝刈り・読み出し・sticky 決定化もすべて serve の既定のまま

**世界の作り方**: フレーム復元は**両者の手札・ライフの真値**を持っている（`_frame_side` は
両席の hand を記録する）ので、determinize で相手手札を再サンプルせず**真値のまま**使い、
不確定なのは山札の並びだけ＝**山札シャッフルを世界**とする（CRN: 同じ seed の並びを全候補で
共有）。「実際にあの場面で何が正解だったか」に最も近い神視点の対照実験になる。

**棲み分け**: `counterfactual_referee`＝gen5 錨・情報集合世界（教師ラベル生成の正本）。
本器＝gen13 本番・真値世界（裁定の最終確認）。両方で同方向なら頑健な裁定。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/serve_referee.py \\
    --marks h1:93,h1:56,h1:30 --worlds 6
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401
import coach_gate as CG  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_game import OPCGGame  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine  # noqa: E402

MAX_STEPS = 400


def _label(d):
    s = str(d.get("action_type"))
    if d.get("card"):
        s += f" {d['card']}"
    if d.get("targets"):
        s += " → " + ",".join(str(t) for t in d["targets"])
    return s


def _shuffle_decks(m, w):
    """世界 w: 山札の並びだけを共有 seed で決める（手札・ライフは復元済みの真値のまま）。"""
    for pid_i, pl in enumerate((m.p1, m.p2)):
        r = np.random.default_rng(70000 + w * 101 + pid_i)
        order = r.permutation(len(pl.deck))
        pl.deck[:] = [pl.deck[int(i)] for i in order]


def rollout_serve(eng, gs, m, mover, rng_seed):
    """両席とも本番 decide（gen13・sims160）で終局まで。返り値 (winner, mover視点ライフ差)。"""
    eng._world_seeds = {}
    rng = np.random.default_rng(rng_seed)
    steps = 0
    while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
        name = gs.current_player(m)
        if name is None:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        mv = eng.decide(m, actor, rng=rng)     # sims は本番既定（SERVE_SIMS）を触らない
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            break
        m = m2
        steps += 1
    me = m.p1 if m.p1.name == mover else m.p2
    opp = m.p2 if m.p1.name == mover else m.p1
    return m.winner, len(me.life or []) - len(opp.life or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default="h1:93,h1:56,h1:30", help="tag:index のカンマ区切り")
    ap.add_argument("--worlds", type=int, default=6)
    args = ap.parse_args()

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    db = _load_db()
    eng = LearnedEngine()                      # 同梱既定＝本番ネット
    gr = OPCGGame(prune_futile=False)          # 候補列挙は無枝刈り（人間の手を必ず含める）
    gs = OPCGGame()                            # 進行は serve 同等

    loaded = {}
    for spec in args.marks.split(","):
        tag, i = spec.split(":"); i = int(i)
        if tag not in loaded:
            raw = RE.load_replay_json(table[tag])
            rec = raw.get("replay", raw)
            loaded[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                           rec["actions"])
        rec, fbi, acts = loaded[tag]
        human = acts[i]
        hkey = (("ATTACK" if human.get("action_type") == "ATTACK_CONFIRM"
                 else human.get("action_type")), human.get("card"))
        built = MG._restore(db, rec, fbi, acts, i)
        if isinstance(built, str):
            print(f"{tag}@{i}: 復元不可 ({built})", flush=True); continue
        m0, actor = built
        name = actor.name if hasattr(actor, "name") else actor
        legal = gr.legal_actions(m0)
        descs = [cpu_ai._describe_move(m0, mv) or {} for mv in legal]
        wins = np.zeros(len(legal)); life = np.zeros(len(legal))
        t0 = time.time()
        for w in range(args.worlds):
            for k, mv in enumerate(legal):
                mb = MG._restore(db, rec, fbi, acts, i)
                if isinstance(mb, str):
                    continue
                mw, _ = mb
                _shuffle_decks(mw, w)          # 同じ w は全候補で同じ並び（CRN）
                ch = gs.apply(mw, mv, name)
                if ch is None:
                    continue
                wn, ld = rollout_serve(eng, gs, ch, name, rng_seed=w * 7919)
                wins[k] += 1 if wn == name else 0
                life[k] += ld
        lifem = life / max(args.worlds, 1)
        order = np.argsort(-(wins * 1000 + lifem))
        print(f"\n=== {tag}@{i} turn{human.get('turn')}（本番仕様: gen13/serve decide・"
              f"{len(legal)}手 × {args.worlds}世界・{time.time()-t0:.0f}s）"
              f" 人間= {_label(human)}", flush=True)
        for k in order:
            d = descs[k]
            at = "ATTACK" if d.get("action_type") == "ATTACK_CONFIRM" else d.get("action_type")
            mark = " ◆人間" if (at, d.get("card")) == hkey else ""
            print(f"   {wins[k]:.0f}/{args.worlds} L{lifem[k]:+.2f}  {_label(d)}{mark}", flush=True)
        print("SERVE_REFEREE " + json.dumps(
            {"tag": tag, "i": i, "human": _label(human),
             "rank": [{"m": _label(descs[k]), "w": int(wins[k]), "l": round(float(lifem[k]), 2)}
                      for k in order]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
