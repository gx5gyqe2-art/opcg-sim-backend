"""マクロ手化 P0: 判断点の型別分布・分岐数・順序重複率の実測（2026-08-24）。

ユーザ設計「慎重型は1枚のドン付与・カウンター1枚・効果対象選択ですら1手＝訪問先が
多すぎて非効率。細かい手を意味のまとまり（登場/ドン付与/バトル/効果起動）に近似すべき」
の定量的裏付けを取る計器。P1（木の候補をマクロ化）の効果測定の基準線。

測るもの（gen15 既定・plan OFF の自己対戦から全判断点を観測）:
  1. 判断窓の型別分布 — 全 decide のうち何割が「原始手の細部」（ドン1枚・カウンター・
     効果選択・ブロッカー）で、何割が「ターンの設計」（メイン窓）か
  2. 窓ごとの分岐数（serve が実際に見る候補数＝枝刈り・ドン箱込みの adapter 列挙）
  3. メイン窓の順序重複率 — 浮ドン k・付与先 m のとき、原始手の並び m^k 通りに対し
     到達しうる配分は C(m+k-1, k) 通りしかない＝木が別枝として読む「同じ未来」の倍率

出力: MACRO_P0_RESULT json ＋ 集計表。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/macro_p0_probe.py \
    --games 8 --seed-base 750000 --workers 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json
import math
import multiprocessing as mp

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai


def window_kind(pending, k_action):
    """判断窓の型（pure）。MAIN_ACTION 以外は pending のアクション名そのまま。"""
    return (pending or {}).get(k_action) or "UNKNOWN"


def order_redundancy(spare, targets):
    """浮ドン spare を targets 体へ1枚ずつ振る原始経路数と配分数の比（pure）。

    経路 = targets^spare（各1枚をどこに置くかの順列）／配分 = C(targets+spare-1, spare)。
    比が大きいほど「同じ未来を別枝として読む」無駄が大きい。"""
    if spare <= 0 or targets <= 0:
        return 1.0
    paths = targets ** spare
    outcomes = math.comb(targets + spare - 1, spare)
    return paths / outcomes


class _Cap:
    def __init__(self, gs, limit):
        self.gs = gs
        self.limit = limit
        self.n = 0
        self.rows = []          # (窓型, 候補数, 選んだ手の型, spare, n_targets)
        self._keys = cpu_ai._pending_keys()

    def on_decision_point(self, ctx):
        pass

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n > self.limit:
            raise _Done()
        m = ctx.manager
        _kp, k_action = self._keys
        kind = window_kind(ctx.pending, k_action)
        try:
            legal = self.gs.legal_actions(m)
        except Exception:
            return
        chosen = (move or {}).get("action_type") or "NONE"
        spare = n_tgt = 0
        if kind == "MAIN_ACTION":
            name = getattr(ctx.actor, "name", None)
            p = m.p1 if getattr(m.p1, "name", None) == name else m.p2
            spare = len(getattr(p, "don_active", []) or [])
            tset = {(x.get("payload") or {}).get("uuid") for x in legal
                    if x.get("action_type") == "ATTACH_DON"}
            n_tgt = len(tset - {None})
        self.rows.append((kind, len(legal), chosen, spare, n_tgt))


class _Done(BaseException):
    pass


_G = {}


def _init(sims):
    from cpu_arena import _load_db
    from opcg_sim.src.core import cpu_learned as CL
    _G["db"] = _load_db()
    _G["eng"] = CL.LearnedEngine(sims=sims)


def _run_game(job):
    seed, sims = job
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    db, eng = _G["db"], _G["eng"]
    la, lb = _leader_pair(db, seed, "random")
    cap = _Cap(eng.game, limit=400)
    seat = make_seat(kind="learned", want_trace=False, sims=sims, engine=eng)
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=seed),
                 observers=(cap,), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=400)
    except _Done:
        pass
    except BaseException as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}", "rows": []}
    return {"seed": seed, "error": None, "rows": cap.rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=750000)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(args.seed_base + i, args.sims) for i in range(args.games)]
    by_kind = collections.defaultdict(lambda: {"n": 0, "branch_sum": 0, "branch_max": 0})
    chosen_ct = collections.Counter()
    redund = []
    n_all = errors = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.sims,)) as pool:
        for res in pool.imap_unordered(_run_game, jobs):
            if res["error"]:
                errors += 1
                print(f"  seed {res['seed']}: {res['error']}", flush=True)
                continue
            for kind, nb, chosen, spare, n_tgt in res["rows"]:
                n_all += 1
                d = by_kind[kind]
                d["n"] += 1
                d["branch_sum"] += nb
                d["branch_max"] = max(d["branch_max"], nb)
                chosen_ct[chosen] += 1
                if kind == "MAIN_ACTION" and spare > 0 and n_tgt > 0:
                    redund.append(order_redundancy(spare, n_tgt))
            print(f"  seed {res['seed']}: 判断点{len(res['rows'])}", flush=True)

    print("\n== 窓型ごとの分布（全 decide に占める割合・平均/最大分岐数）==")
    stats = {}
    for kind, d in sorted(by_kind.items(), key=lambda kv: -kv[1]["n"]):
        share = d["n"] / max(1, n_all)
        avg_b = d["branch_sum"] / max(1, d["n"])
        stats[kind] = {"share": round(share, 3), "avg_branch": round(avg_b, 1),
                       "max_branch": d["branch_max"], "n": d["n"]}
        print(f"  {kind:28s} {share:6.1%}  平均分岐 {avg_b:5.1f}  最大 {d['branch_max']}")
    main_share = stats.get("MAIN_ACTION", {}).get("share", 0.0)
    micro_share = round(1.0 - main_share, 3)
    med_red = sorted(redund)[len(redund) // 2] if redund else 1.0
    max_red = max(redund) if redund else 1.0
    print(f"\n細部窓の割合（メイン以外）: {micro_share:.1%}")
    print(f"メイン窓のドン順序重複率（同一配分に至る原始経路の倍率）: 中央値 {med_red:.0f}x・最大 {max_red:.0f}x")
    print("MACRO_P0_RESULT " + json.dumps(
        {"games": args.games, "decisions": n_all, "errors": errors,
         "micro_share": micro_share, "kinds": stats,
         "don_order_redundancy": {"median": round(med_red, 1), "max": round(max_red, 1),
                                  "n_windows": len(redund)},
         "chosen_types": dict(chosen_ct.most_common(12))}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
