"""rs_ismcts_diag: 木 1 本の方式（世界の引き直し・`redeterminize_every`）の診断（WP `rs-ismcts-diag`）。

`docs/search_efficiency_plan.md` §7.5 の 1〜5 を、**同じ局面の列**で腕を並べて測る。

- 局面: `rs_decide_latency.py` と同じ規約（r13d 両席・random × synth_roles・seed `--seed-base` から
  `--games` 局）。打つ手は主の席（`--main-every`・既定 0＝今の出荷と同じ探索）のもの。
- 各判断点で、腕（`--arms`・例 `0,100000,8,64`）ごとに同じ入力（seed・持ち越し）で `Game.decide` を
  追加で呼ぶ（盤面は変えない）。引き直しの腕には `redet_diag` を付けて内訳（`redet`）を集める。
- 1: `--same A,B`（例 `0,100000`）の 2 腕で、手（sig）・根の訪問数・値が全部一致した判断点の数。
- 2〜4: 引き直しの腕の `redet` の合計（読みの終わり方・自分の手番の食い違い・コミットの食い違い）。
- 5: `--ref-sims` を付けると参照（`--ref-worlds` 本 × `--ref-sims` 回）の根の Q で各腕の選んだ手の
  損（参照の最善の Q − 選んだ手の Q）を測る（探索した判断点＝候補が 2 つ以上のときだけ）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/rs_ismcts_diag.py --games 4 --arms 0,100000,8 \\
      --same 0,100000 --out /tmp/diag.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                          # noqa: E402
import json                                                              # noqa: E402
import sys                                                               # noqa: E402
import time                                                              # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap                                                        # noqa: E402,F401

from opcg_sim.loop import decks as D                                     # noqa: E402
from opcg_sim.loop import driver as DR                                   # noqa: E402
from opcg_sim.loop import engine as E                                    # noqa: E402


def sig(mv):
    """`decide.rs::move_sig` と同じ鍵（action_type・uuid・target_ids・selected_uuids・accepted）。"""
    if mv is None:
        return None
    p = mv.get("payload") or {}
    at = mv.get("action_type") or p.get("action_type")
    return json.dumps([at, p.get("uuid"), p.get("target_ids") or [], p.get("selected_uuids") or [],
                       p.get("accepted")], ensure_ascii=False, sort_keys=True)


def root_stats(out):
    """根の候補の (sig → (n, q))。同じカードの別の実体はグループ（`groups`＝束ねた統計）で読む。"""
    st = out.get("stats") or {}
    legal = st.get("legal") or []
    n = st.get("N") or []
    q = st.get("Q") or []
    res = {}
    groups = out.get("groups") or []
    if groups:
        for g in groups:
            for i in g.get("idxs") or []:
                if i < len(legal):
                    res.setdefault(sig(legal[i]), (float(g["n"]), float(g["q"])))
        return res
    for i, mv in enumerate(legal):
        res.setdefault(sig(mv), (float(n[i]) if i < len(n) else 0.0, float(q[i]) if i < len(q) else 0.0))
    return res


def raw_stats(out):
    st = out.get("stats") or {}
    return (json.dumps([sig(m) for m in st.get("legal") or []]), st.get("N"), st.get("Q"))


def add_into(acc, d):
    for k, v in (d or {}).items():
        if isinstance(v, list):
            cur = acc.setdefault(k, [0] * len(v))
            for i, x in enumerate(v):
                cur[i] += int(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0) + v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=None)
    ap.add_argument("--sims", type=int, default=1280)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--main-every", type=int, default=0, help="打つ手を決める席の引き直しの間隔")
    ap.add_argument("--arms", default="0,100000,8,64")
    ap.add_argument("--same", default=None, help="一致を見る 2 腕（例 0,100000）")
    ap.add_argument("--ref-sims", type=int, default=0)
    ap.add_argument("--ref-worlds", type=int, default=4)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=91000)
    ap.add_argument("--leaders", default="random")
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--max-points", type=int, default=0, help="判断点の上限（0＝無制限）")
    ap.add_argument("--diffs", type=int, default=10, help="一致しなかった判断点を何件記録するか")
    ap.add_argument("--dump", default=None, help="判断点ごとの各腕の手を書き出す（jsonl）")
    ap.add_argument("--score-dump", action="append", default=[],
                    help="LABEL=PATH: 別の実行（別の wheel）の --dump の手も参照で採点する（同じ局面列が前提）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    extra = {}
    for spec_s in args.score_dump:
        label, path = spec_s.split("=", 1)
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for m, mv in row["moves"].items():
                    extra.setdefault(f"{label}:{m}", {})[tuple(row["key"])] = mv
    dump_f = open(args.dump, "w", encoding="utf-8") if args.dump else None

    base = {"budget": int(args.budget)}
    arms = [int(x) for x in args.arms.split(",") if x.strip()]
    specs = {}
    for m in arms:
        kw = dict(base)
        if m:
            kw["redeterminize_every"] = m
            kw["redet_diag"] = True
        specs[m] = E.SeatSpec(args.net, sims=args.sims, **kw)
    ref = None
    if args.ref_sims:
        rkw = dict(base)
        if args.ref_worlds > 1:
            rkw["worlds"] = int(args.ref_worlds)
        ref = E.SeatSpec(args.net, sims=args.ref_sims, **rkw)
    same = [int(x) for x in args.same.split(",")] if args.same else None
    last = {}

    mkw = dict(base)
    if args.main_every:
        mkw["redeterminize_every"] = args.main_every

    class _Spec(E.SeatSpec):
        def decide_opts(self, game_seed, turn, seat, carry=None):
            last["args"] = (game_seed, turn, seat, carry)
            return super().decide_opts(game_seed, turn, seat, carry)

    spec = _Spec(args.net, sims=args.sims, **mkw)
    scored = {k: {"loss_sum": 0.0, "loss_n": 0, "loss_pos": 0, "missing": 0} for k in extra}
    res = {m: {"points": 0, "searched": 0, "redet": {}, "loss_sum": 0.0, "loss_n": 0,
               "loss_pos": 0, "match_ref_best": 0, "loss_missing": 0, "ms": 0.0} for m in arms}
    agree = {"points": 0, "searched": 0, "same_move": 0, "same_all": 0, "diffs": []}
    pts = {"n": 0, "searched": 0}

    class _Stop(Exception):
        pass

    def observer(game, name, turn, step, _out, _move):
        if args.max_points and pts["n"] >= args.max_points:
            raise _Stop()
        pts["n"] += 1
        outs = {}
        for m, sp in specs.items():
            t1 = time.perf_counter()
            outs[m] = json.loads(game.decide(name, sp.decide_opts(*last["args"])))
            res[m]["ms"] += (time.perf_counter() - t1) * 1000.0
            res[m]["points"] += 1
            add_into(res[m]["redet"], outs[m].get("redet"))
        searched = len((outs[arms[0]].get("stats") or {}).get("legal") or []) > 1
        key = (last["args"][0], step, name)
        if dump_f is not None:
            dump_f.write(json.dumps({"key": list(key), "searched": searched,
                                     "moves": {str(m): sig(outs[m].get("move")) for m in arms}},
                                    ensure_ascii=False) + "\n")
        if searched:
            pts["searched"] += 1
        if same:
            a, b = outs[same[0]], outs[same[1]]
            agree["points"] += 1
            agree["searched"] += int(searched)
            sm = sig(a.get("move")) == sig(b.get("move"))
            sa = sm and raw_stats(a) == raw_stats(b) and a.get("commit") == b.get("commit") and a.get("pv") == b.get("pv")
            agree["same_move"] += int(sm)
            agree["same_all"] += int(sa)
            if not sa and len(agree["diffs"]) < args.diffs:
                agree["diffs"].append({"seed": last["args"][0], "turn": turn, "step": step,
                                       "seat": name, "a": sig(a.get("move")), "b": sig(b.get("move"))})
        if ref is not None and searched:
            r = json.loads(game.decide(name, ref.decide_opts(*last["args"])))
            rs = root_stats(r)
            if rs:
                # 参照の最善＝参照の訪問数最多の手（その Q）。損は「参照の最善の Q − 選んだ手の Q」。
                best_k = max(rs, key=lambda k: rs[k][0])
                best = rs[best_k][1]
                for lab, mv_by in extra.items():
                    k = mv_by.get(key)
                    if k in rs and rs[k][0] > 0:
                        loss = best - rs[k][1]
                        scored[lab]["loss_sum"] += loss
                        scored[lab]["loss_n"] += 1
                        scored[lab]["loss_pos"] += int(loss > 1e-9)
                        scored[lab]["match_ref_best"] = scored[lab].get("match_ref_best", 0) + int(k == best_k)
                    else:
                        scored[lab]["missing"] += 1
                for m in arms:
                    k = sig(outs[m].get("move"))
                    if k in rs and rs[k][0] > 0:
                        loss = best - rs[k][1]
                        res[m]["loss_sum"] += loss
                        res[m]["loss_n"] += 1
                        res[m]["loss_pos"] += int(loss > 1e-9)
                        res[m]["match_ref_best"] += int(k == best_k)
                    else:
                        res[m]["loss_missing"] += 1
        for m in arms:
            res[m]["searched"] += int(searched)

    db = D.load_db()
    t0 = time.time()
    games = []
    for g in range(args.games):
        seed = args.seed_base + g
        la, lb = D.leader_pair(db, seed, args.leaders)
        p1, p2 = D.build_pair(db, la, lb, seed, args.decks)
        n0 = pts["n"]
        try:
            r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=observer)
            games.append({"seed": seed, "winner": r["winner"], "turns": r["turns"], "points": pts["n"] - n0})
        except _Stop:
            games.append({"seed": seed, "stopped": True, "points": pts["n"] - n0})
            break
        except Exception as e:                       # noqa: BLE001
            games.append({"seed": seed, "error": f"{type(e).__name__}: {str(e)[:160]}",
                          "points": pts["n"] - n0})
        print(f"[diag] game {g + 1}/{args.games} seed={seed} points={pts['n']} "
              f"searched={pts['searched']} {time.time() - t0:.0f}s", flush=True)
    for m in arms:
        r = res[m]
        r["ms_mean"] = round(r.pop("ms") / max(1, r["points"]), 1)
        if r["loss_n"]:
            r["loss_mean"] = round(r["loss_sum"] / r["loss_n"], 5)
        rd = r["redet"]
        if rd.get("sims"):
            ends = ("expand", "terminal", "dead", "no_legal", "mismatch", "empty")
            rd["end_frac"] = {e: round(rd.get(f"end_{e}", 0) / rd["sims"], 4) for e in ends}
        if rd.get("own_seen"):
            rd["own_diff_frac"] = round(rd["own_diff"] / rd["own_seen"], 4)
            rd["own_diff_frac_ctx"] = {c: (round(rd["own_diff_ctx"][i] / rd["own_seen_ctx"][i], 4)
                                           if rd["own_seen_ctx"][i] else None)
                                       for i, c in enumerate(("after_opp", "after_draw", "after_look", "none"))}
        if rd.get("commit_n"):
            rd["commit_diff_frac"] = round(rd["commit_diff"] / rd["commit_n"], 4)
    data = {"opts": {"sims": args.sims, "budget": args.budget, "main_every": args.main_every,
                     "ref_sims": args.ref_sims, "ref_worlds": args.ref_worlds},
            "games": games, "seed_base": args.seed_base, "points": pts,
            "wall_s": round(time.time() - t0, 1), "arms": {str(m): res[m] for m in arms}}
    if same:
        data["agree"] = agree
    for v in scored.values():
        if v["loss_n"]:
            v["loss_mean"] = round(v["loss_sum"] / v["loss_n"], 5)
    if scored:
        data["scored_dumps"] = scored
    if dump_f is not None:
        dump_f.close()
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
