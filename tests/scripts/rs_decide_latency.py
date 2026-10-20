"""rs_decide_latency: 実対局での 1 手の decide 時間の分布（`docs/search_efficiency_plan.md` §1 の「時間」）。

両席に同じ設定（r13d・`--sims`／`--budget`／`--worlds`／`--redeterminize-every`）を持たせて
`--games` 局を自己対戦で打ち、**1 手ごとの `Game.decide` の実時間**を集めて平均・中央・90%・99%・
最大を出す。盤面は主条件のアリーナと同じ規約（`decks.leader_pair(mode=--leaders)`＋
`build_pair(decks=--decks)`・既定 random × synth_roles）。

- 時間は「直前の手を適用し終えた時刻 → decide が返った時刻」（間に入るのは `pending_json` だけ）。
- 設定が違えば打つ手も変わる＝局面の集合は腕ごとに違う（20 局の分布として比べる）。同じ 1 局面で
  比べたいときは `rs_search_a_latency.py`。
- `--redeterminize-every m`（§7.1・WP `rs-ismcts`）では decide の戻り値の `redet`
  （引き直し・照合・追加の回数＝世界 0 のぶん）を合算し、「照合を省けない比率」
  （`checked / visits`）と「新しい手が追加された回数」も出す。
- `--compare-every 1,8,64` を足すと、主の設定で打つ対局の**同じ局面**で、引き直しの間隔だけを変えた
  decide も追加で呼んで時間を測る（打つ手は主の設定のもの・追加の decide は盤面を変えない・持ち越し
  〔箱コミット〕は主と同じものを渡す）＝腕ごとに局面の集合が違う問題が無い対の比較。結果は `paired`。
- `OPCG_PROF=1` を付けて回すと最後に `opcg_engine.prof_take()` の内訳（区分ごとの秒）も載せる。

§1 の物差しは 1 コアに固定（`taskset -c 0`）して回す。

実行例:
  OPCG_LOG_SILENT=1 taskset -c 0 python tests/scripts/rs_decide_latency.py \\
      --sims 1280 --budget 0 --games 10 --redeterminize-every 8 --out /tmp/lat_m8.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                          # noqa: E402
import json                                                              # noqa: E402
import statistics                                                        # noqa: E402
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

REDET_KEYS = ("redraws", "visits", "checked", "added", "unplayable", "mismatch", "empty")


def quantile(xs, q):
    """最近傍の分位（`xs` は昇順）。"""
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def summarize(ms):
    xs = sorted(ms)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1),
            "median": round(quantile(xs, 0.5), 1), "p90": round(quantile(xs, 0.9), 1),
            "p99": round(quantile(xs, 0.99), 1), "max": round(xs[-1], 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default=None, help="npz（省略＝出荷既定 `engine.DEFAULT_NET`）")
    ap.add_argument("--sims", type=int, default=E.SERVE_SIMS)
    ap.add_argument("--budget", type=int, default=None, help="戦闘箱の枝予算（0＝無制限）")
    ap.add_argument("--worlds", type=int, default=None)
    ap.add_argument("--redeterminize-every", type=int, default=None,
                    help="世界の引き直しの間隔（§7.1・既定 0＝off）")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=91000)
    ap.add_argument("--leaders", default="random")
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--compare-every", default=None,
                    help="同じ局面で追加に測る引き直しの間隔（カンマ区切り・例 1,8,64・0 は off）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    kw = {}
    if args.budget is not None:
        kw["budget"] = int(args.budget)
    if args.worlds and args.worlds > 1:
        kw["worlds"] = int(args.worlds)
    if args.redeterminize_every:
        kw["redeterminize_every"] = int(args.redeterminize_every)
    last = {}

    class _Spec(E.SeatSpec):
        """主の席。直前の decide に渡した (seed, turn, seat, carry) を覚える（対の比較が同じ入力を使う）。"""

        def decide_opts(self, game_seed, turn, seat, carry=None):
            last["args"] = (game_seed, turn, seat, carry)
            return super().decide_opts(game_seed, turn, seat, carry)

    spec = _Spec(args.net, sims=args.sims, **kw)
    compare = [int(x) for x in str(args.compare_every).split(",")] if args.compare_every else []
    cmp_specs = {}
    for m in compare:
        ckw = {k: v for k, v in kw.items() if k != "redeterminize_every"}
        if m:
            ckw["redeterminize_every"] = m
        cmp_specs[m] = E.SeatSpec(args.net, sims=args.sims, **ckw)
    paired = {m: {"ms": [], "redet": {k: 0 for k in REDET_KEYS}} for m in compare}
    paired_main = []
    prof_on = os.environ.get("OPCG_PROF", "").strip() == "1"
    paired_prof = {m: {} for m in compare}
    main_prof = {}

    def _prof_add(bucket):
        for k, v in json.loads(E.engine().prof_take()).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                bucket[k] = bucket.get(k, 0) + v

    db = D.load_db()

    ms = []
    redet = {k: 0 for k in REDET_KEYS}
    games = []
    clock = {"t": 0.0}

    def post(_game, _name, _turn, _step, _move, _events):
        clock["t"] = time.perf_counter()

    def observer(game, name, _turn, _step, out, _move):
        dt = (time.perf_counter() - clock["t"]) * 1000.0
        ms.append(dt)
        for k in REDET_KEYS:
            redet[k] += int((out.get("redet") or {}).get(k) or 0)
        if compare:
            paired_main.append(dt)
            if prof_on:
                _prof_add(main_prof)
            for m, cs in cmp_specs.items():
                opts = cs.decide_opts(*last["args"])
                t1 = time.perf_counter()
                cout = json.loads(game.decide(name, opts))
                paired[m]["ms"].append((time.perf_counter() - t1) * 1000.0)
                if prof_on:
                    _prof_add(paired_prof[m])
                for k in REDET_KEYS:
                    paired[m]["redet"][k] += int((cout.get("redet") or {}).get(k) or 0)
            clock["t"] = time.perf_counter()     # 追加の decide の時間を次の手に混ぜない

    t0 = time.time()
    for g in range(args.games):
        seed = args.seed_base + g
        la, lb = D.leader_pair(db, seed, args.leaders)
        p1, p2 = D.build_pair(db, la, lb, seed, args.decks)
        n0 = len(ms)
        try:
            res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2,
                              observer=observer, post=post)
            games.append({"seed": seed, "winner": res["winner"], "turns": res["turns"],
                          "decides": len(ms) - n0})
        except Exception as e:                       # noqa: BLE001  1 局で計測を止めない
            games.append({"seed": seed, "error": f"{type(e).__name__}: {str(e)[:120]}",
                          "decides": len(ms) - n0})
        print(f"[latency] game {g + 1}/{args.games} seed={seed} decides={len(ms) - n0} "
              f"累計 {summarize(ms)}", flush=True)

    data = {"opts": {"sims": args.sims, **kw}, "games": args.games, "leaders": args.leaders,
            "decks": args.decks, "seed_base": args.seed_base, "net": spec.net,
            "wall_s": round(time.time() - t0, 1), "ms": summarize(ms), "per_game": games}
    if args.redeterminize_every:
        data["redet"] = dict(redet)
        data["redet"]["checked_frac"] = (round(redet["checked"] / redet["visits"], 4)
                                         if redet["visits"] else None)
    if compare:
        # 対の比較: 同じ局面の列。`raw_ms` は局面ごとの実測（シャードを足し合わせて集計し直すため）。
        data["paired"] = {"main": {"every": int(args.redeterminize_every or 0),
                                   "ms": summarize(paired_main),
                                   "raw_ms": [round(x, 2) for x in paired_main]}}
        for m, v in paired.items():
            row = {"ms": summarize(v["ms"]), "raw_ms": [round(x, 2) for x in v["ms"]]}
            if m:
                row["redet"] = dict(v["redet"])
                row["redet"]["checked_frac"] = (round(v["redet"]["checked"] / v["redet"]["visits"], 4)
                                                if v["redet"]["visits"] else None)
            if prof_on:
                row["prof"] = paired_prof[m]
            data["paired"][str(m)] = row
    if prof_on and compare:
        data["paired"]["main"]["prof"] = main_prof
    elif prof_on:
        prof = json.loads(E.engine().prof_take())
        data["prof"] = prof
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[latency] wrote {args.out}")
    print(json.dumps({"opts": data["opts"], "ms": data["ms"], "redet": data.get("redet")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
