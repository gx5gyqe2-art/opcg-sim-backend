"""ターン消化の打ち切り診断（v45・2026-08-08・読み取り専用）。

**問い**: `turn_all` 形式の点（m2@66＝ドン0で攻撃3本＋効果起動が残っているのに初手で
ターンを終えたのがマークの原点）で、CPU は**ターンのどこで打ち切り、なぜ TURN_END を
選ぶのか**。

コーチゲートの `turn_all_rate` は消化できた割合（実測 gen13 で 0.62〜0.69）しか返さないので、
「打ち切り位置」も「prior と value のどちらが原因か」も分からない。本プローブは自ターンの
各決定で `decide(trace=…)` を取り、**TURN_END を選んだ瞬間**を捕まえて次を並べる:

  - **打ち切り位置**: TURN_END 時点で必須アクションを何個消化していたか／何が残っていたか。
  - **visit% と Q**（探索の結果）: TURN_END と、残っている必須アクションの行動価値。
  - **policy prior**（探索前の素の確率）: 同じ2者について。

**読み方（この分解が答えを分ける）**:
  - TURN_END の **Q が最高** なら→ **value の問題**。ネットは「攻撃した後の盤面」を本当に
    悪いと見ている。教師（ターン出口の較正）か符号化の話になる。
  - TURN_END の Q は低いのに **visit% が最高**なら→ **prior の問題**。policy が TURN_END に
    厚い確率を置き、160sims では覆せていない。policy 側の矯正か sims の話になる。
  - どちらでもなく **Q が拮抗**なら→ ネットには区別が付いていない（＝裁定を注入する対象）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/turn_exhaust_probe.py \\
    --point m2@66 --seeds 16 --sims 160
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401
import coach_gate as CG  # noqa: E402
import counterfactual_referee as CR  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402


def _restore(db, tag, idx):
    """コーチゲートと同じ手順で決定点の真盤面を復元する。"""
    if tag not in CR.GAMES:
        replays = {**MG.REPLAYS, **CG.REPLAYS_V2}
        raw = RE.load_replay_json(replays[tag])
        rec = raw.get("replay", raw)
        CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                         rec["actions"])
    built = CR._restore_board(db, tag, idx)
    if isinstance(built, str):
        return None, None, built
    m0, who = built
    return m0, (who if isinstance(who, str) else who.name), None


def _key(d):
    return (d.get("action_type"), d.get("card"))


def _prior_map(pf, mgr, legal):
    """合法手ごとの policy prior（探索前）。等価手が複数あるときは合算する。"""
    if pf is None:
        return {}
    try:
        p = pf(mgr, legal)
    except Exception:
        return {}
    if p is None:
        return {}
    out = collections.defaultdict(float)
    for mv, pv in zip(legal, np.asarray(p).ravel()):
        try:
            out[_key(cpu_ai._describe_move(mgr, mv) or {})] += float(pv)
        except Exception:
            pass
    return dict(out)


def run_point(eng, game, m0, name, required, seeds, sims, max_plies=24):
    """自ターンを指させ、TURN_END を選んだ瞬間の内訳を集める（`turn_all_rate` と同一規約）。"""
    stops, rows, done_hist = [], [], collections.Counter()
    for s in range(seeds):
        eng._world_seeds = {}
        rng = np.random.default_rng(9100 + 97 * s)
        mgr, done = m0, set()
        for _ply in range(max_plies):
            if game.is_terminal(mgr):
                break
            actor_name = game.current_player(mgr)
            if actor_name is None:
                break
            actor = mgr.p1 if mgr.p1.name == actor_name else mgr.p2
            mine = (actor_name == name)
            tr = {} if mine else None
            legal = game.legal_actions(mgr) if mine else None
            pri = _prior_map(eng._priors_fn_cached, mgr, legal) if mine else {}
            mv = eng.decide(mgr, actor, sims=sims, rng=rng, trace=tr)
            if mv is None:
                break
            try:
                d = cpu_ai._describe_move(mgr, mv) or {}
            except Exception:
                d = {"action_type": (mv or {}).get("action_type")}
            if mine and d.get("action_type") == "TURN_END":
                missing = sorted(required - done)
                done_hist[len(done & required)] += 1
                stops.append((len(done & required), tuple(missing)))
                cands = {(_key(c["move"]) if c.get("move") else None): c
                         for c in (tr or {}).get("candidates", [])}
                # **合法手そのもの**を残す。「打ち切った」のか「もう選べなかった」のかは
                # prior/Q を見る前にここで決まる（v45 実測: 多くの seed で TURN_END が
                # 唯一の合法手＝判断以前に必須アクションが合法手から消えている）。
                legal_keys = set()
                for mv2 in (legal or []):
                    try:
                        legal_keys.add(_key(cpu_ai._describe_move(mgr, mv2) or {}))
                    except Exception:
                        pass
                rows.append({"seed": s, "done": len(done & required),
                             "missing": [list(k) for k in missing],
                             "missing_legal": [list(k) for k in missing if k in legal_keys],
                             "n_legal": len(legal or []),
                             "legal_keys": sorted(str(k) for k in legal_keys),
                             "turn_end": cands.get(("TURN_END", None)),
                             "missing_stats": [{"move": list(k), "stat": cands.get(k),
                                                "legal": k in legal_keys,
                                                "prior": round(pri.get(k, float("nan")), 4)}
                                               for k in missing],
                             "turn_end_prior": round(pri.get(("TURN_END", None),
                                                             float("nan")), 4)})
                break
            if mine:
                done.add(_key(d))
            nxt = game.apply(mgr, mv, actor_name)
            if nxt is None:
                break
            mgr = nxt
        else:
            done_hist[len(done & required)] += 1
            stops.append((len(done & required), tuple(sorted(required - done))))
    return stops, rows, done_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="m2@66", help="tag@index（turn_all 形式の点）")
    ap.add_argument("--nets", default="base", help="'value.npz,policy.npz'（'base'=出荷既定）")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--sims", type=int, default=160)
    args = ap.parse_args()

    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine, _priors_fn
    db = _load_db()

    eng = (LearnedEngine() if args.nets == "base"
           else LearnedEngine(value_path=args.nets.split(",")[0],
                              policy_path=(args.nets.split(",") + [None])[1]))
    eng._priors_fn_cached = _priors_fn(eng.pnet, eng.vocab, eng.enc_version)

    tag, idx = args.point.split("@")[0], int(args.point.split("@")[1])
    required = None
    for t, i, accept in CG.VERIFIED_V2 + CG.VERIFIED:
        if t == tag and i == idx:
            required = CG.turn_all_required(accept)
    if required is None:
        print(f"{args.point} は turn_all 形式ではない（本プローブの対象外）")
        return 1

    m0, name, err = _restore(db, tag, idx)
    if m0 is None:
        print(f"{args.point}: 復元不可: {err}")
        return 1
    print(f"=== {args.point}（{name}・必須{len(required)}件・{args.seeds}seed・"
          f"{args.sims}sims）===", flush=True)
    print(f"必須: {sorted(required)}", flush=True)

    stops, rows, hist = run_point(eng, eng.game, m0, name, required, args.seeds, args.sims)
    n = len(stops) or 1
    full = sum(1 for c, _ in stops if c == len(required))
    print(f"\n消化率 {full}/{len(stops)} = {full / n:.2f}（コーチゲートの turn_all_rate 相当）",
          flush=True)
    print("打ち切り位置の分布（必須の消化数 → 回数）:", flush=True)
    for k in sorted(hist):
        print(f"  {k}/{len(required)} 消化で終了: {hist[k]}回", flush=True)

    print("\nTURN_END を選んだ瞬間の内訳（先頭5件）:", flush=True)
    for r in rows[:5]:
        te, tep = r["turn_end"], r["turn_end_prior"]
        te_s = (f"visit%={te['visit_pct']} Q={te['q']}" if te else "探索上位5に不在")
        print(f"  seed{r['seed']}: {r['done']}/{len(required)} 消化 | "
              f"TURN_END {te_s} prior={tep}", flush=True)
        print(f"      合法手{r['n_legal']}件: {r['legal_keys']}", flush=True)
        for m in r["missing_stats"]:
            st = m["stat"]
            st_s = (f"visit%={st['visit_pct']} Q={st['q']}" if st else "探索上位5に不在")
            tag_s = "選べた" if m["legal"] else "**合法手に無い**"
            print(f"      残 {m['move']}: {tag_s} {st_s} prior={m['prior']}", flush=True)

    # 判定の材料: TURN_END と残存必須の Q / prior を平均で並べる
    def _avg(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return round(float(np.mean(v)), 4) if v else None
    te_q = _avg([r["turn_end"]["q"] if r["turn_end"] else None for r in rows])
    te_p = _avg([r["turn_end_prior"] for r in rows])
    ms_q = _avg([m["stat"]["q"] for r in rows for m in r["missing_stats"] if m["stat"]])
    ms_p = _avg([m["prior"] for r in rows for m in r["missing_stats"]])
    # **打ち切りの内訳**: 未消化で終わった seed のうち、残りが合法手に有ったのは何回か。
    unfinished = [r for r in rows if r["done"] < len(required)]
    choosable = [r for r in unfinished if r["missing_legal"]]
    print(f"\n平均: TURN_END Q={te_q} prior={te_p} ／ 残存必須 Q={ms_q} prior={ms_p}", flush=True)
    print(f"未消化 {len(unfinished)}件のうち、残りを**選べたのに選ばなかった**のは "
          f"{len(choosable)}件（残りが合法手に無かった＝判断以前は {len(unfinished) - len(choosable)}件）",
          flush=True)
    print("TURN_EXHAUST_RESULT " + json.dumps(
        {"point": args.point, "required": len(required), "rate": round(full / n, 4),
         "hist": {str(k): v for k, v in sorted(hist.items())},
         "turn_end_q": te_q, "turn_end_prior": te_p,
         "missing_q": ms_q, "missing_prior": ms_p,
         "unfinished": len(unfinished), "unfinished_choosable": len(choosable)},
        ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
