"""人間リプレイとの選択乖離スキャン（v48・2026-08-10・読み取り専用）。

**問い**: 人間が実際に打った対局を復元し、同じ決定点で CPU は**別の手を選ぶか**。

裁定を1点ずつ言葉で確認するのは手間がかかる（v18 は34マークの裁定に人手を要した）。人間が
1局まるごと打ったリプレイがあれば、**その手がそのまま正解ラベル**になり、食い違った点だけを
機械的に抜き出せる。抜き出した点は裁定候補としてレフェリー／コーチゲートに載せられる。

**既存器との棲み分け**（1トピック=1ファイル）:
  - `divergence_probe.py`（v12）は**候補ネット vs 既定ネット**の生成対戦から乖離を採掘する。
  - `mark_referee_verify.py` は既知のマーク点をレフェリーで裁く。
  - 本器は**人間の実対局 vs CPU**。相手は誰でもよく、比較対象は「記録に残った人間の手」。

**判定**: 各人間手番で `mark_gate._restore` により盤面を復元し、`LearnedEngine.decide` を
seeds 回まわして `cpu_ai._describe_move` の (action_type, card, targets) を人間の記録と比べる。
一致率が 0 の点＝CPU が**一度も**人間の手を選ばない点＝最も濃い裁定候補。

**注意（実測で踏んだ落とし穴）**: 攻撃の action_type は経路で違う——CPU は `ATTACK`、
アプリの人間操作は `ATTACK_CONFIRM`（宣言→確定の2段UI）。正規化しないと全攻撃が
「不一致」に化ける。本器は両者を `ATTACK` に寄せて比較する。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/human_replay_divergence.py \\
    --replay h1 --seeds 4 --sims 160
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
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine  # noqa: E402


def _norm(d):
    """比較キー。ATTACK_CONFIRM（人間UI）と ATTACK（CPU）を同一視する。"""
    at = d.get("action_type")
    if at == "ATTACK_CONFIRM":
        at = "ATTACK"
    return (at, d.get("card"), tuple(d.get("targets") or ()))


def _label(d):
    s = str(d.get("action_type"))
    if d.get("card"):
        s += f" {d['card']}"
    if d.get("targets"):
        s += " → " + ",".join(str(t) for t in d["targets"])
    if d.get("selected"):
        s += " [" + ",".join(str(t) for t in d["selected"]) + "]"
    if d.get("accepted") is False:
        s += "（見送り）"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="h1", help="リプレイタグ または JSON ファイルパス")
    ap.add_argument("--net", default="", help="value.npz[,policy.npz]（空＝同梱の既定ネット）")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--skip", default="MULLIGAN,KEEP_HAND",
                    help="比較しない action_type（選択肢が無い/自明な手）")
    args = ap.parse_args()

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    raw = RE.load_replay_json(table.get(args.replay, args.replay))
    rec = raw.get("replay", raw)
    acts = rec["actions"]
    fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    human_pid = next((a.get("player") for a in acts if a.get("src") == "human"), None)
    if human_pid is None:
        print("human の手が無い（自己対戦リプレイ？）"); return 1
    print(f"=== {args.replay}  人間={human_pid}"
          f"（{(rec.get('leaders') or {}).get(human_pid)}）"
          f" / CPU={(rec.get('leaders') or {}).get('p2' if human_pid == 'p1' else 'p1')}"
          f"  seeds={args.seeds} sims={args.sims}")

    db = _load_db()
    if args.net.startswith("neff:"):
        # N系（効果構造符号化）ネット（2026-09-03）: promotion_gate の eng() と同じ注入経路。
        import n_eff_gate
        eng = n_eff_gate.neff_engine(args.net[5:])
    elif args.net.startswith("n1:"):
        import n1_gate
        eng = n1_gate.n1_engine(args.net[3:])
    elif args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0],
                            policy_path=parts[1] if len(parts) > 1 else None)
    else:
        eng = LearnedEngine()

    rows, n_cmp, n_div = [], 0, 0
    for i, a in enumerate(acts):
        if a.get("src") != "human" or a.get("action_type") in skip:
            continue
        built = MG._restore(db, rec, fbi, acts, i)
        if isinstance(built, str) or built is None:
            rows.append({"i": i, "turn": a.get("turn"), "skip": f"復元不可({built})"})
            continue
        m0, actor = built
        name = actor.name if hasattr(actor, "name") else actor
        if name != human_pid:
            continue                                  # 復元手番が人間側でない点は比較しない
        want = _norm(a)
        got = collections.Counter()
        hit = 0
        for s in range(args.seeds):
            eng._world_seeds = {}
            try:
                mv = eng.decide(m0, actor, sims=args.sims,
                                rng=np.random.default_rng(9000 + 97 * i + s))
                d = cpu_ai._describe_move(m0, mv) or {}
            except Exception as e:
                d = {"action_type": f"例外:{type(e).__name__}"}
            got[_label(d)] += 1
            hit += 1 if _norm(d) == want else 0
        n_cmp += 1
        rate = hit / max(args.seeds, 1)
        if rate < 1.0:
            n_div += 1
        rows.append({"i": i, "turn": a.get("turn"), "human": _label(a),
                     "cpu": got.most_common(1)[0][0], "rate": rate,
                     "cpu_all": dict(got)})

    print(f"\n  {'idx':>4} {'turn':>4} {'一致':>5}  人間の手 / CPU の手")
    for r in rows:
        if "skip" in r:
            print(f"  {r['i']:>4} {r['turn']:>4}   ---  {r['skip']}")
            continue
        mark = "" if r["rate"] == 1.0 else ("  ← 乖離" if r["rate"] == 0 else "  ← 一部")
        cpu = "" if r["rate"] == 1.0 else f"  /  CPU: {r['cpu']}"
        print(f"  {r['i']:>4} {r['turn']:>4} {r['rate']:>5.2f}  {r['human']}{cpu}{mark}")
    print(f"\n=== 比較 {n_cmp} 点中 **{n_div} 点で乖離**（CPU が人間と違う手を選ぶ）")
    print("HUMAN_DIVERGENCE " + json.dumps(
        {"replay": args.replay, "compared": n_cmp, "diverged": n_div,
         "points": [{"i": r["i"], "turn": r["turn"], "rate": r["rate"]}
                    for r in rows if "skip" not in r and r["rate"] < 1.0]},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
