"""行動空間の被覆——**枝予算で候補から落ちた手**を数える
（ユーザ指摘 2026-08-12「人間の付与が候補にすら無い」・`docs/measurement.md` §6・読み取り専用）。

**なぜ記録では測れないか**: 棋譜（`n_records`）に残るのは「候補に出た手」だけで、
**枝刈りで出なかった手は記録に現れない**。被覆は「出なかったもの」の量なので、
**エンジンに 2 通り列挙させて差を取る**しかない。

やること: 実自己対戦の各判断点で、同じ盤面・同じ `search_seed`・同じ sims のまま

```
腕 A（既定）   … 箱の枝予算あり（`DecideOptions::budget = BOX_BRANCH_BUDGET`）＝**出荷の挙動**
腕 B（無制限） … opts に `"budget": 0` を渡す（`mod.rs` の `filter(|b| *b > 0)` で None＝無制限）
```

で `decide` を回し、根の候補集合（`stats.legal` の move_sig）を突き合わせる:

```
coverage   = |A ∩ B| / |B|                 A に出た割合（1.0＝落ちていない）
missing    = |B \\ A|                       **予算で消えた手の数**
best_in_A  = B の最善手（訪問最大）が A の候補に在るか   ← 決定に効く被覆
don_cov    = DON_BOX 候補の被覆（付与の列挙が落ちていないか＝2026-08-12 の指摘そのもの）
```

出すもの: 判断点の数・`coverage` の平均と分布・`missing>0` の割合・`best_in_A` が偽の割合・
`budget.exhausted` が立った割合・ターン帯別／`DON_BOX` 別の内訳・腕 B の追加コスト（秒）。

**読み方（事前）**: `missing>0` の割合が 0 なら**枝予算は上限を作っていない**＝被覆は問題でなく、
手当ては評価・方策の側だけになる。`best_in_A` が偽の判断点が 1% でもあれば
**「強い手が候補にすら無い」が現行エンジンで実在する**＝予算の上げ幅を測る価値がある。

**限界・注意**:
- 腕 B を同じ盤面で**追加で 1 回**回すので、この計測の対局は生成の棋譜と**同一の進行にはならない**
  （`decide` は盤面を変えないが、追加の呼び出しで実行順が変わりうる）。測っているのは
  「実自己対戦に現れる判断点での被覆」なので、進行の一致は要らない。
- 腕 B は列挙が無制限なので**遅い**（予算は元々これを止めるためにある）。`--games` は小さく取る。
- `stats.legal` は**箱化後の根の候補**（世界 0 の並び）。同じ手の別の DON 配分は別の候補になる
  （`move_sig` は don_k を含まないので、**`(sig, don_k)` で突き合わせる**）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/action_coverage.py --games 6 --seed-base 9100 \\
    --out ~/coverage.json
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

TURN_BANDS = ("T<=4", "T5-8", "T9+")


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def sig_key(mv):
    """候補の鍵。`move_sig` は DON 配分（don_k）を含まないので**配分も鍵に入れる**。"""
    if not isinstance(mv, dict):
        return json.dumps(mv, ensure_ascii=False, sort_keys=True)
    payload = mv.get("payload") or {}
    k = payload.get("don_k", payload.get("k"))
    return json.dumps([mv.get("action_type"), mv.get("card_uuid"),
                       payload, k], ensure_ascii=False, sort_keys=True)


def is_don(mv):
    return isinstance(mv, dict) and str(mv.get("action_type") or "").startswith("DON")


def _legal(out):
    return ((out.get("stats") or {}).get("legal") or [])


def _visits(out):
    return ((out.get("stats") or {}).get("N") or [])


def best_index(out):
    """腕 B の最善手＝訪問最大の候補の添字（訪問が無ければ None）。"""
    ns = _visits(out)
    if not ns:
        return None
    best = max(range(len(ns)), key=lambda i: float(ns[i]))
    return best if float(ns[best]) > 0 else None


def compare(out_a, out_b):
    """1 判断点の被覆（A＝既定・B＝無制限）。"""
    la, lb = _legal(out_a), _legal(out_b)
    ka = {sig_key(m) for m in la}
    kb = {sig_key(m) for m in lb}
    inter = ka & kb
    bi = best_index(out_b)
    row = {
        "k_a": len(la), "k_b": len(lb),
        "coverage": (len(inter) / len(kb)) if kb else None,
        "missing": len(kb - ka),
        "extra": len(ka - kb),                     # 予算ありの方に在る手＝原則 0（出たら要調査）
        "best_in_a": (sig_key(lb[bi]) in ka) if bi is not None else None,
        "exhausted_a": int(((out_a.get("budget") or {}).get("exhausted") or 0)),
        "exhausted_b": int(((out_b.get("budget") or {}).get("exhausted") or 0)),
    }
    da = {sig_key(m) for m in la if is_don(m)}
    db = {sig_key(m) for m in lb if is_don(m)}
    row["don_k_a"], row["don_k_b"] = len(da), len(db)
    row["don_coverage"] = (len(da & db) / len(db)) if db else None
    row["don_missing"] = len(db - da)
    return row


def run_games(games, seed_base, sims, net, leaders="random", decks="synth_roles",
              max_steps=DR.DEFAULT_MAX_STEPS, own_main_only=True):
    """自己対戦を回し、各判断点で腕 B を追加で 1 回引いて被覆を記録する。"""
    db = D.load_db()
    spec = E.SeatSpec(net=net, sims=sims)
    # 腕 B＝同じネット・同じ sims で**枝予算だけ無制限**（`budget=0` → Rust 側で None）
    spec_b = E.SeatSpec(net=net, sims=sims, budget=0)
    rows = []
    stats = {"games": 0, "aborted": 0, "decides": 0, "skipped": 0, "sec_b": 0.0}

    for g in range(games):
        seed = seed_base + g
        try:
            la, lb = D.leader_pair(db, seed, leaders)
            p1, p2 = D.build_pair(db, la, lb, seed, decks)
        except Exception:                                  # noqa: BLE001
            stats["aborted"] += 1
            continue

        def observe(game, name, turn, step, out, _move, _seed=seed):
            stats["decides"] += 1
            if own_main_only and out.get("kind") not in (None, "main"):
                stats["skipped"] += 1
                return
            if len(_legal(out)) == 0:
                stats["skipped"] += 1
                return
            t0 = time.time()
            try:
                out_b = json.loads(game.decide(
                    name, spec_b.decide_opts(_seed, turn, name, None)))
            except Exception as exc:                       # noqa: BLE001
                rows.append({"error": f"{type(exc).__name__}: {exc}", "turn": turn})
                return
            stats["sec_b"] += time.time() - t0
            row = compare(out, out_b)
            row["turn"] = turn
            row["band"] = turn_band(turn)
            row["kind"] = out.get("kind")
            rows.append(row)

        try:
            DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2,
                        max_steps=max_steps, observer=observe)
            stats["games"] += 1
        except DR.GameAborted:
            stats["aborted"] += 1
    return rows, stats


def block(rows):
    if not rows:
        return None
    ok = [r for r in rows if "error" not in r and r.get("coverage") is not None]
    if not ok:
        return {"n": len(rows), "errors": len(rows)}
    n = len(ok)
    cov = sorted(r["coverage"] for r in ok)
    dc = [r["don_coverage"] for r in ok if r["don_coverage"] is not None]
    bia = [r["best_in_a"] for r in ok if r["best_in_a"] is not None]
    return {
        "n": n, "errors": sum(1 for r in rows if "error" in r),
        "k_a_mean": round(sum(r["k_a"] for r in ok) / n, 2),
        "k_b_mean": round(sum(r["k_b"] for r in ok) / n, 2),
        "coverage_mean": round(sum(cov) / n, 4),
        "coverage_min": round(cov[0], 4),
        "coverage_p05": round(cov[max(int(0.05 * n) - 1, 0)], 4),
        "missing_any": round(sum(1 for r in ok if r["missing"] > 0) / n, 4),
        "missing_mean": round(sum(r["missing"] for r in ok) / n, 3),
        "extra_any": round(sum(1 for r in ok if r["extra"] > 0) / n, 4),
        "best_not_in_a": (round(sum(1 for b in bia if not b) / len(bia), 4) if bia else None),
        "best_checked": len(bia),
        "exhausted_a_any": round(sum(1 for r in ok if r["exhausted_a"] > 0) / n, 4),
        "don_rows": len(dc),
        "don_coverage_mean": (round(sum(dc) / len(dc), 4) if dc else None),
        "don_missing_any": (round(sum(1 for r in ok if r["don_missing"] > 0) / n, 4)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=6, help="回す局数（腕 B は遅いので小さく）")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--sims", type=int, default=E.SERVE_SIMS)
    ap.add_argument("--net", default="", help="空＝出荷既定（`DEFAULT_NET`）")
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real", "purple"))
    ap.add_argument("--decks", default="synth_roles",
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "user"))
    ap.add_argument("--all-kinds", action="store_true",
                    help="main 以外の判断点（窓・対話）も測る（既定は main だけ）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    rows, stats = run_games(args.games, args.seed_base, args.sims, args.net or None,
                            args.leaders, args.decks, own_main_only=not args.all_kinds)
    out = {"params": {"games": args.games, "seed_base": args.seed_base, "sims": args.sims,
                      "net": args.net or "default", "leaders": args.leaders,
                      "decks": args.decks, "all_kinds": bool(args.all_kinds)},
           "run": {**stats, "sec_b": round(stats["sec_b"], 1)},
           "all": block(rows)}
    for b in TURN_BANDS:
        out[b] = block([r for r in rows if r.get("band") == b])
    out["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
