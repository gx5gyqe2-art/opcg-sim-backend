"""PIMC の診断——**strategy fusion が実際に起きているか**と、PIMC の適性 3 性質
（`docs/game_theory.md` §5・§6・§25 の宿題 3／4・読み取り専用）。

PIMC は情報集合から世界を `K` 本引き、**各世界を完全情報ゲームとして**解いて束ねる。
理論上 2 つの誤りが残る（Frank & Basin 以来の指摘）:

- **strategy fusion**: 世界ごとに別の手を選べる前提で価値を作る（実際には区別できない
  状態では同じ手しか打てない）。
- **non-locality**: ある世界の最善が情報集合全体の最適と一致しない。

**本器は「起きているか」を観測する**。`decide` は `worlds>1` のとき **`per_world`**
（世界ごとの根の N/Q と最善手）を返すので、束ねる前の各世界の結論が読める:

```
disagree      = 世界の最善手が全部一致しない判断点の割合         … 融合の必要条件
merged_novel  = **束ねた選択がどの世界の最善でもない**割合        … 融合の直接の観測
q_spread      = 選んだ手の Q の世界間のばらつき（max − min）      … leaf correlation の逆
unmapped      = 世界間で候補の対応が付かなかった数（本来 0）
```

`q_spread` が小さければ **leaf correlation が高い**（どの世界でも同じ側が勝つ）＝PIMC が効く
条件（`game_theory.md` §6）。大きい帯は PIMC が壊れる帯。

**disambiguation factor**（隠れ情報がどれだけ速く公開になるか）も同時に出す:
未見プール（相手の手札＋山＋伏せライフの枚数）がターンごとにどれだけ縮むか。

## 測り方

自己対戦を**候補席だけ `worlds=K`** で回し（既定 `K=8`）、自席の main 行で `per_world` を読む。
`worlds` を上げること自体の強さへの効果は別の測定（アリーナ 2 腕・台帳 §6）＝
**本器は強さではなく「融合が起きているか」だけを見る**。

## 出すもの

帯（ターン帯・`|v0|` の接戦帯）ごとに上の 4 つと、`k_legal`・候補数・世界数。
`disambiguation` は「相手の未見枚数」の平均とターンあたりの減り方。

## 近似

- `per_world` は**世界 0 の `legal` の並び**で返る（対応が付かない手は `unmapped` に数える）。
- 「最善手」は各世界の訪問最大（`N`）。訪問が 0 の世界（sims が薄い）は除く。
- 未見プールは枚数だけ（中身は公平性の規約で使わない・`game_theory.md` §4）。
- 融合の**損の大きさ**は本器では測れない（反事実が要る）＝「起きているか」までが射程。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/pimc_diag.py --games 8 --seed-base 9300 \\
    --worlds 8 --out ~/pimc_diag.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

TURN_BANDS = ("T<=4", "T5-8", "T9+")
V0_BANDS = ("close", "mid", "decided")


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def v0_band(v0, close=0.2, decided=0.6):
    a = abs(float(v0))
    return "close" if a <= close else ("decided" if a > decided else "mid")


def world_best(w):
    """1 世界の最善手の添字（訪問最大・訪問が全部 0 なら None）。"""
    ns = w.get("N") or []
    if not ns:
        return None
    b = int(max(range(len(ns)), key=lambda i: float(ns[i])))
    return b if float(ns[b]) > 0 else None


def merged_best(out):
    """束ねた根の選択（`groups` の訪問最大の代表添字・無ければ None）。"""
    groups = out.get("groups") or []
    if not groups:
        return None
    g = max(groups, key=lambda x: float(x.get("n") or 0.0))
    return int(g["rep"]) if float(g.get("n") or 0.0) > 0 else None


def fusion_of(out):
    """1 判断点 → 融合の観測（世界の不一致・束ねた選択が世界の最善か・Q のばらつき）。"""
    pw = out.get("per_world") or []
    if len(pw) < 2:
        return None
    bests = [world_best(w) for w in pw]
    known = [b for b in bests if b is not None]
    if len(known) < 2:
        return None
    mb = merged_best(out)
    qs = []
    for w in pw:
        q = w.get("Q") or []
        if mb is not None and mb < len(q):
            qs.append(float(q[mb]))
    row = {
        "worlds": len(pw),
        "k_legal": len((out.get("stats") or {}).get("legal") or []),
        "n_cands": len(out.get("groups") or []),
        # **世界の最善手が割れているか**（融合の必要条件）
        "disagree": len(set(known)) > 1,
        "n_distinct_best": len(set(known)),
        # **束ねた選択がどの世界の最善でもない**（融合の直接の観測）
        "merged_novel": (mb is not None and mb not in set(known)),
        # 選んだ手の Q の世界間のばらつき＝leaf correlation の逆
        "q_spread": (round(max(qs) - min(qs), 4) if len(qs) >= 2 else None),
        "q_sd": (round(float(np.std(qs)), 4) if len(qs) >= 2 else None),
        "unmapped": int(sum(int(w.get("unmapped") or 0) for w in pw)),
    }
    return row


def unseen_of(out, game):
    """未見プールの枚数（相手の手札＋山＋伏せライフ）＝disambiguation の材料。"""
    try:
        counts = json.loads(game.deck_counts_json())
    except Exception:                                      # noqa: BLE001
        return None
    return counts


def run_games(games, seed_base, worlds=8, sims=None, net=None,
              leaders="random", decks="synth_roles", max_steps=DR.DEFAULT_MAX_STEPS):
    """候補席だけ `worlds=K` で自己対戦し、自席 main 行の `per_world` を読む。"""
    db = D.load_db()
    spec = E.SeatSpec(net=net, sims=sims or E.SERVE_SIMS, worlds=int(worlds))
    rows = []
    stats = {"games": 0, "aborted": 0, "points": 0, "with_worlds": 0}

    for g in range(games):
        seed = seed_base + g
        try:
            la, lb = D.leader_pair(db, seed, leaders)
            p1, p2 = D.build_pair(db, la, lb, seed, decks)
        except Exception:                                   # noqa: BLE001
            stats["aborted"] += 1
            continue

        def observe(game, _name, turn, _step, out, _move):
            stats["points"] += 1
            row = fusion_of(out)
            if row is None:
                return
            stats["with_worlds"] += 1
            row["turn"] = turn
            row["band"] = turn_band(turn)
            v0 = out.get("v0")
            if v0 is None:
                groups = out.get("groups") or []
                tot = sum(float(x.get("n") or 0.0) for x in groups) or 1.0
                v0 = sum(float(x.get("n") or 0.0) * float(x.get("q") or 0.0)
                         for x in groups) / tot
            row["v0"] = float(v0)
            row["v0_band"] = v0_band(v0)
            row["kind"] = out.get("kind")
            rows.append(row)

        try:
            DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2,
                        max_steps=max_steps, observer=observe)
            stats["games"] += 1
        except DR.GameAborted:
            stats["aborted"] += 1
    return rows, stats


def _rate(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def block(rows):
    if not rows:
        return None
    qs = [r["q_spread"] for r in rows if r.get("q_spread") is not None]
    return {
        "n": len(rows),
        "disagree": _rate(rows, "disagree"),
        "merged_novel": _rate(rows, "merged_novel"),
        "n_distinct_best_mean": _rate(rows, "n_distinct_best"),
        "q_spread_mean": round(float(np.mean(qs)), 4) if qs else None,
        "q_spread_p90": round(float(np.percentile(qs, 90)), 4) if qs else None,
        "q_sd_mean": _rate(rows, "q_sd"),
        "k_legal_mean": _rate(rows, "k_legal"),
        "n_cands_mean": _rate(rows, "n_cands"),
        "unmapped_total": int(sum(r.get("unmapped") or 0 for r in rows)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--worlds", type=int, default=8, help="世界数 K（2 以上で `per_world` が出る）")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--net", default="")
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real", "purple"))
    ap.add_argument("--decks", default="synth_roles",
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "user"))
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    rows, stats = run_games(args.games, args.seed_base, args.worlds, args.sims,
                            args.net or None, args.leaders, args.decks)
    out = {"params": {"games": args.games, "seed_base": args.seed_base,
                      "worlds": args.worlds, "sims": args.sims or E.SERVE_SIMS,
                      "net": args.net or "default", "decks": args.decks},
           "run": stats,
           "all": block(rows),
           "by_turn": {b: block([r for r in rows if r["band"] == b]) for b in TURN_BANDS},
           "by_v0": {b: block([r for r in rows if r["v0_band"] == b]) for b in V0_BANDS},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
