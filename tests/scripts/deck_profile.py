"""デッキの素性——**τ・`c̄(x)` の曲線・θ の尺度**をデッキそのものから出す
（`docs/life_budget.md` §4・§12・記録も対局も要らない・読み取り専用）。

**なぜ記録が要らないか**: デッキは seed から決定論で作れる（`decks.leader_pair` →
`decks.build_pair`）。`life_budget.md` の 3 つの量はデッキの**中身だけ**で決まる:

```
τ        … ライフを受けたとき期待できるトリガーの確率（＝デッキの【トリガー】密度）
c̄(x)     … 超過パワー x を止めるのに要る枚数の期待値（手札 H 枚を引いたときの最小枚数）
θ の尺度 … リーダーのライフ（`θ = A·T − B` と比べる相手＝`L` の出発点）
```

`c̄` は予算の判断そのもの（`S(受ける) − S(守る) = c(x) − c̄`）なので、**デッキごとに違う**なら
「ライフ 3 で守り始める」のような固定のしきい値は原理的に誤り（ユーザ指摘 2026-09-12）。
本器は `c̄(x)` を**デッキの分布から直接**出すので、打ち回しの記録に含まれる交絡が無い。

### 出すもの（デッキ・リーダーごとと、全体の分布）

- `tau` … 【トリガー】を持つ札の割合（50 枚中）。`tau_by_action` でトリガーの中身も割る
  （KO／ライフ回復／付与…＝**受けたときの期待値が「何をくれるか」で変わる**）
- `counter` … 印字カウンターの分布（0／1000／2000…）・平均・**カウンターを持つ札の割合**
- `cbar_curve` … `x = 1000…5000` ごとの **`E[c_min(x)]`**（手札 `--hand` 枚を無作為に引いた
  ときの最小枚数の期待値・止められない確率 `p_cant` も出す）＝**デッキの守りの重さ**
- `counter_events` … 【カウンター】イベントの枚数とコスト（§18 の「ドンが要る守り」の材料）
- `leader_life` … リーダーのライフ（θ と比べる `L` の出発点）
- `roles` … `deck_roles.classify` の型の枚数（除去・ブロッカー・ドロー…＝`K` の材料）

### 近似

- `c̄(x)` は**無作為な手札**の期待値＝実プレイの手札（引いて使った後）ではない。
  実プレイ側の `c̄` は `budget_audit.py` が記録から出すので、**2 つを並べると
  「デッキが持つ守りの重さ」と「実際に手札に在った量」の差**が読める。
- τ は密度であって期待値ではない（トリガーの中身の強さは測らない・`tau_by_action` で型だけ出す）。
- 合成デッキ（`--decks synth_roles`）は seed ごとに違う＝**分布として読む**（1 つの deck の値は
  その seed のもの）。実デッキを見たいときは `--decks user`。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/deck_profile.py --games 300 --seed-base 2320000 \\
    --decks synth_roles --out ~/deck_profile.json
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import deck_roles as DR  # noqa: E402
from opcg_sim.src.models.effect_types import TriggerType  # noqa: E402

X_GRID = (1000, 2000, 3000, 4000, 5000)


def card_facts(m):
    """1 枚 → 予算に効く素性（トリガー・カウンター・種別・コスト・役割）。"""
    abils = tuple(m.abilities or ())
    trig = [a for a in abils if a.trigger == TriggerType.TRIGGER]
    counter_ab = [a for a in abils if a.trigger == TriggerType.COUNTER]
    kind = getattr(getattr(m, "type", None), "name", "")
    return {
        "trigger": bool(trig),
        "trigger_actions": tuple(sorted({_action_name(a) for a in trig})),
        "counter": int(getattr(m, "counter", 0) or 0),
        "counter_event": bool(counter_ab) and kind == "EVENT",
        "cost": int(getattr(m, "cost", 0) or 0),
        "event": kind == "EVENT",
        "roles": tuple(sorted({k.split(":", 1)[0] for k in DR.classify(m)})),
    }


def _action_name(ab):
    """トリガーの中身を 1 語で（効果の先頭の action_type・引けなければ "?"）。"""
    eff = getattr(ab, "effect", None)
    for attr in ("action_type", "action"):
        v = getattr(eff, attr, None)
        if v is not None:
            return getattr(v, "name", str(v))
    # Sequence／Branch などは先頭の要素を覗く
    for attr in ("actions", "steps", "items"):
        seq = getattr(eff, attr, None)
        if seq:
            return _action_name(type("A", (), {"effect": seq[0]})())
    return "?"


def c_min(values, x):
    """`x` を止める最小枚数（大きい順・`budget_audit.c_min` と同じ規則）。止められないなら None。"""
    got = 0.0
    for n, v in enumerate(sorted((float(v) for v in values if v > 0), reverse=True), 1):
        got += v
        if got >= x:
            return n
    return None


def cbar_curve(counters, hand=6, draws=400, rng=None):
    """デッキのカウンター分布 → `x` ごとの `E[c_min(x)]` と「止められない確率」。

    手札 `hand` 枚を**無作為に**引いて（デッキ 50 枚から非復元）最小枚数を数える平均。
    """
    rng = rng or np.random.default_rng(12345)
    arr = np.asarray(counters, np.float64)
    out = {}
    samples = [rng.choice(arr, size=min(hand, len(arr)), replace=False) for _ in range(draws)]
    for x in X_GRID:
        cs = [c_min(s, x) for s in samples]
        ok = [c for c in cs if c is not None]
        out[str(x)] = {"cbar": round(float(np.mean(ok)), 3) if ok else None,
                       "p_cant": round(1.0 - len(ok) / len(cs), 4)}
    return out


def profile_deck(db, leader_id, deck_ids, hand=6, draws=400, rng=None):
    """1 デッキ（リーダー＋50 枚）→ 素性。"""
    facts = [card_facts(db.get_card(c)) for c in deck_ids]
    n = len(facts)
    counters = [f["counter"] for f in facts]
    trig_actions = collections.Counter()
    for f in facts:
        for a in f["trigger_actions"]:
            trig_actions[a] += 1
    roles = collections.Counter()
    for f in facts:
        for r in f["roles"]:
            roles[r] += 1
    ce = [f for f in facts if f["counter_event"]]
    leader = db.get_card(leader_id)
    return {
        "leader": leader_id,
        "leader_life": int(getattr(leader, "life", 0) or 0),
        "n": n,
        "tau": round(sum(1 for f in facts if f["trigger"]) / n, 4),
        "tau_by_action": dict(trig_actions.most_common(8)),
        "counter": {
            "mean": round(float(np.mean(counters)), 1),
            "with_counter": round(sum(1 for c in counters if c > 0) / n, 4),
            "hist": {str(k): v for k, v in sorted(collections.Counter(counters).items())},
        },
        "counter_events": {"n": len(ce),
                           "cost_mean": (round(float(np.mean([f["cost"] for f in ce])), 2)
                                         if ce else None)},
        "cbar_curve": cbar_curve(counters, hand, draws, rng),
        "roles": dict(roles.most_common(10)),
        "cost_mean": round(float(np.mean([f["cost"] for f in facts])), 2),
    }


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.asarray(vals, np.float64)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std()), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "n": len(vals)}


def summarize(profiles):
    """デッキ集合 → 分布（**散らばりが「しきい値はデッキごと」の証拠**）。"""
    out = {"decks": len(profiles),
           "tau": _agg([p["tau"] for p in profiles]),
           "leader_life": _agg([p["leader_life"] for p in profiles]),
           "counter_mean": _agg([p["counter"]["mean"] for p in profiles]),
           "with_counter": _agg([p["counter"]["with_counter"] for p in profiles]),
           "counter_events_n": _agg([p["counter_events"]["n"] for p in profiles]),
           "cost_mean": _agg([p["cost_mean"] for p in profiles])}
    out["cbar"] = {x: _agg([p["cbar_curve"][x]["cbar"] for p in profiles]) for x in
                   (str(v) for v in X_GRID)}
    out["p_cant"] = {x: _agg([p["cbar_curve"][x]["p_cant"] for p in profiles]) for x in
                     (str(v) for v in X_GRID)}
    # リーダーごと（出現数の多い順・**θ の尺度はリーダーのライフで決まる**）
    by_leader = collections.defaultdict(list)
    for p in profiles:
        by_leader[p["leader"]].append(p)
    out["by_leader"] = {
        k: {"n": len(v), "life": v[0]["leader_life"],
            "tau": round(float(np.mean([p["tau"] for p in v])), 4),
            "cbar_3000": round(float(np.mean([p["cbar_curve"]["3000"]["cbar"] for p in v
                                              if p["cbar_curve"]["3000"]["cbar"] is not None])), 3)
            if any(p["cbar_curve"]["3000"]["cbar"] is not None for p in v) else None,
            "with_counter": round(float(np.mean([p["counter"]["with_counter"] for p in v])), 4)}
        for k, v in sorted(by_leader.items(), key=lambda kv: -len(kv[1]))[:12]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200, help="seed の本数（1 seed で 2 デッキ）")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real", "purple"))
    ap.add_argument("--decks", default="synth_roles",
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "user"))
    ap.add_argument("--hand", type=int, default=6, help="`c̄` を測る手札の枚数")
    ap.add_argument("--draws", type=int, default=400, help="1 デッキあたりの手札サンプル数")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    db = D.load_db()
    rng = np.random.default_rng(args.seed_base)
    profiles = []
    failed = 0
    for g in range(args.games):
        seed = args.seed_base + g
        try:
            la, lb = D.leader_pair(db, seed, args.leaders)
            p1, p2 = D.build_pair(db, la, lb, seed, args.decks)
        except Exception:                                  # noqa: BLE001
            failed += 1
            continue
        for leader_id, deck_ids in (p1, p2):
            profiles.append(profile_deck(db, leader_id, deck_ids, args.hand, args.draws, rng))
    out = {"params": {"games": args.games, "seed_base": args.seed_base,
                      "leaders": args.leaders, "decks": args.decks,
                      "hand": args.hand, "draws": args.draws},
           "failed_seeds": failed,
           "summary": summarize(profiles),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
