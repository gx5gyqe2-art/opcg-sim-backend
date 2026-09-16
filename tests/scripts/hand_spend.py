"""**守りで切った札は、手札の中で安い札か**（T64・読み取り専用・2026-09-16）。

ユーザ指摘 2026-09-16「実際は手札のカードの効果や引きの兼ね合いで、高いライフよりも手札の枚数や中身の方が高くつくことがある」。
理論は今、手札の札をどれも `μ`（平均 1 枚）と数え、守る費用を「枚数 c(x) × μ」で置いている。**人は手札の中で一番安い札から切る**
なら、守る費用は「切った札それぞれの価値の和」で、枚数 × μ より安い。ここではそれを記録で確かめる。

```
窓:     相手のターンに来た攻撃への応答（`plan_labels` の take／guard）。手札 = 枠 12〜21（`card_idx`・行の主の手札）
前:     その相手ターンで最初の自分の行の手札          後: 次の自席ターンの最初の main 行の手札（引いた札・ライフの札が増えるのは無視）
切った: 前 − 後（多重集合の差）
v(札):  その札を使ったときの価値（μ を除く）= キャラ: ν(P) + 登場時効果 − コスト·δ ／ イベント: 効果 − コスト·δ ／ ステージ: 能力 1 つ
順位:   手札を v の昇順に並べたときの切った札の位置（0 = 一番安い・1 = 一番高い）
```

**回帰しない・当てはめない**。`v` は `effect_value`／`theory_order` の既存の価格だけ（新定数ゼロ）。

使い方: `python tests/scripts/hand_spend.py --in <n_records ディレクトリ>... [--out x.json]`
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
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import effect_value as EV  # noqa: E402
import guard_afford as GA  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_order import (DELTA, MU, POL_COLS, SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE,  # noqa: E402
                          c_of, incoming_x)
from theory_bridge import ROW_COLS, _extra  # noqa: E402

#: 手札の枠（`guard_afford.SLOT_HAND` と同じ）
SLOT_HAND = GA.SLOT_HAND
#: 超過 x の帯（`attack_response` と同じ切り方）
X_BANDS = ((-1e9, -TO.PWR_EPS, "x<0"), (-TO.PWR_EPS, 1000.0 + TO.PWR_EPS, "0..1000"),
           (1000.0 + TO.PWR_EPS, 2000.0 + TO.PWR_EPS, "1000..2000"), (2000.0 + TO.PWR_EPS, 1e9, ">2000"))


def x_band(x):
    for lo, hi, name in X_BANDS:
        if lo < x <= hi:
            return name
    return "x<0"


def hand_ids(ci_row, idx2cid):
    """行の主の手札の card_id の並び（空の枠は落とす）。"""
    out = []
    for v in np.asarray(ci_row)[SLOT_HAND]:
        cid = idx2cid.get(int(v))
        if cid:
            out.append(str(cid))
    return out


def spent_cards(before, after):
    """多重集合の差＝前に在って後に無い札（引いた札・ライフの札が増える分は無視）。"""
    c = collections.Counter(before)
    c.subtract(collections.Counter(after))
    out = []
    for cid, k in c.items():
        out.extend([cid] * max(0, k))
    return out


def use_value(cid, info, opp_leader_power, r_turns, cards=None):
    """**その札を使ったときの価値（`μ` を除く）**＝切ったときの機会費用。読めなければ `None`。
    **0 で床を打つ**——使わない自由があるので機会費用は負にならない（コストの重い小さな体は 0）。"""
    if not info:
        return None
    cost = float(info.get("cost") or 0.0)
    if info.get("event"):
        v, _u = EV.card_value(cid, EV.ON_PLAY_TRIGGERS, cards=cards)
        return None if v is None else max(0.0, float(v) - cost * DELTA)
    if info.get("stage"):
        return max(0.0, float(EV.ABILITY_UNKNOWN) - cost * DELTA)
    power = float(info.get("power") or 0.0)
    body = TO.nu_of(power, float(opp_leader_power), float(r_turns), is_blocker=bool(info.get("blocker")))
    onplay, unp = EV.card_value(cid, EV.CHAR_ON_PLAY_TRIGGERS, no_ability=0.0, cards=cards)
    if onplay is None:
        if unp and unp[0][0] == "<no_card>":
            onplay = 0.0                                   # 効果 JSON に無い札＝素の体として読む
        else:
            return None                                    # 登場時能力が読めない札は None（0 にしない）
    return max(0.0, float(body) + float(onplay) - cost * DELTA)


def rank_of(values, target):
    """`target` の値が手札（`values`）の昇順の中でどこか（0 = 一番安い・1 = 一番高い・1 枚なら 0.5）。"""
    vs = sorted(values)
    if len(vs) <= 1:
        return 0.5
    below = sum(1 for v in vs if v < target - 1e-12)
    ties = sum(1 for v in vs if abs(v - target) <= 1e-12)
    return (below + 0.5 * max(0, ties - 1)) / (len(vs) - 1)


def collect(dirs, limit_games=0):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    windows = []
    stats = {"games": 0, "windows": 0, "no_after": 0, "no_label": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        life0 = ex["sc"][:, 0]
        labels, _unk = PL.label_game(rows, pol, life0, L, ptr, idx, cards)
        # 席ごとの自席ターンの最初の main 行（後の手札を読む場所）
        first_main = {}
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t >= 1 and PL.is_own_turn(w, t) and int(rows["kind"][i]) == 0 and (w, t) not in first_main:
                first_main[(w, t)] = i
        seen = set()
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or PL.is_own_turn(w, t) or (w, t) in seen:
                continue
            seen.add((w, t))
            lab = int(labels[n])
            if lab < 0:
                stats["no_label"] += 1
                continue
            played = PL.PLAN_CLASSES[lab]
            if played not in ("take", "guard"):
                continue
            j = first_main.get((w, t + 1))
            if j is None:
                stats["no_after"] += 1
                continue
            sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
            before = hand_ids(ci, idx2cid)
            after = hand_ids(ex["ci"][j], idx2cid)
            spent = spent_cards(before, after)
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
            vals = {}
            for cid in set(before):
                vals[cid] = use_value(cid, cards.info(cid), olp, r)
            known = {cid: v for cid, v in vals.items() if v is not None}
            hand_vals = [known[c] for c in before if c in known]
            xs = [x for x in incoming_x(tok) if x >= -TO.PWR_EPS]
            x_max = max(xs) if xs else None
            sp = []
            for cid in spent:
                inf = cards.info(cid) or {}
                sp.append({"cid": cid, "v": known.get(cid), "counter": float(inf.get("counter") or 0.0),
                           "event": bool(inf.get("event")),
                           "rank": rank_of(hand_vals, known[cid]) if cid in known and hand_vals else None})
            windows.append({"played": played, "x_max": x_max, "xb": x_band(x_max) if x_max is not None else "none",
                            "hand_n": len(before), "hand_v_mean": float(np.mean(hand_vals)) if hand_vals else None,
                            "hand_v_min": float(min(hand_vals)) if hand_vals else None,
                            "spent": sp, "my_life": int(round(float(sc[SC_MY_LIFE]))),
                            "c_x": float(c_of(x_max)) if x_max is not None else None})
            stats["windows"] += 1
    return windows, stats


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def block(ws):
    """窓の束 → 切った枚数・順位・価値の和と枚数×μ の比。"""
    sp = [s for w in ws for s in w["spent"]]
    n_sp = [len(w["spent"]) for w in ws]
    ranks = [s["rank"] for s in sp if s["rank"] is not None]
    vs = [s["v"] for s in sp if s["v"] is not None]
    kept_mean = _mean([w["hand_v_mean"] for w in ws])
    o = {"n": len(ws), "spent_per_window": float(np.mean(n_sp)) if n_sp else 0.0,
         "spent_cards": len(sp), "rank_mean": _mean(ranks),
         "share_cheapest_quartile": (float(np.mean([r <= 0.25 for r in ranks])) if ranks else None),
         "v_spent_mean": _mean(vs), "hand_v_mean": kept_mean,
         # **切った札の価値の和 対 枚数 × μ**（1 未満なら「枚数 × μ」は守る費用を高く見ている）
         "v_spent_over_mu": (float(np.sum(vs) / (len(vs) * MU)) if vs else None),
         "event_share": (float(np.mean([s["event"] for s in sp])) if sp else None),
         "counter_mean": _mean([s["counter"] for s in sp]),
         "c_x_mean": _mean([w["c_x"] for w in ws]),
         "life_mean": _mean([w["my_life"] for w in ws])}
    return o


def summarise(windows):
    out = {"all": block(windows)}
    for played in ("guard", "take"):
        ws = [w for w in windows if w["played"] == played]
        out[played] = block(ws)
        for xb in ("0..1000", "1000..2000", ">2000"):
            out["%s|%s" % (played, xb)] = block([w for w in ws if w["xb"] == xb])
    # 切った札の順位の分布（守った窓だけ）
    ranks = [s["rank"] for w in windows if w["played"] == "guard" for s in w["spent"] if s["rank"] is not None]
    if ranks:
        hist = collections.Counter(min(4, int(r * 5)) for r in ranks)
        out["guard_rank_hist"] = {"%.1f-%.1f" % (k / 5, (k + 1) / 5): round(v / len(ranks), 3) for k, v in sorted(hist.items())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    TO.add_nu_mode_arg(ap)
    TO.add_surv_mode_arg(ap)
    TO.add_cbar_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    TO.apply_nu_mode(a)
    TO.apply_surv_mode(a)
    TO.apply_cbar_mode(a)
    t0 = time.time()
    windows, stats = collect(a.src, a.limit_games)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "cbar_mode": a.cbar_mode, "mu": MU,
           "stats": stats, "summary": summarise(windows), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
