"""**場から消える率はパワーの関数か**（T22・生存の複利割引の対・読み取り専用）。

`docs/cpu_theory_gap.md` の **T22**・`game_theory.md` **§14.1**。T21 が要求した測定。

## なぜ要るか——**身代わり項と対になっている**

`nu_of` は `R` ターンぶんの攻撃を足してから `(1 − ko_p)` を**一度だけ**掛ける。
しかも **`ko_p` は定数 0.289**。正しくは**毎ターン生き残る確率を掛けながら足す**
（`Σ lead · s^i`）。そして**高パワーほど死ににくい**はずなので `s` はパワーの関数である。

**片方だけ入れてはいけない**（T21 の結論）:

| 直す項 | どのパワー帯に厚いか |
|---|---|
| 身代わり（T21） | **低パワー**に厚い |
| 生存の複利割引（本件） | **高パワー**に厚い |

**逆を向く**ので、片方だけ入れると全体が悪化する。**現状は 2 つの誤りが中央帯で
打ち消し合っている**——だから両方を揃えてから `nu_measure` で検算する。

## 測り方——**自席ターンの入口を 2 つ並べて、消えた体を数える**

自席ターンの最初の main 行で自分の場のパワーの**多重集合**を取り、
**次の自席ターン**の同じものと比べる。減った分が「消えた体」。

> **枠の番号では追わない**——枠は入れ替わる。**多重集合の差**で数える。

**測っているのは「1 巡で場から消えた率」**であって「相手に KO された率」ではない。
`ν` の割引に要るのは**次のターンに居るか**なので前者が正しいが、**名前は `gone_p` にする**
——自分のコストで退場した体も、効果で戻した体も入る。

## 交絡（読むときに添える）

- **パワーは巡の間に変わりうる**（永続強化）＝「消えた」と「増えた」が同時に立つ。
  **強化が多い帯では過大**に出る。
- **その巡の間に出して、その巡の間に消えた体は見えない**（入口しか見ていない）。
- **最後の自席ターンの後は比べる先が無い**ので、各局の最終巡は数えない
  ＝**負けて場が壊滅した巡が落ちる**＝**全体として過小**に出る向きの偏り。
- **付与ドンは自席ターンの入口には乗っていない**（相手ターン終了時に外れる）ので、
  入口のパワーは素のパワーに近い。

## 読み方（事前登録）

- **パワー帯で `gone_p` が単調に下がれば**、`ko_p` をパワーの関数にする根拠になる。
- **平らなら定数 0.289 のままでよい**＝T21 の身代わり項を**単独で**入れてよいことになる
  （対が要らなくなるので、むしろ話が進む）。
- **単調でない**なら、パワー以外の何か（ブロッカー・効果）が効いている。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/ko_by_power.py --in ~/w39 ~/w42 --out ~/ko.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from theory_order import (KO_P, PWR_EPS, S_IS_BLOCKER, S_IS_CHAR, S_POWER,  # noqa: E402
                          SLOT_OWN_FIELD)

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
#: パワー帯（1000 刻み・`c(x)` 曲線と同じ目盛り）
POWER_BANDS = ((0, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10 ** 9))


def _extra(dd, n):
    return {"tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def _round10(x):
    """`1e4` を掛けて戻した量は 10 の単位に丸める（`measurement.md` 罠 14-5）。"""
    return round(float(x) / 10.0) * 10.0


def band_of(power):
    for lo, hi in POWER_BANDS:
        if lo <= power < hi:
            return "p%d" % (lo // 1000)
    return "p%d" % (POWER_BANDS[-1][0] // 1000)


def field_powers(tok_row, slots=SLOT_OWN_FIELD):
    """自分の場のパワーの多重集合（ブロッカーかも一緒に返す）。"""
    out = []
    for s in range(slots.start, slots.stop):
        if float(tok_row[s, S_IS_CHAR]) <= 0.5:
            continue
        pw = _round10(float(tok_row[s, S_POWER]) * 1e4)
        if pw < -PWR_EPS:
            continue
        out.append((pw, bool(float(tok_row[s, S_IS_BLOCKER]) > 0.5)))
    return out


def lost(before, after):
    """**多重集合の差**＝消えた体（枠の番号では追わない）。"""
    c_a = Counter(after)
    gone = []
    for key in before:
        if c_a[key] > 0:
            c_a[key] -= 1
        else:
            gone.append(key)
    return gone


def collect(dirs, limit_games=0):
    """(局, 席) ごとに自席ターンの入口を並べ、1 巡で消えた体を数える。"""
    recs = []
    stats = {"games": 0, "turns": 0, "pairs": 0, "last_turn_dropped": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        seats = {}
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            if t in seats.setdefault(w, {}):
                continue                      # そのターン最初の main 行だけ
            seats[w][t] = field_powers(ex["tok"][i])
            stats["turns"] += 1
        for w, byturn in seats.items():
            ts = sorted(byturn)
            for a, b in zip(ts, ts[1:]):
                stats["pairs"] += 1
                before = byturn[a]
                c_gone = Counter(lost(before, byturn[b]))
                for key in before:
                    g = 1 if c_gone[key] > 0 else 0
                    if g:
                        c_gone[key] -= 1
                    recs.append({"seed": seed, "turn": a, "power": key[0],
                                 "blocker": key[1], "gone": g})
            stats["last_turn_dropped"] += 1 if ts else 0
    return recs, stats


def _rate(sub):
    if not sub:
        return None
    return {"n": len(sub), "gone_p": round(float(np.mean([r["gone"] for r in sub])), 4),
            "power_mean": round(float(np.mean([r["power"] for r in sub])), 1)}


def _boot(recs, keys, reps=200, seed=0):
    """対局を復元抽出して帯ごとの `gone_p` の CI。"""
    if reps <= 0 or not recs:
        return {k: [None, None] for k in keys}
    by = {}
    for r in recs:
        by.setdefault(r["seed"], []).append(r)
    gids = list(by)
    if len(gids) < 3:
        return {k: [None, None] for k in keys}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in keys}
    for _ in range(int(reps)):
        pick = rng.integers(0, len(gids), len(gids))
        sub = [r for g in pick for r in by[gids[g]]]
        per = {}
        for r in sub:
            per.setdefault(band_of(r["power"]), []).append(r["gone"])
        for k in keys:
            if per.get(k):
                acc[k].append(float(np.mean(per[k])))
    out = {}
    for k, v in acc.items():
        out[k] = ([round(float(np.percentile(v, 2.5)), 4),
                   round(float(np.percentile(v, 97.5)), 4)] if len(v) >= 10 else [None, None])
    return out


def monotone_down(vals):
    """**パワー帯で単調に下がるか**（事前登録した読み方）。"""
    v = [x for x in vals if x is not None]
    return bool(len(v) >= 2 and all(v[i] >= v[i + 1] for i in range(len(v) - 1)))


def spread_of(vals):
    """**帯の最大 − 最小**（2026-09-14 に直した）。

    初版は**最初の帯 − 最後の帯**で測っていた。実測は**山形**（4000〜6000 が峰）で、
    両端だけ見ると 0.043＝「平ら」と出るのに、**峰と谷の差は 0.13 で CI が重ならない**。
    **端点差は「動いていない」の証拠にならない**（罠 22 の親戚＝見る量を間違えている）。
    """
    v = [x for x in vals if x is not None]
    return round(max(v) - min(v), 4) if len(v) >= 2 else None


def summarise(recs, reps=200, seed=0):
    by = {}
    for r in recs:
        by.setdefault(band_of(r["power"]), []).append(r)
    keys = sorted(by, key=lambda k: int(k[1:]))
    out = {"all": _rate(recs), "ko_p_current": KO_P,
           "by_power": {k: _rate(by[k]) for k in keys},
           "by_power_ci95": _boot(recs, keys, reps, seed)}
    # ブロッカーは別に出す（`ν` ではブロック項と二重に効きうる）
    out["by_blocker"] = {("blocker" if b else "vanilla"):
                         _rate([r for r in recs if r["blocker"] == b]) for b in (True, False)}
    vals = [out["by_power"][k]["gone_p"] if out["by_power"][k] else None for k in keys]
    out["monotone_down"] = monotone_down(vals)
    out["spread"] = spread_of(vals)
    # **素の体だけの曲線も出す**——ブロッカーは殴られて死ぬので、
    # 「パワーの効き」とブロッカーの分布が混ざる（実測でブロッカーは 0.36 対 0.26）。
    van = [r for r in recs if not r["blocker"]]
    by_v = {}
    for r in van:
        by_v.setdefault(band_of(r["power"]), []).append(r)
    out["by_power_vanilla"] = {k: _rate(by_v.get(k, [])) for k in keys}
    vv = [out["by_power_vanilla"][k]["gone_p"] if out["by_power_vanilla"][k] else None
          for k in keys]
    out["vanilla_monotone_down"] = monotone_down(vv)
    out["vanilla_spread"] = spread_of(vv)
    # **事前登録した 3 分岐**（docstring の「読み方」）: 単調に下がる／平ら／単調でない
    if out["spread"] is not None:
        out["verdict"] = ("flat_constant_is_fine" if out["spread"] < 0.05 else
                          "power_dependent" if out["monotone_down"] else
                          "non_monotone_something_else_binds")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, stats = collect(a.src, a.limit_games)
    res = {"stats": stats, "rows": len(recs),
           "summary": summarise(recs, a.boot_reps, a.seed),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
