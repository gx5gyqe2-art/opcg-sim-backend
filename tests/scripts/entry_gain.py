"""**登場時の一回性の利得を測る**（`ν` の外にある項・読み取り専用）。

ユーザ指摘 2026-09-14（一覧の 1 番目）「**サーチやドローなどによる手札の量増加または手札の質の
向上**」＋「**ちゃんと支払ったコストと価格が見合っているか**」。`docs/game_theory.md` §14.1.1。

```
play_value = ν ＋ 【登場時の一回性の利得】 − μ − cost·δ
                  ↑ 式に無かった項。ここを測る
```

**実測の `ν` にはこの項は入らない**——`ν` は「体が場に居る間にすること」の傾きで、
サーチ・ドロー・登場時除去は**登場の瞬間に払い出され、カードは手札へ行く**。

## 勝率に回帰しない（ここが設計の要点）

登場時の利得は**他の通貨の増減として現れる**ので、**実際に動いた量を数えて既知の価格を掛ける**
だけで出る。勝率に回帰すると「除去は対象が居るときに打つ」という**選択交絡**がそのまま乗るが、
**通貨の差分は選択交絡と無関係**（何が起きたかを数えているだけ）。

```
利得 = (Δ手札 + 1)·μ          … 引いた／探した分（登場の機械的な −1 を戻す）
     + (Δ自場 − 1)·ν          … 体 1 つ以外に場が増えた分
     + (−Δ相手場)·ν           … 相手の場を減らした分（登場時除去）
     + (−Δ相手ライフ)·λ       … 相手のライフを削った分
     + (Δ自ライフ)·λ          … 自分のライフが動いた分
```

**差分の取り方**: 登場を選んだ main 行 → **同じターンの次の main 行**の盤面。間に在るのは
その登場の解決だけ（効果の対象選択などは `kind != 0` の行）。次の main 行が無い（ターンが
終わった）登場は**測れないので外す**——`coverage` に出す。

## 読み方（事前登録）

`2026-09-14_nu_measure.md` の勘定で、**値付けできていない項が持つべき大きさ**が出ている:

| パワー帯 | 過払い（勝率換算） |
|---|---|
| リーダー未満 | **0.0290** |
| リーダー〜飽和 | 0.0158 |
| 飽和超え | 0.0219 |

- **測った利得がこの値に届けば、勘定が閉じる**＝式に足すべき項はこれで足りる。
- **届かなければ、まだ別の項が在る**（または「過払い」は CPU の漏れ）。
- **`ON_PLAY` の役割を持つ札に利得が集まっているか**も見る（集まっていなければ器を疑う）。

**限界**: `Δ相手場` を相手の `ν` の**平均**で値付けする（消したキャラのパワー帯は追っていない）。
手札の**質**の向上（サーチが無作為な 1 枚を選んだ 1 枚に変える）は `μ` で値付けするので**過小**。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/entry_gain.py --in ~/w32/*/n_records --out ~/entry_gain.json
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
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from nu_calib import SC_OPP_LEADER_POWER, _extra  # noqa: E402
from nu_measure import MU_TRUE, power_band  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_sig", "pol_cid")

#: scalars の列（`nu_measure` と同じ）
SC_MY_LIFE, SC_OPP_LIFE, SC_MY_HAND, SC_MY_FIELD, SC_OPP_FIELD = 0, 1, 6, 8, 9
#: 実測の価格（`docs/game_theory.md` §18）。`ν` は種類別の平均（`nu_measure` の `all`）
LAM_TRUE = 0.1158
NU_MEAN = 0.1230
#: 勘定の穴（`2026-09-14_nu_measure.md`）＝**利得がここに届けば閉じる**
OVERPAY = {"lt_leader": 0.0290, "leader_to_sat": 0.0158, "over_sat": 0.0219}


def deltas(sc_before, sc_after):
    """登場の前後で動いた通貨（**行為者の視点**・符号はそのまま）。"""
    g = {}
    for name, col in (("hand", SC_MY_HAND), ("my_field", SC_MY_FIELD),
                      ("opp_field", SC_OPP_FIELD), ("my_life", SC_MY_LIFE),
                      ("opp_life", SC_OPP_LIFE)):
        g["d_" + name] = float(round(float(sc_after[col]) - float(sc_before[col]), 3))
    return g


def gain_of(d, mu=MU_TRUE, lam=LAM_TRUE, nu=NU_MEAN, body=1.0):
    """通貨の差分 → **登場時の利得**（勝率の単位）。

    機械的な分（手札 −1・自場 +`body`）は `play_value` の他の項が持っているので**戻す**。
    **ステージは体を持たない**ので `body=0`（2026-09-14 に最弱帯へ 124 枚混ざっていた）。
    """
    return (
        (d["d_hand"] + 1.0) * mu            # 引いた／探した分
        + (d["d_my_field"] - body) * nu     # 体（ステージなら 0）以外に場が増えた分
        + (-d["d_opp_field"]) * nu          # 相手の場を減らした分
        + (-d["d_opp_life"]) * lam          # 相手のライフを削った分
        + d["d_my_life"] * lam              # 自分のライフが動いた分
    )


def collect(dirs, limit_games=0):
    """holdout の**打たれた登場** → 前後の通貨の差分（同じターンの次の main 行で測る）。"""
    from opcg_sim.loop import deck_roles as DR
    cards = PL.Cards()
    recs = []
    stats = {"games": 0, "plays": 0, "measurable": 0, "no_next_main": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        # (who, turn) ごとの main 行を順に並べる＝「次の main 行」を引くための索引
        mains = {}
        for pos, i in enumerate(idx):
            if int(rows["kind"][i]) != 0:
                continue
            mains.setdefault((int(rows["who"][i]), int(rows["turn"][i])), []).append(pos)
        for (who, turn), poss in mains.items():
            for a, b in zip(poss, poss[1:] + [None]):
                i = idx[a]
                ch = int(rows["pol_chosen"][i]); k = int(L[i]); j0 = int(ptr[i])
                if ch < 0 or ch >= k:
                    continue
                j = j0 + ch
                try:
                    sig = json.loads(pol["pol_sig"][j])
                except (ValueError, TypeError):
                    continue
                if sig[0] != "PLAY":
                    continue
                cid = str(pol["pol_cid"][j]) or None
                info = cards.info(cid)
                if not info or info.get("event"):
                    continue
                stats["plays"] += 1
                if info.get("stage"):
                    # **ステージは体を持たない**＝`ν` の勘定に入れない。ただし
                    # **登場時の利得だけで出来ている札**なので、別枠で測ると素性が出る
                    stats["stage"] = stats.get("stage", 0) + 1
                if b is None:                       # このターンに次の main 行が無い＝測れない
                    stats["no_next_main"] += 1
                    continue
                sc0 = ex["sc"][i]; sc1 = ex["sc"][idx[b]]
                master = cards.db.get_card(cid) if cid else None
                forms = DR.classify(master) if master is not None else set()
                opl = float(sc0[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                rec = {"seed": int(rows["seed"][idx[0]]), "turn": turn,
                       "band": ("stage" if info.get("stage")
                                else power_band(float(info["power"]), opl)),
                       "cost": float(info.get("cost") or 0),
                       "onplay": bool(any(":ON_PLAY:" in f for f in forms)),
                       "has_ability": bool(getattr(master, "abilities", ()) or ())}
                rec.update(deltas(sc0, sc1))
                recs.append(rec)
                stats["measurable"] += 1
    return recs, stats, games


def _mean_ci(vals, seeds):
    """平均と**対局でクラスタした** 95% CI。"""
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    per = {}
    for v, s in zip(vals, seeds):
        per.setdefault(s, []).append(float(v))
    g = np.array([np.mean(v) for v in per.values()], np.float64)
    m = float(g.mean())
    if len(g) < 2:
        return {"n": len(vals), "games": 1, "mean": round(m, 5), "ci95": None}
    se = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {"n": len(vals), "games": len(g), "mean": round(m, 5), "se": round(se, 5),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


DELTA_KEYS = ("d_hand", "d_my_field", "d_opp_field", "d_my_life", "d_opp_life")


def summarise(recs, mu=MU_TRUE, lam=LAM_TRUE, nu=NU_MEAN, overpay=OVERPAY):
    """帯ごとの通貨の差分と利得、そして**勘定の穴を埋めるか**。"""
    out = {}
    for band in sorted({r["band"] for r in recs}):
        sub = [r for r in recs if r["band"] == band]
        sd = [r["seed"] for r in sub]
        row = {"n": len(sub),
               "deltas": {k: _mean_ci([r[k] for r in sub], sd) for k in DELTA_KEYS},
               "gain": _mean_ci([gain_of(r, mu, lam, nu, 0.0 if band == "stage" else 1.0)
                                 for r in sub], sd),
               "onplay_share": round(float(np.mean([r["onplay"] for r in sub])), 4)}
        need = overpay.get(band)
        g = row["gain"]["mean"]
        if need is not None and g is not None:
            lo, hi = row["gain"]["ci95"] or (None, None)
            row.update(overpay_to_cover=need, covers=round(g / need, 3) if need else None,
                       # **穴を埋めるか**＝利得の CI が過払いを含むか（届いていれば勘定が閉じる）
                       closes_the_books=(None if lo is None else bool(lo <= need <= hi)),
                       reaches=bool(g >= need))
        # ON_PLAY の役割を持つ札に利得が集まっているか（器の妥当性検査）
        for tag, flt in (("with_onplay", True), ("without_onplay", False)):
            s2 = [r for r in sub if r["onplay"] is flt]
            row[tag] = _mean_ci([gain_of(r, mu, lam, nu, 0.0 if band == "stage" else 1.0)
                                 for r in s2], [r["seed"] for r in s2])
        out[band] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--mu", type=float, default=MU_TRUE)
    ap.add_argument("--lam", type=float, default=LAM_TRUE)
    ap.add_argument("--nu", type=float, default=NU_MEAN, help="相手の場を消した分の値付け（平均）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, stats, games = collect(a.src, a.limit_games)
    stats["coverage"] = (round(stats["measurable"] / stats["plays"], 4)
                         if stats["plays"] else None)
    res = {"games": games, "stats": stats,
           "by_band": summarise(recs, a.mu, a.lam, a.nu),
           "prices": {"mu": a.mu, "lam": a.lam, "nu": a.nu},
           "args": {k: v for k, v in vars(a).items() if k != "out"},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
