"""**登場時効果の価格を部品に割って、実現の部品と並べる**（T59・読み取り専用・2026-09-16）。

T54 で「登場時効果 `LOOK`／`PLAY_CARD`／`KO` の登場は価格が実現の 0.54〜0.74」と出た。**登場の行の価格は
体（`ν − μ`）＋登場時効果＋機会費用の和**なので、比だけでは**どの部品が外れているか判らない**。ここでは
`price_realised.collect` の行（`play_parts` つき）を登場時能力の最初の動作の型で切り、

```
価格: body = ν − μ + μ = ν（体の式）   effect = 登場時効果の価格   opportunity = 機会費用
実現: my_body（自分の体の ν_meas の差）  my_hand（μ × 手札の差）  my_life（λ × 自ライフの差）  opp（相手の体・ライフ・手札）  don
```

を**行の平均**で並べる。**回帰しない・当てはめない**。

- `LOOK`（探す）の効果の価格は `μ`（手札 +1）＋**選択の利得** `E[max_k W] − E[W]`。実現の物差し（次の判断点の
  `S_meas`）は**手札 1 枚を `μ` としか数えない**ので、選んだ札の質（選択の利得）は**定義上見えない**。
  見えるのは「札が入ったか」（`found`＝手札の差 ≥ 0＝出した札の分を埋めた）だけ。
- `PLAY_CARD`／`KO` の効果の価格は**体の `ν`**（出す体は `NU_AVG`・KO は対象の式の `ν`）なので、外れは
  T50 の「式の `ν` は帯の 1.1〜1.2 倍（生存の重み `(1−ko_p)·R` 対 生きて迎えたターン数）」と同じ穴。

使い方: `python tests/scripts/onplay_parts.py --in <n_records ディレクトリ>... [--out x.json]`
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import effect_value as EV  # noqa: E402
import price_realised as PR  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_order import MU  # noqa: E402

#: 実現の部品のうち「相手の側」に括るもの
OPP_PARTS = ("opp_body", "opp_life", "opp_hand")
#: 並べる型（他の型も出すが、読むのはこの 3 つ）
FOCUS_ACTS = ("LOOK", "PLAY_CARD", "KO")


def look_k(cid, cards=None):
    """登場時能力が**何枚から選べるか**（`selection_k`・能力ごとの最大）。登場時能力が無ければ 0。"""
    c = (cards or EV._all_cards()).get(str(cid) or "")
    if not c:
        return 0.0
    ks = [EV.selection_k(EV.walk_actions(ab.get("effect")))
          for ab in (c.get("abilities") or []) if ab.get("trigger") in EV.CHAR_ON_PLAY_TRIGGERS]
    return max(ks) if ks else 0.0


def row_parts(r, mu=MU):
    """1 行を「価格の部品」「実現の部品」に直す（`price_realised` の行＝`play_parts` つきの登場の行）。"""
    pp = r["play_parts"]
    body = float(pp["nu_minus_mu"]) + mu                  # 体の式 ν（`ν − μ` に出した札の μ を戻す）
    return {"price": float(r["price"]), "body": body, "effect": float(pp["effect"]),
            "opportunity": float(pp["opportunity"]),
            "real": float(r["real"]), "real_te": float(r["real_te"]),
            "my_body": float(pp["my_body"]), "my_hand": float(pp["my_hand"]), "my_life": float(pp["my_life"]),
            "opp": float(sum(pp[k] for k in OPP_PARTS)), "don": float(pp["don"]),
            # 手札の差 ≥ 0 ＝ 出した札の分を埋めた（探す効果が札を見つけた）。**T69**: 質の補正が載る前の枚数（`my_hand_count`）で読む
            "found": 1.0 if float(pp.get("my_hand_count", pp["my_hand"])) >= -0.5 * mu else 0.0}


def collect(dirs, limit_games=0, cards=None):
    """`price_realised.collect` を回し、登場の行（体つき）だけを型と札で束ねる。"""
    per, stats = PR.collect(dirs, limit_games)
    rows = []
    for rec in per.values():
        for r in rec["rows"]:
            if r["fam"] != "play" or r.get("play_parts") is None:
                continue
            d = row_parts(r)
            d["act"] = r["act"] or "?"
            d["cid"] = r["cid"]
            d["k"] = look_k(r["cid"], cards)
            rows.append(d)
    return rows, stats


def _mean(rows, key):
    return (sum(r[key] for r in rows) / len(rows)) if rows else 0.0


def block(rows):
    """束の平均と、部品ごとの `実現 / 価格`。"""
    keys = ("price", "body", "effect", "opportunity", "real", "real_te", "my_body", "my_hand", "my_life", "opp", "don",
            "found", "k")
    o = {"n": len(rows)}
    o.update({k: round(_mean(rows, k), 4) for k in keys})
    o["ratio_real_over_price"] = round(o["real"] / o["price"], 3) if abs(o["price"]) > 1e-9 else None
    # **体の部品どうし・効果の部品どうし**の比（登場の価格の外れがどこに在るか）
    o["ratio_body"] = round(o["my_body"] / o["body"], 3) if abs(o["body"]) > 1e-9 else None
    # 効果の実現＝体を除いた残り（手札 + 自ライフ + 相手の側 + ドン）に、出した札の −μ を戻す。
    # **出す効果（`PLAY_CARD`）の体は `my_body` に混ざる**ので、その型はこの比ではなく総額で読む
    eff_real = o["my_hand"] + o["my_life"] + o["opp"] + o["don"] + MU
    o["effect_real"] = round(eff_real, 4)
    o["ratio_effect"] = round(eff_real / o["effect"], 3) if abs(o["effect"]) > 1e-9 else None
    return o


def summarise(rows, min_card_rows=8):
    by_act, by_card = {}, {}
    for r in rows:
        by_act.setdefault(r["act"], []).append(r)
        by_card.setdefault((r["act"], r["cid"]), []).append(r)
    out = {"n_rows": len(rows),
           "by_act": {a: block(rs) for a, rs in sorted(by_act.items(), key=lambda kv: -len(kv[1]))},
           "by_card": {"%s|%s" % (a, c): block(rs) for (a, c), rs in sorted(by_card.items(), key=lambda kv: -len(kv[1]))
                       if len(rs) >= min_card_rows and a in FOCUS_ACTS}}
    # **`LOOK` の効果の価格の内訳**: μ（手札 +1）＋選択の利得（k 枚）。選択の利得は物差しに見えない
    lk = by_act.get("LOOK", [])
    if lk:
        prem = [EV._sel_premium(int(r["k"])) for r in lk]
        out["look"] = {"n": len(lk), "k_mean": round(_mean(lk, "k"), 3),
                       "premium_mean": round(sum(prem) / len(prem), 4),
                       "mu": round(MU, 4), "found": round(_mean(lk, "found"), 3),
                       "effect_price_mean": round(_mean(lk, "effect"), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--min-card-rows", type=int, default=8)
    TO.add_nu_mode_arg(ap)
    TO.add_surv_mode_arg(ap)
    TO.add_cbar_mode_arg(ap)
    EV.add_search_price_arg(ap)
    PR.add_hand_meas_arg(ap)
    EV.add_play_now_arg(ap)
    import hand_plan as _HP
    _HP.add_inflow_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    TO.apply_nu_mode(a)
    TO.apply_surv_mode(a)
    TO.apply_cbar_mode(a)
    EV.apply_search_price(a)
    PR.apply_hand_meas(a)
    EV.apply_play_now(a)
    _HP.apply_inflow_mode(a)
    t0 = time.time()
    rows, stats = collect(a.src, a.limit_games)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "search_price": EV.SEARCH_PRICE_MODE,
           "hand_meas": PR.HAND_MEAS_MODE, "play_now": EV.PLAY_NOW_MODE, "inflow": _HP.INFLOW_MODE, "stats": stats,
           "summary": summarise(rows, a.min_card_rows),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
