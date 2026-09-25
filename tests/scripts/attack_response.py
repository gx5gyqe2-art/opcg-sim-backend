"""**攻撃 1 回を相手の応答で割る**（T48・2026-09-16・読み取り専用・記録だけ）。

T47 で**攻撃の価格は実現の 0.7〜0.8 倍（安すぎる）**と判った。式は `min(c(x)·μ, 受ける, ブロック)`＝
「相手が一番安い応答を選ぶ」なので、不足分は**どの応答で**出ているかで分けられる（ユーザ指示「a から詰める」）:

| 応答（記録の差分で判る） | 式の想定 | 実現に出るもの |
|---|---|---|
| **受けた**（相手ライフ−） | `Θ·μ`（リーダー狙い） | `λ − μ`（ライフの札が手に入る）＋トリガー |
| **カウンターした**（相手手札−・ライフ＝） | `c(x)·μ` | 使わせた札 × `μ` |
| **ブロックして倒れた**（ライフ＝・手札＝・相手の体−） | `ν(B)` | 倒れた B の `ν_meas` |
| **何も起きない**（全部＝） | — | 0（無傷でブロック／通らなかった／効果で止まった） |

実現は `price_realised.state_meas` の**部品ごと**（`λ·Δライフ`・`μ·Δ手札`・`Δν_meas`・`δ·Δドン`）に割って出す。
`x`（付けたドン込みの攻撃側パワー − 対象のパワー）の帯ごとに**応答の割合**も出し、式の規則
（`c(x) > Θ` なら受ける）と比べる。**回帰しない**。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/attack_response.py --in ~/w41 --out ~/attack_response.json
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
import guard_afford as GA  # noqa: E402
from price_realised import DELTA, SC_OPP_HAND, don_stock, side_nu_meas  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_bridge import is_decision_row as TB_is_decision_row  # noqa: E402  (D-5)
from theory_bridge import add_decision_row_arg, apply_decision_row  # noqa: E402  (D-5)
import theory_order as _TO  # noqa: E402
from theory_order import (LAM, MU, PWR_EPS, SC_MY_DON, SC_MY_HAND, SC_MY_LEADER_POWER, SC_MY_LIFE,  # noqa: E402
                          SC_OPP_LEADER_POWER, SC_OPP_LIFE, SLOT_OPP_FIELD, SLOT_OWN_FIELD, THETA,
                          add_nu_mode_arg, apply_nu_mode, c_of, opp_bodies_of, own_attackers_of,
                          score_candidate, slot_power, theta_of)

RESPONSES = ("took", "countered", "took_and_countered", "blocker_died", "nothing", "other")
X_BANDS = ((-1e9, -PWR_EPS, "x<0"), (-PWR_EPS, 1000.0 + PWR_EPS, "0..1000"),
           (1000.0 + PWR_EPS, 2000.0 + PWR_EPS, "1000..2000"), (2000.0 + PWR_EPS, 1e9, ">2000"))


#: scalars の列: 相手のアクティブなドン（`don_active_price.py` と同じ・4）
SC_OPP_DON = 4


def x_band(x):
    for lo, hi, name in X_BANDS:
        if lo < x <= hi:
            return name
    return X_BANDS[-1][2]


def parts(sc, tok, sc2, tok2):
    """実現の部品（自席から見た得）。"""
    mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    return {
        "opp_life": LAM * (float(sc[SC_OPP_LIFE]) - float(sc2[SC_OPP_LIFE])),
        "opp_hand": MU * (float(sc[SC_OPP_HAND]) - float(sc2[SC_OPP_HAND])),
        "opp_body": side_nu_meas(tok, SLOT_OPP_FIELD, mlp) - side_nu_meas(tok2, SLOT_OPP_FIELD, mlp),
        "my_life": LAM * (float(sc2[SC_MY_LIFE]) - float(sc[SC_MY_LIFE])),
        "my_hand": MU * (float(sc2[SC_MY_HAND]) - float(sc[SC_MY_HAND])),
        "my_body": side_nu_meas(tok2, SLOT_OWN_FIELD, olp) - side_nu_meas(tok, SLOT_OWN_FIELD, olp),
        "don": DELTA * ((don_stock(sc2, tok2, "me") - don_stock(sc, tok, "me"))
                        - (don_stock(sc2, tok2, "opp") - don_stock(sc, tok, "opp"))),
    }


def parts_mirror(sc, tok, sc2, tok2):
    """**2 行目が相手席の視点で書かれているときの `parts`**（T113・2026-09-19）。

    記録の行は**その行を打つ席の視点**で符号化されている（`sc[0]` はその席のライフ）。
    ターンの括りを「そのターンに残っている最後の行」で閉じると、**その最後の行は相手席の行になりうる**
    （相手の応答窓・箱の commit）。そのとき列の対応だけを入れ替えれば同じ差分が読める:
    ライフ 1↔0・手札 7↔6・自場↔相手場。

    **物差し（リーダーパワー）は 1 行目のものを両辺に使う**——`parts` と同じ規約
    （区間の中で物差しを動かすと体の項が差分ではなくなる）。"""
    mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    return {
        "opp_life": LAM * (float(sc[SC_OPP_LIFE]) - float(sc2[SC_MY_LIFE])),
        "opp_hand": MU * (float(sc[SC_OPP_HAND]) - float(sc2[SC_MY_HAND])),
        "opp_body": side_nu_meas(tok, SLOT_OPP_FIELD, mlp) - side_nu_meas(tok2, SLOT_OWN_FIELD, mlp),
        "my_life": LAM * (float(sc2[SC_OPP_LIFE]) - float(sc[SC_MY_LIFE])),
        "my_hand": MU * (float(sc2[SC_OPP_HAND]) - float(sc[SC_MY_HAND])),
        "my_body": side_nu_meas(tok2, SLOT_OPP_FIELD, olp) - side_nu_meas(tok, SLOT_OWN_FIELD, olp),
        "don": DELTA * ((don_stock(sc2, tok2, "opp") - don_stock(sc, tok, "me"))
                        - (don_stock(sc2, tok2, "me") - don_stock(sc, tok, "opp"))),
    }


def classify(sc, tok, sc2, tok2):
    dl = float(sc[SC_OPP_LIFE]) - float(sc2[SC_OPP_LIFE])
    dh = float(sc[SC_OPP_HAND]) - float(sc2[SC_OPP_HAND])
    nb = sum(1 for s in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop) if float(tok[s, 18]) > 0.5)
    nb2 = sum(1 for s in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop) if float(tok2[s, 18]) > 0.5)
    took, countered = dl > 0.5, dh > 0.5
    if took and countered:
        return "took_and_countered"
    if took:
        return "took"
    if countered:
        return "countered"
    if nb2 < nb:
        return "blocker_died"
    if abs(dl) < 0.5 and abs(dh) < 0.5 and nb2 == nb:
        return "nothing"
    return "other"


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    rows_out = []
    stats = {"games": 0, "attacks": 0, "leader_target": 0, "char_target": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        by_seat = {}
        for n, i in enumerate(order):
            if TB_is_decision_row(rows, pol, L, ptr, i):
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {}
        for w, ns in by_seat.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or not TB_is_decision_row(rows, pol, L, ptr, i):
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            if move_family(sig) != "attack":
                continue
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                continue
            stats["attacks"] += 1
            sc, tok = ex["sc"][i], ex["tok"][i]
            sc2, tok2 = ex["sc"][order[j]], ex["tok"][order[j]]
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]), mode=theta_mode,
                          theta=_TO.theta_take(float(sc[SC_OPP_LIFE]), theta=theta))   # T63: 相手のライフで受ける費用
            ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp,
                   "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), "don_k": 1,
                   "attackers": own_attackers_of(tok, olp), "don_active": float(sc[SC_MY_DON]),
                   "st": _state_of(sc, ex["ci"][i], idx2cid),
                   # **T150f-2**: 見送った登場の価値（`misalloc_play`）に要る手札の card_id 列
                   "hand": _TO.hand_ids_of(ex["ci"][i], idx2cid),
                   "opp_bodies": opp_bodies_of(tok, mlp, max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), th, mu,
                                               ci_row=ex["ci"][i], idx2cid=idx2cid)}
            tl = sig[2] if len(sig) > 2 else None
            si, ti = int(pol["pol_si"][b]), int(pol["pol_ti"][b])
            kdon = max(0, int(pol["pol_k"][b]))
            v = score_candidate(sig, str(pol["pol_cid"][b]) or None,
                                (str(pol["pol_tcid"][b]) or None) if tl else None, ctx, cards,
                                src_power=slot_power(tok, si), tgt_power=slot_power(tok, ti), don_k=kdon)
            if v is None:
                continue
            sp = slot_power(tok, si)
            tp = slot_power(tok, ti)
            leader_target = (ti == 1) or tp is None
            if leader_target:
                tp = olp
                stats["leader_target"] += 1
            else:
                stats["char_target"] += 1
            x = (float(sp) if sp is not None else 0.0) + 1000.0 * kdon - float(tp)
            p = parts(sc, tok, sc2, tok2)
            rows_out.append({"leader": leader_target, "resp": classify(sc, tok, sc2, tok2),
                             "x": x, "xb": x_band(x), "price": float(v), "real": float(sum(p.values())),
                             "parts": p, "theta_says_take": bool(c_of(x) > float(th)),
                             # **T61**: 相手が応答を「選べたか」を読むための状態（攻撃時の相手の手札とアクティブなドン）
                             "opp_hand": int(round(float(sc[SC_OPP_HAND]))), "opp_don": int(round(float(sc[SC_OPP_DON]))),
                             "opp_life": int(round(float(sc[SC_OPP_LIFE]))),
                             # カウンターで切った札の枚数（手札の差分・受けた行はトリガーで増えうる）
                             "cards": float(p["opp_hand"] / MU)})
    return rows_out, stats


def _mean(rs, key):
    return round(float(np.mean([r[key] for r in rs])), 4) if rs else None


def summarise(rows):
    out = {}
    for tgt in ("leader", "char"):
        sub = [r for r in rows if r["leader"] == (tgt == "leader")]
        if len(sub) < 10:
            continue
        o = {"n": len(sub), "price": _mean(sub, "price"), "real": _mean(sub, "real"),
             "by_response": {}, "by_x": {}}
        for resp in RESPONSES:
            rs = [r for r in sub if r["resp"] == resp]
            if len(rs) < 5:
                continue
            o["by_response"][resp] = {
                "n": len(rs), "share": round(len(rs) / len(sub), 4),
                "price": _mean(rs, "price"), "real": _mean(rs, "real"),
                "parts": {k: round(float(np.mean([r["parts"][k] for r in rs])), 4) for k in rs[0]["parts"]},
                "theta_says_take_share": round(float(np.mean([r["theta_says_take"] for r in rs])), 3)}
        for _lo, _hi, name in X_BANDS:
            rs = [r for r in sub if r["xb"] == name]
            if len(rs) < 5:
                continue
            o["by_x"][name] = {"n": len(rs), "price": _mean(rs, "price"), "real": _mean(rs, "real"),
                               "response_shares": {resp: round(sum(1 for r in rs if r["resp"] == resp) / len(rs), 3)
                                                   for resp in RESPONSES},
                               "theta_says_take_share": round(float(np.mean([r["theta_says_take"] for r in rs])), 3)}
        out[tgt] = o
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_decision_row_arg(ap)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    add_nu_mode_arg(ap)
    _TO.add_surv_mode_arg(ap)
    _TO.add_cbar_mode_arg(ap)
    _TO.add_take_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_decision_row(a)
    _TO.apply_surv_mode(a)
    _TO.apply_cbar_mode(a)
    _TO.apply_take_mode(a)
    apply_nu_mode(a)
    t0 = time.time()
    rows, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    res = {"nu_mode": a.nu_mode, "stats": stats, "frozen": {"lambda": LAM, "mu": MU, "delta": DELTA, "theta": a.theta},
           "summary": summarise(rows), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
