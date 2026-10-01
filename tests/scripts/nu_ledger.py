"""**`ν` の中身を 1 つずつ実測と突き合わせる**（T44・2026-09-16・読み取り専用・記録だけ）。

`ν = 攻撃の流量 × (1 − ko_p)`（＋ブロック＋身代わり）が**中盤帯で実測の 1.4 倍・全体で 1.8 倍**（T39）。
どの因子が大きすぎるかを、ユーザ指示「一つずつ確かめてください」に従って**帯ごとに別々に測る**:

| # | 因子 | 式が置いている値 | ここで測るもの |
|---|---|---|---|
| (1) | **攻撃 1 回の価値**（`Θ·μ` の頭打ち） | `min(c(x)·μ, Θ·μ)` | 打った攻撃の**価格 対 実現**（T41 の作法・攻撃手の帯で割る） |
| (2) | **攻撃の回数**（毎ターン殴れる前提） | 1 回/ターン | ターン開始時に場に居た体のうち**そのターン実際に攻撃した割合** |
| (3) | **残りターン `R` と生存** | `R = 4.128`・`1 − ko_p` | (1)×(2)×`R`×(1 − ko_p) を組み直して実測 `ν` と比べる＝残差が (3) |

**回帰しない**。実現の側は A 層の実測（`price_realised.state_meas`）だけ。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/nu_ledger.py --in ~/w41 --out ~/nu_ledger.json
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
from price_realised import NU_MEAS, SAT_OVER_PWR, state_meas  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_bridge import is_decision_row as TB_is_decision_row  # noqa: E402  (D-5)
from theory_bridge import add_decision_row_arg, apply_decision_row  # noqa: E402  (D-5)
from theory_order import (KO_P, MU, PWR_EPS, S_CAN_ATTACK, S_IS_BLOCKER, S_IS_CHAR, S_POWER,  # noqa: E402
                          SC_MY_DON, SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE,
                          SLOT_OWN_FIELD, THETA, add_nu_mode_arg, apply_nu_mode, ko_p_of, nu_of,
                          opp_bodies_of, own_attackers_of, score_candidate, slot_power, theta_of)

#: 残りターン（実測・`nu_calib`）
R_TURNS = 4.128
BANDS = ("lt_leader", "leader_to_sat", "over_sat")


def band_of(power, opp_leader_power):
    x = float(power) - float(opp_leader_power)
    if x < -PWR_EPS:
        return "lt_leader"
    if x <= SAT_OVER_PWR + PWR_EPS:
        return "leader_to_sat"
    return "over_sat"


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    atk = {b: {"price": [], "real": []} for b in BANDS}          # (1) 攻撃 1 回
    atk["leader"] = {"price": [], "real": []}
    # `nu_formula` は定数 `R` = 4.128・`nu_formula_state` は出荷の価格と同じ状態の `R`（`clip(相手ライフ, 1, 5)`・T50 で追加）
    body = {b: {"present": 0, "attacked": 0, "can": 0, "nu_formula": [], "nu_formula_state": [], "ko_p": []}
            for b in BANDS}   # (2)
    stats = {"games": 0, "turns": 0, "atk_rows": 0}
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
            if TB_is_decision_row(rows, pol, L, ptr, i):          # 次の**判断点**で挟む（`price_realised` と同じ・T47）
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {}
        for w, ns in by_seat.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        # ターンごとに「開始時の体」と「そのターン攻撃した枠」を集める
        turn_first = {}
        turn_attacked = {}
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or not TB_is_decision_row(rows, pol, L, ptr, i):
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            key = (w, t)
            if key not in turn_first:
                turn_first[key] = (sc, tok)
                turn_attacked[key] = set()
                stats["turns"] += 1
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            fam = move_family(sig)
            if fam != "attack":
                continue
            si = int(pol["pol_si"][b])
            turn_attacked[key].add(si)
            stats["atk_rows"] += 1
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                continue                                  # ターン最後の行＝相手のターンが挟まる
            th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]), mode=theta_mode, theta=theta)
            ctx = {"theta": th, "mu": mu, "opp_leader_power": olp,
                   "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                   "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), "don_k": 1,
                   "attackers": own_attackers_of(tok, olp), "don_active": float(sc[SC_MY_DON]),
                   "st": _state_of(sc, ex["ci"][i], idx2cid),
                   "opp_bodies": opp_bodies_of(tok, float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                                               max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), th, mu,
                                               ci_row=ex["ci"][i], idx2cid=idx2cid)}
            tl = sig[2] if len(sig) > 2 else None
            v = score_candidate(sig, str(pol["pol_cid"][b]) or None,
                                (str(pol["pol_tcid"][b]) or None) if tl else None, ctx, cards,
                                src_power=slot_power(tok, si), tgt_power=slot_power(tok, pol["pol_ti"][b]),
                                don_k=pol["pol_k"][b])
            if v is None:
                continue
            i2 = order[j]
            real = state_meas(ex["sc"][i2], ex["tok"][i2]) - state_meas(sc, tok)
            sp = slot_power(tok, si)
            bk = "leader" if si == 0 else band_of(sp if sp is not None else 0.0, olp)
            atk[bk]["price"].append(float(v)); atk[bk]["real"].append(float(real))
        # (2) 攻撃の回数——ターン開始時に居た体ごと
        for key, (sc, tok) in turn_first.items():
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
                if float(tok[s, S_IS_CHAR]) <= 0.5:
                    continue
                pw = float(tok[s, S_POWER]) * 1e4
                bk = band_of(pw, olp)
                body[bk]["present"] += 1
                body[bk]["can"] += int(float(tok[s, S_CAN_ATTACK]) > 0.5)
                body[bk]["attacked"] += int(s in turn_attacked[key])
                blk = float(tok[s, S_IS_BLOCKER]) > 0.5
                body[bk]["nu_formula"].append(nu_of(pw, olp, R_TURNS, theta, mu, is_blocker=blk, my_leader_power=mlp))
                body[bk]["nu_formula_state"].append(nu_of(pw, olp, max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), theta, mu,
                                                          is_blocker=blk, my_leader_power=mlp))
                body[bk]["ko_p"].append(ko_p_of(pw))
    return atk, body, stats


def _ci(vals):
    v = np.asarray(vals, np.float64)
    if len(v) < 2:
        return [None, None]
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return [round(float(v.mean() - 1.96 * se), 4), round(float(v.mean() + 1.96 * se), 4)]


def summarise(atk, body):
    out = {"per_attack": {}, "per_body_turn": {}, "compose": {}}
    for bk, d in atk.items():
        if len(d["price"]) < 10:
            continue
        p, q = np.array(d["price"]), np.array(d["real"])
        out["per_attack"][bk] = {"n": len(p), "price": round(float(p.mean()), 4), "real": round(float(q.mean()), 4),
                                 "real_ci95": _ci(q), "ratio": round(float(q.mean() / p.mean()), 3) if p.mean() > 0 else None}
    for bk, d in body.items():
        if d["present"] < 10:
            continue
        out["per_body_turn"][bk] = {"body_turns": d["present"], "attack_rate": round(d["attacked"] / d["present"], 4),
                                    "can_attack_rate": round(d["can"] / d["present"], 4),
                                    "nu_formula_mean": round(float(np.mean(d["nu_formula"])), 4),
                                    "nu_formula_state_mean": round(float(np.mean(d.get("nu_formula_state") or d["nu_formula"])), 4),
                                    "ko_p_band_mean": round(float(np.mean(d["ko_p"])), 4)}
    # (3) 組み直し: 実測の因子 (1)×(2)×R×(1−ko_p) 対 実測 ν 対 式 ν
    for bk in BANDS:
        a, b = out["per_attack"].get(bk), out["per_body_turn"].get(bk)
        if not a or not b:
            continue
        flow = a["real"] * b["attack_rate"]
        comp_const = flow * R_TURNS * (1.0 - KO_P)
        comp_curve = flow * R_TURNS * (1.0 - b["ko_p_band_mean"])
        out["compose"][bk] = {"per_attack_real": a["real"], "attack_rate": b["attack_rate"],
                              "flow_per_turn": round(flow, 4), "R": R_TURNS,
                              "composed_nu_const_kop": round(comp_const, 4),
                              "composed_nu_curve_kop": round(comp_curve, 4),
                              "nu_formula": b["nu_formula_mean"], "nu_measured": NU_MEAS[bk],
                              "formula_over_measured": round(b["nu_formula_mean"] / NU_MEAS[bk], 3),
                              # **出荷の価格と同じ `R`（状態）で引いた式**——定数 `R` の比は状態の `R` より高く出る（T50）
                              "formula_state_over_measured": round(b["nu_formula_state_mean"] / NU_MEAS[bk], 3),
                              "composed_over_measured": round(comp_const / NU_MEAS[bk], 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_decision_row_arg(ap)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_decision_row(a)
    apply_nu_mode(a)
    t0 = time.time()
    atk, body, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    res = {"nu_mode": a.nu_mode, "stats": stats, "frozen": {"R": R_TURNS, "ko_p": KO_P, "nu_meas": NU_MEAS},
           "summary": summarise(atk, body), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
