"""**`ν` の在庫の台帳**——式の項ごとに、記録から測った「その体が実際に積んだもの」と並べる（T50・2026-09-16・読み取り専用）。

T49 で攻撃 1 回の価格は実現に届いた（飽和超え 1.05）のに、**在庫 `ν` は帯の実測の 1.5〜1.7 倍**（中盤／飽和超え）。
式は `ν = (lead·R + 選択肢 + ブロック + 身代わり) × (1 − ko_p)`＝「毎ターン同じ価値で `R` ターン殴り、生存で 1 回割り引く」。
どの因子が在庫を膨らませているかを、**体 1 つを追いかけて**測る（ユーザ指示「(1) ν の在庫の水準」）:

| 因子 | 式が置いている値 | ここで測るもの |
|---|---|---|
| **`R`**（残りターン） | `clip(相手ライフ, 1, 5)`（平均 4.128） | その判断点から対局が終わるまでの**自席ターン数**（今のターンを含む） |
| **生存** | `1 − ko_p(P)` を 1 回 | 同じ枠に同じカードが **k ターン後も居る**割合 `surv(k)`（k = 1..4）→ 生きて迎えたターン数 `Σ_k surv(k)` |
| **流れ** | `lead`（毎ターン） | その体が打った攻撃 1 回の実現（T44 の作法）× 実際に打った回数 |
| **在庫** | 上の積 | **その体が生きている間に打った攻撃の実現の和**（`stock_real`）＝式の `ν` と帯の実測 `ν` の両方と比べる |

体の同一性は**枠 × `card_idx`**で追う（記録に uuid が無い・P8）。同じ枠に同じカードが出直した場合は生きていると誤読する（稀）。
ブロックと身代わりの実現は体に帰属できないので在庫には入れない（式の側の 2 項は分けて出す）。**回帰しない**。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/nu_stock.py --in ~/w41 --out ~/nu_stock.json
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
import theory_order as TO  # noqa: E402
from nu_ledger import BANDS, band_of  # noqa: E402
from price_realised import NU_MEAS, state_meas  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (MU, S_IS_BLOCKER, S_IS_CHAR, S_POWER, SC_MY_DON, SC_MY_LEADER_POWER,  # noqa: E402
                          SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE, SLOT_OWN_FIELD, THETA,
                          add_nu_mode_arg, apply_nu_mode, attack_value_don, ko_p_of, option_value,
                          opp_bodies_of, own_attackers_of, score_candidate, shield_of, slot_power, theta_of)

R_CONST = 4.128
K_MAX = 4


def r_of(opp_life):
    """式が置いている残りターン `R`（橋・`price_realised` と同じ）。"""
    return max(1.0, min(5.0, float(opp_life)))


def formula_terms(power, opp_leader_power, my_leader_power, r, is_blocker, theta=THETA, mu=MU):
    """式 `ν` の項ごとの値（`nu_of(mode="pair")` と同じ材料・同じ和）。"""
    kp = ko_p_of(power)
    lead = attack_value_don(power, opp_leader_power, True, theta, mu)
    opt = option_value(power, opp_leader_power, r, theta, mu, my_leader_power, kp)
    block = (TO.BLOCK_P_BLOCKER if is_blocker else 0.0) * theta * mu
    shield = shield_of(power, opp_leader_power)
    st = TO.surv_turns(r, kp)                     # `once` なら R・`geo` なら Σ(1−ko_p)^t（T60）
    if TO.SURV_MODE == "geo":
        nu = lead * st + opt + (block + shield) * (1.0 - kp)
        weight = st                               # 式の生存の重み＝働くターン数
    else:
        nu = (lead * r + opt + block + shield) * (1.0 - kp)
        weight = (1.0 - kp) * r
    return {"lead": lead, "lead_R": lead * st, "option": opt, "block": block, "shield": shield, "ko_p": kp,
            "surv_weight": weight, "nu": nu}


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    bodies = []            # 体 1 つ（ターン開始時）ごとの記録
    r_by_life = {}         # 相手ライフ → 実際の残り自席ターン
    stats = {"games": 0, "turns": 0, "bodies": 0, "attacks_priced": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        # 判断点（kind 0）の並びと「次の判断点」（同じ席・同じターン）
        by_seat = {}
        for n, i in enumerate(order):
            if int(rows["kind"][i]) == 0:
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {}
        for w, ns in by_seat.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        # 席ごとの自席ターンの並び（開始行）と、そのターンに枠ごとが打った攻撃の実現
        turn_start = {}          # (w, t) -> (sc, tok, ci)
        turn_seq = {}            # w -> [t, ...]
        atk_real = {}            # (w, t, slot) -> [実現, ...]
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            if (w, t) not in turn_start:
                turn_start[(w, t)] = (sc, tok, np.asarray(ex["ci"][i]))
                turn_seq.setdefault(w, []).append(t)
                stats["turns"] += 1
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            if move_family(sig) != "attack":
                continue
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                continue
            si = int(pol["pol_si"][b])
            i2 = order[j]
            real = float(state_meas(ex["sc"][i2], ex["tok"][i2]) - state_meas(sc, tok))
            atk_real.setdefault((w, t, si), []).append(real)
            stats["attacks_priced"] += 1
        # 体を追う
        for w, ts in turn_seq.items():
            for j, t in enumerate(ts):
                sc, tok, ci = turn_start[(w, t)]
                olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
                r_form = r_of(sc[SC_OPP_LIFE])
                r_act = len(ts) - j                       # 今のターンを含む残り自席ターン
                r_by_life.setdefault(int(round(float(sc[SC_OPP_LIFE]))), []).append(r_act)
                th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]), mode=theta_mode, theta=theta)
                for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
                    if float(tok[s, S_IS_CHAR]) <= 0.5:
                        continue
                    pw = float(tok[s, S_POWER]) * 1e4
                    cid_idx = int(ci[s])
                    alive = [1]
                    for kk in range(1, K_MAX + 1):
                        if j + kk >= len(ts):
                            alive.append(None)            # 対局が終わった（死んだのではない）
                            continue
                        _sc2, tok2, ci2 = turn_start[(w, ts[j + kk])]
                        alive.append(int(float(tok2[s, S_IS_CHAR]) > 0.5 and int(ci2[s]) == cid_idx))
                    # 生きている間に打った攻撃の実現（今のターンから、死ぬまで）
                    stock, n_atk, turns_alive = 0.0, 0, 0
                    for kk in range(0, len(ts) - j):
                        if kk > 0:
                            _sc2, tok2, ci2 = turn_start[(w, ts[j + kk])]
                            if not (float(tok2[s, S_IS_CHAR]) > 0.5 and int(ci2[s]) == cid_idx):
                                break
                        turns_alive += 1
                        got = atk_real.get((w, ts[j + kk], s), [])
                        stock += sum(got); n_atk += len(got)
                    terms = formula_terms(pw, olp, mlp, r_form, float(tok[s, S_IS_BLOCKER]) > 0.5, th, mu)
                    bodies.append({"band": band_of(pw, olp), "opp_life": int(round(float(sc[SC_OPP_LIFE]))),
                                   "r_form": r_form, "r_act": r_act, "alive": alive, "turns_alive": turns_alive,
                                   "n_atk": n_atk, "stock_real": stock, **terms})
                    stats["bodies"] += 1
    return bodies, r_by_life, stats


def _m(rows, key, nd=4):
    return round(float(np.mean([r[key] for r in rows])), nd) if rows else None


def summarise(bodies, r_by_life):
    out = {"by_band": {}, "r_by_opp_life": {}}
    for bk in BANDS:
        rs = [b for b in bodies if b["band"] == bk]
        if len(rs) < 20:
            continue
        surv = {}
        for kk in range(1, K_MAX + 1):
            obs = [b["alive"][kk] for b in rs if b["alive"][kk] is not None]
            surv[kk] = round(float(np.mean(obs)), 4) if obs else None
        turns_alive_expected = 1.0 + sum(v for v in surv.values() if v is not None)
        o = {"n": len(rs),
             "R_formula": _m(rs, "r_form"), "R_actual": _m(rs, "r_act"),
             "surv": surv, "turns_alive_from_surv": round(turns_alive_expected, 3),
             "turns_alive_mean": _m(rs, "turns_alive"), "attacks_per_body": _m(rs, "n_atk"),
             "attacks_per_turn_alive": (round(float(sum(b["n_atk"] for b in rs) / max(1, sum(b["turns_alive"] for b in rs))), 4)),
             "real_per_attack": (round(float(sum(b["stock_real"] for b in rs) / max(1, sum(b["n_atk"] for b in rs))), 4)),
             "stock_real": _m(rs, "stock_real"),
             "formula": {k: _m(rs, k) for k in ("lead", "lead_R", "option", "block", "shield", "ko_p", "nu")},
             "nu_measured": NU_MEAS[bk]}
        f = o["formula"]
        # 式の生存の重み（(1−ko_p)·R）対 実測の生きて迎えたターン数
        # 式の生存の重み（`once`: `(1−ko_p)·R`・`geo`: `Σ(1−ko_p)^t`＝`formula_terms` の `surv_weight`・T60）
        o["turn_weight_formula"] = (round(_m(rs, "surv_weight"), 3) if all("surv_weight" in b for b in rs)
                                    else (round(f["lead_R"] / f["lead"] * (1.0 - f["ko_p"]), 3) if f["lead"] else None))
        o["turn_weight_measured"] = o["turns_alive_mean"]
        # 組み直し: 流れ（実現/攻撃 × 攻撃/生きたターン）× 生きたターン
        o["composed_stock"] = round(o["real_per_attack"] * o["attacks_per_turn_alive"] * o["turns_alive_mean"], 4)
        o["composed_stock_rate1"] = round(o["real_per_attack"] * o["turns_alive_mean"], 4)
        o["ratios"] = {"formula_over_measured": round(f["nu"] / NU_MEAS[bk], 3),
                       "stock_real_over_measured": round(o["stock_real"] / NU_MEAS[bk], 3),
                       "formula_over_stock_real": round(f["nu"] / o["stock_real"], 3) if o["stock_real"] else None,
                       "leadR_over_stock_real": round(f["lead_R"] * (1 - f["ko_p"]) / o["stock_real"], 3) if o["stock_real"] else None}
        out["by_band"][bk] = o
    for life, rs in sorted(r_by_life.items()):
        if len(rs) >= 20:
            out["r_by_opp_life"][life] = {"n": len(rs), "R_formula": r_of(life), "R_actual": round(float(np.mean(rs)), 3)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    add_nu_mode_arg(ap)
    TO.add_surv_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)
    TO.apply_surv_mode(a)
    t0 = time.time()
    bodies, r_by_life, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "option_mode": TO.OPTION_MODE, "stats": stats,
           "frozen": {"theta": a.theta, "mu": MU, "nu_meas": NU_MEAS, "R_const": R_CONST},
           "summary": summarise(bodies, r_by_life), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
