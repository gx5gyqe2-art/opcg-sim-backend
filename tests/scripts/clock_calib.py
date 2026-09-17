"""**時計の較正**——「2 本の導火線の競争」の式を、記録の事実と 4 本の方程式で突き合わせる（T51・2026-09-16・読み取り専用）。

```
耐久 E = L + H/c̄ + B          あと何回殴られたら死ぬか（ライフ・手札で止められる回数・ブロッカー）
速度 A = 相手リーダーを越える攻撃手の数（1 ターンに通る攻撃）
時計 T_me = E_opp / A_me,  T_opp = E_me / A_opp,  D = T_opp − T_me,  W = Φ(D/σ_D)
```

| # | 方程式 | 左辺（記録の事実） | 右辺（定義済みの量） | 決まるもの |
|---|---|---|---|---|
| (1) しきい値 | 死ぬまでに実際に止めた回数（通った攻撃 − 失ったライフ） | `H/c̄ + B`（静的）／`(H + 残りターン)/c̄ + B`（引きを足す） | 手札 1 枚が何回ぶんか・ブロッカーの重み |
| (2) 速度 | 自席ターンに実際に通した攻撃数（リーダー狙い・x ≥ 0） | ターン開始時の盤面の `A_me` | `A` の数え方 |
| (3) 時間 | 倒した側の実際の残りターン数（記録の `t_left`） | `E/A` | **この推定器のぶれ `σ_T`**（借り物でなく自前） |
| (4) 勝率 | `D` の帯ごとの実勝率 | `Φ(D/σ_D)`・`σ_D = √2·σ_T` | 形の検算 |

**当てはめない**——係数は既存の実測（`c̄` 1.514）の写しのまま、左辺と右辺を並べるだけ。**引きを足す形**（毎ターン 1 枚引く＝
1 ターンに `1/c̄` 回ぶん耐久が増える＝`T = E/(A − 1/c̄)`）は式の形の候補として並べる（連立で解ける・ユーザ提案）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/clock_calib.py --in ~/w41 --out ~/clock_calib.json
"""
import argparse
import json
import math
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
from theory_bridge import POL_COLS, ROW_COLS, _extra, move_family  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_order import (CBAR, PWR_EPS, S_IS_CHAR, S_POWER, SC_MY_DON, SC_MY_HAND, SC_MY_LEADER_POWER,  # noqa: E402
                          SC_MY_LIFE, SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE, SLOT_OWN_FIELD,
                          count_blockers, incoming_x, opp_chars_of, slot_power)

VARIANTS = ("static", "draws")
D_BINS = ((-1e9, -3.0, "<-3"), (-3.0, -1.0, "-3..-1"), (-1.0, 1.0, "-1..1"), (1.0, 3.0, "1..3"), (3.0, 1e9, ">3"))


def phi_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def d_bin(d):
    for lo, hi, name in D_BINS:
        if lo < d <= hi:
            return name
    return D_BINS[-1][2]


def endurance(life, hand, blockers, cbar=CBAR, draws_turns=0.0):
    """耐久 `E = L + (H + 引き)/c̄ + B`。"""
    return float(life) + (float(hand) + float(draws_turns)) / cbar + float(blockers)


def clock(e, a, variant="static", cbar=CBAR):
    """`T = E/A`（静的）／`T = E/(A − 1/c̄)`（毎ターンの引きで耐久が伸びる・分母の床 0.25）。"""
    a = max(1.0, float(a))
    if variant == "draws":
        return float(e) / max(0.25, a - 1.0 / cbar)
    return float(e) / a


def board_a_me(tok, opp_leader_power):
    """自分のリーダー＋場のキャラのうち相手リーダーを越えるもの（レストの旗は見ない＝次のターンは殴れる）。"""
    n = 1 if (slot_power(tok, 0) or 0.0) >= opp_leader_power - PWR_EPS else 0
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5 and (slot_power(tok, s) or 0.0) >= opp_leader_power - PWR_EPS:
            n += 1
    return n


def row_inputs(sc, tok):
    sc = np.asarray(sc); tok = np.asarray(tok)
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    a_board = board_a_me(tok, olp)
    return {"L_me": float(sc[SC_MY_LIFE]), "L_opp": float(sc[SC_OPP_LIFE]),
            "H_me": float(sc[SC_MY_HAND]), "H_opp": float(sc[SC_OPP_HAND]),
            "B_me": count_blockers(tok), "B_opp": sum(1 for _p, blk in opp_chars_of(tok) if blk),
            "A_me": a_board, "A_opp": sum(1 for x in incoming_x(tok) if x >= -PWR_EPS),
            # **T78**（T77 の横展開・手札の 2 つの価値）: 耐久は**切れる札だけ**・速さは**今出せる通る体**を足す
            "H_me_cut": float(TO.hand_cuttable(tok)),
            "A_me_hand": a_board + TO.hand_attackers(tok, olp, float(sc[SC_MY_DON]))}


def collect(dirs, limit_games=0):
    rows_out = []          # 自席ターン開始行ごと
    rate_rows = []         # (2) 速度: 盤面の A_me 対 実際に通した数
    thr_rows = []          # (1) しきい値: 負けた側の「死ぬまでに止めた回数」対 式
    stats = {"games": 0, "turns": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        turn_start = {}      # (w, t) -> (sc, tok)
        turn_seq = {0: [], 1: []}
        z_of = {}
        passing = {}         # (w, t) -> リーダー狙いで x ≥ 0 の攻撃数（w が打った）
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            z = float(rows["z"][i])
            if z != 0.0:
                z_of[w] = 1.0 if z > 0 else 0.0
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            if (w, t) not in turn_start:
                turn_start[(w, t)] = (sc, tok)
                turn_seq[w].append(t)
                passing.setdefault((w, t), 0)
                stats["turns"] += 1
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            if move_family(sig) != "attack":
                continue
            ti = int(pol["pol_ti"][b])
            tp = slot_power(tok, ti)
            if not (ti == 1 or tp is None):
                continue                                   # キャラ狙いは時計を進めない
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            sp = slot_power(tok, int(pol["pol_si"][b])) or 0.0
            kdon = max(0, int(pol["pol_k"][b]))
            if sp + 1000.0 * kdon - olp >= -PWR_EPS:
                passing[(w, t)] += 1
        if len(z_of) < 2 or not turn_seq[0] or not turn_seq[1]:
            continue
        for w in (0, 1):
            ts = turn_seq[w]
            ts_opp = turn_seq[1 - w]
            won = z_of[w] > 0.5
            for j, t in enumerate(ts):
                sc, tok = turn_start[(w, t)]
                q = row_inputs(sc, tok)
                # (2) 速度（**T78**: 次の自席ターンの実際も並べる——手札から出した体は召喚酔いで次のターンから殴る）
                t_next = ts[j + 1] if j + 1 < len(ts) else None
                rate_rows.append({"A_board": q["A_me"], "A_actual": passing.get((w, t), 0),
                                  "A_hand": q["A_me_hand"],
                                  "A_next": (None if t_next is None else passing.get((w, t_next), 0))})
                # (3) 時間: 倒した側の実際の残りターン
                t_me_act = len(ts) - j                                         # 自分の残り自席ターン（今を含む）
                t_opp_act = sum(1 for tt in ts_opp if tt > t)                  # 相手の残りターン
                rec = {"who": w, "won": won, "t_me_act": t_me_act, "t_opp_act": t_opp_act, **q}
                for v in VARIANTS:
                    e_opp = endurance(q["L_opp"], q["H_opp"], q["B_opp"])
                    e_me = endurance(q["L_me"], q["H_me"], q["B_me"])
                    rec["T_me_" + v] = clock(e_opp, q["A_me"], v)
                    rec["T_opp_" + v] = clock(e_me, q["A_opp"], v)
                rows_out.append(rec)
                # (1) しきい値（負けた側だけ・今から死ぬまで）
                if not won:
                    hits = sum(passing.get((1 - w, tt), 0) for tt in ts_opp if tt > t)
                    stops = hits - q["L_me"]                                   # 通った攻撃のうちライフを減らさなかった数
                    thr_rows.append({"L": q["L_me"], "H": q["H_me"], "B": q["B_me"], "hits": hits,
                                     "stops": max(0.0, stops), "turns_left": t_opp_act,
                                     "H_cut": q["H_me_cut"],
                                     "formula_static": q["H_me"] / CBAR + q["B_me"],
                                     "formula_draws": (q["H_me"] + t_opp_act) / CBAR + q["B_me"],
                                     # **T78**: 耐久に入るのは**切れる札だけ**（T77 の `cuttable`）。
                                     # 引く札も**同じ割合だけ切れる**と置くのが筋（割合はこの行の手札から測る＝新定数ゼロ）
                                     "formula_cut": q["H_me_cut"] / CBAR + q["B_me"],
                                     "formula_cut_draws": (q["H_me_cut"] + t_opp_act) / CBAR + q["B_me"],
                                     "formula_cut_draws_frac": (q["H_me_cut"] + (q["H_me_cut"] / max(1.0, q["H_me"])) * t_opp_act)
                                     / CBAR + q["B_me"]})
    return rows_out, rate_rows, thr_rows, stats


def _mean(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def summarise(rows_out, rate_rows, thr_rows):
    out = {}
    # (1) しきい値
    if thr_rows:
        out["threshold"] = {"n": len(thr_rows),
                            "stops_actual": _mean([r["stops"] for r in thr_rows]),
                            "hits_actual": _mean([r["hits"] for r in thr_rows]),
                            "life_at_row": _mean([r["L"] for r in thr_rows]),
                            "hand_at_row": _mean([r["H"] for r in thr_rows]),
                            "blockers_at_row": _mean([r["B"] for r in thr_rows]),
                            "formula_static": _mean([r["formula_static"] for r in thr_rows]),
                            "formula_draws": _mean([r["formula_draws"] for r in thr_rows]),
                            # **T78**: 切れる札だけで数えた耐久（実測の `stops` に近いのはどちらか）。
                            # **古い行（T78 前の作りの `thr_rows`）にも耐えるように欄が無ければ飛ばす**
                            "hand_cut_at_row": _mean([r["H_cut"] for r in thr_rows if "H_cut" in r]),
                            "formula_cut": _mean([r["formula_cut"] for r in thr_rows if "formula_cut" in r]),
                            "formula_cut_draws": _mean([r["formula_cut_draws"] for r in thr_rows if "formula_cut_draws" in r]),
                            "formula_cut_draws_frac": _mean([r["formula_cut_draws_frac"] for r in thr_rows if "formula_cut_draws_frac" in r]),
                            "mae_static": _mean([abs(r["formula_static"] - r["stops"]) for r in thr_rows]),
                            "mae_draws": _mean([abs(r["formula_draws"] - r["stops"]) for r in thr_rows]),
                            "mae_cut": _mean([abs(r["formula_cut"] - r["stops"]) for r in thr_rows if "formula_cut" in r]),
                            "mae_cut_draws": _mean([abs(r["formula_cut_draws"] - r["stops"]) for r in thr_rows if "formula_cut_draws" in r]),
                            "mae_cut_draws_frac": _mean([abs(r["formula_cut_draws_frac"] - r["stops"])
                                                         for r in thr_rows if "formula_cut_draws_frac" in r]),
                            "stops_per_hand_card": round(float(sum(r["stops"] for r in thr_rows)
                                                               / max(1.0, sum(r["H"] for r in thr_rows))), 4),
                            "implied_cbar_static": round(float(sum(r["H"] for r in thr_rows)
                                                               / max(1e-9, sum(max(0.0, r["stops"] - r["B"]) for r in thr_rows))), 3)}
    # (2) 速度
    if rate_rows:
        ab = np.array([r["A_board"] for r in rate_rows], float); aa = np.array([r["A_actual"] for r in rate_rows], float)
        # **T78**: 次の自席ターンの実際に対して、盤面だけの `A` と手札を足した `A` のどちらが近いか
        nxt = [r for r in rate_rows if r.get("A_next") is not None]
        if nxt:
            an = np.array([r["A_next"] for r in nxt], float)
            abn = np.array([r["A_board"] for r in nxt], float)
            ahn = np.array([r["A_hand"] for r in nxt], float)
            out["rate_next"] = {"n": len(nxt), "A_next_mean": round(float(an.mean()), 4),
                                "A_board_mean": round(float(abn.mean()), 4), "A_hand_mean": round(float(ahn.mean()), 4),
                                "mae_board": round(float(np.abs(abn - an).mean()), 4),
                                "mae_hand": round(float(np.abs(ahn - an).mean()), 4),
                                "bias_board": round(float((abn - an).mean()), 4),
                                "bias_hand": round(float((ahn - an).mean()), 4)}
        out["rate"] = {"n": len(rate_rows), "A_board_mean": round(float(ab.mean()), 4), "A_actual_mean": round(float(aa.mean()), 4),
                       "actual_over_board": round(float(aa.mean() / max(1e-9, ab.mean())), 3),
                       "by_board": {int(k): {"n": int((ab == k).sum()), "actual_mean": round(float(aa[ab == k].mean()), 3)}
                                    for k in sorted(set(ab.astype(int))) if (ab == k).sum() >= 20}}
    # (3) 時間・(4) 勝率
    out["time"] = {}
    out["win_by_D"] = {}
    for v in VARIANTS:
        res = []          # 倒した側の時計の残差（推定 − 実際）
        for r in rows_out:
            if r["won"]:
                res.append(r["T_me_" + v] - r["t_me_act"])
            else:
                res.append(r["T_opp_" + v] - r["t_opp_act"])
        res = np.array(res, float)
        if len(res) < 2:
            continue                                   # 行が無ければ時間・勝率の表は出さない
        sigma_t = float(res.std())
        sigma_d = math.sqrt(2.0) * sigma_t
        d = np.array([r["T_opp_" + v] - r["T_me_" + v] for r in rows_out], float)
        z = np.array([1.0 if r["won"] else 0.0 for r in rows_out], float)
        out["time"][v] = {"n": len(res), "bias": round(float(res.mean()), 3), "sigma_T": round(sigma_t, 3),
                          "mae": round(float(np.abs(res).mean()), 3), "sigma_D": round(sigma_d, 3),
                          "sign_accuracy": round(float(((d > 0) == (z > 0.5)).mean()), 4),
                          # 倒した側だけでなく、全行の「対局の残り」対 min(T)
                          "game_left_actual": _mean([min(r["t_me_act"], r["t_opp_act"] + 0.5) for r in rows_out]),
                          "min_T_hat": _mean([min(r["T_me_" + v], r["T_opp_" + v]) for r in rows_out])}
        bins = {}
        for lo, hi, name in D_BINS:
            m = (d > lo) & (d <= hi)
            if m.sum() >= 20:
                bins[name] = {"n": int(m.sum()), "win_rate": round(float(z[m].mean()), 4),
                              "phi_measured_sigma": round(float(np.mean([phi_cdf(x / sigma_d) for x in d[m]])), 4),
                              "phi_borrowed_sigma": round(float(np.mean([phi_cdf(x / math.sqrt(2.0)) for x in d[m]])), 4)}
        out["win_by_D"][v] = bins
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows_out, rate_rows, thr_rows, stats = collect(a.src, a.limit_games)
    res = {"stats": stats, "frozen": {"cbar": CBAR}, "summary": summarise(rows_out, rate_rows, thr_rows),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
