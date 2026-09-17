"""**交点の橋**——累積した損害の線 `F(t)` と、相手の耐久（しきい値）の線 `Θ(t)` の交点で勝敗を読む（T52・2026-09-16・読み取り専用）。

ユーザ提案「足し上げた価値の関数と、しきい値の関数の交点が勝敗の橋になるんじゃないか」。従来の橋（`theory_bridge`）は
「価格の累積 `ΔG` が勝敗と相関するか」を見たが、価格は帳簿の単位（平均の傾きで換算した勝率）なので「傾き 1」は要らない。
勝敗は**帳簿がしきい値に届くか・どちらが先か**で決まる:

```
F_w(t)  = 席 w が相手に与えた損害の累積（価格の単位: λ×削ったライフ + μ×切らせた札 + ν×倒した体）
Θ_w(t)  = 相手を倒すのに残っている量 = λ·L_opp + μ·H_opp + Σν_meas(相手のアクティブなブロッカー)   （T51 の耐久を価格に）
傾き    = 損害を積む速さ（1 自席ターンあたり）——`hist`＝これまでの実現の平均／`theory`＝今の盤面の攻撃手の価格の和
τ_w     = Θ_w / 傾き            あと何ターンで届くか
勝者    = τ が小さい席（自席が手番なので同数なら自席）・終局 = min τ
```

**当てはめない**（λ・μ・ν・c̄ は写し）。**回帰しない**。測るのは (a) 予測勝者の的中率・(b) 予測終局と実際のずれ（偏り・σ）・
(c) `D = τ_opp − τ_me` の帯ごとの実勝率・(d) 帳簿の単位の検算＝勝った席の終局までの損害 `F` と開始時の `Θ` の比（1 なら単位が合う）。
盤面の時計（T51: 的中 0.59／0.62・σ 1.7〜1.8）と同じ物差しで比べる。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/crossing_bridge.py --in ~/w41 --out ~/crossing.json
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
from attack_response import parts  # noqa: E402
from clock_calib import D_BINS, d_bin  # noqa: E402
from price_realised import nu_meas_of  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (LAM, MU, PWR_EPS, S_IS_BLOCKER, S_IS_CHAR, S_IS_REST, SC_MY_DON, SC_MY_HAND, SLOT_OWN_FIELD,  # noqa: E402
                          SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE,
                          SLOT_OPP_FIELD, THETA, add_nu_mode_arg, apply_nu_mode, attack_value_don,
                          opp_bodies_of, own_attackers_of, score_candidate, slot_power, theta_of)

SLOPES = ("hist", "theory")
SLOPE_FLOOR = 1e-3


#: **耐久の手札項の数え方**（T76・2026-09-17・§0.6 の残り 2）: `count`＝旧（`μ × 枚数`）／`quality`＝**札 1 枚あたりの実価格**
#: （その席の手札の札ごとの `max(ΔH_play, ΔG_guard)`＝T67 の値の平均）を `μ` の代わりに掛ける。**新定数ゼロ**（T64〜T67 の器の写し）。
#: **手札の中身はその席の行にしか無い**（記録に相手の手札は無い）ので、`quality` は守る席の直近の自席ターン開始の行から作る
#: ＝**攻める席が持たない情報を使う**（「本当の耐久なら当たるのか」を先に確かめる段・推定器は次の T）。
#: `play`＝**出す側だけ**（`ΔH_play`）／`guard`＝**守る側だけ**（`ΔG_guard`＝切って止められる分）＝どちらの半分が耐久なのかの切り分け（T76）。
THETA_HAND_MODES = ("count", "quality", "play", "guard")
THETA_HAND_MODE = "count"


def set_theta_hand_mode(mode):
    global THETA_HAND_MODE
    if mode not in THETA_HAND_MODES:
        raise ValueError("theta hand mode は %s のどれか" % (THETA_HAND_MODES,))
    THETA_HAND_MODE = mode


def hand_price_mean(sc, tok_row, ci_row, idx2cid, cards, mu=MU, part="dtotal"):
    """**その席の手札 1 枚あたりの価格**（T76）＝自分の手札の札ごとの `max(ΔH_play, ΔG_guard)`（T67）の平均。
    `part="dh"` なら出す側だけ（守る備えを外した切り分け）。手札が空なら `μ`（旧の数え方）。
    来る攻撃・受ける損・ドンの枠はその席の行から採る（`hand_plan.search_context`）。"""
    import hand_plan as HP
    ctx = HP.search_context(sc, tok_row, ci_row, idx2cid, cards, None)
    items = ctx["hand_items"]
    if not items:
        return float(mu)
    vals = [float(HP.card_deltas(items[:k] + items[k + 1:], it, ctx["caps"], ctx["xs"], ctx["take"])[part])
            for k, it in enumerate(items)]
    return float(np.mean(vals))


def threshold(sc, tok, lam=LAM, mu=MU, g_hand=None):
    """相手の耐久を価格で: `λ·L_opp + g·H_opp + Σν_meas(アクティブなブロッカー)`。
    `g` は手札 1 枚あたりの価格（`None`＝`μ`＝旧・T76 の `quality` では相手の手札から作った実価格）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    g = float(mu if g_hand is None else g_hand)
    th = lam * float(sc[SC_OPP_LIFE]) + g * float(sc[SC_OPP_HAND])
    for s in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5 and float(tok[s, S_IS_BLOCKER]) > 0.5 and float(tok[s, S_IS_REST]) <= 0.5:
            th += nu_meas_of(slot_power(tok, s) or 0.0, mlp)
    return float(th)


def threshold_of_me(sc, tok, lam=LAM, mu=MU, g_hand=None):
    """**自分の耐久**（相手から見たしきい値）: `λ·L_me + g·H_me + Σν_meas(自分のアクティブなブロッカー)`（T75・`g` は T76）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    g = float(mu if g_hand is None else g_hand)
    th = lam * float(sc[SC_MY_LIFE]) + g * float(sc[SC_MY_HAND])
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok[s, S_IS_CHAR]) > 0.5 and float(tok[s, S_IS_BLOCKER]) > 0.5 and float(tok[s, S_IS_REST]) <= 0.5:
            th += nu_meas_of(slot_power(tok, s) or 0.0, olp)
    return float(th)


#: **損害の輪郭の正本**（T75）: `tests/fixtures/harm_profile.json`＝`{"real": [...], "syn": [...]}`（自席ターン番号 j ごとの損害の平均・
#: `harm_profile` の出力・実測の表）。`cross`＝測る記録と別のセットの輪郭を使う（実デッキの記録には合成の輪郭・逆も）。
HARM_PROFILE_PATH = os.path.join(_ROOT, "tests", "fixtures", "harm_profile.json")
HARM_PROFILE_NAMES = ("cross", "real", "syn")
_PROFILES = {}


def load_harm_profiles(path=None):
    """輪郭の表を読む（無ければ空）。"""
    path = path or HARM_PROFILE_PATH
    if path not in _PROFILES:
        try:
            with open(path, encoding="utf-8") as fh:
                _PROFILES[path] = json.load(fh)
        except (OSError, ValueError):
            _PROFILES[path] = {}
    return _PROFILES[path]


def record_kind(dirs):
    """記録のセットの種類（`meta_n_record.json` の `decks`: `user` → `real`・それ以外 → `syn`）。判らなければ `None`。"""
    kinds = set()
    for d in dirs or ():
        try:
            with open(os.path.join(os.path.expanduser(d), "meta_n_record.json"), encoding="utf-8") as fh:
                kinds.add("real" if str(json.load(fh).get("decks")) == "user" else "syn")
        except (OSError, ValueError):
            pass
    return kinds.pop() if len(kinds) == 1 else None


def profile_for(dirs, name="cross", path=None):
    """測る記録に使う輪郭（`cross`＝別のセット・`real`／`syn`＝指定）。無ければ `None`。"""
    prof = load_harm_profiles(path)
    if name == "cross":
        kind = record_kind(dirs)
        name = {"real": "syn", "syn": "real"}.get(kind or "", None)
    if not name:
        return None
    v = prof.get(name)
    return [float(x) for x in v] if v else None


def curve_d_of_row(sc, tok, j, prof, g_hand_of_opp=None, g_hand_of_me=None):
    """**交点の近さ `D`**（T75）＝両席の到達ターンの差 `τ_opp − τ_me`（正なら自分が先に届く）。
    `τ_me` は自分が相手の耐久 `Θ_me` に、`τ_opp` は相手が自分の耐久 `Θ_opp` に、同じ輪郭で積んで届くターン数（相手も同じ自席ターン番号 `j` と置く）。
    `g_hand_of_opp`／`g_hand_of_me` は**その席の手札**の 1 枚あたりの価格（T76・`None` なら `μ`）。1 行からは自分の手札しか読めないので、
    線形の橋では `g_hand_of_me` だけが入る（相手側は `μ` のまま＝非対称・報告で明示する）。"""
    th_me = threshold(sc, tok, g_hand=g_hand_of_opp)
    th_opp = threshold_of_me(sc, tok, g_hand=g_hand_of_me)
    tau_me = tau_from_profile(th_me, int(j), prof)
    tau_opp = tau_from_profile(th_opp, int(j), prof)
    return {"d": float(tau_opp - tau_me), "tau_me": tau_me, "tau_opp": tau_opp, "theta_me": th_me, "theta_opp": th_opp}


def own_turn_index(t):
    """記録のターン番号 `t`（両席で数える・1 始まり）→ 自席ターン番号 `j`（0 始まり）。"""
    return max(0, (int(t) - 1) // 2)


def theory_slope(tok, opp_leader_power, theta=THETA, mu=MU):
    """今の盤面の攻撃手（リーダー＋殴れる体）がリーダーを殴る価格の和＝理論の「1 ターンに積む損害」。"""
    return float(sum(attack_value_don(float(opp_leader_power) + x, opp_leader_power, True, theta, mu)
                     for x in own_attackers_of(tok, opp_leader_power)))


def harm_of(p):
    """実現の部品（`attack_response.parts`）のうち**相手に与えた損害**だけ（相手ライフ・相手手札・相手の体）。"""
    return float(p["opp_life"] + p["opp_hand"] + p["opp_body"])


def predict(theta_me, theta_opp, slope_me, slope_opp):
    """交点までのターン数と予測勝者（自席が手番なので同数なら自席）。"""
    tau_me = theta_me / max(SLOPE_FLOOR, slope_me)
    tau_opp = theta_opp / max(SLOPE_FLOOR, slope_opp)
    return tau_me, tau_opp, (tau_me <= tau_opp)


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    rows_out = []
    ledger = []            # (d) 単位の検算: 勝った席の F_end 対 Θ_start
    turn_harm = []         # 自席ターン番号 j ごとの損害（損害の輪郭＝加速を測る材料）
    stats = {"games": 0, "turns": 0, "rows_bracketed": 0, "theta_hand": THETA_HAND_MODE,
             "g_sum": 0.0, "g_n": 0, "g_fallback": 0,
             "g_win_sum": 0.0, "g_win_n": 0, "g_lose_sum": 0.0, "g_lose_n": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        by_seat = {}
        for n, i in enumerate(order):
            if int(rows["kind"][i]) == 0:
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {}
        for w, ns in by_seat.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        turn_start = {}       # (w, t) -> (sc, tok, ci)
        turn_last = {}        # (w, t) -> その席のそのターン最後の行（T76: 出した後の手札で 1 枚あたりの価格を測る）
        turn_seq = {0: [], 1: []}
        harm = {}             # (w, t) -> 実現の損害の和（そのターン）
        priced = {}           # (w, t) -> 攻撃の価格の和（そのターン）
        z_of = {}
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
            turn_last[(w, t)] = (sc, tok, ex["ci"][i])
            if (w, t) not in turn_start:
                turn_start[(w, t)] = (sc, tok, ex["ci"][i])
                turn_seq[w].append(t)
                harm[(w, t)] = 0.0; priced[(w, t)] = 0.0
                stats["turns"] += 1
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                continue
            i2 = order[j]
            harm[(w, t)] += harm_of(parts(sc, tok, ex["sc"][i2], ex["tok"][i2]))
            stats["rows_bracketed"] += 1
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            if move_family(sig) != "attack":
                continue
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]), mode=theta_mode, theta=theta)
            r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
            ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp, "r_turns": r, "don_k": 1,
                   "attackers": own_attackers_of(tok, olp), "don_active": float(sc[SC_MY_DON]),
                   "st": _state_of(sc, ex["ci"][i], idx2cid),
                   "opp_bodies": opp_bodies_of(tok, mlp, r, th, mu, ci_row=ex["ci"][i], idx2cid=idx2cid)}
            tl = sig[2] if len(sig) > 2 else None
            v = score_candidate(sig, str(pol["pol_cid"][b]) or None, (str(pol["pol_tcid"][b]) or None) if tl else None,
                                ctx, cards, src_power=slot_power(tok, int(pol["pol_si"][b])),
                                tgt_power=slot_power(tok, int(pol["pol_ti"][b])), don_k=int(pol["pol_k"][b]))
            if v is not None:
                priced[(w, t)] += float(v)
        if len(z_of) < 2 or not turn_seq[0] or not turn_seq[1]:
            continue
        # **T76**: 席ごとの手札 1 枚あたりの価格（自席の行からしか読めない）。守る席の直近の自席ターン開始の値を耐久に使う。
        g_self = {}
        if THETA_HAND_MODE != "count":
            part = {"quality": "dtotal", "play": "dh", "guard": "dg"}[THETA_HAND_MODE]
            for w in (0, 1):
                for t in turn_seq[w]:
                    sc, tok, ci = turn_last.get((w, t), turn_start[(w, t)])   # 出した後の手札（ターン最後の行）
                    g_self[(w, t)] = hand_price_mean(sc, tok, ci, idx2cid, cards, mu, part)

        def g_for(defender, t):
            """守る席の手札 1 枚あたりの価格（その席の直近の自席ターン開始・無ければ `None`＝`μ`）。
            **勝った席と負けた席で分けて集計する**（T76 の切り分け: 勝つ席ほど手札を場に出していて 1 枚あたりが安い、を確かめる）。"""
            if THETA_HAND_MODE == "count":
                return None
            prev = [tt for tt in turn_seq[defender] if tt <= t]
            g = g_self.get((defender, prev[-1])) if prev else None
            if g is None:
                stats["g_fallback"] += 1
            else:
                stats["g_sum"] += float(g); stats["g_n"] += 1
                side = "win" if z_of.get(defender, 0.0) > 0.5 else "lose"
                stats["g_%s_sum" % side] += float(g); stats["g_%s_n" % side] += 1
            return g

        # 席ごとの自席ターン開始点で、両席の τ を出す（相手は直前の自分のターン開始の値）
        per_seat = {}
        for w in (0, 1):
            ts = turn_seq[w]
            f_real = 0.0
            for j, t in enumerate(ts):
                sc, tok, _ci = turn_start[(w, t)]
                olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                th_w = threshold(sc, tok, g_hand=g_for(1 - w, t))
                slope_hist = (f_real / j) if j > 0 else None
                slope_theory = theory_slope(tok, olp, theta, mu)
                per_seat[(w, t)] = {"theta": th_w, "slope_hist": slope_hist, "slope_theory": slope_theory,
                                    "f_real": f_real, "t_left": len(ts) - j, "j": j}
                turn_harm.append({"j": j, "harm": harm.get((w, t), 0.0), "slope_theory": slope_theory})
                f_real += harm.get((w, t), 0.0)
            won = z_of[w] > 0.5
            if won and ts:
                th0 = per_seat[(w, ts[0])]["theta"]
                ledger.append({"F_end": f_real, "theta_start": th0,
                               "F_priced_end": sum(priced.get((w, t), 0.0) for t in ts)})
        for w in (0, 1):
            ts = turn_seq[w]; ts_o = turn_seq[1 - w]
            won = z_of[w] > 0.5
            for j, t in enumerate(ts):
                me = per_seat[(w, t)]
                prev_o = [tt for tt in ts_o if tt < t]
                if not prev_o:
                    continue                                   # 相手がまだ 1 ターンも打っていない
                op = per_seat[(1 - w, prev_o[-1])]
                t_opp_act = sum(1 for tt in ts_o if tt > t)
                rec = {"who": w, "won": won, "t_me_act": me["t_left"], "t_opp_act": t_opp_act,
                       "theta_me": me["theta"], "theta_opp": op["theta"], "j_me": me["j"], "j_opp": op["j"],
                       "slope_theory_me": me["slope_theory"], "slope_theory_opp": op["slope_theory"]}
                for sv in SLOPES:
                    s_me = me["slope_" + sv] if me["slope_" + sv] is not None else me["slope_theory"]
                    s_op = op["slope_" + sv] if op["slope_" + sv] is not None else op["slope_theory"]
                    tau_me, tau_opp, pred = predict(me["theta"], op["theta"], s_me, s_op)
                    rec["tau_me_" + sv] = tau_me; rec["tau_opp_" + sv] = tau_opp; rec["pred_" + sv] = pred
                rows_out.append(rec)
    return rows_out, ledger, stats, turn_harm


def harm_profile(turn_harm, j_max=12, min_n=20):
    """**損害の輪郭**＝自席ターン番号 `j` ごとの損害の平均（と理論の傾きの平均）。標本が薄い先は最後の値を伸ばす。"""
    prof, prof_th = [], []
    last, last_th = 0.0, 0.0
    for j in range(j_max + 1):
        hs = [r["harm"] for r in turn_harm if r["j"] == j]
        ts = [r["slope_theory"] for r in turn_harm if r["j"] == j]
        if len(hs) >= min_n:
            last, last_th = float(np.mean(hs)), float(np.mean(ts))
        prof.append(last); prof_th.append(last_th)
    return prof, prof_th


def tau_from_profile(theta, j, prof, scale=1.0):
    """輪郭に沿って損害を積み、`Θ` に届くまでのターン数（端数は比例配分・輪郭の先は最後の値）。"""
    acc = 0.0
    for k in range(200):
        h = prof[min(j + k, len(prof) - 1)] * scale
        if h <= SLOPE_FLOOR:
            h = SLOPE_FLOOR
        if acc + h >= theta:
            return k + (theta - acc) / h
        acc += h
    return 200.0


def summarise(rows_out, ledger, turn_harm=None):
    out = {"n": len(rows_out), "by_slope": {}}
    prof, prof_th = harm_profile(turn_harm or [])
    if turn_harm:
        out["harm_profile"] = {"harm_by_turn": [round(x, 4) for x in prof],
                               "theory_slope_by_turn": [round(x, 4) for x in prof_th]}
        # 輪郭の変種: `curve`＝平均の輪郭のまま／`curve_scaled`＝今の盤面の理論の傾きで輪郭を伸縮
        for r in rows_out:
            for sv, scale_me, scale_op in (
                    ("curve", 1.0, 1.0),
                    ("curve_scaled",
                     r["slope_theory_me"] / max(SLOPE_FLOOR, prof_th[min(r["j_me"], len(prof_th) - 1)]),
                     r["slope_theory_opp"] / max(SLOPE_FLOOR, prof_th[min(r["j_opp"], len(prof_th) - 1)]))):
                tm = tau_from_profile(r["theta_me"], r["j_me"], prof, scale_me)
                to = tau_from_profile(r["theta_opp"], r["j_opp"], prof, scale_op)
                r["tau_me_" + sv] = tm; r["tau_opp_" + sv] = to; r["pred_" + sv] = (tm <= to)
    if ledger:
        fe = np.array([r["F_end"] for r in ledger]); th0 = np.array([r["theta_start"] for r in ledger])
        fp = np.array([r["F_priced_end"] for r in ledger])
        out["ledger"] = {"winners": len(ledger), "F_end_mean": round(float(fe.mean()), 4),
                         "theta_start_mean": round(float(th0.mean()), 4),
                         "F_end_over_theta_start": round(float(fe.mean() / max(1e-9, th0.mean())), 3),
                         "F_priced_end_mean": round(float(fp.mean()), 4),
                         "F_priced_over_F_real": round(float(fp.mean() / max(1e-9, fe.mean())), 3)}
    z = np.array([1.0 if r["won"] else 0.0 for r in rows_out], float)
    for sv in SLOPES + (("curve", "curve_scaled") if turn_harm else ()):
        if not rows_out or ("pred_" + sv) not in rows_out[0]:
            continue
        pred = np.array([r["pred_" + sv] for r in rows_out], bool)
        res = []
        for r in rows_out:
            if r["won"]:
                res.append(min(r["tau_me_" + sv], 30.0) - r["t_me_act"])
            else:
                res.append(min(r["tau_opp_" + sv], 30.0) - r["t_opp_act"])
        res = np.array(res, float)
        d = np.array([min(r["tau_opp_" + sv], 30.0) - min(r["tau_me_" + sv], 30.0) for r in rows_out], float)
        o = {"sign_accuracy": round(float((pred == (z > 0.5)).mean()), 4),
             "bias": round(float(res.mean()), 3), "sigma_T": round(float(res.std()), 3),
             "mae": round(float(np.abs(res).mean()), 3),
             "tau_me_median": round(float(np.median([min(r["tau_me_" + sv], 30.0) for r in rows_out])), 3),
             "win_by_D": {}}
        for lo, hi, name in D_BINS:
            m = (d > lo) & (d <= hi)
            if m.sum() >= 20:
                o["win_by_D"][name] = {"n": int(m.sum()), "win_rate": round(float(z[m].mean()), 4)}
        out["by_slope"][sv] = o
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--theta-hand", default=THETA_HAND_MODE, choices=THETA_HAND_MODES,
                    help="**T76** 耐久の手札項: `count`（既定・`μ × 枚数`）／`quality`（札ごとの `max(ΔH, ΔG)` の平均を掛ける）")
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)
    t0 = time.time()
    set_theta_hand_mode(a.theta_hand)
    rows_out, ledger, stats, turn_harm = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    if stats.get("g_n"):
        stats["g_mean"] = round(stats["g_sum"] / stats["g_n"], 4)
    for side in ("win", "lose"):
        if stats.get("g_%s_n" % side):
            stats["g_%s_mean" % side] = round(stats["g_%s_sum" % side] / stats["g_%s_n" % side], 4)
    res = {"nu_mode": a.nu_mode, "stats": stats, "frozen": {"lambda": LAM, "mu": MU, "theta": a.theta},
           "summary": summarise(rows_out, ledger, turn_harm), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
