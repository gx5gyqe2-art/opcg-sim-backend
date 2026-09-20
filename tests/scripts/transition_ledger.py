#!/usr/bin/env python3
"""**値段の付いていない遷移を数える**（T123・`H`・2026-09-20・`game_theory.md` §17.9.0 の残り 1）。

## 問い

紐付けの法則（§17.9）は **`Σ ΔW = W_end − W₀`**（telescoping）1 本で水準と帰属を結ぶ。
**ところがこの等式は閉じていない**——帳簿は**自席の手**しか積まないが、`W` は**全部の遷移**で動く:

* **引き**（毎ターン 1 枚・規則）
* **【トリガー】とライフ→手札**（受けた 1 枚が手札に入る＝`h = 0.89` の出どころ）
* **ターンの境目**（ドン +2・アンタップ・召喚酔いの解除）——**誰も「手を打っていない」のに時計が動く**
* 効果の遅延解決・KO の後始末

**閉じない限り、判別と較正は原理的に両立しない**（T122 §5 の論証）:
正しい `ΔW` は決着したら黙るので、局ごとの和を勝敗で採点する指標では不利になる。
`Σ ΔW = z − W₀` が厳密なら、静かな帳簿でも合計はちょうど勝敗を当てる。

**だから本器の問いは「どれを値付けすれば閉じるか」**であって、「閉じるか」ではない（閉じるのは恒等式）。

## 測り方（当てはめゼロ・恒等式の配分だけ）

1. 全部の判断行を**記録の順**に並べ、**席 0 の視点**で `W` を出す。
2. 隣り合う行の差 `W_{i+1} − W_i` を**そのまま**取る（**和は `W_last − W_first` に厳密に一致する**＝検算）。
3. その差を 2 つに割る: **`priced`**＝行 `i` で選ばれた手が説明する分（`W(手を当てた状態) − W(状態)`）と
   **`residual`**＝残り。
4. `residual` を **5 つの軸（`Θ_me`・`Θ_opp`・`A_me`・`A_opp`・自席ターン番号 `j`）へシャープレイ値で配る**
   ——**順序に依らず、和が厳密に差に一致する**唯一の配り方（部分集合 2⁵ = 32 通りの評価で出る）。
5. **ターンの境目を跨ぐ差**と**同じターンの中の差**を分けて数える（規則の出どころが違う）。

**新定数ゼロ**（配分は恒等式・`σ_rel` は既測）。**「当てはめて残差を説明する係数」は 1 つも置かない**。

使い方:

    python tests/scripts/transition_ledger.py --in <records_dir> [--games N] [--d-mode curve|clock] [--json out.json]
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import crossing_bridge as CB  # noqa: E402
import guard_afford as GA  # noqa: E402
import kappa_vector as KV  # noqa: E402
import relative_ledger as RL  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import SLOT_OPP_FIELD as TO_SLOT_OPP_FIELD  # noqa: E402
from theory_order import (MU, SC_MY_DON, SC_MY_HAND, SC_MY_LEADER_POWER, SC_MY_LIFE,  # noqa: E402
                          SC_OPP_HAND, SC_OPP_LEADER_POWER, SC_OPP_LIFE, THETA, opp_bodies_of,
                          own_attackers_of, score_candidate, slot_power, theta_of)

#: 配分する軸（`j` も軸に入れる——輪郭の読み出し位置が進むのは**誰の手でもない遷移**）
AXES5 = ("th_me", "th_opp", "a_me", "a_opp", "j")
#: `residual` を規則の出どころで分ける区分
CAUSES = ("turn_boundary", "same_turn")

#: **ターンの境目に値段を付けるか**（T124・残差の 1/3・密度は最悪）。
#:
#: **境目で起きることは全部規則である**（誰も手を打っていないのに時計が動く）:
#:
#: | 規則 | 動く軸 | 既にある量 |
#: |---|---|---|
#: | **引き 1 枚** | 引いた席の `Θ`（手札の項） | `g`＝切れる札 1 枚あたりの価格（T76／T99） |
#: | **アンタップ** | 起きる席の `Θ`（体の項・レストのブロッカーが的に戻る） | `resting_blocker_term`（T96） |
#: | **ドン +2 ＋ 召喚酔いの解除** | 起きる席の `A` | 規則のドンの列の段差 `sched[j+1] − sched[j]`（T114） |
#:
#: **新定数ゼロ**——3 つとも既に測って在る量で、新しい係数は 1 つも置かない。
#: `off`＝境目を値付けしない（T123 の測り方）／`rules`＝上の 3 つを当てる。
#: `rules`＝3 つ全部／**`draw_untap`＝推薦形**（`don` を外す）／`draw`・`untap`・`don`＝内訳。
#:
#: **`don` は外す**（T124 → **T125 で理由を訂正**）。`clock` の読みで悪化する
#: （境目の残差 0.2286 → 0.2651・`priced_search` ではなく `priced_share` が 1 を超える＝行き過ぎ）。
#:
#: **「二重計上」は誤った診断だった**（T125・`2026-09-20_boundary_don_retraction.md`）——実測すると
#: **列の段差は実際の `ΔA` の 0.35 倍しかなく、相関は `−0.27`・符号一致は 30.8%**。
#: **足しすぎではなく、別の量を足していた**。ドン +2 が起きるのは**ターンが渡ってきた席の開始時**で
#: **タイミングの方は正しい**（ユーザ指摘 2026-09-20）。
#: **患部は反対側**——`sched` は**積み上がる歩きの率 `R_j`**（盤面 ＋ 在庫·[j≥2] ＋ 流入·(j−1)）で、
#: 状態の `A`（`seat_slope`＝盤面 ＋ 流入）とは**別の対象**なので、一方の段差が他方の跳びを予測しない。
#: **代わりの量も 2 つ測って両方落ちた**（総額 2.61 倍・相関 +0.60／召喚酔い 3.13 倍・相関 +0.32）＝
#: **境目に速さの項は当てられていない**。`curve` では `A` が時計に入らないので `rules` と `draw_untap` は同じ値になる。
BOUNDARY_MODES = ("off", "rules", "draw_untap", "draw", "untap", "don")
BOUNDARY_MODE = "off"


def set_boundary_mode(name):
    global BOUNDARY_MODE
    if name not in BOUNDARY_MODES:
        raise ValueError("BOUNDARY_MODE は %s のどれか（%r）" % (BOUNDARY_MODES, name))
    BOUNDARY_MODE = name
    return BOUNDARY_MODE


def boundary_dx(tok, ci_row, idx2cid, cards, mlp, g_next, sched, j_next, next_is_seat0):
    """**ターンの境目で規則が動かす分**（席 0 の視点の `Δx`・新定数ゼロ）。

    `tok`／`ci_row` は**境目の直前の行**（前のターンの最後の行）で、**次に動く席はその行の相手**。
    だから起きる側の枠は `SLOT_OPP_FIELD`・リーダーのパワーは `mlp`（T96 と同じ渡し方）。

    * **引き 1 枚** … 引いた席の `Θ` に `g`（切れる札 1 枚あたりの価格）。
      **手札の項は `cuttable_forced` では枚数に線形でない**（守りの強制で上限が付く）ので、
      これは**一次の近似**である——そう書いておく（上限に当たっている行では過大になる）。
    * **アンタップ** … レストのブロッカーが的に戻る（`resting_blocker_term`）。
    * **ドン +2 ＋ 召喚酔いの解除** … 規則のドンの列の段差。`sched` が無ければ 0（`RATE_DON_MODE=off`）。
    """
    use_draw = BOUNDARY_MODE in ("rules", "draw_untap", "draw")
    use_untap = BOUNDARY_MODE in ("rules", "draw_untap", "untap")
    use_don = BOUNDARY_MODE in ("rules", "don")
    d_th = (float(g_next or 0.0) if use_draw else 0.0)
    if use_untap:
        d_th += float(CB.resting_blocker_term(tok, TO_SLOT_OPP_FIELD, mlp, ci_row=ci_row,
                                              idx2cid=idx2cid, cards=cards))
    d_a = 0.0
    if sched and use_don:
        j = max(0, int(j_next))
        lo = sched[min(max(0, j - 1), len(sched) - 1)]
        hi = sched[min(j, len(sched) - 1)]
        d_a = float(hi) - float(lo)
    if next_is_seat0:
        return {"th_me": d_th, "a_me": d_a}
    return {"th_opp": d_th, "a_opp": d_a}


def _swap_state(st):
    """席 1 の視点の状態を**席 0 の視点**に写す（`Θ` と `A` を入れ替え・`j` は共通）。"""
    th_me, th_opp, a_me, a_opp, j = st
    return (th_opp, th_me, a_opp, a_me, j)


def _swap_dx(dx):
    """席 1 の視点の `Δx` を**席 0 の視点**に写す（軸の名前を入れ替えるだけ・符号は変えない）。"""
    m = {"th_me": "th_opp", "th_opp": "th_me", "a_me": "a_opp", "a_opp": "a_me"}
    return {m.get(k, k): v for k, v in dx.items()}


def _mix(st0, st1, keys):
    """`keys` の軸だけ `st1` の値にした中間状態（シャープレイの部分集合の評価に使う）。"""
    out = list(st0)
    for i, name in enumerate(AXES5):
        if name in keys:
            out[i] = st1[i]
    return tuple(out)


def shapley(st0, st1, prof, sigma_rel):
    """**`W(st1) − W(st0)` を 5 つの軸へシャープレイ値で配る**（順序に依らず・和は厳密に差）。

    部分集合 32 通りの `W` を 1 度だけ作って再利用する（`5! = 120` 通りの順序を全部数えるのと同値）。
    **これは配分であって説明ではない**——「どの軸が動いたか」を言うだけで、係数は 1 つも当てはめていない。"""
    names = AXES5
    vals = {}
    for r in range(len(names) + 1):
        for comb in itertools.combinations(names, r):
            key = frozenset(comb)
            vals[key] = RL.w_of(*RL.clocks_of(_mix(st0, st1, key), prof), sigma_rel=sigma_rel)
    out = {n: 0.0 for n in names}
    n = len(names)
    fact = [math.factorial(k) for k in range(n + 1)]
    for r in range(n):
        for comb in itertools.combinations(names, r):
            key = frozenset(comb)
            # そのサイズの部分集合が順序の数え上げで持つ重み（標準のシャープレイの係数）
            wgt = fact[r] * fact[n - r - 1] / fact[n]
            for name in names:
                if name in key:
                    continue
                out[name] += wgt * (vals[key | {name}] - vals[key])
    return out


def _priority(acc):
    """**残差を 3 つに割る**（全部 `|·|` の割合・手当てが別々なので分ける）。

    同じターンの中の残差を**攻撃の行**と**攻撃でない行**に分け、そこへ**ターンの境目**を並べる。3 つで 1 になる。
    **`curve` の読みでは「攻撃でない行」の価格は厳密に 0**（速さの軸が無い）だが、`clock` では価格が付く
    ——だから名前は「値段が付いていない」ではなく「攻撃でない」にしてある（読みによって意味が変わる欄にしない）。"""
    tot = max(1e-12, acc["resid_abs"])
    st_attack = acc["fam_abs"].get("attack", 0.0)
    st_other = sum(v for f, v in acc["fam_abs"].items() if f != "attack")
    return {"attack_rows": round(st_attack / tot, 4),
            "nonattack_rows": round(st_other / tot, 4),
            "turn_boundary": round(acc["by_cause_abs"]["turn_boundary"] / tot, 4)}


def collect(dirs, limit_games=0, theta=THETA, mu=MU):
    """記録を 1 度読んで **`W` の差を `priced` と `residual` に割り、`residual` を軸と区分へ配る**。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    prof = CB.profile_for(dirs)
    if KV.D_MODE == "curve" and not prof:
        raise ValueError("D_MODE=curve なのに損害の輪郭が引けない（%s）" % (dirs,))
    sr = CB.sigma_rel_for(dirs, slope="curve")
    if sr is None:
        raise ValueError("σ_rel が引けない＝黙って別の物差しに落とさない（T118 の規約）")
    TO.set_sigma_rel(sr)
    acc = {"gap": 0.0, "gap_abs": 0.0, "priced": 0.0, "priced_abs": 0.0,
           "resid": 0.0, "resid_abs": 0.0,
           "by_axis": {a: 0.0 for a in AXES5}, "by_axis_abs": {a: 0.0 for a in AXES5},
           "by_cause": {c: 0.0 for c in CAUSES}, "by_cause_abs": {c: 0.0 for c in CAUSES},
           # **原因 × 軸**（ターンの境目で動く軸と、同じターンの中で動く軸は規則の出どころが違う）
           "cross_abs": {c: {a: 0.0 for a in AXES5} for c in CAUSES},
           "cross_n": {c: 0 for c in CAUSES},
           # **行の手の型ごとの残差**——`same_turn` の残差は
           # 「相手が窓で答えた（攻撃の行に偏る）」と「価格が合っていない（出す・効果の行に偏る）」の
           # **2 つが混ざっている**ので、型で割らないと切り分けられない（P8 が開くまでこれが最良の分離）。
           "fam_abs": {}, "fam_n": {}, "fam_priced_abs": {}}
    stats = {"games": 0, "rows": 0, "gaps": 0, "identity_max_err": 0.0,
             "w_first_sum": 0.0, "w_last_sum": 0.0, "z_sum": 0.0, "terminal_sum": 0.0,
             "terminal_abs_sum": 0.0}
    games = 0
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        # 席ごとの速さ（その席の自席ターンの**最初の行**から・`crossing_bridge` の `turn_start` と同じ規約）
        rate_at, g_at, sched_at = {}, {}, {}
        for i in idx:
            if int(r["kind"][i]) != 0:
                continue
            w, t = int(r["who"][i]), int(r["turn"][i])
            if PL.is_own_turn(w, t) and (w, t) not in rate_at:
                rate_at[(w, t)] = KV.rate_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i],
                                                 idx2cid, cards, theta, mu)
                g_at[(w, t)] = KV.g_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid, cards)
                sched_at[(w, t)] = None
                if BOUNDARY_MODE in ("rules", "don") and CB.RATE_DON_MODE != "off":
                    _sc, _tok, _ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
                    _olp = float(np.asarray(_sc)[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                    sched_at[(w, t)] = CB.seat_slope_sched(_sc, _tok, _ci, idx2cid, cards, _olp,
                                                          theta, mu, jmax=int(CB.RACE_CAP))

        def _latest(w, t):
            ts = [tt for (ww, tt) in rate_at if ww == w and tt <= t]
            return (rate_at[(w, max(ts))], g_at[(w, max(ts))]) if ts else None

        seq = []            # (席 0 視点の状態, 手の Δx〔席 0 視点〕, ターン番号, 手の型, 境目の材料)
        z_of = {}
        for i in idx:
            z = float(r["z"][i])
            if z != 0.0:
                z_of[int(r["who"][i])] = 1.0 if z > 0 else 0.0
            if int(r["kind"][i]) != 0:
                continue
            w, t = int(r["who"][i]), int(r["turn"][i])
            if not PL.is_own_turn(w, t):
                continue
            me, op = _latest(w, t), _latest(1 - w, t)
            if me is None or op is None:
                continue
            sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
            st = KV.state_of_row(sc, tok, me[0], op[0], CB.own_turn_index(t),
                                 g_me=me[1], g_opp=op[1])
            stats["rows"] += 1
            # その行で選ばれた手の `Δx`（無ければ空＝値段の付かない行）
            dx = {}; fam = "none"
            k = int(L[i]); ch = int(r["pol_chosen"][i])
            if k >= 1 and 0 <= ch < k:
                b = int(ptr[i]) + ch
                sig = json.loads(pol["pol_sig"][b])
                fam = move_family(sig)
                rt = max(1.0, min(5.0, float(np.asarray(sc)[SC_OPP_LIFE])))
                th = theta_of(tok, float(np.asarray(sc)[SC_MY_LIFE]),
                              float(np.asarray(sc)[SC_MY_DON]), mode="const", theta=theta)
                olp = float(np.asarray(sc)[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                mlp = float(np.asarray(sc)[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
                ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp,
                       "r_turns": rt, "don_k": 1, "attackers": own_attackers_of(tok, olp),
                       "don_active": float(np.asarray(sc)[SC_MY_DON]),
                       "st": _state_of(sc, ci, idx2cid),
                       "opp_bodies": opp_bodies_of(tok, mlp, rt, th, mu, ci_row=ci, idx2cid=idx2cid)}
                tl = sig[2] if len(sig) > 2 else None
                v = score_candidate(sig, str(pol["pol_cid"][b]) or None,
                                    (str(pol["pol_tcid"][b]) or None) if tl else None, ctx, cards,
                                    src_power=slot_power(tok, int(pol["pol_si"][b])),
                                    tgt_power=slot_power(tok, int(pol["pol_ti"][b])),
                                    don_k=int(pol["pol_k"][b]))
                if v is not None:
                    dx = KV.axis_of_move(fam, float(v), sig, str(pol["pol_cid"][b]) or None, cards,
                                         sc, tok, olp, rt, don_k=int(pol["pol_k"][b]))
            if w == 1:
                st, dx = _swap_state(st), _swap_dx(dx)
            seq.append((st, dx, t, fam,
                        (tok, ci, float(np.asarray(sc)[SC_MY_LEADER_POWER]) * 1e4 or 5000.0, w)))
        if len(seq) < 2 or len(z_of) < 2:
            continue
        w_first = RL.w_of(*RL.clocks_of(seq[0][0], prof), sigma_rel=sr)
        w_last = RL.w_of(*RL.clocks_of(seq[-1][0], prof), sigma_rel=sr)
        stats["w_first_sum"] += w_first; stats["w_last_sum"] += w_last
        stats["z_sum"] += z_of.get(0, 0.0)
        stats["terminal_sum"] += z_of.get(0, 0.0) - w_last
        stats["terminal_abs_sum"] += abs(z_of.get(0, 0.0) - w_last)
        run = 0.0
        for (st0, dx0, t0, fam0, bi0), (st1, _dx1, t1, _f1, _b1) in zip(seq, seq[1:]):
            w0 = RL.w_of(*RL.clocks_of(st0, prof), sigma_rel=sr)
            w1 = RL.w_of(*RL.clocks_of(st1, prof), sigma_rel=sr)
            gap = w1 - w0
            run += gap
            dx_use = dict(dx0)
            if BOUNDARY_MODE != "off" and t1 != t0:
                # **境目は「打ち手のいない手」として同じ行に相乗りさせる**（規則の 3 つ）
                tok0, ci0, mlp0, w_row = bi0
                w_next = 1 - w_row                       # 次に動く席＝この行の相手
                ts = [tt for (ww, tt) in rate_at if ww == w_next and tt > t0]
                key_prev = [tt for (ww, tt) in rate_at if ww == w_next and tt <= t0]
                sched = sched_at.get((w_next, max(key_prev))) if key_prev else None
                j_next = CB.own_turn_index(min(ts)) if ts else CB.own_turn_index(t0) + 1
                g_next = g_at.get((w_next, min(ts))) if ts else None
                bdx = boundary_dx(tok0, ci0, idx2cid, cards, mlp0, g_next, sched, j_next,
                                  next_is_seat0=(w_next == 0))
                for kk3, vv3 in bdx.items():
                    dx_use[kk3] = dx_use.get(kk3, 0.0) + vv3
            priced = (RL.w_of(*RL.clocks_of(KV.apply_dx(st0, dx_use), prof), sigma_rel=sr) - w0) if dx_use else 0.0
            resid = gap - priced
            stats["gaps"] += 1
            acc["gap"] += gap; acc["gap_abs"] += abs(gap)
            acc["priced"] += priced; acc["priced_abs"] += abs(priced)
            acc["resid"] += resid; acc["resid_abs"] += abs(resid)
            cause = "turn_boundary" if t1 != t0 else "same_turn"
            acc["by_cause"][cause] += resid; acc["by_cause_abs"][cause] += abs(resid)
            if cause == "same_turn":       # 境目を跨ぐ差は「手の型」の話ではないので混ぜない
                acc["fam_abs"][fam0] = acc["fam_abs"].get(fam0, 0.0) + abs(resid)
                acc["fam_priced_abs"][fam0] = acc["fam_priced_abs"].get(fam0, 0.0) + abs(priced)
                acc["fam_n"][fam0] = acc["fam_n"].get(fam0, 0) + 1
            # **残りを 5 つの軸へ配る**（`priced` が説明した分を引いた状態から `st1` まで）
            base = KV.apply_dx(st0, dx_use) if dx_use else st0
            w_base = RL.w_of(*RL.clocks_of(base, prof), sigma_rel=sr)
            sh = shapley(base, st1, prof, sr)
            acc["cross_n"][cause] += 1
            for a, val in sh.items():
                acc["by_axis"][a] += val; acc["by_axis_abs"][a] += abs(val)
                acc["cross_abs"][cause][a] += abs(val)
            # **配分の恒等式**: シャープレイ値の和は厳密に `W(st1) − W(base)` に一致する
            stats["identity_max_err"] = max(stats["identity_max_err"],
                                            abs(sum(sh.values()) - (w1 - w_base)))
        stats["identity_max_err"] = max(stats["identity_max_err"], abs(run - (w_last - w_first)))
    n = max(1, stats["gaps"]); ng = max(1, stats["games"])
    tot_abs = max(1e-12, acc["gap_abs"])
    out = {"games": stats["games"], "rows": stats["rows"], "gaps": stats["gaps"],
           "d_mode": KV.D_MODE, "boundary_mode": BOUNDARY_MODE, "sigma_rel": round(sr, 4),
           # **恒等式の検算**（配分の和が差に一致すること・telescoping が閉じること）
           "identity_max_abs_error": round(stats["identity_max_err"], 12),
           "w_first_mean": round(stats["w_first_sum"] / ng, 4),
           "w_last_mean": round(stats["w_last_sum"] / ng, 4),
           "z_mean": round(stats["z_sum"] / ng, 4),
           # **末端の隙間**（最後の行の `W` と実際の勝敗の差＝どの遷移にも載らない分）
           "terminal_mean": round(stats["terminal_sum"] / ng, 4),
           "terminal_abs_mean": round(stats["terminal_abs_sum"] / ng, 4),
           # **本題**: 差のうち手が説明する分と残り（絶対値の割合で読む＝符号の打ち消しを避ける）
           "priced_share": round(acc["priced_abs"] / tot_abs, 4),
           "resid_share": round(acc["resid_abs"] / tot_abs, 4),
           "priced_mean": round(acc["priced"] / n, 6), "resid_mean": round(acc["resid"] / n, 6),
           "by_axis_share": {a: round(acc["by_axis_abs"][a] / max(1e-12, sum(acc["by_axis_abs"].values())), 4)
                             for a in AXES5},
           "by_axis_mean": {a: round(acc["by_axis"][a] / n, 6) for a in AXES5},
           "by_cause_share": {c: round(acc["by_cause_abs"][c] / max(1e-12, acc["resid_abs"]), 4)
                              for c in CAUSES},
           "by_cause_mean": {c: round(acc["by_cause"][c] / n, 6) for c in CAUSES},
           "by_cause_gaps": {c: acc["cross_n"][c] for c in CAUSES},
           # **1 つの隙間あたりの残差**（割合ではなく密度で読む＝どの遷移が濃いか）
           "resid_abs_per_gap": {c: round(acc["by_cause_abs"][c] / max(1, acc["cross_n"][c]), 6)
                                 for c in CAUSES},
           # **同じターンの中の残差を手の型で割る**（相手の窓の答え 対 価格の誤り）
           "by_family": {f: {"gaps": acc["fam_n"][f],
                             "resid_per_gap": round(acc["fam_abs"][f] / max(1, acc["fam_n"][f]), 6),
                             "priced_per_gap": round(acc["fam_priced_abs"].get(f, 0.0)
                                                     / max(1, acc["fam_n"][f]), 6),
                             "resid_over_priced": (round(acc["fam_abs"][f]
                                                         / acc["fam_priced_abs"][f], 3)
                                                   if acc["fam_priced_abs"].get(f) else None)}
                         for f in sorted(acc["fam_n"], key=lambda x: -acc["fam_n"][x])},
           # **残差の優先順位**（3 つに割る・各々に別の手当てが要る）:
           # `attack_rows`＝攻撃の行の説明できない分（**相手の窓の答え**と**価格の誤り**が混ざる＝P8 待ち）／
           # `nonattack_rows`＝**攻撃でない行**（体を出す・付与・`TURN_END`・効果）
           #   ——**`curve` の読みではここの価格が厳密に 0**（速さの軸が無いので）。`clock` では価格が付く／
           # `turn_boundary`＝**誰も手を打っていない遷移**（ドン +2・アンタップ・召喚酔いの解除・引き）。
           "resid_priority": _priority(acc),
           # **原因 × 軸**（各原因の中での軸の割合）
           "cross_share": {c: {a: round(acc["cross_abs"][c][a]
                                        / max(1e-12, sum(acc["cross_abs"][c].values())), 4)
                               for a in AXES5} for c in CAUSES}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="値段の付いていない遷移を数える（T123・H）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--d-mode", dest="d_mode", choices=KV.D_MODES, default=None)
    ap.add_argument("--boundary", choices=BOUNDARY_MODES, default=None,
                    help="**T124**: ターンの境目を規則から値付けするか（既定 `off`）")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.d_mode:
        KV.set_d_mode(a.d_mode)
    if a.boundary:
        set_boundary_mode(a.boundary)
    out = collect(a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
