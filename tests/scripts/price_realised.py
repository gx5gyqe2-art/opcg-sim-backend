"""**手の型ごとに「価格」と「実現した価値」を並べる**（T41・2026-09-15・読み取り専用・記録だけ）。

`docs/cpu_theory_gap.md` §0.1 の「目的の一致」——完成＝`v(手)` が手の良し悪しを説明する＝
恒等式 `z − 0.5 ≈ ΔG`（傾き 1）。T40 で `ΔG` が実デッキで反転し（型別では攻撃 0.87・効果 0.16）、
**攻撃と効果・登場の価格の比が現実と合っていない**と判った。本器はその比を**型ごとに直接測る**。

## 何を並べるか

```
価格   v(打った手)                              … 理論の値付け（`theory_order.score_candidate`）
実現   S_meas(次の自分の行) − S_meas(この行)      … 実測の価格で評価した盤面の変化
S_meas = λ·(自ライフ − 相手ライフ) + μ·(自手札 − 相手手札) + δ·(自総ドン − 相手総ドン)
         + Σ ν_meas(自分の体) − Σ ν_meas(相手の体)     ν_meas は帯ごとの実測値（式ではない）
```

**ドンは「総在庫」で数える**（`δ·(自分の総ドン − 相手の総ドン)`・総ドン＝アクティブ＋レスト＋
リーダー付与＋キャラ付与）。初版はアクティブだけを数えて付与の実現が −δ になった（2026-09-15 訂正）——
付与・登場で減ったアクティブは**次のターンに全部戻る**ので在庫の損ではなく、総在庫なら動かない。
**`RAMP_DON`（ドンデッキから追加）だけが総在庫を増やす**＝そこは +δ で実現する。理論の価格が引く
`費用·δ` は**機会費用**（同じターンに他に使えなかった分）で在庫の差分には現れない＝**登場・イベントの
「価格」は機会費用ぶん実現より低く出る**のが正しい読み（`gross`＝足し戻した総額を併記する）。

**同じターンの中の連続する自分の判断点（main 行・kind 0）**で挟む——その間に起きるのは**その手の解決だけ**
（攻撃なら相手の守りの窓・トリガー・自分のアタック時効果の選択）。**自分の選択の行（kind 1/2）は判断点ではない**ので
挟む相手にしない（初版はそこで挟んでいて攻撃の実現が半分消えていた・2026-09-16 訂正）。ターン最後の行（次の自分の行が次のターン）は**相手のターンが
丸ごと挟まる**ので外す。**守りの窓は比べない**（2026-09-15 訂正）——窓は複数行（ブロッカー→カウンター）なので「次の自分の行」が
同じ攻撃の途中になり、しかも**守りの結果は攻撃側の実現（相手が奪ったライフ・使わせた札）に既に入っている**
（攻撃の価格は守り手の最適応答を織り込んだ `min`）＝守りの質は「攻撃側の価格 − 実現」の側に現れる。

**回帰しない**——価格の側と実現の側を別々に出して**比**を見るだけ。実現の側の価格は全部 A 層の実測
（`λ`・`μ`・`δ`・帯ごとの `ν`）で、式（`nu_of`）は通らない＝**価格の欠陥が実現の側に混ざらない**。

## 読み方（事前登録）

- 型ごとの `実現 / 価格` の比が **1 から離れた型が、`ΔG` の反転を作っている型**。
- **攻撃の比が 1 より大きく、効果・登場の比が 1 より小さい**なら、T40 の読み（攻撃が安すぎる／
  効果・登場が高すぎる）が支持される。
- 局ごとの `Σ実現` は `S_meas(終局) − S_meas(開始)` に（外したターン境界を除いて）畳まれる＝
  **`Σ実現 − Σ実現(相手)` が勝敗を傾き ≈ 1 で説明する**なら、実測の価格の水準そのものは正しい。
  そこが外れれば `λ`・`μ`・`ν_meas` の水準の問題で、型の比の問題ではない。

**限界**: 実現は「次の自分の行まで」の変化なので、**その手が将来に残す価値**（体の残りの仕事・
サーチの選択の利得）は `ν_meas`・手札の枚数を通してしか入らない。付与（DON）の実現は攻撃に混ざる。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in ~/w41 --out ~/pr_w41.json
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
import effect_value as EV  # noqa: E402
from theory_bridge import (MOVE_FAMILIES, POL_COLS, ROW_COLS, _extra, _state_of,  # noqa: E402
                           move_family)
import theory_order as _TO  # noqa: E402
from theory_order import (own_attackers_of, play_value, LAM, MU, PWR_EPS, S_IS_CHAR, S_POWER, SC_MY_DON, SC_MY_HAND,  # noqa: E402
                          SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE,
                          SLOT_OPP_FIELD, SLOT_OWN_FIELD, THETA, add_nu_mode_arg, apply_nu_mode,
                          opp_bodies_of, play_cost_term, score_candidate, slot_power, theta_of)

#: 実測の価格（`game_theory.md` §18）——**式ではなく実測**。`δ` は `theory_gate` と同じ値
DELTA = 0.0277
SC_OPP_HAND = 7
#: ドンの列（`encode/scalars.rs`: 2/3 自アクティブ/レスト・4/5 相手・14/15 リーダー付与÷5）と
#: トークンの付与ドン列（`n_rel_feat.S_COLS_V13` の 2 番目・÷5）
SC_DON = {"me": (2, 3, 14), "opp": (4, 5, 15)}
S_ATTACHED_DON = 2
#: **帯ごとの実測 `ν`**（Phase 1a・`2026-09-13_nu_measure` 系・§18）。式 `nu_of` は通さない
NU_MEAS = {"lt_leader": 0.0690, "leader_to_sat": 0.1503, "over_sat": 0.2112}
SAT_OVER_PWR = 2000.0


def nu_meas_of(power, opp_leader_power):
    """帯ごとの実測 `ν`（パワー対相手リーダー）。"""
    x = float(power) - float(opp_leader_power)
    if x < -PWR_EPS:
        return NU_MEAS["lt_leader"]
    if x <= SAT_OVER_PWR + PWR_EPS:
        return NU_MEAS["leader_to_sat"]
    return NU_MEAS["over_sat"]


def side_nu_meas(tok, slots, opp_leader_power):
    """自分／相手の体の `ν_meas` の和。**帯は付与ドンを外した素のパワーで決める**（2026-09-16 訂正・
    ユーザ指摘）——`power_now` は所有者のターンに付与ドンを載せるので、`DON_BOX` で付けた 1000·k が
    次の判断点まで残り、帯が上がって**付けたドンが実現に暗黙に加算**されていた。"""
    tot = 0.0
    for s in range(slots.start, slots.stop):
        if float(tok[s, S_IS_CHAR]) <= 0.5:
            continue
        pw = float(tok[s, S_POWER]) * 1e4 - float(tok[s, S_ATTACHED_DON]) * 5.0 * 1000.0
        if pw < -PWR_EPS:
            continue
        tot += nu_meas_of(pw, opp_leader_power)
    return tot


def don_stock(sc, tok, side="me"):
    """**ドンの総在庫**＝アクティブ＋レスト＋リーダー付与＋キャラ付与（付与しても動かない）。"""
    a, r, ld = SC_DON[side]
    slots = SLOT_OWN_FIELD if side == "me" else SLOT_OPP_FIELD
    attached = sum(float(tok[s, S_ATTACHED_DON]) * 5.0 for s in range(slots.start, slots.stop)
                   if float(tok[s, S_IS_CHAR]) > 0.5)
    return float(sc[a]) + float(sc[r]) + float(sc[ld]) * 5.0 + attached


def state_meas(sc, tok):
    """**実測の価格で評価した盤面**（自席から見た差）。"""
    mlp = float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    return (LAM * (float(sc[SC_MY_LIFE]) - float(sc[SC_OPP_LIFE]))
            + MU * (float(sc[SC_MY_HAND]) - float(sc[SC_OPP_HAND]))
            + DELTA * (don_stock(sc, tok, "me") - don_stock(sc, tok, "opp"))
            + side_nu_meas(tok, SLOT_OWN_FIELD, olp)      # 自分の体は相手のリーダーに対して
            - side_nu_meas(tok, SLOT_OPP_FIELD, mlp))     # 相手の体は自分のリーダーに対して


#: 登場・イベントの価格が引くドンの機会費用（`theory_order.play_value`／`score_candidate` と同じ `0.66·μ`）
DON_COST = 0.66 * MU
#: **後で効く効果**（T53）——次の判断点には出ず、同じターンの後の行（攻撃）に実現が出る動作の型
FLOW_ACTS = frozenset({"ACTIVE_DON", "ATTACH_DON", "GRANT_KEYWORD", "BUFF", "BP_BUFF", "REST"})


def primary_action(cid, triggers=EV.ACTIVATE_TRIGGERS):
    """その契機の最初の能力の最初の動作の型（効果の型の内訳用）。読めなければ `"?"`。"""
    c = EV._all_cards().get(cid) if cid else None
    if not c:
        return "?"
    for ab in (c.get("abilities") or []):
        if (ab.get("trigger") or ab.get("timing")) in triggers:
            acts = EV.walk_actions(ab.get("effect"))
            return str(acts[0].get("type")) if acts else "?"
    return "?"


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const"):
    """(局, 席) ごとに、手の型ごとの価格と実現を足す。"""
    from attack_response import parts          # 遅延 import（attack_response は本器を import する）
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    per = {}
    stats = {"games": 0, "own_rows": 0, "scored": 0, "no_next": 0, "silent": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        order = list(idx)
        # 席ごとの **main 行**（kind 0）の並び——「次の自分の行」は**次の判断点**でなければならない。
        # 攻撃の直後に自分の選択の行（kind 1/2＝アタック時効果の対象・トリガー等）が挟まると、
        # そこで挟むと解決前の盤面を読んでしまい**攻撃の実現が半分消える**（2026-09-16 に実測:
        # リーダー攻撃の実現 0.038 → 0.074・T47）。
        by_seat = {}
        for n, i in enumerate(order):
            if int(rows["kind"][i]) == 0:
                by_seat.setdefault(int(rows["who"][i]), []).append(n)
        nxt = {}
        for w, ns in by_seat.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        # **ターン末の盤面**（T53）＝そのターンの最後の判断点（`TURN_END` の行）。「後で効く」効果
        # （`ACTIVE_DON`・`GRANT_KEYWORD`・`BUFF`・`REST`）の実現は次の判断点には出ず同じターンの攻撃に出るので、
        # 行 → ターン末の差分 `real_te` も持つ（後の行の実現と重なるので**型の和には使わない**・ターン単位の恒等式で読む）
        turn_end_state = {}
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t >= 1 and PL.is_own_turn(w, t) and int(rows["kind"][i]) == 0:
                turn_end_state[(w, t)] = state_meas(ex["sc"][i], ex["tok"][i])   # 後の行で上書き＝最後が残る
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1:
                continue
            z = float(rows["z"][i])
            rec = per.setdefault((seed, w), {"seed": seed, "who": w, "z": None,
                                             "price": {f: 0.0 for f in MOVE_FAMILIES},
                                             "real": {f: 0.0 for f in MOVE_FAMILIES},
                                             "n": {f: 0 for f in MOVE_FAMILIES},
                                             "rows": [], "turns": {}})
            if z != 0.0:
                rec["z"] = 1.0 if z > 0 else 0.0
            sc, tok = ex["sc"][i], ex["tok"][i]
            j = nxt.get(n)
            if j is None:
                stats["no_next"] += 1
                continue
            i2 = order[j]
            if PL.is_own_turn(w, t):
                if int(rows["kind"][i]) != 0:
                    continue
                k = int(L[i]); ch = int(rows["pol_chosen"][i])
                if k < 1 or ch < 0 or ch >= k:
                    continue
                stats["own_rows"] += 1
                if int(rows["turn"][i2]) != t:
                    stats["no_next"] += 1      # ターン最後の行＝相手のターンが挟まる
                    continue
                th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]),
                              mode=theta_mode, theta=theta)
                ctx = {"theta": th, "mu": mu,
                       "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                       "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                       "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), "don_k": 1,
                       "attackers": own_attackers_of(tok, float(sc[SC_OPP_LEADER_POWER]) * 1e4),
                       "don_active": float(sc[SC_MY_DON]),   # 登場の機会費用（T43）
                                              "st": _state_of(sc, ex["ci"][i], idx2cid),
                       "opp_bodies": opp_bodies_of(
                           tok, float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                           max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), th, mu,
                           ci_row=ex["ci"][i], idx2cid=idx2cid)}
                b = int(ptr[i]) + ch
                sig = json.loads(pol["pol_sig"][b])
                tl = sig[2] if len(sig) > 2 else None
                v = score_candidate(sig, str(pol["pol_cid"][b]) or None,
                                    (str(pol["pol_tcid"][b]) or None) if tl else None, ctx, cards,
                                    src_power=slot_power(tok, pol["pol_si"][b]),
                                    tgt_power=slot_power(tok, pol["pol_ti"][b]),
                                    don_k=pol["pol_k"][b])
                fam = move_family(sig)
                if v is None:
                    stats["silent"] += 1
                    continue
                stats["scored"] += 1
                real = state_meas(ex["sc"][i2], ex["tok"][i2]) - state_meas(sc, tok)
                cid = str(pol["pol_cid"][b]) or None
                # **総額**＝価格にドンの機会費用を足し戻したもの（在庫の差分と同じ土俵にする）
                info = cards.info(cid) if cid else None
                cost = float((info or {}).get("cost") or 0) if fam == "play" else 0.0
                # 実際に引かれた費用（`state` なら機会費用・`flat` なら定額）を足し戻す（T43）
                gross = float(v) + (play_cost_term(ctx, cost, mu, th) if fam == "play" else 0.0)
                act = (primary_action(cid) if fam == "effect"
                       else primary_action(cid, EV.CHAR_ON_PLAY_TRIGGERS) if fam == "play" else None)
                real_te = turn_end_state.get((w, t), state_meas(ex["sc"][i2], ex["tok"][i2])) - state_meas(sc, tok)
                # **登場の内訳**（T53）: 価格を「体（ν − μ）」「登場時効果」「機会費用」に、実現を部品に割る
                play_parts = None
                if fam == "play" and info is not None and not (info.get("event") or info.get("stage")):
                    nu_part = play_value(float(info["power"]), 0, ctx["opp_leader_power"], ctx["r_turns"], th, mu,
                                         is_blocker=info.get("blocker"), my_leader_power=ctx["my_leader_power"])
                    cost_part = play_cost_term(ctx, cost, mu, th)
                    play_parts = {"nu_minus_mu": float(nu_part), "effect": float(v) - float(nu_part) + float(cost_part),
                                  "opportunity": float(cost_part), **parts(sc, tok, ex["sc"][i2], ex["tok"][i2])}
            else:
                continue                       # 守りの窓は比べない（docstring）
            rec["price"][fam] += float(v); rec["real"][fam] += float(real); rec["n"][fam] += 1
            rec["rows"].append({"fam": fam, "price": float(v), "real": float(real), "real_te": float(real_te),
                                "gross": float(gross), "act": act, "cid": cid, "turn": t, "play_parts": play_parts})
            # **ターン単位の恒等式**——価格の和 対 「最初の自分の行 → 最後の自分の行」の実現
            tk = rec["turns"].setdefault(t, {"price": 0.0, "first": None, "last": None, "acts": set()})
            tk["price"] += float(v)
            if act:
                tk["acts"].add(act)
            if tk["first"] is None:
                tk["first"] = state_meas(sc, tok)
            tk["last"] = state_meas(ex["sc"][i2], ex["tok"][i2])
    return per, stats


def _slope(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 10 or float(x.var()) <= 0:
        return None
    xd = x - x.mean()
    return float((xd * (y - y.mean())).sum() / (xd * xd).sum())


def _auc(scores, labels):
    from theory_bridge import auc
    return auc(scores, labels)


def _boot_slope(x, y, reps, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y); vals = []
    for _ in range(int(reps)):
        k = rng.integers(0, len(x), len(x))
        v = _slope(x[k], y[k])
        if v is not None:
            vals.append(v)
    return ([round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
            if len(vals) >= 10 else [None, None])


def summarise(per, reps=200, seed=0):
    fams = MOVE_FAMILIES
    out = {"by_family": {}}
    allrows = [r for rec in per.values() for r in rec["rows"]]
    def block(rs):
        p = np.array([r["price"] for r in rs]); q = np.array([r["real"] for r in rs])
        g = np.array([r["gross"] for r in rs])
        te = np.array([r.get("real_te", r["real"]) for r in rs])
        return {"n": len(rs), "price_mean": round(float(p.mean()), 5), "real_mean": round(float(q.mean()), 5),
                "gross_mean": round(float(g.mean()), 5),
                # **行 → ターン末の実現**（T53・後の行の実現と重なるので参考値）
                "real_turn_end_mean": round(float(te.mean()), 5),
                "price_p10_p50_p90": [round(float(np.percentile(p, k)), 4) for k in (10, 50, 90)],
                "real_p10_p50_p90": [round(float(np.percentile(q, k)), 4) for k in (10, 50, 90)],
                "ratio_real_over_price": (round(float(q.mean() / p.mean()), 3)
                                          if abs(float(p.mean())) > 1e-9 else None),
                "ratio_real_over_gross": (round(float(q.mean() / g.mean()), 3)
                                          if abs(float(g.mean())) > 1e-9 else None),
                "corr": round(float(np.corrcoef(p, q)[0, 1]), 4) if p.std() > 0 and q.std() > 0 else None,
                "slope_real_on_price": (round(_slope(p, q), 4) if _slope(p, q) is not None else None)}
    for f in fams:
        rs = [r for r in allrows if r["fam"] == f]
        if len(rs) < 10:
            continue
        out["by_family"][f] = block(rs)
    # 効果の型の内訳（最初の動作の型ごと）
    eff = [r for r in allrows if r["fam"] == "effect"]
    acts = {}
    for r in eff:
        acts.setdefault(r["act"], []).append(r)
    out["effect_by_action"] = {a: block(rs) for a, rs in sorted(acts.items(), key=lambda kv: -len(kv[1]))
                               if len(rs) >= 20}
    # **登場時効果の型の内訳**（T54）——登場の行を登場時能力の最初の動作の型で切る（`?` は能力なし／読めない）
    pacts = {}
    for r in [r for r in allrows if r["fam"] == "play"]:
        pacts.setdefault(r.get("act") or "?", []).append(r)
    out["play_by_onplay_action"] = {a: block(rs) for a, rs in sorted(pacts.items(), key=lambda kv: -len(kv[1]))
                                    if len(rs) >= 20}
    # **T53 (a) ターン単位の恒等式を「後で効く効果が在るターン」と無いターンで分ける**——
    # 流れの効果の価格が正しければ両群の Σ価格/実現 は同じになる（重ね数えをせずに検める）
    grp = {"with_flow_effect": [], "without": []}
    for rec in per.values():
        for tk in rec["turns"].values():
            if tk["first"] is None or tk["last"] is None:
                continue
            key = "with_flow_effect" if (tk.get("acts") or set()) & FLOW_ACTS else "without"
            grp[key].append((tk["price"], tk["last"] - tk["first"]))
    out["turns_by_flow_effect"] = {}
    for key, xs in grp.items():
        if len(xs) >= 20:
            p = np.array([a for a, _b in xs]); q = np.array([b for _a, b in xs])
            out["turns_by_flow_effect"][key] = {"turns": len(xs), "price_mean": round(float(p.mean()), 5),
                                                "real_mean": round(float(q.mean()), 5),
                                                "ratio_real_over_price": round(float(q.mean() / p.mean()), 3) if abs(p.mean()) > 1e-9 else None}
    # **T53 (b) 登場の内訳**——価格の部品と実現の部品の平均
    pp = [r["play_parts"] for r in allrows if r.get("play_parts")]
    if len(pp) >= 20:
        out["play_breakdown"] = {"n": len(pp), **{k: round(float(np.mean([x[k] for x in pp])), 5) for k in pp[0]}}
    # 局ごとの突き合わせ: Σ価格 と Σ実現 が勝敗をどれだけ説明するか（傾き 1 が理想）
    by = {}
    for (sd, w), rec in per.items():
        by.setdefault(sd, {})[w] = rec
    pairs = []
    for sd, seats in by.items():
        a, b = seats.get(0), seats.get(1)
        if a is None or b is None or a["z"] is None or b["z"] is None:
            continue
        na, nb = sum(a["n"].values()), sum(b["n"].values())
        if na < 1 or nb < 1:
            continue
        def turn_sum(r, key):
            return sum((tk["last"] - tk["first"]) if key == "real" else tk["price"]
                       for tk in r["turns"].values() if tk["first"] is not None and tk["last"] is not None)
        pairs.append({"z": a["z"],
                      "dPrice": sum(a["price"].values()) - sum(b["price"].values()),
                      "dReal": sum(a["real"].values()) - sum(b["real"].values()),
                      "dPriceTurn": turn_sum(a, "price") - turn_sum(b, "price"),
                      "dRealTurn": turn_sum(a, "real") - turn_sum(b, "real"),
                      "dPrice_fam": {f: a["price"][f] - b["price"][f] for f in fams},
                      "dReal_fam": {f: a["real"][f] - b["real"][f] for f in fams}})
    out["games"] = len(pairs)
    if len(pairs) >= 10:
        y = [p["z"] for p in pairs]
        for key in ("dPrice", "dReal", "dPriceTurn", "dRealTurn"):
            x = [p[key] for p in pairs]
            out[key] = {"auc": round(_auc(x, y), 4), "slope": (round(_slope(x, y), 4)
                                                             if _slope(x, y) is not None else None),
                        "slope_ci95": _boot_slope(x, y, reps, seed),
                        "mean_abs": round(float(np.mean(np.abs(x))), 4)}
        out["by_family_games"] = {}
        for f in fams:
            xp = [p["dPrice_fam"][f] for p in pairs]; xr = [p["dReal_fam"][f] for p in pairs]
            if float(np.var(xp)) <= 0 and float(np.var(xr)) <= 0:
                continue
            out["by_family_games"][f] = {
                "auc_price": round(_auc(xp, y), 4) if float(np.var(xp)) > 0 else None,
                "auc_real": round(_auc(xr, y), 4) if float(np.var(xr)) > 0 else None,
                "slope_price": (round(_slope(xp, y), 3) if _slope(xp, y) is not None else None),
                "slope_real": (round(_slope(xr, y), 3) if _slope(xr, y) is not None else None)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    add_nu_mode_arg(ap)
    _TO.add_surv_mode_arg(ap)
    _TO.add_cbar_mode_arg(ap)
    ap.add_argument("--flow-pricing", default=None, choices=EV.FLOW_PRICING_MODES,
                    help="**T54** 後で効く効果を付与の行で数える（`option`・既定）か、使った行で数える（`exercise`＝付与の行は 0）か")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)
    _TO.apply_surv_mode(a)
    _TO.apply_cbar_mode(a)
    if a.flow_pricing is not None:
        EV.set_flow_pricing(a.flow_pricing)
    t0 = time.time()
    per, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "flow_pricing": EV.FLOW_PRICING, "stats": stats,
           "frozen": {"lambda": LAM, "mu": MU, "delta": DELTA, "nu_meas": NU_MEAS, "theta": a.theta},
           "summary": summarise(per, a.boot_reps, a.seed), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
