"""**手札の厚い守り手の生き残りの言い過ぎを原因ごとに割る**（2026-10-08・`docs/reports/2026-10-08_survival_split.md`・読み取り専用・診断だけ）。

出発点は `2026-10-08_late_winner.md` §4〜§6.2（守る側の計算が `S` > 0.95 の帯で地平の内の決着を 0.6%／0.7% と言うのに、
観測は ≥ 19%／≥ 26%）。M-2 の計器（`m2_probe.py --clocks-out`・`OPCG_M2_PROBE`）が書いた時計ごとの行（守る側の計算の入力
`din` つき）と、記録そのもの（攻撃・カウンター・ブロック・ライフ・登場）を突き合わせ、守る側の計算を Rust（`m2.resolve`・
診断の入口）で入力を差し替えて解き直す。既定の出力・表・golden には触らない。

測るもの（時計は「自席ターンの頭の行の自分の時計」＝段 1 が今のターン・実際の残りの自席ターン `act` と段が揃うものだけ）:

1. **較正を段で分ける（i）**: 段 `s` の倒れる確率（条件つき＝段 `s` まで生きていたとして）を、予測と観測で比べる。負けた席の時計は
   残りの自席ターン `own_left` で打ち切り（生存）＝勝った席を選ぶ偏り（1c）を入れない（Kaplan–Meier の形）。
2. **連鎖の予測**: 段 `s` に実際に来た盤面（その自席ターンの頭の行）で守る側の計算が言う「今のターンに倒れる確率」`q_s`。
   最初の行からの予測 `h0(s)` と `q_s` の差＝**先の盤面の読みの誤り**、`q_s` と観測の差＝**今のターンの読みの誤り**。
3. **決着のターンに攻撃した体の出どころ（ii）**: 時計の行の時点で盤面に居た／手札に在った／どちらでもない（引いた札・効果で
   出た新しい攻め手）。
4. **守り手の実際（iii）**: そのターンの実際の攻撃の並び（リーダーへの攻撃の超過）で守る側の計算を解き直し、止める本数・切る枚数・
   倒れる確率の予測を、実際（ライフの減り・カウンター・ブロック・決着）と並べる。
5. **解き直し**: 最初の行の守る側の計算の攻撃の並びを (A) 実際の攻撃の並びに差し替える・(A1) 新しい攻め手の攻撃だけ足す・
   (D) 守り手が自分のターンに手札から出した札（地平の内）を最初の手札から抜く、で地平の内の倒れる確率がどれだけ観測に近づくか。

実行例:
  python tests/scripts/survival_split.py --in ~/w41 --clocks real.jsonl --out split_real.json
"""
import argparse
import collections
import json
import math
import os
import sys

os.environ.setdefault("OPCG_M2_PROBE", "1")    # `m2.resolve` は計器が立っているときだけ通る

import numpy as np  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import theory_rs as TR  # noqa: E402

S_POWER, S_COUNTER = 0, 7
OWN_FIELD, HAND = range(2, 7), range(12, 22)
SC_MY_LIFE, SC_OPP_LIFE, SC_MY_HAND, SC_OPP_LP = 0, 1, 6, 13
EPS = 10.0
ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "sig")
POL_COLS = ("pol_sig", "pol_cid", "pol_tcid", "pol_si", "pol_ti", "pol_k")


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32),
            "ci": np.asarray(dd["card_idx"])[:n]}


def _own(w, t):
    return t >= 1 and ((t % 2 == 1) == (w == 0))


def _p(x):
    return float(np.rint(x * 1e4 / EPS) * EPS)


# ---------------------------------------------------------------------------------------------
# 記録の読み

def extract(dirs):
    """記録 → {seed: 局}。局＝{"turns": {(w, t): ターン}, "winner": w}。ターン＝自席ターンの頭の行の盤面と手札・
    そのターンの攻撃（リーダーへ・体へ）・守り手のカウンター／ブロック・ライフの減り・手札から出した札。"""
    rev = TR.idx2cid()
    out = {}
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        z0 = [float(rows["z"][i]) for i in idx if int(rows["who"][i]) == 0]
        winner = 0 if (z0 and z0[0] > 0) else 1
        turns = {}
        seq = list(idx)
        first_play = {}          # uuid → 手札から出した (w, t)
        first_atk = {}           # uuid → 最初に攻撃した (w, t)
        for n, i in enumerate(seq):
            w, t, k = int(rows["who"][i]), int(rows["turn"][i]), int(rows["kind"][i])
            if not _own(w, t):
                continue
            key = (w, t)
            sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
            if key not in turns:
                if k != 0:
                    continue
                if int(L[i]) > 0:
                    s0 = json.loads(pol["pol_sig"][int(ptr[i])])
                    if s0 and s0[0] == "RESOLVE_EFFECT_SELECTION":
                        continue
                turns[key] = {"w": w, "t": t,
                              "field": [rev.get(int(ci[s]), "") for s in OWN_FIELD if int(ci[s])],
                              "hand": [rev.get(int(ci[s]), "") for s in HAND if int(ci[s])],
                              "hand_n": int(round(float(sc[SC_MY_HAND]))),
                              "def_life0": int(round(float(sc[SC_OPP_LIFE]))),
                              "atk": [], "plays": [], "counters": 0, "blocks": 0, "def_life1": None}
            T = turns[key]
            if k != 0 or int(L[i]) < 1:
                continue
            j = int(ptr[i]) + int(rows["pol_chosen"][i])
            sg = json.loads(pol["pol_sig"][j])
            at = sg[0] if sg else None
            if at == "DON_BOX" and sg[2]:
                si, ti, kk = int(pol["pol_si"][j]), int(pol["pol_ti"][j]), max(0, int(pol["pol_k"][j]))
                pw = _p(float(tok[si, S_POWER])) + 1000.0 * kk if 0 <= si < tok.shape[0] else None
                lp = _p(float(sc[SC_OPP_LP])) or 5000.0
                u = sg[1]
                T["atk"].append({"u": u, "cid": str(pol["pol_cid"][j]), "si": si, "ti": ti, "leader": ti == 1, "k": kk,
                                 "x": None if pw is None else pw - lp})
                first_atk.setdefault(u, key)
            elif at == "PLAY":
                si = int(pol["pol_si"][j])
                cv = float(tok[si, S_COUNTER]) * 2000.0 if 0 <= si < tok.shape[0] else 0.0
                T["plays"].append({"u": sg[1], "cid": str(pol["pol_cid"][j]), "counter": round(cv)})
                first_play.setdefault(sg[1], key)
        # 守り手の応答（相手の手番のカウンター／ブロック）とライフの減り
        for n, i in enumerate(seq):
            w, t, k = int(rows["who"][i]), int(rows["turn"][i]), int(rows["kind"][i])
            if _own(w, t) or k == 0:
                continue
            key = (1 - w, t)
            if key not in turns:
                continue
            rs = json.loads(rows["sig"][i]) if rows["sig"][i] else None
            if rs and rs[0] == "SELECT_COUNTER":
                turns[key]["counters"] += 1
            elif rs and rs[0] == "SELECT_BLOCKER":
                turns[key]["blocks"] += 1
        # ターンの後の守り手のライフ（次のターンの最初の行）
        rows_by_turn = collections.OrderedDict()
        for i in seq:
            rows_by_turn.setdefault(int(rows["turn"][i]), i)
        tl = list(rows_by_turn.items())
        for (w, t), T in turns.items():
            nxt = next((i for tt, i in tl if tt > t), None)
            if nxt is None:
                T["def_life1"] = None          # 局がこのターンで終わった
                T["killed"] = (w == winner)
            else:
                ww = int(rows["who"][nxt])
                T["def_life1"] = int(round(float(ex["sc"][nxt][SC_MY_LIFE if ww == 1 - w else SC_OPP_LIFE])))
                T["killed"] = False
        out[seed] = {"turns": turns, "winner": winner, "first_play": first_play, "first_atk": first_atk}
    return out


def classify(G, w, t0, a):
    """攻撃 `a`（ターン `t`）の体が、時計の行（`(w, t0)` の頭）の時点でどこに在ったか: `leader`／`board`／`hand`／`new`。"""
    if a["si"] == 0:
        return "leader"
    u = a["u"]
    fp = G["first_play"].get(u)
    fa = G["first_atk"].get(u)
    if fp is not None and fp[1] < t0:
        return "board"
    if fp is None:
        # 手札から出していない体: 時計の行より前から攻撃していた／盤面に在った（最初の局面）なら盤面、でなければ効果で出た
        if fa is not None and fa[1] < t0:
            return "board"
        T0 = G["turns"].get((w, t0))
        return "board" if (T0 and a["cid"] in T0["field"]) else "new"
    T0 = G["turns"].get((w, t0))
    if T0 is None:
        return "hand"
    if a["cid"] in T0["hand"] or T0["hand_n"] > len(T0["hand"]):   # 手札の枠（10）を越えた分は手札に在ったとみなす（新を下に）
        return "hand"
    return "new"


# ---------------------------------------------------------------------------------------------
# 守る側の計算の解き直し（Rust の `m2.resolve`）

def resolve(din, **ov):
    a = dict(din)
    a.update(ov)
    a = {k: v for k, v in a.items() if k != "no_now"}
    for k in ("cards", "nu"):
        a[k] = [[float(p), float(q)] for p, q in a[k]]
    for k in ("life_types", "draw_types"):
        a[k] = [[float(p), float(q), float(r)] for p, q, r in a[k]]
    for k in ("xs_first", "blk", "rest", "arrive"):
        a[k] = [float(x) for x in a[k]]
    a["seq"] = [[float(x) for x in s] for s in a["seq"]]
    for k in ("don", "life", "lam", "lam_net", "mu", "olp", "mlp", "eps", "feq"):
        a[k] = float(a[k])
    a["turns"] = int(a["turns"])
    a["death"] = bool(a.get("death", False))
    TR.ready()
    return TR.core_call("m2.resolve", a, {})


def pdeath(din, **ov):
    r = resolve(din, death=True, **ov)
    return None if r is None else list(r["harms"])


def p1_death(din, **ov):
    """今のターン（段 1）に倒れる確率。ライフ 0 では値段の工夫が効かない（とどめの段の芯が 0）ので、地平 1 で解いて
    「止められる本数 < 当たる本数」なら 1（ライフ 0 の守り手は 1 本でも通せば倒れる・引く札もライフから手札も無い＝決まる）。"""
    life = float(ov.get("life", din["life"]))
    if round(life) >= 1:
        pd = pdeath(din, **ov)
        return pd[0] if pd else 0.0
    xs = ov.get("xs_first", din["xs_first"])
    n = sum(1 for x in xs if x >= -float(din["eps"]))
    if n == 0:
        return 0.0
    r = resolve(din, turns=1, **ov)
    return 0.0 if float(r["prevented"]) >= n - 1e-9 else 1.0


def hits_per_step(din, **ov):
    """段ごとの受けた本数の期待値（とどめの 1 本は数えない）——値段を「受けた 1 本だけ 1」にする。"""
    nu0 = [[m, 0.0] for m, _ in din["nu"]]
    r = resolve(din, lam=1.0, lam_net=1.0, mu=0.0, nu=nu0, **ov)    # λ = λ_net でとどめの段の芯 (λ − λ_net)·L0 = 0
    return list(r["harms"]), r


def cuts_per_step(din, **ov):
    nu0 = [[m, 0.0] for m, _ in din["nu"]]
    r = resolve(din, lam=0.0, lam_net=0.0, mu=1.0, nu=nu0, **ov)
    return list(r["harms"])


def _within(pd, h):
    return float(sum(pd[:h])) if pd else 0.0


# ---------------------------------------------------------------------------------------------
# 1 本の時計の測り

def _seq_with(din, W, by_step, mode):
    """攻撃の並び（段 2 から）を作り直す。`by_step[s]`＝段 `s` の実際の超過のリスト（`mode`＝"replace" なら段 2..W を実際に
    差し替え・"add" なら元の並びに足す）。W より先は元の並び（守り手が先の段に札を残す読みは変えない）。"""
    seq = [list(x) for x in din["seq"]]
    H = int(din["turns"])
    while len(seq) < max(H - 1, 1):
        seq.append(list(seq[-1]) if seq else [])
    for s in range(2, W + 1):
        if s - 2 >= len(seq):
            break
        if mode == "replace":
            seq[s - 2] = list(by_step.get(s, []))
        else:
            seq[s - 2] = list(seq[s - 2]) + list(by_step.get(s, []))
    return seq


def _drop_cards(cards, vals):
    """手札（`(止める量, ドン)` の列）から、自分のターンに出した札の止める量を 1 枚ずつ抜く（ドン 0 の札を先に）。"""
    cs = [list(c) for c in cards]
    n = 0
    for v in vals:
        if v <= 0:
            continue
        hit = None
        for q, (c, d) in enumerate(cs):
            if abs(c - v) < 1e-6 and d == 0:
                hit = q
                break
        if hit is None:
            for q, (c, d) in enumerate(cs):
                if abs(c - v) < 1e-6:
                    hit = q
                    break
        if hit is not None:
            cs.pop(hit)
            n += 1
    return cs, n


def measure(c, G, by_turn, do_resolve=True):
    seed, w, t0 = int(c["seed"]), int(c["who"]), int(c["t"])
    won = bool(c["winner"])
    act = int(c["act"]) if won else None
    left = int(c["own_left"])
    pd0 = [float(x) for x in c["pdeath"]]
    H = len(pd0)
    din = c["din"]
    last = act if won else left
    W = min(H, last)                          # 見張った段（地平の内・観測のある段）
    ev = bool(won and act <= H)               # 地平の内で倒した
    r = {"seed": seed, "w": w, "t0": t0, "won": won, "act": act, "left": left, "H": H, "W": W, "ev": ev,
         "S": max(0.0, 1.0 - sum(pd0)), "F_W": _within(pd0, W), "pd0": pd0[:W], "cut_exp": float(c.get("cut") or 0.0),
         "hand_cards": len(din["cards"]), "life": float(din["life"])}
    by_step, new_step, src_n = {}, {}, collections.Counter()
    steps = []
    def_plays = []                            # 守り手が自分のターンに出した札の止める量（段 s の前）
    for s in range(1, W + 1):
        t = t0 + 2 * (s - 1)
        T = G["turns"].get((w, t))
        st = {"s": s}
        if T is not None:
            la = [a for a in T["atk"] if a["leader"] and a["x"] is not None]
            xs = [a["x"] for a in la]
            src = [classify(G, w, t0, a) for a in la]
            for sr in src:
                src_n[sr] += 1
            by_step[s] = xs
            new_step[s] = [x for x, sr in zip(xs, src) if sr == "new"]
            st.update({"n_atk": len(xs), "n_hit_x": sum(1 for x in xs if x >= -EPS), "src": src,
                       "n_new": sum(1 for sr in src if sr == "new"),
                       "n_new_hit": sum(1 for x, sr in zip(xs, src) if sr == "new" and x >= -EPS),
                       "counters": T["counters"], "blocks": T["blocks"],
                       "life_lost": (T["def_life0"] - T["def_life1"]) if T["def_life1"] is not None else None,
                       "killed": bool(won and s == act)})
        D = G["turns"].get((1 - w, t + 1))   # 守り手の次の自分のターン（段 s と s+1 の間）
        st["def_play_counters"] = [p["counter"] for p in D["plays"]] if D is not None else []
        cs = by_turn.get((seed, w, t))
        if cs is not None and cs.get("pdeath") is not None:
            st["q1"] = float(cs["pdeath"][0]) if cs["pdeath"] else 0.0
        steps.append(st)
    r["steps"] = steps
    r["src"] = dict(src_n)
    # 決着のターン（勝った席・地平の内）の攻撃の出どころ
    if ev and steps and steps[-1].get("src") is not None:
        r["kill_src"] = steps[-1]["src"]
    if not do_resolve:
        return r
    # 今のターン（段 1）を実際の攻撃で解き直す（iii）——今の盤面・今の守り手の手札はそのまま
    if 1 in by_step:
        pdA1 = pdeath(din, xs_first=by_step[1])
        hA1, _ = hits_per_step(din, xs_first=by_step[1])
        cA1 = cuts_per_step(din, xs_first=by_step[1])
        r["now_act"] = {"q": pdA1[0] if pdA1 else 0.0, "hits": (hA1[0] if hA1 else 0.0) + (pdA1[0] if pdA1 else 0.0),
                        "cuts": cA1[0] if cA1 else 0.0}
        hM, _ = hits_per_step(din)
        cM = cuts_per_step(din)
        r["now_model"] = {"q": pd0[0] if pd0 else 0.0, "hits": (hM[0] if hM else 0.0) + (pd0[0] if pd0 else 0.0),
                          "cuts": cM[0] if cM else 0.0}
    # 解き直し（地平の内・見張った段 W まで）
    base_first = list(din["xs_first"])
    alt = {}
    if by_step:
        xfA = by_step.get(1, base_first)
        alt["A"] = dict(xs_first=xfA, seq=_seq_with(din, W, by_step, "replace"))
        alt["A1"] = dict(xs_first=base_first + new_step.get(1, []), seq=_seq_with(din, W, new_step, "add"))
    drop = [v for st in steps[:-1] for v in st["def_play_counters"]]     # 段 W より前の守り手のターンに出した札
    cardsD, nD = _drop_cards(din["cards"], drop)
    r["D_dropped"] = nD
    alt["D"] = dict(cards=cardsD)
    if "A" in alt:
        alt["AD"] = dict(alt["A"], cards=cardsD)
    for k, ov in alt.items():
        pd = pdeath(din, **ov)
        r["F_" + k] = _within(pd, W) if pd is not None else None
    # 窓の内で受けた本数（とどめの 1 本を含む）: 計画の攻撃・実際の攻撃で解き直した守る側の計算・実際
    hM, _ = hits_per_step(din)
    r["hitsW_model"] = float(sum(hM[:W])) + r["F_W"]
    if "A" in alt:
        hA, _ = hits_per_step(din, **alt["A"])
        r["hitsW_A"] = float(sum(hA[:W])) + (r["F_A"] or 0.0)
        cA = cuts_per_step(din, **alt["A"])
        r["cutsW_A"] = float(sum(cA[:W]))
    cM = cuts_per_step(din)
    r["cutsW_model"] = float(sum(cM[:W]))
    ho, co, ok = 0.0, 0.0, True
    for st in steps:
        if st.get("n_atk") is None:
            ok = False
            break
        co += st["counters"]
        if st["life_lost"] is not None:
            ho += max(0, st["life_lost"])
        elif st["killed"]:
            T = G["turns"].get((w, t0 + 2 * (st["s"] - 1)))
            ho += T["def_life0"] + 1
    if ok:
        r["hitsW_obs"], r["cutsW_obs"] = ho, co
    return r


def chain_rows(c, G, by_turn):
    """段 s（2 以上・s まで倒れていない）ごとの 2×2: 段 s に実際に来た守り手の状態（その自席ターンの頭の行の守る側の計算の入力）に、
    攻撃を (F) 最初の行の計画が段 s に並べた攻撃・(F+new) それに新しい攻め手の実際の攻撃を足したもの・(A) 実際の攻撃、で
    今のターンに倒れる確率を解く。最初の行からの条件つきの予測（両方とも読み）と観測と並べる。"""
    seed, w, t0 = int(c["seed"]), int(c["who"]), int(c["t"])
    won = bool(c["winner"])
    act = int(c["act"]) if won else None
    pd0 = [float(x) for x in c["pdeath"]]
    H = len(pd0)
    W = min(H, act if won else int(c["own_left"]))
    seq0 = c["din"]["seq"]
    S0 = max(0.0, 1.0 - sum(pd0))
    # 逆の順: 最初の行の守る側の計算（守り手の状態は読みのまま）に、実際の攻撃（全部／新しい攻め手だけ）を入れた条件つきの予測
    din0 = c["din"]
    by_step, new_step = {}, {}
    for s in range(1, W + 1):
        T = G["turns"].get((w, t0 + 2 * (s - 1)))
        if T is None:
            continue
        la = [a for a in T["atk"] if a["leader"] and a["x"] is not None]
        by_step[s] = [a["x"] for a in la]
        new_step[s] = [a["x"] for a in la if classify(G, w, t0, a) == "new"]
    base_first = list(din0["xs_first"])
    pdA = pdeath(din0, xs_first=by_step.get(1, base_first), seq=_seq_with(din0, W, by_step, "replace")) if by_step else None
    pdA1 = pdeath(din0, xs_first=base_first + new_step.get(1, []), seq=_seq_with(din0, W, new_step, "add")) if by_step else None

    # 守り手の状態の読み（最初の行の守る側の計算の期待値）: 段 s の頭のライフ・止める札の枚数
    hF, _ = hits_per_step(din0)
    cF = cuts_per_step(din0)
    p_draw = sum(t[2] for t in din0["draw_types"])
    p_life = sum(t[2] for t in din0["life_types"])

    def hz(pd, s):
        if pd is None:
            return None
        F = sum(pd[:s - 1])
        return pd[s - 1] / (1.0 - F) if 1.0 - F > 1e-9 else 1.0
    out = []
    for s in range(2, W + 1):
        t = t0 + 2 * (s - 1)
        cs = by_turn.get((seed, w, t))
        T = G["turns"].get((w, t))
        if cs is None or T is None or not cs.get("din"):
            continue
        F = sum(pd0[:s - 1])
        h0 = pd0[s - 1] / (1.0 - F) if 1.0 - F > 1e-9 else 1.0
        la = [a for a in T["atk"] if a["leader"] and a["x"] is not None]
        xa = [a["x"] for a in la]
        xnew = [a["x"] for a in la if classify(G, w, t0, a) == "new"]
        xf = list(seq0[s - 2]) if s - 2 < len(seq0) else (list(seq0[-1]) if seq0 else [])
        dn = cs["din"]
        pF = p1_death(dn, xs_first=xf)
        pFn = p1_death(dn, xs_first=xf + xnew)
        pA = p1_death(dn, xs_first=xa)
        # 実際の攻撃を分ける: 付けたドンを外した並び（体の素の強さ）・新しい攻め手を抜いた並び
        pA0 = p1_death(dn, xs_first=[a["x"] - 1000.0 * a.get("k", 0) for a in la])
        pAold = p1_death(dn, xs_first=[a["x"] for a in la if classify(G, w, t0, a) != "new"])
        qn = p1_death(dn)
        # 守り手の状態を 1 つずつ最初の行の値へ戻す（実際の攻撃のまま）: ドン・ライフ・ブロッカー・手札（最初の手札から自分の
        # ターンに出した札を抜いたもの＝守りに切った札は戻る）
        drop = [v for ss in range(1, s) for v in ((G["turns"].get((1 - w, t0 + 2 * (ss - 1) + 1)) or {}).get("plays") and
                                                    [p["counter"] for p in G["turns"][(1 - w, t0 + 2 * (ss - 1) + 1)]["plays"]] or [])]
        cardsD, _ = _drop_cards(din0["cards"], drop)
        sw = {"don": dict(don=din0["don"]), "life": dict(life=din0["life"]),
              "blk": dict(blk=din0["blk"], rest=din0["rest"], arrive=din0["arrive"], nu=din0["nu"] + dn["nu"]),
              "hand": dict(cards=cardsD)}
        qsw = {}
        for kk, ov in sw.items():
            try:
                qsw[kk] = p1_death(dn, xs_first=xa, **ov)
            except Exception:  # noqa: BLE001  ブロッカーの余裕の ν が揃わないときは測らない
                qsw[kk] = None
        eh, ec = float(sum(hF[:s - 1])), float(sum(cF[:s - 1]))
        out.append({"seed": seed, "w": w, "t0": t0, "s": s, "S0": S0, "h0": h0,
                    "lifeF_s": float(din0["life"]) - eh, "cardsF_s": len(din0["cards"]) - ec + (s - 1) * p_draw + eh * p_life, "hA_r0": hz(pdA, s), "hA1_r0": hz(pdA1, s),
                    "q_now": qn, "q_F": pF, "q_Fnew": pFn, "q_A": pA, "q_A0": pA0, "q_Aold": pAold,
                    "k_A": int(sum(a.get("k", 0) for a in la)), "k_all": int(sum(a.get("k", 0) for a in T["atk"])),
                    "obs": bool(won and act == s),
                    "n_F": sum(1 for x in xf if x >= -EPS), "n_A": sum(1 for x in xa if x >= -EPS), "n_new": len(xnew),
                    "sumx_F": float(sum(max(0.0, x + 1000.0) for x in xf if x >= -EPS)),
                    "sumx_A": float(sum(max(0.0, x + 1000.0) for x in xa if x >= -EPS)),
                    "cards0": len(din0["cards"]), "cards_s": len(dn["cards"]),
                    "csum0": float(sum(p for p, _ in din0["cards"])), "csum_s": float(sum(p for p, _ in dn["cards"])),
                    "life0": float(din0["life"]), "life_s": float(dn["life"]), "don0": float(din0["don"]), "don_s": float(dn["don"]),
                    "blk0": len(din0["blk"]), "blk_s": len(dn["blk"]), "life0_s": round(float(dn["life"])) < 1,
                    **{"q_A_" + kk: v for kk, v in qsw.items()}})
    return out


def run(clocks, ext, limit=0, do_resolve=True, log_every=500):
    me = [c for c in clocks if c.get("side") == "me" and c.get("plan") and c.get("din") and c.get("pdeath") is not None
          and int(c["seed"]) in ext]
    if limit:
        me = me[:limit]
    by_turn = {(int(c["seed"]), int(c["who"]), int(c["t"])): c for c in clocks if c.get("side") == "me" and "seed" in c}
    out = []
    for n, c in enumerate(me):
        out.append(measure(c, ext[int(c["seed"])], by_turn, do_resolve))
        if log_every and (n + 1) % log_every == 0:
            print("  %d / %d" % (n + 1, len(me)), file=sys.stderr, flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--clocks", required=True)
    ap.add_argument("--out", default="", help="時計ごとの測りの行（JSON lines）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resolve", action="store_true")
    ap.add_argument("--chain", default="", help="段ごとの 2×2 の行（JSON lines）だけを書く")
    a = ap.parse_args(argv)
    ext = extract(a.src)
    clocks = [json.loads(l) for l in open(os.path.expanduser(a.clocks))]
    if a.chain:
        me = [c for c in clocks if c.get("side") == "me" and c.get("plan") and c.get("din") and c.get("pdeath") is not None
              and int(c["seed"]) in ext]
        by_turn = {(int(c["seed"]), int(c["who"]), int(c["t"])): c for c in clocks
                   if c.get("side") == "me" and c.get("plan") and c.get("din")}
        n = 0
        with open(os.path.expanduser(a.chain), "w") as fh:
            for i, c in enumerate(me):
                for row in chain_rows(c, ext[int(c["seed"])], by_turn):
                    fh.write(json.dumps(row) + "\n")
                    n += 1
                if (i + 1) % 500 == 0:
                    print("  %d / %d" % (i + 1, len(me)), file=sys.stderr, flush=True)
        print(n)
        return 0
    recs = run(clocks, ext, a.limit, not a.no_resolve)
    with open(os.path.expanduser(a.out), "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(recs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
