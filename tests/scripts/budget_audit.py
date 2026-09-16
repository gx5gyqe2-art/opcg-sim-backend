"""ライフ予算の監査——**攻撃 1 回ごと**に領域・順序の誤り・単価・貪欲の差を数える
（`docs/life_budget.md` §6 の 1〜3 の実装・読み取り専用・ネットを使わない）。

`life_budget.md` は式を書いたが、**その式で実プレイを勘定した計器が無かった**（2026-09-13 まで）。
本器がそれ。式（同 §1〜§4・§18）:

```
受ける: ライフ −1・手札 +1・トリガー +τ        守る: 手札 −c(x)
G = max(0, N − L − B)      N ≈ A·T            S ≈ H + (N−G) + T − K − G·c̄
S(受ける) − S(守る) = c(x) − c̄                θ(deck, state) = A·T − B
領域 1: G = 0 → 受ける／領域 2: S ≥ 0 → c(x) < c̄ なら守る／領域 3: S < 0 → 別の出口
```

### 攻撃 1 回の単位で取る（**過去の読みより細かい**）

`plan_labels` の take／guard は**相手ターン 1 つの単位**（自席 t−1 と t+1 の開始ライフの差）なので、
「2 回殴られて 1 回守り 1 回受けた」ターンは take に潰れる。予算の式は**攻撃 1 回ごと**の話なので、
本器は行の `kind` と `sig` から窓を切り直す:

- 守り側の `kind==1` の行＝**その攻撃の窓の入口**（盤面はまだ払う前）
- 続く `kind==2` の行＝同じ窓の中の追加の支払い
- 窓の中の `SELECT_COUNTER` の数＝**実際に払った枚数 `c_actual`**・`SELECT_BLOCKER` はブロック

＝`2026-09-13_can_or_wont.md`／`time_plan_map` の守りの数字は**ターン単位の近似**だったことになる。

### 出すもの

- `regimes` … 領域 1／2／3 の内訳と、領域ごとの「最適どおり打った割合」
- `order_errors`（領域 2 のみ） … `c(x) > c̄` を守った割合／`c(x) < c̄` を受けた割合（**予算上の誤り**）
- `pressure` … 超過パワーを 1000 刻みにした帯ごとの `c_actual`・守り率
  ＝**ユーザの「7000 理論」`pressure(k) = c(x+1000k) − c(x) ≈ k` の実測**
- `greedy_gap` … `c_actual − c_min(x)`（**払い過ぎた枚数**・`life_budget.md` §16 の「入口コミット」の差）
- `cbar` … 帯ごとの平均単価 `c̄`（§4 の θ を作る材料）・`theta` … `A·T − B` と `L − θ` 別の守り率
- `by_deck` … 色／注入テンプレート／リーダーのライフ別の `c̄`・`θ`・守り率（**しきい値はデッキごと**の検査）

### 近似（読むときに必ず添える）

- `x` は**その行の相手の枠の最大パワー − 自リーダー**（`race_state._incoming`）＝実際に殴ってきた枠とは
  限らない。`A` は相手のリーダー＋場のキャラ数（起き上がりの判定を入れない＝**上限側**）。
- `B` は**今の起動ブロッカーの数**を「止められる回数」として使う（将来 KO されるぶんは引かない＝上限側）。
- `K`（攻めに要る枚数）は 0 とする＝`S` は**上限**＝領域 3 を過小に数える（§6-3 の「誤りの下限」と同じ扱い）。
- `c(x)` は手札の**印字カウンター**だけで作る（【カウンター】イベントは DON が要る・§18）。
  `--with-events` で「払えるイベントも足した版」も出せる。
- `c̄` は**同じ帯の母集団平均**（その局の将来を覗かない）。帯は `(ターン帯, 自分のライフ)`。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/budget_audit.py \\
    --in ~/w32/n_records/n32_w* --holdout-mod 0 --out ~/budget_w32.json
"""
import argparse
import collections
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

from guard_afford import hand_counters  # noqa: E402  無料／有料の分解はここが正本
from plan_value_map import _pad  # noqa: E402
from race_state import (_extra as _race_extra, _incoming,  # noqa: E402
                        _SLOT_OPP, _T_PWR, _T_CHAR, _PWR_SCALE)
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train import time_labels as TL  # noqa: E402

ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "deck_kinds")
POL_COLS = ("pol_sig", "pol_cid", "pol_tcid")
#: scalars（`encode/scalars.rs`）
SC_L, SC_OPP_L, SC_DON, SC_H, SC_OPP_H = 0, 1, 2, 6, 7
#: tokens の列と枠（`n_rel_feat.S_COLS`）
S_BLOCKER, S_COUNTER, S_IS_CHAR = 6, 7, 18
SLOT_OWN_FIELD, SLOT_OPP_FIELD, SLOT_HAND = slice(2, 7), slice(7, 12), slice(12, 22)
COUNTER_SCALE = 2000.0
TURN_BANDS = ("T<=4", "T5-8", "T9+")
CLOCK_BANDS = ("t<=2", "t3-4", "t5+")
C_TLEFT = 0
PRESSURE_BANDS = ("p<=0", "p1k", "p2k", "p3k", "p4k", "p5k+")
#: パワーの許容（power 単位）。符号化は f16 なので戻したパワーは 2000 が 2000.0002 になる。
#: **許容なしでは「2000 のカウンター 1 枚で 2000 の攻撃を止められない」と数える**＝`c_min` が
#: 系統的に +1 される（実データの大半がちょうど 1000 の倍数）。10 は実カードの刻みよりはるかに小さい。
PWR_EPS = 10.0


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def clock_band(t):
    return "t<=2" if t <= 2.5 else ("t3-4" if t <= 4.5 else "t5+")


def pressure_band(x):
    """超過パワー → カウンター 1 枚（1000）刻みの帯（[`PWR_EPS`] の許容つき）。"""
    if x is None:
        return None
    if x <= PWR_EPS:
        return "p<=0"
    k = int(np.ceil((x - PWR_EPS) / 1000.0))
    return f"p{k}k" if k <= 4 else "p5k+"


def _extra(dd, n):
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    _sc_race, tk_race = _race_extra(dd, n)
    return {"sc": sc, "tok": tok, "race_tk": tk_race,
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64),
            "lives": np.asarray(dd["scalars"])[:n, 0:2].astype(np.float32)}


def c_min(values, x):
    """超過パワー `x` を止めるのに要る**最小の枚数**（大きい値から取る＝枚数最小は貪欲が最適）。

    `values` はカウンター値のリスト（power 単位）。止められないなら None。
    """
    if x is None or x <= PWR_EPS:
        return 0
    got = 0.0
    n = 0
    for v in sorted((float(v) for v in values if float(v) > 0), reverse=True):
        got += v
        n += 1
        if got >= x - PWR_EPS:            # f16 の丸めで「ちょうど足りる」を落とさない
            return n
    return None


def incoming(tk_row, attacker_is_leader=None):
    """飛んでくる攻撃の超過パワー。**殴ってきた枠が分かっているならそれを使う**。

    `race_state._incoming` は相手の枠の**最大**パワーを取る（誰が殴ったか分からない前提）。
    ここでは攻撃宣言の `sig` から「殴ったのはリーダーか」が分かるので、分かっているときは
    その枠のパワーで測る＝`c(x)` の過大評価（＝`c_actual < c_min` の主因）を消す。
    `attacker_is_leader` が None なら従来どおり最大を取る。
    """
    if attacker_is_leader is None:
        return _incoming(tk_row)
    tk = np.asarray(tk_row)
    mine = float(tk[0, _T_PWR]) * _PWR_SCALE
    if attacker_is_leader:
        top = float(tk[1, _T_PWR]) * _PWR_SCALE
    else:
        opp = tk[_SLOT_OPP]
        ch = opp[:, _T_CHAR] > 0.5
        if not ch.any():
            return None
        top = float(opp[ch, _T_PWR].max()) * _PWR_SCALE
    return None if top <= 0.0 else top - mine


def attackers(tok_row):
    """`A` … 相手の 1 ターンあたりの攻撃回数（リーダー＋場のキャラ数）。起動判定を入れない上限。"""
    opp = np.asarray(tok_row)[SLOT_OPP_FIELD]
    return 1 + int((opp[:, S_IS_CHAR] > 0.5).sum())


def blockers(tok_row):
    """`B` … 起動中のブロッカーの数（止められる回数の上限）。"""
    own = np.asarray(tok_row)[SLOT_OWN_FIELD]
    return int((own[:, S_BLOCKER] > 0.5).sum())


def budget(L, H, A, T, B, cbar, K=0.0):
    """§2 の予算。戻り値は (N, G, S)。"""
    N = float(A) * float(T)
    G = max(0.0, N - float(L) - float(B))
    S = float(H) + (N - G) + float(T) - float(K) - G * float(cbar)
    return N, G, S


def regime_of(G, S):
    """§3 の領域。1＝全部受けても生き残る／2＝守る枠があり足りている／3＝足りない。"""
    if G <= 0.0:
        return 1
    return 2 if S >= 0.0 else 3


def optimal_action(regime, cx, cbar):
    """領域ごとの最適。領域 2 だけ単価で決まる。`cx` が None（止められない）なら受けるしかない。"""
    if regime == 1:
        return "take"
    if cx is None:
        return "take"
    if regime == 2:
        return "guard" if cx < cbar else "take"
    return "other"                       # 領域 3＝別の出口（この計器では最適を決めない）


def windows_of(rows, pol, ex, L, ptr, idx, cards, idx2cid, with_events=False):
    """1 局 → 守り側の**攻撃 1 回ごと**の記録（窓の入口の盤面＋実際に払った枚数）。

    **攻撃の的（リーダーか キャラか）も持つ**: 予算の式はライフの話なので、キャラを殴られた窓を
    混ぜると閾値（`x`）が別物になる（実測で `c_actual < c_min` が多発した原因）。的は
    直前の攻撃側の行の `sig`（`ATTACK`／`DON_BOX` の対象）を `plan_labels.move_class` で
    face／board に分ける＝打った手の分類と同じ規則。
    """
    tm, tmask = TL.label_game(rows, ex["lives"], idx)
    u2c = PL.uuid_map(pol, L, ptr, idx)
    zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
    out = []
    cur = None
    target = None
    atk_leader = None
    last_turn = None
    for n, i in enumerate(idx):
        w = int(rows["who"][i]); t = int(rows["turn"][i]); k = int(rows["kind"][i])
        try:
            sj = json.loads(rows["sig"][i])
            at = sj[0]
        except Exception:                                  # noqa: BLE001
            sj, at = None, ""
        if t != last_turn:                                 # ターンが変わったら窓と的を捨てる
            cur, target, atk_leader, last_turn = None, None, None, t
        if t < 1:
            continue
        if PL.is_own_turn(w, t):
            # **手番側（＝攻撃側）の行**。攻撃宣言なら的を覚える（`don` は配分だけで的が無い）。
            # 的は「この後に開く守り側の窓」のもの＝窓の入口より先に読める。
            if at in ("ATTACK", "DON_BOX") and sj is not None:
                cl = PL.move_class(sj, u2c, cards)
                if cl in ("face", "board"):
                    target = cl
                    src = cards.info(u2c.get(sj[1])) if len(sj) > 1 else None
                    atk_leader = None if src is None else bool(src["leader"])
            continue
        if k == 1:
            # 新しい窓の入口（盤面はまだ払う前）
            tok = ex["tok"][i]
            sc = ex["sc"][i]
            free, paid, _slots = hand_counters(tok, ex["ci"][i], idx2cid, cards)
            hand_vals = [float(v) * COUNTER_SCALE
                         for v in np.asarray(tok)[SLOT_HAND][:, S_COUNTER] if float(v) > 0]
            x = incoming(ex["race_tk"][i], atk_leader)
            cur = {
                "turn": t, "who": w, "z": 1.0 if zs.get(w, 0.0) > 0 else 0.0,
                "L": float(sc[SC_L]), "H": float(sc[SC_H]), "don": float(sc[SC_DON]),
                "A": attackers(tok), "B": blockers(tok),
                "T": (float(tm[n][C_TLEFT]) * TL.T_SCALE) if tmask[n] else None,
                "x": x, "free": free, "paid_n": len(paid),
                # `c(x)` は印字カウンターだけ（§18）。`--with-events` は払えるイベントも足す
                "c_min": c_min(_free_values(tok, ex["ci"][i], idx2cid, cards)
                               if not with_events else hand_vals, x),
                "c_actual": 0, "blocked": False, "target": target,
                "atk_leader": atk_leader,
                "deck": _deck_key(rows, i),
            }
            out.append(cur)
        if cur is None:
            continue
        if at == "SELECT_COUNTER":
            cur["c_actual"] += 1
        elif at == "SELECT_BLOCKER":
            cur["blocked"] = True
        elif at == "PASS" and k == 2:
            cur = None                                     # この窓は閉じた
    return out


def _free_values(tok_row, ci_row, idx2cid, cards):
    """手札の**印字カウンター**の値だけ（イベントの上げ幅を含めない）。

    語彙を引けない枠は**符号化の値に落とす**（`guard_afford.hand_counters` と同じ方針）。
    0 として捨てると「止められない窓」を実際より多く数えてしまう＝`cant_stop` が水増しになる。
    落とした値は `max(印字, イベントの上げ幅)` なので、その枠だけ**過大**に働く。
    """
    tok = np.asarray(tok_row)
    hand = tok[SLOT_HAND]
    hand_ci = np.asarray(ci_row)[SLOT_HAND]
    vals = []
    for j in range(hand.shape[0]):
        if float(np.abs(hand[j]).sum()) <= 0.0 or float(hand[j, S_COUNTER]) <= 0.0:
            continue
        cid = idx2cid.get(int(hand_ci[j]))
        inf = cards.info(cid) if cid else None
        if inf is None:
            vals.append(float(hand[j, S_COUNTER]) * COUNTER_SCALE)
            continue
        printed = float(inf.get("counter") or 0.0)
        if printed > 0:
            vals.append(printed)
    return vals


def _deck_key(rows, i):
    """デッキの層（色・注入テンプレート・注入率）。`deck_kinds` が無ければ None。"""
    try:
        d = json.loads(str(rows["deck_kinds"][i]) or "{}")
    except Exception:                                      # noqa: BLE001
        return None
    if not d:
        return None
    return {"colors": "+".join(sorted(d.get("colors") or [])),
            "templates": "+".join(sorted(d.get("templates") or [])) or "none",
            "rate": int(d.get("rate") or 0)}


def collect(dirs, holdout_mod=0, limit_games=0, with_events=False):
    cards = PL.Cards()
    from opcg_sim.learned.vocab import shared_vocab
    idx2cid = {v: k for k, v in shared_vocab().items()}
    wins = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        wins.extend(windows_of(rows, pol, ex, L, ptr, idx, cards, idx2cid, with_events))
    return wins, games


def cbar_table(wins):
    """帯 `(ターン帯, ライフ)` ごとの平均単価 `c̄`（止められた窓の `c_min` の平均）。"""
    acc = collections.defaultdict(list)
    for w in wins:
        if w["c_min"]:
            acc[(turn_band(w["turn"]), int(round(w["L"])))].append(w["c_min"])
    table = {k: float(np.mean(v)) for k, v in acc.items() if len(v) >= 20}
    glob = [w["c_min"] for w in wins if w["c_min"]]
    table["_global"] = float(np.mean(glob)) if glob else 1.0
    return table


def cbar_of(table, w):
    return table.get((turn_band(w["turn"]), int(round(w["L"]))), table["_global"])


def annotate(wins, table):
    """各窓に予算・領域・最適・実際を書き込む。"""
    for w in wins:
        cbar = cbar_of(table, w)
        T = w["T"] if w["T"] is not None else 1.0
        N, G, S = budget(w["L"], w["H"], w["A"], T, w["B"], cbar)
        w.update(cbar=cbar, N=N, G=G, S=S, theta=w["A"] * T - w["B"])
        w["regime"] = regime_of(G, S)
        w["optimal"] = optimal_action(w["regime"], w["c_min"], cbar)
        w["played"] = "guard" if (w["c_actual"] > 0 or w["blocked"]) else "take"
        w["agree"] = (w["played"] == w["optimal"]) if w["optimal"] in ("take", "guard") else None
        w["greedy_gap"] = ((w["c_actual"] - w["c_min"])
                           if (w["c_actual"] > 0 and w["c_min"]) else None)
    return wins


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def block(wins):
    if not wins:
        return None
    out = {"n": len(wins),
           "guard_rate": _rate([w["played"] == "guard" for w in wins]),
           "winrate": _rate([w["z"] for w in wins]),
           "c_actual_mean": round(float(np.mean([w["c_actual"] for w in wins])), 3),
           "c_min_mean": round(float(np.mean([w["c_min"] for w in wins if w["c_min"]])), 3)
           if any(w["c_min"] for w in wins) else None,
           "cant_stop": _rate([w["c_min"] is None for w in wins]),
           "cbar_mean": round(float(np.mean([w["cbar"] for w in wins])), 3),
           "theta_mean": round(float(np.mean([w["theta"] for w in wins])), 2),
           "agree": _rate([w["agree"] for w in wins])}
    gg = [w["greedy_gap"] for w in wins if w["greedy_gap"] is not None]
    out["greedy_gap_mean"] = round(float(np.mean(gg)), 3) if gg else None
    out["greedy_gap_pos"] = _rate([g > 0 for g in gg]) if gg else None
    out["greedy_gap_n"] = len(gg)
    return out


def regimes(wins):
    out = {}
    for r in (1, 2, 3):
        sub = [w for w in wins if w["regime"] == r]
        if not sub:
            continue
        b = block(sub)
        b["share"] = round(len(sub) / len(wins), 4)
        if r == 2:
            # **予算上の順序の誤り**（領域 2 だけ定義される）
            exp = [w for w in sub if w["c_min"]]
            b["order_errors"] = {
                "guarded_expensive": _rate([w["played"] == "guard" for w in exp
                                            if w["c_min"] > w["cbar"]]),
                "took_cheap": _rate([w["played"] == "take" for w in exp
                                     if w["c_min"] < w["cbar"]]),
                "n_expensive": sum(1 for w in exp if w["c_min"] > w["cbar"]),
                "n_cheap": sum(1 for w in exp if w["c_min"] < w["cbar"]),
            }
        out[str(r)] = b
    return out


def pressure(wins):
    """**7000 理論の実測**: 超過パワーの帯ごとに、払った枚数と守り率。"""
    out = {}
    for b in PRESSURE_BANDS:
        sub = [w for w in wins if pressure_band(w["x"]) == b]
        if len(sub) < 20:
            continue
        g = [w for w in sub if w["played"] == "guard"]
        out[b] = {"n": len(sub), "guard_rate": round(len(g) / len(sub), 4),
                  "c_actual_mean_when_guard": (round(float(np.mean([w["c_actual"] for w in g])), 3)
                                               if g else None),
                  "c_min_mean": (round(float(np.mean([w["c_min"] for w in sub if w["c_min"]])), 3)
                                 if any(w["c_min"] for w in sub) else None),
                  "cant_stop": _rate([w["c_min"] is None for w in sub]),
                  "winrate_guard": _rate([w["z"] for w in g]),
                  "winrate_take": _rate([w["z"] for w in sub if w["played"] == "take"])}
    return out


def theta_curve(wins):
    """`L − θ` 別の守り率（§4 の「`L < θ` で守りが要る」の検査）。"""
    out = {}
    for w in wins:
        k = int(np.clip(round(w["L"] - w["theta"]), -4, 4))
        out.setdefault(k, []).append(w)
    return {str(k): {"n": len(v), "guard_rate": _rate([x["played"] == "guard" for x in v]),
                     "winrate": _rate([x["z"] for x in v])}
            for k, v in sorted(out.items()) if len(v) >= 20}


def by_deck(wins, key, min_n=200):
    """デッキの層ごとの `c̄`・`θ`・守り率（**しきい値はデッキごと**の検査）。"""
    acc = collections.defaultdict(list)
    for w in wins:
        d = w.get("deck")
        if not d:
            continue
        acc[str(d.get(key))].append(w)
    return {k: {"n": len(v), "cbar": round(float(np.mean([x["cbar"] for x in v])), 3),
                "theta": round(float(np.mean([x["theta"] for x in v])), 2),
                "guard_rate": _rate([x["played"] == "guard" for x in v]),
                "c_actual_mean": round(float(np.mean([x["c_actual"] for x in v])), 3),
                "agree": _rate([x["agree"] for x in v])}
            for k, v in sorted(acc.items(), key=lambda kv: -len(kv[1])) if len(v) >= min_n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=0, help="0＝全局（ネットを使わないので既定は全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--with-events", action="store_true",
                    help="`c(x)` に【カウンター】イベントの上げ幅も入れる（DON は見ない＝上限側）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    wins, games = collect(args.src, args.holdout_mod, args.limit_games, args.with_events)
    table = cbar_table(wins)
    annotate(wins, table)
    face = [w for w in wins if w["target"] != "board"]
    out = {"games": games, "windows": len(wins), "src": list(args.src),
           "with_events": bool(args.with_events),
           # **主集計はリーダーへの攻撃だけ**（予算の式はライフの話・キャラを殴られた窓は閾値が別物）
           "target_mix": {str(k): sum(1 for w in wins if w["target"] == k)
                          for k in ("face", "board", None)},
           "windows_face": len(face),
           "all": block(face), "regimes": regimes(face),
           "pressure": pressure(face), "theta_curve": theta_curve(face),
           "by_clock": {b: block([w for w in face
                                  if w["T"] is not None and clock_band(w["T"]) == b])
                        for b in CLOCK_BANDS},
           "by_turn": {b: block([w for w in face if turn_band(w["turn"]) == b])
                       for b in TURN_BANDS},
           "by_deck_colors": by_deck(face, "colors"),
           "by_deck_templates": by_deck(face, "templates"),
           "board_windows": block([w for w in wins if w["target"] == "board"]),
           "cbar_global": round(table["_global"], 3),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
