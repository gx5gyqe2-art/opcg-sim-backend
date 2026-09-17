"""**理論の点数は勝敗に換算できるか**（T28・橋・読み取り専用）。

ユーザ提案 2026-09-14「**打った手によるなにかしらの累積スコアを定義して、それを勝敗に紐づける**」。
`docs/cpu_theory_gap.md` **§0.3**（設計）・**§0.4**（暫定値の台帳）。

## なぜ要るか——**勝率は 1 局 1 ビットで、分解能が足りない**

いまの検証は最終的にアリーナの勝率に依るが、**384 局で ±35 Elo**・作っている差は **5 Elo 級**
（`measurement.md` §4）＝**原理的に見えない**。**1 手ごとに値が付けば標本が 2 桁増える。**

## 何を足すか

**各判断点で「その場の最善からいくら落としたか」**（`≤ 0`）を席ごとに足す:

```
S  = Σ_t s_t            ΔS = S_自席 − S_相手席        →  ΔS は勝敗を予測するか
```

**要点は `s_t` が「同じ局面の中の差」であること**——**局面の良し悪しはその場で差し引かれる**ので、
`S` は「どれだけ良い局面に居たか」ではなく**「各決断でいくら取りこぼしたか」**の累計になる。

### 攻め側（自席ターンの本体）

```
s_t = θ(実際に打った手) − max_a θ(a)          候補の中の差
```

### 守り側（相手ターンの窓）——**候補リストは使わない**（ユーザ指摘 2026-09-14）

記録の守りの窓には**候補リストが無い**（実測 0%）。しかし**理論は規則を持っている**
（`c(x) ≤ Θ` なら守る）ので、**状態から助言が出て、記録から実際の行動が出る**:

```
守る費用 = c(x)·μ        受ける費用 = Θ·μ
s_t     = −( 実際に払った費用 − min(2 つのうち払えた方) )
```

> **払えなかった行は誤りと数えない**（`measurement.md` §1「〜しないではなく〜できない」）。
> カウンターを持っていてもドンが足りなければ**守れない**ので、
> **`guard_afford` の予算計算（無料カウンター＋ナップサック）をそのまま使う**。
> ブロッカーが居れば無料で守れる。

## 暫定値（§0.4 の台帳・**感度を付けて回す**）

| | 穴 | この器での扱い |
|---|---|---|
| **P2** | イベント・効果の値付けが無い | `--silent {zero,exclude}` ——**両方出して挟む** |
| **P3** | `Θ` が 2 経路で食い違う | `--theta` ——1.15／1.325／1.88 で回す |
| **P1** | `w(状態)` の式が無い | **帯ごとに傾きを出す**（揃えば実害なし） |

## 読み方（事前登録）

- **`ΔS` が勝敗を予測する**＝理論の点数に**勝率への為替レート**が付く
  （`dP(win)/dΔS`）＝**以後は棋譜を数え直すだけで理論の変更を評価できる**。
- **予測しない**＝理論の順序は勝敗と結びついていない＝**橋は架からない**。
- **攻め側だけ効く／守り側だけ効く**なら、**どちらの半分が正しいか**が分かれる。

**限界**: 観察であって因果ではない。**「勝っている席は良い手を打つ余裕がある」**という
選択交絡は行内の差では消えない（帯で層別して見る）。守りの窓は
**そのターン最初の窓だけ**・来る攻撃は**最大のもの**で近似する。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py --in ~/w41 --out ~/bridge.json
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
from order_acc import band_of  # noqa: E402
import effect_value as EV  # noqa: E402
import theory_order as _TOM  # noqa: E402
from theory_order import (own_attackers_of, MU, PWR_EPS, POL_COLS, SC_MY_DON, SC_MY_LEADER_POWER,  # noqa: E402
                          SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE, THETA,
                          c_of, clock_of_row, incoming_x, opp_bodies_of, opp_chars_of, score_candidate,
                          slot_power,
                          theta_of)


def _TO_W_MODE():
    """今の `w` の形（`theory_order.W_MODE`・T49）——`stats` に刻んで数字の出所を残す。"""
    return _TOM.W_MODE


def _d_bin(d):
    """時計の差 `D` の帯（T49 の検算用の分布）。"""
    d = float(d)
    if d < -3.0:
        return "<-3"
    if d < -1.0:
        return "-3..-1"
    if d <= 1.0:
        return "-1..1"
    if d <= 3.0:
        return "1..3"
    return ">3"

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0",
            "sig")   # `sig` は `PL.label_game`（守り側の take/guard の判定）が要る
#: **P3** の感度（§0.4）——盤面経路 1.325・恒等式 1.88・出荷既定 1.15
THETA_SWEEP = (1.15, 1.325, 1.88)
#: **P2** の扱い（理論が値を付けられなかった行）
SILENT_MODES = ("zero", "exclude")
#: **T28-c**——「払えた」の判定が粗いせいで反転しているのではないかを分ける閾値。
#: 守る力が来る攻撃をこれだけ上回っていれば**余裕で払えた**と見なす（暫定値・感度を取る）。
#: **ブロッカーが居る行は無条件で余裕**（レストするだけでドンを使わない）。
MARGIN_COMFORT = 2000.0
#: **守りの窓の `g`（数える価格）の定義**（T62・2026-09-16・ユーザ指示「まずは理論から固める」→「着手してみてください」）。
#: `paid`＝従来＝**−実際に払った額**（守れば `c(x)·μ`・受ければ `Θ·μ`）／`delta`＝**攻め手の価格 − 実際に払った額**
#: （攻め手の価格＝`min(c(x)·μ, Θ·μ)`＝相手が最安の応答をしたときに失う額）。零和なので同じ移転は攻め手の行で 1 回だけ数え、
#: 受け手の行には見積もりとの差分だけを載せる（`paid` は同じ移転を両席で二重に数え、払える席ほど損に見えた＝守り側の `ΔG` が反対向き）。
GUARD_G_MODES = ("paid", "delta")
GUARD_G_MODE = "delta"
#: **守りの窓で「払った額」をどう読むか**（T64・2026-09-16）。`formula`＝式の費用（守れば `c(x)·μ`・受ければ `Θ·μ`）／
#: `spent`＝**実際に手札から消えた札の価値の和**（`hand_spend.use_value`・切った札の機会費用）＋ 受けたなら `Θ·μ`。
#: 手札の中身は記録の枠 12〜21 に在る（P8 を待たずに読める）。`v` が読めない札は `μ` で数える。
#: **既定は `spent`**（T64・2026-09-16・`2026-09-16_hand_spend.md`）: 守り側の `ΔG` が 0.276 → 0.602／0.429 → 0.574・接戦帯 0.33 → 0.64。
#: 帳簿の「払った額」を式の近似から記録の実額に戻しただけ（定数は増えない）。以前の数字と比べるときは `--guard-cost formula`。
GUARD_COST_MODES = ("formula", "spent")
GUARD_COST_MODE = "spent"


def set_guard_cost_mode(mode):
    global GUARD_COST_MODE
    if mode not in GUARD_COST_MODES:
        raise ValueError("guard cost mode は %s のどれか" % (GUARD_COST_MODES,))
    GUARD_COST_MODE = mode
    return GUARD_COST_MODE


def set_guard_g_mode(mode):
    global GUARD_G_MODE
    if mode not in GUARD_G_MODES:
        raise ValueError("guard g mode は %s のどれか" % (GUARD_G_MODES,))
    GUARD_G_MODE = mode
    return GUARD_G_MODE


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32),
            "ci": np.asarray(dd["card_idx"])[:n]}


def guard_step(tok, sc, played, free, paid, theta=THETA, mu=MU, margin_comfort=None, guard_g=None):
    """**守りの窓 1 つ**の取りこぼし（`≤ 0`）と、判定に使った内訳。

    **払えなかった行は誤りと数えない**——`measurement.md` §1。
    `guard_g`（省略時 `GUARD_G_MODE`）: `g` を `paid`（−払った額）か `delta`（攻め手の価格 − 払った額・T62）で返す。
    両方 `g_paid`／`g_delta` としても返す。
    """
    xs = [x for x in incoming_x(tok) if x >= -PWR_EPS]
    if not xs:
        return None
    x = max(xs)                                   # そのターン最大の攻撃で近似
    blocker = bool((np.asarray(tok)[GA.SLOT_OWN_FIELD][:, GA.S_BLOCKER] > 0.5).any())
    budget = int(round(float(sc[SC_MY_DON])))
    afford_pw = free + GA.knapsack(paid, budget)
    can_guard = bool(blocker or afford_pw >= x)
    cost_take = float(theta) * float(mu)
    cost_guard = float(c_of(x)) * float(mu)
    best = min(cost_take, cost_guard) if can_guard else cost_take
    actual = cost_guard if played == "guard" else cost_take
    # **選んだ行動の変化量 `g`**（T40・2026-09-15）＝**実際に払った費用の符号を返したもの**。
    # 「取りこぼし `s`」と違い**誰の責任かを問わない**——払えずに受けた行も損は損として数える
    # （`s` はそこを 0 にする）。守れないはずの行で守った場合も、払ったのは守りの費用。
    g_paid = -float(actual)
    # **T62**: 攻め手の価格＝相手が最安の応答をしたときに失う額（払えるかは攻め手には見えない＝`min` そのもの）
    price = min(cost_take, cost_guard)
    g_delta = float(price) - float(actual)
    g = g_delta if (GUARD_G_MODE if guard_g is None else guard_g) == "delta" else g_paid
    if played == "guard" and not can_guard:
        # 守れないはずの行で守っている＝予算の見積りが渋い。**誤りにしない**
        actual = best
    # **T28-c**: 余裕の大きさ。**貧しい席を減点していないか**を分けるために出す。
    margin = (float("inf") if blocker else float(afford_pw) - float(x))
    return {"s": -max(0.0, actual - best), "g": g, "g_paid": g_paid, "g_delta": g_delta, "price": float(price),
            "x": x, "can_guard": can_guard, "margin": margin,
            "comfortable": bool(can_guard and margin >= (MARGIN_COMFORT
                                                        if margin_comfort is None
                                                        else float(margin_comfort))),
            "played": played, "theory_says": ("guard" if (can_guard and cost_guard < cost_take)
                                              else "take")}


#: 手の型（T40 の内訳用）。**記録に `ATTACK` は無く攻撃は対象付きの `DON_BOX`**（`theory_order` の注記）
MOVE_FAMILIES = ("attack", "attach", "play", "effect", "end", "other")


def move_family(sig):
    """候補の署名 → 手の型。`DON_BOX` は対象が在れば攻撃・無ければ付与。"""
    if not sig:
        return "other"
    at = sig[0]
    tl = sig[2] if len(sig) > 2 else None
    if at == "ATTACK" or (at == "DON_BOX" and tl):
        return "attack"
    if at in ("DON_BOX", "ATTACH_DON"):
        return "attach"
    if at == "PLAY":
        return "play"
    if at == "ACTIVATE_MAIN":
        return "effect"
    if at == "TURN_END":
        return "end"
    return "other"


def _state_of(sc, ci, idx2cid):
    """判断点の状態（条件の判定用）。**記録だけで作れる**——リーダーは `card_idx` の
    0/1（vocab index）・ステージの有無は 22/23。"""
    try:
        import condition_value as CV
    except Exception:
        return None
    ci = np.asarray(ci)
    return CV.state_from_scalars(sc, idx2cid.get(int(ci[0])), idx2cid.get(int(ci[1])),
                                 my_stage=int(ci[22]) > 0, opp_stage=int(ci[23]) > 0)


def _seat_decks(rec_decks, seed, rows, ex, idx, idx2cid, stats=None):
    """**T68**: その局の席ごとのデッキ `{who: [card_id]}`（seed から復元・最初の自席ターンの手札で検算。合わなければ席を落とす）。"""
    if not rec_decks or int(seed) not in rec_decks:
        return {}
    import hand_spend as HS
    import search_price as SP
    mode, leaders = rec_decks[int(seed)]
    out = {}
    for w in (0, 1):
        i = next((i for i in idx if int(rows["who"][i]) == w and int(rows["turn"][i]) >= 1), None)
        if i is None:
            continue
        d = SP.deck_for_seat(seed, mode, leaders, w, HS.hand_ids(ex["ci"][i], idx2cid))
        if stats is not None:
            stats["search_deck_ok" if d else "search_deck_bad"] = stats.get("search_deck_ok" if d else "search_deck_bad", 0) + 1
        if d:
            out[w] = d
    return out


def _search_ctx(sc, tok, ci_row, idx2cid, cards, deck):
    """**T68**: 探す能力の計画価格の状態（`hand_plan.search_context` ＋ `cards`）。デッキが無ければ `None`。"""
    if deck is None or EV.SEARCH_PRICE_MODE != "plan":
        return None
    import hand_plan as HP
    ctx = HP.search_context(sc, tok, ci_row, idx2cid, cards, deck)
    ctx["cards"] = cards
    return ctx


def _add(rec, band, s, side, g=0.0):
    """**行ごとに帯へ足す**（T28-b）——決着後の雑さが接戦帯に混ざらないようにする。

    `side="grdc"` は**余裕で払えた守りの行だけ**の別勘定（T28-c）＝`grd` と二重に足す。
    `g` は**選んだ手の変化量**（T40）——`s`（最善からの逸脱）と並べて別勘定で足す。
    """
    if side != "grdc":
        rec["s_%s" % side] += float(s)
        rec["g_%s" % side] = rec.get("g_%s" % side, 0.0) + float(g)
        rec["n_%s" % side] += 1
    b = rec["band"].setdefault(band, {"s": 0.0, "n": 0, "s_atk": 0.0, "n_atk": 0,
                                      "s_grd": 0.0, "n_grd": 0,
                                      # **T28-c**: 余裕で払えた守りの行だけの集計
                                      "s_grdc": 0.0, "n_grdc": 0,
                                      # **T40**: 選んだ手の変化量
                                      "g": 0.0, "g_atk": 0.0, "g_grd": 0.0, "g_grdc": 0.0})
    b["s"] += float(s); b["n"] += 1
    b["s_%s" % side] += float(s); b["n_%s" % side] += 1
    if side != "grdc":
        b["g"] += float(g)
    b["g_%s" % side] += float(g)


def ledger_value(score, played_v, mode=None):
    """**数える価格**（`g`・`ΔG`）を帳簿の規約で読み直す（**T58**・ユーザ決定 2026-09-16「それでいきましょうか」）。

    `score()` は打った手の価格を今の `FLOW_PRICING` で返す関数。帳簿の規約（省略時 `effect_value.LEDGER_FLOW_PRICING`＝
    `exercise`）が決める側の規約と同じなら呼び直さず `played_v` をそのまま返す。違えばその規約の下で 1 回だけ読み直す
    （付与の行が 0 になり、行使の行に既に載っている分を 2 度数えない＝罠 31）。読み直しが `None` なら `played_v` に戻す。
    """
    mode = EV.LEDGER_FLOW_PRICING if mode is None else mode
    if mode == EV.FLOW_PRICING:
        return played_v
    with EV.flow_pricing(mode):
        v = score()
    return played_v if v is None else v


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const", nu_targets="leader",
            silent="zero", margin_comfort=None, ledger_pricing=None):
    """(局, 席) ごとに攻め側と守り側の取りこぼしを足す。

    `s`（決める＝行内の最善からの逸脱）は `FLOW_PRICING`、`g`（数える＝`ΔG`）は `ledger_pricing`
    （省略時 `effect_value.LEDGER_FLOW_PRICING`）で読む（T58）。
    """
    ledger_pricing = EV.LEDGER_FLOW_PRICING if ledger_pricing is None else ledger_pricing
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    # **T68**: 探す能力の計画価格に要るデッキ（seed から復元・席ごとに手札で検算）
    import search_price as SP
    rec_decks = SP.record_decks(dirs) if EV.SEARCH_PRICE_MODE == "plan" else {}
    per = {}
    stats = {"games": 0, "atk_rows": 0, "atk_silent": 0, "grd_rows": 0, "grd_no_attack": 0,
             # **T28-c**: 余裕で払えた守りの行の数
             "grd_comfortable": 0,
             # **T49**: 局面の傾き `κ = w(D)/w̄`（攻めの行）——平均が 1 に戻るかが `w(状態)` の検算
             "w_mode": _TO_W_MODE(), "sigma_turn": _TOM.SIGMA_TURN, "kappa_sum": 0.0, "kappa_n": 0,
             # **T58**: 決める価格（`s`）と数える価格（`g`）の規約・読み直した行の数
             "flow_pricing": EV.FLOW_PRICING, "ledger_pricing": ledger_pricing, "ledger_rescored": 0,
             # **T62**: 守りの窓の定義と、自ライフごとの内訳（受けた率・理論が受けろと言う率・`g` の平均）
             "guard_g": GUARD_G_MODE, "grd_by_life": {},
             # **T68**: 探す能力の価格の規約と、デッキを復元できた席／できなかった席の数
             "search_price": EV.SEARCH_PRICE_MODE, "search_deck_ok": 0, "search_deck_bad": 0,
             "d_bins": {"<-3": 0, "-3..-1": 0, "-1..1": 0, "1..3": 0, ">3": 0},
             # **`D` の帯ごとの実勝率**（当てはめない）——時計の推定 `D` が勝敗を順序付けるか・
             # 実測の `W(D)` の傾きが置いた `σ_D` と合うかの検算
             "d_win": {"<-3": [0, 0], "-3..-1": [0, 0], "-1..1": [0, 0], "1..3": [0, 0], ">3": [0, 0]}}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        life0 = ex["sc"][:, 0]
        labels, _unk = PL.label_game(rows, pol, life0, L, ptr, idx, cards)
        seen = set()
        decks = _seat_decks(rec_decks, seed, rows, ex, idx, idx2cid, stats)   # T68
        # **T64**: 守りの窓で実際に消えた札を読むため、席ごとの次の自席ターンの最初の main 行を引く
        first_main = {}
        if GUARD_COST_MODE == "spent":
            for i in idx:
                w0, t0 = int(rows["who"][i]), int(rows["turn"][i])
                if t0 >= 1 and PL.is_own_turn(w0, t0) and int(rows["kind"][i]) == 0 and (w0, t0) not in first_main:
                    first_main[(w0, t0)] = i
        for n, i in enumerate(idx):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1:
                continue
            z = float(rows["z"][i])
            key = (seed, w)
            rec = per.setdefault(key, {"seed": seed, "who": w, "z": None,
                                       "s_atk": 0.0, "s_grd": 0.0, "n_atk": 0, "n_grd": 0,
                                       "g_atk": 0.0, "g_grd": 0.0,        # T40
                                       "g_fam": {}, "n_fam": {},          # T40: 手の型ごとの内訳
                                       "n_silent": 0, "v0": [],
                                       # **T28-b: 行ごとに帯を決めてから足す**
                                       "band": {}})
            if z != 0.0:
                rec["z"] = 1.0 if z > 0 else 0.0
            sc, tok = ex["sc"][i], ex["tok"][i]
            if PL.is_own_turn(w, t):
                if int(rows["kind"][i]) != 0:
                    continue
                k = int(L[i])
                ch = int(rows["pol_chosen"][i])
                if k < 2 or ch < 0 or ch >= k:
                    continue
                stats["atk_rows"] += 1
                # **T63**: 攻撃の価格の受ける費用は**相手のライフ**で読む（`by_life`）。`const` なら従来の `Θ`
                th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]),
                              mode=theta_mode, theta=_TOM.theta_take(float(sc[SC_OPP_LIFE]), theta=theta))
                ctx = {"theta": th, "mu": mu,
                       "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                       "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                       "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), "don_k": 1,
                       "attackers": own_attackers_of(tok, float(sc[SC_OPP_LEADER_POWER]) * 1e4),
                       "don_active": float(sc[SC_MY_DON]),   # 登場の機会費用（T43）
                       # **条件の判定に使う状態**（`condition_value.py`・2026-09-14）。
                       # リーダーとステージは `card_idx` の 0/1 と 22/23 に在る。
                       "st": _state_of(sc, ex["ci"][i], idx2cid),
                       # **T68**: 探す能力の計画価格に要る状態（手札・ドンの枠・来る攻撃・デッキ）。デッキが無ければ `sel(k)` に落ちる
                       "search_ctx": _search_ctx(sc, tok, ex["ci"][i], idx2cid, cards, decks.get(w)),
                       # **効果が取れる相手の体**（価格つき・2026-09-15）——
                       # `ν` は動かさず、**効果の値が盤面で変わる**
                       "opp_bodies": opp_bodies_of(
                           tok, float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                           max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), th, mu,
                           # **枠の素性**（特徴・色・名前）＝除去の絞り込みを判定するため
                           ci_row=ex["ci"][i], idx2cid=idx2cid)}
                if nu_targets == "board":
                    ctx["opp_chars"] = opp_chars_of(tok)
                # **T49**: 局面の傾き。価格（平均の傾きで書いた時計の差分）に掛けて `ΔG` に足す
                ck = clock_of_row(sc, tok)
                kap = float(ck["kappa"])
                stats["kappa_sum"] += kap; stats["kappa_n"] += 1
                stats["d_bins"][_d_bin(ck["d"])] += 1
                if z != 0.0:
                    stats["d_win"][_d_bin(ck["d"])][0] += (1 if z > 0 else 0)
                    stats["d_win"][_d_bin(ck["d"])][1] += 1
                b = int(ptr[i])

                def _score(j, _ctx=ctx, _tok=tok):
                    sig = json.loads(pol["pol_sig"][j])
                    tl = sig[2] if len(sig) > 2 else None
                    return score_candidate(sig, str(pol["pol_cid"][j]) or None,
                                           (str(pol["pol_tcid"][j]) or None) if tl else None,
                                           _ctx, cards,
                                           src_power=slot_power(_tok, pol["pol_si"][j]),
                                           tgt_power=slot_power(_tok, pol["pol_ti"][j]),
                                           don_k=pol["pol_k"][j])
                vals = [_score(j) for j in range(b, b + k)]
                scored = [v for v in vals if v is not None]
                played_v = vals[ch]
                bnd = band_of(abs(float(rows["pol_v0"][i])))
                if played_v is None or len(scored) < 2:
                    stats["atk_silent"] += 1
                    rec["n_silent"] += 1
                    if silent == "zero":
                        # **無言の行を「取りこぼし 0」として母数に入れる**
                        _add(rec, bnd, 0.0, "atk")
                    continue                     # `exclude` は母数にも入れない
                # `s`＝最善からの逸脱（≤ 0）・`g`＝選んだ手の理論値そのもの（T40）
                # `κ` は同じ行の全候補に共通なので順位（最善）は動かず、和の重みだけが局面で変わる
                # **T58**: `g` は帳簿の規約（`exercise`）で読み直す——`s`（決める側）は `option` のまま
                g_v = ledger_value(lambda: _score(b + ch), played_v, ledger_pricing)
                if g_v != played_v:
                    stats["ledger_rescored"] += 1          # 規約で値が動いた行（付与・流れの行）だけ数える
                g_row = float(g_v) * kap
                s_row = (float(played_v) - max(scored)) * kap
                _add(rec, bnd, s_row, "atk", g=g_row)
                fam = move_family(json.loads(pol["pol_sig"][b + ch]))
                rec["g_fam"][fam] = rec["g_fam"].get(fam, 0.0) + g_row
                rec["n_fam"][fam] = rec["n_fam"].get(fam, 0) + 1
                # 逸脱も型ごとに（どの型の取りこぼしが橋を運んでいるか・T41）
                rec.setdefault("s_fam", {})[fam] = rec.get("s_fam", {}).get(fam, 0.0) + s_row
                # その行の最善の型（最善が起動効果だった行の割合を読む）
                bf = move_family(json.loads(pol["pol_sig"][b + int(np.argmax([(-1e9 if v is None else v) for v in vals]))]))
                rec.setdefault("best_fam", {})[bf] = rec.get("best_fam", {}).get(bf, 0) + 1
                rec["v0"].append(abs(float(rows["pol_v0"][i])))
            else:
                if (w, t) in seen or int(labels[n]) < 0:
                    continue
                seen.add((w, t))
                played = PL.PLAN_CLASSES[int(labels[n])]
                if played not in ("take", "guard"):
                    continue
                free, paid, _slots = GA.hand_counters(tok, ex["ci"][i], idx2cid, cards)
                # **T63**: 守りの規則の `Θ` も `--theta-mode` に従う（`max`＝経済の `Θ` と生存のシャドー価格の大きい方）。
                # それまでは定数だけ（攻めの行は `theta_of` を通していたのに守りの窓は通していなかった）
                th_g = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]), mode=theta_mode,
                                theta=_TOM.theta_take(float(sc[SC_MY_LIFE]), theta=theta))   # T63: 自分のライフ
                got = guard_step(tok, sc, played, free, paid, th_g, mu, margin_comfort)
                if got is None:
                    stats["grd_no_attack"] += 1
                    continue
                if GUARD_COST_MODE == "spent":
                    # **T64**: 払った額＝実際に消えた札の価値の和（＋受けたなら `Θ·μ`）。次の自席ターンが無ければ式の費用のまま
                    j = first_main.get((w, t + 1))
                    if j is not None:
                        import hand_spend as HS
                        before = HS.hand_ids(ex["ci"][i], idx2cid)
                        after = HS.hand_ids(ex["ci"][j], idx2cid)
                        olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
                        r_opp = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
                        paid_v = 0.0
                        for cid in HS.spent_cards(before, after):
                            v = HS.use_value(cid, cards.info(cid), olp, r_opp)
                            paid_v += float(mu) if v is None else float(v)
                        actual = paid_v + (float(th_g) * float(mu) if played == "take" else 0.0)
                        stats["grd_spent_rows"] = stats.get("grd_spent_rows", 0) + 1
                        stats["grd_spent_sum"] = stats.get("grd_spent_sum", 0.0) + paid_v
                        got["g_paid"] = -actual
                        got["g_delta"] = got["price"] - actual
                        got["g"] = got["g_delta"] if GUARD_G_MODE == "delta" else got["g_paid"]
                stats["grd_rows"] += 1
                gl = stats["grd_by_life"].setdefault(str(int(round(float(sc[SC_MY_LIFE])))),
                                                     {"n": 0, "took": 0, "says_take": 0, "can_guard": 0,
                                                      "g_paid": 0.0, "g_delta": 0.0, "z_win": 0, "z_n": 0})
                gl["n"] += 1; gl["took"] += int(played == "take"); gl["says_take"] += int(got["theory_says"] == "take")
                gl["can_guard"] += int(got["can_guard"]); gl["g_paid"] += got["g_paid"]; gl["g_delta"] += got["g_delta"]
                if z != 0.0:
                    gl["z_win"] += int(z > 0); gl["z_n"] += 1
                bnd = band_of(abs(float(rows["pol_v0"][i])))
                kap = float(clock_of_row(sc, tok)["kappa"])         # T49（守りの窓も同じ傾き）
                _add(rec, bnd, got["s"] * kap, "grd", g=got["g"] * kap)
                if got["comfortable"]:
                    stats["grd_comfortable"] += 1
                    _add(rec, bnd, got["s"] * kap, "grdc", g=got["g"] * kap)   # **余裕で払えた行だけの別勘定**
    return per, stats


def pair_by_band(per, band):
    """**その帯の行だけ**で 2 席を突き合わせる（T28-b の本体）。"""
    by = {}
    for (seed, w), r in per.items():
        by.setdefault(seed, {})[w] = r
    out = []
    for seed, seats in by.items():
        a, b = seats.get(0), seats.get(1)
        if a is None or b is None or a["z"] is None or b["z"] is None:
            continue
        ba, bb = a["band"].get(band), b["band"].get(band)
        if not ba or not bb or ba["n"] < 1 or bb["n"] < 1:
            continue
        out.append({"seed": seed, "z": a["z"],
                    "dS": ba["s"] - bb["s"],
                    "dS_per_row": ba["s"] / ba["n"] - bb["s"] / bb["n"],
                    "dS_atk": (ba["s_atk"] / ba["n_atk"] if ba["n_atk"] else 0.0)
                              - (bb["s_atk"] / bb["n_atk"] if bb["n_atk"] else 0.0),
                    "dS_grd": (ba["s_grd"] / ba["n_grd"] if ba["n_grd"] else 0.0)
                              - (bb["s_grd"] / bb["n_grd"] if bb["n_grd"] else 0.0),
                    # **T28-c**: 両席が「余裕で払えた」行を持つ局だけで意味を持つ
                    "dS_grdc": ((ba["s_grdc"] / ba["n_grdc"] if ba["n_grdc"] else None)
                                if (ba["n_grdc"] and bb["n_grdc"]) else None),
                    "n_grdc": (ba["n_grdc"], bb["n_grdc"]),
                    # **T40**: 選んだ手の変化量の累積（同じ帯の行だけ）
                    "dG": ba.get("g", 0.0) - bb.get("g", 0.0),
                    "dG_per_row": ba.get("g", 0.0) / ba["n"] - bb.get("g", 0.0) / bb["n"],
                    "dG_atk": (ba.get("g_atk", 0.0) / ba["n_atk"] if ba["n_atk"] else 0.0)
                              - (bb.get("g_atk", 0.0) / bb["n_atk"] if bb["n_atk"] else 0.0),
                    "dG_grd": (ba.get("g_grd", 0.0) / ba["n_grd"] if ba["n_grd"] else 0.0)
                              - (bb.get("g_grd", 0.0) / bb["n_grd"] if bb["n_grd"] else 0.0),
                    "n": ba["n"] + bb["n"], "dn": ba["n"] - bb["n"],
                    "silent": a["n_silent"] + b["n_silent"], "v0": None})
    return out


def pair_games(per, silent="zero"):
    """(局) ごとに 2 席を突き合わせて `ΔS` を作る（全帯まとめ・T28 の形）。"""
    by = {}
    for (seed, w), r in per.items():
        by.setdefault(seed, {})[w] = r
    out = []
    for seed, seats in by.items():
        if len(seats) != 2:
            continue
        a, b = seats.get(0), seats.get(1)
        if a is None or b is None or a["z"] is None or b["z"] is None:
            continue
        if silent == "exclude" and (a["n_silent"] or b["n_silent"]):
            # **無言の行があった局を丸ごと外す**のは厳しすぎるので、
            # `exclude` は「無言の行を母数から外す」＝既に加算していない＝ここでは何もしない。
            pass
        na, nb = a["n_atk"] + a["n_grd"], b["n_atk"] + b["n_grd"]
        if na < 1 or nb < 1:
            continue
        ga, gb = (a.get("g_atk", 0.0), a.get("g_grd", 0.0)), (b.get("g_atk", 0.0), b.get("g_grd", 0.0))
        out.append({"seed": seed, "z": a["z"],
                    "dS": (a["s_atk"] + a["s_grd"]) - (b["s_atk"] + b["s_grd"]),
                    "dS_atk": a["s_atk"] - b["s_atk"],
                    "dS_grd": a["s_grd"] - b["s_grd"],
                    # **1 手あたりに正規化した版**——手数の差が `ΔS` を動かすため。
                    # **先手は手番が 1 つ多い**ので、生の和は席順を拾いうる。
                    "dS_per_row": (a["s_atk"] + a["s_grd"]) / na
                                  - (b["s_atk"] + b["s_grd"]) / nb,
                    # **T40**: 選んだ手の変化量の累積 `ΔG`（`s` と同じ 4 つの形で並べる）
                    "dG": (ga[0] + ga[1]) - (gb[0] + gb[1]),
                    "dG_atk": ga[0] - gb[0],
                    "dG_grd": ga[1] - gb[1],
                    "dG_per_row": (ga[0] + ga[1]) / na - (gb[0] + gb[1]) / nb,
                    # 手の型ごとの内訳（攻め側の 1 手あたり・型の和が `dG_atk` の 1 手あたり版になる）
                    "dG_fam": {f: ((a.get("g_fam", {}).get(f, 0.0) / a["n_atk"] if a["n_atk"] else 0.0)
                                   - (b.get("g_fam", {}).get(f, 0.0) / b["n_atk"] if b["n_atk"] else 0.0))
                               for f in MOVE_FAMILIES},
                    "n_fam": (a.get("n_fam", {}), b.get("n_fam", {})),
                    "dS_fam": {f: ((a.get("s_fam", {}).get(f, 0.0) / a["n_atk"] if a["n_atk"] else 0.0)
                                   - (b.get("s_fam", {}).get(f, 0.0) / b["n_atk"] if b["n_atk"] else 0.0))
                               for f in MOVE_FAMILIES},
                    "best_fam": (a.get("best_fam", {}), b.get("best_fam", {})),
                    "n": na + nb, "dn": na - nb,
                    "silent": a["n_silent"] + b["n_silent"],
                    "v0": float(np.mean(a["v0"])) if a["v0"] else None})
    return out


def auc(scores, labels):
    """順位の AUC（**当てはめない**）。同値は 0.5 として数える。"""
    s = np.asarray(scores, np.float64); y = np.asarray(labels, np.float64)
    pos, neg = (y > 0.5), (y <= 0.5)
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def slope(pairs, key="dS"):
    """**為替レート**——`ΔS` 1 単位あたり勝率がどれだけ動くか（最小二乗）。"""
    x = np.array([p[key] for p in pairs], np.float64)
    y = np.array([p["z"] for p in pairs], np.float64)
    if len(x) < 10 or float(x.var()) <= 0.0:
        return None
    xd = x - x.mean()
    return float((xd * (y - y.mean())).sum() / (xd * xd).sum())


def _boot(pairs, reps=200, seed=0, key="dS"):
    if reps <= 0 or len(pairs) < 10:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(reps)):
        sub = [pairs[k] for k in rng.integers(0, len(pairs), len(pairs))]
        v = slope(sub, key)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return ([round(float(np.percentile(vals, 2.5)), 5),
             round(float(np.percentile(vals, 97.5)), 5)] if len(vals) >= 10 else [None, None])


def _verdict(ci):
    """傾きの 95% CI → 判定（事前登録の 3 値＋`undecided`）。"""
    if ci is None or None in ci:
        return "undecided"
    if ci[0] > 0:
        return "bridge_holds"
    if ci[1] < 0:
        return "bridge_inverted"
    return "no_link"


#: 較正表の等分位の数（5＝五分位。局数が少ないので細かくしない）
CALIB_BINS = 5


def calibration(pairs, key="dG_per_row", bins=CALIB_BINS):
    """**較正表**（T40）——`key` を等分位に切り、各分位の**実勝率**を並べる。

    **回帰で係数を決めない**（決めた瞬間に価格の検算が循環する）。「`ΔG` がこの帯なら
    勝率はこれだけ」を**そのまま表にする**だけ。単調なら「勝敗まで説明する」の第一歩。
    """
    xs = np.array([p[key] for p in pairs], np.float64)
    ys = np.array([p["z"] for p in pairs], np.float64)
    if len(xs) < bins * 2:
        return None
    order = np.argsort(xs, kind="mergesort")
    rows = []
    for k in range(bins):
        sel = order[(len(xs) * k) // bins:(len(xs) * (k + 1)) // bins]
        if len(sel) == 0:
            continue
        rows.append({"bin": k, "n": int(len(sel)),
                     "x_lo": round(float(xs[sel].min()), 5), "x_hi": round(float(xs[sel].max()), 5),
                     "x_mean": round(float(xs[sel].mean()), 5),
                     "win_rate": round(float(ys[sel].mean()), 4)})
    wr = [r["win_rate"] for r in rows]
    return {"bins": rows,
            "monotone": bool(all(wr[i] <= wr[i + 1] for i in range(len(wr) - 1))),
            "spread": round(float(wr[-1] - wr[0]), 4) if wr else None}


def summarise(pairs, reps=200, seed=0):
    if len(pairs) < 10:
        return {"n": len(pairs)}
    out = {"games": len(pairs),
           "dS_mean": round(float(np.mean([p["dS"] for p in pairs])), 5),
           "dS_sd": round(float(np.std([p["dS"] for p in pairs])), 5),
           "silent_rows": int(sum(p["silent"] for p in pairs))}
    # **手数の差が勝敗を説明していないか**——説明していれば `ΔS` は席順を拾っている
    out["dn_mean"] = round(float(np.mean([p["dn"] for p in pairs])), 3)
    out["dn_auc"] = (round(auc([p["dn"] for p in pairs], [p["z"] for p in pairs]), 4)
                     if auc([p["dn"] for p in pairs], [p["z"] for p in pairs]) is not None
                     else None)
    # **T28-c**: 余裕で払えた守りの行だけで引き直す（貧しい席の減点を除く）
    cf = [p for p in pairs if p.get("dS_grdc") is not None and p.get("n_grdc")]
    if len(cf) >= 30:
        for p in cf:
            a, b = p["n_grdc"]
            p["dS_grdc_pr"] = p["dS_grdc"] - 0.0      # 既に 1 手あたり
        out["grd_comfortable_only"] = {
            "games": len(cf),
            "auc": (round(auc([p["dS_grdc"] for p in cf], [p["z"] for p in cf]), 4)
                    if auc([p["dS_grdc"] for p in cf], [p["z"] for p in cf]) is not None
                    else None),
            "slope": (round(slope(cf, "dS_grdc"), 5) if slope(cf, "dS_grdc") else None),
            "slope_ci95": _boot(cf, reps, seed, "dS_grdc")}
    keys = ["dS", "dS_per_row", "dS_atk", "dS_grd"]
    has_g = all("dG_per_row" in p for p in pairs)
    if has_g:
        keys += ["dG", "dG_per_row", "dG_atk", "dG_grd"]
    for key in keys:
        out[key] = {"auc": (round(auc([p[key] for p in pairs],
                                      [p["z"] for p in pairs]), 4)
                            if auc([p[key] for p in pairs], [p["z"] for p in pairs])
                            is not None else None),
                    "slope": (round(slope(pairs, key), 5)
                              if slope(pairs, key) is not None else None),
                    "slope_ci95": _boot(pairs, reps, seed, key)}
    if has_g:
        # **T40**: `ΔG`（選んだ手の変化量）の判定と**較正表**（当てはめない——等分位ごとの実勝率）
        out["gain_verdict"] = _verdict(out["dG_per_row"]["slope_ci95"])
        out["calibration"] = {k: calibration(pairs, k) for k in ("dG_per_row", "dS_per_row")}
    # **P1 の検査**——帯ごとに傾きが揃えば `w` の変動は実害なし
    out["by_band"] = {}
    for nm in ("close", "mid", "decided"):
        sub = [p for p in pairs if p["v0"] is not None and band_of(p["v0"]) == nm]
        if len(sub) >= 50:
            out["by_band"][nm] = {"n": len(sub), "slope": (round(slope(sub), 5)
                                                           if slope(sub) else None)}
    a = out["dS"]["auc"]
    # **事前登録**: `ΔS` が勝敗を予測するか（傾きの CI が 0 を含まないか）——**正規化した版で判定する**
    out["verdict"] = _verdict(out["dS_per_row"]["slope_ci95"])
    out["auc_all"] = a
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA, help="**P3** の暫定値（§0.4）")
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--nu-targets", default="leader", choices=("leader", "board"))
    ap.add_argument("--silent", default="zero", choices=SILENT_MODES,
                    help="**P2** の暫定値——`zero` は無言の行を取りこぼし 0 として母数に入れる／"
                         "`exclude` は母数にも入れない")
    ap.add_argument("--margin-comfort", type=float, default=MARGIN_COMFORT,
                    help="**T28-c** の暫定値——守る力が来る攻撃をこれだけ上回れば「余裕で払えた」")
    ap.add_argument("--nu-mode", default=None, choices=("base", "pair"),
                    help="`ν` の形。**省略時は `theory_order.NU_MODE`（2026-09-15 から `pair`）**。"
                         "**2026-09-15 より前の数字と比べるときは `base` を明示する**")
    ap.add_argument("--cond-unknown", type=float, default=1.0,
                    help="**判らない条件の係数**（§0.4 の感度。1.0＝上限・0.0＝下限）")
    ap.add_argument("--w-mode", default=None, choices=("flat", "clock"),
                    help="**T49** 局面の傾き `κ = w(D)/w̄` を掛けるか。省略時は `theory_order.W_MODE`"
                         "（既定 `flat`＝`κ = 1`・盤面の時計では `clock` は説明力を落とした）")
    ap.add_argument("--sigma-turn", type=float, default=None,
                    help="**T49 の感度**——時計 1 本のぶれ（ターン）。既定は写し（1.0）。合わせ込みには使わない")
    _TOM.add_surv_mode_arg(ap)
    _TOM.add_cbar_mode_arg(ap)
    _TOM.add_take_mode_arg(ap)
    ap.add_argument("--guard-cost", default=None, choices=GUARD_COST_MODES,
                    help="**守りの窓の払った額**（T64）。`spent`＝実際に消えた札の価値の和（＋受けたなら Θ·μ・2026-09-16 から既定）／"
                         "`formula`＝式の費用（以前の数字と比べるとき）")
    ap.add_argument("--guard-g", default=None, choices=GUARD_G_MODES,
                    help="**守りの窓の `g`**（T62）。省略時は `GUARD_G_MODE`（2026-09-16 から `delta`＝攻め手の価格 − 払った額）。"
                         "以前の数字と比べるときは `paid` を明示する")
    ap.add_argument("--flow-pricing", default=None, choices=EV.FLOW_PRICING_MODES,
                    help="**決める価格**（`s`・`ΔS`）の規約。省略時は `effect_value.FLOW_PRICING`（`option`）")
    ap.add_argument("--ledger-pricing", default=None, choices=EV.FLOW_PRICING_MODES,
                    help="**数える価格**（`g`・`ΔG`）の規約（T58）。省略時は `effect_value.LEDGER_FLOW_PRICING`"
                         "（`exercise`＝使った行で 1 回・ユーザ決定 2026-09-16）。"
                         "**2026-09-16 より前の `ΔG` と比べるときは `option` を明示する**")
    EV.add_search_price_arg(ap)
    EV.add_play_now_arg(ap)
    import hand_plan as _HP
    _HP.add_inflow_arg(ap)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    EV.apply_search_price(a)
    EV.apply_play_now(a)
    _HP.apply_inflow_mode(a)

    try:
        import condition_value as CV
        CV.set_unknown_factor(a.cond_unknown)
    except Exception:
        pass
    import theory_order as _TO
    if a.nu_mode is not None:
        _TO.set_nu_mode(a.nu_mode)
    if a.w_mode is not None:
        _TO.set_w_mode(a.w_mode)
    if a.sigma_turn is not None:
        _TO.set_sigma_turn(a.sigma_turn)
    if a.flow_pricing is not None:
        EV.set_flow_pricing(a.flow_pricing)
    _TOM.apply_surv_mode(a)
    _TOM.apply_cbar_mode(a)
    _TOM.apply_take_mode(a)
    if a.guard_g is not None:
        set_guard_g_mode(a.guard_g)
    if a.guard_cost is not None:
        set_guard_cost_mode(a.guard_cost)
    t0 = time.time()
    per, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode, a.nu_targets,
                         a.silent, a.margin_comfort, ledger_pricing=a.ledger_pricing)
    # **T49 の検算**: `κ` の平均（`w` の平均が `w̄` に戻れば 1）
    stats["kappa_mean"] = (round(stats["kappa_sum"] / stats["kappa_n"], 4) if stats["kappa_n"] else None)
    stats["w_mean"] = (round(stats["kappa_mean"] * _TO.W_BAR, 4) if stats["kappa_mean"] is not None else None)
    pairs = pair_games(per, a.silent)
    res = {"stats": stats,
           "provisional": {"P3_theta": a.theta, "P2_silent": a.silent,
                           "T28c_margin": a.margin_comfort, "w_mode": _TO.W_MODE,
                           "flow_pricing": stats["flow_pricing"], "ledger_pricing": stats["ledger_pricing"],
                           "surv_mode": _TO.SURV_MODE, "nu_mode": _TO.NU_MODE,
                           "cbar_mode": _TO.CBAR_MODE, "guard_g": GUARD_G_MODE, "take_mode": _TO.TAKE_MODE,
                           "guard_cost": GUARD_COST_MODE, "search_price": EV.SEARCH_PRICE_MODE,
                           "play_now": EV.PLAY_NOW_MODE, "inflow": _HP.INFLOW_MODE,
                           "note": "§0.4 の暫定値。感度を付けて読む"},
           "summary": summarise(pairs, a.boot_reps, a.seed),
           # **T28-b: 行ごとに帯で切ってから足した版**（判定の主はこちら）
           "per_band": {nm: summarise(pair_by_band(per, nm), a.boot_reps, a.seed)
                        for nm in ("close", "mid", "decided")},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
