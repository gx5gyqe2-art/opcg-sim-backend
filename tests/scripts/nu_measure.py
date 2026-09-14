"""**`ν` を種類別に直接測る**（式の検算・読み取り専用）。

ユーザ指示 2026-09-14「とりあえず検算してみましょうか」。`ν`（場のキャラ 1 体の価格）は
**式を組み立てる前から実測値が在る**——Phase 1a の `hand_value_slope --axis field` が
**0.1087**（勝率/体）を出している。つまり:

> **どんな式を作っても、平均が 0.1087 に合わなければ間違い。**

本器はそれを**種類別に**測って、**項ごとに検算できる**ようにする。やり方は
`hand_value_slope --axis field` と同じ within 推定だが、**説明変数を 1 本から複数本にする**:

```
z  ~  Σ_c  β_c · (種類 c のキャラの体数)        （帯の中で・対局でクラスタした SE）
```

`β_c` が**種類 c のキャラ 1 体の実測価格**＝式の予測と直接比べられる。
1 本だけにすれば Phase 1a の `slope_true` と一致するはず（**器の突き合わせ**になる）。

## 種類の分け方（`--scheme`）

| scheme | 種類 | 何を検算するか |
|---|---|---|
| `all` | 全キャラ 1 本 | **Phase 1a の 0.1087 を再現するか**（器の健全性） |
| `blocker` | ブロッカー／非ブロッカー | ブロック項（実測 `block_p` = 0.773・ブロッカーは 10.4%） |
| `power` | リーダー未満／〜+2000／+2000 以上 | **パワーの階段**（式は 0／0.157／0.180 の 3 値） |
| `power_blocker` | 上の 3 × ブロッカー 2 = 6 | 交互作用（高パワーのブロッカーは繰り返し使えるか） |
| **`lt_split`** | **リーダー未満を「素の体／効果持ち」に割る**＋それ以上 | **T21＝`ν` 最大の穴 0.0690 の中身**（下記） |
| `lt_detail` | 同じ帯を 素／ブロッカー／アタック時／起動／その他 に割る | どの効果が持っているか（分解能は落ちる） |

### T21——**リーダー未満の 0.0690 は効果か、体そのものか**（2026-09-14）

式は `attack_value(x<0) = 0` から **`ν = 0`** と置くが、実測は **0.0690**（CI が 0 を含まない）で
**体数の 40%** がここに居る＝**`ν` 最大の穴**。候補は `game_theory.md` §14.1.1 の
**身代わり（#7）・常在効果（#13）・アタック時効果（#14）**。

**効果を持つかは枠の列で判る**（カード ID に依らない）: `is_blocker_active`・`trig_attack`・
`trig_ko`・`trig_opp_attack`・`act_avail` の 5 つ。どれも立っていない枠が**素の体**で、
**実測でこの帯の 63%** を占める（`threat_next` は連続値かつ盤面由来なので入れない）。

**読み方（事前登録・後から解釈を選ばないために先に書く）**:

- **`lt_plain` が 0.069 前後を持つなら効果説は棄却**＝価値は**体そのもの**に在る
  （身代わり・枠・ドンの器・他の札の「N 体以上」を満たす）。
- **`lt_plain` が 0 に近く効果持ちだけが持つなら効果説が正しい**。
  素の体が 63% なので、効果だけで 0.069 を作るには効果持ちが **0.18 級**を要る。

## 帯（`hand_value_slope --axis field` と同一）

`(自ライフ, 相手ライフ, ターン帯, 手札)`。**説明変数（場のキャラ）は帯に入れない**。

## 読み方（事前登録）

- **`all` が 0.1087 ± CI に入らなければ器がおかしい**（先に直す）。
- **式の予測と実測の差が CI を超える種類が、式の欠けている場所**。
- **とくに「リーダー未満」の実測が 0 でなければ、`attack_value(x<0)=0` は誤り**
  ——式はそこを一律 0 と置いているが、実プレイでは登場候補の 71.8% がこの帯に居る。
- **係数の和が全体と整合するか**も見る（`Σ β_c · 割合_c ≈ β_all`）。

**限界**: これは**観察された傾き**であって因果ではない（勝っている側はキャラが多い）。
Phase 1a の `λ`/`μ`/`δ`/`ν` も同じ性質で、**式が目指しているのは因果**なので、
一致は必要条件であって十分条件ではない。それでも**不一致は式の誤りの証拠**になる。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/nu_measure.py --in ~/w32/*/n_records \\
    --scheme all blocker power --out ~/nu_measure.json
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
from nu_calib import (S_BLOCKER_ACTIVE, S_IS_CHAR, S_POWER, SC_OPP_LEADER_POWER,  # noqa: E402
                      SLOT_OWN_FIELD, _round10)
from theory_order import (MU, NU_TARGET_MODES, SC_MY_LEADER_POWER, THETA,  # noqa: E402
                          nu_of, opp_chars_of)

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
#: scalars の列（`hand_value_slope` と同じ）
SC_MY_LIFE, SC_OPP_LIFE, SC_MY_HAND, SC_TURN = 0, 1, 6, 10
#: 飽和点（`theory_order.saturation_x(1.15)` = 2000）を跨ぐ境目
SAT_OVER = 2000.0
SCHEMES = ("all", "blocker", "power", "power_blocker", "lt_split", "lt_detail")
#: 枠 1 つが「効果を持っている」ことの信号（`n_rel_feat.S_COLS` の列・**カード ID に依らない**）。
#:
#: > **`cond_ok0..3` は使えない**（2026-09-14 に確認）——条件を持たない札も 1 が立つので
#: > **実測で全枠の 100%／96%／89% が 1**。「常在効果を持つか」の代理にならない。
S_TRIG_KO, S_TRIG_ATTACK, S_TRIG_OPP_ATTACK, S_THREAT_NEXT, S_ACT_AVAIL = 14, 15, 16, 17, 21
#: **素の体**＝この 5 つがどれも立っていない枠（実測でリーダー未満の帯の **63%**）。
#:
#: > **`threat_next` は入れない**（2026-09-14 に実測して外した）——**真偽値ではなく連続値**で
#: > （172 種類・`>0` が 58% なのに `>0.5` は 1.1%）、しかも**カードの能力ではなく盤面から
#: > 計算した量**である。`> 0` の真偽で拾うと**6 割の体が「効果持ち」に入って割合が反転する**
#: > （素の体が 62% → 15% に化けた）。**構成比を先に確かめたので気付けた**
#: > （`measurement.md` §14-18）。
EFFECT_SIGNALS = (S_BLOCKER_ACTIVE, S_TRIG_ATTACK, S_TRIG_KO, S_TRIG_OPP_ATTACK, S_ACT_AVAIL)


def signals_of(tok_row, slot):
    """枠 `slot` が立てている効果の信号（列 index の組）。空なら**素の体**。

    **`> 0.5` で見る**——`EFFECT_SIGNALS` は全て 0/1 の旗であり（実測で値の種類が 2）、
    連続値の列を混ぜないための歯止めでもある。
    """
    return tuple(c for c in EFFECT_SIGNALS if float(tok_row[slot, c]) > 0.5)


def turn_band(t):
    """`hand_value_slope`／`theory_residual` と同じ 3 帯。"""
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def band_key(sc_row):
    """帯＝(自ライフ, 相手ライフ, ターン帯, 手札)＝`--axis field` と同一。

    **説明変数（場のキャラの体数）は帯に入れない**——入れると帯の中で動かない。
    """
    return (int(round(float(sc_row[SC_MY_LIFE]))),
            int(round(float(sc_row[SC_OPP_LIFE]))),
            turn_band(int(round(float(sc_row[SC_TURN])))),
            int(round(float(sc_row[SC_MY_HAND]))))


def power_band(power, opp_leader_power):
    """パワーの 3 帯＝**式が 3 値しか返さない境目と同じ**に切る。"""
    x = _round10(float(power) - float(opp_leader_power))
    if x < 0.0:
        return "lt_leader"            # 式では `attack_value = 0` ⇒ `ν = 0`
    if x < SAT_OVER:
        return "leader_to_sat"
    return "over_sat"                 # 飽和＝ここから上はパワーが効かない


def categories(tok_row, opp_leader_power, scheme):
    """自場の枠 → 種類ごとの体数（`scheme` で分け方を替える）。"""
    out = {}
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok_row[s, S_IS_CHAR]) <= 0.0 and float(tok_row[s, S_POWER]) <= 0.0:
            continue
        pw = _round10(float(tok_row[s, S_POWER]) * 1e4)
        blk = float(tok_row[s, S_BLOCKER_ACTIVE]) > 0.0
        if scheme == "all":
            key = "chars"
        elif scheme == "blocker":
            key = "blocker" if blk else "plain"
        elif scheme == "power":
            key = power_band(pw, opp_leader_power)
        elif scheme in ("lt_split", "lt_detail"):
            key = _lt_key(tok_row, s, pw, opp_leader_power, scheme)
        else:
            key = power_band(pw, opp_leader_power) + ("_blk" if blk else "_plain")
        out[key] = out.get(key, 0.0) + 1.0
    return out


def _lt_key(tok_row, slot, pw, opp_leader_power, scheme):
    """**リーダー未満の帯だけを効果の有無で割る**（task #40／T21・2026-09-14）。

    実測の `ν` はこの帯で **0.0690**（CI が 0 を含まない）なのに式は **0** と置く＝
    **`ν` 最大の穴**（体数の 40%）。中身の候補は `game_theory.md` §14.1.1 の
    **身代わり（#7）・常在効果（#13）・アタック時効果（#14）**。

    **読み方（事前登録）**:

    - **`lt_plain`（素の体）が 0.069 前後を持つなら、効果説は棄却**＝価値は**体そのもの**
      （身代わり・枠・ドンの器・他の札の「N 体以上」を満たす）に在る。
    - **`lt_plain` が 0 に近く効果持ちだけが持つなら、効果説が正しい**。
      素の体は帯の 63% を占めるので、効果だけで 0.069 を作るには
      効果持ちが 0.18 級を持たねばならない。

    リーダー以上の帯は 1 本にまとめる（対照＝**帯を分けずに全部入れる**ことで
    除外変数を作らない）。
    """
    if pw - _round10(float(opp_leader_power)) >= 0.0:
        return "ge_leader"
    sig = signals_of(tok_row, slot)
    if not sig:
        return "lt_plain"
    if scheme == "lt_split":
        return "lt_signal"
    if S_BLOCKER_ACTIVE in sig:
        return "lt_blocker"                  # 式が既に持っている項（ブロック）
    if S_TRIG_ATTACK in sig:
        return "lt_attack"                   # アタック時効果（§14.1.1 #14）
    if S_ACT_AVAIL in sig:
        return "lt_act"                      # 未使用の起動メイン（#3）
    return "lt_other"                        # KO 時・被攻撃時


def within_multi(recs, keys, y_key="z", band="band", seed="seed"):
    """帯の中の**多変量** OLS と、**対局でクラスタした**共分散。

    ```
    V = (X'X)^-1 [ Σ_g X_g' e_g e_g' X_g ] (X'X)^-1
    ```

    1 変数にすれば `hand_value_slope.within_slope` と同じ量になる（器の突き合わせ用）。
    """
    by_band = {}
    for r in recs:
        by_band.setdefault(r[band], []).append(r)
    Xs, ys, gs = [], [], []
    used_bands = 0
    for _b, sub in by_band.items():
        if len(sub) < 2:
            continue
        X = np.array([[float(r.get(k, 0.0)) for k in keys] for r in sub], np.float64)
        y = np.array([float(r[y_key]) for r in sub], np.float64)
        X = X - X.mean(0)                      # 帯で中心化（帯の固定効果を落とす）
        y = y - y.mean()
        if not np.any(X.std(0) > 0):
            continue
        Xs.append(X); ys.append(y); gs.extend(r[seed] for r in sub)
        used_bands += 1
    if not Xs:
        return None
    X = np.vstack(Xs); y = np.concatenate(ys)
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        return {"error": "rank_deficient", "n": len(y), "bands": used_bands}
    inv = np.linalg.inv(XtX)
    beta = inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros_like(XtX)
    gs = np.array(gs)
    for g in np.unique(gs):
        m = gs == g
        xg = X[m]; eg = e[m]
        s = xg.T @ eg
        meat += np.outer(s, s)
    V = inv @ meat @ inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return {"n": int(len(y)), "bands": used_bands, "games": int(len(np.unique(gs))),
            "beta": {k: round(float(b), 5) for k, b in zip(keys, beta)},
            "se": {k: round(float(s), 5) for k, s in zip(keys, se)},
            "ci95": {k: [round(float(b - 1.96 * s), 5), round(float(b + 1.96 * s), 5)]
                     for k, b, s in zip(keys, beta, se)},
            "share": {k: round(float(np.mean([r.get(k, 0.0) for r in recs])), 4)
                      for k in keys}}


#: 実測の価格（`docs/game_theory.md` §18・Phase 1a）。**勘定に使うのでここが正本**
MU_TRUE, DELTA_TRUE = 0.0433, 0.0277


def cost_check(played, nu_by_band, mu=MU_TRUE, delta=DELTA_TRUE):
    """**支払ったコストと価格が見合っているか**（ユーザ指摘 2026-09-14）。

    ```
    play_value = ν − μ − cost·δ = 0  ⇒  分岐コスト = (ν − μ) / δ
    ```

    `ν` は**実測値**（種類別）を渡す。**実際に打たれたキャラの平均コスト**と比べて、
    差が**値付けできていない項の大きさ**になる。

    **なぜ差が `ν` の中身ではないのか**（2026-09-14 に自分の診断を訂正した）:
    実測の `ν` は**観察された勝率の傾き**なので、**その体が場に居る間にすることは全部
    入っている**（常在効果もアタック時効果も）。だから差を説明できるのは **`ν` の外**——
    **登場した瞬間の一回性の利得**（サーチ・ドロー・登場時除去）である。カードは**手札**へ
    行くので、原理的に「体が場に居る価値」には入らない。式はこの項を丸ごと欠いていた:

    ```
    play_value = ν ＋ 【登場時の一回性の利得】 − μ − cost·δ
    ```
    """
    out = {}
    for band, a in played.items():
        if not a["n"]:
            continue
        nu = nu_by_band.get(band)
        c = a["cost"] / a["n"]
        row = {"n": a["n"], "cost_mean": round(c, 3), "nu_measured": nu,
               "ability_share": round(a["abil"] / a["n"], 4),
               "onplay_removal_share": round(a["onplay"] / a["n"], 4),
               "blocker_share": round(a["blk"] / a["n"], 4)}
        if nu is not None and delta:
            be = (float(nu) - mu) / delta
            row.update(break_even_cost=round(be, 3), overpay_cost=round(c - be, 3),
                       # **値付けできていない項がこれだけの価値を持たないと勘定が合わない**
                       missing_term_winrate=round((c - be) * delta, 5))
        out[band] = row
    return out


def collect(dirs, limit_games=0, schemes=SCHEMES, cards=None):
    """holdout の自席ターン最初の main 行 → 帯と種類別の体数（と勝敗）。

    `cards` を渡すと**実際に打たれたキャラ**（`pol_chosen`）のコストと素性も集める
    （`cost_check` の材料・パワー帯は同じ切り方）。
    """
    recs = []
    played = {}
    games = 0
    # `cards` を渡すときは候補列も読む（打たれた登場の cid とコストを引くため）
    pol_cols = ("pol_sig", "pol_cid") if cards is not None else ()
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=pol_cols,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = int(rows["seed"][idx[0]])
        seen = set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            if (w, t) in seen:
                continue
            seen.add((w, t))
            z = float(rows["z"][i])
            if z == 0.0:
                continue
            sc = ex["sc"][i]; tok = ex["tok"][i]
            opl = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            rec = {"seed": seed, "band": band_key(sc), "z": 1.0 if z > 0 else 0.0,
                   "opp_leader_power": opl, "turn": t,
                   # **式の予測をこの行の盤面で引くため**に持ち回る（task #39）——
                   # 攻撃項を「対象の max」にすると `ν` が**相手の場に依る**ので、
                   # 代表値 1 つでは引けない（行ごとに引いて平均する）。
                   "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                   "opp_chars": opp_chars_of(tok)}
            for sch in schemes:
                rec.update(categories(tok, opl, sch))
            recs.append(rec)
        if cards is not None:
            _collect_played(rows, pol, ex, L, ptr, idx, cards, played)
    return recs, games, played


def _collect_played(rows, pol, ex, L, ptr, idx, cards, played):
    """**打たれた登場**（`pol_chosen`）のコストと素性をパワー帯ごとに積む。"""
    from opcg_sim.loop import deck_roles as DR
    # **黙って空を返さない**——候補列が無いまま呼ばれるのは呼び側の誤り
    # （2026-09-14 に `pol_cols=()` のまま呼んで、`except` が KeyError を飲んで空になった）
    for need in ("pol_sig", "pol_cid"):
        if need not in pol:
            raise KeyError(f"_collect_played は {need} を要る（iter_games の pol_cols に足す）")
    for i in idx:
        if int(rows["kind"][i]) != 0:
            continue
        ch = int(rows["pol_chosen"][i]); k = int(L[i]); b = int(ptr[i])
        if ch < 0 or ch >= k:
            continue
        j = b + ch
        try:
            sig = json.loads(pol["pol_sig"][j])
        except (ValueError, TypeError):        # 壊れた JSON だけを飲む
            continue
        if sig[0] != "PLAY":
            continue
        cid = str(pol["pol_cid"][j]) or None
        info = cards.info(cid)
        # **ステージは体を持たない**＝「場のキャラ 1 体」の勘定から外す（2026-09-14）
        if not info or info.get("event") or info.get("stage"):
            continue
        master = cards.db.get_card(cid) if cid else None
        forms = DR.classify(master) if master is not None else set()
        band = power_band(float(info["power"]), float(ex["sc"][i][SC_OPP_LEADER_POWER]) * 1e4
                          or 5000.0)
        a = played.setdefault(band, {"n": 0, "cost": 0.0, "blk": 0, "abil": 0, "onplay": 0})
        a["n"] += 1
        a["cost"] += float(info.get("cost") or 0)
        a["blk"] += int(bool(info.get("blocker")))
        a["abil"] += int(bool(getattr(master, "abilities", ()) or ()))
        # **登場時の一回性の利得**の代理（今は除去の型だけ数えられる）
        a["onplay"] += int(any(":ON_PLAY:" in f for f in forms))


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


#: それぞれの種類について、式が返す `ν`（検算の相手）。代表パワーで引く。
PREDICT_POWER = {"lt_leader": 3000.0, "leader_to_sat": 6000.0, "over_sat": 9000.0}


def _kind_spec(k):
    """種類 → (代表パワー, ブロッカーか)。`None` は「母集団の平均（10.4%）で混ぜる」。

    **`power` 帯（`lt_leader` など）は `False`（非ブロッカー）で引く**——混ぜる方が
    母集団としては正しいが、**#39 の前後で比べられるように 2026-09-14 以前と同じにしてある**
    （差は `block_p` 0.104×0.773×Θμ ≈ 0.004 で、本件の差より小さい）。
    """
    if k == "chars":
        return 6000.0, None
    if k in ("blocker", "plain"):
        return 6000.0, (k == "blocker")
    # T21 の細分（`lt_split`／`lt_detail`）。**式の主張はどれも「リーダー未満は 0」**で、
    # ブロッカーだけがブロック項ぶん（0.035）を持つ。`lt_signal`／`ge_leader` は
    # 混ざりものなので**予測を出さない**（代表値を選ぶと恣意になる）。
    if k in ("lt_plain", "lt_attack", "lt_act", "lt_other"):
        return PREDICT_POWER["lt_leader"], False
    if k == "lt_blocker":
        return PREDICT_POWER["lt_leader"], True
    base = k.replace("_blk", "").replace("_plain", "")
    pw = PREDICT_POWER.get(base)
    if pw is None:
        return None, None
    return pw, k.endswith("_blk")


def predict(keys, opp_leader_power=5000.0, r_turns=4.128, theta=THETA, mu=MU,
            targets="leader", recs=None):
    """式の予測値（同じ種類の `ν`）。`R` は実測の 4.128 を既定にする。

    `targets="board"` で**攻撃項を「対象の max」に広げた式**を引く（task #39）。
    そのとき `ν` は**相手の場に依る**ので、代表値 1 つでは引けない——`recs`（`collect` が
    返す行）を渡し、**行ごとに引いてその平均**を予測値にする。**行の盤面の分布ごと**
    突き合わせるので、実測の `β`（同じ行集合の帯内回帰）と同じ母集団で比べられる。
    """
    board = (targets == "board")
    rows = list(recs or ()) if board else []
    if board and not rows:
        board = False                     # 盤面が無いなら従来どおり（黙って壊さない）

    def one(pw, blk, opl, mlp, oc):
        if blk is None:                   # 母集団の平均（ブロッカー 10.4%）で混ぜる
            return (0.104 * one(pw, True, opl, mlp, oc)
                    + 0.896 * one(pw, False, opl, mlp, oc))
        return nu_of(pw, opl, r_turns, theta, mu, is_blocker=blk,
                     opp_chars=oc, my_leader_power=mlp)

    out = {}
    for k in keys:
        pw, blk = _kind_spec(k)
        if pw is None:
            continue
        if not board:
            out[k] = one(pw, blk, opp_leader_power, None, None)
            continue
        out[k] = float(np.mean([one(pw, blk, r["opp_leader_power"], r["my_leader_power"],
                                    r["opp_chars"]) for r in rows]))
    return {k: round(float(v), 5) for k, v in out.items()}


#: Phase 1a が出した実測（`docs/reports/2026-09-13_phase1_w32.md` §1a）
NU_PHASE1A = 0.1087


def check(fit, pred, anchor=NU_PHASE1A):
    """式の予測が実測の CI に入るか（**入らない種類が式の欠けている場所**）。"""
    if not fit or "beta" not in fit:
        return None
    rows = []
    for k, b in fit["beta"].items():
        lo, hi = fit["ci95"][k]
        p = pred.get(k)
        rows.append({"kind": k, "measured": b, "ci95": [lo, hi], "predicted": p,
                     "share_per_turn": fit["share"].get(k),
                     "in_ci": (None if p is None else bool(lo <= p <= hi)),
                     "gap": (None if p is None else round(p - b, 5))})
    rows.sort(key=lambda r: -(abs(r["gap"]) if r["gap"] is not None else -1))
    out = {"rows": rows}
    if "chars" in fit["beta"]:
        lo, hi = fit["ci95"]["chars"]
        out["reproduces_phase1a"] = bool(lo <= anchor <= hi)
        out["phase1a"] = anchor
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--scheme", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    ap.add_argument("--r-turns", type=float, default=4.128, help="式に渡す R（実測の既定）")
    ap.add_argument("--nu-targets", default="leader", choices=NU_TARGET_MODES,
                    help="式の攻撃項をリーダー狙いだけにするか（leader）"
                         "相手の場も対象に含めた max にするか（board・task #39）。"
                         "`board` は**行ごとの盤面で引いて平均**する")
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--cost-check", action="store_true",
                    help="**払ったコストと価格が見合っているか**（`--scheme power` が要る）")
    ap.add_argument("--mu-true", type=float, default=MU_TRUE, help="実測の手札 1 枚（勘定に使う）")
    ap.add_argument("--delta-true", type=float, default=DELTA_TRUE, help="実測のドン 1 個")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    cards = PL.Cards() if a.cost_check else None
    recs, games, played = collect(a.src, a.limit_games, tuple(a.scheme), cards)
    res = {"games": games, "own_turns": len(recs), "fits": {},
           "args": {k: v for k, v in vars(a).items() if k != "out"}}
    keys_of = {"all": ["chars"], "blocker": ["blocker", "plain"],
               "power": ["lt_leader", "leader_to_sat", "over_sat"],
               "power_blocker": [b + s for b in ("lt_leader", "leader_to_sat", "over_sat")
                                 for s in ("_blk", "_plain")],
               "lt_split": ["lt_plain", "lt_signal", "ge_leader"],
               "lt_detail": ["lt_plain", "lt_blocker", "lt_attack", "lt_act", "lt_other",
                             "ge_leader"]}
    for sch in a.scheme:
        keys = [k for k in keys_of[sch] if any(r.get(k) for r in recs)]
        fit = within_multi(recs, keys) if keys else None
        pred = predict(keys, r_turns=a.r_turns, theta=a.theta, mu=a.mu,
                       targets=a.nu_targets, recs=recs)
        res["fits"][sch] = {"fit": fit, "predicted": pred, "check": check(fit, pred)}
    if a.cost_check:
        pf = res["fits"].get("power", {}).get("fit") or {}
        res["cost_check"] = cost_check(played, (pf.get("beta") or {}), a.mu_true, a.delta_true)
    res["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
