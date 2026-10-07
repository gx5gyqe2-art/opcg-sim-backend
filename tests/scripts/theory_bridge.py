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
>
> **G-2（2026-09-25）**: 「払えた」は**規則どおり合計 ≥ 超過 + 1000**（同値は命中・旧の合計 ≥ 超過＝`lenient` の
> 切替は 2026-10-05 に削除）。守る費用を**この手札で実際に失う価値**で測る G-2 の `hand`
> （守る備えは厳密な最大を 1 ラウンド割り引いて読む・`GUARD_S_COST_MODES` の注記）は N-2 の `joint` に置き換わり、
> 2026-10-05 に削除（`claude/theory-switches-final` で再現）。
> **N-2（2026-09-26）**: `GUARD_S_COST_MODE=joint`＝手札の価値を 1 枚 1 役の最適な割り当て（`hand_joint.py`）で読む切替。
> **既定は `joint`**（ユーザ決定 2026-09-26「判断1の続き→(a)」・旧の `c(x)·μ` は `--guard-s-cost curve` で再現）。

## 暫定値（§0.4 の台帳・**感度を付けて回す**）

| | 穴 | この器での扱い |
|---|---|---|
| **P2** | イベント・効果の値付けが無い | 無言の行を取りこぼし 0 として母数に入れる（`zero`・感度の `exclude` は 2026-10-05 に削除） |
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

**段 7（2026-10-07）**: 価格（`s`・`g`）・守りの窓・局面の傾き `κ`・1 局ぶんの行の読みは Rust の局の駆動
（`rust/opcg_engine/src/theory/core/drv_tb.rs`）。Python に残るのは記録の読み・`label_game`・デッキの作り直し・
局の突き合わせ（`pair_games`）と集計（AUC・傾き・ブートストラップ・較正）と JSON。


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
import theory_rs as TR  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0",
            "sig")   # `sig` は `PL.label_game`（守り側の take/guard の判定）が要る
POL_COLS = ("pol_n", "pol_q", "pol_p", "pol_sig", "pol_cid", "pol_tcid", "pol_si",
            "pol_ti", "pol_k")
#: **T28-c**——「払えた」の判定が粗いせいで反転しているのではないかを分ける閾値。
#: 守る力が来る攻撃をこれだけ上回っていれば**余裕で払えた**と見なす（暫定値・感度を取る）。
#: **ブロッカーが居る行は無条件で余裕**（レストするだけでドンを使わない）。
MARGIN_COMFORT = 2000.0
#: **G-2／N-2**: 守りの判断の守る費用（`joint`＝1 枚 1 役の手札の価値の減り・定数・旧 `curve` は凍結ブランチで再現）
GUARD_S_COST_MODE = "joint"


def band_of(v0, close=0.2, decided=0.6):
    """ネットの価値 `|v0|` の帯（接戦／中盤／決着・`order_acc` の写し）。"""
    a = abs(float(v0))
    return "close" if a <= close else ("decided" if a > decided else "mid")


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32),
            "ci": np.asarray(dd["card_idx"])[:n]}


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


def _seat_decks(rec_decks, seed, rows, ex, idx, idx2cid, stats=None):
    """**T68**: その局の席ごとのデッキ `{who: [card_id]}`（seed から復元・最初の自席ターンの手札で検算。合わなければ席を落とす）。"""
    if not rec_decks or int(seed) not in rec_decks:
        return {}
    import deck_refill as DR
    mode, leaders = rec_decks[int(seed)]
    out = {}
    for w in (0, 1):
        i = next((i for i in idx if int(rows["who"][i]) == w and int(rows["turn"][i]) >= 1), None)
        if i is None:
            continue
        d = DR.deck_for_seat(seed, mode, leaders, w, DR.hand_ids(ex["ci"][i], idx2cid))
        if stats is not None:
            stats["search_deck_ok" if d else "search_deck_bad"] = stats.get("search_deck_ok" if d else "search_deck_bad", 0) + 1
        if d:
            out[w] = d
    return out


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const", margin_comfort=None, harm_profile="cross"):
    """(局, 席) ごとに攻め側と守り側の取りこぼしを足す（1 局ぶんは Rust の局の駆動 `theory_bridge`）。

    `s`（決める＝行内の最善からの逸脱）は `FLOW_PRICING`、`g`（数える＝`ΔG`）は `exercise`（T58）で読む。
    """
    ledger_pricing = TR.LEDGER_FLOW_PRICING
    cards = PL.Cards()
    idx2cid = TR.idx2cid()
    # **T68**: 探す能力の計画価格に要るデッキ（seed から復元・席ごとに手札で検算）
    import deck_refill as DR
    rec_decks = DR.record_decks(dirs)
    # **T75**: `W_MODE=curve`（定数）——交点の橋の `D` で `κ` を出す。輪郭は別のセットのもの（`cross`）
    import crossing_bridge as CB
    prof = CB.profile_for(dirs, harm_profile)
    if prof is None:
        raise ValueError("harm profile が無い（%s・%s）" % (harm_profile, CB.HARM_PROFILE_PATH))
    # **T97**: `σ_D = √2 × σ_T` の `σ_T` を**同じ器の実測**から採る（耐久の体の集合ごと・別のセットの値）。
    st = CB.sigma_t_for(dirs, harm_profile)
    if st is None:
        raise ValueError("σ_T が引けない（体の形 %r・%s・%s）＝黙って前の値を使い回さない"
                         % (CB.THETA_BODY_MODE, harm_profile, CB.HARM_PROFILE_PATH))
    TR.set_sigma_turn(st)
    # **T118**: `W_ERR_MODE=rel` なら物差しは `σ_rel × s(τ_me, τ_opp)`（`curve` の読みのもの）。引けなければ落ちる。
    if TR.CLOCK["W_ERR_MODE"] == "rel":
        sr = CB.sigma_rel_for(dirs, harm_profile, slope="curve")
        if sr is None:
            raise ValueError("W_ERR_MODE=rel なのに σ_rel が引けない（%s・%s）＝黙って abs に落とさない"
                             % (harm_profile, CB.HARM_PROFILE_PATH))
        TR.set_sigma_rel(sr)
        # **T98**: `κ = w(D)/w̄` の分母も**同じ器の実測**（`E[w(D)]`）にする。
        wb = CB.w_bar_for(dirs, harm_profile)
        if wb is not None:
            TR.set_w_bar(wb)
    per = {}
    kn_turns = []        # T80: ターンごとの（`D`・生の価格・`ΔW`）＝必要な `κ` を測る材料
    kn_games = []        # T81: 局ごとのターンの並び（窓の広さ・手の型・場の動きで相関を割る）
    stats = {"games": 0, "atk_rows": 0, "atk_silent": 0, "grd_rows": 0, "grd_no_attack": 0,
             # **T28-c**: 余裕で払えた守りの行の数
             "grd_comfortable": 0,
             # **T49**: 局面の傾き `κ = w(D)/w̄`（攻めの行）——平均が 1 に戻るかが `w(状態)` の検算
             "w_mode": TR.W_MODE, "sigma_turn": TR.CLOCK["SIGMA_TURN"], "w_bar": TR.CLOCK["W_BAR"],
             "kappa_sum": 0.0, "kappa_n": 0,
             "clock_hand": TR.CLOCK_HAND_MODE,
             "harm_profile": (harm_profile if TR.W_MODE == "curve" else None),
             # **T76**: 耐久の手札項の数え方（`crossing_bridge.THETA_HAND_MODE`）
             "theta_hand": CB.THETA_HAND_MODE,
             "theta_body": CB.THETA_BODY_MODE,
             # **T84**: 出した体の価格を効き始めるターンに計上するか
             "play_book": "next", "play_deferred": 0, "play_deferred_dropped": 0,
             # **T85**: 付与の帳簿価格の規約と、0 にした行の数
             "attach_ledger": "in_attack", "attach_zeroed": 0,
             # **T86**: 守りの窓の攻め手の価格の取り方と、そのターンの攻撃の本数ごとの内訳
             "guard_price": "max_attack", "grd_by_attacks": {}, "grd_took_n": 0.0,
             # **T87**: 帳簿を実現で書くか・実現で書けた行／書けなかった行（次の行が同じターンに無い＝ターン末）
             "ledger_harm": "realised", "harm_rows": 0, "harm_unbracketed": 0,
             # **T58**: 決める価格（`s`）と数える価格（`g`）の規約・読み直した行の数
             "flow_pricing": TR.RUN["FLOW_PRICING"], "ledger_pricing": ledger_pricing, "ledger_rescored": 0,
             # **T62**: 守りの窓の定義と、自ライフごとの内訳（受けた率・理論が受けろと言う率・`g` の平均）
             "guard_g": "delta", "grd_by_life": {},
             # **G-2**: 守れたかの閾値と、判断の守る費用の測り方
             # `guard_afford` は実際に効いた閾値（2026-10-05 から常に規則どおり）
             "guard_afford": "rule", "guard_afford_requested": "rule",
             "guard_s_cost": GUARD_S_COST_MODE,
             # **T68**: 探す能力の価格の規約と、デッキを復元できた席／できなかった席の数
             "search_price": "plan", "search_deck_ok": 0, "search_deck_bad": 0,
             "d_bins": {"<-3": 0, "-3..-1": 0, "-1..1": 0, "1..3": 0, ">3": 0},
             # **`D` の帯ごとの実勝率**（当てはめない）——時計の推定 `D` が勝敗を順序付けるか・
             # 実測の `W(D)` の傾きが置いた `σ_D` と合うかの検算
             "d_win": {"<-3": [0, 0], "-3..-1": [0, 0], "-1..1": [0, 0], "1..3": [0, 0], ">3": [0, 0]}}
    mc = MARGIN_COMFORT if margin_comfort is None else margin_comfort
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        rows, pol, ex, L, ptr, idx = game
        stats["games"] += 1                                       # `_seat_decks` が `stats` に数える前
        seed = int(rows["seed"][idx[0]])
        labels, _unk = PL.label_game(rows, pol, ex["sc"][:, 0], L, ptr, idx, cards)
        decks = _seat_decks(rec_decks, seed, rows, ex, idx, idx2cid, stats)   # T68
        pin = {"decks": [None if decks.get(w) is None else list(decks.get(w)) for w in (0, 1)],
               "labels": [int(x) for x in labels]}
        c = TR.cfg(theta, mu, theta_mode, prof=prof, nu_targets="leader", GUARD_S_COST_MODE=GUARD_S_COST_MODE,
                   ledger_pricing=str(ledger_pricing), MIRROR_ME=bool(TR.RUN["MIRROR_ME"]), F_PRICING_FIX=True,
                   margin_comfort=float(mc), PLAN_CLASSES=[str(x) for x in PL.PLAN_CLASSES])
        res = TR.game_call("theory_bridge", game, {"cfg": c, "in": pin, "stats": stats, "carry": {}})
        for key, rec in res["per"]:
            key = tuple(key)
            if key in per:
                raise RuntimeError("theory_bridge: per の鍵 %r が 2 局に出た" % (key,))
            per[key] = rec
        kn_turns.extend(res["kn_turns"])
        kn_games.append(res["kn_game"])
        new = res["stats"]
        stats.clear()
        stats.update(new)
    import kappa_needed as KN
    stats["kappa_needed"] = KN.summarise(kn_turns)
    stats["kappa_needed_1turn"] = KN.summarise([dict(t, g0=t["g0_turn"], dW=t["dW_turn"]) for t in kn_turns])
    stats["kappa_split"] = KN.summarise_games(kn_games)        # T81: 相関の中身（窓・型・場の動き）
    stats["last_turn"] = "keep"
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


def pair_games(per):
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




def corr_of(pairs, key="dS"):
    """**尺度に依らない結びつき**（`ΔG` と勝敗の相関・T119）。

    **`slope` は尺度で動く**（帳簿を c 倍すれば 1/c 倍になる）ので、**一律の縮小で「傾きが 1 に寄った」と
    言えてしまう**——実測: 帳簿全体を 0.65 倍すると前向きの傾きは 0.4614 → 0.7098 と 1 へ 54% 寄るが、
    **相関と AUC は 1 ビットも動かない**。だから**較正（傾き）と判別（相関・AUC）は別の 2 つの必要条件**で、
    どちらも出さないと「良くなった」の意味が決まらない。"""
    x = np.array([p[key] for p in pairs], np.float64)
    y = np.array([p["z"] for p in pairs], np.float64)
    if len(x) < 10 or float(x.var()) <= 0.0 or float(y.var()) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def slope_rev(pairs, key="dS"):
    """**後ろ向きの傾き**（勝敗 1 単位あたり帳簿がどれだけ動くか・T119）。

    **これは「恒等式が 1 と言う量」ではない**——`z − 0.5 = ΔG` を**期待値で厳密に満たす**帳簿を作って測ると
    前向きの傾きが 1.00 になり、**後ろ向きは `4·var(p)`（実測 0.15〜0.27）にしかならない**（反証で確認）。
    `rev = 1` を満たせるのは `ΔG ≡ ±0.5` の**全知の帳簿だけ**。
    出す理由は**「11」と「1」を同じ表に並べない**ため（単位が違うものを比べていた）。"""
    x = np.array([p[key] for p in pairs], np.float64)
    y = np.array([p["z"] for p in pairs], np.float64)
    if len(x) < 10 or float(y.var()) <= 0.0:
        return None
    yd = y - y.mean()
    return float((yd * (x - x.mean())).sum() / (yd * yd).sum())


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
                    # **T119**: 尺度に依らない結びつきと、後ろ向きの傾き（**目標値は 1 ではない**）
                    "corr": (round(corr_of(pairs, key), 4)
                             if corr_of(pairs, key) is not None else None),
                    "slope_rev": (round(slope_rev(pairs, key), 5)
                                  if slope_rev(pairs, key) is not None else None),
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
    ap.add_argument("--margin-comfort", type=float, default=MARGIN_COMFORT,
                    help="**T28-c** の暫定値——守る力が来る攻撃をこれだけ上回れば「余裕で払えた」")
    ap.add_argument("--cond-unknown", type=float, default=1.0,
                    help="**判らない条件の係数**（§0.4 の感度。1.0＝上限・0.0＝下限）")
    ap.add_argument("--flow-pricing", default=None, choices=TR.FLOW_PRICING_MODES,
                    help="**決める価格**（`s`・`ΔS`）の規約。省略時は `option`")
    ap.add_argument("--search-value", default=None, choices=TR.SWITCH_VALUES["SEARCH_VALUE_MODE"],
                    help="**N-4** 足した札の値: `legacy`（既定・`max(ΔH, ΔG)`）／`joint`（1 枚 1 役の手札の価値の増え・残す候補）")
    ap.add_argument("--harm-profile", default="cross", choices=("cross", "real", "syn"),
                    help="**T75** 輪郭: `cross`（既定・測る記録と別のセット）／`real`／`syn`（`tests/fixtures/harm_profile.json`）")
    ap.add_argument("--mirror", default=("on" if TR.RUN["MIRROR_ME"] else "off"), choices=("on", "off"),
                    help="**H-4g** 自分の耐久も相手と同じ守る側の計算で読む（既定 on・`off`＝自分は `threshold_of_me`）")
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    TR.set_switch("MIRROR_ME", a.mirror == "on")              # **H-4g**
    if a.search_value is not None:
        TR.set_switch("SEARCH_VALUE_MODE", a.search_value)
    TR.set_switch("UNKNOWN_FACTOR", a.cond_unknown)
    if a.flow_pricing is not None:
        TR.set_switch("FLOW_PRICING", a.flow_pricing)
    t0 = time.time()
    per, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode, a.margin_comfort,
                         harm_profile=a.harm_profile)
    # **T49 の検算**: `κ` の平均（`w` の平均が `w̄` に戻れば 1）
    stats["kappa_mean"] = (round(stats["kappa_sum"] / stats["kappa_n"], 4) if stats["kappa_n"] else None)
    stats["w_mean"] = (round(stats["kappa_mean"] * TR.CLOCK["W_BAR"], 4) if stats["kappa_mean"] is not None else None)
    stats["f_pricing_fixes"] = TR.F_PRICING_FIXES_LABEL                # 値付けの直し（`all`・定数）
    pairs = pair_games(per)
    res = {"stats": stats, "decision_rows": "main",
           "provisional": {"P3_theta": a.theta, "P2_silent": "zero",
                           "T28c_margin": a.margin_comfort, "w_mode": TR.W_MODE,
                           "flow_pricing": stats["flow_pricing"], "ledger_pricing": stats["ledger_pricing"],
                           "surv_mode": TR.SW["SURV_MODE"], "nu_mode": TR.SW["NU_MODE"],
                           "cbar_mode": TR.SW["CBAR_MODE"], "guard_g": "delta", "take_mode": "lethal",
                           "guard_cost": "spent", "search_price": "plan",
                           "guard_afford": "rule", "guard_afford_requested": "rule",   # G-2
                           "guard_s_cost": GUARD_S_COST_MODE,
                           "play_now": "hand", "inflow": "on", "cond_clock": "on",
                           "pricing_fixes": TR.PRICING_FIXES_LABEL,                                      # L
                           "note": "§0.4 の暫定値。感度を付けて読む",
                           "cut_price": TR.SW["CUT_PRICE_MODE"]},                                       # **N-3**
           "summary": summarise(pairs, a.boot_reps, a.seed),
           # **T28-b: 行ごとに帯で切ってから足した版**（判定の主はこちら）
           "per_band": {nm: summarise(pair_by_band(per, nm), a.boot_reps, a.seed)
                        for nm in ("close", "mid", "decided")},
           "seconds": round(time.time() - t0, 1)}
    import crossing_bridge as _CBs
    res["rule_stats"] = dict(_CBs.RULE_STATS)             # **H-4g**: 使った計画ごとの地平の縮み（冷たい実行と同じ数）
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
