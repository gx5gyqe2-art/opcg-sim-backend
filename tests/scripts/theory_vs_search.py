"""**接戦帯で探索は理論のどこから外れるのか**（食い違いの解剖・読み取り専用）。

ユーザ指示 2026-09-13「まずは接戦の時に探索が理論に則っていないことを調べてください。
事実の調査が先です」。`theory_order.py`（T1）は**順序の一致率**を出したが、
**接戦帯で 0.526**＝理論も探索とほとんど合っていない（方策の 0.4975 より良いだけ）。
「どちらが誤っているか」はそこでは分けられない。本器は**その手前の事実**を出す:

```
何が食い違っているのか（どの行動型 vs どの行動型）
食い違いはどれだけ大きいのか（互いの物差しで測った損）
理論の鋭い予言は当たっているのか（当たり外れが明白な 3 件）
```

## 3 つの top1 を並べる

| 記号 | 何 |
|---|---|
| `arg_th` | **理論**の最良手（4 通貨の価格・`theory_order.score_candidate`） |
| `arg_q` | **探索**の最良手（`pol_q` の最大・訪問の下限を満たすものだけ） |
| `arg_played` | **実際に打った手**（`pol_chosen`） |

`arg_q` と `arg_played` がずれるのは正常（選択は訪問数で行うので）。**見たいのは
`arg_th` と `arg_q` のずれ**＝価格と探索の結論の差。

## 食い違いの大きさは**両方の物差しで**測る

```
q_loss  = q(arg_q)  − q(arg_th)     … 理論に従うと探索の物差しでいくら損か
th_loss = th(arg_th) − th(arg_q)    … 探索に従うと理論の物差しでいくら損か
```

**片方だけ大きいなら、大きい側の物差しがその局面を見ている**。両方大きければ本物の対立。
`q_loss` が雑音床（`--q-eps`）を割る食い違いは**どちらでも良い手**＝母数から分けて数える。

## 理論の鋭い予言（当たり外れが明白な 3 件）

価格の式から、**探索がやっていたら理論が正しく探索が誤り**と言える行動が出る:

| # | 予言 | 理由 |
|---|---|---|
| `p1_don_on_zero` | **超過 0 の攻撃にドンを付けても価値は増えない** | `c(0) = c(1000) = 1.00 枚`＝段を跨げない（§14.1） |
| `p2_over_sat` | **飽和点 `x*` を超えて積んでも価値は増えない** | 守り側は `Θ` で受けに回るので `min(c(x), Θ)` が天井 |
| `p3_cannot_connect` | **相手より弱い攻撃は価値 0** | 通らない（`x < 0` ⇒ `c = 0`） |

これは「理論が当たっているか」の検査であって、**探索の Q を正とは仮定しない**
＝T1 の限界（Q を正としていた）を外した見方になる。

## 読み方（事前登録）

- **`top1_th_vs_q` が 0.5 付近**なら「理論と探索は別のことを言っている」。
  そのうえで **`q_loss` と `th_loss` の非対称**を見る:
  `q_loss` ≫ `th_loss` なら理論の取りこぼしが大きい（＝理論が粗い）。逆なら探索の取りこぼし。
- **`cross` の上位 3 型**が「何を巡って食い違っているか」の答え。
- **予言 3 件の頻度が高ければ、探索の側の欠陥が名指しできる**（価格を教える対象が具体になる）。
- `z_link` は**選択交絡がある**（打った手しか観測できない）ので符号だけ読む。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_vs_search.py --in ~/w32/*/n_records \\
    --holdout-mod 7 --out ~/theory_vs_search_w32.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from order_acc import band_of, q_floor  # noqa: E402
from theory_order import (add_nu_mode_arg, apply_nu_mode, MU, THETA, THETA_MODES, NU_TARGET_MODES,  # noqa: E402
                          SC_MY_LIFE, SC_MY_DON,  # noqa: E402
                          SC_MY_LEADER_POWER, SC_OPP_LEADER_POWER,  # noqa: E402
                          SC_OPP_LIFE, PWR_EPS, c_of, opp_chars_of, saturation_x,  # noqa: E402
                          score_candidate, theta_of)

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_n", "pol_q", "pol_p", "pol_sig", "pol_cid", "pol_tcid", "pol_si",
            "pol_ti", "pol_k")

#: 予言の名前（出力の鍵・順序を固定する）
PREDICTIONS = ("p1_don_on_zero", "p2_over_sat", "p3_cannot_connect")
#: トークンの枠（`theory_residual` と同じ・現在パワーは /1e4 で入っている）
SLOT_OWN_FIELD, SLOT_OPP_FIELD = slice(2, 7), slice(7, 12)
S_POWER = 0


def _round10(x):
    """10 の桁で丸める（f32 で 0.7×1e4 が 6999.999… になるため・3 回踏んだ罠）。"""
    return float(np.round(float(x) / PWR_EPS) * PWR_EPS)


def slot_power(tok_row, slot):
    """枠 index → **今のパワー**（`None` なら枠が無い）。

    **印字パワーではなく現在パワーを使う**のが要点——ドンが付いたキャラや強化されたキャラは
    印字では測れず、超過パワー `x` は**まさにその差**を見る量だから
    （`theory_order` は印字しか見ていない＝本器はそこを 1 段直している）。
    """
    if slot is None or int(slot) < 0:
        return None
    s = int(slot)
    if s >= tok_row.shape[0]:
        return None
    return _round10(float(tok_row[s, S_POWER]) * 1e4)


def cand_detail(sig, cid, tcid, si, ti, ctx, cards, tok_row, don_k=None):
    """候補 1 つの**性質**（値ではなく、何をする手なのか）。

    `x` は超過パワー（攻撃・ドン付与だけ）。対象の取り方は
    `theory_order.score_candidate` に揃える（対象が引けない攻撃はリーダー扱い・
    ドン付与はリーダーを相手と見る）が、**パワーは枠から現在値を採る**（無ければ印字に落ちる）。
    `don_k` は**記録の `pol_k`**（その候補の実際の付与枚数・`-1` は DON_BOX でない）。
    """
    at = sig[0] if sig else None
    src = cards.info(cid) if cid else None
    tgt = cards.info(tcid) if tcid else None
    has_target = bool(sig[2]) if (sig and len(sig) > 2) else False
    # **攻撃は `DON_BOX`（対象付き）で来る**（`search/decide.rs::don_box_first_primitive`）
    is_attack = bool(at == "ATTACK" or (at == "DON_BOX" and has_target))
    d = {"at": at, "kind": None, "x": None, "cost": None, "target": None, "x_src": None,
         "is_attack": is_attack}
    d["kind"] = ("attack" if is_attack else
                 ("attach" if at in ("ATTACH_DON", "DON_BOX") else at))
    if src is None:
        return d
    d["cost"] = src.get("cost")
    sp = slot_power(tok_row, si)
    d["x_src"] = "slot" if sp is not None else "printed"
    if sp is None:
        sp = float(src["power"])
    # 記録の `pol_k`（`-1` は DON_BOX でない）を優先し、無ければ `ctx` の仮定に落ちる
    k = float(ctx["don_k"]) if (don_k is None or int(don_k) < 0) else float(don_k)
    d["don_k"] = k
    if is_attack:
        if at == "DON_BOX":
            sp += 1000.0 * k                          # ドンを付けてから殴る
        tp = slot_power(tok_row, ti)
        if tp is None:
            tp = ctx["opp_leader_power"] if tgt is None else float(tgt["power"])
        d["target"] = "leader" if (tgt is None or tgt.get("leader")) else "char"
        d["x"] = _round10(sp - tp)
    elif at in ("ATTACH_DON", "DON_BOX"):
        d["target"] = "leader"
        d["x"] = _round10(sp - ctx["opp_leader_power"])
    return d


def predictions_of(det, theta=THETA, don_k=1):
    """その手が**理論の予言に当たる行動**か（`True` なら「理論はこれを無駄と言う」）。"""
    out = dict.fromkeys(PREDICTIONS, False)
    x = det.get("x")
    if x is None:
        return out
    at = det["at"]
    # **付与枚数は記録に在る**（`pol_k`）＝仮定ではなくその候補の実際の k で判定する
    k = float(det.get("don_k", don_k) or 0.0)
    if at in ("ATTACH_DON", "DON_BOX") and k > 0:
        # 超過 0 の攻撃にドン: c(0) と c(1000) が同じ段なので付けても増えない
        out["p1_don_on_zero"] = bool(0.0 <= x < 1000.0 and
                                     c_of(x + 1000.0 * k) <= c_of(x) + 1e-9)
    out["p2_over_sat"] = bool(x > saturation_x(theta))
    # **攻撃は `DON_BOX`（対象付き）の形で来る**＝`ATTACK` だけ見ていては 1 件も数えられない
    if det.get("is_attack"):
        out["p3_cannot_connect"] = bool(x < 0.0)
    return out


def row_compare(n, q, theory, chosen, n_min=5, q_eps=0.02, n_min_frac=0.05, th_eps=1e-9):
    """1 判断点 → top1 の並びと食い違いの大きさ（値付けできた候補だけで比べる）。

    **`theory_order.row_order` と同じ絞り方**（値付け可 ＆ 訪問の下限）にして、
    順序一致率の母集団と揃える＝2 つの器の数字を並べて読める。

    **理論の最良が同値で並んだ行は判定から外す**（`th_tied`）——`min(c(x), Θ)` は飽和すると
    同値を量産し、付与の増分は平らな段で 0 になるので、**価格は構造的に引き分けを作る**。
    `argmax` は最小 index を返すだけなので、そのまま数えると**器の癖を「探索の食い違い」と
    誤読する**（2026-09-13 に実際に踏んだ。当時は付与枚数を仮定していたので同値がさらに多く、
    同じキャラの k 違いが全部同じ値になっていた＝いまは記録の `pol_k` を使うので区別できる）。

    探索側の物差しは 2 つ出す:
      `i_q` … `Q` の最大（`theory_order` と同じ基準だが**勝者の呪いで上振れする**）
      `i_n` … **訪問の最大＝探索の実際の結論**（選択は訪問で行うので、これが本命）
    """
    n = np.asarray(n, np.float64); q = np.asarray(q, np.float64)
    scored = np.array([t is not None for t in theory], bool)
    keep = scored & (n >= q_floor(n, n_min, n_min_frac))
    ki = np.flatnonzero(keep)
    out = {"k": int(len(n)), "k_kept": int(len(ki))}
    if len(ki) < 2:
        return out
    th = np.array([float(theory[i]) for i in ki], np.float64)
    best = float(th.max())
    out["th_tied"] = bool(int((th >= best - th_eps).sum()) > 1)
    i_th = int(ki[int(np.argmax(th))])
    i_q = int(ki[int(np.argmax(q[ki]))])
    i_n = int(ki[int(np.argmax(n[ki]))])
    out.update(i_th=i_th, i_q=i_q, i_n=i_n,
               i_played=int(chosen) if (0 <= int(chosen) < len(n)) else None,
               q_loss=float(q[i_q] - q[i_th]),
               # 訪問最大を基準にした損＝**勝者の呪いを受けない**側の読み
               q_loss_vs_n=float(q[i_n] - q[i_th]),
               th_loss=float(best - float(theory[i_q])),
               th_loss_vs_n=float(best - float(theory[i_n])),
               # **基準線＝候補を無作為に選んだときの損**（＝最良 − 候補の平均）。
               # これが無いと損の数字は読めない——理論の手の `q_loss` は、理論が Q と無相関な
               # だけでも「最良 − 平均」まで大きく出る。**平均からどれだけ最良側に寄れたか**
               # （`recovered` = 1 − loss/loss_rand・0 が無作為・1 が最良）で読む。
               q_loss_rand=float(q[i_n] - q[ki].mean()),
               th_loss_rand=float(best - th.mean()),
               agree=bool(i_th == i_q),
               agree_n=bool(i_th == i_n),
               # 食い違っても Q の差が雑音床以下＝どちらでも良い手
               tie=bool(abs(float(q[i_q] - q[i_th])) <= q_eps))
    return out


def collect(dirs, holdout_mod=7, limit_games=0, theta=THETA, mu=MU,
            n_min=5, q_eps=0.02, n_min_frac=0.05, don_k=1, bands=("close",), theta_mode="const",
            nu_targets="leader"):
    """holdout の判断点 → 食い違いの記録（既定は接戦帯だけ）。"""
    cards = PL.Cards()
    recs = []
    stats = {"games": 0, "rows": 0, "rows_band": 0, "rows_used": 0, "rows_th_tied": 0,
             "x_from_printed": 0}
    cross = Counter()
    pred_hit = {k: Counter() for k in PREDICTIONS}
    at_dist = Counter()
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                   extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            k = int(L[i])
            if k < 2:
                continue
            stats["rows"] += 1
            band = band_of(float(rows["pol_v0"][i]))
            if bands and band not in bands:
                continue
            stats["rows_band"] += 1
            b = int(ptr[i])
            sc = ex["sc"][i]
            # `Θ` を盤面から出すか定数にするか（`theory_order` と同じ規約・A/B 用）
            th = theta_of(ex["tok"][i], float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]),
                          mode=theta_mode, theta=theta)
            ctx = {"theta": th, "mu": mu,
                   "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                   "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                   "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))),
                   "don_k": don_k}
            tok = ex["tok"][i]
            # `ν` の攻撃項を相手の場も含めた max にするか（`theory_order` と同じ規約・task #39）
            if nu_targets == "board":
                ctx["opp_chars"] = opp_chars_of(tok)
            theory, dets = [], []
            for j in range(b, b + k):
                sig = json.loads(pol["pol_sig"][j])
                cid = str(pol["pol_cid"][j]) or None
                tcid = (str(pol["pol_tcid"][j]) or None) if (len(sig) > 2 and sig[2]) else None
                si, ti = pol["pol_si"][j], pol["pol_ti"][j]
                dk = pol["pol_k"][j]
                theory.append(score_candidate(sig, cid, tcid, ctx, cards,
                                              src_power=slot_power(tok, si),
                                              tgt_power=slot_power(tok, ti), don_k=dk))
                dets.append(cand_detail(sig, cid, tcid, si, ti, ctx, cards, tok, dk))
            r = row_compare(pol["pol_n"][b:b + k], pol["pol_q"][b:b + k], theory,
                            int(rows["pol_chosen"][i]), n_min, q_eps, n_min_frac)
            if "i_th" not in r:
                continue
            # `row_compare` の index は**スライス内**（`pol_chosen` と同じ流儀）＝`b` を引かない
            d_th, d_q, d_n = dets[r["i_th"]], dets[r["i_q"]], dets[r["i_n"]]
            r.update(band=band, seed=seed, turn=int(rows["turn"][i]),
                     z=1.0 if zs.get(int(rows["who"][i]), 0.0) > 0 else 0.0,
                     at_th=d_th["kind"], at_q=d_q["kind"], at_n=d_n["kind"],
                     x_th=d_th["x"], x_q=d_q["x"], x_n=d_n["x"],
                     played_is_th=bool(r.get("i_played") == r["i_th"]))
            # **本物の食い違いだけ**を分類する（同値の行・Q の差が雑音床以下の行は除く）
            if not r["agree_n"] and not r["tie"] and not r["th_tied"]:
                cross[(d_th["kind"], d_n["kind"])] += 1
            # 予言は**探索の結論（訪問最大）**について数える＝理論が「無駄」と言う手を打ったか
            for nm, hit in predictions_of(d_n, theta, don_k).items():
                pred_hit[nm]["hit" if hit else "miss"] += 1
            at_dist[d_n["kind"]] += 1
            recs.append(r)
            stats["rows_used"] += 1
            stats["rows_th_tied"] += int(bool(r["th_tied"]))
            stats["x_from_printed"] += int(d_n.get("x_src") == "printed")
    return recs, stats, cross, pred_hit, at_dist


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def _mean_ci(vals, seeds):
    """平均と**対局でクラスタした** 95% CI（局ごとの平均の分散を使う）。"""
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    per = {}
    for v, s in zip(vals, seeds):
        per.setdefault(s, []).append(float(v))
    g = np.array([np.mean(v) for v in per.values()], np.float64)
    m = float(g.mean())
    if len(g) < 2:
        return {"n": len(vals), "games": len(g), "mean": round(m, 5), "ci95": None}
    se = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {"n": len(vals), "games": len(g), "mean": round(m, 5), "se": round(se, 5),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


def _recovered_ci(rows, loss_key, rand_key, zero_when_agree, reps=400, seed=0):
    """`1 − loss/rand` の**対局クラスタブートストラップ CI**。

    **比の統計量に CI を付けない**と、分母・分子がどちらも小さいので**驚くほど動く**
    ——実測で同じ設定・同じ帯が局の部分集合で 0.17／0.20／0.31 と出た（2026-09-14）。
    """
    if not rows:
        return None
    by = {}
    for r in rows:
        by.setdefault(r["seed"], []).append(r)
    keys = list(by)
    if len(keys) < 3:
        return None
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        sub = [r for k in pick for r in by[keys[k]]]
        loss = float(np.mean([(0.0 if (zero_when_agree and r["agree_n"]) else r[loss_key])
                              for r in sub]))
        rand = float(np.mean([r[rand_key] for r in sub]))
        if rand:
            out.append(1.0 - loss / rand)
    if len(out) < reps // 4:
        return None
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def summarise(recs, cross, pred_hit, at_dist=None, q_eps=0.02, top=8):
    """事実の表（top1 の一致・食い違いの型・損の非対称・予言の頻度・勝敗との符号）。

    **判定に使うのは理論の最良が同値でない行だけ**（`th_tied` を除く）。
    """
    if not recs:
        return {"n": 0}
    ok = [r for r in recs if not r["th_tied"]]
    if not ok:
        return {"n": len(recs), "n_judgeable": 0, "th_tied_share": 1.0}
    sd = [r["seed"] for r in ok]
    real = [r for r in ok if not r["agree_n"] and not r["tie"]]
    rsd = [r["seed"] for r in real]
    out = {"n": len(recs), "n_judgeable": len(ok),
           "th_tied_share": round(1.0 - len(ok) / len(recs), 4),
           "games": len({*sd}),
           # **本命の物差しは訪問最大**（探索の実際の結論）。`q` 版は `theory_order` との対照用
           "top1_th_vs_visits": _mean_ci([1.0 if r["agree_n"] else 0.0 for r in ok], sd),
           "top1_th_vs_qmax": _mean_ci([1.0 if r["agree"] else 0.0 for r in ok], sd),
           "top1_th_vs_played": _mean_ci([1.0 if r["played_is_th"] else 0.0 for r in ok], sd),
           "top1_qmax_vs_visits": _mean_ci(
               [1.0 if r["i_q"] == r["i_n"] else 0.0 for r in ok], sd),
           # 食い違っても Q の差が雑音床以下＝どちらでも良い手
           "tie_share": _mean_ci(
               [1.0 if (not r["agree_n"] and r["tie"]) else 0.0 for r in ok], sd),
           "real_disagreement": len(real),
           # **互いの物差しで測った損**（本物の食い違いの行だけ）
           "q_loss_vs_visits": _mean_ci([r["q_loss_vs_n"] for r in real], rsd),
           "th_loss_vs_visits": _mean_ci([r["th_loss_vs_n"] for r in real], rsd),
           # **無作為な候補の損**（＝最良 − 平均）。損の数字はこれと比べて初めて読める
           "q_loss_random": _mean_ci([r["q_loss_rand"] for r in real], rsd),
           "th_loss_random": _mean_ci([r["th_loss_rand"] for r in real], rsd),
           # `q_loss`（Q 最大を基準）は**勝者の呪いで上振れする**ので参考値として併記する
           "q_loss_vs_qmax_biased": _mean_ci([r["q_loss"] for r in real], rsd),
           "cross_top": [{"theory": a, "search": b, "count": c,
                          "share": round(c / max(1, len(real)), 4)}
                         for (a, b), c in cross.most_common(top)],
           "predictions": {nm: {"hit": c["hit"], "miss": c["miss"],
                                "rate": round(c["hit"] / max(1, c["hit"] + c["miss"]), 4)}
                           for nm, c in pred_hit.items()},
           "q_eps": q_eps}
    # **無作為からどれだけ最良側に寄れたか**（0 = 無作為・1 = 最良）＝損の数字の読み方。
    # **母数は判定できる全行**（一致した行の損は 0）——食い違った行だけで測ると
    # 理論側の損は**定義上 正**になるのに基準線はそうならないので、**理論に不利に偏る**
    # （2026-09-13 に踏んだ: 食い違い限定で −0.51・全行で符号が変わる）。
    okd = [r["seed"] for r in ok]
    out["q_loss_all"] = _mean_ci([r["q_loss_vs_n"] if not r["agree_n"] else 0.0 for r in ok], okd)
    out["th_loss_all"] = _mean_ci([r["th_loss_vs_n"] if not r["agree_n"] else 0.0 for r in ok], okd)
    out["q_loss_random_all"] = _mean_ci([r["q_loss_rand"] for r in ok], okd)
    out["th_loss_random_all"] = _mean_ci([r["th_loss_rand"] for r in ok], okd)
    for side in ("q", "th"):
        for tag, lk, rk in (("", "_loss_all", "_loss_random_all"),
                            ("_on_disagreements", "_loss_vs_visits", "_loss_random")):
            loss = out[f"{side}{lk}"]["mean"]
            rand = out[f"{side}{rk}"]["mean"]
            out[f"{side}_recovered{tag}"] = (round(1.0 - loss / rand, 4)
                                             if (loss is not None
                                                 and rand not in (None, 0.0)) else None)
    # **`*_recovered` には必ず CI を付ける**（2026-09-14）。これは**2 つの小さな平均の比**
    # なので単独では非常に不安定で、**同じ設定でも局の部分集合を替えると 0.17〜0.31 と動く**
    # ——実際にその振れ幅を「設定の差」と読み違えた（`2026-09-14_theta_mode_ab.md` の訂正）。
    # 比は局を復元抽出して作り直す（1 局から多数の行を採るので行で割ってはいけない）。
    rows_for = {"": (ok, "_loss_vs_n", "_loss_rand", True),
                "_on_disagreements": (real, "_loss_vs_n", "_loss_rand", False)}
    for side in ("q", "th"):
        for tag, (sub, lkey, rkey, zero_when_agree) in rows_for.items():
            ci = _recovered_ci(sub, side + lkey, side + rkey, zero_when_agree)
            if ci is not None:
                out[f"{side}_recovered{tag}_ci95"] = ci
    if at_dist:
        tot = sum(at_dist.values())
        out["search_action_mix"] = {a: round(c / tot, 4) for a, c in at_dist.most_common()}
    # 勝敗との符号（**選択交絡あり**＝打った手が理論の手だった行と、そうでない行の勝率差）
    out["z_link"] = {
        "played_is_theory": _mean_ci([r["z"] for r in ok if r["played_is_th"]],
                                     [r["seed"] for r in ok if r["played_is_th"]]),
        "played_is_not_theory": _mean_ci([r["z"] for r in ok if not r["played_is_th"]],
                                         [r["seed"] for r in ok if not r["played_is_th"]])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="const", choices=THETA_MODES,
                    help="`Θ` を定数にするか盤面から出すか（`theory_order` と同じ）")
    ap.add_argument("--nu-targets", default="leader", choices=NU_TARGET_MODES,
                    help="`ν` の攻撃項を対象の max に広げるか（`theory_order` と同じ・task #39）")
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-min-frac", type=float, default=0.05)
    ap.add_argument("--q-eps", type=float, default=0.02, help="Q の雑音床（これ以下の差は同値）")
    ap.add_argument("--don-k", type=int, default=1, help="ドン付与の枚数の仮定（記録に無い）")
    ap.add_argument("--bands", nargs="*", default=["close"],
                    help="見る帯（既定は接戦だけ・`close mid decided` で全部）")
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)

    t0 = time.time()
    res = {"args": {k: v for k, v in vars(a).items() if k != "out"}}
    recs, stats, cross, pred, at_dist = collect(
        a.src, a.holdout_mod, a.limit_games, a.theta, a.mu,
        a.n_min, a.q_eps, a.n_min_frac, a.don_k, tuple(a.bands),
        theta_mode=a.theta_mode, nu_targets=a.nu_targets)
    res["stats"] = stats
    res["summary"] = summarise(recs, cross, pred, at_dist, a.q_eps)
    by_band = {}
    for bn in sorted({r["band"] for r in recs}):
        sub = [r for r in recs if r["band"] == bn]
        cr = Counter((r["at_th"], r["at_n"]) for r in sub
                     if not r["th_tied"] and not r["agree_n"] and not r["tie"])
        mix = Counter(r["at_n"] for r in sub if not r["th_tied"])
        by_band[bn] = summarise(sub, cr, {k: Counter() for k in PREDICTIONS}, mix, a.q_eps)
    res["by_band"] = by_band
    res["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
