"""**理論の順序は探索の Q に勝てるか**（候補 F の前提の直接検査・読み取り専用）。

`docs/cpu_theory_gap.md` §8.1 の **T1**。`docs/game_theory.md` の 4 通貨の価格で候補手を
並べ、探索の `Q` との順序一致を測って、**方策の事前分布 `p` の 0.5016 と比べる**。

```
order_acc(理論, Q)  vs  order_acc(p, Q) = 0.5016（接戦帯・実測）
```

**なぜこれが F の前提か**: F は「価格を方策の順序の教師にする」案である。
教師にする前に、**その価格で並べた順序が探索の結論とどれだけ合うか**を測れる。
合わなければ、価格を教師にしても方策は良くならない＝設計を変える。

## 候補の値付け（`game_theory.md` §14.1）

| 候補 | 値 |
|---|---|
| リーダーへの攻撃 | `min( c(x)·μ, Θ·μ )` … 相手が「守る／受ける」の安い方を選ぶので **min** |
| キャラへの攻撃 | `min( c(x)·μ, ν(対象) )` … 守るか、そのキャラを失うか |
| 登場（キャラ） | `ν(自分のキャラ) − μ − cost·δ` … 手札とドンで場を買う |
| ドン付与 | `( c(x+1000) − c(x) )·μ` … 段を 1 つ上げる。**飽和点 `x*` を超えたら 0** |
| ターン終了 | 0 |
| 起動メイン・イベント | **値付けできない**（効果の中身が要る）＝ペアから外す |

`x = 攻撃側のパワー − 対象のパワー`。`x < 0` は**通らないので 0**（`game_theory.md` §14.1）。

## 読み方（事前登録）

- **`order_acc(理論, Q)` が 0.55 を超えれば、価格は方策の教師として使える**＝F の前提は成立。
  0.50 付近なら、価格で並べても探索の結論に近づかない＝**F の設計を変える**。
- **`coverage`（値付けできたペアの割合）を必ず併記する**。値付けできない候補
  （起動メイン・イベント）を外しているので、coverage が低ければ「攻撃と登場だけの話」になる。
- 比較の相手は**同じ行・同じペア集合での `order_acc(p, Q)`**（本器が両方を同時に出す）。
  別の計測（`order_acc.py`）の 0.5016 とは母集団が違うので、**本器の中で比べる**。

**限界**: 探索の `Q` を正としている（`order_acc.py` と同じ）。`Q` 自体が誤っている可能性は
本器では分けられない。`ν` は近似（下記の `--r-turns` と `--block-p`）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_order.py --in ~/w32/*/n_records \\
    --holdout-mod 7 --out ~/theory_order_w32.json
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
from order_acc import band_of, pair_agree, q_floor  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_n", "pol_q", "pol_p", "pol_sig", "pol_cid", "pol_tcid")

#: 実測の価格（`docs/game_theory.md` §18・勝率の単位）。相手ターンの値を既定にする
#: ——守る／受けるの判断は相手ターンに起きるので、攻撃の値付けはそちらの価格で見る。
MU = 0.0551
LAM = 0.1362
#: 無差別点 `Θ`（枚）＝`λ/μ − 1 − τ_value` の区間 [0.8, 1.5] の中央（`--theta` で変えられる）
THETA = 1.15
#: 費用曲線 `c(x)`＝x を止めるのに要る枚数（合成 800 デッキの実測・§18）
CBAR_CURVE = ((1000, 1.00), (2000, 1.28), (3000, 2.25), (4000, 2.78), (5000, 3.63))
#: 5000 を超えた分の傾き（1000 あたり・実測の平均）
CBAR_SLOPE = 0.66
#: f16 の丸め対策（ちょうど 1000 の倍数が 2000.0002 になる・`budget_audit` と同じ）
PWR_EPS = 10.0
#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs`）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_HAND = 6
SC_TURN = 10
SC_MY_LEADER_POWER, SC_OPP_LEADER_POWER = 12, 13
#: 値付けできる行動（できないものはペアから外す）
SCORABLE = ("ATTACK", "ATTACH_DON", "PLAY", "TURN_END", "DON_BOX")


def c_of(x):
    """費用曲線: 超過 `x` の攻撃を止めるのに要る枚数。

    **`x = 0` でも 1 枚要る**——ルールは「攻撃側のパワー ≥ 対象のパワー」で命中するので、
    生き残るにはカウンターで**上回る**必要がある。したがって
    `x < 0`（そもそも通らない）だけが 0 で、`0 ≤ x ≤ 1000` は 1 枚。
    **ここを 0 にすると、実測で打った攻撃の 38.5% を占める `x ≤ 0` の帯の値付けが狂う**。
    """
    x = float(x)
    if x < -PWR_EPS:
        return 0.0                       # 通らない攻撃＝守る必要が無い
    prev = 0.0
    for thr, cards in CBAR_CURVE:
        if x <= thr + PWR_EPS:
            return cards
        prev = cards
    over = (x - CBAR_CURVE[-1][0]) / 1000.0
    return prev + CBAR_SLOPE * over


def saturation_x(theta=THETA):
    """飽和点 `x* = min{ x : c(x) ≥ Θ }`（`game_theory.md` §14.1）。"""
    for thr, cards in CBAR_CURVE:
        if cards >= theta:
            return float(thr)
    return float(CBAR_CURVE[-1][0])


def attack_value(power, target_power, is_leader, theta=THETA, mu=MU, nu_target=None):
    """攻撃 1 回の価値＝**相手が安い方を選ぶので min**（`game_theory.md` §14.1）。

    リーダー狙い: `min(c(x)·μ, Θ·μ)`。キャラ狙い: `min(c(x)·μ, ν(対象))`。
    `x < 0` は通らないので 0。
    """
    x = float(power) - float(target_power)
    if x < -PWR_EPS:
        return 0.0                       # 通らない＝価値 0（テンポだけ払う・§14.2）
    guard = c_of(x) * mu
    take = (theta * mu) if is_leader else (
        float(nu_target) if nu_target is not None else theta * mu)
    return float(min(guard, take))


def nu_of(power, opp_leader_power, r_turns, theta=THETA, mu=MU, block_p=0.3, ko_p=0.25):
    """場のキャラ 1 体の価格 `ν`（`game_theory.md` §14.1 の近似）。

    残り `r_turns` ターンぶんの攻撃の価値＋ブロックの option value − KO される損。
    `block_p`・`ko_p` は帯によって動く量だが、**順序を測るのが目的なので定数で近似**する
    （全候補に同じ定数が乗るので、同じ種類の候補どうしの順序には影響しない）。
    """
    per_turn = attack_value(power, opp_leader_power, True, theta, mu)
    atk = float(r_turns) * per_turn
    block = block_p * theta * mu                    # 1 回ぶんの攻撃を消す価値
    return atk + block - ko_p * (atk + block)


def attach_value(power, target_power, k=1, theta=THETA, mu=MU):
    """ドン付与 `k` 枚の価値＝**攻撃の価値の増分**（`attack_value` と厳密に整合させる）。

    ```
    attach = ( min(c(x+1000k), Θ) − min(c(x), Θ) ) · μ
    ```

    `Θ` で潰すのは §14.1 の飽和（相手が「受ける」を選んだらそれ以上払わせられない）。
    **段が平らな区間では 0 になる**——実測の曲線は `c(0) = c(1000) = 1.00` なので、
    **超過 0 の攻撃に 1 枚付与しても相手の費用は増えない**（検査できる予測）。
    """
    x0 = float(power) - float(target_power)
    if x0 < -PWR_EPS:
        return 0.0                       # 通らない攻撃は付与しても通らない
    x1 = x0 + 1000.0 * int(k)
    return (min(c_of(x1), theta) - min(c_of(x0), theta)) * mu


def play_value(power, cost, opp_leader_power, r_turns, theta=THETA, mu=MU, delta=None):
    """登場の価値＝`ν − μ − cost·δ`（手札 1 枚とドンで場を買う・§14.1）。

    `δ`（ドン 1 個の価値）の既定は理論値 `Δpressure(1000)·μ ≈ 0.66·μ`（§13）。
    """
    d = (0.66 * mu) if delta is None else float(delta)
    return nu_of(power, opp_leader_power, r_turns, theta, mu) - mu - float(cost) * d


def score_candidate(sig, cid, tcid, ctx, cards):
    """候補 1 つの理論値（値付けできなければ `None`）。

    `sig` は `[action_type, uuid, target_ids, selected_uuids, accepted]`（`record_gen.move_sig`）。
    `ctx` は `{"opp_leader_power", "r_turns", "theta", "mu"}`。
    """
    at = sig[0] if sig else None
    if at not in SCORABLE:
        return None
    if at == "TURN_END":
        return 0.0
    src = cards.info(cid) if cid else None
    tgt = cards.info(tcid) if tcid else None
    theta, mu = ctx["theta"], ctx["mu"]
    if at == "ATTACK":
        if src is None:
            return None
        if tgt is None:                              # 対象のカードが引けない＝リーダー扱い
            return attack_value(src["power"], ctx["opp_leader_power"], True, theta, mu)
        if tgt.get("leader"):
            return attack_value(src["power"], tgt["power"], True, theta, mu)
        nu_t = nu_of(tgt["power"], ctx["my_leader_power"], ctx["r_turns"], theta, mu)
        return attack_value(src["power"], tgt["power"], False, theta, mu, nu_target=nu_t)
    if at in ("ATTACH_DON", "DON_BOX"):
        if src is None:
            return None
        # **`don_k` は記録に無い**（`move_sig` は [action_type, uuid, target_ids,
        # selected_uuids, accepted] の 5 要素で、付与枚数は payload にしか無い）。
        # そこで仮定値 `ctx["don_k"]` を使い、`--don-k` で感度を見られるようにする。
        return attach_value(src["power"], ctx["opp_leader_power"], ctx["don_k"], theta, mu)
    if at == "PLAY":
        if src is None or src.get("event"):
            return None                              # イベントは効果の中身が要る
        return play_value(src["power"], src.get("cost") or 0,
                          ctx["opp_leader_power"], ctx["r_turns"], theta, mu)
    return None


def row_order(n, q, p, theory, n_min=5, q_eps=0.02, p_eps=1e-4, n_min_frac=0.05):
    """1 判断点の順序一致（理論 vs Q・方策 vs Q を**同じペア集合**で）。"""
    n = np.asarray(n, np.float64); q = np.asarray(q, np.float64)
    p = np.asarray(p, np.float64)
    scored = np.array([t is not None for t in theory], bool)
    floor = q_floor(n, n_min, n_min_frac)
    keep = scored & (n >= floor)
    out = {"k": int(len(n)), "k_scored": int(scored.sum()), "k_kept": int(keep.sum())}
    if keep.sum() < 2:
        out.update(th_agree=0, th_pairs=0, p_agree=0, p_pairs=0)
        return out
    t = np.array([float(theory[i]) for i in range(len(theory)) if keep[i]], np.float64)
    a1, n1 = pair_agree(t, q[keep], 0.0, q_eps)
    a2, n2 = pair_agree(p[keep], q[keep], p_eps, q_eps)
    out.update(th_agree=a1, th_pairs=n1, p_agree=a2, p_pairs=n2)
    return out


def collect(dirs, holdout_mod=7, limit_games=0, theta=THETA, mu=MU,
            n_min=5, q_eps=0.02, n_min_frac=0.05, don_k=1):
    cards = PL.Cards()
    recs = []
    stats = {"games": 0, "rows": 0, "rows_used": 0, "cand": 0, "cand_scored": 0}
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
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            k = int(L[i])
            if k < 2:
                continue
            stats["rows"] += 1
            b = int(ptr[i])
            sc = ex["sc"][i]
            ctx = {"theta": theta, "mu": mu,
                   "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                   "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                   "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))),
                   "don_k": don_k}
            theory = []
            for j in range(b, b + k):
                sig = json.loads(pol["pol_sig"][j])
                tcid = None
                tl = sig[2] if len(sig) > 2 else None
                if tl:
                    tcid = str(pol["pol_tcid"][j]) or None
                theory.append(score_candidate(sig, str(pol["pol_cid"][j]) or None,
                                              tcid, ctx, cards))
            stats["cand"] += k
            stats["cand_scored"] += sum(1 for t in theory if t is not None)
            r = row_order(pol["pol_n"][b:b + k], pol["pol_q"][b:b + k], pol["pol_p"][b:b + k],
                          theory, n_min, q_eps, n_min_frac=n_min_frac)
            if r["th_pairs"] == 0 and r["p_pairs"] == 0:
                continue
            r["band"] = band_of(float(rows["pol_v0"][i]))
            r["seed"] = seed
            r["turn"] = int(rows["turn"][i])
            recs.append(r)
            stats["rows_used"] += 1
    return recs, stats


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32)}


def block(recs):
    """帯ごとの集計（**理論と方策を同じペア集合で**比べる）。"""
    if not recs:
        return {"n": 0}
    th_a = sum(r["th_agree"] for r in recs); th_t = sum(r["th_pairs"] for r in recs)
    p_a = sum(r["p_agree"] for r in recs); p_t = sum(r["p_pairs"] for r in recs)
    out = {"n": len(recs), "th_pairs": th_t, "p_pairs": p_t,
           "order_acc_theory": round(th_a / th_t, 4) if th_t else None,
           "order_acc_prior": round(p_a / p_t, 4) if p_t else None,
           "k_mean": round(float(np.mean([r["k"] for r in recs])), 2),
           "coverage_cand": round(float(np.mean([r["k_scored"] / r["k"] for r in recs])), 4),
           "kept_mean": round(float(np.mean([r["k_kept"] for r in recs])), 2)}
    if out["order_acc_theory"] is not None and out["order_acc_prior"] is not None:
        out["gain"] = round(out["order_acc_theory"] - out["order_acc_prior"], 4)
    # 対局でクラスタした SE（1 局から多数の行を採るので行数で割らない）
    by = {}
    for r in recs:
        if r["th_pairs"]:
            by.setdefault(r["seed"], []).append(r["th_agree"] / r["th_pairs"])
    if len(by) >= 2:
        m = np.array([float(np.mean(v)) for v in by.values()], np.float64)
        se = float(m.std(ddof=1) / np.sqrt(len(m)))
        out["games"] = len(m)
        out["theory_se"] = round(se, 4)
        out["theory_ci95"] = [round(float(m.mean()) - 1.96 * se, 4),
                             round(float(m.mean()) + 1.96 * se, 4)]
    return out


def verdict(b):
    """事前登録: **理論の順序が 0.55 を超えれば価格は方策の教師として使える**。"""
    if not b or not b.get("n") or b.get("order_acc_theory") is None:
        return None
    if b["order_acc_theory"] >= 0.55:
        return "price_teaches"
    if b["order_acc_theory"] <= 0.52:
        return "price_does_not_teach"
    return "partly"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA, help="無差別点（枚・既定は実測の中央 1.15）")
    ap.add_argument("--mu", type=float, default=MU, help="手札 1 枚の価格（既定は相手ターンの実測）")
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-min-frac", type=float, default=0.05)
    ap.add_argument("--q-eps", type=float, default=0.02)
    ap.add_argument("--don-k", type=int, default=1,
                    help="DON_BOX の付与枚数の仮定（記録に無いので感度を見る・既定 1）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, stats = collect(a.src, a.holdout_mod, a.limit_games, a.theta, a.mu,
                          a.n_min, a.q_eps, a.n_min_frac, a.don_k)
    allb = block(recs)
    res = {"stats": stats, "all": allb, "verdict": verdict(allb),
           "by_band": {b: block([r for r in recs if r["band"] == b])
                       for b in ("close", "mid", "decided")},
           "saturation_x": saturation_x(a.theta),
           "args": {k: v for k, v in vars(a).items() if k != "out"},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
