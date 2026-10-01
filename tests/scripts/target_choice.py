"""**攻撃の対象選びは正しかったか**（身代わり率の内生性を点検する・読み取り専用）。

ユーザ指摘 2026-09-14:

> 「身代わりが有効な攻撃を吸っているケースでも、攻撃する側が相手の低パワーキャラクターの
> 価値を誤っていた場合は、身代わりに攻撃する必要がないはずなのに攻撃してしまっている
> 場合もあると思います」

**これは `shield_rate.py` の妥当性そのものへの指摘である**。身代わり率は**自己対戦の
攻撃判断から出ている**＝**攻撃側が誤って弱いキャラを殴っていれば、その誤りを
「身代わりの価値」として計上してしまう**。率はゲームの性質ではなく**打ち手の性質**になる。

## 測り方

攻撃 1 本ごとに、**同じ体で**「リーダーを殴る」と「盤面のキャラを殴る」を値付けして比べる:

```
リーダー狙い  vL = min( c(x_lead)·μ , Θ·μ )        x_lead = 実効パワー − 守る側のリーダー
キャラ狙い    vC = min( c(x_char)·μ , ν(対象の帯) )  ν は **実測値**（式の値ではない）
```

- **キャラを殴ったのに `vL > vC`** → **殴る必要が無かった**（身代わり率を水増ししている）
- **リーダーを殴ったのに `max_T vC > vL`** → **殴るべきキャラが居た**（率は過小）

**`ν` は実測（`nu_measure` の帯別）を使う**——式の `ν` はこの帯で 0 なので、
式で判定すると「キャラ狙いは常に誤り」という自明な答えしか出ない。

## 読み方（事前登録）

- **「最適な吸収 / 観測」が 1 を下回れば、ユーザ指摘のとおり身代わり率は水増しされている**。
  1 を上回れば逆＝**CPU はキャラを殴らなさすぎ**で、身代わり項は**過小**。
- **判定は `Θ` に強く依る**ので、**必ず掃引する**（`--theta` を複数渡す）。
  向きが全ての `Θ` で揃って初めて主張になる。

**限界（重い）**:

- **`Θ` を定数で置いている**。実測の `Θ` はライフで **8 倍**動く（`2026-09-14_theta_price.md`）ので、
  **リーサル圏の判定は信用できない**——既定の 1.15 では
  **守る側のライフ 0 でリーダー狙いの 77.9% が「誤り」**という明らかにおかしい答えが出る。
  **`--by-life` でその壊れ方を見る**こと。
- **`ν` の実測も同じ自己対戦から来ている**＝循環が完全には切れていない。
- **盤面の他の要素を見ていない**（ブロッカーの応手・カウンター・除去の誘発・リーサル計算）。
  ＝**「誤り」の絶対水準は読めない。読めるのは向きと感度だけ**。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/target_choice.py --in ~/w39 \\
    --theta 1.15 1.62 2.64 --by-life --out ~/target_choice.json
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
from nu_measure import power_band  # noqa: E402
from shield_rate import (NU_MEASURED, SLOT_FOE_FIELD, SLOT_FOE_LEADER,  # noqa: E402
                         SLOT_MY_LEADER, attacker_power)
from theory_order import (MU, POL_COLS, PWR_EPS, ROW_COLS, S_IS_CHAR,  # noqa: E402
                          THETA, c_of, slot_power)

#: 打つ側から見た「相手のライフ」＝**守る側のライフ**（`scalars` の列）
SC_FOE_LIFE = 1
#: 既定で掃く `Θ`（1.15＝出荷既定・2.64＝実測のライフ 0）
THETA_SWEEP = (1.15, 1.62, 2.64, 5.00)


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def char_value(power, target_power, band, mu=MU, nu=None):
    """キャラ狙いの価値＝`min( c(x)·μ , ν(対象) )`。**`ν` は実測の帯別**。"""
    x = float(power) - float(target_power)
    if x < -PWR_EPS:
        return 0.0
    table = NU_MEASURED if nu is None else nu
    return float(min(c_of(x) * float(mu), table[band]))


def leader_value(x_lead, theta=THETA, mu=MU):
    """リーダー狙いの価値＝`min( c(x)·μ , Θ·μ )`。**`x < 0` は 0**（通らない）。"""
    x = float(x_lead)
    if x < -PWR_EPS:
        return 0.0
    return float(min(c_of(x), float(theta)) * float(mu))


def collect(dirs, limit_games=0, mu=MU):
    """攻撃 1 本 → `(x_lead, キャラを殴ったか, その対象の値, 盤面で最良の値, 守る側ライフ)`。

    **`Θ` に依らない量だけを貯める**ので、掃引は集計側で何度でもやり直せる。
    """
    rows_out = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            k, b = int(L[i]), int(ptr[i])
            ch = int(rows["pol_chosen"][i])
            if ch < 0 or ch >= k:
                continue
            j = b + ch
            try:
                sig = json.loads(pol["pol_sig"][j])
            except (ValueError, TypeError):
                continue
            # **攻撃は `DON_BOX`（対象付き）**＝`ATTACK` を探すと 0 件になる
            if sig[0] != "DON_BOX" or not (len(sig) > 2 and sig[2]):
                continue
            ti = int(pol["pol_ti"][j])
            if ti != SLOT_FOE_LEADER and ti not in SLOT_FOE_FIELD:
                continue
            tok = ex["tok"][i]
            power = attacker_power(tok, int(pol["pol_si"][j]), int(pol["pol_k"][j]))
            if power is None:
                continue
            my_lead = slot_power(tok, SLOT_MY_LEADER) or 5000.0
            best = cur = 0.0
            for s in SLOT_FOE_FIELD:
                if float(tok[s, S_IS_CHAR]) <= 0.5:
                    continue
                tp = slot_power(tok, s) or 0.0
                v = char_value(power, tp, power_band(tp, my_lead), mu)
                best = max(best, v)
                if s == ti:
                    cur = v
            foe_lead = slot_power(tok, SLOT_FOE_LEADER) or 5000.0
            rows_out.append({
                "x_lead": power - foe_lead,
                "hit_char": ti in SLOT_FOE_FIELD,
                "v_actual": cur, "v_best_char": best,
                "foe_life": int(round(float(ex["sc"][i][SC_FOE_LIFE]))),
                "seed": int(rows["seed"][i]),
            })
    return rows_out, games


def judge(recs, theta=THETA, mu=MU):
    """1 つの `Θ` での判定。"""
    hb = ho = fb = fo = 0
    loss = []
    missed = []
    for r in recs:
        v_lead = leader_value(r["x_lead"], theta, mu)
        if r["hit_char"]:
            if v_lead > r["v_actual"] + 1e-9:
                hb += 1
                loss.append(v_lead - r["v_actual"])
            else:
                ho += 1
        else:
            if r["v_best_char"] > v_lead + 1e-9:
                fb += 1
                missed.append(r["v_best_char"] - v_lead)
            else:
                fo += 1
    n_char, n_face = ho + hb, fo + fb
    return {
        "theta": float(theta),
        "char_attacks": n_char, "face_attacks": n_face,
        "char_wrong": hb, "char_wrong_share": round(hb / n_char, 4) if n_char else None,
        "face_wrong": fb, "face_wrong_share": round(fb / n_face, 4) if n_face else None,
        "char_wrong_loss": round(float(np.mean(loss)), 5) if loss else 0.0,
        "face_wrong_missed": round(float(np.mean(missed)), 5) if missed else 0.0,
        # **これが読み取るべき量**: 1 を下回れば身代わり率は水増し・上回れば過小
        "optimal_absorbs_over_observed": (round((ho + fb) / n_char, 3) if n_char else None),
    }


def by_life(recs, theta=THETA, mu=MU, min_n=30):
    """**守る側のライフ別**——定数 `Θ` の壊れ方がここに出る。

    ライフ 0 でリーダー狙いの誤りが跳ね上がるなら、それは CPU ではなく `Θ` の誤りである
    （次の一撃で倒せる場面でリーダーを殴るのが誤りのはずがない）。
    """
    out = {}
    for lf in sorted({r["foe_life"] for r in recs}):
        sub = [r for r in recs if r["foe_life"] == lf]
        if len(sub) < min_n:
            continue
        out[str(lf)] = judge(sub, theta, mu)
    return out


def verdict(sweep):
    """**向きが全ての `Θ` で揃って初めて主張になる**（事前登録の読み方）。"""
    vals = [s["optimal_absorbs_over_observed"] for s in sweep
            if s["optimal_absorbs_over_observed"] is not None]
    if not vals:
        return None
    if all(v > 1.0 for v in vals):
        return "cpu_under_attacks_characters"     # 身代わり項は**過小**
    if all(v < 1.0 for v in vals):
        return "cpu_over_attacks_characters"      # ユーザ指摘のとおり**水増し**
    return "depends_on_theta"                     # 判定できない


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, nargs="+", default=list(THETA_SWEEP),
                    help="掃引する `Θ`（**判定がこれに強く依るので必ず複数**）")
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--by-life", action="store_true",
                    help="守る側のライフ別に出す（**定数 `Θ` の壊れ方が見える**）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, games = collect(a.src, a.limit_games, a.mu)
    sweep = [judge(recs, th, a.mu) for th in a.theta]
    res = {"games": games, "attacks": len(recs), "sweep": sweep, "verdict": verdict(sweep),
           "args": {k: v for k, v in vars(a).items() if k != "out"},
           "seconds": round(time.time() - t0, 1)}
    if a.by_life:
        res["by_life"] = {str(th): by_life(recs, th, a.mu) for th in a.theta}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
