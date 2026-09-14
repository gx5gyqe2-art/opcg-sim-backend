"""**身代わりの率を直接測る**（`ν` の欠けている項・読み取り専用）。

`docs/cpu_theory_gap.md` の **T21**。`game_theory.md` §14.1.1 の **#7「身代わり」**——
**キャラが殴られると、その攻撃はリーダーに通らない**＝ライフを 1 枚守ったのと同じ。
式（`theory_order.nu_of`）は**この項を丸ごと持っていない**。

## なぜこれを測るのか

`ν` の実測は**リーダー未満の帯で 0.0690**（CI が 0 を含まない）なのに式は **0** と置く
＝**`ν` 最大の穴**（体数の 40%）。`nu_measure --scheme lt_split` で
**素の体（効果を持たない枠）が 0.0753 を持つ**ことが判った＝
**価値は効果ではなく体そのものに在る**（効果説は棄却）。

残る候補のうち**身代わりだけは記録から直接率が測れる**:

```
身代わり項 = Σ（吸った攻撃 1 本ごとの価値） / 露出 × R
1 本の価値 = min( c(x_lead)·μ , Θ·μ )      x_lead = 攻撃側のパワー − 守る側のリーダーのパワー
```

**1 本を一律 `Θ·μ` と置いてはいけない**（ユーザ指摘 2026-09-14・初版はそう置いて 22% 過大に出した）:

> 「2000 のパワーを吸うのか、5000 を吸うのか、10000 を吸うのかで身代わりの価値は変わる」

**守った価値は「その攻撃がリーダーに行っていたら払わされた額」**＝`min(c(x_lead)·μ, Θ·μ)` で、
`Θ·μ` は**上限**にすぎない（それ以上高い攻撃は「受ける」を選ぶので払う額は増えない）。
そして決定的なのは **`x_lead < 0` なら価値が 0** だということ——
**相手の 3000 の体は 5000 のリーダーにそもそも通らない**ので、それが自分の 2000 のキャラを
殴っても**リーダーは何も守られていない**。実測で**吸った攻撃の 17.6% がこれ**である
（弱い体ほど多く、リーダー未満の帯では 23.9%）。

**`x_lead` は付与ドンを乗せてから測る**（`DON_BOX` は k 枚付けてから殴るマクロ手・`pol_k`）。

**式を触る前に大きさが判る**ので、`nu_measure` の実測と**突き合わせる検算**になる
（`measurement.md` §14-17＝順序の一致率で A/B しない）。

## 測り方

行は**打っている側の視点**なので、枠 **7〜11 が守る側の場**・枠 **0 が打つ側のリーダー**。

- **被弾**: 選ばれた手が `DON_BOX`（対象付き＝攻撃）で `pol_ti` が 7〜11 の回数。
- **露出**: **手番 1 回につき 1 度だけ**、そのとき守る側の場に居たキャラの数。
- **帯**: 守る側の体のパワー − **打つ側のリーダーのパワー**（`nu_measure.power_band` と同一の
  切り方＝`ν` の実測と同じ帯で比べられる）。

> **`ATTACK` という行動型は記録に無い**（攻撃は全部 `DON_BOX`（対象付き）で来る・
> `theory_order` の冒頭）。**`ATTACK` を探すと 0 件になる**（2026-09-13 の実害）。

## 読み方（事前登録）

- **率がパワーに依らなければ**、身代わりは**全帯に一律**に乗る＝`ν` の全体が上がる。
  式は既に全体で +0.041 過大なので、**一律なら悪化する**。
- **率が低パワーほど高ければ**、身代わりは**穴のある帯に集中する**＝
  **`ν` のパワーに対する形そのもの**を直す項になる（`2026-09-14_nu_measure.md` §3 の
  「両端を切り落として真ん中を盛っている」と符合する）。
- **率と 1 本の価値は逆を向きうる**（2026-09-14 に実測でそうなった）——
  弱い体は**よく殴られるが、弱い攻撃しか吸わない**。
  **積（`rate × value`）で見る**のが正しく、率だけを見ると過大に出る。

**限界**: **観察された率であって因果ではない**——殴られやすい体が場に残りやすい／
残りにくいという選択が入る。**粗の値**であり、**殴られて KO される損は引いていない**
（`nu_of` の `ko_p` が別に持っている。二重に引かないよう、式へ入れるときは
`ko_p` をパワーの関数にするのと**同時に**やる必要がある）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/shield_rate.py --in ~/w39 --out ~/shield.json
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
from theory_order import (MU, POL_COLS, PWR_EPS, ROW_COLS, S_IS_CHAR,  # noqa: E402
                          THETA, c_of, slot_power)

#: 守る側の枠（打っている側の視点では「相手の場」）と、打つ側のリーダーの枠
SLOT_FOE_FIELD = range(7, 12)
SLOT_MY_LEADER = 0
#: 守る側のリーダーの枠（打っている側から見た「相手リーダー」）
SLOT_FOE_LEADER = 1
#: 残りターン数の既定（実測・`nu_measure` と同じ）
R_TURNS = 4.128
BANDS = ("lt_leader", "leader_to_sat", "over_sat")


def _extra(dd, n):
    return {"tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def absorb_value(x_lead, theta=THETA, mu=MU):
    """吸った攻撃 1 本の価値＝**リーダーに行っていたら払わされた額**。

    ```
    min( c(x_lead)·μ , Θ·μ )        x_lead < 0 なら 0
    ```

    **`Θ·μ` は上限**（それ以上高い攻撃は「受ける」を選ぶので払う額は増えない）。
    **`x_lead < 0` は 0**——リーダーに通らない攻撃をキャラが吸っても**何も守っていない**
    （ユーザ指摘 2026-09-14。実測で吸った攻撃の 17.6% がこれ）。
    """
    x = float(x_lead)
    if x < -PWR_EPS:
        return 0.0
    return float(min(c_of(x), float(theta)) * float(mu))


def attacker_power(tok, src_slot, don_k):
    """攻撃側の**実効パワー**＝枠の現在値 ＋ 1000 × 付与枚数（`DON_BOX` は付けてから殴る）。"""
    sp = slot_power(tok, src_slot)
    if sp is None:
        return None
    return sp + 1000.0 * max(int(don_k), 0)


def collect(dirs, limit_games=0, theta=THETA, mu=MU):
    """被弾数・露出・**吸った攻撃の価値の和**（帯ごと）。対局ごとの内訳も返す（CI 用）。"""
    hits = {b: 0 for b in BANDS}
    exposure = {b: 0 for b in BANDS}
    value = {b: 0.0 for b in BANDS}          # Σ min(c(x_lead)μ, Θμ)
    no_connect = {b: 0 for b in BANDS}       # x_lead < 0（リーダーには通らなかった）
    per_game = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        g_hits = {b: 0 for b in BANDS}
        g_exp = {b: 0 for b in BANDS}
        g_val = {b: 0.0 for b in BANDS}
        seen = set()
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            tok = ex["tok"][i]
            lead = slot_power(tok, SLOT_MY_LEADER) or 5000.0
            key = (int(rows["who"][i]), int(rows["turn"][i]))
            if key not in seen:            # **手番 1 回につき 1 度だけ**露出を数える
                seen.add(key)
                for s in SLOT_FOE_FIELD:
                    if float(tok[s, S_IS_CHAR]) > 0.5:
                        g_exp[power_band(slot_power(tok, s) or 0.0, lead)] += 1
            ch = int(rows["pol_chosen"][i])
            k, b = int(L[i]), int(ptr[i])
            if ch < 0 or ch >= k:
                continue
            try:
                sig = json.loads(pol["pol_sig"][b + ch])
            except (ValueError, TypeError):
                continue
            # **攻撃は `DON_BOX`（対象付き）**＝`ATTACK` を探すと 0 件になる
            if sig[0] != "DON_BOX" or not (len(sig) > 2 and sig[2]):
                continue
            ti = int(pol["pol_ti"][b + ch])
            if ti not in SLOT_FOE_FIELD:
                continue
            bd = power_band(slot_power(tok, ti) or 0.0, lead)
            g_hits[bd] += 1
            # **1 本の価値は「リーダーに行っていたら払わされた額」**（一律 Θμ ではない）
            sp = attacker_power(tok, int(pol["pol_si"][b + ch]), int(pol["pol_k"][b + ch]))
            foe_lead = slot_power(tok, SLOT_FOE_LEADER) or 5000.0
            if sp is None:
                continue
            x = sp - foe_lead
            g_val[bd] += absorb_value(x, theta, mu)
            if x < -PWR_EPS:
                no_connect[bd] += 1
        for bd in BANDS:
            hits[bd] += g_hits[bd]
            exposure[bd] += g_exp[bd]
            value[bd] += g_val[bd]
        per_game.append((g_hits, g_exp, g_val))
    return hits, exposure, value, no_connect, per_game, games


def _boot_ci(per_game, band, reps=400, seed=0, num=0, scale=1.0):
    """対局を復元抽出して比の CI を出す（比の統計量なので必ず付ける・§14-15）。

    `num` は分子の取り方＝**0 なら被弾数**（率）・**2 なら吸った価値の和**（身代わり項）。
    分母はどちらも露出。`scale` を掛ければそのまま `× R` した項の CI になる。
    """
    if reps <= 0 or len(per_game) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    h = np.array([g[num][band] for g in per_game], np.float64)
    e = np.array([g[1][band] for g in per_game], np.float64)
    n = len(h)
    out = []
    for _ in range(int(reps)):
        k = rng.integers(0, n, n)
        den = e[k].sum()
        if den > 0:
            out.append(h[k].sum() / den * float(scale))
    if not out:
        return None, None
    return (round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4))


#: `nu_measure` が出した実測の `ν`（帯別・`2026-09-14_nu_measure.md`）と、
#: `theory_order` の式が返す値（`--nu-targets board`・`2026-09-14_nu_targets.md`）。
#: **式に空いている余地 = 実測 − 式**＝身代わり項が入れる大きさ。
NU_MEASURED = {"lt_leader": 0.0690, "leader_to_sat": 0.1503, "over_sat": 0.2112}
NU_FORMULA = {"lt_leader": 0.0008, "leader_to_sat": 0.1603, "over_sat": 0.2185}


def summarise(hits, exposure, value, no_connect, per_game, r_turns=R_TURNS,
              theta=THETA, mu=MU, reps=400):
    """帯ごとの率・**1 本の価値**・身代わり項（粗）。

    ```
    身代わり項（粗） = （吸った価値の和 / 露出）× R
    ```

    **率だけでは出せない**——弱い体はよく殴られるが**弱い攻撃しか吸わない**ので、
    一律 `Θ·μ` を掛けると過大に出る（初版がそれで 22% 高かった・ユーザ指摘 2026-09-14）。
    """
    tm = float(theta) * float(mu)
    rows = {}
    for bd in BANDS:
        h, e, v = hits[bd], exposure[bd], value[bd]
        rate = (h / e) if e else 0.0
        per_absorb = (v / h) if h else 0.0
        gross = (v / e * float(r_turns)) if e else 0.0
        lo, hi = _boot_ci(per_game, bd, reps)
        slo, shi = _boot_ci(per_game, bd, reps, num=2, scale=float(r_turns))
        room = NU_MEASURED[bd] - NU_FORMULA[bd]
        rows[bd] = {
            "hits": h, "exposure": e,
            "rate_per_turn": round(rate, 4),
            "rate_ci95": [lo, hi],
            # **1 本あたりの価値**＝リーダーに行っていたら払わされた額（一律 Θμ ではない）
            "value_per_absorb": round(per_absorb, 5),
            "flat_theta_mu": round(tm, 5),
            "no_connect": no_connect[bd],
            "no_connect_share": round(no_connect[bd] / h, 4) if h else 0.0,
            "absorbs_over_r": round(rate * float(r_turns), 4),
            "shield_gross": round(gross, 4),
            "shield_ci95": [slo, shi],
            # 一律 Θμ で置いたときの値＝**どれだけ過大だったか**を残す
            "shield_gross_flat": round(rate * float(r_turns) * tm, 4),
            "nu_measured": NU_MEASURED[bd], "nu_formula": NU_FORMULA[bd],
            "room_in_formula": round(room, 4),
            "fits_in_room": bool(room > 0),
            "covers_room": (round(gross / room, 3) if room > 0 else None),
        }
    tot_h = sum(hits.values())
    tot_e = sum(exposure.values())
    tot_v = sum(value.values())
    r = [rows[b]["rate_per_turn"] for b in BANDS]
    val = [rows[b]["value_per_absorb"] for b in BANDS]
    g = [rows[b]["shield_gross"] for b in BANDS]
    return {"by_band": rows,
            "rate_all": round(tot_h / tot_e, 4) if tot_e else 0.0,
            "value_per_absorb_all": round(tot_v / tot_h, 5) if tot_h else 0.0,
            "no_connect_share_all": (round(sum(no_connect.values()) / tot_h, 4)
                                     if tot_h else 0.0),
            "theta_mu": round(tm, 4), "r_turns": float(r_turns),
            # **事前登録の読み方**: 低パワーほど厚ければ「形を直す項」・一律なら「悪化する項」
            "rate_falls_with_power": bool(r[0] > r[1] > r[2]),
            # **率と 1 本の価値は逆を向く**（弱い体はよく殴られるが弱い攻撃しか吸わない）
            "value_rises_with_power": bool(val[0] < val[1] < val[2]),
            "shield_falls_with_power": bool(g[0] > g[1] > g[2]),
            "rate_ratio_lt_over_high": (round(r[0] / r[2], 2) if r[2] else None),
            "shield_ratio_lt_over_high": (round(g[0] / g[2], 2) if g[2] else None)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--r-turns", type=float, default=R_TURNS)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--boot-reps", type=int, default=400)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    hits, exposure, value, no_connect, per_game, games = collect(
        a.src, a.limit_games, a.theta, a.mu)
    res = {"games": games,
           "summary": summarise(hits, exposure, value, no_connect, per_game,
                                a.r_turns, a.theta, a.mu, a.boot_reps),
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
