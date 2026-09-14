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
身代わり項 = （残り R ターンで殴られる回数）× Θ·μ
           = （1 手番あたり被弾率）× R × Θ·μ
```

`Θ·μ` は**リーダーへの攻撃 1 回を消す価値**（`attack_value` の `take` と同じ量）。
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
from theory_order import (MU, POL_COLS, ROW_COLS, S_IS_CHAR, THETA,  # noqa: E402
                          slot_power)

#: 守る側の枠（打っている側の視点では「相手の場」）と、打つ側のリーダーの枠
SLOT_FOE_FIELD = range(7, 12)
SLOT_MY_LEADER = 0
#: 残りターン数の既定（実測・`nu_measure` と同じ）
R_TURNS = 4.128
BANDS = ("lt_leader", "leader_to_sat", "over_sat")


def _extra(dd, n):
    return {"tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def collect(dirs, limit_games=0):
    """被弾数と露出（帯ごと）を数える。対局ごとの内訳も返す（クラスタ CI 用）。"""
    hits = {b: 0 for b in BANDS}
    exposure = {b: 0 for b in BANDS}
    per_game = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        g_hits = {b: 0 for b in BANDS}
        g_exp = {b: 0 for b in BANDS}
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
            if ti in SLOT_FOE_FIELD:
                g_hits[power_band(slot_power(tok, ti) or 0.0, lead)] += 1
        for bd in BANDS:
            hits[bd] += g_hits[bd]
            exposure[bd] += g_exp[bd]
        per_game.append((g_hits, g_exp))
    return hits, exposure, per_game, games


def _boot_ci(per_game, band, reps=400, seed=0):
    """対局を復元抽出して率の CI を出す（比の統計量なので必ず付ける・§14-15）。"""
    if reps <= 0 or len(per_game) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    h = np.array([g[0][band] for g in per_game], np.float64)
    e = np.array([g[1][band] for g in per_game], np.float64)
    n = len(h)
    out = []
    for _ in range(int(reps)):
        j = rng.integers(0, n, n)
        den = e[j].sum()
        if den > 0:
            out.append(h[j].sum() / den)
    if not out:
        return None, None
    return (round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4))


#: `nu_measure` が出した実測の `ν`（帯別・`2026-09-14_nu_measure.md`）と、
#: `theory_order` の式が返す値（`--nu-targets board`・`2026-09-14_nu_targets.md`）。
#: **式に空いている余地 = 実測 − 式**＝身代わり項が入れる大きさ。
NU_MEASURED = {"lt_leader": 0.0690, "leader_to_sat": 0.1503, "over_sat": 0.2112}
NU_FORMULA = {"lt_leader": 0.0008, "leader_to_sat": 0.1603, "over_sat": 0.2185}


def summarise(hits, exposure, per_game, r_turns=R_TURNS, theta=THETA, mu=MU, reps=400):
    """帯ごとの率と、そこから出る身代わり項（粗）。"""
    tm = float(theta) * float(mu)
    rows = {}
    for bd in BANDS:
        h, e = hits[bd], exposure[bd]
        rate = (h / e) if e else 0.0
        lo, hi = _boot_ci(per_game, bd, reps)
        room = NU_MEASURED[bd] - NU_FORMULA[bd]
        rows[bd] = {
            "hits": h, "exposure": e,
            "rate_per_turn": round(rate, 4),
            "rate_ci95": [lo, hi],
            "absorbs_over_r": round(rate * float(r_turns), 4),
            "shield_gross": round(rate * float(r_turns) * tm, 4),
            "nu_measured": NU_MEASURED[bd], "nu_formula": NU_FORMULA[bd],
            "room_in_formula": round(room, 4),
            "fits_in_room": bool(room > 0),
        }
    tot_h = sum(hits.values())
    tot_e = sum(exposure.values())
    r = [rows[b]["rate_per_turn"] for b in BANDS]
    return {"by_band": rows,
            "rate_all": round(tot_h / tot_e, 4) if tot_e else 0.0,
            "theta_mu": round(tm, 4), "r_turns": float(r_turns),
            # **事前登録の読み方**: 低パワーほど高ければ「形を直す項」・一律なら「悪化する項」
            "rate_falls_with_power": bool(r[0] > r[1] > r[2]),
            "rate_ratio_lt_over_high": (round(r[0] / r[2], 2) if r[2] else None)}


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
    hits, exposure, per_game, games = collect(a.src, a.limit_games)
    res = {"games": games,
           "summary": summarise(hits, exposure, per_game, a.r_turns, a.theta, a.mu,
                                a.boot_reps),
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
