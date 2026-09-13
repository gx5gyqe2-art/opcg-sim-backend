"""攻め側の予算——**掛けた圧力 vs 掛けられた圧力**（ドン付与の使い切りを数える）
（ユーザ指摘 2026-08-12「CPU はドン付与が苦手」＋「7000 理論」・`docs/life_budget.md` §9・読み取り専用）。

`budget_audit.py` は守り側の勘定。こちらはその鏡。§9 の恒等式:

> 攻撃は必ず相手に損失を強いる（相手の手札 −c か ライフ−1・手札+1）＝「止められる＝無駄」ではない

したがって攻め側の物差しは**相手に払わせる枚数**:

```
force(攻撃 i) = ceil(max(0, x_i) / 1000)        x_i = 自分の攻撃パワー − 相手リーダーのパワー
force_actual  = Σ_i force(実際に打った攻撃)
force_max     = ドンの配り方を最適化したときの Σ force（**同じ攻撃回数・同じドン**）
force_gap     = force_max − force_actual        ← **付与を使い切れていない枚数**
```

`force_gap > 0` は「同じ手数・同じドンで、相手にもっと払わせられたのに払わせていない」＝
**ユーザの指摘（付与が苦手）が実測で成立する形**。さらに `x_i` の分布を 1000 刻みで出すので、
**7000 理論**（1 回の攻撃を 2 枚のカウンターが要る高さに積む）が起きているかも読める。

### 測り方（記録だけ）

自席ターンの**最初の main 行**で盤面を読む（`power_now` は手番側で評価されるので、自席ターンの
自分の枠は攻撃時のパワー・相手リーダーの枠は守りのパワー）:

- `attackers` … 自分の枠のうち `can_attack_now` が立っているもののパワー（リーダー＋場）
- `don` … 使えるドン（`scalars` 列 2・**1 個 = +1000**）
- `force_max` … 「攻撃できる枠のうち高い順に、ドンを 1000 単位で足して `ceil(x/1000)` の和を最大化」
  ＝ドンは 1 個で必ず +1 枚なので **`Σ_i ceil(max(0, x_i)/1000) + don`**（`x_i > 0` の枠が 1 つ以上あるとき）。
  `x_i ≤ 0` の枠にドンを足して 0 を超えさせる分は 1 個目が部分的に無駄になる＝その差も出す。
- `force_actual` … **そのターンに実際に打った攻撃**（`sig` の `ATTACK`／`DON_BOX` の
  **`kind==0` の行だけ**＝箱の中の `ATTACK` は同じ攻撃の続きなので数えない）の
  `ceil(max(0, x)/1000)` の和。x はその攻撃の行の盤面から読む（付与後のパワー）。
- `don_attached` … そのターンの `ATTACH_DON` の行数＝**実際に乗せたドン**（記録から直接）。
- `don_plays` … そのターンに `PLAY` した札のコストの和＝**場に出すのに払ったドン**
  （付与と区別するために要る。これを引かないと「余らせた」を大きく数え過ぎる）。
- `don_idle = max(0, don − don_attached − don_plays)` が**「付与が苦手」の直接の量**＝
  何にも使わずに残したドン。`force_gap` のうちドン側の成分。残りは「殴らなかった分」。

出すもの: `force_actual`／`force_max`／`force_gap` の分布・ターン帯／自分のライフ別・
**`attacks_made` vs `attacks_available`**（そもそも殴っていない回数）・`x` の 1000 刻み分布・
`gap>0` の割合と、`gap` と勝敗の関係。

### 近似（必ず添える）

- 相手のブロッカー・カウンターは**見ない**（`force` は「相手が払わされる**下限**」）。
- `x` は相手**リーダー**のパワー基準＝キャラへの攻撃（`board`）は別に数える（主集計から外す）。
- ドン 1 個 = +1000 は【ドン!!×1】等の追加効果を含まない（**下限側**）。
- `don_plays` はカードのコストだけ＝**【起動メイン】のコスト**（能力側のコスト）を含まない
  ＝`don_idle` は**わずかに過大**（起動を使ったターンで余りを多く見せる）。
- `force_max` は「攻撃回数は実際と同じ」を仮定しない＝**攻撃できた枠を全部使う**上限。
  そのため **`force_gap` は「殴らなかった分」と「ドンを乗せなかった分」の合計**で、
  分解して読む: ドン側は `don_idle`（記録から直接）・攻撃側は `attacks_available − attacks_made`。
- **`force` の単位は「1000 パワー分」**で、カードの枚数そのものではない。相手が実際に払う枚数は
  相手のカウンター分布で決まる（`deck_profile.py` の `c̄(x)` が変換係数＝2000 が多いデッキなら
  同じ `force` で払う**枚数は少ない**）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/attack_budget.py \\
    --in ~/w32/n_records/n32_w* --holdout-mod 0 --out ~/attack_w32.json
"""
import argparse
import collections
import json
import math
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
from race_state import _extra as _race_extra, _SLOT_OWN, _T_PWR, _T_CAN, _T_CHAR, _PWR_SCALE  # noqa: E402,E501

ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "deck_kinds")
POL_COLS = ("pol_sig", "pol_cid", "pol_tcid")
SC_L, SC_OPP_L, SC_DON, SC_H = 0, 1, 2, 6
TURN_BANDS = ("T<=4", "T5-8", "T9+")
X_BANDS = ("x<=0", "x1k", "x2k", "x3k", "x4k", "x5k+")
#: パワーの許容（power 単位）。符号化は f16 なので `power/10000` を戻すと 6000 が 6000.0002 に
#: なる。切り上げをそのまま使うと**ちょうど 1000 の倍数が 1 段上がる**（実データの大半がこれ）
#: ＝`force` が系統的に +1 される。10 は実カードのパワー刻み（1000）よりはるかに小さい。
PWR_EPS = 10.0


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def x_band(x):
    if x is None:
        return None
    if x <= PWR_EPS:
        return "x<=0"
    k = int(math.ceil((x - PWR_EPS) / 1000.0))
    return f"x{k}k" if k <= 4 else "x5k+"


def force(x):
    """相手に払わせる枚数（カウンター 1 枚 = 1000 として切り上げ・[`PWR_EPS`] の許容つき）。"""
    if x is None or x <= PWR_EPS:
        return 0
    return int(math.ceil((x - PWR_EPS) / 1000.0))


def _extra(dd, n):
    _sc_race, tk = _race_extra(dd, n)
    return {"sc": np.asarray(dd["scalars"])[:n, :14].astype(np.float32), "tk": tk}


def my_attackers(tk_row):
    """自席ターンの自分の枠 → 攻撃できる枠のパワー（リーダー＋場）。"""
    tk = np.asarray(tk_row)
    out = []
    if float(tk[0, _T_CAN]) > 0.5:
        out.append(float(tk[0, _T_PWR]) * _PWR_SCALE)
    own = tk[_SLOT_OWN]
    ok = (own[:, _T_CAN] > 0.5) & (own[:, _T_CHAR] > 0.5)
    out += [float(v) * _PWR_SCALE for v in own[ok, _T_PWR]]
    return out


def opp_leader_power(tk_row):
    return float(np.asarray(tk_row)[1, _T_PWR]) * _PWR_SCALE


def max_force(powers, opp_pwr, don):
    """同じ枠・同じドンで掛けられる `Σ force` の上限と、その内訳。

    ドン 1 個は +1000＝**必ず +1 枚**（すでに `x > 0` の枠に足す限り）。`x <= 0` の枠に足すと
    0 を超えるまでの分は無駄になるので、**`x > 0` の枠が 1 つでもあればドンは全部そこへ**乗せる。
    """
    xs = [p - opp_pwr for p in powers]
    base = sum(force(x) for x in xs)
    don = int(max(don, 0))
    if any(x > 0 for x in xs):
        return base + don, base, don
    # どの枠も届かないとき: 一番高い枠にドンを乗せて届かせる（余りが 1 枚ずつになる）
    if not xs:
        return 0, 0, 0
    top = max(xs)
    need = int(math.ceil((PWR_EPS - top) / 1000.0)) if top < PWR_EPS else 0
    usable = max(0, don - max(need - 1, 0))
    return base + max(0, usable), base, don


def turns_of(rows, ex, L, ptr, idx, cards, u2c):
    """1 局 → 自席ターンごとの攻め側の勘定（`u2c`＝uuid → カード ID・呼び出し側で 1 回作る）。"""
    zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
    out = {}
    for i in idx:
        w = int(rows["who"][i]); t = int(rows["turn"][i]); k = int(rows["kind"][i])
        if t < 1 or not PL.is_own_turn(w, t):
            continue
        try:
            sj = json.loads(rows["sig"][i])
            at = sj[0]
        except Exception:                                  # noqa: BLE001
            sj, at = None, ""
        key = (w, t)
        if key not in out and k == 0:
            tk = ex["tk"][i]
            sc = ex["sc"][i]
            powers = my_attackers(tk)
            opp = opp_leader_power(tk)
            don = int(round(float(sc[SC_DON])))
            fmax, base, don_used = max_force(powers, opp, don)
            out[key] = {
                "turn": t, "who": w, "z": 1.0 if zs.get(w, 0.0) > 0 else 0.0,
                "L": float(sc[SC_L]), "opp_L": float(sc[SC_OPP_L]), "H": float(sc[SC_H]),
                "don": don, "attacks_available": len(powers),
                "force_max": fmax, "force_base": base, "don_addable": don_used,
                "force_actual": 0, "attacks_made": 0, "x_list": [],
                "board_attacks": 0, "don_attached": 0, "don_plays": 0, "plays": 0,
            }
        cur = out.get(key)
        if cur is None:
            continue
        if at == "PLAY" and sj is not None:
            # 場に出した札のコスト＝**そのターンにドンを払った分**（付与と区別する）
            inf = cards.info(u2c.get(sj[1])) if len(sj) > 1 else None
            if inf is not None:
                cur["don_plays"] += int(inf.get("cost") or 0)
                cur["plays"] += 1
            continue
        if at == "ATTACH_DON":
            # **付与の直接の観測**（ドンを 1 個乗せた行）＝「使い切ったか」はこれで分かる
            cur["don_attached"] += 1
            continue
        if at in ("ATTACK", "DON_BOX") and sj is not None and k == 0:
            # **攻撃の単位は main 行（`kind==0`）だけ**——箱の中の `ATTACK` 行（`kind==2`）は
            # 同じ攻撃の続きなので数えない（`plan_labels.turn_counts` と同じ規則）。
            # 数えると DON 箱を通った攻撃が全部 2 回になる（2026-09-13・テストで検出）。
            tgt = sj[2][0] if len(sj) > 2 and sj[2] else None
            if tgt is None:
                continue                                   # 配分だけの箱（的が無い）
            inf = cards.info(u2c.get(tgt)) if tgt else None
            if inf is None:
                continue
            if not inf["leader"]:
                cur["board_attacks"] += 1
                continue
            x = _attack_x(ex["tk"][i], sj)
            cur["attacks_made"] += 1
            cur["force_actual"] += force(x)
            cur["x_list"].append(x)
    return list(out.values())


def _attack_x(tk_row, sj):
    """その攻撃の超過パワー。**攻撃した枠**が分かればその枠・分からなければ最大。"""
    tk = np.asarray(tk_row)
    opp = opp_leader_power(tk)
    powers = my_attackers(tk)
    if not powers:
        return None
    return max(powers) - opp


def collect(dirs, holdout_mod=0, limit_games=0):
    cards = PL.Cards()
    rec = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        rec.extend(turns_of(rows, ex, L, ptr, idx, cards, PL.uuid_map(pol, L, ptr, idx)))
    return rec, games


def _m(xs, nd=3):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), nd) if xs else None


def block(recs):
    if not recs:
        return None
    gaps = [r["force_max"] - r["force_actual"] for r in recs]
    return {
        "n": len(recs),
        "force_actual": _m([r["force_actual"] for r in recs]),
        "force_max": _m([r["force_max"] for r in recs]),
        "force_gap": _m(gaps),
        "gap_pos": round(float(np.mean([g > 0 for g in gaps])), 4),
        "don": _m([r["don"] for r in recs]),
        # **付与の使い切り**（`gap` のうちドン側の成分・記録から直接数えた）
        "don_attached": _m([r["don_attached"] for r in recs]),
        "don_plays": _m([r["don_plays"] for r in recs]),
        "plays": _m([r["plays"] for r in recs]),
        # **本当に余らせたドン**＝手持ち − 付与 − 場に出すのに払った分（負は 0 に丸める）
        "don_idle": _m([max(0, r["don"] - r["don_attached"] - r["don_plays"]) for r in recs]),
        "don_idle_pos": round(float(np.mean(
            [r["don"] - r["don_attached"] - r["don_plays"] > 0 for r in recs])), 4),
        "attacks_made": _m([r["attacks_made"] for r in recs]),
        "attacks_available": _m([r["attacks_available"] for r in recs]),
        "no_attack": round(float(np.mean([r["attacks_made"] == 0 for r in recs])), 4),
        "board_attacks": _m([r["board_attacks"] for r in recs]),
        "winrate": _m([r["z"] for r in recs], 4),
        # **付与を使い切ったターンと余らせたターンの勝率**（gap が勝敗に効いているか）
        "winrate_gap0": _m([r["z"] for r, g in zip(recs, gaps) if g <= 0], 4),
        "winrate_gap_pos": _m([r["z"] for r, g in zip(recs, gaps) if g > 0], 4),
    }


def x_dist(recs):
    """打った攻撃の超過パワーの分布（**7000 理論が起きているか**）。"""
    cnt = collections.Counter()
    for r in recs:
        for x in r["x_list"]:
            b = x_band(x)
            if b:
                cnt[b] += 1
    tot = sum(cnt.values()) or 1
    return {b: {"n": cnt.get(b, 0), "share": round(cnt.get(b, 0) / tot, 4)} for b in X_BANDS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=0)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    recs, games = collect(args.src, args.holdout_mod, args.limit_games)
    out = {"games": games, "turns": len(recs), "src": list(args.src),
           "all": block(recs), "x_dist": x_dist(recs),
           "by_turn": {b: block([r for r in recs if turn_band(r["turn"]) == b])
                       for b in TURN_BANDS},
           "by_opp_life": {str(k): block([r for r in recs
                                          if int(round(r["opp_L"])) == k])
                           for k in (1, 2, 3, 4, 5)},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
