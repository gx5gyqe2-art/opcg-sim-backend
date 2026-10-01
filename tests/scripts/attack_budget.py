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
- `don_attached` … `ATTACH_DON` の**行**の数（箱の中の `kind==2` を含む）。
  **付与の一部しか捉えない**（実測 1.32 対 真値 2.94）＝会計には使わない。
  **`ATTACH_DON` は選ばれた手としては一度も現れない**（2026-09-14・300 局 約 1.9 万手で 0 件）
  ——付与は全部 `DON_BOX` の中で起きる。
- `don_box_attached` … `DON_BOX` が乗せたドンの和（記録の **`pol_k`**）＝**実際の付与**。
- `don_plays` … そのターンに `PLAY` した札のコストの和＝**場に出すのに払ったドン**。
- **`don_idle` は状態から直接読む**——自席ターンの**最後の main 行**（`TURN_END` を選んだ行）の
  アクティブなドン＝何にも使わずに残した分。`force_gap` のうちドン側の成分。

> **2026-09-14 の訂正（同じ罠を二度踏まないために残す）**: 以前の `don_idle` は
> **`max(0, don − don_attached − don_plays)` という再構成**だった。`ATTACH_DON` が
> 一度も選ばれないので**付与が丸ごと抜け、余りを 3 倍に見せていた**
> （この器の実測は **2.07** 対 **真値 0.51**・「1 個以上余るターン」74.2% 対 **真値 23.1%**。
> `ATTACH_DON` の**行**を 1.32 引いていたぶんだけ小さく、行も数えなければ 3.38 になる）。
> 波 32 で報告した「ドンを 2.32 個余らせる・77.1% のターン」は**この誤りの産物**。
> **実際は 76.9% のターンでドンを使い切っている**＝ドンは余っていない。
> 教訓は `don_k` の誤りと同じ——**`DON_BOX` は巨視手で、中の原始手は行として現れない**。
> **状態が答えを持っているものを行動の再構成で出さない**（`don_idle_recon_wrong` に並記）。

出すもの: `force_actual`／`force_max`／`force_gap` の分布・ターン帯／自分のライフ別・
**`attacks_made` vs `attacks_available`**（そもそも殴っていない回数）・`x` の 1000 刻み分布・
`gap>0` の割合と、`gap` と勝敗の関係。

### 近似（必ず添える）

- 相手のブロッカー・カウンターは**見ない**（`force` は「相手が払わされる**下限**」）。
- `x` は相手**リーダー**のパワー基準＝キャラへの攻撃（`board`）は別に数える（主集計から外す）。
- ドン 1 個 = +1000 は【ドン!!×1】等の追加効果を含まない（**下限側**）。
- `don_plays` はカードのコストだけ＝**【起動メイン】のコスト**（能力側のコスト）を含まない。
  `don_idle` は直接読みなので影響を受けないが、**会計の残差 `don_unaccounted` に出る**
  （`don − 付与 − コスト − 余り`＝主に起動メインのコスト）。
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

ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen",
            "deck_kinds")
POL_COLS = ("pol_sig", "pol_cid", "pol_tcid", "pol_k")
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


def box_k(rows, pol, ptr, i):
    """その行で選んだ `DON_BOX` が**実際に乗せたドンの枚数**（記録の `pol_k`）。

    `pol_k` は候補ごとの付与枚数で、`DON_BOX` 以外は `-1`。**付与は全部箱の中で起きる**
    （`ATTACH_DON` は一度も選ばれない）ので、**ドンの会計はこの列でしか閉じない**
    ——2026-09-14 に、これを数えずに「余らせたドン 2.32 個」と報告していたのを訂正した。
    """
    if pol is None or "pol_k" not in pol:
        return 0
    b = int(ptr[i]); n = int(rows["pol_len"][i]); ch = int(rows["pol_chosen"][i])
    if n <= 0 or ch < 0 or ch >= n:
        return 0
    k = int(pol["pol_k"][b + ch])
    return k if k > 0 else 0


def turns_of(rows, ex, L, ptr, idx, cards, u2c, pol=None):
    """1 局 → 自席ターンごとの攻め側の勘定（`u2c`＝uuid → カード ID・呼び出し側で 1 回作る）。

    `pol` を渡すと **`DON_BOX` が乗せたドン（`pol_k`）**も数える——付与は**全部箱の中で
    起きる**ので、これが無いと「余らせた」が 3 倍に出る（2026-09-14 に判明）。
    """
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
                "don_box_attached": 0, "don_left": None,
            }
        cur = out.get(key)
        if cur is None:
            continue
        if k == 0:
            # **余らせたドンは状態から直接読む**（最後の main 行＝`TURN_END` を選んだ行の
            # アクティブなドン）。**再構成（`don − 付与 − コスト`）は使わない**——
            # `ATTACH_DON` が一度も選ばれないので付与が丸ごと抜け、余りを 3 倍に見せていた。
            cur["don_left"] = int(round(float(ex["sc"][i][SC_DON])))
        if at == "PLAY" and sj is not None:
            # 場に出した札のコスト＝**そのターンにドンを払った分**（付与と区別する）
            inf = cards.info(u2c.get(sj[1])) if len(sj) > 1 else None
            if inf is not None:
                cur["don_plays"] += int(inf.get("cost") or 0)
                cur["plays"] += 1
            continue
        if at == "ATTACH_DON":
            # **単独の付与**。実測ではこの手は**一度も選ばれない**（2026-09-14・300 局
            # 約 1.9 万手で 0 件）——付与は全部 `DON_BOX` の中で起きるので、
            # 下の `pol_k` の側が実質の観測になる。互換のために残す。
            cur["don_attached"] += 1
            continue
        if at in ("ATTACK", "DON_BOX") and sj is not None and k == 0:
            # **攻撃の単位は main 行（`kind==0`）だけ**——箱の中の `ATTACK` 行（`kind==2`）は
            # 同じ攻撃の続きなので数えない（`plan_labels.turn_counts` と同じ規則）。
            # 数えると DON 箱を通った攻撃が全部 2 回になる（2026-09-13・テストで検出）。
            # **乗せたドンは的の有無に依らず数える**（的の無い箱＝純粋な付与も在る）ので
            # 下の `continue` より前に置く（2026-09-14）。
            cur["don_box_attached"] += box_k(rows, pol, ptr, i)
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
        rec.extend(turns_of(rows, ex, L, ptr, idx, cards, PL.uuid_map(pol, L, ptr, idx), pol))
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
        # **付与の使い切り**（`gap` のうちドン側の成分）。`don_attached` は `ATTACH_DON` の
        # **行**の数（箱の中の `kind==2` を含む）で、**付与の一部しか捉えない**
        # （実測 1.32 対 真値 2.94）＝会計には使わず `don_box_attached` を使う
        "don_attached": _m([r["don_attached"] for r in recs]),
        "don_box_attached": _m([r["don_box_attached"] for r in recs]),
        "don_plays": _m([r["don_plays"] for r in recs]),
        "plays": _m([r["plays"] for r in recs]),
        # **余らせたドンは状態から直接読む**（最後の main 行のアクティブなドン）。
        # 2026-09-14 の訂正: 再構成（`don − 付与 − コスト`）は `ATTACH_DON` が一度も
        # 選ばれないため付与が丸ごと抜け、**余りを 3 倍に見せていた**（2.07 対 真値 0.51）。
        "don_idle": _m([r["don_left"] for r in recs if r["don_left"] is not None]),
        "don_idle_pos": round(float(np.mean(
            [r["don_left"] >= 1 for r in recs if r["don_left"] is not None])), 4),
        # 再構成（**誤り**）を並べて出す＝同じ罠を二度踏まないための突き合わせ
        "don_idle_recon_wrong": _m([max(0, r["don"] - r["don_attached"] - r["don_plays"])
                                    for r in recs]),
        # 会計が閉じるか＝`don ≈ 箱の付与 + コスト + 余り`（差は主に【起動メイン】のコスト）。
        # **`don_attached` は足さない**——箱の中の `ATTACH_DON` 行を数えたもので
        # `don_box_attached`（`pol_k`）と**同じ付与の重複観測**（2026-09-14）。
        "don_unaccounted": _m([
            r["don"] - r["don_box_attached"] - r["don_plays"] - r["don_left"]
            for r in recs if r["don_left"] is not None]),
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
