"""読まれやすさ——**公開情報だけから CPU の手をどれだけ当てられるか**
（`docs/game_theory.md` §3-3・§25 の宿題 2・読み取り専用・ネット不要）。

不完全情報の零和ゲームでは最適戦略は一般に**混合**であり、**決定論の打ち手は適応する相手に
読まれる**。本エンジンは serve で決定論（同じ盤面・同じ seed なら同じ手）。しかも

> **自己対戦アリーナは両者が同じ固定方策なので、この弱点を原理的に測れない**
> （`game_theory.md` §21）。人間と打つときだけ損が出る＝実害の側にある。

そこで「読まれやすさ」を**相手が見えるものだけ**で測る:

```
H(A)            … 行動の周辺エントロピー（何も知らないときの不確かさ）
H(A | 公開状態)  … 公開状態を知ったときの不確かさ
I = H(A) − H(A|公開)   … 相手が公開情報から得る情報量（**大きいほど読まれる**）
acc             … 公開状態から多数決で当てる予測器の的中率（**別の対局で検証**）
```

`acc` が base rate（いちばん多い行動の割合）を大きく超えるなら、**相手は読める**。
`H(A|公開) ≈ 0` なら完全に読める＝混合していない。

## 公開状態（相手が見えるもの）だけを使う

自分のライフ・相手のライフ・**自分の手札の枚数**（中身ではない）・相手の手札枚数・
両者の場のキャラ数・ターン帯・**飛んでくる攻撃の超過パワー帯**（守りのとき）。
手札の中身・伏せライフ・山の順は入れない（`game_theory.md` §4 の規約と同じ向き）。

## 2 つの行動集合

- **守り**（攻撃 1 回ごと・`kind==1` の窓の入口の行）: `guard`（カウンター or ブロッカー）／`take`
- **攻め**（自席ターンの main 行）: `face`／`board`／`develop`／`other`（`plan_labels.move_class`）

## 過学習を避ける（**ここが要**）

多数決の予測器は**対局で分けて**学習・検証する（seed の偶奇で split）。
帯が細かいほど訓練側の的中率は上がるので、**検証側の数字だけを読む**。
帯に現れなかった組み合わせは「全体の多数決」で埋める（相手も同じことしかできない）。

帯の粒度は `--level min|mid|full`。**細かい帯は `I` を上振れさせ、予測器は帯を外す**
（8 局のスモークで帯 144・被覆 2.6%・`lift` が測れなかった）。`bands_per_row` が
0.05 を超えていたら `I` は信用しない（`I_bits_corrected` は Miller–Madow の補正つき）。
`covered`（検証側のうち訓練で見た帯の割合）が 0.8 未満なら、その `acc` は帯の外挿混じり。

## 読み方（事前登録）

- 守りで `acc_test − base_rate` が **+0.15 以上**なら**読まれている**＝混合の欠如が実害になりうる。
- `+0.05` 未満なら公開情報からは読めない（＝隠れ情報が効いている・混合の必要は薄い）。
- `I`（相互情報量）は bit で出す。`I > 0.3 bit` は「公開状態が行動をほぼ決めている」水準。

## 近似

- 「読まれる」は**予測できる**ことであって、**搾取できる**こととは違う（搾取には
  「読んだ上で得をする手」が要る）。本器は前段だけを測る。
- 決定論そのものは測れない（同じ盤面が 2 度出ないので）。代理として公開状態からの予測を使う。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/policy_entropy.py \\
    --in ~/w32/n_records/n32_w* --out ~/policy_entropy.json
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

from race_state import _extra as _race_extra, _incoming  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402

ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len")
POL_COLS = ("pol_sig", "pol_cid", "pol_tcid")
SC_L, SC_OPP_L, SC_H, SC_OPP_H, SC_FIELD, SC_OPP_FIELD, SC_TURN = 0, 1, 6, 7, 8, 9, 10
GUARD_SIGS = ("SELECT_COUNTER", "SELECT_BLOCKER")
PWR_EPS = 10.0


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def x_band(x):
    """飛んでくる超過パワーの帯（公開情報・`PWR_EPS` は f16 の丸めの許容）。"""
    if x is None:
        return "none"
    if x <= PWR_EPS:
        return "x<=0"
    k = int(math.ceil((x - PWR_EPS) / 1000.0))
    return f"x{k}k" if k <= 4 else "x5k+"


def _extra(dd, n):
    sc = np.asarray(dd["scalars"])[:n, :14].astype(np.float32)
    _sc_race, tk = _race_extra(dd, n)
    return {"sc": sc, "tk": tk}


#: 帯の粒度。**細かいほど in-sample の相互情報量は上がるが、予測器は帯を外す**
#: （8 局のスモークで帯 144・被覆 2.6% を実測＝`lift` が測れない）。既定は `mid`。
LEVELS = ("min", "mid", "full")


def hand_bucket(h):
    h = int(round(float(h)))
    return "h0-2" if h <= 2 else ("h3-5" if h <= 5 else "h6+")


def public_band(sc_row, x=None, level="mid"):
    """**相手が見える情報だけ**の帯（`level` で粒度を選ぶ）。

    - `min` … 守り: (自ライフ, 超過パワー帯)／攻め: (自ライフ, 相手ライフ)
    - `mid` … `min` ＋ 相手ライフ・自手札の 3 段・ターン帯（**既定**）
    - `full` … さらに相手手札・両者の場のキャラ数（帯が細かく、予測器は外しやすい）
    """
    L = int(round(float(sc_row[SC_L])))
    oL = int(round(float(sc_row[SC_OPP_L])))
    parts = [L]
    if x is not None:
        parts.append(x_band(x))
    else:
        parts.append(oL)
    if level in ("mid", "full"):
        if x is not None:
            parts.append(oL)
        parts.append(hand_bucket(sc_row[SC_H]))
        parts.append(turn_band(int(round(float(sc_row[SC_TURN])))))
    if level == "full":
        parts += [int(round(float(sc_row[SC_OPP_H]))),
                  int(round(float(sc_row[SC_FIELD]))),
                  int(round(float(sc_row[SC_OPP_FIELD])))]
    return "|".join(str(p) for p in parts)


def entropy(counter):
    """度数 → エントロピー（bit）。"""
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    out = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / n
        out -= p * math.log2(p)
    return out


def conditional_entropy(rows, band_key="band", act_key="action"):
    """`H(A | 帯)`（帯の重みつき平均）と `H(A)`・相互情報量 `I`（bit）。"""
    by = collections.defaultdict(collections.Counter)
    marg = collections.Counter()
    for r in rows:
        by[r[band_key]][r[act_key]] += 1
        marg[r[act_key]] += 1
    n = sum(marg.values())
    h_cond = sum(sum(c.values()) / n * entropy(c) for c in by.values()) if n else 0.0
    h = entropy(marg)
    # **Miller–Madow の上振れ補正**: 帯が多いと `H_cond` は下に偏る（＝`I` が上振れする）。
    # 補正量は `(有効な自由度) / (2 n ln2)`＝帯 × (行動数−1)。帯が n に近いと `I` は意味を失うので、
    # `bands_per_row` も併記して読み手が捨てられるようにする。
    k_act = max(len(marg), 1)
    dof = max(len(by) * (k_act - 1), 0)
    corr = dof / (2.0 * n * math.log(2)) if n else 0.0
    return {"H": round(h, 4), "H_cond": round(h_cond, 4),
            "I_bits": round(h - h_cond, 4),
            "I_bits_corrected": round(max(0.0, h - h_cond - corr), 4),
            "bands": len(by), "n": n,
            "bands_per_row": round(len(by) / n, 4) if n else None}


def majority_predictor(train, test, band_key="band", act_key="action"):
    """帯ごとの多数決を訓練側で作り、**検証側**で的中率を測る（帯が無ければ全体の多数決）。"""
    by = collections.defaultdict(collections.Counter)
    marg = collections.Counter()
    for r in train:
        by[r[band_key]][r[act_key]] += 1
        marg[r[act_key]] += 1
    if not marg:
        return None
    fallback = marg.most_common(1)[0][0]
    table = {b: c.most_common(1)[0][0] for b, c in by.items()}
    hit = seen = 0
    for r in test:
        pred = table.get(r[band_key], fallback)
        hit += int(pred == r[act_key])
        seen += 1
    base = collections.Counter(r[act_key] for r in test)
    base_rate = (base.most_common(1)[0][1] / seen) if seen else None
    return {"n_test": seen, "acc": round(hit / seen, 4) if seen else None,
            "base_rate": round(base_rate, 4) if base_rate is not None else None,
            "lift": round(hit / seen - base_rate, 4) if seen and base_rate is not None else None,
            "bands_seen": len(table),
            "covered": round(sum(1 for r in test if r[band_key] in table) / seen, 4)
            if seen else None}


def collect(dirs, holdout_mod=0, limit_games=0, level="mid"):
    """記録 → 守り（攻撃 1 回ごと）と攻め（main 行）の (帯, 行動)。"""
    cards = PL.Cards()
    dfn, att = [], []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        u2c = PL.uuid_map(pol, L, ptr, idx)
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i]); k = int(rows["kind"][i])
            if t < 1:
                continue
            try:
                sj = json.loads(rows["sig"][i])
                at = sj[0]
            except Exception:                              # noqa: BLE001
                continue
            own = PL.is_own_turn(w, t)
            if own and k == 0:
                cl = PL.move_class(sj, u2c, cards)
                if cl is None or cl == "unknown":
                    continue
                action = cl if cl in ("face", "board", "develop") else "other"
                att.append({"seed": seed, "turn": t, "action": action,
                            "band": public_band(ex["sc"][i], None, level)})
            elif (not own) and k == 1:
                # **攻撃 1 回ごとの窓の入口**＝ここでの選択が守るか受けるか
                action = "guard" if at in GUARD_SIGS else "take"
                x = _incoming(ex["tk"][i])
                dfn.append({"seed": seed, "turn": t, "action": action,
                            "band": public_band(ex["sc"][i], x, level)})
    return att, dfn, games


def analyse(rows):
    """エントロピーと、対局で分けた予測器（seed の偶奇）。"""
    if not rows:
        return None
    out = dict(conditional_entropy(rows))
    train = [r for r in rows if int(r["seed"]) % 2 == 0]
    test = [r for r in rows if int(r["seed"]) % 2 == 1]
    out["predictor"] = majority_predictor(train, test)
    out["train_n"] = len(train)
    out["share"] = {k: round(v / len(rows), 4)
                    for k, v in collections.Counter(r["action"] for r in rows).items()}
    return out


def verdict(a, readable_at=0.15, opaque_at=0.05):
    """事前登録: 検証側の `lift`（的中率 − base rate）で読む。"""
    if not a or not a.get("predictor") or a["predictor"].get("lift") is None:
        return None
    lift = a["predictor"]["lift"]
    return "readable" if lift >= readable_at else ("opaque" if lift < opaque_at else "partly")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=0)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--level", default="mid", choices=LEVELS,
                    help="帯の粒度（細かいほど予測器は帯を外す・既定 mid）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    att, dfn, games = collect(args.src, args.holdout_mod, args.limit_games, args.level)
    out = {"games": games, "src": list(args.src), "level": args.level,
           "defense": analyse(dfn), "attack": analyse(att),
           "seconds": round(time.time() - t0, 1)}
    out["verdict"] = {"defense": verdict(out["defense"]), "attack": verdict(out["attack"])}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
