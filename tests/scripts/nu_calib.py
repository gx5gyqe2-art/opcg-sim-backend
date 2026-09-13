"""**`ν` の 3 つの定数を実測に替える**（場のキャラ 1 体の価格の較正・読み取り専用）。

ユーザ指示 2026-09-13（選択肢 2）。`docs/game_theory.md` §14.1 の `ν` は

```
ν = R · attack_value + block_p · Θ · μ − ko_p · (…)
```

で、**`R`（残りターン）・`block_p`（ブロック確率）・`ko_p`（KO される確率）を私が定数で置いた**
（3.0／0.3／0.25）。`2026-09-13_theory_vs_search.md` §3 で理論側の欠陥が疑われた 2 か所は
どちらも `ν` に関わる:

- **理論が「ターン終了が最善」と言い探索はキャラを出す**（接戦帯の食い違いの 12.0%）
  ＝`play_value = ν − μ − cost·δ` が不当に負になっている疑い＝**`ν` が低すぎる**
- **通らない攻撃（`x<0`）を一律 0 と置いている**＝盤面を動かす価値を見ていない疑い

**3 つとも既存の記録から数えられる**（新しい対局は要らない）。

## 測るもの

| 量 | 数え方 | 現在の定数 |
|---|---|---|
| `blocker_share` | 自場のキャラのうち `is_blocker_active`（トークン列 6）の割合 | —（**そもそも区別していない**） |
| `block_rate_per_window` | `SELECT_BLOCKER` ÷ 守りの窓（`PASS`／`SELECT_COUNTER`／`SELECT_BLOCKER`） | 0.30 |
| `block_per_blocker_turn` | `SELECT_BLOCKER` ÷（自場のアクティブなブロッカー × 相手ターン） | 0.30 |
| `ko_p` | 自席ターンごとに `失った ÷ (居た + 出した)`（次の自席ターンまで） | 0.25 |
| `r_turns_real` | この行から先の自席ターン数（実現値） | `clamp(相手ライフ, 1, 5)` |

**守り／攻めの行は `kind` で分かれる**（`record_gen`: 0=main／1=window／2=commit）。
ブロックとカウンターは **window と commit の行**に出る（`SELECT_BLOCKER`／`SELECT_COUNTER`）。

## 読み方（事前登録）

- **`block_p` は構造的に誤っている可能性が高い**——**ブロッカーでないキャラはブロックできない**
  （`is_blocker_active` はキーワード持ち・非レストだけ立つ）。実測の `blocker_share` が小さければ、
  **定数 0.3 を全キャラに掛けているのが誤り**で、`ν` は**ブロッカーか否かで分けるべき**。
- **`ko_p` が 0.25 より大きければ `ν` は下がる**（＝「理論が低すぎる」の説明にはならない）。
  **小さければ `ν` は上がる**＝§3 の 12.0% の説明になる。
- **`r_turns` の代理（相手ライフ）が実測とずれていれば、そこが一番効く**
  （`R` は `attack_value` に直接掛かるので `ν` の主項）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/nu_calib.py --in ~/w32/*/n_records --out ~/nu_calib.json
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
from theory_order import MU, THETA, attack_value, nu_of  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "sig", "pol_len", "pol_v0")

#: 22 枠の並び（`rust/opcg_engine/src/encode/tokens.rs`）
SLOT_OWN_LEADER, SLOT_OPP_LEADER = 0, 1
SLOT_OWN_FIELD, SLOT_OPP_FIELD = slice(2, 7), slice(7, 12)
#: トークンの列（同上の表）
S_POWER, S_REST, S_BLOCKER_ACTIVE, S_IS_CHAR = 0, 3, 6, 18
#: scalars の列
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_FIELD, SC_OPP_FIELD = 8, 9
SC_OPP_LEADER_POWER = 13

#: 守りの窓で選べる手（これ以外＝効果の解決などは窓として数えない）
GUARD_ACTS = ("PASS", "SELECT_COUNTER", "SELECT_BLOCKER")
#: f32 の丸め（0.7×1e4 が 6999.999… になる・本計画で 4 回踏んだ罠）
PWR_EPS = 10.0


def _round10(x):
    return float(np.round(float(x) / PWR_EPS) * PWR_EPS)


def own_field(tok_row):
    """自場の (キャラ数, アクティブなブロッカー数, パワーの一覧)。"""
    n = blk = 0
    powers = []
    for s in range(SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.stop):
        if float(tok_row[s, S_IS_CHAR]) <= 0.0 and float(tok_row[s, S_POWER]) <= 0.0:
            continue
        n += 1
        powers.append(_round10(float(tok_row[s, S_POWER]) * 1e4))
        if float(tok_row[s, S_BLOCKER_ACTIVE]) > 0.0:
            blk += 1
    return n, blk, powers


def collect(dirs, limit_games=0):
    """記録 → `ν` の 3 定数を数えるための素の計数。"""
    recs = []                      # 自席ターンごと（`ko_p`・`R` 用）
    guard = Counter()              # 守りの窓の行動の内訳
    field = {"chars": 0, "blockers": 0, "rows": 0}
    opp_turn_blockers = 0          # 相手ターンに自分が持っていたアクティブなブロッカーの延べ数
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=(),
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = int(rows["seed"][idx[0]])
        # 自席ターンの最初の main 行だけで盤面を採る（`plan_labels` と同じ流儀）
        own_first, plays, seen_opp_turn = {}, Counter(), set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1:
                continue
            tok = ex["tok"][i]
            n_ch, n_blk, _pw = own_field(tok)
            if PL.is_own_turn(w, t):
                if int(rows["kind"][i]) == 0 and (w, t) not in own_first:
                    own_first[(w, t)] = {"chars": n_ch, "blockers": n_blk,
                                         "opp_life": float(ex["sc"][i][SC_OPP_LIFE]),
                                         "opp_leader_power":
                                             float(ex["sc"][i][SC_OPP_LEADER_POWER]) * 1e4}
                    field["chars"] += n_ch; field["blockers"] += n_blk; field["rows"] += 1
                try:
                    at = json.loads(rows["sig"][i])[0]
                except Exception:
                    at = None
                if at == "PLAY":
                    plays[(w, t)] += 1
            else:
                # 相手ターン: 守りの窓の内訳と、そのとき持っていたブロッカーの延べ数
                if (w, t) not in seen_opp_turn:
                    seen_opp_turn.add((w, t))
                    opp_turn_blockers += n_blk
                try:
                    at = json.loads(rows["sig"][i])[0]
                except Exception:
                    at = None
                if at in GUARD_ACTS:
                    guard[at] += 1
        # 自席ターン → 次の自席ターンで何体失ったか
        for w in (0, 1):
            ts = sorted(t for (ww, t) in own_first if ww == w)
            for a, b in zip(ts, ts[1:]):
                st = own_first[(w, a)]
                had = st["chars"] + plays[(w, a)]
                lost = max(0, had - own_first[(w, b)]["chars"])
                recs.append({"seed": seed, "who": w, "turn": a,
                             "had": had, "lost": lost, "chars": st["chars"],
                             "blockers": st["blockers"], "opp_life": st["opp_life"],
                             "r_real": float(len([x for x in ts if x >= a])),
                             "opp_leader_power": st["opp_leader_power"]})
    return recs, guard, field, opp_turn_blockers, games


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def _mean_ci(vals, seeds):
    """平均と**対局でクラスタした** 95% CI。"""
    if not vals:
        return {"n": 0, "mean": None, "ci95": None}
    per = {}
    for v, s in zip(vals, seeds):
        per.setdefault(s, []).append(float(v))
    g = np.array([np.mean(v) for v in per.values()], np.float64)
    m = float(g.mean())
    if len(g) < 2:
        return {"n": len(vals), "games": 1, "mean": round(m, 5), "ci95": None}
    se = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {"n": len(vals), "games": len(g), "mean": round(m, 5), "se": round(se, 5),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


def calibrate(recs, guard, field, opp_turn_blockers):
    """3 定数の実測値（と現在の定数との対照）。"""
    sd = [r["seed"] for r in recs]
    windows = sum(guard[a] for a in GUARD_ACTS)
    blocks = guard["SELECT_BLOCKER"]
    out = {
        "blocker_share": (round(field["blockers"] / field["chars"], 4)
                          if field["chars"] else None),
        "chars_per_own_turn": (round(field["chars"] / field["rows"], 3)
                               if field["rows"] else None),
        "guard_windows": windows,
        "guard_mix": {a: round(guard[a] / windows, 4) for a in GUARD_ACTS} if windows else {},
        # **窓あたり**のブロック率（守りの窓のうちブロックで応じた割合）
        "block_rate_per_window": round(blocks / windows, 4) if windows else None,
        # **ブロッカー 1 体・相手ターン 1 回あたり**のブロック率（`ν` の式が欲しいのはこちら）
        "block_per_blocker_turn": (round(blocks / opp_turn_blockers, 4)
                                   if opp_turn_blockers else None),
        "opp_turn_blockers": opp_turn_blockers,
        # KO 率＝自席ターン間で失った体数 ÷ 居た体数（出したぶんも分母に入れる）
        "ko_p": _mean_ci([r["lost"] / r["had"] for r in recs if r["had"] > 0],
                         [r["seed"] for r in recs if r["had"] > 0]),
        "r_real": _mean_ci([r["r_real"] for r in recs], sd),
        # 現在の代理（相手ライフを 1..5 に丸めたもの）と実現値のずれ
        "r_proxy": _mean_ci([max(1.0, min(5.0, r["opp_life"])) for r in recs], sd),
        "r_proxy_minus_real": _mean_ci(
            [max(1.0, min(5.0, r["opp_life"])) - r["r_real"] for r in recs], sd),
        "current_constants": {"r_turns": "clamp(opp_life,1,5)", "block_p": 0.3, "ko_p": 0.25},
    }
    return out


def nu_effect(recs, cal, theta=THETA, mu=MU):
    """較正した定数で `ν` を引き直し、**現行との差**を出す（`play_value` の符号が要点）。"""
    ko = cal["ko_p"]["mean"]
    b_blk = cal["block_per_blocker_turn"]
    if ko is None or b_blk is None:
        return None
    old, new_blk, new_plain = [], [], []
    for r in recs:
        if not r["chars"]:
            continue
        p = 5000.0                      # 代表値（キャラ 1 体のパワーは記録から採らず固定にする）
        opl = r["opp_leader_power"] or 5000.0
        r_old = max(1.0, min(5.0, r["opp_life"]))
        r_new = r["r_real"]
        old.append(nu_of(p, opl, r_old, theta, mu, block_p=0.3, ko_p=0.25))
        # **ブロッカーとそれ以外を分ける**（ブロッカーでないキャラはブロックできない）
        new_blk.append(nu_of(p, opl, r_new, theta, mu, block_p=b_blk, ko_p=ko))
        new_plain.append(nu_of(p, opl, r_new, theta, mu, block_p=0.0, ko_p=ko))
    if not old:
        return None
    return {"nu_now": round(float(np.mean(old)), 5),
            "nu_calibrated_blocker": round(float(np.mean(new_blk)), 5),
            "nu_calibrated_plain": round(float(np.mean(new_plain)), 5),
            "ratio_blocker": round(float(np.mean(new_blk)) / float(np.mean(old)), 4),
            "ratio_plain": round(float(np.mean(new_plain)) / float(np.mean(old)), 4),
            "n": len(old)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, guard, field, otb, games = collect(a.src, a.limit_games)
    cal = calibrate(recs, guard, field, otb)
    res = {"games": games, "own_turns": len(recs), "calibration": cal,
           "nu_effect": nu_effect(recs, cal, a.theta, a.mu),
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
