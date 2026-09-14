"""**体はどう退場するのか**（T27・`ν` の勘定を組み直すための入力・読み取り専用）。

`docs/game_theory.md` **§14.1.2**（ユーザ決定 2026-09-14）・`docs/cpu_theory_gap.md` の **T27**。

## なぜ要るか——**退場は損ではなく仕事の完了**

> 「倒されるから出さないという考えではなく、**ライフを守れているから出す**という考えのはず」

旧い式は `ν = (攻撃 + ブロック) × (1 − ko_p)`＝**体が消えるぶん割り引く**。
しかしこのゲームでは、**体が殴られて消えた＝その攻撃がリーダーに通らなかった**＝
**ライフかカウンター札を守っている**。**引くべきではなく足すべきものを引いていた。**

したがって `ko_p` は**割引率ではなく「退場の仕方の分布」**であるべきで、
**それぞれに符号付きの値を掛ける**。本器はその**分布**を測る。

| 退場の仕方 | 見分け方 | 勘定 |
|---|---|---|
| **攻撃で退場** | 相手の選んだ手が**攻撃**で対象枠が自分の場 | **＋身代わり**（リーダーが殴られなかった） |
| **除去で退場** | 相手の選んだ手が**攻撃以外**で対象枠が自分の場 | **＋相手の手札 −1**（相手が払った） |
| **自分の都合** | 相手が自分の場に何もしていないのに消えた | **−**（コスト・6 体目） |
| **生存** | 次の自席ターンにも居る | 攻撃項が続く |

## 測り方

自席ターンの入口を 2 つ並べ（`ko_by_power` と同じ枠）、**その間の相手ターンに
相手が選んだ手**を読む。対象枠は `pol_ti`（選ばれた候補の対象スロット）で、
**7〜11 が自分の場**・**1 が自分のリーダー**（相手の席から見た番号）。

> **枠の番号で追う**——`sig` の uuid では持ち主が判らない（記録に対応表が無い）。
> `pol_si`／`pol_ti` は**席から見たスロット**なので、持ち主が確定する。

**攻撃かどうか**は行動型で判ける——**記録上、攻撃は `DON_BOX`（対象付き）か `ATTACK`** で来る
（`theory_order.score_candidate` の注記）。

> ## **本器は ④（除去要求）を測れない**（2026-09-14 に判明・**0 を結果として読まないこと**）
>
> **除去は `RESOLVE_EFFECT_SELECTION` で来て、対象は `selected_uuids` に入る**
> ——`pol_ti` は **−1** なので、枠で追う本器からは**構造上見えない**。
> 実測で `pol_ti` が相手の場を指すのは **`DON_BOX`（＝攻撃）だけ**だった。
>
> **したがって `removals_per_gone = 0` は「除去が起きていない」という意味ではない。**
> 出力の `removal_observable` が `false` のときは、**④ について何も言えない**。
> 効果の解決行は `unattributed_effect_rows` として数だけ出す（実測で実デッキは合成の 2 倍以上）。
>
> **測るには uuid の持ち主が要る**——記録に対応表は無いが、`sig` から組み立てられる:
> **自席ターンに行動主体（`sig[1]`）として出た uuid はその席のもの**・
> **自分の攻撃の対象（`sig[2]`）は相手のもの**。これは別器（T27-b）に切り出す。

## 吸ったパワーを値に換える（ユーザ指摘 2026-09-14）

> 「**キャラクターが吸ってるパワーも考慮に入れてほしい**」

**2000 を吸うのと 10000 を吸うのとでは価値が違う。** 攻撃側の行から
`x = 攻撃側の実効パワー − 被害者のリーダー` を読み、`min(c(x), Θ)·μ` で値を付ける
（`shield_rate.absorb_value` を共有）。**`x < 0` は 0**——リーダーに通らない攻撃を
キャラが吸っても何も守っていない。

## 「パワー下げ＋攻撃」も除去である（ユーザ指摘 2026-09-14）

> 「**除去要求は効果による除去だけでなくパワー下げ＋攻撃のパターンもある**」

この形は**攻撃行 1 つ**にしか見えないが、**相手はカードを 1 枚余分に払っている**。
同じ巡に**効果の解決**と**自分の場への攻撃**が同居した割合を出す（`debuff_then_attack`）。
**上限の目安**であって帰属ではない（効果が別の用途だった場合も入る）。

**そしてこれは測定そのものを歪める**——**パワーを下げられて生き残った体**は、
多重集合の差では「5000 が消えて 3000 が現れた」＝**消えた**と数えられる。
体の**数**の差は逆に、同じ巡に出した体で埋まると見落とす。**両方出して挟む**
（`gone_bracket`）。

## 交絡（読むときに添える）

- **1 つの相手ターンに複数の出来事が起きる**＝**体 1 体ごとの原因は特定できない**。
  本器が出すのは**ターン単位の突き合わせ**（消えた数 対 攻撃された数・除去された数）。
  **体ごとの帰属ではない**ので、割合として読む。
- **相手が撃ったが消えなかった**（守った・耐えた）も数える——分母が違うので別に出す。
- **各局・各席の最終巡は落ちる**（`ko_by_power` と同じ偏り）。

## 読み方（事前登録）

- **攻撃で消えた割合が大きければ、②（身代わり）が `ν` の主成分**。
- **除去で消えた割合が大きければ、④（除去要求）を項として足す根拠**になる。
- **どちらでもない退場（自分の都合）が大きければ**、そこは**素直に損**である。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/exit_ledger.py --in ~/w39 ~/w42 --out ~/exit.json
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
from ko_by_power import band_of, field_powers, lost  # noqa: E402
from shield_rate import absorb_value, attacker_power  # noqa: E402
from theory_order import PWR_EPS, slot_power  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_sig", "pol_si", "pol_ti", "pol_k")
#: 相手の席から見た**自分の場**の枠（`n_rel_feat` のスロット配置）
OPP_FIELD_SLOTS = range(7, 12)
#: 相手の席から見た**自分のリーダー**の枠
OPP_LEADER_SLOT = 1
#: 記録上、攻撃はこの 2 つで来る（`ATTACK` は候補には出ないが選ばれた手には出る）
ATTACK_TYPES = ("ATTACK", "DON_BOX")
#: 退場の仕方
EXITS = ("attacked", "removed", "own", "survived")


def _extra(dd, n):
    return {"tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def chosen_target(pol, ptr_i, k, chosen):
    """選ばれた候補の (行動型, 対象枠, 候補の index)。選べていなければ `None`。"""
    if chosen is None or chosen < 0 or chosen >= k:
        return None
    j = int(ptr_i) + int(chosen)
    try:
        sig = json.loads(pol["pol_sig"][j])
    except Exception:
        return None
    if not sig:
        return None
    return (sig[0], int(pol["pol_ti"][j]), j)


def absorbed_x(tok, pol, j):
    """**吸われた攻撃の超過パワー `x`**（攻撃側の行から読む）。

    ユーザ指摘 2026-09-14「**キャラクターが吸ってるパワーも考慮に入れてほしい**」。
    **2000 を吸うのか 10000 を吸うのかで身代わりの価値は変わる**——一律に数えてはいけない。

    攻撃側の席の行では **枠 1 が被害者のリーダー**なので、同じ行から
    `x = 攻撃側の実効パワー − 被害者のリーダー` が出る。
    """
    ap = attacker_power(tok, int(pol["pol_si"][j]), int(pol["pol_k"][j]))
    lead = slot_power(tok, OPP_LEADER_SLOT)
    if ap is None or lead is None:
        return None
    return float(ap) - float(lead)


def classify(at, ti):
    """相手の 1 手が**自分の場**に何をしたか。"""
    if ti not in OPP_FIELD_SLOTS:
        return None                              # 自分の場を狙っていない
    return "attacked" if at in ATTACK_TYPES else "removed"


def collect(dirs, limit_games=0):
    """(局, 席) ごとに、自席ターンの入口 2 つと**その間の相手の手**を突き合わせる。"""
    recs = []
    stats = {"games": 0, "pairs": 0, "opp_rows": 0, "no_choice": 0,
             # **効果の解決行**——ここに除去が居るが `pol_ti` からは持ち主が判らない
             "unattributed_effect_rows": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        board = {}                                # (席, ターン) → 場の多重集合
        acts = {}                                 # (席, ターン) → 相手がその席に何をしたか
        absorbed = {}                             # (席, ターン) → 吸った攻撃の `x` の並び
        eff = {}                                  # (席, ターン) → 相手が解決した効果の数
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0:
                continue
            if not PL.is_own_turn(w, t):
                continue        # **相手ターン中の自席の行は「守る側の窓」**＝攻撃はここに無い
            if (w, t) not in board:
                board[(w, t)] = field_powers(ex["tok"][i])
            # **攻撃側の行は攻撃側自身のターンの行**——そこから「相手の場（枠 7〜11）」を
            # 狙った手を数え、**被害者は反対の席**に付ける。
            stats["opp_rows"] += 1
            k = int(L[i])
            if k < 1:
                continue
            got = chosen_target(pol, ptr[i], k, int(rows["pol_chosen"][i]))
            if got is None:
                stats["no_choice"] += 1
                continue
            if got[0] == "RESOLVE_EFFECT_SELECTION":
                stats["unattributed_effect_rows"] += 1
                # **その相手ターンに効果が解決された**という事実だけ記録する。
                # 「パワー下げ＋攻撃」の除去は、この行と攻撃行の**組**で現れる。
                eff.setdefault((1 - w, t), 0)
                eff[(1 - w, t)] += 1
            lab = classify(got[0], got[1])
            if lab:
                acts.setdefault((1 - w, t), Counter())[lab] += 1
                if lab == "attacked":
                    x = absorbed_x(ex["tok"][i], pol, got[2])
                    if x is not None:
                        absorbed.setdefault((1 - w, t), []).append(x)
        seats = {}
        for (w, t) in board:
            seats.setdefault(w, []).append(t)
        for w, ts in seats.items():
            ts.sort()
            for a, b in zip(ts, ts[1:]):
                stats["pairs"] += 1
                before = board[(w, a)]
                gone = lost(before, board[(w, b)])
                # **その巡のあいだに相手が自分の場へ向けた手**（自席ターン a と b の間）
                hits = Counter()
                xs, n_eff = [], 0
                for t in range(a, b + 1):
                    hits += acts.get((w, t), Counter())
                    xs.extend(absorbed.get((w, t), []))
                    n_eff += eff.get((w, t), 0)
                after = board[(w, b)]
                # **多重集合の差は「パワーが下がって生き残った体」を消えたと数える**
                # （ユーザ指摘 2026-09-14）。体の**数**の差は逆に、出した体で埋まると
                # 見落とす。**上限と下限で挟む**。
                n_lo = max(0, len(before) - len(after))
                recs.append({"seed": seed, "turn": a, "n_before": len(before),
                             "n_gone": len(gone), "n_gone_lo": n_lo,
                             "gone_bands": [band_of(p) for p, _bl in gone],
                             "attacked": int(hits.get("attacked", 0)),
                             "removed": int(hits.get("removed", 0)),
                             "absorbed_x": xs, "opp_effects": int(n_eff)})
    return recs, stats


#: **`pol_ti` から除去は見えない**（対象が `selected_uuids` にあるため）。
#: 器がこの事実を持ち、**0 を結果として読ませない**。
REMOVAL_OBSERVABLE = False


def summarise(recs, reps=200, seed=0):
    """**ターン単位の突き合わせ**（体ごとの帰属ではない・割合として読む）。"""
    if not recs:
        return {"n": 0}
    gone = sum(r["n_gone"] for r in recs)
    before = sum(r["n_before"] for r in recs)
    atk = sum(r["attacked"] for r in recs)
    rem = sum(r["removed"] for r in recs)
    # **相手が向けた手が、消えた体をどこまで説明するか**（1 を超えたら守られた分がある）
    out = {"pairs": len(recs), "bodies": before, "gone": gone,
           "gone_p": round(gone / before, 4) if before else None,
           "opp_attacks_on_my_board": atk, "opp_removals_on_my_board": rem,
           "attacks_per_gone": round(atk / gone, 4) if gone else None,
           "removals_per_gone": round(rem / gone, 4) if gone else None,
           "covered_per_gone": round((atk + rem) / gone, 4) if gone else None}
    # **相手が何もしていないのに消えた巡**＝自分の都合の退場（素直な損）
    own = [r for r in recs if r["n_gone"] > 0 and r["attacked"] == 0 and r["removed"] == 0]
    out["gone_with_no_opp_action"] = sum(r["n_gone"] for r in own)
    out["own_exit_share"] = (round(out["gone_with_no_opp_action"] / gone, 4) if gone else None)
    # **吸ったパワー**——一律に数えず `min(c(x), Θ)·μ` で値を付ける（`x < 0` は 0）
    xs = [x for r in recs for x in r["absorbed_x"]]
    if xs:
        vals = [absorb_value(x) for x in xs]
        out["absorbed"] = {
            "n": len(xs),
            "x_mean": round(float(np.mean(xs)), 1),
            "no_connect_share": round(float(np.mean([x < -PWR_EPS for x in xs])), 4),
            "value_mean": round(float(np.mean(vals)), 5),
            "value_total": round(float(np.sum(vals)), 3),
            "value_per_gone": round(float(np.sum(vals)) / gone, 5) if gone else None,
            "by_x_band": {},
        }
        by = {}
        for x, v in zip(xs, vals):
            by.setdefault(band_of(max(0.0, x)), []).append(v)
        out["absorbed"]["by_x_band"] = {
            k: {"n": len(v), "value_mean": round(float(np.mean(v)), 5)}
            for k, v in sorted(by.items(), key=lambda kv: int(kv[0][1:]))}
    # **「パワー下げ＋攻撃」の除去**——効果の解決と攻撃が同じ巡に同居した割合
    atk_turns = [r for r in recs if r["attacked"] > 0]
    out["debuff_then_attack"] = {
        "turns_with_attack": len(atk_turns),
        "also_had_effect": sum(1 for r in atk_turns if r["opp_effects"] > 0),
        "share": (round(sum(1 for r in atk_turns if r["opp_effects"] > 0) / len(atk_turns), 4)
                  if atk_turns else None)}
    # **消えた体の上限と下限**（多重集合の差はパワー低下を損失と数える）
    lo = sum(r["n_gone_lo"] for r in recs)
    out["gone_lo"] = lo
    out["gone_bracket"] = [round(lo / before, 4) if before else None,
                           round(gone / before, 4) if before else None]
    # **④は本器では測れない**——`removal_*` は 0 に張り付くので、**結果として読ませない**
    out["removal_observable"] = REMOVAL_OBSERVABLE
    if not REMOVAL_OBSERVABLE:
        out["removal_note"] = ("除去は RESOLVE_EFFECT_SELECTION で来て対象が selected_uuids に"
                               "入るため pol_ti からは見えない。0 は「起きていない」ではない")
        out["removal_share_of_opp_actions"] = None
    else:
        tot = atk + rem
        out["removal_share_of_opp_actions"] = round(rem / tot, 4) if tot else None
    # パワー帯ごとの退場（T22 の山形をこの枠で読む）
    by = Counter()
    for r in recs:
        by.update(r["gone_bands"])
    out["gone_by_band"] = dict(sorted(by.items(), key=lambda kv: int(kv[0][1:])))
    out["ci95"] = _boot(recs, reps, seed)
    # **事前登録した読み方**——④が測れないうちは②の側だけ判定する
    if not REMOVAL_OBSERVABLE:
        out["verdict"] = "shield_measured_removal_not_observable"
    elif out["removal_share_of_opp_actions"] is not None:
        out["verdict"] = ("removal_term_is_material"
                          if out["removal_share_of_opp_actions"] >= 0.15
                          else "shield_dominates")
    return out


def _boot(recs, reps=200, seed=0):
    keys = ("attacks_per_gone", "removals_per_gone", "own_exit_share")
    if reps <= 0:
        return {k: [None, None] for k in keys}
    by = {}
    for r in recs:
        by.setdefault(r["seed"], []).append(r)
    gids = list(by)
    if len(gids) < 3:
        return {k: [None, None] for k in keys}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in keys}
    for _ in range(int(reps)):
        pick = rng.integers(0, len(gids), len(gids))
        sub = [r for g in pick for r in by[gids[g]]]
        g = sum(r["n_gone"] for r in sub)
        if not g:
            continue
        acc["attacks_per_gone"].append(sum(r["attacked"] for r in sub) / g)
        acc["removals_per_gone"].append(sum(r["removed"] for r in sub) / g)
        acc["own_exit_share"].append(
            sum(r["n_gone"] for r in sub
                if r["n_gone"] > 0 and r["attacked"] == 0 and r["removed"] == 0) / g)
    return {k: ([round(float(np.percentile(v, 2.5)), 4),
                 round(float(np.percentile(v, 97.5)), 4)] if len(v) >= 10 else [None, None])
            for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, stats = collect(a.src, a.limit_games)
    res = {"stats": stats, "summary": summarise(recs, a.boot_reps, a.seed),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
