"""**④ 除去要求を測る**——相手は自分の場に何枚払っているか（T27-b・読み取り専用）。

`docs/game_theory.md` **§14.1.2** の項④・`docs/cpu_theory_gap.md` の **T27-b**。

## なぜ別器が要るか——**枠では見えない**

`exit_ledger` は対象を**枠**（`pol_ti`）で追うので、**除去が見えない**:
除去は `RESOLVE_EFFECT_SELECTION` で来て、**対象は `selected_uuids`** に入り `pol_ti` は −1。
だから前器は `removal_observable = false` を返していた（**0 は「起きていない」ではない**）。

## 鍵——**uuid の持ち主は記録から組み立てられる**

記録に対応表は無いが、**ゲームの規則から決まる**:

| 手掛かり | なぜ持ち主が決まるか |
|---|---|
| **自席ターンの行動主体**（`sig[1]`） | **自分のカードしか動かせない**（攻撃・付与・起動・登場） |
| **自分の攻撃の対象**（`sig[2]`） | **相手のものしか殴れない** |

この 2 つで表を作ってから、`RESOLVE_EFFECT_SELECTION` の `selected_uuids` を引く。
**相手のターンに、自分のカードが選ばれていたら、それが除去要求**である。

> **持ち主が判らない uuid は数えない**（表に載らなかったもの）。
> **表が効いているかは別に検算する**——`PLAY` した uuid が**後で行動しているか**。
> これが高ければ **uuid はゾーンを跨いでも変わらない**＝表は使える。
> **被覆率が 0 でも「表が壊れている」とは限らない**——
> **選ばれたカードが盤面のカードではない**（手札・デッキ・トラッシュ）場合も 0 になる。
> **この 2 つを判定で区別する**（`map_validated`）。

## 何を数えるか

| 量 | 意味 |
|---|---|
| `demand` | **相手が自分のカードを効果で選んだ回数**＝相手が払った労力 |
| `demand_per_turn` | 1 巡あたり |
| `on_play_bundled` | **同じターンに相手が登場もしている**割合＝**カード 1 枚が体と除去を兼ねている** |
| `coverage` | 選ばれた uuid のうち持ち主が判った割合 |

> **`on_play_bundled` が高ければ、④を丸ごと「相手の手札 −1」と数えてはいけない**
> ——**登場時効果の除去は体と抱き合わせ**なので、相手はカード 1 枚で 2 つ得ている。
> その場合、④に付ける値は `μ` より小さい。

## 読み方（事前登録）

- **`demand_per_turn` が大きいほど ④ は `ν` の主成分に近い**。
- **実デッキ ≫ 合成なら、④を入れないと実デッキで最も外す**（前器の示唆と整合）。
- **`coverage` が 0.8 を下回ったとき、意味は 2 通り**——
  **`map_validated` が高ければ「効果は盤面を狙っていない」**（結果である）、
  **低ければ「表が使えない」**（結論を出さない）。

**限界**: 選ばれた＝除去されたとは限らない（レスト・パワー下げ・移動も入る）＝
**「相手が自分の場に効果を向けた回数」の上限**である。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/removal_demand.py --in ~/w41 --out ~/removal.json
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

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_sig",)
#: **自分のカードしか動かせない**行動（行動主体＝その席のもの）
ACTOR_TYPES = ("ATTACK", "DON_BOX", "ATTACH_DON", "ACTIVATE_MAIN", "PLAY")
#: **相手のものしか殴れない**行動（対象＝反対の席のもの）
TARGET_TYPES = ("ATTACK", "DON_BOX")
#: 効果の対象選択（**除去はここに居る**）
SELECT_TYPE = "RESOLVE_EFFECT_SELECTION"
#: 被覆率がこれを下回ったら「除去を数えた」とは言わない
MIN_COVERAGE = 0.8
#: **表の検算**がこれを超えていれば uuid はゾーンを跨ぐ＝表は使える
MAP_VALID_MIN = 0.4


def chosen_sig(pol, ptr_i, k, chosen):
    """選ばれた候補の `sig`（選べていなければ `None`）。"""
    if chosen is None or chosen < 0 or chosen >= k:
        return None
    try:
        sig = json.loads(pol["pol_sig"][int(ptr_i) + int(chosen)])
    except Exception:
        return None
    return sig or None


def own_of(sig, who):
    """1 手から判る持ち主の割り当て `{uuid: 席}`（**ゲームの規則から決まるものだけ**）。"""
    out = {}
    at = sig[0] if sig else None
    if at in ACTOR_TYPES and sig[1]:
        out[str(sig[1])] = who               # 自分のカードしか動かせない
    if at in TARGET_TYPES:
        for u in (sig[2] or ()):
            out[str(u)] = 1 - who            # 相手のものしか殴れない
    return out


def build_owners(rows, pol, L, ptr, idx):
    """**局の全行から持ち主の表を作る**（自席ターンの手だけを使う）。"""
    owners = {}
    for i in idx:
        w, t = int(rows["who"][i]), int(rows["turn"][i])
        if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
            continue
        k = int(L[i])
        if k < 1:
            continue
        sig = chosen_sig(pol, ptr[i], k, int(rows["pol_chosen"][i]))
        if sig:
            owners.update(own_of(sig, w))
    return owners


def collect(dirs, limit_games=0):
    """相手が**自分のカードを効果で選んだ**回数を、局・席・ターンごとに数える。"""
    recs = []
    stats = {"games": 0, "select_rows": 0, "selected_uuids": 0, "owner_known": 0,
             "owners_mapped": 0,
             # **表の検算**——`PLAY` した uuid が後で行動していれば uuid はゾーンを跨ぐ
             "played": 0, "played_later_acted": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        owners = build_owners(rows, pol, L, ptr, idx)
        stats["owners_mapped"] += len(owners)
        pl, ac = set(), set()
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            k = int(L[i])
            if k < 1:
                continue
            sg = chosen_sig(pol, ptr[i], k, int(rows["pol_chosen"][i]))
            if not sg or not sg[1]:
                continue
            if sg[0] == "PLAY":
                pl.add(str(sg[1]))
            elif sg[0] in ACTOR_TYPES:
                ac.add(str(sg[1]))
        stats["played"] += len(pl)
        stats["played_later_acted"] += len(pl & ac)
        demand = {}          # (被害者の席, ターン) → 選ばれた回数
        played = set()       # (席, ターン) 相手がそのターンに登場もしたか
        turns = set()
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            turns.add((w, t))
            k = int(L[i])
            if k < 1:
                continue
            sig = chosen_sig(pol, ptr[i], k, int(rows["pol_chosen"][i]))
            if not sig:
                continue
            if sig[0] == "PLAY":
                played.add((w, t))
            if sig[0] != SELECT_TYPE:
                continue
            stats["select_rows"] += 1
            for u in (sig[3] or ()):
                stats["selected_uuids"] += 1
                o = owners.get(str(u))
                if o is None:
                    continue                      # 持ち主が判らない uuid は数えない
                stats["owner_known"] += 1
                if o == 1 - w:                    # **相手のカードを選んだ**＝除去要求
                    demand[(o, t)] = demand.get((o, t), 0) + 1
        for (victim, t), n in demand.items():
            recs.append({"seed": seed, "turn": t, "victim": victim, "demand": n,
                         "attacker_played": ((1 - victim, t) in played)})
        for (w, t) in turns:
            if (w, t) not in demand:
                recs.append({"seed": seed, "turn": t, "victim": w, "demand": 0,
                             "attacker_played": ((1 - w, t) in played)})
    return recs, stats


def summarise(recs, stats, reps=200, seed=0):
    if not recs:
        return {"n": 0}
    dem = sum(r["demand"] for r in recs)
    hit = [r for r in recs if r["demand"] > 0]
    cov = (stats["owner_known"] / stats["selected_uuids"]) if stats["selected_uuids"] else None
    out = {"turns": len(recs), "demand": dem,
           "demand_per_turn": round(dem / len(recs), 4),
           "turns_with_demand_share": round(len(hit) / len(recs), 4),
           "coverage": round(cov, 4) if cov is not None else None,
           # **登場と同じターンなら、カード 1 枚が体と除去を兼ねている**
           "on_play_bundled": (round(sum(1 for r in hit if r["attacker_played"]) / len(hit), 4)
                               if hit else None)}
    out["ci95"] = _boot(recs, reps, seed)
    # **表が効いているかの検算**——`PLAY` した uuid が後で行動する割合。
    # 高ければ uuid はゾーンを跨いでも変わらない＝**表は使える**。
    val = ((stats["played_later_acted"] / stats["played"]) if stats.get("played") else None)
    out["map_validated"] = round(val, 4) if val is not None else None
    out["owners_mapped"] = stats["owners_mapped"]
    # **被覆率 0 の意味は 2 通り**——表が壊れている／選ばれたのが盤面のカードではない。
    # **検算が通っていれば後者**であり、そのときは「除去が見えない」ではなく
    # **「効果は盤面を狙っていない」**という結果になる。
    if val is not None and val >= MAP_VALID_MIN and cov is not None and cov < MIN_COVERAGE:
        out["verdict"] = "effects_do_not_target_the_board"
    elif val is not None and val < MAP_VALID_MIN:
        out["verdict"] = "owner_map_unusable"
    else:
        out["verdict"] = ("removal_term_is_material" if out["demand_per_turn"] >= 0.15
                          else "removal_term_is_small")
    return out


def _boot(recs, reps=200, seed=0):
    if reps <= 0 or not recs:
        return [None, None]
    by = {}
    for r in recs:
        by.setdefault(r["seed"], []).append(r)
    gids = list(by)
    if len(gids) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(reps)):
        pick = rng.integers(0, len(gids), len(gids))
        sub = [r for g in pick for r in by[gids[g]]]
        if sub:
            vals.append(sum(r["demand"] for r in sub) / len(sub))
    return ([round(float(np.percentile(vals, 2.5)), 4),
             round(float(np.percentile(vals, 97.5)), 4)] if len(vals) >= 10 else [None, None])


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
    res = {"stats": stats, "summary": summarise(recs, stats, a.boot_reps, a.seed),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
