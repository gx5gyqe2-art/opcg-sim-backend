#!/usr/bin/env python3
"""**決着の定義**（T136・2026-09-23・ユーザ指示「勝敗との紐付けは必須で…理論の側に決着の定義が必要かもね」→「それでお願いします。」）。
T117（「今このターンで殺せるか」・2026-09-19）を**決着の定義**として仕切り直したもの。

## 問い

理論には**決着（ここから先はどちらが勝つか決まっている）の定義が無かった**。交点の橋は「先に届いた側」を暗黙に
使い、線形の橋には無く、T117 の「殺せる判定」は**適合率 0.23**（桁で外れている唯一の場所）で止まっていた。
**勝敗の説明は、決着後（勝者は確定・「誰が・いつ」を当てる）と決着前（勝率で言う）に分かれる**——その境目がこの器。

## 定義（規則だけ・完全情報・新定数ゼロ）

> **ターン t の開始時に手番側 w が決着** ⇔ **相手が最善で守っても通る本数 ≥ 相手の残りライフ ＋ 1**。

**「＋1」は規則**（`rules/battle.rs`: **ライフが 0 のときに損害を受けたら負け**＝最後のライフ札を取られても負けではない）。
**T117 の判定（`≥ ライフ`）は 1 本足りない側に外れていた**うえ、**ライフ 0 の行（1 本通れば勝ち）を除外していた**。
本 T の `by_life` 表（ライフ 1 でドン込みの精度 0.89・そのターンに終わった精度は 0.46）がこれを露わにした。

* **通る本数** … 攻撃手（リーダー＋このターン攻撃できる体・`own_attackers_of`）のうち、
  相手の**アクティブなブロッカー**に横取りされない本数（ブロッカーは安い攻撃から止める＝守り手の最良）。
* **止められる本数** … **守り手の実際の手札**（§0.05・**相手席の直近の自席ターンの最後の行**から読む＝T130 と同じ規約）で、
  カウンター値が足りる組を作れる本数。**`LETHAL_STOP_MODE=max`（既定）は「切れるだけ切る」**＝詰みの判定
  （守り手はライフ 0 なら経済を捨てて守る・規則）。**安い攻撃から止め、1 本ごとに使うカウンター値が最小の組を選ぶ**
  （本数を最大にする貪欲）。`econ` は T130 の `attacks_stopped`（受けるより安いときだけ切る）＝比較用。
* **`LETHAL_HAND_MODE`** … `actual`（既定・実際の手札）／`share`（T117 の旧規約＝`相手の手札枚数 × デッキの切れる割合`）。
  **T117 の 0.23 を再現するための切替**。
* **ドン付与** … `--don on`（既定）で**アクティブなドンを安い攻撃から 1 体 4 枚まで**（規則）配って `x` を上げる。
* **カウンターのイベントはドンを払う**（規則）——守り手が使えるのは**相手ターン開始時に残しているアクティブなドン**
  （`sc[SC_OPP_DON_ACTIVE]`）の範囲まで。**イベントの費用の和がそれを超える組は選べない**。
* **守り手の手札の読みは正確**——「最後の判断行で出した札が残っているのでは」と疑って落とす処理を入れて数えたが
  **0 回**（ターン最後の判断行の選択は `TURN_END` で、その行の手札は出した後のもの）。処理は外した。
* **`LETHAL_LIFE_MODE`** … `draw`（既定）＝**受けたライフの札は手札に入る**（規則・`rules/battle.rs` の `dest = Zone::Hand`）ので、
  守り手は**残りライフの枚数**（全部が手札に入ってから、次の損害で負ける）を**デッキの平均カウンター値**の札として追加で持つ
  （**完全情報でもライフの中身の順は読めない**のでデッキ平均＝T91 と同じ規約・新定数ゼロ）。**先に全部持たせる**のは
  守り手に有利な上限＝**宣言は健全側**（取りこぼしは増えうる）。`off`＝旧（T117 と同じ・ライフの札を数えない）。

## 測るもの（真値は「その局を誰が勝ったか」と「いつ終わったか」）

| 名前 | 何か |
|---|---|
| `precision_winner` | 決着と宣言した行のうち、**宣言した席が実際に勝った**割合（**主指標**） |
| `precision_kill` | 決着と宣言した行のうち、**そのターンに終わった**割合（T117 の適合率・比較用） |
| `recall_end` | 勝者の**最後のターン**で決着と宣言できた割合（T117 の再現率） |
| `recall_game` | 勝者が**終局までのどこか**で決着と宣言された局の割合 |
| `lead` | 勝者の最初の宣言から終局までのターン数（先読み・分布） |
| `false_declared` | 敗者の席で宣言した行の数（理論の誤りか CPU の見落としかは記録からは分からない＝内訳を残す） |

**`precision_winner` が 1 に届かない分**は、(a) 定義の誤り（理論）か (b) 詰みが在ったのに CPU が決めなかった（打ち筋・§8.5）。
記録からは分けられないので、**`false_declared` の行で「実際にそのターン何本通ったか」を添えて**後で見分けられるようにする。

## T138a（2026-09-23）: 決着フラグを他の器へ渡す

`collect` の内部ループを共有下請け `_iter_declared_games` に分離し、`settled_map(dirs) -> {(seed, w, t): bool}`
を追加した。**判定の式（`lethal_of_row`）は 1 か所のまま**——`collect` と `settled_map` が別の答えを出す経路が無い。
`two_curves`（T137d）・`win_calib`／`relative_ledger`／`crossing_bridge`（T138b）が「このターンより後は決着後」を
読む入口になる。**この分離で出力は 1 バイトも変わらない**（実記録 w41・合成 w39+w42 で旧版と `json.load` の
辞書が完全一致することを確認済み）。

使い方:

    python tests/scripts/lethal_rule.py --in <records_dir> [--games N] [--don on|off]
                                        [--hand actual|share] [--stop max|econ] [--life draw|off]
                                        [--json out.json] [--dump rows.json]
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import crossing_bridge as CB  # noqa: E402
import guard_afford as GA  # noqa: E402
import hand_guard as HG  # noqa: E402
import hand_plan as HP  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import (MU, PWR_EPS, SC_MY_DON, SC_MY_LIFE, SC_OPP_HAND, SC_OPP_LEADER_POWER,  # noqa: E402
                          SC_OPP_LIFE, THETA, c_of, own_attackers_of)

#: **守り手の手札の読み方**——`actual`（既定・§0.05 完全情報＝相手席の行から実際の値）／`share`（T117 の旧規約）
LETHAL_HAND_MODES = ("actual", "share")
LETHAL_HAND_MODE = "actual"
#: **守り手がどこまで切るか**——`max`（既定・切れるだけ切る＝詰みの判定）／`econ`（受けるより安いときだけ・T130 の規則）
LETHAL_STOP_MODES = ("max", "econ")
LETHAL_STOP_MODE = "max"
#: **受けたライフの札を守り手のカウンターに数えるか**——`draw`（既定・規則）／`off`（T117 の旧規約）
LETHAL_LIFE_MODES = ("draw", "off")
LETHAL_LIFE_MODE = "draw"
#: 1 体に付けられるドンの上限（規則）
DON_PER_BODY = 4
#: 状態スカラーの **相手のアクティブなドン**（`encoder.py` の並び: 0 自ライフ・1 相手ライフ・2 自アクティブ・3 自レスト・4 相手アクティブ）
SC_OPP_DON_ACTIVE = 4


def set_lethal_hand_mode(mode):
    global LETHAL_HAND_MODE
    if mode not in LETHAL_HAND_MODES:
        raise ValueError("lethal hand mode は %s のどれか" % (LETHAL_HAND_MODES,))
    LETHAL_HAND_MODE = mode
    return LETHAL_HAND_MODE


def set_lethal_stop_mode(mode):
    global LETHAL_STOP_MODE
    if mode not in LETHAL_STOP_MODES:
        raise ValueError("lethal stop mode は %s のどれか" % (LETHAL_STOP_MODES,))
    LETHAL_STOP_MODE = mode
    return LETHAL_STOP_MODE


def set_lethal_life_mode(mode):
    global LETHAL_LIFE_MODE
    if mode not in LETHAL_LIFE_MODES:
        raise ValueError("lethal life mode は %s のどれか" % (LETHAL_LIFE_MODES,))
    LETHAL_LIFE_MODE = mode
    return LETHAL_LIFE_MODE


def avg_counter(deck_ids, cards):
    """**デッキ 1 枚あたりの平均カウンター値**（ライフから手札に入る札の読み・T91 の `cut_share` と同じ規約＝完全なデッキ組成）。"""
    vals = []
    for cid in deck_ids or ():
        info = cards.info(cid) or {}
        vals.append(float(info.get("counter") or 0.0))
    return float(np.mean(vals)) if vals else 0.0


def life_cards_as_counters(life, life_counter):
    """**受けたライフの札**（規則: 手札に入る）を守り手のカウンターとして数える。**ライフの札は全部手札に入ってから
    次の損害で負ける**ので `残りライフ` 枚。**先に全部持たせる**（守り手に有利な上限＝宣言は健全側）。"""
    n = int(round(float(life)))
    if n <= 0 or float(life_counter) <= 0.0:
        return []
    return [float(life_counter)] * n


def stops_of_share(n_hand_cut, xs):
    """**T117 の旧規約**: 切れる札の枚数（`cuttable_share × 手札`）から、安い順に `c(x)` 枚ずつ払って何本止まるか。"""
    left = float(n_hand_cut)
    n = 0
    for x in sorted(float(v) for v in (xs or ())):
        need = c_of(x)
        if need <= 0.0 or left < need - 1e-9:
            break
        left -= need
        n += 1
    return n


def stop_min_counter(counters, x, costs=None, budget=float("inf")):
    """超過 `x` を止めるのに**使うカウンター値が最小の組**（`counters` = 各札のカウンター値の列）。
    止められなければ `None`。`x < 0` は止める必要が無い（`()`）。同値は命中するので `x + 1000` 以上が要る。
    `costs`（各札の**イベントの費用**・非イベントは 0）と `budget`（守り手のアクティブなドン）を渡すと、
    **費用の和が予算を超える組は選べない**（規則: カウンターのイベントはドンを払う）。"""
    x = float(x)
    if x < -PWR_EPS:
        return ()
    need = x + 1000.0 - PWR_EPS
    n = min(len(counters), HG.HAND_MAX)
    costs = [0.0] * n if costs is None else list(costs)[:n]
    best, best_idx = None, None
    for r in range(1, n + 1):
        for idx in itertools.combinations(range(n), r):
            s = sum(counters[i] for i in idx)
            if s < need:
                continue
            if sum(costs[i] for i in idx) > budget + 1e-9:
                continue
            if best is None or s < best:
                best, best_idx = s, idx
    return best_idx


def max_stops(counters, xs, costs=None, budget=float("inf")):
    """**守り手が切れるだけ切ったとき止まる本数**（規則・詰みの判定に使う）。

    **安い攻撃から止め、1 本ごとに使うカウンター値が最小の組を選ぶ**——止まる本数を最大にする貪欲。
    `x < 0` の攻撃（通らない）は数えない＝呼ぶ側が `xs` を通る本数に絞っておく。
    `costs`／`budget` はイベントのドン費用と守り手のアクティブなドン（使った分だけ予算が減る）。"""
    pairs = [(float(c), 0.0 if costs is None else float(costs[i]))
             for i, c in enumerate(counters) if float(c) > 0.0]
    left = pairs
    n = 0
    budget = float(budget)
    for x in sorted(float(v) for v in (xs or ())):
        if x < -PWR_EPS:
            continue
        idx = stop_min_counter([c for c, _k in left], x, [k for _c, k in left], budget)
        if idx is None:
            continue                                  # この 1 本は止まらない＝次の（より高い）も止まらないが、確認は続ける
        budget -= sum(left[i][1] for i in idx)
        left = [pc for i, pc in enumerate(left) if i not in idx]
        n += 1
    return n


def attach_don(xs, don):
    """**アクティブなドンを安い攻撃から 1 体 4 枚まで配る**（規則・1 枚 +1000）。付けられない分は捨てる。"""
    xs = sorted(float(v) for v in (xs or ()))
    k = int(round(max(0.0, float(don))))
    i = 0
    while k > 0 and i < len(xs):
        take = min(DON_PER_BODY, k)
        xs[i] += 1000.0 * take
        k -= take
        i += 1
    return xs


def lethal_of_row(sc, tok, with_don=True, defender_counters=None, cut_share=None, take_cost=None,
                  defender_items=None, life_counter=None, defender_costs=None):
    """1 行の判定 `(決着か, 内訳)`。**新定数ゼロ・打ち筋を仮定しない**。

    `defender_counters` … 守り手の実際の手札のカウンター値の列（`LETHAL_HAND_MODE=actual`・**相手席の行から**）。
    `cut_share` … 守り手のデッキの切れる割合（`share` のときだけ・引けなければ 1.0＝悲観側）。
    `defender_items`／`take_cost` … `LETHAL_STOP_MODE=econ` のときだけ（T130 の `attacks_stopped`）。
    `life_counter` … 守り手のデッキの平均カウンター値（`LETHAL_LIFE_MODE=draw`・`actual` のときに要る）。
    `defender_costs` … 守り手の各札の**イベントの費用**（非イベントは 0・無ければ費用ゼロ扱い）。
    予算は `sc[SC_OPP_DON_ACTIVE]`（相手がこのターン開始時に残しているアクティブなドン）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    life = float(sc[SC_OPP_LIFE])
    xs = list(own_attackers_of(tok, olp))
    if with_don:
        xs = attach_don(xs, float(sc[SC_MY_DON]))
    n_block = int(CB._opp_active_blockers(tok))
    xs_sorted = sorted(xs)
    xs_through = xs_sorted[n_block:] if n_block else xs_sorted   # ブロッカーは安い攻撃から横取りする
    xs_through = [x for x in xs_through if x >= -PWR_EPS]        # 通らない攻撃（相手リーダー未満）は数えない
    through = len(xs_through)
    if LETHAL_HAND_MODE == "share":
        n_cut = float(sc[SC_OPP_HAND]) * (1.0 if cut_share is None else float(cut_share))
        stops = stops_of_share(n_cut, xs_through)
        hand_read = round(n_cut, 3)
    else:
        if defender_counters is None:
            raise ValueError("LETHAL_HAND_MODE='actual' なのに守り手の手札が渡されていない（相手席の行から読む）")
        pool = list(defender_counters)
        if LETHAL_LIFE_MODE == "draw":
            if life_counter is None:
                raise ValueError("LETHAL_LIFE_MODE='draw' なのにデッキの平均カウンター値が渡されていない")
            pool += life_cards_as_counters(life, life_counter)
        if LETHAL_STOP_MODE == "econ":
            if defender_items is None or take_cost is None:
                raise ValueError("LETHAL_STOP_MODE='econ' には守り手の items と受ける費用が要る")
            extra = [{"counter": c, "v": None, "cid": "", "cost": 0.0, "event": False} for c in pool[len(defender_counters):]]
            stops = int(HP.attacks_stopped(list(defender_items) + extra, xs_through, take_cost))
        else:
            costs = (list(defender_costs) if defender_costs is not None else [0.0] * len(defender_counters))
            costs += [0.0] * (len(pool) - len(defender_counters))          # ライフの札はキャラ扱い（費用 0）
            stops = max_stops(pool, xs_through, costs, float(sc[SC_OPP_DON_ACTIVE]))
        hand_read = len(defender_counters)
    hits = max(0, through - stops)
    # **規則**: ライフ 0 で損害を受けたら負け＝通る本数が「残りライフ ＋ 1」以上で決着（ライフ 0 なら 1 本）
    return (life >= 0.0 and hits >= life + 1.0), {"xs": len(xs), "blockers": n_block, "through": through,
                                           "stops": int(stops), "hits": int(hits), "life": life,
                                           "hand_read": hand_read, "xs_through": [round(x) for x in xs_through],
                                           "counters": [round(c) for c in (defender_counters or [])],
                                           "opp_don": float(sc[SC_OPP_DON_ACTIVE]),
                                           "don": float(sc[SC_MY_DON]) if with_don else 0.0}


def declare_metrics(games):
    """局ごとの記録 `[(winner, t_end, [(w, t, declared, killed_now)…])…]` から決着の 6 指標を出す（算術だけ）。"""
    n_decl = tp_win = tp_kill = 0
    end_hit = end_total = 0
    game_hit = 0
    leads = []
    false_decl = 0
    both = 0
    for winner, t_end, rows in games:
        first = None
        decl_loser = False
        for w, t, declared, killed_now in rows:
            if not declared:
                continue
            n_decl += 1
            if w == winner:
                tp_win += 1
                if first is None:
                    first = t
            else:
                false_decl += 1
                decl_loser = True
            if killed_now:
                tp_kill += 1
        if winner is not None:
            end_total += 1
            if any(w == winner and t == t_end and d for w, t, d, _k in rows):
                end_hit += 1
            if first is not None:
                game_hit += 1
                leads.append(int(t_end - first))
                if decl_loser:
                    both += 1
    leads_arr = np.asarray(leads, dtype=float) if leads else np.zeros(0)
    return {"declared": n_decl,
            "precision_winner": round(tp_win / max(1, n_decl), 4),
            "precision_kill": round(tp_kill / max(1, n_decl), 4),
            "recall_end": round(end_hit / max(1, end_total), 4),
            "recall_game": round(game_hit / max(1, end_total), 4),
            "false_declared": false_decl,
            "games_both_declared": both,
            "lead": {"n": int(leads_arr.size),
                     "mean": round(float(leads_arr.mean()), 3) if leads_arr.size else None,
                     "median": round(float(np.median(leads_arr)), 1) if leads_arr.size else None,
                     "p_zero": round(float((leads_arr == 0).mean()), 4) if leads_arr.size else None,
                     "hist": {str(int(k)): int(v) for k, v in
                              zip(*np.unique(leads_arr, return_counts=True))} if leads_arr.size else {}}}


def _iter_declared_games(dirs, limit_games=0, with_don=True):
    """**T138a**: `collect`／`settled_map` が共有する下請け——記録を 1 度読んで、局ごとに
    `(seed_g, winner, t_end, rows)` を返す（`rows` = `[(w, t, declared, killed_now, dd, hand_missing, life_counter)]`）。
    `dd` は `lethal_of_row` の内訳（`hand_missing` 行では `{}`）。**判定の式は 1 か所**（`lethal_of_row`）に
    しかない＝`collect` と `settled_map` が別の答えを出す経路が無い。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    import deck_refill as DR
    refill = DR.shares_by_seed(dirs) if LETHAL_HAND_MODE == "share" else {}
    decks = DR.decks_by_seed(dirs) if LETHAL_LIFE_MODE == "draw" else {}
    if LETHAL_LIFE_MODE == "draw" and not decks:
        raise ValueError("LETHAL_LIFE_MODE='draw' なのにデッキが引けない（%s）" % (dirs,))
    avg_cache = {}
    games = 0
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(r["seed"][idx[0]]) if len(idx) else -1
        sh = refill.get(seed_g)
        dk = decks.get(seed_g) if decks else None
        turn_start, turn_last, turn_seq = {}, {}, {0: [], 1: []}
        z_of = {}
        for i in idx:
            w, t = int(r["who"][i]), int(r["turn"][i])
            z = float(r["z"][i])
            if z != 0.0:
                z_of[w] = 1.0 if z > 0 else 0.0
            if int(r["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            turn_last[(w, t)] = (ex["sc"][i], ex["tok"][i], ex["ci"][i])
            if (w, t) not in turn_start:
                turn_start[(w, t)] = (ex["sc"][i], ex["tok"][i], ex["ci"][i])
                turn_seq[w].append(t)
        winner = None
        for w, zz in z_of.items():
            if zz > 0.5:
                winner = w
        t_end = max((t for (w, t) in turn_start), default=-1)
        rows_out = []
        for w in (0, 1):
            for t in turn_seq[w]:
                sc, tok, _ci = turn_start[(w, t)]
                d = 1 - w
                prev = [tt for tt in turn_seq[d] if tt < t]
                counters = items = costs = None
                take = None
                if prev:
                    sc_d, tok_d, ci_d = turn_last.get((d, prev[-1]), turn_start[(d, prev[-1])])
                    items = HP.hand_items(tok_d, ci_d, idx2cid, cards,
                                          float(np.asarray(sc_d)[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0,
                                          max(1.0, min(5.0, float(np.asarray(sc_d)[SC_OPP_LIFE]))))
                    counters = [float(it["counter"]) for it in items]
                    costs = [float(it["cost"]) if it.get("event") else 0.0 for it in items]
                    take = HG.take_cost_of(float(np.asarray(sc_d)[SC_MY_LIFE]), MU)
                elif LETHAL_HAND_MODE == "actual":
                    # 相手がまだ 1 度も打っていない（先手の 1 ターン目）＝手札は初手 5 枚のまま読めない → 宣言しない
                    rows_out.append((w, t, False, False, {}, True, None))
                    continue
                cs = float(sh[d]) if sh is not None else None
                lc = None
                if LETHAL_LIFE_MODE == "draw" and LETHAL_HAND_MODE == "actual":
                    key = (seed_g, d)
                    if key not in avg_cache:
                        if dk is None or dk[d] is None:
                            raise ValueError("seed %d 席 %d のデッキが引けない" % (seed_g, d))
                        avg_cache[key] = avg_counter(dk[d], cards)
                    lc = avg_cache[key]
                declared, dd = lethal_of_row(sc, tok, with_don, counters, cs, take, items, life_counter=lc,
                                             defender_costs=costs)
                killed_now = bool(winner == w and t == t_end)
                rows_out.append((w, t, declared, killed_now, dd, False, lc))
        yield seed_g, winner, t_end, rows_out


def settled_map(dirs, limit_games=0, with_don=True):
    """**T138a**: 局×席×ターンの決着フラグ `{(seed, w, t): declared(bool)}`。**判定は `lethal_of_row` そのもの**
    （`collect` と同じ下請け `_iter_declared_games` を読むだけ）——他の器（`two_curves`・`win_calib`・
    `relative_ledger`・`crossing_bridge`）が「このターンより後は決着後」を読むための入口。
    **手札が読めない行（先手 1 ターン目）は `False`**（宣言しない・`collect` と同じ規約）。"""
    out = {}
    for seed_g, _winner, _t_end, rows in _iter_declared_games(dirs, limit_games, with_don):
        for w, t, declared, _killed_now, _dd, _hand_missing, _lc in rows:
            out[(seed_g, w, t)] = bool(declared)
    return out


def collect(dirs, limit_games=0, with_don=True, dump=None):
    """記録を 1 度読んで決着の指標と内訳を出す。守り手の手札は**相手席の直近の自席ターンの最後の行**から読む。
    `dump` に list を渡すと**宣言した行と勝者の最後のターンの全内訳**を積む（診断用・取りこぼしも読める）。"""
    games_out = []
    stats = {"games": 0, "rows": 0, "hand_missing": 0, "with_don": with_don,
             "hand_mode": LETHAL_HAND_MODE, "stop_mode": LETHAL_STOP_MODE, "life_mode": LETHAL_LIFE_MODE,
             "by_life": {},                           # 相手ライフ別: 宣言数・勝者の宣言数
             "false_rows": []}                        # 敗者の席で宣言した行の内訳（先頭 200 件）
    for seed_g, winner, t_end, rows in _iter_declared_games(dirs, limit_games, with_don):
        stats["games"] += 1
        rows_out = []
        for w, t, declared, killed_now, dd, hand_missing, lc in rows:
            stats["rows"] += 1
            if hand_missing:
                stats["hand_missing"] += 1
                rows_out.append((w, t, declared, killed_now))
                continue
            rows_out.append((w, t, declared, killed_now))
            if declared:
                lk = str(int(dd["life"]))
                bl = stats["by_life"].setdefault(lk, {"declared": 0, "winner": 0, "killed_now": 0})
                bl["declared"] += 1; bl["winner"] += int(winner == w); bl["killed_now"] += int(killed_now)
            if dump is not None and (declared or killed_now):
                # **宣言した行**と**勝者の最後のターン（取りこぼしの診断用）**を積む
                dump.append({"seed": seed_g, "w": w, "t": t, "t_end": int(t_end), "declared": bool(declared),
                             "winner": winner, "life_counter": lc, **dd})
            if declared and winner is not None and w != winner and len(stats["false_rows"]) < 200:
                stats["false_rows"].append({"seed": seed_g, "w": w, "t": t, "t_end": int(t_end), **dd})
        games_out.append((winner, t_end, rows_out))
    out = declare_metrics(games_out)
    out.update({k: v for k, v in stats.items() if k != "false_rows"})
    out["false_rows_sample"] = stats["false_rows"][:20]
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="決着の定義を規則から測る（T136）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--don", default="on", choices=("on", "off"))
    ap.add_argument("--hand", default=None, choices=LETHAL_HAND_MODES,
                    help="守り手の手札の読み方（既定 `actual`＝相手席の行から実際の値・`share`＝T117 の旧規約）")
    ap.add_argument("--stop", default=None, choices=LETHAL_STOP_MODES,
                    help="守り手がどこまで切るか（既定 `max`＝切れるだけ・`econ`＝受けるより安いときだけ）")
    ap.add_argument("--life", default=None, choices=LETHAL_LIFE_MODES,
                    help="受けたライフの札を守り手のカウンターに数えるか（既定 `draw`＝規則・`off`＝T117 の旧規約）")
    ap.add_argument("--json", default="")
    ap.add_argument("--dump", default="", help="宣言した行の全内訳を JSON に書く（診断用）")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.hand:
        set_lethal_hand_mode(a.hand)
    if a.stop:
        set_lethal_stop_mode(a.stop)
    if a.life:
        set_lethal_life_mode(a.life)
    dump = [] if a.dump else None
    out = collect(a.src, a.games, a.don == "on", dump=dump)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
