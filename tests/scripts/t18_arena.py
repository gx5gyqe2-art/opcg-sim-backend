#!/usr/bin/env python3
"""**T142**（2026-09-23）: T18（出口）の器を作り、**10 局の乾式運転**で配線を確かめる。
**判定の本番は T18**（本器は統計的な結論を出さない・勝率は比較しない）。

## 問い

T18 は「理論が純損と言う手を禁じて勝率が動くか」を測る出口。T140（生の橋）・T141（影の介入・
介入なしで率と型を測った）を経て、**実際に打ち回しへ介入する器**が要る。本 T はその器
（`t18_arena.py`）を作り、**小さい規模（10 局）で壊れずに最後まで打てるかだけを確かめる**——
勝率の判定・統計的な結論は次の T18 に残す。

## 介入の機構（ユーザ決定 2026-09-23「完全情報で進めてください」）

**完全情報のまま進める**——両席の手札・デッキが見える前提を崩さない（`cpu_theory_gap.md` §0.05
の 2026-09-17 決定と同じ・T140 の橋も元々完全情報）。相手の手札を伏せる層は本 T では作らない。

**「禁じる」の中身**（T18 の一行「理論が純損と言う手を禁じる」だけでは機構が 1 つに決まらないので、
ここで選んだ形を明記する——**新しい判断基準ではなく、T140／T141 が既に計算している値をそのまま
使う**）: 決定点で `shadow_forbid.shadow_row` の判定が `forbidden`（`played_family != "attach"`——
T141 の申し送りどおり `attach` は対象から除く）なら、**その手を、理論の最善候補（`best_index`）が
指す実際の合法手に置き換える**（`out["groups"][best_index]["rep"]` → `legal[...]`）。
探索が選んだ手そのものを禁じて**探索に選び直させる**（別の理論に依らない代替案）ではなく、
**理論の代替案をそのまま打たせる**形——T141 が既に計算している `best_index` をそのまま使えるので
新しい価格式・新しい探索の変更が要らない（新定数ゼロ）。**この選び方自体が唯一の正解ではない**
（探索に選び直させる代替案は§8 のユーザ判断に残す）。

`driver.run_game(swap=…)` を使う（T20.8.5 の ε 探索と同じ入口・**盤面には介入せず、実対局へ出す
手だけを差し替える**）。`observer` の後・`apply` の前に呼ばれるので、**候補・訪問分布・π は
探索が選んだそのまま**（介入は「打つ手」だけ）。

## 乾式運転で見つかった制約と、その解き方（T142 → T142b）

**T142 の最初の設計は落ちた**——`legal[rep]` をそのまま `apply_game_action` に渡すと、
`ValueError: 不明なアクションです: DON_BOX …` で `GameAborted`。`rust/opcg_engine/src/rules/actions.rs`
の `apply_game_action` が認識するのは**原始手**（`PLAY`／`TURN_END`／`ATTACK`／`ATTACK_CONFIRM`／
`ATTACH_DON`／`ACTIVATE_MAIN`／`RESOLVE_EFFECT_SELECTION`／`MULLIGAN`／`KEEP_HAND`）だけで、
`DON_BOX`（「ドンを k 枚付けてから殴る」箱レベルのマクロ手）は無い。T142 は置き換え先を原始手に
限って動かした（`docs/reports/2026-09-23_t18_arena.md`・介入できたのは禁じられた行の 27.7%／23.5%）。

**T142b（2026-09-23）で解いた——エンジンは変えない**。探索自身が `DON_BOX` を打つときの仕組みを
そのまま Python で真似る:

* 探索は `don_box_first_primitive(box)` の**最初の 1 原始手**だけを打ち、残りを
  `carry.commit = [{"kind": "box", "sig": move_sig(box), "left": 総数 − 1}]` に積む
  （総数 = 付けるドン k 枚 ＋ 殴るなら 1）。次の決定点で `commit_step` が残りを 1 手ずつ
  （`left ≤ 1` かつ的があれば `ATTACK`・それ以外は `ATTACH_DON`）打つ。
* `box_first_primitive`／`box_total`／`expand_replacement` がこの規則の Python 版。`make_swap` は
  置き換え先が `DON_BOX` なら最初の原始手を返し、**`out["commit"]` を書き換える**（`driver.run_game`
  は `swap` の後に `carries[name].put(out)` で `out["commit"]` を読む）。
* **元の手の `commit` は必ず捨てる**——探索が `DON_BOX` を選んでいた場合、置き換えた後にも元の箱の
  残りが次の決定点で打たれてしまう（T142 の器にも潜んでいた欠陥・置き換え先が原始手でも同じ）。
* `commit_step` は `sig` が合法手に見つからなければ**黙って畳む**ので、箱が最後まで打たれたかは
  `make_box_tracker`（`observer`）が数える（`box_completed`／`box_broken`）。

## 予告（T142b・測る前に書く）

1. **20 局とも例外なく最後まで打てる**（`GameAborted` 0 件）——置き換えは常に原始手で届く。
2. **`attach` 以外の禁じられた行はすべて介入できる**（`n_forbidden == n_exempt + n_intervened`・
   `n_no_replacement == 0`）。
3. **置き換えた箱はほぼ最後まで打たれる**（`box_broken` は `box_completed` より十分小さい）——
   壊れるのは途中で盤面が変わって同じ `sig` の合法手が消えたときだけ。

**T144 の切替（`--seq`）**: `shadow_forbid.SEQ_MODE` を `attack`／`attack_le` にすると、純付与は
「それが準備する攻撃の価格」で読まれる。読み替えられた付与の行は `played_reread` が立ち、`attach` でも
対象外にしない（読み替えられない付与は今までどおり対象外）。既定は `off`＝T142b のまま。

使い方:

    python tests/scripts/t18_arena.py --games 10 --seed-base 60000 --decks user [--json out.json]
"""

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.loop import arena as AR  # noqa: E402  （T148: pair_level_ci の正本）
from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
from opcg_sim.loop import record_gen as RG  # noqa: E402
import guard_afford as GA  # noqa: E402
import live_theory as LT  # noqa: E402
import shadow_forbid as SF  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

#: 「禁じる」対象から除く型（T141 の申し送り——系列の価値を見ない `s` の守備範囲の外）。
EXEMPT_FAMILIES = ("attach",)


#: `apply_game_action`／`apply_battle_action` が直接受ける原始手（`rust/opcg_engine/src/rules/actions.rs`）。
PRIMITIVE_ATS = ("PLAY", "TURN_END", "ATTACK", "ATTACK_CONFIRM", "ATTACH_DON", "ACTIVATE_MAIN",
                 "RESOLVE_EFFECT_SELECTION")


def box_total(mv):
    """`DON_BOX` の総原始手数（付与 k 枚＋攻撃形なら 1）。**Rust `search/decide.rs::box_total` と同じ式**。"""
    p = mv.get("payload") or {}
    k = int(float(p.get("don_k") or 0))
    return k + (1 if p.get("target_ids") else 0)


def box_first_primitive(mv):
    """`DON_BOX` → 先頭の原始手。**Rust `search/decide.rs::don_box_first_primitive` と同じ規則**
    （`k ≤ 0` かつ対象あり＝`ATTACK`・それ以外＝`ATTACH_DON`）。`DON_BOX` 以外は素通し。"""
    if mv.get("action_type") != "DON_BOX":
        return mv
    p = mv.get("payload") or {}
    k = int(float(p.get("don_k") or 0))
    uuid = p.get("uuid")
    tids = list(p.get("target_ids") or ())
    if k <= 0 and tids:
        return {"kind": "game", "action_type": "ATTACK", "payload": {"uuid": uuid, "target_ids": tids}}
    return {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": uuid}}


def expand_replacement(rep_move):
    """置き換え先（`legal` の要素）→ `(今打つ原始手, 次の decide に持ち越す残り手順)`。

    **乾式運転で見つけた壁（2026-09-23）への対処**: `DON_BOX` は探索の箱レベルのマクロ手で
    `apply_game_action` は認識しない。探索自身は「先頭の原始手を打ち、残りを
    `[{"kind": "box", "sig": move_sig, "left": total − 1}]` として持ち越す」（`decide.rs` の ⑥ と
    `commit_step`）——**同じ形をここで作る**。Rust 側は変えない（持ち越しの読み手は既存の
    `commit_step`・`move_sig` は `don_k` を含まないので次の decide の合法手に照合できる）。
    原始手でも `DON_BOX` でもない（`SETUP_BOX` 等）なら `(None, None)`＝置き換えない。"""
    at = rep_move.get("action_type")
    if at == "DON_BOX":
        total = box_total(rep_move)
        if total < 1:
            return None, None
        commit = ([{"kind": "box", "sig": RG.move_sig(rep_move), "left": total - 1}]
                  if total > 1 else [])
        return box_first_primitive(rep_move), commit
    if at in PRIMITIVE_ATS:
        # 素の手は残り手順を持たない（`PLAY`／`ACTIVATE_MAIN` の効果対話は次の decide が窓として解く）
        return rep_move, []
    return None, None


def make_swap(cards, idx2cid, theta=THETA, mu=MU, stats=None, exempt=EXEMPT_FAMILIES, seats=None):
    """`driver.run_game(swap=…)` に渡す関数を作る。`stats`（省略可）に介入の実績を積む。

    **盤面には触れない**——`out`（探索の結果・候補・π はそのまま）を読むだけで、返す `move` だけが
    実対局に出る。置き換えは `out["groups"][best_index]["rep"]` が指す `legal` の要素そのもの
    （新しい手を作らない・既存の合法手を選び直すだけ）。

    **T148**: `seats`（省略可・既定 `None`）——介入する席の名前の集合（`driver.run_game` が渡す
    `name`・`{"p1","p2"}` の部分集合）。`None` なら**両席**に介入する（T142／T142b の乾式運転と
    1 ビットも変わらない・後方互換）。**両席に同じ理論で介入すると効果が相殺する**（両方が同じ規則で
    「損な手」を避けるので、片方だけが強くなったわけではない）ので、**T18 の勝率判定には必ず 1 席だけ**
    （`seats={"p1"}` 等）を渡す——`t18_pairs`（下）が seed ごとに `p1`／`p2` を入れ替えて両方測る。"""
    if stats is None:
        stats = {}
    stats.setdefault("n_seen", 0)
    stats.setdefault("n_forbidden", 0)
    stats.setdefault("n_exempt", 0)
    stats.setdefault("n_box_replacement", 0)
    stats.setdefault("box_completed", 0)
    stats.setdefault("box_broken", 0)
    stats.setdefault("pending_box", {})
    stats.setdefault("n_intervened", 0)
    stats.setdefault("n_no_replacement", 0)
    stats.setdefault("interventions", [])

    def swap(game, name, turn, step, out, move):
        if seats is not None and name not in seats:            # **T148**: 介入しない席はそのまま打つ
            return move
        if out.get("kind") != "main":
            return move
        sc, tok, ci = LT.raw_row(game, name)
        cands = LT.raw_candidates(game, name, out)
        if not cands:
            return move
        row = SF.shadow_row(sc, tok, ci, cards, idx2cid, cands, out, move, theta, mu)
        if row is None:
            return move
        stats["n_seen"] += 1
        if not row["forbidden"]:
            return move
        stats["n_forbidden"] += 1
        # **T144**: 付与が攻撃の価格に読み替えられた行（`shadow_forbid.SEQ_MODE` が `off` 以外）は、
        # もう「系列の価値を測れない型」ではないので対象外にしない。読み替えられない付与は今までどおり外す。
        if row["played_family"] in exempt and not row.get("played_reread"):
            stats["n_exempt"] += 1
            return move
        groups = out.get("groups") or []
        legal = (out.get("stats") or {}).get("legal") or []
        bi = row["best_index"]
        if bi is None or bi >= len(groups):
            stats["n_no_replacement"] += 1
            return move
        rep = groups[bi].get("rep")
        if rep is None or rep >= len(legal) or legal[rep] is None:
            stats["n_no_replacement"] += 1
            return move
        new_move, commit = expand_replacement(legal[rep])
        if new_move is None:
            stats["n_no_replacement"] += 1
            return move
        # **探索が元の手のために積んだ残り手順を捨て、置き換え先の残り手順に差し替える**
        # （driver は `swap` の後に `out["commit"]` を持ち越す＝ここで書き換えれば次の decide に届く）。
        # 捨て忘れると、次の decide が**元の箱の続き**（元のカードへのドン付与など）を機械実行する。
        out["commit"] = commit
        if legal[rep].get("action_type") == "DON_BOX":
            stats["n_box_replacement"] += 1
            left = len(commit) and commit[0]["left"]
            if left:
                stats["pending_box"][name] = left       # 次の decide から `commit` が left 回続くはず
            else:
                stats["box_completed"] += 1             # 1 手で終わる箱
        stats["n_intervened"] += 1
        stats["interventions"].append({"turn": turn, "step": step, "who": name,
                                       "played_family": row["played_family"],
                                       "best_family": row["best_family"], "s": row["s"]})
        return new_move

    return swap


def make_box_tracker(stats):
    """`driver.run_game(observer=…)` に渡す観測。**置き換えた箱が最後まで実行されたか**を数える。

    Rust の `commit_step` は持ち越した手順が合法手に照合できないと**黙ってコミットを畳む**
    （「契約違反／消化完了」）＝異常終了しないまま箱が途中で消えうる。`swap` が積んだ
    `pending_box[seat] = left` に対して、その席の次の決定が `kind == "commit"` で `left` 回続けば
    `box_completed`、途中で別の種類の決定が来たら `box_broken`。"""
    stats.setdefault("box_completed", 0)
    stats.setdefault("box_broken", 0)
    stats.setdefault("pending_box", {})

    def observer(game, name, turn, step, out, move):
        left = stats["pending_box"].get(name, 0)
        if not left:
            return
        if out.get("kind") == "commit":
            left -= 1
            if left == 0:
                stats["box_completed"] += 1
                stats["pending_box"].pop(name, None)
            else:
                stats["pending_box"][name] = left
        else:
            stats["box_broken"] += 1
            stats["pending_box"].pop(name, None)

    return observer


def dry_run(seeds, decks_mode, sims=64, net=None, dirichlet_eps=0.25, temp_turns=4, worlds=4,
           theta=THETA, mu=MU, exempt=EXEMPT_FAMILIES, seats=None):
    """`seeds` を介入つきで打ち、局ごとの結果と介入の実績を返す（**乾式運転**・判定はしない）。

    **T148**: `seats`（省略可）を `make_swap` にそのまま渡す（`None`＝両席・後方互換）。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    E.engine()
    search = {"worlds": int(worlds)} if worlds else {}
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps, temp_turns=temp_turns,
                      prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()
    games = []
    for seed in seeds:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, decks_mode)
        stats = {}
        swap = make_swap(cards, idx2cid, theta, mu, stats, exempt, seats=seats)
        aborted = None
        try:
            res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, swap=swap,
                              observer=make_box_tracker(stats))
        except DR.GameAborted as exc:                                  # noqa: BLE001
            aborted = str(exc)
            res = {"winner": None, "turns": None, "steps": None}
        games.append({"seed": seed, "winner": res.get("winner"), "turns": res.get("turns"),
                     "steps": res.get("steps"), "aborted": aborted,
                     "n_seen": stats["n_seen"], "n_forbidden": stats["n_forbidden"],
                     "n_exempt": stats["n_exempt"], "n_box_replacement": stats["n_box_replacement"],
                     "box_completed": stats["box_completed"], "box_broken": stats["box_broken"],
                     "n_intervened": stats["n_intervened"],
                     "n_no_replacement": stats["n_no_replacement"],
                     "interventions": stats["interventions"]})
    return games


#: **T148**: どちらの原始の席名（`driver.run_game` が渡す `name`）の相手か。
_OTHER_SEAT = {"p1": "p2", "p2": "p1"}


def t18_pairs(seeds, decks_mode, sims=64, net=None, dirichlet_eps=0.25, temp_turns=4, worlds=4,
             theta=THETA, mu=MU, exempt=EXEMPT_FAMILIES):
    """**T148**: 1 つの seed につき **2 局**——理論の介入を **p1 だけ**に入れた局と **p2 だけ**に入れた局
    （`opcg_sim.loop.arena` と同じ「同 seed・席を入れ替えたペア」規約・リーダー対は入れ替わらない）。
    **両席に同じ規則で介入すると効果が相殺する**（`make_swap` の docstring）ので、**片席だけへの介入**が
    T18 の勝率判定に要る唯一の形。返り値は局ごとの記録（`dry_run` と同じ形＋`intervened`＝介入された席）。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    E.engine()
    search = {"worlds": int(worlds)} if worlds else {}
    spec = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps, temp_turns=temp_turns,
                      prune_futile=E.GEN_PRUNE_FUTILE, **search)
    db = D.load_db()
    games = []
    for seed in seeds:
        la, lb = D.leader_pair(db, seed, "random")
        for intervened in ("p1", "p2"):
            p1, p2 = D.build_pair(db, la, lb, seed, decks_mode)
            stats = {}
            swap = make_swap(cards, idx2cid, theta, mu, stats, exempt, seats={intervened})
            aborted = None
            try:
                res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, swap=swap,
                                  observer=make_box_tracker(stats))
            except DR.GameAborted as exc:                              # noqa: BLE001
                aborted = str(exc)
                res = {"winner": None, "turns": None, "steps": None}
            winner = res.get("winner")
            games.append({"seed": seed, "intervened": intervened, "winner": winner,
                         "turns": res.get("turns"), "steps": res.get("steps"), "aborted": aborted,
                         # **`score`**＝介入された席の勝率（`opcg_sim.loop.arena.pair_level_ci` の規約）。
                         # `winner is None`（打ち切り・引き分け）は `void`（母数に入れない・0.5 で埋めない）。
                         "score": (None if (aborted or winner is None)
                                  else (1.0 if winner == intervened else 0.0)),
                         "n_seen": stats["n_seen"], "n_forbidden": stats["n_forbidden"],
                         "n_exempt": stats["n_exempt"], "n_box_replacement": stats["n_box_replacement"],
                         "box_completed": stats["box_completed"], "box_broken": stats["box_broken"],
                         "n_intervened": stats["n_intervened"],
                         "n_no_replacement": stats["n_no_replacement"]})
    return games


def summarise_pairs(games):
    """**T148**: `t18_pairs` の出力 → 介入された席の**ペア水準勝率＋95% CI**（`arena.pair_level_ci`・
    同 seed の p1/p2 介入 2 局の平均を 1 ペアのスコアにする）＋配線の健全性（`dry_run` の `summarise` と同じ列）。
    **void（どちらかの局が打ち切り・引き分け）はペアごと母数から外し、件数を必ず載せる**（黙って落とさない）。"""
    if not games:
        return {"n_games": 0, "n_pairs": 0}
    by_seed = {}
    for g in games:
        by_seed.setdefault(g["seed"], {})[g["intervened"]] = g
    pair_scores, void_seeds = [], []
    for seed, by_side in by_seed.items():
        a, b = by_side.get("p1"), by_side.get("p2")
        if a is None or b is None or a["score"] is None or b["score"] is None:
            void_seeds.append(seed)
            continue
        pair_scores.append((a["score"] + b["score"]) / 2.0)
    n_aborted = sum(1 for g in games if g["aborted"])
    n_no_winner = sum(1 for g in games if not g["aborted"] and g["winner"] is None)
    n_seen = sum(g["n_seen"] for g in games)
    n_forbidden = sum(g["n_forbidden"] for g in games)
    n_intervened = sum(g["n_intervened"] for g in games)
    out = {"n_games": len(games), "n_pairs": len(by_seed), "n_void_pairs": len(void_seeds),
           "void_seeds": void_seeds, "n_scored_pairs": len(pair_scores),
           "n_aborted": n_aborted, "n_no_winner": n_no_winner,
           "n_seen": n_seen, "n_forbidden": n_forbidden, "n_intervened": n_intervened,
           "intervene_rate": round(n_intervened / n_seen, 4) if n_seen else None,
           "forbid_rate": round(n_forbidden / n_seen, 4) if n_seen else None}
    if pair_scores:
        out["ci"] = AR.pair_level_ci(pair_scores)
    else:
        out["ci"] = None
    return out


def summarise(games):
    """`dry_run` の局ごとの結果 → 配線の健全性だけを見る表（勝率は出さない・判定しない）。"""
    if not games:
        return {"n_games": 0}
    n_aborted = sum(1 for g in games if g["aborted"])
    n_no_winner = sum(1 for g in games if not g["aborted"] and g["winner"] is None)
    n_seen = sum(g["n_seen"] for g in games)
    n_forbidden = sum(g["n_forbidden"] for g in games)
    n_exempt = sum(g["n_exempt"] for g in games)
    n_box_replacement = sum(g.get("n_box_replacement", 0) for g in games)
    box_completed = sum(g.get("box_completed", 0) for g in games)
    box_broken = sum(g.get("box_broken", 0) for g in games)
    n_intervened = sum(g["n_intervened"] for g in games)
    n_no_replacement = sum(g["n_no_replacement"] for g in games)
    return {
        "n_games": len(games), "n_aborted": n_aborted, "n_no_winner": n_no_winner,
        "n_seen": n_seen, "n_forbidden": n_forbidden, "n_exempt": n_exempt,
        "n_box_replacement": n_box_replacement,
        "box_completed": box_completed, "box_broken": box_broken,
        "n_intervened": n_intervened, "n_no_replacement": n_no_replacement,
        "intervene_rate": round(n_intervened / n_seen, 4) if n_seen else None,
        "forbid_rate": round(n_forbidden / n_seen, 4) if n_seen else None,
        "turns_mean": round(sum(g["turns"] for g in games if g["turns"] is not None)
                            / max(1, len(games) - n_aborted), 2) if len(games) > n_aborted else None,
    }


def build_parser():
    ap = argparse.ArgumentParser(description="T18 の器（T142 の乾式運転／T148 の片席介入・判定は本番の T18）")
    ap.add_argument("--games", type=int, default=10,
                    help="乾式運転（両席介入）の局数。**`--pairs` と同時には使わない**")
    ap.add_argument("--pairs", type=int, default=0,
                    help="**T148**: 片席介入のペア数（seed 1 つにつき p1／p2 に介入した 2 局）。"
                         "指定すると乾式運転ではなく判定用の勝率＋95%% CI を出す")
    ap.add_argument("--seed-base", type=int, required=True,
                    help="既存の記録の seed 帯と重ならない値にする（新しい局を打つ）")
    ap.add_argument("--decks", default="synth", choices=("singleton", "synth", "synth_dig",
                                                          "synth_roles", "user"))
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--seq", default="off", choices=SF.SEQ_MODES,
                    help="**T144** 付与を攻撃の価格に読み替えるか（`off` 以外なら、読み替えた付与は対象外にしない）")
    ap.add_argument("--json", default="")
    ap.add_argument("--result", default="",
                    help="**T148**（`n_loop_ops.md` の規約）: `RESULT.json`（機械可読の納品物）を書くパス")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    SF.set_seq_mode(a.seq)                          # **T144**
    if a.pairs:
        seeds = [a.seed_base + i for i in range(a.pairs)]
        games = t18_pairs(seeds, a.decks, sims=a.sims)
        summary = summarise_pairs(games)
        out = {"games": games, "summary": summary}
        status = "done" if summary.get("ci") is not None else "no_data"
        result = {"status": status, "task": "T148", "seed_base": a.seed_base, "pairs": a.pairs,
                  "decks": a.decks, "seq": a.seq, "summary": summary}
    else:
        seeds = [a.seed_base + i for i in range(a.games)]
        games = dry_run(seeds, a.decks, sims=a.sims)
        summary = summarise(games)
        out = {"games": games, "summary": summary}
        result = {"status": "done", "task": "T142_dry_run", "seed_base": a.seed_base,
                  "games": a.games, "decks": a.decks, "seq": a.seq, "summary": summary}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    if a.result:
        with open(a.result, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
