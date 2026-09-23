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

## 乾式運転で見つかった制約（`DON_BOX` は置き換え先にできない）

**最初の設計は予告 1 で落ちた**——`legal[rep]` をそのまま `apply_game_action` に渡すと、
2 局とも `ValueError: 不明なアクションです: DON_BOX …` で `GameAborted`。原因を
`rust/opcg_engine/src/rules/actions.rs` の `apply_game_action` で確認した——**認識する
`action_type` は `PLAY`／`TURN_END`／`ATTACK`／`ATTACH_DON`／`ACTIVATE_MAIN`／
`RESOLVE_EFFECT_SELECTION`／`MULLIGAN`／`KEEP_HAND` だけで、`DON_BOX` は無い**。`DON_BOX` は
**探索が使う箱レベルのマクロ手**（「ドンを k 枚付けてから殴る」）で、`legal`／`groups` には
値付け・照合のために現れるが、**実際に適用できる原始手への展開（原始化）は探索の内部**にあり、
Python から「この候補を選んで」と渡して原始化させる口は今の API に無い。
**この符号化では攻撃は全部 `DON_BOX` の形で来る**（`theory_order.score_candidate` の docstring
どおり）ので、**`best_family` が `attack`／`attach` の置き換えは行わない**（`make_swap` が
`n_best_is_macro` として数え、元の手をそのまま打たせる）。T141 の実測では理論の最善が `attack`
になる割合が最も高かった（禁じられた行の 55.0%／51.5%）ので、**この制約は介入の対象を大きく
狭める**——**残せるのは最善が `play`／`effect`／`end` のときだけ**。原始化の口を Python 側に開ける
（またはメイン枠の `commit` 機構を経由する）ことは §8 のユーザ判断に残す。

## 予告（測る前に書く・上の制約を踏まえて書き直した）

1. **10 局とも例外なく最後まで打てる**（`GameAborted` が 0 件）——`swap` が返す手は
   **`best_family` が `play`／`effect`／`end` のときだけ**`legal` の実在の要素に置き換わる
   （`attack`／`attach` は置き換えない）ので、規則違反・未対応の手を注入しない。
2. **介入の頻度は T141 の「最善が `play`／`effect`／`end` だった」割合に近い値になる**——実測
   （T141・禁じられた行のうち best_family の内訳）から、`play`＋`effect`＋`end` の合計は
   実 36.7%+6.4%+1.9%=**45.0%**・合成 38.3%+5.5%+4.0%=**47.8%**（`n_forbid` に対する割合）。
   ただし**介入は打ち回しそのものを変える**ので、後続の決定点の分布は T141（無介入）の記録とは
   違う経路を辿る——**近い値になるはず**という予告であって、一致を主張しない。
3. **介入した対局のほうが手数・ターン数が動く**——**方向は予告しない**（今回の置き換え先は
   `attack` を除くので、T141 の「攻撃寄りで速くなる」という直観はそのままは効かない）。

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

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
import guard_afford as GA  # noqa: E402
import live_theory as LT  # noqa: E402
import shadow_forbid as SF  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

#: 「禁じる」対象から除く型（T141 の申し送り——系列の価値を見ない `s` の守備範囲の外）。
EXEMPT_FAMILIES = ("attach",)


def make_swap(cards, idx2cid, theta=THETA, mu=MU, stats=None, exempt=EXEMPT_FAMILIES):
    """`driver.run_game(swap=…)` に渡す関数を作る。`stats`（省略可）に介入の実績を積む。

    **盤面には触れない**——`out`（探索の結果・候補・π はそのまま）を読むだけで、返す `move` だけが
    実対局に出る。置き換えは `out["groups"][best_index]["rep"]` が指す `legal` の要素そのもの
    （新しい手を作らない・既存の合法手を選び直すだけ）。"""
    if stats is None:
        stats = {}
    stats.setdefault("n_seen", 0)
    stats.setdefault("n_forbidden", 0)
    stats.setdefault("n_exempt", 0)
    stats.setdefault("n_best_is_macro", 0)
    stats.setdefault("n_intervened", 0)
    stats.setdefault("n_no_replacement", 0)
    stats.setdefault("interventions", [])

    def swap(game, name, turn, step, out, move):
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
        if row["played_family"] in exempt:
            stats["n_exempt"] += 1
            return move
        # **配線の発見（乾式運転で見つけた・2026-09-23）**: `DON_BOX` は探索が使う **箱レベルの
        # マクロ手**（「k 枚ドンを付けてから殴る」）で、`rust/opcg_engine/src/rules/actions.rs` の
        # `apply_game_action` は `DON_BOX` を**そもそも認識しない**（`ATTACK`／`PLAY`／`ATTACH_DON`／
        # `ACTIVATE_MAIN`／`TURN_END` 等の**原始手**だけを受ける）。この符号化では**攻撃は全部
        # `DON_BOX` の形で来る**（`theory_order.score_candidate` の docstring どおり）ので、
        # `attack`（`DON_BOX` に対象あり）と `attach`（同・対象なし）は `legal[rep]` をそのまま
        # 適用できない——**箱の原始化は探索の内部**にあり、Python から任意の候補を選んで原始化させる
        # 口は今の API に無い。**`best_family` がこの 2 つなら置き換えない**（`attach` は既に
        # `exempt` の既定に入っているが、`best_family`＝置き換え先として選ばれる場合はここで別途止める）。
        if row["best_family"] in ("attack", "attach"):
            stats["n_best_is_macro"] += 1
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
        stats["n_intervened"] += 1
        stats["interventions"].append({"turn": turn, "step": step, "who": name,
                                       "played_family": row["played_family"],
                                       "best_family": row["best_family"], "s": row["s"]})
        return legal[rep]

    return swap


def dry_run(seeds, decks_mode, sims=64, net=None, dirichlet_eps=0.25, temp_turns=4, worlds=4,
           theta=THETA, mu=MU, exempt=EXEMPT_FAMILIES):
    """`seeds` を介入つきで打ち、局ごとの結果と介入の実績を返す（**乾式運転**・判定はしない）。"""
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
        swap = make_swap(cards, idx2cid, theta, mu, stats, exempt)
        aborted = None
        try:
            res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, swap=swap)
        except DR.GameAborted as exc:                                  # noqa: BLE001
            aborted = str(exc)
            res = {"winner": None, "turns": None, "steps": None}
        games.append({"seed": seed, "winner": res.get("winner"), "turns": res.get("turns"),
                     "steps": res.get("steps"), "aborted": aborted,
                     "n_seen": stats["n_seen"], "n_forbidden": stats["n_forbidden"],
                     "n_exempt": stats["n_exempt"], "n_best_is_macro": stats["n_best_is_macro"],
                     "n_intervened": stats["n_intervened"],
                     "n_no_replacement": stats["n_no_replacement"],
                     "interventions": stats["interventions"]})
    return games


def summarise(games):
    """`dry_run` の局ごとの結果 → 配線の健全性だけを見る表（勝率は出さない・判定しない）。"""
    if not games:
        return {"n_games": 0}
    n_aborted = sum(1 for g in games if g["aborted"])
    n_no_winner = sum(1 for g in games if not g["aborted"] and g["winner"] is None)
    n_seen = sum(g["n_seen"] for g in games)
    n_forbidden = sum(g["n_forbidden"] for g in games)
    n_exempt = sum(g["n_exempt"] for g in games)
    n_best_is_macro = sum(g.get("n_best_is_macro", 0) for g in games)
    n_intervened = sum(g["n_intervened"] for g in games)
    n_no_replacement = sum(g["n_no_replacement"] for g in games)
    return {
        "n_games": len(games), "n_aborted": n_aborted, "n_no_winner": n_no_winner,
        "n_seen": n_seen, "n_forbidden": n_forbidden, "n_exempt": n_exempt,
        "n_best_is_macro": n_best_is_macro,
        "n_intervened": n_intervened, "n_no_replacement": n_no_replacement,
        "intervene_rate": round(n_intervened / n_seen, 4) if n_seen else None,
        "forbid_rate": round(n_forbidden / n_seen, 4) if n_seen else None,
        "turns_mean": round(sum(g["turns"] for g in games if g["turns"] is not None)
                            / max(1, len(games) - n_aborted), 2) if len(games) > n_aborted else None,
    }


def build_parser():
    ap = argparse.ArgumentParser(description="T18 の器の乾式運転（T142・判定はしない）")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed-base", type=int, required=True,
                    help="既存の記録の seed 帯と重ならない値にする（新しい局を打つ）")
    ap.add_argument("--decks", default="synth", choices=("singleton", "synth", "synth_dig",
                                                          "synth_roles", "user"))
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    seeds = [a.seed_base + i for i in range(a.games)]
    games = dry_run(seeds, a.decks, sims=a.sims)
    out = {"games": games, "summary": summarise(games)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
