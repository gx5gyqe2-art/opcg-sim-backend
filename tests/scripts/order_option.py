"""攻撃順序の option value——**同じ盤面を別の順序で打ち直して差を測る**
（`docs/game_theory.md` §15・`critique_index.md` §4 の 6 の残り・読み取り専用）。

`game_theory.md` §15（逐次決定と攻撃順序）のとおり、
攻めは**順序の問題**である（同じ 2 回の攻撃でも、どちらを先に通すかで相手の守りの費用が変わる）。
その**取り逃がしの大きさ**は**記録では測れない**——打たなかった順序は棋譜に無い。
`action_coverage.py` と同じ「**エンジンに 2 通りやらせる**」型の計器がこれ。

## 測り方

自己対戦を打ち、**攻撃の候補が 2 つ以上ある判断点**で盤面を `Game.from_hidden` で複製し、
**候補ごとに「その攻撃を最初に打つ」ことを強制**してから、**残りのターンはエンジンに普通に
打たせる**（箱の残り手順 `commit` は捨てる＝差し替え後の対話は既定解決・`driver.run_game`
の `swap` と同じ規約）。自分のターンが終わった時点の盤面を**同じ席から**評価して比べる:

```
opt_gap = max_c V_end(c) − V_end(played)      … 順序で取り逃がした量（勝率の単位）
```

**なぜ席を固定するか**: `V` には席あたり −0.069 級の偏りが在る（`2026-09-13_zero_sum.md`）。
同じ席・同じ時点で**腕どうしを比べる**ので、盤面に依らない偏りは差で消える。
消えないのは「偏りが盤面に依る」場合だけで、それは本器では分けられない（限界）。

**測っているのは「最初の 1 手の選択」の価値**であって、順序の全順列ではない。
残りはエンジンが打つ＝**同じ探索を共有**するので、腕の差は「入口の選択」だけから来る。

## 出すもの

- `opt_gap`（平均・分位点）と `argmax_is_played`（エンジンが最良の入口を選んだ割合）
- `arm_spread`（腕の最大−最小）＝そもそも順序で差が付く盤面なのか
- ターン帯・`|v0|` 帯（接戦帯に集まるか＝`pimc_diag` の融合と同じ場所か）
- 相手ライフ帯（リーサル圏で取り逃がしているか＝`attack_budget` の `don_idle` と同じ場所か）

## 読み方（事前登録）

- **`arm_spread` の平均が 0.02 未満**なら、そもそも順序で勝率が動かない＝
  「攻撃順序の option value」は**量として無い**＝宿題を閉じる（手当ての対象にならない）。
- **`opt_gap` の平均が 0.02 以上**（かつ `argmax_is_played` が 0.7 未満）なら、
  順序の取り逃がしは実在する。**後悔の実測 0.033〜0.084（`measurement.md` §2）と比べる**＝
  同じ桁なら「後悔の主要な成分は攻めの順序」になり、`order_acc` の線（方策の順序）と
  **同じ手当て（価格を順序の教師にする）で両方に効く**という読みになる。
- `opt_gap` が接戦帯に集まるなら `pimc_diag` の融合（接戦帯に限局）と**同じ場所**を指す。

**注意（腕の公平性）**: 打った手も**他の腕と同じ機構で打ち直す**（`played` の腕を別扱いしない）
＝実対局の続きではなく、複製した盤面から同じ手順で回す。こうしないと
「複製の往復で失われるもの」（対話スタック・誘発待ち行列）が腕の差に混ざる。
複製は `has_interaction` が偽の点だけで行う。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/order_option.py --games 8 --seed-base 9500 \\
    --sims 96 --max-points 6 --out ~/order_option.json
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

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
from opcg_sim.loop.record_gen import first_primitive, move_sig  # noqa: E402

#: 「攻撃の入口」とみなす手の種類。`DON_BOX` は**箱**（ドン付与＋攻撃の一括手順）で、
#: `ATTACK` は素の攻撃。どちらも「どの枠で誰を殴るか」を決める手＝順序の選択そのもの。
ATTACK_KINDS = ("ATTACK", "DON_BOX")
TURN_BANDS = ("T<=4", "T5-8", "T9+")


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def v0_band(v):
    a = abs(float(v))
    return "close" if a <= 0.2 else ("mid" if a <= 0.6 else "decided")


def root_value(out):
    """根の値（`v0` が無い版は**群の訪問重み付き Q** で代用・`pimc_diag` と同じ）。"""
    v0 = out.get("v0")
    if v0 is not None:
        return float(v0)
    groups = out.get("groups") or []
    tot = sum(float(g.get("n") or 0.0) for g in groups) or 1.0
    return sum(float(g.get("n") or 0.0) * float(g.get("q") or 0.0) for g in groups) / tot


def attack_candidates(out, max_arms=4):
    """`decide` の出力から**攻撃の入口の候補**を（訪問の多い順に）返す。

    候補の単位は**群**（`groups` の `rep`）だが、腕として回すのは
    **先頭原始手**（`record_gen.first_primitive`＝箱 → ATTACH_DON か ATTACK）。
    箱をそのまま適用することはできない（`DON_BOX` は探索の中だけの合成手で、
    `apply_game_action` は受け取らない）＝**ε 探索の差し替えと同じ規約**で回す
    （残り手順 `commit` は捨て、続きは木が決め直す）。

    **先頭原始手が同じ箱は 1 つの腕に畳む**（例: 同じ枠へ 1 枚／3 枚付与する 2 つの箱は
    どちらも `ATTACH_DON(枠)` から始まる）＝畳まないと「腕の差 0」を自分で作ってしまう。
    畳んだ腕の訪問数は**和**を取る（その入口に木が配った総量）。
    """
    stats = out.get("stats") or {}
    legal = stats.get("legal") or []
    groups = out.get("groups") or []
    by_sig = {}
    for g in groups:
        rep = g.get("rep")
        if rep is None or not (0 <= rep < len(legal)):
            continue
        if legal[rep].get("action_type") not in ATTACK_KINDS:
            continue
        prim = first_primitive(legal[rep])
        key = json.dumps(move_sig(prim), ensure_ascii=False)
        cur = by_sig.get(key)
        if cur is None:
            by_sig[key] = {"sig": key, "move": prim, "n": float(g.get("n") or 0.0),
                           "q": float(g.get("q") or 0.0), "boxes": 1}
        else:
            cur["n"] += float(g.get("n") or 0.0)
            cur["boxes"] += 1
            if float(g.get("n") or 0.0) > 0 and float(g.get("q") or 0.0) > cur["q"]:
                cur["q"] = float(g.get("q") or 0.0)
    cands = sorted(by_sig.values(), key=lambda c: -c["n"])
    return cands[:max_arms]


def played_sig(out):
    """**実際に打った手の先頭原始手**の鍵（`move` は既に原始手なのでそのまま）。"""
    move = out.get("move") or {}
    if not move:
        return None
    return json.dumps(move_sig(first_primitive(move)), ensure_ascii=False)


def apply_move(game, name, move):
    if move.get("kind") == "battle":
        game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
    else:
        game.apply_game_action(name, move["action_type"],
                               json.dumps(move.get("payload") or {}, ensure_ascii=False))


def value_of(eng, game, seat):
    """`seat` の席から見た値。決着していれば ±1（勝率 1/0 の tanh 表現）。"""
    win = game.winner()
    if win is not None:
        return 1.0 if win == seat else -1.0
    enc = json.loads(eng.encode_state(game.hidden_json(), seat))
    if "tok" not in enc and "tokens" in enc:            # `encode_state` は `tokens`・`net_eval` は `tok`
        enc["tok"] = enc.pop("tokens")
    out = json.loads(eng.net_eval(json.dumps(enc, ensure_ascii=False), "[]"))
    return float(out["value"])


#: A/B の 2 本目に使う seed のずらし（続きの世界の引き直し＝**雑音の下限**を測るため）。
SEED_OFF = 1_000_003


def finish_turn(eng, hidden, seat, first_move, spec, seed, turn, max_steps=80):
    """複製した盤面で `first_move` を強制し、**自分のターンが終わるまで**打たせる。

    **強制した手の `commit`（箱の残り手順）は渡さない**＝差し替え後の対話は既定解決
    （`driver.run_game(swap=…)` と同じ規約）。**その後の手の `commit` は運ぶ**
    ——運ばないと続きが実対局より近視眼的になり、腕の値の水準が実プレイから離れる。
    戻り値は「自分のターンが終わった時点の自席の値」。
    """
    game = eng.Game.from_hidden(hidden, "p1", "p2", seed)
    apply_move(game, seat, first_move)
    carry = {}                                        # 席 → `{commit, resact_pending}`
    steps = 0
    while game.winner() is None and steps < max_steps:
        if str(game.turn_player()) != seat:
            break                                     # 自分のターンが終わった
        pending = json.loads(game.pending_json())
        if not pending:
            break
        who = pending["player_id"]
        out = json.loads(game.decide(who, spec.decide_opts(seed, turn, who, carry.get(who))))
        move = out.get("move")
        if move is None:
            break
        carry[who] = {"commit": out.get("commit") or [],
                      "resact_pending": bool(out.get("resact_pending"))}
        apply_move(game, who, move)
        steps += 1
    return value_of(eng, game, seat), steps


def probe_point(eng, hidden, seat, cands, p_sig, spec, seed, turn):
    """1 判断点の全腕を **2 通りの続き**（A／B）で回して 1 行にする。

    **`max − played` をそのまま読んではいけない**（勝者の呪い）。腕の値には続きの探索の
    雑音が乗るので、**腕が全部同じ強さでも `max` は `played` より上に出る**。そこで:

    - `opt_gap_raw` = A で選び A で測る＝**上限**（雑音のぶんだけ上振れする）
    - `opt_gap_honest` = **A で選び B で測る**＝選択の上振れを消した量（これが判定）
    - `gap_random` = A の腕の平均 − `played`＝「入口を無作為に選んだ場合との差」
      （**負なら木の選択は無作為より良い**・雑音に対して不偏）
    - `noise` = 同じ腕の |A − B| の平均＝**雑音の下限**。`arm_spread` がこれと同程度なら
      「順序で差が付いている」ように見えているのは雑音（判定を出さない）
    """
    arms = []
    for c in cands:
        try:
            va, sa = finish_turn(eng, hidden, seat, c["move"], spec, seed, turn)
            vb, _sb = finish_turn(eng, hidden, seat, c["move"], spec, seed + SEED_OFF, turn)
        except Exception as exc:                        # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}", "turn": turn}
        arms.append({"sig": c["sig"], "v_a": round(va, 5), "v_b": round(vb, 5),
                     "n": c["n"], "q": round(c["q"], 5), "boxes": c["boxes"], "steps": sa})
    if len(arms) < 2:
        return None
    a = np.array([x["v_a"] for x in arms], np.float64)
    b = np.array([x["v_b"] for x in arms], np.float64)
    best_a = int(np.argmax(a))
    played = next((k for k, x in enumerate(arms) if x["sig"] == p_sig), None)
    row = {"turn": turn, "band": turn_band(turn), "seat": seat, "arms": len(arms),
           "arm_spread": round(float(a.max() - a.min()), 5),
           "noise": round(float(np.mean(np.abs(a - b))), 5),
           "v_best": round(float(a.max()), 5),
           "played_in_arms": played is not None}
    if played is not None:
        row["v_played"] = arms[played]["v_a"]
        row["opt_gap_raw"] = round(float(a.max() - a[played]), 5)
        row["opt_gap_honest"] = round(float(b[best_a] - b[played]), 5)
        row["gap_random"] = round(float(a.mean() - a[played]), 5)
        row["argmax_is_played"] = bool(best_a == played)
    return row


def run_games(games, seed_base, net=None, sims=96, leaders="random", decks="synth_roles",
              max_points=6, max_arms=4, max_steps=DR.DEFAULT_MAX_STEPS):
    eng = E.engine()
    db = D.load_db()
    spec = E.SeatSpec(net=net, sims=sims)
    rows = []
    stats = {"games": 0, "aborted": 0, "points_seen": 0, "points_probed": 0,
             "skipped_interaction": 0, "errors": 0}

    for g in range(games):
        seed = seed_base + g
        try:
            la, lb = D.leader_pair(db, seed, leaders)
            p1, p2 = D.build_pair(db, la, lb, seed, decks)
        except Exception:                               # noqa: BLE001
            stats["aborted"] += 1
            continue
        picked = {"n": 0}

        def observe(game, name, turn, _step, out, _move):
            if picked["n"] >= max_points:
                return
            cands = attack_candidates(out, max_arms)
            if len(cands) < 2:
                return
            stats["points_seen"] += 1
            if game.has_interaction:
                stats["skipped_interaction"] += 1
                return
            row = probe_point(eng, game.hidden_json(), name, cands,
                              played_sig(out), spec, seed, turn)
            if row is None:
                return
            if "error" in row:
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    rows.append(dict(row, seed=seed))
                return
            row["v0"] = round(root_value(out), 5)
            row["v0_band"] = v0_band(row["v0"])
            row["seed"] = seed
            rows.append(row)
            picked["n"] += 1
            stats["points_probed"] += 1

        try:
            DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2,
                        max_steps=max_steps, observer=observe)
            stats["games"] += 1
        except DR.GameAborted:
            stats["aborted"] += 1
    return rows, stats


def cluster_se(rows, field, key="seed"):
    """**対局でクラスタした** SE と 95% CI（同じ局の判断点は独立でない）。

    1 局から最大 `--max-points` 個の点を採るので、点の数で割ると SE を過小に見る
    （零和の測定で同じ罠を踏んだ・`2026-09-13_zero_sum.md`）。
    """
    by = {}
    for r in rows:
        by.setdefault(r.get(key), []).append(float(r[field]))
    means = np.array([float(np.mean(v)) for v in by.values()], np.float64)
    g = len(means)
    if g < 2:
        return {"games": g, "se": None, "ci95": None}
    se = float(means.std(ddof=1) / np.sqrt(g))
    m = float(means.mean())
    return {"games": g, "se": round(se, 5),
            "ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


def block(rows):
    ok = [r for r in rows if "error" not in r and r.get("opt_gap_honest") is not None]
    if not ok:
        return {"n": 0}
    honest = np.array([r["opt_gap_honest"] for r in ok], np.float64)
    raw = np.array([r["opt_gap_raw"] for r in ok], np.float64)
    rand = np.array([r["gap_random"] for r in ok], np.float64)
    spread = np.array([r["arm_spread"] for r in ok], np.float64)
    noise = np.array([r["noise"] for r in ok], np.float64)
    cl = cluster_se(ok, "opt_gap_honest")
    return {
        "n": len(ok),
        "opt_gap_honest": round(float(honest.mean()), 5),        # **判定に使う量**
        "games": cl["games"],
        "opt_gap_honest_se": cl["se"],                           # **対局でクラスタした** SE
        "opt_gap_honest_ci95": cl["ci95"],
        "opt_gap_raw": round(float(raw.mean()), 5),              # 上限（勝者の呪い込み）
        "gap_random": round(float(rand.mean()), 5),             # 負＝木は無作為より良い
        "arm_spread_mean": round(float(spread.mean()), 5),
        "noise_mean": round(float(noise.mean()), 5),
        "spread_over_noise": round(float(spread.mean() / noise.mean()), 3)
        if noise.mean() > 0 else None,
        "argmax_is_played": round(float(np.mean([r["argmax_is_played"] for r in ok])), 4),
        "arms_mean": round(float(np.mean([r["arms"] for r in ok])), 3),
    }


def by_key(rows, key):
    out = {}
    for r in rows:
        if "error" in r:
            continue
        out.setdefault(r.get(key), []).append(r)
    return {k: block(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def verdict(b):
    """事前登録した読み方。**雑音に対する比を先に見る**（`spread_over_noise`）。"""
    if not b or not b.get("n"):
        return None
    if b.get("spread_over_noise") is not None and b["spread_over_noise"] < 1.5:
        return "noise_floor"                      # 腕の差が雑音と同程度＝判定を出さない
    if b["arm_spread_mean"] < 0.02:
        return "no_option_value"
    ci = b.get("opt_gap_honest_ci95")
    if ci and ci[0] > 0.02:
        return "ordering_leaks"
    if ci and ci[1] < 0.02:
        return "no_option_value"
    return "partly"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--net", default="", help="空＝出荷既定")
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real", "purple"))
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--max-points", type=int, default=6,
                    help="1 局あたりに複製して回す判断点の上限（費用の上限）")
    ap.add_argument("--max-arms", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    rows, stats = run_games(a.games, a.seed_base, net=a.net or None, sims=a.sims,
                            leaders=a.leaders, decks=a.decks,
                            max_points=a.max_points, max_arms=a.max_arms)
    all_b = block(rows)
    ok = [r for r in rows if "error" not in r]
    # 攻撃の入口が 2 つ以上あるのに**攻撃でない手を打った**割合（腕に打った手が無い点）。
    stats["played_in_arms_rate"] = (round(float(np.mean([r["played_in_arms"] for r in ok])), 4)
                                    if ok else None)
    res = {"stats": stats, "all": all_b, "verdict": verdict(all_b),
           "by_turn": by_key(rows, "band"), "by_v0": by_key(rows, "v0_band"),
           "seconds": round(time.time() - t0, 1),
           "args": {k: v for k, v in vars(a).items() if k != "out"}}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            json.dump({"summary": res, "rows": rows}, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
