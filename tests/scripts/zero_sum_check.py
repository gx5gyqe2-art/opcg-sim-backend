"""零和の制約——**同じ盤面を両席から評価して `V_me + V_opp ≈ 0` か**
（`docs/game_theory.md` §2-2・`cpu_theory_gap.md` §3.1・読み取り専用）。

二人零和では**ミニマックス値は反対称**（`V_me(s) = −V_opp(s)`）。これは学習した価値が
満たすべき**制約**であって、当てになる／ならないの問題ではない。破れていれば

- 「守った方が勝ちやすい」のような**すべての読みの土台が歪む**（片側から見た値しか見ていない）
- 探索の窓のミニマックスも壊れる（相手の価値を自分の価値の符号反転として使う場面がある）

にもかかわらず**一度も検査していなかった**（2026-09-13）。本器がそれ。

## 測り方

エンジンは盤面を**席視点**で符号化するので、同じ盤面を 2 回符号化すれば両席の値が出る:

```
h        = game.hidden_json()                   … 盤面（隠れ情報を含む「真の」状態）
enc_me   = opcg_engine.encode_state(h, "p1")    … p1 視点の符号化
enc_opp  = opcg_engine.encode_state(h, "p2")    … p2 視点
V_me     = opcg_engine.net_eval(enc_me,  "[]")["value"]
V_opp    = opcg_engine.net_eval(enc_opp, "[]")["value"]
sum      = V_me + V_opp                         … **0 であるべき量**
```

**注意（近似ではなく仕様）**: 符号化は**公平性の規約**（`game_theory.md` §4）により
相手の手札の中身を渡さない。したがって 2 つの符号化は**同じ情報集合を見ていない**
——p1 は p1 の手札を知り p2 の手札を知らない、p2 はその逆。よって

> `V_me + V_opp` は厳密な 0 ではなく、**「両者が自分の私的情報を見たときの値の和」**。

### **何が 0 であるべきで、何はそうでないか**（判定の要）

- **1 局面ごとの `sum` は 0 でなくてよい**。2 つの値は**違う情報**に条件付けた期待値だから。
  したがって `mean(|sum|)` の大きさは**欠陥の証拠にならない**（私的情報の量そのもの）。
- **平均は 0 であるべき**。`z_p1 = −z_p2` と全期待値の法則から、どんな情報集合で条件付けても
  `E[V_p1] + E[V_p2] = 0`（両者が較正されていれば）。**`mean(sum)` が 0 から有意に離れていれば、
  それは情報の非対称では説明できない＝学習の歪み**。

**標準誤差は対局でクラスタする**（同じ局の行は独立でない）。本器は
「1 局の `sum` の平均」を単位にして局間の sd から SE を出す（`sum_mean_se`・`sum_mean_ci95`）。

**`--blind` は対照として無効だった**（2026-09-13 実測）: 両席の手札を落とすと
`sum_mean` は −0.138 → **−0.665** と**悪化**した。手札が空の盤面は実プレイにほぼ存在せず
（あれば負け濃厚）、**分布外の入力を作っただけ**で私的情報の成分を分離できていない。
残してあるのは「この操作では分離できない」という記録のためで、**判定に使ってはいけない**。
（副産物として: 手札を全部落とすと V が大きく下がる＝ネットは極端な側では
「手札が少ない方が良い」とは思っていない。ただしデータから遠い外挿。）

## 出すもの

- **`sum_mean` と対局クラスタの 95% CI**（**これが判定**）・`sum_sd`
- `abs_mean`／分位点（参考＝私的情報の量の目安・判定には使わない）
- ターン帯・手番別の内訳＝**手番側を高く見る癖**が在るか
- `--blind` の有無での比較

## 読み方（事前登録）

- **`sum_mean` の 95% CI が 0 を含む**なら、零和の制約は破れていない（これまでの読みの土台は無事）。
- **CI が 0 を含まない**なら V は零和を満たしていない＝**片側から見た量（`V_guard − V_take` など）は
  すべてその偏りぶん疑う**。偏りの向きが「手番側を高く見る」なら、`V_guard − V_take` は
  守り（相手ターン）の側で系統的に低く／高く出る。
- `abs_mean` は**大きくても正常**（情報差）。ただし `--blind` で下がらないなら情報差では
  説明できていない＝符号化か学習の非対称を疑う。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/zero_sum_check.py --games 20 --seed-base 9200 \\
    --net opcg_sim/data/learned/nrel_r3.npz --out ~/zero_sum_r3.json
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

TURN_BANDS = ("T<=4", "T5-8", "T9+")
#: 符号化の手札の枠（`n_rel_feat`: 自L・相L・自場5・相場5・手札10）と手札枚数の scalars 列
SLOT_HAND = slice(12, 22)
SC_MY_HAND = 6


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def _engine():
    """masters／net／vocab を読み込んだ `opcg_engine` を返す（`loop.engine` と同じ手順）。"""
    import opcg_engine
    return opcg_engine


def to_net_enc(enc_json):
    """`encode_state` の JSON → `net_eval` が受ける形（**鍵の名前が違う**）。

    `encode_state` は `tokens`・`net_eval`（`net::encoding_from_json`）は `tok` を読む。
    黙って 0 要素として扱われる（`net: tok が 0 要素` で落ちる）ので、ここで移す。
    """
    d = json.loads(enc_json)
    if "tok" not in d and "tokens" in d:
        d["tok"] = d.pop("tokens")
    return d


def value_of(eng, enc):
    """符号化（dict）→ `value`（候補は空＝方策は要らない）。"""
    out = json.loads(eng.net_eval(json.dumps(enc, ensure_ascii=False), "[]"))
    return float(out["value"])


def blind_hand(enc, n_tok=22):
    """手札の枠を 0 にした符号化（私的情報の非対称を消す・`--blind`）。

    `tok` は平らな `[22 * S_DIM]`。手札は枠 12〜21 なので、その区間を 0 にする。
    `scalars` の自手札枚数（列 6）も 0 にする（枚数は公開情報だが、**自分の手札の質**を
    落とすなら枚数も落として両席を同じ扱いにする）。
    """
    d = dict(enc)
    tok = list(d.get("tok") or [])
    if tok and n_tok > 0:
        s = len(tok) // n_tok
        if s * n_tok == len(tok):
            for slot in range(SLOT_HAND.start, SLOT_HAND.stop):
                for k in range(s):
                    tok[slot * s + k] = 0.0
            d["tok"] = tok
    sc = list(d.get("scalars") or [])
    if len(sc) > SC_MY_HAND:
        sc[SC_MY_HAND] = 0.0
        d["scalars"] = sc
    return d


def run_games(games, seed_base, net=None, sims=None, leaders="random",
              decks="synth_roles", blind=False, max_steps=DR.DEFAULT_MAX_STEPS):
    """自己対戦を回し、各判断点で**両席の値**を取る。"""
    eng = _engine()
    db = D.load_db()
    spec = E.SeatSpec(net=net, sims=sims or E.SERVE_SIMS)
    rows = []
    stats = {"games": 0, "aborted": 0, "points": 0, "errors": 0}

    for g in range(games):
        seed = seed_base + g
        try:
            la, lb = D.leader_pair(db, seed, leaders)
            p1, p2 = D.build_pair(db, la, lb, seed, decks)
        except Exception:                                   # noqa: BLE001
            stats["aborted"] += 1
            continue

        game_rows = []

        def observe(game, name, turn, _step, _out, _move):
            stats["points"] += 1
            try:
                h = game.hidden_json()
                enc1 = to_net_enc(eng.encode_state(h, "p1"))
                enc2 = to_net_enc(eng.encode_state(h, "p2"))
                if blind:
                    enc1, enc2 = blind_hand(enc1), blind_hand(enc2)
                v1, v2 = value_of(eng, enc1), value_of(eng, enc2)
            except Exception as exc:                        # noqa: BLE001
                stats["errors"] += 1
                if stats["errors"] <= 3:
                    rows.append({"error": f"{type(exc).__name__}: {exc}", "turn": turn})
                return
            game_rows.append({"turn": turn, "band": turn_band(turn), "seed": seed,
                              "to_move": name, "v_p1": v1, "v_p2": v2, "sum": v1 + v2})

        try:
            res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2,
                              max_steps=max_steps, observer=observe)
            stats["games"] += 1
        except DR.GameAborted:
            stats["aborted"] += 1
            continue
        # **勝敗を後から貼る**（較正を測るのに要る）。`winner=None` は void＝較正には使わない。
        win = res.get("winner")
        for r in game_rows:
            z1 = None if win is None else (1.0 if win == "p1" else 0.0)
            if z1 is None:
                r["z_p1"] = r["z_p2"] = None
            else:
                r["z_p1"], r["z_p2"] = z1, 1.0 - z1
            label_seats(r)
        rows.extend(game_rows)
        if win is None:
            stats["void"] = stats.get("void", 0) + 1
    return rows, stats


def label_seats(r):
    """**手番側（on-distribution）と非手番側（off-distribution）を貼る**。

    訓練の行は「決定点の席」からしか作られないので、**非手番側の評価は構造的に
    教わっていない**（`is_my_turn=0` の行は「攻撃を受けている窓」に偏る）＝
    偏りの切り分けはこの 2 群の較正で付く。席の名前ではなく `to_move` で決まる。
    """
    act, idle = ("p1", "p2") if r["to_move"] == "p1" else ("p2", "p1")
    r["v_act"], r["v_idle"] = r[f"v_{act}"], r[f"v_{idle}"]
    r["z_act"], r["z_idle"] = r.get(f"z_{act}"), r.get(f"z_{idle}")
    return r


def block(rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return {"n": 0, "errors": len(rows)}
    s = np.array([r["sum"] for r in ok], np.float64)
    a = np.abs(s)
    out = {
        "n": len(ok), "errors": len(rows) - len(ok),
        "sum_mean": round(float(s.mean()), 5),       # **判定に使う量**（0 であるべき）
        "sum_sd": round(float(s.std()), 5),
        "abs_mean": round(float(a.mean()), 5),       # 参考（私的情報の量の目安）
        "abs_p50": round(float(np.percentile(a, 50)), 5),
        "abs_p90": round(float(np.percentile(a, 90)), 5),
        "abs_max": round(float(a.max()), 5),
        "v_p1_mean": round(float(np.mean([r["v_p1"] for r in ok])), 5),
        "v_p2_mean": round(float(np.mean([r["v_p2"] for r in ok])), 5),
    }
    out.update(cluster_se(ok))
    out.update(calibration(ok))
    return out


def calibration(rows):
    """**較正**: 手番側（on-distribution）と非手番側（off-distribution）で別に測る。

    `V` は tanh 出力（≈ `2·P(win) − 1`）なので `P̂ = (V+1)/2` と実際の勝敗を比べる。
    手番側が較正されていて非手番側だけ低いなら、`sum` の偏りは
    **「手番でない席を評価するのが分布外」**という測定側の問題（ただし探索は窓で
    相手側の値を使うので、実害はある）。両方低いなら**価値そのものが悲観的**。
    """
    out = {}
    for who in ("act", "idle"):
        sub = [r for r in rows if r.get(f"z_{who}") is not None]
        if not sub:
            out[f"calib_{who}"] = None
            continue
        p = np.array([(r[f"v_{who}"] + 1.0) / 2.0 for r in sub], np.float64)
        z = np.array([r[f"z_{who}"] for r in sub], np.float64)
        out[f"calib_{who}"] = {"n": len(sub), "p_mean": round(float(p.mean()), 4),
                               "z_mean": round(float(z.mean()), 4),
                               "gap": round(float(p.mean() - z.mean()), 4)}
    return out


def cluster_se(rows, key="seed"):
    """**対局でクラスタした** `sum_mean` の SE と 95% CI（同じ局の行は独立でない）。"""
    by = {}
    for r in rows:
        by.setdefault(r.get(key), []).append(r["sum"])
    means = np.array([float(np.mean(v)) for v in by.values()], np.float64)
    g = len(means)
    if g < 2:
        return {"games": g, "sum_mean_se": None, "sum_mean_ci95": None}
    se = float(means.std(ddof=1) / np.sqrt(g))
    m = float(means.mean())
    return {"games": g, "sum_mean_se": round(se, 5),
            "sum_mean_ci95": [round(m - 1.96 * se, 5), round(m + 1.96 * se, 5)]}


def verdict(b):
    """事前登録した読み方: **`sum_mean` の CI が 0 を含むか**（`abs_mean` は判定に使わない）。"""
    if not b or not b.get("n"):
        return None
    ci = b.get("sum_mean_ci95")
    if not ci:
        return "not_enough_games"
    return "antisymmetric" if (ci[0] <= 0.0 <= ci[1]) else "biased"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--net", default="", help="空＝出荷既定")
    ap.add_argument("--sims", type=int, default=None, help="対局を進める探索の sims（値の測定には無関係）")
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real", "purple"))
    ap.add_argument("--decks", default="synth_roles",
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "user"))
    ap.add_argument("--blind", action="store_true",
                    help="両席の手札を符号化から落とす（**対照としては無効**＝分布外の盤面を作るだけ。"
                         "2026-09-13 に実測で判明・記録のために残す）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    # ネットと語彙を読ませる（`loop.engine` の手順をそのまま使う）
    E.load_net(args.net or None)
    rows, stats = run_games(args.games, args.seed_base, args.net or None, args.sims,
                            args.leaders, args.decks, args.blind)
    out = {"params": {"games": args.games, "seed_base": args.seed_base,
                      "net": args.net or "default", "leaders": args.leaders,
                      "decks": args.decks, "blind": bool(args.blind)},
           "run": stats,
           "all": block(rows),
           "by_band": {b: block([r for r in rows if r.get("band") == b]) for b in TURN_BANDS},
           "by_to_move": {k: block([r for r in rows if r.get("to_move") == k])
                          for k in ("p1", "p2")},
           "seconds": round(time.time() - t0, 1)}
    out["verdict"] = verdict(out["all"])
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
