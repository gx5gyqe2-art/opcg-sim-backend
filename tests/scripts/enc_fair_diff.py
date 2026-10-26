"""enc_fair_diff: 探索の根の入力を `reveal=peek` と既定（`fair`）で並べ、特徴ごとの差を測る（WP `enc-audit`）。

`docs/search_efficiency_plan.md` §7.17。1 回きりの計測（テストは付けない）。

各判断点（両席・出荷の設定で打った実対局の decide の直前）で、decide が世界 `i` に使うのと同じ
作り（`Game.encode_world`＝`reveal` の混ぜ直し＋`Pcg32SearchRng(search_seed + i)`）で盤面を引き、
その盤面の符号化（scalars 127・field 10×8・card_idx 24・tokens 22×22・rel_om/rel_oo・extra 33）を
特徴ごとに比べる。比べる組:

- `peek_fair`  … 同じ世界番号 i の peek と fair（**差が出る特徴＝公平化で意味が変わった特徴**）
- `truth_peek` … 本物の盤面と peek（peek が本物から何を変えていたか）
- `truth_fair` … 本物の盤面と fair（隠れた情報を読んでいた特徴＝fair で本物からずれる）
- `fair_fair`  … fair の世界 0 と世界 1（**世界ごとに揺れる**特徴）

あわせて各世界の符号化をネット（既定 r13d・`opcg_engine.net_eval`）に通し、**価値の出力**
（`value`・手番側から見た勝ちの見込み）を判断点ごとに記録する（`values`: 本物・peek・fair の各世界と
その局の勝敗）＝入力の差が出力をどれだけ動かすか・本物の入力のほうが勝敗をよく当てるか（漏れの効き）。

出力 JSON: `{"cfg", "points", "features": {名前: {組: [非零の数, 絶対差の和, 最大, 値の個数]}}, "values"}`。
`--merge` で並列に回した複数の出力を足し、表（非零の割合・平均絶対差）を出す。

実行例（4 本並列・各 5 局）:
  OPCG_LOG_SILENT=1 python tests/scripts/enc_fair_diff.py --seed0 2999900 --games 5 --out a.json
  python tests/scripts/enc_fair_diff.py --merge a.json b.json c.json d.json --table out.md
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                          # noqa: E402
import json                                                              # noqa: E402
import sys                                                               # noqa: E402

import numpy as np                                                       # noqa: E402

import os as _os, sys as _sys                                            # noqa: E401,E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                        # noqa: E402,F401

from opcg_sim.learned import n_rel_feat as F                             # noqa: E402
from opcg_sim.learned.leader_feat import DIMS as LEADER_DIMS             # noqa: E402
from opcg_sim.loop import decks as D                                     # noqa: E402
from opcg_sim.loop import driver as DR                                   # noqa: E402
from opcg_sim.loop import engine as E                                    # noqa: E402

PAIRS = ("peek_fair", "truth_peek", "truth_fair", "fair_fair")
EPS = 1e-6

# scalars 127 列の名前（`rust/opcg_engine/src/encode/scalars.rs` の対応表・`encoder.encode`）。
_SLOTS12 = ["own_L", "opp_L"] + [f"own_f{i}" for i in range(5)] + [f"opp_f{i}" for i in range(5)]
SCALAR_NAMES = (
    ["life_me", "life_opp", "don_active_me", "don_rested_me", "don_active_opp", "don_rested_opp",
     "hand_n_me", "hand_n_opp", "field_n_me", "field_n_opp", "turn_count", "is_my_turn",
     "leader_power_me", "leader_power_opp", "leader_don_me", "leader_don_opp",
     "deck_n_me", "deck_n_opp", "trash_n_me", "trash_n_opp", "koed_me", "koed_opp"]
    + [f"turn1_used.{s}" for s in _SLOTS12] + [f"sick.{s}" for s in _SLOTS12]
    + ["deck_agg.counter_total", "deck_agg.counter_density", "deck_agg.blockers",
       "deck_agg.events", "deck_agg.highcost_char"]
    + ["opp_field.total_power", "opp_field.high", "opp_field.blockers", "playable_chars"]
    + ["hand_agg.counter_total", "hand_agg.counter_cards", "hand_agg.max_counter",
       "hand_agg.blockers", "hand_agg.events"]
    + ["onplay.n_live", "onplay.keep_live", "onplay.n_dead"]
    + ["own_field.total_power", "own_field.high", "own_field.blockers"]
    + ["don_deck_me", "don_deck_opp", "deck_apex.max_power", "deck_apex.max_cost"]
    + [f"leader_me.{d}" for d in LEADER_DIMS] + [f"leader_opp.{d}" for d in LEADER_DIMS]
    + [f"extra.{c}" for c in F.EXTRA_COLS]
)
# `EXTRA_COLS` には同名の列が 2 本ある（役割 `big` の `opp_pool_big`〔列 113〕と脅威の `opp_pool_big`〔列 115〕）。
# 名前で束ねると片方が消えるので、2 本目以降に `#k` を付けて分ける。
_seen = {}
for _k, _n in enumerate(SCALAR_NAMES):
    _seen[_n] = _seen.get(_n, 0) + 1
    if _seen[_n] > 1:
        SCALAR_NAMES[_k] = f"{_n}#{_seen[_n]}"
assert len(SCALAR_NAMES) == 127 and len(set(SCALAR_NAMES)) == 127, len(set(SCALAR_NAMES))
FIELD_COLS = ("cost", "power", "is_rest", "attached_don", "kw_blocker", "kw_rush", "kw_double", "kw_banish")
CARD_IDX_NAMES = (["own_L", "opp_L"] + [f"own_f{i}" for i in range(5)] + [f"opp_f{i}" for i in range(5)]
                  + [f"hand{i}" for i in range(10)] + ["own_stage", "opp_stage"])
TOK_ZONE = tuple(F._ZONE[i] for i in range(F.N_TOK))


def flatten(enc):
    """符号化 dict → {特徴名: np.ndarray（値の並び）}。

    tokens は「ゾーン × S の列」でまとめる（同じゾーンの枠をまたいだ値の並び）。rel は R の列ごと。
    scalars の末尾 33 列は extra と同じもの（`extra` 鍵は数えない）。"""
    out = {}
    sc = np.asarray(enc["scalars"], np.float64)
    for k, n in enumerate(SCALAR_NAMES):
        out[f"sc.{n}"] = sc[k:k + 1]
    fl = np.asarray(enc["field"], np.float64).reshape(10, 8)
    for k, c in enumerate(FIELD_COLS):
        out[f"field.own.{c}"] = fl[:5, k]
        out[f"field.opp.{c}"] = fl[5:, k]
    ci = np.asarray(enc["card_idx"], np.float64)
    for k, n in enumerate(CARD_IDX_NAMES):
        out[f"card_idx.{n}"] = ci[k:k + 1]
    tok = np.asarray(enc["tokens"], np.float64).reshape(F.N_TOK, F.S_DIM)
    for z in ("own_leader", "own_field", "hand", "opp_leader", "opp_field"):
        rows = [i for i in range(F.N_TOK) if TOK_ZONE[i] == z]
        for k, c in enumerate(F.S_COLS):
            out[f"tok.{z}.{c}"] = tok[rows, k]
    for key in ("rel_om", "rel_oo"):
        r = enc.get(key)
        if r is None:
            continue
        r = np.asarray(r, np.float64).reshape(-1, F.R_DIM)
        for k, c in enumerate(F.R_COLS):
            out[f"{key}.{c}"] = r[:, k]
    return out


class Acc:
    def __init__(self):
        self.points = 0
        self.f = {}

    def add(self, pair, a, b):
        for name, va in a.items():
            d = np.abs(va - b[name])
            nz = int((d > EPS).sum())
            ent = self.f.setdefault(name, {p: [0, 0.0, 0.0, 0] for p in PAIRS})[pair]
            ent[0] += nz
            ent[1] += float(d.sum())
            ent[2] = max(ent[2], float(d.max()) if d.size else 0.0)
            ent[3] += int(d.size)


def _value(eng, enc_text):
    """符号化 JSON（`Game.encode` の鍵）→ ネットの価値（`net_eval` は tokens を `tok` で受ける）。"""
    enc = json.loads(enc_text)
    enc["tok"] = enc.pop("tokens")
    return float(json.loads(eng.net_eval(json.dumps(enc), "[]"))["value"])


def run(seed0, games, decks, net, sims, worlds, budget, max_steps):
    eng = E.engine()
    spec = E.SeatSpec(net, sims=sims, budget=budget, worlds=worlds)
    eng.load_net(spec.net)          # net_eval の既定ネット（decide と同じ npz）
    db = D.load_db()
    acc = Acc()
    ctx = {"seed": None, "vals": []}
    values = []

    def obs(game, name, turn, step, out, move):
        s = E.search_seed(ctx["seed"], turn, name)
        raw_t = game.encode_world(name, "truth", 0)
        raw_f = [game.encode_world(name, "fair", s + i) for i in range(worlds)]
        raw_p = [game.encode_world(name, "peek", s + i) for i in range(worlds)]
        ctx["vals"].append({"who": name, "turn": int(turn), "know": json.loads(game.knowledge_json(name)),
                            "truth": _value(eng, raw_t),
                            "peek": [_value(eng, t) for t in raw_p],
                            "fair": [_value(eng, t) for t in raw_f]})
        truth = flatten(json.loads(raw_t))
        fair = [flatten(json.loads(t)) for t in raw_f]
        peek = [flatten(json.loads(t)) for t in raw_p]
        for i in range(worlds):
            acc.add("peek_fair", peek[i], fair[i])
            acc.add("truth_peek", truth, peek[i])
            acc.add("truth_fair", truth, fair[i])
        if worlds >= 2:
            acc.add("fair_fair", fair[0], fair[1])
        acc.points += 1

    res = []
    for seed in range(seed0, seed0 + games):
        ctx["seed"] = seed
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, decks)
        ctx["vals"] = []
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=max_steps, observer=obs)
        for v in ctx["vals"]:
            w = r.get("winner")
            v["z"] = None if w is None else (1.0 if w == v["who"] else -1.0)
            v["seed"] = seed
        values += ctx["vals"]
        res.append({"seed": seed, "winner": r.get("winner"), "turns": r.get("turns"), "steps": r.get("steps")})
        print(f"[game] seed={seed} winner={r.get('winner')} steps={r.get('steps')} points={acc.points}",
              file=sys.stderr, flush=True)
    return {"cfg": {"seed0": seed0, "games": games, "decks": decks, "net": net or "default",
                    "sims": sims, "worlds": worlds, "budget": budget},
            "games": res, "points": acc.points, "features": acc.f, "values": values}


def merge(paths):
    docs = []
    for p in paths:
        with open(p) as f:
            docs.append(json.load(f))
    out = {"cfg": [d["cfg"] for d in docs], "games": sum((d["games"] for d in docs), []),
           "points": sum(d["points"] for d in docs), "features": {},
           "values": sum((d.get("values", []) for d in docs), [])}
    for d in docs:
        for name, ent in d["features"].items():
            dst = out["features"].setdefault(name, {p: [0, 0.0, 0.0, 0] for p in PAIRS})
            for p in PAIRS:
                a, b = dst[p], ent[p]
                dst[p] = [a[0] + b[0], a[1] + b[1], max(a[2], b[2]), a[3] + b[3]]
    return out


def table(doc):
    """非零の割合（値の個数あたり）と平均絶対差。peek_fair の非零割合の大きい順。"""
    rows = []
    for name, ent in doc["features"].items():
        r = {"name": name}
        for p in PAIRS:
            nz, s, mx, n = ent[p]
            r[p] = (nz / n if n else 0.0, s / n if n else 0.0, mx)
        rows.append(r)
    rows.sort(key=lambda r: (-r["peek_fair"][0], -r["truth_fair"][0], -r["fair_fair"][0], r["name"]))
    lines = ["| 特徴 | peek↔fair 非零 | 平均|差| | 最大 | 本物↔peek 非零 | 本物↔fair 非零 | 平均|差| | fair 世界0↔1 非零 | 平均|差| |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pf, tp, tf, ff = r["peek_fair"], r["truth_peek"], r["truth_fair"], r["fair_fair"]
        if max(pf[0], tp[0], tf[0], ff[0]) == 0.0:
            continue
        lines.append(f"| `{r['name']}` | {pf[0]:.3f} | {pf[1]:.4f} | {pf[2]:.3f} | {tp[0]:.3f} | "
                     f"{tf[0]:.3f} | {tf[1]:.4f} | {ff[0]:.3f} | {ff[1]:.4f} |")
    zero = sorted(r["name"] for r in rows if max(r[p][0] for p in PAIRS) == 0.0)
    return "\n".join(lines), zero


def value_summary(vals):
    """価値の出力の差と、勝敗（z）への当たり（二乗誤差）。"""
    if not vals:
        return ""
    def arr(f):
        return np.array([f(v) for v in vals], np.float64)
    pf = np.concatenate([np.abs(np.array(v["peek"]) - np.array(v["fair"])) for v in vals])
    tf = np.concatenate([np.abs(v["truth"] - np.array(v["fair"])) for v in vals])
    ff = arr(lambda v: abs(v["fair"][0] - v["fair"][1]) if len(v["fair"]) > 1 else 0.0)
    tm = arr(lambda v: abs(v["truth"] - float(np.mean(v["fair"]))))
    lines = ["| 組 | 平均|差| | 中位 | 90% 点 | 最大 | 差 >0.05 の割合 |", "|---|---|---|---|---|---|"]
    for lab, d in (("peek↔fair（同じ世界番号）", pf), ("本物↔fair（各世界）", tf),
                   ("fair 世界0↔1", ff), ("本物↔fair 2 本の平均", tm)):
        lines.append(f"| {lab} | {d.mean():.4f} | {np.median(d):.4f} | {np.quantile(d, 0.9):.4f} | "
                     f"{d.max():.4f} | {(d > 0.05).mean():.3f} |")
    zv = [v for v in vals if v.get("z") is not None]
    if zv:
        z = np.array([v["z"] for v in zv])
        def mse(f):
            return float(np.mean((np.array([f(v) for v in zv]) - z) ** 2))
        lines += ["", f"勝敗への二乗誤差（判断点 {len(zv)}・小さいほど当たる）: "
                  f"本物 {mse(lambda v: v['truth']):.4f}・peek（世界平均） {mse(lambda v: float(np.mean(v['peek']))):.4f}・"
                  f"fair（世界平均） {mse(lambda v: float(np.mean(v['fair']))):.4f}・"
                  f"fair（世界 0） {mse(lambda v: v['fair'][0]):.4f}"]
    return "\n".join(lines)


def knowledge_summary(vals):
    """知っている札（`Game.knowledge_json`）の頻度＝v15 で「分かっている分」を足す案の効きの上限。"""
    ks = [v["know"] for v in vals if v.get("know")]
    if not ks:
        return ""
    n = len(ks)
    lines = ["| 領域 | 知っている札が 1 枚以上ある判断点の割合 | 平均枚数 | 領域の平均枚数 |", "|---|---|---|---|"]
    for key, tot in (("opp_hand", "opp_hand_n"), ("opp_deck", "opp_deck_n"), ("opp_life", "opp_life_n"),
                     ("my_deck", "my_deck_n"), ("my_life", "my_life_n"), ("deck_top", "my_deck_n")):
        c = np.array([k[key] for k in ks], np.float64)
        t = np.array([k[tot] for k in ks], np.float64)
        lines.append(f"| `{key}` | {(c > 0).mean():.3f} | {c.mean():.3f} | {t.mean():.2f} |")
    return f"判断点 {n}\n\n" + "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed0", type=int, default=2999900)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--net", default=None)
    ap.add_argument("--sims", type=int, default=E.PLAY_SIMS)
    ap.add_argument("--worlds", type=int, default=E.PLAY_WORLDS)
    ap.add_argument("--budget", type=int, default=E.PLAY_BUDGET)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", nargs="+", default=None)
    ap.add_argument("--table", default=None, help="--merge の結果の表（Markdown）の書き出し先")
    a = ap.parse_args(argv)
    if a.merge:
        doc = merge(a.merge)
        md, zero = table(doc)
        text = (f"判断点 {doc['points']}・局 {len(doc['games'])}\n\n{md}\n\n"
                f"差が 1 度も出なかった特徴（{len(zero)}）: " + ", ".join(f"`{z}`" for z in zero) + "\n\n"
                + value_summary(doc.get("values") or []) + "\n\n"
                + knowledge_summary(doc.get("values") or []) + "\n")
        if a.table:
            with open(a.table, "w") as f:
                f.write(text)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(doc, f, ensure_ascii=False)
        print(text)
        return 0
    doc = run(a.seed0, a.games, a.decks, a.net, a.sims, a.worlds, a.budget, a.max_steps)
    text = json.dumps(doc, ensure_ascii=False)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
