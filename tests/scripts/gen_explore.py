"""§7.73 WP `gen-explore` の段 1 の計測: 後半の低確率の温度で打った生成を、同じ seed の素の生成と対にして数える。

使い方::

    python tests/scripts/gen_explore.py --base <素の生成の part …> --run p10=<part …> --run p20=<part …> --out res.json

数えるもの（局の単位は seed で対にする）:
  - 1 局あたりの「最善でない手」の数（後半の温度で引いて、貪欲に選ぶ手と違った行＝列 explore が 2）
  - その手の根の評価の損（貪欲に選ぶ手の Q − 打った手の Q・手番の視点）
  - 勝敗の荒れ: 同じ seed で勝者が入れ替わった局の割合・先手（p1）の勝率の差・局の長さ
  - (ii)（最善でない手を打った行までの価値の教師を根の評価に替える）で教師が替わる行の割合と、替わる幅
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from opcg_sim.loop import record_gen as G  # noqa: E402


def load_parts(paths):
    """part のディレクトリ（glob 可）→ seed ごとの局の列（dict）。"""
    games = {}
    for pat in paths:
        for d in sorted(glob.glob(os.path.expanduser(pat))):
            for f in sorted(glob.glob(os.path.join(d, "n_record_*.npz"))):
                z = np.load(f)
                cols = {k: z[k] for k in z.files}
                ptr = np.concatenate([[0], np.cumsum(cols["pol_len"].astype(np.int64))])
                for s in np.unique(cols["seed"]):
                    m = np.where(cols["seed"] == s)[0]
                    g = {k: cols[k][m] for k in ("z", "who", "turn", "step", "kind", "pol_len",
                                                 "pol_chosen", "pol_v0", "forced")}
                    if "explore" in cols:
                        g["explore"] = cols["explore"][m]
                    if "z_game" in cols:
                        g["z_game"] = cols["z_game"][m]
                    # 行ごとの候補（n, q）
                    g["cand"] = [(cols["pol_n"][ptr[i]:ptr[i + 1]], cols["pol_q"][ptr[i]:ptr[i + 1]]) for i in m]
                    games[int(s)] = g
    return games


def greedy_index(n, q, q_min_frac=0.125, sims=1280):
    """生成の選び方（`q_min_n`）の近似: 訪問が下限以上の候補の中で Q 最大・無ければ訪問最多。"""
    floor = q_min_frac * sims
    ok = np.where(n >= floor)[0]
    if len(ok) == 0:
        return int(np.argmax(n))
    return int(ok[np.argmax(q[ok])])


def summarize(games, base=None):
    zg = lambda g: g.get("z_game", g["z"])  # noqa: E731
    out = {"games": len(games)}
    late, early, loss, turns, p1w = [], [], [], [], []
    rows = changed = 0
    gap = []
    for s, g in games.items():
        ex = g.get("explore", np.zeros(len(g["z"]), np.int8))
        late.append(int((ex == 2).sum()))
        early.append(int((ex == 1).sum()))
        turns.append(int(g["turn"].max()))
        first = np.argmin(g["step"])
        p1w.append(1.0 if zg(g)[first] * (1 if g["who"][first] == 0 else -1) > 0 else 0.0)
        for i in np.where(ex == 2)[0]:
            n, q = g["cand"][i]
            c = int(g["pol_chosen"][i])
            if c >= 0 and len(n):
                loss.append(float(q[greedy_index(n, q)] - q[c]))
        teach = G.explore_z(zg(g), g["who"], g["step"], g["pol_v0"].astype(np.float32), ex)
        d = np.abs(teach - zg(g))
        rows += len(d)
        changed += int((d > 0).sum())
        gap += d[d > 0].tolist()
    out.update({
        "late_per_game": float(np.mean(late)), "late_games_with_any": float(np.mean(np.array(late) > 0)),
        "early_per_game": float(np.mean(early)),
        "late_q_loss_mean": float(np.mean(loss)) if loss else None,
        "late_q_loss_p90": float(np.percentile(loss, 90)) if loss else None,
        "late_q_loss_share_over_0.1": float(np.mean(np.array(loss) > 0.1)) if loss else None,
        "turns_mean": float(np.mean(turns)), "p1_win": float(np.mean(p1w)),
        "rows": rows, "root_teacher_changed_share": changed / max(rows, 1),
        "root_teacher_gap_mean": float(np.mean(gap)) if gap else 0.0,
    })
    if base is not None:
        common = sorted(set(games) & set(base))
        flips = 0
        for s in common:
            a, b = games[s], base[s]
            wa = zg(a)[np.argmin(a["step"])] * (1 if a["who"][np.argmin(a["step"])] == 0 else -1)
            wb = zg(b)[np.argmin(b["step"])] * (1 if b["who"][np.argmin(b["step"])] == 0 else -1)
            flips += int(np.sign(wa) != np.sign(wb))
        out["paired"] = len(common)
        out["winner_flip_share"] = flips / max(len(common), 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--run", action="append", default=[], help="名前=part（glob・カンマ区切り可）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = load_parts(a.base)
    res = {"base": summarize(base)}
    for r in a.run:
        name, pats = r.split("=", 1)
        g = load_parts(pats.split(","))
        base_c = {s: base[s] for s in g if s in base}
        res[name] = summarize(g, base)
        res[name + "_base_same_seeds"] = summarize(base_c)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
