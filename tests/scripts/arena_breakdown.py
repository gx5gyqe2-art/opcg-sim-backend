"""アリーナ台帳の**対面別内訳**（2026-08-16 ユーザ決定「どの対面が強いかは記録する」）。

なぜ要るか: ランダムリーダー帯（`arena_resume --leaders random`）は総合勝率しか出さないため、
「どのリーダーを握ると強い/弱いか」が判定に一切残らない。汎化の穴（特定の系統だけ打てて
いない）は総合勝率には現れにくく、平均の陰に隠れる。台帳を読み直してリーダー別・対面別に
割り直す**読み取り専用**の集計器。

入力は `arena_resume.py --out` の jsonl（複数指定＝シャードの合算）。各行は
  {"seed":…, "score":0..2, "leaders":[la, lb], "games":[wa, wb]}
で、`leaders`/`games` が無い古い台帳（2026-08-16 以前）でも seed からリーダー対を**再計算**
して集計できる（対面の割当は seed の決定論関数なので復元可能）。ただし `games` が無い行は
2局の内訳が復元できないため、score を両リーダーへ半分ずつ割り当てる（不偏だが分解能は落ちる）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/arena_breakdown.py \\
    --ledger 'tmp_arena/a_p*.jsonl' --leaders random --min-games 6
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import glob
import json

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401


def read_rows(paths):
    """台帳（複数）→ 行のリスト。seed 重複はシャード間の設計ミスなので**落として知らせる**。"""
    rows, seen = [], {}
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seed = int(r["seed"])
            if seed in seen and seen[seed] != path:
                raise SystemExit(f"seed {seed} が {seen[seed]} と {path} で重複（帯設計の誤り）")
            seen[seed] = path
            rows.append(r)
    return rows


def with_leaders(rows, db, mode):
    """`leaders` が無い行を seed から復元して補う（pure に近い変換・db は参照のみ）。"""
    from promotion_gate import _leader_pair
    out = []
    for r in rows:
        r = dict(r)
        if not r.get("leaders") or r["leaders"][0] is None:
            r["leaders"] = list(_leader_pair(db, int(r["seed"]), mode))
        out.append(r)
    return out


def per_leader(rows):
    """リーダー別の {games, wins}（候補席がそのリーダーを握った局だけを数える）。

    `games`（2局の内訳）があれば局単位で正確に割り当て、無ければ score を半分ずつ割る。
    """
    stat = collections.defaultdict(lambda: {"games": 0.0, "wins": 0.0})
    for r in rows:
        la, lb = r["leaders"]
        if la is None:
            continue
        g = r.get("games")
        if g and len(g) == 2:
            stat[la]["games"] += 1.0
            stat[la]["wins"] += float(g[0])
            stat[lb]["games"] += 1.0
            stat[lb]["wins"] += float(g[1])
        else:
            half = float(r["score"]) / 2.0
            stat[la]["games"] += 1.0
            stat[la]["wins"] += half
            stat[lb]["games"] += 1.0
            stat[lb]["wins"] += half
    return stat


def per_matchup(rows):
    """対面別（順序を無視した組）の {pairs, score}。score は候補の勝ち数（0..2）の合計。"""
    stat = collections.defaultdict(lambda: {"pairs": 0, "score": 0.0})
    for r in rows:
        la, lb = r["leaders"]
        if la is None:
            continue
        key = tuple(sorted((la, lb)))
        stat[key]["pairs"] += 1
        stat[key]["score"] += float(r["score"])
    return stat


def _name(db, cid):
    c = db.get_card(cid)
    return f"{cid} {c.name}" if c is not None else cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", nargs="+", required=True, help="jsonl（glob 可・複数=合算）")
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real"),
                    help="台帳を取ったときの対面モード（leaders 欠落行の復元に使う）")
    ap.add_argument("--min-games", type=int, default=4, help="この局数未満のリーダーは表に出さない")
    ap.add_argument("--top", type=int, default=15, help="上位/下位それぞれの表示件数")
    args = ap.parse_args()

    paths = [p for pat in args.ledger for p in sorted(glob.glob(pat))] or args.ledger
    rows = read_rows(paths)
    from cpu_arena import _load_db
    db = _load_db()
    rows = with_leaders(rows, db, args.leaders)

    from arena_parallel import _pair_level_ci
    ci = _pair_level_ci([float(r["score"]) / 2.0 for r in rows])
    print(f"台帳 {len(paths)}本・{len(rows)}ペア（{2 * len(rows)}局）")
    print(f"総合: wr {ci['win_rate']:.4f} CI95[{ci['lo']:.4f},{ci['hi']:.4f}] Elo {ci['elo']:+.1f}")

    lead = per_leader(rows)
    ranked = sorted(((k, v) for k, v in lead.items() if v["games"] >= args.min_games),
                    key=lambda kv: kv[1]["wins"] / kv[1]["games"])
    n_lo = len([1 for _k, v in lead.items() if v["games"] >= args.min_games])
    print(f"\nリーダー別（候補席がそのリーダーを握った局・{args.min_games}局以上の"
          f"{n_lo}リーダー）")
    print("  ▼ 苦手（勝率の低い順）")
    for cid, v in ranked[:args.top]:
        print(f"    {v['wins'] / v['games']:.3f}  ({v['wins']:.1f}/{v['games']:.0f})  {_name(db, cid)}")
    print("  ▲ 得意（勝率の高い順）")
    for cid, v in list(reversed(ranked))[:args.top]:
        print(f"    {v['wins'] / v['games']:.3f}  ({v['wins']:.1f}/{v['games']:.0f})  {_name(db, cid)}")

    mus = per_matchup(rows)
    multi = sorted(((k, v) for k, v in mus.items() if v["pairs"] >= 2),
                   key=lambda kv: -kv[1]["pairs"])
    print(f"\n対面別（2ペア以上を引いた組み合わせ {len(multi)}件・上位{args.top}）")
    for (a, b), v in multi[:args.top]:
        print(f"    {v['score'] / (2 * v['pairs']):.3f}  ({v['pairs']}ペア)  "
              f"{_name(db, a)} × {_name(db, b)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
