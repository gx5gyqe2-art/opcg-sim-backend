"""固定デッキリスト対面のバランス計測（v18・新規マーク採取の対面選定用）。

背景: コーチゲートの旧7点は**単一対局・単一対面**（g3=ナミ vs センゴク・gen4期）由来で、
飽和負け局面のマークは教材にならないことが v16/v17 で確定した（正解が 1/16 の捲り筋になる）。
次のマーク採取は**互角の対面**で行う＝手の差が実際に勝率を分ける局面が多く、
マーク→レフェリー裏取り→学習の全段が機能する。本計器はユーザ提供の固定リスト
（`tests/fixtures/decks/user_decks_20260728.json`）の全ペアを、**同一エンジン（出荷既定）同士**の
CRN ペア（同 seed・席入替）で対戦させ、**デッキ相性そのもの**を測る。

`cpu_arena` の seed 自動生成デッキ（`build_realistic_deck`）とは別トピック＝リストは固定・
`run_game(deck_builder=...)` フックで正確な50枚を毎局再構築する（CardInstance は対局状態を持つ
ため使い回さない）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/matchup_balance_probe.py \
    --pairs 24 --workers 4 --out /tmp/balance.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "fixtures", "decks", "user_decks_20260728.json")

_G = {}


def deck_ids(spec):
    """{card_id: count} → 複製展開した card_id 列（pure・順序は定義順＝決定論）。"""
    out = []
    for cid, n in spec["cards"].items():
        out += [cid] * int(n)
    return out


def classify(wr):
    """対面バランスの分類（pure）。0.5 からの乖離で3段階。"""
    d = abs(wr - 0.5)
    return "互角" if d <= 0.10 else ("やや偏り" if d <= 0.20 else "一方的")


def _init(deck_json):
    from cpu_arena import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    _G["db"] = _load_db()
    _G["decks"] = json.load(open(deck_json))
    _G["eng"] = LearnedEngine()          # 出荷既定（現 gen7）を両席で共有＝腕の差ゼロ
    _G["eng2"] = LearnedEngine()         # 席別インスタンス（世界線キャッシュの分離）


def _play_pair(args):
    """1ペア＝同 seed で AB/BA の2局。デッキ A の勝ち数（0..2）を返す。"""
    a, b, seed = args
    from cpu_arena import _arena_seat
    from game_driver import run_game
    from replay_runner import build_deck_from_ids
    db, decks = _G["db"], _G["decks"]

    def builder(da, db_name):
        def build(_db, _seed):
            l1, c1 = build_deck_from_ids(db, decks[da]["leader"], deck_ids(decks[da]), "p1")
            l2, c2 = build_deck_from_ids(db, decks[db_name]["leader"], deck_ids(decks[db_name]), "p2")
            return l1, c1, l2, c2
        return build

    wins = 0.0
    for da, db_name, a_seat in ((a, b, "p1"), (b, a, "p2")):
        seats = {"p1": _arena_seat("learned", "fair", None, 1, None, None, None, 160, engine=_G["eng"]),
                 "p2": _arena_seat("learned", "fair", None, 1, None, None, None, 160, engine=_G["eng2"])}
        res = run_game(seed, db, seats=seats, deck_builder=builder(da, db_name),
                       legal_moves="skip", invariants="raise")
        if res.winner == a_seat:
            wins += 1.0
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default=FIX)
    ap.add_argument("--pairs", type=int, default=24, help="対面ごとの CRN ペア数（局数は×2）")
    ap.add_argument("--seed-base", type=int, default=940000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="対面の絞り込み（例 nami:shanks）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = list(json.load(open(args.decks)).keys())
    pairings = [tuple(p.split(":")) for p in args.only.split(",")] if args.only else \
        list(itertools.combinations(names, 2))
    t0 = time.time()
    rows = []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.decks,)) as pool:
        for pi, (a, b) in enumerate(pairings):
            seeds = [(a, b, args.seed_base + pi * 1000 + i) for i in range(args.pairs)]
            t1 = time.time()
            sc = list(pool.imap_unordered(_play_pair, seeds))
            w, g = sum(sc), 2 * len(sc)
            wr = w / g
            rows.append({"a": a, "b": b, "wins_a": w, "games": g, "wr_a": round(wr, 4),
                         "verdict": classify(wr)})
            print(f"{a} vs {b}: {w:.0f}/{g} wr({a})={wr:.3f} [{classify(wr)}] "
                  f"({time.time() - t1:.0f}s)", flush=True)
    res = {"pairs": args.pairs, "rows": rows, "sec": int(time.time() - t0)}
    print(f"BALANCE_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
