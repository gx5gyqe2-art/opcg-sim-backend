"""昇格ゲート（旧 `tests/scripts/promotion_gate.py`／`arena_gate.py` の判定部）。

`candidate` が現行 best に段階式 arena で勝った場合のみ昇格を PASS する CLI。
`docs/reports/v5_adoption_20260715.md` §4-1 の「ピーク一過性」（学習が進むとネットが劣化し
最新＝最強でなくなる）への構造的対策で、learner は最新ネットを **candidate** に留め、本ゲートに
勝った場合のみ **best**（生成・出荷の採用元）を更新する＝run をいつ止めてもベストが残る。

サブコマンド:
  promote … 段階式（stage1 で勝ち越さなければ即棄却／stage2 で累計勝率 ≥ --frac）
  band    … 帯層別の一次判定（一次スクリーン＋本判定・`ARENA_GATE_RESULT`）
  smoke   … 1 局を候補同士で回して完走を確かめる（配線の煙試験）

**判定規約は不変**（`opcg_sim.loop.arena` が正本）。アリーナを分散して回すときは
`opcg_sim.loop.arena_shard` ＋ `opcg_sim.loop.arena_merge` を使う（`docs/n_loop_ops.md`）。

実行例:
  OPCG_LOG_SILENT=1 python -m opcg_sim.loop.gate promote --candidate /tmp/cand.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import multiprocessing as mp                                           # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

from opcg_sim.loop import arena as A                                   # noqa: E402
from opcg_sim.loop import decks as D                                   # noqa: E402
from opcg_sim.loop import driver as DR                                 # noqa: E402
from opcg_sim.loop import engine as E                                  # noqa: E402

STAGE2_FRAC = 0.55   # 累計勝率がこの比率以上で昇格


def final_decision(wins: float, games: int, frac: float = STAGE2_FRAC) -> bool:
    """最終判定（pure）: 累計勝率 ≥ frac で昇格（浮動小数の境界は昇格側に丸めない）。"""
    return wins + 1e-9 >= frac * games


def anchor_decision(wins: float, games: int, frac: float = 0.5) -> bool:
    """アンカー判定（血統過適合の検出）: 固定アンカーに**非退行**（勝率 ≥ frac）。

    v6 run の実測: 対 best 連鎖で 3 段昇格した r99 が祖先 gen5 との直接対戦で 0.33 と
    負け越した（閉じた血統内の「親に勝つ特化」が祖先への強さに転移しない）。
    """
    return wins + 1e-9 >= frac * games


def screen_decision(wins: float, games: int, floor: float) -> str:
    """一次スクリーン（pure）: 勝率が floor 未満なら reject。"""
    return "reject" if (wins / max(games, 1)) < floor else "continue"


def _pool(args, cand, best):
    initargs = (cand, best, A.cand_kw_from_args(args), args.leaders, args.decks,
                args.pair_timeout, args.sims)
    return mp.get_context("spawn").Pool(args.workers, initializer=A.init_pool,
                                        initargs=initargs)


def _run(pool, seeds):
    """void（None）を 0 勝扱いにせず落とす＝母数から外す。返り値は有効ペアの勝ち数リスト。"""
    return [s for s in pool.imap_unordered(A.play_pair, seeds) if s is not None]


def cmd_promote(args) -> int:
    t0 = time.time()
    result, promoted, d1 = {}, False, "reject"
    with _pool(args, args.candidate, args.best) as pool:
        sc = _run(pool, [args.seed_base + k for k in range(args.pairs1)])
        wins, games = sum(sc), 2 * len(sc)
        d1 = A.stage1_decision(wins, games)
        print(f"stage1: {wins}/{games} → {d1} ({time.time()-t0:.0f}s)", flush=True)
        if d1 == "continue":
            sc2 = _run(pool, [args.seed_base + args.pairs1 + k for k in range(args.pairs2)])
            wins += sum(sc2)
            games += 2 * len(sc2)
            promoted = final_decision(wins, games, args.frac)
            print(f"stage2: 累計 {wins}/{games} (要{args.frac:.2f}) ({time.time()-t0:.0f}s)",
                  flush=True)
    if promoted and args.anchor is not None:
        # アンカー段: 対 best を超えた candidate だけが来る＝追加コストは昇格候補時のみ。
        with _pool(args, args.candidate, args.anchor) as pool:
            asc = _run(pool, [args.seed_base + 500 + k for k in range(args.anchor_pairs)])
        aw, ag = sum(asc), 2 * len(asc)
        a_ok = anchor_decision(aw, ag, args.anchor_frac)
        print(f"anchor: {aw}/{ag} (要{args.anchor_frac:.2f}) → "
              f"{'OK' if a_ok else 'NG=血統過適合'} ({time.time()-t0:.0f}s)", flush=True)
        result.update(anchor_wins=aw, anchor_games=ag, anchor_ok=a_ok)
        promoted = promoted and a_ok
    result.update(promoted=promoted, wins=wins, games=games,
                  wr=round(wins / max(games, 1), 4), stage1=d1,
                  sec=round(time.time() - t0))
    print("GATE_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if promoted else 1


def cmd_band(args) -> int:
    t_all = time.time()
    res = {"candidate": args.candidate, "baseline": args.best or "default(出荷既定)"}
    with _pool(args, args.candidate, args.best) as pool:
        if args.screen_pairs > 0:
            base = args.seed_base - 500000
            sc = _run(pool, list(range(base, base + args.screen_pairs)))
            swins, sgames = sum(sc), 2 * len(sc)
            verdict = screen_decision(swins, sgames, args.screen_floor)
            print(f"screen: {swins:.1f}/{sgames} wr={swins / max(sgames,1):.3f} → {verdict}",
                  flush=True)
            res["screen"] = {"wins": swins, "games": sgames,
                             "wr": round(swins / max(sgames, 1), 4), "verdict": verdict}
            if verdict == "reject":
                res.update(promoted=False, sec=int(time.time() - t_all))
                print(f"ARENA_GATE_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
                return 1
        all_scores, band_rows = [], []
        for bi, seeds in enumerate(A.plan_bands(args.pairs, args.bands, args.seed_base)):
            t0 = time.time()
            sc = _run(pool, seeds)
            all_scores += sc
            w, g = sum(sc), 2 * len(sc)
            band_rows.append({"band": bi, "wins": w, "games": g,
                              "wr": round(w / max(g, 1), 4)})
            print(f"band{bi}: {w:.1f}/{g} wr={w / max(g,1):.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
    # ペア水準スコア（0/0.5/1）へ正規化してから CI を出す（勝ち数のままだと平均が 2 倍になる）。
    ok, ci = A.final_decision([s / 2.0 for s in all_scores], args.frac)
    res.update({"promoted": ok, "games": 2 * len(all_scores), "wins": sum(all_scores),
                "wr": round(ci["win_rate"], 4),
                "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                "elo": round(ci["elo"], 1), "bands": band_rows,
                "band_spread": round(max(b["wr"] for b in band_rows)
                                     - min(b["wr"] for b in band_rows), 4),
                "sec": int(time.time() - t_all)})
    print(f"ARENA_GATE_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False)
    return 0 if ok else 1


def cmd_smoke(args) -> int:
    """1 局を完走させる（配線の煙試験）。"""
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(args.candidate or None, sims=args.sims)
    la, lb = D.leader_pair(db, args.seed_base, args.leaders)
    t0 = time.time()
    r = DR.run_game(args.seed_base, {"p1": spec, "p2": spec},
                    *D.build_pair(db, la, lb, args.seed_base, args.decks))
    out = {"winner": r["winner"], "turns": r["turns"], "steps": r["steps"],
           "leaders": [la, lb], "sec": round(time.time() - t0, 1)}
    print("SMOKE " + json.dumps(out, ensure_ascii=False), flush=True)
    return 0 if r["winner"] else 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--candidate", default="", help="候補ネットの npz（空=出荷既定）")
        p.add_argument("--best", default="", help="基準ネットの npz（空=出荷既定）")
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--sims", type=int, default=E.SERVE_SIMS)
        p.add_argument("--seed-base", type=int, default=21000)
        p.add_argument("--leaders", default="fixed",
                       choices=("fixed", "random", "real", "purple"))
        p.add_argument("--decks", default="singleton",
                       choices=("singleton", "synth", "synth_dig", "synth_roles", "user"))
        p.add_argument("--pair-timeout", type=int, default=900)
        A.add_cand_args(p)
        return p

    p = common(sub.add_parser("promote", help="段階式の昇格ゲート"))
    p.add_argument("--pairs1", type=int, default=12, help="stage1 のペア数（局数は×2）")
    p.add_argument("--pairs2", type=int, default=38, help="stage2 で追加するペア数")
    p.add_argument("--frac", type=float, default=STAGE2_FRAC)
    p.add_argument("--anchor", default=None, help="固定アンカーの npz（血統過適合の検出）")
    p.add_argument("--anchor-pairs", type=int, default=12)
    p.add_argument("--anchor-frac", type=float, default=0.5)
    p.set_defaults(func=cmd_promote)

    p = common(sub.add_parser("band", help="帯層別の一次判定"))
    p.add_argument("--pairs", type=int, default=400)
    p.add_argument("--bands", type=int, default=4)
    p.add_argument("--screen-pairs", type=int, default=48, help="0 で無効")
    p.add_argument("--screen-floor", type=float, default=0.48)
    p.add_argument("--frac", type=float, default=STAGE2_FRAC)
    p.add_argument("--out", default=None, help="JSON 保存先")
    p.set_defaults(func=cmd_band)

    p = common(sub.add_parser("smoke", help="1 局の完走確認"))
    p.set_defaults(func=cmd_smoke)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
