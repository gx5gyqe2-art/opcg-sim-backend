"""昇格ゲート（v6 柱①）: candidate が現行 best に段階式 arena で勝った場合のみ昇格を PASS する CLI。

`docs/reports/v5_adoption_20260715.md` §4-1。v3/v4/v5 で3回再現した「ピーク一過性」（学習が進むと
ネットが劣化し、最新＝最強でなくなる）への構造的対策。learner は最新ネットを **candidate** に留め、
本ゲートに勝った場合のみ **best**（生成・出荷の採用元）を更新する＝run をいつ止めてもベストが残る。

段階式判定（24局監視 arena は CI±0.19 で判定不能＝v5 実測。判定だけ局数を張る）:
  - stage1: 12ペア=24局（席入替CRN）。**勝ち越し（>50%）で stage2 へ**、五分以下は即棄却
    （真に 55% の candidate が五分以下に沈む確率は ~31%＝次回ゲートで再挑戦できるので許容）。
  - stage2: +38ペア=累計100局。**累計勝率 ≥ 55% で昇格**（AlphaZero evaluator と同水準。
    CI下限>0.5（61/100）まで要求すると微改善が永遠に昇格できないため、比率しきい値にする）。

判定は純関数（stage1_decision / final_decision）＝ `tests/test_promotion_gate.py` が固定する。

実行例（単体・learner からは pd_learn --promote-every 経由で呼ばれる）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/promotion_gate.py \
    --candidate /tmp/cand_v.npz,/tmp/cand_p.npz            # best 未指定＝出荷既定(gen5)
出力: 最終行 `GATE_RESULT {json}`・exit 0=昇格 / 1=棄却。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

STAGE2_FRAC = 0.55   # 累計勝率がこの比率以上で昇格


def stage1_decision(wins: float, games: int) -> str:
    """stage1（少局数の粗いふるい）: 勝ち越しなら 'continue'、五分以下なら 'reject'。"""
    return "continue" if wins * 2 > games else "reject"


def final_decision(wins: float, games: int, frac: float = STAGE2_FRAC) -> bool:
    """最終判定: 累計勝率 ≥ frac で昇格（浮動小数の境界は昇格側に丸めない）。"""
    return wins + 1e-9 >= frac * games


# --- arena 実行（multiprocessing・席入替CRNペア）------------------------------
_G = {}


def _init_pool(cand_spec, best_spec):
    """子プロセス初期化: DB とエンジン2体を1回だけロード（以後の全ペアで共有）。"""
    from cpu_arena import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine

    def eng(spec):
        if not spec:
            return LearnedEngine()   # 出荷既定（現 gen5）
        parts = spec.split(",")
        return LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)
    _G["db"] = _load_db()
    _G["cand"] = eng(cand_spec)
    _G["best"] = eng(best_spec)


def _play_pair(args):
    """1ペア＝同seedで席入替の2局。candidate の勝ち数(0..2)を返す。"""
    seed = args
    from cpu_arena import play_game
    a = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["cand"], p2_engine=_G["best"])
    b = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["best"], p2_engine=_G["cand"])
    return (1.0 if a["winner"] == "p1" else 0.0) + (1.0 if b["winner"] == "p2" else 0.0)


def run_stage(pool, seeds):
    wins = 0.0
    for w in pool.imap_unordered(_play_pair, seeds):
        wins += w
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--best", default="", help="value.npz[,policy.npz]（未指定＝出荷既定 gen5）")
    ap.add_argument("--pairs1", type=int, default=12, help="stage1 のペア数（局数はx2）")
    ap.add_argument("--pairs2", type=int, default=38, help="stage2 で追加するペア数")
    ap.add_argument("--frac", type=float, default=STAGE2_FRAC)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=21000,
                    help="ペアseedの基点（学習roundを混ぜて呼び出し側が変える＝毎回同じ開幕で測らない）")
    args = ap.parse_args()

    t0 = time.time()
    pool = mp.Pool(args.workers, initializer=_init_pool, initargs=(args.candidate, args.best))
    try:
        wins = run_stage(pool, [args.seed_base + k for k in range(args.pairs1)])
        games = args.pairs1 * 2
        d1 = stage1_decision(wins, games)
        print(f"stage1: {wins}/{games} → {d1} ({time.time()-t0:.0f}s)", flush=True)
        promoted = False
        if d1 == "continue":
            wins += run_stage(pool, [args.seed_base + args.pairs1 + k for k in range(args.pairs2)])
            games += args.pairs2 * 2
            promoted = final_decision(wins, games, args.frac)
            print(f"stage2: 累計 {wins}/{games} (要{args.frac:.2f}) ({time.time()-t0:.0f}s)", flush=True)
    finally:
        pool.terminate(); pool.join()
    result = {"promoted": promoted, "wins": wins, "games": games,
              "wr": round(wins / games, 4), "stage1": d1, "sec": round(time.time() - t0)}
    print("GATE_RESULT " + json.dumps(result), flush=True)
    return 0 if promoted else 1


if __name__ == "__main__":
    _sys.exit(main())
