"""評価 v2（L1コア）の係数を SPSA で自己対戦最適化する足場（dev専用・段階導入§9）。

SPSA = Simultaneous Perturbation Stochastic Approximation。**全係数を同時にランダム符号で±摂動**し、
各で自己対戦の勝率を測って**2評価だけで全次元の勾配を推定**→その方向へ更新、を繰り返す。座標降下
（1個ずつ）がノイズ＞効果で詰むのに対し、次元が増えても評価2回で進めるのでノイズの大きい多次元の
評価係数チューニングに向く。

- 対象パラメータ＝`cpu_eval_v2.V2_*`（≈10個）。各を初期値まわりの**乗数 m_i（初期1.0）**で表し、
  m を最適化（スケール差を吸収・`m∈[0.2,5]` にクリップ）。
- 目的 f(m) ＝ v2(m) を ON にした側 vs ベースライン（評価OFF＝成熟J値評価）の**勝率**（席交互）。
- CRN: 1 イテレーション内の f(m+) と f(m−) は**同一ゲームseed集合**で測り、勾配の分散を落とす。

注意: モジュール定数を直接書き換える（単一スレッド arena＝相手と干渉しない）。プロセス内のみ・出荷物не変更。
本スクリプトはあくまで**探索の足場**＝短時間スモークと、長時間バックグラウンド最適化の双方に使う。

実行例:
    OPCG_LOG_SILENT=1 python tests/eval_v2_spsa.py --iters 12 --games 8
    OPCG_LOG_SILENT=1 python tests/eval_v2_spsa.py --iters 1 --games 2   # スモーク
"""
import argparse
import json
import os
import random
import sys

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_eval_v2 as V2
from cpu_arena import _load_db, play_game, win_rate, elo_delta

# 最適化する係数（名前と探索クリップ範囲＝初期値×[lo,hi]）。整数・全体スケールは対象外。
PARAMS = [
    "V2_W_LIFE_PRECIOUS", "V2_W_LIFE_HIGH", "V2_W_DECK",
    "V2_W_DEV", "V2_W_CTR", "V2_W_BODY", "V2_W_TELE",
    "V2_KAPPA", "V2_LAMBDA", "V2_W_DON",
]
M_LO, M_HI = 0.2, 5.0          # 乗数のクリップ（暴走防止）
# チェックポイント（再起動耐性）: best 更新ごとに best 係数を書き出す（落ちても進捗が残る）。
CKPT = os.environ.get("OPCG_SPSA_CKPT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_v2_spsa_best.json"))


def _save_ckpt(init, best_m, best_wr, k):
    try:
        data = {"iter": k, "best_winrate": best_wr,
                "multipliers": {n: m for n, m in zip(PARAMS, best_m)},
                "absolute": {n: init[n] * m for n, m in zip(PARAMS, best_m)}}
        with open(CKPT, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def set_params(init, m):
    """乗数 m を係数へ反映（V2_* を init*m で上書き）。"""
    for name, mi in zip(PARAMS, m):
        setattr(V2, name, init[name] * _clip(mi, M_LO, M_HI))


def winrate_vs_base(db, games, seed0, max_steps):
    """v2 ON vs 評価OFF（base）の勝率を**対照ペア（antithetic）＋コア並列**で測る（分散低減＋高速化）。

    `arena_parallel.paired_play` で各 seed を両席1回ずつ（同 game-seed・`separate_policy_rng`）並列実行。
    現在の評価係数（set_params で設定済みの θ）をワーカーへ `coeffs` で渡す＝その θ で並列対戦。
    games は総局数（pairs=games//2）。v2 側勝率を返す（SPSA の目的関数＝低ノイズ・高速）。
    """
    from arena_parallel import paired_play
    pairs = max(1, games // 2)
    coeffs = {name: float(getattr(V2, name)) for name in PARAMS}   # 親の θ をワーカーへ
    res = paired_play(pairs, seed0=seed0, max_steps=max_steps, coeffs=coeffs)
    return res["win_rate"]


def spsa(iters, games, max_steps, seed0):
    db = _load_db()
    init = {name: float(getattr(V2, name)) for name in PARAMS}
    n = len(PARAMS)
    m = [1.0] * n
    # SPSA ゲイン系列（標準的な定石値）。a/c は勝率スケール(0..1)に合わせ控えめ。
    a, c, A, alpha, gamma = 0.20, 0.15, max(1.0, iters / 10.0), 0.602, 0.101

    set_params(init, m)
    base_wr = winrate_vs_base(db, games, seed0, max_steps)
    best_m, best_wr = list(m), base_wr
    print(f"[init] winrate={base_wr:.3f} (Elo {elo_delta(base_wr):+.0f})")

    for k in range(1, iters + 1):
        ck = c / (k ** gamma)
        ak = a / ((k + A) ** alpha)
        # ランダム符号ベクトル（Rademacher）。CRN のため + と − は同一 game-seed 集合で測る。
        delta = [1.0 if random.random() < 0.5 else -1.0 for _ in range(n)]
        gseed = seed0 + 1000 * k                       # イテレーションごとに別のゲーム集合
        mp = [m[i] + ck * delta[i] for i in range(n)]
        mm = [m[i] - ck * delta[i] for i in range(n)]
        set_params(init, mp); fp = winrate_vs_base(db, games, gseed, max_steps)
        set_params(init, mm); fm = winrate_vs_base(db, games, gseed, max_steps)
        # 勝率を最大化（+方向へ更新）。ghat_i = (fp-fm)/(2 ck delta_i)。
        m = [_clip(m[i] + ak * (fp - fm) / (2 * ck * delta[i]), M_LO, M_HI) for i in range(n)]
        # この m を別 seed で評価（過適合監視）。
        set_params(init, m); wr = winrate_vs_base(db, games, gseed + 7, max_steps)
        tag = ""
        if wr > best_wr:
            best_wr, best_m = wr, list(m); tag = "  <- best"
            _save_ckpt(init, best_m, best_wr, k)     # 再起動耐性: best をその都度保存
        print(f"[iter {k:2d}] f+={fp:.3f} f-={fm:.3f} -> winrate={wr:.3f} "
              f"(Elo {elo_delta(wr):+.0f}){tag}", flush=True)

    set_params(init, best_m)
    print(f"\n=== best: winrate={best_wr:.3f} (Elo {elo_delta(best_wr):+.0f}) ===")
    print("best 係数（初期比 multiplier）:")
    for name, mi in zip(PARAMS, best_m):
        print(f"  {name}: x{mi:.3f}  -> {init[name]*mi:.4f}")
    return best_m, best_wr


def main(argv=None):
    ap = argparse.ArgumentParser(description="評価 v2 係数の SPSA 自己対戦最適化")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--games", type=int, default=8, help="1 評価あたりの自己対戦局数（多いほど低ノイズ・遅い）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args(argv)
    spsa(args.iters, args.games, args.max_steps, args.seed0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
