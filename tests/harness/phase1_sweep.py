"""Phase 1（切り分け実験・docs/reports/cpu_strength_roadmap_20260622.md §4）。

「深く読めば fair が cheat に勝率回復するか」を **horizon 掃引**で測り、限界が**探索か情報欠損か**を
切り分ける。探索ノブ（OPCG_HARD_HORIZON 等）は `cpu_ai` の import 時に確定するので、設定ごとに
**別プロセス**で `cpu_arena.py arena-paired` を起動して掃引する（このスクリプトはその driver）。

判定の核（option 1・確証強化）: 各 horizon は**同一 seed 群**で対戦するので、horizon 間の比較は
**ペア化**される（同一 seed＝同一配り）。深い H と浅い H の **同一 seed ペア差**を取ると配り運の共通分散が
相殺され、各点の広い周辺 CI（±100Elo 級）より遥かに鋭く「深さが効くか」を検定できる（符号検定＋ペア差 CI）。

  - ペア差が**有意に正**（深い方が勝つ seed が多い） → 深さが効く＝**探索が限界**寄り
    （→ Phase 2A 探索路線 / Phase 4 TT 前倒し）。
  - ペア差が**0 近傍/非有意** → 深く読んでも情報欠損を埋められない＝**情報/評価が限界**
    （→ Phase 2 PIMC＝決定化で隠れ情報を補う／Phase 3 評価）。

使い方（重い・実機本走は手動/定期）:
    OPCG_LOG_SILENT=1 python tests/phase1_sweep.py --pairs 40 --horizons 2,4,6 --seed 0
"""
import argparse
import math
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUMMARY = re.compile(r"win_rate=([0-9.]+)\s+Elo=([+-]?\d+)\s+\[([+-]?\d+),\s*([+-]?\d+)\]\s+half=(\d+)")
_DETAIL = re.compile(r"seed=(\d+)\s+p1won=([0-9.]+)\s+p2won=([0-9.]+)\s+pair=([0-9.]+)")


def _sign_test_p(pos: int, neg: int) -> float:
    """符号検定の両側 p 値（ties 無視・Binom(pos+neg, 0.5) の両側裾）。pos+neg==0 は 1.0。"""
    m = pos + neg
    if m == 0:
        return 1.0
    k = max(pos, neg)
    tail = sum(math.comb(m, i) for i in range(k, m + 1)) * (0.5 ** m)
    return min(1.0, 2.0 * tail)


def paired_diff(scores_a: Dict[int, float], scores_b: Dict[int, float]) -> Optional[Dict[str, float]]:
    """同一 seed の b−a ペア差（b=深い H / a=浅い H）。配り運を相殺した深さ効果の検定量。

    勝点 ∈ {0,0.5,1} なので diff ∈ {−1,−0.5,0,0.5,1}。mean_diff の正規近似 CI（n−1 分散）＋符号検定 p。
    共通 seed が無ければ None。
    """
    seeds = sorted(set(scores_a) & set(scores_b))
    diffs = [scores_b[s] - scores_a[s] for s in seeds]
    n = len(diffs)
    if n == 0:
        return None
    mean = sum(diffs) / n
    if n >= 2:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        half = 1.96 * math.sqrt(var / n)
    else:
        half = float("inf")
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return {"n": float(n), "mean_diff": mean, "ci_half": half,
            "n_pos": float(pos), "n_neg": float(neg), "n_tie": float(n - pos - neg),
            "sign_p": _sign_test_p(pos, neg)}


def run_one(horizon: int, pairs: int, seed: int, max_steps: Optional[int]
            ) -> Tuple[Tuple[float, int, int], Dict[int, float]]:
    """1 設定（global horizon=H）で fair[challenger] vs cheat[baseline] の arena-paired を別プロセス実行。

    戻り値 ((win_rate, elo, half_width), {seed: pair_score})。集計行と各 seed の detail 行を回収する。
    """
    env = dict(os.environ)
    env["OPCG_HARD_HORIZON"] = str(horizon)
    env.setdefault("OPCG_LOG_SILENT", "1")
    cmd = [sys.executable, os.path.join(_HERE, "cpu_arena.py"), "arena-paired",
           "--challenger", "hard", "--baseline", "hard",
           "--challenger-policy", "fair", "--baseline-policy", "cheat",
           "--pairs", str(pairs), "--seed", str(seed)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=_HERE)
    summary, scores = None, {}
    for line in out.stdout.splitlines():
        md = _DETAIL.search(line)
        if md:
            scores[int(md.group(1))] = float(md.group(4))
        ms = _SUMMARY.search(line)
        if ms:
            summary = (float(ms.group(1)), int(ms.group(2)), int(ms.group(5)))
    if summary is None:
        raise RuntimeError(f"arena-paired 出力を解析できず (horizon={horizon}):\n{out.stdout}\n{out.stderr}")
    return summary, scores


def sweep(horizons: List[int], pairs: int, seed: int, max_steps: Optional[int]) -> None:
    rows, score_by_h = [], {}
    for h in horizons:
        (wr, elo, half), scores = run_one(h, pairs, seed, max_steps)
        rows.append((h, wr, elo, half))
        score_by_h[h] = scores
        print(f"  horizon={h:>2}  fair-vs-cheat win_rate={wr:.3f}  Elo={elo:+d}  half=±{half}", flush=True)

    print("\n--- ペア差検定（同一 seed・配り運相殺＝確証の核） ---")
    lo, hi = horizons[0], horizons[-1]
    pd = paired_diff(score_by_h[lo], score_by_h[hi])
    significant = False
    if pd:
        print(f"  H{hi} − H{lo} ペア差: mean={pd['mean_diff']:+.3f} 勝点/局 "
              f"(95%CI ±{pd['ci_half']:.3f})  深い方が良い seed {int(pd['n_pos'])}/"
              f"{int(pd['n_pos'] + pd['n_neg'])}（tie {int(pd['n_tie'])}）  符号検定 p={pd['sign_p']:.3f}")
        significant = (pd["sign_p"] < 0.05 and pd["mean_diff"] > 0)

    print("\n--- 切り分け判定 ---")
    if significant:
        print(f"H{lo}→H{hi} のペア差が**有意に正**（p<0.05）→ 深さが効く＝**探索が限界**寄り"
              "（Phase 2A 探索路線 / Phase 4 TT 前倒しを検討）。")
    elif pd and pd["mean_diff"] > 0:
        print(f"H{lo}→H{hi} のペア差は**正の傾向だが非有意**（p={pd['sign_p']:.3f}）→ pairs/seed を増やして再走。"
              "点推定は探索路線寄りだが確証不足。")
    else:
        print("H を上げてもペア差が 0 近傍/非正 → 深く読んでも情報欠損を埋められない"
              "＝**情報/評価が限界**寄り（Phase 2 PIMC＝決定化 / Phase 3 評価へ）。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 切り分け実験: horizon 掃引＋ペア差検定")
    ap.add_argument("--horizons", default="2,4,6", help="カンマ区切りの horizon 値")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print(f"Phase 1 horizon 掃引: horizons={horizons} pairs={args.pairs} seed={args.seed}", flush=True)
    sweep(horizons, args.pairs, args.seed, args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
