"""密ラベル自己対戦コーパスの生成（v16）: 全決定点に value 教師を付ける（間引きゼロ）。

背景（`docs/reports/` の v13/v15 と本計画の実測）: 教師密度が p3期 **102.9 点/局** から
v9 レフェリー期 **2.52 点/局** へ **40.8倍**崩壊しており、gen2→gen5 の世代交代を支えた密度レジームを
gen6 以降は一度も再現していない（gen5 の増分 ≈197,600 行に対し gen6 は 10,159 行）。
「1.6万規模で12方式試して全部無効」という結論は、この**未検証の領域**の外で得られたもの。

本スクリプトは既存の p3 生成コア（`p3_run.selfplay_shard` → `p3_loop.selfplay_game`）を
**無改修で**駆動し、gen6 の自己対戦から全決定点の (z, q_root, turns_left) を集める。
実測 7.98 s/局・**122 行/局**・q_root 有限率 1.00（符号化 v5・sims160・4worker）。

**閉ループにはしない**（生成ネットは gen6 固定・1回きり）: v7 の閉ループは 15,776局・246更新で
昇格ゼロだった＝ピーク一過性と血統過適合の機序を再演しないため。判定は外部（800局アリーナ）。

シャードごとに npz を保存する＝途中終了しても既存分をそのまま学習に使える。
出力は既存バッチスキーマ v2 互換（`ref_finetune_smoke --extra-dirs` がそのまま読める）＋
`kind="dense"`・`q_root`・`turns_left`。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/dense_selfplay_gen.py \
    --target-games 2048 --shard-games 64 --workers 4 --out /tmp/dense_self
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import multiprocessing as mp
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from pd_batch_common import pack_policy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-games", type=int, default=2048)
    ap.add_argument("--shard-games", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=160, help="生成の decide sims（serve と同等）")
    ap.add_argument("--dirichlet-eps", type=float, default=0.15, help="生成の探索ノイズ")
    ap.add_argument("--l1-mix", type=float, default=0.0,
                    help="L1-hard 席の混合比（>0 で gen6 の均衡外の局面を混ぜる。"
                         "注意: L1 席の行は q_root=NaN で z へ退化するため別 --out に分けること）")
    ap.add_argument("--enc-version", type=int, default=5, help="符号化版（gen6=5）")
    ap.add_argument("--base", default="gen6", help="生成に使う同梱世代")
    ap.add_argument("--seed-base", type=int, default=810000)
    ap.add_argument("--rotate-leaders", action="store_true", default=True,
                    help="リーダーを全プールから抽選（学習データの偏り防止・既定ON）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import p3_run as R

    vpath = os.path.join(MODELS, f"{args.base}_value.npz")
    ppath = os.path.join(MODELS, f"{args.base}_policy.npz")
    leaders = None
    if args.rotate_leaders:
        # p3_run.main と同じ作り方（リーダー card_id のプール）。デッキIDではない点に注意。
        from cpu_selfplay import _load_db
        from deckgen import all_leader_ids
        leaders = all_leader_ids(_load_db())
    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "dense_*.npz")))   # 再開時は既存分をスキップ
    print(f"生成開始: base={args.base} ev={args.enc_version} sims={args.sims} "
          f"eps={args.dirichlet_eps} l1_mix={args.l1_mix} 既存シャード={done}", flush=True)

    t_all = time.time()
    tot_rows = tot_games = 0
    with mp.Pool(args.workers, initializer=R._init_worker) as pool:
        shard = done
        while tot_games < args.target_games:
            n = min(args.shard_games, args.target_games - tot_games)
            t0 = time.time()
            vdata, pol, turns, l1g = R.selfplay_shard(
                pool, args.workers, n, args.sims, args.dirichlet_eps, vpath, ppath,
                args.seed_base + shard, ev=args.enc_version, leaders=leaders,
                l1_mix=args.l1_mix)
            if vdata is None:
                print(f"shard{shard}: 生成失敗（スキップ）", flush=True)
                shard += 1
                continue
            rows = len(vdata["value"])
            arrays = {k: vdata[k] for k in ("scalars", "field", "card_idx", "value",
                                            "q_root", "turns_left")}
            arrays["kind"] = np.array(["dense"] * rows)
            arrays.update(pack_policy(pol))
            path = os.path.join(args.out, f"dense_{shard:05d}.npz")
            np.savez_compressed(path, **arrays)
            with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                json.dump({"source": "dense_selfplay", "base": args.base, "games": n,
                           "rows": rows, "sims": args.sims, "eps": args.dirichlet_eps,
                           "l1_mix": args.l1_mix, "l1_games": l1g,
                           "enc_version": args.enc_version, "schema_version": 2,
                           "mean_turns": float(np.mean(turns)) if turns else None}, f)
            tot_rows += rows
            tot_games += n
            dt = time.time() - t0
            qfin = float(np.isfinite(vdata["q_root"]).mean())
            print(f"shard{shard}: {n}局 → {rows}行（{rows / n:.1f}行/局・q_root有限{qfin:.2f}）"
                  f" {dt:.0f}s  累計 {tot_games}局/{tot_rows}行"
                  f"（{(time.time() - t_all) / 60:.0f}分経過）", flush=True)
            shard += 1
    print(f"DENSE_GEN_RESULT {json.dumps({'games': tot_games, 'rows': tot_rows, 'out': args.out, 'min': round((time.time() - t_all) / 60, 1)})}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
