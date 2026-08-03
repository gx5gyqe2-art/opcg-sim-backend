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
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")


def _make_fixed_matchup_game(decks_json, a, b):
    """固定リスト対面用の OPCGGame を作る（v19・対面特化の密ラベル生成）。

    `p3_run._init_worker` が置く `_W["game"]` を差し替える＝`p3_loop.selfplay_game` の
    `game.new_game(db, seed, leaders)` だけがフックで、生成コアは無改修のまま。
    seed 偶奇で席を入れ替える（両デッキを両席から学習分布に入れる）。乱数規約は
    素の `new_game` と同一（`random.seed(seed)` → デッキ構築（固定＝乱数不使用）→
    `start_game()` のシャッフルが global random を消費）。"""
    import json as _json
    from matchup_balance_probe import deck_ids
    from opcg_game import OPCGGame as _G
    specs = _json.load(open(decks_json))
    pair = [(specs[a]["leader"], deck_ids(specs[a])),
            (specs[b]["leader"], deck_ids(specs[b]))]

    class _FixedMatchupGame(_G):
        def new_game(self, db, seed, leaders=None):
            import random as _r
            from replay_runner import build_deck_from_ids
            from opcg_sim.src.core.gamestate import GameManager, Player
            _r.seed(seed)
            (la, ca), (lb, cb) = (pair if seed % 2 == 0 else (pair[1], pair[0]))
            l1, c1 = build_deck_from_ids(db, la, ca, "p1")
            l2, c2 = build_deck_from_ids(db, lb, cb, "p2")
            m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
            m.start_game()
            return m

    from opcg_sim.src.learned.config import GEN_PRUNE_FUTILE
    return _FixedMatchupGame(prune_futile=GEN_PRUNE_FUTILE)


def _init_worker_fixed(decks_json, a, b):
    import p3_run as R
    R._init_worker()
    R._W["game"] = _make_fixed_matchup_game(decks_json, a, b)


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
    ap.add_argument("--mark-frac", type=float, default=0.0,
                    help="マーク局面シード比（v5 §4-2・`mark_seeds.load_mark_boards`）。>0 で各局を"
                         "この確率で人間マークの失敗盤面から開始する＝観測された失敗モードを"
                         "in-distribution 化する（0=従来の turn1 開始のみ）")
    ap.add_argument("--def-force-eps", type=float, default=0.0,
                    help="v26 ε強制防御: 防御窓で確率εのとき訪問分布を無視し『守る手』から一様抽選"
                         "（0=無効。分布の新規性を作る主レバー・`p3_loop._forced_defense_index`）")
    ap.add_argument("--enc-version", type=int, default=5, help="符号化版（gen6=5）")
    ap.add_argument("--base", default="gen6", help="生成に使う同梱世代")
    ap.add_argument("--seed-base", type=int, default=810000)
    ap.add_argument("--rotate-leaders", action="store_true", default=True,
                    help="リーダーを全プールから抽選（学習データの偏り防止・既定ON）")
    ap.add_argument("--matchup", default=None,
                    help="固定リスト対面 'a:b'（user_decks の名前・v19。指定時は rotate 無効・"
                         "seed 偶奇で席入替）")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import p3_run as R

    if os.path.sep in args.base:
        # パス接頭辞（例 /tmp/gen8v6）: 温スタート拡張ネット等、同梱外の生成ネットを使う。
        # 行の符号化版（--enc-version）はネット入力にも使われるため、ネット側の版と一致が必要。
        vpath, ppath = args.base + "_value.npz", args.base + "_policy.npz"
    else:
        vpath = os.path.join(MODELS, f"{args.base}_value.npz")
        ppath = os.path.join(MODELS, f"{args.base}_policy.npz")
    leaders = None
    init, initargs = R._init_worker, ()
    if args.matchup:
        a, b = args.matchup.split(":")
        init, initargs = _init_worker_fixed, (args.decks_json, a, b)
    elif args.rotate_leaders:
        # p3_run.main と同じ作り方（リーダー card_id のプール）。デッキIDではない点に注意。
        from cpu_selfplay import _load_db
        from deckgen import all_leader_ids
        leaders = all_leader_ids(_load_db())
    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "dense_*.npz")))   # 再開時は既存分をスキップ
    print(f"生成開始: base={args.base} ev={args.enc_version} sims={args.sims} "
          f"eps={args.dirichlet_eps} l1_mix={args.l1_mix} matchup={args.matchup} 既存シャード={done}", flush=True)

    t_all = time.time()
    tot_rows = tot_games = 0
    with mp.Pool(args.workers, initializer=init, initargs=initargs) as pool:
        shard = done
        while tot_games < args.target_games:
            n = min(args.shard_games, args.target_games - tot_games)
            t0 = time.time()
            vdata, pol, turns, l1g = R.selfplay_shard(
                pool, args.workers, n, args.sims, args.dirichlet_eps, vpath, ppath,
                args.seed_base + shard, ev=args.enc_version, leaders=leaders,
                l1_mix=args.l1_mix, mark_frac=args.mark_frac,
                def_force_eps=args.def_force_eps)
            if vdata is None:
                print(f"shard{shard}: 生成失敗（スキップ）", flush=True)
                shard += 1
                continue
            rows = len(vdata["value"])
            arrays = {k: vdata[k] for k in ("scalars", "field", "card_idx", "value",
                                            "q_root", "turns_left")}
            # v26 監視: 手札カウンター保有の平均（v6 特徴の先頭＝scalars[SCALARS_V5]）。
            # ε強制防御が効いていれば「カウンターを切った状態」が増え、この値が下がる
            # ＝分布の新規性が実際に生まれているかを走りながら見る唯一の安価な指標。
            hcm = opt = None
            if args.enc_version >= 6:
                import rl_encoder as _E
                hcm = round(float(vdata["scalars"][:, _E.SCALARS_V5].mean()), 5)
            if args.enc_version >= 7:
                # v29 監視: 登場時オプションが「生きている」行の割合（v7 特徴の先頭）。
                # 0 に張り付くなら新特徴は分散を持たず学習の取っ手にならない＝走行中に検知する。
                col = vdata["scalars"][:, _E.SCALARS_V6]
                opt = [round(float((col > 0).mean()), 4), round(float(col.mean()), 5)]
            arrays["kind"] = np.array(["dense"] * rows)
            arrays.update(pack_policy(pol))
            path = os.path.join(args.out, f"dense_{shard:05d}.npz")
            np.savez_compressed(path, **arrays)
            with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                json.dump({"source": "dense_selfplay", "base": args.base, "games": n,
                           "rows": rows, "sims": args.sims, "eps": args.dirichlet_eps,
                           "def_force_eps": args.def_force_eps, "mark_frac": args.mark_frac,
                           "hand_counter_mean": hcm, "live_option": opt,
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
