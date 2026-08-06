"""ターン末専用ヘッドの順位学習（v39・`plan_cf_gen.py` の対）。

**箱の階層ごとに較正を分ける**（v38 の学び）。v38 は同じ value ヘッドへ「戦闘出口の真実」と
「ターン末の真実」を同時に教えようとして失敗した（8点合計 3.06 < 本番 3.44）。真の勝率は盤面
ごとに1つに定まるので両者は論理的には矛盾しないが、実装上は**似た特徴を共有する少数の重み**へ
逆向きの勾配が掛かり、守るべき点のマージンが薄い側（m1@15 は +0.062）から折れる。

本スクリプトは出力を分ける: 胴体（Emb/W1/b1）と既存 value ヘッド（W2/b2）を**凍結**し、
ターン末専用ヘッド（既存ロジットへの残差 MLP・`ValueNet.enable_turn_head`）だけを plancf
コーパスの順位ペアで学習する。既存挙動は bit 単位で不変（学習後に検査して主張する）＝退行のしようがない。
判定は外部（coach_gate/arena_resume）。本スクリプトは順位正答率（学習前後）だけ出す。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/turn_head_finetune.py \
    --dirs /tmp/plancf_all --base gen12 --epochs 8 --lr 1e-3 --out /tmp/cand_v39
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from option_pair_finetune import MODELS, load_pairs_corpus
from ref_finetune_smoke import build_rank_pairs
from opcg_sim.src.core.cpu_learned import warm_start_value, warm_start_policy, _net_enc_version

TRUNK_PARAMS = ("Emb", "W1", "b1", "W2", "b2", "W2t", "b2t", "W_eff")


def snapshot_trunk(net):
    """凍結対象（胴体＋既存ヘッド＋補助ヘッド）の bit スナップショット。"""
    return {k: np.array(getattr(net, k), copy=True)
            for k in TRUNK_PARAMS if getattr(net, k, None) is not None}


def assert_trunk_frozen(net, snap):
    """凍結対象が 1bit も動いていないことを主張する（v39 の設計そのものの検査）。"""
    for k, v in snap.items():
        cur = getattr(net, k)
        assert cur.shape == v.shape and np.array_equal(cur, v), \
            f"凍結したはずの {k} が変化した（ターン末ヘッド学習は胴体に触れてはならない）"


def turn_pair_acc(net, child, pairs):
    """ターン末ヘッドで測ったペア順位正答率（`ref_finetune_smoke.pair_acc` のヘッド版）。"""
    if not pairs:
        return float("nan")
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    rows = np.concatenate([ia, ib])
    pred = net.predict_turn({k: child[k][rows] for k in ("scalars", "field", "card_idx")})
    m = len(pairs)
    return float((pred[:m] > pred[m:]).mean())


def rank_finetune_turn_head(net, child, pairs, epochs=8, lr=2e-4, weight=1.0, margin=0.2,
                            batch_pairs=32, rng_seed=17):
    """順位ヒンジ max(0, margin−(v_a−v_b)) を**ターン末ヘッドのみ**に流す（v39）。

    y の細工でヒンジを MSE 勾配に恒等変換する実装は `rank_finetune`（v12.1）と同じ。違いは
    `backward_turn`（ターン末ヘッドだけの勾配）を使う点＝アンカー（蒸留の錘）が要らない。v33 以降の
    アンカーは「共有重みを動かすと既存挙動が壊れる」ことへの対処であり、そもそも共有重みを
    動かさない本設計では不要になる（錘と教師の綱引き自体が消える）。"""
    rng = np.random.default_rng(rng_seed)
    for _ep in range(epochs):
        order = rng.permutation(len(pairs))
        for s in range(0, len(order), batch_pairs):
            sel = [pairs[k] for k in order[s:s + batch_pairs]]
            ia = np.array([p[0] for p in sel]); ib = np.array([p[1] for p in sel])
            rows = np.concatenate([ia, ib])
            batch = {k: child[k][rows] for k in ("scalars", "field", "card_idx")}
            _, cache = net.forward(batch)
            pred = net.turn_from_cache(cache)
            m = len(sel)
            act = (pred[:m] - pred[m:]) < margin
            if not act.any():
                continue
            B = len(pred)
            y = pred.copy()
            y[:m][act] += (B / 2.0) * weight     # 勝ち側を押し上げ
            y[m:][act] -= (B / 2.0) * weight     # 負け側を押し下げ
            net.step(net.backward_turn(cache, y), lr=lr)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="plancf コーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--base", default="gen12", help="ヘッドを載せる土台（現行本番）")
    ap.add_argument("--enc-version", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="ヘッドのみ（共有重みを一切動かさない）＝共有重み学習 2e-5 より大きく取れる")
    ap.add_argument("--turn-hidden", type=int, default=32, help="ターン末ヘッドの中間層幅")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.25, help="順位ペアに採る z 差の下限")
    ap.add_argument("--globs", default="plancf_*.npz",
                    help="読むシャード種（カンマ区切り）。既定＝ターン末教師のみ")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    child, n_files = load_pairs_corpus([d for d in args.dirs.split(",") if d],
                                       globs=tuple(g for g in args.globs.split(",") if g))
    if child is None:
        print("plancf コーパスが空（--dirs を確認）"); return 1

    vpath = os.path.join(MODELS, f"{args.base}_value.npz")
    ppath = os.path.join(MODELS, f"{args.base}_policy.npz")
    vnet = RN.ValueNet.load(vpath)
    ev0 = _net_enc_version(vnet)
    if ev0 != args.enc_version:
        vnet = warm_start_value(vnet, ev0, args.enc_version)
        print(f"温スタート拡張: v{ev0} → v{args.enc_version}")
    assert vnet.feat_dim == E.feature_dim(args.enc_version), \
        f"入力次元不一致: {vnet.feat_dim} != {E.feature_dim(args.enc_version)}"
    vnet.enable_turn_head(turn_hidden=args.turn_hidden)   # 残差ゼロ＝学習前は現行 value と同一

    pairs = build_rank_pairs(child, delta=args.delta)
    # group 単位で train/val 分割（同一決定点が両側に跨がない＝リークしない）。
    p_tr = [p for p in pairs if p[2] % 7 != 0]
    p_va = [p for p in pairs if p[2] % 7 == 0]
    a0 = turn_pair_acc(vnet, child, p_va)
    print(f"収集: {n_files}シャード・{len(child['value'])}盤面・{len(set(child['group']))}群・"
          f"順位ペア {len(pairs)}（tr {len(p_tr)}/va {len(p_va)}）", flush=True)
    print(f"ターン末順位正答(val) 学習前: {a0:.3f}（残差ゼロ＝現行 value と同一）", flush=True)

    snap = snapshot_trunk(vnet)
    rank_finetune_turn_head(vnet, child, p_tr, epochs=args.epochs, lr=args.lr,
                            margin=args.margin)
    assert_trunk_frozen(vnet, snap)
    a1 = turn_pair_acc(vnet, child, p_va)
    print(f"ターン末順位正答(val) 学習後: {a1:.3f}（学習前 {a0:.3f}）", flush=True)
    print("凍結検査: 胴体・既存ヘッド・補助ヘッドは bit 不変（既存挙動は定義上不変）", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_v = os.path.join(args.out, "value.npz")
    vnet.save(out_v)
    re = RN.ValueNet.load(out_v)
    assert re.vocab_ids == RN.ValueNet.load(vpath).vocab_ids, \
        "保存した候補の vocab_ids が base と一致しない"
    assert re.turn_head and np.array_equal(re.We2, vnet.We2), "ターン末ヘッドが保存されていない"
    out_p = os.path.join(args.out, "policy.npz")
    if ev0 != args.enc_version:
        from opcg_sim.src.learned.policy import PolicyScorer
        warm_start_policy(PolicyScorer.load(ppath), ev0, args.enc_version).save(out_p)
    else:
        shutil.copyfile(ppath, out_p)
    res = {"base": args.base, "files": n_files, "boards": int(len(child["value"])),
           "groups": int(len(set(child["group"]))), "pairs": len(pairs),
           "epochs": args.epochs, "lr": args.lr, "turn_hidden": args.turn_hidden,
           "turn_rank_acc_before": round(a0, 4), "turn_rank_acc_after": round(a1, 4),
           "candidate": f"{out_v},{out_p}"}
    print(f"TURN_HEAD_FINETUNE_RESULT {json.dumps(res)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
