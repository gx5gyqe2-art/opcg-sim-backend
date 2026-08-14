"""出口専用ヘッドの順位学習（v39 ターン末＝`plan_cf_gen.py` の対 / v41 戦闘出口＝`defense_cf_gen.py` の対）。

**箱の階層ごとに較正を分ける**（v38/v40 の学び）。
v38 は同じ value ヘッドへ「戦闘出口の真実」と「ターン末の真実」を同時に教えようとして失敗した
（8点合計 3.06 < 本番 3.44）。真の勝率は盤面ごとに1つに定まるので両者は論理的には矛盾しないが、
実装上は**似た特徴を共有する少数の重み**へ逆向きの勾配が掛かり、守るべき点のマージンが薄い側
（m1@15 は +0.062）から折れる。
v40 は「本体 value を防御CFで直接動かす」腕で、コーチゲートは 8.00/8.00 満点になったのに
アリーナが 0.447 CI[0.409,0.485]（284ペア／568局）＝有意な退行になった。全面順位学習は
**盤面評価そのもの**を全域で動かすため、8点で得た分より他所で失う分が大きい。

本スクリプトは出力を分ける: 胴体（Emb/W1/b1）と既存 value ヘッド（W2/b2）を**凍結**し、
出口専用ヘッド（既存ロジットへの残差 MLP・`ValueNet.enable_exit_head`）だけをコーパスの
順位ペアで学習する。既存挙動は bit 単位で不変（学習後に検査して主張する）＝退行のしようがない。
`--head` が「どの箱の出口か」を選び、それが (a) 読むシャード種 (b) 有効化するヘッド
(c) serve で誰がその値を見るか、の3つを同時に決める:
  turn   … plancf_*.npz / ターン末ヘッド / プラン読み出しとターン静止の出口
  battle … defcf_*.npz  / 戦闘出口ヘッド / 戦闘箱の枝順位づけ（防御窓の読み出し・木の箱化）
判定は外部（coach_gate/arena_resume）。本スクリプトは順位正答率（学習前後）だけ出す。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/exit_head_finetune.py \
    --head battle --dirs /tmp/defcf_all --base gen12 --epochs 8 --lr 1e-3 --out /tmp/cand_v41
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

# 凍結対象＝胴体＋既存 value ヘッド＋補助ヘッド＋**学習対象でない方の出口ヘッド**。
# 後者を入れておくのが要点で、ターン末較正を持つネットへ戦闘出口ヘッドを足しても
# ターン末側が 1bit も動かないことを機械的に主張できる（階層ごとの独立性の検査）。
TRUNK_PARAMS = ("Emb", "W1", "b1", "W2", "b2", "W2t", "b2t", "W_eff")


def _other_head_params(kind):
    return tuple(p for k, spec in RN.EXIT_HEADS.items() if k != kind for p in spec[1:])


def snapshot_trunk(net, kind=None):
    """凍結対象（胴体＋既存ヘッド＋補助ヘッド＋他階層の出口ヘッド）の bit スナップショット。"""
    keys = TRUNK_PARAMS + (_other_head_params(kind) if kind else ())
    return {k: np.array(getattr(net, k), copy=True)
            for k in keys if getattr(net, k, None) is not None}


def assert_trunk_frozen(net, snap):
    """凍結対象が 1bit も動いていないことを主張する（v39 の設計そのものの検査）。"""
    for k, v in snap.items():
        cur = getattr(net, k)
        assert cur.shape == v.shape and np.array_equal(cur, v), \
            f"凍結したはずの {k} が変化した（ターン末ヘッド学習は胴体に触れてはならない）"


def exit_pair_acc(net, child, pairs, kind="turn"):
    """出口ヘッドで測ったペア順位正答率（`ref_finetune_smoke.pair_acc` のヘッド版）。"""
    if not pairs:
        return float("nan")
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    rows = np.concatenate([ia, ib])
    pred = net.predict_exit({k: child[k][rows] for k in ("scalars", "field", "card_idx")}, kind)
    m = len(pairs)
    return float((pred[:m] > pred[m:]).mean())


def rank_finetune_exit_head(net, child, pairs, kind="turn", epochs=8, lr=2e-4, weight=1.0,
                            margin=0.2, batch_pairs=32, rng_seed=17):
    """順位ヒンジ max(0, margin−(v_a−v_b)) を**その出口ヘッドのみ**に流す（v39/v41）。

    y の細工でヒンジを MSE 勾配に恒等変換する実装は `rank_finetune`（v12.1）と同じ。違いは
    `backward_exit`（そのヘッドだけの勾配）を使う点＝アンカー（蒸留の錘）が要らない。v33 以降の
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
            pred = net.exit_from_cache(cache, kind)
            m = len(sel)
            act = (pred[:m] - pred[m:]) < margin
            if not act.any():
                continue
            B = len(pred)
            y = pred.copy()
            y[:m][act] += (B / 2.0) * weight     # 勝ち側を押し上げ
            y[m:][act] -= (B / 2.0) * weight     # 負け側を押し下げ
            net.step(net.backward_exit(cache, y, kind), lr=lr)
    return net


def center_exit_head(net, ref, kind):
    """出口ヘッドの残差ロジットを基準盤面集合 `ref` の上で**平均ゼロ**へ寄せる（v42）。

    なぜ要るか（実測 2026-08-07）: 注入教師で学習したヘッドは、順位を直すのと同時に
    **ほぼ一律のバイアス**を持ちうる（管轄限定注入の腕で平均 +0.457・標準偏差 0.176）。
    箱の中の argmax は全枝に同じ定数を足しても不変なのでバイアスは無害だが、**木の中では
    有害**——戦闘箱ノードの葉見積もり `leaf_v` はこのヘッドの値なので、一律に底上げされると
    非戦闘の葉（本体 value）と**別スケールで比較**され、探索が無差別に戦闘へ入る方向へ歪む。

    tanh 前のロジットへ定数を足す操作は単調なので、**同一の箱の中の順位は厳密に保存される**
    （返り値の shift を検算に使える）。基準は「ヘッドが実際に評価する盤面」＝戦闘出口の
    コーパスを渡すこと（注入コーパスだけで中心化すると3点の偏った分布に合わせてしまう）。"""
    _, _, _, W2n, b2n = RN.EXIT_HEADS[kind]
    _, cache = net.forward(ref)
    _, h = net._exit_hidden_act(cache[8], kind)
    resid = (h @ getattr(net, W2n) + getattr(net, b2n))[:, 0]
    shift = float(resid.mean())
    setattr(net, b2n, getattr(net, b2n) - shift)
    return shift


# 既存呼び出し（v39 のテスト・レポート）向けの薄い別名。
def turn_pair_acc(net, child, pairs):
    return exit_pair_acc(net, child, pairs, "turn")


def rank_finetune_turn_head(net, child, pairs, **kw):
    return rank_finetune_exit_head(net, child, pairs, kind="turn", **kw)


# 箱の階層 → 既定のシャード種（教師と物差しの1対1対応をここに閉じ込める）。
HEAD_GLOBS = {"turn": "plancf_*.npz", "battle": "defcf_*.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="turn", choices=sorted(HEAD_GLOBS),
                    help="どの箱の出口を較正するか（turn=ターン末 / battle=戦闘出口）")
    ap.add_argument("--dirs", required=True, help="出口CFコーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--base", default="gen12", help="ヘッドを載せる土台（現行本番）")
    ap.add_argument("--base-path", default="",
                    help="土台 value.npz[,policy.npz] をパス直指定（MODELS 外の候補ネット＝"
                         "fixture 保全品などに載せる時。--base より優先・2026-08-14）")
    ap.add_argument("--enc-version", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="ヘッドのみ（共有重みを一切動かさない）＝共有重み学習 2e-5 より大きく取れる")
    ap.add_argument("--head-hidden", type=int, default=32, help="出口ヘッドの中間層幅")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.25, help="順位ペアに採る z 差の下限")
    ap.add_argument("--globs", default=None,
                    help="読むシャード種（カンマ区切り）。既定＝--head に対応する教師のみ")
    ap.add_argument("--center-dirs", default="",
                    help="ヘッドの残差ロジットを平均ゼロへ寄せる基準コーパス（カンマ区切り）。"
                         "ヘッドが実際に評価する盤面＝戦闘出口のコーパスを渡す。空＝中心化しない")
    ap.add_argument("--center-glob", default=None,
                    help="基準コーパスのシャード種（既定＝--head に対応する教師）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    globs = args.globs or HEAD_GLOBS[args.head]
    child, n_files = load_pairs_corpus([d for d in args.dirs.split(",") if d],
                                       globs=tuple(g for g in globs.split(",") if g))
    if child is None:
        print(f"コーパスが空（--dirs / --globs={globs} を確認）"); return 1

    if args.base_path:
        _bp = args.base_path.split(",")
        vpath = _bp[0]
        ppath = _bp[1] if len(_bp) > 1 else os.path.join(MODELS, f"{args.base}_policy.npz")
    else:
        vpath = os.path.join(MODELS, f"{args.base}_value.npz")
        ppath = os.path.join(MODELS, f"{args.base}_policy.npz")
    vnet = RN.ValueNet.load(vpath)
    ev0 = _net_enc_version(vnet)
    if ev0 != args.enc_version:
        vnet = warm_start_value(vnet, ev0, args.enc_version)
        print(f"温スタート拡張: v{ev0} → v{args.enc_version}")
    assert vnet.feat_dim == E.feature_dim(args.enc_version), \
        f"入力次元不一致: {vnet.feat_dim} != {E.feature_dim(args.enc_version)}"
    vnet.enable_exit_head(args.head, hidden=args.head_hidden)  # 残差ゼロ＝学習前は現行 value と同一

    pairs = build_rank_pairs(child, delta=args.delta)
    # group 単位で train/val 分割（同一決定点が両側に跨がない＝リークしない）。
    p_tr = [p for p in pairs if p[2] % 7 != 0]
    p_va = [p for p in pairs if p[2] % 7 == 0]
    a0 = exit_pair_acc(vnet, child, p_va, args.head)
    print(f"収集: {n_files}シャード・{len(child['value'])}盤面・{len(set(child['group']))}群・"
          f"順位ペア {len(pairs)}（tr {len(p_tr)}/va {len(p_va)}）", flush=True)
    print(f"{args.head} 出口順位正答(val) 学習前: {a0:.3f}（残差ゼロ＝現行 value と同一）", flush=True)

    snap = snapshot_trunk(vnet, args.head)
    rank_finetune_exit_head(vnet, child, p_tr, kind=args.head, epochs=args.epochs, lr=args.lr,
                            margin=args.margin)
    assert_trunk_frozen(vnet, snap)
    if args.center_dirs:
        ref, n_ref = load_pairs_corpus([d for d in args.center_dirs.split(",") if d],
                                       globs=tuple(g for g in (args.center_glob or
                                                               HEAD_GLOBS[args.head]).split(",")))
        if ref is None:
            print("中心化の基準コーパスが空（--center-dirs を確認）"); return 1
        acc_pre = exit_pair_acc(vnet, child, p_va, args.head)
        shift = center_exit_head(vnet, ref, args.head)
        acc_post = exit_pair_acc(vnet, child, p_va, args.head)
        assert acc_pre == acc_post, "中心化で順位が動いた（単調変換のはずで、あり得ない）"
        print(f"中心化: 残差ロジットを {shift:+.4f} 平行移動"
              f"（基準 {n_ref}シャード {len(ref['value'])}盤面・順位は厳密に不変）", flush=True)
    a1 = exit_pair_acc(vnet, child, p_va, args.head)
    print(f"{args.head} 出口順位正答(val) 学習後: {a1:.3f}（学習前 {a0:.3f}）", flush=True)
    print("凍結検査: 胴体・既存ヘッド・補助ヘッド・他の出口ヘッドは bit 不変"
          "（既存挙動は定義上不変）", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_v = os.path.join(args.out, "value.npz")
    vnet.save(out_v)
    re = RN.ValueNet.load(out_v)
    assert re.vocab_ids == RN.ValueNet.load(vpath).vocab_ids, \
        "保存した候補の vocab_ids が base と一致しない"
    _w2 = RN.EXIT_HEADS[args.head][3]
    assert re.has_exit_head(args.head) and np.array_equal(getattr(re, _w2), getattr(vnet, _w2)), \
        f"{args.head} 出口ヘッドが保存されていない"
    out_p = os.path.join(args.out, "policy.npz")
    if ev0 != args.enc_version:
        from opcg_sim.src.learned.policy import PolicyScorer
        warm_start_policy(PolicyScorer.load(ppath), ev0, args.enc_version).save(out_p)
    else:
        shutil.copyfile(ppath, out_p)
    res = {"head": args.head, "base": args.base, "files": n_files,
           "boards": int(len(child["value"])),
           "groups": int(len(set(child["group"]))), "pairs": len(pairs),
           "epochs": args.epochs, "lr": args.lr, "head_hidden": args.head_hidden,
           "rank_acc_before": round(a0, 4), "rank_acc_after": round(a1, 4),
           "candidate": f"{out_v},{out_p}"}
    print(f"EXIT_HEAD_FINETUNE_RESULT {json.dumps(res)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
