"""オプションペア教師で gen10 を順位微調整する（v31・`option_pair_gen.py` の対）。

v12.1 の順位ヒンジ（`ref_finetune_smoke.rank_finetune`）を、`option_pair_gen` が吐いた
カード単位ペアコーパス（scalars/field/card_idx/value=因果z/group）へ適用する。value 回帰でなく
**同一 group 内の順位** v(勝ち子)>v(負け子) をマージン付きで直接教える＝m4@2 の拮抗する2子を
分離する（回帰は近づけるだけで順位を確定できない・v30 §3-2）。

base=gen10（v7・既に「差を表せる」表現）。policy は base のまま（v12 確定＝policy 微調整は有害）。
判定は外部（coach_gate/arena_resume）。本スクリプトは順位正答率（学習前後・base 対比）だけ出す。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/option_pair_finetune.py \
    --dirs /tmp/optpair --base gen10 --epochs 6 --lr 2e-5 --out /tmp/cand_v31
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import shutil

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from ref_finetune_smoke import (build_rank_pairs, dead_weighted_pairs, pair_acc,
                                rank_finetune, rank_finetune_anchored)
from opcg_sim.src.core.cpu_learned import warm_start_value, warm_start_policy, _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")


PAIR_GLOBS = ("optpair_*.npz", "defcf_*.npz", "plancf_*.npz", "diginj_*.npz")


def load_pairs_corpus(dirs, globs=PAIR_GLOBS):
    """optpair シャード群を child dict（scalars/field/card_idx/value/group/dead_play）へ連結。

    dead_play（v33・不発PLAYフラグ）は旧シャードに無い＝0 で埋める（後方互換）。
    `globs`（v39）は読むシャード種の指定＝**箱の階層ごとに教師を分けて読む**ための seam
    （出口ヘッドの学習は自階層の教師だけを読む＝ターン末 plancf / 戦闘出口 defcf・`exit_head_finetune.py` の `HEAD_GLOBS`）。"""
    keys = ("scalars", "field", "card_idx", "value", "group")
    parts = {k: [] for k in keys}
    dead = []
    n_files = 0
    for d in dirs:
        # v34: 防御窓CF（defcf_*）も同スキーマ（group つき margin_blend ラベル）＝同じ順位学習に流せる。
        # v38: ターン出口CF（plancf_*）も同様（ラベル対象がターン末盤面である点だけが違う）。
        files = []
        for g in globs:
            files += glob.glob(os.path.join(d, g))
        for f in sorted(files):
            z = np.load(f)
            for k in keys:
                parts[k].append(z[k])
            dead.append(z["dead_play"] if "dead_play" in z.files
                        else np.zeros(len(z["value"]), np.float32))
            n_files += 1
    if not n_files:
        return None, 0
    child = {k: np.concatenate(parts[k]) for k in keys}
    child["value"] = child["value"].astype(np.float32)
    child["dead_play"] = np.concatenate(dead).astype(np.float32)
    return child, n_files


def load_anchor(dirs, enc_version, base_net, rows, seed=11, own_turn_only=False):
    """蒸留アンカー（v33）: dense コーパスの一般盤面を読み、scalars を enc_version 幅へ
    ゼロ拡張し、base ネットの予測 y を焼く。

    ゼロ拡張の意味論: append-only 契約の下で、v8 温スタート直後の base は追加列の重みが
    ゼロ＝ゼロ埋め入力での予測は旧版と厳密に一致する。アンカーは「v7 特徴で決まる既存挙動
    （防御較正など）」を固定する錘であり、新特徴（v8 列）の学習は妨げない。

    own_turn_only（v35 層別アンカー）: 相手ターン（＝防御判断側）の盤面をアンカーから除外し、
    自ターン盤面だけを base へ釘付けにする。防御較正はライフ↔手札の交換レートという
    **評価尺度そのもの**を動かす学習であり、dense 盤面の約4割を占める相手ターン盤面を
    MSE で固定すると順位教師の押しが木の深部で打ち消される（v35 実測: 1手先は正解が
    +0.117 上なのに探索後 root Q は PASS が +0.029 上へ逆転）。"""
    keys = ("scalars", "field", "card_idx")
    parts = {k: [] for k in keys}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "dense_*.npz"))):
            z = np.load(f)
            for k in keys:
                parts[k].append(z[k])
    if not parts["scalars"]:
        return None, None
    anchor = {k: np.concatenate(parts[k]) for k in keys}
    if own_turn_only:
        keep = anchor["scalars"][:, E.IDX_IS_MY_TURN] > 0.5
        print(f"層別アンカー: 自ターン {int(keep.sum())}/{len(keep)} 盤面のみ使用"
              f"（相手ターン＝防御系 {int((~keep).sum())} を除外）", flush=True)
        anchor = {k: anchor[k][keep] for k in keys}
    want = E.scalars_dim(enc_version)
    have = anchor["scalars"].shape[1]
    assert have <= want, f"アンカーの符号化が新しすぎる: {have} > {want}"
    if have < want:
        pad = np.zeros((len(anchor["scalars"]), want - have), np.float32)
        anchor["scalars"] = np.concatenate([anchor["scalars"], pad], axis=1)
    rng = np.random.default_rng(seed)
    sel = rng.permutation(len(anchor["scalars"]))[:rows]
    anchor = {k: anchor[k][sel] for k in keys}
    y = np.empty(len(sel), np.float64)
    for s in range(0, len(sel), 4096):
        y[s:s + 4096] = base_net.predict({k: anchor[k][s:s + 4096] for k in keys})
    return anchor, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="optpair コーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--globs", default="",
                    help="読むシャード種（カンマ区切り・空=既定の PAIR_GLOBS）。"
                         "単調性教師 monopair_*.npz など新種を読むときに指定（2026-08-15）")
    ap.add_argument("--base", default="gen10", help="順位微調整の起点（v7 表現）")
    ap.add_argument("--base-path", default="",
                    help="起点 value.npz[,policy.npz] をパス直指定（MODELS 外の候補ネット＝"
                         "fixture 保全品などを起点にする時。--base より優先・2026-08-14）")
    ap.add_argument("--enc-version", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.25, help="順位ペアに採る z 差の下限")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--anchor-dirs", default="",
                    help="蒸留アンカーの dense コーパス（カンマ区切り・空=アンカー無し=v32 挙動）")
    ap.add_argument("--anchor-rows", type=int, default=16000)
    ap.add_argument("--anchor-scale", type=float, default=1.0, help="錘の強さ（lr への係数）")
    ap.add_argument("--anchor-own-turn-only", action="store_true",
                    help="層別アンカー（v35）: 相手ターン＝防御系の盤面をアンカーから除外")
    ap.add_argument("--dead-weight", type=float, default=1.0,
                    help="負け側が不発PLAYのペアの重み（複製倍率・1=無効）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _globs = tuple(g for g in args.globs.split(",") if g) or PAIR_GLOBS
    child, n_files = load_pairs_corpus([d for d in args.dirs.split(",") if d], globs=_globs)
    if child is None:
        print("optpair コーパスが空（--dirs を確認）"); return 1

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

    pairs = build_rank_pairs(child, delta=args.delta)
    # group 単位で train/val 分割（同一決定点が両側に跨がない＝リークしない）。
    p_tr = [p for p in pairs if p[2] % 7 != 0]
    p_va = [p for p in pairs if p[2] % 7 == 0]
    base = RN.ValueNet.load(vpath)
    if _net_enc_version(base) != args.enc_version:
        base = warm_start_value(base, _net_enc_version(base), args.enc_version)
    a0 = pair_acc(vnet, child, p_va)
    ab = pair_acc(base, child, p_va)
    print(f"収集: {n_files}シャード・{len(child['value'])}子盤面・{len(set(child['group']))}群・"
          f"順位ペア {len(pairs)}（tr {len(p_tr)}/va {len(p_va)}・不発行 "
          f"{int(child['dead_play'].sum())}）", flush=True)
    print(f"順位正答(val) 学習前: base={ab:.3f} / cand(=base)={a0:.3f}", flush=True)

    if args.dead_weight > 1:
        n0 = len(p_tr)
        p_tr = dead_weighted_pairs(p_tr, child["dead_play"], k=args.dead_weight)
        print(f"不発ペア重み増し: tr {n0} → {len(p_tr)}", flush=True)
    if args.anchor_dirs:
        anchor, y_anchor = load_anchor([d for d in args.anchor_dirs.split(",") if d],
                                       args.enc_version, base, args.anchor_rows,
                                       own_turn_only=args.anchor_own_turn_only)
        assert anchor is not None, "アンカーコーパスが空（--anchor-dirs を確認）"
        print(f"蒸留アンカー: {len(y_anchor)}盤面・scale={args.anchor_scale}", flush=True)
        rank_finetune_anchored(vnet, child, p_tr, anchor, y_anchor,
                               epochs=args.epochs, lr=args.lr, margin=args.margin,
                               anchor_scale=args.anchor_scale)
    else:
        rank_finetune(vnet, child, p_tr, epochs=args.epochs, lr=args.lr, margin=args.margin)
    a1 = pair_acc(vnet, child, p_va)
    print(f"順位正答(val) 学習後: cand={a1:.3f}（base {ab:.3f}）", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_v = os.path.join(args.out, "value.npz")
    vnet.save(out_v)
    assert RN.ValueNet.load(out_v).vocab_ids == RN.ValueNet.load(vpath).vocab_ids, \
        "保存した候補の vocab_ids が base と一致しない"
    out_p = os.path.join(args.out, "policy.npz")
    # policy は base のまま学習しない（v12）が、**符号化版は value に揃える**（dense_finetune と
    # 同じ 2026-07-31 実害の再発防止: 版違いコピーは `_fit_actions` の行動特徴列ズレで
    # クラッシュせず黙って壊れる＝v30 の arena 0/48 の根本原因）。
    if ev0 != args.enc_version:
        from opcg_sim.src.learned.policy import PolicyScorer
        warm_start_policy(PolicyScorer.load(ppath), ev0, args.enc_version).save(out_p)
    else:
        shutil.copyfile(ppath, out_p)
    res = {"base": args.base, "files": n_files, "children": int(len(child["value"])),
           "groups": int(len(set(child["group"]))), "pairs": len(pairs),
           "anchor": bool(args.anchor_dirs), "anchor_scale": args.anchor_scale,
           "anchor_own_turn_only": args.anchor_own_turn_only,
           "dead_weight": args.dead_weight,
           "rank_acc_base": round(ab, 4), "rank_acc_before": round(a0, 4),
           "rank_acc_after": round(a1, 4),
           "candidate": f"{out_v},{os.path.join(args.out, 'policy.npz')}"}
    print(f"OPTPAIR_FINETUNE_RESULT {json.dumps(res)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
