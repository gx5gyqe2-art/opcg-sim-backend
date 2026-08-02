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
from ref_finetune_smoke import build_rank_pairs, pair_acc, rank_finetune
from opcg_sim.src.core.cpu_learned import warm_start_value, _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")


def load_pairs_corpus(dirs):
    """optpair シャード群を child dict（scalars/field/card_idx/value/group）へ連結。"""
    keys = ("scalars", "field", "card_idx", "value", "group")
    parts = {k: [] for k in keys}
    n_files = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "optpair_*.npz"))):
            z = np.load(f)
            for k in keys:
                parts[k].append(z[k])
            n_files += 1
    if not n_files:
        return None, 0
    child = {k: np.concatenate(parts[k]) for k in keys}
    child["value"] = child["value"].astype(np.float32)
    return child, n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="optpair コーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--base", default="gen10", help="順位微調整の起点（v7 表現）")
    ap.add_argument("--enc-version", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.25, help="順位ペアに採る z 差の下限")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    child, n_files = load_pairs_corpus([d for d in args.dirs.split(",") if d])
    if child is None:
        print("optpair コーパスが空（--dirs を確認）"); return 1

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
          f"順位ペア {len(pairs)}（tr {len(p_tr)}/va {len(p_va)}）", flush=True)
    print(f"順位正答(val) 学習前: base={ab:.3f} / cand(=base)={a0:.3f}", flush=True)

    rank_finetune(vnet, child, p_tr, epochs=args.epochs, lr=args.lr, margin=args.margin)
    a1 = pair_acc(vnet, child, p_va)
    print(f"順位正答(val) 学習後: cand={a1:.3f}（base {ab:.3f}）", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_v = os.path.join(args.out, "value.npz")
    vnet.save(out_v)
    assert RN.ValueNet.load(out_v).vocab_ids == RN.ValueNet.load(vpath).vocab_ids, \
        "保存した候補の vocab_ids が base と一致しない"
    shutil.copyfile(ppath, os.path.join(args.out, "policy.npz"))   # policy は base のまま（v12）
    res = {"base": args.base, "files": n_files, "children": int(len(child["value"])),
           "groups": int(len(set(child["group"]))), "pairs": len(pairs),
           "rank_acc_base": round(ab, 4), "rank_acc_before": round(a0, 4),
           "rank_acc_after": round(a1, 4),
           "candidate": f"{out_v},{os.path.join(args.out, 'policy.npz')}"}
    print(f"OPTPAIR_FINETUNE_RESULT {json.dumps(res)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
