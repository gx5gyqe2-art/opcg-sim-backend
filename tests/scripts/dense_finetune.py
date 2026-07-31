"""密ラベル自己対戦コーパスで gen6 value を追い学習する（v16・`dense_selfplay_gen.py` の対）。

なぜ既存の `ref_finetune_smoke.py` を使わないか: あちらは**レフェリー教師**（採掘決定点・1.6万行・
q_root 無し）の微調整器で、`collect_ref_batches` は q_root/turns_left 列を読まない＝ラベルが
勝敗単独（α=1）へ退化する。v16 の仮説は「gen2→gen5 を支えた**密度レジーム**（102.9 点/局・
混合ラベル y=α·z+(1−α)·q_root ＋残りターン補助損失）を gen6 以降に一度も再現していない」なので、
その学習仕様ごと再現しないと仮説の検証にならない。よって別ファイルに分ける
（CLAUDE.md「1トピック=1ファイル」）。

学習仕様は v4/v5 本走（`pd_learn.py`）と同一:
  - y = α·勝敗 + (1−α)·q_root（`pd_batch_common.mixed_value_label`・α 既定 `V4_LABEL_ALPHA`）
  - 残りターン補助損失（`V4_AUX_TURNS_WEIGHT` / `V4_TURNS_SCALE`）
  - 任意で gen6 予測への distill アンカー（忘却抑制・v5 §4-4b）
**policy は学習しない**（v12 で確定＝policy 微調整は 1 エポックでも対gen6 を 0.33 に落とす）。
出力は value.npz と、基準世代の policy.npz のコピー＝アリーナ/コーチゲートへそのまま渡せる対。

判定は本スクリプトでは行わない。`tests/scripts/arena_gate.py`（800局・帯層別）と
`tests/scripts/coach_gate.py` が判定器（安価な代理指標は v13-v15 でアリーナと逆相関＝一次判定に
使わない）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/dense_finetune.py \
    --dirs /tmp/dense_v16 --epochs 2 --lr 2e-4 --out /tmp/cand_v16
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import shutil
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from pd_batch_common import mixed_value_label, normalize_batch_v2
from opcg_sim.src.learned.config import V4_LABEL_ALPHA, V4_AUX_TURNS_WEIGHT, V4_TURNS_SCALE
from opcg_sim.src.core.cpu_learned import warm_start_value, warm_start_policy, _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")

VKEYS = ("scalars", "field", "card_idx", "value", "q_root", "turns_left")


def load_dense(dirs, log=print):
    """密コーパスの npz 群を連結する。旧 v1 形式（q_root 無し）は v2 へ正規化して受ける。"""
    parts, n_files = {k: [] for k in VKEYS}, 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            arr = normalize_batch_v2({k: z[k] for k in VKEYS if k in z.files})
            for k in VKEYS:
                parts[k].append(arr[k])
            n_files += 1
    if not n_files:
        return None
    out = {k: np.concatenate(parts[k]) for k in VKEYS}
    out["value"] = out["value"].astype(np.float32)
    log(f"収集: {n_files}シャード・{len(out['value'])}行"
        f"（q_root 有限率 {float(np.isfinite(out['q_root']).mean()):.2f}）")
    return out


def build_labels(vdata, alpha=V4_LABEL_ALPHA, turns_scale=V4_TURNS_SCALE):
    """(y, aux) を作る（pure）。q_root が非有限な行は勝敗単独へ退化させる（L1 席等）。

    aux は clip 済み正規化残りターン（NaN は `ValueNet.backward` 側で欠損として除外される）。"""
    q = np.asarray(vdata["q_root"], dtype=np.float32)
    z = np.asarray(vdata["value"], dtype=np.float32)
    q = np.where(np.isfinite(q), q, z)
    y = mixed_value_label(z, q, alpha)
    aux = np.clip(np.asarray(vdata["turns_left"], dtype=np.float32), 0, turns_scale) / turns_scale
    return y, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="密コーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--base", default="gen6", help="温スタート元の同梱世代")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--label-alpha", type=float, default=V4_LABEL_ALPHA,
                    help="y = α·勝敗 + (1−α)·q_root の α（1.0 で勝敗単独＝ref 教師と同じ扱い）")
    ap.add_argument("--aux-weight", type=float, default=V4_AUX_TURNS_WEIGHT)
    ap.add_argument("--turns-scale", type=float, default=V4_TURNS_SCALE)
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="凍結 base 予測への value アンカー MSE（忘却抑制・v5 §4-4b・0で無効）")
    ap.add_argument("--max-rows", type=int, default=0, help=">0 で末尾この行数に制限（新しい順）")
    ap.add_argument("--enc-version", type=int, default=5)
    ap.add_argument("--out", required=True, help="候補ネットの保存先ディレクトリ")
    args = ap.parse_args()

    vdata = load_dense([d for d in args.dirs.split(",") if d])
    if vdata is None:
        print("密コーパスが空（--dirs を確認）"); return 1
    if args.max_rows > 0 and len(vdata["value"]) > args.max_rows:
        vdata = {k: v[-args.max_rows:] for k, v in vdata.items()}
        print(f"末尾 {args.max_rows} 行に制限")

    vpath = os.path.join(MODELS, f"{args.base}_value.npz")
    ppath = os.path.join(MODELS, f"{args.base}_policy.npz")
    vnet = RN.ValueNet.load(vpath)
    ev0 = _net_enc_version(vnet)
    if ev0 != args.enc_version:
        vnet = warm_start_value(vnet, ev0, args.enc_version)   # append-only ゼロ挿入＝恒等
        print(f"温スタート拡張: v{ev0} → v{args.enc_version}")
    assert vnet.feat_dim == E.feature_dim(args.enc_version), (   # pooled/lead/eff 枠を除いた実次元
        f"入力次元不一致: {vnet.feat_dim} != {E.feature_dim(args.enc_version)}")

    y, aux = build_labels(vdata, args.label_alpha, args.turns_scale)
    data = {k: vdata[k] for k in ("scalars", "field", "card_idx")}
    data["value"] = y
    if args.aux_weight > 0:
        data["aux"] = aux
    if args.distill_weight > 0:
        teacher = RN.ValueNet.load(vpath)
        if _net_enc_version(teacher) != args.enc_version:
            teacher = warm_start_value(teacher, _net_enc_version(teacher), args.enc_version)
        data["distill"] = teacher.predict(data)

    print(f"学習: {len(y)}行 base={args.base} epochs={args.epochs} lr={args.lr} "
          f"α={args.label_alpha} aux={args.aux_weight} distill={args.distill_weight}", flush=True)
    t0 = time.time()
    tm, vm = RN.train(vnet, data, epochs=args.epochs, lr=args.lr, batch=args.batch,
                      val_frac=args.val_frac, verbose=True,
                      aux_weight=args.aux_weight, distill_weight=args.distill_weight)

    os.makedirs(args.out, exist_ok=True)
    out_v = os.path.join(args.out, "value.npz")
    vnet.save(out_v)
    # 焼き込み vocab の脱落ガード（2026-07-22 実害: expanded() が vocab_ids を落とし、
    # 候補が現DB由来の別 index で埋め込みを引いて全判定が無効になった）。
    assert RN.ValueNet.load(out_v).vocab_ids == RN.ValueNet.load(vpath).vocab_ids, \
        "保存した候補の vocab_ids が base と一致しない（index ズレで評価が無意味になる）"
    out_p = os.path.join(args.out, "policy.npz")
    # policy は base のまま学習しない（v12: 微調整は有害）が、**符号化版は value に揃える**。
    # 2026-07-31 実害: value だけ v6 へ拡張し policy を v5 のままコピーすると、serve の
    # ctx(v6) を v5 policy の `_fit_actions` が「行動特徴の末尾ズレ」と誤解釈し、行動特徴が
    # 5列ずれた重みで読まれて priors が無意味になる（クラッシュせず黙って壊れる＝arena 0/48）。
    if ev0 != args.enc_version:
        from opcg_sim.src.learned.policy import PolicyScorer
        pnet = warm_start_policy(PolicyScorer.load(ppath), ev0, args.enc_version)  # 恒等拡張
        pnet.save(out_p)
    else:
        shutil.copyfile(ppath, out_p)
    res = {"rows": int(len(y)), "base": args.base, "epochs": args.epochs, "lr": args.lr,
           "label_alpha": args.label_alpha, "aux_weight": args.aux_weight,
           "distill_weight": args.distill_weight,
           "train_mse": round(tm, 5), "val_mse": round(vm, 5),
           "candidate": f"{out_v},{out_p}", "sec": int(time.time() - t0)}
    json.dump(res, open(os.path.join(args.out, "train.json"), "w"), ensure_ascii=False)
    print(f"DENSE_FINETUNE_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
