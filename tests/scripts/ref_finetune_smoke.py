"""v9 フェーズ2 スモーク: レフェリー教師での gen5 微調整（docs/cpu_v9_plan.md §3 の当たり付け）。

v9-label 全枝（claude/v9-label-w*）の教師バッチを収集し、
  1. train/val 分割（決定単位・ハッシュ固定＝再現可能）
  2. gen5 を温スタートして value（z=勝率/捲り率）・policy（バンド上位初手 multi-hot・
     学習時 smooth 床＝「未評価」ハードゼロの緩和）を微調整
  3. 前後評価: val の policy 支持一致率（教師支持集合に argmax が入る率）・KL・value MAE/corr
を LR 候補ごとに報告する。**読み取り専用スモーク**＝同梱ネットは書き換えない
（--out 指定時のみ候補 npz を保存＝後続のゲート運転用）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/ref_finetune_smoke.py \
    --lrs 2e-4,5e-5 --epochs 8 --out /tmp/ref_ft
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import io
import subprocess

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
from az_policy import PolicyScorer, train_policy
from opcg_sim.src.learned.action import ACTION_DIM
from opcg_sim.src.learned.policy import extend_action_dim
from pd_batch_common import unpack_policy
from opcg_sim.src.learned.encoder import scalars_dim, field_dim, known_versions
from opcg_sim.src.core.cpu_learned import _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pad_cols(a, cols):
    """行列/ベクトルを末尾ゼロ埋めで cols 幅に（append-only 特徴＝末尾追加なので単純末尾埋め）。"""
    if a.ndim == 1:
        return a if a.shape[0] >= cols else np.concatenate(
            [a, np.zeros(cols - a.shape[0], a.dtype)])
    return a if a.shape[1] >= cols else np.concatenate(
        [a, np.zeros((len(a), cols - a.shape[1]), a.dtype)], axis=1)


def _pad_ctx(ctx, target_sc):
    """policy ctx=[scalars | field_flat]。旧版 ctx を最新版へ＝scalars 部分の末尾（field の直前）へ
    ゼロ挿入（末尾埋めは field 特徴を汚染するため不可）。"""
    fd = field_dim()
    old_sc = len(ctx) - fd
    if old_sc >= target_sc:
        return ctx
    return np.concatenate([ctx[:old_sc], np.zeros(target_sc - old_sc, ctx.dtype), ctx[old_sc:]])


def _warm_expand(vnet, pnet):
    """gen5（旧符号化版）を教師の最新版へ温スタート拡張。scalars 差を value 入力と policy ctx の
    scalars 末尾へゼロ挿入（恒等）。action 差は呼び出し側の extend_action_dim が担う。"""
    at = scalars_dim(_net_enc_version(vnet))
    d_sc = scalars_dim(max(known_versions())) - at
    if d_sc > 0:
        vnet = vnet.expanded(at, d_sc)
        pnet = pnet.expanded(at, d_sc)
    return vnet, pnet


def collect_ref_batches(workers=("w1", "w2", "w3", "w4", "w5"), log=print):
    """v9-label 枝から全教師バッチを収集して (vdata dict, pol list) に連結する。版が混在
    （旧 v4=51 / 新 v5=55）しても最新版へゼロ埋め統一する（append-only 恒等・cpu_v10）。"""
    tsc = scalars_dim(max(known_versions()))
    S, F, I, Y, K = [], [], [], [], []
    CS, CF, CI, CY = [], [], [], []   # v11 子盤面 value 教師（root 行と独立）
    pol = []
    n_batches = 0
    for w in workers:
        br = f"origin/claude/v9-label-{w}"
        ls = subprocess.run(["git", "-C", REPO, "ls-tree", br + ":p9label", "--name-only"],
                            capture_output=True, text=True)
        for f in ls.stdout.split():
            if not f.startswith("batch_"):
                continue
            raw = subprocess.run(["git", "-C", REPO, "show", f"{br}:p9label/{f}"],
                                 capture_output=True).stdout
            z = np.load(io.BytesIO(raw))
            S.append(_pad_cols(z["scalars"], tsc)); F.append(z["field"]); I.append(z["card_idx"])
            Y.append(z["value"])
            # kind（disagree/sat/blind）: kind 修正前の旧バッチは "" 埋め（重み付け対象外）
            K.append(z["kind"] if "kind" in z.files
                     else np.array([""] * len(z["value"]), dtype="<U8"))
            if "child_value" in z.files:
                CS.append(_pad_cols(z["child_scalars"], tsc)); CF.append(z["child_field"])
                CI.append(z["child_card_idx"]); CY.append(z["child_value"])
            for ctx, am, t in unpack_policy({k: z[k] for k in z.files if k.startswith("pol_")}):
                pol.append((_pad_ctx(ctx, tsc), _pad_cols(am, ACTION_DIM), t))
            n_batches += 1
    if not S:
        return None, None
    vdata = {"scalars": np.concatenate(S), "field": np.concatenate(F),
             "card_idx": np.concatenate(I),
             "value": np.concatenate(Y).astype(np.float32),
             "kind": np.concatenate(K)}
    n_child = 0
    if CY:
        vdata["_child"] = {"scalars": np.concatenate(CS), "field": np.concatenate(CF),
                           "card_idx": np.concatenate(CI),
                           "value": np.concatenate(CY).astype(np.float32)}
        n_child = len(vdata["_child"]["value"])
    log(f"収集: {n_batches}バッチ・教師 {len(vdata['value'])} 決定・子盤面 {n_child}")
    return vdata, pol


def split_idx(n, val_frac=0.15, seed=7):
    """決定単位の train/val 分割（固定 seed＝再現可能）。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def eval_nets(vnet, pnet, vdata, pol, idx):
    """val 指標: value MAE/corr・policy 支持一致率・KL(教師‖net)。"""
    batch = {k: vdata[k][idx] for k in ("scalars", "field", "card_idx")}
    v = vnet.predict(batch)
    z = vdata["value"][idx]
    mae = float(np.abs(v - z).mean())
    corr = float(np.corrcoef(v, z)[0, 1]) if len(idx) > 2 else float("nan")
    agree, kls = 0, []
    for j in idx:
        ctx, am, t = pol[j]
        p = pnet.priors(ctx, am)
        if t[int(np.argmax(p))] > 0:
            agree += 1
        m = t > 0
        kls.append(float(np.sum(t[m] * np.log((t[m] + 1e-9) / (p[m] + 1e-9)))))
    return {"mae": mae, "corr": corr, "agree": agree / max(len(idx), 1),
            "kl": float(np.median(kls))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", default="2e-4,5e-5", help="試す学習率（カンマ区切り）")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--policy-smooth", type=float, default=0.05,
                    help="policy 教師の床（v7 案E・未評価ハードゼロの緩和）")
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="value の忘却対策: 凍結 gen5 予測への distill MSE（v5 §4-4b 機構を流用）")
    ap.add_argument("--policy-selfdistill", type=float, default=0.0,
                    help="policy の忘却対策: gen5 prior を教師とする自己蒸留サンプルを"
                         "ref 教師1件あたりこの比率で混合（mark ガード退行の抑制）")
    ap.add_argument("--skip-policy", action="store_true",
                    help="policy を微調整せず gen5 のまま保存する（v9 既定推奨）。ablation で "
                         "policy 微調整が @64 等の正しい点を壊す犯人と確定（value のみ学習で "
                         "コーチPASS・arena 非退行・2026-07-18）。value は decide の主役で "
                         "素直に学べるため、policy を据え置くのが v9 の正しい形。")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--disagree-weight", type=float, default=1.0,
                    help="kind=disagree（反例）サンプルの policy 学習での複製倍率。1=無効")
    ap.add_argument("--out", default=None, help="候補ネットの保存先（lr ごとのサブ名で保存）")
    ap.add_argument("--base", default="gen6",
                    help="温スタート元の同梱世代（既定 gen6=現既定ネット。gen5 で旧ベース比較）")
    args = ap.parse_args()

    vdata, pol = collect_ref_batches()
    if vdata is None:
        print("教師バッチが見つからない（git fetch 済みか確認）"); return 1
    n = len(vdata["value"])
    tr, va = split_idx(n, args.val_frac)
    print(f"train {len(tr)} / val {len(va)}")

    base_v_path = os.path.join(REPO, "opcg_sim", "data", "learned", f"{args.base}_value.npz")
    base_p_path = os.path.join(REPO, "opcg_sim", "data", "learned", f"{args.base}_policy.npz")
    base = eval_nets(*_warm_expand(RN.ValueNet.load(base_v_path), PolicyScorer.load(base_p_path)),
                     vdata, pol, va)
    print(f"\n[{args.base} 基準] val: value MAE={base['mae']:.3f} corr={base['corr']:.3f}  "
          f"policy 支持一致={base['agree']*100:.0f}% KL={base['kl']:.3f}")

    tr_kind = vdata["kind"][tr]
    tr_vdata = {k: vdata[k][tr] for k in vdata if k not in ("kind", "_child")}
    ch = vdata.get("_child")
    if ch is not None:
        # v11 子盤面教師は全て train へ併合（val は root 決定のみ＝前後比較の指標互換を維持）。
        # decide が比較する「初手後の子盤面」の序列を value に直接教える（@68/@93 の実測根拠）。
        tr_vdata = {k: np.concatenate([tr_vdata[k], ch[k]]) for k in tr_vdata}
        print(f"子盤面教師: +{len(ch['value'])} 行を train に併合")
    tr_pol = [pol[j] for j in tr]
    if args.disagree_weight > 1:
        # 反例（disagree）を複製して policy 学習で重く効かせる（policy_selfdistill と同じ手法）。
        # kind 付きの新バッチのみ対象＝旧バッチ（"" 埋め）は等倍。
        reps = int(round(args.disagree_weight)) - 1
        extra = [tr_pol[j] for j in range(len(tr_pol))
                 if tr_kind[j] == "disagree" for _ in range(reps)]
        n_dis = int((tr_kind == "disagree").sum())
        tr_pol = tr_pol + extra
        print(f"disagree 重み付け: {n_dis} 反例 ×{args.disagree_weight:g} → +{len(extra)} 複製")
    ctx_dim = len(pol[0][0])
    base_v, base_p = _warm_expand(RN.ValueNet.load(base_v_path), PolicyScorer.load(base_p_path))
    if args.distill_weight > 0:
        # 忘却対策（value）: 凍結 gen5 の予測を distill アンカーに（v5 §4-4b の機構を流用）。
        tr_vdata = dict(tr_vdata)
        tr_vdata["distill"] = base_v.predict(
            {k: tr_vdata[k] for k in ("scalars", "field", "card_idx")}).astype(np.float32)
    if args.policy_selfdistill > 0:
        # 忘却対策（policy）: gen5 prior を教師とする自己蒸留サンプルを混合＝ref 教師が
        # 押す場所以外は gen5 の挙動に留める（mark ガード退行の抑制）。
        import math
        n_sd = int(math.ceil(len(tr_pol) * args.policy_selfdistill))
        rng = np.random.default_rng(11)
        idxs = rng.choice(len(tr_pol), size=n_sd, replace=n_sd > len(tr_pol))
        sd = []
        for j in idxs:
            ctx, am, _t = tr_pol[j]
            sd.append((ctx, am, base_p.priors(ctx, am)))
        tr_pol = tr_pol + sd
    for lr in [float(x) for x in args.lrs.split(",")]:
        vnet, pnet = _warm_expand(RN.ValueNet.load(base_v_path), PolicyScorer.load(base_p_path))
        if pnet.in_dim < ctx_dim + ACTION_DIM:
            # v9 行動特徴拡張の温スタート（零行追加＝出力恒等）。新特徴（カウンター値等）は
            # 新形式で記録されたバッチからのみ学習される（旧22次元記録はゼロ埋め）。
            extend_action_dim(pnet, ctx_dim + ACTION_DIM - pnet.in_dim)
        tm, vm = RN.train(vnet, tr_vdata, epochs=args.epochs, lr=lr, batch=64, val_frac=0.1,
                          distill_weight=args.distill_weight)
        if args.skip_policy:
            ce = float("nan")   # policy は gen5 のまま（据え置き＝v9 既定）
        else:
            ce = train_policy(pnet, tr_pol, epochs=args.epochs, lr=lr,
                              smooth=args.policy_smooth)
        after = eval_nets(vnet, pnet, vdata, pol, va)
        print(f"[lr={lr:g}] train: value mse {tm:.3f}→val {vm:.3f}・policy CE {ce:.3f}")
        print(f"          val: value MAE={after['mae']:.3f} corr={after['corr']:.3f}  "
              f"policy 支持一致={after['agree']*100:.0f}% KL={after['kl']:.3f}  "
              f"（Δ一致 {100*(after['agree']-base['agree']):+.0f}pt・ΔMAE {after['mae']-base['mae']:+.3f}）")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            tag = f"lr{lr:g}".replace("-", "m")
            vnet.save(os.path.join(args.out, f"value_{tag}.npz"))
            pnet.save(os.path.join(args.out, f"policy_{tag}.npz"))
            print(f"          saved → {args.out}/*_{tag}.npz")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
