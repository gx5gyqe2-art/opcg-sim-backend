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


def collect_ref_batches(workers=("w1", "w2", "w3", "w4", "w5"), extra_dirs=(), log=print):
    """v9-label 枝から全教師バッチを収集して (vdata dict, pol list) に連結する。版が混在
    （旧 v4=51 / 新 v5=55）しても最新版へゼロ埋め統一する（append-only 恒等・cpu_v10）。
    extra_dirs はローカル npz バッチの追加読み込み（divergence_probe の乖離教師等・v12）。"""
    tsc = scalars_dim(max(known_versions()))
    S, F, I, Y, K = [], [], [], [], []
    CS, CF, CI, CY = [], [], [], []   # v11 子盤面 value 教師（root 行と独立）
    CG = []                           # v12.1 決定点グループ（バッチ跨ぎで一意化・旧バッチは -1）
    _gbase = [0]
    pol = []
    n_batches = 0

    def _ingest(z):
        S.append(_pad_cols(z["scalars"], tsc)); F.append(z["field"]); I.append(z["card_idx"])
        Y.append(z["value"])
        # kind（disagree/sat/blind/diverge）: kind 修正前の旧バッチは "" 埋め（重み付け対象外）
        K.append(z["kind"] if "kind" in z.files
                 else np.array([""] * len(z["value"]), dtype="<U8"))
        if "child_value" in z.files:
            CS.append(_pad_cols(z["child_scalars"], tsc)); CF.append(z["child_field"])
            CI.append(z["child_card_idx"]); CY.append(z["child_value"])
            n_c = len(z["child_value"])
            if "child_group" in z.files and n_c:
                CG.append(z["child_group"].astype(np.int64) + _gbase[0])
                _gbase[0] += int(z["child_group"].max()) + 1
            else:
                CG.append(np.full(n_c, -1, dtype=np.int64))   # 旧バッチ＝グループ不明でペア不能
        for ctx, am, t in unpack_policy({k: z[k] for k in z.files if k.startswith("pol_")}):
            pol.append((_pad_ctx(ctx, tsc), _pad_cols(am, ACTION_DIM), t))

    for w in workers:
        br = f"origin/claude/v9-label-{w}"
        ls = subprocess.run(["git", "-C", REPO, "ls-tree", br + ":p9label", "--name-only"],
                            capture_output=True, text=True)
        for f in ls.stdout.split():
            if not f.startswith("batch_"):
                continue
            raw = subprocess.run(["git", "-C", REPO, "show", f"{br}:p9label/{f}"],
                                 capture_output=True).stdout
            _ingest(np.load(io.BytesIO(raw)))
            n_batches += 1
    import glob as _glob
    for d in extra_dirs:
        for f in sorted(_glob.glob(os.path.join(d, "*.npz"))):
            _ingest(np.load(f))
            n_batches += 1
    if not S:
        return None, None
    vdata = {"scalars": np.concatenate(S), "field": np.concatenate(F),
             "card_idx": np.concatenate(I),
             "value": np.concatenate(Y).astype(np.float32),
             "kind": np.concatenate(K)}
    n_child = 0
    n_grouped = 0
    if CY:
        grp = np.concatenate(CG)
        vdata["_child"] = {"scalars": np.concatenate(CS), "field": np.concatenate(CF),
                           "card_idx": np.concatenate(CI),
                           "value": np.concatenate(CY).astype(np.float32),
                           "group": grp}
        n_child = len(vdata["_child"]["value"])
        n_grouped = int((grp >= 0).sum())
    log(f"収集: {n_batches}バッチ・教師 {len(vdata['value'])} 決定・子盤面 {n_child}"
        f"（グループ付き {n_grouped}）")
    return vdata, pol


def build_rank_pairs(child, delta=0.25, cap_per_group=12):
    """同一決定点（group）の子盤面から z 差 > δ のペア (勝ちidx, 負けidx, group) を作る（pure）。
    v12.1: レフェリーが実測した「初手後の子盤面の順位」だけを教える＝絶対値 z を強制した
    v11 子盤面ラベルの楽観バイアス問題を構造的に回避する。"""
    import collections
    grp = child.get("group")
    if grp is None:
        return []
    z = child["value"]
    by = collections.defaultdict(list)
    for i, g in enumerate(grp):
        if g >= 0:
            by[int(g)].append(i)
    pairs = []
    for g, idxs in by.items():
        got = 0
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                if got >= cap_per_group:
                    break
                a, b = idxs[ai], idxs[bi]
                if abs(z[a] - z[b]) > delta:
                    pairs.append((a, b, g) if z[a] > z[b] else (b, a, g))
                    got += 1
    return pairs


def pair_acc(vnet, child, pairs):
    """ペア順位の正答率（v(勝ち子盤面) > v(負け子盤面) の割合）。"""
    if not pairs:
        return float("nan")
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    rows = np.concatenate([ia, ib])
    pred = vnet.predict({k: child[k][rows] for k in ("scalars", "field", "card_idx")})
    m = len(pairs)
    return float((pred[:m] > pred[m:]).mean())


def rank_finetune(vnet, child, pairs, epochs=4, lr=2e-5, weight=1.0, margin=0.2,
                  batch_pairs=32):
    """子盤面ペアの順位ヒンジ max(0, margin−(v_a−v_b)) で value を微調整（v12.1）。

    実装は ValueNet の公開 API（forward/backward/step）のみ: backward の MSE 勾配
    dpred=(2/B)(pred−y) に対し y=pred±(B/2)·w を与えるとヒンジの ∓w 勾配に恒等変換される
    （コア無改修＝実験は scripts 層に留める）。活性ペア（margin 未達）のみ勾配を流す。"""
    rng = np.random.default_rng(17)
    for _ep in range(epochs):
        order = rng.permutation(len(pairs))
        for s in range(0, len(order), batch_pairs):
            sel = [pairs[k] for k in order[s:s + batch_pairs]]
            ia = np.array([p[0] for p in sel]); ib = np.array([p[1] for p in sel])
            rows = np.concatenate([ia, ib])
            batch = {k: child[k][rows] for k in ("scalars", "field", "card_idx")}
            pred, cache = vnet.forward(batch)
            m = len(sel)
            act = (pred[:m] - pred[m:]) < margin
            if not act.any():
                continue
            B = len(pred)
            y = pred.copy()
            y[:m][act] += (B / 2.0) * weight    # dpred_a = −weight（勝ち側を押し上げ）
            y[m:][act] -= (B / 2.0) * weight    # dpred_b = ＋weight（負け側を押し下げ）
            vnet.step(vnet.backward(cache, y), lr=lr)
    return vnet


def dead_weighted_pairs(pairs, dead_flags, k=3):
    """負け側が「不発PLAY」の順位ペアを k 倍に複製する（v33・pure）。

    m1@3 型（ON_PLAY 持ちを条件不成立で出す＝1枚損のバニラ設置）は一般ペアの海に薄まると
    教師信号が届かない（v32: 296群/1328ペアでも m1@3 の value 差 +0.01 を動かせず）。
    「不発を咎める」ペアだけ重み増しして信号を集中する。k=1 は恒等。"""
    if k <= 1:
        return list(pairs)
    out = list(pairs)
    for p in pairs:
        if dead_flags[p[1]]:                       # p=(勝ちidx, 負けidx, group)
            out.extend([p] * (int(k) - 1))
    return out


def rank_finetune_anchored(vnet, child, pairs, anchor, y_anchor, epochs=4, lr=2e-5,
                           weight=1.0, margin=0.2, batch_pairs=32,
                           anchor_scale=1.0, batch_anchor=192, rng_seed=17):
    """順位ヒンジ＋**蒸留アンカー**で value を微調整する（v33・rank_finetune の後継腕）。

    v32 の負の結果（3回再現）: アンカー無しの順位ヒンジは共有 value を歪め、一般オプション
    順位が上がるほど**防御窓の「素通しが正」較正（m2@12/58）が先に壊れる**。本関数は順位
    バッチごとに「アンカー盤面（dense コーパスの一般盤面）で base の予測値 y_anchor へ引き戻す
    MSE バッチ」を交互に流し、既存挙動を錘で固定したまま順位だけを動かす。

    実装は rank_finetune と同じく ValueNet の公開 API のみ（backward(cache, y) は MSE 勾配＝
    y=y_anchor を与えればそのまま蒸留）。anchor_scale が錘の強さ（lr への係数）。"""
    rng = np.random.default_rng(rng_seed)
    n_anchor = len(y_anchor)
    for _ep in range(epochs):
        order = rng.permutation(len(pairs))
        for s in range(0, len(order), batch_pairs):
            sel = [pairs[k] for k in order[s:s + batch_pairs]]
            ia = np.array([p[0] for p in sel]); ib = np.array([p[1] for p in sel])
            rows = np.concatenate([ia, ib])
            batch = {k: child[k][rows] for k in ("scalars", "field", "card_idx")}
            pred, cache = vnet.forward(batch)
            m = len(sel)
            act = (pred[:m] - pred[m:]) < margin
            if act.any():
                B = len(pred)
                y = pred.copy()
                y[:m][act] += (B / 2.0) * weight
                y[m:][act] -= (B / 2.0) * weight
                vnet.step(vnet.backward(cache, y), lr=lr)
            if n_anchor and anchor_scale > 0:
                ai = rng.integers(0, n_anchor, size=min(batch_anchor, n_anchor))
                ab = {k: anchor[k][ai] for k in ("scalars", "field", "card_idx")}
                _pred_a, cache_a = vnet.forward(ab)
                vnet.step(vnet.backward(cache_a, y_anchor[ai]), lr=lr * anchor_scale)
    return vnet


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
    ap.add_argument("--train-policy", action="store_true",
                    help="policy も微調整する（**既定OFF＝value のみ学習**）。v12 で確定: policy "
                         "微調整は1エポックでも対gen6 アリーナを 0.33 に落とし、value のみは "
                         "80局 0.4875＝無傷（ヌル対照 0.500 で計器健全性も確認）。v9 でも同結論"
                         "（2026-07-18）。新特徴で局面の区別を与えるまで policy 学習は封印")
    ap.add_argument("--skip-policy", action="store_true",
                    help="（後方互換・現在は value のみが既定のため冗長。--train-policy と併用時は"
                         "こちらが優先＝skip）")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--disagree-weight", type=float, default=1.0,
                    help="kind=disagree/diverge（反例）サンプルの policy 学習での複製倍率。1=無効")
    ap.add_argument("--extra-dirs", default=None,
                    help="ローカル教師バッチのディレクトリ（カンマ区切り・divergence_probe --out 等）")
    ap.add_argument("--rank-epochs", type=int, default=0,
                    help="子盤面ペア順位ヒンジの微調整エポック（v12.1・0=無効）。value 学習の後段で適用")
    ap.add_argument("--rank-lr", type=float, default=2e-5)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--rank-margin", type=float, default=0.2)
    ap.add_argument("--rank-delta", type=float, default=0.25,
                    help="ペア成立に要求する子盤面 z 差（レフェリー実測が明確に割れた点のみ教える）")
    ap.add_argument("--diverge-weight", type=float, default=None,
                    help="kind=diverge（乖離教師）専用の複製倍率。未指定は --disagree-weight に従う。"
                         "乖離教師は少数精鋭（候補の失敗分布から採った文脈つき反例）のため"
                         "高倍率で効かせる用途")
    ap.add_argument("--out", default=None, help="候補ネットの保存先（lr ごとのサブ名で保存）")
    ap.add_argument("--base", default="gen6",
                    help="温スタート元の同梱世代（既定 gen6=現既定ネット。gen5 で旧ベース比較）")
    ap.add_argument("--child-frac", type=float, default=0.0,
                    help="子盤面教師の train 混合率（既定0=不使用。全変種で有害の実測＝docs/reports/cpu_v11_child_label_20260723.md）")
    ap.add_argument("--child-pass-only", action="store_true",
                    help="子盤面教師を『手番が渡った初手』（TURN_END/PASS 系＝is_my_turn=0）のみに"
                         "絞る。攻撃/付与系の子盤面は族ベスト値＝楽観バイアスで有害（v11 実測）だが、"
                         "TURN_END は族ベスト＝実測値なので無害＝value の TURN_END 過大評価"
                         "（@64/@68 で +0.6〜0.7）を外科的に矯正する")
    args = ap.parse_args()

    vdata, pol = collect_ref_batches(
        extra_dirs=[d for d in (args.extra_dirs or "").split(",") if d])
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
    if ch is not None and args.child_frac > 0:
        # v11 子盤面教師は train へのみ併合（val は root 決定のみ＝前後比較の指標互換を維持）。
        # decide が比較する「初手後の子盤面」の序列を value に直接教える（@68/@93 の実測根拠）。
        # child_frac<1 は固定 seed の部分サンプル＝全量混合が @64/@137 を壊した実測の切り分け用。
        if args.child_pass_only:
            # is_my_turn（scalars index 11）=0 ＝手番が渡った子盤面（TURN_END/PASS 系）のみ。
            mask = ch["scalars"][:, 11] == 0.0
            ch = {k: v[mask] for k, v in ch.items()}
        n_ch = len(ch["value"])
        keep = np.random.default_rng(13).permutation(n_ch)[:int(round(n_ch * args.child_frac))]
        tr_vdata = {k: np.concatenate([tr_vdata[k], ch[k][keep]]) for k in tr_vdata}
        print(f"子盤面教師: {n_ch} 行中 {len(keep)} 行を train に併合"
              f"（frac={args.child_frac:g}{'・pass-only' if args.child_pass_only else ''}）")
    tr_pol = [pol[j] for j in tr]
    dw_dis = args.disagree_weight
    dw_dvg = args.diverge_weight if args.diverge_weight is not None else dw_dis
    if dw_dis > 1 or dw_dvg > 1:
        # 反例（disagree=採掘・diverge=乖離裁定 v12）を複製して policy 学習で重く効かせる
        # （policy_selfdistill と同じ手法）。kind 付きの新バッチのみ対象＝旧バッチ（"" 埋め）は等倍。
        # diverge は少数（数十点）のため専用倍率で効かせられる。
        _reps = {"disagree": max(int(round(dw_dis)) - 1, 0),
                 "diverge": max(int(round(dw_dvg)) - 1, 0)}
        extra = [tr_pol[j] for j in range(len(tr_pol))
                 for _ in range(_reps.get(tr_kind[j], 0))]
        n_dis = int((tr_kind == "disagree").sum()); n_dvg = int((tr_kind == "diverge").sum())
        tr_pol = tr_pol + extra
        print(f"反例重み付け: disagree {n_dis}×{dw_dis:g}・diverge {n_dvg}×{dw_dvg:g} "
              f"→ +{len(extra)} 複製")
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
                          distill_weight=args.distill_weight) if args.epochs > 0 else (0.0, 0.0)
        if args.rank_epochs > 0 and vdata.get("_child") is not None:
            child_all = vdata["_child"]
            pairs = build_rank_pairs(child_all, delta=args.rank_delta)
            p_tr = [p for p in pairs if p[2] % 7 != 0]   # group 単位で train/val（漏洩防止）
            p_va = [p for p in pairs if p[2] % 7 == 0]
            acc0 = pair_acc(vnet, child_all, p_va)
            rank_finetune(vnet, child_all, p_tr, epochs=args.rank_epochs, lr=args.rank_lr,
                          weight=args.rank_weight, margin=args.rank_margin)
            acc1 = pair_acc(vnet, child_all, p_va)
            print(f"rank微調整: pairs {len(p_tr)}tr/{len(p_va)}va・val順位正答 "
                  f"{acc0:.3f}→{acc1:.3f}")
        if args.skip_policy or not args.train_policy:
            ce = float("nan")   # policy はベース据え置き（value のみ＝v12 確定の既定）
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
