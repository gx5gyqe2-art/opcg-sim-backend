"""同じ局面のドン ±1 の対で、価値に差そのものを教える損失（WP `pair-diff-loss`・`docs/search_efficiency_plan.md` §7.88・
台帳 2+534）。**試し訓練専用**（出荷・Rust・探索は変えない）。

対の npz（`tests/scripts/pair_diff.py merge` が書く）は、同じ盤面の「ドンが 1 枚少ない側（l）」と「多い側（m）」の符号化と、
両側から同じ配り直し 16 本で打ち続けた勝敗の差（価値の目盛り・``d``＝多い側 − 少ない側）を持つ。
損失は価値の損失（勝敗）に **対の予測の差 − 教師の差** の二乗を足す（対の盤面そのものの値は教えない）:

    損失 = 0.5·mean_行((v − z)²) ＋ 倍率 × 0.5·Σ_対((v_m − v_l) − d)² ／ 1 回の価値の行数

**倍率（選ぶ定数なし・§7.83 と同じ決め方）**: 1 周で「対の損失の合計＝それ以外の価値の損失の合計」になる一律の倍率。
訓練の対は 1 周に 1 回ずつ、価値の更新に均等に割り振る（主の並びは動かさない・対は別の乱数列）。訓練を始める前の
ネット（warm-start）で、訓練の価値の行の二乗誤差の平均 ``L_v`` と訓練の対の差の二乗誤差の平均 ``L_d`` を測り、

    倍率 = (訓練の価値の行数 × L_v) ／ (訓練の対の数 × L_d)

にする（行数の比だけの倍率 ``訓練の価値の行数 ／ 訓練の対の数`` も印字する・``count`` はこちらを使う）。

**取り置き**: 対の局 seed % holdout_mod == 0 の対は訓練に使わない（価値の取り置きと同じ局）。
``shapes`` で訓練に使う形を絞れる（形を越えて移るかの腕）。

**既定は 1 ビットも変わらない**: 訓練器の ``--pair-diff`` が無ければ、どの関数も呼ばれず、価値の更新に何も足さない
（`tests/test_pair_diff.py`）。torch backend 専用。
"""
import numpy as np

#: 対の npz の列（l＝少ない側・m＝多い側は同じ位置で対になる）
PAIR_KEYS = ("sc_l", "ci_l", "tok_l", "sc_m", "ci_m", "tok_m", "d", "seed", "shape")
#: balanced＝損失の合計の釣り合い（第一）・count＝行数の比だけ（§7.83 と同じ数え方・損失の大きさを見ない）
PAIR_DIFF_MODES = ("balanced", "count")
#: 評価だけの形（対を作っていない形として読む＝訓練には使わない・取り置きにも訓練の局にも測る）
EVAL_SHAPES = ("mid",)


def load_pairs(paths, holdout_mod=7, shapes=None):
    """対の npz（複数可）→ 連結した dict。``tr``／``va``＝訓練に使う対・取り置きの対の添字。
    評価だけの形（``EVAL_SHAPES``）は訓練に使わない。``shapes`` を渡すと訓練に使う対をその形に絞る（取り置きは全部の形を残す＝評価用）。"""
    parts = []
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            parts.append({k: np.asarray(d[k]) for k in PAIR_KEYS})
    out = {k: np.concatenate([x[k] for x in parts]) for k in PAIR_KEYS}
    hold = out["seed"] % holdout_mod == 0
    use = ~hold & ~np.isin(out["shape"], EVAL_SHAPES)
    if shapes:
        use &= np.isin(out["shape"], list(shapes))
    out["tr"] = np.flatnonzero(use)
    out["va"] = np.flatnonzero(hold)
    out["ev"] = np.flatnonzero(~hold & np.isin(out["shape"], EVAL_SHAPES))      # 訓練の局の評価だけの形
    return out


def take(PR, ix):
    """対の行を [少ない側 B 本, 多い側 B 本] に並べる（float32／int64）。"""
    sc = np.concatenate([PR["sc_l"][ix], PR["sc_m"][ix]]).astype(np.float32)
    ci = np.concatenate([PR["ci_l"][ix], PR["ci_m"][ix]]).astype(np.int64)
    tok = np.concatenate([PR["tok_l"][ix], PR["tok_m"][ix]]).astype(np.float32)
    return sc, ci, tok


def schedule(tr, nv, rng):
    """訓練の対を 1 周に 1 回ずつ、価値の更新 ``nv`` 回に**周の全体へ均等に**割り振る → 長さ ``nv`` の添字の配列の列。
    i 番目（並べ替えた後）の対は更新 floor(i·nv／対の数) 回目に入る（対が更新より少なくても周の頭に寄らない）。"""
    perm = np.asarray(tr)[rng.permutation(len(tr))]
    pos = (np.arange(len(perm), dtype=np.int64) * int(nv)) // max(len(perm), 1)
    return np.split(perm, np.searchsorted(pos, np.arange(1, int(nv))))


def balanced_factor(n_value_rows, n_pairs, l_v, l_d):
    """1 周で対の損失の合計＝それ以外の価値の損失の合計になる倍率（行数と、始める前の損失の大きさから）。"""
    if n_pairs <= 0 or l_d <= 0:
        raise ValueError("--pair-diff balanced: 訓練の対が無いか、対の損失が 0")
    return float(n_value_rows * l_v / (n_pairs * l_d)), float(n_value_rows / n_pairs)


def diff_terms(v, B, d):
    """`v` は [少ない側 B 本, 多い側 B 本] の価値（torch）。対ごとの (予測の差 − 教師の差)²。"""
    return ((v[B:] - v[:B]) - d) ** 2


def predict_diff(value_fn, PR, ix, bs=256):
    """numpy／torch どちらでも: ``value_fn(sc, ci, tok) → 価値の配列``。対ごとの予測の差（多い − 少ない）。"""
    out = []
    for s in range(0, len(ix), bs):
        j = ix[s:s + bs]
        sc, ci, tok = take(PR, j)
        v = np.asarray(value_fn(sc, ci, tok), np.float64)
        out.append(v[len(j):] - v[:len(j)])
    return np.concatenate(out) if out else np.zeros(0)


def fit_stats(dv, dt):
    """差の当てはまり: 二乗誤差・相関・傾き（予測の差を教師の差に回帰）・平均。"""
    dv = np.asarray(dv, np.float64)
    dt = np.asarray(dt, np.float64)
    ok = len(dv) > 2 and np.std(dv) > 0 and np.std(dt) > 0
    return {"n": int(len(dv)), "d_mse": round(float(np.mean((dv - dt) ** 2)), 5) if len(dv) else None,
            "corr": round(float(np.corrcoef(dv, dt)[0, 1]), 4) if ok else None,
            "slope": round(float(np.dot(dv - dv.mean(), dt - dt.mean()) / max(np.dot(dt - dt.mean(), dt - dt.mean()), 1e-12)), 4) if ok else None,
            "dv_mean": round(float(dv.mean()), 4) if len(dv) else None,
            "dt_mean": round(float(dt.mean()), 4) if len(dt) else None}
