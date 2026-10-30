"""「次のターン以降に効く差」を価値に教える 2 案の訓練の足し（WP `value-teach`・
`docs/search_efficiency_plan.md` §7.71・台帳 2+485）。**試し訓練専用**（出荷・Rust・探索は変えない）。

前段（§7.69／§7.70）: ネットはドン 1 枚の差を、どの時点でも打ち続けた差の 3〜6 割にしか見ない。
6キングだけの場で「ロックス海賊団で戻す」は打ち比べで +0.095 なのに根の評価は CPU の手より 0.033 低い。
読み: 価値の教師が局の勝敗だけで粗く、しかも自己対戦の教材に対照（同じ盤面から別の手を打った続き）が無い。

**案 A（資源の先読みの補助の頭・`--res-weight`）**: 既存の補助の頭（§20.8.2 の `aux`＝次の 1 ターンの
ライフ・攻撃）と時間軸の頭（§20.11 の `time_labels`）には**ドンと手札が無い**。同じ後付けの sidecar の流儀で、
各行から見た「次の自分のターン開始時」と「2 つ先の自分のターン開始時」の資源を補助の頭に当てさせる。
列の正本は ``RES_COLS``（ここ 1 か所）。sidecar は ``n_record_XXXXX.res.npz``（`res(D,20 float16)`／
`res_mask(D int8)`）。`dump_io` の pack は触らない＝ここで pack と同じ順に読み、行を揃える（``load_res``）。

**案 B（対照の対で差を教える・`--pair-in`）**: 打ち比べの記録（同じ盤面から手 a と手 b を打ち、16 通りずつ
最後まで打った勝率）を当て直して作った「手 a・手 b を当てた直後の盤面」の対（``tests/scripts/value_teach.py pairs``
が書く npz）で、価値（打ち続けた勝率を価値の目盛り 2w−1 に直したもの）と対の差を教える。
損失＝``pair_value_weight``×0.5·mean((v−t)²)＋``pair_lambda``×0.5·mean(((v_b−v_a)−(t_b−t_a))²)。
価値の更新 ``pair_every`` 回ごとに 1 回、対を ``pair_bs`` 組だけ別の乱数列で引いて更新する（主の並びは動かない）。

**既定は 1 ビットも変わらない**: ``--res-weight 0``・``--pair-in`` 無しなら、どの関数も呼ばれず、
optimizer の持つパラメータも乱数列も今までと同じ（`tests/test_value_teach.py` が確かめる）。
どちらも **torch backend 専用**（numpy backend で立てたら止める）。

CLI（sidecar を書く・冪等）:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.value_teach label --in ~/w60/n_records/n60_w*_p[0-3]
"""
import argparse
import os
import sys
import time

import numpy as np

from opcg_sim.learned.train import plan_labels as PL

# ---------------------------------------------------------------------------
# 案 A: 資源の先読みの教師（sidecar）
# ---------------------------------------------------------------------------
#: 1 時点ぶんの列（自分／相手の順）。目盛りは他の補助教師と同じく 0〜数の範囲に収める。
RES_ITEMS = ("don", "hand", "power", "count", "big")
#: 列の正本。``p1``＝次の自分のターン開始時・``p2``＝2 つ先の自分のターン開始時。
RES_COLS = tuple(f"{side}_{it}_{h}" for h in ("p1", "p2") for it in RES_ITEMS for side in ("my", "opp"))
D_RES = len(RES_COLS)                 # 20
SIDECAR_SUFFIX = ".res.npz"
#: 目盛り（ドン・手札は /10・体数・大型は /5・パワーは符号化と同じ /10000 の合計のまま）
DON_SCALE = 10.0
HAND_SCALE = 10.0
COUNT_SCALE = 5.0
#: 「大型」の線（コスト 6 以上・6キング〔6 コスト〕が入る線）
BIG_COST = 6
ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len")
# 符号化の位置（`encoder.py` の v1 の列・`n_rel_feat` のトークン S の列・枠の並び）
_SC_DON = (2, 3, 4, 5)                # 自アクティブ・自レスト・相手アクティブ・相手レスト
_SC_HAND = (6, 7)
_SC_COUNT = (8, 9)
_OWN_UNITS = (0, 2, 3, 4, 5, 6)       # 自リーダー＋自場 5 枠（付与ドンを数える枠）
_OPP_UNITS = (1, 7, 8, 9, 10, 11)
_OWN_FIELD = (2, 3, 4, 5, 6)
_OPP_FIELD = (7, 8, 9, 10, 11)
_S_POWER, _S_COST, _S_DON = 0, 1, 2


def sidecar_path(shard_path):
    """`.../n_record_00012.npz` → `.../n_record_00012.res.npz`。"""
    return os.path.splitext(shard_path)[0] + SIDECAR_SUFFIX


def snapshot(sc, tok, ci):
    """1 行の盤面（その行の席の視点）→ 自分／相手の資源 10 列（``RES_ITEMS`` × 自分・相手の順）。"""
    sc = np.asarray(sc, np.float32); tok = np.asarray(tok, np.float32); ci = np.asarray(ci)
    on = lambda slots: [s for s in slots if int(ci[s]) != 0]  # noqa: E731
    out = []
    for mine in (True, False):
        units = _OWN_UNITS if mine else _OPP_UNITS
        field = on(_OWN_FIELD if mine else _OPP_FIELD)
        don = sc[_SC_DON[0] if mine else _SC_DON[2]] + sc[_SC_DON[1] if mine else _SC_DON[3]]
        don += float(sum(tok[s, _S_DON] for s in on(units))) * 5.0
        hand = sc[_SC_HAND[0] if mine else _SC_HAND[1]]
        power = float(sum(tok[s, _S_POWER] for s in field))
        count = sc[_SC_COUNT[0] if mine else _SC_COUNT[1]]
        big = sum(1 for s in field if tok[s, _S_COST] * 10.0 >= BIG_COST - 1e-3)
        out.append((don / DON_SCALE, hand / HAND_SCALE, power, count / COUNT_SCALE, big / COUNT_SCALE))
    my, op = out
    return [v for pair in zip(my, op) for v in pair]          # my_don, opp_don, my_hand, …


def label_game(rows, snaps, idx):
    """対局 → 行ごとの (res [n,20] float32, mask [n] int8)。

    `snaps[i]` は行 i の盤面の ``snapshot``（その行の席の視点）。各行の席 w について、
    今より**後**の自分のターンの最初の main 行の盤面を読む（自分の手番の行でも今のターンは入れない）。
    2 つ先が無ければ 1 つ先を使う（`time_labels` の「無ければ最後」と同じ流儀）。
    mask=1 は「次の自分のターンがある・ターン 1 以降」の行。勝敗は使わない（起きたことだけ）。
    """
    n = len(idx)
    out = np.zeros((n, D_RES), np.float32)
    mask = np.zeros(n, np.int8)
    head = {}
    for i in idx:
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        if PL.is_own_turn(w, t) and int(rows["kind"][i]) == 0 and (w, t) not in head:
            head[(w, t)] = i
    turns_of = {w: sorted(t for (ww, t) in head if ww == w) for w in (0, 1)}
    half = D_RES // 2
    for d, i in enumerate(idx):
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        if t < 1:
            continue
        fut = [tt for tt in turns_of[w] if tt > t]
        if not fut:
            continue
        out[d, :half] = snaps[head[(w, fut[0])]]
        out[d, half:] = snaps[head[(w, fut[1] if len(fut) > 1 else fut[0])]]
        mask[d] = 1
    return out, mask


def _snaps(dd, n):
    sc = np.asarray(dd["scalars"])[:n].astype(np.float32)
    tok = np.asarray(dd["tokens"])[:n].astype(np.float32)
    ci = np.asarray(dd["card_idx"])[:n]
    return np.array([snapshot(sc[i], tok[i], ci[i]) for i in range(n)], np.float32).reshape(n, -1)


def build_sidecar(shard_path, force=False):
    """1 シャードの sidecar を書く。戻り値は (書いたか, 行数, mask=1 の行数)。"""
    out_path = sidecar_path(shard_path)
    if os.path.exists(out_path) and not force:
        return False, 0, 0
    rs = msk = None
    for _f, rows, _pol, snaps in PL.iter_shards([], row_cols=ROW_COLS, pol_cols=(),
                                                extra_fn=_snaps, files=[shard_path]):
        n = len(rows["z"])
        rs = np.zeros((n, D_RES), np.float32)
        msk = np.zeros(n, np.int8)
        _L, _ptr, games = PL.games_of(rows)
        for idx in games:
            r_, m_ = label_game(rows, snaps, idx)
            rs[idx] = r_
            msk[idx] = m_
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, res=rs.astype(np.float16), res_mask=msk, cols=np.array(RES_COLS))
    os.replace(tmp, out_path)
    return True, len(msk), int(msk.sum())


def load_res(src_dirs, z_dirs, n_rows):
    """`dump_io.load_dump` の V と**同じ行の並び**で sidecar を読む → (res [N,20] f16, mask [N] i8)・
    どのシャードにも sidecar が無ければ (None, None)。並びは pack と同じ（波の順・`shard_files` の順）。"""
    from opcg_sim.learned.train import dump_io as DIO
    dirs = [d for d in list(src_dirs) + list(z_dirs) if DIO.shard_files(d)]
    files = [f for d in dirs for f in DIO.shard_files(d)]
    if not any(os.path.exists(sidecar_path(f)) for f in files):
        return None, None
    res = np.zeros((n_rows, D_RES), np.float16)
    mask = np.zeros(n_rows, np.int8)
    off = 0
    for f in files:
        with np.load(f, allow_pickle=True) as dd:
            n = int(dd["z"].shape[0])
        sp = sidecar_path(f)
        if os.path.exists(sp):
            with np.load(sp) as ds:
                r = np.asarray(ds["res"], np.float16); m = np.asarray(ds["res_mask"], np.int8)
            if len(r) != n:
                raise ValueError(f"{sp}: 行数がシャードと合わない（{len(r)} != {n}）")
            res[off:off + n] = r
            mask[off:off + n] = m
        off += n
    if off != n_rows:
        raise ValueError(f"sidecar の並びが V と合わない（{off} != {n_rows}）")
    return res, mask


# ---------------------------------------------------------------------------
# 案 B: 対照の対（npz・`tests/scripts/value_teach.py pairs` が書く）
# ---------------------------------------------------------------------------
#: 対の npz の列（a の行と b の行は同じ位置で対になる）
PAIR_KEYS = ("sc_a", "ci_a", "tok_a", "sc_b", "ci_b", "tok_b", "t_a", "t_b", "game", "src")
#: 対の holdout（局の鍵の文字列の和 % 7 == 0）
PAIR_HOLDOUT_MOD = 7


def load_pairs(paths):
    """対の npz（複数可）→ 連結した dict。``t_a``／``t_b`` は価値の目盛り（2w−1）。"""
    parts = []
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            parts.append({k: np.asarray(d[k]) for k in PAIR_KEYS})
    out = {k: np.concatenate([x[k] for x in parts]) for k in PAIR_KEYS}
    out["hold"] = np.array([sum(map(ord, str(g))) % PAIR_HOLDOUT_MOD == 0 for g in out["game"]])
    return out


# ---------------------------------------------------------------------------
# torch（訓練器から呼ぶ）
# ---------------------------------------------------------------------------
#: 案 A の頭のパラメータ名（既存の名前と重ねない・npz には焼かない＝別ファイル ``<out>.res_head.npz``）
RES_PARAMS = ("Wn1", "bn1", "Wn2", "bn2")
RES_HIDDEN = 64
RES_INIT_SEED = 29


def init_res_params(d_e, seed=RES_INIT_SEED):
    """案 A の頭の初期値（別の乱数列＝既存の初期値は動かない）。"""
    rng = np.random.default_rng(seed)
    return {"Wn1": (rng.standard_normal((d_e, RES_HIDDEN)) * np.sqrt(2.0 / d_e)).astype(np.float32),
            "bn1": np.zeros(RES_HIDDEN, np.float32),
            "Wn2": (rng.standard_normal((RES_HIDDEN, D_RES)) * np.sqrt(1.0 / RES_HIDDEN)).astype(np.float32),
            "bn2": np.zeros(D_RES, np.float32)}


def res_loss_terms(pred, res, mask):
    """案 A の損失＝有効行の全要素平均 Huber（`time_loss_terms` と同じ式）。戻り値は (損失, 有効行数)。"""
    import torch
    n = int(mask.sum())
    if n <= 0:
        return pred.sum() * 0.0, 0
    per = torch.nn.functional.huber_loss(pred, res, reduction="none", delta=1.0)
    return (per * mask[:, None]).sum() / (n * pred.shape[1]), n


def pair_loss_terms(v, t_a, t_b, value_weight, lam):
    """案 B の損失。`v` は [a の行 B 本, b の行 B 本] の価値。戻り値は (損失, 価値の mse, 対の差の mse)。"""
    B = t_a.shape[0]
    import torch
    t = torch.cat([t_a, t_b])
    mse = ((v - t) ** 2).mean()
    dd = ((v[B:] - v[:B]) - (t_b - t_a)) ** 2
    dmse = dd.mean()
    return value_weight * 0.5 * mse + lam * 0.5 * dmse, mse, dmse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="sidecar（*.res.npz）を書く・冪等")
    lb.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    lb.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    nw = ns = nr = nk = 0
    for d in args.src:
        for f in PL.shard_files(d):
            wrote, n, ok = build_sidecar(f, force=args.force)
            if not wrote:
                ns += 1
                continue
            nw += 1; nr += n; nk += ok
    print(f"res sidecar 書いた {nw}・飛ばした {ns}・行 {nr}・有効 {nk}（{time.time() - t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
