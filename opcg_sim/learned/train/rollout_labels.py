"""打ち続けの平均を価値の教師にする sidecar（WP `rollout-teach`・`docs/search_efficiency_plan.md` §7.80・台帳 2+518）。

前段（§7.79）: 同じ盤面から見えない札を配り直して、両席とも読み 16 回・世界 1 本で 16 局打ち続けた勝敗の平均は、
ドン 1 枚の差を読み 160 回と同じだけ残し、雑音は勝敗 1 局の 0.18 倍。ここではその平均を、選んだ行だけ
勝敗の代わりに価値の教師にする（残りの行は勝敗のまま・混ぜる割合などの新しい定数は無い）。

sidecar は記録のシャードの隣の ``n_record_XXXXX.roll.npz``（`time_labels` と同じ流儀）:
  ``row``  (M int32)   … シャードの中の行の番号（pack の行の並び＝シャードの行の並び）
  ``mean`` (M float32) … 打ち続けた勝敗（その行の手番の席から見て ±1）の平均＝価値の目盛り
  ``n``    (M int16)   … 平均に使った局数（決着しなかった局は数えない）
  ``seed``／``step`` (M int64) … 検算用（シャードの同じ行の seed・step と一致しなければ止める）
作るのは ``tests/scripts/rollout_teach.py sidecar``。訓練器は ``n_rel_train --z-mode rollout`` で読む
（**既定の ``--z-mode z`` では読まない＝今までと 1 ビットも同じ**）。
"""
import os

import numpy as np

SIDECAR_SUFFIX = ".roll.npz"
KEYS = ("row", "mean", "n", "seed", "step")


def sidecar_path(shard_path):
    """`.../n_record_00012.npz` → `.../n_record_00012.roll.npz`。"""
    return os.path.splitext(shard_path)[0] + SIDECAR_SUFFIX


def write_sidecar(shard_path, row, mean, n, seed, step):
    out = sidecar_path(shard_path)
    order = np.argsort(np.asarray(row), kind="stable")
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, row=np.asarray(row, np.int32)[order], mean=np.asarray(mean, np.float32)[order],
                        n=np.asarray(n, np.int16)[order], seed=np.asarray(seed, np.int64)[order],
                        step=np.asarray(step, np.int64)[order])
    os.replace(tmp, out)
    return out


def load_rollout(src_dirs, z_dirs, n_rows, min_n=1):
    """`dump_io.load_dump` の V と**同じ行の並び**で sidecar を読む → (平均 [N] float32, 印 [N] bool)。

    並びは pack と同じ（`--in` の後に `--z-in`・各ディレクトリの `shard_files` の順）。印は「sidecar に行があり、
    局数が `min_n` 以上」。行の seed・step がシャードと合わなければ ValueError（並びの取り違えを止める）。"""
    from opcg_sim.learned.train import dump_io as DIO
    dirs = [d for d in list(src_dirs) + list(z_dirs) if DIO.shard_files(d)]
    files = [f for d in dirs for f in DIO.shard_files(d)]
    mean = np.zeros(n_rows, np.float32)
    mask = np.zeros(n_rows, bool)
    off = 0
    for f in files:
        sp = sidecar_path(f)
        with np.load(f, allow_pickle=True) as dd:
            n = int(dd["z"].shape[0])
            if os.path.exists(sp):
                seed = np.asarray(dd["seed"][:n], np.int64)
                step = np.asarray(dd["step"][:n], np.int64)
        if os.path.exists(sp):
            with np.load(sp) as ds:
                r = np.asarray(ds["row"], np.int64)
                m = np.asarray(ds["mean"], np.float32)
                k = np.asarray(ds["n"], np.int64)
                if len(r) and (r.min() < 0 or r.max() >= n):
                    raise ValueError(f"{sp}: 行の番号がシャードの外（{n} 行）")
                if not (np.array_equal(seed[r], ds["seed"]) and np.array_equal(step[r], ds["step"])):
                    raise ValueError(f"{sp}: seed・step がシャードの行と合わない")
            ok = k >= min_n
            mean[off + r[ok]] = m[ok]
            mask[off + r[ok]] = True
        off += n
    if off != n_rows:
        raise ValueError(f"sidecar の並びが V と合わない（{off} != {n_rows}）")
    return mean, mask


def apply_teacher(z, mean, mask):
    """印の立つ行だけ勝敗を平均に差し替える → (教師 float32, 差し替えた行数)。印の無い行は勝敗のまま。"""
    out = np.asarray(z, np.float32).copy()
    m = np.asarray(mask, bool)
    out[m] = np.asarray(mean, np.float32)[m]
    return out, int(m.sum())
