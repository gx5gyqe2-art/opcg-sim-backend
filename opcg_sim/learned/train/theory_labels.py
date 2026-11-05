"""見える情報だけの理論の列を価値のネットの入力に足す sidecar（WP `theory-input`・`docs/search_efficiency_plan.md` §7.82・台帳 2+522）。

問い: 遅く効く差（ドン・札・展開の要）は理論がすでに価格として持っている。その読み（2 本の時計・交点の勝率・耐久・
速さ・価格の和）を入力の列にすれば、ネットは遅い方向を自分で見つける必要がなくなるか。列は Rust の
`theory::core::drv_vis`（`tests/scripts/theory_rs.visible_cols`）が **打つ側に見える情報だけ**（その行の符号化と
両席のデッキ）から出す＝記録の行がそのまま盤面（`live_theory` の「生＝記録」）なので、局を打ち直さずに全行で出せる。

sidecar は記録のシャードの隣の ``n_record_XXXXX.theory.npz``（`rollout_labels`／`time_labels` と同じ流儀・全行ぶん）:
  ``th``   (N, D_TH float16) … 列（`COLS` の順・Rust の `VIS_COLS` と同じ）
  ``seed``／``step`` (N int64) … 検算用（シャードの同じ行と一致しなければ止める）
  ``cols`` … 列の名前（検算用）
作るのは ``tests/scripts/theory_input.py sidecar``。訓練器は ``n_rel_train --theory-in`` で読み、
`V["sc"]` の後ろに列を足した束（`ScTheory`）に差し替える（**既定は読まない＝今までと 1 ビットも同じ**）。
使う波に sidecar の無いシャードが 1 つでもあれば止める（列 0 で埋めない＝使う波を揃える）。
"""
import os

import numpy as np

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned.train import dump_io as DIO

SIDECAR_SUFFIX = ".theory.npz"
COLS = NL.TH_COLS
D_TH = NL.D_TH


def sidecar_path(shard_path):
    """`.../n_record_00012.npz` → `.../n_record_00012.theory.npz`。"""
    return os.path.splitext(shard_path)[0] + SIDECAR_SUFFIX


def write_sidecar(shard_path, th, seed, step):
    th = np.asarray(th, np.float32)
    if th.shape[1] != D_TH:
        raise ValueError(f"列数が違う: {th.shape[1]} != {D_TH}")
    out = sidecar_path(shard_path)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, th=th.astype(np.float16), seed=np.asarray(seed, np.int64), step=np.asarray(step, np.int64),
                        cols=np.array(COLS))
    os.replace(tmp, out)
    return out


def load_theory(src_dirs, z_dirs, n_rows, cache_dir=None):
    """`dump_io.load_dump` の V と**同じ行の並び**で sidecar を読む → 列 [N, D_TH] float16。

    並びは pack と同じ（`--in` の後に `--z-in`・各ディレクトリの `shard_files` の順）。sidecar の無いシャードが
    あれば ValueError（列 0 で埋めない）。行の seed・step がシャードと合わなければ ValueError。
    `cache_dir` を渡すと列をそこの `.npy`（memmap）に置いて返す＝RAM に 10.7M 行 × 25 列を持たない（訓練の
    メモリ上限 14GB の余裕のため・2026-11-05 の 2 周目の OOM）。鍵は sidecar の一覧とサイズと mtime。"""
    dirs = [d for d in list(src_dirs) + list(z_dirs) if DIO.shard_files(d)]
    files = [f for d in dirs for f in DIO.shard_files(d)]
    if cache_dir:
        import hashlib
        h = hashlib.sha1()
        for f in files:
            sp = sidecar_path(f)
            if os.path.exists(sp):
                st = os.stat(sp)
                h.update(f"{sp}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"theory_{n_rows}_{h.hexdigest()[:16]}.npy")
        if os.path.exists(path):
            m = np.load(path, mmap_mode="r")
            if m.shape == (n_rows, D_TH):
                return m
        th = np.lib.format.open_memmap(path + ".tmp.npy", mode="w+", dtype=np.float16, shape=(n_rows, D_TH))
    else:
        th = np.zeros((n_rows, D_TH), np.float16)
    off = 0
    missing = []
    for f in files:
        sp = sidecar_path(f)
        with np.load(f, allow_pickle=True) as dd:
            n = int(dd["z"].shape[0])
            seed = np.asarray(dd["seed"][:n], np.int64)
            step = np.asarray(dd["step"][:n], np.int64)
        if not os.path.exists(sp):
            missing.append(sp)
            off += n
            continue
        with np.load(sp) as ds:
            x = np.asarray(ds["th"], np.float16)
            if x.shape != (n, D_TH):
                raise ValueError(f"{sp}: 形がシャードと合わない {x.shape} != {(n, D_TH)}")
            if tuple(ds["cols"]) != COLS:
                raise ValueError(f"{sp}: 列の並びが違う")
            if not (np.array_equal(seed, ds["seed"]) and np.array_equal(step, ds["step"])):
                raise ValueError(f"{sp}: seed・step がシャードの行と合わない")
            th[off:off + n] = x
        off += n
    if off != n_rows:
        raise ValueError(f"sidecar の並びが V と合わない（{off} != {n_rows}）")
    if missing:
        raise ValueError(f"sidecar（*.theory.npz）の無いシャードが {len(missing)} 本（例 {missing[0]}）＝使う波を揃える")
    if cache_dir:
        th.flush()
        del th
        os.replace(path + ".tmp.npy", path)
        return np.load(path, mmap_mode="r")
    return th


class ScTheory(DIO._Waves):
    """`V["sc"]`（memmap の view）の後ろに理論の列を足した束に見せる view（切り出しは普通の ndarray・書き込みはしない）。

    `_Waves` を継承するのは、訓練器の行の切り出し（`n_rel_torch._Rows`・`dump_io.rows_f32`）が memmap の view を
    丸ごと RAM に載せない経路で読むため。"""

    def __init__(self, sc, th):
        self.base = sc
        self.th = th if isinstance(th, np.memmap) else np.asarray(th, np.float16)
        if len(self.th) != len(sc):
            raise ValueError(f"行数が合わない {len(self.th)} != {len(sc)}")
        self.parts = []
        self.cols = None
        self.offsets = np.array([0, len(sc)], np.int64)
        self.shape = (len(sc), int(sc.shape[1]) + D_TH)
        self.dtype = np.dtype(np.float16)
        self.ndim = 2
        self.nbytes = int(getattr(sc, "nbytes", 0)) + int(self.th.nbytes)

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        a = np.concatenate([np.asarray(self.base, np.float16), self.th], 1)
        return a.astype(dtype) if dtype is not None else a

    def __getitem__(self, key):
        if isinstance(key, tuple):
            head, rest = key[0], key[1:]
            return self[head][(slice(None),) + rest]
        if isinstance(key, (int, np.integer)):
            return np.concatenate([np.asarray(self.base[key], np.float16), self.th[key]])
        a = np.asarray(self.base[key], np.float16)
        return np.concatenate([a, np.asarray(self.th[key], np.float16)], 1)


def attach(V, th):
    """V の `sc` を理論の列つきの束に差し替える（他の列は触らない）。"""
    V["sc"] = ScTheory(V["sc"], th)
    return V
