"""dump_io: 棋譜ダンプ（v2／v3／v4）の読み込み（memmap 版・2026-09-07・計画 §18.5）。

`load_dump` は今までの `n_rel_train.load_dump_v2` と**同じ V/P/C の dict** を返す。違いは常駐:

- **V の `tok`/`sc`/`ci`/`z` は RAM に載せない**。波（＝dump のディレクトリ）ごとに
  `cache_dir` の下へ float16／float16／int16／float16 の `.npy` を 1 度だけ書き出し
  （**pack**）、以後は `np.load(mmap_mode="r")` の memmap を束ねた view（`_Waves`）を返す。
  切り出しの形（`V["tok"][bi]`）は変えない＝訓練ループは無改造で動く。
- pack は**シャードの一覧・サイズ・mtime で鍵付け**する（`_wave_key`）。同じ波を後で読み直しても
  作り直さない。波を足しても既存の波の pack は使い回す（鍵はディレクトリごと）。
- **v2 の npz（float32／int64）も同じ関数で読める**。pack を書くときに cast するだけで、
  v3 の npz（既に float16／int16）と**同じ pack になる**（`test_dump_io.py` が両方を照合）。

`P`（方策点）と `C`（候補）は今までどおり RAM（1 行あたり ~140B・`V` の 6% 弱）。方策列は
npz からしか作れない（`pol_sig` の JSON を語彙 index に潰す＝`vocab` 依存）ので pack に入れない。
`z_dirs`（z 専用・π を読まない）は pack があれば npz を 1 バイトも触らずに済む。

**dtype の申告**: 切り出した配列は float16／int16 のまま返る。**使う側で float32 へ上げる**
（`rows_f32`／`n_rel_train.prow`）。fp16 の丸めが forward に与える差は 1 バッチ最大 1.07e-4
（`docs/reports/2026-09-07_train_profile.md` §3）。

**注意（全波規模）**: 切り出しはページキャッシュに乗っている間だけ速い（1 波 100MB 級なら常に
乗る）。全波 5GB 超では実ディスク I/O になり得る＝**時間ではなくメモリのための手段**。

**dump v4 の追加列（2026-09-10・計画 §20.8）**: 補助教師 `aux(D,10)`／`aux_tok(D,6,3)`／
`aux_mask(D)` を pack へ足した（1 行 57B＝`tok` の 6%）。**無い波（v3）は 0・`aux_mask=0`
で埋める**＝波を混ぜても行が揃い、補助損失には入らない。どの波も持っていなければ
`load_dump` は `V["aux"]`／`V["aux_tok"]`／`V["aux_mask"]` に **`None`** を返す（＝訓練側は
補助ヘッドを自動で切る）。`deck_kinds`（WP `rs-removal-decks`）のような文字列列は pack に
入らない（`load_row_col` が npz から直接、pack と同じ行順で読む）。`forced(D int8)`
（WP `rs-eps-explore`・§20.8.5）も同じ扱い＝**pack に入れない・訓練は読まない**（教師では
なく層別の印。読みたいときは `load_row_col(dirs, "forced")`・無い波は `None`）。

**改良方策の材料（2026-09-10・計画 §20.6.1）**: `C["n"]`（訪問の生値）・`C["q"]`（行動価値）・
`C["p"]`（生成役ネットの根の P）と `P["v0"]`（根の価値の推定）も返す。`n`／`q` はどの波にも
ある（dump v2 から）。`p`／`v0` は **dump v4 の追加列**（`pol_p`／`pol_v0`）＝**1 本でも
持たないシャードがあれば `None`**（＝訓練側は warm-start ネットの forward に退避する）。
`C["pi"]`（訪問分布）は**今までどおり**＝既定の教師は 1 bit も変わらない。

**符号化 v14 の 0 埋め（2026-09-11・計画 §20.9 の E・`PACK_VERSION` 3）**: v14 は v13 の列の
**末尾に足しただけ**（`tokens` 22×20→22×22・`scalars` 123→127）。pack は**常に現行の形**で
書き、古い波（v13）は新しい列を **0**（＝情報なし。mask は要らない）で埋める。これで v13 と
v14 の波を 1 本の `_Waves` に束ねられる＝`n_rel_train` は無改造で混ぜて回せる。

**方針ラベル（2026-09-12・計画 §20.10・WP `rs-plan-aux`・`PACK_VERSION` 4）**: シャードの隣の
sidecar `n_record_XXXXX.plan.npz`（`plan_labels.py` が既存の記録から後付けで書く・`plan(D int8)`・
-1＝無し）を pack の `plan` 列に取り込む。**sidecar の無いシャードは -1** で埋める＝損失に入らない。
どの波も持っていなければ `V["plan"]` は `None`（訓練側は方針ヘッドを自動で切る）。sidecar の
一覧・サイズ・mtime も pack の鍵に入れる＝後から sidecar を足せば pack は作り直される。
"""
import glob
import hashlib
import json
import os
import shutil

import numpy as np

from opcg_sim.learned.train.n1_train import _atype_idx    # 候補 action → ATYPES の index

# pack の版（レイアウトを変えたら上げる＝古い pack を無視して作り直す）
PACK_VERSION = 4                       # 4: 方針ラベル `plan` を足した（§20.10・sidecar）
#                                      # 3: 符号化 v14 の形へ 0 埋めで揃える（§20.9 の E）
#                                      # 2: 補助教師の 3 列を足した（dump v4・§20.8）
# V の pack が持つ列と dtype（tok/sc/ci/z は memmap・seed/turn は小さいので RAM に読む）
MMAP_COLS = {"sc": np.float16, "ci": np.int16, "tok": np.float16, "z": np.float16}
RAM_COLS = {"seed": np.int64, "turn": np.int16}
#: 補助教師（dump v4・§20.8.2）。npz に無い波は 0／`aux_mask=0` で埋める＝行は必ず揃う。
AUX_COLS = {"aux": np.float16, "aux_tok": np.float16, "aux_mask": np.int8}
#: 方針ラベル（§20.10・sidecar）。無いシャードは -1＝損失に入らない。
PLAN_COLS = {"plan": np.int8}
PLAN_NONE = -1
CACHE_ENV = "OPCG_DUMP_CACHE"


def _aux_shapes():
    """`aux` 系の 1 行あたりの形（正本は `n_rel` の次元＝`record_gen.AUX_COLS` と同じ数）。"""
    from opcg_sim.learned.n_rel import D_AUX, D_AUX_TOK, N_OPP
    return {"aux": (D_AUX,), "aux_tok": (N_OPP, D_AUX_TOK), "aux_mask": ()}


def target_form():
    """pack が揃える**現行の符号化の形**（`tok` の 1 行の形, `scalars` の列数）。

    符号化 v14（§20.9）は v13 に列を**末尾へ足した**だけ（S 20→22・scalars 123→127）なので、
    古い波は**新しい列を 0 で埋める**だけで現行の形になる（`0`＝情報なし・mask は要らない）。
    こうしておくと波を混ぜても行の形が揃い、`_Waves` が 1 本の配列に見せられる。
    """
    from opcg_sim.learned.n_rel import D_SC, N_TOK
    from opcg_sim.learned.n_rel_feat import S_DIM
    return (N_TOK, S_DIM), D_SC


def default_cache_dir():
    """pack の置き場所（`OPCG_DUMP_CACHE`→`~/.cache/opcg/dump_pack`）。"""
    return os.environ.get(CACHE_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "opcg", "dump_pack")


def shard_files(d):
    return sorted(f for f in glob.glob(os.path.join(d, "n_record_*.npz"))
                  if not f.endswith(".plan.npz"))


def plan_sidecar(f):
    """シャード → 方針ラベルの sidecar のパス（`plan_labels.sidecar_path` と同じ規則）。"""
    return os.path.splitext(f)[0] + ".plan.npz"


def _wave_key(files):
    """シャードの一覧＋サイズ＋mtime（ns）で pack を鍵付けする。"""
    h = hashlib.sha1()
    h.update(f"v{PACK_VERSION}\n".encode())
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
        sc = plan_sidecar(f)
        if os.path.exists(sc):                 # sidecar を足したら pack を作り直す（§20.10）
            st = os.stat(sc)
            h.update(f"{os.path.basename(sc)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def _rows_of(path):
    """1 シャードの行数と列の形（tokens の形・scalars の列数・card_idx の列数）と aux の有無。"""
    with np.load(path, allow_pickle=True) as d:
        if "tokens" not in d.files:
            raise ValueError(f"{path}: dump v2/v3 ではない（tokens 無し）")
        return (int(d["z"].shape[0]), tuple(d["tokens"].shape[1:]),
                int(d["scalars"].shape[1]), int(d["card_idx"].shape[1]),
                all(k in d.files for k in AUX_COLS))


def build_pack(d, cache_dir):
    """波 `d` の V を `cache_dir/<name>_<key>/` へ書き出す（既にあれば何もしない）。

    戻り値は pack のディレクトリ。**シャード 1 本ずつ書く**（float32 の全波を RAM に作らない）。
    `card_idx` は npz の列数のまま持つ（NRel は先頭 22 枠・N系 c は 24 枠を使う＝pack は共用）。
    """
    files = shard_files(d)
    if not files:
        raise ValueError(f"dump のシャードが無い: {d}")
    key = _wave_key(files)
    out = os.path.join(cache_dir, f"{os.path.basename(os.path.abspath(d))}_{key}")
    if os.path.exists(os.path.join(out, "meta.json")):
        return out
    n_tot = 0
    form = None
    per_shard = []
    has_aux = False
    has_plan = False
    for f in files:
        n, ts, sd, cd, aux = _rows_of(f)
        has_plan = has_plan or os.path.exists(plan_sidecar(f))
        if form is None:
            form = (ts, sd, cd)
        elif (ts, sd, cd) != form:
            raise ValueError(f"{f}: 列の形が波の中で揃っていない {(ts, sd, cd)} != {form}")
        per_shard.append(n)
        n_tot += n
        has_aux = has_aux or aux
    src_tok_shape, src_sc_dim, ci_dim = form
    # 符号化 v14（§20.9 の E）: **pack は常に現行の形**にする。古い波（v13＝tokens [22,20]・
    # scalars 123）は新しい列を 0 で埋めて入れる＝波を混ぜても行の形が揃う。
    tok_shape, sc_dim = target_form()
    if len(src_tok_shape) != len(tok_shape) or any(a > b for a, b in zip(src_tok_shape, tok_shape)) \
            or src_sc_dim > sc_dim:
        raise ValueError(
            f"{d}: 現行の符号化より大きい形（tokens {src_tok_shape} > {tok_shape} か "
            f"scalars {src_sc_dim} > {sc_dim}）＝この dump は新しすぎる")
    aux_shapes = _aux_shapes()
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    shapes = {"sc": (n_tot, sc_dim), "ci": (n_tot, ci_dim), "tok": (n_tot,) + tok_shape,
              "z": (n_tot,)}
    shapes.update({k: (n_tot,) + aux_shapes[k] for k in AUX_COLS})
    shapes.update({k: (n_tot,) for k in PLAN_COLS})
    cols = {**MMAP_COLS, **AUX_COLS, **PLAN_COLS}
    mm = {k: np.lib.format.open_memmap(os.path.join(tmp, f"{k}.npy"), mode="w+",
                                       dtype=cols[k], shape=shapes[k])
          for k in cols}
    ram = {k: np.empty(n_tot, dt) for k, dt in RAM_COLS.items()}
    off = 0
    for f, n in zip(files, per_shard):
        with np.load(f, allow_pickle=True) as d_:
            sl = slice(off, off + n)
            ci = d_["card_idx"]
            hi = int(np.abs(ci).max()) if n else 0
            if hi > np.iinfo(np.int16).max:
                raise ValueError(f"{f}: card_idx が int16 に収まらない（max {hi}）")
            # 新しい列は 0 のまま（`open_memmap` は 0 で作られる）＝古い波の 0 埋め。
            sc_ = d_["scalars"]
            tok_ = d_["tokens"]
            mm["sc"][sl, :sc_.shape[1]] = sc_.astype(np.float16)
            mm["ci"][sl] = ci.astype(np.int16)
            mm["tok"][sl, :tok_.shape[1], :tok_.shape[2]] = tok_.astype(np.float16)
            mm["z"][sl] = d_["z"].astype(np.float16)
            ram["seed"][sl] = d_["seed"]
            ram["turn"][sl] = d_["turn"] if "turn" in d_.files else 0
            # 補助教師（v4）。無いシャードは 0＝`aux_mask=0`＝損失に入らない。
            for k, dt in AUX_COLS.items():
                mm[k][sl] = d_[k].astype(dt) if k in d_.files else 0
        # 方針ラベル（§20.10・sidecar）。無いシャードは -1＝損失に入らない。
        sc_path = plan_sidecar(f)
        if os.path.exists(sc_path):
            with np.load(sc_path, allow_pickle=True) as ds:
                pl = np.asarray(ds["plan"], np.int8)[:n]
            if len(pl) != n:
                raise ValueError(f"{sc_path}: 行数がシャードと合わない（{len(pl)} != {n}）")
            mm["plan"][sl] = pl
        else:
            mm["plan"][sl] = PLAN_NONE
        off += n
    assert off == n_tot
    for k in list(mm):
        mm[k].flush()
        del mm[k]
    for k, a in ram.items():
        np.save(os.path.join(tmp, f"{k}.npy"), a)
    with open(os.path.join(tmp, "meta.json"), "w") as fh:
        json.dump({"pack_version": PACK_VERSION, "src": os.path.abspath(d), "key": key,
                   "rows": n_tot, "shards": [os.path.basename(f) for f in files],
                   "shard_rows": per_shard, "ci_dim": ci_dim, "has_aux": bool(has_aux),
                   "has_plan": bool(has_plan),
                   "tok_shape": list(tok_shape), "sc_dim": sc_dim,
                   # 元の npz の形（0 埋めしたかが判る＝v13 の波か v14 の波か）
                   "src_tok_shape": list(src_tok_shape), "src_sc_dim": src_sc_dim,
                   "dtypes": {k: np.dtype(v).name for k, v in
                              {**MMAP_COLS, **AUX_COLS, **PLAN_COLS, **RAM_COLS}.items()}}, fh)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


class _Waves:
    """波ごとの memmap を 1 本の配列に見せる view（`V["tok"][bi]` の形を保つ）。

    実装しているのは訓練ループ・評価帯が使う分だけ: `len`／`shape`／`dtype`／`nbytes`／
    整数・スライス・index 配列・bool マスクでの読み出し（**書き込みはしない**）。
    切り出した結果は普通の ndarray（memmap ではない）＝そのまま float32 へ上げてよい。
    """

    def __init__(self, parts, cols=None):
        self.parts = list(parts)
        n = [int(p.shape[0]) for p in self.parts]
        self.offsets = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
        w = tuple(self.parts[0].shape[1:])
        if cols is not None:                     # 先頭 `cols` 列だけ見せる（card_idx の枠数）
            if not w or cols > w[0]:
                raise ValueError(f"列が足りない: cols={cols} shape={w}")
            w = (cols,) + w[1:]
        self.cols = cols
        self.shape = (int(self.offsets[-1]),) + w
        self.dtype = self.parts[0].dtype
        self.ndim = len(self.shape)
        self.nbytes = int(sum(p.nbytes for p in self.parts))

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        a = np.concatenate([np.asarray(p) for p in self.parts]) if len(self.parts) > 1 \
            else np.asarray(self.parts[0])
        if self.cols is not None:
            a = a[:, :self.cols]
        return a.astype(dtype) if dtype is not None else a

    def _part_of(self, i):
        p = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return p, int(i - self.offsets[p])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            head, rest = key[0], key[1:]
            return self[head][(slice(None),) + rest]
        if isinstance(key, (int, np.integer)):
            p, j = self._part_of(int(key) % len(self))
            row = self.parts[p][j]
            return row[:self.cols] if self.cols is not None else row
        if isinstance(key, slice):
            key = np.arange(*key.indices(len(self)))
        idx = np.asarray(key)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64, copy=False)
        if idx.ndim != 1:
            raise TypeError(f"_Waves は 1 次元の index だけ受ける（{idx.ndim} 次元）")
        out = np.empty((len(idx),) + self.shape[1:], self.dtype)
        c = slice(None) if self.cols is None else slice(0, self.cols)
        if len(self.parts) == 1:
            out[:] = self.parts[0][idx] if self.cols is None else self.parts[0][idx][:, c]
            return out
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        for p in np.unique(which):
            m = which == p
            got = self.parts[p][idx[m] - self.offsets[p]]
            out[m] = got if self.cols is None else got[:, c]
        return out


def _pack_meta(pack):
    with open(os.path.join(pack, "meta.json")) as fh:
        return json.load(fh)


def load_row_col(dirs, name, z_dirs=()):
    """npz の**行ごとの列**を pack と同じ行順で読む（`deck_kinds`／`sig` などの文字列列）。

    pack は数値列しか持たない（文字列は memmap にできない）ので、必要なときだけ npz を開く。
    列を 1 本も持っていなければ `None`（＝`dump_io` は無い列を `None` で返す・§20.8 の契約）。
    """
    out = []
    for d in [d for d in list(dirs) + list(z_dirs) if shard_files(d)]:
        for f in shard_files(d):
            with np.load(f, allow_pickle=True) as dd:
                if name not in dd.files:
                    return None                  # 1 本でも欠けたら揃わない＝無い列として扱う
                out.append(np.asarray(dd[name])[:int(dd["z"].shape[0])])
    return np.concatenate(out) if out else None


def chosen_card_ids(dirs, z_dirs=()):
    """行ごとの「選んだ手の主体カード ID」（判らない行は ""）。pack と同じ行順で返す。

    `sig` は uuid しか持たない＝カードに戻せないので、main 窓の候補列
    （`pol_len`／`pol_chosen`／`pol_cid`）から引く（評価帯の層別・§20.8.2 の 3）。
    """
    waves = [d for d in list(dirs) + list(z_dirs) if shard_files(d)]
    out = []
    for d in waves:
        for f in shard_files(d):
            with np.load(f, allow_pickle=True) as dd:
                pl, pc = dd["pol_len"], dd["pol_chosen"]
                off = np.concatenate([[0], np.cumsum(pl)]).astype(np.int64)
                cid = dd["pol_cid"]
                col = np.full(len(pl), "", dtype=cid.dtype if cid.size else "U1")
                take = np.where((pl >= 1) & (pc >= 0))[0]
                if len(take) and cid.size:
                    col[take] = cid[off[take] + pc[take].astype(np.int64)]
                out.append(col.astype(str))
    return np.concatenate(out) if out else None


def rows_f32(V, bi):
    """V から行を切り出して float32／int64 に上げる（fp16 の pack をそのまま計算に流さない）。"""
    return (np.asarray(V["sc"][bi], np.float32), np.asarray(V["ci"][bi], np.int64),
            np.asarray(V["tok"][bi], np.float32))


def load_dump(dirs, vocab, with_policy=True, z_dirs=(), cache_dir=None, n_tok=None):
    """dump v2／v3 → V（value 行・memmap）, P（方策点）, C（候補）。R は含めない。

    `dirs` は π も z も使う波・`z_dirs` は z 専用（π を読まない）。`cache_dir` は pack の置き場所
    （既定は `default_cache_dir()`）。`n_tok` は `V["ci"]` に出す card_idx の列数
    （既定＝`n_rel.N_TOK`＝22 枠。N系 c の 24 枠が要る評価帯は `n_eff.MAX_CI` を渡す）。
    """
    if n_tok is None:
        from opcg_sim.learned.n_rel import N_TOK
        n_tok = N_TOK
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    waves = [(d, bool(with_policy)) for d in dirs] + [(d, False) for d in z_dirs]
    waves = [(d, p) for d, p in waves if shard_files(d)]
    if not waves:
        raise ValueError(f"dump が見つからない: {dirs} {z_dirs}")
    packs = [(build_pack(d, cache_dir), d, pol) for d, pol in waves]
    V = {k: _Waves([np.load(os.path.join(p, f"{k}.npy"), mmap_mode="r") for p, _d, _x in packs],
                   cols=(n_tok if k == "ci" else None))
         for k in MMAP_COLS}
    for k in RAM_COLS:
        V[k] = np.concatenate([np.load(os.path.join(p, f"{k}.npy")) for p, _d, _x in packs])
    # 補助教師（v4）: **1 本でも持っている波があれば**列を出す（持たない波の行は 0／mask 0）。
    # どの波も持っていなければ `None`＝訓練側は補助ヘッドを自動で切る（§20.8.2）。
    any_aux = any(_pack_meta(p).get("has_aux") for p, _d, _x in packs)
    for k in AUX_COLS:
        V[k] = (_Waves([np.load(os.path.join(p, f"{k}.npy"), mmap_mode="r")
                        for p, _d, _x in packs]) if any_aux else None)
    # 方針ラベル（§20.10）: 1 本でも sidecar を持つ波があれば列を出す（持たない行は -1）。
    any_plan = any(_pack_meta(p).get("has_plan") for p, _d, _x in packs)
    for k in PLAN_COLS:
        V[k] = (_Waves([np.load(os.path.join(p, f"{k}.npy"), mmap_mode="r")
                        for p, _d, _x in packs]) if any_plan else None)
    P = {"row": [], "seed": [], "len": [], "chosen": [], "v0": []}
    C = {"pi": [], "at": [], "cid": [], "tcid": [], "k": [], "si": [], "ti": [],
         "n": [], "q": [], "p": []}
    off_v = 0
    for pack, d, pol in packs:
        with open(os.path.join(pack, "meta.json")) as fh:
            meta = json.load(fh)
        if not pol:
            off_v += int(meta["rows"])
            continue
        for f, n in zip(shard_files(d), meta["shard_rows"]):
            with np.load(f, allow_pickle=True) as dd:
                _read_policy(dd, off_v, vocab, P, C)
            off_v += n
    if P["row"]:
        # 1 本でも持たないシャードがある列は `None`（＝揃わない列は無い列として扱う契約）。
        P = {k: (None if any(a is None for a in v) else np.concatenate(v)) for k, v in P.items()}
        C = {k: (None if any(a is None for a in v) else np.concatenate(v)) for k, v in C.items()}
    else:
        P = {k: np.zeros(0, np.int64) for k in P}
        C = {k: np.zeros(0) for k in C}
    assert off_v == len(V["z"])
    return V, P, C


def _read_policy(d, off_v, vocab, P, C):
    """1 シャードの main 窓の方策点（候補 2 本以上・選択が判っている行）を P/C へ積む。"""
    pl, pc, kind = d["pol_len"], d["pol_chosen"], d["kind"]
    off = np.concatenate([[0], np.cumsum(pl)])
    take = np.where((kind == 0) & (pl >= 2) & (pc >= 0))[0]
    if not len(take):
        return
    P["row"].append((take + off_v).astype(np.int64))
    P["seed"].append(d["seed"][take])
    P["len"].append(pl[take])
    P["chosen"].append(pc[take])
    # 根の価値の推定（dump v4・§20.6.1）。無い波は `None`＝列ごと `None` になる。
    P["v0"].append(d["pol_v0"][take].astype(np.float32) if "pol_v0" in d.files else None)
    idx = np.concatenate([np.arange(off[i], off[i + 1]) for i in take])
    nn = d["pol_n"][idx].astype(np.float64)
    # 改良方策 π' の材料（§20.6.1）。`pol_n`／`pol_q` は dump v2 から常にある。
    C["n"].append(nn.astype(np.float32))
    C["q"].append(d["pol_q"][idx].astype(np.float32))
    C["p"].append(d["pol_p"][idx].astype(np.float32) if "pol_p" in d.files else None)
    segl = np.repeat(np.arange(len(take)), pl[take])
    tot = np.zeros(len(take))
    np.add.at(tot, segl, nn)
    tot = np.maximum(tot, 1e-9)
    C["pi"].append((nn / tot[segl]).astype(np.float32))
    C["at"].append(np.array([_atype_idx(json.loads(s)[0]) for s in d["pol_sig"][idx]], np.int16))
    C["cid"].append(np.array([vocab.get(c, 0) for c in d["pol_cid"][idx]], np.int32))
    C["tcid"].append(np.array([vocab.get(c, 0) for c in d["pol_tcid"][idx]], np.int32))
    C["k"].append(d["pol_k"][idx].astype(np.int16))
    C["si"].append(d["pol_si"][idx].astype(np.int16))
    C["ti"].append(d["pol_ti"][idx].astype(np.int16))
