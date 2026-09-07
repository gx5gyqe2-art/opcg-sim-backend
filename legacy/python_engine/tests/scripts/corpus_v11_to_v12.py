"""コーパスの符号化 v11 → v12 変換（列切り出し・2026-08-15）。

v12 = v9(70列) + リーダー物理要約(24列) ＝ v11 から**リーサル距離Δ3列（列70..72）を抜いた**もの。
v11 の行は [v9 70 | Δ 3 | リーダー 24] の並びなので、`scalars[:, [0:70]+[73:97]]` の切り出しだけで
v12 の教師になる（**対局の再生成は不要**）。他の配列（field/card_idx/value/group/...）は素通し。

なぜ v12 が要るか: v10 のΔはエンジンで台本を再生する実測特徴で ~25ms/盤面。探索は1手で
数百回符号化するため decide が 0.47s(v9) → 13.5s(v11) と本番予算1秒を28倍超過する
（2026-08-15 実測）。リーダー要約はキャッシュ済みで実質ゼロコスト＝**安い側だけを残す**。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/corpus_v11_to_v12.py \
    --dirs /tmp/g15_corpus/part1,/tmp/bb7_corpus --suffix _v12
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import rl_encoder as E  # noqa: E402

# v11 行から v12 を作る列（唯一の正・エンコーダの版レイアウトから導出する）:
# [0, scalars_dim(9)) = v9 の列 ／ [scalars_dim(10), scalars_dim(11)) = リーダー要約24列
# （その間の [scalars_dim(9), scalars_dim(10)) が落とすリーサルΔ3列）。
V12_COLS = (list(range(E.scalars_dim(9)))
            + list(range(E.scalars_dim(10), E.scalars_dim(11))))


def convert_file(src, dst):
    z = np.load(src)
    out = {k: z[k] for k in z.files}
    sc = out["scalars"]
    if sc.shape[1] == E.scalars_dim(12):
        return False                      # 既に v12（冪等）
    assert sc.shape[1] == E.scalars_dim(11), f"{src}: v11(97列) でない（{sc.shape[1]}列）"
    out["scalars"] = sc[:, V12_COLS]
    np.savez_compressed(dst, **out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="変換元ディレクトリ（カンマ区切り）")
    ap.add_argument("--suffix", default="_v12", help="出力ディレクトリの接尾辞")
    ap.add_argument("--glob", default="*.npz")
    args = ap.parse_args()

    assert len(V12_COLS) == E.scalars_dim(12), "列定義が v12 次元と不一致"
    n_files = n_rows = 0
    for d in [x.strip() for x in args.dirs.split(",") if x.strip()]:
        out_dir = d.rstrip("/") + args.suffix
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(d, args.glob))):
            dst = os.path.join(out_dir, os.path.basename(f))
            if convert_file(f, dst):
                n_files += 1
                n_rows += len(np.load(dst)["scalars"])
        print(f"  {d} → {out_dir}", flush=True)
    print(f"CORPUS_V12_DONE files={n_files} rows={n_rows} cols={len(V12_COLS)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
