"""時間軸の教師（dump の行 → 決着までの時間とライフレースの先の姿）と、その sidecar。

計画 §20.11 の候補 H（ユーザ提案 2026-09-12「盤面情報をインプットに、ライフレースを時系列で
見られるように」→「安いやつを順に全てやっていきましょうか」）。V は盤面 1 つに勝率 1 個を返すだけで
**「いつ」を持たない**。ライフレースは速さの勝負なので、**決着までの自席ターン数**と
**2／3 自席ターン先のライフ**を補助ヘッド（`n_rel.time_head`・7 出力・Huber 回帰）に予測させる。
入力も Rust も変えない。教師は既存の記録から後付けで作れる（生成し直しは不要）＝シャードの隣の
sidecar `n_record_XXXXX.time.npz` に `time(D,7 float16)`／`time_mask(D int8)` を書き、`dump_io` が
pack に取り込む。

**列（`TIME_COLS`・すべて「起きたこと」＝良し悪しは入れない）**
  0 t_left        … この行から対局が終わるまでの**自席ターン数** /`T_SCALE`（自席ターンの行は
                    今のターンを含む・相手ターンの行は次の自席ターンから数える）
  1 my_life_p2    … 2 つ先の自席ターン開始時の自分のライフ（対局がそれより前に終われば最後の値）
  2 opp_life_p2   … 同・相手のライフ
  3 my_life_p3    … 3 つ先の自席ターン開始時の自分のライフ
  4 opp_life_p3   … 同・相手のライフ
  5 my_life_end   … 記録の最後の行での自分のライフ（＝決着直前の観測値）
  6 opp_life_end  … 同・相手のライフ
`time_mask=1` は「対局に自席ターンが 1 つ以上あり、勝敗が付いている（z≠0）」行だけ。ターン 0
（マリガン）は 0。

**1 ターン先を入れない理由**: 既存の補助教師（`record_gen.aux_from_ledger`・§20.8.2）が
「次の相手ターンに失うライフ」「次の自分のターンに削るライフ」＝**+1 ターンをすでに持っている**。
H の値打ちは**その先（+2／+3）と決着までの距離**なので、重複する列は作らない。

**「1／2／3 ターン後の勝率」を「ライフの軌跡」に置き換えた理由**: z は対局の中で定数なので
「N ターン後の勝率」を記録から取ると教師が z と同一になる（何も足さない）。探索の自己評価
（`pol_v0`）を将来の行から引く手もあるが、それは観測ではなくブートストラップで、dump v14 の波に
しか無い。**まずは観測できる量（ライフの軌跡と残りターン数）で信号の有無を見る**——出たら
`pol_v0` 版へ進む。

ライフは `scalars[0]`（自分）／`scalars[1]`（相手）を**自席ターン最初の main 行**で読む
（`plan_labels.label_game` と同じ流儀）。視点はその行の `who`。

CLI（sidecar を書く・冪等・既にあれば飛ばす）:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.time_labels --in ~/n32_wave/w*/n_records [--force]
"""
import argparse
import os
import sys
import time

import numpy as np

from opcg_sim.learned.train import plan_labels as PL

TIME_COLS = ("t_left", "my_life_p2", "opp_life_p2", "my_life_p3", "opp_life_p3",
             "my_life_end", "opp_life_end")
D_TIME = len(TIME_COLS)                    # 7（`n_rel.D_TIME` と同じ数）
SIDECAR_SUFFIX = ".time.npz"
#: 残りターン数の目盛り（`aux` のパワーを /10000 で入れるのと同じ流儀）。
T_SCALE = 10.0
#: sidecar を作るのに要る行の列（候補列は要らない＝ライフとターンだけ）
ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len")


def sidecar_path(shard_path):
    """`.../n_record_00012.npz` → `.../n_record_00012.time.npz`。"""
    return os.path.splitext(shard_path)[0] + SIDECAR_SUFFIX


def _lives(dd, n):
    """行ごとの (my_life, opp_life)＝`scalars[:, 0:2]`。"""
    return np.asarray(dd["scalars"])[:n, 0:2].astype(np.float32)


def label_game(rows, lives, idx):
    """対局 → 行ごとの (time [n,7] float32, mask [n] int8)。

    `idx` は手順どおりの行 index（`plan_labels.games_of` が返すもの）。
    """
    n = len(idx)
    out = np.zeros((n, D_TIME), np.float32)
    mask = np.zeros(n, np.int8)
    # 自席ターン開始時のライフ（席ごと・そのターンの最初の main 行）
    life_at = {}
    for i in idx:
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        if PL.is_own_turn(w, t) and int(rows["kind"][i]) == 0 and (w, t) not in life_at:
            life_at[(w, t)] = (float(lives[i][0]), float(lives[i][1]))
    turns_of = {w: sorted(t for (ww, t) in life_at if ww == w) for w in (0, 1)}
    # 決着直前の観測値（記録の最後の行・視点をその席へ合わせる）
    last = idx[-1]
    last_w = int(rows["who"][last])
    last_pair = (float(lives[last][0]), float(lives[last][1]))
    for d, i in enumerate(idx):
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        z = float(rows["z"][i])
        ts = turns_of[w]
        if t < 1 or not ts or z == 0.0:
            continue
        # この行から見た「この先の自席ターン」列（自席ターンの行は今のターンを含む）
        fut = [tt for tt in ts if (tt >= t if PL.is_own_turn(w, t) else tt > t)]
        if not fut:
            continue
        out[d, 0] = len(fut) / T_SCALE
        for k, base in ((2, 1), (3, 3)):                 # +2 → 列 1/2・+3 → 列 3/4
            tt = fut[k] if len(fut) > k else fut[-1]
            my, op = life_at[(w, tt)]
            out[d, base] = my
            out[d, base + 1] = op
        my_end, op_end = last_pair if last_w == w else (last_pair[1], last_pair[0])
        out[d, 5] = my_end
        out[d, 6] = op_end
        mask[d] = 1
    return out, mask


def build_sidecar(shard_path, force=False):
    """1 シャードの sidecar を書く。戻り値は (書いたか, 行数, mask=1 の行数, 列の平均)。"""
    out_path = sidecar_path(shard_path)
    if os.path.exists(out_path) and not force:
        return False, 0, 0, None
    tm = msk = None
    for _f, rows, _pol, lives in PL.iter_shards([], row_cols=ROW_COLS, pol_cols=(),
                                                extra_fn=_lives, files=[shard_path]):
        n = len(rows["z"])
        tm = np.zeros((n, D_TIME), np.float32)
        msk = np.zeros(n, np.int8)
        _L, _ptr, games = PL.games_of(rows)
        for idx in games:
            t_, m_ = label_game(rows, lives, idx)
            tm[idx] = t_
            msk[idx] = m_
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, time=tm.astype(np.float16), time_mask=msk,
                        cols=np.array(TIME_COLS))
    os.replace(tmp, out_path)
    ok = int(msk.sum())
    means = tm[msk > 0].mean(0).tolist() if ok else None
    return True, len(msk), ok, means


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True,
                    help="n_records のディレクトリ（glob 展開済み）")
    ap.add_argument("--force", action="store_true", help="既にある sidecar も書き直す")
    args = ap.parse_args(argv)
    t0 = time.time()
    n_written = n_skipped = n_rows = n_ok = 0
    acc = np.zeros(D_TIME, np.float64)
    for d in args.src:
        for f in PL.shard_files(d):
            wrote, n, ok, means = build_sidecar(f, force=args.force)
            if not wrote:
                n_skipped += 1
                continue
            n_written += 1; n_rows += n; n_ok += ok
            if means is not None:
                acc += np.asarray(means) * ok
    avg = (acc / max(n_ok, 1)).round(3).tolist()
    print(f"sidecar 書いた {n_written}・飛ばした {n_skipped}・行 {n_rows}"
          f"・有効 {n_ok}・平均 {dict(zip(TIME_COLS, avg))}（{time.time()-t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
