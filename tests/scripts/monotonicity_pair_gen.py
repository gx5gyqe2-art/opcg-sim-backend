"""**支配単調性の教師**を生成する（2026-08-15・不変量監査で見つけた既存欠陥への処方）。

`value_invariant_audit.py` が検出した欠陥: 出荷ネットは「他が全く同じで**相手のパワー/ドンが
増えた**盤面」を自分に有利と評価することがある（gen15 で power_opp 39%・don_opp 41%、gen14 も
同水準＝**世代を跨ぐ既存欠陥**）。破れの大きさ 0.02〜0.05 は枝間マージン 0.02〜0.03 と同規模＝
接戦の選択を裏返す力がある。ライフ/手札はほぼ無傷で、**パワーとドンだけが秩序づけられていない**。

本器はその常識を**順位ペア**として教材化する:
  1点（group）= 元盤面と、そこから資源を1単位だけ増やした盤面のペア
  value=+1 が「自分にとって良い側」・−1 が「悪い側」（勝率ではなく**順序の主張**）

**この教師の性質（なぜ物量に向くか）**:
  - **レフェリー不要・対局不要**（ロールアウトを1本も打たない）＝1ペア数ミリ秒で作れる
  - **実デッキに依存しない**（任意の盤面に足すだけ）＝リプレイでも自己対戦でも合成でも良い
  - **ラベルが論理的に正しい**（勝敗ラベルの推定でも人間の裁定でもない＝ブートストラップ問題が無い）
出力スキーマは defcf/vinj/optpair と同一なので `option_pair_finetune`（蒸留アンカー付き順位
学習）にそのまま流せる。**本体 value を動かす**教師なので、必ずアンカー（既存挙動の錘）と
併用し、ns2/ゲート/アリーナで退行を検査すること（v40 の教訓＝本体を動かすと全域が動く）。

**限界**: 資源を「無から足す」ため盤面はやや非現実になりうる（総ドン11 等）。教えているのは
「量の順序」であって「量の価値」ではない（どれだけ良いかは勝敗教師が担当する）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/monotonicity_pair_gen.py \\
    --boards 400 --stride 3 --out /tmp/monopair
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from cpu_selfplay import _load_db  # noqa: E402
from value_invariant_audit import MUTATIONS, sample_boards, _players  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=400)
    ap.add_argument("--stride", type=int, default=3, help="リプレイの採取間隔（小さいほど多く）")
    ap.add_argument("--enc-version", type=int, default=0, help="0=出荷ネットの世代に合わせる")
    ap.add_argument("--kinds", default="",
                    help="生成する種類を絞る（カンマ区切り: life,hand,don,power・空=全部）。"
                         "壊れている次元だけを直したい時に使う＝本体の移動を最小化する"
                         "（2026-08-15: 全種で教えると m1@14〔入口では素通し〕が崩れたため）")
    ap.add_argument("--group-base", type=int, default=800000)
    ap.add_argument("--shard", default="monopair_00000.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.learned import encoder as E
    eng = LearnedEngine()
    ver = args.enc_version or eng.enc_version
    db = _load_db()

    boards = sample_boards(db, args.boards, args.stride)
    print(f"盤面 {len(boards)} 点から単調性ペアを生成（enc_v={ver}）", flush=True)

    want = {k.strip() for k in args.kinds.split(",") if k.strip()}
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "group",
                            "q_root", "turns_left")}
    gi = args.group_base
    kinds = {}

    def push(m, name, val, g):
        e = E.encode(m, name, eng.vocab, version=ver)
        rows["scalars"].append(e["scalars"])
        rows["field"].append(e["field"])
        rows["card_idx"].append(np.asarray(e["card_idx"], dtype=np.int64))
        rows["value"].append(np.float32(val))
        rows["group"].append(np.int64(g))
        rows["q_root"].append(np.float32("nan"))     # 勝敗単独ラベル（エコー遮断・defcf と同規約）
        rows["turns_left"].append(np.float32(0.0))

    for tag, i, m, name in boards:
        _, opp = _players(m, name)
        for key, fn in MUTATIONS:
            if want and key not in want:
                continue
            for side, who in (("me", name), ("opp", opp.name)):
                try:
                    m2 = m.clone()
                except Exception:
                    continue
                if not fn(m2, who):
                    continue
                # 常に `name` 視点で符号化する。自分側の資源増＝+1 が改変後、
                # 相手側の資源増＝+1 は**元盤面**（増えていない方が自分に良い）。
                push(m, name, +1.0 if side == "opp" else -1.0, gi)
                push(m2, name, -1.0 if side == "opp" else +1.0, gi)
                kinds[f"{key}_{side}"] = kinds.get(f"{key}_{side}", 0) + 1
                gi += 1

    if not rows["value"]:
        print("ペアが作れなかった（盤面が採れていない）")
        return 1
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.shard)
    L = max(len(c) for c in rows["card_idx"])
    ci = np.zeros((len(rows["card_idx"]), L), np.int64)
    for k, c in enumerate(rows["card_idx"]):
        ci[k, :len(c)] = c
    np.savez_compressed(
        path,
        scalars=np.stack(rows["scalars"]).astype(np.float32),
        field=np.stack(rows["field"]).astype(np.float32),
        card_idx=ci,
        value=np.array(rows["value"], np.float32),
        group=np.array(rows["group"], np.int64),
        q_root=np.array(rows["q_root"], np.float32),
        turns_left=np.array(rows["turns_left"], np.float32))
    res = {"shard": path, "boards": len(boards), "groups": gi - args.group_base,
           "rows": len(rows["value"]), "enc_version": ver, "kinds": kinds}
    print("MONO_PAIR_RESULT " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
