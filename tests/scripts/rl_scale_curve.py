"""学習evalスパイク: value予測のスケールカーブ（dev・計算資源無し環境での GO/NO-GO 事前フィルタ）。

核心仮説＝「**学習評価器はデータ規模を上げると L1 より正確に勝者を当てられるか**」を、Elo本走でなく
**held-out の勝者予測精度（sign-accuracy）**で測る。outcome をターゲットに、N=10^3→10^4→… で
net を学習し直し、**固定 held-out** 上で net と L1 の sign-acc を比較。
- net が L1 を超え＆**N で伸び続ける** → 規模が効く（本走の価値あり・GO 寄り）。
- 飽和し L1 未満 → 学習evalは規模でも弱い（NO-GO 寄り）。

注意: これは Elo本走の**安い事前フィルタ**であり最終判定ではない（最終は MCTS×Elo・要計算資源）。
実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/rl_scale_curve.py --games 60 --eps 0.3 [--data d.npz]
"""
import argparse

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from cpu_selfplay import _load_db
import rl_encoder as E
import rl_datagen as G
import rl_net as N


def sign_acc(pred, y):
    return float((np.sign(pred) == np.sign(y)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, nargs="+",
                    help="既存 npz（複数可・game id はチャンク跨ぎで一意化して結合）。無ければ生成")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--d-emb", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    db = _load_db()
    vocab = E.build_vocab(db)
    if args.data:
        chunks = []
        goff = 0
        for path in args.data:
            z = np.load(path)
            d = {k: z[k] for k in z.files}
            d["game"] = d["game"].astype(np.int64) + goff   # チャンク跨ぎで game id を一意化
            goff = int(d["game"].max()) + 1
            chunks.append(d)
        data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    else:
        data = G.generate(db, vocab, args.games, args.eps, 400, args.seed0, args.sample_every)
    if data is None:
        print("データ生成0"); return
    n = len(data["value"])
    print(f"総局面 {n}", flush=True)

    # **group split by game**（局面はゲーム内で相関＋勝敗一定＝局面単位split はリークで net を過大評価）。
    rng = np.random.default_rng(0)
    games = np.unique(data["game"])
    rng.shuffle(games)
    # held-out 局面が ~args.holdout になるまでゲームを held-out へ。
    ho_games, cnt = set(), 0
    per_game = max(1, n // max(1, len(games)))
    for gid in games:
        if cnt >= args.holdout:
            break
        ho_games.add(int(gid)); cnt += int((data["game"] == gid).sum())
    is_ho = np.isin(data["game"], list(ho_games))
    ho = {k: data[k][is_ho] for k in data}
    tr = {k: data[k][~is_ho] for k in data}
    yho = ho["value"]
    print(f"split by game: held-out {len(ho_games)}ゲーム / train {len(games)-len(ho_games)}ゲーム", flush=True)

    # L1 ベースライン（held-out 上の勝者予測精度・N 非依存の定数）。
    l1_acc = sign_acc(ho["l1"], yho)
    base = float((yho > 0).mean())   # 多数派ベースライン（参考）
    print(f"held-out {len(yho)}  多数派={max(base, 1-base):.3f}  **L1 sign-acc={l1_acc:.3f}**", flush=True)

    # train 局面をシャッフル（小N でもゲーム多様性を確保・holdout はゲーム単位で分離済み＝リーク無し）。
    tperm = rng.permutation(len(tr["value"]))
    tr = {k: tr[k][tperm] for k in tr}
    ntr = len(tr["value"])
    scales = [s for s in (1000, 3000, 10000, 30000, 100000, 300000) if s <= ntr] or [ntr]
    if scales[-1] != ntr:
        scales.append(ntr)
    print("\nN        net_sign_acc   (vs L1 %+.3f)" % l1_acc, flush=True)
    for Nn in scales:
        sub = {k: tr[k][:Nn] for k in tr}
        net = N.ValueNet(len(vocab), d_emb=args.d_emb, hidden=args.hidden,
                         feat_dim=E.feature_dim(), seed=0)
        N.train(net, sub, epochs=args.epochs, lr=2e-3, batch=256, val_frac=0.05)
        acc = sign_acc(net.predict(ho), yho)
        flag = "  <- L1超え" if acc > l1_acc else ""
        print(f"{Nn:>7d}   {acc:.3f}        ({acc - l1_acc:+.3f}){flag}", flush=True)
    print("\n解釈: net が L1 を超え＆N で上り続ける=規模が効く(GO寄り)/飽和しL1未満=弱い(NO-GO寄り)。"
          "※Elo本走の事前フィルタ・最終判定ではない。")


if __name__ == "__main__":
    main()
