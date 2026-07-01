"""最小1セット通し試走: レバー①②③を最小構成で結線し held-out 勝率を実際に出す。

docs/reports/cpu_rl_generalization_plan_20260701.md D。目的は「配管が通り、held-out 勝率が出るか」
＋「fingerprint 表現＋非線形ネットが未知アーキタイプ(黄)へ転移するか」の一次読み。

構成（最小）:
  ① encoder_v2（効果フィンガープリント state・ID埋め込みなし）
  ② デッキのドメインランダム化の代理: 毎ゲーム train リーダーを無作為選択（黄は held-out へ隔離）
  ③ held-out 勝率ゲート: 学習した value で貪欲1-ply プレイ vs ランダム を、
     in-dist（非黄）と held-out（黄）で対局し勝率を比較

教師は **L1 evaluate**（pre-flight で信頼できると確定した密教師。ランダム続行 outcome はノイズ過多で不可）。
本試走は L1 模倣なので上限は≒L1 だが、狙いは"汎化と配管"の検証（L1超えは outcome自己対戦=次段）。
"""
import argparse
import random

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from rl_encoder_v2 import encode_v2, DIM
from cpu_selfplay import build_deck, _load_db
from probe_generalization import _leaders_split
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai

MOVE_CAP = 20   # 貪欲評価する合法手の上限（コスト制限）


class MLP:
    """1隠れ層 tanh 回帰（Adam・MSE）。入力/教師は train 統計で標準化。"""
    def __init__(self, din, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((din, hidden)) * np.sqrt(2.0 / din)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(1)
        self._m = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self._t = 0
        self.xmu = self.xsd = self.ymu = self.ysd = None

    def fit_norm(self, X, y):
        self.xmu, self.xsd = X.mean(0), X.std(0) + 1e-6
        self.ymu, self.ysd = float(y.mean()), float(y.std()) + 1e-6

    def _fwd(self, Xn):
        Z1 = Xn @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        out = A1 @ self.W2 + self.b2
        return out, (Xn, Z1, A1)

    def train(self, X, y, epochs=60, bs=128, lr=3e-3, rng=None):
        rng = rng or np.random.default_rng(0)
        Xn = (X - self.xmu) / self.xsd
        yn = ((y - self.ymu) / self.ysd).reshape(-1, 1)
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, bs):
                b = idx[s:s + bs]
                out, (xb, Z1, A1) = self._fwd(Xn[b])
                dout = (out - yn[b]) / len(b)
                gW2 = A1.T @ dout; gb2 = dout.sum(0)
                dA1 = dout @ self.W2.T
                dZ1 = dA1 * (1 - A1 ** 2)
                gW1 = xb.T @ dZ1; gb1 = dZ1.sum(0)
                self._step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}, lr)

    def _step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k in g:
            self._m[k] = b1 * self._m[k] + (1 - b1) * g[k]
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g[k] * g[k])
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))

    def value(self, x):
        xn = (x - self.xmu) / self.xsd
        out, _ = self._fwd(xn.reshape(1, -1))
        return float(out[0, 0])   # 標準化 y 空間（順序が保たれれば貪欲には十分）


def _random_game(leaders, db, rng):
    la = leaders[rng.randrange(len(leaders))]
    lb = leaders[rng.randrange(len(leaders))]
    l1, c1 = build_deck(db, "p1", la); l2, c2 = build_deck(db, "p2", lb)
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    return m


def gen_dataset(leaders, db, vocab, fps, n_games, ply_cap, every, rng):
    """ランダムプレイで (encode_v2, L1評価) を収集。"""
    X, Y = [], []
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
            legal = m.get_legal_actions(actor)
            if not legal:
                break
            if ply % every == 0 and m.turn_count >= 2:
                try:
                    X.append(encode_v2(m, nm, vocab, fps))
                    Y.append(float(cpu_ai.evaluate(m, nm)))
                except Exception:
                    pass
            try:
                cpu_ai._apply_move_inplace(m, nm, legal[rng.randrange(len(legal))])
            except Exception:
                break
            ply += 1
    return np.array(X, np.float32), np.array(Y, np.float32)


def _value_state(net, m, me_name, vocab, fps):
    if m.winner is not None:
        return 1e9 if m.winner == me_name else -1e9
    return net.value(encode_v2(m, me_name, vocab, fps))


def greedy_move(net, m, me_name, vocab, fps, rng):
    actor = m.p1 if m.p1.name == me_name else m.p2
    legal = m.get_legal_actions(actor)
    if not legal:
        return None
    best, bv = None, -1e18
    for mv in legal[:MOVE_CAP]:
        clone = cpu_ai._apply_clone(m, me_name, mv)
        if clone is None:
            continue
        v = _value_state(net, clone, me_name, vocab, fps)
        if v > bv:
            bv, best = v, mv
    return best or legal[rng.randrange(len(legal))]


def play_match(net, leaders, db, vocab, fps, n_games, ply_cap, rng, net_is_p1=True):
    """net(貪欲) vs ランダム を対局。net 側の勝率を返す。"""
    wins = 0; done = 0
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        me = "p1" if net_is_p1 else "p2"
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
            legal = m.get_legal_actions(actor)
            if not legal:
                break
            if nm == me:
                mv = greedy_move(net, m, me, vocab, fps, rng)
            else:
                mv = legal[rng.randrange(len(legal))]
            try:
                cpu_ai._apply_move_inplace(m, nm, mv)
            except Exception:
                break
            ply += 1
        if m.winner is not None:
            done += 1
            if m.winner == me:
                wins += 1
    return wins / done if done else float("nan"), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-games", type=int, default=80)
    ap.add_argument("--eval-games", type=int, default=24)
    ap.add_argument("--ply-cap", type=int, default=400)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = FP.build_fingerprints(db)
    train_leaders, held_leaders = _leaders_split(db, "color", 0, rng)
    print(f"encoder_v2 DIM={DIM}  train(非黄)={len(train_leaders)}  held-out(黄)={len(held_leaders)}")

    print(f"① データ生成: {args.data_games} games（train色・L1評価ラベル） ...")
    X, Y = gen_dataset(train_leaders, db, vocab, fps, args.data_games, args.ply_cap, args.every, rng)
    print(f"   states={len(X)}")
    n = len(X); cut = int(n * 0.85)
    idx = nrng.permutation(n); tr, va = idx[:cut], idx[cut:]
    net = MLP(DIM, seed=args.seed)
    net.fit_norm(X[tr], Y[tr])
    net.train(X[tr], Y[tr], epochs=args.epochs, rng=nrng)
    # 学習後の value 予測 R²（標準化空間）で健全性を確認
    def r2(Xs, Ys):
        p = np.array([net.value(x) for x in Xs])
        yn = (Ys - net.ymu) / net.ysd
        return 1 - np.sum((yn - p) ** 2) / (np.sum((yn - yn.mean()) ** 2) + 1e-9)
    print(f"   value R²: in-dist(val)={r2(X[va], Y[va]):+.3f}")

    print(f"③ held-out ゲート: 貪欲value vs ランダム を各 {args.eval_games} games ...")
    wi, di = play_match(net, train_leaders, db, vocab, fps, args.eval_games, args.ply_cap, rng)
    wh, dh = play_match(net, held_leaders, db, vocab, fps, args.eval_games, args.ply_cap, rng)
    print("\n=== 結果 ===")
    print(f"  in-dist(非黄) 勝率 vs ランダム = {wi:.3f}  (n={di})")
    print(f"  held-out(黄)  勝率 vs ランダム = {wh:.3f}  (n={dh})")
    print("\n判定:")
    if not (wi == wi and wh == wh):
        print("  対局が終局せず（ply-cap 到達）。cap↑ か games↑ で再試")
    elif wi > 0.55 and wh > 0.55 and abs(wi - wh) < 0.15:
        print("  配管OK・貪欲valueがランダムに勝ち・held-outでも同等に勝つ＝転移の一次確認")
    elif wi > 0.55 and wh <= 0.52:
        print("  in-distでは勝つが held-out で崩れる＝表現/データが黄へ転移していない")
    else:
        print("  信号弱い（勝率が拮抗）。data-games/epochs/eval-games を増やして再判定")


if __name__ == "__main__":
    main()
