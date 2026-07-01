"""Pre-flight プローブ①: 表現の“汎化 retention”を自己対戦なしで測る（GO/NO-GO）。

docs/reports/cpu_rl_generalization_plan_20260701.md ①。狙い＝**1セット学習にコミットする前に**、
不可逆な選択（カード表現）が分布外へ転移するかを安く判定する。

方法（線形プローブ）:
  1. リーダーを色で分割。held-out = 黄（黒ひげ OP16-080 の色）、train = それ以外の色。
  2. 各リーダーでデッキを組み、**ランダムプレイ**で局面をサンプル（L1バイアスの無い分布）。
  3. 各局面の教師値 = L1 静的評価 cpu_ai.evaluate（deck非依存の位置価値の密な代理）。
  4. 2つの表現でリッジ回帰: R1=識別子bag(card_id multihot)／R2=効果フィンガープリント平均pool。
  5. train色でfit → (a) train色hold-out局面 と (b) 黄局面 で R² を測る。
     retention = R²(黄) / R²(train) が高い表現ほど「未知アーキタイプへ転移する」。

期待: R1 は黄で崩れ（未知IDは全ゼロ特徴→平均予測→R²≈0）、R2 は保つ。
注意: 教師値=L1評価は“表現の汎化”を測る代理であり“強さの天井”ではない（それは本走の held-out 勝率で測る）。
"""
import argparse
import random
import sys

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai

YELLOW = "黄"
E_SCALARS = 14   # encoder.encode の scalars 次元（両表現の共有先頭ブロック）


def _all_leaders(db):
    out = []
    for cid in db.raw_db.keys():
        m = db.get_card(cid)
        if m is not None and m.type.name == "LEADER":
            cols = {getattr(x, "value", x) for x in (getattr(m, "colors", []) or [])}
            out.append((cid, cols))
    return out


def _leaders_split(db, mode, k, rng):
    """held-out の切り方を選ぶ。
      color : 黄を丸ごと訓練から除外（＝未見領域の被覆ゼロ。pre-flight①の設定）
      leader: 黄リーダーの一部だけ held-out・残りの黄は訓練に残す
              （＝ドメインランダム化の代理: 未見“領域”ではなく未見“個体”への転移）
    """
    leaders = _all_leaders(db)
    yellow = [cid for cid, cols in leaders if YELLOW in cols]
    if mode == "color":
        train = [cid for cid, cols in leaders if YELLOW not in cols]
        return train, list(yellow)
    # leader mode: 黄から k 個を held-out、それ以外（他色＋残り黄）は訓練
    ysh = list(yellow); rng.shuffle(ysh)
    held = ysh[:k]
    held_set = set(held)
    train = [cid for cid, _ in leaders if cid not in held_set]
    return train, held


def _zero():
    return np.zeros(FP.CARD_DIM, np.float32)


def _encode_pair(m, nm, vocab, fps, vlen):
    """局面を R1(identity-bag) と R2(fingerprint) の両表現へ符号化して返す。"""
    me = m.p1 if m.p1.name == nm else m.p2
    opp = m.p2 if m.p1.name == nm else m.p1
    scal = E.encode(m, nm, vocab)["scalars"]        # 共有スカラー（両表現に同梱）
    bag = np.zeros(vlen, dtype=np.float32)
    for c in ([me.leader, opp.leader] + list(me.field) + list(opp.field) + list(me.hand)):
        if c is not None:
            bag[vocab.get(getattr(c.master, "card_id", None), 0)] += 1.0
    r1 = np.concatenate([scal, bag])

    def pool(cards):
        vs = [fps.get(getattr(c.master, "card_id", None), _zero()) for c in cards if c is not None]
        return np.mean(vs, axis=0) if vs else _zero()
    r2 = np.concatenate([
        scal,
        fps.get(getattr(me.leader.master, "card_id", None), _zero()) if me.leader else _zero(),
        fps.get(getattr(opp.leader.master, "card_id", None), _zero()) if opp.leader else _zero(),
        pool(list(me.field)), pool(list(opp.field)), pool(list(me.hand)),
    ])
    return r1, r2


def _play_and_sample(leader_ids, db, vocab, fps, n_games, ply_cap, sample_every, rng):
    """ランダムプレイで局面を収集。教師は2種:
      y_eval    = L1 静的評価（密・deck非依存寄り）
      y_outcome = その局面の手番プレイヤーが最終的に勝ったか（±1・AlphaZero流・card信号あり）
    return: R1[N,·], R2[N,·], Yeval[N], Yout[N]（未決着ゲームの Yout は np.nan）
    """
    R1, R2, Yeval, Yout = [], [], [], []
    vlen = len(vocab) + 1
    for _g in range(n_games):
        la = leader_ids[rng.randrange(len(leader_ids))]
        lb = leader_ids[rng.randrange(len(leader_ids))]
        l1, c1 = build_deck(db, "p1", la)
        l2, c2 = build_deck(db, "p2", lb)
        m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
        m.start_game()
        samples = []   # (r1, r2, y_eval, to_move_name)
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            name = pa[0]
            actor = m.p1 if m.p1.name == name else m.p2
            legal = m.get_legal_actions(actor)
            if not legal:
                break
            if ply % sample_every == 0 and m.turn_count >= 2:
                try:
                    ye = float(cpu_ai.evaluate(m, name))
                    r1, r2 = _encode_pair(m, name, vocab, fps, vlen)
                    samples.append((r1, r2, ye, name))
                except Exception:
                    pass
            try:
                cpu_ai._apply_move_inplace(m, name, legal[rng.randrange(len(legal))])
            except Exception:
                break
            ply += 1
        w = m.winner
        for r1, r2, ye, nm in samples:
            R1.append(r1); R2.append(r2); Yeval.append(ye)
            Yout.append(np.nan if w is None else (1.0 if w == nm else -1.0))
    return np.array(R1), np.array(R2), np.array(Yeval), np.array(Yout)


def _ridge(Xtr, ytr, Xte, yte, lam):
    """center-only リッジ（疎な bag 列を z-score で暴発させない）。test の R² を返す。
    R² は y の線形変換で不変なので y は中心化のみ。"""
    mu = Xtr.mean(0)
    Xtr = Xtr - mu; Xte = Xte - mu
    ym = ytr.mean()
    ytr = ytr - ym; yte = yte - ym
    d = Xtr.shape[1]
    w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2)) + 1e-9
    return 1.0 - ss_res / ss_tot


_LAMBDAS = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]


def _ridge_fit_eval(Xtr, ytr, Xva, yva, Xte, yte):
    """λ を検証(in-dist val)で選び、その λ で in-dist val と held-out の R² を返す。"""
    best_lam, best_va = _LAMBDAS[0], -1e9
    for lam in _LAMBDAS:
        r = _ridge(Xtr, ytr, Xva, yva, lam)
        if r > best_va:
            best_va, best_lam = r, lam
    r_out = _ridge(Xtr, ytr, Xte, yte, best_lam)
    return best_va, r_out, best_lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--ply-cap", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--holdout", choices=["color", "leader"], default="color",
                    help="color=黄を丸ごと除外(被覆ゼロ) / leader=黄の一部だけ除外(被覆あり)")
    ap.add_argument("--holdout-k", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = FP.build_fingerprints(db)
    train_leaders, held_leaders = _leaders_split(db, args.holdout, args.holdout_k, rng)
    print(f"holdout={args.holdout}  leaders: train={len(train_leaders)}  held-out={len(held_leaders)}")
    if not held_leaders or not train_leaders:
        print("色分割に失敗"); sys.exit(1)

    print("収集: train色 局面 ...")
    R1tr, R2tr, YEtr, YOtr = _play_and_sample(train_leaders, db, vocab, fps,
                                              args.games, args.ply_cap, args.sample_every, rng)
    print("収集: held-out(黄) 局面 ...")
    R1h, R2h, YEh, YOh = _play_and_sample(held_leaders, db, vocab, fps,
                                          max(args.games // 2, 20), args.ply_cap, args.sample_every, rng)
    S = slice(0, E_SCALARS)   # 共有スカラー列（両表現の先頭 14 次元）

    def probe(teacher_name, Ytr, Yh):
        # 決着した局面のみ（outcome は NaN を除外）
        mtr = ~np.isnan(Ytr); mh = ~np.isnan(Yh)
        R1t, R2t, yt = R1tr[mtr], R2tr[mtr], Ytr[mtr]
        R1e, R2e, yh = R1h[mh], R2h[mh], Yh[mh]
        n = len(yt); cut = int(n * 0.8)
        idx = list(range(n)); rng.shuffle(idx)
        tr, va = idx[:cut], idx[cut:]
        print(f"\n=== 教師={teacher_name}  states: train={n}  held-out={len(yh)} ===")

        def run(name, Xt, Xe):
            r_in, r_out, lam = _ridge_fit_eval(Xt[tr], yt[tr], Xt[va], yt[va], Xe, yh)
            print(f"  {name:20s} R²(in-dist)={r_in:+.3f}  R²(held-out黄)={r_out:+.3f}  (λ={lam})")
            return r_in, r_out

        s_in, s_out = run("scalars-only(基準)", R1t[:, S], R1e[:, S])
        a_in, a_out = run("R1 identity-bag", R1t, R1e)
        b_in, b_out = run("R2 fingerprint", R2t, R2e)
        print(f"  カード表現の上乗せ ΔR²(held-out): R1={a_out - s_out:+.3f}  R2={b_out - s_out:+.3f}"
              f"   [in-dist: R1={a_in - s_in:+.3f} R2={b_in - s_in:+.3f}]")
        return (a_in - s_in, a_out - s_out), (b_in - s_in, b_out - s_out)

    probe("L1 evaluate（密・弱信号）", YEtr, YEh)
    r1o, r2o = probe("game outcome（±1・card信号）", YOtr, YOh)

    print("\n判定（outcome 教師・held-out黄でのカード表現の上乗せ）:")
    print(f"  R1 identity Δ={r1o[1]:+.3f}   R2 fingerprint Δ={r2o[1]:+.3f}")
    if r2o[1] > max(r1o[1], 0) + 0.01:
        print("  → R2(fingerprint) が未知アーキタイプへ転移し R1(identity) は転移しない＝フィンガープリント化 GO")
    elif r2o[1] > r1o[1]:
        print("  → R2 が R1 を上回る（弱GO）。本走の held-out 勝率ゲートで最終判定")
    else:
        print("  → 差が出ない。サンプル増 or 本走 held-out 勝率ゲートで判定")


if __name__ == "__main__":
    main()
