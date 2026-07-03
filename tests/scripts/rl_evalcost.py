"""争点の実測: 「numpy極小NN推論はL1評価より桁違いに重い」は本当か（dev・一回限り）。

レビュアー反論の唯一の経験的前提＝NN1回推論 >> L1評価1回（桁違い）を直接計測で adjudicate する。
中盤局面で (a) L1 cpu_ai.evaluate 1回, (b) net forward batch=1 1回, (c) batched per-sample を比較。
実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/rl_evalcost.py
"""
import random
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db
import rl_encoder as E
import rl_net as N


def _advance(m, n_steps):
    """ランダム手で中盤まで進める（評価対象の現実的な盤面を作る）。"""
    for _ in range(n_steps):
        if m.winner is not None:
            return
        pa = m.pending_actor_action()
        if not pa:
            return
        pid, _ = pa
        actor = m.p1 if m.p1.name == pid else m.p2
        moves = m.get_legal_actions(actor)
        if not moves:
            return
        mv = random.choice(moves)
        try:
            if mv["kind"] == "battle":
                action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
        except Exception:
            return


def _time(fn, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    db = _load_db()
    vocab = E.build_vocab(db)
    random.seed(7)
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    _advance(m, 30)
    pid = "p1"
    net = N.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(), seed=0)

    # ウォームアップ（numpy/BLAS 初期化を除外）。
    for _ in range(50):
        cpu_ai.evaluate(m, pid, see_opp_hand=False)
    enc = E.encode(m, pid, vocab)
    one = {k: np.stack([enc[k]]) for k in ("scalars", "field", "card_idx")}
    for _ in range(50):
        net.predict(one)

    # (a) L1 評価 1回
    t_l1 = _time(lambda: cpu_ai.evaluate(m, pid, see_opp_hand=False), 2000)
    # encode コスト（NN経路は毎回 encode が要る・公平に内訳を出す）
    t_enc = _time(lambda: E.encode(m, pid, vocab), 2000)
    # (b) net forward batch=1
    t_b1 = _time(lambda: net.predict(one), 2000)
    # (c) batched per-sample（CPUでも並列ゲーム/葉をまとめれば効く headroom）
    out = {}
    for B in (16, 64, 256):
        big = {k: np.repeat(one[k], B, axis=0) for k in one}
        per = _time(lambda: net.predict(big), 500) / B
        out[B] = per

    us = 1e6
    print("=== 1局面あたりコスト実測 (中盤盤面・ウォームアップ済) ===")
    print(f"(a) L1 cpu_ai.evaluate      : {t_l1*us:8.1f} µs")
    print(f"(b) net forward batch=1     : {t_b1*us:8.1f} µs   (vs L1 ×{t_b1/t_l1:.2f})")
    print(f"    encode (NN経路の前処理)  : {t_enc*us:8.1f} µs")
    print(f"    NN経路 合計(encode+fwd)  : {(t_enc+t_b1)*us:8.1f} µs   (vs L1 ×{(t_enc+t_b1)/t_l1:.2f})")
    for B, per in out.items():
        print(f"(c) net forward batch={B:<3d}  : {per*us:8.1f} µs/局面 (vs L1 ×{per/t_l1:.2f})")
    print("\n争点判定: (b) が L1 と同オーダー(×0.1〜×10)なら『NN推論は桁違いに重い』は偽。"
          "(c) で batch 化すると per-局面がさらに下がる＝CPUでもバッチでスループット稼げる。")


if __name__ == "__main__":
    main()
