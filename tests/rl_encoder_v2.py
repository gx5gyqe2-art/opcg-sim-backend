"""encoder_v2: 効果フィンガープリント state エンコーダ（レバー① 本体・次期学習の net 入力）。

docs/reports/cpu_rl_generalization_plan_20260701.md ①。pre-flight ① で汎化転移を確認した R2 表現
（scalars ＋ 効果フィンガープリント平均pool）を、再利用可能な固定長 state エンコーダに定式化する。
識別子(card_id)埋め込みは**入れない**（プローブで汎化価値≈0＝OOD負債と確定したため）。

決定的（盤面のみ・RNG不使用）。相手手札の中身は出さない（枚数のみ＝公平）。
"""
import numpy as np

import rl_fingerprint as FP
import rl_encoder as E   # scalars（14次元）は既存エンコーダを流用

DIM = 14 + 5 * FP.CARD_DIM   # scalars ＋ [自L, 相手L, 自場pool, 相手場pool, 自手札pool]


def _zero():
    return np.zeros(FP.CARD_DIM, np.float32)


def encode_v2(manager, me_name, vocab, fps):
    """to-move 視点で局面を DIM 次元の flat ベクトルへ（fps=build_fingerprints の辞書）。"""
    me = manager.p1 if manager.p1.name == me_name else manager.p2
    opp = manager.p2 if manager.p1.name == me_name else manager.p1
    scal = E.encode(manager, me_name, vocab)["scalars"]

    def fp_of(c):
        return fps.get(getattr(c.master, "card_id", None), _zero()) if c is not None else _zero()

    def pool(cards):
        vs = [fp_of(c) for c in cards if c is not None]
        return np.mean(vs, axis=0) if vs else _zero()

    return np.concatenate([
        scal,
        fp_of(me.leader), fp_of(opp.leader),
        pool(list(me.field)), pool(list(opp.field)), pool(list(me.hand)),
    ]).astype(np.float32)
