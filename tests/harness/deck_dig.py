"""掘りカードを保証した合成デッキ（残ドン掘りの方針対照実験用・2026-09-02）。

なぜ要るか: `deck_synth` の合成デッキは**テーマ整合**で採るため、「登場時ドン-Xでドロー」の
コスト1キャラ（サトリ/シュラ等・全5種・全て紫）はテーマ〈空島/神官〉のエネルにしか入らず
（12枚）、他の紫リーダーには0枚。ランダム対面で腕A（`LearnedEngine(residual_dig=True)`）を
回しても、エネルを握った局でしか発火せず対照が空振りする。

本器は合成デッキを組んだ**後**に、そのリーダーで構築可能な掘りカードを一定枚数差し込む
（差し替えは末尾の非掘りカードから・同名4枚まで・50枚維持・決定論）。両席とも同じ規則で
組むので **腕A/腕B は同じデッキ**（CRN）＝差は「終了前に掘ったか」だけに保たれる。
紫を含まないリーダーには差し込めない（色不一致）ため、実験のリーダー母集団は紫を含む
34リーダー（ドン追加効果あり12／なし22）に限る＝`promotion_gate --leaders purple`。

掘りカードの判定は `cpu_learned._is_dig_card`（構造判定・カードID非依存）と同じ。
"""
import random

import deck_synth as DS
from opcg_sim.src.core.cpu_learned import _is_dig_card
from opcg_sim.src.models.models import CardInstance

DEFAULT_INJECT = 8      # 差し込み枚数（2種×4枚＝序盤に1枚は手に来る密度）


def dig_masters(db):
    """DB 中の掘りカード（構造判定・ID順で決定論）。"""
    out = []
    for cid in sorted(db.raw_db.keys()):
        c = db.get_card(cid)
        if c is not None and _is_dig_card(c):
            out.append(c)
    return out


def inject_dig(db, leader, cards, owner, n_inject=DEFAULT_INJECT, seed=0):
    """合成デッキ `cards`（CardInstance×50）に掘りカードを n_inject 枚まで差し込む（pure・決定論）。

    差し替え対象は**掘りカードでない**カードを末尾から（同名の上限を壊さない）。
    リーダーに構築不能（色不一致）なら差し込まず、そのまま返す。返り値 (cards, injected)。"""
    lm = leader.master if hasattr(leader, "master") else leader
    pool = [m for m in dig_masters(db) if DS._legal_for(lm, m)]
    if not pool or n_inject <= 0:
        return list(cards), 0
    masters = [DS._master(x) for x in cards]
    have = {}
    for m in masters:
        have[m.name] = have.get(m.name, 0) + 1
    # 差し込む札を決める（同名4枚まで・種類を回しながら積む）
    plan = []
    i = 0
    while len(plan) < n_inject and i < n_inject * len(pool):
        m = pool[i % len(pool)]
        i += 1
        if have.get(m.name, 0) < DS.MAX_COPIES:
            plan.append(m)
            have[m.name] = have.get(m.name, 0) + 1
    if not plan:
        return list(cards), 0
    # 差し替え位置: 掘りカードでない札を末尾から
    out = list(cards)
    idx = [k for k in range(len(out) - 1, -1, -1) if not _is_dig_card(DS._master(out[k]))]
    n = 0
    for k, m in zip(idx, plan):
        out[k] = CardInstance(m, owner)
        n += 1
    random.Random(seed * 7919 + 31).shuffle(out)
    return out, n


def synth_dig_deck_builder(l1_id, l2_id=None, seed=0, n_inject=DEFAULT_INJECT):
    """`run_game(deck_builder=…)` 互換。`deck_synth.synth_deck_builder` と同じ契約で、
    両席の合成デッキに掘りカードを差し込む。"""
    def _build(db, game_seed):
        l1, c1 = DS.synth_deck(db, l1_id, seed=seed, owner="p1")
        l2, c2 = DS.synth_deck(db, l2_id or l1_id, seed=seed + 1, owner="p2")
        c1, _ = inject_dig(db, l1, c1, "p1", n_inject=n_inject, seed=seed)
        c2, _ = inject_dig(db, l2, c2, "p2", n_inject=n_inject, seed=seed + 1)
        return l1, c1, l2, c2
    return _build
