"""① マリガン方策（deck非依存カーブ規則）の回帰テスト（§H）。

初手は L1 が平坦で汎用探索が無力なため、専用の軽量ルールで「序盤に動けるか」を判定する。
- 序盤キャラ（cost 1..MULL_EARLY_COST_MAX）が居ない初手 → MULLIGAN（引き直し）
- 高コスト（cost>=MULL_HIGH_COST_MIN）に偏った初手 → MULLIGAN
- それ以外 → KEEP
決定的（盤面のみ参照・RNG 不使用）＝CRN／決定論リプレイを壊さない。フラグ OFF で従来へ戻る。
"""
import conftest  # noqa: F401

from engine_helpers import make_game, make_master, make_instance
from opcg_sim.src.models.enums import CardType
from opcg_sim.src.core import cpu_ai


def _set_hand(player, specs):
    """specs=[(cost, type), ...] で player.hand を組む。"""
    player.hand[:] = [make_instance(make_master(card_id=f"C-{i}", cost=c, type=t), owner=player.name)
                      for i, (c, t) in enumerate(specs)]


def _gm_with_hand(specs):
    gm, p1, p2 = make_game()
    _set_hand(p1, specs)
    return gm, p1


CH = CardType.CHARACTER
EV = CardType.EVENT


def test_keep_balanced_curve():
    # 1/2/3/4 コストのキャラ＋イベント＝序盤に動ける健全な初手 → KEEP
    gm, p1 = _gm_with_hand([(1, CH), (2, CH), (3, CH), (4, CH), (2, EV)])
    assert cpu_ai._mulligan_keep(gm, p1.name) is True


def test_mulligan_no_early_play():
    # 序盤に出せるキャラ（cost<=3）が 0＝初動が無い事故初手 → MULLIGAN
    gm, p1 = _gm_with_hand([(4, CH), (5, CH), (4, CH), (6, CH), (4, EV)])
    assert cpu_ai._mulligan_keep(gm, p1.name) is False


def test_mulligan_too_many_highs():
    # 序盤キャラは1枚あるが高コスト（cost>=5）が3枚＝偏った手詰まり初手 → MULLIGAN
    gm, p1 = _gm_with_hand([(2, CH), (5, CH), (6, CH), (7, CH), (3, EV)])
    assert cpu_ai._mulligan_keep(gm, p1.name) is False


def test_keep_when_one_early_and_few_highs():
    # 序盤キャラ1枚＋高コスト2枚（上限内）→ KEEP（境界＝引き直しは明確に悪い手に限る）
    gm, p1 = _gm_with_hand([(3, CH), (5, CH), (6, CH), (2, EV), (2, CH)])
    assert cpu_ai._mulligan_keep(gm, p1.name) is True


def test_empty_hand_keeps():
    gm, p1 = _gm_with_hand([])
    assert cpu_ai._mulligan_keep(gm, p1.name) is True


def test_decide_uses_policy_for_mulligan_moves():
    # MULLIGAN/KEEP_HAND の2択を decide に与えると、方策の判定どおりの action_type を返す。
    gm, p1 = _gm_with_hand([(4, CH), (5, CH), (4, CH), (6, CH), (4, EV)])  # 事故初手＝MULLIGAN
    moves = [{"kind": "game", "action_type": "MULLIGAN", "payload": {}},
             {"kind": "game", "action_type": "KEEP_HAND", "payload": {}}]
    cpu_ai.set_mulligan_override(True)
    try:
        mv = cpu_ai.decide(gm, p1, "hard", moves=moves)
    finally:
        cpu_ai.set_mulligan_override(None)
    assert mv["action_type"] == "MULLIGAN"

    gm2, p1b = _gm_with_hand([(1, CH), (2, CH), (3, CH), (4, CH), (2, EV)])  # 健全＝KEEP
    cpu_ai.set_mulligan_override(True)
    try:
        mv2 = cpu_ai.decide(gm2, p1b, "hard", moves=moves)
    finally:
        cpu_ai.set_mulligan_override(None)
    assert mv2["action_type"] == "KEEP_HAND"


def test_override_off_disables_policy():
    # 方策 OFF（override False）なら mulligan ハンドルを通らず通常探索へ落ちる（=方策で即決しない）。
    # ここでは _mulligan_policy_on() が False を返すことだけ機械的に固定する。
    cpu_ai.set_mulligan_override(False)
    try:
        assert cpu_ai._mulligan_policy_on() is False
    finally:
        cpu_ai.set_mulligan_override(None)
    # override None なら module 既定（既定 ON）へ戻る。
    assert cpu_ai._mulligan_policy_on() == cpu_ai._USE_MULLIGAN_POLICY
