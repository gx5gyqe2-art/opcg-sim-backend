"""継続付与型のバトルKO置換（EB02-030）の配線テスト。

EB02-030「【カウンター】自分のキャラすべては、このターン中、バトルでKOされる場合、代わりに
自分の手札1枚を捨てることができる」は、場に残らないイベント由来のため、被KOキャラ側からは
発生源（トラッシュ済みのイベント）を見つけられない。カウンターとして撃った時点で player へ
this-turn の置換を付与し、_find_replacement がそれを参照することを検証する。

付与型置換も任意（できる）なので、限定A の機構で CONFIRM_OPTIONAL として被KO側へ提示される。
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.enums import Phase
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data", "opcg_cards.json")
_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = CardLoader(DATA)
        _DB.load()
    return _DB


def inst(cid, owner="P1"):
    return CardInstance(db().get_card(cid), owner)


def _setup(grant=True):
    """P1 リーダー(5000) が P2 のキャラ(Nami 2000) をバトルでKOする盤面。
    P2 は EB02-030 由来の this-turn 置換（手札1枚を捨てて代わりにKO回避）を付与済み。"""
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 3
    victim = inst("OP01-016", "P2")  # Nami 2000
    victim.is_rest = True
    p2.field = [victim]
    p2.hand = [inst("OP01-016", "P2"), inst("OP01-016", "P2")]  # 捨て札の的
    if grant:
        gm._register_granted_replacements(p2, inst("EB02-030", "P2"))
    gm.active_battle = {"attacker": p1.leader, "target": victim,
                        "attacker_owner": p1, "target_owner": p2, "counter_buff": 0}
    gm.phase = Phase.BATTLE_COUNTER
    return gm, p1, p2, victim


def test_grant_registration_marks_optional_battle_ko():
    """付与は status=BATTLE_KO・is_optional=True（「できる」）で this-turn 有効として登録される。"""
    gm, p1, p2, victim = _setup()
    assert len(p2.granted_replacements) == 1
    g = p2.granted_replacements[0]
    assert g["status"] == "BATTLE_KO"
    assert g["is_optional"] is True
    assert g["expire_turn"] == gm.turn_count


def test_granted_replacement_offered_and_accept_skips_ko():
    """付与型置換も被KO側へ CONFIRM_OPTIONAL で提示。accept → 手札1枚を捨ててKO回避。"""
    gm, p1, p2, victim = _setup()
    gm.resolve_attack()
    ai = gm.active_interaction
    assert ai is not None and ai.get("action_type") == "CONFIRM_OPTIONAL"
    assert ai.get("player_id") == "P2"
    assert victim in p2.field

    gm.resolve_interaction(p2, {"accepted": True})
    assert gm.active_interaction is None
    assert victim in p2.field          # KO 回避
    assert len(p2.hand) == 1           # 手札1枚を捨てた
    assert len(p2.trash) == 1
    assert gm.active_battle is None


def test_granted_replacement_decline_performs_ko():
    """decline → 付与を使わず本来のKO。キャラはトラッシュへ、手札は減らない。"""
    gm, p1, p2, victim = _setup()
    gm.resolve_attack()
    gm.resolve_interaction(p2, {"accepted": False})
    assert gm.active_interaction is None
    assert victim not in p2.field
    assert victim in p2.trash
    assert len(p2.hand) == 2


def test_no_grant_means_plain_ko():
    """付与が無ければ従来どおり素直にKOされる（中断もしない）。"""
    gm, p1, p2, victim = _setup(grant=False)
    gm.resolve_attack()
    assert gm.active_interaction is None
    assert victim not in p2.field
    assert victim in p2.trash


def test_grant_expires_next_turn():
    """this-turn 付与は次ターン以降は無効（expire_turn 超過で参照されない）。"""
    gm, p1, p2, victim = _setup()
    gm.turn_count += 1  # ターンが進む
    gm.resolve_attack()
    assert gm.active_interaction is None   # 付与は失効 → 提示されない
    assert victim not in p2.field          # 素直にKO
    assert victim in p2.trash


def test_apply_counter_registers_grant():
    """EB02-030 をカウンターとして撃つと、解決時に this-turn 置換が付与される（統合）。"""
    gm, p1, p2, victim = _setup(grant=False)
    # P2 にコスト(2)分のアクティブドンを用意
    for _ in range(2):
        if p2.don_deck:
            p2.don_active.append(p2.don_deck.pop(0))
    eb = inst("EB02-030", "P2")
    p2.hand.append(eb)
    don = p2.don_active[:2]
    gm.apply_counter(p2, eb, don)
    assert any(g["status"] == "BATTLE_KO" for g in p2.granted_replacements)
    assert eb in p2.trash               # イベントはトラッシュへ
