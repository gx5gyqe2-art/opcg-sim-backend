"""GameManager.clone() の高速 deepcopy（CardInstance/DonInstance.__deepcopy__）の正しさ回帰。

CPU 先読みのクローンは clone が支配的コスト（profile で ~96%）。CardInstance/DonInstance に
高速 __deepcopy__ を実装して ~3 倍高速化した（§2.5.2）。本テストは「独立な深いコピー」である
ことを固定する: 可変状態（set/dict/スカラ）は独立・不変の master は共有・同一カードへの複数参照は
クローン後も同一オブジェクト（memo 整合）。
"""
import conftest  # noqa: F401

from engine_helpers import make_game, make_master, make_instance
from opcg_sim.src.models.models import CardInstance, DonInstance, CardMaster
from opcg_sim.src.models.enums import CardType


def test_card_instance_deepcopy_is_independent_and_shares_master():
    import copy
    m = make_master(card_id="T-100", name="テスト", type=CardType.CHARACTER)
    c = make_instance(m, owner="P1")
    c.power_buff = 1234
    c.flags.add("A"); c.timed_keywords.add("ブロッカー")
    c.ability_used_this_turn[0] = 2
    d = copy.deepcopy(c)
    # master は共有（不変）／インスタンスは別物
    assert d.master is c.master
    assert d is not c
    # スカラは値一致
    assert d.power_buff == 1234
    # コンテナは独立（クローンを変えても原本は不変）
    d.flags.add("B"); d.timed_keywords.add("速攻"); d.ability_used_this_turn[0] = 9
    assert "B" not in c.flags
    assert "速攻" not in c.timed_keywords
    assert c.ability_used_this_turn[0] == 2
    # 逆も独立
    c.flags.add("C")
    assert "C" not in d.flags


def test_don_instance_deepcopy_independent():
    import copy
    don = DonInstance("P1")
    don.is_rest = True; don.attached_to = "uuid-x"
    d = copy.deepcopy(don)
    assert d is not don
    assert d.is_rest is True and d.attached_to == "uuid-x"
    d.is_rest = False
    assert don.is_rest is True  # 原本不変


def test_clone_preserves_shared_reference_identity():
    """同一カードを2か所から参照していても、クローン後も同一オブジェクトを指す（memo 整合）。
    クローンの独立性（原本を変えない）も確認する。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="T-200", name="兵", type=CardType.CHARACTER,
                                     power=5000), owner="P1")
    char.power_buff = 1000
    p1.field = [char]
    # 同じインスタンスを別ゾーンからも参照（バトル中の attacker 参照等を模す）
    gm.active_battle = {"attacker": char, "target": p2.leader}

    snap = gm.clone()
    sp1 = snap.p1 if snap.p1.name == p1.name else snap.p2
    cloned_field_card = sp1.field[0]
    # クローン内の2参照は同一オブジェクト
    assert snap.active_battle["attacker"] is cloned_field_card
    # 原本とは別オブジェクト・master 共有
    assert cloned_field_card is not char
    assert cloned_field_card.master is char.master
    # クローンを変えても原本不変
    cloned_field_card.power_buff = 7777
    assert char.power_buff == 1000
