"""ATTACK_DISABLE リーダーと攻撃列挙の整合（bb3 発見の欠陥C・2026-08-13）。

**欠陥**: 「このリーダーはアタックできない」（PASSIVE → timed_flags/flags "ATTACK_DISABLE"・
ビビ OP04-001／レベッカ OP04-039/OP15-039／ベガパンク OP07-097／しらほし OP11-022／
アイスバーグ OP03-058 等の実在7枚）を持つ**リーダー**の攻撃が合法手列挙に載るのに、
`declare_attack` は ValueError で拒否する——場キャラの列挙は ATTACK_DISABLE を見ていたが
リーダーの列挙分岐だけ見ていなかった。実害は API 合法手の不正提示と探索の apply 失敗
（bb3 合成リーダー監査 実測: 300局中25局 APPLY_NONE）。

**固定する性質**: ATTACK_DISABLE を持つリーダーは攻撃者として列挙されない
（flags / timed_flags の両方・実カード OP04-001 でも同じ）。
"""
import conftest  # noqa: F401

import pytest

from engine_helpers import make_game, make_master
from opcg_sim.src.models.models import CardInstance, CardType
from opcg_sim.src.models.enums import Phase


def _setup(gm, p1, p2):
    p1.leader = CardInstance(make_master(card_id="T-L01", type=CardType.LEADER,
                                         power=5000, life=5), p1.name)
    p2.leader = CardInstance(make_master(card_id="T-L02", type=CardType.LEADER,
                                         power=5000, life=5), p2.name)
    gm.turn_count = 5
    gm.turn_player = p1
    gm.phase = Phase.MAIN


def _leader_attacks(gm, p):
    return [mv for mv in gm.get_legal_actions(p)
            if mv.get("action_type") == "ATTACK"
            and (mv.get("payload") or {}).get("uuid") == p.leader.uuid]


def test_leader_attack_disable_timed_flag_not_enumerated():
    gm, p1, p2 = make_game()
    _setup(gm, p1, p2)
    assert _leader_attacks(gm, p1), "前提: 素のリーダー攻撃は列挙される"
    p1.leader.timed_flags.add("ATTACK_DISABLE")
    assert not _leader_attacks(gm, p1), "timed_flags の ATTACK_DISABLE リーダーが列挙された"


def test_leader_attack_disable_flag_not_enumerated():
    gm, p1, p2 = make_game()
    _setup(gm, p1, p2)
    p1.leader.flags.add("ATTACK_DISABLE")
    assert not _leader_attacks(gm, p1), "flags の ATTACK_DISABLE リーダーが列挙された"


def test_real_vivi_leader_cannot_attack_enumeration(db_real):
    """実カード OP04-001 ネフェルタリ・ビビ: 常在「アタックできない」が列挙にも効く。"""
    gm, p1, p2 = make_game()
    _setup(gm, p1, p2)
    m = db_real.get_card("OP04-001")
    if m is None:
        pytest.skip("OP04-001 が DB に無い")
    p1.leader = CardInstance(m, p1.name)
    gm.refresh_passive_state()             # 常在効果 → ATTACK_DISABLE 付与
    assert not _leader_attacks(gm, p1), "ビビ（アタック不可リーダー）の攻撃が列挙された"


@pytest.fixture(scope="module")
def db_real():
    import conftest  # noqa: F401
    from cpu_selfplay import _load_db
    return _load_db()
