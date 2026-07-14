"""EB03-031「トラッシュのイベントの【メイン】効果を発動する」の回帰テスト。

従来 EXECUTE_MAIN_EFFECT は発生源自身の【メイン】を再展開するだけで、トラッシュから
イベントを選んでその効果を実行する機構が無かった（SPEC §6.1 / iter2 報告で要レビュー）。
パーサが EXECUTE_MAIN_EFFECT に対象（自トラッシュ・イベント・コスト7以下・1枚まで）を
付与し、resolver が選んだイベントを source として【メイン】効果を発動する。

対象: EB03-031 ヴィンスモーク・レイジュ
  「【自分のターン中】【登場時】ドン!!-1:自分のリーダーが「サンジ」の場合、
   自分のトラッシュにあるコスト7以下のイベント1枚までの、【メイン】効果を発動する。」
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.enums import TriggerType
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


def _setup(leader="OP02-026", trash=()):
    """leader 既定は「サンジ」(OP02-026)。trash はカード番号の列。"""
    p1 = Player(name="P1", deck=[], leader=inst(leader, "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 2
    src = inst("EB03-031", "P1")
    p1.field = [src]
    p1.trash = [inst(t, "P1") for t in trash]
    p1.deck = [inst("OP01-016", "P1") for _ in range(5)]
    p1.don_active = [DonInstance(owner_id="P1") for _ in range(3)]
    p1.hand = []
    return gm, p1, p2, src


def _fire(gm, p1, src, target_uuids):
    """EB03-031 の ON_PLAY を解決し、コスト→対象選択の対話を順に応答する。

    target_uuids: SELECT_TARGET で選ぶ uuid のリスト（[] で 0 枚＝発動しない）。
    """
    ab = next(a for a in src.master.abilities if a.trigger == TriggerType.ON_PLAY)
    gm.resolve_ability(p1, ab, source_card=src)
    guard = 0
    while gm.active_interaction and guard < 6:
        guard += 1
        at = gm.active_interaction.get("action_type")
        if at == "CONFIRM_OPTIONAL":          # コスト使用確認（自動誘発のコスト句は常に任意）
            gm.resolve_interaction(p1, {"accepted": True})
        elif at == "SELECT_RESOURCE":         # ドン!!-1 コスト
            gm.resolve_interaction(p1, {"selected_uuids": [p1.don_active[0].uuid]})
        elif at == "SELECT_TARGET":            # トラッシュのイベント選択
            gm.resolve_interaction(p1, {"selected_uuids": list(target_uuids)})
        else:
            break


def test_executes_selected_trash_event_main():
    """選んだイベント（OP03-056「カード2枚を引く」）の【メイン】が発動する。"""
    gm, p1, _p2, src = _setup(trash=["OP03-056"])
    event = p1.trash[0]
    _fire(gm, p1, src, [event.uuid])
    assert len(p1.hand) == 2          # イベントの【メイン】が引いた
    assert len(p1.deck) == 3
    assert event in p1.trash          # 効果発動のみ＝イベントはトラッシュに残る
    assert gm.active_interaction is None


def test_up_to_zero_executes_nothing():
    """「1枚まで」なので 0 枚選択（発動しない）も可能で、何も起こらない。"""
    gm, p1, _p2, src = _setup(trash=["OP03-056"])
    _fire(gm, p1, src, [])            # 何も選ばない
    assert len(p1.hand) == 0
    assert gm.active_interaction is None


def test_cost_over_7_event_not_candidate():
    """コスト8のイベント（OP05-058）は「コスト7以下」フィルタで候補にならず発動しない。"""
    gm, p1, _p2, src = _setup(trash=["OP05-058"])  # コスト8 のイベント
    _fire(gm, p1, src, [])
    assert len(p1.hand) == 0


def test_non_event_in_trash_not_candidate():
    """トラッシュのキャラ（非イベント）は候補にならない。"""
    gm, p1, _p2, src = _setup(trash=["OP04-061"])  # コスト3 のキャラクター
    _fire(gm, p1, src, [])
    assert len(p1.hand) == 0


def test_non_sanji_leader_does_not_fire():
    """リーダーが「サンジ」でなければ条件不成立で発動しない（コスト対話も出ない）。"""
    gm, p1, _p2, src = _setup(leader="OP01-001", trash=["OP03-056"])
    ab = next(a for a in src.master.abilities if a.trigger == TriggerType.ON_PLAY)
    gm.resolve_ability(p1, ab, source_card=src)
    assert gm.active_interaction is None
    assert len(p1.hand) == 0
