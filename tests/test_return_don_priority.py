"""RETURN_DON（「ドン!!をドン!!デッキに戻す」）の既定選択は**レスト優先**であることの回帰。

実機プレイテスト（エネル＝OP15-058 のドン返却ループ）で、**レストのドンが有るのにアクティブのドンを
戻す**無駄が観測された。RETURN_DON の対象選択は CPU 探索の対象外で（`_selection_moves` は SELECT_RESOURCE を
列挙しない）、常に `default_interaction_payload`（候補先頭から min 枚）で既定解決される。よって候補順＝優先順位。

戻すなら損の少ない順＝**レスト（今ターン消費済み）＞アクティブ（今ターン使える）＞付与中（+1000を失う）**。
本テストは候補がこの順（レスト先頭）で、既定解決がレストのドンを選ぶことを固定する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_return_don_priority.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.effects.resolver import EffectResolver
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.effect_types import GameAction
from opcg_sim.src.models.enums import ActionType


def _game_with_don(active: int, rested: int):
    gm = GameManager(Player("p1", [], None), Player("p2", [], None))
    gm.turn_player = gm.p1
    gm.turn_count = 4
    p = gm.p2
    p.don_active[:] = [DonInstance(owner_id="p2") for _ in range(active)]
    p.don_rested[:] = [DonInstance(owner_id="p2") for _ in range(rested)]
    for d in p.don_active:
        d.is_rest = False
    for d in p.don_rested:
        d.is_rest = True
    return gm, p


def test_return_don_candidates_rested_first():
    """候補は レスト→アクティブ→付与中 の順（既定解決が一番損の少ないレストを取る）。"""
    gm, p = _game_with_don(active=2, rested=3)
    r = EffectResolver(gm)
    act = GameAction(type=ActionType.RETURN_DON)
    assert r._suspend_for_don_selection(p, act, p.leader, 1) is True
    cands = gm.active_interaction["candidates"]
    # 先頭 rested 3 枚 → 続いて active 2 枚
    assert [c.is_rest for c in cands] == [True, True, True, False, False], \
        f"候補順がレスト先頭でない: {[c.is_rest for c in cands]}"


def test_return_don_default_returns_rested_not_active():
    """既定解決（default_interaction_payload）は**レストのドン**を選ぶ＝アクティブを温存する。"""
    gm, p = _game_with_don(active=2, rested=3)
    r = EffectResolver(gm)
    assert r._suspend_for_don_selection(p, GameAction(type=ActionType.RETURN_DON), p.leader, 1) is True
    pending = gm.get_pending_request()
    payload = gm.default_interaction_payload(pending)
    sel = payload["selected_uuids"]
    assert len(sel) == 1
    rested_uuids = {d.uuid for d in p.don_rested}
    active_uuids = {d.uuid for d in p.don_active}
    assert sel[0] in rested_uuids, "アクティブのドンを戻そうとしている（レストが有るのに）"
    assert sel[0] not in active_uuids


def test_return_don_falls_back_to_active_when_no_rested():
    """レストが無ければアクティブを戻す（候補が尽きないこと＝壊さない）。"""
    gm, p = _game_with_don(active=2, rested=0)
    r = EffectResolver(gm)
    assert r._suspend_for_don_selection(p, GameAction(type=ActionType.RETURN_DON), p.leader, 1) is True
    pending = gm.get_pending_request()
    payload = gm.default_interaction_payload(pending)
    assert payload["selected_uuids"][0] in {d.uuid for d in p.don_active}
