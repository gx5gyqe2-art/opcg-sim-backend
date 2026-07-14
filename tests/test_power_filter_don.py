"""パワー参照対象フィルタと付与ドン!!の扱い — +1000/枚は持ち主のターン中のみ。

付与ドン!!のパワー上昇（+1000/枚）はそのカードの持ち主のターン中だけ有効
（`CardInstance.get_power(is_my_turn)`）。対象フィルタ（「パワーN以下/以上のキャラ」）も
同じ規則で評価しなければならない。従来は matcher が一律 `get_power(True)` で判定して
いたため、相手ターン中に付与ドンが残っているキャラが「パワーN以下」の除去対象から
誤って外れていた（神の裁き OP15-075 の KO が空振りした実対局が発端）。

実行:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_power_filter_don.py -q -s -p no:cacheprovider
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "harness"))

from leader_test_helpers import build, db, add_char
from opcg_sim.src.core.effects.matcher import get_target_cards
from opcg_sim.src.models.effect_types import TargetQuery
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.enums import Zone, Player


def _power_max_query(power_max):
    return TargetQuery(zone=Zone.FIELD, player=Player.OPPONENT,
                       card_type=["CHARACTER"], power_max=power_max)


def test_opponent_attached_don_not_counted_on_my_turn():
    """自ターン中: 相手キャラの付与ドンはパワーに乗らない＝「パワー3000以下」の対象になる。"""
    gm, p1, p2, L = build("OP16-060")  # p1 がターンプレイヤー
    c = add_char(p2, name="コビー", power=2000)
    c.attached_don = 3  # 相手ターン中に付与された残り
    targets = get_target_cards(gm, _power_max_query(3000), L)
    assert c in targets  # 2000 <= 3000（+3000 は持ち主のターンのみ）


def test_own_attached_don_counted_on_own_turn():
    """持ち主のターン中: 自分のキャラの付与ドンはパワーに乗る＝「パワー3000以下」から外れる。"""
    gm, p1, p2, L = build("OP16-060")
    c = add_char(p1, name="自キャラ", power=2000)
    c.attached_don = 3
    q = TargetQuery(zone=Zone.FIELD, player=Player.SELF,
                    card_type=["CHARACTER"], power_max=3000)
    targets = get_target_cards(gm, q, L)
    assert c not in targets  # 2000+3000=5000 > 3000


def test_kami_no_sabaki_kos_don_attached_character():
    """OP15-075 神の裁き【メイン】: 相手ターン付与のドンが残る相手コビー(2000)を KO できる。"""
    gm, p1, p2, L = build("OP15-058")  # リーダー「エネル」(紫)
    koby = add_char(p2, name="コビー", power=2000)
    koby.attached_don = 3
    ev = CardInstance(db().get_card("OP15-075"), p1.name)
    p1.hand.append(ev)
    gm.play_card_action(p1, ev)
    steps = 0
    while gm.active_interaction and steps < 8:
        ia = gm.active_interaction
        at = ia.get("action_type")
        cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
        if at == "SELECT_TARGET" and koby.uuid in cands:
            gm.resolve_interaction(p1, {"selected_uuids": [koby.uuid]})
        elif at == "SELECT_RESOURCE":
            gm.resolve_interaction(p1, {"selected_uuids": cands[:1]})
        else:
            gm.resolve_interaction(p1, {"accepted": True, "selected_uuids": cands[:1]})
        steps += 1
    assert koby not in p2.field
    assert koby in p2.trash
