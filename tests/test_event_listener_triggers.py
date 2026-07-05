"""イベントリスナー誘発 — 他カードの「…が登場した時」「…キャラがKOされた時」。

パーサはタイミングタグ付きの登場リスナーを PASSIVE/YOUR_TURN/OPPONENT_TURN に写像するが、
継続効果の再計算ループは反応型（「…した時」）をスキップするため、従来は発火経路が無く
死んでいた（OP14-041/OP13-100/OP16-079）。同様に ON_KO は KO されたカード自身の【KO時】
しか解決されず、リーダー等の第三者KOリスナー（OP14-041/OP01-061/OP13-002 等の8能力）も
死んでいた。登場/KO イベント地点からリスナーを走査し誘発待ち行列へ積む経路を検証する。

実行:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_event_listener_triggers.py -q -s -p no:cacheprovider
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "harness"))

from leader_test_helpers import build, add_char
from engine_helpers import action
from opcg_sim.src.models.enums import ActionType


def _play_from_hand_via_effect(gm, player, card):
    """効果登場（PLAY_CARD）でキャラを手札から場に出す。"""
    if card in player.field:
        player.field.remove(card)
    if card not in player.hand:
        player.hand.append(card)
    gm.apply_action_to_engine(player, action(ActionType.PLAY_CARD), [card], 0)
    gm._advance_pending_triggers()


def _drain(gm, limit=6):
    """残った対話を既定応答（受諾・先頭選択）で解決する。"""
    steps = 0
    while gm.active_interaction and steps < limit:
        ia = gm.active_interaction
        cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
        responder = gm.p1 if ia.get("player_id") == gm.p1.name else gm.p2
        gm.resolve_interaction(responder, {"accepted": True, "selected_uuids": cands[:1]})
        steps += 1


# ---------------------------------------------------------------------------
# OP14-041 ボア・ハンコック 能力0
# 【相手のターン中】自分のキャラが登場した時、カード1枚を引く。
# ---------------------------------------------------------------------------

def test_op14_041_draws_when_own_char_played_on_opponent_turn():
    """相手のターン中に自分のキャラが（効果で）登場 → カード1枚を引く。"""
    gm, p1, p2, L = build("OP14-041")
    gm.turn_player = p2
    gm.opponent = p1
    ran = add_char(p1, name="ラン", power=3000, traits=["九蛇海賊団"])
    p1.field.remove(ran)
    p1.hand.append(ran)
    hand_before = len(p1.hand)
    _play_from_hand_via_effect(gm, p1, ran)
    assert ran in p1.field
    # 登場で-1、リスナーのドローで+1 → 差引 0
    assert len(p1.hand) == hand_before


def test_op14_041_no_draw_on_own_turn():
    """自分のターン中の登場では引かない（【相手のターン中】タイミング限定）。"""
    gm, p1, p2, L = build("OP14-041")
    ran = add_char(p1, name="ラン", power=3000, traits=["九蛇海賊団"])
    p1.field.remove(ran)
    p1.hand.append(ran)
    hand_before = len(p1.hand)
    _play_from_hand_via_effect(gm, p1, ran)
    assert len(p1.hand) == hand_before - 1  # 登場の-1のみ


def test_op14_041_no_draw_when_opponent_char_played():
    """相手のキャラが登場しても引かない（主語「自分のキャラ」）。"""
    gm, p1, p2, L = build("OP14-041")
    gm.turn_player = p2
    gm.opponent = p1
    foe = add_char(p2, name="ナミ", power=2000)
    hand_before = len(p1.hand)
    _play_from_hand_via_effect(gm, p2, foe)
    assert len(p1.hand) == hand_before


# ---------------------------------------------------------------------------
# OP14-041 ボア・ハンコック 能力1
# 【ドン!!×1】【ターン1回】自分の元々のパワー5000以上の《アマゾン・リリー》/《九蛇海賊団》
# キャラがKOされた時、相手のライフの上から1枚までを、持ち主の手札に加える。
# ---------------------------------------------------------------------------

def _ko(gm, actor, card):
    gm.apply_action_to_engine(actor, action(ActionType.KO), [card], 0)
    gm._advance_pending_triggers()


def test_op14_041_ko_listener_takes_opponent_life():
    """ドン×1の状態で元パワー5000の九蛇キャラがKO → 相手のライフが1枚減る。"""
    gm, p1, p2, L = build("OP14-041")
    L.attached_don = 1
    kuja = add_char(p1, name="マリーゴールド", power=5000, traits=["九蛇海賊団"])
    life_before = len(p2.life)
    _ko(gm, p2, kuja)
    _drain(gm)
    assert len(p2.life) == life_before - 1


def test_op14_041_ko_listener_requires_don():
    """【ドン!!×1】未満（付与ドン0）では発動しない。"""
    gm, p1, p2, L = build("OP14-041")
    L.attached_don = 0
    kuja = add_char(p1, name="マリーゴールド", power=5000, traits=["九蛇海賊団"])
    life_before = len(p2.life)
    _ko(gm, p2, kuja)
    _drain(gm)
    assert len(p2.life) == life_before


def test_op14_041_ko_listener_power_filter():
    """元々のパワー5000未満のキャラのKOでは発動しない。"""
    gm, p1, p2, L = build("OP14-041")
    L.attached_don = 1
    weak = add_char(p1, name="スイート", power=4000, traits=["九蛇海賊団"])
    life_before = len(p2.life)
    _ko(gm, p2, weak)
    _drain(gm)
    assert len(p2.life) == life_before


def test_op14_041_ko_listener_turn_limit():
    """【ターン1回】: 同一ターンに2体KOされても2回目は発動しない。"""
    gm, p1, p2, L = build("OP14-041")
    L.attached_don = 1
    a = add_char(p1, name="マリーゴールド", power=5000, traits=["九蛇海賊団"])
    b = add_char(p1, name="サンダーソニア", power=5000, traits=["九蛇海賊団"])
    life_before = len(p2.life)
    _ko(gm, p2, a)
    _drain(gm)
    _ko(gm, p2, b)
    _drain(gm)
    assert len(p2.life) == life_before - 1  # 1回だけ


# ---------------------------------------------------------------------------
# 第三者KOリスナーの他カード代表: OP01-061 カイドウ
# 【ドン!!×1】【自分のターン中】【ターン1回】相手のキャラがKOされた時、
# ドン!!デッキからドン!!1枚までを、アクティブで追加する。
# ---------------------------------------------------------------------------

def test_op01_061_kaido_ramps_don_when_opponent_char_koed():
    """自分のターン中に相手キャラがKO → ドン!!1枚追加（主語「相手の」の側判定）。"""
    gm, p1, p2, L = build("OP01-061")
    L.attached_don = 1
    foe = add_char(p2, name="ゾロ", power=5000)
    don_before = len(p1.don_active) + len(p1.don_rested)
    _ko(gm, p1, foe)
    _drain(gm)
    assert (len(p1.don_active) + len(p1.don_rested)) - don_before == 1


def test_op01_061_kaido_ignores_own_char_ko():
    """自分のキャラのKOでは発動しない（主語「相手のキャラ」）。"""
    gm, p1, p2, L = build("OP01-061")
    L.attached_don = 1
    own = add_char(p1, name="キング", power=6000)
    don_before = len(p1.don_active) + len(p1.don_rested)
    _ko(gm, p2, own)
    _drain(gm)
    assert (len(p1.don_active) + len(p1.don_rested)) == don_before
