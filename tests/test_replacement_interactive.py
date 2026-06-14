"""置換 sub_effect のネスト中断を UI へ提示する（Phase 2b）回帰テスト。

置換（「（場を離れる/KOされる）代わりに〜できる」）の内側選択は、従来すべて
`_auto_resolve_replacement` が保守的に自動解決していた（人間が選べない）。失われる
外側継続が無い場合（除去アクションが終端・単一対象）に限り、内側の中断を**そのまま
UI へ提示**して被保護側に選ばせる。外側継続が残る場合は従来どおり自動解決する。

対象例: OP05-032「【ターン1回】このキャラがKOされる場合、代わりに「ピーカ」以外の
自分のコスト3以上のキャラ1枚までを、レストにできる」。
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.effects.resolver import EffectResolver
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.effect_types import GameAction, TargetQuery, Sequence, ValueSource
from opcg_sim.src.models.enums import ActionType, Player as P, Zone
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


def _setup():
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 3
    victim = inst("OP05-032", "P2")      # KO 置換持ち
    rest_target = inst("EB01-022", "P2")  # コスト6（レスト候補）
    p2.field = [victim, rest_target]
    return gm, p1, p2, victim, rest_target


def _ko_query():
    # cost_max=5 で victim(OP05-032 コスト4)に一意化（rest_target は コスト6 で除外）。
    # KO 自身の対象選択で中断させず、置換まで到達させる。
    return GameAction(type=ActionType.KO,
                      target=TargetQuery(player=P.OPPONENT, zone=Zone.FIELD,
                                         card_type=["CHARACTER"], cost_max=5),
                      raw_text="相手のコスト5以下のキャラをKO")


def test_terminal_removal_presents_nested_choice():
    """終端の除去 → 置換の内側選択を UI 提示（自動解決しない）。被保護側が選んでレスト。"""
    gm, p1, p2, victim, rest_target = _setup()
    res = EffectResolver(gm)
    res.execution_stack = [_ko_query()]   # 終端（後続なし）→ can_suspend
    res._process_stack(p1, p1.leader)

    ai = gm.active_interaction
    assert ai is not None and ai.get("action_type") == "SELECT_TARGET"
    assert ai.get("player_id") == "P2"     # 被保護側（victim の持ち主）が選ぶ
    assert victim in p2.field              # 置換成立＝KO されていない

    # 被保護側が rest_target を選んで解決 → レストになる
    gm.resolve_interaction(p2, {"selected_uuids": [rest_target.uuid]})
    assert gm.active_interaction is None
    assert rest_target.is_rest is True
    assert victim in p2.field


def test_nonterminal_removal_presents_choice_and_defers_continuation():
    """除去の後続がある（B1: 非終端・単一対象）場合も、置換の内側選択を UI へ提示し、
    後続継続（DRAW）は deferred フレームへ退避。内側選択の解決後に後続が再開する。"""
    gm, p1, p2, victim, rest_target = _setup()
    res = EffectResolver(gm)
    # KO の後にもう1アクション（DRAW）を積む → execution_stack 非空（後続あり）。
    p1.deck = [inst("OP01-016", "P1") for _ in range(3)]
    res.execution_stack = [GameAction(type=ActionType.DRAW, value=ValueSource(base=1), raw_text="1枚引く"),
                           _ko_query()]   # pop 順で KO→DRAW
    res._process_stack(p1, p1.leader)

    ai = gm.active_interaction
    assert ai is not None and ai.get("action_type") == "SELECT_TARGET"
    assert ai.get("player_id") == "P2"      # 被保護側が選ぶ（自動解決しない）
    assert victim in p2.field               # 置換成立＝KO されていない
    assert len(p1.hand) == 0                # 後続 DRAW はまだ実行されていない（退避中）
    assert len(gm._deferred_continuations) == 1  # 外側継続が退避されている

    # 被保護側が rest_target を選んで解決 → レスト後、退避した DRAW が再開される。
    gm.resolve_interaction(p2, {"selected_uuids": [rest_target.uuid]})
    assert gm.active_interaction is None
    assert rest_target.is_rest is True
    assert victim in p2.field
    assert len(p1.hand) == 1                # 後続の DRAW が再開・実行された（外側継続が保たれる）
    assert gm._deferred_continuations == []  # 退避は消化済み


def test_multitarget_removal_defers_remaining_targets():
    """複数対象除去（B2）: 先頭対象の置換が内側選択を提示したら、残対象を退避してループを抜け、
    内側選択の解決後に残対象へ除去を再開する。"""
    gm, p1, p2, victim, rest_target = _setup()
    vanilla = inst("OP01-016", "P2")        # 置換を持たない素のキャラ（コスト1）
    p2.field = [victim, vanilla, rest_target]
    ko = GameAction(type=ActionType.KO, raw_text="相手のキャラ2枚をKO")

    # victim → vanilla の順で 2 体を KO 対象に明示。victim の置換が先に中断する。
    gm.apply_action_to_engine(p1, ko, [victim, vanilla], 0)

    ai = gm.active_interaction
    assert ai is not None and ai.get("action_type") == "SELECT_TARGET"
    assert ai.get("player_id") == "P2"
    assert victim in p2.field               # 置換成立＝KO されていない
    assert vanilla in p2.field              # 残対象はまだ未処理（退避中）
    assert len(gm._deferred_continuations) == 1

    # 内側選択を解決 → 退避していた残対象(vanilla)の KO が再開される。
    gm.resolve_interaction(p2, {"selected_uuids": [rest_target.uuid]})
    assert gm.active_interaction is None
    assert rest_target.is_rest is True
    assert victim in p2.field               # 置換で守られた
    assert vanilla not in p2.field          # 残対象は KO された
    assert vanilla in p2.trash
    assert gm._deferred_continuations == []
