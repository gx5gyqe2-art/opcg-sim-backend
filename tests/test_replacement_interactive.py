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


def test_nonterminal_removal_auto_resolves():
    """除去の後続がある（外側継続が失われる）場合は従来どおり自動解決＝中断を提示しない。"""
    gm, p1, p2, victim, rest_target = _setup()
    res = EffectResolver(gm)
    # KO の後にもう1アクション（DRAW）を積む → execution_stack 非空 → can_suspend=False
    p1.deck = [inst("OP01-016", "P1") for _ in range(3)]
    res.execution_stack = [GameAction(type=ActionType.DRAW, value=ValueSource(base=1), raw_text="1枚引く"),
                           _ko_query()]   # pop 順で KO→DRAW
    res._process_stack(p1, p1.leader)

    assert gm.active_interaction is None    # 自動解決済み・宙吊りなし
    assert victim in p2.field               # 置換成立
    assert victim.is_rest or rest_target.is_rest  # 自動解決で先頭候補をレスト
    assert len(p1.hand) == 1                # 後続の DRAW も実行された（外側継続が保たれる）
