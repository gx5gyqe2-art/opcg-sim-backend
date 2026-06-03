"""エンジンレベルのテスト補助（改善策③: 効果セマンティクスの検証基盤）。

apply_action_to_engine など gamestate の実行系を、実際の GameManager 上で
動かして盤面変化を検証するための最小ヘルパ群。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.effect_types import GameAction, TargetQuery, ValueSource
from opcg_sim.src.models.enums import Attribute, CardType, Color
from opcg_sim.src.models.models import CardInstance, CardMaster


def make_master(
    card_id="T-001",
    name="テスト",
    type=CardType.CHARACTER,
    cost=1,
    power=1000,
    counter=1000,
    attribute=Attribute.SLASH,
    traits=None,
    effect_text="",
    trigger_text="",
    life=0,
    abilities=(),
):
    return CardMaster(
        card_id=card_id,
        name=name,
        type=type,
        colors=[Color.RED],
        cost=cost,
        power=power,
        counter=counter,
        attribute=attribute,
        traits=traits or [],
        effect_text=effect_text,
        trigger_text=trigger_text,
        life=life,
        abilities=abilities,
    )


def make_instance(master=None, owner="P1", **kw):
    if master is None:
        master = make_master()
    return CardInstance(master=master, owner_id=owner, **kw)


def make_player(name="P1", leader_life=5):
    leader = make_instance(
        make_master(card_id="L-001", name=f"{name}リーダー", type=CardType.LEADER, life=leader_life),
        owner=name,
    )
    return Player(name=name, deck=[], leader=leader)


def make_game(p1_name="P1", p2_name="P2"):
    p1 = make_player(p1_name)
    p2 = make_player(p2_name)
    gm = GameManager(p1, p2)
    return gm, p1, p2


def action(type, value=0, status=None, target=None, destination=None):
    return GameAction(
        type=type,
        value=ValueSource(base=value),
        status=status,
        target=target,
        destination=destination,
    )
