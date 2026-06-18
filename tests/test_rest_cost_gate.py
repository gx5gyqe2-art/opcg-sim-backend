"""自己レストを伴うコストの「アクティブでなければ支払えない」ゲートの回帰テスト。

「このステージ/キャラをレストにできる：」を含む【起動メイン】コストは、対象が現在
アクティブ（未レスト）でなければ支払えない（レスト済みは再レストできない）。

不具合: 対象フィルタはレスト状態を問わない（is_rest=None）ため、_can_satisfy_node の
レストコスト充足判定がレスト済みカードも候補に数え、自己レストを伴う起動メインが
レスト後も何度も撃ててしまっていた（ハチノス OP09-099）。
"""
import conftest  # noqa: F401

from engine_helpers import make_game, make_master, make_instance
from opcg_sim.src.core.effects.parser import EffectParser
from opcg_sim.src.core.effects.resolver import EffectResolver
from opcg_sim.src.models.enums import CardType

HACHINOSU_TEXT = (
    "【起動メイン】自分の手札1枚を捨て、このステージをレストにできる："
    "自分のデッキの上から3枚を見て、特徴《黒ひげ海賊団》を持つカード1枚までを公開し、"
    "手札に加える。その後、残りを好きな順番でデッキの下に置く。"
)


def _setup():
    ability = EffectParser().parse_card_text(HACHINOSU_TEXT)[0]
    gm, p1, p2 = make_game()
    stage = make_instance(
        make_master(card_id="OP09-099", name="ハチノス", type=CardType.STAGE,
                    traits=["黒ひげ海賊団"]),
        owner="P1",
    )
    p1.stage = stage
    p1.hand = [make_instance(make_master(name="手札A"), owner="P1")]
    return gm, p1, stage, ability


def test_self_rest_cost_payable_when_active():
    gm, p1, stage, ability = _setup()
    res = EffectResolver(gm)
    assert res._can_satisfy_node(p1, ability.cost, stage) is True


def test_self_rest_cost_unpayable_when_already_rested():
    """レスト済みなら自己レストコストを支払えず、起動メインを再使用できない。"""
    gm, p1, stage, ability = _setup()
    stage.is_rest = True
    res = EffectResolver(gm)
    assert res._can_satisfy_node(p1, ability.cost, stage) is False
