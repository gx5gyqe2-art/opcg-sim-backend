"""ドン!!を対象にしたフリーズの回帰テスト（2026-08-16）。

生成デッキの**交差対面**監査（`tests/scripts/deck_synth_audit.py --cross`）で、
OP10-033 ナミ「相手のレストのドン‼1枚までは、次の相手のリフレッシュフェイズで
アクティブにならない」が対局ごと落としていた:

    AttributeError: 'DonInstance' object has no attribute 'flags'

`FREEZE` は `card.flags` に "FREEZE" を書く**カード専用**のアクションで、ドン!!は
`is_frozen` でフリーズする（`FREEZE_DON`）。「キャラ**か**ドン‼」の択一形（OP07-026）は
Choice に分解されていたが、**ドン!!だけを対象にする形**が素の FREEZE に落ちていた。

  1. パーサ: 対象がコストエリア（＝ドン!!）なら FREEZE_DON（枚数処理）にする。
  2. エンジン: FREEZE ハンドラは flags を持たない対象を is_frozen で受ける
     （取りこぼしても**対局が死なない**ようにする。落ちるとアリーナは1ペア全損）。
"""
import functools

import conftest  # noqa: F401

from engine_helpers import make_game
from opcg_sim.src.models.enums import ActionType, TriggerType
from opcg_sim.src.models.models import CardInstance, DonInstance


@functools.lru_cache(maxsize=1)
def _db():
    from game_driver import load_db
    return load_db()


def _real(card_id, owner="P1"):
    return CardInstance(master=_db().get_card(card_id), owner_id=owner)


NAMI_TEXT = "相手のレストのドン"


def _freeze_action(node):
    """効果ツリーから FREEZE / FREEZE_DON のアクションを1つ取り出す。"""
    subs = getattr(node, "actions", None) or getattr(node, "options", None)
    if subs:
        for s in subs:
            found = _freeze_action(s)
            if found is not None:
                return found
        return None
    if getattr(node, "type", None) in (ActionType.FREEZE, ActionType.FREEZE_DON):
        return node
    for attr in ("if_true", "if_false"):
        sub = getattr(node, attr, None)
        if sub is not None:
            found = _freeze_action(sub)
            if found is not None:
                return found
    return None


def test_don_only_freeze_parses_as_freeze_don():
    """ドン!!だけを対象にするフリーズは FREEZE_DON（枚数処理）になる。"""
    nami = _real("OP10-033")
    assert NAMI_TEXT in nami.master.effect_text          # 実物のテキストで検証している
    ab = next(ab for ab in nami.master.abilities if ab.trigger == TriggerType.ON_PLAY)
    act = _freeze_action(ab.effect)
    assert act is not None
    assert act.type == ActionType.FREEZE_DON
    assert act.status == "OPPONENT"                       # 相手のドン!!を凍らせる
    assert act.value.base == 1


def test_freeze_don_keeps_the_don_rested_for_one_refresh():
    """FREEZE_DON はレストのドン!!を1回ぶんアクティブに戻さない。"""
    gm, p1, p2 = make_game()
    p2.don_rested.extend(DonInstance(owner_id="P2") for _ in range(2))
    nami = _real("OP10-033")
    act = _freeze_action(next(ab for ab in nami.master.abilities
                              if ab.trigger == TriggerType.ON_PLAY).effect)
    gm.apply_action_to_engine(p1, act, [], 1, source_card=nami)
    assert sum(1 for d in p2.don_rested if d.is_frozen) == 1

    gm.refresh_all(p2)
    assert len(p2.don_rested) == 1                        # 凍っていた1枚はレストのまま
    assert p2.don_rested[0].is_frozen is False            # フリーズは1回限り
    assert len(p2.don_active) == 1


def test_freeze_handler_does_not_crash_on_a_don_target():
    """FREEZE が万一ドン!!を対象に受けても落ちない（is_frozen で受ける）。"""
    from opcg_sim.src.core.actions.per_target import freeze
    gm, p1, p2 = make_game()
    don = DonInstance(owner_id="P2")
    don.is_rest = True
    p2.don_rested.append(don)
    freeze(gm, p1, None, don, p2, None, 0, None)
    assert don.is_frozen is True


def test_character_freeze_still_uses_flags():
    """キャラのフリーズは従来どおり flags に載る（本修正で経路が変わっていない）。"""
    from opcg_sim.src.core.actions.per_target import freeze
    gm, p1, p2 = make_game()
    target = _real("OP10-033", owner="P2")
    freeze(gm, p1, None, target, p2, None, 0, None)
    assert "FREEZE" in target.flags
