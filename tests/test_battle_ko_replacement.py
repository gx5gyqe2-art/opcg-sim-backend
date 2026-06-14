"""任意のバトルKO置換（「代わりに〜してもよい/できる」）の確認・拒否（限定A）の回帰テスト。

従来、バトルでKOされる際の任意置換（OP10-034 フランキー等）は常に自動採用され、被KO側が
「代わりの効果を使わずに本来のKOを受ける」選択ができなかった（accepted limitation）。
本テストは、バトルKO置換が CONFIRM_OPTIONAL として被KO側へ提示され、

  - accept  → 置換を実行（本来のKOをスキップ＝キャラは場に残る）
  - decline → 本来のKOを実行（キャラはトラッシュへ）

のいずれも戦闘後処理（_finish_attack）まで正しく完了することを固定する。
"""
import os

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.enums import Phase
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


def _battle_setup():
    """P1 リーダー(5000) が P2 のフランキー(OP10-034, 5000) を攻撃しKOする盤面。
    フランキーは【ターン1回】「バトルでKOされる場合、代わりに自分のライフの上から
    1枚を手札に加えてもよい」を持つ。P2 にライフ1枚を置き、置換を実行可能にする。"""
    p1 = Player(name="P1", deck=[], leader=inst("OP01-001", "P1"))
    p2 = Player(name="P2", deck=[], leader=inst("OP01-001", "P2"))
    gm = GameManager(player1=p1, player2=p2)
    gm.turn_player = p1
    gm.turn_count = 3
    franky = inst("OP10-034", "P2")
    franky.is_rest = True
    p2.field = [franky]
    p2.life = [inst("OP01-016", "P2")]
    gm.active_battle = {"attacker": p1.leader, "target": franky,
                        "attacker_owner": p1, "target_owner": p2, "counter_buff": 0}
    gm.phase = Phase.BATTLE_COUNTER
    return gm, p1, p2, franky


def test_battle_ko_optional_replacement_suspends_for_controller():
    """任意バトルKO置換は被KO側(P2)への CONFIRM_OPTIONAL で戦闘を中断する（まだKOしない）。"""
    gm, p1, p2, franky = _battle_setup()
    gm.resolve_attack()

    ai = gm.active_interaction
    assert ai is not None and ai.get("action_type") == "CONFIRM_OPTIONAL"
    assert ai.get("player_id") == "P2"          # 被KO側（フランキーの持ち主）が決める
    assert franky in p2.field                   # 確認中はまだKOされていない
    assert gm.active_battle is not None          # 戦闘は中断中（後処理は resume で）


def test_battle_ko_replacement_accept_skips_ko():
    """accept → 置換実行（ライフ上1枚を手札へ）。本来のKOはスキップされキャラは場に残る。"""
    gm, p1, p2, franky = _battle_setup()
    gm.resolve_attack()
    gm.resolve_interaction(p2, {"accepted": True})

    assert gm.active_interaction is None
    assert franky in p2.field                   # 置換成立＝KOされていない
    assert franky not in p2.trash
    assert len(p2.hand) == 1                     # ライフ1枚が手札へ
    assert len(p2.life) == 0
    assert gm.active_battle is None             # 戦闘後処理が完了
    assert gm.phase == Phase.MAIN


def test_battle_ko_replacement_decline_performs_real_ko():
    """decline → 置換を使わず本来のKO。キャラはトラッシュへ、ライフは手札に加わらない。"""
    gm, p1, p2, franky = _battle_setup()
    gm.resolve_attack()
    gm.resolve_interaction(p2, {"accepted": False})

    assert gm.active_interaction is None
    assert franky not in p2.field               # 本来のKOが実行された
    assert franky in p2.trash
    assert len(p2.hand) == 0                     # ライフは手札に加わっていない
    assert len(p2.life) == 1
    assert gm.active_battle is None
    assert gm.phase == Phase.MAIN


def test_battle_ko_replacement_headless_default_accepts():
    """ヘッドレス既定応答（index0=accept）は従来の自動採用と一致する（ベースライン不変の担保）。"""
    gm, p1, p2, franky = _battle_setup()
    gm.resolve_attack()
    # default_interaction_payload 相当（accepted=True）で解決
    gm.resolve_interaction(p2, gm.default_interaction_payload())

    assert gm.active_interaction is None
    assert franky in p2.field                   # 既定は置換採用＝従来挙動
    assert len(p2.hand) == 1
