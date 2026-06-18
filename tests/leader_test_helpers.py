"""リーダーカード挙動テスト用ヘルパ。

仕様書 (docs/leader_specs/) のテストケースを pytest 化する際の共通基盤。
実カードDB(CardLoader)から本物のパース済み能力を読み、汎用リッチ盤面の上で
能力を発動し、対話(active_interaction)を駆動して盤面結果を検証する。

設計方針:
  - **テキスト準拠の正しい挙動** をアサートする。現実装にバグがある場合は
    テストが失敗するので、呼び出し側で @pytest.mark.xfail(strict=True) を付ける
    （修正されると xpass → strict で失敗し、マーカー除去を促せる）。
  - 盤面構築は effect_coverage._build_test_state を再利用（ドン10/手札5/トラッシュ10/
    デッキ20/ライフ5/フィールド3 のリッチ盤面、リーダーを配置）。必要に応じて
    各ゾーンを上書きする。

使い方例:
    from leader_test_helpers import build, get_ability, auto_resolve, add_char, leader_power

    def test_xxx():
        gm, p1, p2, leader = build("ST10-003")
        ab = get_ability(leader.master, "ON_ATTACK")
        gm.resolve_ability(p1, ab, leader)
        auto_resolve(gm, p1)                 # 任意コスト受諾・対象自動選択
        assert leader_power(p1) == 7000
"""
import os

import conftest  # noqa: F401  (google スタブ & sys.path)

import effect_coverage as cov
from engine_helpers import make_master, make_instance
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.enums import CardType, Color, Attribute
from opcg_sim.src.utils.loader import CardLoader

_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "opcg_sim", "data", "opcg_cards.json",
)
_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = CardLoader(_DATA)
        _DB.load()
    return _DB


def leader_master(card_id):
    """実カードDBからパース済みの CardMaster を取得する。"""
    m = db().get_card(card_id)
    assert m is not None, f"カードが見つかりません: {card_id}"
    assert m.type == CardType.LEADER, f"{card_id} はリーダーではありません: {m.type}"
    return m


def build(card_id):
    """汎用リッチ盤面を構築し、(gm, p1, p2, leader_instance) を返す。

    p1 がターンプレイヤーでリーダーは card_id。p1 はドン10/手札5/トラッシュ10/
    デッキ20/ライフ5/フィールド3(フィラー)。p2 はフィールド3/手札3/ドン5/デッキ20/ライフ5。
    """
    m = leader_master(card_id)
    gm, p1, p2, source = cov._build_test_state(m)
    return gm, p1, p2, p1.leader


# ---------------------------------------------------------------------------
# 能力選択
# ---------------------------------------------------------------------------

def _trig(ab):
    return ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)


def abilities_of(master, trigger=None):
    abs_ = list(master.abilities or [])
    if trigger is None:
        return abs_
    return [a for a in abs_ if _trig(a) == trigger]


def get_ability(master, trigger, n=0):
    """trigger 種別の n 番目の能力を返す。"""
    matches = abilities_of(master, trigger)
    assert matches, f"{master.card_id} に {trigger} 能力がありません"
    assert n < len(matches), f"{master.card_id} の {trigger} 能力は {len(matches)} 個のみ"
    return matches[n]


# ---------------------------------------------------------------------------
# 盤面構築補助
# ---------------------------------------------------------------------------

_COLOR = {
    "赤": Color.RED, "緑": Color.GREEN, "青": Color.BLUE,
    "紫": Color.PURPLE, "黒": Color.BLACK, "黄": Color.YELLOW,
}
_ATTR = {
    "斬": Attribute.SLASH, "打": Attribute.STRIKE, "射": Attribute.SHOOT,
    "特": Attribute.SPECIAL, "知": Attribute.WISDOM,
}


def make_char(owner, *, name="テスト", cost=1, power=1000, counter=1000,
              traits=None, colors=None, attribute=Attribute.SLASH,
              effect_text="", abilities=(), card_id=None):
    """フィールドに置くキャラの CardInstance を生成する（場には追加しない）。"""
    master = make_master(
        card_id=card_id or f"TC-{name}", name=name, type=CardType.CHARACTER,
        cost=cost, power=power, counter=counter, attribute=attribute,
        traits=traits or [], effect_text=effect_text, abilities=abilities,
    )
    inst = CardInstance(master, owner.name)
    return inst


def add_char(player, **kw):
    """make_char で作ったキャラを player.field に追加し、その instance を返す。"""
    rest = kw.pop("rest", False)
    inst = make_char(player, **kw)
    inst.is_rest = rest
    player.field.append(inst)
    return inst


def clear_field(player):
    player.field = []


def set_life(player, n):
    """ライフ枚数を n に揃える（フィラーで増減）。"""
    cur = len(player.life)
    if n < cur:
        player.life = player.life[:n]
    else:
        for _ in range(n - cur):
            player.life.append(CardInstance(make_master(card_id="LIFE"), player.name))


# ---------------------------------------------------------------------------
# 対話駆動
# ---------------------------------------------------------------------------

def auto_resolve(gm, player, plan=None, limit=20):
    """active_interaction を駆動して解決まで進める。

    plan: 各ステップの payload を順に与えるリスト（省略時は賢い既定）:
      - CONFIRM_OPTIONAL / CONFIRM_TRIGGER → accept (受諾)
      - SELECT_TARGET / SELECT_RESOURCE → constraints.min 枚（最低1枚）を先頭から選択
      - CHOICE → index 0
      - その他 → index 0

    返り値: 実行したステップ数。
    """
    steps = 0
    plan = list(plan or [])
    while gm.active_interaction and steps < limit:
        ia = gm.active_interaction
        at = ia.get("action_type", "")
        if plan:
            payload = plan.pop(0)
        elif at in ("CONFIRM_OPTIONAL", "CONFIRM_TRIGGER"):
            payload = {"accepted": True}
        elif at in ("SELECT_TARGET", "SELECT_RESOURCE"):
            cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            cons = ia.get("constraints") or {}
            mn = cons.get("min", 0)
            mx = cons.get("max", 1)
            if mx is not None and mx < 0:
                n = len(cands)
            else:
                n = max(mn, 1) if cands else 0
                if mx:
                    n = min(n, mx)
            payload = {"selected_uuids": cands[:n], "index": 0}
        elif at == "CHOICE":
            payload = {"selected_uuids": [], "index": 0}
        else:
            payload = {"selected_uuids": [], "index": 0}
        gm.resolve_interaction(player, payload)
        steps += 1
    return steps


def select_uuids(uuids, index=0):
    """SELECT_TARGET/RESOURCE 用 payload を作る。"""
    return {"selected_uuids": list(uuids), "index": index}


def confirm(accepted=True):
    return {"accepted": accepted}


def choose(index):
    return {"selected_uuids": [], "index": index}


# ---------------------------------------------------------------------------
# 観測
# ---------------------------------------------------------------------------

def leader_power(player, my_turn=True):
    return player.leader.get_power(my_turn)


def don_total(player):
    return len(player.don_active) + len(player.don_rested)


def set_don(player, *, active=0, rested=0):
    """player のコストエリアを active 枚・rested 枚に作り直す（付与系テスト用）。

    カード効果「レストのドン‼N枚まで付与」は“既にレスト状態のドン”のみを付与し、
    アクティブのドンは巻き込まない（アクティブのドン付与は基本アクション側の役割）。
    その検証用に、レスト/アクティブの枚数を明示セットする。付与中ドンはクリアする。"""
    player.don_active = [DonInstance(owner_id=player.name) for _ in range(active)]
    rd = []
    for _ in range(rested):
        d = DonInstance(owner_id=player.name)
        d.is_rest = True
        rd.append(d)
    player.don_rested = rd
    player.don_attached_cards = []


def zone_counts(player):
    return {
        "hand": len(player.hand), "field": len(player.field),
        "trash": len(player.trash), "deck": len(player.deck),
        "life": len(player.life),
        "don_active": len(player.don_active), "don_rested": len(player.don_rested),
    }
