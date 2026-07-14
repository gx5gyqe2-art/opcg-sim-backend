"""カード移動・ドン操作・場札上限（GameManager からの移管・ステートレス。第1引数 gm）。"""
from __future__ import annotations

import re

from ..journal import JournaledList
from ...models.models import DonInstance
from ...models.enums import Zone, CardType
from ..rules_constants import FIELD_LIMIT
from ._helpers import _nfc


def _apply_leader_don_deck_rule(gm, player: Player) -> None:
    """リーダーの「ルール上、自分のドン!!デッキはN枚になる」をドン!!デッキ枚数に反映する。
    該当する常在ルールが無ければ既定（10枚）のまま。エネル OP15-058 = 6枚。"""
    leader = getattr(player, "leader", None)
    if not leader or not getattr(leader, "master", None):
        return
    text = _nfc(getattr(leader.master, "effect_text", "") or "")
    m = re.search(_nfc(r"ドン(?:!!|‼)デッキは(\d+)枚"), text)
    if not m:
        return
    n = int(m.group(1))
    player.don_deck = JournaledList(DonInstance(owner_id=player.name) for _ in range(n))

def _find_card_by_uuid(gm, uuid: str) -> Optional[CardInstance]:
    all_players = [gm.p1, gm.p2]
    for p in all_players:
        candidates = []
        if p.leader: candidates.append(p.leader)
        if p.stage: candidates.append(p.stage)
        candidates.extend(p.hand)
        candidates.extend(p.field)
        candidates.extend(p.trash)
        candidates.extend(p.life)
        candidates.extend(p.deck)
        candidates.extend(p.temp_zone)
        for c in candidates:
            if c.uuid == uuid:
                return c
    return None

def _enforce_field_limit(gm, owner: Player) -> None:
    """owner のキャラが上限(FIELD_LIMIT)を超えていれば、超過分を選んでトラッシュさせる。
    他に進行中の対話があるときは起動しない（中断のネストを避ける）。"""
    if gm.active_interaction:
        return
    if len(owner.field) <= FIELD_LIMIT:
        return
    gm._suspend_for_field_overflow(owner)

def _suspend_for_field_overflow(gm, owner: Player) -> None:
    """場のキャラ超過分をトラッシュさせる選択を中断要求として立てる。
    candidates は新規登場分を含む owner の全キャラ（どれをトラッシュしてもよい）。"""
    excess = len(owner.field) - FIELD_LIMIT  # 通常は 1
    gm.active_interaction = {
        "player_id": owner.name,
        "action_type": "FIELD_OVERFLOW_TRASH",
        "message": f"場のキャラクターが上限({FIELD_LIMIT})を超えました。トラッシュするキャラを{excess}枚選んでください。",
        "candidates": list(owner.field),
        "selectable_uuids": [c.uuid for c in owner.field],
        "constraints": {"min": excess, "max": excess},
        "can_skip": False,
        "continuation": {"owner_name": owner.name, "count": excess},
    }

def draw_card(gm, player: Player, count: int = 1):
    for _ in range(count):
        if player.deck:
            card = player.deck.pop(0); player.hand.append(card)
    if not player.deck and not gm.winner: gm.check_victory()

def _find_card_location(gm, card: Card) -> Tuple[Optional[Player], Optional[List[Any]]]:
    for p in [gm.p1, gm.p2]:
        zones = [
            p.hand, p.field, p.life, p.trash, p.deck, p.temp_zone,
            p.don_active, p.don_rested, p.don_attached_cards
        ]
        if p.leader == card: return p, None
        if p.stage == card: return p, None
        for zone in zones:
            if card in zone: return p, zone
    return None, None

def move_card(gm, card: Card, dest_zone: Zone, dest_player: Player, dest_position: str = "BOTTOM"):
    current_owner, current_list = gm._find_card_location(card)
    
    # 領域移動時はステータスをリセット（特にトラッシュ/手札へ戻る場合）。
    # 場を離れると新規状態になるため【ターン1回】の使用回数もリセットする。
    if dest_zone in [Zone.TRASH, Zone.HAND]:
        card.reset_turn_status(clear_usage=True)

    # レスト（横向き）状態は場を離れたら必ず解除する。レストのキャラを手札/トラッシュ/
    # デッキへ戻すと横向きのまま戻ってしまう不具合を防ぐ（is_rest は場でのみ意味を持つ）。
    if dest_zone in [Zone.TRASH, Zone.HAND, Zone.DECK]:
        card.is_rest = False
        
    # フィールドから離れる場合、付与されていたドン‼をレスト状態で持ち主に返す
    if current_owner and current_list is not None and current_list is current_owner.field:
        attached_dons = [d for d in current_owner.don_attached_cards if d.attached_to == card.uuid]
        for don in attached_dons:
            current_owner.don_attached_cards.remove(don)
            don.attached_to = None
            don.is_rest = True
            current_owner.don_rested.append(don)
        card.attached_don = 0
        # 場を離れたら継続効果（timed_power/flags/keywords）を破棄する。
        gm.continuous.drop_for(card.uuid)

    # ライフ領域から離れる場合（手札/トラッシュ/デッキへ移す効果等）、「ライフが離れた時」
    # (ON_LIFE_DECREASE) を待ち行列へ積む。戦闘/効果ダメージは life.pop 済みでここを通らず、
    # それぞれの経路が別途積むため二重計上しない。実際の解決は安全な境界
    # （resolve_ability 完了時 / 対話完了時 / アクション境界）でまとめて行う。
    left_life = (current_owner is not None and current_list is not None
                 and current_list is current_owner.life)
    # 場（フィールド）からの離脱判定。キャラが場を離れた時(ON_LEAVE)誘発に使う。
    left_field = (current_owner is not None and current_list is not None
                  and current_list is current_owner.field
                  and dest_zone != Zone.FIELD)

    if current_list is not None and card in current_list: current_list.remove(card)
    elif current_owner and current_owner.stage == card: current_owner.stage = None

    if left_life:
        gm._enqueue_life_decrease(current_owner, 1)
    if left_field and card.master.type == CardType.CHARACTER:
        gm._enqueue_on_leave(card, current_owner)

    target_list = None
    if dest_zone == Zone.FIELD and card.master.type == CardType.STAGE:
        if dest_player.stage is not None: gm.move_card(dest_player.stage, Zone.TRASH, dest_player)
        dest_player.stage = card
    elif dest_zone == Zone.HAND: target_list = dest_player.hand
    elif dest_zone == Zone.FIELD: target_list = dest_player.field
    elif dest_zone == Zone.TRASH: target_list = dest_player.trash
    elif dest_zone == Zone.LIFE: target_list = dest_player.life
    elif dest_zone == Zone.DECK: target_list = dest_player.deck
    elif dest_zone == Zone.TEMP: target_list = dest_player.temp_zone
    
    if target_list is not None:
        if dest_position == "TOP": target_list.insert(0, card)
        else: target_list.append(card)

def pay_cost(gm, player: Player, cost: int, don_list: Optional[List[DonInstance]] = None):
    if don_list is not None:
        if len(don_list) < cost: raise ValueError("指定されたドン!!の数が不足しています。")
        for don in don_list:
            if don in player.don_active: player.don_active.remove(don); player.don_rested.append(don); don.is_rest = True
            elif don in player.don_attached_cards: player.don_attached_cards.remove(don); player.don_rested.append(don); don.is_rest = True; don.attached_to = None
    else:
        if len(player.don_active) < cost: raise ValueError("ドン!!が不足しています。")
        for _ in range(cost): don = player.don_active.pop(0); player.don_rested.append(don); don.is_rest = True

def _return_one_don(gm, tp: Player, don: DonInstance) -> bool:
    """ドン!!1枚を tp の場（アクティブ/レスト/付与中）から外しドン!!デッキへ戻す。
    付与中だった場合は付与先キャラの attached_don を減らしてパワー上昇を解除する。"""
    if don in tp.don_rested:
        tp.don_rested.remove(don)
    elif don in tp.don_active:
        tp.don_active.remove(don)
    elif don in tp.don_attached_cards:
        tp.don_attached_cards.remove(don)
        host = gm._find_card_by_uuid(don.attached_to) if don.attached_to else None
        if host is not None and getattr(host, "attached_don", 0) > 0:
            host.attached_don -= 1
    else:
        return False
    don.is_rest = False
    don.attached_to = None
    tp.don_deck.append(don)
    return True

def _don_pool_player(gm, player: Player, action: GameAction) -> Player:
    """ドン操作の対象プレイヤー。「相手は…」は status="OPPONENT"、
    対象クエリの player=OPPONENT でも相手を指す。既定は効果の実行者。"""
    opp = gm.p2 if player == gm.p1 else gm.p1
    if getattr(action, 'status', None) == "OPPONENT":
        return opp
    tgt = getattr(action, 'target', None)
    if tgt is not None and getattr(getattr(tgt, 'player', None), 'name', '') == 'OPPONENT':
        return opp
    return player
