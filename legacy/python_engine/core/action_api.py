"""ルールモードのアクション適用ロジック（共通コアパス）。

`/api/game/action`・`/api/game/battle`（`opcg_sim/api/app.py`）のディスパッチ処理を
純粋関数として切り出したもの。HTTP エンドポイント・CPU ドライバ・自己対戦ランナーが
**同一コードパス** を通ることで、AI シミュレーション・自己対戦とルール本番の挙動が
乖離しないことを保証する（docs/SPEC.md §0）。

これらの関数は `manager.action_events` を変化させる（呼び出し側で事前にリセットする）。
ゲーム状態はすべて引数の `manager`（GameManager）上で完結し、ネットワーク/HTTP には依存しない。
"""
from typing import Any, Dict, List, Optional

from ..models.models import CONST
from ..models.enums import TriggerType, Phase

_C_TO_S = CONST.get('c_to_s_interface', {})
_GAME_ACTIONS = _C_TO_S.get('GAME_ACTIONS', {}).get('TYPES', {})
_BATTLE_ACTIONS = _C_TO_S.get('BATTLE_ACTIONS', {}).get('TYPES', {})

# ゲームアクション種別（shared_constants は恒等写像だが将来の改名に備え CONST 経由で解決）
ACT_PLAY = _GAME_ACTIONS.get('PLAY', 'PLAY')
ACT_TURN_END = _GAME_ACTIONS.get('TURN_END', 'TURN_END')
ACT_ATTACK = _GAME_ACTIONS.get('ATTACK', 'ATTACK')
ACT_ATTACK_CONFIRM = _GAME_ACTIONS.get('ATTACK_CONFIRM', 'ATTACK_CONFIRM')
ACT_ATTACH_DON = _GAME_ACTIONS.get('ATTACH_DON', 'ATTACH_DON')
ACT_ACTIVATE_MAIN = _GAME_ACTIONS.get('ACTIVATE_MAIN', 'ACTIVATE_MAIN')
ACT_RESOLVE_SELECTION = _GAME_ACTIONS.get('RESOLVE_EFFECT_SELECTION', 'RESOLVE_EFFECT_SELECTION')
ACT_MULLIGAN = 'MULLIGAN'
ACT_KEEP_HAND = 'KEEP_HAND'

# バトルアクション種別
BACT_SELECT_BLOCKER = _BATTLE_ACTIONS.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
BACT_SELECT_COUNTER = _BATTLE_ACTIONS.get('SELECT_COUNTER', 'SELECT_COUNTER')
BACT_PASS = _BATTLE_ACTIONS.get('PASS', 'PASS')


def _operating_card(player, card_uuid):
    """player のレスト操作対象になりうる場のカード（リーダー/キャラ/ステージ）から uuid 一致を返す。"""
    cards = []
    if player.leader:
        cards.append(player.leader)
    cards.extend(player.field)
    if player.stage:
        cards.append(player.stage)
    return next((c for c in cards if c.uuid == card_uuid), None)


def apply_game_action(manager, player, action_type: str, payload: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """メインフェイズ/ゲーム進行アクションを適用する。

    `player` は行動主体の Player。`payload` は {uuid|card_id, target_ids|target_uuid, ...}。
    末尾でアクション境界処理（_advance_pending_triggers / refresh_passive_state）を実行する。
    `manager.action_events` を返す。事前リセットは呼び出し側の責務。
    """
    payload = payload or {}
    card_uuid = payload.get("uuid") or payload.get("card_id")
    target_ids = payload.get("target_ids", [])
    target_uuid = target_ids[0] if isinstance(target_ids, list) and len(target_ids) > 0 else payload.get("target_uuid")

    current_player = player
    opponent = manager.p2 if current_player == manager.p1 else manager.p1
    operating_card = _operating_card(current_player, card_uuid)
    pid = current_player.name

    if action_type == ACT_PLAY:
        target_card_in_hand = next((c for c in current_player.hand if c.uuid == card_uuid), None)
        if target_card_in_hand:
            manager.action_events.append({"type": "PLAY", "player": pid, "card_name": target_card_in_hand.master.name, "message": f"「{target_card_in_hand.master.name}」を登場"})
            manager.pay_cost(current_player, target_card_in_hand.current_cost)
            manager.play_card_action(current_player, target_card_in_hand)
        else:
            raise ValueError("対象のカードが手札にありません。")
    elif action_type == ACT_TURN_END:
        manager.action_events.append({"type": "TURN_END", "player": pid, "message": f"ターン{manager.turn_count}終了"})
        manager.end_turn()
    elif action_type in (ACT_ATTACK, ACT_ATTACK_CONFIRM):
        if card_uuid == target_uuid:
            raise ValueError("自分自身を攻撃対象に選択することはできません。")
        if not operating_card:
            raise ValueError("アタックするカードが見つかりません。")
        opponent_units = [opponent.leader] + opponent.field
        if opponent.stage:
            opponent_units.append(opponent.stage)
        attack_target = next((c for c in opponent_units if c and c.uuid == target_uuid), None)
        if not attack_target:
            raise ValueError("攻撃対象が見つかりません。")
        manager.action_events.append({"type": "ATTACK", "player": pid, "card_name": operating_card.master.name, "message": f"「{operating_card.master.name}」→「{attack_target.master.name}」攻撃"})
        manager.declare_attack(operating_card, attack_target)
    elif action_type == ACT_ATTACH_DON:
        if not operating_card:
            raise ValueError("ドン!!を付与する対象のカードが見つかりません。")
        if current_player.don_active:
            don = current_player.don_active.pop(0)
            don.attached_to = operating_card.uuid
            current_player.don_attached_cards.append(don)
            operating_card.attached_don += 1
            manager.action_events.append({"type": "ATTACH_DON", "player": pid, "card_name": operating_card.master.name, "message": f"「{operating_card.master.name}」にドン!!付与"})
        else:
            raise ValueError("アクティブなドン!!が不足しています。")
    elif action_type == ACT_ACTIVATE_MAIN:
        if not operating_card:
            raise ValueError("効果を発動するカードが見つかりません。")
        manager.action_events.append({"type": "ACTIVATE_MAIN", "player": pid, "card_name": operating_card.master.name, "message": f"「{operating_card.master.name}」の効果起動"})
        for ability in operating_card.master.abilities:
            if ability.trigger == TriggerType.ACTIVATE_MAIN:
                manager.resolve_ability(current_player, ability, source_card=operating_card)
    elif action_type == ACT_RESOLVE_SELECTION:
        manager.resolve_interaction(current_player, payload)
    elif action_type == ACT_MULLIGAN:
        manager.do_mulligan(current_player)
        manager.action_events.append({"type": "MULLIGAN", "player": pid, "message": "マリガン（手札全交換）"})
    elif action_type == ACT_KEEP_HAND:
        manager.keep_hand(current_player)
        manager.action_events.append({"type": "KEEP_HAND", "player": pid, "message": "手札キープ"})
    else:
        raise ValueError(f"不明なアクションです: {action_type}")

    # 効果でライフが離れた等で積まれた誘発（ON_LIFE_DECREASE 等）をアクション境界で消化する。
    # 中断が残る場合は内部ガードで no-op（対話完了時に resolve_interaction が消化）。
    manager._advance_pending_triggers()
    # アクション境界で盤面依存の常在効果を再計算し、トラッシュ枚数等の変化を即時反映する。
    # 中断が残る場合は refresh_passive_state 内で no-op（対話完了時に反映）。
    manager.refresh_passive_state()
    return manager.action_events


def apply_battle_action(manager, player, action_type: str, card_uuid: Optional[str] = None) -> List[Dict]:
    """戦闘アクション（ブロック/カウンター/パス）を適用する。`manager.action_events` を返す。"""
    pid = player.name
    try:
        manager._validate_action(player, action_type)
    except Exception as ve:
        if action_type != BACT_PASS:
            raise ve

    if action_type == BACT_SELECT_BLOCKER:
        blocker = next((c for c in player.field if c.uuid == card_uuid), None)
        if blocker:
            manager.action_events.append({"type": "BLOCK", "player": pid, "card_name": blocker.master.name, "message": f"「{blocker.master.name}」でブロック"})
        manager.handle_block(blocker)
    elif action_type == BACT_SELECT_COUNTER:
        counter_card = next((c for c in player.hand if c.uuid == card_uuid), None)
        if counter_card:
            manager.action_events.append({"type": "COUNTER", "player": pid, "card_name": counter_card.master.name, "message": f"「{counter_card.master.name}」でカウンター(+{counter_card.master.counter or 0})"})
        manager.apply_counter(player, counter_card)
    elif action_type == BACT_PASS:
        manager.action_events.append({"type": "PASS", "player": pid, "message": "パス"})
        # パスはフェーズで意味が異なる。ブロックステップでのパスは「ブロックしない」であり、
        # その後カウンターステップへ進む必要がある。従来は常に apply_counter(None) を呼んで
        # 即 resolve_attack していたため、ブロックしないとカウンターステップが飛ばされていた。
        if manager.phase == Phase.BLOCK_STEP:
            manager.handle_block(None)        # ブロックしない → カウンターステップへ
        else:
            manager.apply_counter(player, None)  # カウンターしない → 攻撃解決
    else:
        raise ValueError(f"不明な戦闘アクションです: {action_type}")
    return manager.action_events
