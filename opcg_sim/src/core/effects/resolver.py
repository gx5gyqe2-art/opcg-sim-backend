from typing import Optional, List, Any, Dict
from ...models.enums import ActionType, Zone, ConditionType, CompareOperator
from ...models.effect_types import EffectAction, Condition
from ...utils.logger_config import log_event

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..gamestate import GameManager, Player, CardInstance

def check_condition(game_manager: 'GameManager', player: 'Player', condition: Optional[Condition], source_card: 'CardInstance') -> bool:
    if not condition: return True
    from .matcher import get_target_cards
    
    res = False
    if condition.target:
        matches = get_target_cards(game_manager, condition.target, source_card)
        res = len(matches) > 0
        log_event("DEBUG", "resolver.check_condition_target", f"Target condition: {len(matches)} matches", player=player.name)
    elif condition.type == ConditionType.LIFE_COUNT:
        val = len(player.life)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.HAND_COUNT:
        val = len(player.hand)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.DON_COUNT:
        val = len(player.don_active) + len(player.don_rested)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.TRASH_COUNT:
        val = len(player.trash)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.HAS_TRAIT:
        # フィールドのキャラまたは自身が特徴を持つか
        has_in_field = any(condition.value in c.master.traits for c in player.field)
        has_in_source = (source_card and condition.value in source_card.master.traits)
        res = has_in_field or has_in_source
    elif condition.type == ConditionType.LEADER_NAME:
        res = player.leader and player.leader.master.name == condition.value
    
    log_event("INFO", "resolver.condition_result", f"Condition [{condition.raw_text}]: {res}", player=player.name)
    return res

def execute_action(
    game_manager: 'GameManager', 
    player: 'Player', 
    action: EffectAction, 
    source_card: 'CardInstance', 
    effect_context: Optional[Dict[str, Any]] = None
) -> bool:
    from .matcher import get_target_cards
    if effect_context is None: effect_context = {}

    if not check_condition(game_manager, player, action.condition, source_card):
        return True

    targets = []
    # 再開時の選択済みID確認
    selected_uuids = effect_context.get("selected_uuids")

    if action.target:
        if action.target.select_mode == "REFERENCE":
            # 指示語の解決
            last_uuid = effect_context.get("last_target_uuid")
            if last_uuid:
                ref_card = game_manager._find_card_by_uuid(last_uuid)
                if ref_card: targets = [ref_card]
            log_event("DEBUG", "resolver.resolve_reference", f"Resolved reference to: {[t.name for t in targets]}", player=player.name)
        else:
            # ターゲット候補の取得
            candidates = get_target_cards(game_manager, action.target, source_card)
            
            # --- 修正ポイント: サーチ（TEMPゾーン）の場合は常にインタラクションを発生させる ---
            is_search = (action.target.zone == Zone.TEMP) or (action.source_zone == Zone.TEMP)
            
            # 選択が必要かどうかの判定
            # 1. 自動選択モードでない (ALL, SOURCE, SELF)
            # 2. 候補が存在する OR サーチである（0枚でも「なし」を確認させるため）
            should_interact = action.target.select_mode not in ["ALL", "SOURCE", "SELF"] and (len(candidates) > 0 or is_search)

            if should_interact:
                if selected_uuids is None:
                    # ユーザー選択待ちへ移行
                    log_event("INFO", "resolver.suspend", f"Selection required for {action.type}. Candidates: {len(candidates)}", player=player.name)
                    
                    # サーチの場合、候補(candidates)には「条件に合うカード」しか入っていない可能性がある
                    # ユーザーには「めくったカード全て(temp_zone)」を見せる必要がある
                    display_candidates = candidates
                    if is_search:
                        display_candidates = player.temp_zone # 全量表示
                    
                    game_manager.active_interaction = {
                        "player_id": player.name,
                        "action_type": "SEARCH_AND_SELECT",
                        "message": action.raw_text or "対象を選択してください",
                        # フロントエンドには表示用リスト(candidates)を渡す
                        "candidates": display_candidates, 
                        "selectable_uuids": [c.uuid for c in candidates], # 実際に選べるのはマッチしたカードのみ
                        "can_skip": True, # サーチは基本的に任意（見つからない場合など）
                        "continuation": {
                            "action": action,
                            "source_card_uuid": source_card.uuid,
                            "effect_context": effect_context
                        }
                    }
                    return False # 中断
                
                # 再開後: 選択されたIDからオブジェクトを復元
                # candidatesは再取得されているので、そこから選ぶ
                targets = [c for c in candidates if c.uuid in selected_uuids]
            else:
                # 選択不要（全対象など）の場合は候補をそのままターゲットにする
                targets = candidates

    # 直前のターゲットとして記憶（指示語解決用）
    if targets and action.target and action.target.tag == "last_target":
        effect_context["last_target_uuid"] = targets[0].uuid

    # アクション実行
    self_execute(game_manager, player, action, targets)

    # 連鎖アクション（then_actions）の実行
    if action.then_actions:
        for sub in action.then_actions:
            # 再帰実行。中断が発生したら False を返してスタックを巻き戻す
            if not execute_action(game_manager, player, sub, source_card, effect_context):
                return False
                
    return True

def self_execute(game_manager, player, action, targets):
    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)
    elif action.type == ActionType.LOOK:
        # デッキからTEMPへ移動
        moved_count = 0
        for _ in range(action.value):
            if player.deck:
                card = player.deck.pop(0)
                player.temp_zone.append(card)
                moved_count += 1
        log_event("INFO", "resolver.look", f"Moved {moved_count} cards to temp_zone", player=player.name)
    elif action.type == ActionType.KO:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.TRASH, owner)
    elif action.type == ActionType.MOVE_TO_HAND:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.HAND, owner)
    elif action.type == ActionType.TRASH:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.TRASH, owner)
    elif action.type == ActionType.DECK_BOTTOM:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.DECK, owner, dest_position="BOTTOM")
    elif action.type == ActionType.BUFF:
        for t in targets: t.power_buff += action.value
    elif action.type == ActionType.REST:
        for t in targets: t.is_rest = True
    elif action.type == ActionType.ACTIVE:
        for t in targets: t.is_rest = False
    elif action.type == ActionType.ATTACH_DON:
        # ドン付与ロジック
        if targets and player.don_active:
            don = player.don_active.pop(0)
            target_card = targets[0]
            don.attached_to = target_card.uuid
            player.don_attached_cards.append(don)
            target_card.attached_don += 1
