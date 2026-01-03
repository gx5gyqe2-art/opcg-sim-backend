from typing import Optional, List, Any, Dict
from ...models.enums import ActionType, Zone, PendingMessage, CardType
from ...models.effect_types import EffectAction
from ...utils.logger_config import log_event

# 循環参照回避のため、型ヒントでのみ GameManager を参照
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..gamestate import GameManager, Player, CardInstance

def execute_action(
    game_manager: 'GameManager', 
    player: 'Player', 
    action: EffectAction, 
    source_card: 'CardInstance', 
    effect_context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    アクションを実行する。ユーザー入力が必要な場合は中断し False を返す。
    完了した場合は True を返す。
    """
    # インポート (循環参照回避のため関数内でインポート)
    from .matcher import get_target_cards
    
    # --- 1. ターゲット解決フェーズ ---
    targets = []
    
    # コンテキストから選択済みのUUIDを取得 (再開時用)
    selected_uuids = effect_context.get("selected_uuids") if effect_context else None
    
    if action.target:
        # まず候補を全て取得
        candidates = get_target_cards(game_manager, action.target, source_card)
        
        # ユーザー選択が必要か判定
        needs_selection = False
        
        # モードによる判定: "ALL"(全体), "SOURCE"(自身), "SELF"(自プレイヤー) 等は選択不要
        if action.target.select_mode in ["ALL", "SOURCE", "SELF"]:
            needs_selection = False
        else:
            # 候補が存在する場合のみ選択が必要
            if len(candidates) > 0:
                needs_selection = True
                # ※本来は「〜まで選ぶ」のような任意選択かどうかの判定も必要だが、
                # ここでは「候補があればユーザーに選ばせる」とする
        
        if needs_selection:
            if selected_uuids is None:
                # --- [中断] ユーザー入力待ち ---
                log_event("INFO", "resolver.suspend", f"Suspending for target selection: {action.type}", player=player.name)
                
                game_manager.active_interaction = {
                    "player_id": player.name,
                    "action_type": "SEARCH_AND_SELECT", # 汎用的な選択アクションとして設定
                    "message": "対象を選択してください",
                    "selectable_uuids": [c.uuid for c in candidates],
                    "can_skip": False, # TODO: 効果が任意(Optional)ならTrueにする
                    # 再開用コンテキスト
                    "continuation": {
                        "action": action,
                        "source_card_uuid": source_card.uuid,
                    }
                }
                return False
            else:
                # --- [再開] 選択結果の適用 ---
                targets = [c for c in candidates if c.uuid in selected_uuids]
                log_event("INFO", "resolver.resume", f"Resumed with targets: {[t.name for t in targets]}", player=player.name)
        else:
            # 選択不要（全対象など）
            targets = candidates

    # --- 2. アクション実行フェーズ ---
    
    # ターゲットがない場合でも実行されるアクションのためにリストを空で初期化
    if not targets and not action.target:
        targets = []

    # 各アクションタイプごとの処理
    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)

    elif action.type == ActionType.KO:
        for card in targets:
            owner, _ = game_manager._find_card_location(card)
            if owner:
                game_manager.move_card(card, Zone.TRASH, owner)
                log_event("INFO", "resolver.action", f"KO'd {card.name}", player=player.name)

    elif action.type == ActionType.TRASH:
        # 手札を捨てるコストやハンデスなど
        for card in targets:
             owner, _ = game_manager._find_card_location(card)
             if owner:
                game_manager.move_card(card, Zone.TRASH, owner)
                log_event("INFO", "resolver.action", f"Trashed {card.name}", player=player.name)

    elif action.type == ActionType.BUFF:
        # パワーバフ (ターン終了時まで)
        for card in targets:
            card.power_buff += action.value
            log_event("INFO", "resolver.buff", f"Buffed {card.name} by {action.value}", player=player.name)

    elif action.type == ActionType.REST:
        for card in targets:
            card.is_rest = True
            log_event("INFO", "resolver.action", f"Rested {card.name}", player=player.name)
            
    elif action.type == ActionType.ACTIVE:
        for card in targets:
            card.is_rest = False
            log_event("INFO", "resolver.action", f"Set {card.name} to Active", player=player.name)
            
    elif action.type == ActionType.MOVE_TO_HAND:
        for card in targets:
            owner, _ = game_manager._find_card_location(card)
            if owner:
                game_manager.move_card(card, Zone.HAND, owner)
                log_event("INFO", "resolver.action", f"Returned {card.name} to hand", player=player.name)

    # --- 3. 連鎖アクション (Then Actions) ---
    if action.then_actions:
        for sub_action in action.then_actions:
            # コンテキストは使い回さない（次のアクションはまた新規の選択になるため None を渡す）
            success = execute_action(game_manager, player, sub_action, source_card, None)
            if not success:
                # 連鎖の途中で中断が発生した場合、全体の処理もここで止める
                return False 

    return True
