from typing import Dict, List
from ...models.effect_types import (
    Ability, Sequence, GameAction, TargetQuery, ValueSource, Branch, Choice, Condition, _nfc
)
from ...models.enums import TriggerType, ActionType, Zone, ConditionType, CompareOperator

def get_manual_ability(card_id: str) -> List[Ability]:
    """カードIDから手動定義された効果リストを取得する。定義がなければ空リストを返す。"""
    return MANUAL_EFFECTS.get(card_id, [])

# 【重要】ここにカードIDごとの効果定義を追加してください
# ※IDは opcg_cards.json の "number" (品番) と完全一致する必要があります
MANUAL_EFFECTS: Dict[str, List[Ability]] = {
    
    # 例: シャルリア宮 (ST13-004 と仮定)
    "ST13-004": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                # 1. 山札の上から1枚を見て、手札に加える
                # (Lookで見ずに直接移動でも良いが、ログの正確さのためにLook->Moveとする)
                GameAction(
                    type=ActionType.LOOK,
                    value=1,
                    source_zone=Zone.DECK,
                    dest_zone=Zone.TEMP,
                    raw_text="山札の上から1枚を見る"
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, select_mode="ALL", player="SELF"), 
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.HAND,
                    raw_text="手札に加える"
                ),
                # 2. 残りをトラッシュ (今回は1枚見て1枚引くので残りは無いが、記述通りにするなら)
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(group="REMAINING_CARDS"),
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.TRASH,
                    raw_text="残りをトラッシュに置く"
                ),
                # 3. 手札を1枚捨てる
                GameAction(
                    type=ActionType.DISCARD,
                    target=TargetQuery(target_player="SELF", zone=Zone.HAND, count=1),
                    source_zone=Zone.HAND,
                    dest_zone=Zone.TRASH,
                    raw_text="自分の手札1枚を捨てる"
                )
            ])
        )
    ],

    # 例: 少女 (OP01-001等のIDに合わせてください)
    # 登場時: 相手のキャラ1枚までを、このターン中、パワー-2000
    "OP01-006": [ 
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(target_player="OPPONENT", zone=Zone.FIELD, type="CHARACTER", count=1),
                value=ValueSource(base=-2000),
                raw_text="相手のキャラ1枚までを、パワー-2000"
            )
        )
    ],
    
    # 随時追加...
}
