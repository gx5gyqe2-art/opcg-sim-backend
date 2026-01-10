from typing import Dict, List
from ...models.effect_types import (
    Ability, Sequence, GameAction, TargetQuery, ValueSource, Branch, Choice, Condition
)
from ...models.enums import TriggerType, ActionType, Zone, ConditionType, CompareOperator, Player

def get_manual_ability(card_id: str) -> List[Ability]:
    """カードIDから手動定義された効果リストを取得する"""
    return MANUAL_EFFECTS.get(card_id, [])

# カードごとの効果定義
MANUAL_EFFECTS: Dict[str, List[Ability]] = {
    
    # ----------------------------------------------------
    # リーダー: イム (OP13-079)
    # ----------------------------------------------------
    "OP13-079": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            condition=Condition(type=ConditionType.TURN_LIMIT, value=1), # ターン1回
            cost=Choice(
                message="コストを選択してください",
                options=[
                    # 1. 自分の特徴《天竜人》を持つキャラをトラッシュに置く
                    GameAction(
                        type=ActionType.TRASH,
                        target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, traits=["天竜人"], count=1),
                        raw_text="自分の特徴《天竜人》を持つキャラをトラッシュに置く"
                    ),
                    # 2. 手札1枚をトラッシュに置く
                    GameAction(
                        type=ActionType.TRASH,
                        target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1),
                        raw_text="手札1枚をトラッシュに置く"
                    )
                ]
            ),
            effect=GameAction(
                type=ActionType.DRAW,
                value=ValueSource(base=1),
                raw_text="カード1枚を引く"
            )
        )
    ],

    # ----------------------------------------------------
    # シャルリア宮 (OP13-086)
    # ----------------------------------------------------
    "OP13-086": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=ValueSource(base=3),
                    target=TargetQuery(zone=Zone.DECK, player=Player.SELF, count=3),
                    destination=Zone.TEMP,
                    raw_text="自分のデッキの上から3枚を見る"
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, traits=["天竜人"], count=1), 
                    destination=Zone.HAND,
                    raw_text="「シャルリア宮」以外の特徴《天竜人》を持つカード1枚までを公開し、手札に加える"
                ),
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL"),
                    destination=Zone.TRASH,
                    raw_text="残りをトラッシュに置く"
                ),
                GameAction(
                    type=ActionType.DISCARD,
                    target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1),
                    destination=Zone.TRASH,
                    raw_text="自分の手札1枚を捨てる"
                )
            ])
        )
    ],

    # ----------------------------------------------------
    # チャルロス聖 (OP13-087)
    # ----------------------------------------------------
    "OP13-087": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=GameAction(
                type=ActionType.TRASH,
                target=TargetQuery(player=Player.SELF, zone=Zone.DECK, count=1),
                raw_text="自分のデッキの上から1枚をトラッシュに置く"
            )
        )
    ],

    # ----------------------------------------------------
    # ミョスガルド聖 (OP13-092)
    # ----------------------------------------------------
    "OP13-092": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            condition=Condition(type=ConditionType.LIFE_COUNT, operator=CompareOperator.LE, value=3), # LE (<=)
            effect=GameAction(
                type=ActionType.PLAY_CARD,
                target=TargetQuery(zone=Zone.TRASH, player=Player.SELF, card_type=["STAGE"], cost_max=1, traits=["聖地マリージョア"], count=1),
                destination=Zone.FIELD,
                raw_text="自分のトラッシュからコスト1の特徴《聖地マリージョア》を持つステージカード1枚までを、登場させる"
            )
        )
    ],
    
    # ----------------------------------------------------
    # トップマン・ウォーキュリー聖 (OP13-089)
    # ----------------------------------------------------
    "OP13-089": [
        Ability(
            trigger=TriggerType.ON_KO,
            effect=GameAction(
                type=ActionType.DRAW,
                value=ValueSource(base=1),
                raw_text="カード1枚を引く"
            )
        )
    ],

    # ----------------------------------------------------
    # ジェイガルシア・サターン聖 (OP13-083)
    # ----------------------------------------------------
    "OP13-083": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=ValueSource(base=5),
                    target=TargetQuery(zone=Zone.DECK, player=Player.SELF, count=5),
                    destination=Zone.TEMP
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, traits=["五老星"], count=1),
                    destination=Zone.HAND,
                    raw_text="特徴《五老星》を持つカード1枚までを公開し、手札に加える"
                ),
                GameAction(
                    type=ActionType.DECK_BOTTOM,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL"),
                    destination=Zone.DECK,
                    raw_text="残りを好きな順番でデッキの下に置く"
                )
            ])
        )
    ],

    # ----------------------------------------------------
    # イーザンバロン・V・ナス寿郎聖 (OP13-080)
    # ----------------------------------------------------
    "OP13-080": [
        Ability(
            trigger=TriggerType.ON_ATTACK,
            condition=Condition(type=ConditionType.TRASH_COUNT, operator=CompareOperator.GE, value=10), # GE (>=)
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], count=1),
                value=ValueSource(base=-2000),
                raw_text="相手のキャラ1枚までを、このターン中、パワー-2000"
            )
        )
    ],

    # ----------------------------------------------------
    # マーカス・マーズ聖 (OP13-091)
    # ----------------------------------------------------
    "OP13-091": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            cost=GameAction(
                type=ActionType.DISCARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1),
                raw_text="自分の手札1枚を捨てることができる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=5),
                raw_text="相手の元々のコスト5以下のキャラ1枚までを、KOする"
            )
        )
    ],

    # ----------------------------------------------------
    # "五老星"ここに!!! (OP13-096)
    # ----------------------------------------------------
    "OP13-096": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=ValueSource(base=3),
                    target=TargetQuery(zone=Zone.DECK, player=Player.SELF, count=3),
                    destination=Zone.TEMP
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, traits=["天竜人"], count=1),
                    destination=Zone.HAND
                ),
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL"),
                    destination=Zone.TRASH
                )
            ])
        )
    ],

    # ----------------------------------------------------
    # 世界の均衡など…永遠には保てぬのだ (OP13-097)
    # ----------------------------------------------------
    "OP13-097": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=GameAction(
                type=ActionType.REST,
                target=TargetQuery(player=Player.SELF, zone=Zone.DON_DECK, count=5), 
                raw_text="自分のドン‼5枚をレストにできる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=6),
                raw_text="相手の元々のコスト6以下のキャラ1枚までを、KOする"
            )
        )
    ],

    # ----------------------------------------------------
    # 虚の玉座 (OP13-099)
    # ----------------------------------------------------
    "OP13-099": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=Sequence(actions=[
                GameAction(type=ActionType.REST, target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, names=["虚の玉座"], count=1)),
                GameAction(type=ActionType.REST, target=TargetQuery(player=Player.SELF, zone=Zone.DON_DECK, count=3))
            ]),
            effect=GameAction(
                type=ActionType.PLAY_CARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.HAND, traits=["五老星"], colors=["Black"]),
                destination=Zone.FIELD,
                raw_text="手札から黒の特徴《五老星》を持つキャラカード1枚までを登場させる"
            )
        )
    ]
}
