from typing import Dict, List
from ...models.effect_types import (
    Ability, Sequence, GameAction, TargetQuery, ValueSource, Branch, Choice, Condition
)
from ...models.enums import TriggerType, ActionType, Zone, ConditionType, CompareOperator

def get_manual_ability(card_id: str) -> List[Ability]:
    """カードIDから手動定義された効果リストを取得する。定義がなければ空リストを返す。"""
    return MANUAL_EFFECTS.get(card_id, [])

# カードIDごとの効果定義
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
                        target=TargetQuery(target_player="SELF", zone=Zone.FIELD, traits=["天竜人"], count=1),
                        raw_text="自分の特徴《天竜人》を持つキャラをトラッシュに置く"
                    ),
                    # 2. 手札1枚をトラッシュに置く
                    GameAction(
                        type=ActionType.TRASH,
                        target=TargetQuery(target_player="SELF", zone=Zone.HAND, count=1),
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
                    value=3,
                    source_zone=Zone.DECK,
                    dest_zone=Zone.TEMP,
                    raw_text="自分のデッキの上から3枚を見る"
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, traits=["天竜人"], count=1, exclude_ids=["OP13-086"]), # 自身(シャルリア宮)以外
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.HAND,
                    raw_text="「シャルリア宮」以外の特徴《天竜人》を持つカード1枚までを公開し、手札に加える"
                ),
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(group="REMAINING_CARDS"),
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.TRASH,
                    raw_text="残りをトラッシュに置く"
                ),
                GameAction(
                    type=ActionType.DISCARD,
                    target=TargetQuery(target_player="SELF", zone=Zone.HAND, count=1),
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
                target=TargetQuery(target_player="SELF", zone=Zone.DECK, count=1),
                raw_text="自分のデッキの上から1枚をトラッシュに置く"
            )
        )
        # ブロッカーはキーワード能力として自動処理されるため記述不要（あるいはTriggerType.BLOCKERとして追加も可）
    ],

    # ----------------------------------------------------
    # ミョスガルド聖 (OP13-092)
    # ----------------------------------------------------
    "OP13-092": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            condition=Condition(type=ConditionType.LIFE_COUNT, operator=CompareOperator.LTE, value=3), # ライフ3枚以下
            effect=GameAction(
                type=ActionType.PLAY_CARD,
                target=TargetQuery(zone=Zone.TRASH, type="STAGE", cost=1, traits=["聖地マリージョア"], count=1),
                source_zone=Zone.TRASH,
                dest_zone=Zone.FIELD,
                raw_text="自分のトラッシュからコスト1の特徴《聖地マリージョア》を持つステージカード1枚までを、登場させる"
            )
        )
    ],
    
    # ----------------------------------------------------
    # トップマン・ウォーキュリー聖 (OP13-089)
    # ----------------------------------------------------
    "OP13-089": [
        # 常時効果（トラッシュ7枚以上で場を離れない）はStatusEffectとして別途実装が必要だが、
        # ここではトリガー効果のみ定義
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
                    value=5,
                    source_zone=Zone.DECK,
                    dest_zone=Zone.TEMP
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, traits=["五老星"], count=1),
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.HAND,
                    raw_text="特徴《五老星》を持つカード1枚までを公開し、手札に加える"
                ),
                GameAction(
                    type=ActionType.DECK_BOTTOM,
                    target=TargetQuery(group="REMAINING_CARDS"),
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.DECK,
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
            condition=Condition(type=ConditionType.TRASH_COUNT, operator=CompareOperator.GTE, value=10), # トラッシュ10枚以上
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(target_player="OPPONENT", zone=Zone.FIELD, type="CHARACTER", count=1),
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
                target=TargetQuery(target_player="SELF", zone=Zone.HAND, count=1),
                raw_text="自分の手札1枚を捨てることができる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(target_player="OPPONENT", zone=Zone.FIELD, type="CHARACTER", cost=5, operator=CompareOperator.LTE),
                raw_text="相手の元々のコスト5以下のキャラ1枚までを、KOする"
            )
        )
    ],

    # ----------------------------------------------------
    # "五老星"ここに!!! (OP13-096)
    # ----------------------------------------------------
    "OP13-096": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN, # イベントのメイン効果
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=3,
                    source_zone=Zone.DECK,
                    dest_zone=Zone.TEMP
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, traits=["天竜人"], count=1, exclude_ids=["OP13-096"]),
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.HAND
                ),
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(group="REMAINING_CARDS"),
                    source_zone=Zone.TEMP,
                    dest_zone=Zone.TRASH
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
                target=TargetQuery(target_player="SELF", zone=Zone.DON, count=5), # ドン!!5枚レスト
                raw_text="自分のドン‼5枚をレストにできる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(target_player="OPPONENT", zone=Zone.FIELD, type="CHARACTER", cost=6, operator=CompareOperator.LTE),
                raw_text="相手の元々のコスト6以下のキャラ1枚までを、KOする"
            )
            # ※「自分の場のキャラが、特徴《天竜人》を持つキャラのみの場合」という条件はConditionクラスで表現しきれない場合があるため、一旦コストのみ実装
        )
    ],

    # ----------------------------------------------------
    # 虚の玉座 (OP13-099)
    # ----------------------------------------------------
    "OP13-099": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=Sequence(actions=[
                GameAction(type=ActionType.REST, target=TargetQuery(target_player="SELF", zone=Zone.FIELD, card_id="OP13-099", count=1)), # 自身をレスト
                GameAction(type=ActionType.REST, target=TargetQuery(target_player="SELF", zone=Zone.DON, count=3))
            ]),
            effect=GameAction(
                type=ActionType.PLAY_CARD,
                target=TargetQuery(target_player="SELF", zone=Zone.HAND, traits=["五老星"], color="Black"),
                # ※「自分の場のドン‼の枚数以下のコストを持つ」という動的条件は現状のTargetQueryで表現が難しいが、
                # プレイヤーに委ねる形（選択肢でフィルタしない）で実装
                source_zone=Zone.HAND,
                dest_zone=Zone.FIELD,
                raw_text="手札から黒の特徴《五老星》を持つキャラカード1枚までを登場させる"
            )
        )
    ]
}
