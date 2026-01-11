from typing import Dict, List
from ...models.effect_types import (
    Ability, Sequence, GameAction, TargetQuery, ValueSource, Branch, Choice, Condition
)
from ...models.enums import TriggerType, ActionType, Zone, ConditionType, CompareOperator, Player, Color

def get_manual_ability(card_id: str) -> List[Ability]:
    return MANUAL_EFFECTS.get(card_id, [])

MANUAL_EFFECTS: Dict[str, List[Ability]] = {
    "OP05-097": [
        Ability(
            trigger=TriggerType.YOUR_TURN,
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(
                    player=Player.SELF,
                    zone=Zone.HAND,
                    traits=["天竜人"],
                    cost_min=2,
                    select_mode="ALL"
                ),
                value=ValueSource(base=-1),
                status="COST_REDUCTION",
                raw_text="自分が手札から登場させるコスト2以上の特徴《天竜人》を持つキャラカードの支払うコストは1少なくなる"
            )
        )
    ],
    "OP13-079": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            condition=Condition(type=ConditionType.TURN_LIMIT, value=1),
            cost=Choice(
                message="コストを選択してください",
                option_labels=[
                    "自分の特徴《天竜人》を持つキャラをトラッシュ",
                    "手札1枚をトラッシュ"
                ],
                options=[
                    GameAction(
                        type=ActionType.TRASH,
                        target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, traits=["天竜人"], count=1, save_id="cost_char"),
                        raw_text="自分の特徴《天竜人》を持つキャラをトラッシュに置く"
                    ),
                    GameAction(
                        type=ActionType.TRASH,
                        target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, save_id="cost_hand"),
                        raw_text="手札1枚をトラッシュに置く"
                    )
                ]
            ),
            effect=GameAction(
                type=ActionType.DRAW,
                value=ValueSource(base=1),
                raw_text="カード1枚を引く"
            )
        ),
        Ability(
            trigger=TriggerType.GAME_START,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.PLAY_CARD,
                    target=TargetQuery(
                        zone=Zone.DECK, 
                        player=Player.SELF, 
                        card_type=["STAGE"], 
                        traits=["聖地マリージョア"], 
                        count=1,
                        save_id="imu_start_play",
                        is_up_to=True
                    ),
                    destination=Zone.FIELD,
                    raw_text="ゲーム開始時、自分のデッキから特徴《聖地マリージョア》を持つステージカード1枚までを、登場させる"
                ),
                GameAction(
                    type=ActionType.SHUFFLE,
                    raw_text="デッキをシャッフルする"
                )
            ])
        )
    ],
    "OP13-082": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            condition=Condition(type=ConditionType.LEADER_NAME, value="イム"),
            cost=Sequence(actions=[
                GameAction(
                    type=ActionType.REST,
                    target=TargetQuery(player=Player.SELF, zone=Zone.COST_AREA, count=1, is_strict_count=True, is_rest=False, save_id="gorosei_cost_don"),
                    raw_text="自分のドン!!1枚をレストにする"
                ),
                GameAction(
                    type=ActionType.DISCARD,
                    target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, save_id="gorosei_cost_hand"),
                    raw_text="自分の手札1枚を捨てる"
                )
            ]),
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["CHARACTER"], select_mode="ALL"),
                    raw_text="自分のキャラすべてをトラッシュに置く"
                ),
                GameAction(
                    type=ActionType.PLAY_CARD,
                    target=TargetQuery(
                        player=Player.SELF, 
                        zone=Zone.TRASH, 
                        traits=["五老星"], 
                        power_max=5000, 
                        power_min=5000, 
                        count=5, 
                        is_up_to=True, 
                        save_id="gorosei_play"
                    ),
                    destination=Zone.FIELD,
                    raw_text="自分のトラッシュからパワー5000のカード名の異なる特徴《五老星》を持つキャラカード5枚までを、登場させる"
                )
            ])
        )
    ],
    "OP13-086": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=ValueSource(base=3),
                    destination=Zone.TEMP,
                    raw_text="自分のデッキの上から3枚を見る"
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, traits=["天竜人"], count=1, save_id="shalria_select", is_up_to=True), 
                    destination=Zone.HAND,
                    raw_text="「シャルリア宮」以外の特徴《天竜人》を持つカード1枚までを公開し、手札に加える"
                ),
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL", count=99),
                    destination=Zone.TRASH,
                    raw_text="残りをトラッシュに置く"
                ),
                GameAction(
                    type=ActionType.DISCARD,
                    target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, save_id="shalria_discard"),
                    destination=Zone.TRASH,
                    raw_text="自分の手札1枚を捨てる"
                )
            ])
        )
    ],
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
    "OP13-090": [  # チャルロス聖の追加
        Ability(
            trigger=TriggerType.ON_PLAY,
            cost=GameAction(
                type=ActionType.REST,
                target=TargetQuery(player=Player.SELF, zone=Zone.COST_AREA, count=1, is_strict_count=True, is_rest=False, save_id="charlos_cost_don"),
                raw_text="ドン!!1枚をレストにできる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(
                    player=Player.OPPONENT, 
                    zone=Zone.FIELD, 
                    card_type=["CHARACTER"], 
                    cost_max=4, 
                    count=1, 
                    is_up_to=True, 
                    save_id="charlos_ko_target"  # ログのエラー原因だったsave_idを定義
                ),
                raw_text="相手のコスト4以下のキャラ1枚までを、KOする"
            )
        )
    ],
    "OP13-092": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            condition=Condition(type=ConditionType.LIFE_COUNT, operator=CompareOperator.LE, value=3),
            effect=GameAction(
                type=ActionType.PLAY_CARD,
                target=TargetQuery(zone=Zone.TRASH, player=Player.SELF, card_type=["STAGE"], cost_max=1, traits=["聖地マリージョア"], count=1, save_id="myosgard_revive", is_up_to=True),
                destination=Zone.FIELD,
                raw_text="自分のトラッシュからコスト1の特徴《聖地マリージョア》を持つステージカード1枚までを、登場させる"
            )
        )
    ],
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
    "OP13-083": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=ValueSource(base=5),
                    destination=Zone.TEMP
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, traits=["五老星"], count=1, save_id="saturn_select", is_up_to=True),
                    destination=Zone.HAND,
                    raw_text="特徴《五老星》を持つカード1枚までを公開し、手札に加える"
                ),
                GameAction(
                    type=ActionType.DECK_BOTTOM,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL", count=99),
                    destination=Zone.DECK,
                    raw_text="残りを好きな順番でデッキの下に置く"
                )
            ])
        )
    ],
    "OP13-080": [
        Ability(
            trigger=TriggerType.ON_ATTACK,
            condition=Condition(type=ConditionType.TRASH_COUNT, operator=CompareOperator.GE, value=10),
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], count=1, save_id="nasujuro_debuff", is_up_to=True),
                value=ValueSource(base=-2000),
                raw_text="相手のキャラ1枚までを、このターン中、パワー-2000"
            )
        )
    ],
    "OP13-091": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            cost=GameAction(
                type=ActionType.DISCARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, save_id="mars_cost", is_up_to=True),
                raw_text="自分の手札1枚を捨てることができる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=5, save_id="mars_ko", is_up_to=True),
                raw_text="相手の元々のコスト5以下のキャラ1枚までを、KOする"
            )
        )
    ],
    "OP13-084": [
        Ability(
            trigger=TriggerType.YOUR_TURN,
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(
                    player=Player.SELF, 
                    zone=Zone.FIELD, 
                    traits=["五老星"], 
                    card_type=["CHARACTER"], 
                    select_mode="ALL"
                ),
                value=ValueSource(base=7000),
                status="POWER_OVERRIDE",
                raw_text="自分の特徴《五老星》を持つキャラすべてのパワーを7000にする"
            )
        )
    ],
    "OP13-096": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.LOOK,
                    value=ValueSource(base=3),
                    destination=Zone.TEMP
                ),
                GameAction(
                    type=ActionType.MOVE_TO_HAND,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, traits=["天竜人"], count=1, save_id="event_select", is_up_to=True),
                    destination=Zone.HAND
                ),
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL", count=99),
                    destination=Zone.TRASH
                )
            ])
        )
    ],
    "OP13-097": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=GameAction(
                type=ActionType.REST,
                target=TargetQuery(player=Player.SELF, zone=Zone.COST_AREA, count=5, save_id="event_cost_don", is_up_to=True), 
                raw_text="自分のドン‼5枚をレストにできる"
            ),
            effect=GameAction(
                type=ActionType.KO,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=6, save_id="event_ko", is_up_to=True),
                raw_text="相手の元々のコスト6以下のキャラ1枚までを、KOする"
            )
        )
    ],
    "OP13-099": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            condition=Condition(type=ConditionType.TURN_LIMIT, value=1),
            cost=Sequence(actions=[
                GameAction(
                    type=ActionType.REST,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, names=["虚の玉座"], count=1, is_strict_count=True, save_id="throne_rest"),
                    raw_text="このステージをレストにする"
                ),
                GameAction(
                    type=ActionType.REST,
                    target=TargetQuery(player=Player.SELF, zone=Zone.COST_AREA, count=3, is_strict_count=True, is_rest=False, save_id="throne_don_rest"),
                    raw_text="ドン!!3枚をレストにする"
                )
            ]),
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.PLAY_CARD,
                    target=TargetQuery(
                        player=Player.SELF, 
                        zone=Zone.HAND, 
                        traits=["五老星"], 
                        colors=["黒"], 
                        card_type=["CHARACTER"],
                        count=1, 
                        is_up_to=True,
                        save_id="throne_play",
                        cost_max_dynamic="DON_COUNT_FIELD"
                    ),
                    destination=Zone.FIELD,
                    raw_text="手札から自分の場のドン!!の枚数以下のコストを持つ黒の特徴《五老星》を持つキャラカード1枚までを, 登場させる"
                )
            ])
        )
    ],
}
