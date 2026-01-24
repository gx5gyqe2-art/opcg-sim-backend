from typing import Dict, List
from ...models.effect_types import (
    Ability, Sequence, GameAction, TargetQuery, ValueSource, Branch, Choice, Condition
)
from ...models.enums import TriggerType, ActionType, Zone, ConditionType, CompareOperator, Player, Color

def get_manual_ability(card_id: str) -> List[Ability]:
    return MANUAL_EFFECTS.get(card_id, [])

MANUAL_EFFECTS: Dict[str, List[Ability]] = {
    # --- 既存のカード (イム/五老星など) ---
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
                    select_mode="ALL",
                    count=-1
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
                        target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, traits=["天竜人"], count=1, save_id="imu_cost_char"),
                        raw_text="自分の特徴《天竜人》を持つキャラをトラッシュに置く"
                    ),
                    GameAction(
                        type=ActionType.TRASH,
                        target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, save_id="imu_cost_hand"),
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
                # 盤面リセット：自動解決（count=-1, select_mode="ALL"）
                GameAction(
                    type=ActionType.TRASH,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["CHARACTER"], select_mode="ALL", count=-1),
                    raw_text="自分のキャラすべてをトラッシュに置く"
                ),
                # トラッシュ蘇生：is_unique_name=True で名称重複排除
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
                        is_unique_name=True,
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
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL", count=-1),
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
                target=TargetQuery(
                    player=Player.SELF, 
                    zone=Zone.DECK, 
                    count=1, 
                    select_mode="ALL"
                ),
                raw_text="自分のデッキの上から1枚をトラッシュに置く"
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
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL", count=-1),
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
                    select_mode="ALL",
                    count=-1
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
                    target=TargetQuery(zone=Zone.TEMP, player=Player.SELF, select_mode="ALL", count=-1),
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
                    raw_text="手札から自分の場のドン!!の枚数以下のコストを持つ黒の特徴《五老星》を持つキャラカード1枚までを、登場させる"
                )
            ])
        )
    ],
    
    # --- 新規追加: Nami.json 関連カード ---
    "OP06-106": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Choice(
                message="効果を使用しますか？",
                option_labels=["使用する", "使用しない"],
                options=[
                    Sequence(actions=[
                        GameAction(
                            type=ActionType.MOVE_CARD,
                            target=TargetQuery(player=Player.SELF, zone=Zone.LIFE, count=1, select_mode="CHOOSE"),
                            destination=Zone.HAND,
                            raw_text="自分のライフの上か下から1枚を手札に加える"
                        ),
                        GameAction(
                            type=ActionType.MOVE_CARD,
                            target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE", is_up_to=True),
                            destination=Zone.LIFE,
                            raw_text="自分の手札1枚までを、ライフの上に加える"
                        )
                    ]),
                    GameAction(type=ActionType.OTHER, raw_text="何もしない")
                ]
            )
        )
    ],
    "OP06-047": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.DECK_BOTTOM,
                    target=TargetQuery(player=Player.OPPONENT, zone=Zone.HAND, select_mode="ALL", count=-1),
                    raw_text="相手は自身の手札すべてをデッキに戻す"
                ),
                GameAction(
                    type=ActionType.SHUFFLE,
                    target=TargetQuery(player=Player.OPPONENT, zone=Zone.DECK),
                    raw_text="デッキをシャッフルする"
                ),
                GameAction(
                    type=ActionType.DRAW,
                    target=TargetQuery(player=Player.OPPONENT, zone=Zone.DECK),
                    value=ValueSource(base=5),
                    raw_text="その後、相手はカード5枚を引く"
                )
            ])
        )
    ],
    "P-096": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(type=ActionType.DRAW, value=ValueSource(base=1), raw_text="カード1枚を引く"),
                GameAction(
                    type=ActionType.DISCARD,
                    target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE"),
                    raw_text="自分の手札1枚を捨てる"
                )
            ])
        ),
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            condition=Condition(type=ConditionType.TURN_LIMIT, value=1),
            effect=GameAction(
                type=ActionType.ATTACH_DON,
                target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, names=["ナミ"], count=1, select_mode="CHOOSE"),
                value=ValueSource(base=1),
                is_rest=True,
                raw_text="自分の「ナミ」1枚にレストのドン‼1枚までを、付与する"
            )
        )
    ],
    "PRB02-016": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=Sequence(actions=[
                GameAction(
                    type=ActionType.REST,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, is_strict_count=True, count=1, ref_id="self"),
                    raw_text="このキャラをレストにする"
                ),
                Choice(
                    message="コスト（ライフを手札に加える）を支払いますか？",
                    option_labels=["支払う", "支払わない"],
                    options=[
                        GameAction(
                            type=ActionType.MOVE_CARD,
                            target=TargetQuery(player=Player.SELF, zone=Zone.LIFE, count=1, select_mode="CHOOSE"),
                            destination=Zone.HAND,
                            raw_text="自分のライフの上か下から1枚を手札に加える"
                        ),
                        GameAction(type=ActionType.OTHER, raw_text="何もしない")
                    ]
                )
            ]),
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["LEADER", "CHARACTER"], count=1, select_mode="CHOOSE", is_up_to=True),
                value=ValueSource(base=3000),
                raw_text="自分のリーダーかキャラ1枚までを、このターン中、パワー+3000"
            )
        )
    ],
    "OP04-056": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            effect=GameAction(
                type=ActionType.DECK_BOTTOM,
                target=TargetQuery(player=Player.ALL, zone=Zone.FIELD, card_type=["CHARACTER"], count=1, select_mode="CHOOSE", is_up_to=True),
                raw_text="キャラ1枚までを、持ち主のデッキの下に置く"
            )
        )
    ],
    "OP12-051": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=Sequence(actions=[
                GameAction(type=ActionType.REST, target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, is_strict_count=True, count=1, ref_id="self"), raw_text="このキャラをレストにする"),
                GameAction(type=ActionType.DISCARD, target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE"), raw_text="自分の手札1枚を捨てる")
            ]),
            effect=GameAction(
                type=ActionType.BUFF, # DEBUFF -> BUFF
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, cost_max=4, count=1, select_mode="CHOOSE", is_up_to=True), 
                status="BLOCKER_DISABLE", 
                raw_text="相手のコスト4以下のキャラ1枚までを、このターン中、【ブロッカー】を発動できない"
            )
        )
    ],
    "OP07-116": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.BUFF,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["LEADER", "CHARACTER"], count=1, select_mode="CHOOSE", is_up_to=True),
                    value=ValueSource(base=1000),
                    raw_text="自分のリーダーかキャラ1枚までを、このターン中、パワー+1000"
                ),
                Branch(
                    condition=Condition(type=ConditionType.LIFE_COUNT, player=Player.OPPONENT, value=2, operator=CompareOperator.LE),
                    true_action=GameAction(
                        type=ActionType.REST,
                        target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=4, count=1, select_mode="CHOOSE", is_up_to=True),
                        raw_text="相手のライフが2枚以下の場合、相手のコスト4以下のキャラ1枚までを、レストにする"
                    )
                )
            ])
        )
    ],
    "OP06-104": [
        Ability(
            trigger=TriggerType.ON_KO,
            condition=Condition(type=ConditionType.LIFE_COUNT, player=Player.OPPONENT, value=3, operator=CompareOperator.LE),
            effect=GameAction(
                type=ActionType.MOVE_CARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.DECK, count=1, select_mode="TOP"),
                destination=Zone.LIFE,
                raw_text="相手のライフが3枚以下の場合、自分のデッキの上から1枚までを、ライフの上に加える"
            )
        )
    ],
    "PRB02-008": [
        Ability(
            trigger=TriggerType.ON_KO,
            effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=2), raw_text="カード2枚を引く")
        )
    ],
    "EB03-053": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(
                    type=ActionType.ATTACH_DON,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["LEADER"], count=1, select_mode="CHOOSE"),
                    value=ValueSource(base=1),
                    is_rest=True,
                    raw_text="自分のリーダーにレストのドン‼1枚までを、付与する"
                ),
                Branch(
                    condition=Condition(type=ConditionType.LIFE_COUNT, player=Player.OPPONENT, value=3, operator=CompareOperator.GE),
                    true_action=GameAction(
                        type=ActionType.MOVE_CARD,
                        target=TargetQuery(player=Player.OPPONENT, zone=Zone.LIFE, count=1, select_mode="TOP"),
                        destination=Zone.HAND,
                        raw_text="相手のライフが3枚以上の場合、相手のライフの上から1枚までを、持ち主の手札に加える"
                    )
                )
            ])
        ),
        Ability(
            trigger=TriggerType.ON_KO,
            cost=GameAction(type=ActionType.OTHER, raw_text="自分のライフの上から1枚を表向きにする"),
            effect=GameAction(
                type=ActionType.PLAY_CARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.HAND, card_type=["CHARACTER"], power_max=6000, count=1, select_mode="CHOOSE", is_up_to=True),
                raw_text="自分の手札からパワー6000以下のキャラカード1枚までを、登場させる"
            )
        )
    ],
    "OP08-047": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            cost=GameAction(
                type=ActionType.MOVE_CARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["CHARACTER"], count=1, select_mode="CHOOSE", exclude_ids=["self"]),
                destination=Zone.HAND,
                raw_text="このキャラ以外の自分のキャラ1枚を持ち主の手札に戻す"
            ),
            effect=GameAction(
                type=ActionType.MOVE_CARD,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=6, count=1, select_mode="CHOOSE", is_up_to=True),
                destination=Zone.HAND,
                raw_text="相手のコスト6以下のキャラ1枚までを、持ち主の手札に戻す"
            )
        )
    ],
    "EB03-055": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            cost=GameAction(
                type=ActionType.TRASH,
                target=TargetQuery(player=Player.SELF, zone=Zone.LIFE, count=1, select_mode="TOP"),
                raw_text="自分のライフの上から1枚をトラッシュに置く"
            ),
            effect=Branch(
                condition=Condition(type=ConditionType.LEADER_TRAIT, value="麦わらの一味", player=Player.SELF),
                true_action=GameAction(
                    type=ActionType.MOVE_CARD,
                    target=TargetQuery(player=Player.SELF, zone=Zone.DECK, count=2, select_mode="TOP", is_up_to=True),
                    destination=Zone.LIFE,
                    raw_text="自分のリーダーが特徴《麦わらの一味》を持つ場合、自分のデッキの上から2枚までを、ライフの上に加える"
                )
            )
        ),
        Ability(
            trigger=TriggerType.ON_KO,
            condition=Condition(type=ConditionType.CONTEXT, value="OPPONENT_TURN"),
            effect=GameAction(
                type=ActionType.DAMAGE,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.LIFE, count=1),
                raw_text="相手に1ダメージを与えてもよい"
            )
        )
    ],
    "OP13-042": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            effect=Sequence(actions=[
                GameAction(type=ActionType.DRAW, value=ValueSource(base=2), raw_text="カード2枚を引く"),
                GameAction(type=ActionType.DISCARD, target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE"), raw_text="自分の手札1枚を捨てる"),
                GameAction(
                    type=ActionType.ATTACH_DON,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["LEADER"], count=1, select_mode="CHOOSE"),
                    value=ValueSource(base=2),
                    is_rest=True,
                    raw_text="自分のリーダーにレストのドン‼2枚までを、付与する"
                ),
                GameAction(
                    type=ActionType.ATTACH_DON,
                    target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["CHARACTER"], count=1, select_mode="CHOOSE"),
                    value=ValueSource(base=2),
                    is_rest=True,
                    raw_text="自分のキャラ1枚にレストのドン‼2枚までを、付与する"
                )
            ])
        )
    ],
    "OP11-041": [
        Ability(
            trigger=TriggerType.ON_LIFE_DECREASE,
            condition=Condition(type=ConditionType.AND, args=[
                 Condition(type=ConditionType.CONTEXT, value="SELF_TURN"),
                 Condition(type=ConditionType.HAND_COUNT, player=Player.SELF, value=7, operator=CompareOperator.LE),
                 Condition(type=ConditionType.TURN_LIMIT, value=1)
            ]),
            effect=Choice(
                 message="効果を使用しますか？（1枚引く）",
                 option_labels=["使用する", "使用しない"],
                 options=[
                     GameAction(type=ActionType.DRAW, value=ValueSource(base=1), raw_text="カード1枚を引く"),
                     GameAction(type=ActionType.OTHER, raw_text="何もしない")
                 ]
            )
        ),
        Ability(
            trigger=TriggerType.OPPONENT_ATTACK,
            condition=Condition(type=ConditionType.AND, args=[
                 Condition(type=ConditionType.HAS_DON, value=1, operator=CompareOperator.GE),
                 Condition(type=ConditionType.TURN_LIMIT, value=1)
            ]),
            effect=Choice(
                 message="効果を使用しますか？（手札1枚捨ててパワー+2000）",
                 option_labels=["使用する", "使用しない"],
                 options=[
                     Sequence(actions=[
                         GameAction(type=ActionType.DISCARD, target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE"), raw_text="自分の手札1枚を捨てる"),
                         GameAction(
                            type=ActionType.BUFF,
                            target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["LEADER"], count=1, ref_id="self"),
                            value=ValueSource(base=2000),
                            raw_text="このリーダーは、このターン中、パワー+2000"
                         )
                     ]),
                     GameAction(type=ActionType.OTHER, raw_text="何もしない")
                 ]
            )
        )
    ],
    "OP03-048": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            condition=Condition(type=ConditionType.LEADER_NAME, value="ナミ", player=Player.SELF),
            effect=GameAction(
                type=ActionType.MOVE_CARD,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=5, count=1, select_mode="CHOOSE", is_up_to=True),
                destination=Zone.HAND,
                raw_text="自分のリーダーが「ナミ」の場合、相手のコスト5以下のキャラ1枚までを、持ち主の手札に戻す"
            )
        )
    ]
}