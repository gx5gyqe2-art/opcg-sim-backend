import json
import os
from typing import Dict, List
from ...models.effect_types import Ability
from ...utils.logger_config import log_event

# グローバル変数として生成データを保持
GENERATED_EFFECTS: Dict[str, List[Ability]] = {}

def load_generated_effects(json_path: str = "opcg_sim/data/generated_effects.json"):
    """
    LLM生成された効果定義JSONをロードしてメモリにキャッシュする。
    サーバー起動時（app.pyなど）に呼び出すことを想定。
    """
    global GENERATED_EFFECTS
    
    if not os.path.exists(json_path):
        log_event("WARNING", "catalog.load_skip", f"Generated effects file not found: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        count = 0
        for card_id, effects_data in data.items():
            if not isinstance(effects_data, list):
                continue
                
            abilities = []
            for effect_dict in effects_data:
                # effect_types.py で実装した from_dict を使用してオブジェクト化
                try:
                    ability = Ability.from_dict(effect_dict)
                    abilities.append(ability)
                except Exception as e:
                    log_event("ERROR", "catalog.parse_error", f"Failed to parse effect for {card_id}: {e}")
            
            if abilities:
                GENERATED_EFFECTS[card_id] = abilities
                count += 1
            
        log_event("INFO", "catalog.load_success", f"Loaded {count} generated card effects from {json_path}")
        
    except Exception as e:
        log_event("ERROR", "catalog.load_error", f"Failed to load generated effects: {str(e)}")

def get_ability(card_id: str) -> List[Ability]:
    """
    カードIDに対応する効果定義を取得する。
    
    優先順位:
    1. 手動定義 (MANUAL_EFFECTS) - 開発者が手動で修正・実装したもの（最優先）
    2. 自動生成 (GENERATED_EFFECTS) - LLMによって生成されたもの
    """
    # 1. 手動定義を確認
    if card_id in MANUAL_EFFECTS:
        return MANUAL_EFFECTS[card_id]
    
    # 2. 自動生成データを確認
    if card_id in GENERATED_EFFECTS:
        return GENERATED_EFFECTS[card_id]
        
    return []

def get_manual_ability(card_id: str) -> List[Ability]:
    """手動オーバーライド(MANUAL_EFFECTS)のみを返す。

    効果定義の主役は parser.py（日本語テキスト→Ability 変換）。
    LLM生成データ(generated_effects.json)は精度が低いため、ここでは
    手動定義のみをオーバーライドとして扱い、生成データは参照しない。
    """
    return MANUAL_EFFECTS.get(card_id, [])


# --- 以下、既存の手動定義データ (MANUAL_EFFECTS) ---
# ※ 以前のファイル内容にある MANUAL_EFFECTS の定義をそのまま維持してください。
# ここでは紙幅の都合上、省略していますが、実際のファイルには必ず含めてください。

from ...models.effect_types import (
    Sequence, GameAction, TargetQuery, ValueSource, Branch, Choice, Condition
)
from ...models.enums import TriggerType, ActionType, Zone, ConditionType, CompareOperator, Player, Color

# parser.py で正確に再現できるカードは削除済み（16エントリ削除）。
# 残っているのは parser.py では対応困難な特殊効果のみ。
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
    "OP12-051": [
        Ability(
            trigger=TriggerType.ACTIVATE_MAIN,
            cost=Sequence(actions=[
                GameAction(type=ActionType.REST, target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, is_strict_count=True, count=1, ref_id="self"), raw_text="このキャラをレストにする"),
                GameAction(type=ActionType.DISCARD, target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE", save_id="op12_051_cost"), raw_text="自分の手札1枚を捨てる")
            ]),
            effect=GameAction(
                type=ActionType.BUFF,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, cost_max=4, count=1, select_mode="CHOOSE", is_up_to=True, save_id="op12_051_debuff"),
                status="BLOCKER_DISABLE",
                duration="THIS_TURN",
                raw_text="相手のコスト4以下のキャラ1枚までを、このターン中、【ブロッカー】を発動できない"
            )
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
    "OP08-047": [
        Ability(
            trigger=TriggerType.ON_PLAY,
            cost=GameAction(
                type=ActionType.MOVE_CARD,
                target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["CHARACTER"], count=1, select_mode="CHOOSE", exclude_ids=["self"], save_id="op08_047_cost"),
                destination=Zone.HAND,
                raw_text="このキャラ以外の自分のキャラ1枚を持ち主の手札に戻す"
            ),
            effect=GameAction(
                type=ActionType.MOVE_CARD,
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=6, count=1, select_mode="CHOOSE", is_up_to=True, save_id="op08_047_effect"),
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
                if_true=GameAction(
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
                     GameAction(type=ActionType.RULE_PROCESSING, raw_text="何もしない")
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
                         GameAction(type=ActionType.DISCARD, target=TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1, select_mode="CHOOSE", save_id="op11_041_discard"), raw_text="自分の手札1枚を捨てる"),
                         GameAction(
                            type=ActionType.BUFF,
                            target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, card_type=["LEADER"], count=1, ref_id="self"),
                            value=ValueSource(base=2000),
                            duration="THIS_TURN",
                            raw_text="このリーダーは、このターン中、パワー+2000"
                         )
                     ]),
                     GameAction(type=ActionType.RULE_PROCESSING, raw_text="何もしない")
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
                target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, card_type=["CHARACTER"], cost_max=5, count=1, select_mode="CHOOSE", is_up_to=True, save_id="op03_048_effect"),
                destination=Zone.HAND,
                raw_text="自分のリーダーが「ナミ」の場合、相手のコスト5以下のキャラ1枚までを、持ち主の手札に戻す"
            )
        )
    ]
}