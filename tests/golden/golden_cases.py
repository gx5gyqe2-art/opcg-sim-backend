"""ゴールデンコーパス（効果セマンティクスの回帰スイート）。

各ケースは:
  id      : 一意な識別子
  text    : カード効果テキスト（生）
  as_trigger: トリガー欄として解析するか（任意, 既定 False）
  expect  : Ability ごとの「期待 summary（部分仕様）」のリスト
            tests/golden/summarize.py の matches_expected で部分一致判定する

新しいルールを TDD で作るときは、ここにケースを追加して失敗させ、
atoms.py にルールを足して緑にする、というサイクルで進める。
実プレイ可能デッキ(imu/nami)のカードを優先的に網羅していく。
"""

CASES = [
    # ----- 基本ドロー ------------------------------------------------------
    {
        "id": "draw_on_play_1",
        "text": "【登場時】カード1枚を引く。",
        "expect": [
            {"trigger": "ON_PLAY", "effect": {"kind": "action", "type": "DRAW", "value": 1}}
        ],
    },
    {
        "id": "draw_on_ko_2",
        "text": "【KO時】カード2枚を引く。",
        "expect": [
            {"trigger": "ON_KO", "effect": {"kind": "action", "type": "DRAW", "value": 2}}
        ],
    },
    # ----- 複数能力（/ 区切り） -------------------------------------------
    {
        "id": "multi_ability",
        "text": "【登場時】カード1枚を引く。 / 【KO時】カード1枚を引く。",
        "expect": [
            {"trigger": "ON_PLAY", "effect": {"type": "DRAW"}},
            {"trigger": "ON_KO", "effect": {"type": "DRAW"}},
        ],
    },
    # ----- 自己レストをコストにしたドロー ---------------------------------
    {
        "id": "self_rest_cost_draw",
        "text": "【起動メイン】このキャラをレストにできる：カード1枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {"kind": "action", "type": "REST", "target": {"ref_id": "self"}},
                "effect": {"kind": "action", "type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件付きパワー増減（以下比較） ---------------------------------
    {
        "id": "cond_power_buff_le",
        "text": "【自分のターン中】自分のライフが3枚以下の場合、このリーダーのパワー+1000。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "LIFE_COUNT", "operator": "LE", "value": 3},
                "effect": {"kind": "action", "type": "BUFF", "value": 1000},
            }
        ],
    },
    # ----- 相手キャラの条件付き KO（OP13-091 マーズ聖の効果本体） ----------
    {
        "id": "ko_opponent_cost5_upto",
        "text": "相手の元々のコスト5以下のキャラ1枚までを、KOする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "KO",
                    "target": {"player": "OPPONENT", "cost_max": 5, "is_up_to": True},
                }
            }
        ],
    },
    # ----- 手札を捨てる（単体） -------------------------------------------
    {
        "id": "discard_hand_1",
        "text": "自分の手札1枚を捨てる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DISCARD",
                    "target": {"zone": "HAND", "player": "SELF", "count": 1},
                }
            }
        ],
    },
    # ----- ドロー後に手札を捨てる（逐次, P-096 少女の登場時） --------------
    {
        "id": "draw_then_discard",
        "text": "【登場時】カード1枚を引き、自分の手札1枚を捨てる。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "DRAW", "value": 1},
                        {"type": "DISCARD", "target": {"zone": "HAND"}},
                    ],
                },
            }
        ],
    },
    # ----- コスト増減（相手キャラ, 現状 OTHER → 修正対象） -----------------
    {
        "id": "cost_reduction_opponent",
        "text": "相手のキャラ1枚までを、このターン中、コスト-2。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "COST_REDUCTION",
                    "value": -2,
                    "target": {"player": "OPPONENT", "card_type": ["CHARACTER"], "is_up_to": True},
                }
            }
        ],
    },
    # ----- このカードを登場させる（トリガー自己登場, 対象=自身） -----------
    {
        "id": "play_self_trigger",
        "text": "このカードを登場させる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "PLAY_CARD",
                    "target": {"ref_id": "self"},
                }
            }
        ],
    },
    # ----- デッキシャッフル（現状 OTHER → 修正対象） ----------------------
    {
        "id": "shuffle_deck",
        "text": "デッキをシャッフルする。",
        "expect": [{"effect": {"kind": "action", "type": "SHUFFLE"}}],
    },
    # ----- 残りをデッキの下へ（分割で truncate され OTHER 化していた） -----
    {
        "id": "remaining_deck_bottom",
        "text": "残りを好きな順番でデッキの下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DECK_BOTTOM",
                    "target": {"zone": "TEMP"},
                }
            }
        ],
    },
    # ----- ドン!!返却をコストにしたドロー（ドン‼-1） ----------------------
    {
        "id": "don_return_cost",
        "text": "【起動メイン】ドン‼-1：カード1枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {"kind": "action", "type": "RETURN_DON", "value": 1},
                "effect": {"kind": "action", "type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- ドン!!をレストで追加（従来 OTHER） -----------------------------
    {
        "id": "don_add_rested",
        "text": "ドン‼デッキからドン‼1枚までを、レストで追加する。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "RAMP_DON",
                    "status": "RESTED",
                    "value": 1,
                }
            }
        ],
    },
    # ----- カウンターのパワー付与（OP13-097 世界の均衡） -------------------
    {
        "id": "counter_power_buff_3000",
        "text": "【カウンター】自分のリーダーを、このバトル中、パワー+3000。",
        "expect": [
            {
                "trigger": "COUNTER",
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "value": 3000,
                    "target": {"player": "SELF"},
                },
            }
        ],
    },
]
