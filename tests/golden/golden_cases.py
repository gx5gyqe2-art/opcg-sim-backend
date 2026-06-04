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
    # ----- 除去保護: 相手の効果で場を離れない（条件付き PASSIVE） ----------
    {
        "id": "prevent_leave_conditional",
        "text": "自分のトラッシュが7枚以上ある場合、このキャラは相手の効果で場を離れない。",
        "expect": [
            {
                "trigger": "PASSIVE",
                "condition": {"type": "TRASH_COUNT", "operator": "GE", "value": 7},
                "effect": {"kind": "action", "type": "PREVENT_LEAVE", "status": "LEAVE"},
            }
        ],
    },
    # ----- 除去保護: バトルでKOされない -----------------------------------
    {
        "id": "prevent_battle_ko",
        "text": "このキャラは、このターン中、バトルでKOされない。",
        "expect": [
            {"effect": {"kind": "action", "type": "PREVENT_LEAVE", "status": "BATTLE_KO"}}
        ],
    },
    # ----- このバトル中のパワー付与（duration=THIS_BATTLE） --------------
    {
        "id": "battle_power_buff_duration",
        "text": "自分のリーダーを、このバトル中、パワー+2000。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "value": 2000,
                    "duration": "THIS_BATTLE",
                }
            }
        ],
    },
    # ----- アタック制限（このターン中） -----------------------------------
    {
        "id": "attack_disable_this_turn",
        "text": "相手のキャラ1枚までは、このターン中、アタックできない。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "ATTACK_DISABLE",
                    "duration": "THIS_TURN",
                    "target": {"player": "OPPONENT"},
                }
            }
        ],
    },
    # ----- アタック制限（次の相手のターン終了時まで→複数ターン） -----------
    {
        "id": "attack_disable_next_turn",
        "text": "相手のキャラ1枚までは、次の相手のターン終了時まで、アタックできない。",
        "expect": [
            {"effect": {"kind": "action", "type": "ATTACK_DISABLE", "duration": "UNTIL_NEXT_TURN_END"}}
        ],
    },
    # ----- トリガー: このカードの【メイン】効果を発動する -----------------
    {
        "id": "execute_main_trigger",
        "text": "このカードの【メイン】効果を発動する。",
        "as_trigger": True,
        "expect": [
            {"trigger": "TRIGGER", "effect": {"kind": "action", "type": "EXECUTE_MAIN_EFFECT"}}
        ],
    },
    # ----- キーワード付与: このキャラは【ブロッカー】を得る ----------------
    #   構造分解で keyword タグが脱落し「このキャラはを得る」になっていた既知バグ。
    #   keyword タグを保持し GRANT_KEYWORD(status=キーワード名) を生成する。
    {
        "id": "grant_keyword_blocker_self",
        "text": "【起動メイン】このキャラは【ブロッカー】を得る。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "effect": {
                    "kind": "action",
                    "type": "GRANT_KEYWORD",
                    "status": "ブロッカー",
                    "target": {"player": "SELF"},
                },
            }
        ],
    },
    # ----- キーワード付与: このターン中【速攻】を得る（duration 付き） -------
    {
        "id": "grant_keyword_rush_this_turn",
        "text": "このキャラは、このターン中、【速攻】を得る。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "GRANT_KEYWORD",
                    "status": "速攻",
                    "duration": "THIS_TURN",
                }
            }
        ],
    },
    # ----- ライフ操作: デッキの上からライフへ（回復, deck→life） ----------
    #   従来は HEAL だが target=LIFE で対象選択待ちに陥っていた。target=None で
    #   エンジンがデッキ上から value 枚をライフに加える。
    {
        "id": "life_recover_from_deck",
        "text": "自分のデッキの上から1枚までを、ライフの上に加える。",
        "expect": [
            {"effect": {"kind": "action", "type": "HEAL", "value": 1, "target": None}}
        ],
    },
    # ----- ライフ操作: ライフ→手札（上か下から, dest=LIFE 誤りを修正） -------
    {
        "id": "life_to_hand_top_or_bottom",
        "text": "自分のライフの上か下から1枚を手札に加えることができる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "destination": "HAND",
                    "target": {"zone": "LIFE", "player": "SELF"},
                }
            }
        ],
    },
    # ----- ライフ操作: 手札→ライフ（hand→life） ----------------------------
    {
        "id": "hand_to_life_top",
        "text": "自分の手札1枚までを、ライフの上に加える。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "destination": "LIFE",
                    "target": {"zone": "HAND", "player": "SELF", "is_up_to": True},
                }
            }
        ],
    },
    # ----- ライフ操作: ライフ→トラッシュ（life→trash） --------------------
    {
        "id": "life_to_trash_opponent",
        "text": "相手のライフの上から1枚までを、トラッシュに置く。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "TRASH",
                    "target": {"zone": "LIFE", "player": "OPPONENT", "is_up_to": True},
                }
            }
        ],
    },
    # ----- ライフ操作: 表向き/裏向き（FACE_UP_LIFE, 従来 OTHER） -----------
    {
        "id": "life_face_up",
        "text": "自分のライフの上から1枚を表向きにできる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "FACE_UP_LIFE",
                    "status": "UP",
                    "target": {"zone": "LIFE", "player": "SELF"},
                }
            }
        ],
    },
    {
        "id": "life_face_down_all",
        "text": "自分のライフすべてを裏向きにする。",
        "expect": [
            {"effect": {"kind": "action", "type": "FACE_UP_LIFE", "status": "DOWN"}}
        ],
    },
    # ----- ドン操作: レストのドンをリーダー/キャラに付与（ATTACH_DON） -----
    {
        "id": "don_attach_rested",
        "text": "自分のリーダーかキャラ1枚にレストのドン‼1枚までを、付与する。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "ATTACH_DON",
                    "value": 1,
                    "status": "RESTED",
                    "target": {"zone": "FIELD", "player": "SELF",
                               "card_type": ["LEADER", "CHARACTER"]},
                }
            }
        ],
    },
    # ----- ドン操作: ドンをアクティブにする（ACTIVE_DON, 枚数ベース） -------
    {
        "id": "don_set_active",
        "text": "自分のドン‼2枚までを、アクティブにする。",
        "expect": [
            {"effect": {"kind": "action", "type": "ACTIVE_DON", "value": 2, "target": None}}
        ],
    },
    # ----- ドン操作: ドンをレストにする（REST_DON, 多くはコスト） ----------
    {
        "id": "don_set_rest",
        "text": "自分のドン‼2枚をレストにできる。",
        "expect": [
            {"effect": {"kind": "action", "type": "REST_DON", "value": 2, "target": None}}
        ],
    },
    # ----- ドン操作: 場のドンをドンデッキに戻す（RETURN_DON, 従来 OTHER） --
    {
        "id": "don_return_to_deck",
        "text": "自分の場のドン‼を1枚以上ドン‼デッキに戻すことができる。",
        "expect": [
            {"effect": {"kind": "action", "type": "RETURN_DON", "value": 1, "target": None}}
        ],
    },
    # ----- ドン操作: 相手が自身のドンをドンデッキに戻す（player=OPPONENT） --
    {
        "id": "don_return_opponent",
        "text": "相手は自身の場のドン‼1枚をドン‼デッキに戻す。",
        "expect": [
            {"effect": {"kind": "action", "type": "RETURN_DON", "value": 1, "status": "OPPONENT"}}
        ],
    },
    # ----- 条件: リーダーが『X』を含む特徴を持つ（LEADER_TRAIT, 従来 GENERIC） -
    {
        "id": "cond_leader_trait_bracket",
        "text": "自分のリーダーが『白ひげ海賊団』を含む特徴を持つ場合、カード1枚を引く。",
        "expect": [
            {
                "condition": {"type": "LEADER_TRAIT", "value": "白ひげ海賊団"},
                "effect": {"kind": "action", "type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: 盤面のキャラ枚数（FIELD_COUNT, 従来 GENERIC） --------------
    {
        "id": "cond_field_count_chars",
        "text": "自分のキャラが3枚以上いる場合、カード1枚を引く。",
        "expect": [
            {
                "condition": {"type": "FIELD_COUNT", "operator": "GE", "value": 3, "player": "SELF"},
                "effect": {"kind": "action", "type": "DRAW"},
            }
        ],
    },
    # ----- 条件: デッキ枚数（DECK_COUNT, 従来 GENERIC） --------------------
    {
        "id": "cond_deck_count",
        "text": "自分のデッキが20枚以下の場合、カード1枚を引く。",
        "expect": [
            {"condition": {"type": "DECK_COUNT", "operator": "LE", "value": 20}, "effect": {"type": "DRAW"}}
        ],
    },
    # ----- 条件: リーダーが多色（LEADER_COLOR, 従来 GENERIC） --------------
    {
        "id": "cond_leader_multicolor",
        "text": "自分のリーダーが多色の場合、カード1枚を引く。",
        "expect": [
            {"condition": {"type": "LEADER_COLOR", "value": "多色"}, "effect": {"type": "DRAW"}}
        ],
    },
    # ----- 置換効果: KOされる場合、代わりに手札を捨てる（REPLACE_EFFECT） ----
    {
        "id": "replace_on_ko_discard",
        "text": "このキャラがKOされる場合、代わりに自分の手札1枚を捨てる。",
        "expect": [
            {
                "trigger": "PASSIVE",
                "effect": {
                    "kind": "action",
                    "type": "REPLACE_EFFECT",
                    "status": "LEAVE",
                    "sub_effect": {"type": "DISCARD", "target": {"zone": "HAND"}},
                },
            }
        ],
    },
    # ----- 置換効果: バトルでKOされる場合、代わりに（status=BATTLE_KO） ------
    {
        "id": "replace_battle_ko",
        "text": "このキャラは、このターン中、バトルでKOされる場合、代わりに自分の手札1枚を捨てることができる。",
        "expect": [
            {"effect": {"kind": "action", "type": "REPLACE_EFFECT", "status": "BATTLE_KO"}}
        ],
    },
    # ----- 自己トラッシュ（このキャラをトラッシュに置く, 最頻出 OTHER 49件） ----
    #   KO ではなく単なる移動。対象は自身(SOURCE)。多くはコストで使われる。
    {
        "id": "trash_self_cost",
        "text": "このキャラをトラッシュに置くことができる。",
        "expect": [
            {"effect": {"kind": "action", "type": "TRASH", "target": {"player": "SELF"}}}
        ],
    },
    # ----- 自己アクティブ（このキャラをアクティブにする, OTHER 27件） -----------
    {
        "id": "active_self",
        "text": "このキャラをアクティブにする。",
        "expect": [
            {"effect": {"kind": "action", "type": "ACTIVE", "target": {"player": "SELF"}}}
        ],
    },
    # ----- ステージをレスト（このステージをレストにできる, 「できる」取りこぼし 20件） -
    {
        "id": "rest_stage_can",
        "text": "このステージをレストにできる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "REST",
                    "target": {"zone": "FIELD", "card_type": ["STAGE"]},
                }
            }
        ],
    },
    # ----- デッキの上からトラッシュ（mill, TRASH_FROM_DECK, OTHER 11件） --------
    {
        "id": "mill_deck_top",
        "text": "自分のデッキの上から2枚をトラッシュに置く。",
        "expect": [
            {"effect": {"kind": "action", "type": "TRASH_FROM_DECK", "value": 2, "target": None}}
        ],
    },
    # ----- 残りをトラッシュへ（remaining→trash, OTHER 18件） -------------------
    {
        "id": "remaining_trash",
        "text": "残りをトラッシュに置く。",
        "expect": [
            {"effect": {"kind": "action", "type": "TRASH", "target": {"zone": "TEMP"}}}
        ],
    },
    # ----- 手札公開: 「自分の手札からイベント2枚を公開することができる」（OTHER 解消） ------
    {
        "id": "reveal_hand_events",
        "text": "自分の手札からイベント2枚を公開することができる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "REVEAL",
                    "target": {"zone": "HAND", "player": "SELF", "card_type": ["EVENT"], "count": 2, "is_up_to": True},
                }
            }
        ],
    },
    # ----- 手札公開: 「パワー8000のキャラ1枚を公開できる」（誤 BUFF → REVEAL に修正） --------
    {
        "id": "reveal_hand_power_char",
        "text": "自分の手札からパワー8000のキャラカード1枚を公開できる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "REVEAL",
                    "target": {"zone": "HAND", "player": "SELF", "card_type": ["CHARACTER"], "power_max": 8000},
                }
            }
        ],
    },
    # ----- 非自己アクティブ: 「自分のキャラ1枚までをアクティブにする」（4件 fallback） -------
    {
        "id": "active_target_self_char",
        "text": "自分のキャラ1枚までを、アクティブにする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "ACTIVE",
                    "target": {"player": "SELF", "card_type": ["CHARACTER"], "is_up_to": True},
                }
            }
        ],
    },
    # ----- ブロッカー無効: 「相手はこのバトル中【ブロッカー】を発動できない」（4件 OTHER） ----
    {
        "id": "blocker_disable_this_battle",
        "text": "相手は、このバトル中、【ブロッカー】を発動できない。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "BLOCKER_DISABLE",
                    "duration": "THIS_BATTLE",
                    "target": {"player": "OPPONENT"},
                }
            }
        ],
    },
    # ----- 速攻（自然言語）: 「登場したターンにキャラへアタックできる」（4件 OTHER） ----------
    {
        "id": "rush_natural_keyword",
        "text": "このキャラは登場したターンにキャラへアタックできる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "GRANT_KEYWORD",
                    "status": "速攻",
                    "duration": "PERMANENT",
                }
            }
        ],
    },
    # ----- mill「置き」活用形: 「デッキの上からN枚をトラッシュに置き、シャッフルする」 -------
    {
        "id": "mill_deck_conjunctive",
        "text": "自分のデッキの上から2枚をトラッシュに置き、デッキをシャッフルする。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "TRASH_FROM_DECK", "value": 2},
                        {"type": "SHUFFLE"},
                    ],
                }
            }
        ],
    },
    # ----- バウンス: 「（コストN以下の）キャラを持ち主の手札に戻す」（OPPONENT デフォルト） ---
    {
        "id": "bounce_to_owner_opponent",
        "text": "コスト3以下のキャラ1枚までを、持ち主の手札に戻す。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BOUNCE",
                    "target": {"player": "OPPONENT", "card_type": ["CHARACTER"], "cost_max": 3, "is_up_to": True},
                }
            }
        ],
    },
    # ----- バウンス: 「自分のキャラを持ち主の手札に戻す」（SELF 明示） ----------
    {
        "id": "bounce_self_to_hand",
        "text": "自分のキャラ1枚を持ち主の手札に戻すことができる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BOUNCE",
                    "target": {"player": "SELF", "card_type": ["CHARACTER"]},
                }
            }
        ],
    },
    # ----- デッキ下送り: 「（コストN以下の）キャラを持ち主のデッキの下に置く」 ----
    {
        "id": "deck_bottom_to_owner",
        "text": "コスト2以下のキャラ1枚までを、持ち主のデッキの下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DECK_BOTTOM",
                    "target": {"player": "OPPONENT", "card_type": ["CHARACTER"], "cost_max": 2, "is_up_to": True},
                }
            }
        ],
    },
    # ----- デッキ下送り: 「自分の手札N枚をデッキの下に置く」 -------------------
    {
        "id": "hand_to_deck_bottom",
        "text": "自分の手札2枚を好きな順番でデッキの下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DECK_BOTTOM",
                    "target": {"zone": "HAND", "player": "SELF", "count": 2},
                }
            }
        ],
    },
    # ----- デッキ下送り: 「相手は自身の手札1枚をデッキの下に置く」 --------------
    {
        "id": "opp_hand_to_deck_bottom",
        "text": "相手は自身の手札1枚をデッキの下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DECK_BOTTOM",
                    "target": {"zone": "HAND", "player": "OPPONENT", "count": 1},
                }
            }
        ],
    },
    # ----- 残り→デッキ上か下（上か下選択, 保守的に DECK_BOTTOM）-----------------
    {
        "id": "remaining_deck_top_or_bottom",
        "text": "残りをデッキの上か下に置く。",
        "expect": [
            {"effect": {"kind": "action", "type": "DECK_BOTTOM", "target": {"zone": "TEMP"}}}
        ],
    },
    # ----- 手札から登場させる（PLAY_CARD from HAND） --------------------------
    {
        "id": "play_from_hand_cost_filter",
        "text": "自分の手札からコスト2以下のキャラカード1枚までを、登場させる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "PLAY_CARD",
                    "target": {"zone": "HAND", "player": "SELF", "cost_max": 2, "is_up_to": True},
                    "destination": "FIELD",
                }
            }
        ],
    },
    # ----- トラッシュからレストで登場させる（PLAY_CARD from TRASH, RESTED） ------
    {
        "id": "play_from_trash_rested",
        "text": "自分のトラッシュからコスト4以下のキャラカード1枚までを、レストで登場させる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "PLAY_CARD",
                    "target": {"zone": "TRASH", "player": "SELF", "cost_max": 4, "is_up_to": True},
                    "destination": "FIELD",
                    "status": "RESTED",
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
