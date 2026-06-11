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
    # ----- レスト制限（次の相手のターン終了時まで、レストにできない） -------
    #   対象キャラはレストにできない＝アタック/ブロックができない（どちらも本体をレストにするため）。
    {
        "id": "rest_restrict_next_turn",
        "text": "相手のコスト5以下のキャラ1枚までは、次の相手のターン終了時まで、レストにできない。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "PREVENT_REST",
                    "duration": "UNTIL_NEXT_TURN_END",
                    "target": {"player": "OPPONENT", "cost_max": 5, "is_up_to": True},
                }
            }
        ],
    },
    # ----- レスト制限（次の相手のエンドフェイズ終了時まで・複数枚） ----------
    {
        "id": "rest_restrict_endphase_multi",
        "text": "相手のコスト7以下のキャラ3枚までは、次の相手のエンドフェイズ終了時まで、レストにできない。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "PREVENT_REST",
                    "duration": "UNTIL_NEXT_TURN_END",
                    "target": {"player": "OPPONENT", "cost_max": 7, "is_up_to": True},
                }
            }
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
    # ----- サーチ: デッキを見て公開し手札に加える（LOOK+grab+remaining に構造修正） --------
    #   従来は構造分解で LOOK が欠落し、対象が誤って FIELD/BOUNCE(count=4) になっていた。
    #   parser.py が「デッキの上からN枚を見て、」で分割し、count 誤取得も解消。
    {
        "id": "deck_search_to_hand",
        "text": "【登場時】自分のデッキの上から4枚を見て、コスト4以上のカード1枚までを公開し、手札に加える。残りを好きな順番でデッキの下に置く。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 4},
                        {
                            "type": "MOVE_CARD",
                            "destination": "HAND",
                            "target": {"zone": "TEMP", "player": "SELF", "cost_min": 4, "count": 1, "is_up_to": True},
                        },
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                },
            }
        ],
    },
    # ----- scry: デッキを見て並び替えデッキへ戻す（LOOK+temp_to_deck, temp リーク無し） -----
    {
        "id": "deck_scry_rearrange",
        "text": "【起動メイン】自分のデッキの上から3枚を見て、好きな順番に並び替え、デッキの上か下に置く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 3},
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                },
            }
        ],
    },
    # ----- デッキ公開→条件付き登場（LOOK+play_from_temp+remaining, temp リーク無し） -----
    #   従来は「公開し、…登場させる」が1原子句化し、レガシーが PLAY_CARD の対象を
    #   FIELD/DECK に誤推定していた。parser.py が「…を公開し、」で分割し、look_deck が
    #   LOOK→TEMP、play_from_temp が TEMP→FIELD、残りを DECK_BOTTOM(TEMP) が戻す。
    {
        "id": "deck_reveal_play_cost",
        "text": "【登場時】自分のデッキの上から1枚を公開し、コスト2のキャラカード1枚までを、登場させる。その後、残りをデッキの上か下に置く。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 1},
                        {
                            "type": "PLAY_CARD",
                            "destination": "FIELD",
                            "target": {"zone": "TEMP", "player": "SELF", "cost_max": 2, "is_up_to": True},
                        },
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                },
            }
        ],
    },
    # ----- デッキ公開→特徴フィルタで登場（『白ひげ海賊団』） -------------------------------
    {
        "id": "deck_reveal_play_trait",
        "text": "【登場時】自分のデッキの上から1枚を公開し、コスト4以下の『白ひげ海賊団』を含む特徴を持つキャラカード1枚までを、登場させる。その後、残りをデッキの上か下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 1},
                        {
                            "type": "PLAY_CARD",
                            "destination": "FIELD",
                            "target": {"zone": "TEMP", "cost_max": 4, "traits": ["白ひげ海賊団"], "is_up_to": True},
                        },
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                }
            }
        ],
    },
    # ----- デッキ公開→レストで登場（status=RESTED, 連用形「登場させ、」も同型） -----------
    {
        "id": "deck_reveal_play_rested",
        "text": "【アタック時】自分のデッキの上から1枚を公開し、コスト2のキャラカード1枚までを、レストで登場させる。その後、残りをデッキの上か下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 1},
                        {"type": "PLAY_CARD", "destination": "FIELD", "status": "RESTED", "target": {"zone": "TEMP", "cost_max": 2}},
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                }
            }
        ],
    },
    # ----- デッキから直接登場（サーチ→登場, play_from_deck, zone=DECK） --------------------
    #   従来はレガシーフォールバックで zone=FIELD/TEMP に誤ターゲット（盤面 no-op）だった。
    {
        "id": "play_from_deck_named",
        "text": "【KO時】自分のデッキから「スマイリー」1枚までを、登場させ、デッキをシャッフルする。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "PLAY_CARD", "destination": "FIELD",
                         "target": {"zone": "DECK", "player": "SELF", "names": ["スマイリー"], "is_up_to": True}},
                        {"type": "SHUFFLE"},
                    ],
                },
            }
        ],
    },
    # ----- サーチ（特徴フィルタ）: 見て特徴Xのカードを手札に加える --------------------------
    {
        "id": "deck_search_trait",
        "text": "【登場時】自分のデッキの上から5枚を見て、特徴《麦わらの一味》を持つカード1枚までを公開し、手札に加える。残りを好きな順番でデッキの下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 5},
                        {"type": "MOVE_CARD", "destination": "HAND", "target": {"zone": "TEMP", "traits": ["麦わらの一味"], "is_up_to": True}},
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                }
            }
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
    # ----- 手札捨て＋ステージレストをコストにした起動効果 ------------------
    # 「自分の手札1枚を捨て、このステージをレストにできる」が split_pattern の
    # 「捨て、」で分割され「自分の手札1枚を」が動詞なし断片化していた問題を修正。
    # (?<=捨て)、 に変更することで「捨て」を前クローズに残す。
    {
        "id": "discard_rest_stage_cost",
        "text": "【起動メイン】自分の手札1枚を捨て、このステージをレストにできる：カード1枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {
                    "kind": "seq",
                    "actions": [
                        {"kind": "action", "type": "DISCARD"},
                        {"kind": "action", "type": "REST"},
                    ],
                },
                "effect": {"kind": "action", "type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 手札捨て＋このキャラをトラッシュをコストにした起動効果 ----------
    {
        "id": "discard_trash_self_cost",
        "text": "【起動メイン】自分の手札1枚を捨て、このキャラをトラッシュに置くことができる：カード1枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {
                    "kind": "seq",
                    "actions": [
                        {"kind": "action", "type": "DISCARD"},
                        {"kind": "action", "type": "TRASH"},
                    ],
                },
                "effect": {"kind": "action", "type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 効果テキスト中の「捨ててもよい」（任意 discard） ----------------
    {
        "id": "discard_optional",
        "text": "【登場時】自分の手札1枚を捨ててもよい：カード2枚を引く。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "cost": {"kind": "action", "type": "DISCARD"},
                "effect": {"kind": "action", "type": "DRAW", "value": 2},
            }
        ],
    },
    # ----- 手札→デッキ下（hand_to_deck） -----------------------------------
    {
        "id": "hand_to_deck_1",
        "text": "【ドン‼×1】【起動メイン】【ターン1回】カード1枚を引き、自分の手札1枚をデッキの上か下に置く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"kind": "action", "type": "DRAW", "value": 1},
                        {"kind": "action", "type": "DECK_BOTTOM", "target": {"zone": "HAND"}},
                    ],
                },
            }
        ],
    },
    # ----- ライフ→手札（もよい形）（life_to_hand_optional） ----------------
    {
        "id": "life_to_hand_optional",
        "text": "【メイン】自分の手札から「エドワード・ニューゲート」1枚までを、登場させる。その後、自分のライフの上か下から1枚を手札に加えてもよい。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"kind": "action", "type": "PLAY_CARD"},
                        {"kind": "action", "type": "MOVE_CARD", "target": {"zone": "LIFE"}, "destination": "HAND"},
                    ],
                },
            }
        ],
    },
    # ----- ドン!! スペース表記（ドン !!-1）（don_return_space） ------------
    {
        "id": "don_return_space",
        "text": "【起動メイン】【ターン1回】ドン !!-1：相手のドン!!1枚までを、レストにする。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {"kind": "action", "type": "RETURN_DON", "value": 1},
            }
        ],
    },
    # ----- 公開→条件付き登場（インライン条件: LOOK→Branch(REVEALED_CARD_TRAIT)→PLAY_CARD(TEMP)）-----
    #   「デッキの一番上を公開し」を独立クローズに分割して LOOK(→TEMP) 化し、
    #   「そのカードが…の場合」を REVEALED_CARD_TRAIT のインライン Branch として保持する
    #   （アビリティ条件へ lift しない＝公開を先に実行してから条件評価できる）。
    #   従来は条件が lift され公開(LOOK)が消失し、PLAY_CARD が zone=FIELD に誤ターゲットしていた。
    {
        "id": "play_revealed_rested",
        "text": "自分のデッキの一番上を公開し、そのカードがコスト4以下の特徴《王下七武海》を持つキャラカードの場合、レストで登場させてもよい。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 1},
                        {
                            "kind": "branch",
                            "condition": {"type": "REVEALED_CARD_TRAIT"},
                            "if_true": {
                                "type": "PLAY_CARD",
                                "status": "RESTED",
                                "target": {"zone": "TEMP"},
                            },
                        },
                    ],
                },
            }
        ],
    },
    # ----- アクティブキャラへのアタック付与（PERMANENT）-------------------
    {
        "id": "attack_active_permanent",
        "text": "【ドン‼×2】このキャラは相手のアクティブのキャラにもアタックできる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "GRANT_KEYWORD",
                    "status": "ATTACK_ACTIVE",
                    "duration": "PERMANENT",
                },
            }
        ],
    },
    # ----- アクティブキャラへのアタック付与（THIS_TURN, 対象付き）----------
    {
        "id": "attack_active_this_turn",
        "text": "【登場時】自分の特徴《SWORD》を持つ、リーダーかキャラ1枚までは、このターン中、アクティブのキャラにもアタックできる。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "action",
                    "type": "GRANT_KEYWORD",
                    "status": "ATTACK_ACTIVE",
                    "duration": "THIS_TURN",
                },
            }
        ],
    },
    # ----- フリーズ: 「次の相手のリフレッシュフェイズでアクティブにならない」 -------
    {
        "id": "freeze_rested_char",
        "text": "【登場時】相手のレストのキャラ1枚までは、次の相手のリフレッシュフェイズでアクティブにならない。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "action",
                    "type": "FREEZE",
                    "target": {"player": "OPPONENT", "is_up_to": True},
                },
            }
        ],
    },
    # ----- 効果無効: 「相手のキャラ1枚までを、このターン中、効果を無効にする」 -----
    {
        "id": "negate_effect_char",
        "text": "【登場時】相手のキャラ1枚までを、このターン中、効果を無効にする。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "action",
                    "type": "NEGATE_EFFECT",
                    "target": {"player": "OPPONENT", "is_up_to": True},
                    "duration": "THIS_TURN",
                },
            }
        ],
    },
    # ----- 効果無効: 「相手のリーダーかキャラ1枚までを、このターン中、効果を無効にする」 --
    {
        "id": "negate_effect_leader_or_char",
        "text": "【起動メイン】相手のリーダーかキャラ1枚までを、このターン中、効果を無効にする。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "effect": {
                    "kind": "action",
                    "type": "NEGATE_EFFECT",
                    "target": {"player": "OPPONENT", "is_up_to": True},
                    "duration": "THIS_TURN",
                },
            }
        ],
    },
    # ----- ルール処理: 「ルール上、このカードはカード名を「X」としても扱う」 ----------
    {
        "id": "rule_card_alias",
        "text": "ルール上、このカードはカード名を「ウソップ」としても扱う。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "RULE_PROCESSING",
                },
            }
        ],
    },
    # ----- 自己制限: 「自分は、このターン中、自分の効果でライフを手札に加えられない」 --
    {
        "id": "self_cannot_life_to_hand",
        "text": "自分は、このターン中、自分の効果でライフを手札に加えられない。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "RULE_PROCESSING",
                },
            }
        ],
    },
    # ----- ライフ→手札（枚数形）: 「自分のライフ1枚を手札に加えることができる」 ------
    {
        "id": "life_to_hand_count_form",
        "text": "自分のライフ1枚を手札に加えることができる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "target": {"zone": "LIFE"},
                    "destination": "HAND",
                },
            }
        ],
    },
    # ----- 自己トラッシュ（短縮形）: 「このキャラをトラッシュに」（置く省略）---------
    {
        "id": "trash_self_short",
        "text": "【起動メイン】このキャラをトラッシュに：カード2枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {"kind": "action", "type": "TRASH"},
                "effect": {"kind": "action", "type": "DRAW", "value": 2},
            }
        ],
    },
    # ----- ライフ→トラッシュ（もよい形）: 「ライフの上から1枚をトラッシュに置いてもよい」 --
    {
        "id": "life_to_trash_optional",
        "text": "代わりに自分のライフの上から1枚をトラッシュに置いてもよい。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "TRASH",
                    "target": {"zone": "LIFE", "is_up_to": True},
                },
            }
        ],
    },
    # ===== 新条件タイプ（GENERIC 分類拡充） =====
    # ----- 条件: このキャラがレストの（SOURCE_STATE / IS_RESTED） -----------
    {
        "id": "cond_source_is_rested",
        "text": "【自分のターン中】このキャラがレストの場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "SOURCE_STATE", "value": "IS_RESTED"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: このキャラがアクティブの（SOURCE_STATE / IS_ACTIVE） --------
    {
        "id": "cond_source_is_active",
        "text": "【自分のターン中】このキャラがアクティブの場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "SOURCE_STATE", "value": "IS_ACTIVE"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: このキャラが登場したターンの（SOURCE_STATE / ENTERED_THIS_TURN） -
    {
        "id": "cond_source_entered_this_turn",
        "text": "【自分のターン中】このキャラが登場したターンの場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "SOURCE_STATE", "value": "ENTERED_THIS_TURN"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: このキャラのパワーが7000以上の（SOURCE_STATE / POWER）--------
    {
        "id": "cond_source_power_ge",
        "text": "【自分のターン中】このキャラのパワーが7000以上の場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "SOURCE_STATE", "operator": "GE"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: 場のキャラが特定の特徴のみ（FIELD_ALL_TRAIT）-----------------
    {
        "id": "cond_field_all_trait",
        "text": "【自分のターン中】自分の場のキャラが、特徴《天竜人》を持つキャラのみの場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "FIELD_ALL_TRAIT", "player": "SELF"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: 特定キャラが場にいる（HAS_CHARACTER / 存在）-----------------
    {
        "id": "cond_has_character_present",
        "text": "【自分のターン中】自分の「ルフィ」がいる場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "HAS_CHARACTER", "operator": "GE", "player": "SELF"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: 特定キャラが場にいない（HAS_CHARACTER / 不在）---------------
    {
        "id": "cond_has_character_absent",
        "text": "【自分のターン中】自分の「ルフィ」がいない場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "HAS_CHARACTER", "operator": "EQ", "player": "SELF"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: リーダーの属性（LEADER_ATTRIBUTE）--------------------------
    {
        "id": "cond_leader_attribute",
        "text": "【自分のターン中】自分のリーダーが属性(斬)を持つ場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "LEADER_ATTRIBUTE", "value": "斬", "player": "SELF"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 条件: レストのカード枚数（RESTED_COUNT）---------------------------
    {
        "id": "cond_rested_count",
        "text": "【自分のターン中】自分のレストのカードが8枚以上ある場合、カード1枚を引く。",
        "expect": [
            {
                "trigger": "YOUR_TURN",
                "condition": {"type": "RESTED_COUNT", "operator": "GE", "value": 8, "player": "SELF"},
                "effect": {"type": "DRAW", "value": 1},
            }
        ],
    },
    # ===== A1: self_to_hand — このカードを手札に加える =====
    # 「このカード/このキャラカードを手札に加える（ことができる）」は MOVE_CARD(SOURCE→HAND)。
    # 従来は search_to_hand が zone=TEMP に誤設定していた（D detector 対象8枚）。
    {
        "id": "self_to_hand_trigger",
        "text": "【トリガー】このカードを手札に加える。",
        "expect": [
            {
                "trigger": "TRIGGER",
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "destination": "HAND",
                    # zone が TEMP ではないこと（SOURCE モード: zone は FIELD デフォルト）。
                    # search_to_hand が誤って zone=TEMP に設定していた bug を検出する。
                    "target": {"player": "SELF", "zone": "FIELD"},
                },
            }
        ],
    },
    {
        "id": "self_to_hand_ko",
        "text": "【KO時】このキャラカードを手札に加えることができる。",
        "expect": [
            {
                "trigger": "ON_KO",
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "destination": "HAND",
                    "target": {"player": "SELF", "zone": "FIELD"},
                },
            }
        ],
    },
    # ===== A2: trash_to_deck_ordered — トラッシュから好きな順番でデッキ下へ =====
    # 従来は temp_to_deck が zone=TEMP に誤設定していた（D detector 対象~18枚）。
    {
        "id": "trash_to_deck_ordered",
        "text": "自分のトラッシュのカード3枚を好きな順番でデッキの下に置くことができる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DECK_BOTTOM",
                    "target": {"zone": "TRASH", "player": "SELF", "count": 3},
                }
            }
        ],
    },
    # ===== A2b: field_char_to_deck_ordered — コスト付きキャラを持ち主デッキ下へ =====
    # 従来は temp_to_deck が zone=TEMP に誤設定していた（OP06-058 等）。
    {
        "id": "field_char_to_deck_ordered",
        "text": "コスト6以下のキャラ2枚までを、好きな順番で持ち主のデッキの下に置く。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "DECK_BOTTOM",
                    "target": {
                        "player": "OPPONENT",
                        "card_type": ["CHARACTER"],
                        "cost_max": 6,
                        "is_up_to": True,
                    },
                }
            }
        ],
    },
    # ===== C11: hand_to_life — dest_position フィールド =====
    # GameAction に dest_position が追加されたことで hand_to_life がライフ上/下を正確に区別できる。
    {
        "id": "hand_to_life_top",
        "text": "自分の手札1枚をライフの上に加える。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "target": {"zone": "HAND", "player": "SELF"},
                    "destination": "LIFE",
                    "dest_position": "TOP",
                }
            }
        ],
    },
    {
        "id": "hand_to_life_bottom",
        "text": "自分の手札1枚をライフの下に加える。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "target": {"zone": "HAND", "player": "SELF"},
                    "destination": "LIFE",
                    "dest_position": "BOTTOM",
                }
            }
        ],
    },
    # ===== D12d: rest_self — このキャラをレストにする（効果文脈） =====
    {
        "id": "rest_self_effect",
        "text": "【登場時】このキャラをレストにする。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "action",
                    "type": "REST",
                    "target": {"player": "SELF"},
                },
            }
        ],
    },
    # ===== D12b: trash_to_hand — トラッシュからカードを手札に =====
    {
        "id": "trash_event_to_hand",
        "text": "【登場時】自分のトラッシュのイベント1枚までを、手札に加える。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "action",
                    "type": "MOVE_CARD",
                    "target": {"zone": "TRASH", "player": "SELF", "is_up_to": True},
                    "destination": "HAND",
                },
            }
        ],
    },
    # ===== ko: KOできる（任意KO, OP05-060系, OTHER 2件） =====
    {
        "id": "ko_optional_trait",
        "text": "自分の特徴《王下七武海》を持つキャラ1枚をKOできる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "KO",
                    "target": {"player": "SELF", "traits": ["王下七武海"]},
                }
            }
        ],
    },
    # ===== B5: scry-1 デッキの上か下に置く =====
    # 「公開し、デッキの上か下に置く」→ LOOK + DECK_BOTTOM(TEMP)。
    # scry_place ルールで OTHER だった 2 件を修正。
    {
        "id": "scry_one_deck_top_or_bottom",
        "text": "【登場時】自分のデッキの上から1枚を公開し、デッキの上か下に置く。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 1},
                        {"type": "DECK_BOTTOM", "target": {"zone": "TEMP"}},
                    ],
                },
            }
        ],
    },
    # ===== B6: 登場させた場合 → PREV_ACTION condition =====
    # 「登場させた場合、【速攻】を得る」→ Branch(PREV_ACTION=PLAYED_CARD, GRANT_KEYWORD)。
    # legacy parser の _parse_condition_obj が「場合」を strip 済みのテキストに対して
    # 「場合」の有無を再チェックしていたバグを修正。
    {
        "id": "prev_action_played_card_rush",
        "text": "【メイン】自分のデッキの上から1枚を公開し、コスト5以下のキャラカード1枚までを、登場させてもよい。登場させた場合、そのキャラは、このターン中、【速攻】を得る。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "LOOK", "value": 1},
                        {"type": "PLAY_CARD"},
                        {
                            "kind": "branch",
                            "condition": {"type": "PREV_ACTION"},
                            "if_true": {"type": "GRANT_KEYWORD", "status": "速攻"},
                        },
                    ],
                }
            }
        ],
    },
    # ===== 裾野OTHER: 選択型トラッシュ（trash_target） =====
    # 「自分のキャラ1枚をトラッシュに置く」— このキャラ以外の選択型。trash_self(SOURCE)とは別。
    {
        "id": "trash_target_own_char",
        "text": "【起動メイン】自分のキャラ1枚をトラッシュに置くことができる：カード2枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {
                    "kind": "action",
                    "type": "TRASH",
                    "target": {"player": "SELF", "zone": "FIELD"},
                },
                "effect": {"kind": "action", "type": "DRAW", "value": 2},
            }
        ],
    },
    # 特徴フィルタ『』付きの選択型トラッシュ。
    {
        "id": "trash_target_trait",
        "text": "【KO時】自分の『白ひげ海賊団』を含む特徴を持つキャラ1枚をトラッシュに置くことができる。",
        "expect": [
            {
                "trigger": "ON_KO",
                "effect": {
                    "kind": "action",
                    "type": "TRASH",
                    "target": {"player": "SELF", "traits": ["白ひげ海賊団"]},
                },
            }
        ],
    },
    # ===== 裾野OTHER: パワー設定/上書き（set_power → BUFF+POWER_OVERRIDE） =====
    # 「パワー0にする」— 静的なパワー上書き。power_buff(±N)は「にする」を除外しているため別ルール。
    {
        "id": "set_power_zero",
        "text": "相手のキャラ1枚までを、このターン中、パワー0にする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "POWER_OVERRIDE",
                    "value": 0,
                    "duration": "THIS_TURN",
                    "target": {"player": "OPPONENT", "is_up_to": True},
                },
            }
        ],
    },
    # ===== 構造的難所: select断片（「…を選ぶ。選んだキャラは…」） =====
    # 「（対象）を選ぶ」→ SELECT(save_id) ／ 後続「選んだキャラ」→ ref_id="selected_card"。
    {
        "id": "select_then_attack_disable",
        "text": "【登場時】相手のコスト6以下のキャラ1枚までを選ぶ。選んだキャラは、このターン中、アタックできない。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "SELECT", "target": {"player": "OPPONENT", "cost_max": 6, "is_up_to": True}},
                        {"type": "ATTACK_DISABLE", "target": {"ref_id": "selected_card"}},
                    ],
                },
            }
        ],
    },
    # ===== 構造的難所: trigger断片（「〈timing〉時、発動できる」埋め込みトリガー） =====
    # 「相手がアタックした時、発動できる」→ ディスパッチ対象 ON_OPP_ATTACK へ。OTHER は消える。
    {
        "id": "text_trigger_opp_attack",
        "text": "【ターン1回】相手がアタックした時、発動できる。相手のリーダーかキャラ1枚までを、このターン中、パワー-1000。",
        "expect": [
            {
                "trigger": "ON_OPP_ATTACK",
                "condition": {"type": "TURN_LIMIT"},
                "effect": {"kind": "action", "type": "BUFF", "value": -1000},
            }
        ],
    },
    # 非ディスパッチ timing × PASSIVE → ACTIVATE_MAIN（常時誤発動を避け手動発動可能に）。
    {
        "id": "text_trigger_passive_to_main",
        "text": "このキャラが相手の効果でレストになった時、発動できる。このキャラをトラッシュに置き、カード2枚を引くことができる。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "TRASH"},
                        {"type": "DRAW", "value": 2},
                    ],
                },
            }
        ],
    },
    # ===== 構造的難所 C7: ライフ scry（対話選択 Choice ツリー） =====
    # 「自分か相手のライフの上から1枚までを見て、ライフの上か下に置く」→
    #   Choice[自分/相手/見ない] → 各 Seq[LOOK_LIFE → Choice[上/下に置く]]。
    {
        "id": "life_scry_top_or_bottom",
        "text": "【登場時】自分か相手のライフの上から1枚までを見て、ライフの上か下に置く。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "choice",
                    "options": [
                        {"kind": "seq", "actions": [
                            {"type": "LOOK_LIFE"},
                            {"kind": "choice", "options": [
                                {"type": "MOVE_CARD", "destination": "LIFE", "dest_position": "TOP"},
                                {"type": "MOVE_CARD", "destination": "LIFE", "dest_position": "BOTTOM"},
                            ]},
                        ]},
                        {"kind": "seq", "actions": [
                            {"type": "LOOK_LIFE"},
                            {"kind": "choice", "options": [
                                {"type": "MOVE_CARD", "destination": "LIFE", "dest_position": "TOP"},
                                {"type": "MOVE_CARD", "destination": "LIFE", "dest_position": "BOTTOM"},
                            ]},
                        ]},
                        {"kind": "seq", "actions": []},
                    ],
                },
            }
        ],
    },
    # ===== 構造的難所: モーダル選択「以下から1つを選ぶ」（ ・項目を Choice の options へ） =====
    #   従来は ` / ` で別 Ability に分割→破棄され options が空だった。
    {
        "id": "modal_choice_two_options",
        "text": "【登場時】以下から1つを選ぶ。 / ・相手のコスト4以下のキャラ1枚までを、KOする。 / ・カード2枚を引く。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "choice",
                    "options": [
                        {"kind": "action", "type": "KO",
                         "target": {"player": "OPPONENT", "cost_max": 4, "is_up_to": True}},
                        {"kind": "action", "type": "DRAW", "value": 2},
                    ],
                },
            }
        ],
    },
    # 選択肢の前段に条件ゲート（「…の場合、以下から1つを選ぶ」）→ ability.condition へ lift。
    {
        "id": "modal_choice_condition_gate",
        "text": "【メイン】自分のリーダーが多色の場合、以下から1つを選ぶ。 / ・相手のコスト4以下のキャラ1枚までを、持ち主の手札に戻す。 / ・カード2枚を引く。",
        "expect": [
            {
                "condition": {"type": "LEADER_COLOR"},
                "effect": {
                    "kind": "choice",
                    "options": [
                        {"kind": "action", "type": "BOUNCE"},
                        {"kind": "action", "type": "DRAW", "value": 2},
                    ],
                },
            }
        ],
    },
    # 選択肢の片方が自前の条件 Branch を持つ（OP14-069 同型: 条件付き KO ／ レスト制限）。
    {
        "id": "modal_choice_option_with_branch",
        "text": "【登場時】以下から1つを選ぶ。 / ・自分のリーダーが特徴《ドンキホーテ海賊団》を持つ場合、相手のコスト8以下のキャラ1枚までを、KOする。 / ・相手のコスト7以下のキャラ3枚までは、次の相手のエンドフェイズ終了時まで、レストにできない。",
        "expect": [
            {
                "trigger": "ON_PLAY",
                "effect": {
                    "kind": "choice",
                    "options": [
                        {"kind": "branch",
                         "condition": {"type": "LEADER_TRAIT"},
                         "if_true": {"kind": "action", "type": "KO"}},
                        {"kind": "action", "type": "PREVENT_REST", "duration": "UNTIL_NEXT_TURN_END"},
                    ],
                },
            }
        ],
    },
    # ----- 相手への除去＋自己バウンス（TRIGGER, TARGET_SIDE 監査フラグ対応） ----
    # 「相手の…をKOし、このカードを手札に加える」は Sequence に分割され、
    # 前段の KO 対象は OPPONENT、後段の自己バウンスは SOURCE になる。
    # 分割しないと self_to_hand が丸呑みし相手キャラの KO が消失していた。
    {
        "id": "trigger_ko_opp_then_self_bounce",
        "text": "相手のコスト1以下のキャラ1枚までを、KOし、このカードを手札に加える。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "KO", "target": {"player": "OPPONENT", "cost_max": 1}},
                        {"type": "MOVE_CARD", "target": {"select_mode": "SOURCE"}},
                    ],
                }
            }
        ],
    },
    # 「相手の…をレストにし、このカードを手札に加える」も同型（レスト＋自己バウンス）。
    {
        "id": "trigger_rest_opp_then_self_bounce",
        "text": "相手のコスト2以下のキャラ1枚までを、レストにし、このカードを手札に加える。",
        "expect": [
            {
                "effect": {
                    "kind": "seq",
                    "actions": [
                        {"type": "REST", "target": {"player": "OPPONENT", "cost_max": 2}},
                        {"type": "MOVE_CARD", "target": {"select_mode": "SOURCE"}},
                    ],
                }
            }
        ],
    },
    # ----- 動的コスト上限: ライフ枚数依存（COST_LIMIT 監査フラグ対応） --------
    # 「相手のライフの枚数以下のコストを持つ相手のキャラ」→ cost_max_dynamic=LIFE_COUNT_OPPONENT
    {
        "id": "cost_limit_opp_life_ko",
        "text": "相手のライフの枚数以下のコストを持つ相手のキャラ1枚までを、KOする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "KO",
                    "target": {
                        "player": "OPPONENT",
                        "cost_max_dynamic": "LIFE_COUNT_OPPONENT",
                        "is_up_to": True,
                    },
                }
            }
        ],
    },
    # 「自分のライフの枚数以下のコストを持つ相手のキャラ」→ LIFE_COUNT_SELF
    {
        "id": "cost_limit_self_life_ko",
        "text": "自分のライフの枚数以下のコストを持つ相手のキャラ1枚までを、KOする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "KO",
                    "target": {
                        "player": "OPPONENT",
                        "cost_max_dynamic": "LIFE_COUNT_SELF",
                        "is_up_to": True,
                    },
                }
            }
        ],
    },
    # 「お互いのライフの合計枚数以下のコストを持つ相手のキャラ」→ LIFE_COUNT_BOTH
    {
        "id": "cost_limit_both_life_ko",
        "text": "お互いのライフの合計枚数以下のコストを持つ相手のキャラ1枚までを、KOする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "KO",
                    "target": {
                        "player": "OPPONENT",
                        "cost_max_dynamic": "LIFE_COUNT_BOTH",
                        "is_up_to": True,
                    },
                }
            }
        ],
    },
    # ----- C9 同値パワー（発動時スナップショット, DURATION/OTHER 監査対応） -------
    # 「このキャラの元々のパワーは、このターン中、相手のリーダーと同じパワーになる」
    {
        "id": "power_equalize_opp_leader",
        "text": "このキャラの元々のパワーは、このターン中、相手のリーダーと同じパワーになる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "POWER_OVERRIDE",
                    "duration": "THIS_TURN",
                    "target": {"select_mode": "SOURCE"},
                },
            }
        ],
    },
    # 「選んだキャラと同じパワーになる」→ ref=selected。
    {
        "id": "power_equalize_selected",
        "text": "このキャラの元々のパワーは、このターン中、選んだキャラと同じパワーになる。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "POWER_OVERRIDE",
                    "duration": "THIS_TURN",
                    "target": {"select_mode": "SOURCE"},
                },
            }
        ],
    },
    # 「このターン中、コスト0にする」— COST_OVERRIDE で base_cost_override をセット。
    {
        "id": "set_cost_zero_this_turn",
        "text": "相手の元々の効果のないキャラ1枚までを、このターン中、コスト0にする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "COST_OVERRIDE",
                    "value": 0,
                    "duration": "THIS_TURN",
                    "target": {"player": "OPPONENT", "is_up_to": True},
                },
            }
        ],
    },
    # 「元々のパワー7000にする」— base_power_override に静的値をセット。
    {
        "id": "set_power_base_value",
        "text": "自分のリーダーかキャラ1枚までを、このターン中、元々のパワー7000にする。",
        "expect": [
            {
                "effect": {
                    "kind": "action",
                    "type": "BUFF",
                    "status": "POWER_OVERRIDE",
                    "value": 7000,
                    "duration": "THIS_TURN",
                    "target": {"player": "SELF", "is_up_to": True},
                },
            }
        ],
    },
    # ----- 効果ダメージ（DEAL_DAMAGE） -----------------------------------
    # 「相手に N ダメージを与える」: 相手リーダーへ N ダメージ（旧 OTHER）。
    {
        "id": "deal_damage_opponent",
        "text": "【メイン】相手に1ダメージを与える。",
        "expect": [
            {"effect": {"kind": "action", "type": "DEAL_DAMAGE", "value": 1,
                        "target": {"player": "OPPONENT"}}}
        ],
    },
    # 「自分は N ダメージを受ける」: 自分リーダーへ N ダメージ。
    {
        "id": "deal_damage_self",
        "text": "【登場時】自分は1ダメージを受ける。",
        "expect": [
            {"effect": {"kind": "action", "type": "DEAL_DAMAGE", "value": 1,
                        "target": {"player": "SELF"}}}
        ],
    },
    # ----- 相手デッキを覗く（LOOK + OPPONENT, 後続なしの純粋な公開） --------
    {
        "id": "look_opponent_deck_top",
        "text": "【アタック時】相手のデッキの上から1枚を見る。",
        "expect": [
            {"effect": {"kind": "action", "type": "LOOK", "value": 1, "status": "OPPONENT"}}
        ],
    },
    # ----- 複合除去保護: 「相手の効果で、KOされずレストにされない」 ---------
    {
        "id": "prevent_ko_and_rest",
        "text": "このキャラは相手の効果で、KOされずレストにされない。",
        "expect": [
            {"effect": {"kind": "seq", "actions": [
                {"kind": "action", "type": "PREVENT_LEAVE"},
                {"kind": "action", "type": "PREVENT_REST"},
            ]}}
        ],
    },
    # 「このキャラは相手の効果でレストにされない」: 単独のレスト不可保護。
    {
        "id": "prevent_rest_self",
        "text": "このキャラは相手の効果でレストにされない。",
        "expect": [
            {"effect": {"kind": "action", "type": "PREVENT_REST",
                        "target": {"select_mode": "SOURCE"}}}
        ],
    },
    # ----- 除外フィルタ: 「「◯◯」以外のキャラ1枚」 ------------------------
    {
        "id": "exclude_named_character",
        "text": "【登場時】相手の「モンキー・D・ルフィ」以外のキャラ1枚までを、持ち主のデッキの下に置く。",
        "expect": [
            {"effect": {"kind": "action",
                        "target": {"player": "OPPONENT", "exclude_names": ["モンキー・D・ルフィ"]}}}
        ],
    },
    # ----- ライフ並び替え（ORDER_LIFE） ----------------------------------
    # 「（自分の）ライフすべてを見て、好きな順番で置く」: ライフ内を任意順に並べ替え。
    {
        "id": "order_life_self",
        "text": "【登場時】自分のライフすべてを見て、好きな順番で置く。",
        "expect": [
            {"effect": {"kind": "action", "type": "ORDER_LIFE", "status": None}}
        ],
    },
    # 「相手のライフすべてを見て、好きな順番で置く」: 相手のライフを並べ替え（妨害）。
    {
        "id": "order_life_opponent",
        "text": "相手のライフすべてを見て、好きな順番で置く。",
        "expect": [
            {"effect": {"kind": "action", "type": "ORDER_LIFE", "status": "OPPONENT"}}
        ],
    },
    # ----- イベント発動（EXECUTE_EVENT） ---------------------------------
    # 「自分の手札から（条件）イベント1枚までを、発動する」: 手札のイベントを発動。
    {
        "id": "execute_event_from_hand",
        "text": "【登場時】自分の手札から特徴《ドレスローザ》を持つイベント1枚までを、発動する。",
        "expect": [
            {"effect": {"kind": "action", "type": "EXECUTE_EVENT",
                        "target": {"zone": "HAND", "player": "SELF", "traits": ["ドレスローザ"]}}}
        ],
    },
    # ----- 二択「〜するか、〜する」（Choice AST） -------------------------
    {
        "id": "choice_suruka_two_option",
        "text": "【メイン】カード1枚を引くか、相手のキャラ1枚をレストにする。",
        "expect": [
            {"effect": {"kind": "choice", "options": [
                {"kind": "action", "type": "DRAW", "value": 1},
                {"kind": "action", "type": "REST"},
            ]}}
        ],
    },
    # ----- ドン!!複合コスト「自分のドン‼N枚と…をレストにできる」 ----------
    # コスト前半の断片「自分のドン‼N枚と」を REST_DON 化（従来 OTHER）。
    {
        "id": "don_and_rest_self_cost",
        "text": "【起動メイン】自分のドン‼1枚とこのキャラをレストにできる：カード1枚を引く。",
        "expect": [
            {
                "trigger": "ACTIVATE_MAIN",
                "cost": {"kind": "seq", "actions": [
                    {"kind": "action", "type": "REST_DON", "value": 1},
                    {"kind": "action", "type": "REST", "target": {"ref_id": "self"}},
                ]},
                "effect": {"kind": "action", "type": "DRAW", "value": 1},
            }
        ],
    },
    # ----- 勝利宣言「自分はゲームに勝利する」→ VICTORY（即時勝利） --------
    {
        "id": "declare_victory",
        "text": "【メイン】自分はゲームに勝利する。",
        "expect": [
            {"effect": {"kind": "action", "type": "VICTORY", "status": None}}
        ],
    },
    # ----- 手札全戻し「自分の手札すべてをデッキに戻し、シャッフル」 --------
    {
        "id": "hand_all_to_deck",
        "text": "【登場時】自分の手札すべてをデッキに戻し、デッキをシャッフルする。",
        "expect": [
            {"effect": {"kind": "seq", "actions": [
                {"kind": "action", "type": "DECK_BOTTOM", "target": {"zone": "HAND", "player": "SELF"}},
                {"kind": "action", "type": "SHUFFLE"},
            ]}}
        ],
    },
    # ----- 活用形/「てもよい」の取りこぼし補完 ----------------------------
    # bounce: 「…手札に戻してもよい」（任意形）
    {
        "id": "bounce_optional_form",
        "text": "【登場時】相手のコスト5以下のキャラ1枚までを、持ち主の手札に戻してもよい。",
        "expect": [
            {"effect": {"kind": "action", "type": "BOUNCE", "target": {"player": "OPPONENT", "is_up_to": True}}}
        ],
    },
    # hand→デッキの下: 「…デッキの下に置いてもよい」（任意・並び替え）
    {
        "id": "hand_to_deck_bottom_optional",
        "text": "【登場時】自分の手札すべてを好きな順番でデッキの下に置いてもよい。",
        "expect": [
            {"effect": {"kind": "action", "type": "DECK_BOTTOM", "target": {"zone": "HAND", "player": "SELF"}}}
        ],
    },
    # ドン!!デッキ返却: 「相手は…ドン‼1枚をドン‼デッキに戻してもよい」
    {
        "id": "don_return_deck_optional",
        "text": "相手は自身のアクティブのドン‼1枚をドン‼デッキに戻してもよい。",
        "expect": [
            {"effect": {"kind": "action", "type": "RETURN_DON", "value": 1, "status": "OPPONENT"}}
        ],
    },
    # ----- 「任意の枚数」可変選択（is_up_to + 大きめ count で 0..N 選択） ----
    {
        "id": "ko_any_number_optional",
        "text": "【メイン】自分のコスト2以下の特徴《スリラーバーク海賊団》を持つキャラを任意の枚数KOしてもよい。",
        "expect": [
            {"effect": {"kind": "action", "type": "KO",
                        "target": {"player": "SELF", "is_up_to": True}}}
        ],
    },
    {
        "id": "bounce_any_number_optional",
        "text": "【カウンター】自分の場のキャラを任意の枚数手札に戻してもよい。",
        "expect": [
            {"effect": {"kind": "action", "type": "BOUNCE",
                        "target": {"player": "SELF", "is_up_to": True}}}
        ],
    },
    # ----- 自分デッキトップの公開（条件評価用, LOOK で TEMP へ→reclaim） --------
    {
        "id": "reveal_self_deck_top",
        "text": "【登場時】自分のデッキの上から1枚を公開する。",
        "expect": [
            {"effect": {"kind": "action", "type": "LOOK", "value": 1, "status": None}}
        ],
    },
    # ----- コスト節先頭の条件を ability.condition へ引き上げ ----------------
    {
        "id": "cost_prefix_condition_lifted",
        "text": "【起動メイン】自分のリーダーが「しらほし」の場合、このキャラをレストにできる：相手のコスト3以下のキャラ1枚までを、KOする。",
        "expect": [
            {"trigger": "ACTIVATE_MAIN",
             "condition": {"type": "LEADER_NAME"},
             "cost": {"kind": "action", "type": "REST", "target": {"ref_id": "self"}},
             "effect": {"kind": "action", "type": "KO"}}
        ],
    },
    # ----- 丸数字コスト（①＝ドン1枚レスト） --------------------------------
    {
        "id": "don_cost_circled_one",
        "text": "【起動メイン】①：このキャラをアクティブにする。",
        "expect": [
            {"trigger": "ACTIVATE_MAIN",
             "cost": {"kind": "action", "type": "REST_DON", "value": 1}}
        ],
    },
    # ----- ライフを見て1枚をデッキ上へ（LIFE→DECK top, ST13-004 前段） --------
    {
        "id": "life_view_to_deck_top",
        "text": "自分のライフすべてを見て、1枚を自分のデッキの上に置く。",
        "expect": [
            {"effect": {"kind": "action", "type": "MOVE_CARD", "destination": "DECK",
                        "dest_position": "TOP", "target": {"zone": "LIFE", "player": "SELF", "count": 1}}}
        ],
    },
    # ----- サーチ結果をライフへ（TEMP→LIFE, look_deck の後段） --------------
    {
        "id": "search_temp_to_life",
        "text": "カード1枚までを、ライフの上に加える。",
        "expect": [
            {"effect": {"kind": "action", "type": "MOVE_CARD", "destination": "LIFE",
                        "dest_position": "TOP", "target": {"zone": "TEMP", "is_up_to": True}}}
        ],
    },
    # ----- 登場制限 PASSIVE（手札のこのカードは効果で登場できない） ----------
    {
        "id": "no_effect_play_passive",
        "text": "手札のこのカードは、効果で登場できない。",
        "expect": [
            {"effect": {"kind": "action", "type": "RESTRICTION", "status": "NO_EFFECT_PLAY"}}
        ],
    },
    # ----- レスト登場 PASSIVE（自分のキャラはレストで登場する） -------------
    {
        "id": "rested_play_passive",
        "text": "自分のキャラカードはレストで登場する。",
        "expect": [
            {"effect": {"kind": "action", "type": "RESTRICTION", "status": "RESTED_PLAY"}}
        ],
    },
    # ----- 付与ドンをコストエリアへ（MOVE_ATTACHED_DON） ------------------
    {
        "id": "move_attached_don_to_cost",
        "text": "自分の付与されているドン‼合計2枚をコストエリアにレストで戻すことができる。",
        "expect": [
            {"effect": {"kind": "action", "type": "MOVE_ATTACHED_DON", "value": 2}}
        ],
    },
    # ----- アタック対象変更（REDIRECT_ATTACK） ----------------------------
    {
        "id": "redirect_attack_selected",
        "text": "選んだキャラにアタックの対象を変更する。",
        "expect": [
            {"effect": {"kind": "action", "type": "REDIRECT_ATTACK"}}
        ],
    },
    # ----- 共有対象の二択「Xを、AかB」（加えるか登場させる）-------------------
    {
        "id": "shared_target_choice_life_or_play",
        "text": "【登場時】自分のトラッシュからコスト4以下の特徴《スリラーバーク海賊団》を持つキャラカード1枚までを、ライフの上に表向きで加えるか登場させる。",
        "expect": [
            {"effect": {"kind": "choice", "options": [
                {"kind": "action", "type": "MOVE_CARD", "destination": "LIFE"},
                {"kind": "action", "type": "PLAY_CARD"},
            ]}}
        ],
    },
    # ----- デッキの下に「置いてもよい」(て形, OP07-042 置換 sub_effect) ----------
    {
        "id": "deck_bottom_te_form",
        "text": "自分の、「ゲッコー・モリア」以外のキャラ1枚を持ち主のデッキの下に置いてもよい。",
        "expect": [
            {"effect": {"kind": "action", "type": "DECK_BOTTOM"}}
        ],
    },
    # ----- 自己登場の連用形断片「このカードを登場させ、…」(OP08-113) -------------
    {
        "id": "play_self_continuative",
        "text": "【トリガー】このカードを登場させ、相手のコスト3以下のキャラ1枚までを、KOする。",
        "expect": [
            {"effect": {"kind": "seq", "actions": [
                {"kind": "action", "type": "PLAY_CARD", "target": {"ref_id": "self"}},
                {"kind": "action", "type": "KO"},
            ]}}
        ],
    },
    # ----- ドン追加「追加し、」の後段を落とさない（OP09-022 MISSING_ACTION） -------
    {
        "id": "ramp_then_play",
        "text": "【起動メイン】ドン!!デッキからドン!!1枚までを、レストで追加し、自分の手札からコスト5以下のキャラカード1枚までを、登場させる。",
        "expect": [
            {"effect": {"kind": "seq", "actions": [
                {"kind": "action", "type": "RAMP_DON"},
                {"kind": "action", "type": "PLAY_CARD"},
            ]}}
        ],
    },
    # ----- 自ライフ上の公開（FACE_UP_LIFE, OP15-119） ----------------------------
    {
        "id": "reveal_own_life_top",
        "text": "自分のライフの上から1枚までを公開する。",
        "expect": [
            {"effect": {"kind": "action", "type": "FACE_UP_LIFE",
                        "target": {"zone": "LIFE", "player": "SELF"}}}
        ],
    },
    # ----- 自己効果無効「は」(受動・no-op, OP05-100 / OP09-081 前段) --------------
    {
        "id": "self_effect_negated_noop",
        "text": "この効果は無効になる。",
        "expect": [
            {"effect": {"kind": "action", "type": "RULE_PROCESSING"}}
        ],
    },
]
