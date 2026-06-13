# 効果パーサ (Parser V2) — 合成ルールレジストリ

「日本語テキスト → 中間表現(IR)」変換の実装ドキュメント。構成・動作・開発フロー・
主要メカニズムを記述する。

## 概要

- **中間表現(IR)**: `Ability / GameAction / TargetQuery / Condition / ValueSource`。
  `resolver` が実行する。V2 の対象は「日本語テキスト → IR」の変換部のみ。
- **インターフェース**: `parse_card_text() -> List[Ability]`（レガシーと同一）。
  `resolver` / `gamestate` / `loader` は差し替えに無改修。
- **構造分解**（トリガー/コスト/条件/逐次/分岐/選択肢）はレガシー `parser.py` を再利用し、
  **原子句の解釈のみ**をルールレジストリに置き換える。未対応句はレガシー
  `_parse_atomic_action()` にフォールバックし `unmatched` に記録する。

## 構成

```
opcg_sim/src/core/effects/
├── parser.py            # レガシー（構造分解＝トリガー/コスト/条件/逐次/選択肢 を担当）
├── parser_v2.py         # EffectParserV2: レガシーを継承し原子句解釈だけ差し替え
└── rules/
    ├── base.py          # Rule / RuleRegistry / ParseContext / @rule デコレータ
    ├── atoms.py         # 宣言的な原子アクションルール
    └── __init__.py      # default_registry を公開し atoms を自動登録
```

```
tests/
├── golden/
│   ├── golden_cases.py  # ゴールデンコーパス（効果セマンティクスの期待値）
│   └── summarize.py     # AST→指紋(summary) 変換＋部分一致判定
├── test_golden.py       # ゴールデン・ランナー（pytest / 単体実行 両対応）
└── effect_diagnostics.py# 全カードに適用し、未対応句ランキング等を可視化
```

### 動作の流れ（EffectParserV2）

1. レガシーと同じ手順でカードテキストを能力単位に分割し、トリガー・コスト・条件・
   逐次(Sequence)・分岐(Branch)・選択肢(Choice) の **構造** を組み立てる。
2. 葉に当たる **原子句** の解釈時に `default_registry.apply()` を呼ぶ。
   - ルールが一致 → そのルールが構築した `GameAction` を採用（`rule_hits` に記録）
   - 不一致 → レガシー `_parse_atomic_action()` にフォールバック（`unmatched` に記録）
   - フォールバック結果が `ActionType.OTHER` の原子句は `fallback_other` に記録

## 開発フロー（TDD）

1. **診断で標的を選ぶ**
   ```bash
   OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40
   ```
2. **ゴールデンケースを追加** — `tests/golden/golden_cases.py` に `text` と期待 `summary`
   ```bash
   OPCG_LOG_SILENT=1 python tests/test_golden.py
   ```
3. **ルールを追加** — `opcg_sim/src/core/effects/rules/atoms.py` に `@rule(...)`
4. **回帰確認＋カバレッジ確認**
   ```bash
   python -m pytest tests/ -p no:capture -q
   OPCG_LOG_SILENT=1 python tests/compare_parsers.py
   OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py
   OPCG_LOG_SILENT=1 python tests/full_card_audit.py --regen   # 挙動を意図的に変えた場合
   ```

ルールは `matches→build` を兼ねた関数として書ける（`@rule` デコレータ）。
`priority` が大きいほど先に試行される。

## パーサの選択（環境変数）

`loader.make_parser()` がパーサを生成する。既定は `EffectParserV2`。IR・インターフェースが
同一のため `resolver` / `gamestate` は差し替えに無改修。

```
OPCG_PARSER=v2       # 既定。EffectParserV2 を使用
OPCG_PARSER=legacy   # 従来の EffectParser を使用
```

V2 読み込みに失敗した場合は自動的にレガシーへ退避する。

## atoms.py ルール

ルールは `@rule(name, priority)` で `default_registry` に登録され、priority 降順で試行される。
**完全な一覧と最新の priority・条件は `atoms.py` の各 `@rule` デコレータと docstring が正**。
以下は領域別の概観（網羅ではない）。

| 領域 | 代表ルール |
|---|---|
| ドロー/KO | `draw` / `ko` |
| デッキ参照・サーチ | `look_deck` / `search_deck_to_hand` / `search_to_hand` / `temp_to_deck` / `look_opp_deck` / `reveal_deck_top` |
| 登場 | `play_card_from_zone` / `play_from_deck` / `play_from_temp` / `play_revealed` / `play_self` / `dual_tier_play_from_trash` |
| 移動 | `bounce` / `self_to_hand` / `trash_to_hand` / `hand_to_deck` / `deck_bottom_general` / `remaining_deck_bottom` / `remaining_deck_top_or_bottom` / `remaining_trash` / `mill_deck` |
| ライフ操作 | `life_recover` / `life_to_hand` / `hand_to_life` / `field_char_to_life` / `life_to_trash` / `life_cards_to_trash` / `life_scry_top` / `life_face` / `life_to_deck_top` / `order_life` / `reveal_own_life_top` / `search_to_life` |
| ドン操作 | `don_add` / `don_attach` / `don_set_rest` / `don_set_active` / `don_return` / `don_return_deck` / `don_cost_circled` / `don_rest_cost_fragment` / `move_attached_don` |
| パワー/コスト | `power_buff` / `set_power` / `power_swap` / `power_equalize` / `set_cost` / `cost_change`（対象は `_buff_target`） |
| 状態/制限 | `rest`(+self/cost) / `active`(+self) / `attack_active` / `freeze_target` / `attack_disable` / `rest_restrict` / `blocker_disable` / `self_cannot` / `rush_natural` |
| 効果無効 | `negate_effect`（他動詞） / `self_effect_disabled`（自動詞） / `self_effect_negated_noop`（受動・no-op） / `scoped_negate_opp_onplay`（相手【登場時】無効） |
| 除去保護/置換 | `prevent_leave` / `prevent_leave_and_keyword` / `prevent_ko_and_rest` / `prevent_rest_self` |
| 選択/トラッシュ | `select_target` / `trash_target` / `trash_self` / `discard` / `reveal_hand` |
| その他/メカニクス | `grant_keyword` / `deal_damage` / `execute_main` / `execute_event` / `shuffle` / `declare_cost`(C8) / `declare_victory` / `win_on_deckout`(C10) / `redirect_attack` / `rested_play_passive` / `no_effect_play_passive` / `rule_processing` / `bare_number_cost_noop` |

### 「持ち主の〜」系除去の対象側（側無指定 → ALL）

「（コストN以下の）キャラ…を**持ち主の**手札／デッキの下／ライフに戻す・置く・加える」
（`bounce` / `deck_bottom_general` / `field_char_to_life`）で、**側の明示（自分の／相手の）が
無い**場合の対象は、テキスト準拠で**自分・相手の両方**（`Player.ALL`）。戻り先は常に対象
カードの持ち主のゾーン。除去の既定選択（CPU・自己対戦・監査ハーネス）が自分のキャラを
自爆対象にしないよう、`matcher.get_target_cards` は `Player.ALL` 候補を**「相手→自分」順**で
並べる（先頭＝相手キャラ）。UI 上は両側から選択可能。

例外:
- 「相手の」「自分の」明示はそれぞれ `OPPONENT` / `SELF` として尊重する。
- 「この（カード／キャラ／ステージ）」= 自己参照（コスト等）は ALL 既定の対象外（発生源自身）。

### `field_char_to_life`（場のキャラ → 持ち主のライフ）

「（コストN以下の）キャラ…を持ち主のライフの上（か下）に（表向き／裏向きで）加える」を
`MOVE_CARD(dest=LIFE)` 化する（OP03-123 等）。`life_face`（自ライフの反転）より高 priority で
先取りする。「上か下」は上下選択の `Choice`、片側のみなら固定。`face_up` は表向き／裏向きを反映。
「キャラ」明示が無い置換文脈（「（…場を離れる場合、）代わりに…ライフに加える」OP11-101）は
対象＝発生源自身（`select_mode="SOURCE"`）。

### 二択「AするかB、する」の Choice 化

`parser._parse_suruka_choice`。動詞終止形（u 段かな）直後の「か、」だけを境界に2アクションへ分割し
`Choice` 化する。名詞並列（「自分か相手」「リーダーかキャラ」）は語尾で除外し、左右がともに
実行系アクションに解釈できる場合のみ Choice 化する。共有対象の二択は `_parse_shared_target_choice`。

## parser.py の主要な構造分解

原子句ルール以外に、`parser.py` の構造分解レベルで以下を扱う。

### 連用形「〜し、」連結句の Sequence 分割

連用形連結（「相手のキャラ1枚をKOし、自分のキャラ1枚を手札に戻す」等）を Sequence に分割する。
`_parse_to_node` の split_pattern に `(?<=KOし)、`/`(?<=レストにし)、`/`(?<=戻し)、`/`(?<=追加し)、`
等の境界を持つ。

### モーダル選択「以下から1つを選ぶ」

- `parse_card_text`: Choice 導入セグメントに続く「・」選択肢セグメントを `\n` で再結合する。
- `_parse_to_node`: `。` 分割より前に Choice を検出し `_parse_choice` で生成する。
- `_parse_choice`: option 内の条件分岐も Branch 化、`相手は…選ぶ` は `Choice.player=OPPONENT`。

### 段階効果「…の枚数によって以下の効果をそれぞれ適用する」

`_parse_apply_each`。head から参照ゾーン（TRASH/LIFE/HAND/DECK）を判定し、各「・N枚以上…」項目を
`Branch[ZONE_COUNT>=N] → 効果` に変換した Sequence を生成する。

### サーチ構造分解

「デッキの上からN枚を見て、M枚を手札に加える。残りを…」形式を `look_deck`（LOOK）→
`search_to_hand`（TEMP→HAND）→ `temp_to_deck`（残り戻し）の3段に分割する。読点 `、` を `。` に
置換して LOOK クローズを独立させる。

> **TEMP リーク注意**: LOOK は候補を `temp_zone` に移すため、後続で TEMP を消費する必要がある。
> 消費されなかった TEMP は解決完了時に `resolver._reclaim_temp_to_deck_top` がデッキトップへ戻す。

## 継続効果（期間付き効果）— `effects/continuous.py`

「このバトル中」「このターン中」「次の相手のターン終了時まで」のように、適用後に特定タイミングで
失効する効果を `ContinuousEffectManager` が管理する。

- CardInstance の専用フィールド `timed_power` / `timed_cost` / `timed_flags` / `timed_keywords` に
  反映する。`reset_turn_status()` ではクリアしない。
- `get_power()` は `timed_power` を加算。アタック制限は `timed_flags` を参照。
- 失効は `expire(event)` を **バトル終了**（`resolve_attack`）と **ターン終了**（`end_turn`）で呼ぶ。
- Duration: `THIS_BATTLE` / `THIS_TURN` / `UNTIL_NEXT_TURN_END` / `PERMANENT`。

## 除去保護（PREVENT_LEAVE）

カードが場を離れる瞬間に介入する効果（「相手の効果で場を離れない」「バトルでKOされない」）。

- `GameManager._active_protection(card, status_values)` が除去の瞬間に評価する。
- フック点: 効果除去（KO/DISCARD/BOUNCE 等の `status="LEAVE"`）/ バトルKO（`status="BATTLE_KO"`）。

## 置換効果（REPLACE_EFFECT）

「このキャラがKOされる/場を離れる場合、代わりに〜」の処理。

- パーサ: 「代わりに」＋「このキャラ…される/場を離れる」を検出し、置換アクションを
  `GameAction.sub_effect` に保持した `REPLACE_EFFECT`（PASSIVE）を生成する。
- エンジン: `_active_replacement(card, status_values)` が条件評価＋実行可能性（`_can_satisfy_node`）を
  確認し、満たせば置換を実行して本来の除去をスキップする。
- 置換 `sub_effect` の実行・実行可能性判定の **source は「離れるカード」(`card`)** とする
  （条件評価／ターン1回管理は能力保持カード `protector` のまま）。これにより「代わりに
  （そのカードを）ライフに加える」等が、保護者ではなく離れるカード自身を対象に取る
  （OP11-101: 離れる《超新星》キャラを持ち主のライフへ。`field_char_to_life` の SOURCE 解決）。
- 置換 `sub_effect` が対象選択/任意確認で中断した場合は `_auto_resolve_replacement` が同期解決する
  （任意=accept、対象=有効候補を自動選択）。除去解決中に走るため、解決前後で外側の
  `active_interaction` を保全する。フロントへ選択を提示する完全な対話化は未実装。
