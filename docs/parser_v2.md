# 効果パーサ刷新 (Parser V2) — 合成ルールレジストリ

カード効果が「想定通り実行されない」問題への長期的アーキテクチャ刷新。
本ドキュメントは設計方針・構成・開発フロー・ルール一覧をまとめる。

## 背景と方針

旧 `parser.py` は `_detect_action_type()` の巨大な if 連鎖で原子句を解釈しており、

- 順序依存で壊れやすい
- どの表現が未対応か分からない（サイレント失敗）
- 単体テストしづらい

という課題があった。診断では **`ActionType.OTHER`（=解析できたが何もしない）が約940件**
存在し、これが「効果が動かない」主因の一つと判明した。

刷新の核となる判断：

1. **中間表現(IR)は流用する。** `Ability / GameAction / TargetQuery / Condition / ValueSource`
   等はよく設計されており、`resolver` がそのまま実行できる。刷新対象は
   「日本語テキスト → IR」の変換部のみ。
2. **インターフェースを保つ。** `parse_card_text() -> List[Ability]` を維持するため、
   `resolver` / `gamestate` は無改修。`loader` の差し替えも1行で済む。
3. **段階的移行。** 新パーサは「構造分解」は旧実装を再利用し、刷新の主眼である
   **原子句の解釈だけ**をルールレジストリに置き換える。未対応句は旧実装へ
   フォールバックしつつ記録するので、**本番は決して壊れない**。

## 構成

```
opcg_sim/src/core/effects/
├── parser.py            # レガシー（構造分解＝トリガー/コスト/条件/逐次/選択肢 を担当）
├── parser_v2.py         # EffectParserV2: レガシーを継承し原子句解釈だけ差し替え
└── rules/
    ├── base.py          # Rule / RuleRegistry / ParseContext / @rule デコレータ
    ├── atoms.py         # 宣言的な原子アクションルール（ここを育てる）
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

## 開発フロー（TDD）

新しい表現に対応するときは次のサイクルで進める。

1. **診断で標的を選ぶ**
   ```bash
   OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40
   ```
   「未対応句ランキング」上位（＝頻度が高い表現）から着手する。

2. **ゴールデンケースを追加して赤にする**
   `tests/golden/golden_cases.py` に `text` と期待 `summary` を書く。
   ```bash
   OPCG_LOG_SILENT=1 python tests/test_golden.py
   ```

3. **ルールを足して緑にする**
   `opcg_sim/src/core/effects/rules/atoms.py` に `@rule(...)` を1つ追加。

4. **回帰確認＋カバレッジ確認**
   ```bash
   python -m pytest tests/ -p no:capture -q
   OPCG_LOG_SILENT=1 python tests/compare_parsers.py
   OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py
   # 挙動ベースラインが変わった場合は差分確認後に更新
   OPCG_LOG_SILENT=1 python tests/full_card_audit.py --regen
   ```

ルールは `matches→build` を兼ねた関数として書ける（`@rule` デコレータ）。
`priority` が大きいほど先に試行される（具体的・限定的なルールを高く）。

## 本番への適用（有効化済み）

`loader.make_parser()` がパーサを生成し、**既定で EffectParserV2** を使用する。
IR・インターフェースが同一のため resolver / gamestate は無改修。

### ロールバック

環境変数で即時に従来パーサへ戻せる（コード変更・再デプロイ不要）:

```
OPCG_PARSER=legacy   # 従来の EffectParser を使用
OPCG_PARSER=v2       # 既定。EffectParserV2 を使用
```

V2 読み込みに失敗した場合も自動的にレガシーへ退避する（フェイルセーフ）。

## 現況

| 指標 | 値 |
|---|---|
| 原子句カバレッジ（ルール命中率） | **98.8%** |
| `ActionType.OTHER`（実行時に何もしない句） | **39** |
| atoms.py ルール数 | **73** |
| ゴールデンコーパス | **~134件** |
| テスト総数 | **259（全緑）** |
| レガシー vs V2 退行 | **0** |

## atoms.py ルール一覧（73ルール）

priority 降順で試行される（大きいほど先）。

| ルール名 | priority | 概要 |
|---|---|---|
| `rule_processing` | 100 | メタルール処理（ターン1回制限等のルール句） |
| `draw` | 80 | ドロー（N枚引く） |
| `ko` | 80 | KO（フィールドの対象をKO） |
| `look_deck` | 80 | デッキの上からN枚を見る（LOOK、TEMP移動） |
| `search_deck_to_hand` | 80 | デッキサーチ→手札 |
| `search_to_hand` | 78 | TEMP→手札（サーチ後半） |
| `temp_to_deck` | 78 | TEMP→デッキ（サーチ残り戻し） |
| `scry_place` | 78 | ライフ上/下への置き直し |
| `mill_deck` | 75 | デッキトップN枚をトラッシュ（mill） |
| `discard` | 75 | 手札から捨てる |
| `don_add` | 75 | ドン追加（アクティブ/レスト） |
| `don_attach` | 75 | ドン付与（キャラへ） |
| `life_recover` | 75 | ライフ回復（デッキ→ライフ） |
| `life_to_hand` | 75 | ライフ→手札 |
| `hand_to_life` | 75 | 手札→ライフ |
| `life_to_trash` | 75 | ライフ→トラッシュ |
| `life_cards_to_trash` | 75 | ライフ複数→トラッシュ |
| `life_scry_top` | 75 | ライフ上から見て戻す |
| `life_face` | 75 | ライフを表/裏向きにする |
| `don_set_rest` | 72 | ドンをレストにする |
| `don_set_active` | 72 | ドンをアクティブにする |
| `don_return_deck` | 72 | ドンをドンデッキに戻す |
| `don_return` | 70 | ドン返却（コストエリアから） |
| `play_card_from_zone` | 70 | ゾーンからカードを登場/発動 |
| `play_from_deck` | 70 | デッキからカードを登場 |
| `play_from_temp` | 70 | TEMP（公開中）からカードを登場 |
| `play_revealed` | 70 | 公開したカードを登場 |
| `play_self` | 70 | 自分自身を登場させる |
| `reveal_hand` | 68 | 手札を公開する |
| `trash_to_hand` | 67 | トラッシュ→手札 |
| `hand_to_deck` | 67 | 手札→デッキ |
| `self_to_hand` | 67 | 自分自身を手札に戻す |
| `bounce` | 65 | バウンス（フィールド→手札） |
| `deck_bottom_general` | 65 | デッキの下に置く |
| `remaining_deck_bottom` | 63 | 残りをデッキの下に置く |
| `remaining_deck_top_or_bottom` | 63 | 残りをデッキの上か下に置く |
| `remaining_trash` | 63 | 残りをトラッシュに置く |
| `set_power` | 62 | パワーを特定値に設定（POWER_OVERRIDE） |
| `power_swap` | 61 | 2キャラのパワーを入れ替える（SWAP_POWER） |
| `power_equalize` | 60 | パワーを参照カードと同値に（POWER_OVERRIDE+REFERENCE_POWER） |
| `set_cost` | 60 | コストを特定値に設定（COST_OVERRIDE） |
| `cost_change` | 58 | コスト増減（+N / -N / N少なくなる） |
| `power_buff` | 55 | パワー増減（+N / -N） |
| `trash_target` | 53 | 対象をトラッシュに置く |
| `trash_self` | 52 | 自身をトラッシュに置く |
| `active_target` | 50 | 対象をアクティブにする |
| `active_self` | 50 | 自身をアクティブにする |
| `rest` | 50 | 対象をレストにする |
| `rest_self` | 50 | 自身をレストにする |
| `rest_self_cost` | 50 | コストとして自身をレスト |
| `attack_active` | 48 | アタック後アクティブになる |
| `freeze_target` | 46 | 凍結（アクティブにならない、FREEZE） |
| `select_target` | 45 | 対象選択（選択フラグのみ） |
| `negate_effect` | 44 | 効果を無効にする（他動詞・対象指定あり） |
| `self_effect_disabled` | 64 | 効果が無効になる（自動詞・自身の効果が無効化される） |
| `grant_keyword` | 40 | キーワード能力を付与（ブロッカー/速攻等） |
| `prevent_leave_and_keyword` | 70 | 場を離れない＋キーワード付与の複合 |
| `prevent_leave` | 35 | 場を離れない / KOされない（保護） |
| `rest_restrict` | 38 | レストにできない制限 |
| `attack_disable` | 37 | アタックできない制限 |
| `blocker_disable` | 36 | ブロッカー発動制限 |
| `self_cannot` | 33 | 自己制限（ドロー不可/アタック不可等、期間付き） |
| `rush_natural` | 30 | 速攻（登場時アタック可） |
| `execute_main` | 28 | このカードの【メイン】効果を発動 |
| `shuffle` | 25 | シャッフル |
| `declare_cost` | 90 | コスト宣言インタラクション（C8 メカニクス） |
| `win_on_deckout` | 20 | デッキアウト時勝利置換（C10 メカニクス） |
| `look_opp_deck` | 81 | 相手デッキ上 N 枚の覗き見（LOOK+OPPONENT・盤面不変） |
| `order_life` | 77 | ライフ並び替え（ORDER_LIFE・「好きな順番で置く」） |
| `execute_event` | 71 | 手札イベントの発動（EXECUTE_EVENT・効果解決→トラッシュ） |
| `prevent_ko_and_rest` | 67 | 複合除去保護（PREVENT_LEAVE + PREVENT_REST） |
| `prevent_rest_self` | 66 | レスト不可保護（自身が相手効果でレストされない） |
| `deal_damage` | 55 | 効果ダメージ（「相手にNダメージを与える／自分はN受ける」） |

> 上記に加え、`set_power` 等の既存ルールに細かな活用形対応（`life_recover` の連用形「加え」、
> `prevent_leave` の「KOされず」連用形）を追記。対象解析(`matcher.parse_target`)には
> `exclude_names`（「「◯◯」以外」の除外）を新設した。

### 二択「AするかB、する」の Choice 化

`parser._parse_suruka_choice` を追加。動詞終止形（u 段かな）直後の「か、」だけを境界に
2 アクションへ分割し `Choice` 化する。名詞並列（「自分か相手」「リーダーかキャラ」）は
語尾で除外され、左右がともに実行系アクションに解釈できる場合のみ Choice 化する（過検知防止）。
3 択（「KOするか、戻すか、置く」）は二択のネストで表現される。

## 主要な parser.py 構造修正

原子句ルール以外に、parser.py の構造分解レベルでも重要な修正が入っている。

### 連用形「〜し、」連結句の Sequence 分割

「相手のキャラ1枚をKOし、自分のキャラ1枚を手札に戻す」のような連用形連結を
Sequence に正しく分割。`(?<=KOし)、`/`(?<=レストにし)、`/`(?<=戻し)、` 等を
`_parse_to_node` の分割パターンに追加。

### モーダル選択「以下から1つを選ぶ」の構造修正

`以下から1つを選ぶ` を含む全カードが `Choice(options=[])` になる silent no-op を修正。
- `parse_card_text`: Choice 導入セグメントに続く選択肢セグメントを `\n` で再結合
- `_parse_to_node`: `。` 分割より前に Choice を検出し `_parse_choice` で生成
- `_parse_choice`: option 内の条件分岐も Branch 化、`相手は…選ぶ` は `Choice.player=OPPONENT`

### サーチ構造分解

「デッキの上からN枚を見て、M枚を手札に加える。残りを…」形式のサーチを
`look_deck`（LOOK）→ `search_to_hand`（TEMP→HAND）→ `temp_to_deck`（残り戻し）の
3段に正しく分割。読点 `、` を `。` に置換して LOOK クローズを独立させる。

> **TEMP リーク注意**: LOOK は候補を temp_zone に移すため、後続で TEMP を必ず消費すること。
> 消費しないと temp_zone にカードが取り残されデッキから消失する。

## 継続効果（期間付き効果）— `effects/continuous.py`

「このバトル中」「このターン中」「次の相手のターン終了時まで」のように、適用後に
特定タイミングで失効する効果を `ContinuousEffectManager` が一元管理する。

- 効果は CardInstance の専用フィールド `timed_power` / `timed_cost` / `timed_flags` /
  `timed_keywords` に反映。`reset_turn_status()` でクリアされない。
- `get_power()` は `timed_power` を加算。アタック制限は `timed_flags` を参照。
- 失効は `expire(event)` を **バトル終了**（`resolve_attack`）と
  **ターン終了**（`end_turn`）のフックで呼ぶ。
- 対応 Duration: `THIS_BATTLE` / `THIS_TURN` / `UNTIL_NEXT_TURN_END` / `PERMANENT`

## 除去保護（PREVENT_LEAVE）

「相手の効果で場を離れない」「バトルでKOされない」のように、カードが場を離れる
瞬間に介入して除去を無効化する効果。

- `GameManager._active_protection(card, status_values)` が除去の瞬間にその場評価する。
- フック点: 効果除去（KO/DISCARD/BOUNCE 等の `status="LEAVE"`）/ バトルKO（`status="BATTLE_KO"`）

## 置換効果（REPLACE_EFFECT）

「このキャラがKOされる/場を離れる場合、代わりに〜」の処理。

- パーサ: 「代わりに」＋「このキャラ…される/場を離れる」を検出し、
  置換アクションを `GameAction.sub_effect` に保持した `REPLACE_EFFECT`（PASSIVE）を生成。
- エンジン: `_active_replacement(card, status_values)` が条件ライブ評価＋実行可能性を確認し、
  満たせば置換を実行して本来の除去をスキップ。
- 制約: `sub_effect` が対象選択を必要とする場合（複数枚捨てる等）は UI が出ない（E14）。
  「できる」型の任意置換 UI も未対応（E15）。
