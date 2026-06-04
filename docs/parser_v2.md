# 効果パーサ刷新 (Parser V2) — 合成ルールレジストリ

カード効果が「想定通り実行されない」問題（改善策⑥）への長期的アーキテクチャ刷新。
本ドキュメントは設計方針・構成・開発フローをまとめる。

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
   OPCG_LOG_SILENT=1 python tests/test_golden.py        # 全ケース緑
   OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py # 命中率↑/フォールバック率↓
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

### 有効化前の安全確認（`tests/compare_parsers.py`）

全カードでレガシー出力と V2 出力を比較し、分類する:
- 完全一致 / 改善(OTHER解消) / その他差異 / **退行(新規OTHER)**

退行が出たら非0終了するため CI ゲートに使える。有効化時点の結果:
**完全一致 730 / 改善 449 / その他差異 1194 / 退行 0**。
「その他差異」は主に「DRAW の不要な target が None になる」「バトル中バフに
duration=THIS_BATTLE が付く」等の改善・等価変化であることを確認済み。

## 既知の課題（診断が表面化させたもの）

- `ActionType.OTHER` 約940件：`resolver`/`gamestate` 側に実行処理が無いため、
  解析できても盤面が変わらない。ルール整備と並行して
  `apply_action_to_engine()` の対応アクション拡充が必要（改善策④）。
- ~~「このキャラはを得る」等：キーワード付与（【ブロッカー】を得る 等）の
  セグメント分割でキーワードが脱落するケース。~~ **対応済み**（`grant_keyword` ルール）。
  原因は `parse_ability` の `clean_text = re.sub('【.*?】','',…)` が
  キーワード能力タグ（【ブロッカー】等）まで一括除去していたこと。除去対象を
  トリガー/注釈タグに限定し（7種のキーワードタグは保持）、`grant_keyword` ルールが
  `GRANT_KEYWORD(status=キーワード名, duration)` を生成する。あわせて
  【速攻：キャラ】内部の「：」がコスト区切りと誤認される問題も修正
  （タグ内部をマスクしてコロン判定）。146 句が `GRANT_KEYWORD` に解決され、
  従来は誤って `BUFF`（キーワード脱落）に落ちていた。
- ~~ライフ操作（デッキ→ライフ／ライフ→手札／手札→ライフ／ライフ→トラッシュ／
  表・裏向き）~~ **対応済み**（`life_recover` / `life_to_hand` / `hand_to_life` /
  `life_to_trash` / `life_face`）。`life_to_hand` は legacy が「ライフの上か下から…
  手札に加える」を `destination=LIFE` と誤判定していた（実質 no-op）バグを修正。
  `life_recover` は対象選択待ちを避けるため `target=None`（エンジンの HEAL が
  デッキ上から value 枚をライフへ）。`life_face` は `FACE_UP_LIFE(status=UP/DOWN)`。
  残: 「ライフを見て上か下に置く」等の look-and-place 系は複雑なため未対応。
- ~~ドン!!操作（付与／アクティブ／レスト／ドンデッキに戻す）~~ **対応済み**
  （`don_attach` / `don_set_active` / `don_set_rest` / `don_return_deck`）。ドン!!は
  均質なため枚数(value)ベースで処理し、無意味な対象選択中断を避ける（付与のみ付与先を
  対象に持つ）。「相手は…」は `status="OPPONENT"`。あわせてエンジンの
  `REST_DON` 実行系が欠落しており【ドン!!×N】コストが実質 no-op だったバグも修正。
- `コスト-N` 系のコスト操作：未対応。`COST_CHANGE` 実装と合わせて対応予定。

## エンジン実行系（apply_action_to_engine）の拡充 — 改善策④

パーサが正しい ActionType を生成しても、`gamestate.apply_action_to_engine` に
実行処理が無ければ盤面は変わらない（=`ActionType.OTHER` 同様のサイレント失敗）。
そこで実行系を `tests/test_effects_engine.py`（実 GameManager 上で盤面変化を検証）
で守りながら拡充する。

実装済み（gamestate.apply_action_to_engine）:
- `RETURN_DON`（`ドン‼-N`）: 場のドン!!を N 枚ドン!!デッキへ戻す（レスト→アクティブ→付与の順）
- `RAMP_DON` の `status="RESTED"` 対応（`レストで追加`）: レスト状態でコストエリアへ
- `GRANT_KEYWORD`: `status` のキーワードを継続効果（`timed_keywords`）として付与
  （duration で失効、passive リセットで消えない）
- `FACE_UP_LIFE`: `status="DOWN"` で裏向き、それ以外で表向きに `is_face_up` を切替
- `REST_DON`（新規）/`ACTIVE_DON`（枚数ベース）: アクティブ↔レストを value 枚切替。
  `REST_DON` 欠落で【ドン!!×N】コストが no-op だったのを修正。
- `ATTACH_DON`: value 枚を付与（従来1枚固定）。`status="RESTED"` でレストのまま付与。
- `RETURN_DON`: `status="OPPONENT"`／対象 player で相手のドンも対象化。
- `TRASH_FROM_DECK`（新規）: デッキの上から value 枚をトラッシュへ送る（mill）。対象選択は
  させず枚数ベース、`status="OPPONENT"` で相手デッキを対象化。従来は ActionType が生成されても
  実行系が無くサイレント no-op だった（OTHER 指標には現れない「効果が動かない」例）。
- `PLAY_CARD` + `status="RESTED"`: フィールドへの登場後に `is_rest=True` をセット。
  「レストで登場させる」（トラッシュ/手札からの効果登場）で機能。
- `REVEAL`（新規）: カードを公開する（情報開示）。盤面は動かさず公開した事実をログに残す。
  「自分の手札から…を公開する」（条件成立の証明等）で機能。

実装済み（resolver）:
- `EXECUTE_MAIN_EFFECT`（`このカードの【メイン】効果を発動する`）: source_card 自身の
  ACTIVATE_MAIN 能力の効果を実行スタックへ展開して再発動（コストは支払わない）。
  自己参照による無限ループは `_main_expanded` フラグで1回に限定。
- **`TURN_LIMIT` enforce（【ターン1回】）**: `resolve_ability` が条件・コスト成立時に
  `source_card.ability_used_this_turn` を加算し、上限到達後は発動を抑止する。カウンタは
  `reset_turn_status`（毎ターン境界）でクリアされるためターン単位で機能する。
- **条件の fail-safe＋分類拡充**: 解釈不能な `OTHER` 条件は False（誤発動防止）。`GENERIC`
  （未分類だが実在する条件）は誤抑制回避のため許容＋ログとしつつ、評価可能なクラスタを
  実条件へ分類して誤発動源を削減（GENERIC 251→132）。新規分類: `LEADER_TRAIT`（『X』記法）、
  `FIELD_COUNT`（盤面のキャラ枚数, レスト/特徴/コスト/プレイヤーのフィルタ対応。数値は
  フィルタと枚数が混在し得るため閾値は「M枚」側から取る保守設計）、`DECK_COUNT`（デッキ枚数）、
  `LEADER_COLOR`（多色＝2色以上）。

## 継続効果（期間付き効果）の管理 — `effects/continuous.py`

「このバトル中」「このターン中」「次の相手のターン終了時まで」のように、適用後に
特定タイミングで失効する効果を `ContinuousEffectManager` が一元管理する。

設計（既存エンジン非破壊）:
- 効果は CardInstance の *専用フィールド* `timed_power` / `timed_flags` に反映。
  これらは `reset_turn_status()` でクリアされない＝ターン境界を跨いで存続できる。
  既存の `power_buff` / `flags`（ターン境界でリセット）とは独立で衝突しない。
- `get_power()` は `timed_power` を加算。アタック制限は `timed_flags` の
  `ATTACK_DISABLE` を `declare_attack` で参照。
- 失効は `expire(event)` を **バトル終了**（`resolve_attack`）と
  **ターン終了**（`end_turn`）のフックで呼ぶ。リセット後の再適用が不要。

対応 Duration: `THIS_BATTLE` / `THIS_TURN` / `UNTIL_NEXT_TURN_END` / `PERMANENT`。
対応 kind: `POWER`（timed_power）/ `COST`（timed_cost）/ `FLAG`（timed_flags）/
`KEYWORD`（timed_keywords）。
ルーティング: `apply_action_to_engine` で BUFF(duration=THIS_BATTLE) と
ATTACK_DISABLE/RESTRICTION、および GRANT_KEYWORD を継続効果として登録する。
キーワードは `timed_keywords` に保持され `_apply_passive_effects` のリセットで消えず
（`has_keyword()` で本来＋付与分を参照）、`drop_for` で場を離れる際に破棄される。

これにより「このバトル中バフが同ターンの後続バトルへ誤って持ち越す」バグや、
「次の相手のターン終了時までアタック不可」のような複数ターン跨ぎ制限が正しく動く。

## 除去保護（PREVENT_LEAVE）

「相手の効果で場を離れない」「バトルでKOされない」のように、カードが場を離れる
瞬間に介入して除去を無効化する効果。多くは条件付き PASSIVE（例: トラッシュ7枚以上）。

設計（ライブ評価・ラッチしない）:
- `GameManager._active_protection(card, status_values)` が、除去の瞬間に対象カードの
  PASSIVE 能力（effect が PREVENT_LEAVE）を走査し、条件を `EffectResolver._check_condition`
  でその場評価する。スナップショット的なフラグを持たないため、条件変動に正しく追随。
- フック点:
  - 効果除去（`apply_action_to_engine` の KO/DISCARD/BOUNCE/MOVE/DECK_* 等）:
    相手の効果で・フィールド上の対象に対してのみ `status="LEAVE"` 保護を確認。
  - バトルKO（`resolve_attack`）: `status="BATTLE_KO"` 保護を確認。
- パーサ: `prevent_leave` ルールが `場を離れない`→LEAVE / `KOされない`→BATTLE_KO を生成。

## 置換効果（REPLACE_EFFECT）

「このキャラが(バトルで)?KOされる/場を離れる場合、代わりに〜」を、除去を別の行動に
置き換える効果として扱う（除去保護の枠組みの拡張）。

設計:
- パーサ: `parse_ability` が「代わりに」＋「このキャラ…される/場を離れる」を検出し、
  置換アクションを `GameAction.sub_effect` に保持した `REPLACE_EFFECT`（PASSIVE）を生成。
  `バトル`→`status=BATTLE_KO` / それ以外→`status=LEAVE`。「…される場合」はゲート条件では
  なくトリガー文脈なので `ability.condition` には載せない。
- エンジン: `GameManager._active_replacement(card, status_values)` が除去の瞬間に PASSIVE の
  `REPLACE_EFFECT` を走査し、条件（ライブ評価）と置換の実行可能性（`_can_satisfy_node`、
  例: 捨てる手札があるか）を満たせば置換を実行し、本来の除去をスキップする。
- フック点: 効果除去（`status="LEAVE"`）/ バトルKO（`status="BATTLE_KO"`）。
- MVP制約: 自身（このキャラ）のみ対象。他キャラを守る型・置換が対象選択で中断する場合・
  「できる」の任意選択UIは今後の課題。

## 現況（ルール42種 + エンジン/resolver/継続効果/除去保護/置換効果/サーチ構造修正 時点）

- 原子句カバレッジ（ルール命中率）: **約92.5%**
- `ActionType.OTHER`（実行時に何もしない句）: **234**（開始時 942 から約75%削減）
- テスト: `test_parser.py`(8) + `test_golden.py`(62) + `test_effects_engine.py`(43)
  + `test_gameplay_smoke.py`(2) = 全115件緑
- レガシー vs V2 全カード比較: 退行(新規OTHER)=0 を維持
- パーサルール: draw / ko / rest / rest_self_cost / power_buff / discard /
  cost_change / play_self / shuffle / remaining_deck_bottom / don_return / don_add /
  execute_main / attack_disable / prevent_leave / grant_keyword /
  life_recover / life_to_hand / hand_to_life / life_to_trash / life_face /
  don_attach / don_set_active / don_set_rest / don_return_deck /
  trash_self / active_self / mill_deck / remaining_trash /
  bounce / deck_bottom_general / remaining_deck_top_or_bottom / play_card_from_zone /
  active_target / blocker_disable / rush_natural / reveal_hand /
  **look_deck / search_to_hand / temp_to_deck**（＋ rest/mill_deck を拡張、
  ＋ parser.py に「デッキの上からN枚を見て、」の構造分割を追加）

### サーチ構造分解の修正（parser.py `_parse_to_node`）

「デッキの上からN枚を見て、…M枚までを公開し、手札に加える。残りを…」のような **サーチ** は、
従来「見て、」が分割境界に無く 1 原子句化していたため、`parse_target` が「N枚」を count に
誤取得し、LOOK も欠落して誤った BOUNCE を生成していた。`_parse_to_node` で
「デッキの上から\d+枚を見て、」の読点を「。」に置換して **LOOK を独立クローズ化** し、
`look_deck`(→LOOK) / `search_to_hand`(TEMP→HAND) / `temp_to_deck`(残り TEMP→デッキ) /
`remaining_deck_bottom` が各クローズを解釈する。デッキ文脈に限定してライフ等の「見て、」へは
影響させず、`compare_parsers` で退行=0 を確認済み（改善 449→735）。

> **TEMP リーク注意**: LOOK は候補を temp_zone に移すため、後続で TEMP を必ず消費する
> （grab／残り戻し）こと。消費しないと temp_zone にカードが取り残されデッキから消失する。

### 残課題（今後・長い裾野）

残る OTHER は頻度の低い多様な専用効果に分散（上位でも 1 表現あたり 10 件前後）。
費用対効果は逓減しており、今後は次の2方針が現実的:
1. ライフ操作（表/裏向き）・デッキ操作（並び替え）・ドン!!可変返却などを順次ルール化。
2. V2 を本番有効化し、実デッキ(imu/nami)で回しながら不足を golden に追加して個別対応。
- COST/KEYWORD の duration 対応は `_apply_passive_effects` の再計算と統合する設計が必要。
