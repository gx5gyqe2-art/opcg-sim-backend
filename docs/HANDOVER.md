# カード効果システム — 構成リファレンス

本書は `opcg-sim-backend` の **カード効果処理システム** の構成・実行方法・実装上の不変条件を、
独立したレビューのために事実ベースで記述する。設計詳細は `docs/parser_v2.md` を参照。

---

## 1. アーキテクチャ

### 効果処理のパイプライン

```
カードDB(日本語テキスト)
   │  loader.py: _create_card_master() / make_parser()
   ▼
                [ EffectParserV2 ]
                        │ 構造分解(レガシー流用)
                        │ + 原子句のみ rules で解釈
                        │ + 未対応はレガシーへフォールバック
                        ▼
                   Ability(IR)
                        │ trigger / condition / cost / effect
                        ▼  ゲーム中、該当タイミングで
                  resolver.py（EffectResolver）
                        │  AST を実行スタックで処理（対象選択/任意確認は中断/再開）
                        ▼
                  gamestate.py（apply_action_to_engine / continuous / 除去保護
                        │         / 誘発キュー _pending_triggers）
                        ▼
                     盤面更新
```

### 誘発能力の待ち行列（_pending_triggers）

ライフ公開【トリガー】・`ON_LIFE_DECREASE` 等の誘発能力は `gamestate._pending_triggers` へ積み、
`_advance_pending_triggers()` が1件ずつ消化する（`_battle_triggers` と同型）。中断（対話）が立てば
`resolve_interaction` が解決後に再開する。【トリガー】は「発動できる」（任意）のため
`optional=True` で積み、`CONFIRM_TRIGGER` 対話で使う/使わないを確認してから解決する
（複数枚＝ダブルアタック等でも中断を跨いで消失しない）。ドレインは戦闘解決末尾・効果ダメージ末尾・
対話完了時・API アクション境界で行う。

### パーサ構成

`EffectParserV2` は `EffectParser`(レガシー) を**継承**し、`_parse_atomic_action()` のみを
オーバーライドする。トリガー判定・コスト分離・逐次/分岐/選択肢の構造分解はレガシーを使う。

- 原子句は `default_registry.apply(ctx)` でルール優先解釈する。
- どのルールも一致しなければレガシー実装にフォールバックし、その句を `unmatched` に記録する。
- フォールバック結果が `ActionType.OTHER`（実行系のない句）になった原子句は `fallback_other` に記録する。

### 中間表現(IR)

`models/effect_types.py` に定義。`Ability`（trigger / condition / cost / effect）を頂点に、
効果ツリーは `GameAction` / `Sequence` / `Branch` / `Choice` の組み合わせ。
`GameAction.sub_effect` は置換効果（REPLACE_EFFECT）の置換アクションを保持する。

### 継続効果（期間付き効果）

`effects/continuous.py` の `ContinuousEffectManager`。

- `CardInstance` の専用フィールド `timed_power` / `timed_cost` / `timed_flags` /
  `timed_keywords` に反映する。これらは `reset_turn_status()` でクリアされない。既存の
  `power_buff`/`cost_buff`/`flags`/`current_keywords`（ターン境界 or passive 再計算でリセット）
  とは別フィールド。
- kind: `POWER` / `COST` / `FLAG` / `KEYWORD`。Duration: `THIS_BATTLE` / `THIS_TURN` /
  `UNTIL_NEXT_TURN_END` / `PERMANENT`（場を離れるまで持続）。
- 失効は `expire(event)` を **バトル終了**(`resolve_attack`)・**ターン終了**(`end_turn`)で呼ぶ。
  カードが場を離れる際は `move_card` が `drop_for(uuid)` を呼んで当該分を破棄する。
- 参照側: `get_power()`=`timed_power` 加算、`current_cost`=`timed_cost` 加算、
  `has_keyword()`=`current_keywords ∪ timed_keywords`、アタック制限=`timed_flags`。
- **効果無効化**は `FLAG "EFFECTS_DISABLED"`（timed_flags）として付与する。`reset_turn_status`
  でクリアされないため「このターン中」/「次の相手のターン終了時まで」の無効化が途中のアクション
  （戦闘の `target.reset_turn_status` 等）で解除されない。参照側は
  `CardInstance.is_effect_negated`（= `ability_disabled` または timed_flags の `EFFECTS_DISABLED`）。
  能力発動ガード・キーワード判定・除去保護の走査はこのプロパティを見る。

### 除去保護（PREVENT_LEAVE）と置換効果（REPLACE_EFFECT）

`gamestate._active_protection(card, status)` / `_active_replacement(card, status)`。除去が
起こる瞬間に対象の PASSIVE 能力を走査し、条件（例: トラッシュ7枚以上）を
`EffectResolver._check_condition` でその場で評価する（フラグをラッチしない）。

- 保護 `PREVENT_LEAVE`: `status="LEAVE"`（相手の効果で**あらゆる**除去から場を離れない）/
  `"EFFECT_KO"`（**KO 限定**＝「効果でKOされない」。手札に戻す/山札へ送る等の非KO除去には効かない）/
  `"BATTLE_KO"`（バトルでKOされない）。除去ディスパッチ（`apply_action_to_engine`）は KO アクション
  には `("LEAVE","EFFECT_KO")`、非KO除去（bounce/deck/hand 等）には `("LEAVE",)` のみを照合する。
- 置換 `REPLACE_EFFECT`: 「代わりに〜」。実行可能性（`_can_satisfy_node`）も満たせば
  `sub_effect`（置換アクション）を実行し本来の除去をスキップする。同じ `LEAVE`/`BATTLE_KO`
  フックで動作する（保護を先に判定、無ければ置換を判定）。
- 置換 `sub_effect` が対象選択/任意確認で中断した場合は、`_auto_resolve_replacement` が
  同期的に解決する（任意=accept、対象=有効候補を自動選択）。除去解決中に走るため、
  解決前後で外側の `active_interaction` を保全する。

---

## 2. ファイルマップ

### 本番コード

| パス | 役割 |
|---|---|
| `opcg_sim/src/core/effects/parser.py` | レガシーパーサ（構造分解を担当・V2が継承） |
| `opcg_sim/src/core/effects/parser_v2.py` | V2 パーサ。原子句をレジストリ化＋フォールバック記録 |
| `opcg_sim/src/core/effects/rules/base.py` | `Rule`/`RuleRegistry`/`ParseContext`/`@rule` |
| `opcg_sim/src/core/effects/rules/atoms.py` | 原子アクションルール群 |
| `opcg_sim/src/core/effects/continuous.py` | 継続効果マネージャ |
| `opcg_sim/src/core/effects/matcher.py` | 対象指定の解析(`parse_target`)・実体化(`get_target_cards`) |
| `opcg_sim/src/core/effects/resolver.py` | IR の実行（EXECUTE_MAIN_EFFECT 等もここ） |
| `opcg_sim/src/core/gamestate.py` | ゲームエンジン本体（apply_action_to_engine / 除去保護 / 継続効果フック） |
| `opcg_sim/src/models/effect_types.py` | IR 定義（Ability/GameAction/TargetQuery/Condition…）。`GameAction.sub_effect`（置換用）/`GameAction.face_up`（ライフへの向き）/`Ability.cost_optional`（任意コスト） |
| `opcg_sim/src/models/models.py` | CardMaster/CardInstance（`timed_power`/`timed_cost`/`timed_flags`/`timed_keywords`、`has_keyword()`、`is_effect_negated`） |
| `opcg_sim/src/models/enums.py` | ActionType/TriggerType/Zone/ConditionType… |
| `opcg_sim/src/utils/loader.py` | カードDB/デッキ読込・`make_parser()` ファクトリ |

### テスト・ツール

| パス | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサの単体テスト |
| `tests/golden/golden_cases.py` | ゴールデンコーパス（効果セマンティクスの期待 summary） |
| `tests/golden/summarize.py` | AST→指紋(summary) 変換＋部分一致判定 |
| `tests/test_golden.py` | ゴールデン・ランナー（pytest / 単体実行 両対応） |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化テスト |
| `tests/test_realdeck_play.py` | 実カード効果の盤面変化・保護・対話テスト |
| `tests/test_mistarget_guard.py` | ミスターゲット/lift 検出器の回帰ガード（A/B 及び C/D 上限） |
| `tests/test_full_card_audit.py` | 全カード構造不変条件ゲート（EXCEPTION/CARD_LOSS/TEMP_LEAK） |
| `tests/test_full_card_baseline.py` | 全カード挙動ベースライン回帰（`full_card_baseline.json` と比較） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ |
| `tests/effect_diagnostics.py` | 未対応句/OTHER ランキングの可視化 |
| `tests/text_execution_audit.py` | テキスト↔実行不一致の全カード監査（フラグ別） |
| `tests/full_card_audit.py` | 全カード構造不変条件検証＋挙動ベースライン生成（`--regen` で更新） |
| `tests/quality_map.py` | NO_CHANGE/WARN の細分類 |
| `tests/effect_coverage.py` | 全カード実行カバレッジ（SKIP/ERROR/INTERACTIVE/EXECUTED/NO_CHANGE） |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |
| `tests/mistarget_diagnostics.py` | ミスターゲット/lift 候補の検出 |
| `tests/interactive_target_audit.py` | INTERACTIVE 対象の自動監査（TargetQuery とテキストの照合。`--top N`） |
| `tests/leader_spec_probe.py` | リーダー1枚分のテキスト/パース結果(AST要約)/実行観測(classify)を出力（`<ID>`/`--set`/`--all`/`--json`） |
| `tests/leader_test_helpers.py` | リーダー挙動テスト用ヘルパ（実DBの能力を汎用盤面で発動・対話駆動・盤面観測） |
| `tests/test_leader_*.py` | 全137リーダーの挙動テスト（セット別13ファイル）。期待挙動をアサートし、現挙動と異なる項目は xfail でマーク |
| `docs/leader_specs/` | リーダー効果仕様（セット別13本＋README/_GUIDE/_TEST_GUIDE。既知の挙動差異は ISSUES.md） |
| `full_card_baseline.json` | 全能力の実行シグネチャ凍結（挙動ベースライン） |

### フロントエンド（opcg-sim-frontend）

| パス | 役割 |
|---|---|
| `src/game/types.ts` | `BaseCard` に `trigger_text`/`ability_disabled`/`is_frozen` |
| `src/api/types.ts` | `ActionEvent` 型・`GameActionResult.action_events` フィールド |
| `src/api/client.ts` | `sendAction`/`sendBattleAction` の戻り値に `action_events` を含める |
| `src/game/actions.ts` | `useGameAction` の `addEventLog` コールバック |
| `src/layout/layout.config.ts` | `BADGE_FROZEN_*`・`BADGE_NEGATE_*` 色定数 |
| `src/ui/CardRenderer.tsx` | `is_frozen`/`ability_disabled` の Pixi 半透明オーバーレイ |
| `src/ui/CardDetailSheet.tsx` | 状態バッジ（凍結/効果無効）・`trigger_text` ブロック |
| `src/ui/ActionLog.tsx` | 効果解決ログパネル（右上固定・折りたたみ式） |
| `src/ui/EffectToast.tsx` | 効果適用の一時トースト（`action_events` を上部に短時間表示） |
| `src/screens/RealGame.tsx` | `eventLog`/`effectToasts` ステート・`<ActionLog>`/`<EffectToast>` レンダリング・対話 UI |
| `shared_constants.json` | `TRIGGER_TEXT`/`ABILITY_DISABLED`/`IS_FROZEN` を `CARD_PROPERTIES` に追加 |

---

## 3. ルール追加・検証フロー

```bash
# 1) 未対応句/OTHER ランキング・ミスターゲット候補の確認
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40
OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py --top 40

# 2) ゴールデンケースを追加（tests/golden/golden_cases.py に text と期待 summary）
OPCG_LOG_SILENT=1 python tests/test_golden.py

# 3) ルールを追加（opcg_sim/src/core/effects/rules/atoms.py に @rule）
#    エンジン側の実行が必要なら gamestate/resolver も実装し test_effects_engine に検証追加

# 4) 回帰・退行・カバレッジ確認
python -m pytest tests/ -p no:capture -q
OPCG_LOG_SILENT=1 python tests/compare_parsers.py       # レガシー比の新規OTHER（退行）
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py    # 命中率 / OTHER 数
OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py # A/B/C/D 検出器

# 5) 実行カバレッジ・監査
OPCG_LOG_SILENT=1 python tests/effect_coverage.py
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show INTERACTIVE
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --card OP01-001
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --flag DURATION

# 6) 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
OPCG_LOG_SILENT=1 python tests/full_card_audit.py --regen   # 挙動を意図的に変えた場合に更新
```

ルールは `@rule(name, priority)` で関数登録する。`priority` が大きいほど先に試行する。
不一致なら `None`、一致なら `EffectNode`（`GameAction`/`Sequence` 等）を返す。

---

## 4. 運用（環境変数）

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_PARSER` | `v2` | `legacy` でレガシーパーサに切り替える（再デプロイ不要）。V2 読込失敗時も自動でレガシーへ退避する |
| `OPCG_LOG_SILENT` | （未設定） | `1` で stdout ログを抑止（テスト/診断用）。resolver の実行レポートも抑制される |

---

## 5. 実装上の不変条件・注意点

- **本番パスは loader 経由**。効果定義は EffectParserV2 の自動解析に一本化
  （`loader._create_card_master`。旧 catalog.py の手動オーバーライドは廃止済み）。
- **テキスト正規化**: パーサは NFC、loader の DataCleaner は NFKC を使う箇所がある。
  全角/半角・`!!`/`‼`(U+203C)・各種マイナス記号の揺れがある（ルールの正規表現は両対応にする）。
- **pytest の出力キャプチャ**: logger が `sys.stdout` を直接掴むため、`pytest` は
  `-p no:capture` で実行する。`OPCG_LOG_SILENT=1` 併用推奨。
- **`timed_*`（power/cost/flags/keywords）は `reset_turn_status` でクリアしない**。リセット対象に
  追加すると複数ターン跨ぎ効果・付与キーワード・期間付きコストが消える。
- **`_apply_passive_effects` は cost_buff/current_keywords を毎回リセット**する（power_buff/flags は
  しない）。期間付きの COST/KEYWORD は `timed_cost`/`timed_keywords` に載せる（直接 cost_buff/
  current_keywords へ加えると passive 再計算で消える）。INSTANT/PASSIVE のコスト・キーワードは
  reset+reapply で機能する。`_apply_passive_effects` の Step2/3 は `player.stage` を含む。
- **CardMaster は frozen dataclass**。abilities は生成時に確定する。テストで能力を差し替える場合は
  `make_master(..., abilities=(...))` で構築する。
- **新パーサの ActionType は V2 有効時のみ生成**。`OPCG_PARSER=legacy` 時はレガシー解釈に戻る。
- **`gamestate.get_debug_snapshot()` は `CardMaster.card_id` を使う**。`.id` 属性は存在しない。
- **全カード挙動ベースライン `full_card_baseline.json`** は現状の挙動を凍結したもの。挙動を変える
  際は差分をレビューして `full_card_audit.py --regen` で更新する。
- **`parse_target` の対象側(player)判定は期間/タイミング句の「相手の」を除外する**。
  「(次の)相手の(ターン/エンドフェイズ)(終了時)(まで/中)」を player 判定から除去している。
  duration 表現を増やす際は `matcher.py` の除去正規表現に追加する。
- **隠しゾーン（ライフ/デッキ）の対象は通常「上から自動取得」する**（情報リーク防止）。自分の
  ライフ等を明示公開して選ぶ効果は `TargetQuery.flags` に `"REVEAL_SELECT"` を付け、対話選択に切り替える。
- **「持ち主の…」での OPPONENT 補正は「自分の(キャラ/リーダー)」明示を尊重する**（`deck_bottom_general` 等）。
  明示が無いときだけ相手既定にする。
- **ドン付与(ATTACH_DON)の付与先は `parse_target` で解析**し特徴/名前/コスト絞り込みを拾う。
  `is_rest` のみ明示リセットする。
- **自己バフの対象は `atoms._buff_target`**（power_buff/set_power/cost_change）。主語が
  「この(キャラ/リーダー/カード)」なら `SOURCE` を返し、それ以外は `parse_target` に委ねる。
- **置換 sub_effect の中断は `_auto_resolve_replacement` が同期解決する**（任意=accept、対象=自動選択）。
  置換は除去解決の最中（`apply_action_to_engine` 内）に走る。`active_interaction` は単一スロット設計。
- **スコープ付き相手効果無効は `Player.negate_onplay_until`**。「相手の【登場時】効果は無効になる」は
  parser が【登場時】を非タグ化して保全し、`DISABLE_ABILITY status=OPP_ONPLAY` を生成、apply 時に
  相手プレイヤーへ期限(turn_count)を設定、`play_card_action` が期間中の ON_PLAY をスキップする。
  対応スコープは現状【登場時】(ON_PLAY)のみ。
- **`parser._parse_to_node` の split_pattern が Sequence 分割境界を定義する**（`。` / `その後、` /
  連用形の `(?<=置き)、` `(?<=KOし)、` `(?<=追加し)、` 等）。連用接続で後段アクションが同一原子句に
  飲まれる場合はここに境界を追加する。

---

## 6. 参考

- `docs/parser_v2.md` — 設計詳細・ルール一覧
- 全カード監査の起点: `OPCG_LOG_SILENT=1 python tests/text_execution_audit.py` /
  `tests/effect_diagnostics.py --top 40`
- 2デッキ回帰: `python -m pytest tests/test_realdeck_play.py -p no:capture -q`
- 全カード回帰: `python -m pytest tests/test_full_card_baseline.py tests/test_full_card_audit.py -p no:capture -q`

---

---

## 7. 継続効果・パッシブ再計算

- **`passive_power` / `passive_power_override` / `passive_counter` は再計算レイヤ**。
  `_apply_passive_effects` の Step1 が毎回 0/None にリセットして再適用する。即時効果は
  `power_buff` / `base_power_override`（`reset_turn_status` で失効）に載せ、両者を混ぜない。
- **`_apply_passive_effects` は `active_interaction` 中は何もしない**（リセットだけ走って再適用
  できず資産が消えるのを防ぐ）。Step2/3 は `player.stage` も対象に含む。
- **Step2' が OPPONENT_TURN を非ターンプレイヤーのカードに適用**する（cost_buff/passive_power
  再計算レイヤなのでターンが替われば自然に消える）。
- **`refresh_passive_state()` を API アクション境界で呼ぶ**（盤面依存の常在パワー等を即時反映）。
  中断中・`_in_passive_recalc` 中は no-op（多重適用・無限再帰防止）。
- **反応型「…された時」を含む常在は `_is_reactive_passive` で再計算から除外**する（毎回発動防止）。
- **`timed_*`（power/cost/flags/keywords）は `reset_turn_status` でクリアしない**。期間付きの
  COST/KEYWORD は `timed_cost`/`timed_keywords` に載せる（cost_buff/current_keywords へ直接加えると
  passive 再計算で消える）。

## 8. 誘発・対話・コスト

- **誘発は `_pending_triggers` キュー経由**。`move_card` がライフ離脱を `ON_LIFE_DECREASE` として
  積み、API 境界/対話完了/戦闘・効果ダメージ末尾でドレインする（戦闘・効果ダメージは `life.pop`
  済みのため二重計上しない）。
- **【トリガー】（ライフ公開）は `CONFIRM_TRIGGER` で確認**してから解決する。複数枚はキューで保持し
  中断跨ぎで消えない。
- **【ドン‼×N】= `Condition(HAS_DON, value=N, GE)`**。`_check_condition` は `source_card.attached_don`
  を見る（コストエリアの active ドンではない）。本文明示の「ドン!!N枚をレストにする」コストは REST_DON。
- **任意コスト能力は `Ability.cost_optional`**（コスト句に「できる/してもよい」）。自動誘発
  （ON_PLAY/ON_ATTACK/ON_OPP_ATTACK/ON_BLOCK 等）は発動前に `CONFIRM_OPTIONAL` で使用確認する。
  `ACTIVATE_MAIN`（自発起動）は対象外。使用回数は accept 後に消費（拒否では未消費）。
- **ターン内イベント追跡**: `gamestate._turn_events` + `record_turn_event` + `EVENT_THIS_TURN` 条件。
  「〈イベント〉した時、…でなければ発動しない」型の誘発条件を、イベント発生フラグで評価する。
- **文脈依存「N枚につき」は `PREV_ACTION_COUNT` 動的値**（`_last_action_count`＝直前アクションの
  対象枚数）。ドンの増減は targets を介さないため、REST_DON/ACTIVE_DON/RETURN_DON は実処理枚数を
  記録してこの値に使う。「N枚につき+X」の対象実体化スケーリングは `ValueSource.count_query`/`COUNT_QUERY`。
- **遅延効果「ターン終了時、」は `GameAction.delay="TURN_END"`** → `pending_end_of_turn` に積み
  `end_turn` で解決。`end_turn` はターンプレイヤーの TURN_END に加え非ターンプレイヤーの
  OPP_TURN_END も自動発火する（`_fire_turn_end_triggers`）。

## 9. 対象解決・除去保護・置換

- **`parse_target` は主語修飾（特徴/コスト上限/枚数）を保全**する。「相手が選び」等の従属節は
  player 判定から除去して `TargetQuery.chooser` へ。期間/タイミング句の「(次の)相手の(ターン/
  エンドフェイズ)(終了時)(まで/中)」は player 判定から除外する（増やす際は除去正規表現に追加）。
- **隠しゾーン（ライフ/デッキ）の対象は上から自動取得**（情報リーク防止）。明示公開して選ぶ効果は
  `TargetQuery.flags` に `"REVEAL_SELECT"` を付け対話選択に切り替える。
- **「他の／このキャラ以外」→ `EXCLUDE_SOURCE`**（`get_target_cards` がソース自身を除外）。
- **`TargetQuery.zone` はリスト対応**（手札かトラッシュ等）。
- **coreference「そのキャラ／そのカード」**: FIELD 選択を `saved_targets['selected_card']` に
  自動保存し、後続 ref がそれを参照する。ref が未保存なら対象なし。能力内に選択 producer が
  あるか否かで「選択への後方参照」と「文脈的指示（置換被害者・誘発主体）」を静的に切り分ける。
- **除去保護 `PREVENT_LEAVE`**: 期間付きは継続フラグ `PREVENT_<status>`、PASSIVE はマーカーのまま
  除去時に走査（範囲保護はリーダー/フィールド/ステージも走査）。`LEAVE`/`EFFECT_KO` をバケツ分離
  （「効果でKOされない」は KO 限定で、手札戻し等の非KO除去には効かない）。
- **置換 sub_effect は `_auto_resolve_replacement` が同期解決**する（任意=accept、対象=自動選択）。
  置換は除去解決の最中（`apply_action_to_engine` 内）に走る。`active_interaction` は単一スロット設計。
- **効果無効は `is_effect_negated` を参照**する（能力発動ガード・blocker・ON_PLAY 発火・除去保護の
  protector スキップ・deckout 走査）。`NEGATE_EFFECT` は継続効果 `EFFECTS_DISABLED` 化。スコープ付き
  相手効果無効は `Player.negate_onplay_until`（現状【登場時】のみ）。
- **自己制限（self_cannot）は `player.restrictions`（key→{expire, min_cost}）に記録し各地点で enforce**:
  `CANNOT_PLAY_CHARACTER`(min_cost対応)/`CANNOT_PLAY_FROM_HAND`/`CANNOT_ATTACK_LEADER`/
  `CANNOT_DRAW_BY_EFFECT`/`CANNOT_ACTIVATE_DON`/`CANNOT_LIFE_TO_HAND`（`SELF_RESTRICTION_KEYS`）。
  「このターン中」は `turn_count <= expire` の遅延失効。

## 10. 値・符号・その他の解析要点

- 全角符号（＋/－/−/‐）は NFC/NFKC の揺れに両対応する（`_SIGN`/`_to_int`、上限判定）。
- 丸数字コスト（➁/③）は枚数として解釈する（先頭丸数字＋NFKC 分解後の素数字も）。
- パワー/コストの「N以上」「N以下」「NからM」を `power_min`/`power_max`/`cost_min`/`cost_max` に、
  接尾辞なし「コストNの」はちょうど N（cost_min=cost_max=N）に解釈する。
- 「ライフがN枚になるように」は `TargetQuery.count_dynamic="DOWN_TO_N"`。
- 「まで」無しのちょうど N 枚コストは `is_strict_count`（N枚未満では支払えない）。
- temp 回収先は `_temp_origin` 属性（"LIFE"=ライフ上、未設定=デッキトップ）。ライフへの向きは
  `GameAction.face_up`。
- デッキ配置の上下選択・並び替え／ライフ並べ替えは `ARRANGE_DECK` 対話（`status="ARRANGE"`/
  `dest_position`）。ヘッドレス既定（payload 空）では現状順・デッキ下に解決され挙動不変。
- 選択グループ分配（「N枚を選び、1枚を…、残りを…」）は `select_distribute` →
  SELECT(グループ保存)＋`GROUP_FIRST`/`REMAINING`。
- Sequence の分割境界は `_parse_to_node` の split_pattern（`。`/`その後、`/連用形の `(?<=置き)、` 等）。
- 複合条件「Aがいて、Bの場合」「Aを持ち、…」は `AND` に分割する。
- 「〜の代わりに〜」は択一（先行効果の条件を `AND(cond, NOT(後続条件))` に書き換え。`ConditionType.NOT`）。
  ただし置換（「される/離れる代わりに」= `REPLACE_EFFECT`）は別経路。
- 「（このリーダー/キャラが）バトルしている場合」は `SOURCE_STATE "IN_BATTLE"`（`active_battle` 参照）。
- DECLARE_COST の相手デッキトップ公開は resume フックで行う（AST に LOOK は無い）。

## 11. リーダー効果仕様とテスト

- `docs/leader_specs/` にセット別仕様（基本／効果テキスト／期待挙動／テストケース）。索引は
  `docs/leader_specs/README.md`、テスト方針・ヘルパ API は `_TEST_GUIDE.md`。
- 挙動テストは `tests/test_leader_*.py`（`tests/leader_test_helpers.py` の盤面構築・対話駆動・観測 API）。
  期待挙動をアサートし、現挙動と異なる項目は `@pytest.mark.xfail` で表現する。
- 現在の既知の挙動差異は `docs/leader_specs/ISSUES.md`。

## 12. 監査・品質ゲート

| ツール | 役割 | 合格条件 |
|---|---|---|
| `tests/full_card_audit.py` | 全カード構造不変条件＋挙動シグネチャ生成（`--regen` で `full_card_baseline.json` 更新） | EXCEPTION / CARD_LOSS / TEMP_LEAK = 0 |
| `tests/test_full_card_baseline.py` | 挙動ベースライン回帰 | `full_card_baseline.json` と一致 |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分 | 新規 OTHER（退行）= 0 |
| `tests/test_quality_gates.py` | NO_CHANGE/WARN/SELECT_MISMATCH 等のラチェット | 設定閾値以内 |
| `tests/interactive_target_audit.py` | INTERACTIVE 対象と TargetQuery/テキストの照合 | 疑い 0 |
| `tests/condition_synth.py` / `tests/battle_coverage.py` / `tests/effect_coverage.py` | 条件合成発動・戦闘発火・実行カバレッジ | ERROR 0 |

挙動を変更したら、差分をレビューのうえ `full_card_audit.py --regen` でベースラインを更新し、
上記ゲートを通す。`condition_synth` の合成盤面は実評価器（`_check_condition`/`_can_satisfy_node`）で
再検証する。

## 13. フロントエンド UI の要点（RealGame）

- **アクションボタンの表示可否は `getAvailableActions` に一元化**する（CardDetailSheet と
  CardActionMenu の双方に反映。location だけで判定するとステージに攻撃/ドン付与が出る）。
- **ミニメニューは `gameState` / `pendingRequest.request_id` の変化で自動クローズ**する（盤面再構築で
  アンカー座標が古くなるため。攻撃ターゲティング開始時も閉じる）。
- **ライフの横向き描画はレンダリング用コピー `{ ...c, is_rest: true }`** を使い、onClick・詳細シートには
  元のカードオブジェクトを渡す。
- **`life[0]` が山の一番上**（バックエンドはダメージ時 `life.pop(0)`、HEAL は append）。BoardSide は
  逆順 addChild で `life[0]` を最前面・最上段に描画する。裏向きライフは `eventMode='none'` でタップ無効。
- **`Player.to_dict` はライフを `is_face_up` でシリアライズ**する。裏向きライフ/相手手札のカード識別
  情報（name/card_id/text）は送信されるため、対人戦対応時はマスキングの検討が必要。
- 並び替えモード（`maxSelect<0`）は Pointer Events のドラッグ&ドロップ（`setPointerCapture`＋6px 閾値＋
  矩形ヒットテスト、`touch-action: none`）。`CardSelectModal` の `allowPosition` で上下配置を確定する。
