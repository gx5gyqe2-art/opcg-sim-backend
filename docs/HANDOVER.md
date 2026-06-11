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
[ catalog.py 手動定義があれば優先 ] ─ なければ ─▶ [ EffectParserV2 ]
   │                                                   │ 構造分解(レガシー流用)
   │                                                   │ + 原子句のみ rules で解釈
   │                                                   │ + 未対応はレガシーへフォールバック
   ▼                                                   ▼
   └──────────────▶  Ability(IR) ◀──────────────────────┘
                        │ trigger / condition / cost / effect
                        ▼  ゲーム中、該当タイミングで
                  resolver.py（EffectResolver）
                        │  AST を実行スタックで処理（対象選択は中断/再開）
                        ▼
                  gamestate.py（apply_action_to_engine / continuous / 除去保護）
                        ▼
                     盤面更新
```

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

### 除去保護（PREVENT_LEAVE）と置換効果（REPLACE_EFFECT）

`gamestate._active_protection(card, status)` / `_active_replacement(card, status)`。除去が
起こる瞬間に対象の PASSIVE 能力を走査し、条件（例: トラッシュ7枚以上）を
`EffectResolver._check_condition` でその場で評価する（フラグをラッチしない）。

- 保護 `PREVENT_LEAVE`: `status="LEAVE"`（相手の効果で場を離れない）/ `"BATTLE_KO"`
  （バトルでKOされない）。
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
| `opcg_sim/src/core/effects/catalog.py` | 手動オーバーライド(MANUAL_EFFECTS) |
| `opcg_sim/src/core/gamestate.py` | ゲームエンジン本体（apply_action_to_engine / 除去保護 / 継続効果フック） |
| `opcg_sim/src/models/effect_types.py` | IR 定義（Ability/GameAction/TargetQuery/Condition…）。`GameAction.sub_effect`（置換用） |
| `opcg_sim/src/models/models.py` | CardMaster/CardInstance（`timed_power`/`timed_cost`/`timed_flags`/`timed_keywords`、`has_keyword()`） |
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
| `tests/test_realdeck_play.py` | 実デッキ(imu/nami)での盤面変化・保護・対話テスト |
| `tests/test_gameplay_smoke.py` | 実デッキでのゲーム進行スモーク |
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

- **本番パスは loader 経由**。catalog(手動定義) > parser(V2) の優先順位（`loader._create_card_master`）。
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

## 7. 2026-06 カード効果再現性向上の変更点（card-effect-bugs ブランチ）

### 修正した根本原因（コード実証済み）

| # | 根本原因 | 主な修正箇所 |
|---|---|---|
| RC-1 | 全角符号（＋/－/−/‐）が NFC で畳まれず `[+-]` 正規表現が不一致 | atoms.py `_SIGN`/`_to_int`、matcher.py 上限判定 |
| RC-2 | 制限/付与系ルールが対象をハードコードし主語修飾（特徴/コスト上限/枚数）を破棄 | atoms.py `_subject_target`、gamestate `_active_protection`（範囲保護の走査＋期間付き保護フラグ） |
| RC-3 | 「相手が選び」等の従属節が対象側判定を汚染 | matcher.py 除去リスト＋`TargetQuery.chooser`（選択者指定） |
| RC-4 | 「N枚につき+X」がフラット値になる | `ValueSource.count_query`＋`COUNT_QUERY` 動的値（毎回実体化して数える） |
| RC-6 | 「ライフがN枚になるように」が「N枚だけ」になる | `TargetQuery.count_dynamic="DOWN_TO_N"` |
| - | **PASSIVE バフが再計算のたびに累積**（+1000 が際限なく増える） | `CardInstance.passive_power`/`passive_power_override`/`passive_counter`（再計算レイヤ） |
| - | **対話中断中の再計算がバフを消す**（リセットだけ走り再適用が中断ガードで空振り） | `_apply_passive_effects` 冒頭で中断中は skip |
| - | 無タグ反応型「…KOされた時」が PASSIVE 扱いで毎回発動 | `_detect_trigger` の ON_KO/ON_ATTACK 写像＋`_is_reactive_passive` ガード |
| - | コスト上限修飾「ライフの…枚数以下の」がゾーン検出を汚染（KOの代わりにライフを墓地送り） | matcher.py ゾーン検出も除去後テキストで実施 |
| - | 「このカードの【登場時】効果を発動する」が常に ACTIVATE_MAIN を展開 | 参照タグの非タグ化保全＋`_expand_main_effect(ref_trigger)` |
| - | 複数タグのみのセグメント（【自分のターン中】【登場時】/）が本体を共有しない | parser.py セグメント共有の複数タグ対応 |
| - | ライフ公開→条件付き登場が no-op（FACE_UP のみで TEMP 未経由） | LOOK_LIFE 経由＋`_temp_origin="LIFE"` 回収 |
| - | 中断→再開経路で `save_id` 保存がスキップ | resolver `_resolve_targets` 再開パス |

### 新しい不変条件

- **`passive_power` / `passive_power_override` / `passive_counter` は再計算レイヤ**。
  `_apply_passive_effects` Step1 が毎回リセットして再適用する。即時効果は従来どおり
  `power_buff` / `base_power_override`（reset_turn_status で失効）へ。両者を混ぜないこと。
- **`_apply_passive_effects` は `active_interaction` 中に呼ばれても何もしない**。
  リセットだけ走って再適用できないため（資産消失防止）。
- **PREVENT_LEAVE は期間付きなら継続フラグ `PREVENT_<status>`**（timed_flags）として付与される。
  PASSIVE はマーカーのまま除去時走査（範囲保護はリーダー/フィールド/ステージも走査される）。
- **temp 回収先は `_temp_origin` 属性で決まる**（"LIFE"=ライフ上、未設定=デッキトップ）。
- **DECLARE_COST の相手デッキトップ公開は resume フックで行う**（AST に LOOK は無い。
  mistarget 検出器 C は DECLARE_COST 保持カードを除外済み）。

### 監査ハーネスの強化（tests/effect_coverage.py）

- H-1: ステータス差分測定（power/cost/keywords/flags/rest/don 構成）
- H-2: ON_PLAY の登場アーティファクト控除（プレイ自体の盤面変化を除外）
- H-3: CHOICE 全パス列挙（上限8）
- H-4: SELECT_TARGET 候補のテキスト照合（SELECT_MISMATCH）
- `tests/test_quality_gates.py`: ラチェット式ゲート
  （WARN_DIRECTION=0 / STAT_ONLY=0 / NO_IMPL=0 / SELECT_MISMATCH≤2 / フォールバック=0）

### 既知の残課題（優先順）

1. **mistarget D 残り4枚**（OP10-058/OP10-022/OP08-118/OP06-086）:
   「2枚までを選び、1枚を…、残りを…」の選択グループ分配（選択結果の REMAINING 参照）が未実装。
2. **OPPONENT_TURN / TURN_END 系トリガーの実プレイ配線**: resolve_ability 直呼びでは動くが、
   ターン進行からの自動発火経路の網羅検証は未実施（Phase 4 H-6 バトルシナリオ網羅で対応予定）。
3. **ドン付与の相手プール**（OP15-015）: 「相手のレストのドン‼を付与」が自分のドンを使う。
4. **遅延効果**（OP03-005 サッチ / OP13-024 ゴードン）: 「このターン終了時、…」が即時実行される。
5. **文脈依存の「N枚につき」**（捨てたカード1枚につき等）: 直前アクションの結果数参照が未実装
   （フラット値のまま）。
6. **「他の「X」」の自己除外**（EB02-018 バギー）: 「自分のキャラの他の「バギー」がいない」が
   ソース自身を数えてしまう。
7. **二重制約/複数ゾーンの対象**（EB03-049/OP06-086/OP13-079）: 「コスト6以下とコスト4以下を
   1枚ずつ」「手札か場の…」の Choice/Sequence 分配が部分対応（interactive_target_audit に残存）。

---

## 8. 2026-06 Phase 4/5（深層ハーネス＋検出分の修正）

### 追加した検出ハーネス

| ツール | 役割 | 結果 |
|---|---|---|
| `tests/condition_synth.py` (H-5) | 条件/コストを満たす盤面を合成して発動・分類。実評価器で再検証し合成漏れを除外 | 490 未検証能力 → 1233 実行確認 / SATISFIED_NO_CHANGE 9 |
| `tests/battle_coverage.py` (H-6) | declare_attack→handle_block→counter を駆動し ON_ATTACK/ON_BLOCK/ON_OPP_ATTACK/COUNTER を実戦発火 | 494 発火 / ERROR 0 |
| `effect_coverage._zone_fingerprint`/`_moved` | 枚数で相殺されるグロスのカード移動を検出（ドロー+手札→デッキ cost 等） | 偽 NO_CHANGE を排除 |

### Phase 5 で修正した実バグ

- **ON_BLOCK 未発火**（14枚）: `handle_block` が【ブロック時】能力を発動していなかった。
- **複合条件「Aがいて、Bの場合」の誤読**（~9枚）: 単一条件＋最初の数値で誤分類していた
  （例:「コスト8以上のキャラがいて、手札6枚以下」→ `HAND_COUNT>=8`）。`_parse_condition_obj`
  が連結部で2分割して `AND` を構成するようにした。

### 新しい不変条件・運用

- **`passive_*`（passive_power / passive_power_override / passive_counter）は再計算レイヤ**。
  即時効果の `power_buff`/`base_power_override`/`cost_buff` とは別で、`_apply_passive_effects` が
  毎回 0/None にリセットして再適用する。
- **H-4〜H-7 のゲートは `tests/test_quality_gates.py`**: SATISFIED_NO_CHANGE≤9 /
  BATTLE_NO_CHANGE=0 / battle ERROR=0 / interactive_audit≤11 をラチェット固定。
- **`condition_synth` の合成盤面は実評価器（`_check_condition`/`_can_satisfy_node`）で再検証する**。
  合成しきれない条件型（DON_COUNT_COMPARE/PREV_ACTION/色フィルタ等）は UNHANDLED に落とし、
  真バグ候補（SATISFIED_NO_CHANGE）に混ぜない。
