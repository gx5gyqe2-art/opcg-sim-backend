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

### 既知の残課題（優先順）→ 2026-06 解消

1. ✅ **選択グループ分配**（OP08-118 等）: 「N枚を選び、1枚を…、残りを…」を `select_distribute`
   ルールで SELECT(グループ保存)＋GROUP_FIRST/REMAINING に分解。resolver がグループの先頭/残余を
   参照する（field 分配）。OP06-086/OP10-058 は二ティア/公開(TEMP)経由で近似。
2. ✅ **OPPONENT_TURN / TURN_END 系トリガーの実プレイ配線**: `end_turn` を `_fire_turn_end_triggers`
   に分離し、ターンプレイヤーの TURN_END に加え非ターンプレイヤーの **OPP_TURN_END** も自動発火する。
3. ✅ **ドン付与の相手プール**（OP15-015）: `don_attach` がドン枚数句直前の「相手の」を検出し
   status に "OPP" を付与、エンジンが相手のドンプールから付与する（status `RESTED_OPP`）。
4. ✅ **遅延効果**（OP03-005 / OP13-024）: `GameAction.delay="TURN_END"`。parser_v2 が「ターン終了時、/に」
   を遅延マークし、resolver が `pending_end_of_turn` に積み、`end_turn` で解決する。
5. ✅ **文脈依存の「N枚につき」**（捨てたカード1枚につき等）: `PREV_ACTION_COUNT` 動的値。resolver が
   `_last_action_count`(直前アクションの対象枚数)を記録し、`get_dynamic_value` が参照する。
6. ✅ **「他の「X」」の自己除外**（EB02-018 等）: matcher が「他の／このキャラ以外」で `EXCLUDE_SOURCE`
   フラグを立て、`get_target_cards` がソース自身を候補から除外する。
7. ✅ **二重制約/複数ゾーンの対象**（EB03-049/OP03-096/OP13-079）: `TargetQuery.zone` のリスト対応
   （手札かトラッシュ等）、`dual_tier_play_from_trash` が特徴/名前/ゾーンを両ティアで共有、
   `_parse_target_alternative_choice` が「AかB、<動詞>」を制約別 Choice に分解。

> 監査 `interactive_target_audit` は raw_text 共有の兄弟（Choice/二択/二ティア/自己コスト）を集約
> 判定する精度改善で誤検知を排し、ラチェットを **0** に締結（`test_quality_gates`）。

### 残る近似・未対応（軽微）

- ドン付与済み枚数依存「付与されているドン1枚につき」・「カード名の異なる」系の動的値は未対応
  （対象固有/名前集合の別機構が必要）。
- OP06-086 の「コスト4以下と2以下を1枚ずつ選び1枚登場・残りレスト」は二ティア＋REMAINING で
  近似（厳密な選択集合分配ではない）。

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

---

## 9. 2026-06 RealGame UI 改善（realgame-ui-improvements ブランチ）

対戦画面（RealGame）の UI/UX 改善。backend PR #23 / frontend PR #23 で main へマージ済み。

### Backend の変更（1行）

- `gamestate.py` `Player.to_dict`: ライフを `_format_card(c, c.is_face_up)` でシリアライズ
  （従来は常に `False` 固定で、FACE_UP_LIFE で表になったライフをクライアントが判別できなかった）。
- **情報リーク注記**: `_format_card` は `is_face_up` を上書きするのみで、裏向きライフ/相手手札の
  カード識別情報（name/card_id/text）は従来から全送信されている。本変更でリーク範囲は拡大しないが、
  対人戦対応時はマスキングの検討が必要（trigger/candidates フローへの影響に注意）。

### Frontend の変更

| パス | 変更内容 |
|---|---|
| `src/game/cardTypes.ts` | **新規**。`normalizeCardType` — カード種別の英/日表記（'STAGE'/'ステージ' 等）を正規化 |
| `src/game/cardActions.ts` | **新規**。`getAvailableActions(card, location, isMyTurn, activeDonCount)` — 実行可能アクション（登場/攻撃/ドン付与/効果起動）の一元判定。攻撃・ドン付与はリーダー/キャラのみ、起動メインは種別不問 |
| `src/ui/CardActionMenu.tsx` | **新規**。カードタップ時にカード近傍へ出すミニアクションメニュー（画面端クランプ・上半分タップは下に/下半分は上に表示・ドン付与は枚数ステッパー内蔵） |
| `src/ui/CardDetailSheet.tsx` | `renderButtons` を `getAvailableActions` ベースに書き換え（ステージに攻撃/ドン付与が出るバグの修正箇所） |
| `src/ui/BoardSide.tsx` | ライフを仮想1枚+枚数バッジから **個別カードの横向き重ね表示** に変更（90°回転は `is_rest: true` のレンダリング用コピーで実現）。両陣営に適用。枚数バッジ併記 |
| `src/ui/CardRenderer.tsx` | `options.onClick` のシグネチャを `(pos: {x,y}) => void` に変更（pointertap の `e.global` を渡す。autoDensity+全画面キャンバスのため CSS px と一致） |
| `src/screens/RealGame.tsx` | `actionMenu` ステート追加。操作可能カードはミニメニュー、操作不可カード（相手/ライフ/トラッシュ等）は従来どおり直接詳細シート |
| `src/ui/CardSelectModal.tsx` | 並び替えモード（`maxSelect < 0`）の小さい↑↓ボタンを廃止し、Pointer Events の **ドラッグ&ドロップ並び替え** に置換（追加依存なし） |
| `src/layout/layout.config.ts` | `Z_INDEX.MINI_MENU: 1500` 追加（NOTIFICATION/OVERLAY より上、SHEET より下） |

### 実装上の不変条件・注意点（UI）

- **アクションボタンの表示可否は `getAvailableActions` に一元化**。新アクションを追加する場合は
  ここに足すこと（CardDetailSheet と CardActionMenu の双方に反映される）。location だけで判定する
  実装に戻すとステージ攻撃バグが再発する。
- **ミニメニューは `gameState` / `pendingRequest.request_id` の変化で必ず自動クローズ**する
  （RealGame の useEffect）。PIXI 盤面は状態変化のたびに全再構築されカード位置が変わるため、
  アンカー座標が古くなるのを防ぐ。攻撃ターゲティング開始時（handleAction ATTACK 分岐）も閉じる。
- **ライフの横向き描画はレンダリング用コピー `{ ...c, is_rest: true }`** を使う。onClick・詳細シートには
  **元のカードオブジェクト**を渡すこと（コピーを渡すとレスト状態が誤表示される）。
- **`life[0]` が山の一番上**（バックエンドはダメージ時 `life.pop(0)`、HEAL は append）。BoardSide は
  逆順 addChild で `life[0]` を最前面・最上段に描画する。
- **裏向きライフは `eventMode = 'none'` でタップ無効**。表向きのみタップ → 既存フロー（location 'life'
  → isOperatable=false → 詳細シート直行）。
- **`createCardContainer` の onClick は座標を受け取る**。座標不要の呼び出し元（Sandbox 等）は
  引数を無視するラムダでよい。
- **並び替えモードのドラッグ**は `setPointerCapture` + 6px 閾値 + `getBoundingClientRect` の矩形
  ヒットテストで `selected`（=配置順）を splice 移動する。グリッドは `selected` 順で描画されるため
  ライブ並べ替え自体が挿入位置のフィードバックになる。アイテムは `touch-action: none`（タッチ対応）。

### 残課題（UI）

- ライフ重ね間隔・ミニメニュー幅などの微調整は実機目視が未実施（型チェック/lint/バックエンド
  テスト 342 件は通過済み）。
- 対人戦対応時: 裏向きライフ/相手手札のカード識別情報マスキング（上記リーク注記）。

---

## 10. 2026-06 対話化と自己制限のエンフォース

### 課題3(3a): 自己制限（self_cannot）のエンフォース

「自分は、このターン中、…できない」を従来の `RULE_PROCESSING` no-op から実エンフォースへ。
parser(`self_cannot`)が述語を制限キーへ写像し、`apply_action_to_engine` が `player.restrictions`
（key→{expire, min_cost}）に記録、各地点で enforce する。`gamestate.SELF_RESTRICTION_KEYS`：

| キー | 述語 | enforce 地点 |
|---|---|---|
| `CANNOT_PLAY_CHARACTER`(min_cost対応) | キャラ（コストN以上）を登場できない | `play_card_action` |
| `CANNOT_PLAY_FROM_HAND` | 手札からカードをプレイできない | `play_card_action` |
| `CANNOT_ATTACK_LEADER` | リーダーにアタックできない | `declare_attack` |
| `CANNOT_DRAW_BY_EFFECT` | 自分の効果でカードを引けない | `DRAW` |
| `CANNOT_ACTIVATE_DON` | キャラの効果でドン‼をアクティブにできない | `ACTIVE_DON` |
| `CANNOT_LIFE_TO_HAND` | 自分の効果でライフを手札に加えられない | `MOVE_CARD`(life→hand) |

- 「このターン中」= 現ターンのみ有効（`turn_count <= expire` の遅延失効。`negate_onplay_until` と同方式）。
- `attack_disable` から「自分は…リーダーにアタックできない」を除外し `self_cannot` に委譲。
- 述語を判別できない自己制限（「デッキに入れられない」=構築ルール）は従来どおり no-op。
- テスト: `tests/test_self_cannot.py`(16)。

### 課題2(2a/2b): デッキ配置の上下選択・並び替え／ライフ並べ替えの対話化

「好きな順番でデッキの上か下に置く」「ライフすべてを見て好きな順番で置く」を、従来の
現状順・デッキ下固定から **ARRANGE_DECK 対話**へ。

- **parser**: `temp_to_deck`/`remaining_deck_bottom`/`remaining_deck_top_or_bottom`/`hand_to_deck` に
  `status="ARRANGE"`（順序選択）と `dest_position`（"TOP"/"BOTTOM"/"CHOOSE"）を付与。
- **resolver**: `_maybe_suspend_arrange`/`_suspend_for_arrange` が DECK_BOTTOM(ARRANGE/CHOOSE) と
  ORDER_LIFE を中断（`action_type="ARRANGE_DECK"`, `allow_reorder`/`allow_position`）。
  `_resolve_targets` は REMAINING を「残り全部」として選択中断せず、ARRANGE_DECK 一本に集約。
- **gamestate**: `resolve_interaction` の `ARRANGE_DECK` 分岐で順序適用＋上/下配置（TOP は逆順
  insert で先頭が最上面）、ライフ再整列。DECK_BOTTOM ハンドラが `dest_position` を尊重。
- **frontend**: `CardSelectModal` に `allowPosition`（「デッキの上へ/下へ」確定ボタン）を追加、
  既存の DnD 並び替えモード(`maxSelect<0`)と併用。RealGame の `showSearchModal`/`handleSelectionResolve`
  が ARRANGE_DECK を処理し `{selected_uuids(配置順), position}` を送信。
- **不変条件**: ヘッドレス(`_smart_drain` の既定 payload=selected_uuids 空/position 無し)では
  現状順・デッキ下に解決され**挙動不変**（baseline/品質ゲート維持）。テスト: `tests/test_arrange_deck.py`(7)。

### 残課題（この回で未着手）

- 課題2(2c): 置換効果（`_auto_resolve_replacement`）の対話化は高リスクのため見送り（現状は自動解決）。
- バックエンド全 365 passed・品質ゲート緑。フロントは tsc/eslint 通過、実機目視は未実施。
