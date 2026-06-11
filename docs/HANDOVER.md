# 引き継ぎ資料 — カード効果システム刷新

最終更新: 2026-06-11 / ブランチ: `claude/handoff-materials-review-46flvn`

このドキュメントは `opcg-sim-backend` の **カード効果処理システム** を引き継ぐための資料です。
設計詳細は `docs/parser_v2.md`、本書はその上位のオリエンテーション（全体像・運用・残タスク）を担います。

---

## 1. 背景と目的

**課題**: ゲーム中にカード効果が想定通り実行されない場合が多数あった。

**原因（診断で判明）**:
1. 旧 `parser.py` は巨大な if 連鎖で原子句を解釈しており、順序依存・サイレント失敗・
   テスト困難という構造的課題があった。
2. パーサが `ActionType.OTHER`（=解析できても実行系が無く何もしない）に落ちる句が
   **約940件**存在した。
3. エンジン側（`gamestate.apply_action_to_engine` / `resolver`）に未実装のアクションが多く、
   正しい型を出しても盤面が変わらなかった。
4. 「このバトル中」「次の相手のターン終了時まで」等の**期間付き効果**を管理する機構が無かった。

**対応方針**: 中間表現(IR)とインターフェース(`parse_card_text`)を維持したまま、
日本語→IR 変換を**合成ルールレジストリ方式**へ刷新し、エンジン実行系を拡充。
段階的・非破壊で移行し、最終的に新パーサ(V2)を本番有効化した。

---

## 2. 現在の状態

| 指標 | 刷新開始時 | **現在** |
|---|---|---|
| 原子句カバレッジ（ルール命中率） | 0% | **99%超** |
| `ActionType.OTHER`（実行時に何もしない句） | 942 | **6** |
| 未分類条件 `GENERIC`（誤発動の温床） | — | **1** |
| atoms.py ルール数 | 0 | **83** |
| テスト総数 | 17 | **284（全緑）** |
| 本番パーサ | レガシー | **EffectParserV2（既定）** |

### 監査フラグ（`tests/text_execution_audit.py` による全カード検証）

| フラグ | 件数 | 意味 |
|---|---|---|
| FLAG_OTHER | **11** | 未実装句（裾野の専用効果が残存） |
| FLAG_HIDDEN_LEAK | **0** | 隠しゾーン情報リーク（全解決済み） |
| FLAG_DURATION | **0** | 期間不一致（全解決済み） |
| FLAG_COST_LIMIT | **0** | 動的コスト上限未設定（全解決済み） |
| FLAG_TARGET_SIDE | **0** | 対象プレイヤー逆（全解決済み） |
| FLAG_MISSING_ACTION | **1** | 動詞に対応アクション無し（複雑な分岐効果） |

### 構造不変条件（`tests/full_card_audit.py` による全2652枚検証）

| 不変条件 | 件数 |
|---|---|
| EXCEPTION（例外発生） | **0** |
| CARD_LOSS（カード消失） | **0** |
| TEMP_LEAK（tempリーク） | **0** |

### 実行カバレッジ（`tests/effect_coverage.py` による全カード走査）

| 分類 | 件数 | 意味 |
|---|---|---|
| SKIP | 325 | 能力なし |
| ERROR | **0** | 例外発生（全解決済み） |
| INTERACTIVE | 459 | 対象選択が必要（`interactive_target_audit.py` で対象の正しさを自動監査済み） |
| EXECUTED | 2241 | 盤面変化確認済み |
| NO_CHANGE | 452 | 条件未達/コスト不成立/PASSIVE/測定限界（実バグ=0確認済み） |

---

## 3. アーキテクチャ全体像

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

### V2 の設計思想（最重要）

`EffectParserV2` は `EffectParser`(レガシー) を**継承**し、`_parse_atomic_action()` だけを
オーバーライドする。トリガー判定・コスト分離・逐次/分岐/選択肢の構造分解はレガシーをそのまま使う。

- 原子句は `default_registry.apply(ctx)` でルール優先解釈
- どのルールも当たらなければ **レガシー実装にフォールバック**し、その句を `unmatched` に記録
- → 本番は決して壊れない。未対応表現は診断で可視化され、ルール追加で burn down できる

### 継続効果（期間付き効果）

`effects/continuous.py` の `ContinuousEffectManager`。

- `CardInstance` の専用フィールド `timed_power` / `timed_cost` / `timed_flags` /
  `timed_keywords` に反映。**これらは `reset_turn_status()` でクリアされない**
  （ターン境界を跨いで存続する鍵）。既存の `power_buff`/`cost_buff`/`flags`/`current_keywords`
  （ターン境界 or passive 再計算でリセット）とは独立で衝突しない。
- kind: `POWER` / `COST` / `FLAG` / `KEYWORD`。Duration: `THIS_BATTLE` / `THIS_TURN` /
  `UNTIL_NEXT_TURN_END` / `PERMANENT`（場を離れるまで持続）。
- 失効は `expire(event)` を **バトル終了**(`resolve_attack`)・**ターン終了**(`end_turn`)で呼ぶ。
  カードが場を離れる際は `move_card` が `drop_for(uuid)` を呼び、その分を破棄する。
- 参照側: `get_power()`=`timed_power` 加算、`current_cost`=`timed_cost` 加算、
  `has_keyword()`=`current_keywords ∪ timed_keywords`、アタック制限=`timed_flags`。

### 除去保護（PREVENT_LEAVE）と置換効果（REPLACE_EFFECT）

`gamestate._active_protection(card, status)` / `_active_replacement(card, status)`。除去が
起こる瞬間に対象の PASSIVE 能力を走査し、条件（例: トラッシュ7枚以上）を
`EffectResolver._check_condition` で**ライブ評価**する（フラグをラッチしないので条件変動に追随）。
- 保護 `PREVENT_LEAVE`: `status="LEAVE"`（相手の効果で場を離れない）/ `"BATTLE_KO"`
  （バトルでKOされない）。
- 置換 `REPLACE_EFFECT`: 「代わりに〜」。実行可能性（`_can_satisfy_node`）も満たせば
  `sub_effect`（置換アクション）を実行し本来の除去をスキップ。同じ `LEAVE`/`BATTLE_KO`
  フックに相乗り（保護を先に判定、無ければ置換を判定）。

---

## 4. ファイルマップ

### 本番コード

| パス | 役割 |
|---|---|
| `opcg_sim/src/core/effects/parser.py` | レガシーパーサ（構造分解を担当・V2が継承） |
| `opcg_sim/src/core/effects/parser_v2.py` | **新パーサ**。原子句をレジストリ化＋フォールバック記録 |
| `opcg_sim/src/core/effects/rules/base.py` | `Rule`/`RuleRegistry`/`ParseContext`/`@rule` |
| `opcg_sim/src/core/effects/rules/atoms.py` | **原子アクションルール群（ここを育てる）** |
| `opcg_sim/src/core/effects/continuous.py` | 継続効果マネージャ |
| `opcg_sim/src/core/effects/matcher.py` | 対象指定の解析(`parse_target`)・実体化(`get_target_cards`) |
| `opcg_sim/src/core/effects/resolver.py` | IR の実行（EXECUTE_MAIN_EFFECT 等もここ） |
| `opcg_sim/src/core/effects/catalog.py` | 手動オーバーライド(MANUAL_EFFECTS, 13枚) |
| `opcg_sim/src/core/gamestate.py` | ゲームエンジン本体（apply_action_to_engine / 除去保護 / 継続効果フック） |
| `opcg_sim/src/models/effect_types.py` | IR 定義（Ability/GameAction/TargetQuery/Condition…）。`GameAction.sub_effect`（置換用） |
| `opcg_sim/src/models/models.py` | CardMaster/CardInstance（`timed_power`/`timed_cost`/`timed_flags`/`timed_keywords`、`has_keyword()`） |
| `opcg_sim/src/models/enums.py` | ActionType/TriggerType/Zone… |
| `opcg_sim/src/utils/loader.py` | カードDB/デッキ読込・`make_parser()` ファクトリ |

### テスト・ツール

| パス | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサの単体テスト（8件） |
| `tests/golden/golden_cases.py` | **ゴールデンコーパス（効果セマンティクスの期待値, ~124件）** |
| `tests/golden/summarize.py` | AST→指紋(summary) 変換＋部分一致判定 |
| `tests/test_golden.py` | ゴールデン・ランナー（pytest / 単体実行 両対応） |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化テスト |
| `tests/test_realdeck_play.py` | 実デッキ(imu/nami)での盤面変化・保護・対話テスト |
| `tests/test_gameplay_smoke.py` | 実デッキでのゲーム進行スモーク |
| `tests/test_mistarget_guard.py` | 隠れミスターゲット/lift 不具合の回帰ガード（A/B=0・C/D 上限） |
| `tests/test_full_card_audit.py` | 全カード構造不変条件ゲート（EXCEPTION/CARD_LOSS/TEMP_LEAK=0） |
| `tests/test_full_card_baseline.py` | 全カード挙動ベースライン回帰（`full_card_baseline.json` と比較） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ |
| `tests/effect_diagnostics.py` | **未対応句/OTHER ランキングの可視化** |
| `tests/text_execution_audit.py` | **テキスト↔実行不一致の全カード監査**（フラグ別ランキング） |
| `tests/full_card_audit.py` | **全カード構造不変条件検証＋挙動ベースライン生成**（`--regen` で更新） |
| `tests/quality_map.py` | NO_CHANGE/WARN の細分類（真のバグ=0 確認済み） |
| `tests/effect_coverage.py` | 全カード実行カバレッジ（SKIP/ERROR/INTERACTIVE/EXECUTED/NO_CHANGE） |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |
| `tests/mistarget_diagnostics.py` | 隠れミスターゲット/lift 不具合の検出 |
| `tests/interactive_target_audit.py` | **INTERACTIVE 対象の自動監査**（解釈済み TargetQuery をテキストと照合し、対象側/コスト上限/枚数/特徴の不一致候補を検出。`--top N`） |
| `full_card_baseline.json` | 全3152能力の実行シグネチャ凍結（挙動ベースライン） |

### フロントエンド（opcg-sim-frontend）

| パス | 役割 |
|---|---|
| `src/game/types.ts` | `BaseCard` に `trigger_text`/`ability_disabled`/`is_frozen` を追加 |
| `src/api/types.ts` | `ActionEvent` 型・`GameActionResult.action_events` フィールド |
| `src/api/client.ts` | `sendAction`/`sendBattleAction` の戻り値に `action_events` を含める |
| `src/game/actions.ts` | `useGameAction` に `addEventLog` コールバックを追加 |
| `src/layout/layout.config.ts` | `BADGE_FROZEN_BG/CSS`・`BADGE_NEGATE_BG/CSS` 色定数を追加 |
| `src/ui/CardRenderer.tsx` | `is_frozen`/`ability_disabled` の Pixi 半透明オーバーレイを追加 |
| `src/ui/CardDetailSheet.tsx` | 状態バッジ（凍結/効果無効）・`trigger_text` ブロックを追加 |
| `src/ui/ActionLog.tsx` | **効果解決ログパネル**（新規。右上固定・折りたたみ式） |
| `src/screens/RealGame.tsx` | `eventLog` ステート・`<ActionLog>` レンダリング・対話 UI |
| `shared_constants.json` | `TRIGGER_TEXT`/`ABILITY_DISABLED`/`IS_FROZEN` を `CARD_PROPERTIES` に追加 |

---

## 5. 開発フロー（ルール追加 TDD サイクル）

```bash
# 1) 標的を選ぶ（OTHERランキング上位＝効果が動かない直接原因）
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40
#    OTHER に出ない隠れ不具合（ミスターゲット/lift）はこちらで棚卸し
OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py --top 40

# 2) ゴールデンケースを追加して赤にする
#    tests/golden/golden_cases.py に text と期待 summary を書く
OPCG_LOG_SILENT=1 python tests/test_golden.py

# 3) ルールを足して緑にする
#    opcg_sim/src/core/effects/rules/atoms.py に @rule を1つ追加
#    （エンジン側の実行が必要なら gamestate/resolver も実装し test_effects_engine に検証追加）

# 4) 回帰・退行・カバレッジ確認
python -m pytest tests/ -p no:capture -q
OPCG_LOG_SILENT=1 python tests/compare_parsers.py      # 退行(新規OTHER)=0 を維持
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py   # 命中率↑/OTHER↓
OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py # 隠れ不具合 A/B=0・C/D↓ を確認

# 5) 実行カバレッジ・監査で状況確認
OPCG_LOG_SILENT=1 python tests/effect_coverage.py                        # 全体サマリ
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show INTERACTIVE     # 手動テスト優先リスト
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --card OP01-001        # 1枚を詳細確認
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py                   # フラグ別集計
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --flag DURATION   # 個別フラグ詳細

# 6) 全カード構造不変条件・挙動ベースラインの確認
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
#    意図的な挙動改善後はベースラインを更新する
OPCG_LOG_SILENT=1 python tests/full_card_audit.py --regen
```

ルールは `@rule(name, priority)` で関数登録。`priority` が大きいほど先に試行
（具体的・限定的なルールを高く）。不一致なら `None`、一致なら `GameAction` を返す。

---

## 6. 運用（環境変数）

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_PARSER` | `v2` | `legacy` でレガシーパーサへ**即ロールバック**（再デプロイ不要） |
| `OPCG_LOG_SILENT` | （未設定） | `1` で stdout ログ抑止（テスト/診断用。バッファ蓄積は維持） |

**ロールバック手順**: 本番で問題が出たら Cloud Run の環境変数に `OPCG_PARSER=legacy` を
設定するだけ。V2 読込失敗時も自動でレガシーへ退避する（フェイルセーフ）。

---

## 7. 残タスク（優先度順）

### A. OTHER 6件の裾野 burn down（継続）

`effect_diagnostics.py` 起点で継続。**方針: catalog は使わず parser/エンジン拡張で対応**（2026-06-11）。

- ✅ **済（parser/エンジン+テスト）**: ライフ並び替え(ORDER_LIFE) / イベント発動(EXECUTE_EVENT) /
  効果ダメージ / 相手・自分デッキ閲覧 / 複合除去保護 / 除外フィルタ / ドン複合コスト(REST_DON) /
  勝利宣言(VICTORY) / 手札全戻し / 活用形・「てもよい」 / 「任意の枚数」可変選択 / 「KOしてもよい」 /
  **共有対象二択(「加えるか登場」, MISSING 3→1)** / **REDIRECT_ATTACK** / **MOVE_ATTACHED_DON** /
  **レスト登場(RESTED_PLAY)** / **登場制限(NO_EFFECT_PLAY)**
- ✅ **追加で済**: ライフ→デッキ上(LIFE_TO_DECK) / サーチ結果をライフへ(TEMP→LIFE) /
  丸数字コスト①➀(REST_DON)
- **残6件（各1カードの深い構造/エンジン作業 or 見送り。詳細は下記）**:
  - OP07-042「代わりに〜できる」任意置換: **E14/E15（見送り中）**
  - OP05-100「この効果は無効になる」: **実質no-op（OTHER=何もしない=正しい）**
  - OP11-103: コスト節内の条件「リーダーが「しらほし」の場合」→ ability 条件への抽出（構造）
  - OP15-119: 新トリガー「相手がイベント/ブロッカーを発動した時」（構造）
  - OP06-086: dual-tier（コスト4以下と2以下を1枚ずつ選び登場）= 二重選択
  - OP15-092: トラッシュ枚数で段階効果（条件パッシブの Sequence-of-Branch）
  - OP09-081: 「自分/相手の【登場時】効果は無効になる」= scoped 効果無効（範囲修飾・要慎重設計）

### B. MISSING_ACTION 3件（複雑な分岐パターン）

「〜するか、〜する」形式の二択は `parser._parse_suruka_choice` で Choice 化済み
（動詞終止形 u 段かな直後の「か、」を境界に分割。OP05-096 等が解決）。残る3件は
「以下から1つを…」でも「するか」でもない不規則な分岐（例: 数値条件で分岐する複合効果）で、
個別の構造解析が必要。`text_execution_audit.py --flag MISSING_ACTION` で棚卸しする。

### C. 置換効果（REPLACE_EFFECT）の残 — E14/E15【見送り中（2026-06-11 判断）】

**判断の根拠**: (1) 現状の card DB に「複数選択を伴う複雑な置換」カードが実質存在しない、
(2) `GameManager.active_interaction` が**単一 continuation のみ保持する設計**で、置換が他効果の
解決中に発生する「ネストした中断」を表現できない（核アーキテクチャ改修が必要）。受益カードが
無い段階での高リスク改修を避け、該当カード出現時に着手する方針。

**E14 置換実行が対象選択で中断する場合の挙動**
- 置換 `sub_effect` の実行中に `_suspend_for_target_selection` が起動する場合
  （例: 「代わりに手札2枚を選んで捨てる」）。現状は `_active_replacement` が同期 auto-execute。
  → REPLACE_EFFECT をスタックに積み直し、`active_interaction` を呼び出し側へ伝播させる設計変更。
  ネストした中断は単一 continuation 設計の拡張（continuation スタック化）が前提。

**E15 任意（「できる」）の選択 UI**
- 「代わりに〜できる」形の置換の yes/no UI。E14 の suspend/resume 実装後に
  既存 `_suspend_for_optional_confirmation` でほぼ自動サポート。

### D. INTERACTIVE 71件の手動検証

`effect_coverage.py --show INTERACTIVE` で対象カードを確認し、実プレイで意味的正しさを検証する。
自動テストでは検証できない「テキスト通りの対象/数値か」はこのルートで詰める。

### E. 隠れミスターゲット（C/D detector）

`tests/test_mistarget_guard.py` の上限（現在 C≤8 / D≤8）。
横展開是正でカードが減ったら上限値を新しい実測値に下げて固定する。

### F. フロント lint 削減（現状 137件: error 129 / warning 8）

- ✅ 安全分は処理済み: 未使用 catch バインディング6件→`catch {}`、eslint に `argsIgnorePattern '^_'`
  追加、`logger.ts` の `any`→`unknown`（lint 148→137、ビルド緑維持）
- **`@typescript-eslint/no-explicit-any` 94件**: `SandboxGame.tsx`/`localLogic.ts`/`BoardSide.tsx`/
  `DeckBuilder.tsx`/`RealGame.tsx` 等。ドメイン型（CardInstance vs CardData、API レスポンス
  スキーマ）の設計判断が必要なため**人間レビュー必須**。`as any` キャストと API デコード周りが中心。
- **`react-hooks/exhaustive-deps` 8件**（warning）: 依存配列追加で再レンダリング挙動が変わり得る
  ものを含む（WebSocket 接続・初期化 useEffect 等）。closure 挙動を確認してから個別対応。

### G. 効果適用アニメーション（視覚QA 必須）

- KO/バウンス/ドローの視覚フィードバック追加
- フロント: `src/screens/RealGame.tsx` の action_events を消費してアニメ発火
- ブラウザ確認が必須。自動テスト不可。

---

## 8. 注意点・落とし穴

- **本番パスは loader 経由**。catalog(手動定義) > parser(V2) の優先順位（`loader._create_card_master`）。
- **テキスト正規化**: パーサは NFC、loader の DataCleaner は NFKC を使う箇所がある。
  全角/半角・`!!`/`‼`(U+203C)・各種マイナス記号の揺れに注意（ルールの正規表現は両対応にする）。
- **pytest の出力キャプチャ**: logger が `sys.stdout` を直接掴むため、`pytest` は
  `-p no:capture` で実行する。`OPCG_LOG_SILENT=1` 併用推奨。
- **`timed_*`（power/cost/flags/keywords）は reset_turn_status でクリアしない**設計。ここを
  「リセット対象に追加」してしまうと複数ターン跨ぎ効果・付与キーワード・期間付きコストが壊れる。
- **`_apply_passive_effects` は cost_buff/current_keywords を毎回リセット**する（power_buff/flags
  はしない）。期間付きの COST/KEYWORD はこのリセットを避けるため `timed_cost`/`timed_keywords`
  に載せている（直接 cost_buff/current_keywords へ加えると即消える）。INSTANT/PASSIVE の
  コスト・キーワードは従来どおり reset+reapply で機能する。
- **CardMaster は frozen dataclass**。abilities は生成時に確定。テストで能力を差し替える
  場合は `make_master(..., abilities=(...))` で構築する。
- **新パーサの効果は V2 有効化後にのみ反映**。`OPCG_PARSER=legacy` 時はレガシー解釈に戻る
  （= 新 ActionType は生成されない）。
- **`OPCG_LOG_SILENT=1` は resolver の実行レポートも抑制する**。診断スクリプト実行時は
  必ず `OPCG_LOG_SILENT=1` を付けること。
- **`gamestate.get_debug_snapshot()` は `CardMaster.card_id` を使う**。`.id` 属性は存在しない。
- **`_apply_passive_effects` の Step2/3 は `player.stage` を含む**。ステージ効果（コスト軽減等）は
  正しく再計算される（かつて除外されていたバグは修正済み）。
- **全カード挙動ベースライン `full_card_baseline.json`** は「現状の挙動」を凍結したもの。
  バグ修正で挙動が変わる際は差分をレビューして `full_card_audit.py --regen` で更新する。
- **`parse_target` の対象側(player)判定は期間/タイミング句の「相手の」に注意**。「次の相手のターン
  終了時まで」等の duration を player 判定から除外している（しないと「自分のリーダーを…パワー+N」が
  OPPONENT 強化になる）。新しい duration 表現を増やす際は `matcher.py` の除去正規表現に追加する。
- **隠しゾーン（ライフ/デッキ）からの「見て選ぶ」は `TargetQuery.flags` に `"REVEAL_SELECT"` を付ける**。
  resolver は通常ライフ/デッキを「上から自動取得」する（情報リーク防止）が、自分のライフを明示公開して
  選ぶ効果（「ライフすべてを見て1枚をデッキ上に置く」等）はこの flag で対話選択に切り替える。
- **「持ち主の…」での OPPONENT 補正は「自分の(キャラ/リーダー)」明示を尊重する**こと
  （`deck_bottom_general` 等）。明示が無いときだけ相手既定にする。
- **ドン付与(ATTACH_DON)の付与先は `parse_target` で解析**し特徴/名前/コスト絞り込みを拾う
  （手動構築すると filter が脱落する）。`is_rest` 漏れだけ明示リセットする。

### 直近のミスターゲット是正（2026-06-11, `interactive_target_audit.py` 起点）
- 「自分のキャラを持ち主のデッキの下に置く」が相手対象になる不具合（deck_bottom）を修正。
- 「次の相手のターン終了時まで」で自己バフ(パワー/コスト)が相手強化になる不具合（parse_target）を修正。
- ドン付与の付与先フィルタ脱落を修正。
- ライフ→デッキの「見て選ぶ」を自動取得から対話選択(`REVEAL_SELECT`)へ修正。

---

## 9. 設計判断と根拠

- **合成ルールレジストリ（vs 形式文法/構造化データ主導）**: 実カードテキストは
  半構造的で揺れが大きく、形式文法は脆く段階移行しづらい。ルールレジストリは
  「未対応はレガシーへフォールバック」で非破壊・漸進的に移行でき、各ルールが単体テスト可能。
- **IR/インターフェース維持**: resolver/gamestate を無改修にでき、リスクと差分を最小化。
- **継続効果を専用フィールドで実装（vs 既存 power_buff/flags 流用）**: 既存はターン境界で
  リセットされるため複数ターン跨ぎ効果と衝突する。専用フィールド＋イベント失効で
  reapply 不要のクリーンな設計にした。
- **除去保護をライブ評価（vs フラグのラッチ）**: 条件（トラッシュ枚数等）が変動するため、
  除去の瞬間に評価する方が正確。

---

## 10. 参考

- `docs/parser_v2.md` — 設計詳細・ルール一覧・現況
- 計測の起点: `OPCG_LOG_SILENT=1 python tests/text_execution_audit.py`（不一致監査）/
  `tests/effect_diagnostics.py --top 40`（OTHER）
- 2デッキ回帰: `python -m pytest tests/test_realdeck_play.py -p no:capture -q`
- 全カード回帰: `python -m pytest tests/test_full_card_baseline.py tests/test_full_card_audit.py -p no:capture -q`
