# 引き継ぎ資料 — カード効果システム刷新

最終更新: 2026-06-10 / ブランチ: `claude/handoff-docs-review-o0zx7t`

このドキュメントは、本リポジトリ（opcg-sim-backend）の **カード効果処理の刷新作業** を
引き継ぐための資料です。詳細な設計は `docs/parser_v2.md` を、本書はその上位の
オリエンテーション（全体像・運用・残タスク）を担います。

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

## 2. 現在の状態（main ブランチ時点）

| 指標 | 刷新開始時 | **現在** |
|---|---|---|
| 原子句カバレッジ（ルール命中率） | 0% | **98.3%** |
| `ActionType.OTHER`（実行時に何もしない句） | 942 | **58** |
| 未分類条件 `GENERIC`（誤発動の温床） | — | **1** |
| パーサルール数 | 0 | **58+** |
| テスト総数 | 17 | **210（全緑）** |
| 本番パーサ | レガシー | **EffectParserV2（既定）** |

- 全2652カードの能力構築・実デッキ(imu/nami)ロード・ゲーム開始〜数ターン進行を確認済み。
- レガシー vs V2 の全カード比較で **退行(新規OTHER)=0** を一貫して維持。
- 隠れミスターゲット detector: A=0 / B=0 / C≤8 / D≤8。

### 実行カバレッジ（`tests/effect_coverage.py` による全カード走査、2026-06-10 時点）

| 分類 | 件数 | 意味 |
|---|---|---|
| SKIP | 325 | 能力なし |
| **ERROR** | **67** | 例外発生 → エンジン修正が必要（最優先） |
| **INTERACTIVE** | **385** | 手動テスト必須リスト（対象選択が残留） |
| EXECUTED | 2271 | 盤面変化確認済み（うち WARN=292：方向不一致の疑い） |
| NO_CHANGE | 429 | 条件未達 or OTHER の疑い |

WARN はほとんどが「モーダル/分岐の副作用で別パスが実行された」ケースで誤アラートが多い。
本当に壊れているカードの特定には `--card <ID>` で個別確認を推奨。

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
| `tests/golden/golden_cases.py` | **ゴールデンコーパス（効果セマンティクスの期待値, 88件）** |
| `tests/golden/summarize.py` | AST→指紋(summary) 変換＋部分一致判定 |
| `tests/test_golden.py` | ゴールデン・ランナー（79件） |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化テスト（67件） |
| `tests/test_gameplay_smoke.py` | 実デッキでのゲーム進行スモーク（2件） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ |
| `tests/effect_diagnostics.py` | **未対応句/OTHER ランキングの可視化** |
| `tests/mistarget_diagnostics.py` | **隠れミスターゲット/lift 不具合の検出（OTHER に出ない不具合）** |
| `tests/test_mistarget_guard.py` | 上記の回帰ガード（A/B detector=0 固定・C/D 上限） |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |
| `tests/effect_coverage.py` | **全カード実行カバレッジ**。全トリガー×全カードを GameManager 上で発動し SKIP/ERROR/INTERACTIVE/EXECUTED/NO_CHANGE に分類。手動テスト優先リストの生成と自動方向性検証（`_soft_assert`）を担う |

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
| `src/screens/RealGame.tsx` | `eventLog` ステート・`<ActionLog>` レンダリング・overlay 改善 |
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
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider
OPCG_LOG_SILENT=1 python tests/compare_parsers.py      # 退行(新規OTHER)=0 を維持
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py   # 命中率↑/OTHER↓
OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py # 隠れ不具合 A/B=0・C/D↓ を確認

# 5) 実行カバレッジで ERROR/INTERACTIVE を確認（修正サイクルの確認に使う）
OPCG_LOG_SILENT=1 python tests/effect_coverage.py                        # 全体サマリ
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show ERROR           # 例外一覧
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show INTERACTIVE     # 手動テスト優先リスト
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --card OP01-001        # 1枚を詳細確認
OPCG_LOG_SILENT=1 python tests/effect_coverage.py --trigger ON_PLAY      # トリガー絞り込み
```

ルールは `@rule(name, priority)` で関数登録。`priority` が大きいほど先に試行
（具体的・限定的なルールを高く）。不一致なら `None`、一致なら `EffectNode` を返す。

---

## 6. 運用（環境変数）

| 環境変数 | 既定 | 用途 |
|---|---|---|
| `OPCG_PARSER` | `v2` | `legacy` でレガシーパーサへ**即ロールバック**（再デプロイ不要） |
| `OPCG_LOG_SILENT` | （未設定） | `1` で stdout ログ抑止（テスト/診断用。バッファ蓄積は維持） |

**ロールバック手順**: 本番で問題が出たら Cloud Run の環境変数に `OPCG_PARSER=legacy` を
設定するだけ。V2 読込失敗時も自動でレガシーへ退避する（フェイルセーフ）。

---

## 7. 既知の課題・残タスク（優先度順）

### タスク概要（A〜G カテゴリ）

| カテゴリ | 内容 | 状態 |
|---|---|---|
| **A** | 隠れミスターゲット横展開（C detector 残 8枚） | A3 残 |
| B | デッキ公開→登場クラスタの残フォロー | B4 残 |
| C | 専用メカニクス・難所（要設計） | C7残・C8・C9・C10 残 |
| D | 裾野 OTHER 継続 burn down（現状 **OTHER=58**） | 継続中 |
| E | 置換効果（REPLACE_EFFECT）の残 | E14・E15 残 |
| F | アーキテクチャ整理（catalog 縮退） | 未着手 |
| G | 回帰防止（C/D detector 上限の随時更新） | C≤8/D≤8 に更新済み |

---

### A. 隠れミスターゲット横展開

**背景**: `OTHER` には現れないが**意味的に壊れている**不具合が2類型存在する:
- **ミスターゲット型**: レガシーフォールバック句が ActionType は正しく出すが**対象 zone を文言依存で誤推定**（実行はされるが盤面が誤る）。
- **条件 lift 型**: インライン条件が先頭ゲート条件 lift でアビリティ条件へ引き上げられ、**前段の LOOK が消失**し順序矛盾。

いずれも `effect_diagnostics.py`（OTHER カウント）や `compare_parsers.py`（新規OTHER検知）では**捕捉できない**。
検出ツール: `tests/mistarget_diagnostics.py`（A/B/C/D の4 detector）。回帰ガード: `tests/test_mistarget_guard.py`（A/B=0 固定・C/D 上限確認）。

**A3 コスト宣言メカニクス**（C detector 8枚中大半: OP11系6枚）
  - 「任意のコストを宣言し、相手のデッキの上から1枚を公開する」— 専用 ActionType が必要な難所（→ C8 参照）。

> 着手サイクル: `mistarget_diagnostics.py --top 40` で標的選定 → golden 追加（赤）→ ルール/エンジン是正（緑）→ `test_mistarget_guard.py` の上限を新実測値に下げて固定 → commit。

---

### B. デッキ公開→登場クラスタの残フォロー

**B4 ライフ公開系**（OP10-022/ST13-007/010/014, 4枚）
  - 「自分のライフの上から1枚を公開し、…の場合、登場させてもよい」。登場元が**ライフ**のためライフ→TEMP 公開と未登場カードのライフ戻しが要る。C7「ライフ look-and-place」と同系統でエンジン拡張が必要。

---

### C. 専用メカニクス・難所（要設計）

単純なルール追加では解決しない、設計が必要な残件:

**C7 ライフ look-and-place（残）**
  - 完了: 「自分か相手のライフの上から1枚を見て、ライフの上か下に置く」（8カード）→ `LOOK_LIFE` + 対話選択 `Choice`。
  - **残**: ライフ全見て並び替え（「好きな順番で置く」）・デッキ→ライフ追加の連用形断片・ライフ公開→登場(B4)・公開→パワー参照。

**C8 コスト宣言メカニクス**（OP11系6枚+）
  - 「任意のコストを宣言し、相手のデッキの上から1枚を公開する」。ゲーム独自の「宣言」インタラクション＋専用 ActionType の設計が必要。

**C9 「パワーが相手と同じになる」**（3件）
  - 動的な対象パワー参照（`base_power_override` をゲーム中に動的評価）。未実装。

**C10 「自分は敗北する代わりに勝利する」**（2件）
  - 勝敗条件の置換。ゲームエンジン全体を跨ぐ設計変更が必要。

---

### D. 裾野 OTHER 継続

**D12 OTHERランキング継続 burn down**（現状 **OTHER=58**）
  - `effect_diagnostics.py` 起点で継続。残るものは構造的難所・専用メカニクスが中心（単純ルール追加では解きにくい）。
  - 対応済み（代表）: 自己トラッシュ/自己アクティブ/ステージレスト/mill/残りトラッシュ/bounce/deck_bottom/play_card_from_zone/reveal_hand/サーチ・scry 構造分解/FREEZE/NEGATE_EFFECT/自己制限/hand_to_deck/play_revealed/attack_active/trash_to_hand/rest_self/set_power/trash_target/ライフ scry/select断片/trigger断片/レスト制限/モーダル選択。

---

### E. 置換効果（REPLACE_EFFECT）の残

自己型・保護者型（`_active_replacement` がフィールド全体をスキャン）まで実装済み。残件:

**E14 置換実行が対象選択で中断する場合の挙動検証**
  - 置換 `sub_effect` の実行中に `_suspend_for_target_selection` が起動する場合（複数候補の捨て札等）の処理を要確認。

**E15 任意（「できる」）の選択 UI**
  - 「代わりに〜できる」形の置換（プレイヤーが選択）の UI は未提供。現状: 取れるなら実行。

---

### F. アーキテクチャ整理

**F16 catalog 縮退**
  - `opcg_sim/src/core/effects/catalog.py` の `MANUAL_EFFECTS`（13枚）。
  - パーサが賢くなった分、1枚ずつ golden で検証しながら削れる（parser で正しく生成できれば手動定義を削除）。

---

### G. 回帰防止（継続）

**G19 C/D detector 上限の随時更新**
  - `tests/test_mistarget_guard.py` の `assert len(...) <= N`（現在: C≤8 / D≤8）。
  - 横展開是正でカードが減ったら上限値を新しい実測値に下げて固定する。

---

## 8. 注意点・落とし穴

- **本番パスは loader 経由**。catalog(手動定義) > parser(V2) の優先順位（`loader._create_card_master`）。
- **テキスト正規化**: パーサは NFC、loader の DataCleaner は NFKC を使う箇所がある。
  全角/半角・`!!`/`‼`(U+203C)・各種マイナス記号の揺れに注意（ルールの正規表現は両対応にする）。
- **pytest の出力キャプチャ**: logger が `sys.stdout` を直接掴むため、`pytest` は
  `-s`（キャプチャ無効）で実行する。`OPCG_LOG_SILENT=1` 併用推奨。
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
- **`OPCG_LOG_SILENT=1` は resolver の実行レポートも抑制する**。`resolver.py` の
  `_log_execution_report` / `_log_failure_snapshot` は `OPCG_LOG_SILENT` 未設定時に
  stdout へ JSON ブロックを出力する（AI 向けのデバッグログ）。診断スクリプト実行時は
  必ず `OPCG_LOG_SILENT=1` を付けること。
- **`gamestate.get_debug_snapshot()` は `CardMaster.card_id` を使う**。`.id` 属性は存在しない
  （2026-06-10 修正済み）。フィラーカード等カスタム CardMaster を使うテストでも影響なし。

---

## 9. 設計判断と根拠（なぜこの方式か）

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

- `docs/parser_v2.md` — 設計詳細・ルール一覧・現況・残課題
- 計測の起点コマンド: `OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40`
