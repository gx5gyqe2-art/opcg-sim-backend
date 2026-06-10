# 引き継ぎ資料 — カード効果システム刷新

最終更新: 2026-06-10 / ブランチ: `claude/system-issues-review-45ftlh`

> **§15 セッション追記（全カード効果再現に向けた監査 burn-down ＋ 設計案件 C8/C9/C10）**
> を末尾に追加。監査フラグ COST_LIMIT 15→0 / DURATION 48→24 / TARGET_SIDE 32→16 /
> OTHER 57→50。C8（コスト宣言・数値入力UI込み）・C9（同値パワー・スナップショット）・
> C10（敗北→勝利の置換）を実装。詳細は §15 を参照。

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

> **更新（§11 セッション後）**: 下表の ERROR=67 は **0 になった**（任意コスト不成立で
> 例外を投げていたのが主因。§11 参照）。

| 分類 | 件数 | 意味 |
|---|---|---|
| SKIP | 325 | 能力なし |
| **ERROR** | **67 → 0** | 例外発生 → エンジン修正が必要（最優先）。§11 で解消 |
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

## 11. 本セッションの修正（realgame 効果処理バグ・実プレイ2デッキ対応）

ブランチ `claude/realgame-effect-processing-bugs-h98p18`。「realgame で効果がテキスト通りに
動かない」課題に対し、**IR レベルの指標(98.3%)では表面化しない実行レベルのバグ** を
実際にカードを発動させて発見・修正した。スコープは合意のうえ **実際にロードされる2デッキ
(`imu.json` / `nami.json`、29ユニーク＋リーダー2) を end-to-end で正しく** ＋ **フロントの
対話処理を網羅的に**。

### 重要な前提（落とし穴の追加）

- **実行カバレッジの ERROR は「任意コスト不成立で例外」が主因**だった。OPCG の `：` の前の
  コスト句は常に任意で、払えなければ能力が発生しないだけ。例外にすると ON_PLAY 任意コストを
  持つカードを出した瞬間にゲームが落ちる。→ `resolver.resolve_ability` を **スキップ**に変更。
- **ゾーン枚数だけの検証では net-zero スワップ（ライフ⇔手札）やパワー/コスト/キーワード変化を
  見逃す**。検証は **カード identity・power・keyword・temp リーク**まで見ること。
- **EVENT カードの効果が対象選択で中断する場合、resume 時に `_find_card_by_uuid` で見つかる
  必要がある**（＝発動中はまだ手札にあること）。テストで event を素の `resolve_ability` に
  渡すと resume できないので、`play_card_action` 経由 or 手札に置いて検証する。

### バックエンドの修正（全227テスト緑・退行(新規OTHER)=0 維持）

| # | 不具合 | 修正箇所 |
|---|---|---|
| 1 | **任意(できる)コスト不成立で `ValueError`** → ゲームが落ちる | `resolver.py resolve_ability`（raise→return）。ERROR 67→0 |
| 2 | **マリガン導入でゲームループ破綻**（start_game後 end_turn が期待アクション不一致で例外） | `test_gameplay_smoke.py`（マリガン確定を挟む）。API/`do_mulligan`は既存で正常 |
| 3 | **STAGE の YOUR_TURN/PASSIVE が一切発動しない** | `gamestate._apply_passive_effects` の Step2/3 が `player.stage` を除外していた。聖地マリージョア(コスト軽減)・虚の玉座(リーダー+1000) |
| 4 | **五老星の全体トラッシュ不発**（`自分のキャラすべてをトラッシュに置く`→OTHER） | `rules/atoms.py trash_target` が節末 `…トラッシュに`(動詞欠落)も拾うよう拡張 |
| 5 | **自己レスト複合コストの欠落**（`このキャラをレストにし、<追加コスト>` で REST 以外が破棄） | `parser.py _parse_cost_node` が残りも解析し Sequence 化。お玉・ヒナ |
| 6 | **除去保護の脱落**（`場を離れず、【X】を得る` が GRANT_KEYWORD のみ生成） | `rules/atoms.py` に `prevent_leave_and_keyword`(p70) 追加＋`prevent_leave` が「離れず」も拾う。`gamestate._active_protection/_active_replacement` を効果ツリー走査(`_find_action`)に。トラッシュ7枚以上の五老星(ナス寿郎/ウォーキュリー/マーズ) |
| 7 | **共有トリガーの効果欠落**（`【メイン】/【カウンター】<効果>` で ACTIVATE_MAIN 側が None） | `parser.py parse_card_text` が本体なし先頭トリガータグに次セグメントの本体を共有。焔裂き |
| 8 | **DEAL_DAMAGE 未実装**（`相手にNダメージ` が no-op） | `gamestate.apply_action_to_engine` に DEAL_DAMAGE 実装（ライフ→手札・トリガー発火・勝敗）。ニコ・ロビン |

- 検証: `tests/test_realdeck_play.py`（**新規・17件**）に 2デッキの主要効果の盤面/パワー/
  キーワード/保護をアサート。`effect_coverage` の ERROR=0 を確認。`compare_parsers` 退行=0。

### フロントエンドの修正（`opcg-sim-frontend`・ビルド通過）

`src/screens/RealGame.tsx`。バックエンドが出す全 `pending_request` 種別を UI で解決可能に。

- **ブロッカー選択不能（致命的）**: `isBoardSelectMode` に `SELECT_BLOCKER` を追加。従来
  「ブロッカーを選択」要求で盤面カードがクリックできず**パスしかできない＝ブロック不能**だった。
- **攻撃対象の未検証**: 攻撃対象を相手のリーダー/フィールド/ステージに限定。無効カードクリックで
  サーバエラー＆ターゲティング復帰不能だったのを防止。
- **ターゲティング競合**: 新しい選択要求の到着時に攻撃ターゲティング状態を解除（二重表示防止）。

### 残課題（次の担当へ）

- **2デッキ以外の全カード**は未横断（本セッションは2デッキに限定）。`effect_coverage --card <ID>`
  起点で同じ手順（identity/power/keyword 込みで検証）を横展開する。
- フロント: `ORDER_CARDS`（「好きな順番でデッキの下に置く」）は現状エンジンが順不同で自動配置。
  厳密な並び替え UI は未実装（戦略影響が小さいため後回し）。効果適用の視覚フィードバック・
  `as any` の型整理（既存 lint エラー）も残。
- WARN 付き EXECUTED(約312) は大半が誤検知だが、2デッキ外の本当の方向不一致は要個別確認。

---

## 13. 監査ハーネス起点の体系的修正（本番テストで判明したバグ）

「1枚ずつ直す whack-a-mole では収束しない」課題に対し、**テキスト↔実行の不一致を全カードで
自動検出する監査ハーネス**を導入し、根本原因カテゴリを一括修正した。

### 監査ハーネス `tests/text_execution_audit.py`（新規・検出基盤）

全2652カード／2デッキで AST とテキストを突き合わせ、フラグ別に不一致をランキング出力する。
```bash
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py            # 全体集計
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --deck imu nami
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --flag DURATION   # 個別フラグ詳細
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --card OP11-041
```
フラグ: `FLAG_OTHER`（未実装句）/`FLAG_HIDDEN_LEAK`（隠しゾーンの位置指定を選択させて中身が
見える・**実行時計測**）/`FLAG_DURATION`（期間不一致）/`FLAG_COST_LIMIT`（動的コスト未制限）/
`FLAG_TARGET_SIDE`（対象 player 逆）/`FLAG_MISSING_ACTION`（動詞に対応アクション無し・advisory）。
ベースライン→現在: HIDDEN_LEAK **86→0** / COST_LIMIT 20→15 / OTHER 57 / DURATION 49→48 / TARGET_SIDE 32。

### 根本原因カテゴリ修正

| # | 不具合（報告） | 根本原因と修正 |
|---|---|---|
| **致命** | **ブロッカーが一度も発動できない** | `loader` が `master.keywords` を一切設定せず `has_keyword('ブロッカー')` が常に False → `has_blocker` 常に False → BLOCK_STEP に入らない。`loader._extract_static_keywords` で effect_text から【ブロッカー/速攻/ダブルアタック/バニッシュ】を抽出（「を得る/発動できない/持つ」は除外）。 |
| C1 | EB03-055 等が隠しゾーンの中身を見て選べる | `resolver._resolve_targets`: DECK/LIFE 直接ターゲットは中断せず**上から count 枚を位置指定で自動取得**（対話は LOOK→TEMP 経由のみ）。情報リークを全カード一括撲滅。 |
| C2 | OP11-041 の「ターン中」がバトル中限定に | catalog の BUFF に `duration=THIS_TURN` 付与 ＋ engine が THIS_TURN/UNTIL_NEXT_TURN_END のパワー増減も継続効果(`timed_power`)に載せる（被攻撃リーダーの `reset_turn_status` で消えない）。 |
| C3 | OP13-099 がコスト無制限で登場 | `matcher.parse_target` が「（場の）ドン!!の枚数以下のコスト」→ `cost_max_dynamic=DON_COUNT_FIELD`。 |
| C4 | カウンターでエラー（`期待:CHOICE`） | `declare_attack` が ON_ATTACK/ON_OPP_ATTACK トリガーの中断を無視して防御フェイズへ進み、未解決 interaction と SELECT_COUNTER が衝突。トリガーを待ち行列で順次解決し、全解決後にフェイズ遷移（`_advance_battle_triggers`、`resolve_interaction` から再開）。 |

### フロント（`opcg-sim-frontend`）

- `isBoardSelectMode` に `SELECT_BLOCKER` 追加（守備側がブロッカーをクリックで選べる）。
- 攻撃対象を相手リーダー/フィールド/ステージに限定・攻撃ターゲティングと選択の競合解消。
- 守備側（手番でない側）の選択時に「🛡 P2 の防御選択」を明示。

### 検証

- `tests/test_realdeck_play.py` を 24件に拡充（ブロッカー keyword/flow・カウンター衝突・
  duration 存続・動的コスト・隠しゾーン非peek 等）。全 **232 緑**・退行(新規OTHER)=0。
- 監査フラグは `text_execution_audit.py` で継続計測（残りは長い裾野＝2デッキ外の多様な専用効果）。

### 残課題（次の担当へ）

- 監査の DURATION(48)/COST_LIMIT(15)/TARGET_SIDE(32)/OTHER(57) の多くは2デッキ外。
  同じ手順（監査でフラグ→カテゴリ修正→回帰）で burn down する。
- C4 と同型の「能力ループ中に suspend しても止まらない」構造は `play_card_action`＋
  `_apply_passive_effects`・`end_turn` にも残る（declare_attack のみ対応済み）。
- フロント: ORDER_CARDS（厳密な並び替え）・効果適用アニメ・PWA キャッシュ更新導線。

---

## 14. 参考

- `docs/parser_v2.md` — 設計詳細・ルール一覧・現況・残課題
- 計測の起点: `OPCG_LOG_SILENT=1 python tests/text_execution_audit.py`（不一致監査）/
  `tests/effect_diagnostics.py --top 40`（OTHER）
- 2デッキ回帰: `OPCG_LOG_SILENT=1 python -m pytest tests/test_realdeck_play.py -q -s -p no:cacheprovider`

---

## 15. 本セッションの修正（全カード効果再現に向けた監査 burn-down ＋ 設計案件）

ブランチ `claude/system-issues-review-45ftlh`。「全カードの効果再現」を目標に、監査ハーネス
(`text_execution_audit.py`)のフラグをカテゴリ単位で burn-down し、設計が必要な難所
(C8/C9/C10)を実装した。全テスト **232→244 緑**、退行(新規OTHER)=0、A/B=0・C/D≤8 を維持。

### 監査フラグの推移

| フラグ | 開始 | 現在 | 主因と対応 |
|---|---|---|---|
| COST_LIMIT | 15 | **0** | ライフ枚数依存の動的コスト上限（LIFE_COUNT_OPPONENT/SELF/BOTH） |
| DURATION | 48 | **24** | 連用形チェーン分割・コスト0セット(COST_OVERRIDE)・自己制限の期間保持・C9 |
| TARGET_SIDE | 32 | **16** | 相手除去＋自己バウンスの分割・SOURCE対象時の検知器誤検知除外 |
| OTHER | 57 | **50** | C8/C10/コスト0セット 等で解消（残りは長い裾野） |

### 根本原因カテゴリ修正

| # | 不具合 | 修正 |
|---|---|---|
| 1 | **連用形「〜し、」連結句の丸呑み**（前段の相手除去/デバフが消失） | `parser._parse_to_node` の Sequence 分割に `(?<=KOし)、`/`(?<=レストにし)、`/`(?<=戻し)、`/`(?<=\d)し、` を追加。ko/rest/bounce/don_set_rest を分割後の末尾連用形に対応。TRIGGER の「相手をKO/レスト/戻し→自己バウンス」やスクアード型「パワー-Nし→ライフ手札」を正しく Sequence 化 |
| 2 | **ライフ枚数の動的コスト上限が未制限** | `matcher.parse_target` に `cost_max_dynamic=LIFE_COUNT_*`、`get_target_cards` で評価。コスト修飾句中の「お互い/相手/自分の」がプレイヤー判定へ漏れる問題も `player_text` で除去 |
| 3 | **「コスト0にする」が OTHER** | `set_cost` ルール＋`COST_OVERRIDE`。`models.base_cost_override`（base_power_override と対称・reset_turn_status で失効）を追加 |

### 設計案件（ユーザー意思決定のうえ実装）

- **C8 コスト宣言メカニクス**（OP11系6枚, *数値入力UI*）: 「任意のコストを宣言し、相手の
  デッキの上から1枚を公開する。…宣言したコストと同じ場合、…」。
  `ActionType.DECLARE_COST` / `ConditionType.DECLARED_COST_MATCH` / `PendingMessage.DECLARE_COST`
  を新設。resolver は数値入力インタラクション(`_suspend_for_cost_declaration`)へ中断し、
  `gamestate.resolve_interaction` が宣言値を記録＋相手デッキトップを公開(`last_revealed_card`)
  してから再開。フロント(`RealGame.tsx`)に 0〜max のボタン式オーバーレイを追加し
  `declared_value` を送出。
- **C9 同値パワー**（*発動時スナップショット*）: 「このキャラの元々のパワーは、…相手の
  リーダー/選んだキャラ/アタックしているキャラと同じパワーになる」。`power_equalize` ルールが
  `BUFF+POWER_OVERRIDE`、値は `ValueSource(dynamic_source="REFERENCE_POWER", ref_id=...)`。
  `gamestate.get_dynamic_value` が発動時に参照カードの現在パワーで解決（以後の変動に追随しない）。
- **C10 勝敗置換**: 「自分のデッキが0枚になった場合、自分は敗北する代わりに勝利する」(OP03-040
  ナミ等2枚)。`win_on_deckout` ルールが `VICTORY+status="REPLACE_DECKOUT_LOSS"`、
  `check_victory` がデッキアウト時に本人の PASSIVE を走査して敗北を勝利へ置換。

### 任意効果の発動可否（「〜してもよい」yes/no 確認）

ユーザー要望「置換に限らず〈〜できる〉系効果を発動するか選べるようにしたい」に対し、
**通常の効果解決パス**（resolver の suspend/resume が使える）で任意効果の yes/no 確認を実装した。

- `GameAction.is_optional`。`parser._parse_logic_block` が効果文脈で「してもよい／てもよい」
  終止の GameAction に付与（コスト/REPLACE_EFFECT/DECLARE_COST/OTHER は除外。多義の「できる」は
  注釈/コスト/キーワード/トリガー宣言を誤検知するため**明示マーカーのみ**拾う）。
- resolver は実行前に `CONFIRM_OPTIONAL` インタラクションへ中断（`_suspend_for_optional_confirmation`）。
  yes で発動・no でスキップ（`resume_optional`）。確認状態は context の `_confirmed_optionals`
  （id 集合）で持ち、共有ノード(CardMaster)を汚さない。
- フロント(`RealGame.tsx`)に「発動する／発動しない」オーバーレイを追加し `accepted` を送出。
- 既知の制限: 現状は **明示マーカー「してもよい/てもよい」** のみ。より曖昧な「〜できる」
  （文脈依存で任意の場合がある）の拾い上げは誤発動リスクがあるため未対応。必要なら効果動詞ごとに
  ホワイトリスト方式で段階追加する。

### 残課題（次の担当へ）

- **E15 任意置換「代わりに〜できる」の選択UI**（上記の通常パスの確認とは別件）: パーサは既に
  `REPLACE_EFFECT(sub_effect)` を正しく生成し、現状は自動実行（取れるなら実行）。対話的な
  yes/no を挟むには、**同期的な除去フロー（`apply_action_to_engine` の KO/leave・`resolve_attack`
  から呼ばれる `_active_replacement`）の途中で中断・再開する機構**が必要。これは §7 E14
  「置換が対象選択で中断する場合」と同じ構造的ブロッカーで、除去/保護フローの suspend/resume
  対応という独立した中規模リファクタを要する。多くの任意置換はプレイヤーに有利な置換のため
  自動実行で実害は小さいが、UI 化は本リファクタとセットで着手するのが安全。
- 監査の DURATION(24)/TARGET_SIDE(16)/OTHER(50)/MISSING_ACTION(118) の残りは 2デッキ外の
  多様な専用効果と検知器のアドバイザリが中心。同手順（監査でフラグ→カテゴリ修正→回帰）で継続。
- C9 の `attacker` 参照は `active_battle` 経由。バトル外コンテキストでの同値参照は要確認。
