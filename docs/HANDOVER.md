# 引き継ぎ資料 — カード効果システム刷新

最終更新: 2026-06-03 / 対象ブランチ: `main`（PR #4 マージ済み）

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

## 2. 現在の状態（このPR時点）

| 指標 | 開始時 | 現在 |
|---|---|---|
| 原子句カバレッジ（ルール命中率） | 0% | 約70.4%（grant_keyword + ライフ操作5種 + ドン操作4種） |
| `ActionType.OTHER`（実行時に何もしない句） | 942 | 342（約64%減） |
| テスト総数 | 17 | 64（全緑） |
| 本番パーサ | レガシー | **EffectParserV2（既定）** |

> 追記1（キーワード付与の修正）: 「このキャラは【ブロッカー】を得る」等が構造分解で
> キーワードを脱落させ誤って `BUFF` に落ちていたバグを修正。`parse_ability` の
> タグ一括除去をトリガー/注釈タグに限定し（キーワード能力タグは保持）、
> `grant_keyword` ルールで `GRANT_KEYWORD` を生成（146 句）。
>
> 追記2（ライフ操作）: デッキ→ライフ／ライフ→手札／手札→ライフ／ライフ→トラッシュ／
> 表・裏向き をルール化（`life_*`）。`life_to_hand` は legacy が「ライフの上か下から
> …手札に加える」を `destination=LIFE` と誤判定していたバグを修正。`FACE_UP_LIFE`
> のエンジン実行も追加。
>
> 追記3（ドン!!操作）: 付与／アクティブ／レスト／ドンデッキに戻す をルール化
> （`don_*`）。ドンは均質なため枚数(value)ベースで処理。エンジンに `REST_DON`
> 実行系が欠落しており【ドン!!×N】コストが no-op だったバグを修正。`ATTACH_DON`
> を複数枚＋レスト付与対応、`RETURN_DON` を相手対象対応に拡張。詳細は
> `docs/parser_v2.md`。

- 全2652カードの能力構築・実デッキ(imu/nami)ロード・ゲーム開始〜数ターン進行を確認済み。
- レガシー vs V2 の全カード比較で **退行(新規OTHER)=0**。

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

- `CardInstance.timed_power` / `timed_flags` に反映。**これらは `reset_turn_status()` で
  クリアされない**（ターン境界を跨いで存続する鍵）。既存の `power_buff`/`flags`
  （ターン境界でリセット）とは独立で衝突しない。
- 失効は `expire(event)` を **バトル終了**(`resolve_attack`)・**ターン終了**(`end_turn`)で呼ぶ。
- Duration: `THIS_BATTLE` / `THIS_TURN` / `UNTIL_NEXT_TURN_END`。

### 除去保護（PREVENT_LEAVE）

`gamestate._active_protection(card, status)`。除去が起こる瞬間に対象の PASSIVE 能力を走査し、
条件（例: トラッシュ7枚以上）を `EffectResolver._check_condition` で**ライブ評価**する
（フラグをラッチしないので条件変動に追随）。
- `status="LEAVE"`: 相手の効果で場を離れない（KO/bounce/trash 等の除去時）
- `status="BATTLE_KO"`: バトルでKOされない（`resolve_attack` の戦闘KO時）

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
| `opcg_sim/src/models/effect_types.py` | IR 定義（Ability/GameAction/TargetQuery/Condition…） |
| `opcg_sim/src/models/models.py` | CardMaster/CardInstance（`timed_power`/`timed_flags` 追加済） |
| `opcg_sim/src/models/enums.py` | ActionType/TriggerType/Zone… |
| `opcg_sim/src/utils/loader.py` | カードDB/デッキ読込・`make_parser()` ファクトリ |

### テスト・ツール

| パス | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサの単体テスト（8件） |
| `tests/golden/golden_cases.py` | **ゴールデンコーパス（効果セマンティクスの期待値）** |
| `tests/golden/summarize.py` | AST→指紋(summary) 変換＋部分一致判定 |
| `tests/test_golden.py` | ゴールデン・ランナー（21件） |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化テスト（12件） |
| `tests/test_gameplay_smoke.py` | 実デッキでのゲーム進行スモーク（2件） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ |
| `tests/effect_diagnostics.py` | **未対応句/OTHER ランキングの可視化** |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |

---

## 5. 開発フロー（ルール追加 TDD サイクル）

```bash
# 1) 標的を選ぶ（OTHERランキング上位＝効果が動かない直接原因）
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40

# 2) ゴールデンケースを追加して赤にする
#    tests/golden/golden_cases.py に text と期待 summary を書く
OPCG_LOG_SILENT=1 python tests/test_golden.py

# 3) ルールを足して緑にする
#    opcg_sim/src/core/effects/rules/atoms.py に @rule を1つ追加
#    （エンジン側の実行が必要なら gamestate/resolver も実装し test_effects_engine に検証追加）

# 4) 回帰・退行・カバレッジ確認
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider
OPCG_LOG_SILENT=1 python tests/compare_parsers.py     # 退行(新規OTHER)=0 を維持
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py  # 命中率↑/OTHER↓
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

1. **裾野の OTHER（約342件）のルール化** — 頻度は低く多様（上位でも10件前後/表現）。
   `effect_diagnostics.py` の「OTHER化する原子句ランキング」を起点に継続。
   候補: デッキ並び替え（デッキの上か下に置く）、ライフを見て上か下に置く
   （look-and-place）、公開して手札に加える 等。
   - キーワード付与（【ブロッカー】等を得る）は **対応済み**（`grant_keyword`）。
     ただし `GRANT_KEYWORD` は `current_keywords` に直接加算するため、`THIS_TURN` 等の
     duration 失効と、`_apply_passive_effects` の `current_keywords` 毎回リセットとの
     統合は未対応（下記2と同根の課題）。
   - ライフ操作（デッキ↔ライフ↔手札／トラッシュ／表・裏向き）は **対応済み**（`life_*`）。
     残: 「ライフを見て上か下に置く」等の look-and-place 系。なお MOVE_CARD は
     `dest_position` フィールドを持たず常に末尾（下）へ入るため「ライフの上に加える」の
     上下区別は未対応（既存制約）。
   - ドン!!操作（付与／アクティブ／レスト／ドンデッキに戻す）は **対応済み**（`don_*`、
     枚数ベース）。残: REST_DON をコストにする句の充足判定（現状 target=None のため
     `_can_satisfy_node` がドン枚数を検証せず常に True）。
2. **COST/KEYWORD の duration 対応** — 現状 `_apply_passive_effects` がこれらを毎回
   再計算するため、継続効果マネージャと統合する設計が必要（POWER/FLAG は対応済）。
3. **置換効果（「代わりに〜」）** — 「KOされる場合、代わりに手札を捨てる」等。
   除去保護の枠組みを拡張して置換に対応する余地。
4. **ターン1回制限の enforce** — `CardInstance.ability_used_this_turn` は定義/リセット
   されるが未参照。`TURN_LIMIT` 条件が常に True（`resolver._check_condition`）。
5. **条件の fail-safe 化** — `GENERIC`/未対応条件が True に倒れる（誤発動の温床）。
6. **catalog の縮退** — parser が賢くなった分、`MANUAL_EFFECTS`(13枚) を1枚ずつ
   golden で検証しながら削れる。

---

## 8. 注意点・落とし穴

- **本番パスは loader 経由**。catalog(手動定義) > parser(V2) の優先順位（`loader._create_card_master`）。
- **テキスト正規化**: パーサは NFC、loader の DataCleaner は NFKC を使う箇所がある。
  全角/半角・`!!`/`‼`(U+203C)・各種マイナス記号の揺れに注意（ルールの正規表現は両対応にする）。
- **pytest の出力キャプチャ**: logger が `sys.stdout` を直接掴むため、`pytest` は
  `-s`（キャプチャ無効）で実行する。`OPCG_LOG_SILENT=1` 併用推奨。
- **`timed_power`/`timed_flags` は reset_turn_status でクリアしない**設計。ここを
  「リセット対象に追加」してしまうと複数ターン跨ぎ効果が壊れる。
- **`_apply_passive_effects` は cost_buff/current_keywords を毎回リセット**するが
  power_buff/flags はリセットしない。継続効果に COST/KEYWORD を載せる際はこの相互作用に注意。
- **CardMaster は frozen dataclass**。abilities は生成時に確定。テストで能力を差し替える
  場合は `make_master(..., abilities=(...))` で構築する。
- **新パーサの効果は V2 有効化後にのみ反映**。`OPCG_PARSER=legacy` 時はレガシー解釈に戻る
  （= 新 ActionType は生成されない）。

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
- PR #4 — 本作業一式
- 計測の起点コマンド: `OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40`
