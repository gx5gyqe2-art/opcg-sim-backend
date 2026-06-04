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
- `ドン!!-1` / `コスト-N` 系のコスト操作：未対応。`COST_CHANGE` 実装と合わせて対応予定。

## エンジン実行系（apply_action_to_engine）の拡充 — 改善策④

パーサが正しい ActionType を生成しても、`gamestate.apply_action_to_engine` に
実行処理が無ければ盤面は変わらない（=`ActionType.OTHER` 同様のサイレント失敗）。
そこで実行系を `tests/test_effects_engine.py`（実 GameManager 上で盤面変化を検証）
で守りながら拡充する。

実装済み（gamestate.apply_action_to_engine）:
- `RETURN_DON`（`ドン‼-N`）: 場のドン!!を N 枚ドン!!デッキへ戻す（レスト→アクティブ→付与の順）
- `RAMP_DON` の `status="RESTED"` 対応（`レストで追加`）: レスト状態でコストエリアへ

実装済み（resolver）:
- `EXECUTE_MAIN_EFFECT`（`このカードの【メイン】効果を発動する`）: source_card 自身の
  ACTIVATE_MAIN 能力の効果を実行スタックへ展開して再発動（コストは支払わない）。
  自己参照による無限ループは `_main_expanded` フラグで1回に限定。

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

対応 Duration: `THIS_BATTLE` / `THIS_TURN` / `UNTIL_NEXT_TURN_END`。
ルーティング: `apply_action_to_engine` で BUFF(duration=THIS_BATTLE) と
ATTACK_DISABLE/RESTRICTION を継続効果として登録する。

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

## 現況（ルール16種 + エンジン/resolver/継続効果/除去保護 時点）

- 原子句カバレッジ（ルール命中率）: **約60%**（grant_keyword 追加で 57%→60.1%）
- `ActionType.OTHER`（実行時に何もしない句）: **942 → 421 に削減（約55%減）**
  （キーワード付与は従来 OTHER ではなく誤 BUFF だったため OTHER 数は不変。146 句が
  `BUFF`→`GRANT_KEYWORD` に正される改善）
- テスト: `test_parser.py`(8) + `test_golden.py`(23) + `test_effects_engine.py`(13)
  + `test_gameplay_smoke.py`(2) = 全46件緑
- レガシー vs V2 全カード比較: 退行(新規OTHER)=0 を維持
- パーサルール: draw / ko / rest / rest_self_cost / power_buff / discard /
  cost_change / play_self / shuffle / remaining_deck_bottom / don_return / don_add /
  execute_main / attack_disable / prevent_leave / **grant_keyword**

### 残課題（今後・長い裾野）

残る OTHER は頻度の低い多様な専用効果に分散（上位でも 1 表現あたり 10 件前後）。
費用対効果は逓減しており、今後は次の2方針が現実的:
1. ライフ操作（表/裏向き）・デッキ操作（並び替え）・ドン!!可変返却などを順次ルール化。
2. V2 を本番有効化し、実デッキ(imu/nami)で回しながら不足を golden に追加して個別対応。
- COST/KEYWORD の duration 対応は `_apply_passive_effects` の再計算と統合する設計が必要。
