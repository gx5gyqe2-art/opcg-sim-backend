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

## 本番への適用（段階的）

現状 `loader.py` はレガシー `EffectParser` を使用しており、V2 は **オプトイン**。
十分にルールが育ち回帰スイートが揃った段階で、`loader.py` の
`EffectParser()` を `EffectParserV2()` に差し替えるだけで切替できる
（IR・インターフェースが同一のため resolver 等は無改修）。

## 既知の課題（診断が表面化させたもの）

- `ActionType.OTHER` 約940件：`resolver`/`gamestate` 側に実行処理が無いため、
  解析できても盤面が変わらない。ルール整備と並行して
  `apply_action_to_engine()` の対応アクション拡充が必要（改善策④）。
- 「このキャラはを得る」等：キーワード付与（【ブロッカー】を得る 等）の
  セグメント分割でキーワードが脱落するケース。専用ルールで対応予定。
- `ドン!!-1` / `コスト-N` 系のコスト操作：未対応。`COST_CHANGE` 実装と合わせて対応予定。

## 現況（ルール9種時点）

- 原子句カバレッジ（ルール命中率）: **約47%**
- `ActionType.OTHER`（実行時に何もしない句）: **942 → 796 に削減**
- ゴールデンケース: 13件（全緑）/ 既存 `tests/test_parser.py` も維持・緑
- ルール: draw / ko / rest / rest_self_cost / power_buff / discard /
  cost_change / play_self / shuffle / remaining_deck_bottom

### 次の最優先ターゲット（診断の OTHER ランキングより）

resolver / gamestate 側の対応（改善策④）も必要なため本増分では未着手：

- `このカードの効果を発動する`（77）= トリガーの自己メイン再発動（`EXECUTE_MAIN_EFFECT` の実装が必要）
- `ドン‼-N` / `レストで追加`（計100超）= ドン!! の返却・レスト追加（`RETURN_DON` 等の実装が必要）
- `このキャラはアタックできない` / `バトルでKOされない` = 制限・常時効果（`RESTRICTION`/`PREVENT_LEAVE` の実装が必要）
