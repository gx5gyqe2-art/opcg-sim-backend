# 効果検証トリアージ報告 — イテレーション1

本書は「カード効果の正しさ検証」初回イテレーションの**トリアージ報告**である（計画:
`docs/CPU_BATTLE_PLAN.md` の効果検証ハーネス、本イテレーションは**報告のみ・修正はしない**）。
期待挙動マニフェスト × 実行差分、および CPU 対 CPU 自己対戦による in-context 検証の結果をまとめる。

実施日基準のカード規模: 2652 枚（うち効果保持 **2327 枚 / 能力 3168 件**）。

---

## 0. 使ったツール（本イテレーションで追加）

| ツール | 役割 | 主な実行 |
|---|---|---|
| `tests/expected_effects.py` | 期待挙動マニフェスト生成（カード×能力の「期待する動き」を AST から機械生成） | `python tests/expected_effects.py --regen` → `tests/expected_effects.json` |
| `tests/effect_oracle.py` | 期待 vs テキスト/AST の静的整合性コンパレータ（既存ゲートが拾わない高シグナルのみ） | `python tests/effect_oracle.py --json /tmp/oracle.json` |
| `tests/cpu_selfplay.py --oracle` | 自己対戦の各ステップに `snapshot_diff`（盤面前後差分）を記録（in-context 検証） | `python tests/cpu_selfplay.py --oracle --out trace.jsonl` |

既存資産を再利用: `tests/golden/summarize.py`（指紋化）、`tests/effect_coverage.py`（方向マップ/盤面差分）、
`opcg_sim/src/core/invariants.py`（実行時不変条件）、`docs/leader_specs/`（人手オラクル）。

---

## 1. 検証層のサマリ（現状の健全性）

### 1.1 静的層（パース・方向）＝ ほぼクリーン
- `text_execution_audit`: `FLAG_OTHER/DURATION/COST_LIMIT/TARGET_SIDE = 0`、`FLAG_MISSING_ACTION = 4`（§2.4 参照）。
- `effect_diagnostics`: 未対応(フォールバック)句・OTHER化句 = **0**。
- `effect_oracle` `HAS_OTHER = 0`（未実装句の回帰なし）。
- `test_quality_gates`: 全合格（NO_IMPL / DIRECTION / SELECT_MISMATCH / INTERACTIVE不一致 = 0）。

### 1.2 動的層（実戦シーケンス・実行時安全）＝ クリーン
- CPU 対 CPU 自己対戦でインバリアント違反 **0**:
  - random 方策 300 ゲーム（seed 20000〜）: 300/300 完走・違反 0。
  - AI 方策 hard 8 ゲーム（seed 30000〜）: 8/8 完走・違反 0。
- `--oracle` で各ステップの `snapshot_diff` を記録可能（効果ごとの盤面前後差分を後追い検証できる）。

> 含意: 「クラッシュ／カード消失／中断リーク／場超過／スタック」等の**重篤な実行時異常は現状検出されない**。
> 残るのは「実装済みだが**意味的・ルール的に細部が正しいか**」の領域であり、以下が候補。

---

## 2. 検出された候補（カテゴリ別・優先度順）

### 2.1 【高】PER_TURN_LIMIT_GAP — 「ターン1回」が AST/実行に反映されていない（18枚）

**症状**: テキストに【ターン1回】があるが、パース済み能力の条件に `TURN_LIMIT` が無い。
18 枚中 **15 枚が「代わりに…」置換効果**（KO/場を離れる の置換）、残り 3 枚も保護/リダイレクト系の常在。

**根拠（要確認だが強い疑い）**:
- `TURN_LIMIT` の使用回数 enforce は `EffectResolver.resolve_ability`（`resolver.py:44-76`）にのみ存在。
- 置換効果は `GameManager._active_replacement`（`gamestate.py:1527`）が別経路で処理し、
  `ability_used_this_turn` を**参照も加算もしていない**。かつ当該 18 枚は条件に `TURN_LIMIT` を持たない。
- → 置換/保護系の【ターン1回】は**1ターンに複数回発動できてしまう可能性**（公式は1回まで）。

**該当カード（18）**:
EB01-008 リトルオーズJr. / OP05-032 ピーカ / OP05-100 エネル / OP07-029 バジル・ホーキンス /
OP07-042 ゲッコー・モリア / OP10-034 フランキー / OP10-037 リム / OP10-074 ピーカ /
OP10-118 モンキー・D・ルフィ / OP12-053 ボルサリーノ / OP13-017 モンキー・D・ドラゴン /
OP13-046 ビスタ / OP14-092 Mr.3(ギャルディーノ) / PRB02-002 トラファルガー・ロー /
ST09-010 ポートガス・D・エース / ST15-005 ポートガス・D・エース / ST20-002 シャーロット・クラッカー /
ST22-012 マルコ

**分類**: `DATA_GAP`/`LIKELY_BUG`（要確認）。
**確認リプロ（次イテレーション）**: 同一ターンに当該キャラの置換を 2 回誘発させ、2 回目が無効化されるかを
検証する単体テストを `tests/test_leader_*` / `engine_helpers` で作る（KO を 2 回試みる盤面）。

### 2.2 【低・要精査】UP_TO_GAP — 「〜まで」表記だが is_up_to 対象なし（203枚・大半ノイズ）

検出 203 枚のうち **154 枚は「ドン!!N枚まで」**（DON ランプの上限であり、カード選択の `is_up_to` とは別物）。
検出器の正規表現が粗く、**DON 文言を除外していない**ため誤検知が多い。実質レビュー対象は残り **約49枚**
（カード選択の「までを」が `is_up_to` に落ちていない疑い）。

**分類**: 大半 `FALSE_POSITIVE`（検出器の精度不足）。**アクション**: 検出器を改良し
「ドン!!…まで」「アクティブに…まで」等の非カード選択を除外してから再評価する（次イテレーション）。

### 2.3 【誤検知】MISSING_ACTION — 「公開」を engine が LOOK_LIFE で実装（4枚）

`text_execution_audit FLAG_MISSING_ACTION = 4`: OP10-022 / ST13-007 / ST13-010 / ST13-014。
いずれも「ライフの上から1枚を**公開**」を engine は `LOOK_LIFE` アクションで実装しており、
監査のキーワードマップが `LOOK_LIFE` を「公開」に対応づけていないための**誤検知**。

**分類**: `FALSE_POSITIVE`。**アクション**: `text_execution_audit` の動詞マップに `LOOK_LIFE`/`FACE_UP_LIFE`
を「公開」の許容アクションとして追加（次イテレーション）。

### 2.4 既知の差異（既出・本イテレーション対象外）
`docs/leader_specs/ISSUES.md` の 2 件（EB01-001 カウンター付与の盤面非反映、ST03-001 側無指定 BOUNCE の対象側）。
xfail で固定済み。

---

## 3. 優先度まとめ

| 優先 | カテゴリ | 件数 | 性質 | 次アクション |
|---|---|---|---|---|
| **高** | PER_TURN_LIMIT_GAP | 18 | 置換/保護系【ターン1回】の未 enforce 疑い | 単体テストで確認→ enforce 実装 |
| 中 | UP_TO_GAP（精査後） | ~49 | カード選択の 0枚可 取りこぼし疑い | 検出器改良→個別確認 |
| 低 | MISSING_ACTION | 4 | 監査マップの誤検知 | 監査の動詞マップ補正 |
| 低 | UP_TO_GAP（DON文言） | ~154 | 検出器ノイズ | 検出器で除外 |
| — | 既知差異(ISSUES) | 2 | xfail 固定済み | 既存方針どおり |

---

## 4. 再現方法（このレポートの数値の出し方）

```bash
cd opcg-sim-backend
# 期待マニフェスト生成
OPCG_LOG_SILENT=1 python tests/expected_effects.py --regen
# 静的オラクル（候補抽出）
OPCG_LOG_SILENT=1 python tests/effect_oracle.py --json /tmp/oracle.json
OPCG_LOG_SILENT=1 python tests/effect_oracle.py --category PER_TURN_LIMIT_GAP
# 動的（自己対戦・in-context）
OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --games 300 --seed 20000
OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --policy ai --difficulty hard --games 8 --seed 30000
OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --oracle --seed 2 --out /tmp/trace.jsonl  # snapshot_diff
# 個別カードの期待挙動
OPCG_LOG_SILENT=1 python tests/expected_effects.py --card OP10-034
```

---

## 5. 次イテレーション（修正フェーズ・別途合意）

1. **PER_TURN_LIMIT_GAP（最優先）**: 置換/保護経路（`_active_replacement` 等）に per-turn 使用回数の
   enforce を追加するか、パーサで当該能力に `TURN_LIMIT` 条件を付与する。golden/leader_spec に
   失敗テストを先行追加 → 修正 → `full_card_audit.py --regen` → ラチェット強化。
2. **検出器改良**: UP_TO_GAP の DON 文言除外、MISSING_ACTION の動詞マップ補正で誤検知を解消し、
   真の候補だけを残す。
3. **自己対戦オラクルの深化**: `snapshot_diff` に加え、効果発動ごとの expected-vs-actual（マニフェスト参照）
   突合を per-step で行い、in-context の意味的乖離も自動 anomaly 化する。
