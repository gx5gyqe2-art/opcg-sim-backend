# 効果検証トリアージ報告 — イテレーション1

本書は「カード効果の正しさ検証」初回イテレーション（2026-06）の**トリアージ報告**である（特定時点の
スナップショット。検証ハーネスの仕様は [`docs/TEST_SPEC.md`](../TEST_SPEC.md) §3.1。本イテレーションは
報告のみ・修正はしない方針で実施したが、PER_TURN_LIMIT_GAP は確定バグとして別途修正済み＝§2.1）。
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

## 1. 検証層のサマリ（現状）

### 1.1 静的層（パース・方向）
- `text_execution_audit`: `FLAG_OTHER/DURATION/COST_LIMIT/TARGET_SIDE = 0`、`FLAG_MISSING_ACTION = 4`（§2.4 参照）。
- `effect_diagnostics`: 未対応(フォールバック)句・OTHER化句 = **0**。
- `effect_oracle` `HAS_OTHER = 0`（未実装句の回帰なし）。
- `test_quality_gates`: 全合格（NO_IMPL / DIRECTION / SELECT_MISMATCH / INTERACTIVE不一致 = 0）。

### 1.2 動的層（実戦シーケンス・実行時安全）
- CPU 対 CPU 自己対戦でインバリアント違反 **0**:
  - random 方策 300 ゲーム（seed 20000〜）: 300/300 完走・違反 0。
  - AI 方策 hard 8 ゲーム（seed 30000〜）: 8/8 完走・違反 0。
- `--oracle` で各ステップの `snapshot_diff` を記録可能（効果ごとの盤面前後差分を後追い検証できる）。

> 含意: 「クラッシュ／カード消失／中断リーク／場超過／スタック」等の**重篤な実行時異常は現状検出されない**。
> 残るのは「実装済みだが**意味的・ルール的に細部が正しいか**」の領域であり、以下が候補。

---

## 2. 検出された候補（カテゴリ別）

### 2.1 【バグ確定→修正済み】PER_TURN_LIMIT_GAP — 置換/保護系【ターン1回】が未 enforce（18枚）

> **修正済み（このイテレーションで対応）**: 下記の根本原因 1・2 を両方修正。検証:
> 新規テスト `tests/test_turn_limit_replacement.py`（置換3枚＋保護OP10-118）が緑、
> 既存スイート 733 passed / 1 skipped / 2 xfailed、`full_card_audit` ゲート 0、
> 挙動ベースライン（`full_card_baseline.json`）・golden 不変（初回発動は変わらないため）。
> オラクル検出は **18 → 1**（残 1=OP10-118 の inline「ターンに1回」表記で AST 未付与だが、
> 挙動は raw_text フォールバックで enforce 済み＝表現上の軽微な負債のみ）。


**症状**: テキストに【ターン1回】があるが、パース済み能力の条件に `TURN_LIMIT` が無い。
18 枚中 **15 枚が「代わりに…」置換効果**（KO/場を離れる の置換）、残り 3 枚も保護/リダイレクト系の常在。

**検証結果（バグ確定）**: 同一ターンに置換を 2 回発火させる実機テストで、代表 4 枚中 3 枚が**2 回とも発動**:
- OP10-034 フランキー: 手札 5→6→7 / ライフ 5→4→3（ライフ回収が 2 回）。
- ST09-010 ポートガス・D・エース: ライフ→トラッシュが 2 回。
- OP10-074 ピーカ: 置換 fired=True が 2 回。
- （OP13-046 ビスタは検証用手札にコスト対象が無く `_can_satisfy_node` で不成立＝判定不能。反証ではない）

**根本原因（コードで確定・二重の取りこぼし）**:
1. パーサが置換能力（「代わりに…」）に `TURN_LIMIT` 条件を**付与していない**（18枚すべて parsed=False）。
2. 仮に付与されても `_check_condition` の `TURN_LIMIT` 分岐は**常に True**を返す（`resolver.py:667-669`、
   enforce は `resolve_ability` に委譲）。
3. 置換/保護経路（`GameManager._active_replacement` `gamestate.py:1527` / `_active_protection`）は
   `resolve_ability` を通らず、`ability_used_this_turn` を**参照も加算もしない**。
→ 置換/保護系の【ターン1回】は**1ターンに何度でも発動**する（公式は1回まで）＝**実バグ**。

**修正内容**:
1. `parser.py:480-483`: 自己置換（「このキャラ」）で `final_condition = None` としていたのを
   `final_condition = turn_limit_cond` に変更し、【ターン1回】の使用回数制限を保持する。
2. `gamestate.py`: `_active_replacement` / `_active_protection` に per-turn 使用回数の enforce を追加
   （`_ability_turn_limit` で条件 TURN_LIMIT または raw_text の【ターン1回】表記から上限を取得し、
   `ability_used_this_turn` を参照/加算。ターン境界の `reset_turn_status(clear_usage=True)` でリセット）。
   inline「ターンに1回」（parser が拾わない表記）も raw_text 併用でカバー。

**該当カード（18）**:
EB01-008 リトルオーズJr. / OP05-032 ピーカ / OP05-100 エネル / OP07-029 バジル・ホーキンス /
OP07-042 ゲッコー・モリア / OP10-034 フランキー / OP10-037 リム / OP10-074 ピーカ /
OP10-118 モンキー・D・ルフィ / OP12-053 ボルサリーノ / OP13-017 モンキー・D・ドラゴン /
OP13-046 ビスタ / OP14-092 Mr.3(ギャルディーノ) / PRB02-002 トラファルガー・ロー /
ST09-010 ポートガス・D・エース / ST15-005 ポートガス・D・エース / ST20-002 シャーロット・クラッカー /
ST22-012 マルコ

**分類**: `DATA_GAP`/`LIKELY_BUG`。

### 2.2 UP_TO_GAP — 「〜まで」表記だが is_up_to 対象なし（203枚）

検出 203 枚のうち **154 枚は「ドン!!N枚まで」**（DON ランプの上限であり、カード選択の `is_up_to` とは別物）。
検出器の正規表現が粗く、DON 文言を除外していないため誤検知が多い。残り **約49枚**
（カード選択の「までを」が `is_up_to` に落ちていない疑い）。

**分類**: 大半 `FALSE_POSITIVE`（検出器の精度不足）。

### 2.3 MISSING_ACTION — 「公開」を engine が LOOK_LIFE で実装（4枚）

`text_execution_audit FLAG_MISSING_ACTION = 4`: OP10-022 / ST13-007 / ST13-010 / ST13-014。
いずれも「ライフの上から1枚を**公開**」を engine は `LOOK_LIFE` アクションで実装しており、
監査のキーワードマップが `LOOK_LIFE` を「公開」に対応づけていないための**誤検知**。

**分類**: `FALSE_POSITIVE`。

### 2.4 既知の差異（既出・本イテレーション対象外）
`docs/leader_specs/ISSUES.md` の 2 件（EB01-001 カウンター付与の盤面非反映、ST03-001 側無指定 BOUNCE の対象側）。
xfail で固定済み。

---

## 3. カテゴリまとめ

| カテゴリ | 件数 | 判定 |
|---|---|---|
| PER_TURN_LIMIT_GAP | 18→**修正済** | バグ確定→修正（置換/保護系【ターン1回】を enforce、検出 18→1。残1は表現負債・挙動は正常） |
| UP_TO_GAP（精査後） | ~49 | カード選択の 0枚可 取りこぼし疑い |
| MISSING_ACTION | 4 | 誤検知（公開=LOOK_LIFE で実装済み） |
| UP_TO_GAP（DON文言） | ~154 | 誤検知（検出器ノイズ） |
| 既知差異(ISSUES) | 2 | xfail 固定済み |

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
