# テスト仕様書 — opcg-sim-backend

本書は `opcg-sim-backend` の**テスト仕様書**である。対になる **システム仕様書** は
[`docs/SPEC.md`](SPEC.md)。リーダー個別のテスト方針は [`docs/leader_specs/_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md)、
既知の挙動差異は [`docs/leader_specs/ISSUES.md`](leader_specs/ISSUES.md)。

---

## 1. テスト戦略・原則

- **テキスト/公式ルール準拠の期待挙動をアサートする**（現実装に迎合しない）。
- 期待と現挙動が一致 → 通常テスト。差異がある → `@pytest.mark.xfail`（解消で xpass→strict なら赤に転じ、マーカーを外して通常テスト化）。
- **挙動を変えたら全カード挙動ベースライン（`full_card_baseline.json`）を再生成**し、差分をレビューして品質ゲートを通す。

### 実行方法（重要）
logger が `sys.stdout` を直接掴むため、pytest はキャプチャ無効で実行する。

```bash
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider
# （旧表記の -p no:capture でも可。-s/-p no:capture を付けないと I/O error になる）
```

合格条件: 出力が `passed` / `xfailed` / `skipped` のみ。`failed` / `xpassed` を残さない。

---

## 2. テストスイート一覧

### コアルール（ターン/戦闘/召喚酔い/場上限）
| ファイル | 役割 |
|---|---|
| `tests/test_rules_summoning_field_limit.py` | **召喚酔い/速攻**（登場ターン攻撃不可・速攻例外・リーダー非対象）と**場5体上限**（6体目で `FIELD_OVERFLOW_TRASH` 強制トラッシュ／効果登場でも発火／境界）の検証 |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化（プレイ/アタック/ブロック/カウンター/効果解決） |
| `tests/test_realdeck_play.py` | 実カードでの盤面変化・除去保護・対話 |
| `tests/test_self_cannot.py` | 自己制限（CANNOT_*）の enforce |
| `tests/test_arrange_deck.py` | デッキ配置/並び替え対話 |

### オンライン対戦（ルーム/WS）
| ファイル | 役割 |
|---|---|
| `tests/test_rule_online.py` | ルール対戦のルーム生成→デッキ選択→開始→アクションの WS 同期、開始の ready ガード（`load_deck_mixed` をモックし Firestore 非依存） |

### カード効果（パーサ/ゴールデン/全カード）
| ファイル | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサ単体 |
| `tests/test_golden.py` / `tests/golden/*` | ゴールデンコーパス（AST 指紋の部分一致） |
| `tests/test_full_card_audit.py` | 全カード構造不変条件ゲート（EXCEPTION/CARD_LOSS/TEMP_LEAK=0） |
| `tests/test_full_card_baseline.py` | 全カード挙動ベースライン回帰（`full_card_baseline.json` と一致） |
| `tests/test_mistarget_guard.py` | ミスターゲット/lift 検出器の回帰ガード |
| `tests/test_quality_gates.py` | NO_CHANGE/WARN/SELECT_MISMATCH のラチェット |
| `tests/test_cpu_selfplay.py` | CPU 対 CPU 自己対戦の完走・決定論・clone 非破壊・合法手適用・インバリアント検出 |

### リーダー効果（全137枚）
| ファイル | 役割 |
|---|---|
| `tests/test_leader_*.py`（13本） | 全リーダーの挙動テスト。テキスト準拠の期待をアサートし、差異は xfail。方針は [`_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md) |
| `tests/leader_test_helpers.py` | リーダー挙動テスト用ヘルパ（盤面構築・対話駆動・観測） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ（`make_game`/`make_instance`/`make_master`/`action`） |

---

## 3. 診断・監査ツール（pytest 外）

| ツール | 役割 |
|---|---|
| `tests/effect_diagnostics.py` | 未対応句/OTHER ランキングの可視化（`--top N`） |
| `tests/mistarget_diagnostics.py` | ミスターゲット/lift 候補の検出（`--top N`） |
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |
| `tests/effect_coverage.py` | 全カード実行カバレッジ（SKIP/ERROR/INTERACTIVE/EXECUTED/NO_CHANGE） |
| `tests/text_execution_audit.py` | テキスト↔実行不一致の全カード監査（フラグ別） |
| `tests/interactive_target_audit.py` | INTERACTIVE 対象と TargetQuery/テキストの照合 |
| `tests/full_card_audit.py` | 全カード構造不変条件検証＋挙動ベースライン生成（`--regen` で更新） |
| `tests/cpu_selfplay.py` | 決定論的 CPU 対 CPU 自己対戦（効果検証ハーネス）。`--seed N`/`--games K` で再現実行し、各ステップでインバリアント検出（FIELD_LIMIT/DON_CONSERVATION/UUID_DUPLICATE/STUCK/TEMP_ZONE_LEAK 等）、違反時はトレース末尾とリプロ seed を出力。`--out trace.jsonl` で機械可読トレース、`--verbose` で 1 手ずつ表示。`--p1-leader/--p2-leader` でリーダー指定 |
| `tests/leader_spec_probe.py` | リーダー1枚のテキスト/AST要約/実行観測の出力（`<ID>`/`--set`/`--all`/`--json`） |
| `tests/condition_synth.py` / `tests/battle_coverage.py` | 条件合成発動 / 戦闘発火カバレッジ |

---

## 4. ルール追加・検証フロー

```bash
# 1) 未対応句/ミスターゲット候補の確認
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py --top 40
OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py --top 40

# 2) ゴールデンケース追加（tests/golden/golden_cases.py に text と期待 summary）
OPCG_LOG_SILENT=1 python tests/test_golden.py

# 3) ルール追加（opcg_sim/src/core/effects/rules/atoms.py に @rule）
#    エンジン実行が要るなら gamestate/resolver も実装し test_effects_engine に検証追加
#    コアルール（ターン/戦闘等）の変更は gamestate.py を直接修正し test_rules_* に検証追加

# 4) 回帰・退行・カバレッジ
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider
OPCG_LOG_SILENT=1 python tests/compare_parsers.py        # レガシー比の新規OTHER（退行）
OPCG_LOG_SILENT=1 python tests/effect_diagnostics.py     # 命中率 / OTHER 数

# 5) 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
OPCG_LOG_SILENT=1 python tests/full_card_audit.py --regen   # 挙動を意図的に変えた場合に更新
```

`@rule(name, priority)` で関数登録（priority 大ほど先に試行、不一致は `None`、一致は `EffectNode`）。

---

## 5. 品質ゲート

| ツール | 合格条件 |
|---|---|
| `tests/full_card_audit.py` | EXCEPTION / CARD_LOSS / TEMP_LEAK = 0 |
| `tests/test_full_card_baseline.py` | `full_card_baseline.json` と一致 |
| `tests/compare_parsers.py` | 新規 OTHER（退行）= 0 |
| `tests/test_quality_gates.py` | 設定閾値以内 |
| `tests/interactive_target_audit.py` | 疑い 0 |
| `tests/condition_synth.py` / `tests/battle_coverage.py` / `tests/effect_coverage.py` | ERROR 0 |

挙動を変更したら差分をレビューのうえ `full_card_audit.py --regen` でベースライン更新し、上記ゲートを通す。

---

## 6. 直近の変更で追加されたテスト（参考）

- **オンライン対戦**: `tests/test_rule_online.py`（2件）。ルーム生成→WS購読→SET_DECK→START→`/api/game/action` のブロードキャスト同期、開始の両者 ready ガードを検証。
- **コアルール修正**: `tests/test_rules_summoning_field_limit.py`（9件）。召喚酔い/速攻、場5体上限（強制トラッシュ＝`FIELD_OVERFLOW_TRASH`）を検証。
- これらの修正に伴い `full_card_baseline.json` を更新（`OP06-086`: ON_PLAY で場が6体になる挙動が5体上限により `INTERACTIVE`＝選択待ちへ変化）。

全スイート基準値: **711 passed / 1 skipped / 2 xfailed**（既知差異 ISSUES.md の2件が xfail）。

---

## 7. 既知の挙動差異
リーダー効果のテキスト準拠期待と現挙動の差異は [`docs/leader_specs/ISSUES.md`](leader_specs/ISSUES.md) に集約
（各項目は対応する `tests/test_leader_*.py` の xfail で固定）。差異が解消されればマーカーを外して通常テスト化する。
