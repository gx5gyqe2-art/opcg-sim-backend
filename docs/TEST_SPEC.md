# テスト仕様書 — opcg-sim-backend

本書は `opcg-sim-backend` の**テスト仕様書**である。対になる **システム仕様書** は
[`docs/SPEC.md`](SPEC.md)。リーダー個別のテスト方針は [`docs/leader_specs/_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md)、
既知の挙動差異は [`docs/leader_specs/ISSUES.md`](leader_specs/ISSUES.md)。

---

## 1. テスト戦略・原則

- **効果の意味的正しさ（テキスト準拠で正しく発動するか）は、自動テストではなく
  「デッキ単位の手動検証」で担保する**（→ §8）。本書のテスト群は
  「**壊れていないこと**」——クラッシュ／カード消失／場超過を起こさない、
  既存挙動が退行しない——の保証に役割を絞る。
- **挙動を変えたら全カード挙動ベースライン（`full_card_baseline.json`）を再生成**し、
  差分をレビューして品質ゲートを通す。

### ⚠️ 注意：「成功するが何もしない」効果の死角
`RULE_PROCESSING`（「ルール上、〜になる」等の常在ルール注記）は**実行時 no-op** で、
resolver は `success = True` を返す。エラー・フォールバック・OTHER のいずれにもならず、
**構造監査も挙動ベースラインも素通りする**。「パースできた＝動く」ではない。

- 実例：リーダー OP15-058 エネル「ルール上、自分のドン!!デッキは6枚になる」が
  長期間 **未適用（10枚のまま）** だった。`RULE_PROCESSING` が no-op で、ドン!!デッキ
  枚数は別経路（`GameManager` 構築時）で初期化し直さないと既定の10枚のままになるため。
- 教訓：`RULE_PROCESSING` に落ちる能力は、**別経路でルールが強制されているかを必ず
  実機で確認**する。セットアップ／経済ルール（ドン!!デッキ枚数等）は per-ability の
  盤面差分の外側にあるので、**ゲーム不変条件として個別テストを足す**こと。

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

### カード効果（パーサ/ゴールデン/全カード・回帰/安定性）
| ファイル | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサ単体 |
| `tests/test_golden.py` / `tests/golden/*` | ゴールデンコーパス（AST 指紋の部分一致） |
| `tests/test_full_card_audit.py` | 全カード構造不変条件ゲート（EXCEPTION/CARD_LOSS/TEMP_LEAK=0） |
| `tests/test_full_card_baseline.py` | 全カード挙動ベースライン回帰（`full_card_baseline.json` と一致） |
| `tests/test_verified_decks.py` | **手動検証済みデッキの効果回帰**（§8）。ベースラインが捕捉できない常在ルール（RULE_PROCESSING）・ON_LEAVE 誘発・勝利条件・ドンデッキ枚数・カード名別名・持続時間等を意味的に固定 |
| `tests/test_cpu_selfplay.py` | CPU 対 CPU 自己対戦の完走・決定論・clone 非破壊・合法手適用・インバリアント検出 |

### リーダー効果（全137枚）
| ファイル | 役割 |
|---|---|
| `tests/test_leader_*.py`（13本） | 全リーダーの挙動テスト（既存の回帰アンカー）。方針は [`_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md) |
| `tests/leader_test_helpers.py` | リーダー挙動テスト用ヘルパ（盤面構築・対話駆動・観測） |
| `tests/engine_helpers.py` | 最小 GameManager 構築ヘルパ（`make_game`/`make_instance`/`make_master`/`action`） |

---

## 3. 診断・監査ツール（pytest 外）

| ツール | 役割 |
|---|---|
| `tests/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |
| `tests/full_card_audit.py` | 全カード構造不変条件検証＋挙動ベースライン生成（`--regen` で更新） |
| `tests/cpu_selfplay.py` | 決定論的 CPU 対 CPU 自己対戦。`--seed N`/`--games K` で再現実行し、各ステップでインバリアント検出（FIELD_LIMIT/DON_CONSERVATION/UUID_DUPLICATE/STUCK/TEMP_ZONE_LEAK 等）、違反時はトレース末尾とリプロ seed を出力。`--out trace.jsonl` で機械可読トレース、`--verbose` で 1 手ずつ表示。`--p1-leader/--p2-leader` でリーダー指定 |
| `tests/leader_spec_probe.py` | リーダー1枚のテキスト/AST要約/実行観測の出力（`<ID>`/`--set`/`--all`/`--json`）。手動検証（§8）の補助に使う |

---

## 4. 変更・回帰検証フロー

```bash
# 1) ルール追加（opcg_sim/src/core/effects/rules/atoms.py に @rule）
#    エンジン実行が要るなら gamestate/resolver も実装し test_effects_engine に検証追加
#    コアルール（ターン/戦闘等）の変更は gamestate.py を直接修正し test_rules_* に検証追加

# 2) 回帰・退行
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider
OPCG_LOG_SILENT=1 python tests/compare_parsers.py        # レガシー比の新規OTHER（退行）

# 3) 全カード構造不変条件・挙動ベースライン
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

挙動を変更したら差分をレビューのうえ `full_card_audit.py --regen` でベースライン更新し、上記ゲートを通す。

---

## 6. 直近の変更で追加されたテスト（参考）

- **オンライン対戦**: `tests/test_rule_online.py`（2件）。ルーム生成→WS購読→SET_DECK→START→`/api/game/action` のブロードキャスト同期、開始の両者 ready ガードを検証。
- **コアルール修正**: `tests/test_rules_summoning_field_limit.py`（9件）。召喚酔い/速攻、場5体上限（強制トラッシュ＝`FIELD_OVERFLOW_TRASH`）を検証。
- これらの修正に伴い `full_card_baseline.json` を更新（`OP06-086`: ON_PLAY で場が6体になる挙動が5体上限により `INTERACTIVE`＝選択待ちへ変化）。

---

## 7. 既知の挙動差異
リーダー効果のテキスト準拠期待と現挙動の差異は [`docs/leader_specs/ISSUES.md`](leader_specs/ISSUES.md) に集約
（各項目は対応する `tests/test_leader_*.py` の xfail で固定）。差異が解消されればマーカーを外して通常テスト化する。

---

## 8. 効果の正しさ検証（デッキ単位の手動方式）

効果の意味的正しさ（テキスト準拠で正しく発動するか）は、自動オラクル／監査では検出
しきれない細部が多い。そこで**実際に組んだデッキを起点に、カードを1枚ずつ実装と
突合する手動方式**を採用する。

手順:

1. フロントの**デッキビルダーからデッキを「検証向け Markdown」でエクスポート**する
   （リーダー＋各カードを「枚数 番号 名前 / 効果テキスト / トリガー」で列挙）。
2. 各カードについて、効果テキストを実装（`parser.py` / `resolver.py` /
   `rules/atoms.py` / `matcher.py` / `gamestate.py`）の挙動と突合する。
   AST のダンプだけで判断せず、**実機（実効パワー・条件評価・対象選択・盤面差分）
   まで確認**する（§1 の `RULE_PROCESSING` 死角に注意）。
3. バグ確定なら修正し、可能なら同型テンプレートのカードへ横展開する。挙動を変えた
   場合は §4・§5 の回帰フロー（ベースライン再生成・退行ゼロ・構造ゲート）を通す。
4. リーダーの常在「ルール」効果（ドン!!デッキ枚数等）は per-ability 差分に現れない
   ため、**ゲーム不変条件として個別テストを足す**。

検証で固定した挙動は `tests/test_verified_decks.py` に1ケースずつ集約する（ベースライン
が見られない常在ルール・ON_LEAVE・勝利条件・別名・持続時間等の意味的回帰ガード）。
新しいデッキを検証して挙動を直したら、同ファイルに対応するアサートを追記すること。
