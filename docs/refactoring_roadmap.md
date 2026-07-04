# リファクタリング実装ロードマップ（①〜⑤ 横断）

5つの詳細設計を実装フェーズへ移すための**横断実行順序**。各設計内の PR 分割
（A/B/C/D/E/F 系）はそれぞれの設計書に定義済み。本書はそれらを跨ぐ依存・衝突の交通整理と
着手順序を定める索引。

## 実装状況（2026-07-04 時点）

**backend の Wave 0〜4 は完了**（①ディスパッチ化＋GameManager分割／②API分割／④契約一本化／
⑤tests再編・logging・except・ruff）。挙動不変（ベースライン無変更）・全ゲート green で各 PR をマージ済み。

- ✅ **①A**（A-1〜A-3）: `apply_action_to_engine` のレジストリ・ディスパッチ化（文字列比較45分岐→enum レジストリ）。
- ✅ **②C**（C-1〜C-5）: `api/app.py` を `create_app()` シェル＋分離モジュール（config/resources/state/
  presenters/ws/services）＋`routers.py` へ分割。**追加 followup で `routers/` をドメイン別パッケージへ分割済み**。
- ✅ **④D**（D-2〜D-4/D-6）: 共有定数ローダ一本化・request_id 決定化・API 契約生成（`tools/export_contract.py`＋
  `contract/`＋`/health` schema_hash＋ラチェット）。
- ✅ **⑤E**（E-1〜E-7）: tests/ 再編・logging 一元化・エンジン内 except の診断ログ・ruff E722 ラチェット。
- ✅ **①B**（B-1〜B-6）: `GameManager` を `core/engine/`（8 モジュール・ステートレス関数＋1行デリゲート）へ分割。
  `gamestate.py` を ~2355→790行（-66%）。

**未実施（低ROI・別 driver 待ちで意図的に保留）**:
- ⑤E-6 の engine 外（parser/cpu_*/learned）への except 型狭め拡張。**裸 except の再発防止は E-7 の
  ruff E722 ラチェットで達成済み**（残るは非裸の防御的 catch＝AI/探索経路のため型狭めは退行リスク）。
- ①B 設計書 §2-8 の A フェーズ由来 micro-cleanup（一部は「単純統合で挙動が変わる」と設計書自身が警告）。
- ③ frontend（`refactoring_realgame.md` F-1〜F-6）は別リポジトリのトラック。

## 設計書インデックス

| # | 設計書 | PR 群 | 主対象 |
|---|---|---|---|
| ① | `refactoring_gamestate.md` | A-1〜A-3 / B-1〜B-6 | `core/gamestate.py`（ディスパッチ化＋GameManager分割） |
| ② | `refactoring_api_app.md` | C-1〜C-5 | `api/app.py`（ルータ/サービス/状態分離） |
| ③ | frontend `refactoring_realgame.md` | F-1〜F-6 | `RealGame.tsx`（フック分割＋盤面レイアウト共通化） |
| ④ | `refactoring_shared_contract.md` | D-1〜D-6 | 契約一本化（shared_constants 同期＋API型生成） |
| ⑤ | `refactoring_tests_and_errors.md` | E-1〜E-7 | tests/ 再編＋例外ログ化 |

## 計画原則

1. **衝突ゾーンは単一オーナー制**（1ファイル＝1ワークストリームが連続占有）:
   - `gamestate.py`: ①が端から端まで占有。⑤E-6 の gamestate 内 except 修正は①B の移設に相乗り。
   - `api/app.py`: 小さな基盤修正（④D-2 の /health、⑤E-5 の except）を先に着地→②の大分割が修正済みコードを運ぶ。
   - 定数ローダ（models.py/schemas.py）: ④D-2 で1回だけ一本化→②はそれを import。
   - frontend `types.ts`: ④D-5（生成化）を先→③は生成物を import。
2. **基盤を先、構造移動を後**（横断的小修正を先着地させ、大移動が修正済みコードを運ぶ）。
3. **ゲートは設計書どおり**: エンジン接触（①A/B・⑤E-6）は `-m slow`＋bench±5%、API 系は
   test_api.py、フロントは手動スモーク。ベースライン再生成は禁止。

## 実行ウェーブ（backend クリティカルパス）

```
Wave 0  D-1(front) · E-1              … 独立・低リスク（並行可）          ✅（backend 分）
Wave 1  D-2 → (E-5) → E-2 → E-3 → E-4 … 横断基盤（ローダ/ロガー/tests再編）✅
Wave 2  A-1 → A-2 → A-3               … ①ディスパッチ化（gamestate 占有） ✅
Wave 3  C-1 → C-2..C-5 → D-3 → D-4 → D-6 … ②API分割＋④契約                ✅
Wave 4  B-1 → .. → B-6 (+E-6) → E-7   … ①GameManager分割＋except仕上げ＋ruff ✅
followup routers/ ドメイン別分割（②C-5 の後続）                          ✅
```

依存グラフ（要点）:
```
D-1 ─(独立)          E-1 ─(独立)
D-2 ─▶ C-1 ─▶ C-2..C-5 ─▶ D-3 ─▶ D-4 ─▶ D-5(front)
E-5 ─▶ C-1
A-1 ─▶ A-2 ─▶ A-3 ─▶ B-1..B-6 ─▶ E-6 ─▶ E-7
```

**クリティカルパス**: `D-2/E-5 → ①A → ②C → ④D → ①B → E-7`。

## フロント並行トラック（backend と独立）

`F-1`(vitest+boardLayout・独立) → `F-2` → … → `F-6`。
`D-5`（frontend 型生成）は F が `types.ts` を触り終えた後、または F-1 と並行で先行着地。

## 着手順序（推奨）

| 順 | PR | リポジトリ | 状態 |
|---|---|---|---|
| 1 | D-1 定数乖離3キー解消 | frontend | 別リポトラック |
| 2 | E-1 fixtures 移設 | backend | ✅ 完了 |
| 3 | D-2 定数ローダ一本化＋/health | backend | ✅ 完了 |
| 4 | E-5 logging 一元化 | backend | ✅ 完了 |
| 5 | A-1 actions パッケージ＋プレイヤーレベルハンドラ | backend | ✅ 完了 |
| 6〜 | A-2/A-3 → C-1..C-5 → D-3/D-4/D-6 → B-1..B-6 → E-6/E-7 → routers 分割 | backend | ✅ 完了 |

> backend の①②④⑤は完了。残りは上記「実装状況」の**未実施**（低ROI・保留）と ③ frontend トラック。

各PRはマージ後、次PRは最新 `origin/main` から作業ブランチを切り直す
（マージ済みブランチは再利用しない）。
