# リファクタリング詳細設計④: フロント／バックエンド契約の一本化（shared_constants 同期＋API型生成）

- 対象:
  - `shared_constants.json`（backend ルート／frontend ルートに**手動コピーで二重存在**）
  - backend: `opcg_sim/api/schemas.py`・`opcg_sim/src/models/models.py`（定数ローダ重複）
  - frontend: `src/api/types.ts`（95行・手書きミラー）・`src/game/types.ts`（149行・同）
- 目的: 「契約の正本はバックエンド」を機械で強制する。定数の乖離を解消・再発防止し、
  API 型を pydantic スキーマからの**コード生成**に置き換える。
- ステータス: 設計（実装は本書承認後に別PRで段階実施。フロント側の変更は
  `opcg-sim-frontend` 側の PR として出す）
- 関連: `docs/refactoring_api_app.md`（②・config.py の CONST 一本化と接続）、
  frontend `docs/refactoring_realgame.md`（③）

---

## 0. 現状の問題（調査結果）

### 0-1. shared_constants.json の乖離（実測）

- backend 版と frontend 版は `CARD_PROPERTIES` で既に diff がある。backend のみ:
  **`TRIGGER_TEXT` / `ABILITY_DISABLED` / `IS_FROZEN`**（frontend 版に無い）。
- 同期手段はシンボリックリンク・スクリプト・CI チェックのいずれも無く、完全な手動コピー。
- 用途は両側で契約の中心: backend は pydantic の Field alias（schemas.py 全域）と
  action_api のアクション名解決、frontend は API キー解決（client.ts）・アクション種別
  （c_to_s_interface）・カード属性キー。**ここがズレると API の読み書きが静かに壊れる**。

### 0-2. 定数ローダの重複（backend 内部で3実装）

| 実装 | 場所 | 特徴 |
|---|---|---|
| `load_shared_constants()` | `src/models/models.py:12` | 単一パス・失敗沈黙・**ハードコードのフォールバック定数辞書つき** |
| `load_shared_constants()` | `api/schemas.py:8` | 候補3パス（`../../`・`../`・`/app/`）・裸 except |
| `get_const()` | `api/app.py:38` | 単一パス（②の config.py へ集約予定） |

### 0-3. API 型の手書きミラーとドリフトの実例

- frontend `src/api/types.ts:8` のコメント「`request_id` … 現状バックエンドからは
  送られていません」は**誤り**（ドリフトの証拠）。backend は `get_pending_request()` で
  `"request_id": str(uuid.uuid4())` を送っている（gamestate.py L549, L570）。
- さらにこの request_id は**取得のたびに再生成**される。同一の選択要求でも WS/ポーリングの
  たびに別 ID になるため、frontend が request_id を「同一要求の識別子」として使っている箇所
  （選択リセット effect・モーダルの React key）が、状態再取得のたびに発火/再マウントし得る。
  契約上の欠陥（§3-3 で修正）。
- `options` の型も両側で不一致: backend `List[Any]`、frontend 型定義は
  `{label, value}[]`、実際の使用箇所（RealGame）は `as unknown as string[]` にキャスト。
  三者バラバラで、真実はコードを実行しないと分からない。
- backend `schemas.py` には `GameStateSchema / PendingRequestSchema / GameActionResultSchema`
  等の pydantic モデルが既にあるが、`extra='allow'`＋validate 失敗時 raw dict フォールバック
  （build_game_result_hybrid）のため、**スキーマが実際の応答より狭くても誰も気づかない**。

## 1. 設計原則

1. **契約の正本＝バックエンド**。`shared_constants.json` と pydantic スキーマから
   フロントの成果物（定数コピー・TS 型）を**生成**する。手書きミラーを撤廃する。
2. **生成物はコミットし、CI で「再生成して差分ゼロ」を検証**する（full_card_baseline と
   同じラチェット思想。ビルド時生成にしないのは、差分レビューを可能にするためと、
   フロント CI がバックエンドのチェックアウトを必要としないようにするため）。
3. **リポジトリ横断の同期は開発環境で完結**させる: 本開発環境（Claude セッション）には
   backend / frontend が同居しているため、同期スクリプトは `../opcg-sim-frontend` への
   生成で足りる。GitHub Actions 越しのクロスリポジトリ検証は PAT が用意されるまで
   オプション扱い（CLAUDE.md の既存の PAT 補足と整合）。
4. **ランタイム安全網**: 生成が漏れて乖離したまま deploy された場合に備え、
   `/health` に契約ハッシュを載せ、フロント起動時に自分の埋め込みハッシュと照合する。

## 2. 成果物の全体像

```
opcg-sim-backend/
├── shared_constants.json          # 正本（場所は現行どおりルート。参照パスを変えない）
├── contract/
│   ├── api_schema.json            # pydantic モデル群の JSON Schema（生成物・コミット）
│   └── manifest.json              # { constants_sha256, schema_sha256, generated_at_commit }
├── opcg_sim/src/utils/shared_constants.py   # ローダ一本化（core/api 共用）
└── opcg_sim/tools/export_contract.py        # スキーマ/manifest 生成 + フロントへの同期

opcg-sim-frontend/
├── shared_constants.json          # 生成物（backend からコピー。手動編集禁止のヘッダ注記）
├── src/api/generated/
│   ├── api-types.ts               # api_schema.json からの生成物（コミット）
│   └── contract-manifest.ts       # ハッシュ埋め込み（/health 照合用）
├── src/api/types.ts               # generated からの再エクスポート＋フロント固有の拡張のみ
└── package.json                   #   "gen:contract": json-schema-to-typescript 実行
```

## 3. バックエンド側の設計

### 3-1. ローダ一本化 — `opcg_sim/src/utils/shared_constants.py`

- 探索パスは schemas.py の候補リスト方式を正とし、`load_shared_constants() -> dict` と
  `constants_hash() -> str`（正規化 JSON の sha256 先頭12桁）を提供。読込失敗は
  `logging.warning`（沈黙をやめる。空 dict フォールバックの挙動は不変）。
- `models.py` / `schemas.py` / （②実施後は）`api/config.py` はこれを import。
  models.py の**ハードコード・フォールバック辞書はここへ1箇所だけ移す**（3重定義の解消）。
- 注意: models.py は import 時に CONST を評価する（enum 変換等）ため、循環を避けて
  `utils` は他モジュールに依存しない葉モジュールとする。

### 3-2. スキーマ補強（型生成の前提）

生成される TS 型の品質はスキーマの正確さで決まる。生成の前に schemas.py を実態へ合わせる:

- `PendingRequestSchema` に実際に送っているフィールドを追加:
  `request_id: Optional[str]`、`source_card_uuid: Optional[str]`、
  `allow_position: Optional[bool]`、`allow_reorder: Optional[bool]`、
  `constraints`（`min/max/source_label/render_mode` を持つ `ConstraintsSchema` に構造化）、
  `options`（実態調査の上 `List[str]` か `List[OptionSchema]` に確定 — 現在
  backend `List[Any]` / frontend 型 `{label,value}[]` / 実使用 `string[]` の三者不一致を解消）。
- `CpuStepResultSchema`（`GameActionResultSchema` ＋ `cpu_acted / cpu_event / waiting_for`）と
  `ActionEventSchema` を新設（現在フロントだけが型を持っている応答の正本化）。
- **契約テスト（ラチェット）を新設** `tests/test_api_contract.py`:
  API スモーク（test_api.py と同じ TestClient 経路）で得た実応答を各スキーマで
  `model_validate(strict でなくとも extra='forbid' 相当の検査)` し、
  (a) validate が**フォールバック経路を踏まずに**成功すること、
  (b) 応答に**スキーマ未定義のキーが現れないこと**（現れたら追加を強制）を検証する。
  これにより「スキーマ＝実態」が以後崩れない。

### 3-3. request_id の安定化（小さな挙動修正・別コミット）

- 現状: `get_pending_request()` が呼ばれるたびに `uuid.uuid4()` を再生成（L549, 570）。
- 変更: interaction/選択要求の**生成時**に一度だけ採番して interaction dict に保持し、
  `get_pending_request()` は保持値を返す（無ければ従来どおり採番＝互換）。
- 効果: フロントの「request_id 変化＝新しい要求」というセマンティクス（選択リセット
  effect・モーダル key）が正しく機能する。エンジンのベースライン（full_card_baseline）は
  API 層の request_id を含まないため影響なし。**挙動修正なので構造リファクタとは
  コミットを分け**、PR 説明に明記する。

### 3-4. `opcg_sim/tools/export_contract.py`

```
python -m opcg_sim.tools.export_contract [--sync-frontend ../opcg-sim-frontend]
```

1. schemas.py の公開モデル（GameStateSchema / PendingRequestSchema /
   GameActionResultSchema / CpuStepResultSchema / CardSchema / ActionEventSchema …）を
   `model_json_schema(by_alias=True)` で `$defs` に集約 → `contract/api_schema.json`。
2. `contract/manifest.json` を更新（constants/schema の sha256）。
3. `--sync-frontend` 指定時（開発環境用）:
   - `shared_constants.json` をフロントへコピー（先頭に「生成物・手動編集禁止・
     正本は opcg-sim-backend」ヘッダを付けられないため、`docs/README` と
     フロント側 `src/api/generated/` の README 注記で明示）。
   - `contract/api_schema.json` もフロントへ渡し、フロント側 `npm run gen:contract` を案内
     （Node 依存の生成はフロント側 package.json の責務にする＝backend は Node 非依存を維持）。
- **CI ゲート（backend）**: export を再実行して `contract/` に git diff が無いことを検証する
  ステップを追加（スキーマを変えたのに export を忘れた PR を落とす）。

### 3-5. /health の拡張（ランタイム安全網）

```json
{ "status": "ok", "constants_loaded": true,
  "constants_hash": "ab12cd34ef56", "schema_hash": "0123abcd4567" }
```

- 追加キーのみ（既存キー不変＝契約互換）。frontend は起動時の `checkHealth()`（既存呼び出し）
  でハッシュを照合し、不一致なら `console.warn`＋開発ビルドでは errorToast を表示する。

## 4. フロントエンド側の設計

### 4-1. 型生成 — `npm run gen:contract`

- devDependency: `json-schema-to-typescript`（実行時依存なし・生成物コミット）。
- 入力: backend の `contract/api_schema.json`（同期スクリプトが配置）
  → 出力: `src/api/generated/api-types.ts` と `contract-manifest.ts`。
- **OpenAPI（openapi.json）からの生成を採らない理由**: 現行 app.py のルートは
  `req: Any = Body(...)` / `Dict[str, Any]` で response_model 宣言も無く、
  openapi.json にはほぼ型情報が無い。②（routers 分割）完了後に response_model を
  付与すれば `openapi-typescript` へ発展できるが、それを待たずに
  「pydantic モデル → JSON Schema → TS」で今の正本（schemas.py）から直接生成する。
  ②完了後の発展経路として設計上互換（生成元が openapi.json に変わるだけ）。

### 4-2. 既存手書き型の移行

| 現行（手書き） | 移行後 |
|---|---|
| `src/api/types.ts` の PendingRequest / GameActionResult / ActionEvent / CpuStepResult | `generated/api-types.ts` からの再エクスポート。**フロント固有のリクエスト型**（GameActionRequest / BattleActionRequest の extra 形など送信側の便宜型）は手書きのまま types.ts に残す（送信契約は backend が `Dict[str,Any]` で受けており生成元が無いため。②の routers 分割で受信側モデルが定義され次第、生成へ切替） |
| `src/game/types.ts` の GameState / PlayerState / CardInstance | `generated` の GameStateSchema 系へ**段階移行**。フロントが便宜上足しているフィールド（仮想ゾーンカード等）は `VirtualZoneCard` として handwritten 拡張に分離済みのため共存可能 |
| `types.ts:8` の「request_id は送られていない」コメント | §3-3 の安定化とともに削除（実態と一致させる） |

- 移行は import 置換のみで**ランタイム挙動を変えない**（型レベルの変更に限定）。
  型エラーが出た箇所は「契約とフロント実装のズレが顕在化した箇所」なので、
  握りつぶさず個別に判断してコミットを分ける（例: `options` の string[] 化）。

### 4-3. 定数コピーの検証

- `npm run gen:contract` は型生成に加えて、`shared_constants.json` の sha256 が
  `contract-manifest.ts` の値と一致するかを検証（不一致＝同期漏れを即検出）。
- frontend CI に `gen:contract` 再実行→差分ゼロ検証ステップを追加
  （backend 側と対になるラチェット。バックエンドのチェックアウト不要＝コミット済み
  api_schema.json を入力にするため self-contained）。

## 5. 運用ルールの更新（CLAUDE.md への追記・両リポジトリ）

- backend: 「`shared_constants.json`・`api/schemas.py` を変更したら
  `python -m opcg_sim.tools.export_contract --sync-frontend ../opcg-sim-frontend` を実行し、
  フロント側の生成物更新を**同じ作業単位でフロントリポジトリの PR として出す**」。
- frontend: 「`shared_constants.json` と `src/api/generated/` は生成物。手動編集禁止。
  変更が必要な場合は backend 側の正本を変更して同期する」。
- 将来オプション（PAT 導入後）: frontend CI から backend リポジトリを checkout して
  manifest ハッシュを照合するクロスリポジトリ検証ジョブ。現時点では設計のみ。

## 6. 移行手順（PR 分割）

| PR | リポジトリ | 内容 | ゲート |
|---|---|---|---|
| D-1 | frontend | **即時修正**: shared_constants.json を backend 版に更新（3キー追加・削除キーなしを確認済み） | tsc / eslint / build（追加キーのみ＝既存参照に影響なし） |
| D-2 | backend | ローダ一本化（utils/shared_constants.py）＋ /health 拡張 | 全テスト・test_api.py（/health は既存キー不変） |
| D-3 | backend | スキーマ補強＋契約テスト（test_api_contract.py）＋ request_id 安定化（別コミット） | 全テスト＋新契約テスト。ベースライン無変更 |
| D-4 | backend | export_contract.py＋contract/ 生成物＋CI ラチェット | 契約テスト・再生成差分ゼロ |
| D-5 | frontend | gen:contract 導入＋generated/ 生成＋types.ts / game/types.ts の再エクスポート移行＋CI ラチェット | tsc / eslint / build / vitest（③で導入済みなら）＋手動スモーク（API 通信系） |
| D-6 | 両方 | CLAUDE.md 運用ルール追記・docs 更新（SPEC.md の契約章） | — |

依存関係: D-1 は独立・即実施可。D-3 は ②（api/app.py 分離）と独立に実施可能だが、
②の C-1（config.py）が先に入る場合は D-2 のローダ一本化先を config.py に合わせる
（設計は両立するよう §3-1 を葉モジュールにしてある）。

## 7. リスクと対策

| リスク | 対策 |
|---|---|
| スキーマ補強で validate が通らなくなり raw dict フォールバックが増える | 契約テスト（§3-2）が「フォールバックを踏まない」ことを直接検証。補強は実応答のインベントリ（テストで収集）に基づいて行う |
| `options` 等の型確定でフロントの実装ズレが顕在化 | 顕在化が目的。型エラー箇所は個別コミットで判断（握りつぶしの `as` 追加を禁止） |
| request_id 安定化の副作用（フロントの再マウント頻度が変わる） | 変わるのは「同一要求での不要な再マウントが消える」方向のみ。オンライン/CPU の手動スモークで選択 UI の連続操作を確認 |
| 生成物のコミット忘れ | 両リポジトリの CI に「再生成して差分ゼロ」ラチェットを追加（§3-4, §4-3） |
| フロント固有拡張と生成型の衝突 | 生成物は `src/api/generated/` に隔離し、手書きは再エクスポート層のみ。生成物への手動編集は CI ラチェットが検出 |

## 8. 非目標

- フル OpenAPI 化（response_model の全ルート付与）: ②の routers 分割後の発展課題。
  本設計の生成パイプラインは生成元を openapi.json に差し替え可能な構造にしてある。
- ルールロジック二重実装（frontend localLogic.ts）の解消: 洗い出し H-2。定数参照の
  ハードコード（INITIAL_DON_COUNT 等）だけは D-5 の際に shared_constants 参照へ置換してよい
  （挙動不変・値同一のため）。
- WS メッセージ（STATE_UPDATE 等）のスキーマ化: REST と同型部分は本設計でカバーされる。
  ルーム系メタ（ready_states 等）の正本化は②の presenters 分離後に追加する。

## 9. 完了条件

- shared_constants.json の diff（backend vs frontend）が 0、かつ CI ラチェットで再発不能。
- 定数ローダが backend 内で1実装（フォールバック辞書も1箇所）。
- frontend の受信 API 型がすべて生成物由来（手書きミラー 0。送信便宜型のみ手書き）。
- 契約テスト green（応答がスキーマで完全記述されている）＋ request_id が要求単位で安定。
- /health ハッシュ照合が機能（意図的に乖離させると warn が出ることを確認）。
