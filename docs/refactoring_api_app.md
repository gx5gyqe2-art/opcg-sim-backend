# リファクタリング詳細設計②: api/app.py のルータ／サービス／状態ストア分離

- 対象: `opcg_sim/api/app.py`（1,002行）
- 目的: HTTP/WS 層に凝集した「ルーティング・ビジネスロジック・状態管理・WS配信・CPU思考駆動」を
  責務ごとのモジュールへ分離する。**API 契約（レスポンス形・エラー形・ETag・WS メッセージ形）は
  一切変えない**。
- ステータス: 設計（実装は本書承認後に別PRで段階実施）
- 関連: `docs/refactoring_gamestate.md`（項目①）、`docs/SPEC.md`（API 仕様）、`tests/test_api.py`（API 契約テスト）

---

## 0. 現状の責務インベントリ（行マップ）

| 責務 | 現在地（app.py の行） | 行数感 |
|---|---|---|
| ブートストラップ（sys.path ハック・二重 import・lifespan・CORS） | L1–61 | ~60 |
| 定数/設定（CONST 読込・パス・IMAGE_VERSION・CARDS_ETAG） | L38–47, 237–264 | ~40 |
| Firestore クライアント（沈黙初期化） | L63–65 | 3 |
| WS 接続管理 ×2（ConnectionManager / GameConnectionManager） | L67–156 | ~90 |
| 状態レジストリ（RULE_ROOMS / GAMES / SANDBOX_GAMES / CPU_GAMES） | L163, 231–235 | ~10 |
| ルーム→WS ペイロード整形（_rule_room_meta / build_rule_message / broadcast_rule_state） | L166–202 | ~40 |
| レスポンス整形（build_game_result_hybrid） | L205–229 | ~25 |
| デッキ読込（_load_deck_doc / load_deck_mixed / _deck_preview） | L268–285, 930–943 | ~35 |
| リプレイ記録（REPLAY_SCHEMA / _replay_enabled / _replay_record_action / _capture_final_winner） | L305–339 | ~35 |
| CPU 思考駆動（_ponder_* / _speculate_* / _kick_* / _cached_cpu_move / _plan_segment） | L459–635 | ~180 |
| ルート: game 系（create/action/state/battle） | L288–457 | ~130 |
| ルート: cpu/step・replay | L638–750 | ~110 |
| ルート: cards/assets | L753–771 | ~20 |
| ルート: deck CRUD | L773–819 | ~45 |
| ルート: sandbox（list/create/action/WS） | L821–879 | ~60 |
| ルート: rule ルーム（create/list/action/WS） | L889–999 | ~110 |
| health | L1001–1002 | 2 |

問題の核心:
1. **状態（4レジストリ）・整形・配信・思考駆動・ルーティングが同一モジュール**で、変更影響が読めない。
2. `_ponder_plan`/`_speculate_plan`（`asyncio.to_thread` + clone）という**並行処理の要**が
   ルート定義の間に埋まっており、レビュー単位として危険。
3. import が `sys.path.append` + try/except の二重経路で、実行ディレクトリ依存。
4. 初期化失敗（firestore / SandboxManager / worker）の沈黙握りつぶし。

## 1. 設計原則

1. **API 契約の完全不変**: `tests/test_api.py`（契約テスト）が green のまま。レスポンス JSON の
   キー・形・エラー整形・ETag/304・WS メッセージ（STATE_UPDATE 形）・OPTIONS ハンドラの応答を
   1バイトも変えない。
2. **プロセス内メモリ状態は「仕様」として維持**: Cloud Run 単一コンテナ・揮発前提は SPEC 記載の
   現行設計（リプレイの docstring L736 にも明記）。本リファクタでは外部ストア化（Redis 等）は
   **しない**が、状態アクセスを 1 モジュールに集約し、将来の差し替え点を1箇所にする。
3. **並行処理の意味論を保存**: ponder/speculate は本番有効（Dockerfile で
   `OPCG_PLAN_CACHE=1 / OPCG_PONDER=1 / OPCG_PONDER_SPEC=1`）。
   「clone は**メインスレッドで原子的に**」「journal はスレッドローカル」「世代 gen による
   supersede」「合法性ゲートで stale を弾く」という不変条件をコードコメントごと逐語移動する。
   `test_plan_cache.py` / `test_journal_concurrency.py` がガード。
4. **FastAPI の慣用へ寄せる**: ルートは `APIRouter` へ分割し、`app.py` は組み立て
   （create_app）だけにする。ただし DI コンテナや `Depends` の全面導入はしない
   （モジュールシングルトンを 1 箇所に集約すれば現規模では十分。過剰設計を避ける）。

## 2. 新パッケージ構成

```
opcg_sim/api/
├── app.py            # 組み立てのみ: FastAPI 生成・CORS・lifespan・ルータ登録・後方互換エイリアス（~80行）
├── config.py         # CONST 読込（schemas.py と一本化）・パス・env フラグ・IMAGE_VERSION 計算
├── resources.py      # プロセス常駐リソース: card_db（CardLoader）・firestore db・CARDS_ETAG
├── state.py          # 対局レジストリ: GAMES / SANDBOX_GAMES / CPU_GAMES / RULE_ROOMS
├── presenters.py     # build_game_result_hybrid / _rule_room_meta / build_rule_message
├── ws.py             # ConnectionManager / GameConnectionManager / broadcast_rule_state
├── services/
│   ├── __init__.py
│   ├── decks.py      # _load_deck_doc / load_deck_mixed / _deck_preview / デッキ CRUD 本体
│   ├── games.py      # 対局生成（create の本体）・_resolve_first_player
│   ├── replay.py     # REPLAY_SCHEMA / _replay_enabled / _replay_record_action / _capture_final_winner
│   └── cpu_driver.py # _plan_segment / _cached_cpu_move / _ponder_* / _speculate_* / _kick_* / step 本体
├── routers/
│   ├── __init__.py   # ALL_ROUTERS のリスト（app.py が一括 include）
│   ├── game.py       # /api/game/create|action|state|battle（+ OPTIONS）
│   ├── cpu.py        # /api/game/cpu/step・/api/game/{id}/replay（+ OPTIONS）
│   ├── cards.py      # /api/cards・/api/assets/version・/health
│   ├── decks.py      # /api/deck/*
│   ├── sandbox.py    # /api/sandbox/*・/ws/sandbox/{id}
│   └── rule.py       # /api/rule/*・/ws/game/{id}
├── schemas.py        # 既存（CONST 読込は config.py へ委譲するよう変更）
└── decide_client.py  # 既存（変更なし）
```

依存の向き（上→下のみ・循環なし）:

```
routers → services → state / presenters / ws / resources / config
app.py  → routers（include）+ 後方互換エイリアス
core（gamestate/action_api/cpu_ai）へは services からのみ到達
```

## 3. 各モジュールの設計

### 3-1. config.py — 設定と定数の一本化

- `load_shared_constants()` を**ここに一本化**する。現在は app.py（L38–42）と schemas.py（L8–22）に
  重複し、探索パスも異なる。schemas.py の候補リスト方式（`../../` / `../` / `/app/`）を正とし、
  **失敗時は logging.warning を出して空 dict**（現行の完全沈黙をやめる。応答形は不変:
  `CONST.get(..., fallback)` パターンがフォールバックを担保している）。
- env フラグの述語を集約: `plan_cache_enabled()` / `ponder_enabled()` / `speculate_enabled()`
  （現 L459–463, 539–542, 696 の `os.environ.get` 直読み3箇所＋インライン1箇所を統一。
  **毎回 env を読む現行セマンティクスを維持**する—Dockerfile コメントに「即ロールバックは
  フラグを 0 に」とあり、起動時固定化は不可）。
- `BASE_DIR / DATA_DIR / CARD_DB_PATH / REPLAY_SCHEMA / IMAGE_VERSION`（`_compute_image_version` ごと移動）。

### 3-2. resources.py — プロセス常駐リソース

```python
# 現 L63–65, 237–243 を移動。初期化失敗の「沈黙」をやめ、原因をログに残す（挙動は不変: db=None 継続）
logger = logging.getLogger("opcg.api")

def _init_firestore():
    if firestore is None:
        logger.warning("google-cloud-firestore 未導入: デッキ CRUD は無効")
        return None
    try:
        return firestore.Client()
    except Exception:
        logger.warning("Firestore 初期化失敗: デッキ CRUD は無効", exc_info=True)
        return None

db = _init_firestore()
card_db = CardLoader(CARD_DB_PATH); card_db.load(); card_db.load_cache()
CARDS_ETAG = f'"{card_db.db_hash()}"'

def materialize_all_cards():
    """遅延パースの全件実体化（現 L347–348, 761–762, 975–976 に3回コピペされている処理を集約）。"""
    if len(card_db.cards) < len(card_db.raw_db):
        for card_id in card_db.raw_db.keys():
            card_db.get_card(card_id)
```

### 3-3. state.py — 対局レジストリ（将来の差し替え点）

```python
# 現 L163, 231–235 の4 dict を移動。型は現行のまま（dict）。
# 本リファクタでは「置き場所の集約」のみ行い、外部ストア化・ロック導入・dataclass 化はしない（§6 非目標）。
GAMES: Dict[str, GameManager] = {}
SANDBOX_GAMES: Dict[str, "SandboxManager"] = {}
CPU_GAMES: Dict[str, Dict[str, Any]] = {}
RULE_ROOMS: Dict[str, Dict[str, Any]] = {}

def clear_all():   # テスト用（test_api.py の per-test クリアを1関数に）
    for reg in (GAMES, SANDBOX_GAMES, CPU_GAMES, RULE_ROOMS):
        reg.clear()
```

> RULE_ROOMS のエントリ形（room dict のキー構成）は現 L158–162 のコメントを
> docstring として移設し、**形の正本**をここに置く。

### 3-4. ws.py — WebSocket 接続管理

- `ConnectionManager`（sandbox 用）と `GameConnectionManager`（ルール対戦用）を逐語移動。
- `delayed_cleanup` が参照する `SANDBOX_GAMES` / `RULE_ROOMS` / `GAMES` は `state` から import。
- `broadcast_rule_state`（L194–202）もここへ（presenters の `build_rule_message` を使う）。
- 送信失敗の `except Exception: pass`（L103, 152）は**現行維持**（切断済みソケットへの送信失敗は
  正常系。コメントでその旨を明記する）。接続直後の初期状態送信失敗（L81, 126）は
  `logger.debug` を追加。

### 3-5. presenters.py — レスポンス/WS ペイロード整形

- `build_game_result_hybrid`（L205–229）/ `_rule_room_meta`（L166–174）/
  `build_rule_message`（L177–191）を逐語移動。
- これらは**契約の中心**（validate 失敗時に raw dict へフォールバックする防御も含めて不変）。
  1行も変えないことをレビュー観点とする。

### 3-6. services/ — ビジネスロジック

**decks.py**: `_load_deck_doc` / `load_deck_mixed` / `_deck_preview` ＋ deck CRUD の本体
（save/delete/get/list のルート内ロジック）。Firestore 依存はすべてこの1ファイルに閉じる。

**games.py**: `/api/game/create` の本体（deck 読込→Player/GameManager 生成→CPU_GAMES 登録→
cpu_trace 時の種固定）と `_resolve_first_player`。`/api/rule/action` の START 分岐が行う
対局生成（L975–984）も**同じ関数を呼ぶ**よう共通化する（現在は create とほぼ同型の重複）。
- 共通化関数: `create_rule_game(p1_deck, p2_deck, *, first_player, vs_cpu, ...) -> (game_id, manager)`。
  ただし seed 固定・CPU メタ登録などの分岐条件は現行と完全同値に保つ。

**replay.py**: `REPLAY_SCHEMA` / `_replay_enabled` / `_replay_record_action` /
`_capture_final_winner` ＋ `/api/game/{id}/replay` の descriptor 組み立て。
「opt-in 時のみ・観測専用・進行不変・例外安全」の設計コメントを逐語維持。

**cpu_driver.py**（最重要・~250行）: `_plan_segment` / `_cached_cpu_move` / `_ponder_enabled` /
`_ponder_plan` / `_kick_ponder` / `_speculate_enabled` / `_speculate_compute` / `_speculate_plan` /
`_kick_speculate` ＋ `/api/game/cpu/step` の本体（`step(game_id) -> dict` として切り出し、
`_waiting_for` はその内部関数のまま維持）。
- **並行処理の不変条件**（§1-3）をモジュール docstring に昇格して明文化する。
- `meta["plan_cache"]` の形（queue/task/spec_queue/spec_task/spec_gen/spec_hits/spec_misses）を
  docstring に記載（現在は読まないと分からない）。

### 3-7. routers/ — ルーティング（薄い皮）

各ルートは「リクエスト取り出し → service 呼び出し → presenter で整形 → WS broadcast → 返却」の
5行程度に薄くする。例（game.py）:

```python
router = APIRouter()

@router.options("/api/game/action")
async def options_game_action(): return {"status": "ok"}

@router.post("/api/game/action")
async def game_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id")
    manager = state.GAMES.get(game_id)
    if not manager:
        return presenters.build_game_result_hybrid(None, game_id, success=False,
                error_code=..., error_msg="指定されたゲームが見つかりません。")
    try:
        manager.action_events = []
        ...  # replay 記録 → action_api 委譲（現行 L396–404 逐語）
        result = presenters.build_game_result_hybrid(manager, game_id, success=True)
        await ws.broadcast_rule_state(game_id)
        cpu_driver.kick_ponder(game_id); cpu_driver.kick_speculate(game_id)
        return result
    except Exception as e:
        return presenters.build_game_result_hybrid(manager, game_id, success=False,
                error_code=..., error_msg=str(e))
```

- **OPTIONS ハンドラは現行どおり全て残す**（フロントの CORS preflight 契約）。
- ルートの `try/except Exception → 整形済みエラー応答` パターンは現行維持
  （HTTP 200 + success:false が契約。HTTPException 化はしない）。

### 3-8. app.py — 組み立てと後方互換

```python
def create_app() -> FastAPI:
    app = FastAPI(title="OPCG Simulator API v1.7", lifespan=_lifespan)
    app.add_middleware(CORSMiddleware, ...)
    for r in routers.ALL_ROUTERS:
        app.include_router(r)
    return app

app = create_app()

# ---- 後方互換エイリアス（テスト・外部スクリプトの参照点）----
# tests/test_api.py は `from opcg_sim.api import app as A` の後、A.GAMES / A.card_db /
# A.load_deck_mixed 等を参照・monkeypatch する。移行を1PRで完結させるため再エクスポートを残す。
from .state import GAMES, SANDBOX_GAMES, CPU_GAMES, RULE_ROOMS       # noqa: F401
from .resources import card_db, db                                   # noqa: F401
from .services.decks import load_deck_mixed                          # noqa: F401
...
```

- Dockerfile の `uvicorn opcg_sim.api.app:app` は**無変更で動く**。
- `sys.path.append` ハック（L18–20）と try/except 二重 import（L22–30）は**撤去**し、
  相対 import（`from .schemas import ...`、`from ..src.core.sandbox import SandboxManager`）へ
  一本化する。SandboxManager の握りつぶし import（失敗すると `sandbox_create` が
  `"SandboxManager" not in globals()` で落ちる現行の防御 L845）は、正規 import の一本化に伴い
  **通常 import に変更**（import 不能ならプロセス起動時に落ちて原因が即分かる方が正しい。
  現に同パッケージ内なので失敗する正当な理由がない）。

### 3-9. test_api.py の追従（機械的・契約は不変）

- monkeypatch 対象: `A.load_deck_mixed` → **呼び出し箇所が services/decks 内になる**ため、
  `monkeypatch.setattr(deck_service, "load_deck_mixed", stub)` に変更（1行）。
  ルータ/サービスは `load_deck_mixed` を **from-import せずモジュール属性経由で呼ぶ**
  規約とし、パッチが必ず効くようにする（`decks.load_deck_mixed(...)` 形式）。
- レジストリのクリア: `state.clear_all()` に置換（1行）。
- アサーション（契約検証）は**一切変更しない**。

## 4. 併せて直す項目と、直さない項目

### 4-1. 本リファクタに含める（低リスク・観測性のみの変更）

| 項目 | 現状 | 変更 |
|---|---|---|
| sys.path ハック・二重 import | L18–30 | 撤去・相対 import 一本化（§3-8） |
| firestore 初期化の沈黙 | L63–65 | logger.warning + 継続（db=None 挙動は不変） |
| worker 起動失敗の沈黙 | L53–56 | logger.warning + 継続（フォールバック挙動は不変） |
| CONST 読込の重複（app/schemas） | 2実装 | config.py へ一本化 |
| カードDB全件実体化のコピペ×3 | L347/761/975 | `resources.materialize_all_cards()` へ集約 |
| `_deck_preview` の `except Exception: pass` | L941 | logger.debug 追加（None 返却は不変） |
| print デバッグ | L82 | logger 化 |

### 4-2. 非目標（別Issue/別設計とする）

- **状態の外部ストア化**（Redis/Firestore への GAMES 退避）・水平スケール対応。
  現行は Cloud Run 単一インスタンス前提の設計（SPEC 準拠）。state.py への集約が将来の受け皿。
- **RULE_ROOMS / CPU_GAMES / plan_cache の dataclass 化**（`Dict[str, Any]` の型付け）。
  形の正本を docstring 化するまでに留める（洗い出し【中】項目として別途）。
- **明示ロックの導入**。asyncio シングルスレッド＋to_thread は clone 済みスナップショットのみに
  触れる現設計で安全（test_journal_concurrency.py がガード）。ロック追加はレイテンシと
  デッドロックのリスクを新規に持ち込むため、外部ストア化とセットで検討する。
- 認証・レート制限・OpenAPI スキーマ整備（フロント型生成の前提整備は洗い出し【中】M-2 で別途）。

## 5. 移行手順（PR 分割）

| PR | 内容 | 主なゲート |
|---|---|---|
| C-1 | `config.py` / `resources.py` / `state.py` 新設。app.py はそこから import（ルートは未分割）。sys.path ハック撤去・import 一本化・沈黙初期化のログ化 | 全テスト（特に test_api.py 46+件）・**uvicorn 起動スモーク**（`uvicorn opcg_sim.api.app:app` を起動し /health 応答確認） |
| C-2 | `presenters.py` / `ws.py` 分離。schemas.py の CONST を config へ委譲 | test_api.py・WS テスト（sandbox/rule の WS 契約） |
| C-3 | `services/`（decks / games / replay）分離。rule START の対局生成を games.create と共通化 | test_api.py・test_realdeck_play.py |
| C-4 | `services/cpu_driver.py` 分離（ponder/speculate/plan_cache を逐語移動） | test_api.py・**test_plan_cache.py・test_journal_concurrency.py**・`OPCG_PLAN_CACHE=1 OPCG_PONDER=1 OPCG_PONDER_SPEC=1` での手動スモーク（CPU対戦を1局通す） |
| C-5 | `routers/` 分割・app.py を create_app + 互換エイリアスへ縮小。test_api.py のパッチ対象更新 | 全ゲート・起動スモーク・SPEC.md の API 章更新 |

各PRの diff は「移動＋import 変更」が主で、新規ロジックは C-3 の対局生成共通化のみ
（それも既存2実装の同型部分の抽出）。`git diff --color-moved=dimmed-zebra` で移動検証。

## 6. リスクと対策

| リスク | 対策 |
|---|---|
| ponder/speculate の並行バグ再発（過去に間欠クラッシュ→journal スレッドローカル化で解消済み） | cpu_driver.py へ**逐語移動**（ロジック変更ゼロ）。「clone はメインスレッド」「gen で supersede」の不変条件を docstring 化。test_journal_concurrency.py を C-4 のゲートに明記 |
| monkeypatch が効かなくなる（from-import による束縛） | 「サービス関数はモジュール属性経由で呼ぶ」規約（§3-9）＋ test_api.py のパッチ対象を同PRで更新 |
| import 一本化による起動環境差（ローカル直接実行 vs Docker） | C-1 に uvicorn 起動スモークを追加。Dockerfile の起動コマンドは不変なので Cloud Run 影響なし |
| 循環 import（state ↔ ws ↔ presenters） | 依存を §2 の一方向に固定（ws → presenters → state。逆方向 import 禁止）。broadcast_rule_state が manager.winner で room status を書く処理（L200–201）は ws.py 内に残し、presenters は純関数に保つ |
| OPTIONS/CORS 契約の欠落 | ルータ移設時に OPTIONS ハンドラの本数を移行前後で機械比較（`grep -c "@.*options"`） |
| SandboxManager import の正規化で起動失敗が顕在化 | 意図した挙動変更（沈黙→即失敗）として PR 説明に明記。パッケージ内 import なので実環境で失敗する経路はない |

## 7. 完了条件（達成状況）

- ✅ app.py は `create_app()` シェル（約85行・CORS＋lifespan＋include_router＋後方互換エイリアスのみ）。
- ✅ ルート関数はロジックを config/resources/state/presenters/ws/services へ委譲する薄い皮。
- ✅ `sys.path` 操作 0（C-1 で撤去）、import フォールバックは firestore の任意依存のみ、初期化の沈黙 `except: pass` 0。
- ✅ test_api.py 系（契約テスト）green。**デッキ読込のみ monkeypatch 対象が `services.decks` に移動**
  （ルートがサービスモジュール属性経由で呼ぶため。C-5 でテスト側も追従済み）。
- ✅ 全品質ゲート green（全スイート 1083 passed・構造監査 0・ベースライン無変更）。
- ✅ `docs/SPEC.md` の API 章がモジュール構成（`routers/` パッケージ・分離モジュール）を反映。

> C-5 は当初リスク管理のため単一 `routers.py` で着地し、後続 followup で本設計どおり
> **`routers/` のドメイン別パッケージ（game/cpu/cards/decks/sandbox/rule）** へ分割完了（§3-7 の構成に一致）。
