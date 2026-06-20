# PyPy 移行 ランブック（2026-06-20）

CPU(hard) 探索を PyPy で **~2.1x（実測・改変ゼロ・挙動ビット一致）** 高速化し、浮いた分を horizon に回すための
移行手順書。実測の裏付けは [`reports/cpu_search_accel_pypy_20260620.md`](cpu_search_accel_pypy_20260620.md)、
インフラ前提は Cloud Run（asia-northeast1）＋Firestore＋GCS＋対戦状態インメモリ常駐。

> 設計メモ＝点の報告（追記・改変しない）。手順の実施で仕様が変われば `docs/SPEC.md §2.5` に追従反映する。

## 0. 方針

- **エンジン本体（`src/core/**`）は stdlib のみ＝PyPy で今すぐ動く**（import・探索とも実証済み）。
- 採用の唯一の障壁は**配信スタックの PyPy 互換**：`pydantic-core`（Rust/PyO3）・`grpcio`（C 拡張・google-cloud 依存）。
  PyPy で**歴史的に難物**＝ここを避ける方式を本命にする。
- **本命＝方式B（プロセス分離）**：Web/永続化は CPython のまま、**探索エンジンだけ PyPy ワーカー**で走らせる。
  互換問題を完全回避し、ロールバックも 1 フラグで可能。
- **方式A（単一プロセス全 PyPy）**は依存が PyPy で入れば最も単純。まず Phase 0 で可否だけ安く判定する。
- **不変条件**：移行はインフラ最適化であって**カード挙動・CPU 方策を変えない**。全ゲート（pytest・構造監査・
  full_card_baseline）を **PyPy/CPython 双方で緑**にすることを完了条件にする。

## Phase 0 — 互換スパイク（半日・方式A可否の判定）

公式 PyPy イメージで**全依存が install できるか**だけを見る（コードは触らない）。

```bash
# 公式 PyPy（apt 版は 3.9 と古い。デプロイは公式イメージの 3.10/3.11 を使う）
docker run --rm -it pypy:3.11-slim bash      # 取得不可なら pypy:3.10-slim
  pip install fastapi uvicorn websockets python-multipart requests
  pip install pydantic                       # ← pydantic-core が PyPy で建つか（最大の山）
  pip install google-cloud-firestore google-cloud-storage   # ← grpcio が PyPy で建つか（第二の山）
  pypy3 -c "import pydantic, google.cloud.firestore, fastapi; print('A viable')"
```

- **全部通る → 方式A**（Phase 1A へ）。最小工数。
- **pydantic-core / grpcio が落ちる（想定大）→ 方式B**（Phase 1B へ）。

判定だけの spike。**install ログを報告に残す**（採否の根拠）。

## Phase 1A — 単一プロセス移行（方式Aが通った場合）

1. `Dockerfile` のベースを `FROM pypy:3.11-slim` に変更（`pypy3` が既定 python）。
2. `requirements.txt` は不変。`build_card_cache` は PyPy 上で実行（stdlib のみ＝問題なし）。
3. `CMD` は `uvicorn` 据え置き（uvicorn は PyPy 対応）。`--http h11`（uvloop/httptools は C 拡張で PyPy 不可の
   可能性 → pure-Python の h11 を明示）。
4. → Phase 2（デプロイ）/ Phase 3（ゲート）へ。

> 方式Aは一括移行ゆえロールバックはイメージ差し戻し。Phase 0 が通っても uvloop 等の C 拡張で詰まり得るので、
> 詰まったら方式B へ退避する。

## Phase 1B — プロセス分離移行（本命・方式Bが通らなくても成立）

**構成**：1 コンテナ内で **CPython の FastAPI（既存・無改造）** が、**常駐 PyPy ワーカー**へ `decide` を委譲する。

```
[Cloud Run container]
  CPython uvicorn (FastAPI / pydantic / firestore / GCS)   ← 既存スタックそのまま
        │  decide 要求（pickle した GameManager）
        ▼  ローカル IPC（unix socket / multiprocessing.connection）
  PyPy worker（src/core のみ・常駐＝JIT フルウォーム）       ← 探索だけ高速化
        ▲  選択手（dict）を返す
```

### 1B-1 ワーカー（PyPy 側・新規 `tools/decide_worker.py`）
- 起動時に `src/core` を import（カードキャッシュもロード）。
- ループ：IPC で **(pickle 化 GameManager, actor_name, difficulty, rng_seed)** を受け取り、
  `cpu_ai.decide_guarded(...)` を実行し、**選択手 dict** を返す。
- **依存ゼロ**（stdlib のみ）＝PyPy で確実に動く。

### 1B-2 ブリッジ（CPython 側・`api/app.py` の cpu/step ハンドラ）
- 既存の同期 `decide` 呼び出しを、ワーカーへの **送受信 1 往復**に置換（薄いクライアント関数）。
- **`_USE_PYPY_WORKER` フラグ**で旧経路（インプロセス decide）へ即時フォールバック＝ロールバック容易。
- ワーカー未起動／IPC 失敗時は**自動でインプロセス decide にフォールバック**（可用性優先）。

### 1B-3 直列化（要設計の肝）
- **state 受け渡し**：`GameManager` を `pickle`（engine は stdlib-only＝CPython→PyPy 間 pickle 互換。`__deepcopy__`
  実装済み＝picklable な構造のはず。**要確認＝Phase 3 の round-trip テスト**で機械照合）。
- **コスト**：decide 1 回あたり pickle+IPC 1 往復（≒ clone 1 回ぶん・数 ms）。探索内部は ~280 clone なので
  **相対的に微小**＝PyPy の節約（~120ms/decide）が十分上回る。
- **決定性**：`decide_guarded` の RNG はワーカーへ **seed を渡して再現**（タイブレークの乱数も一致させる）。
  これを守れば**手選択は現行と完全一致**（CPU 方策不変）。

### 1B-4 ワーカーのライフサイクル
- コンテナ起動時に PyPy ワーカーを subprocess で常駐起動（Cloud Run は 1 コンテナ＝同居で十分。サイドカーは任意）。
- **常駐＝JIT フルウォーム**＝実測 ~2.1x がそのまま効く（インメモリ常駐運用なので min-instances≥1 前提と整合）。
- ヘルスチェック＋クラッシュ時自動再起動＋（再起動直後は warmup 前なのでフォールバック許容）。

## Phase 2 — Cloud Run デプロイ変更

| 項目 | 変更 | 理由 |
|---|---|---|
| ベースイメージ | A:`pypy:3.11-slim` / B:CPython 据え置き＋PyPy 同梱 | 方式別 |
| **min-instances** | **≥1 を維持/明示** | インメモリ対戦状態＋**PyPy JIT 常時ウォーム**（cold で warmup ペナルティ） |
| concurrency | 現行維持（探索は単一スレッド前提） | `decide` は同期・原子的（SPEC §2.5.2） |
| vCPU | 据え置き（深さに転用）or 削減（コスト減） | 2.1x の使い道は方針次第 |
| asia-northeast1 | 不変 | — |

> Cloud Run の min/max-instances・concurrency・vCPU は**リポジトリ外（コンソール/IaC）**。デプロイ前に現行値を確認し、
> 上表の前提（min≥1）と齟齬がないか点検する。

## Phase 3 — 検証ゲート（完了条件）

すべて緑で初めて切替。**CPython と PyPy の双方で**回す。

```bash
# 1) 全テスト・構造監査・品質ゲート（CLAUDE.md 必須フラグ）
OPCG_LOG_SILENT=1 pypy3   -m pytest tests/ -q -s -p no:cacheprovider   # 全 pass
OPCG_LOG_SILENT=1 python  -m pytest tests/ -q -s -p no:cacheprovider   # 退行なし確認
OPCG_LOG_SILENT=1 pypy3   tests/full_card_audit.py                     # EXCEPTION/CARD_LOSS/TEMP_LEAK=0

# 2) 速度の再確認（移行後）
OPCG_LOG_SILENT=1 pypy3   tests/bench_decide.py                        # ~2.1x を再現

# 3)（方式B 専用）pickle round-trip 一致：
#    GameManager → pickle → unpickle で clone と完全等価（既存 deep_diff/不変条件を流用）
#    ＋ ブリッジ経由 decide が インプロセス decide と「同一手」（決定性 seed 固定）
```

- **挙動ビット一致**：`test_full_card_baseline.py`（挙動ベースライン）・`test_cpu_make_unmake.py`（方策一致）が
  PyPy でも緑＝カード挙動・CPU 手選択が不変。
- **A/B（任意・深さ転用するなら）**：horizon を +1 して `tests/cpu_arena.py` で「退行なし／+Elo」を確認
  （horizon 3→4 と同じ枠組み・席交互・独立2シード群）。

## Phase 4 — 切替とロールバック

- **切替**：方式B は `_USE_PYPY_WORKER=on`、方式A はイメージ差し替え。**段階リリース**（Cloud Run のトラフィック
  分割＝新リビジョンへ 10%→50%→100%）で本番影響を限定。
- **ロールバック**：
  - 方式B：`_USE_PYPY_WORKER=off`（即・インプロセス decide へ）。
  - 方式A：前リビジョンへトラフィック 100% 戻し（Cloud Run リビジョン管理）。
- **監視**：cpu/step のレイテンシ（p50/p90）・ワーカー再起動回数・フォールバック発生率。

## リスクと対策（要点）

| リスク | 影響 | 対策 |
|---|---|---|
| pydantic-core/grpcio が PyPy 不可 | 方式A 不成立 | Phase 0 で先に判定→方式B（互換問題を回避） |
| uvloop/httptools 等 C 拡張 | 方式A の uvicorn 起動不可 | `--http h11`・`--loop asyncio`（pure-Python） |
| GameManager が pickle 不可 | 方式B の IPC 不成立 | Phase 3 の round-trip テストで早期検出・不可なら to_dict+再構築 API を新設 |
| cold start で JIT 未ウォーム | 初手だけ遅い | min-instances≥1＋ワーカー常駐＋warmup ゲート＋未ウォーム時フォールバック |
| RNG 不一致で手が変わる | 方策の微差 | seed をワーカーへ渡し決定性維持＝手選択一致を Phase 3 で照合 |

## 推奨ルート（要約）

**Phase 0（半日・方式判定）→ 想定どおりなら方式B → Phase 3 全緑 → Phase 4 段階切替。**
方式B は配信スタック無改造・1 フラグでロールバック可・挙動不変ゲート完備＝**最小リスクで ~2.1x**。
深さ転用（horizon +1）は切替が安定してから A/B で別途。

> 出所：本ブランチ `tests/bench_decide.py`（spike）／計測環境 CPython 3.11.15・PyPy 7.3.15(3.9.18)。
