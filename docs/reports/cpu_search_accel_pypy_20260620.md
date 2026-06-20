# CPU 探索 高速化 調査（2026-06-20）— 手順と対照

CPU（hard）思考の更なる高速化＝**「速くした分を horizon（思考の深さ）に回す」**ための調査スナップショット。
本書は**手順（再現可能なベンチ）**と**対照（手段ごとの倍率・リスク・方策影響）**をまとめる。実装は含まない
（点の報告＝追記・改変しない）。仕様の正本は [`docs/SPEC.md` §2.5.2](../SPEC.md)。

## 0. 結論（要約）

- **PyPy で実測 ~2.1x（コード改変ゼロ・挙動ビット一致）**。エンジン本体は **stdlib のみ**で PyPy 上で
  そのまま動く（import・探索とも実証）。配信スタック（FastAPI/pydantic-core/grpcio）の PyPy 互換だけが採用前の課題。
- 換算アンカー（過去実測）: **compute ~1.7x ≒ horizon +1 ≒ +58 Elo**（horizon 3→4・予算 90→150）。
  → **PyPy 単体で horizon +1（=5手）を据え置きレイテンシで取れる**目算。算法系の策（差分評価/LMR 等）と**乗算**。

## 1. 前提：現状のコスト構造（既出・SPEC §2.5.2）

| 段階 | 施策 | 効果 |
|---|---|---|
| clone 高速化 | `CardInstance/DonInstance.__deepcopy__` | ~3x |
| clone 除去 | make/unmake（journaling・`_USE_MAKE_UNMAKE`） | per-node 4.3x → hybrid 3.57x |
| clone 以外 | `@dataclass(eq=False)` 同一性比較＋軽量 `pending_actor_action` | 通算 ~4.2x（decide ~1176→~278ms*） |

\* horizon=3 時点。現行 horizon=4（予算150）の中盤 decide は本調査の CPython 計測で **median ~223ms**。

**現在の支配項は clone ではなく apply＋evaluate**。特に `_search` は子の並べ替え（ビーム選別）のため
**毎ノードで全合法手ぶん `_score_move_1ply`→`evaluate` を呼ぶ**（`cpu_ai.py:1271-1279`）＝eval が分岐数×ノード数走る。
置換表は健全キーで転置率 ≤0.5%・~3%/node オーバーヘッドで**実測不採用**（SPEC §2.5.2・再提案しない）。

## 2. 検証手順（再現可能）

ベンチ: `tests/bench_decide.py`（hard×hard 決定論セルフプレイ・各 decide の所要を計測・JIT 暖機ゲートは計測外）。

```bash
# 1) PyPy 入手（この環境では downloads.python.org が host_not_allowed のため apt 経由）
sudo apt-get install -y pypy3            # → PyPy 7.3.15 / Python 3.9 言語レベル

# 2) ホットパスが PyPy で import 通過するか（stdlib-only の確認）
pypy3 -c "import opcg_sim.src.core.gamestate, opcg_sim.src.core.cpu_ai, opcg_sim.src.core.journal; print('OK')"

# 3) 同一スクリプトで CPython と PyPy を計測（同一シードで挙動一致を担保）
OPCG_LOG_SILENT=1 python  tests/bench_decide.py
OPCG_LOG_SILENT=1 pypy3   tests/bench_decide.py
```

確認できた素性:
- ホットパス（`cpu_ai`/`gamestate`/`journal`/`sandbox`）は **import が stdlib のみ**（native 依存ゼロ）。
- `pydantic`/`google-cloud`/`fastapi` は **API 層（`api/app.py`・`api/schemas.py`）に隔離**＝エンジンは引かない。
- 決定論一致: CPython・PyPy とも **同一 337 step / 280 decide** を再生＝**同じ手を選択（挙動ビット一致）**。

## 3. 対照①：CPython vs PyPy 実測（同一対局）

| 指標 | CPython 3.11.15 | PyPy 3.9（warm） | 倍率 |
|---|---|---|---|
| per-decide median | 222.6 ms | **101.6 ms** | **2.19x** |
| per-decide mean | 250.7 ms | 118.6 ms | 2.11x |
| per-decide p90 | 493.4 ms | 239.0 ms | 2.06x |
| wall（3ゲーム計） | 70.2 s | 33.3 s | 2.11x |

- warm = 5 ゲーム暖機後。**本番サーバは常時稼働＝フルウォーム相当**なのでこの値が実効。
- 暖機 2 ゲームでも median 117.5ms（~1.9x）＝暖機を増やすほど僅かに伸びる。
- **正直な訂正**: 着手前の口頭見積り「3〜8x」に対し**実測は ~2.1x**。本エンジンは効果リゾルバの多態 dispatch が
  多く JIT が最も得意な形ではないため、これが現実値。それでも**改変ゼロ・挙動不変で 2x** は高 ROI。

## 4. 対照②：高速化手段の総覧

倍率は「同深さでの速度」。換算: **~1.7x ≒ horizon +1 ≒ +約58 Elo**。

| 手段 | 速度（同深さ） | 何に効くか | 方策影響 | 工数/リスク | 状態 |
|---|---|---|---|---|---|
| **PyPy**（ランタイム交換） | **~2.1x（実測）** | 全体（解釈実行） | **不変**（IEEE-754 同一・実測一致） | 低（配信スタック互換のみ） | **本調査で実証** |
| 差分評価（incremental eval） | ~1.3〜1.6x（推定） | 支配項 eval を毎回ゼロ走査→差分 | **不変**（全ノード機械照合可） | 中 | 未着手 |
| 遅延評価（lazy ordering） | ~1.3〜1.8x（推定） | 並べ替えの full eval を安価プロキシ化 | 近似（A/B 要） | 中 | 未着手 |
| parked journaled 化 | ~1.1〜1.3x（推定） | 残 clone（decide の 12〜30%）を ~4x 安く | **不変**（既存 journal ゲート流用） | 中 | SPEC「残（任意）」 |
| LMR / PVS / killer | 速度より **horizon +~1 相当** | node 削減・深さ増幅 | 近似（A/B 要） | 中〜低 | 未着手 |
| mypyc（型付き→C拡張） | ~1.5〜4x（一般値） | 属性アクセス重 OOP | **不変** | 中（型注釈＋compile） | 未検証 |
| root 並列化 | wall-clock ~2〜3x | レイテンシ直撃 | 不変だが**要決定性担保** | 高（pickle/GIL/再現性） | 未着手 |
| native 部分書換（Rust/C++） | eval 数x／実効 ~1.5〜2x | eval/ordering を native 常駐 | **要差分オラクル** | 大（divergence risk） | 非推奨 |
| 置換表（transposition） | — | 同一盤面の再探索省略 | 不変 | — | **実測不採用** |

**乗算性**: PyPy（ランタイム）× 差分評価/LMR（算法）は独立に効く＝積み上げで **horizon +2** が射程。

## 5. 採用前の課題（PyPy）

エンジン本体は PyPy で動く。残る判断点は配信スタックのみ:

1. **依存の PyPy 互換**: `pydantic-core`（Rust/PyO3）・`google-cloud`＝`grpcio`（C 拡張）。
   pydantic-core は近年 PyPy wheel あり、**grpcio は PyPy で歴史的に難物**。
2. **アーキテクチャ2択**:
   - **A. 単一プロセス**: pydantic-core/grpcio が PyPy で install できるか検証（pypi 到達可）。
   - **B. プロセス分離（堅い）**: **探索エンジンを PyPy ワーカー**、FastAPI は CPython のまま IPC で `decide` を渡す。
     エンジンが stdlib-only ゆえ分離が自然で、配信スタックの PyPy 互換問題を**完全回避**。
3. **PyPy バージョン**: apt 版は 3.9（古め）。公式 PyPy 3.10/3.11 ならもう少し伸びる可能性。
   この環境は `downloads.python.org` が `host_not_allowed`＝whitelist か vendoring が必要。

## 6. 推奨順序

1. **PyPy 採否を決める**（最大 ROI・改変ゼロ・挙動不変）。配信は **B. プロセス分離**が安全本命。
2. **差分評価**（方策不変ゲートで安全）＝現支配項 eval に直撃。
3. 余力で **LMR / lazy eval** を A/B Elo（horizon 3→4 で使った枠組み）で「退行なし」確認しつつ horizon 拡大。
4. **parked journaled 化**で残 clone を刈る（挙動不変・既存ゲート流用）。

> 検証データの出所: `tests/bench_decide.py`（spike・本ブランチにコミット済み）。
> 計測機: 本実行環境・CPython 3.11.15 / PyPy 7.3.15（3.9.18）。
