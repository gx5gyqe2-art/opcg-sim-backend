# リファクタリング詳細設計⑤: tests/ ディレクトリ再編＋例外握りつぶしのログ化

- 対象:
  - `tests/`（Python 133ファイル: test_* 74 / conftest 1 / 非テスト 58、＋大容量データ）
  - src/api 全域の例外処理（裸 except 3・沈黙 `except: pass` 系 27・print 5）
- 目的: 「テスト」「テスト基盤ライブラリ」「実験/計測スクリプト」「フィクスチャ」の境界を
  ディレクトリで明示する。例外の沈黙握りつぶしを「型指定＋ログ or 理由コメント」に統一し、
  再発を lint ラチェットで防ぐ。**テストの内容・エンジン挙動は一切変えない**。
- ステータス: 設計（実装は本書承認後に別PRで段階実施）
- 関連: ②`docs/refactoring_api_app.md`（app.py の except は②C-1 で実施）、
  ④`docs/refactoring_shared_contract.md`（schemas.py の裸 except はローダ一本化で消滅）

---

# Part A: tests/ ディレクトリ再編

## A-0. 現状の構造分析（実測・設計の前提）

### 重要な発見: 非テスト58本は「不要物」ではなく2種類に分かれる

import 関係を全数調査した結果:

**(1) テスト基盤ライブラリ（テストから import される・28本）** — 移動は import に波及する

| ライブラリ | import するテスト（例） |
|---|---|
| `engine_helpers` / `leader_test_helpers` | リーダー回帰テスト全13ファイル・エンジン系テスト多数 |
| `cpu_selfplay` | CPU 系テスト約15ファイル（selfplay 駆動の共通基盤） |
| `cpu_arena` / `cpu_replay` / `deckgen` / `turn_solver` | test_journal / test_cpu_arena / test_deckgen 等 |
| `full_card_audit` / `structural_invariants` / `effect_oracle` / `expected_effects` / `quality_map` | **品質ゲートの本体**（test_full_card_audit / test_full_card_baseline / test_structural_gate / test_effect_oracle_gate / test_quality_gates） |
| `rl_encoder` / `rl_net` / `rl_datagen` | test_rl_* / test_cpu_learned |
| `az_loop` / `az_mcts` / `az_net` / `az_mcts_tree` / `az_policy` | test_az_* / test_opcg_adapter |
| `opcg_game` / `opcg_action` / `tictactoe` / `gate_a_tictactoe` / `gate_b_opcg` / `p2_gen0` / `p3_loop` / `phase1_sweep` | test_gate_* / test_p2_harness / test_p3_* / test_phase1_sweep |

**(2) スタンドアロン実験・計測・監査 CLI（どのテストからも import されない・30本）** — 自由に移動可

`arena_parallel`(※depth_arena が import) / `depth_arena` / `thinktime_arena` / `bench_decide` /
`_profile_decide` / `_e2e_worker_smoke` / `lethal_audit` / `lethal_regret` / `monotonicity_sweep` /
`rl_evalcost` / `rl_scale_curve` / `rl_throughput` / `p3_gate` / `p3_probe` / `p3_run` / `p3_vs_l1` /
`gate_b_diag` / `battle_coverage` / `effect_coverage` / `effect_diagnostics` / `false_path_coverage` /
`coverage_report` / `card_spec_probe` / `leader_spec_probe` / `compare_parsers` / `condition_synth` /
`sample_audit` / `interactive_target_audit` / `mistarget_diagnostics` / `text_execution_audit`

**(3) データ**: `expected_effects.json`（4.98MB）/ `full_card_baseline.json`（193KB）/ `golden/`
— 参照は6ファイルのみ（effect_oracle / expected_effects / full_card_audit /
test_full_card_baseline / test_verified_buckets / test_verified_decks ＋ test_golden の golden/）。

### import の仕組み（制約）

- 全非テストファイルが冒頭で `import conftest` を実行し、conftest.py が
  (a) リポジトリルートを sys.path に挿入、(b) google.cloud のスタブ化、を行う。
- テストは基盤ライブラリを**ベア import**（`import cpu_selfplay`）している。
  つまり「tests/ ディレクトリが sys.path にあること」が暗黙の前提。

## A-1. 新レイアウト

```
tests/
├── conftest.py            # pytest フック＋sys.path 追加（harness/scripts を path に載せる）
├── test_*.py              # 74本（変更なし・直下に残す）
├── harness/               # (1) テスト基盤ライブラリ 28本
│   ├── _bootstrap.py      #   conftest から抽出した共通ブート（ルート sys.path＋google スタブ）
│   ├── engine_helpers.py, cpu_selfplay.py, full_card_audit.py, ...
├── scripts/               # (2) スタンドアロン実験/計測/監査 CLI 30本
│   ├── bench_decide.py, depth_arena.py, p3_run.py, ...
└── fixtures/              # (3) データ
    ├── expected_effects.json, full_card_baseline.json
    └── golden/
```

### 設計判断とその理由

1. **すべて tests/ 配下に残す**（トップレベル `scripts/` や `opcg_sim/testing/` パッケージ化は
   採らない）。理由: (a) 基盤/スクリプトはテスト専用依存（pytest 前提の conftest ブート）を
   共有しており、製品パッケージに混ぜない方が依存が明瞭、(b) sys.path 制御を conftest 1箇所に
   閉じられる、(c) 将来パッケージ化（`from tests.harness import x`）への移行を妨げない。
2. **テストファイル74本は 1 文字も変更しない**。conftest.py が
   `tests/harness` と `tests/scripts` を sys.path に追加することで、既存のベア import
   （`import cpu_selfplay` / `import full_card_audit`）をそのまま解決させる。
   移行リスクを「conftest の3行＋ファイル移動」に閉じ込める。
3. **`import conftest` の置換**: 移動する58本は先頭の `import conftest` を
   `import _bootstrap` に機械置換する（harness/scripts どちらも同階層に _bootstrap を配置、
   scripts 側は `sys.path` に harness を足す処理も _bootstrap が担う）。
   conftest.py 自身も _bootstrap を import する（ブートロジックの単一ソース化。
   pyo3 の `BaseException` 捕捉はスタブ注入の正当な例外として _bootstrap にコメント付きで残す）。
4. **直接実行の互換**: `python tests/harness/full_card_audit.py` /
   `python tests/scripts/bench_decide.py` は sys.path[0]=自ディレクトリ＋ _bootstrap で
   従来どおり単体実行できる（現行の `python tests/full_card_audit.py` と同じ使い勝手）。
5. **データパス**: fixtures/ への移動に伴い、参照6ファイル＋test_golden の相対パスを更新
   （`os.path.join(os.path.dirname(__file__), ...)` 形式なので機械的）。
   ※ git 履歴上の 5MB は移動では消えない。目的は境界の明示であり、サイズ削減は掲げない。

## A-2. ドキュメント・運用の追従（同一PR内で実施）

- `CLAUDE.md` 品質ゲートのコマンド: `python tests/full_card_audit.py`
  → `python tests/harness/full_card_audit.py`（--regen も同様）。
- `docs/TEST_SPEC.md`: tests/ 配下のパス参照が **83箇所**ある。§実行方法・§ツール一覧を
  新レイアウトに更新（機械置換＋目視）。`docs/README.md`（3箇所）も同様。
- `CLAUDE.md` に配置規約を1行追記:
  「テストが import するものは `tests/harness/`、単体実行の実験/計測は `tests/scripts/`、
  データは `tests/fixtures/` に置く」。

## A-3. 移行手順（PR 分割）と検証

| PR | 内容 | 検証 |
|---|---|---|
| E-1 | `fixtures/` 新設・データ3点移動＋参照パス更新 | 全テスト（baseline/oracle/golden が読めること＝そのまま検証になる）・`full_card_audit` 実行 |
| E-2 | `_bootstrap.py` 抽出＋`scripts/` 移動（30本・テスト import なし） | 全テスト＋スクリプト起動スモーク（各ファイル `python -c "import ..."` 相当を一括実行するチェックスクリプトで import エラー 0 を確認。depth_arena→arena_parallel の同階層 import も対象） |
| E-3 | `harness/` 移動（28本）＋conftest の sys.path 追加 | 全テスト（74本無変更で green ＝ベア import 解決の証明）・`-n auto` 並列で確認（xdist の path 伝播） |
| E-4 | CLAUDE.md / TEST_SPEC.md / docs/README.md のパス更新＋配置規約追記 | 文書内パスの grep 検査（`tests/[a-z_]+\.py` が旧位置を指していないこと） |

- リスク: pytest-xdist のワーカーが conftest の sys.path 変更を継承しない環境差
  → conftest はワーカーでも import されるため理論上安全だが、E-3 のゲートで
  `-n auto` を明示検証項目にする。
- ロールバックは PR 単位の revert で完結（src/ 無変更のためエンジンリスクなし）。

---

# Part B: 例外握りつぶしのログ化

## B-0. 現状インベントリ（実測）

| 分類 | 箇所 | 現状 |
|---|---|---|
| **裸 `except:`** | `resolver.py:148`（asdict 失敗）/ `resolver.py:160`（デバッグ出力全体）/ `schemas.py:20`（定数読込） | 3箇所。型無指定で KeyboardInterrupt 等まで飲む |
| **初期化の沈黙** | `app.py:29`（SandboxManager import）/ `:55`（worker 起動）/ `:65`（firestore）/ `models.py:19`（定数読込） | 失敗しても無言で None/空 dict 継続 → 後段で原因不明の障害 |
| **通信の沈黙** | `app.py:103,152`（WS send）/ `:126`（WS 初期状態送信） | 切断済みソケットへの送信失敗＝正常系（沈黙は正当） |
| **エンジン内の防御** | `gamestate.py:1025,2076` / `parser.py:183` / `cpu_ai.py:948` / `cpu_learned.py:129` / `learned/action.py:63` | `except Exception: pass` 系。効果解決・探索の防御だが、なぜ握るかの記録なし |
| **ワーカー/クライアント** | `decide_client.py:101,132` / `tools/decide_worker.py:101` | フォールバック設計（意図的） |
| **print デバッグ** | `resolver.py:137,139,159`（EXECUTION_REPORT / DEBUG_SNAPSHOT）他 計5箇所 | `OPCG_LOG_SILENT` でゲートされた stdout 出力。`docs/LOGGING.md` の「唯一のログ＝CPU思考トレース」と表現が不整合 |

## B-1. 例外処理ポリシー（プロジェクト規約として docs/LOGGING.md に正本化）

1. **裸 `except:` は禁止**（例外: `tests/harness/_bootstrap.py` の pyo3 PanicException 対策
   `except BaseException` のみ、理由コメント必須）。
2. すべての except は **(a) 型を指定し、(b) ログを出すか「なぜ沈黙が正しいか」の
   コメントを付ける**。無言の `pass` を許すのは、コメントで正当化された箇所のみ。
3. ログは標準 `logging` を使う。`logging.getLogger("opcg.<領域>")`（`opcg.api` /
   `opcg.engine` / `opcg.debug`）。レベル指針:
   - `warning`: 初期化失敗・フォールバック発動（firestore / worker / 定数読込 / sandbox）
   - `debug`: 通信の正常系失敗（WS send）・エンジン内防御の作動
   - 例外情報は `exc_info=True` で添付
4. **`OPCG_LOG_SILENT=1` は従来どおり全出力を抑止**する（テストスイートの必須フラグ・
   挙動不変）。実装はルートロガー `opcg` へのハンドラ登録時に一括判定（print の個別
   ゲートを廃止して一元化）。
5. `docs/LOGGING.md` を更新: 「撤去したのは log_event/GCS/Slack の**イベント配信基盤**。
   標準 logging による WARNING 以上の運用ログと、`opcg.debug`（旧 print の
   EXECUTION_REPORT / DEBUG_SNAPSHOT、OPCG_LOG_SILENT ゲート付き）は存続」と明文化し、
   実装と文書の不整合を解消する。

## B-2. 箇所別の処方（挙動不変を原則）

| 箇所 | 処方 |
|---|---|
| `resolver.py:148` | `except Exception:` に狭め、`ability_dump = str(ability)` フォールバック維持（ログ不要・デバッグ出力内の防御） |
| `resolver.py:160` / print 3箇所 | `_log_execution_report/_log_failure_snapshot` を `opcg.debug` ロガーへ移行（出力内容・ゲート挙動は同一）。外側 except は `except Exception: logger.debug(...)` |
| `schemas.py:20` | ④のローダ一本化（utils/shared_constants.py）で消滅。④未実施の場合のみ `except (OSError, json.JSONDecodeError):`＋warning に修正 |
| `models.py:19` | 同上（④に委譲） |
| `app.py` 全13箇所 | ②C-1（resources.py / ws.py 分離）の処方に従う（初期化=warning／WS send=コメント付き沈黙）。**②実施前に⑤が先行する場合は現位置で同じ処方を適用**（②の移動時にログ行ごと移る） |
| `gamestate.py:1025,2076` / `parser.py:183` | 型を実際に起こり得る例外へ狭め（実装時に到達条件を確認）、`opcg.engine` の debug ログ＋「なぜ継続してよいか」コメントを付す。**探索ホットパス注意**: make/unmake 探索中に高頻度到達する except だった場合、ログは `OPCG_LOG_SILENT` に加えて探索中フラグでの抑止を検討（実装時に `-m slow`＋bench_decide で確認） |
| `cpu_ai.py:948` / `cpu_learned.py:129` / `learned/action.py:63` | 探索/推論の防御。型を狭め、理由コメント必須。ログは debug（探索中の頻度を bench で確認してから） |
| `decide_client.py` / `decide_worker.py` | フォールバック設計＝意図的。warning 1回（初回フォールバック時のみ、以後抑止）＋コメントで維持 |

## B-3. 再発防止ラチェット — ruff の最小導入

- devDependency 相当として `ruff` を CI に追加し、**選択ルールを絞って**開始する:
  `E722`（裸 except 禁止）のみ。既存コードは B-2 で 0 になるため即 green。
- 設定は `pyproject.toml` の `[tool.ruff.lint] select = ["E722"]`（他ルールは範囲外＝
  本リファクタで議論しない。将来 `BLE001` 等へ広げる余地はコメントで残す）。
- CI ステップ: `pip install ruff && ruff check opcg_sim/`（tests/ は対象外から開始）。

## B-4. 移行手順（PR 分割）と検証

| PR | 内容 | 検証 |
|---|---|---|
| E-5 | logging 一元化（`opcg` ロガー＋OPCG_LOG_SILENT 一括ゲート）＋resolver の print/裸 except 置換＋LOGGING.md 更新 | 全テスト（`OPCG_LOG_SILENT=1 -s` で出力ゼロ＝従来同様）・ベースライン無変更 |
| E-6 | エンジン内 except の型狭め＋コメント/ログ（gamestate / parser / cpu_* / learned） | 全テスト＋**`-m slow`（journal）＋bench_decide ±5%**（ホットパス接触のため）・構造監査 EXCEPTION=0 維持 |
| E-7 | ruff E722 ラチェットを CI へ追加 | CI green（E722 違反 0） |

②・④との実施順序: app.py / schemas.py / models.py の該当箇所は②C-1・④D-2 と重複する。
**先に実施した側が処方を適用し、後発は「適用済みであることの確認」に読み替える**
（処方内容は本書 B-2 と ②§4-1・④§3-1 で同一になるよう整合済み）。

## 完了条件

- tests/ 直下が `test_*.py`＋`conftest.py`＋3サブディレクトリのみ。
- テストファイル74本が無変更のまま全スイート green（`-n auto` 並列含む）。
- 裸 `except:` 0（ruff E722 が CI で強制）。コメント無しの沈黙 `except: pass` 0。
- src/api の `print()` 0（`opcg.debug` ロガー経由・OPCG_LOG_SILENT 挙動は不変）。
- CLAUDE.md / TEST_SPEC.md / README / LOGGING.md が新レイアウトと例外ポリシーを反映。
