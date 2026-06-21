# PyPy 移行 Phase 0 互換スパイク 実行結果（2026-06-20）

[`pypy_migration_runbook_20260620.md`](pypy_migration_runbook_20260620.md) の **Phase 0（配信スタックが PyPy で
install できるか）** を実行した結果の記録（点の報告）。**結果：方式A（単一プロセス全 PyPy）の2大依存が PyPy で建たない**。

## 環境
- PyPy **7.3.15（Python 3.9 言語レベル）**＝apt 版（`downloads.python.org` がこの環境で `host_not_allowed` のため
  公式 3.11 イメージは未取得・Docker デーモンも未稼働）。pypi.org は到達可。
- 手順：`pypy3 -m venv` → `pip install` を依存グループ別に実行し、build/import 可否を判定。

## 結果

| 依存 | 判定 | 詳細 |
|---|---|---|
| `uvicorn` / `websockets` / `requests` / `h11` / `python-multipart` | ✅ install＋import OK | 純 Python＝PyPy で無問題 |
| **`pydantic-core`**（`fastapi`→`pydantic` v2 の必須依存） | ❌ **build 失敗** | **PyPy wheel 無し**→Rust ソースビルド→`maturin/cargo` が `the configured PyPy interpreter version (3.9) is lower than PyO3's minimum supported version (3.11)` で拒否 |
| **`grpcio`**（`google-cloud-firestore`/`storage` の土台） | ❌ **build 未完（300s timeout）** | **PyPy wheel 無し**→巨大 C 拡張のソースビルドが長大。歴史的に PyPy で難物 |
| エンジン `opcg_sim/src/core/**` | ✅（既出・再掲） | stdlib-only＝PyPy で import・探索とも実証（~2.1x） |

## 判定

- **方式A（単一プロセス・全 PyPy）**：配信スタックの2大依存 `pydantic-core`・`grpcio` が
  PyPy でクリーンに入らない（wheel 無し／ソースビルドが拒否・長大）。
- **版の注記**：テストは PyPy 3.9。PyPy **3.11** なら `pydantic-core` は Rust ソースから建つ*可能性*があるが、
  wheel 無し＝毎回ソースビルド・PyO3 の PyPy 対応は実験的・`grpcio` は別途長大ビルド。
  3.11 で方式A を検証するなら Rust toolchain 同梱イメージでの再スパイクが前提。
