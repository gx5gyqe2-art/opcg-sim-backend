"""ログのシンク分離（Phase 3・`logger_config`）の回帰テスト。

検証項目（`_resolve_sinks` の自動分岐＋明示指定、`file` シンクの書き出し）:
  - 明示 `OPCG_LOG_SINK` が最優先。
  - 未指定の自動分岐: 本番（GCS 可）=gcs/slack/stdout ／ ローカル（GCS 不可）=file/stdout。
  - `OPCG_LOG_SILENT=1` は stdout/file を外す（テスト・従来同値／バッファ蓄積は別途維持）。
  - `file` シンクは {LOG_DIR}/{session}.jsonl へ 1 行 JSON を追記する。
"""
import json
import os

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.utils import logger_config as LC


def _resolve(monkeypatch, *, sink_env=None, silent=False, gcs=False, slack=False):
    """環境/モジュール状態を差し替えて _resolve_sinks() を評価する。"""
    if sink_env is None:
        monkeypatch.delenv("OPCG_LOG_SINK", raising=False)
    else:
        monkeypatch.setenv("OPCG_LOG_SINK", sink_env)
    monkeypatch.setattr(LC, "LOG_SILENT", silent)
    monkeypatch.setattr(LC, "SLACK_BOT_TOKEN", "tok" if slack else None)
    monkeypatch.setattr(LC, "_storage_client", object() if gcs else None)
    monkeypatch.setattr(LC, "BUCKET_NAME", "bucket" if gcs else "")
    return LC._resolve_sinks()


def test_explicit_sink_env_wins(monkeypatch):
    assert _resolve(monkeypatch, sink_env="file") == {"file"}
    assert _resolve(monkeypatch, sink_env="gcs,slack") == {"gcs", "slack"}
    # 明示時も SILENT は stdout/file を外す。
    assert _resolve(monkeypatch, sink_env="stdout,file,gcs", silent=True) == {"gcs"}


def test_default_local_uses_file(monkeypatch):
    """ローカル（GCS 不可・非サイレント）の既定は file＋stdout（GCS 往復なし）。"""
    assert _resolve(monkeypatch, gcs=False, slack=False) == {"file", "stdout"}


def test_default_prod_uses_gcs_slack(monkeypatch):
    """本番（GCS 可）の既定は gcs＋slack＋stdout＝従来同値。file は付かない。"""
    assert _resolve(monkeypatch, gcs=True, slack=True) == {"gcs", "slack", "stdout"}


def test_silent_strips_stdout_and_file(monkeypatch):
    """テスト（OPCG_LOG_SILENT=1）はローカルでも stdout/file を出さない（従来同値）。"""
    assert _resolve(monkeypatch, gcs=False, slack=False, silent=True) == set()


def test_file_sink_writes_jsonl(monkeypatch, tmp_path):
    monkeypatch.setattr(LC, "LOG_DIR", str(tmp_path))
    LC._write_file_sink({LC.K["SESSION"]: "sessX", LC.K["MESSAGE"]: "hello",
                         LC.K["ACTION"]: "game.test"})
    p = tmp_path / "sessX.jsonl"
    assert p.exists()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec[LC.K["ACTION"]] == "game.test"
