"""opcg ロガーの一元設定。

従来 `resolver.py` の `print()` が個別に `OPCG_LOG_SILENT` を判定していたのを、`opcg` ロガーの
ハンドラ登録で一元的にゲートする。

- `OPCG_LOG_SILENT=1`（テスト/診断の必須フラグ）: `opcg.*` の全出力を抑止する（従来挙動）。
- 非サイレント: `opcg.debug`（旧 print の EXECUTION_REPORT/DEBUG_SNAPSHOT）と WARNING 以上を
  stdout へ出す。書式は `%(message)s`＝旧 print と同一のマーカー付き生テキストを保つ。

`configure_opcg_logging()` は `opcg_sim` パッケージ import 時に一度だけ実行する（冪等）。
本モジュールは os/sys/logging のみに依存する葉。
"""
import os
import sys
import logging

_configured = False


def configure_opcg_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger("opcg")
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)

    if os.environ.get("OPCG_LOG_SILENT"):
        # 全抑止（NullHandler＋CRITICAL 超）。デバッグ print スナップショットも警告も出さない。
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.CRITICAL + 1)
    else:
        handler = logging.StreamHandler(sys.stdout)
        # 旧 print と同一の生テキスト出力（EXECUTION_REPORT/DEBUG_SNAPSHOT のマーカーを保つ）。
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        # 旧 print 相当は DEBUG。従来どおり非サイレントで出力する。
        root.setLevel(logging.DEBUG)
