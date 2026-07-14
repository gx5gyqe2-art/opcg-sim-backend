# opcg ロガーの一元設定を import 時に確定する（OPCG_LOG_SILENT の抑止挙動を含む）。
from .src.utils.logging_setup import configure_opcg_logging

configure_opcg_logging()
