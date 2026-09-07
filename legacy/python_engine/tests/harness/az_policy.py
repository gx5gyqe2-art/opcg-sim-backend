"""委譲shim: 本番 `legacy.python_engine.learned.policy` を単一の正とする（重複解消・旧は import 様式のみ差分）。"""
import sys
from legacy.python_engine.learned import policy as _m
sys.modules[__name__] = _m
