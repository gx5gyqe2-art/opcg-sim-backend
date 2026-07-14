"""委譲shim: 本番 `opcg_sim.src.learned.policy` を単一の正とする（重複解消・旧は import 様式のみ差分）。"""
import sys
from opcg_sim.src.learned import policy as _m
sys.modules[__name__] = _m
