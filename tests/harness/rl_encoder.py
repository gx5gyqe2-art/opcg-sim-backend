"""委譲shim: 本番 `opcg_sim.src.learned.encoder` を単一の正とする（重複解消・旧は完全同一コピー）。"""
import sys
from opcg_sim.src.learned import encoder as _m
sys.modules[__name__] = _m
