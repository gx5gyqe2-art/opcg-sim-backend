"""Share the attacker plans (`rule_don_solve`) and the defender DP (`rule_guard_plan_ex`) across processes.

The values are pure functions of their keys (the caches inside crossing_bridge), so loading them changes
nothing but time. PLAN_CACHE=<prefix>: load every <prefix>*.pkl at import; PLAN_CACHE_OUT=<file>: dump at exit.
"""
import atexit
import glob
import os
import pickle
import sys

import crossing_bridge as CB

_STRIDE = int(os.environ.get("GAME_STRIDE") or 1)
if _STRIDE > 1:                       # every n-th game only (the synthetic corpus: compute budget)
    _orig_iter = CB.PL.iter_games

    def _strided(*a, **k):
        for g, item in enumerate(_orig_iter(*a, **k)):
            if g % _STRIDE == 0:
                yield item
    CB.PL.iter_games = _strided

_P = os.environ.get("PLAN_CACHE")
if _P:
    n0 = 0
    for f in sorted(glob.glob(_P + "*.pkl")):
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception as e:  # a half-written shard: skip it
            print("plan_cache: skip %s (%s)" % (f, e), file=sys.stderr)
            continue
        CB._RULE_DON_CACHE.update(d.get("don", {}))
        CB._RULE_EX_CACHE.update(d.get("ex", {}))
        n0 += 1
    print("plan_cache: loaded %d files, %d plans, %d dp" % (n0, len(CB._RULE_DON_CACHE), len(CB._RULE_EX_CACHE)),
          file=sys.stderr)

_OUT = os.environ.get("PLAN_CACHE_OUT")


def _dump():
    if not _OUT:
        return
    tmp = _OUT + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"don": dict(CB._RULE_DON_CACHE)}, fh, protocol=4)   # the DP cache is too large to keep
    os.replace(tmp, _OUT)
    print("plan_cache: dumped %d plans to %s" % (len(CB._RULE_DON_CACHE), _OUT), file=sys.stderr)


atexit.register(_dump)

if _OUT:
    _orig_solve = CB.rule_don_solve
    _N = {"new": 0}

    def _solve(*a, **k):
        n0 = len(CB._RULE_DON_CACHE)
        r = _orig_solve(*a, **k)
        if len(CB._RULE_DON_CACHE) > n0:
            _N["new"] += 1
            if _N["new"] % 100 == 0:
                _dump()
        return r
    CB.rule_don_solve = _solve
