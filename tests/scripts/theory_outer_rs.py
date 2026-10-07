"""**理論の器の Rust 全移植・段 4（2026-10-07）**: 守る側の外側と耐久 `Θ` の切替（`OPCG_THEORY_OUTER=py|rs|both`・
省略時は `OPCG_THEORY_CORE` と同じ）と呼び出しの記録。段 3 の `theory_core_rs` の続き（同じ深さ・同じ記録の器を使う）。

入口（段 5〜6 の Python〔行の読み・局の駆動〕から呼ばれるもの・プロファイルの実測）:
`crossing_bridge` の `attacker_ctx`・`rule_don_plan_for`・`threshold_parts_side`・`threshold_of_me_parts`・`opp_blockers_of`・
`theory_slope_parts`・`hand_groups`・`purse_series`・`tau_grow`、`hand_plan.hand_items`／`search_context`、
`theory_bridge.guard_hand_reading`、`cut_price.curve_of_row`（→ 曲線の代理）。

* `py`（既定）: 何もしない。
* `rs`: Rust だけで解く。攻め手の財布（`actx`）と曲線は **Rust の物**（`ax:<n>`／`cv:<n>`）で、Python は代理（読むだけの
  財布の写し `_Actx`・`CutCurve` の部分クラス `RsCurve`〔`ḡ`・`L`・`Lx`・`set_loss` は Rust〕）を持つ。
* `both`: Python と Rust の両方で解き、型とビットが違えば `AssertionError`（戻りは Python の物に Rust の番地を添えたもの）。
  `RULE_STATS`／`EX_SPEED_STATS` の増分（Rust は順に返す）と、財布ごとの覚え書き `_gain`（癖ごと写した）も毎回比べる。

記録（`OPCG_THEORY_CORE_CAPTURE`）: 入口と核の中の `rule_don_solve` を書く。財布は `_gain` 込みの dict・`_RULE_DON_CACHE` の
読みは `pre` の `rdc`（順に依らない再生）。**移植の間だけの道具**（段 7 で Python の理論と一緒に消す）。
"""
import atexit
import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_core_rs as TCR  # noqa: E402
import theory_rs as RS  # noqa: E402

_OUTER = {"mode": "py"}
_dumps = TCR._dumps


# ---------------------------------------------------------------------------------------------------------------
# 符号化

def _hr_enc(v):
    """`HandRead` → `{"__hr__": 値, …}`（数の部分クラスなので `RS.enc` だと属性が落ちる）。"""
    return {"__hr__": float(v), "cards": [list(c) for c in v.cards], "don": float(v.don), "n_hand": int(v.n_hand),
            "life_types": [list(t) for t in v.life_types], "draw_types": [list(t) for t in v.draw_types],
            "arrive": list(v.arrive)}


def _is_hr(v):
    return type(v).__name__ == "HandRead" and isinstance(v, float)


class _Actx(dict):
    """攻め手の財布の代理（`both`: Python の財布そのもの／`rs`: Rust の財布の写し・読むだけ）。`_rs`＝Rust の番地。"""

    def __del__(self):
        rid = self.__dict__.get("_rs")
        if rid is not None:
            try:
                TCR._rs_call("ax.free", {"d": [["ax", {"obj": rid}]]}, {})
            except Exception:                           # noqa: BLE001
                pass


def _enc4(v, rs):
    """引数の符号化: `rs=True` は Rust へ（財布は番地）、偽は記録へ（財布は `_gain` 込みの dict）。"""
    if _is_hr(v):
        return RS.enc(_hr_enc(v))
    if isinstance(v, _Actx):
        if rs and "_rs" in v.__dict__:
            return {"obj": v._rs}
        return RS.enc(dict(v))
    if rs and isinstance(v, dict) and "key" in v and "cand" in v and "att1" in v:
        raise RuntimeError("theory outer: Rust の番地の無い財布（dict）を Rust に渡そうとした")
    return TCR._enc(v)


def _args4(argd, rs):
    return {"d": [[k, _enc4(x, rs)] for k, x in argd.items()]}


# ---------------------------------------------------------------------------------------------------------------
# 文脈と計数

def _modes_outer(gl):
    import rd_kernel as RDK
    import theory_order as TO
    return {"THETA_HAND_MODE": gl.get("THETA_HAND_MODE"), "THETA_SIDE_MODE": gl.get("THETA_SIDE_MODE"),
            "RD_KERNEL": RDK.MODE, "RATE_DECAY_MODE": gl.get("RATE_DECAY_MODE"),
            "EX_STATE_BUDGET": None if "EX_STATE_BUDGET" not in gl else str(gl.get("EX_STATE_BUDGET")),
            "THETA_BODY_MODE": gl.get("THETA_BODY_MODE"), "SLOPE_TAKE_MODE": gl.get("SLOPE_TAKE_MODE"),
            "ATTACK_DON_MAX": str(TO.ATTACK_DON_MAX)}


def _cb_globals(gl):
    """切替を読む `crossing_bridge` の大域（器が `crossing_bridge` 自身なら `__main__` の写し・他は import した物）。"""
    if gl is not None and "THETA_HAND_MODE" in gl:
        return gl
    return vars(sys.modules["crossing_bridge"])


def _g4(gl):
    g = TCR._g()
    g["modes"] = dict(g["modes"], **{k: v for k, v in _modes_outer(_cb_globals(gl)).items() if v is not None})
    return g


def _rs_call_ev(name, a_enc, g):
    import json
    payload = _dumps({"d": [["a", a_enc], ["g", RS.enc(g)], ["pre", []]]})
    try:
        out = json.loads(TCR._engine().theory_core_call(name, payload, False))
    except Exception as e:                              # noqa: BLE001
        raise RuntimeError("theory outer: %s を Rust が解けない: %s" % (name, e))
    d = {k: x for k, x in out["d"]}
    ev = [(k, TCR._dec(v)) for k, v in d.get("ev", [])]
    return d["r"], [int(x) for x in d["cs"]], ev


def _stats_dicts(gl):
    cb = _cb_globals(gl)
    return cb["RULE_STATS"], cb["EX_SPEED_STATS"]


def _apply_ev(gl, ev, rs_d=None, ex_d=None):
    if not ev:
        return
    a, b = _stats_dicts(gl)
    rs_d = a if rs_d is None else rs_d
    ex_d = b if ex_d is None else ex_d
    for k, v in ev:
        if k.startswith("ex:"):
            k = k[3:]
            ex_d[k] = ex_d.get(k, 0) + v
        else:
            rs_d[k] = rs_d.get(k, 0) + v


def _snap(gl):
    a, b = _stats_dicts(gl)
    return dict(a), dict(b)


def _ev_check(name, gl, before, ev):
    """`both`: Python が動かした計数（前 → 後）と、前に Rust の増分を順に足したものが同じか。"""
    a, b = _stats_dicts(gl)
    ra, rb = dict(before[0]), dict(before[1])
    _apply_ev(gl, ev, ra, rb)
    if _dumps(RS.enc(ra)) != _dumps(RS.enc(dict(a))) or _dumps(RS.enc(rb)) != _dumps(RS.enc(dict(b))):
        TCR._stat(name, "both_mismatch_stats")
        raise AssertionError("theory outer both: %s の計数が違う\n  py=%r / %r\n  rs=%r / %r" % (name, dict(a), dict(b), ra, rb))


# ---------------------------------------------------------------------------------------------------------------
# 戻りの扱い

def _curve_summary(cv):
    return {"n": int(cv.valuer.n), "cand": list(cv.cand), "h0": cv.h0, "share": cv.share, "mu": cv.mu,
            "cids": list(cv.cids), "reserve": cv.reserve, "xs_future": list(cv.xs_future), "mlp": cv.mlp}


def _dec_result(name, r_enc):
    """Rust の戻り → Python の値（財布・曲線は代理に）。"""
    if name == "cb.attacker_ctx":
        if r_enc is None:
            return None
        d = {k: x for k, x in r_enc["d"]}
        ax = _Actx(TCR._dec(d["d"]))
        ax._rs = d["id"]["obj"]
        return ax
    if name == "cp.curve_of_row":
        if r_enc is None:
            return None
        d = {k: x for k, x in r_enc["d"]}
        return _CLS["rs"](d["id"]["obj"], TCR._dec(d["s"]))
    return TCR._dec(r_enc)


def _res_enc(name, r):
    if name == "cb.attacker_ctx":
        return None if r is None else RS.enc({k: v for k, v in r.items() if k != "_gain"})
    if name == "cp.curve_of_row":
        return None if r is None else RS.enc(_curve_summary(r))
    return _enc4(r, False)


def _rs_cmp(name, r_rs_enc):
    if r_rs_enc is None or name not in ("cb.attacker_ctx", "cp.curve_of_row"):
        return r_rs_enc
    d = {k: x for k, x in r_rs_enc["d"]}
    return d["d"] if name == "cb.attacker_ctx" else d["s"]


def _wrap_both_result(name, r_py, r_rs_enc):
    """`both`: Python の物に Rust の番地を添える。"""
    if r_py is None or name not in ("cb.attacker_ctx", "cp.curve_of_row"):
        return r_py
    rid = {k: x for k, x in r_rs_enc["d"]}["id"]["obj"]
    if name == "cb.attacker_ctx":
        ax = _Actx(r_py)
        ax._rs = rid
        return ax
    r_py.__class__ = _CLS["both"]
    r_py._rs = rid
    return r_py


def _gain_check(argd):
    """`both`: 財布の `_gain`（Python）と Rust の財布の覚え書きが同じか（癖の写しの見張り）。"""
    for v in argd.values():
        if isinstance(v, _Actx) and "_rs" in v.__dict__:
            r, _cs, _ev = _rs_call_ev("ax.dict", {"d": [["ax", {"obj": v._rs}]]}, {})
            g_rs = {k: x for k, x in r["d"]}.get("_gain")
            g_py = RS.enc(v["_gain"]) if "_gain" in v else None
            if _dumps(g_rs) != _dumps(g_py):
                TCR._stat("ax._gain", "both_mismatch")
                raise AssertionError("theory outer both: 財布の _gain が違う\n  py=%s\n  rs=%s"
                                     % (_dumps(g_py)[:800], _dumps(g_rs)[:800]))
            TCR._stat("ax._gain", "both_checked")


def _call_outer(name, orig, bind, args, kw, gl):
    mode = _OUTER["mode"]
    cap = bool(TCR._CFG["dir"])
    argd = bind(args, kw)
    g = _g4(gl)
    if mode == "rs":
        r_enc, cs, ev = _rs_call_ev(name, _args4(argd, True), g)
        TCR._add_cond_stats(cs)
        _apply_ev(gl, ev)
        TCR._stat(name, "rs_calls")
        return _dec_result(name, r_enc)
    a_cap = _args4(argd, False) if cap else None
    a_rs = _args4(argd, True) if mode == "both" else None
    fr = {"start": TCR._SEQ[0] + 1, "pre": [], "seen": set()}
    TCR._FRAMES.append(fr)
    cs0 = TCR._cond_stats()
    st0 = _snap(gl)
    TCR._DEPTH[0] += 1
    try:
        r = orig(*args, **kw)
    finally:
        TCR._DEPTH[0] -= 1
        TCR._pop_frame(fr)
    cs = [x - y for x, y in zip(TCR._cond_stats(), cs0)]
    r_enc = _res_enc(name, r)
    pre = TCR._frame_pre(fr) if cap else None
    ev = None
    if mode == "both":
        r_rs, cs_rs, ev = _rs_call_ev(name, a_rs, g)
        TCR._stat(name, "both_checked")
        a, b = _dumps(r_enc), _dumps(_rs_cmp(name, r_rs))
        if a != b or cs_rs != cs:
            TCR._stat(name, "both_mismatch")
            _dump_mismatch(name, a_rs, a, b, cs, cs_rs)
            raise AssertionError("theory outer both: %s が違う\n  a=%s\n  py=%s cs=%s\n  rs=%s cs=%s"
                                 % (name, _dumps(a_rs)[:2000], a[:1500], cs, b[:1500], cs_rs))
        _ev_check(name, gl, st0, ev)
        _gain_check(argd)
        r = _wrap_both_result(name, r, r_rs)
    if cap:
        r_cap = r_enc
        if name == "cp.curve_of_row" and r is not None:
            _CAP_IDS[0] += 1
            r.__dict__["_cap"] = "cv:%d" % _CAP_IDS[0]
            r_cap = {"d": [["id", {"obj": r._cap}], ["s", r_enc]]}
        _record4(name, a_cap, g, pre, r_cap, cs, ev)
    return r


def _dump_mismatch(name, a_rs, py, rs, cs, cs_rs):
    """`OPCG_THEORY_OUTER_DUMP=<file>` なら不一致の全文を書く（調べる用）。"""
    p = os.environ.get("OPCG_THEORY_OUTER_DUMP")
    if p:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(_dumps({"fn": name, "a": a_rs, "py": py, "rs": rs, "cs": cs, "cs_rs": cs_rs}) + chr(10))


def _outer_wrapper(name, orig):
    bind = TCR._binder(orig)
    gl = getattr(orig, "__globals__", None)

    @functools.wraps(orig)
    def w(*args, **kw):
        if TCR._DEPTH[0] > 0 or _OUTER["mode"] == "py" and not TCR._CFG["dir"]:
            return orig(*args, **kw)
        return _call_outer(name, orig, bind, args, kw, gl)
    w.__theory_core_orig__ = orig
    return w


def _rds_wrapper(orig):
    """`rule_don_solve`（核の中・記録だけ）: 財布は `_gain` 込みの dict で書く。"""
    bind = TCR._binder(orig)
    gl = orig.__globals__

    @functools.wraps(orig)
    def w(*args, **kw):
        if not TCR._CFG["dir"]:
            return orig(*args, **kw)
        argd = bind(args, kw)
        g = _g4(gl)
        a_cap = {"d": [[k, (RS.enc(dict(x)) if k == "actx" else _enc4(x, False))] for k, x in argd.items()]}
        fr = {"start": TCR._SEQ[0] + 1, "pre": [], "seen": set()}
        TCR._FRAMES.append(fr)
        cs0 = TCR._cond_stats()
        try:
            r = orig(*args, **kw)
        finally:
            TCR._pop_frame(fr)
        cs = [x - y for x, y in zip(TCR._cond_stats(), cs0)]
        _record4("cb.rule_don_solve", a_cap, g, TCR._frame_pre(fr), RS.enc(r), cs, None)
        return r
    w.__theory_core_orig__ = orig
    return w


_CAP_IDS = [0]


def _record4(name, a_enc, g, pre, r_enc, cs, ev):
    """`TCR._record` ＋ 計数の増分 `ev`（`both` の通しでは Rust の増分＝Python と同じと確かめた物・`py` の記録では `None`）。"""
    curve = name == "cp.curve_of_row" or name.startswith("cv.")
    TCR._stat(name, "calls")
    if not curve:
        # 曲線の行は重複を除かない（曲線ごとに記録の順に解き直す・1 つの曲線の中の覚え書きが履歴に依る＝E52）
        key = _dumps([a_enc, RS.enc(g), pre])
        seen = TCR._SEEN.setdefault(name, set())
        if key in seen:
            return
        seen.add(key)
    TCR._stat(name, "recorded")
    rec = {"fn": name, "a": a_enc, "g": RS.enc(g), "pre": pre, "r": r_enc, "cs": cs}
    if ev is not None:
        rec["ev"] = [[k, RS.enc(v)] for k, v in ev]
    TCR._write("cp.curve" if curve else name, rec)


class _RdcLog(TCR.LogDict):
    """`_RULE_DON_CACHE`（`get` も記録する）。"""

    def get(self, k, d=None):
        if dict.__contains__(self, k):
            return self[k]
        return d


def _install_rdc(ns):
    c = ns.get("_RULE_DON_CACHE")
    if c is not None and not isinstance(c, _RdcLog):
        ns["_RULE_DON_CACHE"] = _RdcLog("rdc", c)


#: (名前, モジュール, 属性)
OUTER = [
    ("cb.attacker_ctx", "crossing_bridge", "attacker_ctx"),
    ("cb.rule_don_plan_for", "crossing_bridge", "rule_don_plan_for"),
    ("cb.threshold_parts_side", "crossing_bridge", "threshold_parts_side"),
    ("cb.threshold_of_me_parts", "crossing_bridge", "threshold_of_me_parts"),
    ("cb.opp_blockers_of", "crossing_bridge", "opp_blockers_of"),
    ("cb.theory_slope_parts", "crossing_bridge", "theory_slope_parts"),
    ("cb.hand_groups", "crossing_bridge", "hand_groups"),
    ("cb.purse_series", "crossing_bridge", "purse_series"),
    ("cb.tau_grow", "crossing_bridge", "tau_grow"),
    ("hp.hand_items", "hand_plan", "hand_items"),
    ("hp.search_context", "hand_plan", "search_context"),
    ("tb.guard_hand_reading", "theory_bridge", "guard_hand_reading"),
    ("cp.curve_of_row", "cut_price", "curve_of_row"),
]


# ---------------------------------------------------------------------------------------------------------------
# 曲線の代理

def _g_cv(name):
    """曲線の呼び出しの文脈。`defending` は窓の `CUT_PRICER` を入れてから `ḡ` を読む＝その `ḡ` はまだ無い。Python も `L` を
    覚えた後の `ḡ` しか読めない（読めば再帰する）ので、`ḡ` だけは文脈なしで渡し、Rust は `L` が未定なら誤りにする（`nested`）。"""
    try:
        return TCR._g(), []
    except RuntimeError:
        if name != "cv.gbar":
            raise
        import theory_order as TO
        saved = (TO.CUT_PRICER, TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD)
        TO.CUT_PRICER, TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD = None, None, None
        try:
            g = TCR._g()
        finally:
            TO.CUT_PRICER, TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD = saved
        return g, [("nested", True)]


def _cv_call(rid, name, extra=()):
    g, more = _g_cv(name)
    a = {"d": [["cv", {"obj": rid}]] + [[k, RS.enc(v)] for k, v in list(extra) + more]}
    r, _cs, _ev = _rs_call_ev(name, a, g)
    return TCR._dec(r)


class _CurveMixin:
    def _rs_gbar(self):
        return _cv_call(self._rs, "cv.gbar")

    def _rs_L(self):
        return _cv_call(self._rs, "cv.L")

    def _rs_Lx(self, x):
        return _cv_call(self._rs, "cv.Lx", [("x", float(x))])

    def _rs_set_loss(self, keep, S):
        return _cv_call(self._rs, "cv.set_loss", [("keep", sorted(int(i) for i in keep)), ("S", sorted(int(i) for i in S))])

    def __del__(self):
        rid = self.__dict__.get("_rs")
        if rid is not None:
            try:
                TCR._rs_call("cv.free", {"d": [["cv", {"obj": rid}]]}, {})
            except Exception:                           # noqa: BLE001
                pass


_CLS = {}


def _mk_curve_classes():
    import cut_price as CP
    base = CP.CutCurve

    class RsCurve(_CurveMixin, base):
        """`rs`: Rust の曲線だけ（`valuer` は持たない）。"""

        def __init__(self, rid, s):
            self._rs = rid
            self.valuer = None
            self._n = int(s["n"])
            self.cand = list(s["cand"])
            self.n0 = len(self.cand)
            self.h0 = float(s["h0"])
            self.share = float(s["share"])
            self.mu = float(s["mu"])
            self.cids = list(s["cids"])
            self.reserve = None if s["reserve"] is None else float(s["reserve"])
            self._L = None
            self.xs_future = list(s["xs_future"])
            self.mlp = float(s["mlp"])

        @property
        def gbar(self):
            key = (self.reserve, CP.CUT_PRICE_MODE)
            got = self.__dict__.get("_gbar_memo")
            if got is not None and got[0] == key:
                return got[1]
            g = float(self._rs_gbar())
            self._gbar_memo = (key, g)
            return g

        def full(self):
            return frozenset(range(self._n))

        def L(self):
            if self._L is None:
                self._L = [float(x) for x in self._rs_L()]
            return self._L

        def Lx(self, x):
            return float(self._rs_Lx(x))

        def set_loss(self, keep, S):
            return float(self._rs_set_loss(frozenset(keep), frozenset(S) & frozenset(keep)))

    class BothCurve(_CurveMixin, base):
        """`both`: Python の曲線そのもの（`__class__` を差し替える）＋ Rust の曲線と突き合わせる。"""

        def _chk(self, what, a, b):
            if _dumps(RS.enc(a)) != _dumps(RS.enc(b)):
                TCR._stat("cv." + what, "both_mismatch")
                _dump_mismatch("cv." + what, None, _dumps(RS.enc(a)), _dumps(RS.enc(b)), None, None)
                raise AssertionError("theory outer both: 曲線の %s が違う py=%r rs=%r" % (what, a, b))
            TCR._stat("cv." + what, "both_checked")

        @staticmethod
        def _py(fn, *a):
            TCR._DEPTH[0] += 1
            try:
                return fn(*a)
            finally:
                TCR._DEPTH[0] -= 1

        def _run(self, what, fn, args, extra, rs):
            """Python で解き（深さ +1・記録の枠つき）、Rust と比べ、記録する。"""
            cap = bool(TCR._CFG["dir"]) and "_cap" in self.__dict__
            g, more = _g_cv("cv." + what)
            extra = list(extra) + more
            fr = {"start": TCR._SEQ[0] + 1, "pre": [], "seen": set()}
            TCR._FRAMES.append(fr)
            cs0 = TCR._cond_stats()
            TCR._DEPTH[0] += 1
            try:
                v = fn(*args)
            finally:
                TCR._DEPTH[0] -= 1
                TCR._pop_frame(fr)
            cs = [x - y for x, y in zip(TCR._cond_stats(), cs0)]
            self._chk(what, v, rs())
            if cap:
                a = {"d": [["cv", {"obj": self._cap}]] + [[k, RS.enc(x)] for k, x in extra]}
                _record4("cv." + what, a, g, TCR._frame_pre(fr), RS.enc(v), cs, None)
            return v

        @property
        def gbar(self):
            if TCR._DEPTH[0] > 0 or self.__dict__.get("_gbar_memo") is not None:
                return base.gbar.fget(self)
            return self._run("gbar", base.gbar.fget, (self,), [], self._rs_gbar)

        def L(self):
            if TCR._DEPTH[0] > 0 or self._L is not None:
                return base.L(self)
            return self._run("L", base.L, (self,), [], self._rs_L)

        def Lx(self, x):
            if TCR._DEPTH[0] > 0:
                return base.Lx(self, x)
            return self._run("Lx", base.Lx, (self, x), [("x", float(x))], lambda: self._rs_Lx(x))

        def set_loss(self, keep, S):
            if TCR._DEPTH[0] > 0:
                return base.set_loss(self, keep, S)
            kk, ss = sorted(int(i) for i in keep), sorted(int(i) for i in frozenset(S) & frozenset(keep))
            return self._run("set_loss", base.set_loss, (self, keep, S), [("keep", kk), ("S", ss)],
                             lambda: self._rs_set_loss(frozenset(keep), frozenset(S) & frozenset(keep)))

    _CLS["rs"], _CLS["both"] = RsCurve, BothCurve


# ---------------------------------------------------------------------------------------------------------------
# 取り付け

def install(mode):
    """段 4 の入口を差し替える（`theory_core_rs.install` が呼ぶ・核の差し替えの後）。"""
    import importlib
    _OUTER["mode"] = mode
    if mode == "py" and not TCR._CFG["dir"]:
        return
    for m in sorted({m for _n, m, _a in OUTER}):
        importlib.import_module(m)
    _mk_curve_classes()
    fns = {(m, a): getattr(sys.modules[m], a) for _n, m, a in OUTER}
    for name, m, a in OUTER:
        TCR._patch_everywhere(fns[(m, a)], _outer_wrapper(name, fns[(m, a)]))
    cb = sys.modules["crossing_bridge"]
    TCR._patch_everywhere(cb.rule_don_solve, _rds_wrapper(cb.rule_don_solve))
    if TCR._CFG["dir"]:
        _install_rdc(vars(cb))
        only = os.environ.get("OPCG_THEORY_CAPTURE_ONLY")
        if only == "outer":
            # 段 4 の行だけを書く（段 3 の核の行は段 3 の記録で確かめ済み・ディスクを食う）
            w0 = TCR._write
            keep = tuple(n for n, _m, _a in OUTER) + ("cb.rule_don_solve", "cv.", "cp.curve")

            def _w(name, rec, _w0=w0):
                if name.startswith(keep):
                    _w0(name, rec)
            TCR._write = _w
    if mode == "rs" and os.environ.get("OPCG_PLAN_STORE"):
        _rs_call_ev("store.open", {"d": [["path", os.environ["OPCG_PLAN_STORE"]]]}, {})
        atexit.register(_store_report)


def _store_report():
    try:
        r, _cs, _ev = _rs_call_ev("store.report", {"d": []}, {})
        if r is not None:
            print("plan_store_rs: %r" % (TCR._dec(r),), file=sys.stderr)
    except Exception:                                   # noqa: BLE001
        pass


def patch_main(ns, mod):
    """器を `__main__` として走らせるとき、器自身の写しの段 4 の入口を差し替える。"""
    if _OUTER["mode"] == "py" and not TCR._CFG["dir"]:
        return 0
    n = 0
    for name, m, a in OUTER:
        if m == mod and a in ns:
            orig = ns[a]
            w = _outer_wrapper(name, orig)
            for k, v in list(ns.items()):
                if v is orig:
                    ns[k] = w
                    n += 1
    if mod == "crossing_bridge" and "rule_don_solve" in ns:
        orig = ns["rule_don_solve"]
        w = _rds_wrapper(orig)
        for k, v in list(ns.items()):
            if v is orig:
                ns[k] = w
        if TCR._CFG["dir"]:
            _install_rdc(ns)
    return n
