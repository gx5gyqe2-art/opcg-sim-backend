"""**理論の器の Rust 全移植・段 3（2026-10-07）**: 値付けの核の切替（`OPCG_THEORY_CORE=py|rs|both`）と呼び出しの記録。

核（`theory_order`・`effect_value`・`hand_plan`・`hand_spend`・`search_price`・`hand_joint`・`theory_bridge.joint_valuer` に
またがる 1 つの相互再帰）の**外から呼ばれる入口**を差し替える:

* `py`（既定）: 何もしない（器は Python のまま）。
* `rs`: 入口を Rust の核（`opcg_engine.theory_core_call`）で解く。核の中の呼び出しは全部 Rust の中で閉じる。
* `both`: Python と Rust の両方で解き、**型とビットが 1 つでも違えば `AssertionError`**。戻りは Python のもの。
  Python の核の中で入れ子に呼ばれた入口は Python だけで解く（Rust は一番外の呼び出しだけを受ける＝覚え書きの状態が
  Python と同じ順で育つ）。

`OPCG_THEORY_CORE_CAPTURE=<dir>` なら入口と、核の中の主な関数（`attack_value_don`・`option_value`・`attack_stream`・
`search_value`・`card_gain`・`free_value`）の呼び出しを記録する（`<dir>/<src>/<tool>/<名前>.jsonl`）。
1 行＝`{"fn", "a", "g", "pre", "r", "cs"}`（記録の形 `theory_rs.enc`）。`pre`＝丸めた鍵の覚え書き（`option_value`・
`card_gain`・`a_of`）のうち、その呼び出しが**前から在った値を読んだ**もの＝`cargo test` は覚え書きを空にしてから
入れて解く（順に依らない再生・`rust/opcg_engine/src/theory/core/tests_core.rs`）。同じ入力（`fn`・`a`・`g`・`pre`）は 1 度だけ。

起動は `theory_capture_run.py`（器を `__main__` として走らせ、器自身が持つ入口〔`theory_bridge.joint_valuer`〕も差し替える）。
**移植の間だけの道具**（段 7 で Python の理論と一緒に消す）。
"""
import atexit
import functools
import inspect
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_rs as RS  # noqa: E402

_CFG = {"mode": "py", "dir": "", "installed": False}
_DEPTH = [0]          # Python の核の中を走っている深さ（>0 なら入口は Python だけ）
_SEQ = [0]            # 丸めた鍵の覚え書きへの書き込みの通し番号
_FRAMES = []          # 記録中の呼び出し（前もっての中身を集める）
_SEEN = {}
_FH = {}
_STATS = {}
_JV_IDS = [0]


def _stat(name, key, n=1):
    s = _STATS.setdefault(name, {})
    s[key] = s.get(key, 0) + n


# ---------------------------------------------------------------------------------------------------------------
# 符号化

def _hooks():
    """`Cards` と効果の木と核の物（`JointValuer` の代理）を名前で。"""
    import theory_capture  # noqa: F401  (`Cards` の物を名前で符号化する差し替えを入れる)
    import effect_value as EV
    RS.register_obj(EV._all_cards(), "effects")


def _enc(v):
    if isinstance(v, (_RsJV, _BothJV)):
        return {"obj": "jv:%d" % v._cap_id}
    return RS.enc(v)


def _enc_args(a):
    return {"d": [[k, _enc(x)] for k, x in a.items()]}


def _dumps(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _dec(v):
    if isinstance(v, list):
        return [_dec(x) for x in v]
    if isinstance(v, dict):
        if "obj" in v and len(v) == 1:
            nm = v["obj"]
            if nm == "Cards":
                return RS._cards()
            if nm == "effects":
                import effect_value as EV
                return EV._all_cards()
            raise ValueError("戻りに名前の物 %r" % nm)
        if "f" in v and len(v) == 1 or "t" in v and len(v) == 1:
            return RS.dec(v)
        if "d" in v and len(v) == 1:
            return {_dec_key(k): _dec(x) for k, x in v["d"]}
        return RS.dec(v)
    return v


def _dec_key(k):
    k = _dec(k)
    return tuple(k) if isinstance(k, list) else k


# ---------------------------------------------------------------------------------------------------------------
# 文脈（入口ごとに Rust へ渡す大域）

def _g():
    import cut_price as CP
    import effect_value as EV
    import search_price as SP
    import theory_order as TO
    import condition_value as CV
    pr = TO.CUT_PRICER
    gbar = None
    if pr is not None:
        view = (getattr(pr, "__defaults__", None) or (None,))[0]
        if view is None or getattr(view, "kind", None) != "avg":
            raise RuntimeError("theory core: CUT_PRICER が 1 枚あたり一定の窓（avg）でない")
        memo = view.curve.__dict__.get("_gbar_memo")
        if memo is None or memo[0] != (view.curve.reserve, CP.CUT_PRICE_MODE):
            raise RuntimeError("theory core: CUT_PRICER の ḡ がまだ決まっていない")
        gbar = float(memo[1])
    modes = {"NU_MODE": TO.NU_MODE, "SURV_MODE": TO.SURV_MODE, "OPTION_MODE": TO.OPTION_MODE, "CBAR_MODE": TO.CBAR_MODE,
             "SPEED_MEMO": str(bool(TO.SPEED_MEMO)), "ATTACK_ABILITY_MODE": TO.ATTACK_ABILITY_MODE,
             "PASSIVE_BODY_MODE": TO.PASSIVE_BODY_MODE, "DEFENDER_POWER_MODE": TO.DEFENDER_POWER_MODE,
             "ATTACK_DON_COST_MODE": TO.ATTACK_DON_COST_MODE,
             "COST_AFFORD_MODE": EV.COST_AFFORD_MODE,
             "F_PRICING_FIX": "all" if EV.F_PRICING_FIX == frozenset(EV.F_PRICING_FIXES) else ",".join(sorted(EV.F_PRICING_FIX)),
             "CUT_PRICE_MODE": CP.CUT_PRICE_MODE, "CUT_TAKE_MODE": CP.CUT_TAKE_MODE,
             "SEARCH_VALUE_MODE": SP.SEARCH_VALUE_MODE, "DECK_COUNTER_MODE": SP.DECK_COUNTER_MODE,
             "UNKNOWN_FACTOR": float(CV.UNKNOWN_FACTOR)}
    return {"CUT_PRICER": gbar, "CUT_PRICER_KEY": TO.CUT_PRICER_KEY, "CUT_TAKE_CARD": TO.CUT_TAKE_CARD,
            "CUT_OTHER_SIDE": int(TO.CUT_OTHER_SIDE), "OPTION_DEPTH": int(TO._OPTION_DEPTH),
            "FLOW_PRICING": EV.FLOW_PRICING, "OPAQUE_UPPER": bool(EV._OPAQUE_UPPER[0]), "modes": modes}


# ---------------------------------------------------------------------------------------------------------------
# 丸めた鍵の覚え書き（前もっての中身を集める）

class LogDict(dict):
    """`__getitem__` で読んだ値のうち、記録中の呼び出しより前に書かれたものを集める dict。"""

    def __init__(self, name, src):
        super().__init__(src)
        self.name = name
        self.wseq = {}

    def __setitem__(self, k, v):
        _SEQ[0] += 1
        self.wseq[k] = _SEQ[0]
        super().__setitem__(k, v)

    def __getitem__(self, k):
        v = super().__getitem__(k)
        w = self.wseq.get(k, 0)
        for fr in _FRAMES:
            if w < fr["start"] and (self.name, k) not in fr["seen"]:
                fr["seen"].add((self.name, k))
                fr["pre"].append([self.name, k, v])
        return v

    def clear(self):
        super().clear()
        self.wseq.clear()


def _install_logdicts():
    import deck_refill as DR
    import search_price as SP
    import theory_order as TO
    if not isinstance(TO._OPTION_CACHE, LogDict):
        TO._OPTION_CACHE = LogDict("option", TO._OPTION_CACHE)
    if not isinstance(SP._GAIN, LogDict):
        SP._GAIN = LogDict("gain", SP._GAIN)
    if not isinstance(DR._FLOW, LogDict):
        DR._FLOW = LogDict("flow", DR._FLOW)


# ---------------------------------------------------------------------------------------------------------------
# Rust を呼ぶ

def _engine():
    return RS.engine()


def _rs_call(name, a_enc, g):
    payload = _dumps({"d": [["a", a_enc], ["g", RS.enc(g)], ["pre", []]]})
    try:
        out = json.loads(_engine().theory_core_call(name, payload, False))
    except Exception as e:                              # noqa: BLE001
        raise RuntimeError("theory core: %s を Rust が解けない: %s" % (name, e))
    d = {k: x for k, x in out["d"]}
    return d["r"], [int(x) for x in d["cs"]]


def _cond_stats():
    import effect_value as EV
    return [EV.COND_STATS["true"], EV.COND_STATS["false"], EV.COND_STATS["unknown"]]


def _add_cond_stats(cs):
    import effect_value as EV
    EV.COND_STATS["true"] += cs[0]
    EV.COND_STATS["false"] += cs[1]
    EV.COND_STATS["unknown"] += cs[2]


def _write(name, rec):
    d = _CFG["dir"]
    fh = _FH.get(name)
    if fh is None:
        os.makedirs(d, exist_ok=True)
        fh = open(os.path.join(d, name.replace(".", "_") + ".jsonl"), "a", encoding="utf-8")
        _FH[name] = fh
    fh.write(_dumps(rec) + "\n")


def _record(name, a_enc, g, pre, r_enc, cs):
    try:
        key = _dumps([a_enc, RS.enc(g), pre])
    except Exception:                                   # noqa: BLE001
        _stat(name, "unencodable")
        return
    seen = _SEEN.setdefault(name, set())
    _stat(name, "calls")
    if key in seen:
        return
    seen.add(key)
    _stat(name, "recorded")
    _write(name, {"fn": name, "a": a_enc, "g": RS.enc(g), "pre": pre, "r": r_enc, "cs": cs})


def _binder(fn):
    sig = inspect.signature(fn)

    def bind(args, kw):
        ba = sig.bind(*args, **kw)
        ba.apply_defaults()
        return dict(ba.arguments)
    return bind


def _pop_frame(fr):
    """記録中の枠を同一性で外す（`list.remove` は `==` で比べる＝中身の同じ外側の枠を外してしまう）。"""
    for i in range(len(_FRAMES) - 1, -1, -1):
        if _FRAMES[i] is fr:
            del _FRAMES[i]
            return


def _frame_pre(fr):
    return [[n, RS.enc(k), RS.enc(v)] for n, k, v in fr["pre"]]


def _call_routed(name, orig, bind, args, kw, transform=None):
    """一番外の入口: py（記録だけ）／rs／both。"""
    mode = _CFG["mode"]
    cap = bool(_CFG["dir"])
    try:
        argd = bind(args, kw)
        g = _g()
        a_enc = _enc_args(transform(argd) if transform else argd)
    except RuntimeError:
        raise
    except Exception as e:                              # noqa: BLE001
        if mode == "rs":
            raise RuntimeError("theory core: %s の引数を符号化できない: %r" % (name, e))
        _stat(name, "unencodable")
        return orig(*args, **kw)
    if mode == "rs":
        r_enc, cs = _rs_call(name, a_enc, g)
        _add_cond_stats(cs)
        _stat(name, "rs_calls")
        return _dec(r_enc)
    fr = {"start": _SEQ[0] + 1, "pre": [], "seen": set()}
    _FRAMES.append(fr)
    cs0 = _cond_stats()
    _DEPTH[0] += 1
    try:
        r = orig(*args, **kw)
    finally:
        _DEPTH[0] -= 1
        _pop_frame(fr)
    cs = [x - y for x, y in zip(_cond_stats(), cs0)]
    r_enc = _enc(r)
    if cap:
        _record(name, a_enc, g, _frame_pre(fr), r_enc, cs)
    if mode == "both":
        r_rs, cs_rs = _rs_call(name, a_enc, g)
        _stat(name, "both_checked")
        if _dumps(r_rs) != _dumps(r_enc) or cs_rs != cs:
            _stat(name, "both_mismatch")
            raise AssertionError("theory core both: %s が違う\n  a=%s\n  py=%s cs=%s\n  rs=%s cs=%s"
                                 % (name, _dumps(a_enc)[:3000], _dumps(r_enc)[:800], cs, _dumps(r_rs)[:800], cs_rs))
    return r


def _call_internal(name, orig, bind, args, kw):
    """核の中の関数（記録だけ・切り替えない）。"""
    try:
        argd = bind(args, kw)
        g = _g()
        a_enc = _enc_args(argd)
    except Exception:                                   # noqa: BLE001
        _stat(name, "unencodable")
        return orig(*args, **kw)
    fr = {"start": _SEQ[0] + 1, "pre": [], "seen": set()}
    _FRAMES.append(fr)
    cs0 = _cond_stats()
    try:
        r = orig(*args, **kw)
    finally:
        _pop_frame(fr)
    cs = [x - y for x, y in zip(_cond_stats(), cs0)]
    try:
        _record(name, a_enc, g, _frame_pre(fr), _enc(r), cs)
    except Exception:                                   # noqa: BLE001
        _stat(name, "unencodable_result")
    return r


def _wrapper(name, orig, bind_fn=None, transform=None):
    bind = _binder(bind_fn or orig)

    @functools.wraps(orig)
    def w(*args, **kw):
        if _DEPTH[0] > 0 or _CFG["mode"] == "py" and not _CFG["dir"]:
            return orig(*args, **kw)
        return _call_routed(name, orig, bind, args, kw, transform)
    w.__theory_core_orig__ = orig
    return w


def _internal_wrapper(name, orig):
    bind = _binder(orig)

    @functools.wraps(orig)
    def w(*args, **kw):
        if not _CFG["dir"] or _DEPTH[0] == 0 and _CFG["mode"] == "rs":
            return orig(*args, **kw)
        return _call_internal(name, orig, bind, args, kw)
    w.__theory_core_orig__ = orig
    return w


# ---------------------------------------------------------------------------------------------------------------
# JointValuer の代理

class _RsJV:
    """Rust の `JointValuer`（`rs`）。"""

    def __init__(self, rid, n, cap_id):
        self._rid, self.n, self._cap_id = rid, n, cap_id

    def _call(self, name, keep):
        a = {"d": [["jv", {"obj": self._rid}], ["keep", None if keep is None else sorted(int(i) for i in keep)]]}
        r, _cs = _rs_call(name, a, _g())
        return _dec(r)

    def value(self, keep=None):
        return self._call("jv.value", keep)

    def loss(self, S):
        full = frozenset(range(self.n))
        return max(0.0, self.value(full)[0] - self.value(full - frozenset(S))[0])

    def __del__(self):
        try:
            _rs_call("jv.free", {"d": [["jv", {"obj": self._rid}]]}, {})
        except Exception:                               # noqa: BLE001
            pass


class _BothJV:
    """Python の `JointValuer` と Rust の写しを並べて持つ（`both`／記録）。"""

    def __init__(self, py, rid, cap_id):
        self._py, self._rid, self._cap_id = py, rid, cap_id
        self.n = py.n

    def value(self, keep=None):
        k = frozenset(range(self.n)) if keep is None else frozenset(keep)
        a_enc = {"d": [["jv", {"obj": "jv:%d" % self._cap_id}], ["keep", sorted(int(i) for i in k)]]}
        g = _g()
        fr = {"start": _SEQ[0] + 1, "pre": [], "seen": set()}
        _FRAMES.append(fr)
        cs0 = _cond_stats()
        _DEPTH[0] += 1
        try:
            r = self._py.value(k)
        finally:
            _DEPTH[0] -= 1
            _pop_frame(fr)
        cs = [x - y for x, y in zip(_cond_stats(), cs0)]
        r_enc = _enc(r)
        if _CFG["dir"]:
            _record("jv.value", a_enc, g, _frame_pre(fr), r_enc, cs)
        if _CFG["mode"] == "both":
            a_rs = {"d": [["jv", {"obj": self._rid}], ["keep", sorted(int(i) for i in k)]]}
            r_rs, cs_rs = _rs_call("jv.value", a_rs, g)
            _stat("jv.value", "both_checked")
            if _dumps(r_rs) != _dumps(r_enc) or cs_rs != cs:
                _stat("jv.value", "both_mismatch")
                raise AssertionError("theory core both: jv.value が違う keep=%s py=%s rs=%s" % (sorted(k), r_enc, r_rs))
        return r

    def loss(self, S):
        full = frozenset(range(self.n))
        return max(0.0, self.value(full)[0] - self.value(full - frozenset(S))[0])


def _hand_enc(hand):
    return {k: v for k, v in hand.items() if k != "_joint"}


def _joint_valuer_wrapper(orig):
    def w(hand):
        got = hand.get("_joint")
        if got is not None and got[0] is hand.get("inflow") and got[1] is hand["slots"]:
            return got[2]                                           # 同じ読みの使い回し（Python と同じ規則）
        if _DEPTH[0] > 0 or _CFG["mode"] == "py" and not _CFG["dir"]:
            return orig(hand)
        mode = _CFG["mode"]
        g = _g()
        a_enc = {"d": [["hand", RS.enc(_hand_enc(hand))]]}
        _JV_IDS[0] += 1
        cap_id = _JV_IDS[0]
        rid = None
        if mode in ("rs", "both"):
            r, _cs = _rs_call("tb.joint_valuer", a_enc, g)
            rid = r["obj"]
        if _CFG["dir"]:
            _write("tb.joint_valuer", {"fn": "tb.joint_valuer", "a": a_enc, "g": RS.enc(g), "pre": [],
                                       "r": {"obj": "jv:%d" % cap_id}, "cs": [0, 0, 0]})
            _stat("tb.joint_valuer", "recorded")
        if mode == "rs":
            prox = _RsJV(rid, len(hand["slots"]), cap_id)
        else:
            _DEPTH[0] += 1
            try:
                py = orig(hand)
            finally:
                _DEPTH[0] -= 1
            prox = _BothJV(py, rid, cap_id)
        hand["_joint"] = (hand.get("inflow"), hand["slots"], prox)
        return prox
    w.__theory_core_orig__ = orig
    return w


# ---------------------------------------------------------------------------------------------------------------
# 対象

def _nu_bind():
    import theory_order as TO
    return TO.nu_of


#: (名前, モジュール, 属性, 引数の名前を読む関数〔省略＝その関数〕)
ROUTED = [
    ("to.attack_value", "theory_order", "attack_value", None),
    ("to.nu_of", "theory_order", "nu_of", None),
    ("to._nu_of_other_side", "theory_order", "_nu_of_other_side", "nu_of"),
    ("to._don_cost_total", "theory_order", "_don_cost_total", None),
    ("ev.card_value", "effect_value", "card_value", None),
    ("ev.continuous_self_mods", "effect_value", "continuous_self_mods", None),
    ("hs.use_value", "hand_spend", "use_value", None),
    ("hp.apply_inflow", "hand_plan", "apply_inflow", None),
    ("dr.a_of", "deck_refill", "a_of", None),
]
#: 核の中の関数（記録だけ）
INTERNAL = [
    ("to.attack_value_don", "theory_order", "attack_value_don"),
    ("to.option_value", "theory_order", "option_value"),
    ("to.attack_stream", "theory_order", "attack_stream"),
    ("sp.search_value", "search_price", "search_value"),
    ("sp.card_gain", "search_price", "card_gain"),
    ("hs.free_value", "hand_spend", "free_value"),
]


def _patch_everywhere(orig, new):
    n = 0
    for m in list(sys.modules.values()):
        d = getattr(m, "__dict__", None)
        if not d:
            continue
        for k, v in list(d.items()):
            if v is orig:
                d[k] = new
                n += 1
    return n


def _apply_card_value_args(a):
    # `card_value(..., cards=...)` の `cards` は効果の木（名前で）。触らずに渡す。
    return a


def make_wrappers(namespace_fns):
    """`namespace_fns`＝{(モジュール, 属性): 関数}（器自身の `__main__` の写しを差し替えるため）→ {同じ鍵: 包み}。"""
    import importlib
    out = {}
    for name, mod, attr, bind_attr in ROUTED:
        orig = namespace_fns.get((mod, attr))
        if orig is None:
            continue
        bfn = getattr(importlib.import_module(mod), bind_attr) if bind_attr else None
        bfn = getattr(bfn, "__theory_core_orig__", bfn)
        out[(mod, attr)] = _wrapper(name, orig, bfn)
    for name, mod, attr in INTERNAL:
        orig = namespace_fns.get((mod, attr))
        if orig is not None:
            out[(mod, attr)] = _internal_wrapper(name, orig)
    jv = namespace_fns.get(("theory_bridge", "joint_valuer"))
    if jv is not None:
        out[("theory_bridge", "joint_valuer")] = _joint_valuer_wrapper(jv)
    return out


def install(mode=None, capture_dir=None):
    """入口を差し替える（`mode`／`capture_dir` は省略時に環境変数 `OPCG_THEORY_CORE`・`OPCG_THEORY_CORE_CAPTURE`）。"""
    if _CFG["installed"]:
        return
    mode = mode if mode is not None else os.environ.get("OPCG_THEORY_CORE", "py")
    if mode not in ("py", "rs", "both"):
        raise ValueError("OPCG_THEORY_CORE は py／rs／both")
    outer = os.environ.get("OPCG_THEORY_OUTER") or mode
    if outer not in ("py", "rs", "both"):
        raise ValueError("OPCG_THEORY_OUTER は py／rs／both")
    cap = capture_dir if capture_dir is not None else os.environ.get("OPCG_THEORY_CORE_CAPTURE", "")
    if cap:
        cap = os.path.join(cap, os.environ.get("OPCG_THEORY_CAPTURE_SRC", "cap"), os.environ.get("OPCG_THEORY_CAPTURE_TOOL", "run"))
    _CFG.update(mode=mode, dir=cap)
    if mode == "py" and outer == "py" and not cap:
        _CFG["installed"] = True
        return
    import importlib
    mods = sorted({m for _n, m, _a, _b in ROUTED} | {m for _n, m, _a in INTERNAL} | {"theory_bridge", "cut_price"})
    for m in mods:
        importlib.import_module(m)
    _hooks()
    _install_logdicts()
    if mode in ("rs", "both") or outer in ("rs", "both"):
        RS.load_all()
        RS.load_effects()
    fns = {}
    for _n, m, a, _b in ROUTED:
        fns[(m, a)] = getattr(sys.modules[m], a)
    for _n, m, a in INTERNAL:
        fns[(m, a)] = getattr(sys.modules[m], a)
    fns[("theory_bridge", "joint_valuer")] = sys.modules["theory_bridge"].joint_valuer
    for key, w in make_wrappers(fns).items():
        _patch_everywhere(fns[key], w)
    import theory_outer_rs as TOR                       # 段 4（守る側の外側と耐久）
    TOR.install(outer)
    _CFG["installed"] = True
    atexit.register(_finish)


def patch_main(ns, tool_path):
    """器を `__main__` として走らせるとき、器自身の写し（`ns`）が持つ入口を差し替える（`theory_capture_run.py`）。"""
    import theory_outer_rs as TOR
    if not _CFG["installed"] or (_CFG["mode"] == "py" and TOR._OUTER["mode"] == "py" and not _CFG["dir"]):
        return 0
    mod = os.path.splitext(os.path.basename(tool_path))[0]
    n0 = TOR.patch_main(ns, mod)
    fns = {}
    for _n, m, a, _b in ROUTED:
        if m == mod and a in ns:
            fns[(m, a)] = ns[a]
    for _n, m, a in INTERNAL:
        if m == mod and a in ns:
            fns[(m, a)] = ns[a]
    if mod == "theory_bridge" and "joint_valuer" in ns:
        fns[("theory_bridge", "joint_valuer")] = ns["joint_valuer"]
    n = n0
    for key, w in make_wrappers(fns).items():
        orig = fns[key]
        for k, v in list(ns.items()):
            if v is orig:
                ns[k] = w
                n += 1
    return n


def stats():
    return {k: dict(v) for k, v in sorted(_STATS.items())}


def _finish():
    sp = os.environ.get("OPCG_THEORY_CORE_STATS")
    if sp:
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"mode": _CFG["mode"], "stats": stats()}, fh, ensure_ascii=False, indent=1)
    for fh in _FH.values():
        fh.close()
    _FH.clear()
    if _CFG["dir"]:
        os.makedirs(_CFG["dir"], exist_ok=True)
        with open(os.path.join(_CFG["dir"], "_stats.json"), "w", encoding="utf-8") as fh:
            json.dump({"mode": _CFG["mode"], "stats": stats()}, fh, ensure_ascii=False, indent=1)
