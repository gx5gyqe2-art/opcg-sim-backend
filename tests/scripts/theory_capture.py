"""**理論の器の Rust 全移植・段 1（2026-10-06）**: 呼び出しの記録の器（`OPCG_THEORY_CAPTURE`）と、両方で解いて比べる運転。

本物の通し（8 器 × 実・合成の記録）の途中で、**移した葉の関数の入力と出力**をビットのまま書き出す。`cargo test` が
Python 無しでそれを Rust で解き直し、ビットで比べる（`rust/opcg_engine/src/theory/tests_leaves.rs`）。

使い方（器そのものは 1 字も変えない・`theory_capture_run.py` が器の前に `install()` してから器を `__main__` として走らせる）:

    OPCG_THEORY_CAPTURE=<dir> OPCG_THEORY_CAPTURE_SRC=w41 OPCG_THEORY_CAPTURE_TOOL=crossing_bridge \\
        python tests/scripts/theory_capture_run.py tests/scripts/crossing_bridge.py --in <rec> --limit-games 5 --json out.json
    # 両方で解いて比べる（違えば AssertionError・出力は Python のまま）
    OPCG_THEORY_BOTH=1 python tests/scripts/theory_capture_run.py ...

1 行 = `{"fn": 葉の名前, "a": 引数（名前つき・既定値込み）, "g": 読む大域, "s": 核の答え [[名前, 引数, 戻り], …], "r": 戻り}`
（記録の形は `theory_rs.enc`）。**同じ入力（型・ビットごと）は 1 度だけ**書く。

**核（段 3・まだ Python）に頼る葉**（`deck_refill.a_of` → `attack_value_don`、`leader_power_opp_turn` → `_state_of`・
`continuous_self_mods`、`CutCurve.L` → `JointValuer.value`）は、その呼び出しの引数と答えを `s` に順に残す
（Rust は同じ順・同じ引数で引く＝核の呼び方まで突き合わせる）。

**移植の間だけの道具**（段 7 で Python の理論と一緒に消す）。
"""
import atexit
import functools
import inspect
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_rs as RS  # noqa: E402

_STACK = []          # 記録中の枠（核の答えを受ける）
_SEEN = {}           # 名前 → 見た入力の鍵
_STATS = {}          # 名前 → 数
_FH = {}
_CFG = {}


class _Skip(Exception):
    pass


def _k(v):
    """型とビットごとの鍵（同じ入力を 1 度だけ書く）。"""
    if v is None or type(v) is bool or type(v) is str:
        return v if type(v) is not bool else ("b", v)
    if type(v) is int:
        return ("i", v)
    if type(v) is float:
        return ("f", v.hex())
    nm = RS._obj_name(v)
    if nm is not None:
        return ("o", nm)
    if type(v) is tuple:
        return ("t",) + tuple(_k(x) for x in v)
    if type(v) is list:
        return ("l",) + tuple(_k(x) for x in v)
    if type(v) is dict:
        if len(v) > 1000:
            RS.enc(v)                                  # 語彙なら名前を覚える（違えば例外）
            nm = RS._obj_name(v)
            if nm is not None:
                return ("o", nm)
        return ("d",) + tuple((_k(a), _k(b)) for a, b in v.items())
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return ("nd", a.dtype.str, a.shape, a.tobytes())
    if isinstance(v, np.generic):
        return ("np", v.dtype.str, v.tobytes())
    if isinstance(v, slice):
        return ("sl", v.start, v.stop, v.step)
    if isinstance(v, (set, frozenset)):
        return ("s",) + tuple(sorted(_k(x) for x in v))
    e = RS.enc(v)                                      # 名前の物（札の原本・カード表）か、符号化できなければ例外
    return ("e", json.dumps(e))


def _register_known():
    """大域の物（カード表・盤面の分布）の名前。"""
    import theory_order as TO
    bd = TO.load_opp_boards()
    RS.register_obj(bd, "opp_boards")
    for r, lst in bd.items():
        RS.register_obj(lst, "opp_boards:%d" % int(r))


def _cards_hook(v):
    t = type(v)
    return t.__name__ == "Cards" and t.__module__.endswith("plan_labels")


_orig_enc = RS.enc


def _enc(v):
    if _cards_hook(v):
        return {"obj": "Cards"}
    return _orig_enc(v)


RS.enc = _enc                     # `Cards` の物は名前で（Rust は大域のカード表を使う）
_orig_obj_name = RS._obj_name


def _obj_name(v):
    if _cards_hook(v):
        return "Cards"
    return _orig_obj_name(v)


RS._obj_name = _obj_name


def _stat(name, key, n=1):
    s = _STATS.setdefault(name, {})
    s[key] = s.get(key, 0) + n


def _write(name, rec):
    d = _CFG["dir"]
    fh = _FH.get(name)
    if fh is None:
        os.makedirs(d, exist_ok=True)
        fh = open(os.path.join(d, name + ".jsonl"), "a", encoding="utf-8")
        _FH[name] = fh
    fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def _check_both(name, rec):
    pay = json.dumps({"a": rec["a"], "g": rec["g"], "s": rec["s"]}, ensure_ascii=False, separators=(",", ":"))
    try:
        got = RS.engine().theory_leaf_call(name, pay)
    except Exception as e:                              # noqa: BLE001
        raise AssertionError("theory both: %s を Rust が解けない: %s" % (name, e))
    want = json.dumps(rec["r"], ensure_ascii=False, separators=(",", ":"))
    _stat(name, "both_checked")
    if got != want:
        _stat(name, "both_mismatch")
        raise AssertionError("theory both: %s が違う\n  a=%s\n  py=%s\n  rs=%s" % (name, pay[:2000], want[:500], got[:500]))


class Spec:
    def __init__(self, name, mod, attr, glob=(), subs=(), when=None, xform=None, kind="func", probe=None):
        self.name, self.mod, self.attr, self.probe = name, mod, attr, probe
        self.glob, self.subs, self.when, self.xform, self.kind = tuple(glob), frozenset(subs), when, xform, kind


def _binder(fn):
    sig = inspect.signature(fn)
    names = list(sig.parameters)
    defaults = {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}
    simple = all(p.kind == p.POSITIONAL_OR_KEYWORD for p in sig.parameters.values())

    def bind(args, kw):
        if simple and len(args) <= len(names):
            out = dict(zip(names, args))
            for k, v in kw.items():
                if k in out or k not in names:
                    raise TypeError("bind")
                out[k] = v
            for k in names:
                if k not in out:
                    if k not in defaults:
                        raise TypeError("bind")
                    out[k] = defaults[k]
            return {k: out[k] for k in names}
        ba = sig.bind(*args, **kw)
        ba.apply_defaults()
        return dict(ba.arguments)
    return bind


def _globals(spec):
    return {k: fn() for k, fn in spec.glob}


def _record(spec, argd, run):
    """入力を鍵にして 1 度だけ: 符号化 → 呼ぶ（核の答えを受ける枠つき）→ 書く（→ 両方の運転なら Rust と比べる）。"""
    name = spec.name
    _stat(name, "calls")
    try:
        g = _globals(spec)
        key = (_k(argd), _k(g))
    except Exception:                                   # noqa: BLE001
        _stat(name, "unencodable")
        return run()
    seen = _SEEN.setdefault(name, set())
    if key in seen:
        return run()
    if spec.probe is not None and spec.probe(argd):
        # 覚え書きから返る（本体が走らない＝核の答えが無い）＝記録しない。値を決める入力が最初と同じかは `probe` が数える（E33）。
        return run()
    try:
        a_enc, g_enc = RS.enc(argd), RS.enc(g)
    except Exception:                                   # noqa: BLE001
        _stat(name, "unencodable")
        return run()
    fr = {"want": spec.subs, "subs": []}
    _STACK.append(fr)
    try:
        r = run()
    finally:
        _STACK.pop()
    try:
        rec = {"fn": name, "a": a_enc, "g": g_enc, "s": fr["subs"], "r": RS.enc(r)}
    except Exception:                                   # noqa: BLE001
        _stat(name, "unencodable_result")
        return r
    seen.add(key)
    _stat(name, "recorded")
    if _CFG.get("dir"):
        _write(name, rec)
    if _CFG.get("both"):
        _check_both(name, rec)
    return r


def _func_wrapper(spec, orig):
    bind = _binder(orig)

    @functools.wraps(orig)
    def w(*args, **kw):
        try:
            argd = bind(args, kw)
            if spec.when is not None and not spec.when(argd):
                return orig(*args, **kw)
            if spec.xform is not None:
                argd = spec.xform(argd)
        except Exception:                               # noqa: BLE001
            _stat(spec.name, "unbound")
            return orig(*args, **kw)
        return _record(spec, argd, lambda: orig(*args, **kw))
    w.__theory_capture_orig__ = orig
    return w


def _sub_stub(name, orig):
    """核の呼び出しの記録（記録中の枠が欲しがるときだけ・答えはそのまま返す）。"""
    bind = _binder(orig)

    @functools.wraps(orig)
    def w(*args, **kw):
        if not _STACK or not any(name in fr["want"] for fr in _STACK):
            return orig(*args, **kw)
        r = orig(*args, **kw)
        try:
            e = [name, RS.enc(bind(args, kw)), RS.enc(r)]
        except Exception:                               # noqa: BLE001
            e = [name, {"d": []}, {"obj": "unencodable"}]
        for fr in _STACK:
            if name in fr["want"]:
                fr["subs"].append(e)
        return r
    w.__theory_capture_orig__ = orig
    return w


# ---------------------------------------------------------------------------------------------------------------
# 対象（段 2 の葉・既定の経路で走るもの）

def _g(mod, attr):
    def get():
        return getattr(sys.modules[mod], attr)
    return get


def _ffix():
    import effect_value as EV
    return bool(EV._ffix("attached_don_cond"))


def _cut_pricer_on():
    import theory_order as TO
    return TO.CUT_PRICER is not None


def _blockers_xform(a):
    ctx = a["ctx"]
    bodies = [{"blocker": b.get("blocker"), "is_rest": b.get("is_rest"), "power": b["power"], "nu": b["nu"]}
              for b in (ctx.get("opp_bodies") or ()) if b.get("blocker") and not b.get("is_rest")]
    return {"ctx": {"opp_bodies": bodies}}


def _items_xform(a):
    a = dict(a)
    a["items"] = [{"counter": it.get("counter"), "event": it.get("event"), "cost": it.get("cost")} for it in a["items"]]
    return a


def _boards_xform(a):
    import theory_order as TO
    a = dict(a)
    b = a.get("boards")
    if b is not None and RS._obj_name(b) is None and b is not TO._OPP_BOARDS:
        raise _Skip("盤面の分布が大域の表でない")
    return a


_A_OF_FIRST = {}     # 覚え書きの鍵 → 値を決める正確な入力（最初に本体が走った呼び出しの）


def _a_of_hit(a):
    """`deck_refill.a_of` の覚え書き（`_FLOW`）に当たるか（鍵は `a_of` と同じ式）。当たるなら、値を決める正確な入力
    （山・相手リーダーのパワー・ドンの枠・速攻・付与・Θ・μ・値段の文脈）が最初に入れたものと同じかを数える（E33）。"""
    import deck_refill as DR
    import theory_order as TO
    olp = float(a["opp_leader_power"])
    cap = None if a["don"] is None else int(round(float(a["don"])))
    key = (tuple(a["deck_ids"]), round(olp, 1), cap, bool(a["rush_only"])) + (() if a["with_don"] else ("bare",))
    if TO.CUT_PRICER is not None:
        key = (key + (TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD is not None)) if TO._cut_cache_ok() else None
    exact = (tuple(a["deck_ids"]), olp.hex(), cap, bool(a["rush_only"]), bool(a["with_don"]), _k(a["theta"]), _k(a["mu"]),
             TO.CUT_PRICER_KEY if TO.CUT_PRICER is not None else None,
             TO.CUT_TAKE_CARD if TO.CUT_PRICER is not None else None)
    if key is None:
        return False
    if key in DR._FLOW:
        _stat("dr.a_of", "memo_hit_same_value_inputs" if _A_OF_FIRST.get(key) == exact else "memo_hit_RISKY")
        return True
    _A_OF_FIRST[key] = exact
    return False


TO_ = "theory_order"
SPECS = [
    Spec("to.theta_take", TO_, "theta_take"),
    Spec("to.turn_weights", TO_, "turn_weights", glob=[("SURV_MODE", _g(TO_, "SURV_MODE"))]),
    Spec("to.surv_turns", TO_, "surv_turns", glob=[("SURV_MODE", _g(TO_, "SURV_MODE"))]),
    Spec("to.cbar_of", TO_, "cbar_of"),
    Spec("to.c_of", TO_, "c_of", glob=[("CBAR_MODE", _g(TO_, "CBAR_MODE"))]),
    Spec("to.slot_power", TO_, "slot_power"),
    Spec("to.slot_don", TO_, "slot_don"),
    Spec("to.incoming_x", TO_, "incoming_x"),
    Spec("to.count_blockers", TO_, "count_blockers"),
    Spec("to.opp_chars_of", TO_, "opp_chars_of"),
    Spec("to.own_attackers_of", TO_, "own_attackers_of"),
    Spec("to.hand_ids_of", TO_, "hand_ids_of"),
    Spec("to.card_identity", TO_, "card_identity"),
    Spec("to.ko_p_of", TO_, "ko_p_of"),
    Spec("to.power_band_of", TO_, "power_band_of"),
    Spec("to.shield_of", TO_, "shield_of"),
    Spec("to._attack_bound", TO_, "_attack_bound",
         glob=[("CUT_PRICER", _cut_pricer_on), ("CUT_TAKE_CARD", _g(TO_, "CUT_TAKE_CARD"))]),
    Spec("to.blockers_of", TO_, "blockers_of", xform=_blockers_xform),
    Spec("to.clock_scale", TO_, "clock_scale"),
    Spec("to.mover_shift", TO_, "mover_shift"),
    Spec("to.whole_clock_scale", TO_, "whole_clock_scale"),
    Spec("to._upper", TO_, "_upper"),
    Spec("to.whole_turn_race_prob", TO_, "whole_turn_race_prob"),
    Spec("to.prob_of_d", TO_, "prob_of_d", glob=[("W_ERR_MODE", _g(TO_, "W_ERR_MODE")), ("SIGMA_REL", _g(TO_, "SIGMA_REL")),
                                               ("SIGMA_D", _g(TO_, "SIGMA_D"))]),
    Spec("to.w_of_d", TO_, "w_of_d", glob=[("SIGMA_D", _g(TO_, "SIGMA_D"))]),
    Spec("to.state_factor", TO_, "state_factor", glob=[("W_MODE", _g(TO_, "W_MODE")), ("SIGMA_REL", _g(TO_, "SIGMA_REL")),
                                                       ("SIGMA_D", _g(TO_, "SIGMA_D")), ("W_BAR", _g(TO_, "W_BAR"))]),
    Spec("to.theta_of", TO_, "theta_of", glob=[("CBAR_MODE", _g(TO_, "CBAR_MODE"))]),
    Spec("to.board_theta", TO_, "board_theta", glob=[("CBAR_MODE", _g(TO_, "CBAR_MODE"))]),
    Spec("to.leader_power_opp_turn", TO_, "leader_power_opp_turn", subs=("tb._state_of", "ev.continuous_self_mods")),
    Spec("to.defender_power", TO_, "defender_power", glob=[("DEFENDER_POWER_MODE", _g(TO_, "DEFENDER_POWER_MODE"))],
         subs=("tb._state_of", "ev.continuous_self_mods")),
    Spec("to.attack_don_cost", TO_, "attack_don_cost", glob=[("ATTACK_DON_COST_MODE", _g(TO_, "ATTACK_DON_COST_MODE"))],
         when=lambda a: sys.modules[TO_].ATTACK_DON_COST_MODE == "off", xform=lambda a: {"k": a["k"]}),
    Spec("cb.opp_attackers_of", "crossing_bridge", "opp_attackers_of"),
    Spec("cb._opp_active_blockers", "crossing_bridge", "_opp_active_blockers"),
    Spec("cb._own_active_blockers", "crossing_bridge", "_own_active_blockers"),
    Spec("pr.nu_meas_of", "price_realised", "nu_meas_of"),
    Spec("pr.side_nu_meas", "price_realised", "side_nu_meas"),
    Spec("pr.don_stock", "price_realised", "don_stock"),
    Spec("pr.don_attached", "price_realised", "don_attached"),
    Spec("pr.quality_correction", "price_realised", "quality_correction"),
    Spec("dr.is_cuttable", "deck_refill", "is_cuttable"),
    Spec("dr.body_of", "deck_refill", "body_of"),
    Spec("dr.cut_share", "deck_refill", "cut_share"),
    Spec("dr.removal_harm", "deck_refill", "removal_harm", xform=_boards_xform),
    Spec("dr.card_effect_harm", "deck_refill", "card_effect_harm", xform=_boards_xform),
    Spec("dr.e_of", "deck_refill", "e_of", xform=_boards_xform),
    Spec("dr.a_of", "deck_refill", "a_of", subs=("to.attack_value_don", "to.attack_value"), probe=_a_of_hit),
    Spec("lr.avg_counter", "lethal_rule", "avg_counter", glob=[("AVG_COUNTER_MODE", _g("lethal_rule", "AVG_COUNTER_MODE"))]),
    Spec("lr.life_cards_as_counters", "lethal_rule", "life_cards_as_counters"),
    Spec("lr.stop_min_counter", "lethal_rule", "stop_min_counter"),
    Spec("lr.max_stops", "lethal_rule", "max_stops"),
    Spec("lr.attach_don", "lethal_rule", "attach_don"),
    Spec("hg.counter_of", "hand_guard", "counter_of"),
    Spec("hg.guard_cost_min_v", "hand_guard", "guard_cost_min_v"),
    Spec("hg.guard_value", "hand_guard", "guard_value"),
    Spec("hg.delta_g", "hand_guard", "delta_g"),
    Spec("hg.take_cost_of", "hand_guard", "take_cost_of"),
    Spec("ga.knapsack", "guard_afford", "knapsack"),
    Spec("ga.hand_counters", "guard_afford", "hand_counters"),
    Spec("hs.hand_ids", "hand_spend", "hand_ids"),
    Spec("cp.cuttable_indices", "cut_price", "cuttable_indices", xform=_items_xform),
    Spec("cp._multiset", "cut_price", "_multiset"),
    Spec("cp._minus", "cut_price", "_minus"),
    Spec("cp.sum_in", "cut_price", "sum_in"),
    Spec("cp.CutCurve.L", "cut_price", "CutCurve.L", kind="L", subs=("cp.valuer_value",)),
    Spec("cp.CutCurve.Lx", "cut_price", "CutCurve.Lx", kind="Lx"),
    Spec("cp.CutCurve.gbar", "cut_price", "CutCurve.gbar", kind="gbar"),
    Spec("cp.CutView.price", "cut_price", "CutView.price", kind="price"),
    Spec("cv.family_of", "condition_value", "family_of"),
    Spec("cv.compare", "condition_value", "compare"),
    Spec("cv.offset_threshold", "condition_value", "offset_threshold"),
    Spec("cv._int_value", "condition_value", "_int_value"),
    Spec("cv._mine", "condition_value", "_mine"),
    Spec("cv._decide_interval", "condition_value", "_decide_interval"),
    Spec("cv.has_don_requirement", "condition_value", "has_don_requirement"),
    Spec("cv.holds", "condition_value", "holds",
         glob=[("FFIX_ATTACHED_DON", _ffix), ("UNKNOWN_FACTOR", _g("condition_value", "UNKNOWN_FACTOR"))]),
    Spec("cv.factor", "condition_value", "factor",
         glob=[("FFIX_ATTACHED_DON", _ffix), ("UNKNOWN_FACTOR", _g("condition_value", "UNKNOWN_FACTOR"))]),
    Spec("cv._field_ids", "condition_value", "_field_ids"),
    Spec("cv.leader_info", "condition_value", "leader_info", when=lambda a: a["cards"] is None),
    Spec("cv.state_from_scalars", "condition_value", "state_from_scalars", when=lambda a: a["cards"] is None),
    Spec("sp.eligible_deck_cards", "search_price", "eligible_deck_cards", when=lambda a: a["st"] is None,
         xform=lambda a: {**a, "cards": None}),
]
#: 核（段 3）の答えを記録する口: (名前, モジュール, 属性, 差し替えるモジュール〔その葉が引く名前空間だけ〕)
SUBS = [
    ("to.attack_value_don", TO_, "attack_value_don", ("deck_refill",)),
    ("to.attack_value", TO_, "attack_value", ("deck_refill",)),
    ("tb._state_of", "theory_bridge", "_state_of", ("theory_bridge",)),
    ("ev.continuous_self_mods", "effect_value", "continuous_self_mods", ("effect_value",)),
]


def _patch_everywhere(orig, new):
    """`orig` を持つ全モジュールの名前を `new` に替える（`from X import f` の写しも）。"""
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


def _method_wrapper(spec, cls, attr):
    import cut_price as CP                          # noqa: F401
    orig = cls.__dict__[attr]
    if spec.kind == "L":
        def L(self):
            if self._L is not None:
                return orig(self)
            argd = {"n_slots": int(self.valuer.n), "cand": [int(c) for c in self.cand]}
            vv = self.valuer.value

            def run():
                def rec_value(keep, *a, **kw):
                    r = vv(keep, *a, **kw)
                    e = ["cp.valuer_value", RS.enc({"keep": sorted(int(i) for i in keep)}), RS.enc((float(r[0]),))]
                    for fr in _STACK:
                        if "cp.valuer_value" in fr["want"]:
                            fr["subs"].append(e)
                    return r
                self.valuer.value = rec_value
                try:
                    return orig(self)
                finally:
                    try:
                        del self.valuer.value
                    except AttributeError:
                        self.valuer.value = vv
            return _record(spec, argd, run)
        return L
    if spec.kind == "Lx":
        def Lx(self, x):
            r = orig(self, x)
            if self._L is not None:
                _record(spec, {"L": list(self._L), "n0": int(self.n0), "mu": float(self.mu), "x": x}, lambda: r)
            return r
        return Lx
    if spec.kind == "gbar":
        fget = orig.fget

        def gbar(self):
            r = fget(self)
            _record(spec, {"L": None if self._L is None else list(self._L), "n0": int(self.n0), "mu": float(self.mu),
                           "reserve": self.reserve}, lambda: r)
            return r
        return property(gbar)
    if spec.kind == "price":
        def price(self, k, mu=None):
            r = orig(self, k, mu)
            if getattr(self, "kind", None) == "avg":
                _record(spec, {"k": k, "kind": "avg", "gbar": float(self.curve.gbar)}, lambda: r)
            return r
        return price
    raise ValueError(spec.kind)


def install(capture_dir=None, both=None):
    """葉と核の口を差し替える。`capture_dir`／`both` は省略時に環境変数（`OPCG_THEORY_CAPTURE`・`OPCG_THEORY_BOTH`）。"""
    if _CFG.get("installed"):
        return
    cap = capture_dir if capture_dir is not None else os.environ.get("OPCG_THEORY_CAPTURE", "")
    if cap:
        src = os.environ.get("OPCG_THEORY_CAPTURE_SRC", "cap")
        tool = os.environ.get("OPCG_THEORY_CAPTURE_TOOL", "run")
        cap = os.path.join(cap, src, tool)
    _CFG["dir"] = cap
    _CFG["both"] = bool(int(os.environ.get("OPCG_THEORY_BOTH", "0"))) if both is None else bool(both)
    import importlib
    for mod in sorted({s.mod for s in SPECS} | {s[1] for s in SUBS}):
        importlib.import_module(mod)
    _register_known()
    if _CFG["both"]:
        RS.load_all()
    for name, mod, attr, where in SUBS:
        orig = getattr(sys.modules[mod], attr)
        stub = _sub_stub(name, orig)
        for w in where:
            m = sys.modules[w]
            for k, v in list(vars(m).items()):
                if v is orig:
                    setattr(m, k, stub)
    for spec in SPECS:
        m = sys.modules[spec.mod]
        if spec.kind == "func":
            orig = getattr(m, spec.attr)
            orig = getattr(orig, "__theory_capture_orig__", orig)
            _patch_everywhere(orig, _func_wrapper(spec, orig))
        else:
            cls_name, attr = spec.attr.split(".")
            cls = getattr(m, cls_name)
            setattr(cls, attr, _method_wrapper(spec, cls, attr))
    _CFG["installed"] = True
    atexit.register(_finish)


def stats():
    return {k: dict(v) for k, v in sorted(_STATS.items())}


def _finish():
    for fh in _FH.values():
        fh.close()
    _FH.clear()
    if _CFG.get("dir"):
        os.makedirs(_CFG["dir"], exist_ok=True)
        with open(os.path.join(_CFG["dir"], "_stats.json"), "w", encoding="utf-8") as fh:
            json.dump({"both": _CFG.get("both"), "stats": stats()}, fh, ensure_ascii=False, indent=1)
