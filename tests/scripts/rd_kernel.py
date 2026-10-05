"""**Rust 化・第 1 段／第 2a 段（2026-10-05）**: `rule_don` の守る側の計算（動的計画・段ごとの状態の数え方・地平の選び方）と、
**第 2a 段**で計画の列挙と採点・歩き（`rules_sched`・`walk_crossing`）・地平の試行のループを
Rust の核（`rust/opcg_engine/src/theory/`）で回すための Python 側の継ぎ目。

**値は 1 ビットも変えない**（設計: `docs/reports/2026-10-05_rust_kernel_stage1.md`・正本のモデルは Python の
`crossing_bridge._rule_guard_plan_ex` と、その原文の写し `tests/harness/rule_don_ref.py`）。

切替は環境変数 `OPCG_RD_KERNEL`（`crossing_bridge` が import 時に 1 回読む・`set_mode` でテストが切り替える）:

* `py`   — 今の Python の解き方（**既定**。Rust の一致が実測で固まるまで）。
* `rs`   — Rust の核。wheel が無い／古い／入口の版が違うと**落ちる**（黙って Python に戻らない）。
* `auto` — wheel が新しければ Rust・無ければ Python（stderr に 1 回だけ知らせる）。
* `both` — 両方で解いて**ビットで比べ**、違えば例外（Python の結果を返す）。測定の通し（慣らし運転）用。
* `ref`  — 速くする前の解き方の原文（`rule_don_ref.solve`）。遅い・1 問の調べ直し用。

継ぎ目は `crossing_bridge.py` の末尾の 1 行（`rd_kernel.install(globals())`）だけで、解き方の関数の原文は 1 文字も
変えない（`functools.wraps` で包むので `inspect.getsource` は元の原文を返す＝版の指紋のテストはそのまま通る）。
包むのは 3 つ: `_rule_guard_plan_ex`（動的計画）と `_ex_fit_horizon`（段ごとの数え方と地平の選び方）と
**`rule_don_solve`（第 2a 段: `rs` なら覚え書きと出す札の組〔`_rule_don_masks`＝`rules_steps` は Python のまま・第 2b 段〕と
最初の地平〔`model_horizon`〕だけ Python で作り、試行のループから先を `opcg_engine.rd_solve` 1 回で解く）**。
計画の辞書（`a_time`・`theta_parts` ほか）は Python が `_rule_don_solve` と同じ順・同じ式で組み立てる。

**古い wheel の検出**: `make test` は wheel を作り直さない。Rust 側が `(入口の版, src/theory の原文のハッシュ)` を
返し、ここで版と（ソースの木が有れば）ハッシュを突き合わせる。違えば `rs`／`both` は落ち、`auto` は Python へ。

**計画の覚え書きの鍵**: `kernel_tag()`（`py` か `rs:<版>:<ハッシュ>`）を `plan_store` の鍵に入れる——Rust の核の誤りが
Python の作った覚え書きを汚さず、核を直せば別の鍵になる。
"""
import atexit
import functools
import glob
import json
import os
import sys

#: Rust 側の入口の版（`theory::API`）と一致させる。入口の形を変えたら両方上げる。
RD_KERNEL_API = 2
MODES = ("py", "rs", "auto", "both", "ref")

MODE = os.environ.get("OPCG_RD_KERNEL", "py") or "py"
if MODE not in MODES:
    raise SystemExit("OPCG_RD_KERNEL は %s のどれか（今: %r）" % ("/".join(MODES), MODE))

STATS = {"calls": 0, "rs_calls": 0, "both_checked": 0, "fit_calls": 0, "solve_calls": 0, "solve_checked": 0}
_G = {}              # crossing_bridge の大域の辞書（install が差し替える・生きた辞書）
_ORIG = {}           # 元の関数
_WARNED = [False]
_STATUS = []         # [(ok, reason, module)] の 1 要素（1 回だけ調べる）
_ATT = {"memo": None, "d": None}
_GLOB = {"d": None}


# ----------------------------------------------------------------------------------------------
# wheel の確認

def source_hash(root=None):
    """`rust/opcg_engine/src/theory/**/*.rs` の原文のハッシュ（`build.rs` と同じ式: FNV-1a 64・相対パス順）。木が無ければ None。"""
    base = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rust", "opcg_engine",
                                "src", "theory")
    base = os.path.abspath(base)
    if not os.path.isdir(base):
        return None
    files = sorted(os.path.relpath(p, base).replace(os.sep, "/")
                   for p in glob.glob(os.path.join(base, "**", "*.rs"), recursive=True))
    h = 0xcbf29ce484222325
    for rel in files:
        with open(os.path.join(base, rel), "rb") as fh:
            body = fh.read()
        for b in rel.encode() + b"\0" + body + b"\0":
            h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def status():
    """`(ok, 理由, opcg_engine)`。結果は 1 回だけ調べて覚える。"""
    if _STATUS:
        return _STATUS[0]
    try:
        import opcg_engine as E
    except Exception as e:                                   # noqa: BLE001
        res = (False, "opcg_engine が import できない: %r" % (e,), None)
    else:
        try:
            api, h = E.rd_kernel_version()
        except AttributeError:
            res = (False, "wheel が古い（rd_kernel_version が無い）: `make rust-develop` で作り直す", None)
        else:
            src = source_hash()
            if api != RD_KERNEL_API:
                res = (False, "入口の版が違う（wheel=%s・期待=%s）: `make rust-develop`" % (api, RD_KERNEL_API), None)
            elif src is not None and src != h:
                res = (False, "wheel が古い（src/theory のハッシュ wheel=%s・ソース=%s）: `make rust-develop`" % (h, src), None)
            else:
                res = (True, "ok", E)
    _STATUS.append(res)
    return res


def engine():
    """Rust の核（`opcg_engine`）。使えなければ理由つきで落ちる。"""
    ok, why, E = status()
    if not ok:
        raise RuntimeError("Rust の核が使えない: " + why)
    return E


def set_mode(m):
    """テスト・測定が切り替える（`rs`／`both` は wheel が無いとここで落ちる）。"""
    global MODE
    if m not in MODES:
        raise ValueError(m)
    if m in ("rs", "both"):
        engine()
    MODE = m


def _mode():
    """今の呼び出しで使う経路: `py`／`rs`／`both`（`auto` と `ref` はここで解く）。"""
    m = MODE
    if m == "auto":
        ok, why, _E = status()
        if ok:
            return "rs"
        if not _WARNED[0]:
            _WARNED[0] = True
            print("rd_kernel: Rust の核を使えない（%s）→ Python で解く" % why, file=sys.stderr)
        return "py"
    return "py" if m == "ref" else m


def kernel_tag():
    """計画の覚え書きの鍵に入れる「どの解き方か」。Python の結果を返す経路（`py`・`both`・`ref`）は `py`。"""
    if _mode() == "rs":
        _ok, _why, E = status()
        api, h = E.rd_kernel_version()
        return "rs:%s:%s" % (api, h)
    return "py"


# ----------------------------------------------------------------------------------------------
# 守る側の動的計画

def _defender(g=None):
    used = (g or _G)["_EX_USED"]
    memo = used["memo"]
    E = engine()
    if memo is None:                       # 窓・テスト・感度の測定: 呼び出しをまたいで覚える（予算は無い）
        d = _GLOB["d"]
        if d is None or d.n_states() > 600000:
            d = _GLOB["d"] = E.RdDefender(None)
        return d
    if _ATT["memo"] is not memo:           # 試行ごとに空の覚え書きで始める（Python の `{}` と同じ単位）
        _ATT["memo"] = memo
        _ATT["d"] = E.RdDefender(used["limit"])
    return _ATT["d"]


def reset_global():
    """窓・テスト用の覚え書き（呼び出しをまたぐ）を捨てる（値は変わらない）。"""
    _GLOB["d"] = None


def _fl(seq):
    return [float(x) for x in seq or ()]


def guard_rs(cards, don, xs_first, seq, blk_margins, life, turns, life_types, lam, lam_net, mu, olp, mlp,
             rest_blk=(), arrive_blk=(), draw_types=(), defender=None, g=None):
    """Rust の核で `_rule_guard_plan_ex` と同じ辞書を返す。予算を超えたら `None`。"""
    g = g or _G
    d = defender if defender is not None else _defender(g)
    nu_fn = g["nu_meas_of"]
    nu = {}
    for m in list(blk_margins or ()) + list(rest_blk or ()) + list(arrive_blk or ()):
        m = float(m)
        if m not in nu:
            nu[m] = float(nu_fn(m + olp, mlp))
    r = d.solve([(float(c), float(dd)) for c, dd in cards or ()], float(don), _fl(xs_first),
                [_fl(s) for s in seq], _fl(blk_margins), float(life), None if turns is None else int(turns),
                [(float(c), float(dd), float(p)) for c, dd, p in life_types or ()],
                [(float(c), float(dd), float(p)) for c, dd, p in draw_types or ()],
                float(lam), float(lam_net), float(mu), float(olp), float(mlp), _fl(rest_blk), _fl(arrive_blk),
                list(nu.items()), float(g["PWR_EPS"]), float(g["_FEQ"]))
    if r is None:
        return None
    cut, st, alive, prev, harms, theta, nu_all = r
    if not (blk_margins or rest_blk or arrive_blk):
        nu_all = 0                         # Python の `sum(空)` は整数の 0（辞書の `repr` が違う）
    return {"cut": cut, "stopped": st, "alive": alive, "prevented": prev, "harms": tuple(harms), "theta": theta,
            "nu_all": nu_all}


def _mismatch(what, a, b):
    raise AssertionError("rd_kernel: Rust と Python の結果が違う（%s）\n  py=%r\n  rs=%r" % (what, a, b))


def _wrap_guard(orig, g):
    @functools.wraps(orig)
    def wrapped(cards, don, xs_first, seq, blk_margins, life, turns, life_types, lam, lam_net, mu, olp, mlp,
                rest_blk=(), arrive_blk=(), draw_types=()):
        m = _mode()
        if m == "py":
            return orig(cards, don, xs_first, seq, blk_margins, life, turns, life_types, lam, lam_net, mu, olp, mlp,
                        rest_blk, arrive_blk, draw_types)
        STATS["calls"] += 1
        args = (cards, don, xs_first, seq, blk_margins, life, turns, life_types, lam, lam_net, mu, olp, mlp,
                rest_blk, arrive_blk, draw_types)
        if m == "rs":
            STATS["rs_calls"] += 1
            out = guard_rs(*args, g=g)
            if out is None:
                raise g["_ModelBudget"]()
            return out
        # both: Rust を先に（失敗しても Python の覚え書きには触らない）→ Python → ビットで比べる
        d = _defender(g)
        rs = guard_rs(*args, defender=d, g=g)
        used = g["_EX_USED"]
        try:
            py = orig(*args)
        except g["_ModelBudget"]:
            if rs is not None:
                _mismatch("Python は予算超え・Rust は収まった", "budget", rs)
            if used["memo"] is not None and d.n_states() != len(used["memo"]):
                _mismatch("予算超えの時点の状態の数", len(used["memo"]), d.n_states())
            raise
        STATS["both_checked"] += 1
        if rs is None:
            _mismatch("Rust は予算超え・Python は収まった", py, "budget")
        if repr(py) != repr(rs):
            _mismatch("動的計画の結果", py, rs)
        if used["memo"] is not None and d.n_states() != len(used["memo"]):
            _mismatch("状態の数", len(used["memo"]), d.n_states())
        return py
    return wrapped


# ----------------------------------------------------------------------------------------------
# 段ごとの数え方・地平の選び方

def count_layers(cards, don, blk, life, life_types, rest_blk, arrive_blk, draw_types, roots, cap, lim, g=None):
    """Rust の `_ex_count_layers`（`roots` は `[(今のターンの攻撃, 並び)]`）。"""
    E = engine()
    return E.rd_count_layers(
        [(float(c), float(d)) for c, d in cards or ()], float(don), _fl(blk), float(life),
        [(float(c), float(d), float(p)) for c, d, p in life_types or ()], _fl(rest_blk), _fl(arrive_blk),
        [(float(c), float(d), float(p)) for c, d, p in draw_types or ()],
        [(_fl(xf), [_fl(s) for s in ls]) for xf, ls in roots], int(cap), int(lim), float((g or _G)["PWR_EPS"]))


def _wrap_fit(orig, g):
    @functools.wraps(orig)
    def wrapped(cards_d, don_d, blk, life, actx, life_types, draw_types, arrive, masks, h_fail, lim):
        m = _mode()
        if m == "py":
            return orig(cards_d, don_d, blk, life, actx, life_types, draw_types, arrive, masks, h_fail, lim)
        STATS["fit_calls"] += 1
        roots = []
        for mk in masks:
            for xf in g["_mask_firsts"](mk, actx):
                roots.append((xf, mk["later_seq"]))
        sizes = count_layers(cards_d, don_d, blk, life, life_types, tuple(actx.get("rest_blk") or ()), arrive,
                             draw_types, roots, h_fail, lim, g=g)
        fit = engine().rd_fit_horizon(sizes, int(lim), int(h_fail))
        if m == "both":
            py = orig(cards_d, don_d, blk, life, actx, life_types, draw_types, arrive, masks, h_fail, lim)
            if py != fit:
                _mismatch("選んだ地平", py, fit)
            return py
        return fit
    return wrapped


# ----------------------------------------------------------------------------------------------
# 取り込み（テストの金型）: `OPCG_RD_CAPTURE=<jsonl>` が有れば解いた問題を書き出す

def enc(v):
    """ビットを保つ符号化: 浮動小数は 16 桁の 16 進・組と辞書は型を残す（`dec` で戻る）。"""
    import struct
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return {"f": "%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]}
    if isinstance(v, tuple):
        return {"t": [enc(x) for x in v]}
    if isinstance(v, list):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {"d": [[enc(k), enc(x)] for k, x in v.items()]}
    if hasattr(v, "item"):                                   # numpy のスカラー
        return enc(v.item())
    raise TypeError("enc: %r" % (type(v),))


def dec(v):
    import struct
    if isinstance(v, list):
        return [dec(x) for x in v]
    if isinstance(v, dict):
        if "f" in v:
            return struct.unpack("<d", struct.pack("<Q", int(v["f"], 16)))[0]
        if "t" in v:
            return tuple(dec(x) for x in v["t"])
        if "d" in v:
            return {dec(k) if not isinstance(dec(k), list) else tuple(dec(k)): dec(x) for k, x in v["d"]}
    return v


def _capture_wrapper(orig, path, every, limit, _G):
    import random
    rng = random.Random(int(os.environ.get("OPCG_RD_CAPTURE_SEED", "7")))
    seen = set()
    n = [0]
    fh = open(path, "a", encoding="utf-8")
    src = os.environ.get("OPCG_RD_CAPTURE_SRC", "cap")

    @functools.wraps(orig)
    def wrapped(cards_d, don_d, blk, life, actx, turns=None, life_types=(), draw_types=(), arrive=()):
        out = orig(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)
        if n[0] >= limit:
            return out
        ax = {k: v for k, v in actx.items() if k != "_gain"}
        k = repr((cards_d, don_d, blk, life, sorted(ax.items(), key=lambda kv: kv[0]), turns, life_types,
                  draw_types, arrive, _G["EX_STATE_BUDGET"]))
        if k in seen or rng.random() > 1.0 / every:
            return out
        seen.add(k)
        n[0] += 1
        plan = out[2]
        rec = {"src": "%s#%d" % (src, n[0]), "budget": _G["EX_STATE_BUDGET"],
               "args": enc([list(cards_d or ()), don_d, list(blk or ()), life, ax, turns, tuple(life_types or ()),
                            tuple(draw_types or ()), tuple(arrive or ())]),
               "cp": enc(tuple(actx.get("cp", (None, None)))), "expect": enc(out)}
        # 動的計画の 1 呼び出し（採った計画の守り）: 入力と出力をビットで残す（Rust 単体の再生用）
        try:
            pr = _G["_prices_of"](actx)
            res = _G["rule_guard_plan_ex"](cards_d, don_d, plan["xs_first"], None, blk, life, plan["horizon"],
                                           life_types, pr, later_seq=plan["later_seq"],
                                           rest_blk=tuple(actx.get("rest_blk") or ()), arrive_blk=arrive,
                                           draw_types=draw_types)
            nu = {}
            for m in list(blk or ()) + list(actx.get("rest_blk") or ()) + list(arrive or ()):
                nu[float(m)] = float(_G["nu_meas_of"](float(m) + float(pr["olp"]), float(pr["mlp"])))
            rec["dp"] = {"cards": enc([list(c) for c in cards_d or ()]), "don": enc(float(don_d)),
                         "xs_first": enc(list(plan["xs_first"])),
                         "seq": enc([list(s) for s in plan["later_seq"]]), "blk": enc(list(blk or ())),
                         "life": enc(float(life)), "turns": int(plan["horizon"]),
                         "life_types": enc([list(t) for t in life_types or ()]),
                         "draw_types": enc([list(t) for t in draw_types or ()]),
                         "prices": enc([float(pr[k]) for k in ("lam", "lam_net", "mu", "olp", "mlp")]),
                         "rest": enc(list(actx.get("rest_blk") or ())), "arrive": enc(list(arrive or ())),
                         "nu": enc([[m, v] for m, v in nu.items()]), "eps": enc(float(_G["PWR_EPS"])),
                         "feq": enc(float(_G["_FEQ"])), "expect": enc(res)}
        except Exception as e:                               # noqa: BLE001
            rec["dp_error"] = repr(e)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        return out
    return wrapped


# ----------------------------------------------------------------------------------------------
# 第 2a 段: 計画の列挙と採点・歩き・試行のループ（`rule_don_solve` の冷たい経路）

_RACE_CAP = {}


def _race_cap(g):
    """`tau_grow` の `cap` の既定（定義の時点の `RACE_CAP`・`walk_crossing` は渡さない）。"""
    f = g["tau_grow"]
    if id(f) not in _RACE_CAP:
        import inspect
        _RACE_CAP[id(f)] = float(inspect.signature(f).parameters["cap"].default)
    return _RACE_CAP[id(f)]


def _glob_defender():
    d = _GLOB["d"]
    if d is None or d.n_states() > 600000:
        d = _GLOB["d"] = engine().RdDefender(None)
    return d


def _res_dict(r, any_blk):
    cut, st, alive, prev, harms, theta, nu_all = r
    if not any_blk:
        nu_all = 0                         # Python の `sum(空)` は整数の 0（辞書の `repr` が違う）
    return {"cut": cut, "stopped": st, "alive": alive, "prevented": prev, "harms": tuple(harms), "theta": theta,
            "nu_all": nu_all}


def run_rs(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, masks, h0, g=None):
    """Rust の核で試行のループから先を解く。`(計画の組 (cut, stopped, plan), 使った地平 or None, 開示)`。
    `plan` は `_rule_don_solve` と同じ辞書（同じキーの順）で、`horizon`／`horizon0` はまだ付けない。"""
    g = g or _G
    E = engine()
    prices = g["_prices_of"](actx)
    pr = tuple(float(prices[k]) for k in ("lam", "lam_net", "mu", "olp", "mlp"))
    rest = tuple(actx.get("rest_blk") or ())
    nu_fn = g["nu_meas_of"]
    nu = {}
    for m in list(blk or ()) + list(rest) + list(arrive or ()):
        m = float(m)
        if m not in nu:
            nu[m] = float(nu_fn(m + pr[3], pr[4]))
    budget = actx["budget"]
    flow = actx.get("flow") or [0.0] * (budget + 1)
    att1 = actx["att1"]
    mk = [(m["cost"], m["b"], m["later_seq"], m["hits1"], m["p_atk"], m["p_eff"], m["caps"],
           [(s["paid"], s["eff"], s["fb"]) for s in m["steps"]]) for m in masks]
    lim = g["EX_STATE_BUDGET"]
    no_now = bool(actx.get("no_attack_now"))
    h, mi, ks, paid, incr, val, tau, sched, r, st = E.rd_solve(
        [(float(c), float(d)) for c, d in cards_d or ()], float(don_d), _fl(blk), float(life),
        None if turns is None else int(turns),
        [(float(c), float(d), float(p)) for c, d, p in life_types or ()],
        [(float(c), float(d), float(p)) for c, d, p in draw_types or ()], _fl(arrive), _fl(rest),
        [float(x) for _s, x in att1], int(budget), flow, no_now, pr, list(nu.items()),
        float(g["PWR_EPS"]), float(g["_FEQ"]), actx["ds"], actx.get("a_tab") or [0.0], actx.get("ar_tab") or [0.0],
        actx.get("e_tab") or [0.0], float(g["SLOPE_FLOOR"]), _race_cap(g), mk, int(h0 or 0),
        None if lim is None else int(lim), bool(g["EX_LAYER_COUNT"]), _glob_defender())
    m = masks[mi]
    res = _res_dict(r, blk or rest or arrive)
    n = len(att1)
    hits1 = m["hits1"]
    xf = () if no_now else (tuple(float(att1[q][1]) + 1000.0 * ks[q] for q in range(n)) + tuple(hits1[n:]))
    hh = res["harms"]
    plan = {"atk": m["p_atk"], "rush": m["p_rush"], "eff": m["p_eff"], "incr": incr,
            "paid": float(paid), "play": tuple(m["play"]), "k": tuple(ks),
            "xs_first": tuple(xf), "later_seq": m["later_seq"],
            "alive": res["alive"], "value": val, "tau": tau,
            "harm_steps": tuple(hh), "sched": tuple(sched), "theta": res["theta"],
            "cut": res["cut"], "stopped": res["stopped"]}
    # 以下 `_rule_don_solve` の末尾と同じ式・同じ順
    L0 = int(max(0, round(float(life))))
    lam = float(prices["lam"]); mu = float(prices["mu"])
    plan["theta_parts"] = (lam * L0, mu * float(res["cut"]), float(res["nu_all"]))
    tau = float(plan["tau"]); th = float(plan["theta"])
    sched = plan["sched"]
    plan["a_time"] = (th / tau) if tau > 1e-12 else max(float(sched[0]) if sched else 0.0, g["SLOPE_FLOOR"])
    plan["a_turn"] = float(sched[0]) if sched else 0.0
    plan["attach_lead"] = 0.0
    plan["attach"] = 0.0
    plan["rest"] = rest
    plan["arrive"] = tuple(arrive or ())
    plan["draw_types"] = tuple(draw_types or ())
    return (res["cut"], res["stopped"], plan), h, st


def _keys(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, g):
    """`rule_don_solve` の覚え書きの鍵（外側＝`"w"` つき）と、`_rule_don_solve` の予算なしの鍵を作る関数。"""
    head = (tuple(sorted((float(c), float(d)) for c, d in cards_d or ())), float(don_d),
            tuple(sorted(float(m) for m in blk or ())), int(max(0, round(float(life)))))
    tail = (tuple(life_types or ()), tuple(draw_types or ()), tuple(sorted(float(m) for m in arrive or ())), actx["key"])
    outer = head + (None if turns is None else int(turns),) + tail + (g["EX_STATE_BUDGET"], "w")
    return outer, (lambda h: head + (int(h),) + tail)


def _put(cache, key, val):
    if len(cache) > 100000:
        cache.clear()
    cache[key] = val


def solve_rs(cards_d, don_d, blk, life, actx, turns=None, life_types=(), draw_types=(), arrive=(), g=None,
             cached=True):
    """`rule_don_solve` と同じ答え（覚え書き・計画のディスクの覚え書き・`EX_SPEED_STATS` も同じに扱う）を Rust の核で。
    `cached=False` は覚え書きを読まず書かず開示も足さない（`both` の比べる側）。`(答え, 開示)` を返す。"""
    g = g or _G
    if tuple(actx.get("cp", (None, None))) != g["cut_context_key"]():
        raise RuntimeError("攻め手の財布（attacker_ctx）と守る側の計算が別の値段の文脈で作られている（同じ `CP.defending` の中で呼ぶ）")
    okey, ikey = _keys(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, g)
    cache = g["_RULE_DON_CACHE"]
    if cached and okey in cache:
        return cache[okey], None
    store = g["PLAN_STORE"]
    dkey = None
    if cached and store is not None:
        dkey = store.key_of(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)
        hit = store.get(dkey)
        if hit is not None:
            _put(cache, okey, hit)
            return hit, None
    masks = g["_rule_don_masks"](cards_d, blk, life, actx, life_types)
    h0 = None
    if turns is None:
        h0 = g["model_horizon"](actx, blk, int(max(0, round(float(life)))), arrive)
    # 予算なしの試行（地平を渡された・予算が無い）は Python も `_rule_don_solve` の覚え書きを読み書きする（同じ鍵）
    unb = turns if turns is not None else (h0 if g["EX_STATE_BUDGET"] is None else None)
    out = cache.get(ikey(unb)) if (cached and unb is not None) else None
    st = None
    if out is None:
        out, h, st = run_rs(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, masks, h0, g)
        if cached and unb is not None:
            _put(cache, ikey(unb), out)
    else:
        h = unb
    if turns is None:
        out[2]["horizon"] = h
        out[2]["horizon0"] = h0
    if cached:
        if st is not None:
            es = g["EX_SPEED_STATS"]
            for k, v in zip(("attempt_fail", "attempt_skipped", "count_calls", "count_fallback"), st[:4]):
                es[k] += v
        _put(cache, okey, out)
        if dkey is not None:
            store.put(dkey, out)
    return out, st


_SPEED_KEYS = ("attempt_fail", "attempt_skipped", "count_calls", "count_fallback")


def _wrap_solve(orig, g):
    @functools.wraps(orig)
    def wrapped(cards_d, don_d, blk, life, actx, turns=None, life_types=(), draw_types=(), arrive=()):
        if MODE == "ref":
            import rule_don_ref as REF                       # 遅延（crossing_bridge を import し終えてから）
            return REF.solve(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)
        m = _mode()
        if m == "py":
            return orig(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)
        STATS["solve_calls"] += 1
        if m == "rs":
            return solve_rs(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, g=g)[0]
        # both: Rust を先に（覚え書きに触らない）→ Python → `repr` で比べる（冷たい解きなら開示の増分も）
        rs, st = solve_rs(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, g=g, cached=False)
        okey, _ik = _keys(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive, g)
        fresh = okey not in g["_RULE_DON_CACHE"] and g["PLAN_STORE"] is None and turns is None \
            and g["EX_STATE_BUDGET"] is not None
        before = [g["EX_SPEED_STATS"][k] for k in _SPEED_KEYS]
        py = orig(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)
        if repr(py) != repr(rs):
            _mismatch("計画（rule_don_solve）", py, rs)
        if fresh and st is not None:
            delta = tuple(g["EX_SPEED_STATS"][k] - b for k, b in zip(_SPEED_KEYS, before))
            if delta != tuple(st[:4]):
                _mismatch("試行の開示（EX_SPEED_STATS の増分）", delta, tuple(st[:4]))
        STATS["solve_checked"] += 1
        return py
    return wrapped


def install(g):
    """`crossing_bridge` の大域（`globals()`）の解き方の関数を包む。関数の原文は変えない。"""
    global _G
    if not _G:
        _G = g                              # 最初の写し（生きた大域の辞書そのもの・予算や切替は呼ぶたびに最新で読む）
    for name, wrap in (("_rule_guard_plan_ex", _wrap_guard), ("_ex_fit_horizon", _wrap_fit),
                       ("rule_don_solve", _wrap_solve)):
        _ORIG.setdefault(id(g), {})[name] = g[name]
        g[name] = wrap(g[name], g)
    path = os.environ.get("OPCG_RD_CAPTURE")
    if path:
        g["rule_don_solve"] = _capture_wrapper(g["rule_don_solve"], path,
                                               int(os.environ.get("OPCG_RD_CAPTURE_EVERY", "1")),
                                               int(os.environ.get("OPCG_RD_CAPTURE_MAX", "100000")), g)
    g["RD_KERNEL_TAG"] = kernel_tag


def _report():
    if STATS["calls"] or STATS["fit_calls"]:
        print("rd_kernel: mode=%s %r" % (MODE, STATS), file=sys.stderr)


atexit.register(_report)
