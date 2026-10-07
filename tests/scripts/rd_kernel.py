"""**Rust 化（2026-10-05）**: `rule_don` の守る側の計算（動的計画・段ごとの状態の数え方・地平の選び方）・計画の列挙と採点・
歩き・地平の試行のループを回す Rust の核（`rust/opcg_engine/src/theory/`）の Python 側の継ぎ目。

**第 3 段（2026-10-05）**: Rust が**唯一の速い解き方**になった（速くした Python の解き方は消した・設計 §9 の (B)）。
Python に残るのは: 覚え書き（`crossing_bridge._RULE_DON_CACHE`）・計画のディスクの覚え書き（`plan_store`）・出す札の組
（`_rule_don_masks`＝`rules_steps`＋`_attach_gain`・第 2b 段まで Python）・最初の地平（`model_horizon`）・ν の表・計画の
辞書の組み立て（`crossing_bridge.rule_don_solve`）。答え合わせの相手は速くする前の原文 `tests/harness/rule_don_ref.py`（REF）。

切替は環境変数 `OPCG_RD_KERNEL`（import 時に 1 回読む・`set_mode` でテストが切り替える）:

* `rs`   — Rust の核（**既定**）。wheel が無い／古い／入口の版が違うと**落ちる**（黙って他の解き方に戻らない・
  `make test` も wheel を要る）。
* `ref`  — 速くする前の解き方の原文（`rule_don_ref`）。遅い・1 問の調べ直し用。
* `both` — Rust と原文の両方で解いて `repr` を比べ、違えば例外（**Rust の結果を返す**＝出力は `rs` と同じ）。遅い。

旧の `py`（速くした Python）と `auto`（Rust が無ければ `py`）は**消した**（`py` が無いので `auto` の戻り先も無い）。
指定すると理由つきで落ちる。

**古い wheel の検出**: `make test` は wheel を作り直さない。Rust 側が `(入口の版, src/theory の原文のハッシュ)` を
返し、ここで版と（ソースの木が有れば）ハッシュを突き合わせる。違えば落ちる。

**計画の覚え書きの鍵**: `kernel_tag()`（`rs:<版>:<ハッシュ>`）を `plan_store` の鍵に入れる——Rust を直せば別の鍵。
"""
import atexit
import functools
import glob
import json
import os
import sys

#: Rust 側の入口の版（`theory::API`）と一致させる。入口の形を変えたら両方上げる。
RD_KERNEL_API = 2
MODES = ("rs", "ref", "both")
#: 第 3 段で消した切替の値（指定されたら理由つきで落ちる）
REMOVED = {"py": "速くした Python の解き方は第 3 段（2026-10-05）で消した＝`rs`（既定）か、原文の `ref` を使う",
           "auto": "`auto` は Rust が無いときの戻り先（`py`）ごと消した＝`rs`（既定・wheel は `make rust-develop`）"}


def _check_mode(m):
    if m in REMOVED:
        raise ValueError("OPCG_RD_KERNEL=%s は使えない: %s" % (m, REMOVED[m]))
    if m not in MODES:
        raise ValueError("OPCG_RD_KERNEL は %s のどれか（今: %r）" % ("/".join(MODES), m))
    return m


try:
    MODE = _check_mode(os.environ.get("OPCG_RD_KERNEL", "rs") or "rs")
except ValueError as _e:
    raise SystemExit(str(_e))

STATS = {"guard_calls": 0, "solve_calls": 0, "guard_checked": 0, "solve_checked": 0}
_G = {}              # crossing_bridge の大域の辞書（install が覚える・生きた辞書）
_STATUS = []         # [(ok, reason, module)] の 1 要素（1 回だけ調べる）
_GLOB = {"d": None}


# ----------------------------------------------------------------------------------------------
# wheel の確認

#: 守る側の計算の原文（`build.rs` の `KERNEL_FILES` と同じ・全移植の段 3〔2026-10-07〕から理論の移植のファイルは入れない）
KERNEL_FILES = ("defender.rs", "layers.rs", "numeric.rs", "plans.rs", "pyapi.rs", "sched.rs", "succ.rs", "table.rs")


def source_hash(root=None):
    """`rust/opcg_engine/src/theory/` の守る側の計算の原文（`KERNEL_FILES`）のハッシュ（`build.rs` と同じ式: FNV-1a 64・
    相対パス順）。木が無ければ None。"""
    base = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rust", "opcg_engine",
                                "src", "theory")
    base = os.path.abspath(base)
    if not os.path.isdir(base):
        return None
    files = sorted(os.path.relpath(p, base).replace(os.sep, "/")
                   for p in glob.glob(os.path.join(base, "**", "*.rs"), recursive=True)
                   if os.path.relpath(p, base).replace(os.sep, "/") in KERNEL_FILES)
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
    _check_mode(m)
    if m in ("rs", "both"):
        engine()
    MODE = m


def kernel_tag():
    """計画の覚え書きの鍵に入れる「どの核か」（`rs:<入口の版>:<src/theory のハッシュ>`）。"""
    _ok, _why, E = status()
    if E is None:
        return "rs:none"
    api, h = E.rd_kernel_version()
    return "rs:%s:%s" % (api, h)


# ----------------------------------------------------------------------------------------------
# 原文（REF）: `ref`／`both` の比べる相手

def ref_module(g=None):
    """`rule_don_ref`（速くする前の原文）を、**今動いている** `crossing_bridge` の切替で読めるようにして返す
    （`crossing_bridge` が `__main__` として走っていても、REF が import した別の写しではなくこちらの大域を写す）。"""
    harness = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness")
    if harness not in sys.path:                              # 器を直接走らせたとき（tests/harness が道に無い）
        sys.path.append(harness)
    import rule_don_ref as REF                               # 遅延（crossing_bridge を import し終えてから）
    g = g or _G
    mod = sys.modules.get(g.get("__name__")) if g else None
    if mod is not None and vars(mod) is g:
        REF.CB = mod
    REF._sync()
    return REF


def _mismatch(what, a, b):
    raise AssertionError("rd_kernel: Rust と原文（rule_don_ref）の結果が違う（%s）\n  ref=%r\n  rs=%r" % (what, a, b))


def check(what, rs, ref):
    """`both`: `repr` で比べる（違えば例外）。"""
    if repr(rs) != repr(ref):
        _mismatch(what, ref, rs)
    STATS["solve_checked" if what == "rule_don_solve" else "guard_checked"] += 1


# ----------------------------------------------------------------------------------------------
# 守る側の動的計画（`crossing_bridge.rule_guard_plan_ex` が呼ぶ）

def glob_defender():
    """呼び出しをまたいで覚える守る側の計算（予算なし・窓・テスト・計画の列挙の外の呼び出し用）。値は問題だけで決まる。"""
    d = _GLOB["d"]
    if d is None or d.n_states() > 600000:
        d = _GLOB["d"] = engine().RdDefender(None)
    return d


def reset_global():
    """呼び出しをまたぐ覚え書きを捨てる（値は変わらない）。"""
    _GLOB["d"] = None


def _fl(seq):
    return [float(x) for x in seq or ()]


def guard_rs(cards, don, xs_first, seq, blk_margins, life, turns, life_types, lam, lam_net, mu, olp, mlp,
             rest_blk=(), arrive_blk=(), draw_types=(), defender=None, g=None):
    """Rust の核で守る側の最善の守りを解く（`seq` は段ごとに昇順に並べた攻撃の並び）。予算を超えたら `None`
    （`defender` を渡さなければ予算なしの共有の覚え書き＝`None` にならない）。"""
    g = g or _G
    STATS["guard_calls"] += 1
    d = defender if defender is not None else glob_defender()
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
    return res_dict(r, blk_margins or rest_blk or arrive_blk)


def res_dict(r, any_blk):
    cut, st, alive, prev, harms, theta, nu_all = r
    if not any_blk:
        nu_all = 0                         # Python の `sum(空)` は整数の 0（辞書の `repr` が違う）
    return {"cut": cut, "stopped": st, "alive": alive, "prevented": prev, "harms": tuple(harms), "theta": theta,
            "nu_all": nu_all}


def count_layers(cards, don, blk, life, life_types, rest_blk, arrive_blk, draw_types, roots, cap, lim, eps=None):
    """Rust の段ごとの状態の数え方（`roots` は `[(今のターンの攻撃, 並び)]`・テスト用）。"""
    E = engine()
    return E.rd_count_layers(
        [(float(c), float(d)) for c, d in cards or ()], float(don), _fl(blk), float(life),
        [(float(c), float(d), float(p)) for c, d, p in life_types or ()], _fl(rest_blk), _fl(arrive_blk),
        [(float(c), float(d), float(p)) for c, d, p in draw_types or ()],
        [(_fl(xf), [_fl(s) for s in ls]) for xf, ls in roots], int(cap), int(lim),
        float(_G["PWR_EPS"] if eps is None else eps))


# ----------------------------------------------------------------------------------------------
# 計画の列挙と採点・歩き・試行のループ（`crossing_bridge.rule_don_solve` が呼ぶ）

_RACE_CAP = {}


def race_cap(tau_grow):
    """`tau_grow` の `cap` の既定（定義の時点の `RACE_CAP`・`walk_crossing` は渡さない）。"""
    if id(tau_grow) not in _RACE_CAP:
        import inspect
        _RACE_CAP[id(tau_grow)] = float(inspect.signature(tau_grow).parameters["cap"].default)
    return _RACE_CAP[id(tau_grow)]


def rd_solve(args):
    """`opcg_engine.rd_solve(*args, 共有の覚え書き)`。`args` は `crossing_bridge._rd_solve_args` が作る。"""
    STATS["solve_calls"] += 1
    return engine().rd_solve(*args, glob_defender())


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


def install(g):
    """`crossing_bridge` の大域（`globals()`）を覚える（wheel は最初に解くときに確かめる＝古ければそこで落ちる）。
    `OPCG_RD_CAPTURE` が有れば `rule_don_solve` に取り込みの包みを掛ける。"""
    global _G
    if not _G:
        _G = g                              # 最初の写し（生きた大域の辞書そのもの・予算や切替は呼ぶたびに最新で読む）
    path = os.environ.get("OPCG_RD_CAPTURE")
    if path:
        g["rule_don_solve"] = _capture_wrapper(g["rule_don_solve"], path,
                                               int(os.environ.get("OPCG_RD_CAPTURE_EVERY", "1")),
                                               int(os.environ.get("OPCG_RD_CAPTURE_MAX", "100000")), g)
    g["RD_KERNEL_TAG"] = kernel_tag


def _report():
    if MODE != "rs" and (STATS["guard_calls"] or STATS["solve_calls"]):
        print("rd_kernel: mode=%s %r" % (MODE, STATS), file=sys.stderr)


atexit.register(_report)
