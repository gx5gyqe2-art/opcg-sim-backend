"""**理論の器の Rust 全移植・段 5／6（2026-10-07）**: 行の読みと局の駆動の切替（`OPCG_THEORY_ROWS=py|rs|both`・既定 `py`）。

* `py`（既定）: 何もしない（器は 1 字も変わらない）。
* `rs`: 各器の `collect` のループの本体（1 局ぶん）を Rust が解く。Python は記録を読み（`plan_labels.iter_games`）・
  入力の側（合成デッキの作り直し・`label_game`・決着の旗の表…）を作って **1 局の枠**（`theory_rs.frame_of`）と一緒に渡し、
  戻った行の表を器の入れ物（`rows_out`・`stats`・`per` …）へ入れる。集計・ブートストラップ・AUC・較正・JSON は Python のまま。
* `both`: Python の本体（核も外側も Python＝覚え書きは Python だけで育つ）と Rust の局の駆動（覚え書きは Rust だけで育つ）を
  両方解き、1 局ごとに行の表・`stats`・`RULE_STATS` の増分を**型とビット**で比べる（違えば `AssertionError`）。

仕組み: `plan_labels.iter_games` を差し替え、呼び手が器の `collect`（`(ファイル名, 関数名)` で見分ける・器を `__main__` で
走らせても同じ）なら、呼び手の局所変数（`frame.f_locals`＝器の入れ物）を読んで 1 局ずつ Rust を呼ぶ。`rs` では 1 局も
Python へ渡さない（ループの本体は走らない）。`lethal_rule.settled_map` は関数ごと差し替える。

記録（`OPCG_THEORY_ROWS_CAPTURE=<dir>`）: 局ごとに `{tool, src, game, frame, payload, result}` を `<dir>/<src>/<tool>.jsonl.gz` に書く
（`cargo test` が Python 無しに同じ順で解き直す）。**移植の間だけの道具**（段 7 で Python の理論と一緒に消し、器は Rust だけを呼ぶ形にする）。
"""
import copy
import gzip
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_rs as RS  # noqa: E402

_CFG = {"mode": "py", "installed": False, "cap": "", "fh": None, "games": {}, "checks": 0, "rows": 0}
_REAL = {}


def mode():
    return _CFG["mode"]


# ---------------------------------------------------------------------------------------------------------------
# 受け渡し

def _engine():
    return RS.engine()


def _clock():
    import theory_order as TO
    return {"W_ERR_MODE": TO.W_ERR_MODE, "SIGMA_REL": (None if TO.SIGMA_REL is None else float(TO.SIGMA_REL)),
            "SIGMA_D": float(TO.SIGMA_D), "W_MODE": TO.W_MODE, "W_BAR": float(TO.W_BAR)}


def _cb_mod():
    """切替を読む `crossing_bridge`（器が `crossing_bridge` 自身なら `__main__` の写し）。"""
    m = sys.modules.get("__main__")
    if m is not None and os.path.basename(getattr(m, "__file__", "") or "") == "crossing_bridge.py":
        return m
    import crossing_bridge
    return crossing_bridge


def _g():
    import theory_core_rs as TCR
    import theory_outer_rs as TOR
    g = TCR._g()
    g["modes"] = dict(g["modes"], **{k: v for k, v in TOR._modes_outer(vars(_cb_mod())).items() if v is not None})
    return g


def _cfg(loc, prof=None, theta_mode="const"):
    import theory_order as TO
    cb = _cb_mod()
    th = loc.get("theta", TO.THETA)
    mu = loc.get("mu", TO.MU)
    return {"theta": float(th), "mu": float(mu), "theta_mode": str(loc.get("theta_mode", theta_mode)),
            "clock": _clock(), "THETA_SIDE_MODE": cb.THETA_SIDE_MODE, "RATE_DECAY_MODE": cb.RATE_DECAY_MODE,
            "prof": (None if prof is None else [float(x) for x in prof])}


def frame_enc(fr):
    """枠 → 記録の形（`cargo test` の `game::frame_of_pyval` が読む）。"""
    names, snames = fr.names()
    cols = []
    for n in names:
        dt, sh, raw = fr.col(n)
        cols.append([n, {"nd": dt, "s": list(sh), "h": bytes(raw).hex()}])
    strs = [[n, fr.strcol(n)] for n in snames]
    d = fr.decks()
    return {"d": [["cols", cols], ["strs", strs], ["u2c", [list(x) for x in fr.u2c()]],
                  ["decks", None if d is None else [list(d[0]), list(d[1])]]]}


def call(tool, game, payload):
    """1 局を Rust で解く。戻り＝記録の形をほどいた dict。"""
    rows, pol, ex, L, ptr, idx = game
    fr = RS.frame_of(rows, pol, ex, L, ptr, idx)
    pl = dict(payload, tool=tool, g=_g())
    txt = RS.dumps(pl)
    try:
        out = _engine().theory_game_call(fr, txt)
    except Exception as e:                               # noqa: BLE001
        raise RuntimeError("theory rows: %s の局を Rust が解けない: %s" % (tool, e))
    res = RS.dec(json.loads(out))
    if _CFG["cap"]:
        _capture(tool, fr, txt, out)
    return res


def _counters():
    """Rust が増分で返す大域の計数（`RULE_STATS`・`EX_SPEED_STATS`・`COND_STATS`）"""
    import effect_value as EV
    cb = vars(_cb_mod())
    return cb["RULE_STATS"], cb["EX_SPEED_STATS"], EV.COND_STATS


def _apply_ev(res, dicts=None):
    """`RULE_STATS`／`EX_SPEED_STATS` の増分を順に足し（段 4 と同じ式・`ex:` は `EX_SPEED_STATS`）、条件の計数の差分を足す。"""
    rs_d, ex_d, cond = dicts if dicts is not None else _counters()
    for k, v in res.get("ev") or []:
        if k.startswith("ex:"):
            k = k[3:]
            ex_d[k] = ex_d.get(k, 0) + v
        else:
            rs_d[k] = rs_d.get(k, 0) + v
    cs = res.get("cs") or [0, 0, 0]
    for k, n in zip(("true", "false", "unknown"), cs):
        cond[k] = cond.get(k, 0) + n


def _counters_snap():
    return tuple(dict(d) for d in _counters())


def _counters_check(tool, before, res):
    """`both`: 前の計数に Rust の増分を足したものが、Python の本体が動かした後の計数と同じか。"""
    want = tuple(dict(d) for d in before)
    _apply_ev(res, want)
    check(tool, "RULE_STATS/EX_SPEED_STATS/COND_STATS", tuple(dict(d) for d in _counters()), want)


def _capture(tool, fr, payload_txt, out_txt):
    if _CFG["fh"] is None:
        src = os.environ.get("OPCG_THEORY_CAPTURE_SRC", "cap")
        d = os.path.join(_CFG["cap"], src)
        os.makedirs(d, exist_ok=True)
        name = os.environ.get("OPCG_THEORY_CAPTURE_TOOL", tool)
        _CFG["fh"] = gzip.open(os.path.join(d, name + ".rows.jsonl.gz"), "wt", encoding="utf-8")
    n = _CFG["games"].get(tool, 0)
    _CFG["games"][tool] = n + 1
    line = json.dumps({"tool": tool, "game": n, "frame": frame_enc(fr), "payload": json.loads(payload_txt),
                       "result": json.loads(out_txt)}, ensure_ascii=False, separators=(",", ":"))
    _CFG["fh"].write(line + "\n")


def _close():
    if _CFG["fh"] is not None:
        _CFG["fh"].close()
        _CFG["fh"] = None
    sp = os.environ.get("OPCG_THEORY_ROWS_STATS")
    if sp:
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump({"mode": _CFG["mode"], "games_checked": _CFG["checks"], "rows_checked": _CFG["rows"],
                       "games": _CFG["games"]}, fh)


# ---------------------------------------------------------------------------------------------------------------
# 比べる（`both`）

def _e(v):
    return json.dumps(RS.enc(_plain(v)), ensure_ascii=False, sort_keys=False)


def _plain(v):
    """比べる前の正規化: `HandRead` は数・`set` は並べた list・numpy の配列はそのまま（`enc` が読む）。"""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_plain(x) for x in v)
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if type(v).__name__ == "HandRead":
        return float(v)
    return v


def check(tool, what, py, rs):
    a, b = _e(py), _e(rs)
    if a != b:
        dump = os.environ.get("OPCG_THEORY_ROWS_DUMP")
        if dump:
            with open(dump, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"tool": tool, "what": what, "py": a, "rs": b}, ensure_ascii=False) + "\n")
        raise AssertionError("theory rows: %s の %s が Python と Rust で違う:\n py=%s\n rs=%s" % (tool, what, a[:1500], b[:1500]))
    _CFG["rows"] += 1


# ---------------------------------------------------------------------------------------------------------------
# 局の駆動（器ごと）

def _seed_of(game):
    rows, _pol, _ex, _L, _ptr, idx = game
    return int(rows["seed"][idx[0]]) if len(idx) else -1


class _Driver:
    """器ごとの局の駆動の型。`prep`（入力の側・Python）・`payload`・`apply`（`rs`）・`snap`／`compare`（`both`）を持つ。"""
    tool = None

    def __init__(self, frame):
        self.fr = frame
        self.loc = frame.f_locals
        self.carry = {}

    def refresh(self):
        self.loc = self.fr.f_locals
        return self.loc

    def run(self, real):
        limit = int(self.loc.get("limit_games") or 0)
        games = 0
        for game in real:
            games += 1
            if limit and games > limit:
                if _CFG["mode"] == "both":
                    yield game
                return
            if _CFG["mode"] == "rs":
                self.step_rs(game, games)
                _CFG["checks"] += 1
                continue
            snap = self.snap()
            res = self.solve(game, games, snap["stats_rs"])
            ctr = _counters_snap()
            yield game
            self.refresh()
            self.compare(snap, res)
            _counters_check(self.tool, ctr, res)
            _CFG["checks"] += 1
        # `rs`: 1 局も渡さない（器のループの本体は走らない）

    def step_rs(self, game, n):
        st = self.loc["stats"]
        res = self.solve(game, n, st)
        _apply_ev(res)
        self.apply(res)
        new = res["stats"]
        st.clear()
        st.update(new)

    def snap(self):
        return {"stats_rs": copy.deepcopy(self.loc["stats"])}


def _decks_pair(seat_decks, seed):
    """`KV._deck_pair(seat_decks, seed)`（無ければ `None`）"""
    import kappa_vector as KV
    p = KV._deck_pair(seat_decks, seed)
    return None if p is None else [None if d is None else list(d) for d in p]


class KvDriver(_Driver):
    tool = "kappa_vector"

    def solve(self, game, n, stats):
        loc = self.loc
        pin = {"decks": _decks_pair(loc["seat_decks"], _seed_of(game))}
        return call(self.tool, game, {"cfg": _cfg(loc, loc["prof"]), "in": pin, "stats": stats, "carry": {}})

    def apply(self, res):
        out = res["out"]
        if out is None:
            return
        self.loc["zs"].append(out["z0"])
        arms = self.loc["arms"]
        for k, v in zip(arms, out["acc"]):
            arms[k].append(v)

    def snap(self):
        s = super().snap()
        s["n"] = len(self.loc["zs"])
        return s

    def compare(self, snap, res):
        loc = self.loc
        check(self.tool, "stats", loc["stats"], res["stats"])
        out = res["out"]
        n = snap["n"]
        if out is None:
            check(self.tool, "appended", len(loc["zs"]) - n, 0)
            return
        check(self.tool, "z0", loc["zs"][n:], [out["z0"]])
        check(self.tool, "acc", [loc["arms"][k][n:] for k in loc["arms"]], [[v] for v in out["acc"]])


class TlDriver(_Driver):
    tool = "transition_ledger"

    def solve(self, game, n, stats, acc=None):
        loc = self.loc
        pin = {"decks": _decks_pair(loc["seat_decks"], _seed_of(game))}
        cfg = dict(_cfg(loc, loc["prof"]), sr=float(loc["sr"]))
        return call(self.tool, game, {"cfg": cfg, "in": pin, "stats": stats,
                                      "carry": {"acc": loc["acc"] if acc is None else acc}})

    def step_rs(self, game, n):
        super().step_rs(game, n)

    def apply(self, res):
        acc = self.loc["acc"]
        new = res["carry"]["acc"]
        acc.clear()
        acc.update(new)

    def snap(self):
        s = super().snap()
        s["acc"] = copy.deepcopy(self.loc["acc"])
        return s

    def run(self, real):
        # `both`: Rust は前の `acc` の写しから解く
        self._acc_snap = None
        return super().run(real)

    def compare(self, snap, res):
        check(self.tool, "stats", self.loc["stats"], res["stats"])
        check(self.tool, "acc", self.loc["acc"], res["carry"]["acc"])


#: `(ファイル名, 関数名)` → 局の駆動
DRIVERS = {("kappa_vector.py", "collect"): KvDriver, ("transition_ledger.py", "collect"): TlDriver}


def _iter_games_proxy(*args, **kwargs):
    real = _REAL["iter_games"](*args, **kwargs)
    fr = sys._getframe(1)
    key = (os.path.basename(fr.f_code.co_filename), fr.f_code.co_name)
    drv = DRIVERS.get(key)
    if drv is None or _CFG["mode"] == "py":
        return real
    return drv(fr).run(real)


def install(mode_=None):
    """切替を入れる（`mode` は省略時に `OPCG_THEORY_ROWS`）。`rs`／`both` では Python の側の核と外側は Python のまま
    （`both` の比べる相手は覚え書きが Python だけで育った値・E69）。"""
    if _CFG["installed"]:
        return
    m = mode_ if mode_ is not None else (os.environ.get("OPCG_THEORY_ROWS") or "py")
    if m not in ("py", "rs", "both"):
        raise ValueError("OPCG_THEORY_ROWS は py／rs／both")
    _CFG["mode"] = m
    _CFG["installed"] = True
    if m == "py":
        return
    _CFG["cap"] = os.environ.get("OPCG_THEORY_ROWS_CAPTURE", "")
    RS.load_all()
    RS.load_effects()
    from opcg_sim.learned.train import plan_labels as PL
    _REAL["iter_games"] = PL.iter_games
    PL.iter_games = _iter_games_proxy
    if os.environ.get("OPCG_PLAN_STORE") and m == "rs":
        import theory_outer_rs as TOR
        TOR._rs_call_ev("store.open", {"d": [["path", os.environ["OPCG_PLAN_STORE"]]]}, {})
        import atexit
        atexit.register(TOR._store_report)
    import atexit
    atexit.register(_close)
