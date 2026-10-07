"""**理論の計算は Rust だけ**（全移植の段 7・2026-10-07・`docs/reports/2026-10-07_port_stage7.md`）——理論の器が Rust を呼ぶ口。

理論の式（値付けの核・耐久 Θ・速さ・守る側の計算・行の読み・局ごとの `collect` の中身）は全部
`rust/opcg_engine/src/theory/` にあり、Python に写しは無い。Python に残るのは（ユーザ決定 2026-10-06／07）:
CLI・記録の読み込みと入力の準備（`label_game`・合成デッキの作り直し・決着の旗の表・損害の輪郭と σ の表）・
カード DB（パーサ）からカード表と効果の木を Rust へ渡すこと・集計（ブートストラップ・AUC・較正）と JSON の書き出し。

この模块が持つもの:

* **受け渡し**: カード表と語彙（`load_cards`）・効果の木（`load_effects`・`opcg_sim/data/opcg_effects.json`）・
  盤面の分布の fixture（`load_opp_boards`）・1 局の枠（`frame_of`）・記録の形の符号化（`enc`／`dec`）。
* **切替の置き場**（`SW`）: Rust へ渡す文脈 `g` の切替の値。**既定の枝と、残す候補 2 つ**（ドンの付け違いの費用の 3 つの形
  `ATTACK_DON_COST_MODE`・探す効果の 1 枚 1 役 `SEARCH_VALUE_MODE=joint`）だけを選べる。ほかの値は `set_switch` が誤りにする
  （Rust の `entry::apply_g` も誤りを返す）。候補は環境変数 `OPCG_THEORY_SWITCHES="名前=値,…"` でも選べる。
  旧い既定・保留の値の再現は凍結ブランチ `claude/theory-switches-final`。
* **器の集計が読む定数**（`MU`・`THETA`・時計の `SIGMA_D` ほか）と時計の設定（`set_w_err_mode`・`set_sigma_rel`・
  `set_sigma_turn`・`set_w_bar`）。
* **呼び出し**: `game_call`（1 局ぶんの局の駆動・`theory_game_call`）・`rows_call`（局をまたがない行の関数・`theory_rows_call`）。
  局の駆動が返す計数の増分は `RULE_STATS`／`EX_SPEED_STATS`／`COND_STATS` に足す。
* **計画のディスクの覚え書き**: `OPCG_PLAN_STORE=<dir>` なら Rust の覚え書き（`core::store`）を開く。

記録の形（`enc`／`dec`）: 浮動小数は 16 桁の 16 進（ビット）・組は `{"t": …}`・dict は `{"d": [[k, v], …]}`・numpy の配列は
`{"nd": dtype, "s": 形, "h": 生のバイト}`・numpy のスカラーは `{"npf"|"npi": …}`・大域の物は `{"obj": 名前}`
（Rust の `theory::pyval` が同じ形を読み書きする）。
"""
import atexit
import json
import math
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EFFECTS_PATH = os.path.join(_ROOT, "opcg_sim", "data", "opcg_effects.json")
OPP_BOARDS_PATH = os.path.join(_ROOT, "tests", "fixtures", "opp_boards.json")

_ENGINE = {}


# ---------------------------------------------------------------------------------------------------------------
# 器の集計と出力が読む定数（値は Rust の `theory::leaves_to`・`core::to` と同じ・実測の写し）

MU = 0.0551
LAM = 0.1362
H_LIFE_TO_HAND = 0.89
THETA = round((LAM - H_LIFE_TO_HAND * MU) / MU, 4)
R_TURNS = 4.128
DELTA = 0.0277
#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs`）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_HAND, SC_OPP_HAND = 6, 7
SC_MY_LEADER_POWER, SC_OPP_LEADER_POWER = 12, 13
#: 整数ターンの床（`crossing_bridge.sigma_rel_mle`）
TURN_ROUND_VAR = 1.0 / 12.0
TURN_ROUND_MEAN = -0.5
#: 出力の欄に書く定数（波C／段 7 で切替は消えた）
W_MODE = "curve"
SETTLE_COND_MODE = "whole"
SIGMA_FLOOR_MODE = "off"
CLOCK_HAND_MODE = "off"
LEDGER_FLOW_PRICING = "exercise"
#: 値付けの直し（`F_PRICING_FIX=all`・定数）の出力の欄（旧 `effect_value.pricing_fixes_label()`／`apply_f_pricing_fixes`）
PRICING_FIXES_LABEL = "draw_n,either_side,optional_block,base_power,opp_subject,revealed_src,choice_max"
F_PRICING_FIXES_LABEL = "hand_board,state_filters,branch_then,attached_don_cond,look_return,trash_pool"
FLOW_PRICING_MODES = ("option", "exercise")

# ---------------------------------------------------------------------------------------------------------------
# 時計（`theory_order` の写しだった大域の設定・Rust へは `clock()` で渡す）

W_ERR_MODES = ("abs", "rel")
CLOCK = {"W_ERR_MODE": "rel", "SIGMA_REL": None, "SIGMA_TURN": 1.0, "SIGMA_D": math.sqrt(2.0) * 1.0,
         "W_BAR": 0.5 / R_TURNS}


def set_w_err_mode(mode):
    if mode not in W_ERR_MODES:
        raise ValueError("w err mode は %s のどれか" % (W_ERR_MODES,))
    CLOCK["W_ERR_MODE"] = mode
    return mode


def set_sigma_rel(value):
    CLOCK["SIGMA_REL"] = None if value is None else float(value)
    return CLOCK["SIGMA_REL"]


def set_sigma_turn(turns):
    CLOCK["SIGMA_TURN"] = float(turns)
    CLOCK["SIGMA_D"] = math.sqrt(2.0) * CLOCK["SIGMA_TURN"]
    return CLOCK["SIGMA_D"]


def set_w_bar(value):
    CLOCK["W_BAR"] = float(value)
    return CLOCK["W_BAR"]


def clock():
    return {"W_ERR_MODE": CLOCK["W_ERR_MODE"], "SIGMA_REL": (None if CLOCK["SIGMA_REL"] is None else float(CLOCK["SIGMA_REL"])),
            "SIGMA_D": float(CLOCK["SIGMA_D"]), "W_MODE": W_MODE, "W_BAR": float(CLOCK["W_BAR"])}


# ---------------------------------------------------------------------------------------------------------------
# 切替（Rust の `entry::apply_g` が受ける値だけ）

#: `g["modes"]`（並びは Rust の計画の覚え書きの鍵に入る＝変えない）
SW = {"NU_MODE": "pair", "SURV_MODE": "geo", "OPTION_MODE": "dist", "CBAR_MODE": "strict", "SPEED_MEMO": "True",
      "ATTACK_ABILITY_MODE": "off", "PASSIVE_BODY_MODE": "off", "DEFENDER_POWER_MODE": "rule",
      "ATTACK_DON_COST_MODE": "off", "COST_AFFORD_MODE": "check", "F_PRICING_FIX": "all",
      "CUT_PRICE_MODE": "joint", "CUT_TAKE_MODE": "gbar", "SEARCH_VALUE_MODE": "legacy", "DECK_COUNTER_MODE": "rules",
      "UNKNOWN_FACTOR": 1.0, "THETA_HAND_MODE": "rule_don", "THETA_SIDE_MODE": "legacy", "RD_KERNEL": "rs",
      "RATE_DECAY_MODE": "off", "EX_STATE_BUDGET": "300000", "THETA_BODY_MODE": "blockers", "SLOPE_TAKE_MODE": "life",
      "ATTACK_DON_MAX": "10"}
#: 選べる値（ここに無い名前は固定）。`UNKNOWN_FACTOR` は数（判らない条件の係数・感度）
SWITCH_VALUES = {"ATTACK_DON_COST_MODE": ("off", "opportunity", "misalloc", "misalloc_play"),
                 "SEARCH_VALUE_MODE": ("legacy", "joint"),
                 "THETA_SIDE_MODE": ("legacy", "symmetric"),
                 "RATE_DECAY_MODE": ("off", "ko")}
#: 局の駆動の設定（`g` の外）
RUN = {"FLOW_PRICING": "option", "PRE_SETTLE_MODE": "off", "MIRROR_ME": True}
RUN_VALUES = {"FLOW_PRICING": FLOW_PRICING_MODES, "PRE_SETTLE_MODE": ("off", "on"), "MIRROR_ME": (True, False)}


def set_switch(name, value):
    """切替を 1 つ立てる。Rust に移していない値は `ValueError`（黙って既定で解かない）。"""
    if name == "UNKNOWN_FACTOR":
        SW[name] = float(value)
        return SW[name]
    if name in RUN_VALUES:
        if value not in RUN_VALUES[name]:
            raise ValueError("%s=%r は Rust に無い（%r）" % (name, value, RUN_VALUES[name]))
        RUN[name] = value
        return value
    ok = SWITCH_VALUES.get(name)
    if name not in SW:
        raise ValueError("切替 %r は無い" % (name,))
    if ok is None:
        if value != SW[name]:
            raise ValueError("%s は %r だけ（%r は移していない・凍結ブランチ claude/theory-switches-final で再現）"
                             % (name, SW[name], value))
        return value
    if value not in ok:
        raise ValueError("%s=%r は Rust に無い（%r・旧い値は凍結ブランチ claude/theory-switches-final で再現）" % (name, value, ok))
    SW[name] = value
    return value


def _switches_from_env():
    txt = os.environ.get("OPCG_THEORY_SWITCHES", "")
    for part in [p for p in txt.split(",") if p.strip()]:
        k, _, v = part.partition("=")
        set_switch(k.strip(), v.strip())


_switches_from_env()


def g():
    """Rust の局の駆動へ渡す文脈（段 3／4 の `g`・既定の文脈＝値段の窓なし）。"""
    modes = dict(SW)
    modes["UNKNOWN_FACTOR"] = float(modes["UNKNOWN_FACTOR"])
    return {"CUT_PRICER": None, "CUT_PRICER_KEY": None, "CUT_TAKE_CARD": None, "CUT_OTHER_SIDE": 0, "OPTION_DEPTH": 0,
            "FLOW_PRICING": RUN["FLOW_PRICING"], "OPAQUE_UPPER": False, "modes": modes}


def cfg(theta=THETA, mu=MU, theta_mode="const", prof=None, **extra):
    """局の駆動の設定（`core::drive::cfg_of`）。`extra` は器ごとの値（並びは呼び手の順）。"""
    out = {"theta": float(theta), "mu": float(mu), "theta_mode": str(theta_mode), "clock": clock(),
           "THETA_SIDE_MODE": SW["THETA_SIDE_MODE"], "RATE_DECAY_MODE": SW["RATE_DECAY_MODE"],
           "prof": (None if prof is None else [float(x) for x in prof])}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Rust の口

def engine():
    if "m" not in _ENGINE:
        import opcg_engine
        if not hasattr(opcg_engine, "theory_game_call"):
            raise RuntimeError("opcg_engine の wheel が古い（theory_game_call が無い）——`make rust-develop` で作り直す")
        _ENGINE["m"] = opcg_engine
    return _ENGINE["m"]


def ready():
    """カード表・盤面の分布・効果の木を Rust へ（1 度だけ）・計画の覚え書きを開く。"""
    if "ready" in _ENGINE:
        return engine()
    load_cards()
    load_opp_boards()
    load_effects()
    _ENGINE["ready"] = True
    path = os.environ.get("OPCG_PLAN_STORE")
    if path:
        core_call("store.open", {"path": path})
        atexit.register(_store_report)
    return engine()


def core_call(name, args, ctx=None):
    """核の入口を 1 つ（`core::entry`）。`args` は名前つきの全部の引数（既定値も渡す）・`ctx` は文脈 `g`
    （省略時は空＝計画の覚え書きの `store.*` の形）。"""
    payload = json.dumps({"d": [["a", enc(args)], ["g", enc({} if ctx is None else ctx)], ["pre", []]]},
                         ensure_ascii=False, separators=(",", ":"))
    out = json.loads(engine().theory_core_call(name, payload, False))
    return dec({k: x for k, x in out["d"]}.get("r"))


def attack_value(power, target_power, is_leader, theta=None, mu=None, nu_target=None, blockers=None):
    """攻撃 1 回の価値（`min(守る費用, 受ける費用, …)`・Rust の `to::attack_value`）——手計算の試験が呼ぶ入口。"""
    ready()
    a = {"power": float(power), "target_power": float(target_power), "is_leader": bool(is_leader),
         "theta": float(THETA if theta is None else theta), "mu": float(MU if mu is None else mu),
         "nu_target": None if nu_target is None else float(nu_target),
         "blockers": None if blockers is None else [[float(p), float(q)] for p, q in blockers]}
    return core_call("to.attack_value", a, g())


def _store_report():
    try:
        r = core_call("store.report", {})
        if r is not None:
            print("plan_store_rs: %r" % (r,), file=sys.stderr)
    except Exception:                                   # noqa: BLE001
        pass


RULE_STATS = {}
EX_SPEED_STATS = {"attempt_fail": 0, "attempt_skipped": 0, "count_calls": 0, "count_fallback": 0}
COND_STATS = {"true": 0, "false": 0, "unknown": 0}


def reset_cond_stats():
    for k in COND_STATS:
        COND_STATS[k] = 0


def apply_ev(res):
    """局の駆動が返した `RULE_STATS`／`EX_SPEED_STATS` の増分（`ex:` は後者）を順に足し、条件の計数の差分を足す。"""
    for k, v in res.get("ev") or []:
        if k.startswith("ex:"):
            k = k[3:]
            EX_SPEED_STATS[k] = EX_SPEED_STATS.get(k, 0) + v
        else:
            RULE_STATS[k] = RULE_STATS.get(k, 0) + v
    cs = res.get("cs") or [0, 0, 0]
    for k, n in zip(("true", "false", "unknown"), cs):
        COND_STATS[k] = COND_STATS.get(k, 0) + n


def game_call(tool, game, payload, counters=True, raw=False):
    """1 局（`plan_labels.iter_games` の 1 つ、または `frame_of` の枠）を Rust の局の駆動で解く。戻り＝ほどいた dict
    （`raw` なら JSON を読んだだけの記録の形）。`counters` なら計数の増分を足す。"""
    ready()
    fr = game if hasattr(game, "names") else frame_of(*game)
    pl = dict(payload, tool=tool, g=g())
    try:
        out = engine().theory_game_call(fr, dumps(pl))
    except Exception as e:                               # noqa: BLE001
        raise RuntimeError("theory: %s の局を Rust が解けない: %s" % (tool, e))
    j = json.loads(out)
    res = dec(j)
    if counters:
        apply_ev(res)
    return (res, j) if raw else res


def seed_of(game):
    """1 局（`iter_games` の 1 つ）の seed（行が無ければ −1）"""
    rows, _pol, _ex, _L, _ptr, idx = game
    return int(rows["seed"][idx[0]]) if len(idx) else -1


def deck_list(p):
    """`(席 0 のデッキ, 席 1 のデッキ)`（無ければ `None`）→ 受け渡しの形（list の list）"""
    return None if p is None else [None if d is None else list(d) for d in p]


def settled_in(settled, seed):
    """この局の宣言した行 `[[w, t], …]`（`settled` が無ければ `None`）"""
    if settled is None:
        return None
    return [[w, t] for (sd, w, t), v in settled.items() if sd == seed and v]


def rows_call(fn, rs, **kw):
    """局をまたがない行の関数（`core::drive::rows_call`）。`rs`＝行の数の組の列。"""
    ready()
    pl = dict(kw, fn=fn, cfg=cfg(), rs=rs)
    try:
        out = engine().theory_rows_call(dumps(pl))
    except Exception as e:                               # noqa: BLE001
        raise RuntimeError("theory: %s を Rust が解けない: %s" % (fn, e))
    return dec(json.loads(out))


def prob_of_d(rs, sigma_d=None, scale_mode="hyp", mover=False):
    """`W`（時計の差 `D` から勝率・`[(D, τ_me|None, τ_opp|None), …]`）。式は Rust の `leaves_to::prob_of_d`。"""
    rows = [[float(d), None if a is None else float(a), None if b is None else float(b)] for d, a, b in rs]
    return rows_call("to.prob_of_d", rows, sigma_d=(None if sigma_d is None else float(sigma_d)),
                     scale_mode=str(scale_mode), mover=bool(mover))


def clock_scale(pairs, mode="hyp"):
    """1 次同次な局面の尺度 `s`（`[(τ_me, τ_opp), …]`）。"""
    return rows_call("to.clock_scale", [[float(a), float(b)] for a, b in pairs], mode=str(mode))


def whole_clock_scale(ts):
    """`whole` の時計 1 本の幅の尺度 `max(1, τ)`（`[τ, …]`）。"""
    return rows_call("to.whole_clock_scale", [float(t) for t in ts])


def tau_from_profile(rows, prof):
    """輪郭に沿って `Θ` に届くまでのターン数（`[(Θ, j, scale, r, shield, shield_rate, refill, step), …]`）。"""
    rs = [[float(a[0]), int(a[1])] + [float(x) for x in a[2:]] for a in rows]
    return rows_call("cb.tau_from_profile", rs, prof=[float(x) for x in prof])


# ---------------------------------------------------------------------------------------------------------------
# 記録の形

#: 大域の物の識別（`id` → 名前）
_OBJS = {}


def _vocab_rev():
    if "rev" not in _ENGINE:
        from opcg_sim.learned.vocab import shared_vocab
        _ENGINE["rev"] = {i: c for c, i in shared_vocab().items()}
    return _ENGINE["rev"]


def idx2cid():
    """語彙の index → card_id（`plan_labels` の語彙と同じ）。"""
    return dict(_vocab_rev())


class Unencodable(TypeError):
    pass


def enc(v):
    """ビットを保つ符号化（`dec` で戻る・Rust の `pyval::from_capture` が読む）。"""
    if v is None or isinstance(v, (bool, int, str)) and not isinstance(v, np.generic):
        if isinstance(v, int) and not isinstance(v, bool) and not (-(1 << 63) <= v < (1 << 63)):
            raise Unencodable("int が 64 bit を越える")
        return v
    if isinstance(v, float):
        return {"f": "%016x" % struct.unpack("<Q", struct.pack("<d", v))[0]}
    hit = _OBJS.get(id(v))
    if hit is not None and hit[0] is v:
        return {"obj": hit[1]}
    if isinstance(v, tuple):
        return {"t": [enc(x) for x in v]}
    if isinstance(v, list):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {"d": [[enc(k), enc(x)] for k, x in v.items()]}
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        dt = a.dtype.str.lstrip("<|=")
        if a.dtype.byteorder == ">":
            raise Unencodable("big endian")
        return {"nd": dt, "s": list(a.shape), "h": a.tobytes().hex()}
    if isinstance(v, np.floating):
        return {"npf": v.dtype.str.lstrip("<|="), "h": "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]}
    if isinstance(v, (np.integer, np.bool_)):
        return {"npi": v.dtype.str.lstrip("<|="), "v": int(v)}
    if isinstance(v, (set, frozenset)):
        return sorted(enc(x) for x in v)
    raise Unencodable("enc: %r" % (type(v),))


def dec(v):
    if isinstance(v, list):
        return [dec(x) for x in v]
    if isinstance(v, dict):
        if "f" in v and len(v) == 1:
            return struct.unpack("<d", struct.pack("<Q", int(v["f"], 16)))[0]
        if "t" in v and len(v) == 1:
            return tuple(dec(x) for x in v["t"])
        if "d" in v and len(v) == 1:
            out = {}
            for k, x in v["d"]:
                k = dec(k)
                out[tuple(k) if isinstance(k, list) else k] = dec(x)
            return out
        if "nd" in v:
            return np.frombuffer(bytes.fromhex(v["h"]), dtype=np.dtype("<" + v["nd"])).reshape(v["s"]).copy()
        if "npf" in v:
            return np.dtype(v["npf"]).type(struct.unpack("<d", struct.pack("<Q", int(v["h"], 16)))[0])
        if "npi" in v:
            return np.dtype(v["npi"]).type(v["v"])
        if "obj" in v:
            return ("<obj>", v["obj"])
    return v


def dumps(v):
    return json.dumps(enc(v), ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------------------------------------------
# カード表・効果の木・fixture

def _sv(x):
    return getattr(x, "value", x)


def card_record(m):
    """1 枚の札 → Rust の `TCard` の JSON（理論が原本から読む項目と `Cards.info` の 9 項目）。"""
    from opcg_sim.learned import n_rel_feat as NF
    cid = m.card_id
    info = _cards().info(cid)
    prof = NF.profile(m)
    thr = []
    for t in prof.get("thr") or ():
        if len(t) < 5:
            raise ValueError("thr の形が想定外: %r" % (t,))
        thr.append([None if t[0] is None else t[0], None if t[1] is None else t[1], bool(t[2]), str(t[3]), bool(t[4])])
    return {"id": str(cid), "info": info,
            "type": getattr(getattr(m, "type", None), "name", ""),
            "power": float(getattr(m, "power", 0) or 0),
            "cost": int(getattr(m, "cost", 0) or 0),
            "counter": float(getattr(m, "counter", 0) or 0),
            "keywords": [str(k) for k in (getattr(m, "keywords", ()) or ())],
            "traits": [str(_sv(t)) for t in (getattr(m, "traits", None) or [])],
            "colors": [str(_sv(c)) for c in (getattr(m, "colors", None) or [])],
            "names": [str(n) for n in (getattr(m, "all_names", None) or [getattr(m, "name", "")])],
            "attribute": str(_sv(getattr(m, "attribute", "")) or ""),
            "counter_event": float(prof.get("counter_event") or 0.0),
            "thr": thr}


def _db():
    from opcg_sim.loop import decks as D
    return D.load_db()


def _cards():
    if "cards" not in _ENGINE:
        from opcg_sim.learned.train import plan_labels as PL
        _ENGINE["cards"] = PL.Cards()
    return _ENGINE["cards"]


def card_table():
    db = _db()
    cards = [card_record(m) for _cid, m in sorted(db.cards.items()) if m is not None]
    vocab = sorted([[int(i), str(c)] for i, c in _vocab_rev().items()])
    return {"cards": cards, "vocab": vocab}


def load_cards():
    """カード表と語彙を Rust へ（1 度だけ）。戻り＝札の数。"""
    if "cards_loaded" not in _ENGINE:
        txt = json.dumps(card_table(), ensure_ascii=False, separators=(",", ":"))
        _ENGINE["cards_loaded"] = engine().theory_load_cards(txt)
    return _ENGINE["cards_loaded"]


def load_effects(path=EFFECTS_PATH):
    if "effects_loaded" not in _ENGINE:
        with open(path, encoding="utf-8") as fh:
            _ENGINE["effects_loaded"] = engine().theory_load_effects(fh.read())
    return _ENGINE["effects_loaded"]


def load_opp_boards(path=OPP_BOARDS_PATH):
    if "boards_loaded" not in _ENGINE:
        _ENGINE["boards_loaded"] = engine().theory_load_opp_boards(path)
    return _ENGINE["boards_loaded"]


# ---------------------------------------------------------------------------------------------------------------
# 局の枠

ROW_NUM_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_NUM_COLS = ("pol_n", "pol_q", "pol_p", "pol_si", "pol_ti", "pol_k")
STR_COLS = ("sig", "pol_sig", "pol_cid", "pol_tcid")


def _col(name, a):
    a = np.ascontiguousarray(a)
    return (name, a.dtype.str.lstrip("<|="), list(a.shape), a.tobytes())


def frame_of(rows, pol, ex, L, ptr, idx, u2c=None, decks=None):
    """1 局（`plan_labels.iter_games` の 1 つ）→ `TheoryFrame`。行は局の順（`idx`）に並べ直し、候補は各行の範囲を連結する。"""
    from opcg_sim.learned.train import plan_labels as PL
    idx = np.asarray(idx, np.int64)
    cols = []
    for k in ROW_NUM_COLS:
        if k in rows:
            cols.append(_col(k, np.asarray(rows[k])[idx]))
    for k, v in (ex or {}).items():
        cols.append(_col(k, np.asarray(v)[idx]))
    pidx = np.concatenate([np.arange(int(ptr[i]), int(ptr[i]) + int(L[i])) for i in idx]) if len(idx) else np.zeros(0, np.int64)
    for k in POL_NUM_COLS:
        if k in pol:
            cols.append(_col(k, np.asarray(pol[k])[pidx]))
    strs = []
    for k in STR_COLS:
        src = rows if k in rows else pol
        if k in src:
            sel = idx if src is rows else pidx
            strs.append((k, [str(x) for x in np.asarray(src[k])[sel]]))
    if u2c is None:
        u2c = PL.uuid_map(pol, L, ptr, idx)
    deck_t = None if decks is None else (list(decks[0]), list(decks[1]))
    return engine().TheoryFrame(cols, strs, [(str(a), str(b)) for a, b in u2c.items()], deck_t)


def row_frame(sc, tok, ci, cands=(), who=0, turn=1):
    """生の 1 行（`sc`／`tok` は float32・`ci` は int64）と候補の記述子（`sig`・`cid`・`tcid`・`si`・`ti`・`k`）→ 1 行の枠
    （T18 の器が生の局面を Rust に読ませる・`live_theory.raw_row` の形）。"""
    n = len(cands)
    rows = {"who": np.array([who], np.int64), "turn": np.array([turn], np.int64), "seed": np.array([0], np.int64),
            "z": np.array([0.0], np.float32), "kind": np.array([0], np.int64), "pol_len": np.array([n], np.int64),
            "pol_chosen": np.array([-1], np.int64), "pol_v0": np.array([0.0], np.float32)}
    pol = {"pol_sig": np.array([json.dumps(c["sig"]) for c in cands] or [""], object)[:n],
           "pol_cid": np.array([c["cid"] or "" for c in cands] or [""], object)[:n],
           "pol_tcid": np.array([c["tcid"] or "" for c in cands] or [""], object)[:n],
           "pol_si": np.array([int(c["si"]) for c in cands], np.int64),
           "pol_ti": np.array([int(c["ti"]) for c in cands], np.int64),
           "pol_k": np.array([int(c["k"]) for c in cands], np.int64)}
    ex = {"sc": np.asarray(sc, np.float32)[None], "tok": np.asarray(tok, np.float32)[None],
          "ci": np.asarray(ci, np.int64)[None]}
    return frame_of(rows, pol, ex, np.array([n]), np.array([0]), np.array([0]), u2c={})
