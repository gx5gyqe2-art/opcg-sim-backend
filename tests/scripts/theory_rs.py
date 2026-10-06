"""**理論の器の Rust 全移植・段 1（2026-10-06）**: Python → Rust の入力の受け渡し（`docs/reports/2026-10-06_port_stage1_2.md`）。

境界（計画 §4.1・ユーザ決定 2026-10-06）: 記録の読み込み・合成デッキの作り直し・カード DB（パーサ）・集計と JSON の書き出しは
Python に残る。Rust へ渡すのは:

* **カード表と語彙**（1 度だけ・`load_cards()`）——`PL.Cards.info` の 9 項目と、理論が原本（`load_db().get_card`）から読む
  項目（種別・パワー・コスト・印字カウンター・キーワード・特徴・色・名前・属性）と `n_rel_feat.profile` の
  `counter_event`／`thr`。`check_cards()` が Rust の写しと項目ごとに突き合わせる。
* **効果の木**（1 度だけ・`load_effects()`・`opcg_sim/data/opcg_effects.json` の文字列そのまま）。`check_effects()`。
* **fixture**（`harm_profile.json`・`opp_boards.json`）——Rust が自分で読む。`check_fixtures()` が全部の値をビットで比べる。
* **局の枠**（`frame_of(...)`・行の `scalars`／`tokens` は器が見る float32 のまま・候補・uuid→card・デッキ）。
  `check_frame()` が往復の恒等を確かめる。

記録の形（`enc`／`dec`）: 浮動小数は 16 桁の 16 進（ビット）・組は `{"t": …}`・dict は `{"d": [[k, v], …]}`・numpy の配列は
`{"nd": dtype, "s": 形, "h": 生のバイト}`・numpy のスカラーは `{"npf"|"npi": …}`・大域の物（カード表・語彙・盤面の分布）は
`{"obj": 名前}`。`theory_capture.py`（呼び出しの記録）と Rust の `theory::pyval` が同じ形を読み書きする。
"""
import json
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
HARM_PROFILE_PATH = os.path.join(_ROOT, "tests", "fixtures", "harm_profile.json")

_ENGINE = {}


def engine():
    if "m" not in _ENGINE:
        import opcg_engine
        if not hasattr(opcg_engine, "theory_leaf_call"):
            raise RuntimeError("opcg_engine の wheel が古い（theory_leaf_call が無い）——`make rust-develop` で作り直す")
        _ENGINE["m"] = opcg_engine
    return _ENGINE["m"]


# ---------------------------------------------------------------------------------------------------------------
# 記録の形

#: 大域の物の識別（`id` → 名前）。`register_obj` で足す。
_OBJS = {}


def register_obj(obj, name):
    _OBJS[id(obj)] = (obj, name)


def _vocab_rev():
    if "rev" not in _ENGINE:
        from opcg_sim.learned.vocab import shared_vocab
        _ENGINE["rev"] = {i: c for c, i in shared_vocab().items()}
    return _ENGINE["rev"]


def _obj_name(v):
    hit = _OBJS.get(id(v))
    if hit is not None and hit[0] is v:
        return hit[1]
    return None


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
    nm = _obj_name(v)
    if nm is not None:
        return {"obj": nm}
    if isinstance(v, tuple):
        return {"t": [enc(x) for x in v]}
    if isinstance(v, list):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        if len(v) > 1000 and v == _vocab_rev():
            register_obj(v, "idx2cid")
            return {"obj": "idx2cid"}
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
    if isinstance(v, slice):
        return {"t": ["slice", v.start, v.stop]}
    if isinstance(v, (set, frozenset)):
        return sorted(enc(x) for x in v)
    cid = getattr(v, "card_id", None)
    if cid is not None and hasattr(v, "abilities"):
        return {"obj": "card:%s" % cid}
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
    """1 枚の札 → Rust の `TCard` の JSON（Python の器が読む値をそのまま・型変換も同じ式）。"""
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


def card_table_json():
    return json.dumps(card_table(), ensure_ascii=False, separators=(",", ":"))


def load_cards():
    """カード表と語彙を Rust へ（1 度だけ）。戻り＝札の数。"""
    if "cards_loaded" not in _ENGINE:
        _ENGINE["cards_loaded"] = engine().theory_load_cards(card_table_json())
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


def load_all():
    load_cards()
    load_opp_boards()
    return engine()


def _same(a, b):
    """記録の形のまま（＝浮動小数はビットで）等しいか。"""
    return json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)


def check_cards():
    """Rust の写しが Python の表と項目ごとに一致するか。戻り＝札の数（違えば例外）。"""
    tab = card_table()
    load_cards()
    got = dec(json.loads(engine().theory_dump_cards()))
    want_cards = [{**c, "thr": [list(map(lambda x: float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x, r))
                                for r in c["thr"]]} for c in tab["cards"]]
    if len(got["cards"]) != len(want_cards):
        raise AssertionError("札の数が違う")
    for a, b in zip(got["cards"], want_cards):
        if _same(enc(a), enc(b)):
            continue
        raise AssertionError("札 %s が違う: %r != %r" % (b["id"], a, b))
    if [list(x) for x in got["vocab"]] != tab["vocab"]:
        raise AssertionError("語彙が違う")
    return len(want_cards)


def check_effects(path=EFFECTS_PATH):
    """効果の木の写しが `json.load` の結果と型・値ごとに一致するか（浮動小数はビット）。"""
    load_effects(path)
    with open(path, encoding="utf-8") as fh:
        want = json.load(fh)["cards"]
    got = engine().theory_dump_effects()
    if got != json.dumps(enc(want), ensure_ascii=False, separators=(",", ":")):
        raise AssertionError("効果の木の写しが違う")
    return len(want)


def _floats_of(v, out):
    if isinstance(v, float):
        out.append(v)
    elif isinstance(v, dict):
        for k, x in v.items():
            _floats_of(x, out)
    elif isinstance(v, list):
        for x in v:
            _floats_of(x, out)
    return out


def check_fixtures():
    """fixture を Rust が読んだ値が Python の `json.load` と全部の値でビット一致するか。戻り＝比べた浮動小数の数。"""
    n = 0
    for p in (HARM_PROFILE_PATH, OPP_BOARDS_PATH):
        with open(p, encoding="utf-8") as fh:
            want = json.load(fh)
        got = engine().theory_read_fixture(p)
        if got != json.dumps(enc(want), ensure_ascii=False, separators=(",", ":")):
            raise AssertionError("fixture %s の読みが違う" % p)
        n += len(_floats_of(want, []))
    # 盤面の分布の型つきの表（`load_opp_boards` と同じ形）
    with open(OPP_BOARDS_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    want = {int(k): [[float(m), [[float(p), bool(b)] for p, b in bodies]] for m, bodies in v]
            for k, v in (raw.get("by_r") or {}).items()}
    got = dec(json.loads(load_opp_boards()))
    if json.dumps(enc(got)) != json.dumps(enc(want)):
        raise AssertionError("盤面の分布の表が違う")
    return n


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


def check_frame(fr, rows, pol, ex, L, ptr, idx):
    """枠の往復が恒等か（全部の列のバイト・文字列・uuid→card）。"""
    from opcg_sim.learned.train import plan_labels as PL
    idx = np.asarray(idx, np.int64)
    for k, v in list((ex or {}).items()) + [(k, rows[k]) for k in ROW_NUM_COLS if k in rows]:
        a = np.ascontiguousarray(np.asarray(v)[idx])
        dt, sh, raw = fr.col(k)
        if dt != a.dtype.str.lstrip("<|=") or list(sh) != list(a.shape) or bytes(raw) != a.tobytes():
            raise AssertionError("列 %s の往復が違う" % k)
    for k in ("sc", "tok"):
        if k in (ex or {}):
            a = np.asarray(ex[k])[idx]
            for j in range(len(idx)):
                if fr.f32_row(k, j) != [float(x) for x in a[j].ravel()]:
                    raise AssertionError("%s の行 %d の float32 → float が違う" % (k, j))
    u2c = PL.uuid_map(pol, L, ptr, idx)
    if fr.u2c() != [(str(a), str(b)) for a, b in u2c.items()]:
        raise AssertionError("uuid→card の往復が違う")
    return True


def check_dirs(dirs, limit_games=0):
    """記録の局ごとに枠を作って往復を確かめる（器と同じ `iter_games`・`theory_bridge._extra` の float32）。戻り＝(局, 行)。"""
    from opcg_sim.learned.train import plan_labels as PL
    import theory_bridge as TB
    g = n = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=TB.ROW_COLS, pol_cols=TB.POL_COLS, extra_fn=TB._extra):
        g += 1
        if limit_games and g > limit_games:
            break
        fr = frame_of(rows, pol, ex, L, ptr, idx)
        check_frame(fr, rows, pol, ex, L, ptr, idx)
        n += len(idx)
    return g if not limit_games else min(g, limit_games), n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="段 1 の受け渡しの恒等の確認（カード表・効果の木・fixture・局の枠）")
    ap.add_argument("--in", dest="inp", nargs="*", default=[])
    ap.add_argument("--limit-games", type=int, default=0)
    a = ap.parse_args(argv)
    print("cards", check_cards())
    print("effects", check_effects())
    print("fixture_floats", check_fixtures())
    if a.inp:
        print("frames(games, rows)", check_dirs(a.inp, a.limit_games))
    return 0


if __name__ == "__main__":
    sys.exit(main())
