"""符号化オラクル（`docs/rust_engine_plan.md` §12.2・WP `rs-p4-encode` の受け入れ道具）。

**問い**: Rust の符号化 v13（`rust/opcg_engine/src/encode/`）は Python 版
（`learned/encoder.py::encode(version=13)`／`learned/n_rel_feat.py::encode_rel`／
`learned/n_eff.py::build_eff_tables`）と**同じ盤面・同じ視点で同じ数**を出すか。

やること:
  1. `rs_diff_replay.Recorder` で局を打ち（random／l1）、各行の `hidden`（記録 v5＝完全な内部状態）を
     集めて **等間隔に `--boards` 局面**を抜く（P3 の問合せオラクルと同じ取り方）。
  2. 各局面 × **両視点**（p1/p2）で Python 側の
     `encode(v13)`（scalars 123・field 10×8・card_idx 24）と
     `encode_rel`（tokens 22×20・rel_om 16×6×5・rel_oo 16×16×5・extra 29）を
     `with_relations` の **有無の両モード**で計算する。
  3. 同じ `hidden` を Rust の `opcg_engine.encode_state(hidden, seat, opts)` に渡して照合する。
  4. 語彙全行（2,652＋PAD）のカード表（`build_eff_tables` の 5 表）を
     `opcg_engine.eff_tables(start, count)` と照合する。
  5. `RS_ENCODE {...}` の 1 行を stdout に出す。

**不一致は列名で報告する**（`scalars.leader_power_me`／`tokens[3].can_attack_now`／
`rel_om[1][0].ko_gap`／`extra.guard_per_card`／`cardtab.ab[1234][2][45]` のように、
どの式が違うかが 1 行で分かる形）。

許容: float は 1e-6（`--tol` で変更可）・`card_idx` は整数一致。

`opcg_engine` が無い／`encode_state` を持たない古い拡張では全行 unimplemented として集計する
（黙って緑にしない）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_encode_oracle.py --boards 200
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_encode_oracle.py \\
      --boards 10 --games 1 --policy random --verbose     # 開発中の速い一次チェック
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402

from harness.game_driver import DEFAULT_MAX_STEPS, InvariantError, load_db, make_seat, run_game  # noqa: E402
from harness.rs_record import manager_from_hidden  # noqa: E402
from rs_diff_replay import Recorder, board_dict, canon, first_diff  # noqa: E402

from opcg_sim.src.learned import encoder as E  # noqa: E402
from opcg_sim.src.learned import n_eff, n_rel_feat  # noqa: E402
from opcg_sim.src.learned.leader_feat import DIMS as LEADER_DIMS  # noqa: E402

try:                        # Rust 拡張は未導入でも動く（その場合は全行 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_EFFECTS = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")

ENC_VERSION = 13


# --- 効果 JSON（Rust が読むカード定義）------------------------------------------

def ensure_effects_json(path: str, db) -> str:
    """効果構造 JSON を用意する（無い／カード DB と食い違うなら作り直す）。

    Rust と Python が**同じカード DB**を見ていることが照合の前提（`rs_query_oracle` と同じ）。
    """
    want = db.db_hash()
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                head = json.load(f)
            if head.get("source", {}).get("db_hash") == want:
                return path
            print(f"[effects] {path} はカード DB と食い違う（db_hash 不一致）ので再生成する")
        except (OSError, ValueError):
            print(f"[effects] {path} が読めないので再生成する")
    else:
        print(f"[effects] {path} が無いので生成する")
    subprocess.run([_sys.executable, "-m", "opcg_sim.tools.export_effects_json", "--out", path],
                   cwd=_REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    return path


# --- 列名（不一致をどの式かで報告するための対応表）--------------------------------

_SLOT12 = (["leaderM", "leaderO"]
           + [f"fieldM{i}" for i in range(E.MAX_FIELD)]
           + [f"fieldO{i}" for i in range(E.MAX_FIELD)])
_DECK_AGG = ["counter_total", "counter_density", "blockers", "events", "highcost_char"]
_FIELD_AGG = ["total_power", "high_power", "blockers"]
_HAND_AGG = ["counter_total", "counter_cards", "max_counter", "blockers", "events"]

SCALAR_COLS = (
    ["life_me", "life_opp", "don_active_me", "don_rested_me", "don_active_opp", "don_rested_opp",
     "hand_me", "hand_opp", "field_me", "field_opp", "turn_count", "is_my_turn",
     "leader_power_me", "leader_power_opp"]                                        # v1: 0..14
    + ["leader_don_me", "leader_don_opp"]                                          # v2: 14..16
    + ["deck_me", "deck_opp", "trash_me", "trash_opp", "koed_me", "koed_opp"]      # v3a: 16..22
    + [f"used_{s}" for s in _SLOT12]                                               # v3b: 22..34
    + [f"sick_{s}" for s in _SLOT12]                                               # v3b: 34..46
    + [f"deck_{k}" for k in _DECK_AGG]                                             # v4: 46..51
    + [f"oppfield_{k}" for k in _FIELD_AGG]                                        # v5: 51..54
    + ["playable_chars"]                                                           # v5: 54
    + [f"hand_{k}" for k in _HAND_AGG]                                             # v6: 55..60
    + ["onplay_live", "onplay_keep", "onplay_dead"]                                # v7: 60..63
    + [f"myfield_{k}" for k in _FIELD_AGG]                                         # v8: 63..66
    + ["don_deck_me", "don_deck_opp", "deck_apex_power", "deck_apex_cost"]         # v9: 66..70
    + [f"leaderfeat_me.{d}" for d in LEADER_DIMS]                                  # v11/v12: 70..82
    + [f"leaderfeat_opp.{d}" for d in LEADER_DIMS]                                 # v11/v12: 82..94
    + [f"extra.{c}" for c in n_rel_feat.EXTRA_COLS]                                # v13: 94..123
)

FIELD_COLS = [f"field[{r}].{c}"
              for r in range(2 * E.MAX_FIELD)
              for c in (["cost", "power", "is_rest", "attached_don"]
                        + [f"kw_{k}" for k in E.KEYWORDS])]

CARD_IDX_COLS = (["card_idx.leaderM", "card_idx.leaderO"]
                 + [f"card_idx.fieldM{i}" for i in range(E.MAX_FIELD)]
                 + [f"card_idx.fieldO{i}" for i in range(E.MAX_FIELD)]
                 + [f"card_idx.hand{i}" for i in range(E.MAX_HAND)]
                 + ["card_idx.stageM", "card_idx.stageO"])

TOKEN_COLS = [f"tokens[{i}].{c}"
              for i in range(n_rel_feat.N_TOK) for c in n_rel_feat.S_COLS]
REL_OM_COLS = [f"rel_om[{i}][{j}].{c}"
               for i in range(n_rel_feat.N_OWN)
               for j in range(n_rel_feat.N_OPP)
               for c in n_rel_feat.R_COLS]
REL_OO_COLS = [f"rel_oo[{i}][{j}].{c}"
               for i in range(n_rel_feat.N_OWN)
               for j in range(n_rel_feat.N_OWN)
               for c in n_rel_feat.R_COLS]
EXTRA_COLS = [f"extra.{c}" for c in n_rel_feat.EXTRA_COLS]

assert len(SCALAR_COLS) == E.scalars_dim(ENC_VERSION), (len(SCALAR_COLS), E.scalars_dim(ENC_VERSION))
assert len(FIELD_COLS) == E.field_dim()
assert len(CARD_IDX_COLS) == 2 + 2 * E.MAX_FIELD + E.MAX_HAND + 2


# --- 盤面の収集（`rs_query_oracle` と同じ取り方）-----------------------------------

def collect_boards(db, games: int, policy: str, seed_base: int, max_steps: int) -> list:
    """`games` 局を打ち、全行（setup＋各行動後）の `hidden`／`state` を集める。"""
    rows = []
    kind = "random" if policy == "random" else "ai"
    for i in range(games):
        seed = seed_base + i
        seats = {"p1": make_seat(kind=kind), "p2": make_seat(kind=kind)}
        rec = Recorder(seed, policy, record_hidden=True)
        try:
            run_game(seed, db, seats=seats, observers=(rec,), max_steps=max_steps)
        except (InvariantError, Exception) as e:  # noqa: BLE001 - 局が落ちてもそこまでの行は使える
            print(f"[boards] seed={seed} policy={policy} で打ち切り: {type(e).__name__}: {e}")
        if rec.setup is not None:
            rows.append({"seed": seed, "policy": policy, "hidden": rec.setup["hidden"],
                         "state": rec.setup["state"]})
        for step in rec.steps:
            if "hidden" in step:
                rows.append({"seed": seed, "policy": policy, "hidden": step["hidden"],
                             "state": step["state"]})
    return rows


def evenly(rows: list, n: int) -> list:
    """等間隔に n 件抜く（n >= len(rows) なら全部）。"""
    if n >= len(rows) or n <= 0:
        return list(rows)
    step = len(rows) / n
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]


# --- 照合 -----------------------------------------------------------------------

def compare(cols, py, rs, tol, totals, firsts, where, exact=False):
    """1 本の配列を列名つきで照合する（不一致は最初の 1 件を `firsts` に積む）。"""
    py = np.asarray(py, np.float64).ravel()
    rs = np.asarray(rs, np.float64).ravel()
    if py.shape != rs.shape:
        totals["shape_mismatch"] += 1
        if not firsts:
            firsts.append({"kind": "shape", "where": where,
                           "python_len": int(py.size), "rust_len": int(rs.size)})
        return
    totals["values"] += int(py.size)
    totals["nonzero"] += int(np.count_nonzero(py))
    bad = (py != rs) if exact else ~np.isclose(py, rs, rtol=0.0, atol=tol, equal_nan=True)
    n_bad = int(bad.sum())
    if not n_bad:
        return
    totals["mismatch"] += n_bad
    idx = int(np.argmax(bad))
    col = cols[idx] if idx < len(cols) else f"?[{idx}]"
    totals["mismatch_by_col"][col] = totals["mismatch_by_col"].get(col, 0) + 1
    if not firsts:
        firsts.append({"kind": "value", "where": where, "column": col, "index": idx,
                       "python": float(py[idx]), "rust": float(rs[idx]),
                       "n_bad_in_array": n_bad})


def rust_encode(hidden, seat, skip_relations):
    opts = json.dumps({"skip_relations": bool(skip_relations), "skip_onplay": False})
    out = opcg_engine.encode_state(
        json.dumps(hidden, ensure_ascii=False, default=str), seat, opts)
    return json.loads(out)


def check_board(db, row, board_index, vocab, tol, totals, firsts, verbose=False):
    hidden = row["hidden"]
    manager = manager_from_hidden(db, hidden)

    # 自己検査: 復元した盤面が記録の `state` と一致するか（Python 側だけで閉じる）。
    got = dict(board_dict(manager))
    got.pop("pending_request", None)
    exp = dict(row["state"])
    exp.pop("pending_request", None)
    if json.dumps(canon(exp), sort_keys=True, ensure_ascii=False, default=str) != \
            json.dumps(canon(got), sort_keys=True, ensure_ascii=False, default=str):
        totals["restore_mismatch"] += 1
        if not firsts:
            firsts.append({"kind": "restore", "board": board_index, "seed": row["seed"],
                           "path": first_diff(canon(exp), canon(got))})
        return

    for seat in ("p1", "p2"):
        # Python 側（毎回まっさらな盤面から＝符号化の副作用を持ち越さない）。
        # `with_pending=True`＝要求（pending request）を塞がない。符号化 v7 の登場時スキャンと
        # `_leader_act_avail` は要求と合法手を読むので、塞いだままだと Python 側が常に
        # (0,0,0)／0.0 に潰れ、**照合が空振りする**（Rust だけが本物を計算してしまう）。
        manager = manager_from_hidden(db, hidden, with_pending=True)
        enc = E.encode(manager, seat, vocab, version=ENC_VERSION)
        manager = manager_from_hidden(db, hidden, with_pending=True)
        rel_on = n_rel_feat.encode_rel(manager, seat, with_relations=True)
        manager = manager_from_hidden(db, hidden, with_pending=True)
        rel_off = n_rel_feat.encode_rel(manager, seat, with_relations=False)

        if opcg_engine is None or not hasattr(opcg_engine, "encode_state"):
            totals["unimplemented"] += 1
            continue
        try:
            got_on = rust_encode(hidden, seat, skip_relations=False)
            got_off = rust_encode(hidden, seat, skip_relations=True)
        except NotImplementedError as e:
            totals["unimplemented"] += 1
            if not firsts:
                firsts.append({"kind": "unimplemented", "board": board_index, "detail": str(e)})
            continue
        except ValueError as e:      # 契約違反（ハーネス側／Rust 側のバグ）＝黙って通さない
            totals["bad_payload"] += 1
            if not firsts:
                firsts.append({"kind": "bad_payload", "board": board_index, "detail": str(e)})
            continue

        w = f"board={board_index} seed={row['seed']} policy={row['policy']} seat={seat}"
        compare(SCALAR_COLS, enc["scalars"], got_on["scalars"], tol, totals, firsts,
                f"{w} scalars")
        compare(FIELD_COLS, enc["field"], got_on["field"], tol, totals, firsts, f"{w} field")
        compare(CARD_IDX_COLS, enc["card_idx"], got_on["card_idx"], tol, totals, firsts,
                f"{w} card_idx", exact=True)
        compare(TOKEN_COLS, rel_on["tokens"], got_on["tokens"], tol, totals, firsts,
                f"{w} tokens(rel=on)")
        compare(REL_OM_COLS, rel_on["rel_om"], got_on["rel_om"], tol, totals, firsts,
                f"{w} rel_om")
        compare(REL_OO_COLS, rel_on["rel_oo"], got_on["rel_oo"], tol, totals, firsts,
                f"{w} rel_oo")
        compare(EXTRA_COLS, rel_on["extra"], got_on["extra"], tol, totals, firsts, f"{w} extra")

        # with_relations=False: tokens/extra は同値・関係は返さない（Python は None）。
        compare(TOKEN_COLS, rel_off["tokens"], got_off["tokens"], tol, totals, firsts,
                f"{w} tokens(rel=off)")
        compare(EXTRA_COLS, rel_off["extra"], got_off["extra"], tol, totals, firsts,
                f"{w} extra(rel=off)")
        compare(SCALAR_COLS, enc["scalars"], got_off["scalars"], tol, totals, firsts,
                f"{w} scalars(rel=off)")
        if rel_off["rel_om"] is not None or got_off["rel_om"]:
            totals["mismatch"] += 1
            if not firsts:
                firsts.append({"kind": "rel_off_not_empty", "where": w,
                               "python_none": rel_off["rel_om"] is None,
                               "rust_len": len(got_off["rel_om"])})
        totals["views"] += 1
        # 空振り検査（`rs_query_oracle` の nonempty_targets と同じ役目）: 実測が要る列
        # （v7 の登場時スキャン・extra[0] のリーダー起動）が**実際に非零になった**視点の数。
        # ここが 0 のまま mismatch=0 なら、両側とも既定値を返しているだけの恐れがある。
        if enc["scalars"][60] > 0:
            totals["onplay_live_views"] += 1
        if enc["scalars"][62] > 0:
            totals["onplay_dead_views"] += 1
        if rel_on["extra"][0] > 0:
            totals["leader_act_views"] += 1
        if verbose:
            print(f"[view] {w} ok", flush=True)


def check_card_table(db, vocab, tol, totals, firsts, chunk=256):
    """語彙全行のカード表（`build_eff_tables` の 5 表）を Rust と照合する。"""
    stats, ab, abm, pwr, isl = n_eff.build_eff_tables(db, vocab)
    n = stats.shape[0]
    totals["cardtab_rows"] = 0
    if opcg_engine is None or not hasattr(opcg_engine, "eff_tables"):
        totals["unimplemented"] += 1
        return
    start = 0
    while start < n:
        out = json.loads(opcg_engine.eff_tables(start, chunk))
        if out["n"] != n:
            totals["shape_mismatch"] += 1
            if not firsts:
                firsts.append({"kind": "cardtab_rows", "python": n, "rust": out["n"]})
            return
        cnt = out["count"]
        if cnt <= 0:
            break
        end = start + cnt
        cols_stats = [f"cardtab.stats[{r}][{c}]"
                      for r in range(start, end) for c in range(n_eff.STATS_DIM)]
        cols_ab = [f"cardtab.ab[{r}][{k}][{c}]"
                   for r in range(start, end)
                   for k in range(n_eff.MAX_AB) for c in range(n_eff.ABILITY_DIM)]
        cols_abm = [f"cardtab.abm[{r}][{k}]"
                    for r in range(start, end) for k in range(n_eff.MAX_AB)]
        cols_pwr = [f"cardtab.pwr[{r}]" for r in range(start, end)]
        cols_isl = [f"cardtab.isl[{r}]" for r in range(start, end)]
        compare(cols_stats, stats[start:end], out["stats"], tol, totals, firsts, "cardtab.stats")
        compare(cols_ab, ab[start:end], out["ab"], tol, totals, firsts, "cardtab.ab")
        compare(cols_abm, abm[start:end], out["abm"], tol, totals, firsts, "cardtab.abm")
        compare(cols_pwr, pwr[start:end], out["pwr"], tol, totals, firsts, "cardtab.pwr")
        compare(cols_isl, isl[start:end], out["isl"], tol, totals, firsts, "cardtab.isl")
        totals["cardtab_rows"] += cnt
        start = end


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="符号化 v13（scalars/field/card_idx・tokens/rel/extra・カード表）を"
                    "Python と Rust で照合する")
    ap.add_argument("--boards", type=int, default=200, help="照合する局面数（既定 200）")
    ap.add_argument("--games", type=int, default=10, help="局面を採る局数（方策ごと・既定 10）")
    ap.add_argument("--policy", choices=["random", "l1", "both"], default="both",
                    help="局面を採る方策（既定 both＝random と l1 の両方から等分に採る）")
    ap.add_argument("--seed-base", type=int, default=940000, help="seed の基点（既定 940000）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="1局の上限ステップ")
    ap.add_argument("--effects", default=DEFAULT_EFFECTS,
                    help="効果構造 JSON（Rust のカード定義表・カード DB と食い違えば再生成する）")
    ap.add_argument("--tol", type=float, default=1e-6, help="float の許容差（既定 1e-6）")
    ap.add_argument("--no-cards", action="store_true", help="カード表の照合を省く（開発中の絞り込み用）")
    ap.add_argument("--verbose", action="store_true", help="局面ごとの進捗を出す")
    args = ap.parse_args(argv)

    db = load_db()
    effects_path = ensure_effects_json(args.effects, db)
    vocab_ids = n_eff.default_vocab_ids()
    vocab = E.vocab_from_ids(vocab_ids)
    if opcg_engine is not None and hasattr(opcg_engine, "load_masters"):
        n = opcg_engine.load_masters(effects_path)
        print(f"[effects] load_masters({effects_path}) -> {n} cards")
    if opcg_engine is not None and hasattr(opcg_engine, "set_vocab"):
        n = opcg_engine.set_vocab(json.dumps(vocab_ids))
        print(f"[vocab] set_vocab -> {n} ids")

    totals = {"values": 0, "mismatch": 0, "nonzero": 0, "views": 0, "unimplemented": 0,
              "bad_payload": 0, "shape_mismatch": 0, "restore_mismatch": 0,
              "cardtab_rows": 0, "onplay_live_views": 0, "onplay_dead_views": 0,
              "leader_act_views": 0, "mismatch_by_col": {}}
    firsts = []

    t0 = time.time()
    if not args.no_cards:
        check_card_table(db, vocab, args.tol, totals, firsts)
        print(f"[cardtab] rows={totals['cardtab_rows']} mismatch={totals['mismatch']} "
              f"({time.time() - t0:.1f}s)")

    policies = ["random", "l1"] if args.policy == "both" else [args.policy]
    rows = []
    for i, policy in enumerate(policies):
        t1 = time.time()
        got = collect_boards(db, args.games, policy, args.seed_base + 1000 * i, args.max_steps)
        print(f"[boards] policy={policy} games={args.games} rows={len(got)} "
              f"({time.time() - t1:.1f}s)")
        rows.append(got)
    per_policy = max(1, args.boards // len(rows)) if rows else 0
    boards = []
    for got in rows:
        boards.extend(evenly(got, per_policy))
    boards = boards[:args.boards] if args.boards > 0 else boards

    t1 = time.time()
    for i, row in enumerate(boards):
        try:
            check_board(db, row, i, vocab, args.tol, totals, firsts, verbose=False)
        except Exception as e:  # noqa: BLE001 - ハーネス自身の事故も集計に載せる
            totals["bad_payload"] += 1
            if not firsts:
                firsts.append({"kind": "harness_error", "board": i,
                               "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        if args.verbose and (i + 1) % 10 == 0:
            print(f"[boards] {i + 1}/{len(boards)} values={totals['values']} "
                  f"mismatch={totals['mismatch']} ({time.time() - t1:.1f}s)", flush=True)

    summary = {
        "boards": len(boards),
        "views": totals["views"],
        "cols_scalars": E.scalars_dim(ENC_VERSION),
        "cols_field": E.field_dim(),
        "cols_card_idx": len(CARD_IDX_COLS),
        "cols_tokens": n_rel_feat.N_TOK * n_rel_feat.S_DIM,
        "cols_rel_om": n_rel_feat.N_OWN * n_rel_feat.N_OPP * n_rel_feat.R_DIM,
        "cols_rel_oo": n_rel_feat.N_OWN * n_rel_feat.N_OWN * n_rel_feat.R_DIM,
        "cols_extra": n_rel_feat.EXTRA_DIM,
        "cardtab_rows": totals["cardtab_rows"],
        "vocab": len(vocab_ids),
        "values": totals["values"],
        "nonzero": totals["nonzero"],
        "onplay_live_views": totals["onplay_live_views"],
        "onplay_dead_views": totals["onplay_dead_views"],
        "leader_act_views": totals["leader_act_views"],
        "mismatch": totals["mismatch"],
        "mismatch_by_col": dict(sorted(totals["mismatch_by_col"].items(),
                                       key=lambda kv: -kv[1])[:10]),
        "shape_mismatch": totals["shape_mismatch"],
        "restore_mismatch": totals["restore_mismatch"],
        "unimplemented": totals["unimplemented"],
        "bad_payload": totals["bad_payload"],
        "tol": args.tol,
        "games": args.games,
        "policy": args.policy,
        "seed_base": args.seed_base,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "seconds": round(time.time() - t0, 1),
        "first": firsts[0] if firsts else None,
    }
    print("RS_ENCODE " + json.dumps(summary, ensure_ascii=False))
    bad = (totals["mismatch"] or totals["bad_payload"] or totals["shape_mismatch"]
           or totals["restore_mismatch"] or totals["unimplemented"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
