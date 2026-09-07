"""forward オラクル（`docs/rust_engine_plan.md` §12.2・WP `rs-p4-net` の受け入れ道具）。

**問い**: Rust の NRel forward（`net/{npz,nrel}.rs`）は Python 版（`learned/n_rel.py::NRelNet` の
`value` と `nrel_priors`）と**同じ符号化に同じ値**を返すか。

符号化 WP（`rs-p4-encode`）とは**独立に**検証する: 入力は Python の
`NRelValueAdapter.encode_state`（scalars/card_idx/tokens/関係）を JSON で Rust へ渡す。
候補（方策の入力）も Python の探索用合法手（`adapter.OPCGGame.legal_actions`）をそのまま渡し、
uuid → card_id と 22 枠 index の解決だけハーネスが付ける（盤面の解決は符号化／探索 WP の担当）。
**Rust が計算するのは forward そのもの**（カード表・トークン・本体・候補素性 139・予算 3・
seg-softmax）＝この WP の範囲だけを見る。

やること:
  1. `rs_query_oracle.collect_boards` で局を打ち（random／l1）、記録 v5 の `hidden` から
     等間隔に `--boards` 局面を抜く（P3 の問合せオラクルと同じ取り方）。
  2. カード表の元（`n_eff.build_eff_tables` の 5 表＋`n_rel_feat.profile_table` の `ret_don`）を
     npz へ書き、`opcg_engine.load_net(npz, tables_npz)` に読ませる（Rust の npz 読みも通す）。
  3. 各局面 × 両視点で Python の `value`（`adapter.predict_state`）と、手番側の `priors`
     （`n_rel.nrel_priors`）を取り、同じ符号化で Rust の `net_eval` を呼んで照合する。
  4. `RS_NET {...}` の 1 行を stdout に出す（許容 1e-5）。

`opcg_engine` が無い／`net_eval` を持たない古い拡張では全件 unimplemented として集計する
（黙って緑にしない）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_net_oracle.py --boards 200
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_net_oracle.py \\
      --boards 20 --games 2 --policy random --verbose     # 開発中の速い一次チェック
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402

from harness.game_driver import DEFAULT_MAX_STEPS, load_db  # noqa: E402
from harness.rs_record import manager_from_hidden  # noqa: E402
from rs_query_oracle import collect_boards, evenly  # noqa: E402

from opcg_sim.src.learned import n_eff as NE  # noqa: E402
from opcg_sim.src.learned import n_rel as NR  # noqa: E402
from opcg_sim.src.learned import n_rel_feat as NRF  # noqa: E402
from opcg_sim.src.learned.adapter import OPCGGame  # noqa: E402

try:                        # Rust 拡張は未導入でも動く（その場合は全件 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_NET = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "learned", "nrel_a1.npz")


# --- Rust へ渡す入力 -------------------------------------------------------------

def dump_tables(path: str, adapter) -> None:
    """カード表の元（`build_eff_tables` の 5 表＋`ret_don`）を npz で書く。

    Rust の `net::card_table` はここから `card_table()` を計算する＝**カード表の計算そのものは
    Rust 側**で、Python から渡すのは符号化 WP が持つ「表の元」だけ。
    """
    net = adapter.net
    np.savez_compressed(path, STATS=net.STATS, AB=net.AB, ABM=net.ABM, PWR=net.PWR, ISL=net.ISL,
                        RET=np.asarray(adapter.ptab_ret, np.float32))


def encoding_json(sc, ci, tok, rel_om, rel_oo, extra) -> str:
    """`encode/mod.rs` の `Encoding` と同じ鍵（B=1 の 1 視点ぶん）。"""
    return json.dumps({
        "scalars": np.asarray(sc, np.float32)[0].tolist(),
        "card_idx": np.asarray(ci)[0].tolist(),
        "tok": np.asarray(tok, np.float32)[0].tolist(),
        "rel_om": np.asarray(rel_om, np.float32)[0].tolist(),
        "rel_oo": np.asarray(rel_oo, np.float32)[0].tolist(),
        "extra": np.asarray(extra, np.float32).tolist(),
    })


def legal_json(manager, me_name: str, legal: list) -> str:
    """探索用合法手 ＋ 盤面で解けた識別（card_id／22 枠 index）。

    Python 側 `n_eff._cand_row`／`n_rel._cand_rows` が uuid から引くもの（カードの `card_id` と
    `NR._slots` 上の位置）を**同じ規則で**先に解く。Rust は解決済みの識別を受け取り、
    素性 139 と予算 3 の**計算**だけを行う（uuid の解決は `rs-p4-encode`／`rs-p4-legal` の担当）。
    """
    uidx = NE._uuid_index(manager)
    me = manager.p1 if manager.p1.name == me_name else manager.p2
    opp = manager.p2 if me is manager.p1 else manager.p1
    smap = {getattr(c, "uuid", None): i for i, c in enumerate(NRF._slots(me, opp)) if c is not None}

    def card_id(card):
        return getattr(getattr(card, "master", None), "card_id", None) if card is not None else None

    rows = []
    for mv in legal:
        p = mv.get("payload") or {}
        su = mv.get("card_uuid") or p.get("uuid")
        tids = p.get("target_ids") or []
        tu = tids[0] if tids else None
        rows.append({
            "action_type": mv.get("action_type") or "",
            "payload": {"don_k": p.get("don_k"), "target_ids": list(tids)},
            "card_id": card_id(uidx.get(su)),
            "target_card_id": card_id(uidx.get(tu)),
            "si": int(smap.get(su, -1)),
            "ti": int(smap.get(tu, -1)),
        })
    return json.dumps({"moves": rows}, ensure_ascii=False, default=str)


# --- 1 局面の照合 --------------------------------------------------------------

def restore_battle_owners(manager, hidden: dict) -> None:
    """復元した盤面を「手番と合法手が引ける」状態にする。

    2 点だけ `manager_from_hidden`（P1 の契約）の外側で補う:
      - `active_battle` の所在の持ち主（記録 v3 以降の `attacker_owner`／`target_owner`）。
        復元器は盤面 dict の照合に要らないので入れていないが、`pending_actor_action` は
        BLOCK/COUNTER 窓でこれを読む。
      - `get_pending_request` の封じ（復元器が入れる恒常 None のラムダ）を外す。封じの理由は
        上の 2 欄が無いと KeyError になることなので、戻した後は本物を呼んでよい。外さないと
        `get_legal_actions` が空を返し、方策の照合が**空振りで緑になる**。
    共有ハーネス（`tests/harness/rs_record.py`）は触らず、本オラクルの中だけで完結させる。
    """
    ab = (hidden.get("manager") or {}).get("active_battle")
    if ab and manager.active_battle is not None:
        for key in ("attacker_owner", "target_owner"):
            name = ab.get(key)
            if name:
                manager.active_battle[key] = manager.p1 if name == manager.p1.name else manager.p2
    manager.__dict__.pop("get_pending_request", None)


def check_board(db, row: dict, game, adapter, priors_fn, board_index: int, tol: float,
                totals: dict, firsts: list) -> None:
    manager = manager_from_hidden(db, row["hidden"])
    restore_battle_owners(manager, row["hidden"])
    pa = manager.pending_actor_action()
    actor = pa[0] if pa else None
    for name in (manager.p1.name, manager.p2.name):
        sc, ci, tok, om, oo, R = adapter.encode_state(manager, name)
        enc_json = encoding_json(sc, ci, tok, om, oo, R["extra"])
        py_value = adapter.predict_state(manager, name)

        py_priors, moves_json, n_moves = None, "[]", 0
        if name == actor:
            legal = game.legal_actions(manager)
            if legal:
                n_moves = len(legal)
                moves_json = legal_json(manager, name, legal)
                py_priors = priors_fn(manager, legal)
                if py_priors is None:
                    totals["priors_error"] += 1
                    if not firsts:
                        firsts.append({"kind": "python_priors_none", "board": board_index,
                                       "seed": row["seed"], "moves": n_moves})
                    return

        if opcg_engine is None or not hasattr(opcg_engine, "net_eval"):
            totals["unimplemented"] += 1
            continue
        try:
            out = json.loads(opcg_engine.net_eval(enc_json, moves_json))
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

        # value（両視点）
        totals["values"] += 1
        err = abs(float(out["value"]) - float(py_value))
        totals["value_max_abs_err"] = max(totals["value_max_abs_err"], err)
        if not (err <= tol):
            totals["mismatch"] += 1
            totals["value_mismatch"] += 1
            if not firsts:
                firsts.append({"kind": "value", "board": board_index, "seed": row["seed"],
                               "seat": name, "python": float(py_value),
                               "rust": float(out["value"]), "abs_err": err})
        elif abs(float(py_value)) > 1e-6:
            totals["nonzero_values"] += 1

        # priors（手番側のみ＝Python の `nrel_priors` が定義される視点）
        got = list(out.get("priors") or [])
        if py_priors is None:
            if got:
                totals["mismatch"] += 1
                if not firsts:
                    firsts.append({"kind": "priors_unexpected", "board": board_index,
                                   "seat": name, "rust_len": len(got)})
            continue
        exp = [float(x) for x in np.asarray(py_priors).reshape(-1)]
        if len(got) != len(exp):
            totals["mismatch"] += 1
            totals["priors_mismatch"] += 1
            if not firsts:
                firsts.append({"kind": "priors_len", "board": board_index, "seat": name,
                               "python": len(exp), "rust": len(got)})
            continue
        totals["priors_rows"] += 1
        totals["priors_cands"] += len(exp)
        # 空振り検査: どの種類の手を実際に通したか（素性 139 の action onehot と
        # 対象・don_k の枝がどれだけ踏まれたか）。
        for mv in legal:
            at = mv.get("action_type") or ""
            totals["cand_kinds"][at] = totals["cand_kinds"].get(at, 0) + 1
            pl = mv.get("payload") or {}
            if pl.get("target_ids"):
                totals["cands_with_target"] += 1
            if at == "DON_BOX" and pl.get("don_k"):
                totals["cands_don_k"] += 1
        worst = max((abs(a - b) for a, b in zip(got, exp)), default=0.0)
        totals["priors_max_abs_err"] = max(totals["priors_max_abs_err"], worst)
        if not (worst <= tol):
            totals["mismatch"] += 1
            totals["priors_mismatch"] += 1
            if not firsts:
                q = max(range(len(exp)), key=lambda i: abs(got[i] - exp[i]))
                firsts.append({"kind": "priors", "board": board_index, "seed": row["seed"],
                               "seat": name, "cand": q, "moves": n_moves,
                               "python": exp[q], "rust": got[q], "abs_err": worst})
        elif len(exp) > 1:
            totals["multi_cand_rows"] += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="NRel forward（value/priors）を Python と Rust で照合する")
    ap.add_argument("--boards", type=int, default=200, help="照合する局面数（既定 200）")
    ap.add_argument("--games", type=int, default=10, help="局面を採る局数（方策ごと・既定 10）")
    ap.add_argument("--policy", choices=["random", "l1", "both"], default="both",
                    help="局面を採る方策（既定 both＝random と l1 の両方から等分に採る）")
    ap.add_argument("--seed-base", type=int, default=900000, help="seed の基点（既定 900000）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="1局の上限ステップ")
    ap.add_argument("--net", default=DEFAULT_NET, help="NRel の npz（既定 nrel_a1.npz）")
    ap.add_argument("--tol", type=float, default=1e-5, help="許容絶対誤差（既定 1e-5）")
    ap.add_argument("--verbose", action="store_true", help="局面ごとの進捗を出す")
    args = ap.parse_args(argv)

    db = load_db()
    t0 = time.time()
    adapter, priors_fn, vocab = NR.load_serve_parts(args.net, db)
    print(f"[net] {args.net} hidden={adapter.net.W1.shape[1]} "
          f"ablate={sorted(adapter.net.ablate)} vocab={len(adapter.vocab_ids)} "
          f"({time.time() - t0:.1f}s)")

    tmp = tempfile.TemporaryDirectory(prefix="rs_net_oracle_")
    tables_path = _os.path.join(tmp.name, "eff_tables.npz")
    dump_tables(tables_path, adapter)
    summary_load = None
    if opcg_engine is not None and hasattr(opcg_engine, "load_net"):
        summary_load = json.loads(opcg_engine.load_net(args.net, tables_path))
        ok_vocab = summary_load.get("vocab_ids") == list(adapter.vocab_ids)
        ok_rows = summary_load.get("card_table_rows") == len(adapter.tab)
        print(f"[net] load_net -> hidden={summary_load.get('hidden')} "
              f"ablate={summary_load.get('ablate')} rows={summary_load.get('card_table_rows')} "
              f"vocab_ok={ok_vocab}")
        if not (ok_vocab and ok_rows):
            print("RS_NET " + json.dumps({"boards": 0, "mismatch": 1,
                                          "first": {"kind": "load_net", "vocab_ok": ok_vocab,
                                                    "rows_ok": ok_rows}}))
            return 1

    policies = ["random", "l1"] if args.policy == "both" else [args.policy]
    rows = []
    for i, policy in enumerate(policies):
        t0 = time.time()
        got = collect_boards(db, args.games, policy, args.seed_base + 1000 * i, args.max_steps)
        print(f"[boards] policy={policy} games={args.games} rows={len(got)} "
              f"({time.time() - t0:.1f}s)")
        rows.append(got)
    per_policy = max(1, args.boards // len(rows)) if rows else 0
    boards = []
    for got in rows:
        boards.extend(evenly(got, per_policy))
    boards = boards[:args.boards] if args.boards > 0 else boards

    game = OPCGGame()
    totals = {"values": 0, "priors_rows": 0, "priors_cands": 0, "mismatch": 0,
              "value_mismatch": 0, "priors_mismatch": 0, "priors_error": 0,
              "unimplemented": 0, "bad_payload": 0, "harness_error": 0,
              "value_max_abs_err": 0.0, "priors_max_abs_err": 0.0,
              "nonzero_values": 0, "multi_cand_rows": 0,
              "cand_kinds": {}, "cands_with_target": 0, "cands_don_k": 0}
    firsts = []
    t0 = time.time()
    for i, row in enumerate(boards):
        try:
            check_board(db, row, game, adapter, priors_fn, i, args.tol, totals, firsts)
        except Exception as e:  # noqa: BLE001 - ハーネス自身の事故も集計に載せる
            totals["harness_error"] += 1
            if not firsts:
                firsts.append({"kind": "harness_error", "board": i,
                               "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        if args.verbose and (i + 1) % 10 == 0:
            print(f"[boards] {i + 1}/{len(boards)} values={totals['values']} "
                  f"priors={totals['priors_rows']} mismatch={totals['mismatch']} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    summary = {
        "boards": len(boards),
        "values": totals["values"],                    # 局面×両視点
        "priors_rows": totals["priors_rows"],          # 手番側の候補集合の数
        "priors_cands": totals["priors_cands"],        # 候補の総数
        "value_max_abs_err": float(f"{totals['value_max_abs_err']:.3e}"),
        "priors_max_abs_err": float(f"{totals['priors_max_abs_err']:.3e}"),
        "mismatch": totals["mismatch"],
        "value_mismatch": totals["value_mismatch"],
        "priors_mismatch": totals["priors_mismatch"],
        "priors_error": totals["priors_error"],
        "unimplemented": totals["unimplemented"],
        "bad_payload": totals["bad_payload"],
        "harness_error": totals["harness_error"],
        # 空振り検査: value が 0 でない件数・候補が 2 つ以上あった件数（1 候補は常に確率 1.0）
        "nonzero_values": totals["nonzero_values"],
        "multi_cand_rows": totals["multi_cand_rows"],
        "cand_kinds": dict(sorted(totals["cand_kinds"].items())),
        "cands_with_target": totals["cands_with_target"],
        "cands_don_k": totals["cands_don_k"],
        "tol": args.tol,
        "net": _os.path.basename(args.net),
        "ablate": sorted(adapter.net.ablate),
        "vocab": len(vocab),
        "games": args.games,
        "policy": args.policy,
        "seed_base": args.seed_base,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "seconds": round(time.time() - t0, 1),
        "first": firsts[0] if firsts else None,
    }
    print("RS_NET " + json.dumps(summary, ensure_ascii=False))
    tmp.cleanup()
    bad = (totals["mismatch"] or totals["bad_payload"] or totals["unimplemented"]
           or totals["harness_error"] or totals["priors_error"] or not totals["values"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
