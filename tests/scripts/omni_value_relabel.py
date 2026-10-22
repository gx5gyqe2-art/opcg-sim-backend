"""全部見える探索の評価を価値の教師として付け直す（`docs/search_efficiency_plan.md` §7.16・WP `omni-value` 段階 1）。

記録（`n_records/n4N_wKK_pX`・dump v5）の**全判断点**について、その判断点の本当の盤面で
`Game.decide` を **`omniscient=true`・sims 320・worlds 1・budget 0・ネット r13d** で回し、
根の価値（手番の席から見た値・訪問で重み付けた根の Q の平均＝記録の `pol_v0` と同じ式）を
`v_omni` として書く。

**盤面の復元**: dump v5 の npz は `hidden`（完全な内部状態）を持たない（列は符号化・候補・z・
`sig`／`forced_sig` などだけ）。そこで**局の seed から対局を作り直し、記録した手を順に当て直す**
（探索は回さない）。対局の乱数は `Game(seed)` の PCG32 だけで決まる（`driver.py` の決定論の契約）。
探索を回して同じ手を引き直す道は使えない——波 43 の後に既定が `reveal=fair` になり、高速化も入った
ので、同じ seed でも 5 手目で別の手になる（実測・notes）。

当て直す手は記録の `sig`（箱レベルの move_sig）から作る:
  - `forced`≠0 の行は `forced_sig`（ε で差し替えた実際の手）
  - `DON_BOX` は先頭原始手（付与 k>0 なら ATTACH_DON・k=0 なら ATTACK・`record_gen.first_primitive`）
  - `SETUP_BOX` は素の PLAY／ACTIVATE_MAIN
  - 効果の対話（`RESOLVE_EFFECT_SELECTION`）は合法手が既定解決 1 本だけなので、`sig` の
    `selected_uuids`／`accepted` を既定の payload に書き込む。CHOICE の番号・ARRANGE_DECK の上／下は総当たり
  - `sig` で決まらない手（カウンターに使う札〔battle の手は uuid を sig に持たない〕・CHOICE の番号）は
    **候補を全部当てて、次の行の符号化が記録と一致するものだけ残す**（小さなビーム）
**検算**: 各行で、復元した盤面の符号化（`tokens`・`scalars`）が記録とビット一致することを確かめる
（一致しなければその局の残りの行は `ok=0`）。

使い方:
  python tests/scripts/omni_value_relabel.py run --in ~/n43/n_records/n43_w01_p0 --workers 4
  python tests/scripts/omni_value_relabel.py run --in "~/n43/n_records/n43_w01_p*" --out-dir n1_results/omni_value
  python tests/scripts/omni_value_relabel.py summarize --omni "n1_results/omni_value/omni_*.npz"
出力 `omni_<dir>.npz`（入力の行と同じ並び）の列: `seed`・`step`・`turn`・`who`・`kind`（行の対応鍵と層別用）、
`z`（記録の勝敗）、`v_omni`（float32・根の探索が無い点〔効果の窓など〕と復元に失敗した行は NaN）、
`n_omni`（根の訪問の和）、`v_raw`（r13d の素の価値＝探索なし・公平な入力）、`v0`（記録の `pol_v0`）、
`ok`（1＝復元できた）、`sec`（その行の decide の秒）。訓練側は `n_rel_train --z-mode omni|mix --omni-in` で読む。
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OPCG_LOG_SILENT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from opcg_sim.loop import decks as D, driver as DR, engine as E, record_gen as RG  # noqa: E402

OUT_DIR = os.path.join("n1_results", "omni_value")
#: ビームの上限（sig で決まらない手の分岐。超えたら先頭から残す）。
BEAM_MAX = 64
COLS = ("seed", "step", "turn", "who", "kind", "z", "v_omni", "n_omni", "v_raw", "v0", "ok", "sec")

_G = {}


# --- 盤面の復元 ------------------------------------------------------------------------------

def applied_sig(row):
    """記録の行 → 実対局に出た手の move_sig（箱は先頭原始手へ）。"""
    if row["forced"]:
        return json.loads(row["forced_sig"])
    sig = json.loads(row["sig"])
    if sig[0] == "DON_BOX":
        k = int(row["k"])
        if k > 0 or not sig[2]:
            return ["ATTACH_DON", sig[1], [], [], None]
        return ["ATTACK", sig[1], sig[2], [], None]
    return sig


def candidates(game, who, sig):
    """`sig` に当てはまる手（合法手そのもの＋既定の payload に sig を書き込んだもの）。"""
    legal = json.loads(game.legal_json(who))
    out, seen = [], set()

    def add(mv):
        key = json.dumps(mv, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(mv)

    for mv in legal:
        if RG.move_sig(mv) == sig:
            add(mv)
    at, uuid, tids, sel, acc = sig
    if at == "SETUP_BOX":
        for mv in legal:
            if mv.get("action_type") in ("PLAY", "ACTIVATE_MAIN") \
                    and (mv.get("payload") or {}).get("uuid") == uuid:
                add(mv)
    pend = None
    for mv in legal:
        if mv.get("kind") != "game" or mv.get("action_type") != at:
            continue
        p = dict(mv.get("payload") or {})
        if uuid is not None:
            p["uuid"] = uuid
        if list(p.get("target_ids") or ()) != list(tids):
            p["target_ids"] = list(tids)
        if list(p.get("selected_uuids") or ()) != list(sel):
            p["selected_uuids"] = list(sel)
        if acc is not None:
            p["accepted"] = acc
        new = dict(mv, payload=p)
        if RG.move_sig(new) != sig:
            continue
        add(new)
        if pend is None:
            pend = json.loads(game.pending_json()) or {}
        if pend.get("action") == "CHOICE":
            for i in range(len(pend.get("options") or [])):
                add(dict(mv, payload=dict(p, index=i)))
        # 上／下の選択（探索は `position` を BOTTOM／TOP で分ける・`search/adapter.rs`）も sig に載らない
        if pend.get("action") == "ARRANGE_DECK" and pend.get("allow_position"):
            for pos in ("BOTTOM", "TOP"):
                add(dict(mv, payload=dict(p, position=pos)))
    return out


def apply_move(game, who, mv):
    if mv.get("kind") == "battle":
        game.apply_battle_action(who, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(who, mv["action_type"],
                               json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def enc_matches(game, who, row):
    enc = json.loads(game.encode(who))
    tok = np.asarray(enc["tokens"], np.float32).astype(np.float16).reshape(row["tokens"].shape)
    sc = np.asarray(enc["scalars"], np.float32).astype(np.float16)
    return np.array_equal(tok, row["tokens"]) and np.array_equal(sc, row["scalars"]), enc


def root_value(out):
    """decide の出力 → (根の価値, 訪問の和)。根の探索が無い点は (nan, 0)。"""
    groups = out.get("groups") or []
    ntot = sum(float(g["n"]) for g in groups)
    if out.get("kind") != "main" or ntot <= 0:
        return float("nan"), 0.0
    return sum(float(g["n"]) * float(g["q"]) for g in groups) / ntot, ntot


# --- 1 局 -------------------------------------------------------------------------------------

def _init(net, sims, worlds, budget, decks):
    eng = E.engine()
    _G["net"] = E.load_net(net)
    _G["eng"] = eng
    _G["db"] = D.load_db()
    _G["opts"] = {"net": _G["net"], "sims": int(sims), "worlds": int(worlds),
                  "budget": int(budget), "omniscient": True}
    _G["decks"] = decks


def relabel_game(job):
    """1 局ぶんの行 → 列の dict（行は記録の並び）と局の診断。"""
    seed, rows = job
    n = len(rows)
    res = {k: np.zeros(n, np.float32) for k in ("v_omni", "n_omni", "v_raw", "sec")}
    res["v_omni"][:] = np.nan
    res["v_raw"][:] = np.nan
    res["ok"] = np.zeros(n, np.int8)
    diag = {"seed": seed, "rows": n, "fail": None, "beam_max": 1}
    db, eng = _G["db"], _G["eng"]
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, _G["decks"])
    # ビームの要素は (盤面, ここまでに当てた手)。`Game.fork()` は対局の乱数を写さない（山札を混ぜる
    # 効果の後で記録と食い違う・実測）ので、分岐は seed から手の列を当て直して作る。
    beam = [(DR.new_game(seed, p1, p2), [])]
    for j, row in enumerate(rows):
        who = "p1" if row["who"] == 0 else "p2"
        if row["step"] != j:
            diag["fail"] = f"step の抜け（行 {j} が step {row['step']}）"
            break
        live, enc = [], None
        for g, hist in beam:
            ok, e = enc_matches(g, who, row)
            if ok:
                live.append((g, hist))
                enc = enc or e
        if not live:
            diag["fail"] = f"符号化が一致しない（step {j}）"
            break
        beam = live
        game = beam[0][0]
        res["ok"][j] = 1
        # 素の価値（探索なし・公平な入力）。`net_eval` は既定ネット（最初に読んだ 1 本）＝r13d。
        enc["tok"] = enc.pop("tokens")
        res["v_raw"][j] = float(json.loads(eng.net_eval(json.dumps(enc), "[]"))["value"])
        opts = dict(_G["opts"], search_seed=E.search_seed(seed, int(row["turn"]), who))
        t0 = time.time()
        out = json.loads(game.decide(who, json.dumps(opts)))       # decide は盤面を変えない
        res["sec"][j] = time.time() - t0
        res["v_omni"][j], res["n_omni"][j] = root_value(out)
        if j == n - 1:
            break
        sig = applied_sig(row)
        nxt = []
        for g, hist in beam:
            cs = candidates(g, who, sig)
            # 先頭以外の候補は seed から当て直した盤面に当てる（先頭は手元の盤面を進める＝最後に）
            for mv in cs[1:]:
                h = DR.new_game(seed, p1, p2)
                try:
                    for w, m in hist:
                        apply_move(h, w, m)
                    apply_move(h, who, mv)
                except Exception:                                  # noqa: BLE001  合わない候補は捨てる
                    continue
                nxt.append((h, hist + [(who, mv)]))
            if cs:
                try:
                    apply_move(g, who, cs[0])
                    nxt.append((g, hist + [(who, cs[0])]))
                except Exception:                                  # noqa: BLE001
                    pass
            if len(nxt) >= BEAM_MAX:
                break
        if not nxt:
            diag["fail"] = f"手が当てられない（step {j}・sig {row['sig']}）"
            break
        diag["beam_max"] = max(diag["beam_max"], len(nxt))
        beam = nxt[:BEAM_MAX]
    return res, diag


# --- 1 ディレクトリ ---------------------------------------------------------------------------

def load_rows(path):
    """npz → 局ごとの行（必要な列だけ・記録の並びのまま）。"""
    with np.load(path, allow_pickle=False) as f:         # 列は 1 度だけ読む（NpzFile は引くたびに展開する）
        z = {k: f[k] for k in ("z", "seed", "who", "step", "turn", "kind", "sig", "forced", "forced_sig",
                               "pol_len", "pol_chosen", "pol_k", "pol_v0", "tokens", "scalars")}
    off = np.concatenate([[0], np.cumsum(z["pol_len"])]).astype(np.int64)
    pc = z["pol_chosen"].astype(np.int64)
    pk = z["pol_k"]
    games, order = {}, []
    for r in range(len(z["z"])):
        s = int(z["seed"][r])
        if s not in games:
            games[s] = []
            order.append(s)
        k = int(pk[off[r] + pc[r]]) if (z["pol_len"][r] > 0 and pc[r] >= 0) else -1
        games[s].append({"who": int(z["who"][r]), "step": int(z["step"][r]), "turn": int(z["turn"][r]),
                         "sig": str(z["sig"][r]), "forced": int(z["forced"][r]),
                         "forced_sig": str(z["forced_sig"][r]), "k": k,
                         "tokens": z["tokens"][r], "scalars": z["scalars"][r]})
    meta = {"seed": z["seed"], "step": z["step"], "turn": z["turn"], "who": z["who"],
            "kind": z["kind"], "z": z["z"], "v0": z["pol_v0"]}
    return order, games, meta


def run_dir(d, out_dir, pool, limit_games=0):
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(d, "n_record_*.npz")))
    files = [f for f in files if not (f.endswith(".plan.npz") or f.endswith(".time.npz"))]
    cols = {k: [] for k in COLS}
    diags = []
    jobs, tags = [], []
    for f in files:
        order, games, meta = load_rows(f)
        if limit_games:
            order = order[:max(0, limit_games - len(jobs))]
        keep = np.isin(meta["seed"], np.asarray(order, np.int64))
        for k in ("seed", "step", "turn", "who", "kind", "z", "v0"):
            cols[k].append(np.asarray(meta[k])[keep])
        jobs += [(s, games[s]) for s in order]
        tags += [os.path.basename(f)] * len(order)
        if limit_games and len(jobs) >= limit_games:
            break
    # 局をまとめて流す（局の長さがばらつくのでファイルごとに待たない）。並びは imap が保つ
    for i, (res, diag) in enumerate(pool.imap(relabel_game, jobs, chunksize=1)):
        diag["file"] = tags[i]
        diags.append(diag)
        for k in ("v_omni", "n_omni", "v_raw", "sec", "ok"):
            cols[k].append(res[k])
        if (i + 1) % 40 == 0 or i + 1 == len(jobs):
            print(f"{os.path.basename(d)}: {i + 1}/{len(jobs)} 局 {time.time() - t0:.0f}s", flush=True)
    arr = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    arr["seed"] = arr["seed"].astype(np.int64)
    arr["step"] = arr["step"].astype(np.int32)
    arr["v_omni"] = arr["v_omni"].astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"omni_{os.path.basename(os.path.normpath(d))}.npz")
    np.savez_compressed(out, **arr)
    info = {"dir": d, "out": out, "rows": int(len(arr["z"])), "games": len(diags),
            "ok_rows": int(arr["ok"].sum()), "omni_rows": int(np.isfinite(arr["v_omni"]).sum()),
            "failed_games": [g for g in diags if g["fail"]],
            "beam_max": max([g["beam_max"] for g in diags] or [0]),
            "decide_sec_sum": float(arr["sec"].sum()), "wall_sec": round(time.time() - t0, 1)}
    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    return info


def cmd_run(a):
    dirs = []
    for pat in a.src:
        pat = os.path.expanduser(pat)
        dirs += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.net, a.sims, a.worlds, a.budget, a.decks)) as pool:
        for d in dirs:
            info = run_dir(d, a.out_dir, pool, a.limit_games)
            print(json.dumps({k: v for k, v in info.items() if k != "failed_games"},
                             ensure_ascii=False), flush=True)
            print(f"  復元に失敗した局 {len(info['failed_games'])}", flush=True)
    return 0


# --- 読み取り（相関） -----------------------------------------------------------------------

def _corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(arr):
    """`v_omni` と z・素の価値・記録の根の価値の相関（全体とターン帯別）。"""
    m = np.isfinite(arr["v_omni"]) & (arr["ok"] == 1)
    vo, z, vr, v0, turn = (arr["v_omni"][m], arr["z"][m], arr["v_raw"][m],
                           arr["v0"][m].astype(np.float32), arr["turn"][m])
    main0 = arr["kind"][m] == 0

    def block(sel):
        return {"n": int(sel.sum()),
                "corr_omni_z": _corr(vo[sel], z[sel]),
                "corr_raw_z": _corr(vr[sel], z[sel]),
                "corr_v0_z": _corr(v0[sel & main0], z[sel & main0]),
                "corr_omni_raw": _corr(vo[sel], vr[sel]),
                "corr_omni_v0": _corr(vo[sel & main0], v0[sel & main0]),
                "mse_omni_z": float(np.mean((vo[sel] - z[sel]) ** 2)) if sel.any() else float("nan"),
                "mse_raw_z": float(np.mean((vr[sel] - z[sel]) ** 2)) if sel.any() else float("nan"),
                "sign_omni_z": float(np.mean((vo[sel] > 0) == (z[sel] > 0))) if sel.any() else float("nan"),
                "sign_raw_z": float(np.mean((vr[sel] > 0) == (z[sel] > 0))) if sel.any() else float("nan"),
                "mean_abs_omni_minus_raw": float(np.mean(np.abs(vo[sel] - vr[sel]))) if sel.any() else float("nan")}

    out = {"all": block(np.ones(len(vo), bool))}
    for lo, hi in ((0, 2), (3, 4), (5, 6), (7, 8), (9, 99)):
        out[f"turn_{lo}-{hi}"] = block((turn >= lo) & (turn <= hi))
    out["rows_total"] = int(len(arr["z"]))
    out["rows_omni"] = int(m.sum())
    out["sec_per_point"] = float(arr["sec"][m].mean()) if m.any() else float("nan")
    return out


def cmd_summarize(a):
    files = sorted(glob.glob(os.path.expanduser(a.omni)))
    parts = [dict(np.load(f)) for f in files]
    arr = {k: np.concatenate([p[k] for p in parts]) for k in COLS}
    s = summarize(arr)
    s["files"] = files
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--in", dest="src", nargs="+", required=True, help="記録のディレクトリ（glob 可）")
    r.add_argument("--out-dir", default=OUT_DIR)
    r.add_argument("--net", default=os.path.join(E.MODELS, "nrel_r13d.npz"), help="既定 r13d（同梱）")
    r.add_argument("--sims", type=int, default=320)
    r.add_argument("--worlds", type=int, default=1)
    r.add_argument("--budget", type=int, default=0)
    r.add_argument("--decks", default="synth_roles", help="記録の meta_n_record.json の decks と同じもの")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit-games", type=int, default=0, help="0＝全局（計測用に先頭 N 局だけ）")
    s = sub.add_parser("summarize")
    s.add_argument("--omni", required=True, help="omni_*.npz の glob")
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
